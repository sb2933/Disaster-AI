from pathlib import Path

import pandas as pd
import pytest

from agents.seismic.features import build_features
from agents.seismic.predict import build_evidence, compute_confidence
from schemas.agent_output import AgentOutput, DataWindow

LAT, LON = 29.0, 84.0
RAW_COLUMNS = ["event_id", "time", "lat", "lon", "depth_km", "mag", "mag_type"]


def _row(event_id: str, date_str: str, mag: float) -> dict:
    return {
        "event_id": event_id, "time": f"{date_str}T00:00:00+00:00",
        "lat": LAT, "lon": LON, "depth_km": 10.0, "mag": mag, "mag_type": "mb",
    }


def test_as_of_extends_date_range_past_last_event(tmp_path: Path):
    """A quiet spell after the last recorded event must still get zero-filled
    feature rows through `as_of`, instead of the range silently stopping at
    whenever the last earthquake happened to occur (see predict.py)."""
    rows = [_row(f"filler{i}", f"2020-01-{i:02d}", 3.5) for i in range(1, 6)]
    catalog = pd.DataFrame(rows, columns=RAW_COLUMNS)
    raw_path = tmp_path / "catalog.csv"
    catalog.to_csv(raw_path, index=False)

    as_of = pd.Timestamp("2020-03-01")
    features = build_features(raw_path, tmp_path / "features.parquet", as_of=as_of)
    assert features["date"].max() == as_of

    features_no_as_of = build_features(raw_path, tmp_path / "features2.parquet")
    assert features_no_as_of["date"].max() == pd.Timestamp("2020-01-05")


def test_compute_confidence_monotonic_and_bounded():
    assert compute_confidence(0) == pytest.approx(0.2)
    assert compute_confidence(50) == pytest.approx(1.0)
    assert compute_confidence(500) == pytest.approx(1.0)  # saturates, never exceeds 1.0
    assert compute_confidence(0) < compute_confidence(10) < compute_confidence(50)


def test_build_evidence_falls_back_when_nothing_notable():
    row = pd.Series({
        "count_7d": 0.0, "relative_rate_7d": float("nan"), "max_mag_30d": 0.0,
        "days_since_max_mag_30d": float("nan"), "b_value_valid": 0, "b_value": float("nan"),
        "neighbor_count_7d": 0.0,
    })
    evidence = build_evidence(row, mc=4.5)
    assert evidence == ["No significant recent seismicity detected"]


def test_build_evidence_mentions_recent_activity():
    row = pd.Series({
        "count_7d": 14.0, "relative_rate_7d": 4.1, "max_mag_30d": 5.2,
        "days_since_max_mag_30d": 3.0, "b_value_valid": 1, "b_value": 0.95,
        "neighbor_count_7d": 2.0,
    })
    evidence = build_evidence(row, mc=4.5)
    joined = " | ".join(evidence)
    assert "14" in joined and "4.1" in joined
    assert "5.2" in joined and "3 days ago" in joined
    assert "0.95" in joined
    assert "2" in joined


def test_predict_output_is_valid_agent_output():
    """A record built the way predict.py builds it must validate against the
    shared AgentOutput contract."""
    from datetime import datetime, timezone

    output = AgentOutput(
        agent="seismic", region_id="r-6_c2", timestamp=datetime.now(timezone.utc),
        hazard="earthquake_m5plus_7d", risk_score=0.0061, confidence=0.968,
        evidence=build_evidence(pd.Series({
            "count_7d": 0.0, "relative_rate_7d": float("nan"), "max_mag_30d": 4.1,
            "days_since_max_mag_30d": 10.0, "b_value_valid": 0, "b_value": float("nan"),
            "neighbor_count_7d": 0.0,
        }), mc=4.5),
        data_window=DataWindow(start=pd.Timestamp("2026-08-06").date(), end=pd.Timestamp("2026-10-04").date()),
    )
    assert 0.0 <= output.risk_score <= 1.0
