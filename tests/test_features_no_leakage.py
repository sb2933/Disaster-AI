from pathlib import Path

import pandas as pd

from agents.seismic.features import FEATURE_COLUMNS, build_features

LAT, LON = 29.0, 84.0
RAW_COLUMNS = ["event_id", "time", "lat", "lon", "depth_km", "mag", "mag_type"]


def _row(event_id: str, date_str: str, mag: float) -> dict:
    return {
        "event_id": event_id,
        "time": f"{date_str}T00:00:00+00:00",
        "lat": LAT,
        "lon": LON,
        "depth_km": 10.0,
        "mag": mag,
        "mag_type": "mb",
    }


def _build_catalog(extra_future_events: list[dict]) -> pd.DataFrame:
    rows = [_row(f"filler{i}", f"2020-01-{i:02d}", 3.5) for i in range(1, 10)]
    rows.append(_row("target", "2020-06-15", 4.5))
    rows += extra_future_events
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _only_cell_with_events(features: pd.DataFrame) -> str:
    counts = features.groupby("cell_id")["count_1d"].sum()
    return counts[counts > 0].idxmax()


def test_features_before_cutoff_unaffected_by_later_events(tmp_path: Path):
    """Features for days up to a cutoff must be identical whether or not the
    catalog contains events that happen after that cutoff."""
    base_catalog = _build_catalog(extra_future_events=[])
    future_events = [_row("future1", "2020-09-01", 6.0), _row("future2", "2020-09-02", 5.5)]
    catalog_with_future = _build_catalog(extra_future_events=future_events)

    base_path = tmp_path / "base.csv"
    future_path = tmp_path / "with_future.csv"
    base_catalog.to_csv(base_path, index=False)
    catalog_with_future.to_csv(future_path, index=False)

    feat_base = build_features(base_path, tmp_path / "base.parquet")
    feat_future = build_features(future_path, tmp_path / "future.parquet")

    cell_id = _only_cell_with_events(feat_base)
    # feat_base's date range ends at its last event (2020-06-15); only dates
    # within both runs' ranges can be compared.
    cutoff = pd.Timestamp("2020-06-15")

    a = feat_base[(feat_base["cell_id"] == cell_id) & (feat_base["date"] <= cutoff)].set_index("date").sort_index()
    b = feat_future[(feat_future["cell_id"] == cell_id) & (feat_future["date"] <= cutoff)].set_index("date").sort_index()

    assert list(a.index) == list(b.index)
    for col in FEATURE_COLUMNS:
        pd.testing.assert_series_equal(a[col], b[col], check_names=False)


def test_feature_window_does_not_include_forecast_day_plus_one(tmp_path: Path):
    """A spike the day AFTER the forecast day must not appear in that day's features."""
    catalog = _build_catalog(extra_future_events=[_row("day_after", "2020-06-16", 7.0)])
    raw_path = tmp_path / "catalog.csv"
    catalog.to_csv(raw_path, index=False)
    features = build_features(raw_path, tmp_path / "features.parquet")

    cell_id = _only_cell_with_events(features)
    day_t = features[(features["cell_id"] == cell_id) & (features["date"] == "2020-06-15")].iloc[0]
    assert day_t["max_mag_30d"] == 4.5  # the M7.0 on 06-16 must not leak into day 06-15's features
