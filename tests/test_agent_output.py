from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from schemas.agent_output import AgentOutput, DataWindow


def _valid_kwargs(**overrides):
    kwargs = dict(
        agent="seismic",
        region_id="cell_0142",
        timestamp=datetime(2026, 10, 3, tzinfo=timezone.utc),
        hazard="earthquake_m5plus_7d",
        risk_score=0.08,
        confidence=0.6,
        evidence=["14 M3+ events in 72h, 4.1x background rate"],
        data_window=DataWindow(start=date(2026, 8, 4), end=date(2026, 10, 3)),
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_output_round_trips():
    output = AgentOutput(**_valid_kwargs())
    assert output.risk_score == 0.08
    parsed = AgentOutput.model_validate_json(output.model_dump_json())
    assert parsed == output


@pytest.mark.parametrize("field,value", [("risk_score", 1.5), ("risk_score", -0.1), ("confidence", 1.1)])
def test_out_of_range_scores_rejected(field, value):
    with pytest.raises(ValidationError):
        AgentOutput(**_valid_kwargs(**{field: value}))


def test_naive_timestamp_assumed_utc():
    output = AgentOutput(**_valid_kwargs(timestamp=datetime(2026, 10, 3)))
    assert output.timestamp.tzinfo is not None
    assert output.timestamp.utcoffset().total_seconds() == 0


def test_data_window_end_before_start_rejected():
    with pytest.raises(ValidationError):
        DataWindow(start=date(2026, 10, 3), end=date(2026, 8, 4))


def test_missing_required_field_rejected():
    kwargs = _valid_kwargs()
    del kwargs["region_id"]
    with pytest.raises(ValidationError):
        AgentOutput(**kwargs)
