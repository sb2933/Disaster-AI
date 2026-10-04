"""Shared output contract for all disaster-intelligence agents.

Every agent in this system (seismic, flood, landslide, ...) must emit its
forecasts as a list of ``AgentOutput`` records. Keeping this contract
agent-agnostic lets a future aggregator/orchestrator consume any agent's
output uniformly, without knowing about that agent's internals.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from pydantic import BaseModel, Field, field_validator


class DataWindow(BaseModel):
    """The time span of input data actually used to produce a forecast.

    Recording this explicitly (rather than just "as of timestamp") makes it
    possible to audit exactly which observations a given risk_score is
    conditioned on.
    """

    start: date = Field(..., description="Earliest date of data used, inclusive.")
    end: date = Field(..., description="Latest date of data used, inclusive.")

    @field_validator("end")
    @classmethod
    def _end_not_before_start(cls, end: date, info) -> date:
        start = info.data.get("start")
        if start is not None and end < start:
            raise ValueError("data_window.end must not be before data_window.start")
        return end


class AgentOutput(BaseModel):
    """A single agent's hazard forecast for one region at one point in time.

    This is intentionally generic: ``hazard`` names the hazard type
    (e.g. "earthquake_m5plus_7d"), ``region_id`` is an agent-defined spatial
    unit (e.g. a grid cell id), and ``risk_score`` / ``confidence`` are both
    normalised to [0, 1] so outputs from different agents are comparable at
    a glance even though the underlying models differ completely.
    """

    agent: str = Field(..., min_length=1, description="Agent identifier, e.g. 'seismic'.")
    region_id: str = Field(..., min_length=1, description="Agent-defined spatial unit id, e.g. a grid cell id.")
    timestamp: datetime = Field(..., description="UTC time the forecast was produced.")
    hazard: str = Field(..., min_length=1, description="Hazard type and horizon, e.g. 'earthquake_m5plus_7d'.")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability of the hazard occurring in the forecast window.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent's confidence in risk_score itself (e.g. based on data sufficiency), not the risk_score's value.")
    evidence: list[str] = Field(default_factory=list, description="Human-readable strings justifying the score, e.g. '14 M3+ events in 72h, 4.1x background rate'.")
    data_window: DataWindow = Field(..., description="Span of input data used to compute this forecast.")

    @field_validator("timestamp")
    @classmethod
    def _timestamp_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    model_config = {
        "json_schema_extra": {
            "example": {
                "agent": "seismic",
                "region_id": "cell_0142",
                "timestamp": "2026-10-03T00:00:00Z",
                "hazard": "earthquake_m5plus_7d",
                "risk_score": 0.08,
                "confidence": 0.6,
                "evidence": [
                    "14 M3+ events in 72h, 4.1x background rate",
                    "max magnitude last 30d: 4.3 (6 days ago)",
                ],
                "data_window": {"start": "2026-08-04", "end": "2026-10-03"},
            }
        }
    }
