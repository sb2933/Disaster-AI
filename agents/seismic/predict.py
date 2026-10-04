"""Daily live forecast run for the seismic agent.

Fetches recent events, rebuilds the feature grid from the full catalog,
loads the trained model (+ its scaler/calibrator from the checkpoint),
and writes one AgentOutput per active Nepal grid cell to
reports/latest_forecast.json.

Run: python -m agents.seismic.predict [--skip-fetch]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import torch
from pyproj import Transformer

from agents.seismic.features import (
    CELL_SIZE_M,
    PROJ_STRING,
    DEFAULT_FEATURES_PATH,
    build_features,
    load_catalog,
    parse_cell_id,
    prepare_grid,
)
from agents.seismic.model import SeismicForecastModel
from agents.seismic.train import FEATURE_COLUMNS, transform_features
from pipelines.fetch_usgs import DEFAULT_RAW_PATH, fetch_recent
from schemas.agent_output import AgentOutput, DataWindow

logger = logging.getLogger(__name__)

HAZARD = "earthquake_m5plus_7d"
AGENT_NAME = "seismic"
DEFAULT_CHECKPOINT_PATH = Path("models/seismic_checkpoint.pt")
DEFAULT_FORECAST_PATH = Path("reports/latest_forecast.json")

# Test region, per the project spec: Nepal.
NEPAL_BBOX = {"min_lat": 26.3, "max_lat": 30.5, "min_lon": 80.0, "max_lon": 88.2}

# A cell's forecast confidence saturates once it has this many historical
# events -- this is the same threshold used for trusting the rolling
# b-value estimate in features.py, reused here as a general "enough data to
# say something" heuristic. Below that, confidence degrades linearly.
CONFIDENCE_SATURATION_EVENTS = 50
CONFIDENCE_FLOOR = 0.2


def get_nepal_cells(cell_ids: set[str]) -> list[str]:
    """Active cells whose centre falls inside the Nepal test-region bounding box."""
    inverse = Transformer.from_crs(PROJ_STRING, "EPSG:4326", always_xy=True)
    nepal_cells = []
    for cell_id in cell_ids:
        row, col = parse_cell_id(cell_id)
        x_m, y_m = (col + 0.5) * CELL_SIZE_M, (row + 0.5) * CELL_SIZE_M
        lon, lat = inverse.transform(x_m, y_m)
        if NEPAL_BBOX["min_lat"] <= lat <= NEPAL_BBOX["max_lat"] and NEPAL_BBOX["min_lon"] <= lon <= NEPAL_BBOX["max_lon"]:
            nepal_cells.append(cell_id)
    return nepal_cells


def compute_confidence(event_count: int) -> float:
    """Simple data-sufficiency heuristic: more historical events -> more confidence
    in the forecast itself (not in the risk_score's value). Saturates at
    CONFIDENCE_SATURATION_EVENTS events, consistent with features.py's
    b-value reliability threshold."""
    fraction = min(event_count, CONFIDENCE_SATURATION_EVENTS) / CONFIDENCE_SATURATION_EVENTS
    return float(CONFIDENCE_FLOOR + (1.0 - CONFIDENCE_FLOOR) * fraction)


def build_evidence(row: pd.Series, mc: float) -> list[str]:
    evidence = []
    if row["count_7d"] > 0:
        rate_str = ""
        if pd.notna(row["relative_rate_7d"]) and row["relative_rate_7d"] > 0:
            rate_str = f", {row['relative_rate_7d']:.1f}x background rate"
        evidence.append(f"{row['count_7d']:.0f} M{mc:.1f}+ events in the last 7 days{rate_str}")
    if pd.notna(row["max_mag_30d"]) and row["max_mag_30d"] > 0:
        evidence.append(
            f"max magnitude {row['max_mag_30d']:.1f} in the last 30 days, "
            f"{row['days_since_max_mag_30d']:.0f} days ago"
        )
    if row["b_value_valid"]:
        evidence.append(f"rolling b-value {row['b_value']:.2f} (365-day estimate)")
    if row["neighbor_count_7d"] > 0:
        evidence.append(f"{row['neighbor_count_7d']:.0f} events in neighbouring cells in the last 7 days")
    if not evidence:
        evidence.append("No significant recent seismicity detected")
    return evidence


def build_cell_feature_arrays(features_df: pd.DataFrame, scaler: dict) -> dict[str, dict]:
    """Per-cell, date-sorted (dates, scaled features) arrays, plus the latest raw row
    (for evidence text). Unlike train.build_cell_arrays, this needs no label column --
    we are forecasting the unknown future, not training on known outcomes."""
    df = features_df.sort_values(["cell_id", "date"]).reset_index(drop=True)
    scaled = transform_features(df, scaler)
    arrays: dict[str, dict] = {}
    for cell_id, group in df.groupby("cell_id", sort=False):
        idx = group.index.to_numpy()
        arrays[cell_id] = {
            "dates": group["date"].to_numpy(),
            "scaled_features": scaled[idx],
            "latest_row": group.iloc[-1],
        }
    return arrays


def run(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    raw_catalog_path: Path = DEFAULT_RAW_PATH,
    features_path: Path = DEFAULT_FEATURES_PATH,
    out_path: Path = DEFAULT_FORECAST_PATH,
    skip_fetch: bool = False,
    fetch_days: int = 2,
) -> list[AgentOutput]:
    if not skip_fetch:
        logger.info("Fetching last %d days of events...", fetch_days)
        fetch_recent(days=fetch_days, out_path=raw_catalog_path)
    else:
        logger.info("Skipping fetch, using existing catalog at %s", raw_catalog_path)

    as_of = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
    logger.info("Rebuilding features from the full catalog (as_of=%s)...", as_of.date())
    features_df = build_features(raw_catalog_path, features_path, as_of=as_of)

    catalog = load_catalog(raw_catalog_path)
    assigned, active_cells, date_range, mc_lookup = prepare_grid(catalog, as_of=as_of)
    event_counts = assigned.groupby("cell_id").size()
    latest_mc = mc_lookup[max(mc_lookup)]

    nepal_cells = [c for c in get_nepal_cells(active_cells) if c in features_df["cell_id"].unique()]
    logger.info("%d active Nepal cells to forecast", len(nepal_cells))
    if not nepal_cells:
        logger.warning("No active cells fall inside the Nepal bounding box; nothing to forecast.")
        return []

    ckpt = torch.load(checkpoint_path, weights_only=False)
    config = ckpt["config"]
    model = SeismicForecastModel(
        n_features=len(FEATURE_COLUMNS), seq_len=config["seq_len"],
        lstm_hidden=config["lstm_hidden"], lstm_layers=config["lstm_layers"],
        transformer_heads=config["transformer_heads"], transformer_layers=config["transformer_layers"],
        transformer_ff=config["transformer_ff"], dropout=config["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    calibrator = ckpt["calibrator"]
    seq_len = config["seq_len"]

    cell_arrays = build_cell_feature_arrays(features_df[features_df["cell_id"].isin(nepal_cells)], ckpt["scaler"])

    outputs: list[AgentOutput] = []
    now = datetime.now(timezone.utc)
    for cell_id in nepal_cells:
        arr = cell_arrays[cell_id]
        if len(arr["dates"]) < seq_len:
            logger.warning("Cell %s has only %d days of history (<%d); skipping.", cell_id, len(arr["dates"]), seq_len)
            continue

        seq = arr["scaled_features"][-seq_len:]
        with torch.no_grad():
            raw_prob = torch.sigmoid(model(torch.from_numpy(seq).unsqueeze(0))).item()
        risk_score = float(calibrator.transform([raw_prob])[0])

        latest_row = arr["latest_row"]
        latest_date = pd.Timestamp(arr["dates"][-1])
        confidence = compute_confidence(int(event_counts.get(cell_id, 0)))
        evidence = build_evidence(latest_row, latest_mc)

        output = AgentOutput(
            agent=AGENT_NAME,
            region_id=cell_id,
            timestamp=now,
            hazard=HAZARD,
            risk_score=risk_score,
            confidence=confidence,
            evidence=evidence,
            data_window=DataWindow(start=(latest_date - timedelta(days=seq_len - 1)).date(), end=latest_date.date()),
        )
        outputs.append(output)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([o.model_dump(mode="json") for o in outputs], f, indent=2)
    logger.info("Wrote %d forecasts to %s", len(outputs), out_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the daily seismic forecast for Nepal.")
    parser.add_argument("--skip-fetch", action="store_true", help="Reuse the existing catalog instead of fetching new events.")
    parser.add_argument("--fetch-days", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_FORECAST_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run(checkpoint_path=args.checkpoint, out_path=args.out, skip_fetch=args.skip_fetch, fetch_days=args.fetch_days)


if __name__ == "__main__":
    main()
