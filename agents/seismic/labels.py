"""Forecast labels for the seismic agent.

label = 1 if an event with magnitude >= LABEL_MIN_MAG occurs in the SAME
cell during days t+1..t+7 inclusive (relative to forecast day t); 0 if no
such event occurs; NaN if the catalog does not extend far enough into the
future to know the answer (i.e. date t+7 is after the last day in the
catalog) -- those rows are "unknown", not "no event", and must not be
treated as negatives during training.

Uses the exact same grid/active-cell definition as features.py (via
`prepare_grid`) so labels line up row-for-row with features on
(cell_id, date).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from agents.seismic.features import DEFAULT_RAW_PATH, load_catalog, prepare_grid

logger = logging.getLogger(__name__)

LABEL_MIN_MAG = 5.0
LABEL_HORIZON_DAYS = 7

DEFAULT_LABELS_PATH = Path("data/processed/labels.parquet")


def _labels_for_cell(
    event_dates: np.ndarray, date_range: pd.DatetimeIndex, catalog_max_date: pd.Timestamp
) -> np.ndarray:
    window_start = date_range + pd.Timedelta(days=1)
    window_end = date_range + pd.Timedelta(days=LABEL_HORIZON_DAYS)
    has_future_data = window_end.values <= np.datetime64(catalog_max_date)

    if len(event_dates):
        lo = np.searchsorted(event_dates, window_start.values, side="left")
        hi = np.searchsorted(event_dates, window_end.values, side="right")
        counts = hi - lo
    else:
        counts = np.zeros(len(date_range), dtype=int)

    return np.where(has_future_data, (counts > 0).astype(float), np.nan)


def build_labels(
    raw_catalog_path: Path = DEFAULT_RAW_PATH,
    out_path: Path = DEFAULT_LABELS_PATH,
    active_cell_cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute labels for every (cell_id, date) in the active grid and write to parquet.

    `active_cell_cutoff` must match whatever was passed to build_features --
    see features.select_active_cells -- so labels and features cover the
    exact same set of cells.
    """
    catalog = load_catalog(raw_catalog_path)
    assigned, active_cells, date_range, _ = prepare_grid(catalog, active_cell_cutoff=active_cell_cutoff)
    catalog_max_date = date_range.max()

    strong = assigned[assigned["mag"] >= LABEL_MIN_MAG]
    strong_by_cell = {
        cell_id: np.sort(group["date"].to_numpy()) for cell_id, group in strong.groupby("cell_id")
    }
    logger.info(
        "%d cells, %d M%.1f+ events in the catalog overall", len(active_cells), len(strong), LABEL_MIN_MAG
    )

    rows = []
    for cell_id in sorted(active_cells):
        event_dates = strong_by_cell.get(cell_id, np.array([], dtype="datetime64[ns]"))
        labels = _labels_for_cell(event_dates, date_range, catalog_max_date)
        rows.append(pd.DataFrame({"cell_id": cell_id, "date": date_range, "label": labels}))

    labels_df = pd.concat(rows, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(out_path, index=False)

    n_known = labels_df["label"].notna().sum()
    n_positive = (labels_df["label"] == 1).sum()
    logger.info(
        "Wrote %d label rows to %s (%d known, %d positive, %.4f%% positive rate)",
        len(labels_df), out_path, n_known, n_positive, 100 * n_positive / max(n_known, 1),
    )
    return labels_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build M5+/7-day forecast labels.")
    parser.add_argument("--raw-catalog", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--active-cell-cutoff", type=str, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cutoff = pd.Timestamp(args.active_cell_cutoff) if args.active_cell_cutoff else None
    build_labels(args.raw_catalog, args.out, active_cell_cutoff=cutoff)


if __name__ == "__main__":
    main()
