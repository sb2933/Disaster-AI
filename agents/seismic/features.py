"""Daily per-cell seismicity features for the Himalaya/Nepal forecasting grid.

Pipeline (see `build_features` for the full orchestration):
  1. Project events onto a 50x50 km grid using an equal-area projection
     (see PROJ_STRING below for the choice).
  2. Estimate magnitude of completeness (Mc) per 5-year period via the
     maximum-curvature method with the standard MAXC+0.2 correction
     (Woessner & Wiemer, 2005), and mark each event "complete" if its
     magnitude is >= the Mc of its own period.
  3. Keep "active" cells (>=5 events anywhere in the catalog) plus their
     8 Moore neighbours, to bound the size of the daily grid.
  4. For each active cell and each day, compute rolling seismicity
     features using only events known as of that day.

Causality
---------
A forecast "issued" at the end of day t uses all events with origin time
on or before day t. It predicts activity for days t+1..t+7 (see
labels.py). The one deliberate exception is the long-term background
rate, which uses only events strictly BEFORE day t -- so that a burst of
activity on day t itself cannot inflate its own baseline and mute the
very relative-rate spike it should be flagging.

Projection choice
-----------------
We use a Lambert Azimuthal Equal-Area (LAEA) projection centred on the
training region (lat 25-36N, lon 70-97E -> centre at 30.5N, 83.5E).
LAEA preserves area exactly at the centre and keeps area distortion
under ~1-2% across the full training+test extent, which is what matters
here: a "50x50 km cell" must mean approximately the same physical area
everywhere in the grid, or event-density features would be biased by
latitude/longitude.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

logger = logging.getLogger(__name__)

# --- Grid ---------------------------------------------------------------
PROJ_STRING = "+proj=laea +lat_0=30.5 +lon_0=83.5 +datum=WGS84 +units=m +no_defs"
CELL_SIZE_M = 50_000.0
ACTIVE_MIN_EVENTS = 5

# --- Magnitude of completeness ------------------------------------------
MC_EPOCH_YEAR = 1975
MC_PERIOD_YEARS = 5
MAXC_CORRECTION = 0.2
MAG_BIN_WIDTH = 0.1
MC_MIN_EVENTS_FOR_TRUST = 100  # below this, log a warning; estimate is still used

# --- Feature windows ------------------------------------------------------
COUNT_WINDOWS_DAYS = (1, 7, 30)
MAXMAG_WINDOW_DAYS = 30
ENERGY_WINDOW_DAYS = 30
BVALUE_WINDOW_DAYS = 365  # trailing 1-year window: balances recency against reaching BVALUE_MIN_EVENTS
BVALUE_MIN_EVENTS = 50
BVALUE_BIN_CORRECTION = 0.05  # half the 0.1-magnitude bin width, standard Aki MLE correction

DEFAULT_RAW_PATH = Path("data/raw/usgs_catalog.csv")
DEFAULT_FEATURES_PATH = Path("data/processed/features.parquet")

FEATURE_COLUMNS = [
    "count_1d",
    "count_7d",
    "count_30d",
    "background_rate",
    "relative_rate_7d",
    "max_mag_30d",
    "days_since_max_mag_30d",
    "energy_log10_30d",
    "b_value",
    "b_value_valid",
    "neighbor_count_7d",
    "neighbor_max_mag_30d",
]


def load_catalog(path: Path) -> pd.DataFrame:
    """Load a raw USGS catalog CSV and add a UTC-midnight `date` column."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df["date"] = df["time"].dt.tz_localize(None).dt.floor("D")
    return df


def build_transformer() -> Transformer:
    """Lon/lat (EPSG:4326) -> equal-area metres transformer, see PROJ_STRING."""
    return Transformer.from_crs("EPSG:4326", PROJ_STRING, always_xy=True)


def assign_grid_cells(df: pd.DataFrame, transformer: Transformer) -> pd.DataFrame:
    """Project each event and assign it a stable `cell_id` of the form 'r{row}_c{col}'."""
    df = df.copy()
    x_m, y_m = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["row"] = np.floor(y_m / CELL_SIZE_M).astype(int)
    df["col"] = np.floor(x_m / CELL_SIZE_M).astype(int)
    df["cell_id"] = "r" + df["row"].astype(str) + "_c" + df["col"].astype(str)
    return df


def parse_cell_id(cell_id: str) -> tuple[int, int]:
    row_part, col_part = cell_id.split("_")
    return int(row_part[1:]), int(col_part[1:])


def neighbour_ids(cell_id: str) -> list[str]:
    """The 8 Moore-neighbourhood cell ids surrounding `cell_id`."""
    row, col = parse_cell_id(cell_id)
    return [
        f"r{row + dr}_c{col + dc}"
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def estimate_mc_maxcurvature(mags: np.ndarray, bin_width: float = MAG_BIN_WIDTH) -> float:
    """Magnitude of completeness via maximum curvature (mode of the FMD), no correction applied."""
    if len(mags) == 0:
        return float("nan")
    binned = np.round(np.round(mags / bin_width) * bin_width, 2)
    values, counts = np.unique(binned, return_counts=True)
    return float(values[np.argmax(counts)])


def assign_completeness(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, float]]:
    """Estimate Mc per 5-year period (MAXC + 0.2) and flag each event as complete or not."""
    df = df.copy()
    df["period_start"] = MC_EPOCH_YEAR + (
        (df["date"].dt.year - MC_EPOCH_YEAR) // MC_PERIOD_YEARS
    ) * MC_PERIOD_YEARS

    mc_lookup: dict[int, float] = {}
    for period, group in df.groupby("period_start"):
        mc_maxc = estimate_mc_maxcurvature(group["mag"].to_numpy())
        mc = mc_maxc + MAXC_CORRECTION
        mc_lookup[int(period)] = mc
        if len(group) < MC_MIN_EVENTS_FOR_TRUST:
            logger.warning(
                "Period starting %s has only %d events; Mc estimate (%.2f) may be unreliable.",
                period, len(group), mc,
            )
        else:
            logger.info("Period starting %s: Mc=%.2f (n=%d)", period, mc, len(group))

    df["mc"] = df["period_start"].map(mc_lookup)
    df["is_complete"] = df["mag"] >= df["mc"]
    return df, mc_lookup


def select_active_cells(
    df_assigned: pd.DataFrame, min_events: int = ACTIVE_MIN_EVENTS, cutoff_date: pd.Timestamp | None = None
) -> set[str]:
    """Cells with >= min_events events, plus their 8 neighbours.

    `cutoff_date`: if given, only events with date <= cutoff_date count
    toward the >= min_events threshold. Pass the training period's end date
    here to define the grid using only information that would have been
    available at training time -- otherwise a cell that only became active
    due to a LATER (e.g. test-period) sequence gets included in the grid
    with the benefit of hindsight, which is a form of look-ahead bias in
    which cells exist at all (not in any one day's feature values).
    """
    counted = df_assigned if cutoff_date is None else df_assigned[df_assigned["date"] <= cutoff_date]
    counts = counted.groupby("cell_id").size()
    core = set(counts[counts >= min_events].index)
    expanded = set(core)
    for cell_id in core:
        expanded.update(neighbour_ids(cell_id))
    return expanded


def prepare_grid(
    catalog: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    active_cell_cutoff: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, set[str], pd.DatetimeIndex, dict[int, float]]:
    """Shared grid/active-cell/Mc setup, reused identically by features.py and labels.py.

    By default the date range runs through the catalog's LAST RECORDED EVENT,
    which is what historical training/labelling wants (there is no reason to
    manufacture days beyond the data you fetched). For a live daily forecast,
    pass `as_of=<today>` so quiet recent days with zero events still get a
    (zero-filled) row instead of the feature table silently stopping at
    whenever the last earthquake happened to occur -- otherwise a multi-day
    quiet spell makes the forecast look stale by that many days.

    `active_cell_cutoff`: see select_active_cells -- pass the training
    period's end date to define the grid without hindsight from val/test
    period seismicity.

    Returns (assigned_catalog, active_cell_ids, full_date_range, mc_lookup).
    """
    transformer = build_transformer()
    assigned = assign_grid_cells(catalog, transformer)
    assigned, mc_lookup = assign_completeness(assigned)
    active_cells = select_active_cells(assigned, cutoff_date=active_cell_cutoff)
    end_date = assigned["date"].max()
    if as_of is not None and as_of > end_date:
        end_date = as_of
    date_range = pd.date_range(assigned["date"].min(), end_date, freq="D")
    return assigned, active_cells, date_range, mc_lookup


def _mc_series_for_dates(date_range: pd.DatetimeIndex, mc_lookup: dict[int, float]) -> pd.Series:
    periods = MC_EPOCH_YEAR + ((date_range.year - MC_EPOCH_YEAR) // MC_PERIOD_YEARS) * MC_PERIOD_YEARS
    return pd.Series([mc_lookup.get(int(p), np.nan) for p in periods], index=date_range)


def _days_since_max(window: np.ndarray) -> float:
    """For a trailing window ending "today" (last element), days since the most recent max."""
    if np.all(np.isnan(window)):
        return np.nan
    reversed_window = window[::-1]  # index 0 = today, counting backwards
    return float(np.nanargmax(reversed_window))


def _compute_cell_features(
    sub: pd.DataFrame, date_range: pd.DatetimeIndex, mc_by_date: pd.Series
) -> pd.DataFrame:
    """Build the daily (non-neighbour) feature columns for one cell's events."""
    if len(sub):
        daily_max_mag_all = sub.groupby("date")["mag"].max().reindex(date_range)
        complete = sub[sub["is_complete"]]
    else:
        daily_max_mag_all = pd.Series(np.nan, index=date_range)
        complete = sub

    if len(complete):
        daily_count_complete = complete.groupby("date").size().reindex(date_range, fill_value=0).astype(float)
        daily_magsum_complete = complete.groupby("date")["mag"].sum().reindex(date_range, fill_value=0.0)
        energy = 10.0 ** (1.5 * complete["mag"] + 4.8)
        daily_energy_complete = energy.groupby(complete["date"]).sum().reindex(date_range, fill_value=0.0)
    else:
        daily_count_complete = pd.Series(0.0, index=date_range)
        daily_magsum_complete = pd.Series(0.0, index=date_range)
        daily_energy_complete = pd.Series(0.0, index=date_range)

    count_1d = daily_count_complete
    count_7d = daily_count_complete.rolling(7, min_periods=1).sum()
    count_30d = daily_count_complete.rolling(30, min_periods=1).sum()

    # Background rate deliberately excludes day t itself (see module docstring).
    cumulative_before = daily_count_complete.cumsum().shift(1).fillna(0.0)
    days_elapsed_before = pd.Series(np.arange(len(date_range), dtype=float), index=date_range)
    background_rate = cumulative_before / days_elapsed_before.replace(0.0, np.nan)
    relative_rate_7d = (count_7d / 7.0) / background_rate.replace(0.0, np.nan)

    max_mag_30d = daily_max_mag_all.rolling(MAXMAG_WINDOW_DAYS, min_periods=1).max()
    days_since_max_mag_30d = daily_max_mag_all.rolling(
        MAXMAG_WINDOW_DAYS, min_periods=1
    ).apply(_days_since_max, raw=True)

    energy_sum_30d = daily_energy_complete.rolling(ENERGY_WINDOW_DAYS, min_periods=1).sum()
    energy_log10_30d = np.log10(energy_sum_30d.replace(0.0, np.nan))

    windowed_count_b = daily_count_complete.rolling(BVALUE_WINDOW_DAYS, min_periods=1).sum()
    windowed_magsum_b = daily_magsum_complete.rolling(BVALUE_WINDOW_DAYS, min_periods=1).sum()
    mean_mag_b = windowed_magsum_b / windowed_count_b.replace(0.0, np.nan)
    denom = mean_mag_b - (mc_by_date - BVALUE_BIN_CORRECTION)
    b_value_valid = (windowed_count_b >= BVALUE_MIN_EVENTS) & (denom > 0)
    b_value = (np.log10(np.e) / denom.where(denom > 0)).where(b_value_valid)

    return pd.DataFrame(
        {
            "count_1d": count_1d,
            "count_7d": count_7d,
            "count_30d": count_30d,
            "background_rate": background_rate,
            "relative_rate_7d": relative_rate_7d,
            "max_mag_30d": max_mag_30d,
            "days_since_max_mag_30d": days_since_max_mag_30d,
            "energy_log10_30d": energy_log10_30d,
            "b_value": b_value,
            "b_value_valid": b_value_valid.astype(int),
        },
        index=date_range,
    )


def build_features(
    raw_catalog_path: Path = DEFAULT_RAW_PATH,
    out_path: Path = DEFAULT_FEATURES_PATH,
    as_of: pd.Timestamp | None = None,
    active_cell_cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Full orchestration: load catalog -> grid -> per-cell daily features -> neighbour features -> parquet.

    `as_of`: see prepare_grid -- pass today's date for a live forecast so
    quiet recent days are represented, instead of the range stopping at the
    last recorded event's date.
    `active_cell_cutoff`: see select_active_cells -- pass the training
    period's end date to avoid hindsight in which cells are in the grid.
    """
    catalog = load_catalog(raw_catalog_path)
    assigned, active_cells, date_range, mc_lookup = prepare_grid(catalog, as_of=as_of, active_cell_cutoff=active_cell_cutoff)
    mc_by_date = _mc_series_for_dates(date_range, mc_lookup)
    logger.info("%d active cells, %d days in range", len(active_cells), len(date_range))

    grouped = assigned[assigned["cell_id"].isin(active_cells)].groupby("cell_id")
    cell_frames: dict[str, pd.DataFrame] = {}
    for cell_id in sorted(active_cells):
        sub = grouped.get_group(cell_id) if cell_id in grouped.groups else assigned.iloc[0:0]
        cell_frames[cell_id] = _compute_cell_features(sub, date_range, mc_by_date)

    result_frames = []
    for cell_id, feat in cell_frames.items():
        neighbours = [n for n in neighbour_ids(cell_id) if n in cell_frames]
        if neighbours:
            neighbor_count_7d = pd.concat(
                [cell_frames[n]["count_7d"] for n in neighbours], axis=1
            ).sum(axis=1)
            neighbor_max_mag_30d = pd.concat(
                [cell_frames[n]["max_mag_30d"] for n in neighbours], axis=1
            ).max(axis=1)
        else:
            neighbor_count_7d = pd.Series(0.0, index=date_range)
            neighbor_max_mag_30d = pd.Series(np.nan, index=date_range)

        feat = feat.copy()
        feat["neighbor_count_7d"] = neighbor_count_7d
        feat["neighbor_max_mag_30d"] = neighbor_max_mag_30d
        feat["cell_id"] = cell_id
        result_frames.append(feat)

    features = pd.concat(result_frames).rename_axis("date").reset_index()
    features = features[["cell_id", "date"] + FEATURE_COLUMNS]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    logger.info("Wrote %d feature rows to %s", len(features), out_path)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily per-cell seismicity features.")
    parser.add_argument("--raw-catalog", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--active-cell-cutoff", type=str, default=None, help="e.g. 2014-12-31 to define active cells using only data up to this date.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cutoff = pd.Timestamp(args.active_cell_cutoff) if args.active_cell_cutoff else None
    build_features(args.raw_catalog, args.out, active_cell_cutoff=cutoff)


if __name__ == "__main__":
    main()
