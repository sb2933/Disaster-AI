"""Train and evaluate the seismic forecasting model, Baseline A and ETAS.

Time-based split (never shuffled across time): train on 1975-2014,
validate on 2015-2019 (which happens to include the 2015 Gorkha
sequence -- an out-of-training-sample stress test), test on 2020-today.

Run: python -m agents.seismic.train [--quick]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve, precision_recall_curve
from torch.utils.data import DataLoader, Dataset

from agents.seismic.baselines import (
    EtasParams,
    build_cell_event_index,
    fit_baseline_a,
    fit_etas,
    predict_baseline_a,
    predict_etas,
)
from agents.seismic.features import (
    FEATURE_COLUMNS,
    MAXMAG_WINDOW_DAYS,
    assign_grid_cells,
    build_transformer,
    load_catalog,
)
from agents.seismic.model import FocalLoss, SeismicForecastModel

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Config -- every tunable setting for this pipeline lives here.
# --------------------------------------------------------------------- #
CONFIG = {
    "features_path": "data/processed/features.parquet",
    "labels_path": "data/processed/labels.parquet",
    "raw_catalog_path": "data/raw/usgs_catalog.csv",
    "checkpoint_path": "models/seismic_checkpoint.pt",
    "metrics_path": "reports/metrics.json",
    "reports_dir": "reports",
    # Grid / forecast definition (must match features.py / labels.py)
    "cell_size_km": 50,
    "horizon_days": 7,
    # Time-based split -- never shuffled across time.
    "train_start": "1975-01-01", "train_end": "2014-12-31",
    "val_start": "2015-01-01", "val_end": "2019-12-31",
    "test_start": "2020-01-01", "test_end": "2026-10-03",
    "random_seed": 42,
    # Sequence model
    "seq_len": 60,
    "lstm_hidden": 64,
    "lstm_layers": 1,
    "transformer_heads": 4,
    "transformer_layers": 1,
    "transformer_ff": 128,
    "dropout": 0.1,
    # Training
    "undersample_negative_ratio": 10,  # negatives:positives kept in TRAIN split only
    "batch_size": 256,
    "lr": 1e-3,
    "max_epochs": 30,
    "early_stopping_patience": 5,
    "loss": "focal",  # "focal" or "weighted_bce"
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    # Evaluation
    "target_recall_for_threshold": 0.8,  # per-model operating threshold chosen on validation
    "label_min_mag": 5.0,
    "n_bootstrap": 100,  # block-bootstrap resamples (by calendar month) for ROC/PR-AUC CIs;
    # kept modest since the effective resampling unit is ~70 test-period months, not
    # millions of rows -- precision gains beyond ~100-200 resamples are marginal
    "alarm_budgets": (0.005, 0.01, 0.05),  # fixed-alarm-budget table: fraction of cell-days alarmed
    # Gorkha 2015 case study
    "gorkha_lat": 28.23,
    "gorkha_lon": 84.73,
    "gorkha_start": "2015-01-01",
    "gorkha_end": "2015-06-30",
}

FEATURE_FILL_VALUES = {
    "count_1d": 0.0,
    "count_7d": 0.0,
    "count_30d": 0.0,
    "background_rate": 0.0,  # no prior history to estimate a baseline from
    "relative_rate_7d": 0.0,  # can't compute a ratio without history -> "no anomaly"
    "max_mag_30d": 0.0,  # no event recorded in the window
    "days_since_max_mag_30d": float(MAXMAG_WINDOW_DAYS),  # "at least this many days" sentinel
    "energy_log10_30d": 0.0,  # no energy released; sentinel, well below any real value (~9+)
    "b_value": 1.0,  # neutral/typical tectonic default; b_value_valid flags when this is a guess
    "neighbor_count_7d": 0.0,
    "neighbor_max_mag_30d": 0.0,
    "etas_prob": 0.0,  # only present in the ETAS-augmented experiment; never actually NaN, defensive fill
}
# Right-skewed count/rate columns are log1p'd before z-scoring.
LOG1P_COLUMNS = ["count_1d", "count_7d", "count_30d", "neighbor_count_7d", "relative_rate_7d", "background_rate"]
# b_value_valid is a 0/1 flag, left unscaled.
NO_SCALE_COLUMNS = ["b_value_valid"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------- #
# Data loading, imputation, scaling
# --------------------------------------------------------------------- #


def load_merged_dataset(config: dict) -> pd.DataFrame:
    features = pd.read_parquet(config["features_path"])
    labels = pd.read_parquet(config["labels_path"])
    merged = features.merge(labels, on=["cell_id", "date"], how="inner")
    merged = merged[merged["label"].notna()].reset_index(drop=True)
    return merged


def fit_scaler(train_df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS) -> dict:
    """Fit log1p+z-score stats on the TRAIN split only (never on val/test).

    `feature_columns` lets a caller extend the standard feature set (e.g. the
    ETAS-probability-augmented experiment in run()) without touching the
    canonical FEATURE_COLUMNS used by features.py/labels.py/baselines.py.
    """
    x = train_df[feature_columns].fillna(FEATURE_FILL_VALUES).copy()
    for col in LOG1P_COLUMNS:
        if col in x.columns:
            x[col] = np.log1p(x[col].clip(lower=0))
    scale_cols = [c for c in feature_columns if c not in NO_SCALE_COLUMNS]
    mean = x[scale_cols].mean()
    std = x[scale_cols].std().replace(0, 1.0)
    return {"mean": mean.to_dict(), "std": std.to_dict(), "scale_cols": scale_cols, "feature_columns": feature_columns}


def transform_features(df: pd.DataFrame, scaler: dict) -> np.ndarray:
    feature_columns = scaler.get("feature_columns", FEATURE_COLUMNS)
    x = df[feature_columns].fillna(FEATURE_FILL_VALUES).copy()
    for col in LOG1P_COLUMNS:
        if col in x.columns:
            x[col] = np.log1p(x[col].clip(lower=0))
    for col in scaler["scale_cols"]:
        x[col] = (x[col] - scaler["mean"][col]) / scaler["std"][col]
    return x[feature_columns].to_numpy(dtype=np.float32)


def build_cell_arrays(df: pd.DataFrame, scaler: dict) -> dict[str, dict]:
    """One contiguous, date-sorted (dates, scaled features, labels) array per cell.

    Every active cell has a row for every day in the full catalog date
    range (see features.py), so these arrays are gap-free and index `i`
    always means "i days after this cell's first row".
    """
    df = df.sort_values(["cell_id", "date"]).reset_index(drop=True)
    feature_matrix = transform_features(df, scaler)
    arrays: dict[str, dict] = {}
    for cell_id, group in df.groupby("cell_id", sort=False):
        idx = group.index.to_numpy()
        arrays[cell_id] = {
            "dates": group["date"].to_numpy(),
            "features": feature_matrix[idx],
            "labels": group["label"].to_numpy(dtype=np.float32),
        }
    return arrays


def build_sample_index(cell_arrays: dict, start_date: str, end_date: str, seq_len: int) -> tuple[list, np.ndarray, np.ndarray]:
    """All (cell_id, row_idx) pairs with a full `seq_len`-day lookback, for forecast
    days within [start_date, end_date]. Returns (cell_ids, row_idxs, labels)."""
    start, end = np.datetime64(start_date), np.datetime64(end_date)
    cell_ids: list[str] = []
    row_idxs_parts, labels_parts = [], []
    for cell_id, arr in cell_arrays.items():
        dates = arr["dates"]
        in_range = np.where((dates >= start) & (dates <= end))[0]
        valid = in_range[in_range >= seq_len - 1]
        if len(valid) == 0:
            continue
        cell_ids.extend([cell_id] * len(valid))
        row_idxs_parts.append(valid)
        labels_parts.append(arr["labels"][valid])
    row_idxs = np.concatenate(row_idxs_parts) if row_idxs_parts else np.array([], dtype=int)
    labels = np.concatenate(labels_parts) if labels_parts else np.array([], dtype=np.float32)
    return cell_ids, row_idxs, labels


def undersample_quiet_days(cell_ids: list, row_idxs: np.ndarray, labels: np.ndarray, ratio: float, seed: int) -> tuple[list, np.ndarray, np.ndarray]:
    """Keep all positives, randomly subsample negatives to `ratio`:1. TRAIN split only."""
    rng = np.random.default_rng(seed)
    pos_mask = labels == 1
    pos_positions = np.where(pos_mask)[0]
    neg_positions = np.where(~pos_mask)[0]
    n_keep_neg = min(len(neg_positions), int(len(pos_positions) * ratio))
    kept_neg = rng.choice(neg_positions, size=n_keep_neg, replace=False)
    keep = np.concatenate([pos_positions, kept_neg])
    rng.shuffle(keep)
    kept_cell_ids = [cell_ids[i] for i in keep]
    return kept_cell_ids, row_idxs[keep], labels[keep]


class CellSequenceDataset(Dataset):
    def __init__(self, cell_arrays: dict, cell_ids: list, row_idxs: np.ndarray, seq_len: int):
        self.cell_arrays = cell_arrays
        self.cell_ids = cell_ids
        self.row_idxs = row_idxs
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.cell_ids)

    def __getitem__(self, i: int):
        cell_id = self.cell_ids[i]
        row_idx = int(self.row_idxs[i])
        arr = self.cell_arrays[cell_id]
        seq = arr["features"][row_idx - self.seq_len + 1 : row_idx + 1]
        label = arr["labels"][row_idx]
        return torch.from_numpy(seq.copy()), torch.tensor(label, dtype=torch.float32)


# --------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------- #


def train_model(config: dict, cell_arrays: dict) -> tuple[SeismicForecastModel, dict]:
    seq_len = config["seq_len"]
    train_ids, train_rows, train_labels = build_sample_index(cell_arrays, config["train_start"], config["train_end"], seq_len)
    train_ids, train_rows, train_labels = undersample_quiet_days(
        train_ids, train_rows, train_labels, config["undersample_negative_ratio"], config["random_seed"]
    )
    val_ids, val_rows, val_labels = build_sample_index(cell_arrays, config["val_start"], config["val_end"], seq_len)
    logger.info(
        "Train (undersampled): %d rows (%d positive). Val: %d rows (%d positive).",
        len(train_ids), int(train_labels.sum()), len(val_ids), int(val_labels.sum()),
    )

    train_loader = DataLoader(
        CellSequenceDataset(cell_arrays, train_ids, train_rows, seq_len),
        batch_size=config["batch_size"], shuffle=True, num_workers=0,
    )
    val_dataset = CellSequenceDataset(cell_arrays, val_ids, val_rows, seq_len)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    n_features = next(iter(cell_arrays.values()))["features"].shape[1]
    model = SeismicForecastModel(
        n_features=n_features, seq_len=seq_len,
        lstm_hidden=config["lstm_hidden"], lstm_layers=config["lstm_layers"],
        transformer_heads=config["transformer_heads"], transformer_layers=config["transformer_layers"],
        transformer_ff=config["transformer_ff"], dropout=config["dropout"],
    )
    if config["loss"] == "focal":
        loss_fn = FocalLoss(alpha=config["focal_alpha"], gamma=config["focal_gamma"])
    else:
        pos_weight = (train_labels == 0).sum() / max((train_labels == 1).sum(), 1)
        from agents.seismic.model import WeightedBCELoss
        loss_fn = WeightedBCELoss(positive_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_pr_auc = -np.inf
    best_state = None
    patience_left = config["early_stopping_patience"]

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y)
        epoch_loss /= max(len(train_ids), 1)

        val_probs = _predict_model(model, val_loader)
        val_pr_auc = average_precision_score(val_labels, val_probs) if val_labels.sum() > 0 else float("nan")
        logger.info("Epoch %d: train_loss=%.5f val_pr_auc=%.5f", epoch, epoch_loss, val_pr_auc)

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = config["early_stopping_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping at epoch %d (best val_pr_auc=%.5f)", epoch, best_pr_auc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_probs = _predict_model(model, val_loader)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_probs, val_labels)

    return model, {
        "calibrator": calibrator,
        "val_probs_raw": val_probs, "val_labels": val_labels,
        "best_val_pr_auc": best_pr_auc,
    }


@torch.no_grad()
def _predict_model(model: SeismicForecastModel, loader: DataLoader) -> np.ndarray:
    model.eval()
    outputs = []
    for x, _ in loader:
        outputs.append(torch.sigmoid(model(x)).numpy())
    return np.concatenate(outputs) if outputs else np.array([])


def predict_model_for_index(model: SeismicForecastModel, cell_arrays: dict, cell_ids: list, row_idxs: np.ndarray, seq_len: int, batch_size: int) -> np.ndarray:
    loader = DataLoader(CellSequenceDataset(cell_arrays, cell_ids, row_idxs, seq_len), batch_size=batch_size, shuffle=False, num_workers=0)
    return _predict_model(model, loader)


def dates_for_index(cell_arrays: dict, cell_ids: list, row_idxs: np.ndarray) -> np.ndarray:
    """The forecast-day date for each (cell_id, row_idx) pair in a sample index."""
    cell_ids_arr = np.asarray(cell_ids)
    out = np.empty(len(cell_ids), dtype="datetime64[ns]")
    for cid in set(cell_ids):
        mask = cell_ids_arr == cid
        out[mask] = cell_arrays[cid]["dates"][row_idxs[mask]]
    return out


# --------------------------------------------------------------------- #
# Metrics, threshold selection, plots
# --------------------------------------------------------------------- #


def find_threshold_for_recall(val_probs: np.ndarray, val_labels: np.ndarray, target_recall: float) -> float:
    """Smallest probability threshold whose alarms still capture >= target_recall of
    validation positives. Chosen once per model on validation, then frozen for test."""
    order = np.argsort(-val_probs)
    sorted_labels = val_labels[order]
    sorted_probs = val_probs[order]
    n_pos = sorted_labels.sum()
    if n_pos == 0:
        return float(np.median(val_probs))
    cum_hits = np.cumsum(sorted_labels)
    recall = cum_hits / n_pos
    idx = np.searchsorted(recall, target_recall)
    idx = min(idx, len(sorted_probs) - 1)
    return float(sorted_probs[idx])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    alarmed = y_prob >= threshold
    alarm_rate = float(alarmed.mean())
    n_pos = y_true.sum()
    hit_rate = float(y_true[alarmed].sum() / n_pos) if n_pos > 0 else float("nan")
    probability_gain = hit_rate / alarm_rate if alarm_rate > 0 else float("nan")
    negatives = y_true == 0
    false_alarm_rate = float(alarmed[negatives].mean()) if negatives.sum() > 0 else float("nan")

    return {
        "roc_auc": float(roc_auc), "pr_auc": float(pr_auc), "brier_score": float(brier),
        "threshold": float(threshold), "alarm_rate": alarm_rate, "hit_rate": hit_rate,
        "probability_gain": float(probability_gain), "false_alarm_rate": false_alarm_rate,
        "n_samples": int(len(y_true)), "n_positive": int(n_pos),
    }


def bootstrap_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    dates: np.ndarray,
    metric_fns: dict[str, "callable"],
    n_bootstrap: int = 100,
    seed: int = 0,
) -> dict[str, tuple[float, float, float]]:
    """Block-bootstrap 95% CIs for one or more metrics at once, resampling whole
    CALENDAR MONTHS with replacement (one shared set of resampled datasets, reused
    for every metric in `metric_fns`, so a single pass covers e.g. both ROC-AUC
    and PR-AUC).

    Earthquake occurrence is heavily clustered in time (aftershock sequences),
    so individual cell-days are not independent -- resampling single rows
    (the usual bootstrap) would understate uncertainty by treating correlated
    days as independent evidence. Resampling whole months keeps each
    sequence's internal correlation intact while still letting the overall
    mix of quiet/active months vary across resamples.

    Returns {name: (point_estimate, ci_lower_2.5pct, ci_upper_97.5pct)}. A
    bootstrap draw where only one class is present (e.g. zero positive months
    sampled) is skipped for all metrics, since ROC-AUC/PR-AUC are undefined
    there.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    points = {name: float(fn(y_true, y_prob)) for name, fn in metric_fns.items()}

    months = pd.PeriodIndex(pd.DatetimeIndex(dates), freq="M")
    month_to_idx: dict = {}
    for i, m in enumerate(months):
        month_to_idx.setdefault(m, []).append(i)
    month_idx_arrays = [np.array(v) for v in month_to_idx.values()]
    n_months = len(month_idx_arrays)

    rng = np.random.default_rng(seed)
    estimates: dict[str, list[float]] = {name: [] for name in metric_fns}
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, n_months, size=n_months)
        idx = np.concatenate([month_idx_arrays[j] for j in sampled])
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        for name, fn in metric_fns.items():
            try:
                estimates[name].append(float(fn(yt, yp)))
            except ValueError:
                continue

    results = {}
    for name, point in points.items():
        vals = estimates[name]
        if not vals:
            results[name] = (point, float("nan"), float("nan"))
        else:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            results[name] = (point, float(lo), float(hi))
    return results


def fixed_alarm_budget_table(
    y_true: np.ndarray, y_prob: np.ndarray, budgets: tuple[float, ...] = (0.005, 0.01, 0.05)
) -> list[dict]:
    """For each alert budget (fraction of cell-days the agent is allowed to alarm on),
    the threshold that spends exactly that budget on the TEST set and the fraction of
    actual M5+ events caught by alarming at/above it. Ties in y_prob (e.g. many
    identical zero-probability days) can make the actual alarm rate overshoot the
    requested budget -- actual_alarm_rate is reported alongside budget for that reason.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_pos = int(y_true.sum())
    rows = []
    for budget in budgets:
        threshold = float(np.quantile(y_prob, 1 - budget))
        alarmed = y_prob >= threshold
        caught = int(y_true[alarmed].sum())
        rows.append(
            {
                "budget": budget,
                "threshold": threshold,
                "actual_alarm_rate": float(alarmed.mean()),
                "events_caught": caught,
                "events_total": n_pos,
                "fraction_caught": caught / n_pos if n_pos > 0 else float("nan"),
            }
        )
    return rows


def molchan_curve(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sweep every threshold: (fraction of space-time alarmed, fraction of events missed)."""
    order = np.argsort(-y_prob)
    y_sorted = np.asarray(y_true)[order]
    n = len(y_sorted)
    n_pos = y_sorted.sum()
    alarm_rate = np.arange(1, n + 1) / n
    if n_pos == 0:
        return alarm_rate, np.full(n, np.nan)
    miss_rate = 1 - np.cumsum(y_sorted) / n_pos
    return alarm_rate, miss_rate


def plot_comparison(results: dict[str, dict], reports_dir: Path) -> None:
    """results[name] = {"y_true":..., "y_prob":..., "threshold":...}"""
    reports_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curve -- M5+/7-day forecast"); plt.legend(); plt.tight_layout()
    plt.savefig(reports_dir / "roc_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 6))
    for name, r in results.items():
        precision, recall, _ = precision_recall_curve(r["y_true"], r["y_prob"])
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall curve -- M5+/7-day forecast"); plt.legend(); plt.tight_layout()
    plt.savefig(reports_dir / "pr_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 6))
    for name, r in results.items():
        try:
            frac_pos, mean_pred = calibration_curve(r["y_true"], r["y_prob"], n_bins=10, strategy="quantile")
            plt.plot(mean_pred, frac_pos, marker="o", label=name)
        except ValueError:
            logger.warning("Skipping calibration curve for %s (degenerate bins).", name)
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfectly calibrated")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration (reliability) curve"); plt.legend(); plt.tight_layout()
    plt.savefig(reports_dir / "calibration_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 6))
    for name, r in results.items():
        alarm_rate, miss_rate = molchan_curve(r["y_true"], r["y_prob"])
        plt.plot(alarm_rate, miss_rate, label=name)
    plt.plot([0, 1], [1, 0], "k--", linewidth=0.8, label="random")
    plt.xlabel("Fraction of space-time alarmed"); plt.ylabel("Miss rate")
    plt.title("Molchan diagram"); plt.legend(); plt.tight_layout()
    plt.savefig(reports_dir / "molchan_diagram.png", dpi=150); plt.close()


def plot_gorkha_case_study(dates: np.ndarray, series: dict[str, np.ndarray], mainshock_date: str, reports_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    for name, probs in series.items():
        plt.plot(dates, probs, label=name)
    plt.axvline(np.datetime64(mainshock_date), color="red", linestyle="--", label="Gorkha mainshock (M7.8)")
    plt.xlabel("Date"); plt.ylabel("P(M5+ in next 7 days)")
    plt.title("Gorkha 2015 case study -- epicentre cell, using only data available each day")
    plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
    reports_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(reports_dir / "gorkha_case_study.png", dpi=150); plt.close()


# --------------------------------------------------------------------- #
# ETAS prediction over a dataframe slice (helper, orchestration-specific)
# --------------------------------------------------------------------- #


def predict_etas_for_df(
    df: pd.DataFrame, cell_event_index: dict, params: EtasParams, t0: pd.Timestamp, horizon: int
) -> pd.Series:
    if len(df) == 0:
        return pd.Series(dtype=float)
    parts = []
    for cell_id, group in df.groupby("cell_id"):
        event_days, event_mags = cell_event_index.get(cell_id, (np.array([]), np.array([])))
        forecast_days = (group["date"] - t0).dt.days.to_numpy(dtype=float)
        preds = predict_etas(params, cell_id, event_days, event_mags, forecast_days, horizon=horizon)
        parts.append(pd.Series(preds, index=group.index))
    return pd.concat(parts).reindex(df.index)


# --------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------- #


def run(config: dict, extra_feature_columns: list[str] | None = None) -> dict:
    set_seed(config["random_seed"])
    reports_dir = Path(config["reports_dir"])
    horizon = config["horizon_days"]

    logger.info("Loading merged features+labels dataset...")
    merged = load_merged_dataset(config)
    date_mask = lambda start, end: (merged["date"] >= start) & (merged["date"] <= end)  # noqa: E731
    train_df = merged[date_mask(config["train_start"], config["train_end"])]
    val_df = merged[date_mask(config["val_start"], config["val_end"])]
    test_df = merged[date_mask(config["test_start"], config["test_end"])]
    logger.info("train=%d val=%d test=%d rows", len(train_df), len(val_df), len(test_df))

    # --- Baseline A --- (independent of ETAS/the extra-feature experiment)
    logger.info("Fitting Baseline A (logistic regression)...")
    baseline_a_model = fit_baseline_a(train_df)
    baseline_a_val_probs = predict_baseline_a(baseline_a_model, val_df)
    baseline_a_test_probs = predict_baseline_a(baseline_a_model, test_df)

    # --- Baseline B: ETAS --- (fit on train period only, see fit_etas)
    logger.info("Fitting Baseline B (ETAS)...")
    catalog = load_catalog(Path(config["raw_catalog_path"]))
    assigned = assign_grid_cells(catalog, build_transformer())
    train_events = assigned[date_mask_catalog(assigned, config["train_start"], config["train_end"])][["cell_id", "date", "mag"]]
    etas_params = fit_etas(train_events)
    t0 = train_events["date"].min()
    cell_event_index = build_cell_event_index(assigned[["cell_id", "date", "mag"]], etas_params.mc, t0)
    etas_val_probs = predict_etas_for_df(val_df, cell_event_index, etas_params, t0, horizon).to_numpy()
    etas_test_probs = predict_etas_for_df(test_df, cell_event_index, etas_params, t0, horizon).to_numpy()

    # --- Optional experiment: add the ETAS forecast probability as a model input feature ---
    # Uses the SAME frozen (train-period-only) etas_params and the causally-growing
    # cell_event_index as the ETAS baseline above -- so this is exactly "what ETAS
    # would have forecast, as of that day", not a leaked/omniscient value.
    if extra_feature_columns and "etas_prob" in extra_feature_columns:
        logger.info("Computing ETAS probability feature for all %d rows (ETAS-augmented experiment)...", len(merged))
        merged = merged.copy()
        merged["etas_prob"] = predict_etas_for_df(merged, cell_event_index, etas_params, t0, horizon).to_numpy()
        train_df = merged[date_mask(config["train_start"], config["train_end"])]
        val_df = merged[date_mask(config["val_start"], config["val_end"])]
        test_df = merged[date_mask(config["test_start"], config["test_end"])]

    feature_columns = FEATURE_COLUMNS + [c for c in (extra_feature_columns or []) if c not in FEATURE_COLUMNS]
    scaler = fit_scaler(train_df, feature_columns)
    cell_arrays = build_cell_arrays(merged, scaler)

    # --- Deep model ---
    logger.info("Training the LSTM+Transformer model (%d input features)...", len(feature_columns))
    model, train_extra = train_model(config, cell_arrays)
    calibrator = train_extra["calibrator"]
    model_val_probs = calibrator.transform(train_extra["val_probs_raw"])
    model_val_labels = train_extra["val_labels"]

    test_ids, test_rows, test_labels = build_sample_index(cell_arrays, config["test_start"], config["test_end"], config["seq_len"])
    model_test_probs_raw = predict_model_for_index(model, cell_arrays, test_ids, test_rows, config["seq_len"], config["batch_size"])
    model_test_probs = calibrator.transform(model_test_probs_raw)
    model_test_dates = dates_for_index(cell_arrays, test_ids, test_rows)

    # --- Thresholds (chosen on validation, frozen for test), metrics, bootstrap CIs, alarm table ---
    results = {}
    for name, val_probs, val_labels, test_probs, test_labels_, test_dates_ in [
        ("baseline_a", baseline_a_val_probs, val_df["label"].to_numpy(), baseline_a_test_probs, test_df["label"].to_numpy(), test_df["date"].to_numpy()),
        ("etas", etas_val_probs, val_df["label"].to_numpy(), etas_test_probs, test_df["label"].to_numpy(), test_df["date"].to_numpy()),
        ("model", model_val_probs, model_val_labels, model_test_probs, test_labels, model_test_dates),
    ]:
        threshold = find_threshold_for_recall(val_probs, val_labels, config["target_recall_for_threshold"])
        metrics = compute_metrics(test_labels_, test_probs, threshold)

        cis = bootstrap_cis(
            test_labels_, test_probs, test_dates_,
            {"roc_auc": roc_auc_score, "pr_auc": average_precision_score},
            n_bootstrap=config["n_bootstrap"], seed=config["random_seed"],
        )
        metrics["roc_auc_ci95"] = [cis["roc_auc"][1], cis["roc_auc"][2]]
        metrics["pr_auc_ci95"] = [cis["pr_auc"][1], cis["pr_auc"][2]]
        metrics["alarm_budget_table"] = fixed_alarm_budget_table(test_labels_, test_probs, budgets=config["alarm_budgets"])

        results[name] = {"metrics": metrics, "y_true": test_labels_, "y_prob": test_probs}
        logger.info("%s test metrics: %s", name, metrics)

    metrics_out = {name: r["metrics"] for name, r in results.items()}
    Path(config["metrics_path"]).parent.mkdir(parents=True, exist_ok=True)
    with open(config["metrics_path"], "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info("Wrote metrics to %s", config["metrics_path"])

    plot_comparison({name: {"y_true": r["y_true"], "y_prob": r["y_prob"]} for name, r in results.items()}, reports_dir)

    # --- Gorkha 2015 case study ---
    logger.info("Running Gorkha 2015 case study...")
    point_df = pd.DataFrame({"lat": [config["gorkha_lat"]], "lon": [config["gorkha_lon"]]})
    epicenter_cell = assign_grid_cells(point_df, build_transformer())["cell_id"].iloc[0]
    gorkha_dates = pd.date_range(config["gorkha_start"], config["gorkha_end"], freq="D")

    if epicenter_cell in cell_arrays:
        cell_dates = cell_arrays[epicenter_cell]["dates"]
        row_idxs = np.searchsorted(cell_dates, gorkha_dates.to_numpy())
        valid = (row_idxs < len(cell_dates)) & (cell_dates[np.clip(row_idxs, 0, len(cell_dates) - 1)] == gorkha_dates.to_numpy())
        valid &= row_idxs >= config["seq_len"] - 1
        row_idxs = row_idxs[valid]
        plot_dates = gorkha_dates[valid]

        model_probs = calibrator.transform(
            predict_model_for_index(model, cell_arrays, [epicenter_cell] * len(row_idxs), row_idxs, config["seq_len"], config["batch_size"])
        )
        gorkha_df = merged[(merged["cell_id"] == epicenter_cell) & merged["date"].isin(plot_dates)].sort_values("date")
        baseline_a_gorkha = predict_baseline_a(baseline_a_model, gorkha_df)
        event_days_g, event_mags_g = cell_event_index.get(epicenter_cell, (np.array([]), np.array([])))
        forecast_days_g = (plot_dates - t0).to_numpy().astype("timedelta64[D]").astype(float)
        etas_gorkha = predict_etas(etas_params, epicenter_cell, event_days_g, event_mags_g, forecast_days_g, horizon)

        plot_gorkha_case_study(
            plot_dates.to_numpy(),
            {"model": model_probs, "baseline_a": baseline_a_gorkha, "etas": etas_gorkha},
            "2015-04-25",
            reports_dir,
        )
        logger.info("Gorkha epicentre cell: %s", epicenter_cell)
    else:
        logger.warning("Epicentre cell %s not in active cell set; skipping Gorkha case study plot.", epicenter_cell)

    # --- Checkpoint ---
    checkpoint_path = Path(config["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "calibrator": calibrator,
            "config": config,
            "feature_columns": feature_columns,
            "baseline_a_model": baseline_a_model,
            "etas_params": etas_params,
            "etas_t0": t0,
        },
        checkpoint_path,
    )
    logger.info("Saved checkpoint to %s", checkpoint_path)

    return metrics_out


def date_mask_catalog(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    return (df["date"] >= start) & (df["date"] <= end)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the seismic forecasting model.")
    parser.add_argument("--quick", action="store_true", help="Fast correctness check: 2 epochs, patience 1.")
    parser.add_argument(
        "--with-etas-feature", action="store_true",
        help="Experiment: add the ETAS forecast probability as an extra model input feature. "
        "Writes to separate output paths so it never overwrites the original run's artifacts.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = dict(CONFIG)
    if args.quick:
        config["max_epochs"] = 2
        config["early_stopping_patience"] = 1

    extra_feature_columns = None
    if args.with_etas_feature:
        extra_feature_columns = ["etas_prob"]
        config["metrics_path"] = "reports/metrics_etas_augmented.json"
        config["checkpoint_path"] = "models/seismic_checkpoint_etas_augmented.pt"
        config["reports_dir"] = "reports/etas_augmented"

    run(config, extra_feature_columns=extra_feature_columns)


if __name__ == "__main__":
    main()
