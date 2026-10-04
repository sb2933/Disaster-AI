"""Two simple baseline forecasters for the seismic agent.

Baseline A -- logistic regression
----------------------------------
A logistic regression on two of the daily per-cell features from
features.py: `relative_rate_7d` and `max_mag_30d`. NaNs are imputed before
fitting (see `_prepare_logistic_features`): a missing relative rate means
there was no prior history to compare against (treated as "no anomaly",
fill 0.0), and a missing 30-day max magnitude means no event was recorded
in that window (filled with 0.0, well below any real magnitude).

Baseline B -- simplified temporal ETAS
---------------------------------------
A single-region Epidemic-Type Aftershock Sequence (ETAS, Ogata 1988)
model, fit once on the training period, used to forecast each cell's
M5+ probability. This is deliberately simplified, and every
simplification is an explicit assumption stated here:

1. Temporal-only kernel, no explicit spatial kernel. The 5 kernel
   parameters (mu, K, alpha, c, p) are fit on the POOLED regional
   sequence of >=Mc events (all cells combined) -- per-cell catalogs are
   far too sparse to fit a 5-parameter kernel reliably. The fitted
   kernel SHAPE is then shared across all cells; only the background
   rate is allocated per cell (assumption 2), and at prediction time
   each cell's self-excitation term uses only that cell's OWN past
   events. This ignores cross-cell triggering (a real aftershock
   sequence spilling into neighbouring cells).
2. The fitted background rate mu is a single regional number, allocated
   to each cell in proportion to that cell's share of total training-
   period >=Mc events: mu_cell = mu * (cell_count / total_count). This
   assumes a temporally stationary, and simply-allocated, background.
3. The triggering kernel is the standard Omori-Utsu law
   g(dt) = (p-1)/c * (1 + dt/c)^-p (normalised to integrate to 1 on
   [0, inf)), with productivity kappa(m) = K * exp(alpha * (m - Mc)).
   Parameters are fit by maximum likelihood (Ogata 1988 temporal
   log-likelihood) via scipy.optimize, using only M>=Mc events.
4. A single GLOBAL Gutenberg-Richter b-value (Aki MLE, fit on the same
   training events) converts the model's rate of >=Mc events into a
   rate of >=5.0 events: lambda_M5(t) = lambda_Mc(t) * 10^(-b*(5-Mc)).
   This assumes GR holds with one stationary, spatially uniform b.
5. P(>=1 M5+ event in the 7-day forecast window) is computed with a
   Poisson approximation, P = 1 - exp(-Lambda), where Lambda is the
   closed-form Omori-Utsu integral of lambda_M5 over the forecast
   window, using only events known as of the forecast day (the
   conditioning history is frozen at the forecast origin -- we do not
   simulate hypothetical new events occurring inside the forecast
   window itself, since those are exactly what we are forecasting).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agents.seismic.features import MAXC_CORRECTION, estimate_mc_maxcurvature

logger = logging.getLogger(__name__)

LABEL_MIN_MAG = 5.0
BVALUE_BIN_CORRECTION = 0.05


# ------------------------------------------------------------------ #
# Baseline A: logistic regression
# ------------------------------------------------------------------ #

LOGISTIC_FEATURE_COLUMNS = ["relative_rate_7d", "max_mag_30d"]


def _prepare_logistic_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df[LOGISTIC_FEATURE_COLUMNS].copy()
    x["relative_rate_7d"] = x["relative_rate_7d"].fillna(0.0)
    x["max_mag_30d"] = x["max_mag_30d"].fillna(0.0)
    return x


def fit_baseline_a(train_df: pd.DataFrame) -> Pipeline:
    """Fit the logistic regression baseline. `train_df` needs LOGISTIC_FEATURE_COLUMNS + 'label'.

    Deliberately NOT using class_weight="balanced": with this catalog's extreme
    imbalance (~3600 negatives per positive), balanced reweighting shifts the
    fitted intercept so far that predict_proba outputs ~0.46-1.0 for literally
    every row (confirmed empirically), destroying calibration even though
    ROC-AUC (rank-based, insensitive to this) looks fine. Plain unweighted MLE
    logistic regression keeps predict_proba calibrated to the true base rate,
    which the threshold/false-alarm-rate/probability-gain metrics depend on.
    """
    x = _prepare_logistic_features(train_df)
    y = train_df["label"].astype(int)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(x, y)
    logger.info(
        "Baseline A fit on %d rows (%d positive). Coefficients: %s",
        len(x), int(y.sum()), dict(zip(LOGISTIC_FEATURE_COLUMNS, pipeline["clf"].coef_[0])),
    )
    return pipeline


def predict_baseline_a(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    x = _prepare_logistic_features(df)
    return pipeline.predict_proba(x)[:, 1]


# ------------------------------------------------------------------ #
# Baseline B: simplified temporal ETAS
# ------------------------------------------------------------------ #


@dataclass
class EtasParams:
    mu: float  # regional background rate, events/day (>= mc)
    K: float  # productivity scale
    alpha: float  # productivity magnitude-scaling
    c: float  # Omori-Utsu c (days)
    p: float  # Omori-Utsu p (> 1)
    b_value: float  # global Gutenberg-Richter b-value
    mc: float  # magnitude of completeness used for this fit
    cell_weights: dict[str, float] = field(default_factory=dict)  # share of total >=mc events, per cell

    @property
    def gr_factor(self) -> float:
        """P(M>=5 | M>=mc) under Gutenberg-Richter: converts a >=mc rate into a >=5 rate."""
        return 10.0 ** (-self.b_value * (LABEL_MIN_MAG - self.mc))


def aki_b_value(mags: np.ndarray, mc: float) -> float:
    """Aki (1965) maximum-likelihood b-value estimate, with the standard half-bin correction."""
    mean_mag = float(np.mean(mags))
    denom = mean_mag - (mc - BVALUE_BIN_CORRECTION)
    return float(np.log10(np.e) / denom)


def _omori_g(dt: np.ndarray, c: float, p: float) -> np.ndarray:
    """Omori-Utsu decay density, normalised to integrate to 1 over [0, inf)."""
    return (p - 1) / c * (1 + dt / c) ** (-p)


def _omori_G(dt: np.ndarray, c: float, p: float) -> np.ndarray:
    """Omori-Utsu CDF: integral of _omori_g from 0 to dt (dt assumed >= 0)."""
    return 1 - (1 + dt / c) ** (1 - p)


def _etas_neg_log_likelihood(x: np.ndarray, times: np.ndarray, mags: np.ndarray, mc: float, T: float) -> float:
    mu = np.exp(x[0])
    K = np.exp(x[1])
    alpha = x[2]
    c = np.exp(x[3])
    p = 1.0 + np.exp(x[4])  # enforce p > 1, required for the Omori integral to converge

    kappa = K * np.exp(alpha * (mags - mc))

    dt = times[:, None] - times[None, :]  # dt[i, j] = t_i - t_j
    causal = dt > 0
    g = np.zeros_like(dt)
    g[causal] = _omori_g(dt[causal], c, p)
    triggered = (g * kappa[None, :]).sum(axis=1)

    lam_at_events = mu + triggered
    log_lik = np.sum(np.log(lam_at_events))

    compensator = mu * T + np.sum(kappa * _omori_G(T - times, c, p))
    return -(log_lik - compensator)


#  Bounds on the reparametrised vector x = [log(mu), log(K), alpha, log(c), log(p-1)].
#  Without bounds, Nelder-Mead/BFGS can wander into regions where c or p blow up to
#  astronomical values (observed empirically on sparse training samples), causing
#  overflow in exp() with no likelihood gradient to pull them back. These ranges
#  cover all physically plausible ETAS parameters (c: 1e-5 to 50 days, p: 1.0001 to 21).
_ETAS_BOUNDS = [
    (np.log(1e-8), np.log(10.0)),   # mu: events/day
    (np.log(1e-8), np.log(50.0)),  # K
    (0.01, 5.0),                    # alpha
    (np.log(1e-5), np.log(50.0)),  # c (days)
    (np.log(1e-4), np.log(20.0)),  # p - 1
]


def _fit_etas_kernel(times: np.ndarray, mags: np.ndarray, mc: float, T: float) -> tuple[float, float, float, float, float]:
    """MLE fit of (mu, K, alpha, c, p) via scipy, optimising in a reparametrised space
    that keeps mu, K, c > 0 and p > 1 automatically, and bounded (see _ETAS_BOUNDS) to
    keep the search inside physically plausible ETAS parameter ranges."""
    n = len(times)
    x0 = np.array([np.log(max(n / T, 1e-6)), np.log(0.1), 1.0, np.log(0.01), np.log(0.2)])
    x0 = np.clip(x0, [b[0] for b in _ETAS_BOUNDS], [b[1] for b in _ETAS_BOUNDS])
    result = minimize(
        _etas_neg_log_likelihood,
        x0,
        args=(times, mags, mc, T),
        method="L-BFGS-B",
        bounds=_ETAS_BOUNDS,
        options={"maxiter": 2000},
    )
    if not result.success:
        logger.warning("ETAS kernel MLE did not fully converge: %s", result.message)
    for (lo, hi), value, name in zip(_ETAS_BOUNDS, result.x, ["log(mu)", "log(K)", "alpha", "log(c)", "log(p-1)"]):
        if np.isclose(value, lo, atol=1e-3) or np.isclose(value, hi, atol=1e-3):
            logger.warning(
                "ETAS parameter %s converged AT its bound (%.4g in [%.4g, %.4g]). This typically means "
                "the training catalog is too sparse/short (e.g. dominated by one sequence) to identify "
                "this parameter -- known ETAS MLE degeneracy, not a bug. Treat this fit with caution.",
                name, value, lo, hi,
            )
    mu, K, alpha, c, p = result.x
    return float(np.exp(mu)), float(np.exp(K)), float(alpha), float(np.exp(c)), float(1.0 + np.exp(p))


def fit_etas(train_events: pd.DataFrame) -> EtasParams:
    """Fit the simplified ETAS baseline on the training period only.

    `train_events` must have columns ['cell_id', 'date', 'mag'] and should
    already be restricted to the training time window (and training
    region, if different from the test region) by the caller.
    """
    mc = estimate_mc_maxcurvature(train_events["mag"].to_numpy()) + MAXC_CORRECTION
    complete = train_events[train_events["mag"] >= mc].sort_values("date").reset_index(drop=True)
    if len(complete) < 50:
        logger.warning("Only %d events >= Mc=%.2f in the training period; ETAS fit will be unstable.", len(complete), mc)

    t0 = complete["date"].min()
    times = (complete["date"] - t0).dt.days.to_numpy(dtype=float)
    mags = complete["mag"].to_numpy()
    T = float((complete["date"].max() - t0).days) + 1.0

    mu, K, alpha, c, p = _fit_etas_kernel(times, mags, mc, T)
    b_value = aki_b_value(mags, mc)

    cell_counts = complete.groupby("cell_id").size()
    cell_weights = (cell_counts / cell_counts.sum()).to_dict()

    logger.info(
        "ETAS fit: mu=%.5f K=%.5f alpha=%.3f c=%.5f p=%.3f b=%.3f mc=%.2f (n=%d events, %d cells)",
        mu, K, alpha, c, p, b_value, mc, len(complete), len(cell_weights),
    )
    return EtasParams(mu=mu, K=K, alpha=alpha, c=c, p=p, b_value=b_value, mc=mc, cell_weights=cell_weights)


def build_cell_event_index(events_df: pd.DataFrame, mc: float, t0: pd.Timestamp) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-cell (days-since-t0, magnitude) arrays for events with mag >= mc, sorted by time.

    `t0` must be the same reference date used when fitting the ETAS kernel's
    `times` array, so that day offsets are on a consistent scale.
    """
    complete = events_df[events_df["mag"] >= mc]
    index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cell_id, group in complete.groupby("cell_id"):
        days = (group["date"] - t0).dt.days.to_numpy(dtype=float)
        order = np.argsort(days)
        index[cell_id] = (days[order], group["mag"].to_numpy()[order])
    return index


def predict_etas(
    params: EtasParams,
    cell_id: str,
    event_days: np.ndarray,
    event_mags: np.ndarray,
    forecast_days: np.ndarray,
    horizon: int = 7,
) -> np.ndarray:
    """P(>=1 M5+ event in (t, t+horizon]) for each forecast day t in `forecast_days`.

    `event_days`/`event_mags` must be this cell's FULL known >=mc event
    history (sorted); for each forecast day, only events with
    event_days <= t are used, so this is safe to call with the complete
    history even when `forecast_days` spans multiple points in time.
    """
    mu_cell = params.mu * params.cell_weights.get(cell_id, 0.0)
    kappa = params.K * np.exp(params.alpha * (event_mags - params.mc))
    gr_factor = params.gr_factor

    probs = np.empty(len(forecast_days), dtype=float)
    for idx, t in enumerate(forecast_days):
        mask = event_days <= t
        lo = np.clip((t + 1) - event_days[mask], a_min=0, a_max=None)
        hi = (t + horizon) - event_days[mask]
        contrib = kappa[mask] * (_omori_G(hi, params.c, params.p) - _omori_G(lo, params.c, params.p))
        lam_mc_integral = mu_cell * horizon + float(contrib.sum())
        lam_m5_integral = max(lam_mc_integral, 0.0) * gr_factor
        probs[idx] = 1.0 - np.exp(-lam_m5_integral)
    return probs


if __name__ == "__main__":
    # Quick smoke test on whatever sample data currently sits in data/processed
    # and data/raw -- NOT the official evaluation (that is train.py's job, with
    # a proper time-based train/val/test split, calibration and metrics report).
    import sys

    from sklearn.metrics import average_precision_score, roc_auc_score

    from agents.seismic.features import assign_grid_cells, build_transformer, load_catalog

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    features = pd.read_parquet("data/processed/features.parquet")
    labels = pd.read_parquet("data/processed/labels.parquet")
    merged = features.merge(labels, on=["cell_id", "date"], how="inner")
    merged = merged[merged["label"].notna()].reset_index(drop=True)

    split_date = merged["date"].quantile(0.8, interpolation="nearest")
    train_df = merged[merged["date"] < split_date]
    test_df = merged[merged["date"] >= split_date]
    logger.info("Smoke-test split at %s: %d train rows, %d test rows", split_date, len(train_df), len(test_df))

    if train_df["label"].sum() == 0 or test_df["label"].sum() == 0:
        logger.warning("Too few positive labels in this sample for a meaningful smoke test; skipping metrics.")
        sys.exit(0)

    model_a = fit_baseline_a(train_df)
    pred_a = predict_baseline_a(model_a, test_df)
    logger.info(
        "Baseline A smoke test: ROC-AUC=%.3f PR-AUC=%.4f",
        roc_auc_score(test_df["label"], pred_a), average_precision_score(test_df["label"], pred_a),
    )

    catalog = load_catalog(Path("data/raw/usgs_catalog.csv"))
    assigned = assign_grid_cells(catalog, build_transformer())
    train_events = assigned[assigned["date"] < split_date][["cell_id", "date", "mag"]]
    etas_params = fit_etas(train_events)

    t0 = train_events["date"].min()
    cell_index = build_cell_event_index(assigned[["cell_id", "date", "mag"]], etas_params.mc, t0)

    etas_preds = []
    for cell_id, group in test_df.groupby("cell_id"):
        event_days, event_mags = cell_index.get(cell_id, (np.array([]), np.array([])))
        forecast_days = (group["date"] - t0).dt.days.to_numpy(dtype=float)
        preds = predict_etas(etas_params, cell_id, event_days, event_mags, forecast_days)
        etas_preds.append(pd.Series(preds, index=group.index))
    etas_pred_series = pd.concat(etas_preds).reindex(test_df.index)

    logger.info(
        "Baseline B (ETAS) smoke test: ROC-AUC=%.3f PR-AUC=%.4f",
        roc_auc_score(test_df["label"], etas_pred_series), average_precision_score(test_df["label"], etas_pred_series),
    )
