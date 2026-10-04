# Seismic Forecasting Agent (Phase 1 of a disaster-intelligence system)

For each 50x50 km grid cell in Nepal and each day, this agent estimates the
**probability** that an M5.0+ earthquake will occur in that cell within the
next 7 days.

**This is probabilistic forecasting, not deterministic earthquake
prediction.** No model here (or anywhere in seismology) can tell you that an
earthquake *will* happen at a specific place and time. What it can do is say
how anomalous current seismicity looks relative to a cell's own history, and
turn that into a calibrated probability — the same kind of statement as a
weather forecast's "30% chance of rain," not a warning siren. Treat every
`risk_score` in this repo's output accordingly: as a shift in odds worth
weighing alongside other evidence, never as a prediction to act on alone.

## How it works

- **Region**: trained on the wider Himalayan belt (lat 25-36N, lon 70-97E),
  evaluated on Nepal (lat 26.3-30.5N, lon 80-88.2E).
- **Data**: the USGS earthquake catalog (M3.0+, 1975-present).
- **Grid**: an equal-area (Lambert Azimuthal Equal-Area) 50x50 km grid, kept
  to "active" cells (>=5 historical events) plus their neighbours, to keep
  the dataset CPU-sized.
- **Features**: per cell, per day — event counts (1/7/30-day), relative rate
  vs. long-term background, 30-day max magnitude and days since, 30-day
  seismic energy, a rolling Gutenberg-Richter b-value, and the same stats
  for the 8 neighbouring cells. Every feature is computed causally — see
  `agents/seismic/features.py`'s module docstring for the exact cutoff
  rules, and `tests/test_features_no_leakage.py` for the test that enforces
  it.
- **Label**: did an M5.0+ event occur in that cell during the next 7 days
  (`NaN`, not `0`, when the catalog doesn't yet extend far enough to know).
- **Models**: a logistic-regression baseline, a simplified temporal ETAS
  (self-exciting point process) baseline, and the primary model — an
  LSTM + Transformer encoder over a 60-day feature sequence, trained with
  focal loss and isotonic-calibrated on a held-out validation period. See
  `agents/seismic/baselines.py` and `agents/seismic/train.py` for the exact
  assumptions behind each.
- **Output contract**: every forecast is an `AgentOutput` record
  (`schemas/agent_output.py`) — the shared schema this agent and any future
  disaster-intelligence agents in this system will emit.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

## Running the pipeline

Run these in order. Each step reads the previous step's output from
`data/` or `models/`.

1. **Fetch the historical catalog** (one-time, ~90 seconds for the full
   1975-present Himalayan belt catalog):
   ```bash
   python -m pipelines.fetch_usgs historical --start 1975-01-01 --end 2026-10-03
   ```

2. **Build features and labels** (~35s and ~5s respectively on the full
   catalog):
   ```bash
   python -m agents.seismic.features
   python -m agents.seismic.labels
   ```

3. **Fit the baselines** (optional standalone smoke test; `train.py` below
   fits and evaluates them as part of the full pipeline anyway):
   ```bash
   python -m agents.seismic.baselines
   ```

4. **Train and evaluate** the LSTM+Transformer model, Baseline A and ETAS,
   all on a strict time-based split (train 1975-2014, validate 2015-2019,
   test 2020-present — never shuffled across time). This is the expensive
   step (roughly 45 minutes on CPU for the full catalog, dominated by
   per-epoch validation passes). Writes `reports/metrics.json`, comparison
   plots (ROC, PR, calibration, Molchan diagram), the Gorkha-2015 case
   study plot, and `models/seismic_checkpoint.pt`:
   ```bash
   python -m agents.seismic.train
   ```
   Use `--quick` (2 epochs, patience 1) to sanity-check the pipeline runs
   before committing to a full run.

5. **Run a daily forecast** for Nepal — fetches the last 2 days of events,
   rebuilds features through today, loads the trained checkpoint, and
   writes `reports/latest_forecast.json` (one `AgentOutput` per active
   Nepal cell):
   ```bash
   python -m agents.seismic.predict
   ```

## Tests

```bash
pytest
```

Covers label-horizon correctness, absence of future-data leakage in
features (verified by re-running feature generation on a truncated catalog
and checking every causal feature is bit-for-bit identical), and
`AgentOutput` schema validation.

## Known limitations

- Baseline A (logistic regression on 2 features) has weak discriminative
  power (test ROC-AUC ~0.59) — expected, since it's a deliberately simple
  baseline, not a competitor to the full model.
- The simplified ETAS baseline shares one kernel shape across all cells
  (fit on the pooled regional catalog) and allocates background rate by
  historical event share — see `agents/seismic/baselines.py` for the full
  list of stated assumptions.
- `b_value` is only computable (needs >=50 events in a rolling 365-day
  window) for a small fraction of cell-days — mostly the most active cells
  during major sequences. It's `NaN` (with a `b_value_valid` flag)
  everywhere else, by design.
- All three models' test-set probability gain is modest in absolute terms,
  as is inherent to this problem: M5+ earthquakes in any given 50km cell in
  any given week are intrinsically rare and only partially forecastable
  with current science.
