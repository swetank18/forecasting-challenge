# Cloud Workload Forecasting

Forecasts the last two hours of aggregate demand (`active_vm_count`, `aggregate_cpu_utilization`) for 10,000 Azure VMs. It uses a leakage-safe rolling-origin backtest to pick the model and turns the forecast into a capacity rule.

Results and reasoning are in [`reports/final_report.md`](reports/final_report.md). The original challenge brief is in [`ASSIGNMENT.md`](ASSIGNMENT.md), [`DATASET.md`](DATASET.md) and [`SUBMISSION.md`](SUBMISSION.md).

## Setup

Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`pyproject.toml` has version ranges. `requirements-lock.txt` pins the exact versions the results were produced with (`pip install -r requirements-lock.txt && pip install -e . --no-deps`).

## Data

Download `data-v1.tar.gz` from the [data-v1 release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1) and extract it into `challenge_data/`:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
(cd challenge_data && sha256sum -c SHA256SUMS)
```

With the GitHub CLI: `gh release download data-v1 -R atomityhq/forecasting-challenge`. The data is not committed.

## Generate predictions

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

This takes about 10 seconds and, in order:

1. loads and validates the readings and prints the quality checks;
2. builds both targets on the full 5-minute grid, then splits off the last 24 timestamps as the holdout;
3. runs the rolling-origin backtest on the training window only and picks a model per target;
4. refits the chosen models on the training window and forecasts the 24 holdout timestamps;
5. writes `predictions.csv` (`timestamp,active_vm_count,aggregate_cpu_utilization`, no index);
6. only then scores the holdout against the forecast and persistence, for reporting. Nothing from this step goes back into the model.

It also writes backtest tables, the model choice, forecast bands and the holdout evaluation to `outputs/` (git-ignored).

Options: `--horizon` (default 24), `--outputs-dir`, `--quiet`. Nothing depends on 10,000 VMs, 134 timestamps or a 110-point training window. The split, backtest origins and season length all come from the data, and the pipeline shrinks the backtest gracefully on small datasets.

## Backtests, notebooks, tests

```bash
python -m cwfc.train --data challenge_data      # backtest tables only, into outputs/
python -m pytest -q
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

- `01_data_understanding`: schema, quality checks, cadence, shared timestamps, lifetimes, target construction
- `02_eda_workload`: CPU distribution, temporal structure, hourly cycle, segments, peaks
- `03_backtest_results`: metric tables, error by horizon, worst fold, intervals, capacity rules, holdout

The notebooks import everything from `src/cwfc` and save the report figures to `reports/figures/`.

The tests cover target construction (dedup, invalid values, gaps), the split, the backtest plan on short series, leakage, and the prediction contract. The leakage tests check that features and forecasts don't change when future data is corrupted. The smoke test runs the whole CLI on a synthetic dataset of 5 VMs × 40 timestamps.

## Layout

```
src/cwfc/
  config.py      horizon, cadence, backtest settings
  io.py          loading and data quality checks
  targets.py     target construction, segment helpers
  split.py       train/holdout split, rolling origins
  baselines.py   persistence, moving averages, drift, seasonal naive
  models.py      ETS, seasonal ETS, ARIMA (AIC per fold), level+season, direct ridge
  features.py    origin-safe lag and rolling features
  backtest.py    rolling-origin loop, summary table, model selection
  metrics.py     MAE, RMSE, MASE, peak MAE, coverage
  intervals.py   residual-quantile bands, capacity rule scoring
  train.py       backtest CLI
  predict.py     prediction CLI
```

## Main decisions

- **`active_vm_count` gets persistence.** Every VM reports at every timestamp, so the series is constant at 10,000. Every model ties at zero error, and the simplest one wins the tie.
- **`aggregate_cpu_utilization` gets seasonal ETS** (additive level + 12-step season). The differenced series shows a clear hourly cycle (lag-12 ACF 0.61). ETS beat persistence by 12% MAE in backtest and by 4.6% on the holdout.
- **Validation:** an expanding window on the training data only, 13 origins every 15 minutes from t = 48, each forecasting 24 steps, with the same origins for every model. Selection uses mean fold MASE, with ties going to the simpler model.
- **No daily or weekly features.** 11 hours of data can't validate them. The within-hour phase is known in advance at any origin and is the only calendar signal used.
- **Intervals** come from per-step quantiles of backtest errors. The recommended capacity rule is a flat level at the highest p95 band value over each 2-hour window.
- **The ML model was kept and reported, but not chosen.** A direct ridge on lag and rolling features didn't beat persistence with ~110 points.

## Assumptions

- Timestamps are seconds from the trace start on a 5-minute grid. Holdout timestamps are the last 24 grid points.
- A VM is active at `t` if it has a finite `avg_cpu` reading at `t`. Duplicate readings for a VM are averaged, and CPU is clipped to [0, 100].
- If a timestamp were missing entirely, its targets would be linearly interpolated and flagged. This release has no gaps.
- `active_vm_count` predictions are rounded to integers. CPU predictions are clipped to [0, 100].
- Metadata deletion times are future information and are never used as features.
- The holdout is used once, for final scoring. It never feeds fitting, feature construction, tuning or selection.
