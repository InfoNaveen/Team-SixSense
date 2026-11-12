# DISCOM Load Forecasting — Features & Evaluation Harness

> **Owner:** Features + Evaluation  
> **Scope:** `eval/` and `features/`  
> **Not in scope:** data cleaning, LightGBM/Prophet/LSTM models, peak classifier, procurement, dashboard.

---

## Quick Start for the Modeling Team

```python
from eval.backtest import run_backtest
from eval.metrics  import compute_all_metrics

# 1. Load your clean, hourly DataFrame
import pandas as pd
df = pd.read_parquet("data/clean_hourly.parquet")  # DatetimeIndex, 'actual' column required

# 2. Plug in your model
def my_fit_predict(train_df, test_df, horizon):
    """
    Train on train_df, return predictions for test_df.

    Parameters
    ----------
    train_df : pd.DataFrame — DatetimeIndex, columns include 'actual' and all feature cols
    test_df  : pd.DataFrame — same schema, 'actual' may be NaN (don't use it!)
    horizon  : "day_ahead" | "week_ahead"

    Returns
    -------
    pd.DataFrame indexed like test_df with columns: p10, p50, p90  (float, MW)
    """
    # ... your model here ...
    pass

# 3. Run the backtest (writes results/forecasts_day_ahead.parquet)
results = run_backtest(
    fit_predict=my_fit_predict,
    horizon="day_ahead",
    df=df,
    model_name="lgbm_v1",          # label that appears in metrics.csv
)

# 4. Compute all metrics (writes results/metrics.csv + results/missed_peaks.csv)
metrics = compute_all_metrics(horizon="day_ahead", raw_df=df)
print(metrics)
```

---

## Detailed Contract

### `eval/backtest.py`

#### `get_folds(df) → list[dict]`

| Field | Type | Description |
|---|---|---|
| `fold_id` | int | 0-based index (oldest month first) |
| `month_label` | str | e.g. `"2025-11"` |
| `train` | DataFrame | All rows **before** the test month |
| `test` | DataFrame | All rows **in** that calendar month |
| `origin` | Timestamp | Last timestamp in `train` |
| `horizon_h` | int | Number of test rows (hours) |

- 12 folds, one per calendar month, so every season appears.
- Expanding window — train grows each fold; no random splits.
- Minimum 672 h (4 weeks) of training required; folds below this are skipped with a warning.

#### `run_backtest(fit_predict, horizon, df, results_dir, model_name)`

- Calls `fit_predict(train_df, test_df, horizon)` **once per fold** (retrain each fold).
- Validates that the return has `p10`, `p50`, `p90` columns and the same length as `test_df`.
- Writes `results/forecasts_{horizon}.parquet`.

#### `fit_predict` signature

```python
def fit_predict(
    train_df: pd.DataFrame,   # DatetimeIndex; columns include 'actual'
    test_df:  pd.DataFrame,   # DatetimeIndex; 'actual' may be absent or NaN
    horizon:  str,            # "day_ahead" | "week_ahead"
) -> pd.DataFrame:            # columns: p10, p50, p90; indexed like test_df
```

> **Important:** `test_df` is passed so the model knows the forecast timestamps.
> Do **not** read `test_df["actual"]` — it may be NaN or absent. Using it = leakage.

---

### `features/build_features.py`

#### `build_features(df, horizon, origin) → pd.DataFrame`

Call this inside `fit_predict` to get the full feature matrix:

```python
from features.build_features import build_features

def my_fit_predict(train_df, test_df, horizon):
    origin   = train_df.index[-1]
    combined = pd.concat([train_df, test_df]).sort_index()
    feats    = build_features(combined, horizon=horizon, origin=origin)

    # Drop rows with NaN features (head of series, insufficient lag history)
    feats = feats.dropna()

    X_train = feats.loc[train_df.index].dropna()
    y_train = train_df.loc[X_train.index, "actual"]
    X_test  = feats.loc[test_df.index]

    # ... train, predict, return DataFrame with p10/p50/p90 ...
```

#### Feature columns produced

| Group | Column(s) | Notes |
|---|---|---|
| Thermal | `cdh`, `cdh_24h_sum`, `heat_index` | Rothfusz regression; fallback to `apparent_temp_c` |
| Temp lags | `temp_lag_24h`, `_48h`, `_72h`, `temp_roll_72h_mean` | Past only |
| Load lags (day-ahead) | `load_lag_48h`, `load_lag_168h`, `load_roll_7d_mean`, `load_roll_7d_max` | — |
| Load lags (week-ahead) | `load_lag_168h`, `load_lag_336h`, `load_roll_7d_mean`, `load_roll_7d_max` | — |
| Calendar | `hour`, `day_of_week`, `month`, `is_weekend`, `is_holiday`, `days_to_festival`, `days_since_festival`, `is_oddeven`, `is_lockdown` | — |
| Trend | `day_index` | Float days since first row in df |

#### Leakage guarantee

`build_features` automatically calls `assert_no_leakage(features, origin, raw_df)` before returning.  
If any load feature at `t > origin` was computed from actuals at `t' > origin`, an `AssertionError` is raised with a detailed message.

---

### `eval/metrics.py`

#### `compute_all_metrics(horizon, results_dir, raw_df) → pd.DataFrame`

Reads `results/forecasts_{horizon}.parquet` (and optionally `results/peak_flags.parquet`).  
Writes:
- `results/metrics.csv` — one row per `(model, horizon, segment_type, segment_value)`
- `results/missed_peaks.csv` — hours where actual > P90 or peak flag was missed

#### Segments computed automatically

| `segment_type` | Values |
|---|---|
| `overall` | `all` |
| `season` | `winter`, `spring`, `summer`, `monsoon` |
| `hour_band` | `night`, `morning`, `afternoon`, `evening` |
| `peak_type` | `peak` (top-25th-percentile), `off_peak` |

#### Metrics columns in `metrics.csv`

| Column | Formula |
|---|---|
| `mape_pct` | Mean \|actual−p50\| / \|actual\| × 100 |
| `rmse_mw` | √( mean (actual−p50)² ) |
| `pinball_p10` | Pinball loss at q=0.1 on P10 |
| `pinball_p50` | Pinball loss at q=0.5 on P50 (= MAE/2) |
| `pinball_p90` | Pinball loss at q=0.9 on P90 |
| `picp` | Fraction of actuals in [P10, P90]; target ≥ 0.80 |
| `peak_timing_error_min` | Mean \|hour of actual peak − hour of P50 peak\| × 60 |
| `peak_magnitude_error_mw` | Mean (actual − P50) at the actual peak hour |
| `peak_precision` | TP/(TP+FP) on is_predicted_peak (needs peak_flags.parquet) |
| `peak_recall` | TP/(TP+FN) on is_predicted_peak |
| `peak_f1` | Harmonic mean of precision and recall |
| `improvement_mape_pct_pct` | (baseline_mape − model_mape) / baseline_mape × 100 |
| `improvement_rmse_mw_pct` | Same for RMSE |
| `improvement_pinball_p50_pct` | Same for pinball P50 |

#### `peak_flags.parquet` interface (Modeling team supplies this)

```
Index   : DatetimeIndex (hourly, same as forecasts)
Columns : is_actual_peak    bool — ground truth from peak classifier
          is_predicted_peak bool — your model's peak flag output
```

---

## Bundled Baselines

Run these to establish reference performance before plugging in LightGBM/Prophet:

```bash
# Seasonal-naïve
python -m eval.backtest --parquet data/clean_hourly.parquet --horizon day_ahead --baseline seasonal_naive

# Linear regression
python -m eval.backtest --parquet data/clean_hourly.parquet --horizon day_ahead --baseline linear

# Then compute metrics
python -m eval.metrics --horizon day_ahead --raw-parquet data/clean_hourly.parquet
```

---

## Input DataFrame Schema

| Column | Type | Required | Notes |
|---|---|---|---|
| `actual` | float | ✅ | MW load. Future rows may be NaN. |
| `temp_c` | float | ✅ | Dry-bulb °C |
| `humidity_pct` | float | ✅ | Relative humidity 0–100 |
| `apparent_temp_c` | float | ⬜ | Feels-like °C (Rothfusz fallback) |
| `is_holiday` | int/bool | ✅ | 1 on public holidays |
| `is_lockdown` | int/bool | ⬜ | 1 during lockdown periods |
| `is_oddeven` | int/bool | ⬜ | 1 during odd-even traffic days |
| `festival_name` | str | ⬜ | Name string on festival day, else `""` |

Index must be a `pd.DatetimeIndex` at **hourly** frequency (`freq="H"` or `"h"`).

---

## File Layout

```
DISCOM/
├── data/
│   └── clean_hourly.parquet      ← provided by Data Engineering
├── eval/
│   ├── __init__.py
│   ├── backtest.py               ← this team
│   └── metrics.py                ← this team
├── features/
│   ├── __init__.py
│   └── build_features.py         ← this team
├── results/
│   ├── forecasts_day_ahead.parquet
│   ├── forecasts_week_ahead.parquet
│   ├── metrics.csv
│   ├── missed_peaks.csv
│   └── peak_flags.parquet        ← provided by Modeling team
└── tests/
    └── test_no_leakage.py        ← this team
```

---

## Running Tests

```bash
# Leakage test (standalone)
python -m pytest tests/test_no_leakage.py -v

# Full smoke-test with synthetic data (no real data needed)
python tests/test_no_leakage.py --smoke
```

