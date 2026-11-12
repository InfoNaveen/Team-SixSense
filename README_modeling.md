# Modeling Team — Integration Guide

> **Owner**: Features + Evaluation  
> **Files you care about**: `eval/backtest.py` · `features/build_features.py` · `eval/metrics.py`  
> **You must not modify**: these files directly — open a PR and tag the F+E owner.

---

## 1. What we own vs what you own

| Concern | Owner |
|---------|-------|
| `eval/backtest.py` — fold generation, harness loop, parquet output | **F+E** |
| `features/build_features.py` — all feature engineering | **F+E** |
| `eval/metrics.py` — all metric computation and CSV reports | **F+E** |
| `models/` — LightGBM, Prophet, LSTM, any new model | **Modeling** |
| `data/peak_flags.parquet` — peak classifier output | **Modeling / Classifier** |
| `data/processed.parquet` — cleaned hourly timeseries | **Data** |

---

## 2. The `fit_predict` contract

Your model must expose **one callable** with this exact signature:

```python
def fit_predict(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    horizon:  str,          # "day_ahead" or "week_ahead"
) -> pd.DataFrame:
    ...
```

### Input DataFrames

| Column | Type | Description |
|--------|------|-------------|
| `ds` | `datetime64[ns]` | Hourly timestamp (India/Kolkata, stored as UTC-naive or tz-aware) |
| `load_mw` | `float` | Actual grid load in MW (NaN in test_df is acceptable) |
| `temp_c` | `float` | Dry-bulb temperature °C |
| `rh_pct` | `float` | Relative humidity % (may be missing) |
| `apparent_temp_c` | `float` | Fallback apparent temperature °C (may be missing) |
| *feature columns* | `float/int` | Pre-built by `build_features.py` if you call it |

### Output DataFrame (required columns)

| Column | Type | Description |
|--------|------|-------------|
| `ds` | `datetime64[ns]` | Must match `test_df["ds"]` exactly |
| `yhat` | `float` | P50 (median) point forecast in MW |
| `yhat_lower` | `float` | P10 lower bound in MW |
| `yhat_upper` | `float` | P90 upper bound in MW |

> **Critical**: `yhat_lower` and `yhat_upper` must be **true probabilistic quantiles** — do not clip them to equal `yhat`. The harness computes PICP, so wrong intervals directly hurt your score.

---

## 3. Using the feature pipeline

Call `build_features` **inside** `fit_predict`, on the training window only:

```python
from features.build_features import build_features, assert_no_leakage

def fit_predict(train_df, test_df, horizon):
    origin = train_df["ds"].max()

    # Build features (causal — only uses data up to origin)
    feat = build_features(
        train_df,
        horizon=horizon,
        holidays=HOLIDAYS,          # list[pd.Timestamp]
        festivals=FESTIVALS,        # list[(name, pd.Timestamp)]
        lockdown_ranges=LOCKDOWNS,  # list[(start_str, end_str)]
        origin=origin,
    )
    assert_no_leakage(feat, origin=origin)   # will raise if leakage detected

    # Build test features (use the SAME origin so lags are computed correctly)
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    test_feat = build_features(all_df, horizon=horizon, origin=test_df["ds"].max())
    test_feat = test_feat[test_feat["ds"].isin(test_df["ds"])]

    # ... fit model on feat, predict on test_feat ...
```

> **Important for week-ahead**: the model receives `load_lag_168h` and `load_lag_336h` only.  
> **Important for day-ahead**: the model receives `load_lag_48h` and `load_lag_168h` only.  
> These choices enforce the minimum-lag rule so forecasts are operationally valid.

---

## 4. Running the backtest

```python
from eval.backtest import run_backtest
from models.my_model import fit_predict   # your callable

# Day-ahead
results_da = run_backtest(fit_predict, horizon="day_ahead")
# → writes results/forecasts_day_ahead.parquet automatically

# Week-ahead
results_wa = run_backtest(fit_predict, horizon="week_ahead")
# → writes results/forecasts_week_ahead.parquet
```

The harness will:
1. Load `data/processed.parquet` (or use a `df` kwarg you pass).
2. Create 12 monthly folds (last 12 calendar months).
3. Call `fit_predict(train_df, test_df, horizon)` for each fold.
4. Assert no data leakage between train and test.
5. Merge actuals and write the combined parquet.

---

## 5. Computing metrics

```python
from eval.metrics import compute_all_metrics

compute_all_metrics(
    forecasts_path  = "results/forecasts_day_ahead.parquet",
    baseline_path   = "results/forecasts_day_ahead_baseline_snaive.parquet",
    horizon         = "day_ahead",
    model_name      = "lgbm_v1",          # your model label
    peak_flags_path = "results/peak_flags.parquet",   # optional
    context_flags   = {                   # optional
        "2025-04-14": "heatwave",
        "2024-11-01": "diwali festival",
    },
)
```

This writes:
- `results/metrics.csv` — master table (one row per model × horizon)
- `results/metrics_by_season.csv`
- `results/metrics_by_hour.csv`
- `results/metrics_by_peaktype.csv`
- `results/missed_peaks.csv`

---

## 6. Baselines (pre-computed for reference)

We supply two baselines you must beat:

| Baseline | File | Notes |
|----------|------|-------|
| Seasonal-Naive | `results/forecasts_{horizon}_baseline_snaive.parquet` | Same hour last week (median over all Mondays at 14:00, etc.) |
| Linear Regression | `results/forecasts_{horizon}_baseline_linreg.parquet` | Cyclic time encodings + trend |

Run them yourself:
```bash
python -m eval.backtest --horizon both
```

---

## 7. Data contract (`data/processed.parquet`)

The cleaned parquet (produced by the Data team) must have:

| Column | Required | Notes |
|--------|----------|-------|
| `ds` | ✅ | Hourly, no gaps, UTC-naive or tz-aware |
| `load_mw` | ✅ | No NaN in training rows |
| `temp_c` | ✅ | Dry-bulb temperature °C |
| `rh_pct` | ➖ optional | For heat index; fallback to `apparent_temp_c` |
| `apparent_temp_c` | ➖ optional | Fallback for heat index |

Minimum history: **13 calendar months** (12 test months + at least 1 month of training for fold 0).

---

## 8. File outputs summary

```
results/
  forecasts_day_ahead.parquet             ← your model (day-ahead)
  forecasts_week_ahead.parquet            ← your model (week-ahead)
  forecasts_day_ahead_baseline_snaive.parquet
  forecasts_day_ahead_baseline_linreg.parquet
  forecasts_week_ahead_baseline_snaive.parquet
  forecasts_week_ahead_baseline_linreg.parquet
  metrics.csv                             ← master metrics table
  metrics_by_season.csv
  metrics_by_hour.csv
  metrics_by_peaktype.csv
  missed_peaks.csv
  peak_flags.parquet                      ← FROM Modeling/Classifier
```

---

## 9. Quick sanity checklist before submitting

- [ ] `fit_predict` returns all three columns: `yhat`, `yhat_lower`, `yhat_upper`
- [ ] `yhat_lower < yhat < yhat_upper` for the vast majority of rows
- [ ] `assert_no_leakage(feat, origin)` passes without raising
- [ ] Backtest runs for both `day_ahead` and `week_ahead`
- [ ] `metrics.csv` contains your model name in both horizon rows
- [ ] PICP is between 0.70 and 0.90 (target ≈ 0.80 for P10–P90)
- [ ] MAPE improvement vs snaive is positive for both horizons
