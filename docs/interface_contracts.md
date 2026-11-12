# Interface Contracts & Team Checkpoints (H1 Locked)

**Project**: Probabilistic Peak-Aware Load Forecasting & Procurement Copilot for Delhi's DISCOM  
**Event**: CodeSpark 2026 (12-hour Hackathon, Team of 4)

## 1. Strict Folder Ownership
- **Lead (P1)**: `procurement/`, `docs/`, `integration/`, `pitch/`, `results/procurement_week.json`
- **Data & Dashboard (P2)**: `data/`, then `app/`
- **Features & Evaluation (P3)**: `features/`, `eval/`, `results/metrics.csv`, `results/missed_peaks.csv`
- **Modeling (P4)**: `models/`, `results/forecasts_day.parquet`, `results/forecasts_week.parquet`, `results/peak_flags.parquet`

> Rule: Never write into another person's folder. Communicate strictly through the contracts below and `results/` files.

---

## 2. Locked Contracts

### Contract 1: `data/clean_hourly.parquet` (Data / P2 produces)
- Index or column `ts`: `datetime64`, Asia/Kolkata (`IST`), hourly, unique, sorted.
- `load_mw` (`float`): Megawatts; missing values imputed and logged in `data/imputation_log.md`.
- `load_mw_imputed_flag` (`bool`): `True` where `load_mw` was filled.
- `temp_c`, `rh_pct`, `apparent_temp_c` (`float`): Hourly temperature (°C), relative humidity (%), apparent temperature (°C).
- `is_holiday` (`bool`), `festival_name` (`str`), `is_oddeven` (`bool`), `is_lockdown` (`bool`).

### Contract 2: `features/build_features.py` (Features / P3 produces)
```python
def build_features(df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """
    horizon in {"day", "week"}
    Indexed by ts; feature columns plus target column "load_mw".
    Leakage rule: uses only data available at forecast origin (09:00 IST on D-1).
    Load lags must be strictly older than forecast origin.
    """
```

### Contract 3: `eval/backtest.py` (Features / P3 produces) & `fit_predict` (Modeling / P4 supplies)
```python
def get_folds(df: pd.DataFrame) -> list[dict]:
    """Each dict: fold, train_end, test_start, test_end (expanding window, monthly test blocks, last 12 months)."""

def run_backtest(fit_predict, horizon: str) -> pd.DataFrame:
    """Writes results/forecasts_{horizon}.parquet."""

# Modeling supplies:
def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame with columns p10, p50, p90 indexed like X_test (with p10 <= p50 <= p90)."""
```

### Contract 4: `results/` (File-based communication only)
- `results/forecasts_day.parquet`, `results/forecasts_week.parquet`:
  `ts, fold, horizon, model, p10, p50, p90, actual`
- `results/peak_flags.parquet`:
  `date, fold, peak_prob, peak_pred (bool), peak_hour_pred, peak_actual (bool), peak_hour_actual, threshold_mw`
- `results/metrics.csv` (Features writes):
  `horizon, model, fold, metric, value`
- `results/missed_peaks.csv` (Features writes):
  `date, actual_peak_mw, pred_p50_peak_mw, pred_p90_peak_mw, hour_error_min, note`
- `results/procurement_week.json` (Lead writes):
  `schedule (24x7 values), p90_exposure_mw, cost_model, cost_flat, saving`

### Contract 5: `procurement/procurement.py` (Lead / P1 produces)
```python
def simulate_week(forecast_df: pd.DataFrame, params: dict) -> dict:
    """Simulates P90 buy schedule vs P50 and flat schedule; returns Contract #4 dict."""
```

### Contract 6: `app/app.py` (Data / P2 produces)
- Reads `results/` only; never imports model or feature code. Must run 100% offline.

---

## 3. Default Assumptions (Change Only via Lead)
- **CDH base**: `25 °C` (`max(0, temp_c - 25)`).
- **Peak day threshold**: Daily maximum load > 90th percentile of daily maxima in the **training window only**.
- **Forecast origin**: `09:00 IST` on `D-1` (Day-ahead = 24h of day `D`; Week-ahead = 168h from `00:00` of day `D`).
- **Weather source**: Forecast archive if available; otherwise actual weather as proxy, stated transparently in `docs/model_card.md`.

## 4. Matched Package Versions
| Package | Version | Purpose |
| --- | --- | --- |
| `lightgbm` | `4.7.0` | Quantile regression ($\alpha \in \{0.1, 0.5, 0.9\}$) & peak classifier |
| `pandas` | `3.0.6` | Data manipulation, contracts, schema handling |
| `numpy` | `2.5.3` | Matrix operations, non-crossing sorting, conformal residuals |
| `scikit-learn` | `1.9.1` | Peak classification metrics (Precision, Recall, F1) |
| `scipy` | `1.18.1` | Quantile routines & statistical operations |
| `pyarrow` | `26.0.1` | Reading and writing `results/*.parquet` tables |

## 5. Checkpoints
- **H1**: Schema & contracts locked (this document).
- **H3**: Clean dataset (`data/clean_hourly.parquet`) + walk-forward harness + baselines.
- **H6**: Features + quantile models + peak classifier + `results/*` populated for all folds.
- **H8**: **FEATURE FREEZE** — full accuracy table, PICP calibration check, procurement simulation on real P90.
- **H10**: Dashboard integrated on real results; model card & memo complete.
- **H12**: Fresh-clone E2E run verified, 2 pitch rehearsals complete, submit.

