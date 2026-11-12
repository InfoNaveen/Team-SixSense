# Model Card — Probabilistic Peak-Aware Load Forecasting

> CodeSpark 2026 · Delhi DISCOM Load Forecasting & Procurement Copilot

---

## 1. Model Details

| Field | Value |
|-------|-------|
| Model type | LightGBM Quantile Regression (3 separate models: alpha = 0.10, 0.50, 0.90) |
| Framework | `lightgbm` via scikit-learn API |
| Output | Hourly P10, P50, P90 demand forecasts (MW) |
| Horizons | Day-ahead (24 h from 00:00 day D) · Week-ahead (168 h from 00:00 day D) |
| Forecast origin | 09:00 IST on D-1 |
| Peak classifier | [FILL: binary classifier or threshold on P90 — describe approach used by P4] |
| Training objective | Pinball (quantile) loss at each alpha |

---

## 2. Intended Use

- **Primary**: Day-ahead and week-ahead probabilistic load forecasting for Delhi DISCOM procurement planning.
- **Users**: DISCOM procurement desk, scheduling coordinators.
- **NOT for**: Real-time grid control, financial trading, or safety-critical load-shedding decisions without human oversight.

---

## 3. Training Data

| Item | Detail |
|------|--------|
| Load source | [FILL: SLDC Delhi / Grid-India / Kaggle — cite exact URL and date range] |
| Weather source | Open-Meteo Historical Weather API (hourly, Delhi coords 28.61°N 77.23°E) |
| Date range | [FILL: e.g. 2019-01-01 to 2025-10-31] |
| Granularity | Hourly (8760 h/year × N years) |
| Total rows | [FILL: e.g. ~52,000 rows after cleaning] |

### Data Quality & Imputation

| Issue | Count | Method | Notes |
|-------|-------|--------|-------|
| Missing load hours | [FILL] | [FILL: linear interp / forward fill / dropped] | Flagged via `load_mw_imputed_flag` |
| Duplicate timestamps | [FILL] | Kept first occurrence | IST has no DST, so duplicates indicate meter errors |
| Outlier load values | [FILL] | [FILL: capped at ±4σ / manual review] | |
| Missing weather hours | [FILL] | [FILL: interpolation] | |

> **Honesty note on weather data**: We use **actual observed weather** as a proxy for weather forecasts. Historical forecast archives with hourly Delhi resolution were not freely available. Our accuracy therefore represents an **upper bound**; production deployment would degrade by the weather forecast error. This is a known limitation, stated here transparently.

---

## 4. Features

| Feature | Source | Leakage-safe? |
|---------|--------|---------------|
| `cdh` (cooling-degree-hours, base 25°C) | Weather | ✓ Uses D-1 weather at origin |
| `apparent_temp_c` / heat index | Weather | ✓ |
| `temp_lag_24h`, `temp_lag_48h` | Weather | ✓ |
| `load_lag_24h`, `load_lag_48h`, `load_lag_168h` | Load | ✓ All lags ≥24 h from target |
| `load_rolling_7d_mean`, `load_rolling_7d_std` | Load | ✓ Window ends before origin |
| `hour_of_day`, `day_of_week`, `month` | Calendar | ✓ Deterministic |
| `is_holiday`, `festival_name` | Calendar | ✓ Known in advance |
| `is_oddeven`, `is_lockdown` | Event flags | ✓ Known policy dates |
| `trend` (linear counter) | Time | ✓ |
| [FILL: any additional features P3 added] | | |

---

## 5. Evaluation Protocol

- **Method**: Walk-forward (expanding-window) backtest. **Never a random split.**
- **Fold structure**: [FILL: e.g. 12 monthly folds, training expands, test = 1 month]
- **Test period**: [FILL: last 12 months of available data]
- **Baselines compared**: Seasonal-naive, linear regression, [FILL: others]
- **Metrics**: See next section

---

## 6. Metrics

### 6a. Overall accuracy (P50 point forecast)

| Horizon | Model | MAPE (%) | RMSE (MW) | vs Seasonal-Naive |
|---------|-------|----------|-----------|-------------------|
| Day-ahead | LightGBM | [FILL] | [FILL] | [FILL: -X%] |
| Day-ahead | Seasonal-naive | [FILL] | [FILL] | baseline |
| Week-ahead | LightGBM | [FILL] | [FILL] | [FILL: -X%] |
| Week-ahead | Seasonal-naive | [FILL] | [FILL] | baseline |

### 6b. Interval calibration

| Horizon | PICP (%) target=80% | Mean interval width (MW) |
|---------|---------------------|--------------------------|
| Day-ahead | [FILL] | [FILL] |
| Week-ahead | [FILL] | [FILL] |

> PICP = Prediction Interval Coverage Probability. If intervals are well-calibrated, PICP ≈ 80% for P10–P90 band.

### 6c. Peak-event detection

| Metric | Value |
|--------|-------|
| Precision | [FILL] |
| Recall | [FILL] |
| F1 | [FILL] |
| Peak hour MAE (minutes) | [FILL] |

---

## 7. Known Failure Modes

1. **Sudden weather shifts**: Model misses peaks when temperature jumps 5°C+ within 24 h (thermal inertia lag not fully captured).
2. **Festival effects**: Diwali and Eid load patterns differ from training; few examples per festival.
3. **Post-lockdown regime shift**: COVID lockdowns (2020-21) create a distributional break; model may under-weight the new regime.
4. **Week-ahead degradation**: Accuracy drops significantly for hours 96–168 due to weather forecast uncertainty propagation.
5. [FILL: additional failure modes observed during backtest — see docs/why_peaks_are_hard.md and results/missed_peaks.csv]

---

## 8. Ethical Considerations

- **Grid reliability**: Under-procurement recommendations can cause load-shedding. This model should **assist**, not replace, human dispatchers.
- **No personal data**: All data is aggregated grid-level demand; no individual consumer data.
- **Equity**: Peak events disproportionately affect areas with poor infrastructure. Over-procurement is safer for vulnerable communities.

---

## 9. Limitations & Recommendations

- Weather-as-proxy inflates reported accuracy (see honesty note above).
- Model is trained on Delhi only; not transferable to other DISCOMs without retraining.
- Procurement cost rates are **placeholder assumptions** (Rs 5/12/2.5 per kWh) — actual DISCOM tariffs will change results.
- [FILL: any additional limits from P3/P4 analysis]

---

*Last updated: [FILL: date] · Team Lead*
