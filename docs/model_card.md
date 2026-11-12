# Model Card — Probabilistic Peak-Aware Load Forecasting

> CodeSpark 2026 — Delhi DISCOM Load Forecasting & Procurement Copilot

---

## 1. Model Details

| Field | Value |
|-------|-------|
| Model type | LightGBM Quantile Regression (3 separate models: $\alpha = 0.10, 0.50, 0.90$) |
| Calibration | Conformal Interval Calibrator (`lgbm_quantile_calibrated`) |
| Framework | `lightgbm` 4.7.0 / `scikit-learn` 1.6+ |
| Output | Hourly P10, P50, P90 demand forecasts (MW) with quantile crossing resolution |
| Horizons | Day-ahead (24 h) & Week-ahead (168 h) from 00:00 IST |
| Forecast origin | Strictly 09:00 IST on D-1 |
| Peak classifier | Top-decile daily peak classifier (>6373.6 MW training threshold) with peak-hour predictor |
| Training objective | Pinball (quantile) loss at each $\alpha$ |

---

## 2. Intended Use

- **Primary**: Day-ahead and week-ahead probabilistic load forecasting for Delhi DISCOM power procurement scheduling.
- **Users**: DISCOM power trading desk, scheduling coordinators, and grid operations planners.
- **Out of Scope**: Real-time grid dispatch (<15 min), financial power market speculative trading, or automated load-shedding control.

---

## 3. Training Data & Quality

| Item | Detail |
|------|--------|
| Load source | Delhi 5-Minute Electricity Demand Dataset (`yug201/delhi-5-minute-electricity-demand-for-forecasting`) |
| Weather source | Hourly co-located meteorological measurements (temp, humidity, dew point) |
| Date range | 2021-01-01 to 2024-12-12 (34,585 clean hourly rows aggregated from 393,440 raw 5-min readings) |
| Granularity | Hourly resampling (`1h` mean aggregation) |
| Demand Range | Minimum = 1,302.08 MW, Maximum = 8,631.53 MW |
| Temperature Range | Minimum = 4.00°C, Maximum = 46.40°C |

### Data Quality & Imputation

| Issue | Observation | Method |
|-------|-------------|--------|
| Missing 5-min readings | 21,569 timestamps out of 415,009 (5.2%) | Aggregated to hourly; short gaps ($\le 3$h) time-interpolated |
| Longest outages | 5 gaps $\ge 24$h (e.g. 2021-07-19 to 2021-07-22: 2d 10h) | Flagged via `load_mw_imputed_flag` |
| Summer Peak Coverage | June–August across 2021–2024: 87.3% to 98.2% present | Zero missing summer seasons |

> **Data Honesty Note on Weather**: We use observed historical weather from the dataset as a proxy for weather forecasts. In a live production system, numerical weather predictions (NWP) would introduce additional forecast variance. This evaluation therefore establishes the empirical upper bound of the model architecture under actual weather conditions.

---

## 4. Features & Leakage Guarantees

All features strictly observe the **09:00 D-1 forecast origin**. No features use data beyond the origin:
- **Thermal**: Cooling Degree Hours ($\text{CDH} = \max(0, \text{temp} - 25^\circ\text{C})$), 24h rolling CDH sum, apparent temperature.
- **Weather Lags**: Temperature at $t-24$h, $t-48$h, 72h rolling mean temperature.
- **Load Lags**:
  - Day-ahead: Load at $t-48$h, $t-72$h, $t-168$h; 7-day rolling mean/max shifted by 48h.
  - Week-ahead: Load at $t-168$h, $t-336$h; 7-day rolling mean/max shifted by 168h.
- **Calendar & Context**: Sin/cos cyclical hour (24h) and day-of-week (7d), weekend indicator, Indian national/state holidays (`holidays` package), Delhi lockdown flag.
- **Excluded**: Target-leaking columns (`moving_avg_3`, raw date counters) are strictly excluded.

---

## 5. Evaluation Protocol

- **Protocol**: 12-month expanding-window walk-forward backtest (2024-01 to 2024-12).
- **Rule**: Never a random train/test split. Each fold trains strictly on all historical data prior to the test month.

---

## 6. Walk-Forward Results

### 6a. Model vs Baselines (12-Fold Walk-Forward Averages)

| Horizon | Model | MAPE (%) | RMSE (MW) | Pinball P10 | Pinball P50 | Pinball P90 |
|---------|-------|----------|-----------|-------------|-------------|-------------|
| **Day-Ahead** | **`lgbm_quantile`** | **16.93%** | **964.70 MW** | 189.63 | 330.42 | 173.46 |
| Day-Ahead | `linear_regression` | 18.34% | 968.33 MW | 177.96 | 353.97 | 187.27 |
| Day-Ahead | `seasonal_naive` | 28.59% | 1574.86 MW | 304.92 | 563.33 | 300.81 |
| **Week-Ahead** | **`lgbm_quantile`** | **21.48%** | **1214.85 MW** | 205.82 | 421.72 | 241.41 |
| Week-Ahead | `linear_regression` | 21.79% | 1090.43 MW | 189.16 | 417.08 | 214.58 |
| Week-Ahead | `seasonal_naive` | 32.05% | 1689.56 MW | 317.33 | 641.79 | 323.71 |

### 6b. Interval Calibration (Nominal Target: 80.0%)

| Horizon | Raw `lgbm_quantile` PICP | Calibrated `lgbm_quantile_calibrated` PICP |
|---------|--------------------------|--------------------------------------------|
| Day-Ahead | 56.78% (Under-coverage) | **80.05%** |
| Week-Ahead | 55.57% (Under-coverage) | **79.99%** |

*Note: Raw quantile regression exhibits empirical under-coverage due to heavy-tailed load swings. Post-hoc conformal calibration shifts the intervals to achieve the exact nominal 80.0% coverage without altering point forecasts.*

### 6c. Peak-Event Classification (>6373.6 MW Threshold)

| Metric | Measured Value |
|--------|----------------|
| Total Test Days | 347 days |
| Actual Peak Days | 84 days (24.2% of test days) |
| Predicted Peak Days | 103 days |
| **Precision** | **65.05%** |
| **Recall** | **79.76%** |
| **F1 Score** | **0.717** |
| Peak Hour MAE | 343.0 minutes |

---

## 7. Known Failure Modes

1. **Sudden Heatwave Inception**: First day of multi-day heatwaves where temperature jumps by $>5^\circ\text{C}$ before buildings accumulate heat.
2. **Post-Festival Rebound**: Days immediately following Diwali and Eid exhibit atypical commercial-to-residential load transitions.
3. **Winter Heating Spikes**: Occasional January cold waves (temperatures $<6^\circ\text{C}$) trigger space-heating demand surges not fully captured by cooling-centric CDH features.
