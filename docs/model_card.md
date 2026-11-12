# Model Card: Delhi DISCOM Probabilistic Peak-Aware Load Forecaster

## 1. Model Details
- **Project**: Probabilistic Peak-Aware Load Forecasting & Procurement Copilot for Delhi's DISCOM (CodeSpark 2026)
- **Forecasting Models**:
  - **Day-Ahead (24h)** & **Week-Ahead (168h)** Quantile Regressors: LightGBM (`lightgbm==4.7.0`, `objective="quantile"`, $\alpha \in \{0.1, 0.5, 0.9\}$), post-processed via per-hour quantile sorting ($P_{10} \le P_{50} \le P_{90}$) and conformal interval calibration if raw PICP deviates from ~80%.
  - **Peak-Event Classifier**: LightGBM binary classifier predicting whether daily max load exceeds the training-window 90th percentile, paired with peak-hour prediction.
  - **Baselines**: (1) Seasonal-Naive (same hour prior week, $t - 168$h), (2) Linear Regression on calendar + temperature features.
- **Owner / Contacts**: Team of 4 (Lead, Data/Dashboard, Features/Eval, Modeling).

## 2. Intended Use & Out-of-Scope Uses
- **Intended Use**: Day-ahead and week-ahead probabilistic power procurement planning, peak-day alerting, and tail-risk budgeting for a distribution utility (DISCOM).
- **Out-of-Scope**: Real-time SCADA dispatch, sub-second grid frequency control, automated breaker switching, or electricity spot-price forecasting.

## 3. Data Sources & Weather Proxy Disclosure
- **Load Data**: Historical Delhi hourly load (SLDC Delhi / Grid-India POSOCO or documented fallback; verify final source in `data/imputation_log.md`).
- **Weather Data**: Open-Meteo historical hourly weather (`temp_c`, `rh_pct`, `apparent_temp_c`).
  - **Honest Proxy Disclosure**: *[SELECT AT H8: Forecast Archive Used / Actual Weather Used as Proxy]*. If actual weather is used as a proxy at forecast time, reported error metrics represent an optimistic lower bound because real numerical weather prediction (NWP) errors at 24h–168h lead times are omitted.
- **Calendar & Events**: Hand-curated Delhi holiday, lunar festival (Diwali, Holi, Eid), odd-even traffic, and COVID-19 lockdown flags.

## 4. Key Modeling Assumptions
1. **Forecast Origin**: `09:00 IST` on `D-1`. All historical load features (lags, rolling means/maxima) strictly precede `09:00 IST` on `D-1` to eliminate target leakage.
2. **CDH Base**: `25 °C` (`max(0, temp_c - 25)`).
3. **Peak Day Threshold**: Computed strictly from the **90th percentile of daily maxima inside each fold's training window** (never using test-set distribution).
4. **Timezone**: `Asia/Kolkata` (IST, no DST); missing/duplicate timestamps are checked and logged rather than silently dropped.

## 5. Evaluation Protocol (Walk-Forward Only)
- **Zero Random Splits**: Evaluated strictly via expanding-window walk-forward validation across the **last 12 monthly test blocks** so every Delhi season (winter trough ~2,000 MW, spring, extreme summer peak ~8,300 MW, monsoon humidity) is tested out-of-sample.
- **Results Summary** *(auto-populated from `results/metrics.csv` at H8/H10)*:
  - Day-ahead (24h) MAPE / RMSE / Pinball / PICP vs. Seasonal-Naive & Linear Regression: `[PENDING H6/H8 RESULTS]`
  - Week-ahead (168h) MAPE / RMSE / Pinball / PICP vs. Seasonal-Naive & Linear Regression: `[PENDING H6/H8 RESULTS]`
  - Peak-Day Precision / Recall / F1 & Peak-Timing Error (minutes): `[PENDING H6/H8 RESULTS]`

## 6. Known Limitations & Failure Modes
1. **Sudden Heatwaves / Unprecedented Maxima**: Tree-based models (LightGBM) cannot extrapolate beyond historical load-temperature support without explicit linear/trend terms or residual calibration.
2. **Shifted Lunar Festivals**: Diwali, Eid, and Holi shift across Gregorian dates each year; industrial/residential drop-offs also depend on day-of-week adjacency.
3. **Thermal Inertia Saturation**: Multi-day heat spells (`>43 °C` for 72h+) cause structural wall heat retention that simple 24h temperature lags underestimate.
4. **Data Imputation Artifacts**: Outage zeros or meter drop-outs filled via interpolation (`load_mw_imputed_flag == True`) can dampen sharp peak spikes if not flagged carefully.
5. **Tariff Assumptions in Procurement**: Procurement rupee savings rely on explicit placeholder tariff assumptions (`Rs 5.0` contracted / `Rs 12.0` DSM / `Rs 2.5` stranded per kWh) rather than confidential DISCOM PPA contracts.

