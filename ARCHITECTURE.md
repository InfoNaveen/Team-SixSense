# System Architecture — Delhi DISCOM Load Forecasting & Procurement Copilot

**CodeSpark 2026** · Probabilistic Peak-Aware Energy Analytics & Hedging Engine

---

## 1. High-Level Architecture Overview

```
               [Raw 5-Min CSV: Power Demand + Weather]
                                 │
                                 ▼
                     [data/download_raw.py]
                                 │
                                 ▼
                      [pipeline.py (Master)]
                                 │
     ┌───────────────────────────┴───────────────────────────┐
     │                                                       │
     ▼                                                       ▼
[Data Cleaning & Imputation]                [Strict Leakage-Free Features]
- Hourly aggregation                        - Forecast Origin: 09:00 D-1
- Outlier & gap detection                   - CDH (Base 25°C) & Apparent Temp
- Contract: clean_hourly.parquet            - Weather lags (t-24h, t-48h, roll-72h)
                                            - Load lags: day (≥48h), week (≥168h)
                                            - Calendar & Cyclical sin/cos encodings
                                 │
                                 ▼
             [Expanding-Window Walk-Forward Evaluation]
             - 12 Monthly test blocks (Never a random split)
             - Models:
                 * Seasonal Naive (t-168h baseline)
                 * Linear Regression (Ridge baseline)
                 * LightGBM Quantile (P10, P50, P90 pinball loss)
                 * Conformal Calibrator (PICP nominal ~80%)
                                 │
     ┌───────────────────────────┴───────────────────────────┐
     │                                                       │
     ▼                                                       ▼
[Peak-Event Classification]                 [Risk-Aware Procurement Copilot]
- Top-decile threshold (>6373.6 MW)         - Schedule = P90 (Risk-Averse)
- Peak day & hour prediction                - Cost = Sched*PPA + Under*DSM + Over*Stranded
- Missed peak diagnostic tracking           - Benchmark vs Naive Flat Schedule
                                 │
                                 ▼
             [Locked Results Contracts in results/]
             - forecasts_{day,week}.csv / .parquet
             - metrics.csv & accuracy_table.csv
             - peak_flags.csv / .parquet & peak_summary.csv
             - missed_peaks.csv
             - procurement_week.json & procurement_week.csv
                                 │
                                 ▼
             [Streamlit Operational Dashboard: app.py]
             (5 Offline Tabs reading results/ ONLY)
             1. Forecast (Hero P10-P90 band & peak markers)
             2. Accuracy vs baselines (MAPE / RMSE per horizon)
             3. Error heatmap (Hour × Season MAE)
             4. Missed peaks (Drill-down into failure dates)
             5. Procurement (Weekly savings & tariff sensitivity)
```

---

## 2. Component Directory Structure & Team Boundaries

| Component | Primary File | Role & Ownership |
|---|---|---|
| **Pipeline Runner** | `pipeline.py` | Single source of truth: data prep, features, backtest, metrics, procurement. |
| **Data Ingestion** | `data/download_raw.py` | Downloads 5-min Kaggle dataset into `data/raw/`. |
| **Clean Dataset** | `data/clean_hourly.parquet` | Contract 1: Hourly IST timestamped demand, weather, and calendar flags. |
| **Dashboard** | `app.py` | Single source of truth: 5-tab offline Streamlit visualizer reading `results/`. |
| **Procurement Engine** | `procurement/procurement.py` | Cost simulation, P90 vs Flat comparison, and 2D rate sensitivity. |
| **Unit Tests** | `procurement/test_procurement.py` | 7 known-answer unit tests verifying procurement math and schemas. |
| **Model Documentation** | `docs/model_card.md` | Model card covering architecture, data, metrics, and limitations. |
| **Analytical Brief** | `docs/why_peaks_are_hard.md` | 1-page empirical analysis of non-linear weather, festivals, and thermal inertia. |
| **Procurement Memo** | `docs/procurement_memo.md` | Weekly executive memo for representative summer peak week. |

---

## 3. Strict Machine Learning & Data Contracts

1. **Temporal Integrity**: All training and testing use expanding-window walk-forward validation with monthly test blocks. Random splitting is strictly prohibited.
2. **Forecast Origin Enforcement**: The forecast origin is fixed at 09:00 on D-1. Features for day-ahead dispatch (D 00:00 to 23:00) only access load information available prior to the origin (load lags $\ge 48$h). Week-ahead dispatch accesses load lags $\ge 168$h.
3. **Target Leakage Prohibition**: Raw calendar index columns (`year`, `month`, `day`, `hour`, `minute`) and moving target averages (`moving_avg_3`) are strictly barred from the feature matrix.
4. **Quantile Crossing Resolution**: The LightGBM quantile models enforce strict monotonicity ($P_{10} \le P_{50} \le P_{90}$) via post-prediction sorting.
5. **Dashboard Isolation**: `app.py` reads exclusively from `results/` artifacts and never imports modeling, feature, or training logic.
