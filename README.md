# Probabilistic Peak-Aware Load Forecasting & Procurement Copilot for Delhi's DISCOM

**CodeSpark 2026** · 12-Hour Hackathon · Team SixSense

An end-to-end, peak-aware probabilistic electricity demand forecasting and procurement optimization system for Delhi's electricity distribution companies (DISCOMs).

---

## Quick Start (Fresh Clone)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download raw 5-minute dataset from Kaggle into data/raw/
python data/download_raw.py

# 3. Run the full pipeline (12-fold expanding walk-forward backtest + peak classifier + procurement)
python pipeline.py data/raw

# 4. Run unit tests
python procurement/test_procurement.py

# 5. Launch the 5-tab operational Streamlit dashboard
streamlit run app.py
```

To run a fast 3-fold smoke test:
```bash
python pipeline.py data/raw --folds 3
```

---

## System Architecture & Flow

Detailed system architecture and data guarantees are documented in [ARCHITECTURE.md](file:///c:/Users/Naveen%20Patil/OneDrive/Documents/DISCOM/ARCHITECTURE.md).

```
[Raw 5-Min Data: data/raw/]
          │
          ▼
[pipeline.py (Master Pipeline)]
    ├── 1. Resample to hourly clean data (data/clean_hourly.parquet)
    ├── 2. Build leakage-free features (09:00 D-1 forecast origin)
    ├── 3. Expanding-window walk-forward backtest (12 monthly folds)
    │      ├── seasonal_naive baseline
    │      ├── linear_regression baseline
    │      ├── lgbm_quantile (P10, P50, P90)
    │      └── lgbm_quantile_calibrated (conformal interval adjustment)
    ├── 4. Peak-event classifier (>6,373.6 MW threshold) & peak hour predictor
    ├── 5. Procurement cost simulation (P90 dynamic schedule vs naive flat)
    └── 6. Export locked contract deliverables to results/
          │
          ▼
[app.py (Streamlit Dashboard)]
    ├── Tab 1: Forecast (Hero P10-P90 band & peak markers)
    ├── Tab 2: Accuracy vs baselines (MAPE / RMSE / Pinball)
    ├── Tab 3: Error heatmap (Hour × Season MAE)
    ├── Tab 4: Missed peaks (Diagnostic drill-down)
    └── Tab 5: Procurement (₹48.45 Cr summer peak savings & tariff sensitivity)
```

---

## Contract Deliverables in `results/`

All modules strictly adhere to the locked file contracts:
- `results/forecasts_day.csv` & `results/forecasts_day.parquet`: Columns `ts, fold, horizon, model, p10, p50, p90, actual`
- `results/forecasts_week.csv` & `results/forecasts_week.parquet`: Columns `ts, fold, horizon, model, p10, p50, p90, actual`
- `results/metrics.csv` & `results/accuracy_table.csv`: Walk-forward performance across models and horizons
- `results/peak_flags.csv` & `results/peak_flags.parquet`: Peak probabilities and actual peak events
- `results/peak_summary.csv`: Precision, Recall, F1 score, and peak timing error
- `results/missed_peaks.csv`: Diagnostic log of missed peak events
- `results/procurement_week.json` & `results/procurement_week.csv`: 24×7 buy schedule, tail risk exposure, and tariff sensitivity grid

---

## Comprehensive Documentation

- [ARCHITECTURE.md](file:///c:/Users/Naveen%20Patil/OneDrive/Documents/DISCOM/ARCHITECTURE.md) — Architectural overview, data contracts, and design principles.
- [docs/model_card.md](file:///c:/Users/Naveen%20Patil/OneDrive/Documents/DISCOM/docs/model_card.md) — Complete model card with empirical 12-fold walk-forward metrics, data card, and limitations.
- [docs/why_peaks_are_hard.md](file:///c:/Users/Naveen%20Patil/OneDrive/Documents/DISCOM/docs/why_peaks_are_hard.md) — 1-page analytical brief detailing non-linear weather sensitivity, festivals, and thermal inertia with real data evidence.
- [docs/procurement_memo.md](file:///c:/Users/Naveen%20Patil/OneDrive/Documents/DISCOM/docs/procurement_memo.md) — Executive procurement memo for the summer peak week detailing ₹48.45 Cr in simulated savings.
- [docs/pitch_outline.md](file:///c:/Users/Naveen%20Patil/OneDrive/Documents/DISCOM/docs/pitch_outline.md) — 3-minute pitch script, slide text, demo click path, and Q&A defense.
- [docs/assumptions.md](file:///c:/Users/Naveen%20Patil/OneDrive/Documents/DISCOM/docs/assumptions.md) — Locked project assumptions (CDH base 25°C, 09:00 D-1 origin, rate cards).
