# Probabilistic Peak-Aware Load Forecasting & Procurement Copilot for Delhi's DISCOM

> **One-liner:** A probabilistic ML pipeline that forecasts Delhi's electricity demand with uncertainty bands, flags peak-risk days, and recommends cost-optimal procurement strategies — turning gigawatts of guesswork into data-driven decisions.

**Event:** CodeSpark 2026 · 12-hour Hackathon · 60 Teams

---

## Team Structure

| Role | Responsibility |
|------|---------------|
| **Lead** | Architecture, integration, final pipeline, pitch |
| **Data / Dashboard** | Data cleaning, EDA, Streamlit dashboard |
| **Features / Eval** | Feature engineering, walk-forward evaluation, failure analysis |
| **Modeling** | LightGBM quantile models, peak classifier, hyperparameter tuning |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                                   │
│  SLDC / Grid-India (load)  +  Open-Meteo (weather)                      │
└──────────────────────┬──────────────────────────────────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  data/raw/          │
            │  load_raw.csv       │
            │  weather_raw.csv    │
            └────────┬────────────┘
                     │  clean_and_merge.py
                     ▼
            ┌─────────────────────┐
            │  data/               │
            │  clean_hourly.parquet│
            └────────┬────────────┘
                     │  features.py
                     ▼
            ┌─────────────────────┐
            │  Feature Matrix      │
            │  (CDH, heat index,   │
            │   lags, calendar...) │
            └────────┬────────────┘
                     │
            ┌────────┴────────┐
            │                 │
            ▼                 ▼
   ┌────────────────┐  ┌──────────────┐
   │ LightGBM       │  │ Peak Day     │
   │ Quantile Reg.  │  │ Classifier   │
   │ (P10/P50/P90)  │  │              │
   └───────┬────────┘  └──────┬───────┘
           │                  │
           └────────┬─────────┘
                    ▼
           ┌────────────────┐
           │  results/       │
           │  forecasts.csv  │
           │  metrics.json   │
           │  missed_peaks…  │
           └───────┬────────┘
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
 ┌──────────────┐  ┌───────────────────┐
 │  Streamlit   │  │  Procurement      │
 │  Dashboard   │  │  Copilot          │
 │  (app.py)    │  │  (procurement.py) │
 └──────────────┘  └───────────────────┘
```

---

## Folder Structure

```
DISCOM/
├── data/
│   ├── raw/                    # Raw CSVs (load + weather)
│   └── clean_hourly.parquet    # Cleaned, merged hourly data
├── docs/
│   ├── README.md               # This file
│   ├── model_card.md           # Google-style model card
│   ├── assumptions.md          # All assumptions in one place
│   ├── pitch_outline.md        # 5-minute pitch structure
│   └── failure_analysis_template.md  # Honest failure analysis
├── models/                     # Saved model artifacts (.pkl)
├── results/                    # Forecasts, metrics, plots
│   ├── forecasts.csv
│   ├── metrics.json
│   └── missed_peaks.csv
├── src/
│   ├── clean_and_merge.py      # Data cleaning pipeline
│   ├── features.py             # Feature engineering
│   ├── train.py                # Model training
│   ├── evaluate.py             # Walk-forward evaluation
│   ├── procurement.py          # Procurement cost optimizer
│   └── peak_classifier.py     # Peak day classification
├── app.py                      # Streamlit dashboard
├── run_all.py                  # One-click pipeline runner
├── requirements.txt            # Python dependencies
└── config.py                   # Global constants & parameters
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (data → features → train → evaluate → results)
python run_all.py

# 3. Launch the dashboard (optional)
streamlit run app.py
```

---

## Contract Summary — Procurement Tiers

| # | Contract Type | Rate (₹/kWh) | When Used | Risk Level |
|---|--------------|---------------|-----------|------------|
| 1 | **Base PPA** | ₹3.50 | Always-on contracted capacity (3,500 MW) | Lowest |
| 2 | **Day-Ahead Market (DAM)** | ₹5.00 | Procured D-1 based on P50 forecast | Low |
| 3 | **Buffer Procurement** | ₹7.50 | Extra capacity (P50 → P90 gap) for peak-risk days | Medium |
| 4 | **Real-Time / DSM** | ₹12.00 | Emergency balancing when actuals exceed procurement | Highest |
| 5 | **Surplus Sell-back** | ₹2.00 | Revenue from over-procurement sold back to market | Credit |
| 6 | **Net Procurement Cost** | — | Total cost across all tiers minus sell-back | Composite |

---

## Hackathon Checkpoint Timeline

| Checkpoint | Hour | Milestone |
|-----------|------|-----------|
| **H1** | 1 | Data cleaned, `clean_hourly.parquet` ready, EDA notebook done |
| **H3** | 3 | Feature engineering complete, baseline model (naive + linear) trained |
| **H6** | 6 | LightGBM quantile models trained, walk-forward backtest running |
| **H8** | 8 | Evaluation metrics computed, peak classifier tuned, procurement copilot coded |
| **H10** | 10 | Dashboard functional, failure analysis written, model card filled |
| **H12** | 12 | Final polish, pitch rehearsed, code cleaned, submission ready |

---

## License

Hackathon project — CodeSpark 2026. For demonstration and evaluation purposes.
