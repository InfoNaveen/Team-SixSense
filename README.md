# Probabilistic Peak-Aware Load Forecasting & Procurement Copilot (Delhi DISCOM)

CodeSpark 2026 | Domain 4 PS 2 | Team of 4

## Overview
An end-to-end, leakage-free probabilistic load forecasting and day-ahead/week-ahead power procurement copilot for Delhi's DISCOM:
- **Hourly P10 / P50 / P90 load forecasts** for Day-Ahead (24h) and Week-Ahead (168h) horizons.
- **Peak-event classifier** flagging top-decile peak days and expected peak hour.
- **Walk-forward (rolling-origin) backtest** across the last 12 monthly blocks (zero random splits).
- **Procurement advisor** converting P90 intervals into an hourly buy schedule and quantifying rupee savings vs. a naive flat schedule.
- **Offline Streamlit + Plotly dashboard** reading solely from `results/`.

## Strict Folder Ownership
| Role | Owns | Hands off to |
| --- | --- | --- |
| **Lead (P1)** | `procurement/`, `docs/`, `integration/`, `pitch/`, `results/procurement_week.json` | Everyone (contracts, integration, memo, card, pitch) |
| **Data & Dashboard (P2)** | `data/`, `app/` | Features (`data/clean_hourly.parquet`), Lead (dashboard) |
| **Features & Eval (P3)** | `features/`, `eval/`, `results/metrics.csv`, `results/missed_peaks.csv` | Modeling (features + harness), Dashboard & Lead |
| **Modeling (P4)** | `models/`, `results/forecasts_*.parquet`, `results/peak_flags.parquet` | Dashboard & Lead |

See [docs/interface_contracts.md](docs/interface_contracts.md) for the locked schemas and checkpoints.

## Quickstart (Fresh Clone)
```bash
pip install -r requirements.txt

# 1. Validate whatever contract artifacts exist so far
python integration/validate_contracts.py --mode available

# 2. Run procurement simulation on dummy stub data (before H6 real forecasts exist)
python -m procurement.procurement --dummy --out results/procurement_week.json

# 3. Run full integration pipeline (once Data, Features, and Modeling modules are ready)
python integration/run_pipeline.py

# 4. Launch dashboard (Data/P2 owns app/app.py)
streamlit run app/app.py
```

## Key Assumptions
- **Forecast origin**: 09:00 IST on D-1 (day-ahead covers 24h of day D; week-ahead covers 168h from 00:00 of day D).
- **Cooling-degree-hours (CDH) base**: 25 °C.
- **Peak day definition**: Daily maximum load > 90th percentile of daily maxima in the training window only.
- **Procurement cost rates (marked assumptions)**: Contracted day-ahead Rs 5.0/kWh; DSM shortfall Rs 12.0/kWh; stranded surplus Rs 2.5/kWh.

