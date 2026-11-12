# Probabilistic Peak-Aware Load Forecasting & Procurement Copilot for Delhi's DISCOM

**CodeSpark 2026** · 12-Hour Hackathon · Team of 4

An end-to-end, peak-aware probabilistic electricity demand forecasting and procurement optimization system for Delhi's electricity distribution companies (DISCOMs).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Check teammate contracts status
python scripts/check_contracts.py

# 3. Test procurement engine with dummy forecasts
python stubs/test_procurement_smoke.py

# 4. Run full pipeline (runs whatever modules teammates have delivered)
python run_all.py

# 5. Launch Streamlit dashboard
streamlit run app/app.py
```

---

## Documentation

Comprehensive project documentation is available in [`docs/`](file:///c:/Users/Naveen%20Patil/OneDrive/Documents/DISCOM/docs/):

- [docs/README.md](file:///c:/Users/Naveen%20Patil/OneDrive/Documents/DISCOM/docs/README.md) — Architecture, contracts, pipeline overview, and directory structure.
- [docs/assumptions.md](file:///c:/Users/Naveen%20Patil/OneDrive/Documents/DISCOM/docs/assumptions.md) — Complete list of all operational and modeling assumptions (CDH base 25°C, 09:00 D-1 origin, rate cards, etc.).
- [docs/model_card.md](file:///c:/Users/Naveen%20Patil/OneDrive/Documents/DISCOM/docs/model_card.md) — Model card documenting data, quantile regression setup, evaluation, and limitations.
- [docs/pitch_outline.md](file:///c:/Users/Naveen%20Patil/OneDrive/Documents/DISCOM/docs/pitch_outline.md) — 5-minute timed pitch script with slides and demo cues for the hackathon judges.
- [docs/failure_analysis_template.md](file:///c:/Users/Naveen%20Patil/OneDrive/Documents/DISCOM/docs/failure_analysis_template.md) — Honest post-hoc failure analysis framework for missed peak demand days.

---

## Team Ownership & Architecture

Strict folder ownership ensures zero merge conflicts:

| Folder | Owner | Role & Deliverables |
|---|---|---|
| `procurement/` | **Lead** | Procurement cost simulation (`simulate_week`), weekly optimizer |
| `docs/` | **Lead** | Model card, assumptions, failure analysis, pitch outline |
| Root & `scripts/` | **Lead** | `run_all.py`, contract validator, stubs, integration harness |
| `data/` | **Data (P2)** | Data ingestion, cleaning (`clean_hourly.parquet`), calendar flags |
| `features/` | **Features (P3)** | Feature matrix generation (`build_features`), weather/load lags |
| `eval/` | **Features (P3)** | Expanding-window walk-forward backtest (`get_folds`, `run_backtest`) |
| `models/` | **Modeling (P4)** | Quantile LightGBM models (`fit_predict` for p10/p50/p90), peak classifier |
| `app/` | **Data (P2)** | Streamlit dashboard (reads `results/` only) |
| `results/` | **Shared** | Contractual parquet, CSV, and JSON deliverables |

