# Pitch Deck Outline (7 Slides — 3 Minutes)

*Owner: Lead (P1) — CodeSpark 2026*

1. **Slide 1 — Title & The DISCOM Tail-Risk Problem**
   - Delhi load swings 2,000 MW (winter) to 8,300 MW (summer).
   - Point forecasts hide tail risk: under-buying triggers Rs 12/kWh DSM & blackouts; over-buying triggers stranded capacity charges.

2. **Slide 2 — Leakage-Free Data & Physics-Informed Features**
   - Canonical IST hourly load + Open-Meteo weather + Delhi festival/event calendar.
   - Forecast origin fixed at **09:00 IST on D-1**: strict lag gating, CDH (base 25 °C), Heat Index, 72h thermal inertia.

3. **Slide 3 — Walk-Forward Evaluation & Baseline Comparison**
   - Expanding window across 12 monthly folds (zero random splits).
   - Per-horizon (24h & 168h) MAPE, RMSE, Pinball loss, and % improvement over Seasonal-Naive & Linear Regression.

4. **Slide 4 — Peak-Event Classifier & Honest Uncertainty (PICP)**
   - Top-decile peak-day classifier (Precision / Recall / F1) + peak-hour prediction (timing error in minutes).
   - Non-crossing P10/P50/P90 + conformal calibration hitting ~80% PICP.

5. **Slide 5 — Procurement Copilot & Rupee Savings Story**
   - Hourly P90 buy schedule vs. P50 and Flat average-P90 schedule.
   - Weekly cost breakdown, shortfall hours avoided, rupee savings, and tariff sensitivity table.

6. **Slide 6 — Honest Failure Analysis: Why Peaks Are Hard**
   - Evidence from `missed_peaks.csv`: sudden heatwaves, humidity spikes, shifted festivals.
   - Transparent disclosure of weather proxy & next steps (NWP ensembles, storm nowcasting).

7. **Slide 7 — Reproducible Architecture & Live Demo**
   - Modular 4-owner contract pipeline + offline Streamlit dashboard (`results/` only).

