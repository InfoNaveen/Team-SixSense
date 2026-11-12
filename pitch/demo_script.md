# 3-Minute Demo Script & Judge Q&A Playbook

*Owner: Lead (P1) — CodeSpark 2026*

## Timed Demo Flow (3:00 Total)

### 0:00–0:30 | Problem in One Sentence
- **Say**: "Delhi's power demand swings from 2,000 MW in winter to 8,300 MW on summer evenings. A DISCOM buying power a day or week ahead on a single point forecast—or a flat block—either bleeds money on stranded surplus at night or gets hit by punitive Rs 12/kWh DSM penalties and blackout risk during evening peaks. We built a **probabilistic, peak-aware load forecaster and procurement copilot**."

### 0:30–1:15 | Hero Chart (Dashboard View 1)
- **Click**: Open Streamlit dashboard (`app/app.py`) -> Day-Ahead / Week-Ahead Forecast tab.
- **Show**: Actual load vs. P50 curve with shaded **P10–P90 prediction band** and red **peak-event markers**.
- **Say**: "Notice how our band widens specifically during volatile summer evening ramps instead of staying artificially uniform, and our peak-event classifier flags top-decile peak days and exact peak hours ahead of 09:00 IST D-1."

### 1:15–1:45 | Accuracy Table vs. Baselines & PICP (Dashboard View 2)
- **Click**: Evaluation & Error Heatmap tab.
- **Show**: Per-horizon metrics table (MAPE, RMSE, Pinball loss, PICP, Peak Precision/Recall/F1, Peak Timing Error in minutes) against Seasonal-Naive and Linear Regression, plus the **Hour × Season error heatmap**.
- **Say**: "We never used a random split—everything is evaluated across 12 expanding walk-forward monthly folds covering every Delhi season. Our P10–P90 band achieves ~80% empirical coverage (PICP) after conformal calibration, and beats both baselines on RMSE, Pinball loss, and peak recall."

### 1:45–2:30 | Procurement Panel & Rupee Savings (Dashboard View 4)
- **Click**: Procurement Copilot tab (`results/procurement_week.json`).
- **Show**: 24×7 P90 buy schedule, `p90_exposure_mw` tail buffer, and cost comparison (`cost_model` vs. `cost_flat` and `cost_p50`).
- **Say**: "Translating P90 into an hourly procurement schedule keeps total contracted energy identical to a flat average-P90 block, while slashing both night-time stranded surplus and evening DSM shortfall penalties—delivering simulated weekly savings of **Rs [X] Crore ([Y]%)** with near-zero shortfall hours."

### 2:30–3:00 | Missed-Peak Drill-Down & Honesty Slide (Dashboard View 3)
- **Click**: Missed-Peak Drill-Down tab (`results/missed_peaks.csv`).
- **Show**: Pick a real missed-peak day, show forecast vs. actual and root cause (e.g., sudden heatwave onset, monsoon humidity surge, or shifted festival).
- **Say**: "Where does it fail? Here is a real day we missed and why: [cite actual date & reason]. Next, we would plug in archived NWP ensemble forecasts and intraday storm nowcasts. Thank you!"

---

## Likely Judge Questions & Short Answers

1. **"Did you use a random train/test split?"**
   - "No—strict expanding-window walk-forward validation only across the last 12 monthly blocks, retraining every fold. Zero future load lags past the 09:00 D-1 forecast origin."
2. **"Is your weather data leakage-free?"**
   - "We state transparently in `docs/model_card.md` whether archived weather forecasts or actual weather-as-proxy were used, and how proxy error acts as a lower bound on live NWP uncertainty."
3. **"Are your prediction intervals honest?"**
   - "Yes—we report empirical PICP against the 80% target for a P10–P90 interval, enforce $P_{10} \le P_{50} \le P_{90}$ sorting, and apply conformal residual calibration when raw intervals under-cover."
4. **"Why should we trust your rupee savings number?"**
   - "Every tariff parameter (Rs 5/kWh contracted, Rs 12/kWh DSM, Rs 2.5/kWh stranded) is explicitly marked as an assumption, and `procurement_week.json` includes a full sensitivity grid showing savings hold across DSM rates from Rs 8 to Rs 18/kWh."
5. **"Where does your model fail?"**
   - "Our `missed_peaks.csv` drill-down highlights specific dates of sudden heatwave build-up, high-humidity pre-monsoon nights, and shifted lunar festivals."

