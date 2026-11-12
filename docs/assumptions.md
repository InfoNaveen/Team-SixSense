# Assumptions & Design Decisions

> All assumptions made in this project are documented here for transparency and reproducibility. Judges and reviewers should consult this file to understand the choices behind our pipeline.

---

## 1. Forecast Origin

- **Assumption:** All forecasts are issued at **09:00 IST on D-1** (the day before the target day).
- **Rationale:** This is a standard day-ahead planning horizon for DISCOM procurement. By 09:00, the previous day's full load data and morning weather observations are available.
- **Implication:** Any feature using load or weather data must only use observations *before* 09:00 on D-1. This means:
  - `load_lag_24h` for hour H on day D uses load at hour H on day D-1 (available only if H < 09:00 on D-1; otherwise we use D-2).
  - In practice, the safest lag is **≥24 hours** from the forecast target.

---

## 2. Cooling Degree Hours (CDH)

- **Base temperature:** **25°C**
- **Formula:** `CDH = max(0, temperature_2m − 25)`
- **Rationale:** 25°C is a commonly used cooling threshold for Indian climates. Below this, air conditioning load is minimal. Above it, AC demand rises approximately linearly.
- **Source:** Standard practice in Indian power-sector demand studies.

---

## 3. Peak Day Definition

- **Definition:** A day is classified as a **peak day** if its daily maximum load exceeds the **90th percentile** of all daily maxima in the *training set*.
- **Rationale:** The 90th percentile captures extreme demand days (roughly ~36 days/year) without being so rare as to have too few positive examples for classification.
- **Threshold recalculation:** The threshold is recalculated at each walk-forward step using only training data (no test leakage).

---

## 4. Weather Data — Actual as Proxy

- **Assumption:** We use **actual observed weather** (from Open-Meteo) in place of weather forecasts.
- **Rationale:** Historical weather *forecast* archives with hourly resolution for Delhi are not freely available for the full 2019–2025 period.
- **Impact:** Our forecast accuracy represents an **upper bound**. In production, weather forecast errors would degrade performance.
- **Mitigation:** This is stated transparently in the model card, pitch, and results interpretation.

---

## 5. Timezone

- **Timezone:** **IST (UTC+05:30)**
- **No daylight saving:** India does not observe DST, so no clock-change adjustments are needed.
- **All timestamps** in the pipeline are in IST. Raw data is converted to IST during cleaning if sourced in UTC.

---

## 6. Procurement Rates

| Contract Tier | Rate (₹/kWh) | Description |
|--------------|---------------|-------------|
| Base PPA (long-term) | **₹3.50** | Always-on contracted capacity, cheapest tier |
| Day-Ahead Market (DAM) | **₹5.00** | Procured D-1 based on P50 forecast |
| Buffer Procurement | **₹7.50** | Extra capacity for gap between P50 and P90 on peak-risk days |
| Real-Time / DSM | **₹12.00** | Emergency procurement when actuals exceed all planned procurement |
| Surplus Sell-back | **₹2.00** | Revenue recovered by selling excess procurement back to market |

- **Rationale:** These are simplified, illustrative rates that reflect the general cost hierarchy in Indian power markets (base < DAM < buffer < real-time). Actual DISCOM tariffs are more complex.
- **Base contracted capacity:** **3,500 MW** (approximate scale for a Delhi DISCOM).

---

## 7. Walk-Forward Backtest Design

- **Strategy:** Expanding window (training data grows over time; no data is dropped).
- **Test blocks:** **Monthly** — each calendar month is a test block.
- **Number of folds:** **12** (last 12 months of available data).
- **Retraining:** Model is retrained once at the start of each fold using all prior data.
- **No shuffling:** Temporal order is strictly preserved.
- **No validation set tuning within walk-forward:** Hyperparameters are fixed before the backtest to prevent test leakage.

---

## 8. No Future Leakage

- **Strict rule:** At forecast time (09:00 D-1), the model may only use:
  - Load data from **D-2 and earlier** (full days only, to be safe).
  - Weather data from **D-2 and earlier** (actual observations).
  - Weather data for **D** is used as *proxy for forecasts* (see Assumption 4).
- **Lag features:** All load/temperature lags are computed relative to the target hour and verified to be ≥24 hours in the past from the forecast origin.

---

## 9. Missing Data Imputation

- **Linear interpolation:** Applied for gaps of **≤3 consecutive hours**.
- **Gaps > 3 hours:** Flagged and optionally excluded from training (not imputed, to avoid injecting artificial patterns).
- **Rationale:** Short gaps (1–3 hours) are common in SCADA data and can be safely interpolated. Longer gaps may indicate outages or data failures where interpolation would be misleading.

---

## 10. Additional Assumptions

| Topic | Assumption |
|-------|-----------|
| **Holiday calendar** | Based on Indian national holidays + Delhi state holidays. Manual list curated for 2019–2025. |
| **COVID flag** | Binary flag for national lockdown period (25 Mar 2020 – 31 May 2020). Extended restrictions (Jun–Dec 2020) are *not* separately flagged. |
| **Peak season** | April–September (summer months when AC-driven demand dominates). |
| **Model family** | LightGBM chosen for speed, interpretability (feature importance), and native quantile support. No deep learning due to hackathon time constraints. |
| **Quantile levels** | P10, P50, P90 chosen to give an 80% prediction interval (industry standard for probabilistic forecasting). |
| **No ensemble** | Single LightGBM per quantile (not an ensemble of different model families) to keep the pipeline simple and debuggable within 12 hours. |

---

*Last updated: November 2026*
