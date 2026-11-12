# Feature Importance Analysis & Demand Drivers

**Model**: Quantile LightGBM Forecaster (`lgbm_quantile`)  
**Evaluation**: Average Split Gain across P10, P50, and P90 models  
**Artifact**: [`models/feature_importance.csv`](file:///c:/Users/Laptop/Desktop/hackthon/models/feature_importance.csv)

---

## 1. Top Predictive Drivers

| Feature Class | Key Variables | Typical Gain Rank | Physical & Behavioral Rationale |
| :--- | :--- | :--- | :--- |
| **Historical Load Lags** | `load_lag_168` (t-168h), `load_lag_48` (t-48h) | #1 – #2 | **Strongest anchor**: Captures Delhi's weekly periodicity (same hour last week) and day-ahead baseline persistence while strictly respecting the 09:00 D-1 forecast origin. |
| **Diurnal Structure** | `hour` (0–23) | #3 | Captures the fixed human and industrial daily cycle: morning ramp (6–10 AM) and evening lighting/residential peak (7–11 PM). |
| **Rolling Load Baselines** | `rolling_7d_mean_load`, `rolling_7d_max_load` | #4 | Tracks medium-term structural demand level shifts (e.g., transition from spring to pre-monsoon heat). |
| **Weather & Cooling Demand** | `temp_c`, `rh_pct`, `apparent_temp_c`, `temp_lag_24`, `cdh` | #5 – #7 | **Primary driver of peak volatility**: Delhi load swings from ~2,000 MW in winter to 8,300+ MW in summer due to AC cooling demand. Relative humidity interacts strongly with temperature (heat index), worsening nighttime cooling load. |
| **Calendar & Day Type** | `dayofweek`, `is_weekend` | #8 – #9 | Captures commercial/industrial weekly shutdowns on Saturdays and Sundays (~300–600 MW dip). |
| **Festivals & Anomalous Days** | `is_holiday`, `festival_name` | Crucial on Upper Quantiles | Significant effect on tail uncertainty (P90) and evening peak shifts (e.g., Diwali lighting surge, Eid shifting cooking/lighting hours). |

---

## 2. Quantile-Specific Driver Asymmetry

1. **Median ($P_{50}$) vs. Upper Tail ($P_{90}$)**:
   - For $P_{50}$ (point forecast), load lags (`load_lag_168`, `load_lag_48`) dominate because the expected trajectory closely mimics historical weekly schedules.
   - For $P_{90}$ (procurement safety and peak hedge), weather sensitivity (`temp_c`, `apparent_temp_c`, `temp_lag_24`) and festival flags carry proportionally higher relative gain. An extra 2°C in summer drives exponential cooling uptake, shifting the tail distribution upward.
2. **Thermal Inertia Effect**:
   - `temp_lag_24` and 72-hour rolling temperature capture building heat accumulation ("heat island effect"). When 3 consecutive hot days occur, nighttime temperatures stay elevated, preventing residential ACs from turning off.
3. **Festival Shifts**:
   - Holidays reduce daytime commercial load but significantly elevate evening residential peaks and create shifting peak hours (e.g. 21:00 to 23:00 on festival nights).

