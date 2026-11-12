# Why Peaks Are Hard

> One-page evidence brief for judges and the model card.

---

## 1. The problem in one number

Delhi's hourly demand swings from ~2,000 MW (January night) to ~8,300 MW (June evening). A model that is "good on average" — MAPE 4% — can still miss a 7,500 MW peak by 300 MW, triggering Rs 36 lakh/hour in DSM penalties (at Rs 12/kWh).

---

## 2. Why peaks are structurally harder to forecast

### 2a. Temperature sensitivity is non-linear

Below ~25°C, cooling load is zero. Above 25°C, every additional degree adds 200–400 MW of air-conditioning demand. At 45°C (a typical Delhi June afternoon), the relationship saturates and becomes noisy. A 2°C weather forecast error at 43°C causes a ~600 MW load error — but the same 2°C error at 20°C causes almost nothing.

> **Evidence**: [FILL from backtest — scatter plot of forecast error vs temperature, showing wider spread above 35°C]

### 2b. Thermal inertia (multi-day heat buildup)

Peak demand doesn't only depend on *today's* temperature. Three consecutive 44°C days produce higher peaks than a single 44°C day after a cool spell, because buildings absorb heat. Our CDH (cooling-degree-hours) feature captures this partially, but multi-day heat accumulation is hard to model with 24–48 h lags alone.

> **Evidence**: [FILL — compare missed peaks vs 3-day cumulative CDH; expect higher miss rate when CDH rises sharply]

### 2c. Festival and holiday effects

| Event | Typical load impact | Training examples |
|-------|-------------------|-------------------|
| Diwali | -15% to -25% (residential exodus + closures) | ~6 (one per year) |
| Eid | -10% to -15% | ~6 |
| Republic Day | -5% | ~6 |
| School summer start | +5% to +10% sustained | ~6 |
| Odd-even driving rules | Unclear | 2–3 events total |

With only 5–6 examples per festival, the model memorises rather than generalises. A festival falling on a different day-of-week or temperature regime than seen in training can cause large errors.

> **Evidence**: [FILL from missed_peaks.csv — list dates near Diwali/Eid where model missed]

### 2d. Regime shifts (COVID, new capacity)

The 2020-21 lockdowns created an unprecedented demand collapse. Post-lockdown recovery followed a different trajectory than pre-2020. The model's expanding training window includes lockdown data, which can bias estimates downward for 2022+ peaks.

> **Evidence**: [FILL — fold-by-fold RMSE showing whether accuracy improves when more post-lockdown data enters training]

---

## 3. What our model does about it

| Problem | Mitigation |
|---------|-----------|
| Temperature non-linearity | CDH feature (base 25°C), apparent temperature, heat index |
| Thermal inertia | Rolling 3-day and 7-day temperature means as features |
| Festival rarity | Festival flag features; still limited by training count |
| Regime shift | Expanding window naturally weights recent data more heavily as training grows |
| Extreme uncertainty | P90 interval captures upside risk; procurement uses P90 schedule |

---

## 4. Our honest failures

> **Populated after backtest results are final.** See [`results/missed_peaks.csv`](file:///c:/Users/Naveen%20Patil/OneDrive/Documents/DISCOM/results/missed_peaks.csv).

| Date | Actual Peak (MW) | P50 Forecast (MW) | P90 Forecast (MW) | Hour Error (min) | Likely Cause |
|------|-------------------|--------------------|--------------------|------------------|-------------|
| [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL: e.g., sudden temp jump / festival] |
| [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |

---

## 5. What we would do with more time

1. **Conformal calibration** of P10/P90 intervals post-hoc to fix PICP drift.
2. **Temperature forecast archive** (Open-Meteo forecast API) instead of actual-as-proxy.
3. **Festival-aware model stacking**: separate sub-model trained only on festival windows.
4. **TFT (Temporal Fusion Transformer)** for interpretable attention over multi-day heat buildup.

---

*This document supports the model card and the pitch's "honest failure analysis" section.*

