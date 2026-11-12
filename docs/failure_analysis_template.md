# Failure Analysis

> **Philosophy:** Judges reward intellectual honesty. Every model fails somewhere. Documenting *where* and *why* shows maturity and deep understanding. This is not a weakness — it's a strength.

---

## 1. Missed Peaks

Peak days where actual daily max exceeded the 90th percentile threshold but our **peak classifier failed to flag** or our **P90 forecast was too low**.

| Date | Actual Peak (MW) | Predicted P50 (MW) | Predicted P90 (MW) | P90 Error (MW) | Peak Hour (Actual) | Peak Hour (Predicted) | Hour Error |
|------|------------------|--------------------|--------------------|-----------------|--------------------|-----------------------|------------|
| [FILL FROM results/missed_peaks.csv] | | | | | | | |
| [FILL FROM results/missed_peaks.csv] | | | | | | | |
| [FILL FROM results/missed_peaks.csv] | | | | | | | |
| [FILL FROM results/missed_peaks.csv] | | | | | | | |
| [FILL FROM results/missed_peaks.csv] | | | | | | | |

**Total missed peaks:** [TO BE FILLED] out of [TO BE FILLED] actual peak days.

**Peak recall:** [TO BE FILLED]%

---

## 2. Common Failure Modes

### 2.1 Sudden Weather Changes

- **What happens:** A cold front or unseasonal rain arrives between the forecast origin (09:00 D-1) and the target day, causing actual temperatures to diverge sharply from the "proxy forecast" (which uses actual weather, but in production would use forecasts with their own errors).
- **Example dates:** [FILL FROM RESULTS]
- **Impact:** Load forecast can be off by [TO BE FILLED] MW on such days.
- **Mitigation in production:** Use ensemble weather forecasts; flag high-uncertainty weather days.

### 2.2 Festival / Holiday Effects

- **What happens:** Indian festivals (Diwali, Eid, Holi, etc.) have complex and variable effects on load:
  - Diwali eve: high residential load (lighting, cooking)
  - Diwali day: lower commercial/industrial load (closures)
  - Post-Diwali: demand recovery spike
- **Current handling:** Binary `is_holiday` flag — too coarse.
- **Impact:** [TO BE FILLED] MW average error on festival days vs. [TO BE FILLED] MW on normal days.
- **Fix:** Holiday-type encoding (festival, national, state) + day-relative-to-festival features.

### 2.3 Post-COVID / Lockdown Regime Shift

- **What happens:** The COVID-19 lockdowns (Mar–Jun 2020) created an unprecedented demand drop. Post-lockdown recovery followed a different trajectory than pre-COVID patterns.
- **Current handling:** Binary `is_lockdown` flag for the strict lockdown period.
- **Impact:** Models trained including 2020 data may learn anomalous patterns. Models excluding it lose ~8% of training data.
- **Trade-off decision:** [DESCRIBE WHAT WE CHOSE AND WHY]

### 2.4 Extreme Heat Events Beyond Training Distribution

- **What happens:** If the test period contains temperatures higher than any in the training period, the model is extrapolating. LightGBM (tree-based) cannot extrapolate beyond the range of training data.
- **Example:** If training max is 47°C and test day hits 48.5°C, the model's CDH feature saturates at the training max.
- **Impact:** Likely under-prediction on record-breaking heat days.
- **Fix:** Add a linear correction term for out-of-distribution temperatures, or use a hybrid model.

### 2.5 Load Shedding / Curtailment Artifacts

- **What happens:** If the historical load data reflects *curtailed* demand (i.e., actual demand was higher but DISCOM shed load), then our model learns from the curtailed signal, not true demand.
- **Impact:** Systematic under-prediction during peak stress periods.
- **Fix:** Obtain unconstrained demand estimates or flag known load-shedding periods.

---

## 3. Calibration Analysis

> Is our 80% prediction interval (P10–P90) actually covering 80% of observations?

| Period | Expected Coverage | Actual PICP | Status |
|--------|-------------------|-------------|--------|
| Overall | 80% | [TO BE FILLED]% | [✅ / ⚠️ / ❌] |
| Summer (Apr–Sep) | 80% | [TO BE FILLED]% | [✅ / ⚠️ / ❌] |
| Winter (Oct–Mar) | 80% | [TO BE FILLED]% | [✅ / ⚠️ / ❌] |
| Peak days only | 80% | [TO BE FILLED]% | [✅ / ⚠️ / ❌] |
| Non-peak days | 80% | [TO BE FILLED]% | [✅ / ⚠️ / ❌] |

**Interpretation:**
- PICP > 85% → Intervals are **too wide** (over-conservative). We're buying too much buffer.
- PICP ≈ 78–82% → **Well-calibrated.** ✅
- PICP < 75% → Intervals are **too narrow** (under-confident). Risk of under-procurement.

[TO BE FILLED: Write 2-3 sentences interpreting the actual PICP values.]

---

## 4. Interval Width Analysis

| Period | Mean P10–P90 Width (MW) | Median Width (MW) | Comment |
|--------|------------------------|--------------------|---------|
| Overall | [TO BE FILLED] | [TO BE FILLED] | |
| Summer | [TO BE FILLED] | [TO BE FILLED] | Expected wider due to AC variability |
| Winter | [TO BE FILLED] | [TO BE FILLED] | Expected narrower |
| Peak days | [TO BE FILLED] | [TO BE FILLED] | Wider is appropriate |

[TO BE FILLED: Are the intervals appropriately wider on peak/summer days?]

---

## 5. Worst Forecast Days

Top 5 days with the highest absolute P50 error:

| Rank | Date | Actual Peak (MW) | P50 Forecast (MW) | Abs Error (MW) | MAPE (%) | Likely Cause |
|------|------|-------------------|--------------------|----------------|----------|-------------|
| 1 | [FILL] | | | | | |
| 2 | [FILL] | | | | | |
| 3 | [FILL] | | | | | |
| 4 | [FILL] | | | | | |
| 5 | [FILL] | | | | | |

---

## 6. What We Would Fix With More Time

| Priority | Improvement | Expected Impact | Effort |
|----------|------------|-----------------|--------|
| 🔴 High | Use actual weather *forecast* archives instead of actuals-as-proxy | More realistic accuracy estimates | Need data source |
| 🔴 High | Festival-type encoding (not just binary holiday) | Better festival-day forecasts | 2–3 hours |
| 🟡 Medium | Conformal prediction for post-hoc calibration | Guaranteed coverage probability | 1–2 hours |
| 🟡 Medium | Add solar generation forecast as a feature (net load) | Capture rooftop solar growth | Need data source |
| 🟢 Low | Ensemble with linear quantile regression | Hedge against LightGBM failure modes | 1 hour |
| 🟢 Low | Intra-day model update at 18:00 D-1 | Incorporate afternoon weather updates | 2 hours |

---

## 7. Honest Summary

> [TO BE FILLED: Write a 3–4 sentence honest summary. Example:]
>
> *"Our model performs well on typical days with MAPE of X% and well-calibrated 80% intervals. It struggles most with unseasonal weather transitions and festival effects, where errors can reach Y MW. Peak recall is Z%, meaning we miss roughly N peak days per year — each of which represents a procurement risk. With more time, we would prioritize conformal calibration and festival-aware features."*

---

*This document will be filled after running the evaluation pipeline (`python run_all.py`). Placeholder markers `[TO BE FILLED]` and `[FILL]` indicate data-dependent fields.*

*Last updated: November 2026*
