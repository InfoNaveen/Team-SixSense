# Speaker Notes — Round 1 Pitch
**Project**: Probabilistic, Peak-Aware Load Forecasting & Procurement Copilot for Delhi's DISCOM
**Format**: 6 slides, 30 seconds spoken per slide (3 minutes total)

## Slide 1: The problem: the cost of not knowing the tail risk
*Target Duration: ~30 seconds*

> Delhi's power demand fluctuates violently between two thousand megawatts in winter and over eight thousand in summer. Standard point forecasts hide the tail risk. When DISCOMs under-buy, they pay punitive DSM penalties of twelve rupees per unit. When they over-buy, capacity sits stranded at two-point-five rupees. Certainty is an illusion; managing the spread is where money is saved.

---

## Slide 2: Our solution: three parts: probabilistic forecast, peak-day detector, procurement advisor
*Target Duration: ~30 seconds*

> We built an integrated copilot with three interconnected engines. First, a quantile gradient booster forecasting P10, P50, and P90 demand day-ahead and week-ahead. Second, a binary peak classifier alerting operators when daily peaks breach the 90th percentile. Third, an automated procurement advisor translating the P90 trajectory into an optimal dispatch schedule.

---

## Slide 3: Architecture: boxes-and-arrows pipeline
*Target Duration: ~30 seconds*

> Our architecture enforces strict modular separation. We ingest Delhi five-minute grid telemetry and meteorological data, engineer temperature cooling degrees and calendar lags, and train walk-forward LightGBM quantile models without data leakage. The downstream modules consume standardized parquet contracts to power metrics, peak classification, procurement optimization, and the interactive dashboard.

---

## Slide 4: The hero visual: actual vs P50 with P10-P90 band
*Target Duration: ~30 seconds*

> This hero chart illustrates week-ahead hourly performance. The shaded blue band represents the P10-to-P90 uncertainty interval, and black traces actual load. Notice how the P90 envelope expands proactively during the critical seven-to-eleven PM evening peak ramp. The red crosses mark daily peaks, successfully bounded by our probabilistic corridor to eliminate supply deficits.

---

## Slide 5: The money story: cost of P90 vs flat schedule
*Target Duration: ~30 seconds*

> Here is the financial impact: across one sample week, a conventional flat schedule incurs ₹581.8 crore in total power outlay. The P90-shaped schedule reduces this to ₹533.4 crore, delivering ₹48.5 crore in weekly savings, or 8.3 percent. By minimizing high-penalty real-time DSM exposure, peak-aware procurement protects DISCOM operating margins.

---

## Slide 6: Why you can trust it: validation, baselines, and honest limits
*Target Duration: ~30 seconds*

> Our pipeline relies entirely on walk-forward temporal splits evaluated against seasonal-naive and linear baselines, achieving 19.2% MAPE and 68.1% interval coverage. Transparently, our current limitations are: weather inputs serve as a forecast proxy, intervals require conformal calibration, and peak timing has one-hour granularity. Immediate next steps are conformal re-calibration and live weather API integration.

---
