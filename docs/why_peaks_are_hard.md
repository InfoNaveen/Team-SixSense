# Why Peaks Are Hard: Failure Analysis & Physical Drivers in Delhi's Grid

*One-page technical note (Deliverable D5) — CodeSpark 2026*

## 1. Why Average Accuracy (MAPE / RMSE) Lies to a DISCOM
Delhi's grid demand swings by **>4× across the year**—from roughly **2,000 MW on winter nights** to **~8,300 MW on extreme summer evenings**. A point model can post an impressive 3–4% annual MAPE while failing on the **top 10% of peak days** that drive 80% of DSM penalty risk, transformer overloads, and rotational load-shedding.

## 2. Three Physical & Calendar Drivers of Peak Misses

### A. Non-Linear Weather & Humidity Amplification
- Below ~25 °C, cooling load is near zero; above 25 °C, air-conditioning demand rises sharply and non-linearly with **Cooling-Degree-Hours (CDH)** and **Heat Index (apparent temperature)**.
- During Delhi's pre-monsoon and monsoon transition (July–August), dry-bulb temperature may drop from 45 °C to 37 °C while **relative humidity spikes above 70%**, pushing apparent temperature and latent AC compressor load higher than temperature alone predicts.

### B. Multi-Day Thermal Inertia (Heat Build-Up)
- Building envelopes and urban concrete retain heat across consecutive hot nights (high minimum temperatures at 02:00–05:00 IST).
- Day 1 of a heatwave at 43 °C behaves very differently from **Day 4 of consecutive 43 °C heat** with 32 °C overnight lows, driving evening peaks (22:00–23:30 IST) when residential AC load stacks on top of commercial cooling.
- Lagged 48–72h rolling mean temperatures capture much of this inertia, but abrupt heatwave onsets or sudden evening dust storms/thunderstorms (*andhi*, causing 1,000+ MW sudden drops) remain hardest to predict 24–168h out.

### C. Shifted Festivals & Regime Anomalies
- **Lunar / Cultural Calendar**: Major load anomalies (Diwali lighting + commercial shutdown, Holi midday drop, Eid) shift by 10–15 days each year on the Gregorian calendar, reducing the usefulness of raw same-day-last-year lookups (`t - 8760h`).
- **Structural Regime Breaks**: Odd-Even policies, lockdowns, and multi-year appliance growth (year-on-year AC penetration) shift the load-temperature slope over time.

## 3. Evidence from Our Missed-Peak Days (`results/missed_peaks.csv`)
*(Populated after H6/H8 walk-forward backtest from `results/missed_peaks.csv`)*

| Date | Actual Peak (MW) | Pred P50 Peak (MW) | Pred P90 Peak (MW) | Hour Error (min) | Root Cause / Note |
| --- | --- | --- | --- | --- | --- |
| `[PENDING H6/H8]` | `—` | `—` | `—` | `—` | Run `integration/run_pipeline.py` to inspect top missed peaks |

## 4. What We Would Do Next (Production Roadmap)
1. **Archived NWP Ensembles**: Replace actual-weather proxy with true archived ECMWF/GFS ensemble weather forecasts so interval width reflects real weather forecast uncertainty at 24h vs 168h.
2. **Tail-Weighted / Peak-Hour Loss**: Add sample weights on high-CDH summer evenings and train dedicated peak-magnitude quantile adjustments.
3. **Outage & Thunder-Storm Nowcasting**: Pair the D-1 procurement baseline with intraday 2–6h satellite/radar nowcasts for sudden *andhi* load drops.

