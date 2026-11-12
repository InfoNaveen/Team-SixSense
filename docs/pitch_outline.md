# Pitch Outline — 3-Minute Presentation

> **Title**: Probabilistic Peak-Aware Load Forecasting & Procurement Copilot for Delhi's DISCOM

---

## Slide 1: Hook  (0:00 – 0:30)

**Visual**: Single number on screen: "₹36 lakh per hour"

**Script**:
> "Delhi's demand swings from 2,000 to 8,300 megawatts. When a DISCOM under-procures by just 300 MW for one hour, the penalty is ₹36 lakh at DSM rates. Over-procure by 300 MW? That's ₹7.5 lakh per hour wasted on stranded power. A single-number forecast hides this risk completely. We built a system that quantifies it."

---

## Slide 2: The Problem  (0:30 – 1:00)

**Visual**: Split screen — left shows "Point forecast: 5,200 MW ✓", right shows "But actual range: 4,800 – 5,800 MW" with red zone highlighted.

**Script**:
> "Current scheduling uses a point forecast. The procurement desk buys a fixed flat amount. When demand spikes — and it does, every summer evening — they're caught short. When it dips — every winter night — they've paid for power nobody uses. The problem isn't forecast accuracy. It's that a single number has no concept of risk."

**Key point**: Frame it as a *risk problem*, not just an accuracy problem.

---

## Slide 3: Our Solution  (1:00 – 1:45)

**Visual**: Shaded band chart — P10 (light blue) / P50 (solid line) / P90 (light red) with actual overlaid.

**Script**:
> "We produce P10, P50, P90 forecasts — not one number but a probability range. We train three LightGBM quantile models using walk-forward backtesting, never a random split. Our features: cooling-degree-hours, heat index, temperature lags, load lags respecting the 09:00 D-1 forecast origin, festival and holiday flags, and trend.
>
> Then we add two tools on top: a peak-event classifier that flags whether tomorrow will exceed the 90th percentile, and a procurement copilot that turns the P90 forecast into an hour-by-hour buying schedule."

---

## Slide 4: Demo  (1:45 – 2:30)

**Demo click path** (Streamlit dashboard):

1. **Hero chart**: Shaded P10/P50/P90 bands for a summer week. Point to where actual sits inside the band. "See — 80% coverage."
2. **Peak flags**: Click on a flagged peak day. Show peak_prob, predicted vs actual peak hour. "This day was flagged 24 hours in advance."
3. **Accuracy table**: Show MAPE/RMSE per horizon, LightGBM vs baselines. "We beat seasonal-naive by [FILL]%."
4. **Procurement panel**: Click on "Sample Week". Show three-strategy comparison: P90 vs P50 vs Flat. "P90 schedule saves Rs [FILL] versus flat — and it only missed the peak by [FILL] MW."
5. **Missed peaks**: Drill down into a missed peak. "Here's where we failed and why — a festival effect not in our training data."

**Fallback**: If dashboard has issues, present the charts as static screenshots.

---

## Slide 5: Results  (2:30 – 2:50)

**Visual**: Results table

| Metric | Day-Ahead | Week-Ahead |
|--------|-----------|------------|
| MAPE vs naive | [FILL: -X%] | [FILL: -X%] |
| PICP (target 80%) | [FILL]% | [FILL]% |
| Peak recall | [FILL]% | — |
| Procurement saving (P90 vs flat) | — | Rs [FILL] ([FILL]%) |

**Script**:
> "We beat both baselines on both horizons. Our 80% intervals actually contain [FILL]% of actuals — that's calibrated, not overconfident. Peak recall is [FILL]%. And the procurement copilot saved Rs [FILL] on a single sample week using placeholder rates."

---

## Slide 6: Why Us  (2:50 – 3:00)

**Script**:
> "Three things. One: rigorous walk-forward evaluation, never a random split. Two: honest failure analysis — we show you the days we missed and why. Three: a reproducible pipeline — clone the repo, run one command, get every result. Thank you."

---

---

## Judge Q&A — Prepared Answers

### Q: "Did you use a random train-test split?"
> "No, never. We use expanding-window walk-forward backtesting with monthly test blocks. Every prediction is made using only data available before the forecast origin. We can show the fold structure in the code."

### Q: "How do you prevent data leakage?"
> "Three controls. First: load lags are all ≥24h behind the target. Second: the forecast origin is 09:00 D-1 and features only use data available by then. Third: walk-forward folds enforce temporal ordering — no future data in training."

### Q: "Your intervals — are they calibrated?"
> "We check PICP: the fraction of actuals inside the P10–P90 band. Target is 80%. Ours is [FILL]%. If it drifts, we would apply a conformal post-hoc shift — simple, effective, and it's in our stretch goals. [If done: We did apply conformal calibration and improved PICP from X% to Y%.]"

### Q: "Where does the model fail?"
> "Three places. Sudden multi-degree temperature jumps within 24 hours — our lags don't capture intra-day weather shifts. Festivals with very few training examples — Diwali load patterns differ year to year. And the week-ahead horizon degrades past 96 hours because we use actual weather as a proxy for forecasts, which masks the real weather uncertainty."

### Q: "Is the savings number trustworthy?"
> "The shape is trustworthy — hour-varying procurement beats flat procurement because demand varies. The exact rupee number depends on the assumed rates, which are placeholders. We provide a sensitivity table that shows the saving across a range of DSM and stranded-cost rates. The qualitative conclusion holds across all settings."

### Q: "What would you do with more time?"
> "Four things: use Open-Meteo's forecast archive instead of actual weather, apply conformal calibration for tighter intervals, train a festival-specific sub-model, and add a TFT for interpretable multi-horizon attention."

### Q: "Why LightGBM and not deep learning?"
> "Twelve-hour hackathon. LightGBM trains in seconds, quantile regression is built-in, and it handles tabular features well. We considered TFT as a stretch goal. For this data volume (~50K rows), gradient boosting is competitive with transformers and much faster to iterate."

---

*Last updated: [FILL: date and time] · Lead*
