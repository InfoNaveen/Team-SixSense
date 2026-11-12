# Data + Dashboard (P2) Hour-by-Hour Checklist

- **H0–H1**: Agree on `data/clean_hourly.parquet` schema (Contract #1). Locate real Delhi hourly load CSV (SLDC/Kaggle) or confirm fallback by H1.
- **H1–H3**: Run `python data/build_dataset.py --load-csv <file> --fetch-open-meteo` (or fallback). Verify `data/clean_hourly.parquet` passes `validate_clean_hourly()` and inspect `data/imputation_log.md` + `data/sources.md`. Hand off to Features (P3) by H2–H3.
- **H3–H6**: Build `app/app.py` on dummy data (`results/forecasts_week_DUMMY.parquet` or built-in fallback + `results/procurement_week.json`) so all 4 views render offline.
- **H6–H8**: Switch `app/app.py` to real `results/forecasts_*.parquet`, `results/peak_flags.parquet`, `results/metrics.csv`, `results/missed_peaks.csv`, and `results/procurement_week.json`.
- **H8–H10**: Polish layout, verify zero imports from `models/` or `features/`, verify 100% offline execution.
- **H10–H12**: Freeze code, test fresh-clone Streamlit launch (`streamlit run app/app.py`), rehearse demo click-path with Lead.

