# Data Cleaning & Imputation Log (`data/clean_hourly.parquet`)

- **Hourly Window (IST)**: `2020-01-01 00:00:00+05:30` to `2024-12-31 23:00:00+05:30` (`43,848` hours)
- **Load Source**: FALLBACK: Synthetic Delhi hourly load (2,000-8,300 MW diurnal + CDH + thermal inertia)
- **Weather Source**: FALLBACK: Synthetic Delhi hourly weather (offline default)

> **[FALLBACK NOTICE — FOR MODEL CARD]**: One or more inputs used the clearly-labelled synthetic fallback generator because no local raw CSV / live Open-Meteo connection was supplied. Replace via `--load-csv` and `--fetch-open-meteo` when real files are placed locally.

## Summary of Load Gap & Outlier Handling
- Total imputed load hours (`load_mw_imputed_flag == True`): **19** (0.043%)
- Short gaps (<= 6h, linear interpolation): **9** hours
- Longer gaps (> 6h, same-hour-last-week `t - 168h`): **10** hours
- Outage / telemetry spike outliers flagged & re-imputed: **6** hours

## Detailed Per-Column Operations

| Column | Count | Method | Reason | Notes |
| --- | ---: | --- | --- | --- |
| `load_mw` | 2 | Mean aggregation by ts | Duplicate hourly timestamps in raw feed | — |
| `load_mw` | 13 | Reindexed to complete IST hourly grid | Missing timestamps or raw NaN readings before outlier check | — |
| `load_mw` | 6 | Flagged -> NaN -> imputed | Outage (<500 MW or <20% 7d median) or telemetry spike (>11,000 MW / >3,000 MW jump) | Sample flagged timestamps: 2020-01-18 02:00:00+05:30, 2020-01-20 04:00:00+05:30, 2020-07-12 04:00:00+05:30, 2021-08-22 03:00:00+05:30, 2022-06-18 04:00:00+05:30 |
| `load_mw` | 9 | Linear interpolation (<=6h) | Short telemetry/clock gaps | — |
| `load_mw` | 10 | Same-hour-last-week (t - 168h) | Longer outages (>6h) or leading/trailing short gaps | — |
| `temp_c` | 0 | None needed | Complete hourly weather coverage | — |
| `rh_pct` | 0 | None needed | Complete hourly weather coverage | — |
| `apparent_temp_c` | 0 | None needed | Complete hourly weather coverage | — |
