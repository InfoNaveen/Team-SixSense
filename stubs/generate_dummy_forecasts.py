#!/usr/bin/env python
"""
generate_dummy_forecasts.py  –  H1 deliverable for teammates
=============================================================
Creates dummy forecast parquets in the EXACT contract format so that
Features (P3), Modeling (P4), and Data/Dashboard (P2) can develop
against real file shapes before the pipeline produces real results.

Outputs (in results/):
  - forecasts_day_DUMMY.parquet   (24 h × 12 folds)
  - forecasts_week_DUMMY.parquet  (168 h × 12 folds)
  - peak_flags_DUMMY.parquet
  - missed_peaks_DUMMY.csv
  - metrics_DUMMY.csv

Contract 4 columns:
  forecasts_{horizon}.parquet : ts, fold, horizon, model, p10, p50, p90, actual
  peak_flags.parquet          : date, fold, peak_prob, peak_pred, peak_hour_pred,
                                peak_actual, peak_hour_actual, threshold_mw
  metrics.csv                 : horizon, model, fold, metric, value
  missed_peaks.csv            : date, actual_peak_mw, pred_p50_peak_mw,
                                pred_p90_peak_mw, hour_error_min, note

Run:
    python stubs/generate_dummy_forecasts.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

np.random.seed(2026)


# ---------------------------------------------------------------------------
# Realistic Delhi-like hourly load shape
# ---------------------------------------------------------------------------
def _delhi_load_shape(ts: pd.DatetimeIndex) -> np.ndarray:
    """Generate a plausible Delhi hourly load curve (MW).

    Shape drivers:
      - Seasonal cycle  : ~2000 MW winter → ~7500 MW summer
      - Daily cycle     : morning ramp, evening peak
      - Random noise    : ±200 MW hour-to-hour
    """
    day_of_year = ts.dayofyear.values.astype(float)
    hour = ts.hour.values.astype(float)

    # Seasonal: peaks around day 170 (mid-June)
    seasonal = 4750 + 2750 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # Diurnal: bimodal – morning ramp 06-10, evening peak 18-22
    diurnal = (
        0.85
        + 0.10 * np.exp(-0.5 * ((hour - 9) / 2) ** 2)    # morning
        + 0.15 * np.exp(-0.5 * ((hour - 20) / 2.5) ** 2)  # evening
        - 0.10 * np.exp(-0.5 * ((hour - 4) / 2) ** 2)     # night dip
    )

    load = seasonal * diurnal + np.random.normal(0, 200, size=len(ts))
    return np.clip(load, 1500, 9000)


# ---------------------------------------------------------------------------
# Generate forecast DataFrames
# ---------------------------------------------------------------------------
def _make_forecast_df(
    horizon: str, n_folds: int = 12
) -> pd.DataFrame:
    """Build a dummy forecast DataFrame for the given horizon."""
    hours_per_fold = 24 if horizon == "day" else 168
    model_name = "lgbm_quantile_DUMMY"

    rows = []
    # 12 monthly folds starting 2024-07-01
    fold_starts = pd.date_range("2024-07-01", periods=n_folds, freq="MS")

    for fold_idx, start in enumerate(fold_starts, 1):
        ts_range = pd.date_range(start, periods=hours_per_fold, freq="h")
        actual = _delhi_load_shape(ts_range)

        # Simulated forecast: P50 ≈ actual + small bias; P10/P90 bracket it
        noise_p50 = np.random.normal(0, 150, size=hours_per_fold)
        p50 = actual + noise_p50
        p10 = p50 - np.abs(np.random.normal(350, 80, size=hours_per_fold))
        p90 = p50 + np.abs(np.random.normal(400, 90, size=hours_per_fold))

        for i in range(hours_per_fold):
            rows.append(
                {
                    "ts": ts_range[i],
                    "fold": fold_idx,
                    "horizon": horizon,
                    "model": model_name,
                    "p10": round(float(p10[i]), 1),
                    "p50": round(float(p50[i]), 1),
                    "p90": round(float(p90[i]), 1),
                    "actual": round(float(actual[i]), 1),
                }
            )

    df = pd.DataFrame(rows)
    # Enforce contract: p10 <= p50 <= p90
    df["p10"] = df[["p10", "p50"]].min(axis=1)
    df["p90"] = df[["p90", "p50"]].max(axis=1)
    return df


def _make_peak_flags(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Derive peak-flag dummy from the day-ahead forecasts."""
    df = forecast_df.copy()
    df["date"] = df["ts"].dt.date

    daily = df.groupby(["date", "fold"]).agg(
        actual_peak=("actual", "max"),
        pred_p90_peak=("p90", "max"),
        pred_p50_peak=("p50", "max"),
        peak_hour_actual=("actual", "idxmax"),
        peak_hour_pred=("p90", "idxmax"),
    ).reset_index()

    # Threshold = 90th percentile of daily maxima (training proxy)
    threshold = float(np.percentile(daily["actual_peak"], 90))

    daily["threshold_mw"] = threshold
    daily["peak_actual"] = daily["actual_peak"] > threshold
    daily["peak_prob"] = np.clip(
        (daily["pred_p90_peak"] - threshold) / 500 + 0.5, 0, 1
    ).round(3)
    daily["peak_pred"] = daily["peak_prob"] > 0.5

    # Resolve peak hours back to hour-of-day
    for col in ["peak_hour_actual", "peak_hour_pred"]:
        daily[col] = daily[col].map(
            lambda idx: int(df.loc[idx, "ts"].hour) if idx in df.index else 0
        )

    return daily[
        [
            "date", "fold", "peak_prob", "peak_pred", "peak_hour_pred",
            "peak_actual", "peak_hour_actual", "threshold_mw",
        ]
    ]


def _make_missed_peaks(peak_flags: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Missed peaks: days where peak_actual=True but peak_pred=False."""
    missed = peak_flags[peak_flags["peak_actual"] & ~peak_flags["peak_pred"]].copy()
    if missed.empty:
        # Guarantee at least a couple for template purposes
        missed = peak_flags[peak_flags["peak_actual"]].head(3).copy()
        missed["peak_pred"] = False

    df = forecast_df.copy()
    df["date"] = df["ts"].dt.date

    rows = []
    for _, row in missed.iterrows():
        day_data = df[df["date"] == row["date"]]
        if day_data.empty:
            continue
        rows.append(
            {
                "date": row["date"],
                "actual_peak_mw": round(float(day_data["actual"].max()), 1),
                "pred_p50_peak_mw": round(float(day_data["p50"].max()), 1),
                "pred_p90_peak_mw": round(float(day_data["p90"].max()), 1),
                "hour_error_min": abs(int(row["peak_hour_actual"]) - int(row["peak_hour_pred"])) * 60,
                "note": "DUMMY – replace with real failure analysis",
            }
        )
    return pd.DataFrame(rows)


def _make_metrics(day_df: pd.DataFrame, week_df: pd.DataFrame) -> pd.DataFrame:
    """Generate plausible dummy metrics (not real evaluation)."""
    rows = []
    model = "lgbm_quantile_DUMMY"
    for horizon, fdf in [("day", day_df), ("week", week_df)]:
        for fold in fdf["fold"].unique():
            fold_data = fdf[fdf["fold"] == fold]
            actual = fold_data["actual"].values
            p50 = fold_data["p50"].values

            mape = float(np.mean(np.abs((actual - p50) / np.clip(actual, 1, None)))) * 100
            rmse = float(np.sqrt(np.mean((actual - p50) ** 2)))
            picp = float(np.mean((actual >= fold_data["p10"].values) & (actual <= fold_data["p90"].values))) * 100

            for metric, value in [("MAPE_%", mape), ("RMSE_MW", rmse), ("PICP_%", picp)]:
                rows.append(
                    {
                        "horizon": horizon,
                        "model": model,
                        "fold": fold,
                        "metric": metric,
                        "value": round(value, 2),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("[dummy-gen] Generating dummy forecast files for contract testing ...")

    day_df = _make_forecast_df("day")
    week_df = _make_forecast_df("week")

    # Write forecasts
    day_path = RESULTS / "forecasts_day_DUMMY.parquet"
    week_path = RESULTS / "forecasts_week_DUMMY.parquet"
    day_df.to_parquet(day_path, index=False)
    week_df.to_parquet(week_path, index=False)
    print(f"  -> {day_path.name}  ({len(day_df)} rows, cols: {list(day_df.columns)})")
    print(f"  -> {week_path.name} ({len(week_df)} rows, cols: {list(week_df.columns)})")

    # Also write as the "real" filenames so procurement can run immediately
    day_df.to_parquet(RESULTS / "forecasts_day.parquet", index=False)
    week_df.to_parquet(RESULTS / "forecasts_week.parquet", index=False)
    print("  -> Also written as forecasts_day.parquet / forecasts_week.parquet (for pipeline testing)")

    # Peak flags (from day-ahead data)
    pf = _make_peak_flags(day_df)
    pf_path = RESULTS / "peak_flags_DUMMY.parquet"
    pf.to_parquet(pf_path, index=False)
    pf.to_parquet(RESULTS / "peak_flags.parquet", index=False)
    print(f"  -> {pf_path.name}   ({len(pf)} rows)")

    # Missed peaks
    mp = _make_missed_peaks(pf, day_df)
    mp_path = RESULTS / "missed_peaks_DUMMY.csv"
    mp.to_csv(mp_path, index=False)
    mp.to_csv(RESULTS / "missed_peaks.csv", index=False)
    print(f"  -> {mp_path.name}  ({len(mp)} rows)")

    # Metrics
    met = _make_metrics(day_df, week_df)
    met_path = RESULTS / "metrics_DUMMY.csv"
    met.to_csv(met_path, index=False)
    met.to_csv(RESULTS / "metrics.csv", index=False)
    print(f"  -> {met_path.name}     ({len(met)} rows)")

    print("\n[dummy-gen] Done. All contract files are now available for teammates.")
    print("            Replace _DUMMY files with real outputs as pipeline matures.")


if __name__ == "__main__":
    main()

