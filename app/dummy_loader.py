"""
Results & Dummy Fallback Loader for app/app.py (Data + Dashboard / P2).

Reads results/ files only. Never imports models/, features/, eval/, data/, or procurement/.
If real results/ files (or results/forecasts_week_DUMMY.parquet) are missing, builds
in-memory dummy DataFrames matching Contract #4 so the dashboard runs offline at H3-H6.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5, 6):
        return "Summer"
    if month in (7, 8, 9):
        return "Monsoon"
    return "Post-Monsoon"


def _make_dummy_forecasts(horizon: str) -> pd.DataFrame:
    """Create a 12-month dummy walk-forward forecast table matching Contract #4."""
    rng = np.random.default_rng(101 if horizon == "week" else 202)
    ts = pd.date_range("2024-01-01 00:00:00", "2024-12-31 23:00:00", freq="h", tz="Asia/Kolkata")
    # Sample 7 days per month across 12 folds to keep dummy payload fast & responsive
    keep_mask = ts.day <= 7
    ts = ts[keep_mask]

    hour = ts.hour.to_numpy()
    doy = ts.dayofyear.to_numpy()
    fold = ts.month.to_numpy()

    seasonal = 4200.0 + 1850.0 * np.sin(2.0 * np.pi * (doy - 100.0) / 365.25)
    diurnal = (
        650.0 * np.exp(-0.5 * ((hour - 15.0) / 3.0) ** 2)
        + 1200.0 * np.exp(-0.5 * ((hour - 22.0) / 2.3) ** 2)
        - 600.0 * np.exp(-0.5 * ((hour - 4.5) / 2.0) ** 2)
    )
    actual = np.clip(seasonal + diurnal + rng.normal(0.0, 140.0, len(ts)), 1950.0, 8300.0)

    # Main LightGBM model
    spread = (230.0 if horizon == "week" else 170.0) + 80.0 * np.exp(-0.5 * ((hour - 22.0) / 2.8) ** 2)
    p50_lgb = actual + rng.normal(0.0, 130.0 if horizon == "week" else 90.0, len(ts))
    q_lgb = np.sort(np.vstack([p50_lgb - spread, p50_lgb, p50_lgb + spread]), axis=0)

    df_lgb = pd.DataFrame({
        "ts": ts,
        "fold": fold,
        "horizon": horizon,
        "model": "lightgbm_quantile",
        "p10": np.round(q_lgb[0], 2),
        "p50": np.round(q_lgb[1], 2),
        "p90": np.round(q_lgb[2], 2),
        "actual": np.round(actual, 2),
    })
    return df_lgb


def _make_dummy_peak_flags(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    df = forecasts_df.copy()
    df["date"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d")
    daily = (
        df.groupby(["date", "fold"], as_index=False)
        .agg(
            actual_max=("actual", "max"),
            p90_max=("p90", "max"),
            peak_hour_actual=("actual", lambda s: int(df.loc[s.idxmax(), "ts"].hour)),
            peak_hour_pred=("p50", lambda s: int(df.loc[s.idxmax(), "ts"].hour)),
        )
    )
    threshold_mw = float(np.quantile(daily["actual_max"], 0.90))
    daily["peak_actual"] = daily["actual_max"] >= threshold_mw
    daily["peak_prob"] = np.clip((daily["p90_max"] - threshold_mw + 300.0) / 600.0, 0.02, 0.98).round(3)
    daily["peak_pred"] = daily["peak_prob"] >= 0.5
    daily["threshold_mw"] = round(threshold_mw, 2)
    return daily[[
        "date", "fold", "peak_prob", "peak_pred", "peak_hour_pred",
        "peak_actual", "peak_hour_actual", "threshold_mw",
    ]]


def _make_dummy_metrics() -> pd.DataFrame:
    rows = [
        # Day-ahead
        ("day", "seasonal_naive", "all", "MAPE", 6.85),
        ("day", "seasonal_naive", "all", "RMSE", 412.4),
        ("day", "seasonal_naive", "all", "Pinball", 148.2),
        ("day", "linear_regression", "all", "MAPE", 5.40),
        ("day", "linear_regression", "all", "RMSE", 328.7),
        ("day", "linear_regression", "all", "Pinball", 112.5),
        ("day", "lightgbm_quantile", "all", "MAPE", 3.15),
        ("day", "lightgbm_quantile", "all", "RMSE", 198.4),
        ("day", "lightgbm_quantile", "all", "Pinball", 64.8),
        ("day", "lightgbm_quantile", "all", "PICP", 0.81),
        ("day", "lightgbm_quantile", "all", "Peak_Precision", 0.84),
        ("day", "lightgbm_quantile", "all", "Peak_Recall", 0.79),
        ("day", "lightgbm_quantile", "all", "Peak_F1", 0.81),
        ("day", "lightgbm_quantile", "all", "Peak_Timing_Error_Min", 42.0),
        # Week-ahead
        ("week", "seasonal_naive", "all", "MAPE", 7.45),
        ("week", "seasonal_naive", "all", "RMSE", 458.9),
        ("week", "seasonal_naive", "all", "Pinball", 165.0),
        ("week", "linear_regression", "all", "MAPE", 5.95),
        ("week", "linear_regression", "all", "RMSE", 361.2),
        ("week", "linear_regression", "all", "Pinball", 124.7),
        ("week", "lightgbm_quantile", "all", "MAPE", 4.10),
        ("week", "lightgbm_quantile", "all", "RMSE", 254.6),
        ("week", "lightgbm_quantile", "all", "Pinball", 81.3),
        ("week", "lightgbm_quantile", "all", "PICP", 0.79),
        ("week", "lightgbm_quantile", "all", "Peak_Precision", 0.78),
        ("week", "lightgbm_quantile", "all", "Peak_Recall", 0.74),
        ("week", "lightgbm_quantile", "all", "Peak_F1", 0.76),
        ("week", "lightgbm_quantile", "all", "Peak_Timing_Error_Min", 68.0),
    ]
    return pd.DataFrame(rows, columns=["horizon", "model", "fold", "metric", "value"])


def _make_dummy_missed_peaks() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "date": "2024-06-05",
            "actual_peak_mw": 8042.0,
            "pred_p50_peak_mw": 7490.0,
            "pred_p90_peak_mw": 7760.0,
            "hour_error_min": 60.0,
            "note": "DUMMY: Sudden 4-day heatwave onset with 32C overnight minimum (thermal inertia spike).",
        },
        {
            "date": "2024-07-03",
            "actual_peak_mw": 7610.0,
            "pred_p50_peak_mw": 7180.0,
            "pred_p90_peak_mw": 7450.0,
            "hour_error_min": 120.0,
            "note": "DUMMY: Pre-monsoon humidity surge (RH > 75%) shifted peak later to 23:00 IST.",
        },
    ])


def _make_dummy_procurement() -> dict[str, Any]:
    return {
        "schedule": [[5200.0] * 7 for _ in range(24)],
        "p90_exposure_mw": 310.0,
        "cost_model": 5_100_000_000.0,
        "cost_flat": 5_550_000_000.0,
        "saving": 450_000_000.0,
        "saving_pct_vs_flat": 8.11,
        "cost_p50": 4_960_000_000.0,
        "is_dummy_data": True,
        "assumptions": {
            "contracted_rate_rs_per_kwh": 5.0,
            "dsm_rate_rs_per_kwh": 12.0,
            "stranded_rate_rs_per_kwh": 2.5,
        },
    }


def load_dashboard_bundle() -> dict[str, Any]:
    """
    Load all results/ artifacts, seamlessly using DUMMY files or in-memory stubs
    for any artifact not yet produced by teammates.
    """
    sources_status: dict[str, str] = {}

    # 1. Forecasts (week & day)
    week_real = RESULTS_DIR / "forecasts_week.parquet"
    week_dummy_file = RESULTS_DIR / "forecasts_week_DUMMY.parquet"
    day_real = RESULTS_DIR / "forecasts_day.parquet"

    if week_real.exists():
        forecasts_week = pd.read_parquet(week_real)
        sources_status["forecasts_week"] = "REAL (results/forecasts_week.parquet)"
    elif week_dummy_file.exists():
        forecasts_week = pd.read_parquet(week_dummy_file)
        sources_status["forecasts_week"] = "DUMMY FILE (results/forecasts_week_DUMMY.parquet)"
    else:
        forecasts_week = _make_dummy_forecasts("week")
        sources_status["forecasts_week"] = "DUMMY IN-MEMORY (awaiting H6 forecasts_week.parquet)"

    if day_real.exists():
        forecasts_day = pd.read_parquet(day_real)
        sources_status["forecasts_day"] = "REAL (results/forecasts_day.parquet)"
    else:
        forecasts_day = _make_dummy_forecasts("day")
        sources_status["forecasts_day"] = "DUMMY IN-MEMORY (awaiting H6 forecasts_day.parquet)"

    # 2. Peak flags
    peak_path = RESULTS_DIR / "peak_flags.parquet"
    if peak_path.exists():
        peak_flags = pd.read_parquet(peak_path)
        sources_status["peak_flags"] = "REAL (results/peak_flags.parquet)"
    else:
        peak_flags = _make_dummy_peak_flags(forecasts_week)
        sources_status["peak_flags"] = "DUMMY IN-MEMORY (awaiting H6 peak_flags.parquet)"

    # 3. Metrics
    metrics_path = RESULTS_DIR / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        sources_status["metrics"] = "REAL (results/metrics.csv)"
    else:
        metrics = _make_dummy_metrics()
        sources_status["metrics"] = "DUMMY IN-MEMORY (awaiting H8 metrics.csv)"

    # 4. Missed peaks
    missed_path = RESULTS_DIR / "missed_peaks.csv"
    if missed_path.exists():
        missed_peaks = pd.read_csv(missed_path)
        sources_status["missed_peaks"] = "REAL (results/missed_peaks.csv)"
    else:
        missed_peaks = _make_dummy_missed_peaks()
        sources_status["missed_peaks"] = "DUMMY IN-MEMORY (awaiting H8 missed_peaks.csv)"

    # 5. Procurement JSON
    proc_path = RESULTS_DIR / "procurement_week.json"
    if proc_path.exists():
        procurement = json.loads(proc_path.read_text(encoding="utf-8"))
        tag = "DUMMY STUB FILE" if procurement.get("is_dummy_data") else "REAL"
        sources_status["procurement"] = f"{tag} (results/procurement_week.json)"
    else:
        procurement = _make_dummy_procurement()
        sources_status["procurement"] = "DUMMY IN-MEMORY (awaiting procurement_week.json)"

    return {
        "forecasts_day": forecasts_day,
        "forecasts_week": forecasts_week,
        "peak_flags": peak_flags,
        "metrics": metrics,
        "missed_peaks": missed_peaks,
        "procurement": procurement,
        "sources_status": sources_status,
        "season_helper": _season_from_month,
    }

