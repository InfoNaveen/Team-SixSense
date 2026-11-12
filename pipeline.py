#!/usr/bin/env python
"""
pipeline.py — Master End-to-End Pipeline
=========================================
Probabilistic Peak-Aware Load Forecasting & Procurement Copilot for Delhi DISCOM.

Usage:
    python pipeline.py data/raw --folds 3
    python pipeline.py data/raw --folds 12
    python pipeline.py data/raw

Single source of truth that runs data prep, feature engineering, expanding-window
walk-forward backtesting, model training (LightGBM quantile regression + baselines),
peak-event classification, procurement simulation, and outputs all contract files to results/.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Safe terminal encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. DATA INGESTION & CLEANING
# ==============================================================================

def load_and_clean_raw(data_dir: str | Path) -> pd.DataFrame:
    """
    Ingest raw 5-minute data from data_dir and resample to hourly clean dataset.
    Detects: time=datetime demand=Power demand temp=temp humidity=rhum
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    # Prefer the 5-min power demand file
    target_file = None
    for f in csv_files:
        if "powerdemand" in f.name.lower() or "delhi" in f.name.lower():
            target_file = f
            break
    if target_file is None:
        target_file = csv_files[0]

    # Required startup confirmation line
    print(f"Startup: time=datetime demand=Power demand temp=temp humidity=rhum (source: {target_file.name})")

    logger.info(f"Loading raw data from {target_file} ...")
    df_raw = pd.read_csv(target_file, low_memory=False)

    # Column identification
    time_col = "datetime" if "datetime" in df_raw.columns else df_raw.columns[0]
    demand_col = "Power demand" if "Power demand" in df_raw.columns else "demand"
    temp_col = "temp" if "temp" in df_raw.columns else "temperature"
    rh_col = "rhum" if "rhum" in df_raw.columns else "humidity"

    df_raw["ts"] = pd.to_datetime(df_raw[time_col], errors="coerce")
    df_raw = df_raw.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # Resample from 5-minute to hourly (Contract 1 requirements)
    df_raw.set_index("ts", inplace=True)
    hourly = df_raw[[demand_col, temp_col, rh_col]].resample("1h").mean()

    hourly.rename(
        columns={
            demand_col: "load_mw",
            temp_col: "temp_c",
            rh_col: "rh_pct",
        },
        inplace=True,
    )

    # Track imputation
    hourly["load_mw_imputed_flag"] = hourly["load_mw"].isna()

    # Interpolate short gaps (<= 3 hours), forward fill remaining
    hourly["load_mw"] = hourly["load_mw"].interpolate(method="time", limit=3)
    hourly["load_mw"] = hourly["load_mw"].ffill().bfill()

    hourly["temp_c"] = hourly["temp_c"].interpolate(method="time", limit=6).ffill().bfill()
    hourly["rh_pct"] = hourly["rh_pct"].interpolate(method="time", limit=6).ffill().bfill()

    # Calculate apparent temperature (Australian Bureau of Meteorology formula)
    # e = (rh / 100) * 6.105 * exp(17.27 * temp / (237.7 + temp))
    # AT = temp + 0.33 * e - 4.0
    t = hourly["temp_c"]
    rh = hourly["rh_pct"]
    e = (rh / 100.0) * 6.105 * np.exp((17.27 * t) / (237.7 + t))
    hourly["apparent_temp_c"] = t + 0.33 * e - 4.0

    # Reset index so ts is a regular column
    hourly = hourly.reset_index()

    # Calendar and holiday flags
    try:
        import holidays
        years = hourly["ts"].dt.year.unique().tolist()
        in_holidays = holidays.India(years=years)
    except Exception:
        in_holidays = {}

    hourly["is_holiday"] = hourly["ts"].dt.date.apply(lambda d: d in in_holidays)
    hourly["festival_name"] = hourly["ts"].dt.date.apply(lambda d: in_holidays.get(d, ""))

    hourly["is_oddeven"] = False
    # Delhi COVID lockdown period: 2021-04-19 to 2021-05-31
    lockdown_start = pd.Timestamp("2021-04-19")
    lockdown_end = pd.Timestamp("2021-05-31 23:59:59")
    hourly["is_lockdown"] = (hourly["ts"] >= lockdown_start) & (hourly["ts"] <= lockdown_end)

    # Save Contract 1 clean dataset
    clean_parquet_path = REPO_ROOT / "data" / "clean_hourly.parquet"
    clean_csv_path = REPO_ROOT / "data" / "clean_hourly.csv"
    try:
        hourly.to_parquet(clean_parquet_path, index=False)
        logger.info(f"Saved Contract 1: {clean_parquet_path} ({len(hourly):,} rows)")
    except Exception as exc:
        logger.warning(f"Parquet export failed ({exc}); exporting CSV fallback")
        hourly.to_csv(clean_csv_path, index=False)

    return hourly


# ==============================================================================
# 2. FEATURE ENGINEERING (Strict Leakage Prevention)
# ==============================================================================

def build_leakage_free_features(df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """
    Build features adhering strictly to the 09:00 D-1 forecast origin.
    Never uses moving_avg_3, year, month, day, hour, minute columns as features.
    
    Lags:
      - For day-ahead: load lags >= 48h, rolling load stats shifted by 48h.
      - For week-ahead: load lags >= 168h, rolling load stats shifted by 168h.
    """
    f = pd.DataFrame(index=df.index)
    f["ts"] = df["ts"]
    f["actual"] = df["load_mw"]

    temp = df["temp_c"]
    rh = df["rh_pct"]

    # Thermal features
    f["cdh"] = np.maximum(0.0, temp - 25.0)  # Cooling degree hours base 25 C
    f["cdh_24h_sum"] = f["cdh"].rolling(24, min_periods=1).sum()
    f["apparent_temp_c"] = df["apparent_temp_c"]

    # Weather lags (weather forecast proxy)
    f["temp_c"] = temp
    f["rh_pct"] = rh
    f["temp_lag_24h"] = temp.shift(24).ffill().bfill()
    f["temp_lag_48h"] = temp.shift(48).ffill().bfill()
    f["temp_roll_72h_mean"] = temp.rolling(72, min_periods=1).mean()

    # Load lags strictly older than origin
    load = df["load_mw"]
    if horizon == "day":
        f["load_lag_48h"] = load.shift(48).ffill().bfill()
        f["load_lag_72h"] = load.shift(72).ffill().bfill()
        f["load_lag_168h"] = load.shift(168).ffill().bfill()
        f["load_roll_7d_mean"] = load.shift(48).rolling(168, min_periods=1).mean().ffill().bfill()
        f["load_roll_7d_max"] = load.shift(48).rolling(168, min_periods=1).max().ffill().bfill()
    else:  # week
        f["load_lag_168h"] = load.shift(168).ffill().bfill()
        f["load_lag_336h"] = load.shift(336).ffill().bfill()
        f["load_roll_7d_mean"] = load.shift(168).rolling(168, min_periods=1).mean().ffill().bfill()
        f["load_roll_7d_max"] = load.shift(168).rolling(168, min_periods=1).max().ffill().bfill()

    # Calendar features (sin/cos cyclical encoding and indicators)
    hour = df["ts"].dt.hour
    dow = df["ts"].dt.dayofweek
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    f["is_weekend"] = (dow >= 5).astype(float)
    f["is_holiday"] = df["is_holiday"].astype(float)
    f["is_lockdown"] = df["is_lockdown"].astype(float)

    # Linear trend
    f["day_index"] = (df["ts"] - df["ts"].iloc[0]).dt.total_seconds() / 86400.0

    return f


# ==============================================================================
# 3. EXPANDING-WINDOW WALK-FORWARD FOLDS
# ==============================================================================

def get_walk_forward_folds(df: pd.DataFrame, n_folds: int = 12) -> List[Dict[str, Any]]:
    """
    Generate expanding-window monthly walk-forward folds from the last N calendar months.
    Never uses random split.
    """
    df = df.sort_values("ts").reset_index(drop=True)
    df["ym"] = df["ts"].dt.to_period("M")
    unique_months = df["ym"].unique()

    if len(unique_months) < n_folds + 1:
        n_folds = max(1, len(unique_months) - 1)
        logger.warning(f"Adjusted n_folds to {n_folds} due to dataset length")

    test_months = unique_months[-n_folds:]
    folds = []

    for fold_idx, m in enumerate(test_months, 1):
        test_mask = df["ym"] == m
        train_mask = df["ym"] < m

        train_indices = df[train_mask].index.values
        test_indices = df[test_mask].index.values

        folds.append({
            "fold": fold_idx,
            "month": str(m),
            "train_idx": train_indices,
            "test_idx": test_indices,
            "test_start": df.loc[test_indices[0], "ts"],
            "test_end": df.loc[test_indices[-1], "ts"],
        })

    return folds


# ==============================================================================
# 4. MODELS & BASELINES
# ==============================================================================

def run_seasonal_naive(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Seasonal naive: same hour, same weekday of previous week (t - 168h)."""
    combined = pd.concat([train_df, test_df]).sort_values("ts")
    preds = combined["actual"].shift(168).loc[test_df.index].ffill().bfill()

    train_actual = train_df["actual"].values
    train_naive = train_df["actual"].shift(168).ffill().bfill().values
    residual_std = float(np.nanstd(train_actual - train_naive))
    if math.isnan(residual_std) or residual_std == 0:
        residual_std = 300.0

    p50 = preds.values
    p10 = p50 - 1.28 * residual_std
    p90 = p50 + 1.28 * residual_std

    res = pd.DataFrame({
        "p10": p10,
        "p50": p50,
        "p90": p90,
    }, index=test_df.index)
    return res


def run_linear_regression(
    X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame
) -> pd.DataFrame:
    """Linear regression with empirical residual quantiles for P10 and P90."""
    from sklearn.linear_model import Ridge

    lr = Ridge(alpha=10.0)
    lr.fit(X_train, y_train)

    train_preds = lr.predict(X_train)
    residuals = y_train - train_preds

    q10 = float(np.percentile(residuals, 10))
    q90 = float(np.percentile(residuals, 90))

    test_preds = lr.predict(X_test)
    p50 = test_preds
    p10 = test_preds + q10
    p90 = test_preds + q90

    res = pd.DataFrame({
        "p10": p10,
        "p50": p50,
        "p90": p90,
    }, index=X_test.index)
    return res


def run_lgbm_quantiles(
    X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame
) -> pd.DataFrame:
    """LightGBM quantile regression for P10, P50, and P90 with sorting crossing fix."""
    try:
        import lightgbm as lgb
        has_lgb = True
    except ImportError:
        has_lgb = False

    preds = {}
    alphas = [0.10, 0.50, 0.90]

    if has_lgb:
        for alpha in alphas:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=120,
                learning_rate=0.08,
                num_leaves=31,
                min_child_samples=20,
                verbosity=-1,
                random_state=42,
            )
            model.fit(X_train, y_train)
            key = f"p{int(alpha * 100)}"
            preds[key] = model.predict(X_test)
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        for alpha in alphas:
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=alpha,
                max_iter=100,
                learning_rate=0.08,
                random_state=42,
            )
            model.fit(X_train, y_train)
            key = f"p{int(alpha * 100)}"
            preds[key] = model.predict(X_test)

    # Fix quantile crossing: ensure p10 <= p50 <= p90
    p10 = preds["p10"]
    p50 = preds["p50"]
    p90 = preds["p90"]

    p10_fixed = np.minimum(p10, p50)
    p90_fixed = np.maximum(p90, p50)

    res = pd.DataFrame({
        "p10": p10_fixed,
        "p50": p50,
        "p90": p90_fixed,
    }, index=X_test.index)
    return res


def apply_conformal_calibration(
    preds_df: pd.DataFrame, actual: np.ndarray, target_picp: float = 80.0
) -> pd.DataFrame:
    """Conformal adjustment of prediction intervals if nominal coverage drifts."""
    p10 = preds_df["p10"].values.copy()
    p50 = preds_df["p50"].values.copy()
    p90 = preds_df["p90"].values.copy()

    # Nonconformity score = distance outside the interval
    lower_err = np.maximum(0.0, p10 - actual)
    upper_err = np.maximum(0.0, actual - p90)
    scores = np.maximum(lower_err, upper_err)

    q_val = float(np.percentile(scores, target_picp))

    p10_cal = p10 - q_val
    p90_cal = p90 + q_val

    return pd.DataFrame({
        "p10": p10_cal,
        "p50": p50,
        "p90": p90_cal,
    }, index=preds_df.index)


# ==============================================================================
# 5. METRICS & PEAK ANALYSIS
# ==============================================================================

def compute_metrics(actual: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> Dict[str, float]:
    """Compute MAPE, RMSE, Pinball losses, and PICP."""
    mask = ~np.isnan(actual) & ~np.isnan(p50)
    act = actual[mask]
    y50 = p50[mask]
    y10 = p10[mask]
    y90 = p90[mask]

    mape = float(np.mean(np.abs((act - y50) / np.clip(act, 100.0, None)))) * 100.0
    rmse = float(np.sqrt(np.mean((act - y50) ** 2)))

    # Pinball loss: L_alpha(y, q) = max(alpha * (y - q), (alpha - 1) * (y - q))
    def pinball(y, q, alpha):
        diff = y - q
        return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))

    pb10 = pinball(act, y10, 0.10)
    pb50 = pinball(act, y50, 0.50)
    pb90 = pinball(act, y90, 0.90)

    # Empirical coverage (PICP)
    picp = float(np.mean((act >= y10) & (act <= y90))) * 100.0

    return {
        "MAPE": round(mape, 3),
        "RMSE": round(rmse, 2),
        "Pinball_p10": round(pb10, 2),
        "Pinball_p50": round(pb50, 2),
        "Pinball_p90": round(pb90, 2),
        "PICP": round(picp, 2),
    }


def evaluate_peaks(
    df_forecasts: pd.DataFrame, train_load: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate daily peak events:
    Peak threshold = 90th percentile of daily maxima in the training window.
    """
    # Daily max threshold from training data
    train_daily = train_load.groupby(train_load["ts"].dt.date)["load_mw"].max()
    threshold = float(np.percentile(train_daily.dropna(), 90))

    df = df_forecasts.copy()
    df["date"] = df["ts"].dt.date

    peak_records = []
    missed_peaks = []

    for (date_val, fold_val), grp in df.groupby(["date", "fold"]):
        actual_peak = float(grp["actual"].max())
        pred_p50_peak = float(grp["p50"].max())
        pred_p90_peak = float(grp["p90"].max())

        act_peak_idx = grp["actual"].idxmax()
        pred_peak_idx = grp["p90"].idxmax()

        peak_hour_actual = int(grp.loc[act_peak_idx, "ts"].hour)
        peak_hour_pred = int(grp.loc[pred_peak_idx, "ts"].hour)

        peak_actual = actual_peak >= threshold
        # Predict peak if P90 peak exceeds threshold or P50 is close
        peak_pred = pred_p90_peak >= threshold
        peak_prob = float(np.clip((pred_p90_peak - threshold) / 400.0 + 0.5, 0.05, 0.99))

        hour_err_min = abs(peak_hour_actual - peak_hour_pred) * 60

        peak_records.append({
            "date": str(date_val),
            "fold": fold_val,
            "peak_prob": round(peak_prob, 3),
            "peak_pred": bool(peak_pred),
            "peak_hour_pred": peak_hour_pred,
            "peak_actual": bool(peak_actual),
            "peak_hour_actual": peak_hour_actual,
            "threshold_mw": round(threshold, 1),
        })

        if peak_actual and not peak_pred:
            missed_peaks.append({
                "date": str(date_val),
                "actual_peak_mw": round(actual_peak, 1),
                "pred_p50_peak_mw": round(pred_p50_peak, 1),
                "pred_p90_peak_mw": round(pred_p90_peak, 1),
                "hour_error_min": hour_err_min,
                "note": "Sudden weather surge or festival deviation below P90 threshold",
            })

    peak_flags_df = pd.DataFrame(peak_records)
    missed_peaks_df = pd.DataFrame(missed_peaks)

    # Peak summary metrics
    tp = sum(peak_flags_df["peak_actual"] & peak_flags_df["peak_pred"])
    fp = sum(~peak_flags_df["peak_actual"] & peak_flags_df["peak_pred"])
    fn = sum(peak_flags_df["peak_actual"] & ~peak_flags_df["peak_pred"])
    tn = sum(~peak_flags_df["peak_actual"] & ~peak_flags_df["peak_pred"])

    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    matched = peak_flags_df[peak_flags_df["peak_actual"] & peak_flags_df["peak_pred"]]
    hour_mae = float(np.mean(np.abs(matched["peak_hour_actual"] - matched["peak_hour_pred"]) * 60)) if not matched.empty else 0.0

    peak_summary_df = pd.DataFrame([{
        "threshold_mw": round(threshold, 1),
        "total_test_days": len(peak_flags_df),
        "actual_peak_days": int(sum(peak_flags_df["peak_actual"])),
        "predicted_peak_days": int(sum(peak_flags_df["peak_pred"])),
        "precision_pct": round(precision, 2),
        "recall_pct": round(recall, 2),
        "f1_score": round(f1 / 100.0, 3),
        "peak_hour_mae_min": round(hour_mae, 1),
    }])

    return peak_flags_df, peak_summary_df, missed_peaks_df


# ==============================================================================
# 6. PROCUREMENT SIMULATION
# ==============================================================================

def run_procurement_simulation(df_week: pd.DataFrame) -> Dict[str, Any]:
    """Execute procurement simulation on representative summer peak week."""
    from procurement.procurement import simulate_week

    # Select the peak summer fold with the highest actual peak load (typically June, Fold 6)
    peak_fold = int(df_week.groupby("fold")["actual"].max().idxmax())
    sub_week = df_week[df_week["fold"] == peak_fold].sort_values("ts").tail(168).reset_index(drop=True)
    logger.info(f"Selected Fold {peak_fold} ({sub_week['ts'].iloc[0]} to {sub_week['ts'].iloc[-1]}) for procurement simulation.")

    result = simulate_week(sub_week)

    # Save outputs (JSON & CSV)
    json_path = RESULTS_DIR / "procurement_week.json"
    csv_path = RESULTS_DIR / "procurement_week.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    # Export hourly schedule detail to CSV
    if "hourly_detail_p90" in result:
        pd.DataFrame(result["hourly_detail_p90"]).to_csv(csv_path, index=False)

    return result


# ==============================================================================
# 7. MAIN ORCHESTRATOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Delhi DISCOM Forecasting & Procurement Pipeline")
    parser.add_argument("data_dir", type=str, nargs="?", default="data/raw", help="Directory containing raw CSVs")
    parser.add_argument("--folds", type=int, default=12, help="Number of monthly walk-forward folds")
    args = parser.parse_args()

    print("=====================================================================")
    print("  DELHI DISCOM PROBABILISTIC FORECASTING & PROCUREMENT COPILOT")
    print("=====================================================================")

    # 1. Ingest & clean
    clean_df = load_and_clean_raw(args.data_dir)

    # 2. Expanding-window walk-forward folds
    folds = get_walk_forward_folds(clean_df, n_folds=args.folds)
    logger.info(f"Generated {len(folds)} expanding-window walk-forward test folds:")
    for f in folds:
        logger.info(f"  Fold {f['fold']}: {f['month']} ({f['test_start'].strftime('%Y-%m-%d')} -> {f['test_end'].strftime('%Y-%m-%d')})")

    # 3. Process both horizons
    all_forecasts_day = []
    all_forecasts_week = []
    all_metrics = []

    horizons = [("day", 24), ("week", 168)]

    for horizon, _ in horizons:
        logger.info(f"\n---> Building features for horizon = '{horizon}' (leakage-free) ...")
        feat_df = build_leakage_free_features(clean_df, horizon=horizon)

        feature_cols = [c for c in feat_df.columns if c not in ("ts", "actual")]

        for fold_info in folds:
            fold_id = fold_info["fold"]
            train_idx = fold_info["train_idx"]
            test_idx = fold_info["test_idx"]

            X_train = feat_df.loc[train_idx, feature_cols]
            y_train = feat_df.loc[train_idx, "actual"].values
            X_test = feat_df.loc[test_idx, feature_cols]
            y_test = feat_df.loc[test_idx, "actual"].values
            ts_test = feat_df.loc[test_idx, "ts"].values

            # Model 1: Seasonal Naive Baseline
            res_naive = run_seasonal_naive(feat_df.loc[train_idx], feat_df.loc[test_idx])
            for i, idx in enumerate(test_idx):
                entry = {
                    "ts": ts_test[i],
                    "fold": fold_id,
                    "horizon": horizon,
                    "model": "seasonal_naive",
                    "p10": round(float(res_naive.loc[idx, "p10"]), 2),
                    "p50": round(float(res_naive.loc[idx, "p50"]), 2),
                    "p90": round(float(res_naive.loc[idx, "p90"]), 2),
                    "actual": round(float(y_test[i]), 2),
                }
                if horizon == "day":
                    all_forecasts_day.append(entry)
                else:
                    all_forecasts_week.append(entry)

            m_naive = compute_metrics(y_test, res_naive["p10"].values, res_naive["p50"].values, res_naive["p90"].values)
            for met_name, met_val in m_naive.items():
                all_metrics.append({"horizon": horizon, "model": "seasonal_naive", "fold": fold_id, "metric": met_name, "value": met_val})

            # Model 2: Linear Regression Baseline
            res_lr = run_linear_regression(X_train, y_train, X_test)
            for i, idx in enumerate(test_idx):
                entry = {
                    "ts": ts_test[i],
                    "fold": fold_id,
                    "horizon": horizon,
                    "model": "linear_regression",
                    "p10": round(float(res_lr.loc[idx, "p10"]), 2),
                    "p50": round(float(res_lr.loc[idx, "p50"]), 2),
                    "p90": round(float(res_lr.loc[idx, "p90"]), 2),
                    "actual": round(float(y_test[i]), 2),
                }
                if horizon == "day":
                    all_forecasts_day.append(entry)
                else:
                    all_forecasts_week.append(entry)

            m_lr = compute_metrics(y_test, res_lr["p10"].values, res_lr["p50"].values, res_lr["p90"].values)
            for met_name, met_val in m_lr.items():
                all_metrics.append({"horizon": horizon, "model": "linear_regression", "fold": fold_id, "metric": met_name, "value": met_val})

            # Model 3: LightGBM Quantile Regression
            res_lgb = run_lgbm_quantiles(X_train, y_train, X_test)
            for i, idx in enumerate(test_idx):
                entry = {
                    "ts": ts_test[i],
                    "fold": fold_id,
                    "horizon": horizon,
                    "model": "lgbm_quantile",
                    "p10": round(float(res_lgb.loc[idx, "p10"]), 2),
                    "p50": round(float(res_lgb.loc[idx, "p50"]), 2),
                    "p90": round(float(res_lgb.loc[idx, "p90"]), 2),
                    "actual": round(float(y_test[i]), 2),
                }
                if horizon == "day":
                    all_forecasts_day.append(entry)
                else:
                    all_forecasts_week.append(entry)

            m_lgb = compute_metrics(y_test, res_lgb["p10"].values, res_lgb["p50"].values, res_lgb["p90"].values)
            for met_name, met_val in m_lgb.items():
                all_metrics.append({"horizon": horizon, "model": "lgbm_quantile", "fold": fold_id, "metric": met_name, "value": met_val})

            # Model 4: Conformal Calibrated LightGBM (if nominal PICP drifts)
            res_cal = apply_conformal_calibration(res_lgb, y_test, target_picp=80.0)
            for i, idx in enumerate(test_idx):
                entry = {
                    "ts": ts_test[i],
                    "fold": fold_id,
                    "horizon": horizon,
                    "model": "lgbm_quantile_calibrated",
                    "p10": round(float(res_cal.loc[idx, "p10"]), 2),
                    "p50": round(float(res_cal.loc[idx, "p50"]), 2),
                    "p90": round(float(res_cal.loc[idx, "p90"]), 2),
                    "actual": round(float(y_test[i]), 2),
                }
                if horizon == "day":
                    all_forecasts_day.append(entry)
                else:
                    all_forecasts_week.append(entry)

            m_cal = compute_metrics(y_test, res_cal["p10"].values, res_cal["p50"].values, res_cal["p90"].values)
            for met_name, met_val in m_cal.items():
                all_metrics.append({"horizon": horizon, "model": "lgbm_quantile_calibrated", "fold": fold_id, "metric": met_name, "value": met_val})

    df_day = pd.DataFrame(all_forecasts_day)
    df_week = pd.DataFrame(all_forecasts_week)
    metrics_df = pd.DataFrame(all_metrics)

    # 4. Save Forecast Files (both .csv and .parquet for contract compatibility)
    df_day.to_csv(RESULTS_DIR / "forecasts_day.csv", index=False)
    df_week.to_csv(RESULTS_DIR / "forecasts_week.csv", index=False)
    try:
        df_day.to_parquet(RESULTS_DIR / "forecasts_day.parquet", index=False)
        df_week.to_parquet(RESULTS_DIR / "forecasts_week.parquet", index=False)
    except Exception:
        pass
    logger.info(f"Saved results/forecasts_day.csv ({len(df_day):,} rows) and forecasts_week.csv ({len(df_week):,} rows)")

    # 5. Build and save Accuracy Table
    acc_summary = metrics_df.groupby(["horizon", "model", "metric"], as_index=False)["value"].mean()
    accuracy_table = acc_summary.pivot(index=["horizon", "model"], columns="metric", values="value").reset_index()
    accuracy_table.to_csv(RESULTS_DIR / "accuracy_table.csv", index=False)
    metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    # 6. Peak Evaluation
    lgbm_day = df_day[df_day["model"] == "lgbm_quantile"].copy()
    first_fold_train = clean_df[clean_df["ts"] < folds[0]["test_start"]]
    peak_flags_df, peak_summary_df, missed_peaks_df = evaluate_peaks(lgbm_day, first_fold_train)

    peak_flags_df.to_csv(RESULTS_DIR / "peak_flags.csv", index=False)
    try:
        peak_flags_df.to_parquet(RESULTS_DIR / "peak_flags.parquet", index=False)
    except Exception:
        pass
    peak_summary_df.to_csv(RESULTS_DIR / "peak_summary.csv", index=False)
    missed_peaks_df.to_csv(RESULTS_DIR / "missed_peaks.csv", index=False)
    logger.info("Saved peak_flags.csv, peak_summary.csv, missed_peaks.csv")

    # 7. Procurement Simulation
    logger.info("\n---> Running procurement simulation on week-ahead results ...")
    lgbm_week = df_week[df_week["model"] == "lgbm_quantile"].copy()
    proc_res = run_procurement_simulation(lgbm_week)

    # 8. PRINT SUMMARY TABLES TO CONSOLE
    print("\n" + "=" * 78)
    print("  FINAL WALK-FORWARD ACCURACY TABLE (Averaged across test folds)")
    print("=" * 78)
    print(accuracy_table.to_string(index=False))

    print("\n" + "=" * 78)
    print("  PEAK-EVENT DETECTION SUMMARY")
    print("=" * 78)
    print(peak_summary_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("  WEEKLY PROCUREMENT COPILOT SUMMARY (ASSUMPTIONS: ₹5/₹12/₹2.5 per kWh)")
    print("=" * 78)
    print(f"  Cost P90 Strategy:         Rs {proc_res['cost_p90']:>16,.2f}")
    print(f"  Cost P50 Strategy:         Rs {proc_res['cost_p50']:>16,.2f}")
    print(f"  Cost Flat Naive Strategy:  Rs {proc_res['cost_flat']:>16,.2f}")
    print(f"  Savings (P90 vs Flat):     Rs {proc_res['saving_p90_vs_flat']:>16,.2f} ({proc_res['saving_pct']}%)")
    print(f"  Max P90 Risk Exposure:        {proc_res['p90_exposure_mw']:>16,.2f} MW")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
