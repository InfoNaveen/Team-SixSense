"""
models/pipeline.py
==================
Modeling Pipeline Runner for Person C (Modeling).

Handles:
    - Day-ahead and week-ahead quantile forecasts (raw + conformal calibrated).
    - Peak thresholding, classification, and peak-hour prediction.
    - Exporting contract-compliant parquet files to results/:
        * results/forecasts_day.parquet
        * results/forecasts_week.parquet
        * results/peak_flags.parquet
    - Exports feature importance and calibration documentation.

Adapts automatically:
    If eval/backtest.py and features/build_features.py exist, uses the shared team harness.
    Otherwise, runs against the synthetic walk-forward dummy generator so results/
    files remain valid and available for Dashboard (Data) and Procurement (Lead).
"""

from __future__ import annotations
import os
import sys
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# Add repo root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.quantile_lgbm import QuantileLGBMForecaster, fit_predict, fit_predict_day, fit_predict_week
from models.calibration import ConformalCalibrator, compute_picp, compute_pinball_loss
from models.peak_classifier import (
    compute_training_peak_threshold,
    PeakClassifier,
    generate_peak_flags_table,
    save_peak_flags,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("models.pipeline")


def _run_with_shared_harness(horizon: str) -> Optional[pd.DataFrame]:
    """Attempt to run through Features' eval.backtest harness if available."""
    try:
        import eval.backtest as bt  # type: ignore
        if hasattr(bt, "run_backtest"):
            logger.info(f"Using shared backtest harness for horizon='{horizon}'...")
            fit_fn = fit_predict_day if horizon == "day" else fit_predict_week
            df_res = bt.run_backtest(fit_fn, horizon)
            return df_res
    except ImportError:
        logger.info("Shared backtest harness (eval/backtest.py) not found yet.")
    except Exception as e:
        logger.warning(f"Error calling shared backtest harness: {e}")
    return None


def generate_contract_backtest(
    n_days: int = 150,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs a 3-fold expanding walk-forward backtest simulating the contract

    when Features' eval/backtest.py is not yet committed.
    """
    from models.test_models_dummy import generate_synthetic_load_data

    logger.info("Generating contract-compliant walk-forward backtest (3 folds)...")
    df = generate_synthetic_load_data(n_days=n_days, seed=seed)

    feature_cols = [
        "temp_c", "rh_pct", "apparent_temp_c", "cdh",
        "hour", "dayofweek", "is_weekend", "is_holiday",
        "load_lag_48", "load_lag_168", "temp_lag_24",
        "rolling_7d_mean_load", "rolling_7d_max_load"
    ]

    # Define 3 walk-forward folds (monthly test blocks)
    # Fold 1: train 0..60 days, val 60..70 days, test 70..90 days
    # Fold 2: train 0..80 days, val 80..90 days, test 90..110 days
    # Fold 3: train 0..100 days, val 100..110 days, test 110..130 days
    fold_configs = [
        {"fold": 1, "train_days": 60, "val_days": 10, "test_days": 20},
        {"fold": 2, "train_days": 80, "val_days": 10, "test_days": 20},
        {"fold": 3, "train_days": 100, "val_days": 10, "test_days": 20},
    ]

    all_day_forecasts: List[pd.DataFrame] = []
    all_week_forecasts: List[pd.DataFrame] = []
    all_peak_flags: List[pd.DataFrame] = []
    cal_reports: List[str] = []

    for cfg in fold_configs:
        f_idx = cfg["fold"]
        t_end = cfg["train_days"] * 24
        v_end = t_end + cfg["val_days"] * 24
        test_end = v_end + cfg["test_days"] * 24

        train_df = df.iloc[:t_end]
        val_df = df.iloc[t_end:v_end]
        test_df = df.iloc[v_end:test_end]

        X_train, y_train = train_df[feature_cols], train_df["load_mw"]
        X_val, y_val = val_df[feature_cols], val_df["load_mw"]
        X_test, y_test = test_df[feature_cols], test_df["load_mw"]

        # 1. Day-ahead quantile model
        forecaster_day = QuantileLGBMForecaster(horizon="day", random_state=42 + f_idx)
        forecaster_day.fit(X_train, y_train)
        preds_day = forecaster_day.predict(X_test)

        # Conformal interval calibration on validation fold
        val_preds_day = forecaster_day.predict(X_val)
        calibrator_day = ConformalCalibrator(target_coverage=0.80, method="asymmetric")
        calibrator_day.fit(y_val, val_preds_day["p10"], val_preds_day["p90"])
        preds_day_cal = calibrator_day.calibrate_dataframe(preds_day)

        picp_raw = compute_picp(y_test, preds_day["p10"], preds_day["p90"])
        picp_cal = compute_picp(y_test, preds_day_cal["p10"], preds_day_cal["p90"])
        pin_raw = compute_pinball_loss(y_test, preds_day["p10"], preds_day["p50"], preds_day["p90"])["pinball_mean"]
        pin_cal = compute_pinball_loss(y_test, preds_day_cal["p10"], preds_day_cal["p50"], preds_day_cal["p90"])["pinball_mean"]

        cal_reports.append(
            f"| Fold {f_idx} (Day) | {picp_raw*100:.1f}% | {picp_cal*100:.1f}% | {pin_raw:.2f} | {pin_cal:.2f} | "
            f"c_low={calibrator_day.c_low:.1f}, c_high={calibrator_day.c_high:.1f} |"
        )

        # Build day forecast rows (both raw and calibrated)
        df_raw_day = pd.DataFrame({
            "ts": test_df["ts"].values,
            "fold": f_idx,
            "horizon": "day",
            "model": "lgbm_quantile",
            "p10": preds_day["p10"].values,
            "p50": preds_day["p50"].values,
            "p90": preds_day["p90"].values,
            "actual": y_test.values,
        })
        df_cal_day = pd.DataFrame({
            "ts": test_df["ts"].values,
            "fold": f_idx,
            "horizon": "day",
            "model": "lgbm_quantile_calibrated",
            "p10": preds_day_cal["p10"].values,
            "p50": preds_day_cal["p50"].values,
            "p90": preds_day_cal["p90"].values,
            "actual": y_test.values,
        })
        all_day_forecasts.extend([df_raw_day, df_cal_day])

        # 2. Week-ahead quantile model
        forecaster_week = QuantileLGBMForecaster(horizon="week", random_state=142 + f_idx)
        forecaster_week.fit(X_train, y_train)
        preds_week = forecaster_week.predict(X_test)

        val_preds_week = forecaster_week.predict(X_val)
        calibrator_week = ConformalCalibrator(target_coverage=0.80, method="asymmetric")
        calibrator_week.fit(y_val, val_preds_week["p10"], val_preds_week["p90"])
        preds_week_cal = calibrator_week.calibrate_dataframe(preds_week)

        df_raw_week = pd.DataFrame({
            "ts": test_df["ts"].values,
            "fold": f_idx,
            "horizon": "week",
            "model": "lgbm_quantile",
            "p10": preds_week["p10"].values,
            "p50": preds_week["p50"].values,
            "p90": preds_week["p90"].values,
            "actual": y_test.values,
        })
        df_cal_week = pd.DataFrame({
            "ts": test_df["ts"].values,
            "fold": f_idx,
            "horizon": "week",
            "model": "lgbm_quantile_calibrated",
            "p10": preds_week_cal["p10"].values,
            "p50": preds_week_cal["p50"].values,
            "p90": preds_week_cal["p90"].values,
            "actual": y_test.values,
        })
        all_week_forecasts.extend([df_raw_week, df_cal_week])

        # 3. Peak classifier & peak flags
        threshold_mw = compute_training_peak_threshold(
            y_train_hourly=train_df["load_mw"],
            dates_train=train_df["ts"],
            quantile=0.90,
        )

        hourly_for_peaks = pd.DataFrame({
            "ts": test_df["ts"].values,
            "p50": preds_day["p50"].values,
            "actual": y_test.values,
        })

        df_peak = generate_peak_flags_table(
            df_hourly_test=hourly_for_peaks,
            threshold_mw=threshold_mw,
            fold=f_idx,
            horizon="day",
            ts_col="ts",
            p50_col="p50",
            actual_col="actual",
        )
        all_peak_flags.append(df_peak)

    # Save feature importance from latest fold
    df_imp = forecaster_day.get_feature_importances()
    df_imp.to_csv("models/feature_importance.csv", index=False)
    logger.info("Saved models/feature_importance.csv")

    # Update calibration documentation
    joined_reports = "\n".join(cal_reports)
    cal_notes_content = f"""# Interval Calibration Notes

## Methodology: Conformal Quantile Regression (CQR) Shift
To ensure honest uncertainty quantification and satisfy the hackathon PICP criteria (~80% coverage for P10-P90 interval), we implement a finite-sample, distribution-free conformal calibration shift on the walk-forward validation window.

### Nonconformity Definition
- Lower residual: $E^{{\\text{{low}}}}_i = \\hat{{q}}_{{0.1}}(x_i) - y_i$
- Upper residual: $E^{{\\text{{high}}}}_i = y_i - \\hat{{q}}_{{0.9}}(x_i)$

Conformal quantile offsets $c_{{\\text{{low}}}}, c_{{\\text{{high}}}}$ are estimated at the $(1 - \\alpha/2)(1 + 1/n)$ quantile level ($\\alpha = 0.20$ for 80% coverage).
Calibrated intervals are:
$$P_{{10}}^{{\\text{{cal}}}} = \\hat{{q}}_{{0.1}} - c_{{\\text{{low}}}}$$
$$P_{{90}}^{{\\text{{cal}}}} = \\hat{{q}}_{{0.9}} + c_{{\\text{{high}}}}$$
Post-shift monotonicity sorting ($p_{{10}} \\le p_{{50}} \\le p_{{90}}$) is strictly applied.

## Empirical Walk-Forward Calibration Results
| Fold / Window | Raw PICP | Calibrated PICP | Raw Pinball (MW) | Calibrated Pinball (MW) | Conformal Shifts |
| :--- | :--- | :--- | :--- | :--- | :--- |
{joined_reports}

## Production Contract
- Raw predictions saved as: `model="lgbm_quantile"`
- Calibrated predictions saved as: `model="lgbm_quantile_calibrated"`
"""
    with open("models/calibration_notes.md", "w", encoding="utf-8") as f:
        f.write(cal_notes_content)
    logger.info("Saved models/calibration_notes.md")

    df_forecasts_day = pd.concat(all_day_forecasts, ignore_index=True)
    df_forecasts_week = pd.concat(all_week_forecasts, ignore_index=True)
    df_peak_flags = pd.concat(all_peak_flags, ignore_index=True)

    return df_forecasts_day, df_forecasts_week, df_peak_flags


def run_pipeline():
    """Main pipeline execution routine."""
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Attempt using shared harness first
    day_res = _run_with_shared_harness("day")
    week_res = _run_with_shared_harness("week")

    if day_res is not None and week_res is not None:
        logger.info("Backtest harness successfully executed from eval/backtest.py.")
        # Save results directly
        day_res.to_parquet("results/forecasts_day.parquet", index=False)
        week_res.to_parquet("results/forecasts_week.parquet", index=False)
    else:
        logger.info("Executing self-contained walk-forward modeling pipeline...")
        df_day, df_week, df_peak = generate_contract_backtest()

        # Enforce contract column selection
        contract_cols_forecast = ["ts", "fold", "horizon", "model", "p10", "p50", "p90", "actual"]
        df_day = df_day[contract_cols_forecast]
        df_week = df_week[contract_cols_forecast]

        contract_cols_peak = [
            "date", "fold", "peak_prob", "peak_pred",
            "peak_hour_pred", "peak_actual", "peak_hour_actual", "threshold_mw"
        ]
        df_peak = df_peak[contract_cols_peak]

        df_day.to_parquet("results/forecasts_day.parquet", index=False)
        df_week.to_parquet("results/forecasts_week.parquet", index=False)
        df_peak.to_parquet("results/peak_flags.parquet", index=False)

        logger.info(f"Saved results/forecasts_day.parquet ({len(df_day)} rows)")
        logger.info(f"Saved results/forecasts_week.parquet ({len(df_week)} rows)")
        logger.info(f"Saved results/peak_flags.parquet ({len(df_peak)} rows)")

    logger.info("Pipeline execution finished successfully.")


if __name__ == "__main__":
    run_pipeline()
