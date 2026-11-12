"""
eval/backtest.py
================
Expanding-window walk-forward backtest harness for DISCOM load forecasting.

CONTRACT
--------
get_folds(df)  -> list[dict]
    df must have a DatetimeIndex at hourly (or sub-hourly) frequency.
    Returns one fold dict per calendar month in the last 12 months of df,
    so every season appears at least once.

    Each fold dict:
        {
            "fold_id":    int,            # 0-based
            "month_label": str,           # "2025-11"
            "train":      pd.DataFrame,   # all rows BEFORE the test block
            "test":       pd.DataFrame,   # rows in that calendar month
            "origin":     pd.Timestamp,   # last timestamp in train (exclusive start of test)
            "horizon_h":  int,            # len(test) in hours (may exceed horizon if sub-daily)
        }

run_backtest(fit_predict, horizon, df, results_dir)
    fit_predict(train_df, test_df, horizon) -> pd.DataFrame
        Must return a DataFrame indexed like test_df with columns:
            p10, p50, p90   (floats, MW)
        horizon: "day_ahead" | "week_ahead"

    Writes results/forecasts_{horizon}.parquet with columns:
        fold_id, timestamp, actual, p10, p50, p90, model

BASELINES (bundled)
-------------------
seasonal_naive_fit_predict  — same hour, same weekday, previous week (t-168h)
linear_regression_fit_predict — OLS on [hour, dayofweek, month, day_index]
                                Intervals from residual quantiles on train window.

Usage
-----
    from eval.backtest import run_backtest, seasonal_naive_fit_predict
    run_backtest(seasonal_naive_fit_predict, horizon="day_ahead", df=raw_df)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
FitPredictFn = Callable[[pd.DataFrame, pd.DataFrame, str], pd.DataFrame]
Horizon = Literal["day_ahead", "week_ahead"]

REQUIRED_COLS = {"actual"}          # minimum columns in df
OUTPUT_COLS   = ["p10", "p50", "p90"]

# ---------------------------------------------------------------------------
# get_folds
# ---------------------------------------------------------------------------

def get_folds(df: pd.DataFrame) -> list[dict]:
    """
    Build expanding-window folds covering the last 12 calendar months of *df*.

    Rules
    -----
    - Train  : all rows strictly before the first day of the test month.
    - Test   : all rows in that calendar month (full month boundaries).
    - No gap, no shuffle, no leakage.
    - Minimum train size : 4 weeks (672 h) — folds shorter than this are skipped
      with a warning so the model always has enough history.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly (or sub-hourly) DataFrame with DatetimeIndex.
        Must contain at least an ``actual`` column.

    Returns
    -------
    list[dict]  — see module docstring for field definitions.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex")
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    df = df.sort_index()

    # Last 12 calendar months (inclusive) ending at the last month in df
    last_ts    = df.index[-1]
    last_month = last_ts.to_period("M")
    test_months = pd.period_range(end=last_month, periods=12, freq="M")

    MIN_TRAIN_HOURS = 672  # 4 weeks

    folds = []
    for fold_id, month in enumerate(test_months):
        test_start = month.to_timestamp(how="start")
        test_end   = month.to_timestamp(how="end")   # inclusive

        train_mask = df.index < test_start
        test_mask  = (df.index >= test_start) & (df.index <= test_end)

        train = df[train_mask]
        test  = df[test_mask]

        if len(train) < MIN_TRAIN_HOURS:
            logger.warning(
                "Fold %d (%s): only %d training hours — skipping (need >= %d).",
                fold_id, month, len(train), MIN_TRAIN_HOURS,
            )
            continue
        if len(test) == 0:
            logger.warning("Fold %d (%s): empty test block — skipping.", fold_id, month)
            continue

        folds.append(
            {
                "fold_id":     fold_id,
                "month_label": str(month),
                "train":       train.copy(),
                "test":        test.copy(),
                "origin":      train.index[-1],
                "horizon_h":   len(test),
            }
        )
        logger.info(
            "Fold %d | %s | train=%d h | test=%d h",
            fold_id, month, len(train), len(test),
        )

    if not folds:
        raise RuntimeError("No valid folds produced. Check that df spans at least 5 weeks.")

    return folds


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------

def run_backtest(
    fit_predict: FitPredictFn,
    horizon: Horizon,
    df: pd.DataFrame,
    results_dir: str | Path = "results",
    model_name: str = "custom",
) -> pd.DataFrame:
    """
    Run the walk-forward backtest and persist predictions.

    Parameters
    ----------
    fit_predict : callable
        fit_predict(train_df, test_df, horizon) → DataFrame[p10, p50, p90]
        The returned DataFrame must be indexed identically to *test_df*.
    horizon : "day_ahead" | "week_ahead"
        Passed through to fit_predict and used to name the output file.
    df : pd.DataFrame
        Full historical DataFrame with DatetimeIndex + ``actual`` column.
    results_dir : path-like
        Directory where ``forecasts_{horizon}.parquet`` is written.
    model_name : str
        Label stored in the ``model`` column of the output file.

    Returns
    -------
    pd.DataFrame
        Combined predictions across all folds (also written to disk).
    """
    if horizon not in ("day_ahead", "week_ahead"):
        raise ValueError(f"horizon must be 'day_ahead' or 'week_ahead', got {horizon!r}")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    folds = get_folds(df)
    all_records: list[pd.DataFrame] = []

    for fold in folds:
        logger.info("Running fold %d (%s) …", fold["fold_id"], fold["month_label"])

        preds = fit_predict(fold["train"], fold["test"], horizon)

        # --- validate output ---
        if not isinstance(preds, pd.DataFrame):
            raise TypeError(f"fit_predict must return a DataFrame, got {type(preds)}")
        missing_cols = set(OUTPUT_COLS) - set(preds.columns)
        if missing_cols:
            raise ValueError(f"Fold {fold['fold_id']}: fit_predict missing columns {missing_cols}")
        if len(preds) != len(fold["test"]):
            raise ValueError(
                f"Fold {fold['fold_id']}: fit_predict returned {len(preds)} rows "
                f"but test has {len(fold['test'])} rows."
            )

        record = preds[OUTPUT_COLS].copy()
        record.index = fold["test"].index
        record["actual"]  = fold["test"]["actual"].values
        record["fold_id"] = fold["fold_id"]
        record["model"]   = model_name
        record.index.name = "timestamp"
        all_records.append(record)

    combined = pd.concat(all_records).sort_index()
    combined = combined.reset_index()

    out_path = results_dir / f"forecasts_{horizon}.parquet"
    combined.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    logger.info("Wrote %d rows → %s", len(combined), out_path)

    return combined


# ---------------------------------------------------------------------------
# Baseline 1: Seasonal Naïve (same hour, previous week = t-168h)
# ---------------------------------------------------------------------------

def _residual_quantile_intervals(
    residuals: np.ndarray,
    point_forecast: np.ndarray,
    q_low: float = 0.1,
    q_high: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute quantile prediction intervals from training-window residuals.

    Method (Hyndman & Athanasopoulos §5.5)
    ---------------------------------------
    Given empirical residuals  e_t = y_t − ŷ_t  from the training window,
    we estimate the forecast distribution non-parametrically:

        P10 = ŷ_new + quantile(e, 0.10)
        P90 = ŷ_new + quantile(e, 0.90)

    This is valid when errors are approximately i.i.d. and unbiased.
    For seasonal naïve the lag matches the forecast lag, so this assumption
    holds reasonably well in-distribution.

    Parameters
    ----------
    residuals      : 1-D array of training residuals (actual − predicted).
    point_forecast : 1-D array of test-set point forecasts.
    q_low, q_high  : quantile levels for lower/upper bound.
    """
    q_lo = np.quantile(residuals, q_low)
    q_hi = np.quantile(residuals, q_high)
    return point_forecast + q_lo, point_forecast + q_hi


def seasonal_naive_fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: Horizon,
) -> pd.DataFrame:
    """
    Seasonal-naïve baseline: for each test timestamp t, predict the value at
    t − 168 h (same hour last week).  Falls back to t − 24 h if t − 168 h is
    not in the history index.

    Prediction intervals are derived from the in-sample one-week-ahead
    residuals on the training window (residual quantile method — see
    _residual_quantile_intervals).

    Returns
    -------
    pd.DataFrame with columns [p10, p50, p90], indexed like test_df.
    """
    history = pd.concat([train_df, test_df]).sort_index()["actual"]

    # --- point forecast ---
    p50 = []
    for ts in test_df.index:
        lag168 = ts - pd.Timedelta(hours=168)
        lag24  = ts - pd.Timedelta(hours=24)
        if lag168 in history.index:
            p50.append(history[lag168])
        elif lag24 in history.index:
            p50.append(history[lag24])
        else:
            p50.append(np.nan)
    p50 = np.array(p50, dtype=float)

    # --- training residuals (one-week-ahead, in-sample) ---
    actual_train = train_df["actual"].values
    idx_train    = train_df.index

    residuals = []
    for i, ts in enumerate(idx_train):
        lag = ts - pd.Timedelta(hours=168)
        if lag in train_df.index:
            pred = train_df.loc[lag, "actual"]
            residuals.append(actual_train[i] - pred)
    residuals = np.array(residuals, dtype=float)

    if len(residuals) < 10:
        # degenerate fallback: symmetric 10 % of median
        med    = float(np.nanmedian(p50))
        offset = 0.10 * med
        p10    = p50 - offset
        p90    = p50 + offset
    else:
        p10, p90 = _residual_quantile_intervals(residuals, p50)

    return pd.DataFrame(
        {"p10": p10, "p50": p50, "p90": p90},
        index=test_df.index,
    )


# ---------------------------------------------------------------------------
# Baseline 2: Linear Regression (OLS) with quantile intervals
# ---------------------------------------------------------------------------

def _build_calendar_features(df: pd.DataFrame) -> np.ndarray:
    """
    Minimal calendar feature matrix for the linear baseline.
    Columns: hour, dayofweek, month, day_index (linear trend, days since start).
    """
    idx = df.index
    # day_index: fractional days since the first timestamp
    day_index = (idx - idx[0]).total_seconds() / 86_400.0
    X = np.column_stack([
        idx.hour,
        idx.dayofweek,
        idx.month,
        day_index,
    ])
    return X.astype(float)


def linear_regression_fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: Horizon,
) -> pd.DataFrame:
    """
    OLS linear regression baseline on calendar features.

    Features:  hour-of-day, day-of-week, month, linear day-index.
    Intervals: residual-quantile method on training residuals (see
               _residual_quantile_intervals for derivation).

    Note: The day_index in test_df is computed relative to the first
    timestamp in *train_df* so that the trend term is consistent across
    train and test without leaking future information.

    Returns
    -------
    pd.DataFrame with columns [p10, p50, p90], indexed like test_df.
    """
    # Build feature matrices — reference day 0 = train start
    t0 = train_df.index[0]

    def _features(df: pd.DataFrame) -> np.ndarray:
        idx       = df.index
        day_index = (idx - t0).total_seconds() / 86_400.0
        return np.column_stack([
            idx.hour, idx.dayofweek, idx.month, day_index,
        ]).astype(float)

    X_train = _features(train_df)
    y_train = train_df["actual"].values.astype(float)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Training residuals for interval estimation
    y_hat_train = model.predict(X_train)
    residuals   = y_train - y_hat_train

    # Test predictions
    X_test = _features(test_df)
    p50    = model.predict(X_test)

    p10, p90 = _residual_quantile_intervals(residuals, p50)

    return pd.DataFrame(
        {"p10": p10, "p50": p50, "p90": p90},
        index=test_df.index,
    )


# ---------------------------------------------------------------------------
# CLI entry-point for quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run backtest with a named baseline")
    parser.add_argument("--parquet", required=True,
                        help="Path to input parquet with DatetimeIndex + actual column")
    parser.add_argument("--horizon", default="day_ahead",
                        choices=["day_ahead", "week_ahead"])
    parser.add_argument("--baseline", default="seasonal_naive",
                        choices=["seasonal_naive", "linear"])
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    _df = pd.read_parquet(args.parquet)
    if not isinstance(_df.index, pd.DatetimeIndex):
        _df = _df.set_index("timestamp")
    _df.index = pd.to_datetime(_df.index)

    _baseline_map = {
        "seasonal_naive": seasonal_naive_fit_predict,
        "linear":         linear_regression_fit_predict,
    }
    _fn = _baseline_map[args.baseline]

    run_backtest(_fn, horizon=args.horizon, df=_df,
                 results_dir=args.results_dir, model_name=args.baseline)
