"""
models/peak_classifier.py
=========================
Peak-Event Detection and Peak-Hour Prediction.

Contains:
    1. Threshold calculation strictly on training folds (90th percentile of daily maxima).
    2. Binary LightGBM classifier for day's peak exceeding top-decile threshold.
    3. Peak-hour predictor (argmax of P50 curve and optional multi-class hour classifier).
    4. Generator and writer for results/peak_flags.parquet per the contract.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

# Lightweight laptop-friendly parameters for binary peak classifier
DEFAULT_CLASSIFIER_PARAMS: Dict[str, Any] = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 20,
    "max_depth": 5,
    "min_child_samples": 15,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "scale_pos_weight": 1.5,  # Minor upweight for top-decile imbalance (~10% positive)
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}


def compute_training_peak_threshold(
    y_train_hourly: Union[pd.Series, np.ndarray],
    dates_train: Union[pd.Series, np.ndarray, pd.DatetimeIndex],
    quantile: float = 0.90,
) -> float:
    """Computes the peak load threshold (e.g., 90th percentile of daily maxima)

    strictly on training data for a fold.

    Parameters
    ----------
    y_train_hourly : Series or array
        Hourly load values in training fold.
    dates_train : Series, array, or DatetimeIndex
        Dates or timestamps corresponding to y_train_hourly.
    quantile : float
        Quantile threshold for peak day (default 0.90).

    Returns
    -------
    float
        Threshold load in MW.
    """
    ts = pd.to_datetime(dates_train)
    date_vals = ts.dt.date if hasattr(ts, "dt") else ts.date
    df = pd.DataFrame({"y": np.asarray(y_train_hourly).ravel(), "date": date_vals})
    daily_maxima = df.groupby("date")["y"].max().dropna()
    if len(daily_maxima) == 0:
        raise ValueError("Cannot compute peak threshold on empty training series.")
    threshold = float(np.quantile(daily_maxima.values, quantile))
    logger.info(f"Computed fold training peak threshold (q={quantile}): {threshold:.2f} MW from {len(daily_maxima)} days")
    return threshold


def predict_peak_hour_from_p50(
    df_hourly_preds: pd.DataFrame,
    p50_col: str = "p50",
    ts_col: Optional[str] = "ts",
) -> pd.DataFrame:
    """Predicts peak hour and magnitude for each day as the argmax of the hourly P50 load curve.

    This preserves temporal consistency with the quantile forecaster.

    Parameters
    ----------
    df_hourly_preds : pd.DataFrame
        Hourly predictions with timestamp and p50 column.
    p50_col : str
        Column name for median load forecast.
    ts_col : str, optional
        Column name for timestamp. If None, uses df_hourly_preds.index.

    Returns
    -------
    pd.DataFrame
        Daily DataFrame indexed by date with columns:
        ['peak_hour_pred', 'peak_magnitude_pred']
    """
    df = df_hourly_preds.copy()
    if ts_col and ts_col in df.columns:
        ts_series = pd.to_datetime(df[ts_col])
    else:
        ts_series = pd.to_datetime(df.index)

    df["date"] = ts_series.dt.date
    df["hour"] = ts_series.dt.hour

    daily_records = []
    for date, group in df.groupby("date"):
        max_idx = group[p50_col].idxmax()
        peak_hour = int(group.loc[max_idx, "hour"])
        peak_mag = float(group.loc[max_idx, p50_col])
        daily_records.append({"date": date, "peak_hour_pred": peak_hour, "peak_magnitude_pred": peak_mag})

    return pd.DataFrame(daily_records)


class PeakClassifier:
    """Binary LightGBM classifier for predicting whether daily peak will exceed the 90th percentile threshold."""

    def __init__(self, params: Optional[Dict[str, Any]] = None, decision_threshold: float = 0.5):
        self.params = {**DEFAULT_CLASSIFIER_PARAMS}
        if params:
            self.params.update(params)
        self.decision_threshold = decision_threshold
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names_: List[str] = []
        self.peak_threshold_mw_: Optional[float] = None

    def fit(
        self,
        X_daily_train: pd.DataFrame,
        y_daily_peak_train: Union[pd.Series, np.ndarray],
        peak_threshold_mw: float,
    ) -> "PeakClassifier":
        """Fits binary classifier where target = (daily_peak > peak_threshold_mw)."""
        self.peak_threshold_mw_ = peak_threshold_mw
        self.feature_names_ = list(X_daily_train.columns)

        y_binary = (np.asarray(y_daily_peak_train) > peak_threshold_mw).astype(int)

        self.model = lgb.LGBMClassifier(objective="binary", **self.params)
        self.model.fit(X_daily_train, y_binary)
        return self

    def predict_proba(self, X_daily: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PeakClassifier is not fitted.")
        return self.model.predict_proba(X_daily[self.feature_names_])[:, 1]

    def predict(self, X_daily: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X_daily)
        return (probs >= self.decision_threshold).astype(bool)


def evaluate_peak_metrics(df_peak_flags: pd.DataFrame) -> Dict[str, float]:
    """Computes peak detection performance metrics:

    Precision, Recall, F1, Mean Timing Error (minutes), Mean Magnitude Error (MW).
    """
    metrics: Dict[str, float] = {}

    if "is_peak_day_actual" in df_peak_flags.columns and "is_peak_day_pred" in df_peak_flags.columns:
        y_true = df_peak_flags["is_peak_day_actual"].astype(int)
        y_pred = df_peak_flags["is_peak_day_pred"].astype(int)

        metrics["peak_precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["peak_recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["peak_f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    if "peak_hour_actual" in df_peak_flags.columns and "peak_hour_pred" in df_peak_flags.columns:
        valid = df_peak_flags["peak_hour_actual"].notna() & df_peak_flags["peak_hour_pred"].notna()
        if valid.sum() > 0:
            # Timing error in minutes
            timing_diff = (df_peak_flags.loc[valid, "peak_hour_pred"] - df_peak_flags.loc[valid, "peak_hour_actual"]).abs() * 60.0
            metrics["peak_timing_mae_minutes"] = float(timing_diff.mean())

    if "peak_magnitude_actual" in df_peak_flags.columns and "peak_magnitude_pred" in df_peak_flags.columns:
        valid = df_peak_flags["peak_magnitude_actual"].notna() & df_peak_flags["peak_magnitude_pred"].notna()
        if valid.sum() > 0:
            mag_diff = (df_peak_flags.loc[valid, "peak_magnitude_pred"] - df_peak_flags.loc[valid, "peak_magnitude_actual"]).abs()
            metrics["peak_magnitude_mae_mw"] = float(mag_diff.mean())

    return metrics


def generate_peak_flags_table(
    df_hourly_test: pd.DataFrame,
    threshold_mw: float,
    fold: int = 0,
    horizon: str = "day",
    classifier: Optional[PeakClassifier] = None,
    X_daily_test: Optional[pd.DataFrame] = None,
    ts_col: str = "ts",
    p50_col: str = "p50",
    actual_col: Optional[str] = "actual",
) -> pd.DataFrame:
    """Builds the canonical peak_flags table for results/peak_flags.parquet.

    Contract: date, fold, peak_prob, peak_pred (bool), peak_hour_pred, peak_actual (bool), peak_hour_actual, threshold_mw
    """
    df = df_hourly_test.copy()
    if ts_col in df.columns:
        ts_series = pd.to_datetime(df[ts_col])
    else:
        ts_series = pd.to_datetime(df.index)

    df["date"] = ts_series.dt.date
    df["hour"] = ts_series.dt.hour

    daily_rows = []
    has_actual = actual_col in df.columns and df[actual_col].notna().any()

    for date, group in df.groupby("date"):
        # Prediction from P50 curve
        max_p50_idx = group[p50_col].idxmax()
        p_hour_pred = int(group.loc[max_p50_idx, "hour"])
        p_mag_pred = float(group.loc[max_p50_idx, p50_col])

        # Default classification: argmax/max P50 exceeds threshold
        is_peak_pred = bool(p_mag_pred > threshold_mw)
        peak_prob = 1.0 if is_peak_pred else 0.0

        p_hour_act = None
        is_peak_act = False
        p_mag_act = None

        if has_actual:
            max_act_idx = group[actual_col].idxmax()
            p_hour_act = int(group.loc[max_act_idx, "hour"])
            p_mag_act = float(group.loc[max_act_idx, actual_col])
            is_peak_act = bool(p_mag_act > threshold_mw)

        row: Dict[str, Any] = {
            "date": pd.to_datetime(date),
            "fold": fold,
            "peak_prob": float(peak_prob),
            "peak_pred": is_peak_pred,
            "peak_hour_pred": p_hour_pred,
            "peak_actual": is_peak_act,
            "peak_hour_actual": p_hour_act,
            "threshold_mw": float(threshold_mw),
            # Supplementary metrics for error drilldowns
            "peak_magnitude_pred": round(p_mag_pred, 2),
            "peak_magnitude_actual": round(p_mag_act, 2) if p_mag_act is not None else np.nan,
        }
        daily_rows.append(row)

    df_out = pd.DataFrame(daily_rows)

    # If classifier provided, use its probabilities and predictions
    if classifier is not None and X_daily_test is not None and len(X_daily_test) == len(df_out):
        probs = classifier.predict_proba(X_daily_test)
        preds = classifier.predict(X_daily_test)
        df_out["peak_prob"] = probs
        df_out["peak_pred"] = preds

    return df_out


def save_peak_flags(df_peak_flags: pd.DataFrame, output_path: str = "results/peak_flags.parquet") -> None:
    """Save peak flags table to parquet."""
    df_peak_flags.to_parquet(output_path, index=False)
    logger.info(f"Saved peak flags table ({len(df_peak_flags)} rows) to {output_path}")
