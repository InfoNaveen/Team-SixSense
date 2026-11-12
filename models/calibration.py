"""
models/calibration.py
=====================
Conformal Prediction Interval Calibration for Probabilistic Load Forecasting.

Ensures the P10-P90 interval achieves honest ~80% coverage (PICP ~ 0.80)
via distribution-free Conformal Quantile Regression (CQR) shift.

Exposes:
    compute_picp(y_true, p10, p90) -> float
    compute_pinball_loss(y_true, p10, p50, p90) -> dict
    ConformalCalibrator (class)
    calibrate_forecasts(df_forecasts, y_val, p10_val, p90_val, target_coverage=0.80) -> pd.DataFrame
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_picp(
    y_true: Union[pd.Series, np.ndarray],
    p10: Union[pd.Series, np.ndarray],
    p90: Union[pd.Series, np.ndarray],
) -> float:
    """Prediction Interval Coverage Probability (PICP).

    Share of actual values falling inside [p10, p90].
    Target is 0.80 (80%).
    """
    y = np.asarray(y_true).ravel()
    low = np.asarray(p10).ravel()
    high = np.asarray(p90).ravel()

    valid_mask = ~(np.isnan(y) | np.isnan(low) | np.isnan(high))
    if valid_mask.sum() == 0:
        return 0.0

    inside = (y[valid_mask] >= low[valid_mask]) & (y[valid_mask] <= high[valid_mask])
    return float(np.mean(inside))


def compute_pinball_loss(
    y_true: Union[pd.Series, np.ndarray],
    p10: Union[pd.Series, np.ndarray],
    p50: Union[pd.Series, np.ndarray],
    p90: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """Pinball loss for quantiles 0.1, 0.5, 0.9 and their mean.

    pinball(q) = mean(max(q * (y - yhat), (q - 1) * (y - yhat)))
    """
    y = np.asarray(y_true).ravel()
    preds = {0.1: np.asarray(p10).ravel(), 0.5: np.asarray(p50).ravel(), 0.9: np.asarray(p90).ravel()}

    losses: Dict[str, float] = {}
    valid = ~np.isnan(y)
    for q, yhat in preds.items():
        v = valid & ~np.isnan(yhat)
        diff = y[v] - yhat[v]
        loss = np.maximum(q * diff, (q - 1.0) * diff)
        losses[f"pinball_p{int(q*100)}"] = float(np.mean(loss))

    losses["pinball_mean"] = float(np.mean([losses["pinball_p10"], losses["pinball_p50"], losses["pinball_p90"]]))
    return losses


class ConformalCalibrator:
    """Finite-sample distribution-free Conformal Quantile Calibrator.

    Computes conformal correction offsets on a validation window so that
    coverage on unseen test data matches target_coverage (default 80%).
    """

    def __init__(self, target_coverage: float = 0.80, method: str = "asymmetric"):
        """Parameters

        ----------
        target_coverage : float
            Nominal coverage, 0.80 for [p10, p90].
        method : str
            'asymmetric' (independent lower/upper tail shifts) or 'symmetric' (single offset).
        """
        self.target_coverage = target_coverage
        self.alpha = 1.0 - target_coverage  # 0.20 for 80%
        self.method = method
        self.c_low: float = 0.0
        self.c_high: float = 0.0
        self.c_sym: float = 0.0
        self.val_picp_before: float = 0.0
        self.val_picp_after: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self,
        y_val: Union[pd.Series, np.ndarray],
        p10_val: Union[pd.Series, np.ndarray],
        p90_val: Union[pd.Series, np.ndarray],
    ) -> "ConformalCalibrator":
        """Fit conformal shift parameters using validation ground truth and raw predictions."""
        y = np.asarray(y_val).ravel()
        p10 = np.asarray(p10_val).ravel()
        p90 = np.asarray(p90_val).ravel()

        mask = ~(np.isnan(y) | np.isnan(p10) | np.isnan(p90))
        y, p10, p90 = y[mask], p10[mask], p90[mask]
        n = len(y)
        if n == 0:
            raise ValueError("No valid calibration samples provided.")

        self.val_picp_before = compute_picp(y, p10, p90)

        if self.method == "asymmetric":
            # Lower tail error: want P(y < p10 - c_low) <= alpha/2 (0.10)
            # e_low = p10 - y. We want p10 - c_low <= y => e_low <= c_low.
            # Quantile level: (1 - alpha/2) * (1 + 1/n)
            q_level = min(1.0, (1.0 - self.alpha / 2.0) * (n + 1.0) / n)
            e_low = p10 - y
            e_high = y - p90

            self.c_low = float(np.quantile(e_low, q_level, method="higher" if hasattr(np, "quantile") else "linear"))
            self.c_high = float(np.quantile(e_high, q_level, method="higher" if hasattr(np, "quantile") else "linear"))

            cal_p10 = p10 - self.c_low
            cal_p90 = p90 + self.c_high
        else:
            # Symmetric CQR score: S_i = max(p10_i - y_i, y_i - p90_i)
            # S_i > 0 when y_i is outside interval
            scores = np.maximum(p10 - y, y - p90)
            q_level = min(1.0, (1.0 - self.alpha) * (n + 1.0) / n)
            self.c_sym = float(np.quantile(scores, q_level, method="higher" if hasattr(np, "quantile") else "linear"))

            cal_p10 = p10 - self.c_sym
            cal_p90 = p90 + self.c_sym

        self.val_picp_after = compute_picp(y, cal_p10, cal_p90)
        self.is_fitted = True

        logger.info(
            f"Conformal Calibration fitted (n={n}): "
            f"PICP before={self.val_picp_before*100:.1f}%, after={self.val_picp_after*100:.1f}% "
            f"[c_low={self.c_low:.2f}, c_high={self.c_high:.2f}, c_sym={self.c_sym:.2f}]"
        )
        return self

    def transform(
        self,
        p10: Union[pd.Series, np.ndarray],
        p50: Union[pd.Series, np.ndarray],
        p90: Union[pd.Series, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply conformal shift to test predictions and guarantee non-crossing."""
        if not self.is_fitted:
            raise RuntimeError("ConformalCalibrator is not fitted.")

        p10_arr = np.asarray(p10, dtype=float).copy()
        p50_arr = np.asarray(p50, dtype=float).copy()
        p90_arr = np.asarray(p90, dtype=float).copy()

        if self.method == "asymmetric":
            cal_p10 = p10_arr - self.c_low
            cal_p90 = p90_arr + self.c_high
        else:
            cal_p10 = p10_arr - self.c_sym
            cal_p90 = p90_arr + self.c_sym

        # Enforce non-crossing: p10 <= p50 <= p90
        # If shift caused inversion, sort row-wise
        stacked = np.column_stack([cal_p10, p50_arr, cal_p90])
        sorted_q = np.sort(stacked, axis=1)
        return sorted_q[:, 0], sorted_q[:, 1], sorted_q[:, 2]

    def calibrate_dataframe(
        self,
        df_forecast: pd.DataFrame,
        p10_col: str = "p10",
        p50_col: str = "p50",
        p90_col: str = "p90",
    ) -> pd.DataFrame:
        """Calibrate a forecast DataFrame and return new DataFrame with calibrated intervals."""
        df_out = df_forecast.copy()
        p10_c, p50_c, p90_c = self.transform(
            df_out[p10_col].values,
            df_out[p50_col].values,
            df_out[p90_col].values,
        )
        df_out[p10_col] = p10_c
        df_out[p50_col] = p50_c
        df_out[p90_col] = p90_c
        return df_out


def calibrate_forecasts(
    df_forecast: pd.DataFrame,
    y_val: Union[pd.Series, np.ndarray],
    p10_val: Union[pd.Series, np.ndarray],
    p90_val: Union[pd.Series, np.ndarray],
    target_coverage: float = 0.80,
    method: str = "asymmetric",
) -> Tuple[pd.DataFrame, ConformalCalibrator]:
    """One-shot calibration helper:

    Fits calibrator on validation predictions and applies to df_forecast.

    Returns
    -------
    df_calibrated : pd.DataFrame
    calibrator : ConformalCalibrator
    """
    calibrator = ConformalCalibrator(target_coverage=target_coverage, method=method)
    calibrator.fit(y_val, p10_val, p90_val)
    df_cal = calibrator.calibrate_dataframe(df_forecast)
    return df_cal, calibrator

