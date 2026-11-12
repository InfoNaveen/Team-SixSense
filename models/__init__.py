"""
models package
==============
Probabilistic Forecasting, Peak Event Classification, Interval Calibration, and Benchmarks.
"""

from models.quantile_lgbm import (
    QuantileLGBMForecaster,
    fit_predict,
    fit_predict_day,
    fit_predict_week,
    sort_quantiles,
)
from models.calibration import (
    ConformalCalibrator,
    compute_picp,
    compute_pinball_loss,
    calibrate_forecasts,
)
from models.peak_classifier import (
    PeakClassifier,
    compute_training_peak_threshold,
    predict_peak_hour_from_p50,
    generate_peak_flags_table,
    evaluate_peak_metrics,
    save_peak_flags,
)
from models.prophet_benchmark import fit_predict as fit_predict_prophet
from models.lstm_benchmark import fit_predict as fit_predict_lstm

__all__ = [
    "QuantileLGBMForecaster",
    "fit_predict",
    "fit_predict_day",
    "fit_predict_week",
    "sort_quantiles",
    "ConformalCalibrator",
    "compute_picp",
    "compute_pinball_loss",
    "calibrate_forecasts",
    "PeakClassifier",
    "compute_training_peak_threshold",
    "predict_peak_hour_from_p50",
    "generate_peak_flags_table",
    "evaluate_peak_metrics",
    "save_peak_flags",
    "fit_predict_prophet",
    "fit_predict_lstm",
]
