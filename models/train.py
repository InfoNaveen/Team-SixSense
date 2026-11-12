"""
models/train.py
===============
Entry point exposing fit_predict for backtesting contract.
Delegates to Quantile LightGBM models.
"""

from models.quantile_lgbm import fit_predict, fit_predict_day, fit_predict_week

__all__ = ["fit_predict", "fit_predict_day", "fit_predict_week"]
