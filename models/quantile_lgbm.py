"""
models/quantile_lgbm.py
=======================
Probabilistic Load Forecasting using LightGBM Quantile Regression.

Exposes:
    fit_predict(X_train, y_train, X_test, horizon="day") -> pd.DataFrame[p10, p50, p90]
    QuantileLGBMForecaster (class)

Ensures monotonic non-crossing quantiles: p10 <= p50 <= p90 per row.
Tailored for fast CPU execution on standard laptops.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)

# Default hyperparameter configurations for laptop execution (<30s per fold)
DEFAULT_PARAMS_DAY: Dict[str, Any] = {
    "n_estimators": 250,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}

DEFAULT_PARAMS_WEEK: Dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 45,
    "max_depth": 8,
    "min_child_samples": 30,
    "subsample": 0.80,
    "subsample_freq": 1,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}

IGNORE_COLUMNS = {"ts", "timestamp", "datetime", "date", "load_mw", "target", "actual"}


def _preprocess_features(
    X: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Clean feature DataFrame, dropping non-feature timestamp/target columns

    and casting object/string columns to 'category' for LightGBM.
    """
    X_clean = X.copy()
    drop_cols = [c for c in X_clean.columns if c.lower() in IGNORE_COLUMNS]
    if drop_cols:
        X_clean = X_clean.drop(columns=drop_cols)

    cats = list(categorical_features) if categorical_features is not None else []
    for c in X_clean.columns:
        if X_clean[c].dtype == "object" or X_clean[c].dtype.name == "category" or X_clean[c].dtype == "bool":
            X_clean[c] = X_clean[c].astype("category")
            if c not in cats:
                cats.append(c)

    return X_clean, cats


def sort_quantiles(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort three quantile predictions row-wise to eliminate quantile crossing.

    Enforces p10 <= p50 <= p90 everywhere.
    """
    stacked = np.column_stack([p10, p50, p90])
    sorted_q = np.sort(stacked, axis=1)
    return sorted_q[:, 0], sorted_q[:, 1], sorted_q[:, 2]


class QuantileLGBMForecaster:
    """Manages 3 LightGBM quantile regression models (alpha=0.1, 0.5, 0.9)."""

    def __init__(
        self,
        horizon: str = "day",
        quantiles: Tuple[float, float, float] = (0.1, 0.5, 0.9),
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ):
        self.horizon = horizon.lower()
        self.quantiles = sorted(quantiles)
        self.random_state = random_state

        base_params = DEFAULT_PARAMS_DAY if self.horizon in ("day", "day-ahead", "24h") else DEFAULT_PARAMS_WEEK
        self.params = {**base_params}
        if params:
            self.params.update(params)
        self.params["random_state"] = self.random_state

        self.models_: Dict[float, lgb.LGBMRegressor] = {}
        self.feature_names_: List[str] = []
        self.categorical_features_: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> "QuantileLGBMForecaster":
        """Fit three separate quantile models (alpha=0.1, 0.5, 0.9)."""
        X_clean, cats = _preprocess_features(X, categorical_features)
        self.feature_names_ = list(X_clean.columns)
        self.categorical_features_ = cats

        eval_clean = None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_clean, _ = _preprocess_features(X_val, self.categorical_features_)
            eval_clean = (X_val_clean, np.asarray(y_val))

        y_arr = np.asarray(y)

        for q in self.quantiles:
            logger.info(f"Fitting LightGBM quantile={q} for horizon={self.horizon}...")
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                **self.params,
            )
            fit_kwargs: Dict[str, Any] = {}
            if eval_clean is not None:
                fit_kwargs["eval_set"] = [eval_clean]
                fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=25, verbose=False)]

            model.fit(
                X_clean,
                y_arr,
                categorical_feature=self.categorical_features_ if self.categorical_features_ else "auto",
                **fit_kwargs,
            )
            self.models_[q] = model

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict p10, p50, p90 and enforce non-crossing by row-wise sorting.

        Returns DataFrame with columns ['p10', 'p50', 'p90'].
        """
        if not self.models_:
            raise RuntimeError("Forecaster has not been fitted yet.")

        X_clean, _ = _preprocess_features(X, self.categorical_features_)
        # Align columns
        for c in self.feature_names_:
            if c not in X_clean.columns:
                X_clean[c] = np.nan
        X_clean = X_clean[self.feature_names_]

        raw_preds: Dict[float, np.ndarray] = {}
        for q in self.quantiles:
            raw_preds[q] = self.models_[q].predict(X_clean)

        p10, p50, p90 = sort_quantiles(raw_preds[0.1], raw_preds[0.5], raw_preds[0.9])

        index = X.index if hasattr(X, "index") else None
        return pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}, index=index)

    def get_feature_importances(self, importance_type: str = "gain") -> pd.DataFrame:
        """Compute feature importances across all three quantiles."""
        if not self.models_:
            raise RuntimeError("Forecaster has not been fitted yet.")

        df_imp = pd.DataFrame({"feature": self.feature_names_})
        for q, model in self.models_.items():
            col_name = f"importance_p{int(q*100)}_{importance_type}"
            df_imp[col_name] = model.booster_.feature_importance(importance_type=importance_type)

        imp_cols = [c for c in df_imp.columns if c.startswith("importance_")]
        df_imp["importance_mean"] = df_imp[imp_cols].mean(axis=1)
        df_imp = df_imp.sort_values(by="importance_mean", ascending=False).reset_index(drop=True)
        return df_imp


def fit_predict(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_test: pd.DataFrame,
    horizon: str = "day",
    params: Optional[Dict[str, Any]] = None,
    categorical_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Contract function:

    Fits LightGBM quantile regression models (0.1, 0.5, 0.9) on X_train, y_train
    and predicts on X_test. Enforces non-crossing p10 <= p50 <= p90.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series or np.ndarray
        Training load target (load_mw)
    X_test : pd.DataFrame
        Test features for forecast window
    horizon : str
        'day' (24 h) or 'week' (168 h)
    params : dict, optional
        Custom hyperparameter overrides
    categorical_features : list of str, optional
        List of categorical column names

    Returns
    -------
    pd.DataFrame
        Columns: ['p10', 'p50', 'p90'] aligned with X_test index.
    """
    forecaster = QuantileLGBMForecaster(
        horizon=horizon,
        quantiles=(0.1, 0.5, 0.9),
        params=params,
    )
    forecaster.fit(X_train, y_train, categorical_features=categorical_features)
    return forecaster.predict(X_test)


def fit_predict_day(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Fixed-signature wrapper for day-ahead horizon."""
    return fit_predict(X_train, y_train, X_test, horizon="day")


def fit_predict_week(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Fixed-signature wrapper for week-ahead horizon."""
    return fit_predict(X_train, y_train, X_test, horizon="week")


