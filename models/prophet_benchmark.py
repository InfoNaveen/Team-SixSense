"""
models/prophet_benchmark.py
===========================
Benchmark Model: Additive Seasonal Time-Series Model with Indian Festivals ("prophet").

Exposes:
    fit_predict(X_train, y_train, X_test, horizon="day") -> pd.DataFrame[p10, p50, p90]

Supports both:
    1. Direct facebook/prophet Prophet library (with interval_width=0.80).
    2. High-speed native Fourier + Festival Ridge/Quantile additive fallback if
       cmdstanpy C++ binaries are unavailable on Windows.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from models.quantile_lgbm import sort_quantiles

logger = logging.getLogger(__name__)


def _fit_predict_native_prophet(
    ts_train: pd.Series,
    y_train: pd.Series,
    ts_test: pd.Series,
    festival_train: Optional[pd.Series] = None,
    festival_test: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Use the official Prophet package."""
    from prophet import Prophet  # type: ignore

    df_train = pd.DataFrame({"ds": pd.to_datetime(ts_train), "y": np.asarray(y_train, dtype=float)})

    # Initialize Prophet with 80% uncertainty interval (corresponds to P10-P90)
    m = Prophet(
        interval_width=0.80,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
    )
    m.add_country_holidays(country_name="IN")

    if festival_train is not None:
        df_train["festival"] = festival_train.values
        m.add_regressor("festival")

    m.fit(df_train)

    df_future = pd.DataFrame({"ds": pd.to_datetime(ts_test)})
    if festival_test is not None:
        df_future["festival"] = festival_test.values

    fcst = m.predict(df_future)
    p10 = fcst["yhat_lower"].values
    p50 = fcst["yhat"].values
    p90 = fcst["yhat_upper"].values

    p10, p50, p90 = sort_quantiles(p10, p50, p90)
    return pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}, index=ts_test.index if hasattr(ts_test, "index") else None)


def _fit_predict_fourier_prophet(
    ts_train: pd.Series,
    y_train: pd.Series,
    ts_test: pd.Series,
    festival_train: Optional[pd.Series] = None,
    festival_test: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Robust native additive seasonal model matching Prophet's formulation:

    y(t) = trend + sum(Fourier_diurnal) + sum(Fourier_weekly) + festival + error.
    Used when C++ Stan runtime is unavailable.
    """
    from sklearn.linear_model import Ridge

    def _make_fourier(dates: pd.Series) -> pd.DataFrame:
        dt = pd.to_datetime(dates)
        h = dt.dt.hour.values
        d = dt.dt.dayofweek.values
        y_d = dt.dt.dayofyear.values

        feats: Dict[str, np.ndarray] = {
            "trend": np.arange(len(dates), dtype=float),
            "sin_hour": np.sin(2 * np.pi * h / 24),
            "cos_hour": np.cos(2 * np.pi * h / 24),
            "sin_hour2": np.sin(4 * np.pi * h / 24),
            "cos_hour2": np.cos(4 * np.pi * h / 24),
            "sin_dow": np.sin(2 * np.pi * d / 7),
            "cos_dow": np.cos(2 * np.pi * d / 7),
            "sin_year": np.sin(2 * np.pi * y_d / 365.25),
            "cos_year": np.cos(2 * np.pi * y_d / 365.25),
            "is_weekend": (d >= 5).astype(float),
        }
        return pd.DataFrame(feats)

    X_tr = _make_fourier(ts_train)
    X_te = _make_fourier(ts_test)

    if festival_train is not None and festival_test is not None:
        X_tr["festival"] = pd.to_numeric(festival_train, errors="coerce").fillna(0).values
        X_te["festival"] = pd.to_numeric(festival_test, errors="coerce").fillna(0).values

    model = Ridge(alpha=1.0)
    y_tr_arr = np.asarray(y_train, dtype=float)
    model.fit(X_tr, y_tr_arr)

    p50 = model.predict(X_te)
    residuals = y_tr_arr - model.predict(X_tr)
    # 80% interval corresponds to 1.282 standard deviations under normality
    std_res = np.std(residuals)
    z = 1.28155

    p10 = p50 - z * std_res
    p90 = p50 + z * std_res

    p10, p50, p90 = sort_quantiles(p10, p50, p90)
    return pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}, index=ts_test.index if hasattr(ts_test, "index") else None)


def fit_predict(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_test: pd.DataFrame,
    horizon: str = "day",
) -> pd.DataFrame:
    """Prophet benchmark contract function.

    Returns DataFrame with columns ['p10', 'p50', 'p90'] aligned with X_test index.
    """
    ts_col = None
    for col in ["ts", "timestamp", "datetime"]:
        if col in X_train.columns:
            ts_col = col
            break

    if ts_col:
        ts_train = X_train[ts_col]
        ts_test = X_test[ts_col]
    elif hasattr(X_train, "index") and isinstance(X_train.index, pd.DatetimeIndex):
        ts_train = pd.Series(X_train.index)
        ts_test = pd.Series(X_test.index)
    else:
        # Generate dummy hourly sequence if no timestamps provided
        ts_train = pd.date_range("2024-01-01", periods=len(X_train), freq="h")
        ts_test = pd.date_range(ts_train[-1] + pd.Timedelta(hours=1), periods=len(X_test), freq="h")

    fest_train = X_train["is_holiday"] if "is_holiday" in X_train.columns else None
    fest_test = X_test["is_holiday"] if "is_holiday" in X_test.columns else None

    # Try official Prophet package, fall back to native Fourier additive model
    try:
        import prophet  # type: ignore
        return _fit_predict_native_prophet(ts_train, y_train, ts_test, fest_train, fest_test)
    except Exception as e:
        logger.info(f"Official prophet package unavailable ({e}), using native Fourier-Prophet benchmark.")
        return _fit_predict_fourier_prophet(ts_train, y_train, ts_test, fest_train, fest_test)

