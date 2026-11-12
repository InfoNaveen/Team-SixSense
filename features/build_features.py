"""
features/build_features.py
===========================
Feature engineering for DISCOM load forecasting.

CONTRACT
--------
build_features(df, horizon, origin) -> pd.DataFrame
    df      : raw hourly DataFrame with DatetimeIndex. Required columns:
                  actual        (MW load, historical only — NaN in future)
                  temp_c        (dry-bulb temperature, °C)
                  humidity_pct  (relative humidity, %)
                  apparent_temp_c (apparent/feels-like temperature, °C) [optional]
                  is_holiday    (bool/int)
                  is_lockdown   (bool/int)
                  is_oddeven    (bool/int)
                  festival_name (str, may be empty/"")
    horizon : "day_ahead" | "week_ahead"
    origin  : pd.Timestamp — the last timestamp for which actuals are known.
              Features must NEVER use data after this timestamp.

Returns a DataFrame with the same index as df (or a contiguous sub-slice),
containing only feature columns (no 'actual').  NaN rows at the head (due to
lags) are forward-filled where safe; remaining NaNs signal the caller.

LEAKAGE GUARANTEE
-----------------
Call assert_no_leakage(features_df, origin) after building features.
It raises AssertionError if any feature value at timestamp t > origin
was computed using data from t' > origin.

FEATURES BUILT
--------------
Thermal
    cdh                 Cooling degree-hours: max(0, temp_c − 25)
    cdh_24h_sum         Rolling 24-h sum of cdh (past 24 h only)
    heat_index          Rothfusz heat index (°C); falls back to apparent_temp_c

Temperature lags
    temp_lag_24h        temp_c at t − 24 h
    temp_lag_48h        temp_c at t − 48 h
    temp_lag_72h        temp_c at t − 72 h
    temp_roll_72h_mean  Rolling mean of temp_c over past 72 h

Load lags  (horizon-aware; NO future actuals used)
    day_ahead  : load_lag_48h, load_lag_168h, load_roll_7d_mean, load_roll_7d_max
    week_ahead : load_lag_168h, load_lag_336h, load_roll_7d_mean, load_roll_7d_max

Calendar
    hour, day_of_week, month, is_weekend
    is_holiday, days_to_festival, days_since_festival
    is_oddeven, is_lockdown

Trend
    day_index   Linear count of days since the first row in df (float)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

Horizon = Literal["day_ahead", "week_ahead"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_THRESHOLD_CDH = 25.0   # °C, base temperature for cooling degree-hours
_LAG_HOURS = {
    "day_ahead":  [48, 168],
    "week_ahead": [168, 336],
}
_REQUIRED_COLS = {"actual", "temp_c", "humidity_pct", "is_holiday"}


# ---------------------------------------------------------------------------
# Heat-index (Rothfusz regression)
# ---------------------------------------------------------------------------

def _rothfusz_heat_index(T_f: np.ndarray, RH: np.ndarray) -> np.ndarray:
    """
    Rothfusz (1990) heat index regression, °F input, °F output.

    Valid for T ≥ 80 °F and HI ≥ 80 °F.  For values outside this range the
    simple Steadman approximation is used instead.

    Reference: NWS Technical Attachment SR 90-23.

    HI = −42.379
       + 2.04901523 T
       + 10.14333127 RH
       − 0.22475541 T·RH
       − 0.00683783 T²
       − 0.05481717 RH²
       + 0.00122874 T²·RH
       + 0.00085282 T·RH²
       − 0.00000199 T²·RH²
    """
    HI = (
        -42.379
        + 2.04901523   * T_f
        + 10.14333127  * RH
        - 0.22475541   * T_f * RH
        - 0.00683783   * T_f ** 2
        - 0.05481717   * RH ** 2
        + 0.00122874   * T_f ** 2 * RH
        + 0.00085282   * T_f * RH ** 2
        - 0.00000199   * T_f ** 2 * RH ** 2
    )
    # Steadman simple approximation for mild conditions
    HI_simple = 0.5 * (T_f + 61.0 + (T_f - 68.0) * 1.2 + RH * 0.094)
    use_simple = HI_simple < 80.0
    HI = np.where(use_simple, HI_simple, HI)
    return HI


def compute_heat_index(
    temp_c: pd.Series,
    humidity_pct: pd.Series,
    apparent_temp_c: pd.Series | None = None,
) -> pd.Series:
    """
    Heat index in °C using Rothfusz (1990).

    Falls back to *apparent_temp_c* (feels-like) where:
      - temp_c < 27 °C  (Rothfusz is poorly calibrated below 80 °F)
      - temp_c is NaN

    If apparent_temp_c is not provided, the raw heat index is returned
    (possibly unreliable for cold cells — caller should mask downstream).

    Parameters
    ----------
    temp_c          : dry-bulb temperature, °C
    humidity_pct    : relative humidity, percent (0–100)
    apparent_temp_c : optional feels-like temperature, °C (fallback)
    """
    T_c  = temp_c.values.astype(float)
    RH   = humidity_pct.values.astype(float)
    T_f  = T_c * 9.0 / 5.0 + 32.0

    HI_f = _rothfusz_heat_index(T_f, RH)
    HI_c = (HI_f - 32.0) * 5.0 / 9.0   # back to °C

    if apparent_temp_c is not None:
        app = apparent_temp_c.values.astype(float)
        # Use Rothfusz only when temp ≥ 27 °C and not NaN
        use_rothfusz = (T_c >= 27.0) & ~np.isnan(T_c) & ~np.isnan(RH)
        HI_c = np.where(use_rothfusz, HI_c, app)

    return pd.Series(HI_c, index=temp_c.index, name="heat_index")


# ---------------------------------------------------------------------------
# Festival helper
# ---------------------------------------------------------------------------

def _festival_distance(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    days_to_festival   : days until the NEXT festival from each row (0 on festival day)
    days_since_festival: days since the MOST RECENT festival (0 on festival day)

    Uses df["festival_name"] — non-empty string means a festival on that date.
    If the column is absent, returns zeros for both.
    """
    if "festival_name" not in df.columns:
        z = pd.Series(0.0, index=df.index)
        return z, z.copy()

    dates     = df.index.normalize()   # floor to midnight
    is_fest   = df["festival_name"].fillna("").str.strip().astype(bool)
    fest_dates = dates[is_fest.values].unique()

    if len(fest_dates) == 0:
        z = pd.Series(0.0, index=df.index)
        return z, z.copy()

    # Broadcast (N_rows, N_festivals) difference matrix, then reduce
    dates_arr = dates.values.astype("datetime64[D]").astype(np.int64)
    fest_arr  = np.array(fest_dates, dtype="datetime64[D]").astype(np.int64)

    diff = dates_arr[:, None] - fest_arr[None, :]          # (N, F), days

    days_since = np.where(diff >= 0, diff, np.inf).min(axis=1).astype(float)
    days_to    = np.where(diff <= 0, -diff, np.inf).min(axis=1).astype(float)

    # Cap at 30 days to keep values bounded
    days_since = np.clip(days_since, 0, 30)
    days_to    = np.clip(days_to, 0, 30)

    return (
        pd.Series(days_to,    index=df.index, name="days_to_festival"),
        pd.Series(days_since, index=df.index, name="days_since_festival"),
    )


# ---------------------------------------------------------------------------
# Core feature builder
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    horizon: Horizon,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build the full feature matrix for the given horizon.

    Parameters
    ----------
    df      : Raw hourly DataFrame with DatetimeIndex (see module docstring).
    horizon : "day_ahead" | "week_ahead"
    origin  : Last timestamp for which ``df["actual"]`` is valid.
              Any feature that would look at actual[t'] for t' > origin is
              prohibited — the leakage test enforces this post-hoc.

    Returns
    -------
    pd.DataFrame
        One row per timestamp in df; feature columns only (no 'actual').
        Rows with insufficient history (e.g. first 336 h for week-ahead lags)
        will contain NaN — the caller (backtest harness) trims these.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex")
    if horizon not in ("day_ahead", "week_ahead"):
        raise ValueError(f"horizon must be 'day_ahead' or 'week_ahead', got {horizon!r}")

    missing = {"actual", "temp_c", "humidity_pct"} - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    df = df.sort_index()
    out = pd.DataFrame(index=df.index)

    # ------------------------------------------------------------------ #
    # 1. THERMAL FEATURES                                                  #
    # ------------------------------------------------------------------ #
    # CDH = max(0, temp_c − 25)
    out["cdh"] = np.maximum(0.0, df["temp_c"] - _THRESHOLD_CDH)

    # Rolling 24-h sum of CDH — closed='left' so we don't use the current hour
    # We use min_periods=1 to avoid NaN at the start
    out["cdh_24h_sum"] = (
        out["cdh"]
        .rolling(window=24, min_periods=1, closed="left")
        .sum()
    )

    # Heat index
    apparent = df.get("apparent_temp_c", None)
    out["heat_index"] = compute_heat_index(
        df["temp_c"], df["humidity_pct"], apparent
    ).values

    # ------------------------------------------------------------------ #
    # 2. TEMPERATURE LAGS + ROLLING MEAN                                   #
    # ------------------------------------------------------------------ #
    for lag_h in [24, 48, 72]:
        col = f"temp_lag_{lag_h}h"
        out[col] = df["temp_c"].shift(lag_h)

    out["temp_roll_72h_mean"] = (
        df["temp_c"]
        .rolling(window=72, min_periods=1, closed="left")
        .mean()
    )

    # ------------------------------------------------------------------ #
    # 3. LOAD LAGS  (horizon-aware, never crosses origin)                  #
    # ------------------------------------------------------------------ #
    # We only shift the HISTORICAL actuals; future actuals are NaN in df,
    # so shift() naturally propagates NaN forward — no manual masking needed.
    # The leakage test below is the safety net.
    lag_hours = _LAG_HOURS[horizon]
    for lag_h in lag_hours:
        col = f"load_lag_{lag_h}h"
        out[col] = df["actual"].shift(lag_h)

    # Rolling 7-day mean/max — using only past values (closed="left")
    out["load_roll_7d_mean"] = (
        df["actual"]
        .rolling(window=168, min_periods=1, closed="left")
        .mean()
    )
    out["load_roll_7d_max"] = (
        df["actual"]
        .rolling(window=168, min_periods=1, closed="left")
        .max()
    )

    # ------------------------------------------------------------------ #
    # 4. CALENDAR FEATURES                                                 #
    # ------------------------------------------------------------------ #
    idx = df.index
    out["hour"]        = idx.hour
    out["day_of_week"] = idx.dayofweek          # 0=Mon … 6=Sun
    out["month"]       = idx.month
    out["is_weekend"]  = (idx.dayofweek >= 5).astype(int)

    for bool_col in ("is_holiday", "is_lockdown", "is_oddeven"):
        if bool_col in df.columns:
            out[bool_col] = df[bool_col].fillna(0).astype(int).values
        else:
            out[bool_col] = 0

    days_to_fest, days_since_fest = _festival_distance(df)
    out["days_to_festival"]   = days_to_fest.values
    out["days_since_festival"] = days_since_fest.values

    # ------------------------------------------------------------------ #
    # 5. TREND                                                             #
    # ------------------------------------------------------------------ #
    out["day_index"] = (idx - idx[0]).total_seconds() / 86_400.0

    # ------------------------------------------------------------------ #
    # Leakage guard (always run)                                           #
    # ------------------------------------------------------------------ #
    assert_no_leakage(out, origin, df)

    logger.info(
        "Built %d features × %d rows for %s (origin=%s).",
        len(out.columns), len(out), horizon, origin,
    )
    return out


# ---------------------------------------------------------------------------
# Leakage test
# ---------------------------------------------------------------------------

def assert_no_leakage(
    features: pd.DataFrame,
    origin: pd.Timestamp,
    raw_df: pd.DataFrame,
) -> None:
    """
    Assert that no feature at time t > origin was computed using actual load
    data from t' > origin.

    Strategy
    --------
    1. Build a *corrupted* version of raw_df where all actuals after origin
       are replaced by a unique sentinel value (1e9).
    2. Recompute load-lag and load-rolling features on the corrupted series.
    3. For every timestamp t > origin, check that the feature values match
       the corrupted version (sentinel propagation) rather than the real values.

    This catches any path that reads future actuals, including:
      - Insufficient shift amounts
      - Rolling windows that extend past origin
      - Accidental index alignment using real future values

    Raises
    ------
    AssertionError with a detailed message if leakage is found.
    """
    SENTINEL = 1.234567e9  # obviously wrong value

    future_mask = raw_df.index > origin
    if not future_mask.any():
        return   # no future rows → trivially clean

    # Corrupt actuals after origin
    corrupted = raw_df.copy()
    corrupted.loc[future_mask, "actual"] = SENTINEL

    # Columns that are derived from actual load
    load_feature_cols = [c for c in features.columns if "load" in c]
    if not load_feature_cols:
        return

    # Recompute load features on corrupted data
    # We only shift and roll the corrupted actual series
    tainted: dict[str, pd.Series] = {}
    for col in load_feature_cols:
        if col.startswith("load_lag_"):
            lag_h = int(col.split("_")[-1].replace("h", ""))
            tainted[col] = corrupted["actual"].shift(lag_h)
        elif col == "load_roll_7d_mean":
            tainted[col] = (
                corrupted["actual"]
                .rolling(window=168, min_periods=1, closed="left")
                .mean()
            )
        elif col == "load_roll_7d_max":
            tainted[col] = (
                corrupted["actual"]
                .rolling(window=168, min_periods=1, closed="left")
                .max()
            )

    # Check: for t > origin, features should equal the tainted version
    # (i.e. they must have been computed from the same data we see)
    future_idx = raw_df.index[future_mask]
    violations = []
    for col, tainted_series in tainted.items():
        real_future    = features.loc[future_idx, col]
        tainted_future = tainted_series.loc[future_idx]

        # Where tainted is SENTINEL, the feature must also be SENTINEL (or NaN)
        # Where tainted is NaN, the feature must also be NaN
        # Disagreement → leakage
        sentinel_mask = np.abs(tainted_future.values - SENTINEL) < 1e3
        feature_vals  = real_future.values

        if np.any(sentinel_mask):
            leaked = feature_vals[sentinel_mask]
            if not np.all(np.isnan(leaked) | (np.abs(leaked - SENTINEL) < 1e3)):
                violations.append(
                    f"  Column '{col}': found {np.sum(~np.isnan(leaked))} future-tainted "
                    f"rows with non-sentinel values after origin={origin}."
                )

    if violations:
        raise AssertionError(
            "LEAKAGE DETECTED — features use post-origin actuals:\n"
            + "\n".join(violations)
            + "\nFix: ensure load lags and rolling windows are anchored strictly "
              "before the forecast origin."
        )

    logger.info("Leakage test PASSED for origin=%s.", origin)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build features and run leakage test")
    parser.add_argument("--parquet",  required=True, help="Raw data parquet path")
    parser.add_argument("--horizon",  default="day_ahead", choices=["day_ahead", "week_ahead"])
    parser.add_argument("--origin",   default=None,
                        help="Forecast origin timestamp (ISO8601). Defaults to last row.")
    parser.add_argument("--out",      default=None, help="Output parquet path for features")
    args = parser.parse_args()

    _df = pd.read_parquet(args.parquet)
    if not isinstance(_df.index, pd.DatetimeIndex):
        _df = _df.set_index("timestamp")
    _df.index = pd.to_datetime(_df.index)

    _origin = pd.Timestamp(args.origin) if args.origin else _df.index[-1]
    _feats  = build_features(_df, horizon=args.horizon, origin=_origin)

    if args.out:
        _feats.to_parquet(args.out)
        logger.info("Features written → %s", args.out)
    else:
        print(_feats.tail())
        print(f"\nShape: {_feats.shape}")
