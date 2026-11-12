"""
eval/metrics.py
===============
Evaluation metrics for DISCOM load-forecast backtest results.

INPUTS
------
results/forecasts_{horizon}.parquet
    Columns: timestamp, actual, p10, p50, p90, fold_id, model
    Written by eval/backtest.run_backtest().

results/peak_flags.parquet  (optional, for peak precision/recall/F1)
    Columns: timestamp, is_actual_peak (bool), is_predicted_peak (bool)
    Written by the peak-classifier team (Modeling scope, not ours).

OUTPUTS
-------
results/metrics.csv
    One row per (model, horizon, segment) with all scalar metrics.

results/missed_peaks.csv
    Rows where actual peak exceeded P90 or the peak flag was missed,
    with a short auto-generated note ("heatwave", "festival", etc.).

PUBLIC API
----------
compute_all_metrics(horizon, results_dir, raw_df) -> pd.DataFrame
    Compute every metric and write both output files.
    Returns the metrics DataFrame.

METRIC DEFINITIONS (also in docstrings below)
---------------------------------------------
MAPE        Mean Absolute Percentage Error (%)
RMSE        Root Mean Squared Error (MW)
Pinball     Quantile (tilted) loss for q ∈ {0.1, 0.5, 0.9}
PICP        Prediction Interval Coverage Probability — fraction of actuals
            falling in [P10, P90]; target ≥ 80%.
Peak timing error   minutes between actual and predicted (P50) hourly peak
Peak magnitude error  MW difference at the actual peak hour vs. P50
Peak P/R/F1 Precision, Recall, F1 on is_predicted_peak vs. is_actual_peak
Improvement %   (baseline_metric − model_metric) / baseline_metric × 100
                Positive = better than seasonal-naïve.

SEGMENTS
--------
season      (winter/spring/summer/monsoon)   based on month
hour_band   (night 22-5, morning 6-11, afternoon 12-17, evening 18-21)
peak_type   (peak = top-25th-percentile actual load hours; off_peak = rest)
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
# Segment helpers
# ---------------------------------------------------------------------------

_SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3:  "spring", 4: "spring", 5: "spring",
    6:  "summer", 7: "summer", 8: "summer",
    9:  "monsoon",10: "monsoon",11: "monsoon",
}

def _assign_season(ts: pd.Series) -> pd.Series:
    """Map a Series of Timestamps to Indian meteorological season strings."""
    return ts.dt.month.map(_SEASON_MAP)


def _assign_hour_band(ts: pd.Series) -> pd.Series:
    """
    Map hour-of-day to named band:
        night     22–05  (inclusive)
        morning   06–11
        afternoon 12–17
        evening   18–21
    """
    h = ts.dt.hour
    bands = pd.Series("other", index=ts.index, dtype=str)
    bands[h.isin(range(22, 24)) | h.isin(range(0, 6))]  = "night"
    bands[h.isin(range(6, 12))]                          = "morning"
    bands[h.isin(range(12, 18))]                         = "afternoon"
    bands[h.isin(range(18, 22))]                         = "evening"
    return bands


def _assign_peak_type(actual: pd.Series, threshold_pct: float = 75.0) -> pd.Series:
    """
    Label each row 'peak' if the actual load is above the given percentile
    (default 75th), else 'off_peak'.

    Using an empirical threshold keeps the definition data-driven and
    consistent across all models evaluated on the same test set.
    """
    threshold = np.percentile(actual.dropna(), threshold_pct)
    return np.where(actual >= threshold, "peak", "off_peak")


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (%).

    MAPE = (1/N) Σ |y_t − ŷ_t| / |y_t| × 100

    NaN where actual == 0 (avoid division by zero); those rows are excluded
    from the mean (consistent with industry practice for load forecasting).

    Returns NaN if no valid rows.
    """
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(actual != 0, np.abs(actual - predicted) / np.abs(actual), np.nan)
    return float(np.nanmean(pct) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Root Mean Squared Error (same units as actual — MW).

    RMSE = sqrt( (1/N) Σ (y_t − ŷ_t)² )

    Penalises large errors more than MAE; appropriate for grid operations
    where peak deviations have disproportionate cost.
    """
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.nanmean((actual - predicted) ** 2)))


def pinball_loss(actual: np.ndarray, predicted: np.ndarray, q: float) -> float:
    """
    Pinball (quantile) loss for quantile level *q* ∈ (0, 1).

    L_q(y, ŷ) = (1/N) Σ  { q · (y_t − ŷ_t)      if y_t ≥ ŷ_t
                           { (1−q) · (ŷ_t − y_t)   otherwise

    Equivalent to the mean of the tilted absolute deviation.
    At q=0.5 this equals MAE/2.

    For well-calibrated intervals:
        q=0.1 → P10 should underestimate 10 % of the time
        q=0.9 → P90 should overestimate 90 % of the time
    """
    if not 0 < q < 1:
        raise ValueError(f"q must be in (0, 1), got {q}")
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err       = actual - predicted
    loss      = np.where(err >= 0, q * err, (q - 1) * err)
    return float(np.nanmean(loss))


def picp(actual: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    """
    Prediction Interval Coverage Probability.

    PICP = (1/N) Σ 𝟙[ P10_t ≤ y_t ≤ P90_t ]

    Target: PICP ≥ 0.80 for a nominal 80 % prediction interval.
    Values below 0.80 indicate the interval is too narrow (overconfident).
    Values well above 0.80 suggest the interval is too wide (wasteful for
    procurement decisions).

    Returns a value in [0, 1].
    """
    actual = np.asarray(actual, dtype=float)
    p10    = np.asarray(p10,    dtype=float)
    p90    = np.asarray(p90,    dtype=float)
    covered = (actual >= p10) & (actual <= p90)
    return float(np.nanmean(covered.astype(float)))


def peak_timing_error_minutes(actual: pd.Series, p50: pd.Series) -> float:
    """
    Peak timing error in minutes, averaged over days.

    For each calendar day:
        t_actual  = hour of maximum actual load
        t_pred    = hour of maximum P50
        error_d   = |t_actual − t_pred| × 60   (minutes)

    Returns the mean across days.  NaN days (no data) are excluded.

    A one-hour shift = 60 min error.  Target: < 60 min for day-ahead.
    """
    df = pd.DataFrame({"actual": actual.values, "p50": p50.values}, index=actual.index)
    df["date"] = df.index.normalize()

    daily_actual_peak = df.groupby("date")["actual"].idxmax()
    daily_p50_peak    = df.groupby("date")["p50"].idxmax()

    errors = []
    for date in daily_actual_peak.index:
        t_a = daily_actual_peak.get(date)
        t_p = daily_p50_peak.get(date)
        if pd.notna(t_a) and pd.notna(t_p):
            diff_h = abs(t_a.hour - t_p.hour)
            errors.append(diff_h * 60)

    return float(np.nanmean(errors)) if errors else np.nan


def peak_magnitude_error_mw(actual: pd.Series, p50: pd.Series) -> float:
    """
    Peak magnitude error in MW, averaged over days.

    For each calendar day:
        h*       = hour of maximum actual load
        mag_err  = actual[h*] − p50[h*]   (signed, positive = under-forecast)

    Returns the mean signed error.  Use abs() externally if needed.
    Convention: positive → model under-estimated the peak (more critical for
    procurement).
    """
    df = pd.DataFrame({"actual": actual.values, "p50": p50.values}, index=actual.index)
    df["date"] = df.index.normalize()

    daily_peak_hour = df.groupby("date")["actual"].idxmax()

    errors = []
    for date, peak_ts in daily_peak_hour.items():
        if pd.notna(peak_ts) and peak_ts in df.index:
            errors.append(df.loc[peak_ts, "actual"] - df.loc[peak_ts, "p50"])

    return float(np.nanmean(errors)) if errors else np.nan


def peak_precision_recall_f1(
    is_actual_peak: pd.Series,
    is_predicted_peak: pd.Series,
) -> dict[str, float]:
    """
    Precision, Recall, and F1-score for peak-hour detection.

    Definitions (binary classification — 'peak' = positive class):
        TP = predicted peak AND actual peak
        FP = predicted peak AND NOT actual peak
        FN = NOT predicted peak AND actual peak

        Precision = TP / (TP + FP)   fraction of flagged hours that are real peaks
        Recall    = TP / (TP + FN)   fraction of real peaks that are flagged
        F1        = 2 × P × R / (P + R)

    Returns dict with keys: precision, recall, f1.
    Returns NaN for undefined cases (e.g. no predicted peaks).
    """
    a = np.asarray(is_actual_peak,    dtype=bool)
    p = np.asarray(is_predicted_peak, dtype=bool)

    TP = np.sum(a & p)
    FP = np.sum(~a & p)
    FN = np.sum(a & ~p)

    precision = float(TP / (TP + FP)) if (TP + FP) > 0 else np.nan
    recall    = float(TP / (TP + FN)) if (TP + FN) > 0 else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else np.nan
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def improvement_pct(model_metric: float, baseline_metric: float) -> float:
    """
    Percentage improvement of *model* over *baseline* (lower-is-better metric).

    improvement = (baseline − model) / baseline × 100

    Positive → model is better.
    Negative → model is worse.
    Returns NaN if baseline is 0.
    """
    if baseline_metric == 0:
        return np.nan
    return float((baseline_metric - model_metric) / baseline_metric * 100)


# ---------------------------------------------------------------------------
# Single-segment metric bundle
# ---------------------------------------------------------------------------

def _compute_metric_row(
    subset: pd.DataFrame,
    peak_flags: pd.DataFrame | None,
    model: str,
    horizon: str,
    segment_type: str,
    segment_value: str,
) -> dict:
    """
    Compute all scalar metrics for one (model, horizon, segment) slice.

    Returns a flat dict suitable for one row of the metrics CSV.
    """
    actual = subset["actual"].values.astype(float)
    p10    = subset["p10"].values.astype(float)
    p50    = subset["p50"].values.astype(float)
    p90    = subset["p90"].values.astype(float)

    row = {
        "model":         model,
        "horizon":       horizon,
        "segment_type":  segment_type,
        "segment_value": segment_value,
        "n_hours":       len(subset),
        # Point metrics
        "mape_pct":    mape(actual, p50),
        "rmse_mw":     rmse(actual, p50),
        # Quantile metrics
        "pinball_p10": pinball_loss(actual, p10, q=0.1),
        "pinball_p50": pinball_loss(actual, p50, q=0.5),
        "pinball_p90": pinball_loss(actual, p90, q=0.9),
        # Interval
        "picp":        picp(actual, p10, p90),
        # Peak
        "peak_timing_error_min": peak_timing_error_minutes(
            subset["actual"], subset["p50"]
        ),
        "peak_magnitude_error_mw": peak_magnitude_error_mw(
            subset["actual"], subset["p50"]
        ),
    }

    # Peak P/R/F1 (requires peak_flags)
    if peak_flags is not None:
        joined = subset.join(peak_flags, how="left")
        if "is_actual_peak" in joined.columns and "is_predicted_peak" in joined.columns:
            valid = joined[["is_actual_peak", "is_predicted_peak"]].dropna()
            prf   = peak_precision_recall_f1(
                valid["is_actual_peak"], valid["is_predicted_peak"]
            )
            row.update({
                "peak_precision": prf["precision"],
                "peak_recall":    prf["recall"],
                "peak_f1":        prf["f1"],
            })
    else:
        row.update({"peak_precision": np.nan, "peak_recall": np.nan, "peak_f1": np.nan})

    return row


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def compute_all_metrics(
    horizon: Horizon,
    results_dir: str | Path = "results",
    raw_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute all metrics and write results/metrics.csv and results/missed_peaks.csv.

    Parameters
    ----------
    horizon     : "day_ahead" | "week_ahead"
    results_dir : directory containing forecasts_{horizon}.parquet and
                  (optionally) peak_flags.parquet; outputs written here too.
    raw_df      : Optional original DataFrame (DatetimeIndex + columns including
                  festival_name, temp_c) used to annotate missed-peak notes.
                  If None, notes fall back to generic labels.

    Returns
    -------
    pd.DataFrame  — the metrics table (also written to metrics.csv).
    """
    results_dir = Path(results_dir)

    # --- Load forecasts ---
    fc_path = results_dir / f"forecasts_{horizon}.parquet"
    if not fc_path.exists():
        raise FileNotFoundError(
            f"Could not find {fc_path}. Run run_backtest() first."
        )
    fc = pd.read_parquet(fc_path)
    fc["timestamp"] = pd.to_datetime(fc["timestamp"])
    fc = fc.set_index("timestamp").sort_index()

    # --- Load peak flags (optional) ---
    peak_path = results_dir / "peak_flags.parquet"
    peak_flags: pd.DataFrame | None = None
    if peak_path.exists():
        peak_flags = pd.read_parquet(peak_path)
        peak_flags.index = pd.to_datetime(peak_flags.index)
    else:
        logger.warning("peak_flags.parquet not found — peak P/R/F1 will be NaN.")

    # --- Attach segment labels ---
    fc["season"]    = _assign_season(fc.index.to_series())
    fc["hour_band"] = _assign_hour_band(fc.index.to_series())
    fc["peak_type"] = _assign_peak_type(fc["actual"])

    models = fc["model"].unique()
    all_rows: list[dict] = []

    for model in models:
        mdf = fc[fc["model"] == model]

        # Overall (no segmentation)
        all_rows.append(
            _compute_metric_row(mdf, peak_flags, model, horizon, "overall", "all")
        )

        # By season
        for season, sdf in mdf.groupby("season"):
            all_rows.append(
                _compute_metric_row(sdf, peak_flags, model, horizon, "season", season)
            )

        # By hour band
        for band, bdf in mdf.groupby("hour_band"):
            all_rows.append(
                _compute_metric_row(bdf, peak_flags, model, horizon, "hour_band", band)
            )

        # By peak type
        for ptype, pdf in mdf.groupby("peak_type"):
            all_rows.append(
                _compute_metric_row(pdf, peak_flags, model, horizon, "peak_type", ptype)
            )

    metrics_df = pd.DataFrame(all_rows)

    # --- Compute improvement % vs seasonal-naïve baseline ---
    baseline_model = "seasonal_naive"
    if baseline_model in models:
        baseline_rows = metrics_df[metrics_df["model"] == baseline_model].set_index(
            ["segment_type", "segment_value"]
        )
        for col in ["mape_pct", "rmse_mw", "pinball_p50"]:
            imp_col = f"improvement_{col}_pct"
            metrics_df[imp_col] = metrics_df.apply(
                lambda r: improvement_pct(
                    r[col],
                    baseline_rows.loc[(r["segment_type"], r["segment_value"]), col]
                    if (r["segment_type"], r["segment_value"]) in baseline_rows.index
                    else np.nan,
                ),
                axis=1,
            )
    else:
        logger.warning(
            "No '%s' model found in forecasts — improvement %% columns will be NaN.",
            baseline_model,
        )

    # --- Write metrics.csv ---
    out_metrics = results_dir / "metrics.csv"
    metrics_df.to_csv(out_metrics, index=False, float_format="%.4f")
    logger.info("Wrote metrics → %s", out_metrics)

    # --- Missed peaks ---
    _write_missed_peaks(fc, raw_df, results_dir)

    return metrics_df


# ---------------------------------------------------------------------------
# Missed-peaks report
# ---------------------------------------------------------------------------

def _auto_note(ts: pd.Timestamp, raw_df: pd.DataFrame | None) -> str:
    """
    Generate a short one-word note for a missed-peak event.
    Priority: festival > heatwave (temp ≥ 42 °C) > holiday > generic.
    """
    if raw_df is None or ts not in raw_df.index:
        return "unknown"

    row = raw_df.loc[ts]

    # Festival
    if "festival_name" in raw_df.columns:
        fest = str(row.get("festival_name", "")).strip()
        if fest:
            return f"festival:{fest}"

    # Heatwave
    if "temp_c" in raw_df.columns:
        t = row.get("temp_c", np.nan)
        if pd.notna(t) and t >= 42.0:
            return "heatwave"
        if pd.notna(t) and t >= 38.0:
            return "hot_day"

    # Holiday / lockdown / odd-even
    for flag in ("is_lockdown", "is_holiday", "is_oddeven"):
        if flag in raw_df.columns and row.get(flag, 0):
            return flag.replace("is_", "")

    return "demand_spike"


def _write_missed_peaks(
    fc: pd.DataFrame,
    raw_df: pd.DataFrame | None,
    results_dir: Path,
) -> None:
    """
    Identify and write results/missed_peaks.csv.

    A day is a "missed peak" if any of the following is true for the
    model's best-performing version (lowest MAPE model in fc):
        1. actual load exceeded P90 (interval miss)
        2. is_actual_peak=True but is_predicted_peak=False (flag miss)

    Columns in output:
        date, timestamp_of_peak, actual_mw, p90_mw, miss_type, note
    """
    # Use the first model's data as representative (worst-case approach)
    model_order = fc["model"].value_counts().index.tolist()
    primary_model = model_order[0]
    mdf = fc[fc["model"] == primary_model].copy()

    records = []

    # --- Type 1: actual > P90 ---
    above_p90 = mdf[mdf["actual"] > mdf["p90"]]
    for ts, row in above_p90.iterrows():
        note = _auto_note(ts, raw_df)
        records.append(
            {
                "date":              ts.date(),
                "timestamp_of_peak": ts,
                "actual_mw":         row["actual"],
                "p90_mw":            row["p90"],
                "miss_type":         "above_p90",
                "note":              note,
            }
        )

    # --- Type 2: flag miss (requires peak_flags) ---
    peak_path = results_dir / "peak_flags.parquet"
    if peak_path.exists():
        peak_flags = pd.read_parquet(peak_path)
        peak_flags.index = pd.to_datetime(peak_flags.index)
        joined = mdf.join(peak_flags, how="left")
        flag_miss = joined[
            joined.get("is_actual_peak", False) & ~joined.get("is_predicted_peak", True)
        ]
        for ts, row in flag_miss.iterrows():
            if ts not in above_p90.index:   # avoid duplicate
                note = _auto_note(ts, raw_df)
                records.append(
                    {
                        "date":              ts.date(),
                        "timestamp_of_peak": ts,
                        "actual_mw":         row["actual"],
                        "p90_mw":            row.get("p90", np.nan),
                        "miss_type":         "flag_miss",
                        "note":              note,
                    }
                )

    if records:
        missed_df = pd.DataFrame(records).sort_values("timestamp_of_peak")
        out_path  = results_dir / "missed_peaks.csv"
        missed_df.to_csv(out_path, index=False)
        logger.info(
            "Wrote %d missed-peak events → %s", len(missed_df), out_path
        )
    else:
        logger.info("No missed peaks detected (all actuals within P10-P90 band).")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute backtest metrics")
    parser.add_argument("--horizon",     default="day_ahead",
                        choices=["day_ahead", "week_ahead"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-parquet", default=None,
                        help="Optional path to raw data parquet (for missed-peak notes)")
    args = parser.parse_args()

    _raw = None
    if args.raw_parquet:
        _raw = pd.read_parquet(args.raw_parquet)
        if not isinstance(_raw.index, pd.DatetimeIndex):
            _raw = _raw.set_index("timestamp")
        _raw.index = pd.to_datetime(_raw.index)

    _metrics = compute_all_metrics(
        horizon=args.horizon,
        results_dir=args.results_dir,
        raw_df=_raw,
    )
    print(_metrics.to_string(index=False))
