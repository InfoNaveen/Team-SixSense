"""
Contract Validator for CodeSpark 2026 Delhi DISCOM Copilot (Lead / P1).

Validates schema compliance across all fixed handoff artifacts without
modifying or reimplementing teammate modules.

Usage:
    python integration/validate_contracts.py --mode available
    python integration/validate_contracts.py --mode strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

CLEAN_HOURLY_COLS = {
    "load_mw",
    "load_mw_imputed_flag",
    "temp_c",
    "rh_pct",
    "apparent_temp_c",
    "is_holiday",
    "festival_name",
    "is_oddeven",
    "is_lockdown",
}

FORECAST_COLS = {"ts", "fold", "horizon", "model", "p10", "p50", "p90", "actual"}
PEAK_FLAGS_COLS = {
    "date",
    "fold",
    "peak_prob",
    "peak_pred",
    "peak_hour_pred",
    "peak_actual",
    "peak_hour_actual",
    "threshold_mw",
}
METRICS_COLS = {"horizon", "model", "fold", "metric", "value"}
MISSED_PEAKS_COLS = {
    "date",
    "actual_peak_mw",
    "pred_p50_peak_mw",
    "pred_p90_peak_mw",
    "hour_error_min",
    "note",
}
PROCUREMENT_KEYS = {"schedule", "p90_exposure_mw", "cost_model", "cost_flat", "saving"}


def _check_columns(actual_cols: set[str], required_cols: set[str], label: str) -> list[str]:
    missing = sorted(required_cols - actual_cols)
    if missing:
        return [f"{label}: missing columns {missing}"]
    return []


def validate_clean_hourly(path: Path) -> list[str]:
    errors: list[str] = []
    df = pd.read_parquet(path)
    cols = set(df.columns)
    if "ts" not in cols and df.index.name != "ts":
        errors.append("data/clean_hourly.parquet: missing 'ts' column or index")
    errors.extend(_check_columns(cols, CLEAN_HOURLY_COLS, "data/clean_hourly.parquet"))

    ts = pd.to_datetime(df["ts"] if "ts" in cols else df.index)
    if ts.duplicated().any():
        errors.append("data/clean_hourly.parquet: duplicate timestamps found")
    if not ts.is_monotonic_increasing:
        errors.append("data/clean_hourly.parquet: timestamps are not sorted ascending")
    return errors


def validate_forecasts(path: Path) -> list[str]:
    errors: list[str] = []
    df = pd.read_parquet(path)
    cols = set(df.columns)
    if "ts" not in cols and df.index.name == "ts":
        cols = cols | {"ts"}
    errors.extend(_check_columns(cols, FORECAST_COLS, str(path.relative_to(REPO_ROOT))))
    if {"p10", "p50", "p90"}.issubset(df.columns):
        crossing = ((df["p10"] > df["p50"]) | (df["p50"] > df["p90"])).sum()
        if crossing > 0:
            errors.append(f"{path.name}: found {int(crossing)} quantile-crossing rows")
    return errors


def validate_peak_flags(path: Path) -> list[str]:
    df = pd.read_parquet(path)
    return _check_columns(set(df.columns), PEAK_FLAGS_COLS, "results/peak_flags.parquet")


def validate_metrics(path: Path) -> list[str]:
    df = pd.read_csv(path)
    return _check_columns(set(df.columns), METRICS_COLS, "results/metrics.csv")


def validate_missed_peaks(path: Path) -> list[str]:
    df = pd.read_csv(path)
    return _check_columns(set(df.columns), MISSED_PEAKS_COLS, "results/missed_peaks.csv")


def validate_procurement_json(path: Path) -> list[str]:
    errors: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(PROCUREMENT_KEYS - set(data.keys()))
    if missing:
        errors.append(f"results/procurement_week.json: missing keys {missing}")
    sched = data.get("schedule")
    if not isinstance(sched, list) or len(sched) != 24 or any(len(r) != 7 for r in sched):
        errors.append("results/procurement_week.json: 'schedule' must be a 24x7 nested list")
    return errors


def validate_all(mode: str = "available") -> tuple[bool, list[str], list[str]]:
    """
    Validate all contract artifacts.

    Returns (passed, messages, errors).
    """
    checks = [
        (REPO_ROOT / "data" / "clean_hourly.parquet", validate_clean_hourly),
        (REPO_ROOT / "results" / "forecasts_day.parquet", validate_forecasts),
        (REPO_ROOT / "results" / "forecasts_week.parquet", validate_forecasts),
        (REPO_ROOT / "results" / "peak_flags.parquet", validate_peak_flags),
        (REPO_ROOT / "results" / "metrics.csv", validate_metrics),
        (REPO_ROOT / "results" / "missed_peaks.csv", validate_missed_peaks),
        (REPO_ROOT / "results" / "procurement_week.json", validate_procurement_json),
    ]

    messages: list[str] = []
    errors: list[str] = []

    for path, fn in checks:
        rel = str(path.relative_to(REPO_ROOT))
        if not path.exists():
            if mode == "strict":
                errors.append(f"MISSING: {rel}")
            else:
                messages.append(f"[PENDING] {rel} (not yet generated)")
            continue

        file_errors = fn(path)
        if file_errors:
            errors.extend(file_errors)
        else:
            messages.append(f"[OK]      {rel}")

    return len(errors) == 0, messages, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DISCOM hackathon contract files.")
    parser.add_argument(
        "--mode",
        choices=["available", "strict"],
        default="available",
        help="'available' skips missing teammate files; 'strict' requires all contracts.",
    )
    args = parser.parse_args()

    ok, messages, errors = validate_all(mode=args.mode)
    for msg in messages:
        print(msg)
    for err in errors:
        print(f"[ERROR]   {err}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

