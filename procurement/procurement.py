"""
Procurement Cost Simulation & Schedule Builder (Lead / P1).

Contract #5:
    simulate_week(forecast_df: pd.DataFrame, params: dict) -> dict

Contract #4 (results/procurement_week.json):
    schedule (24x7 values), p90_exposure_mw, cost_model, cost_flat, saving

All cost parameters are explicit assumptions per Requirements.pdf (Section 6):
    - Contracted day-ahead rate: Rs 5.0 / kWh
    - Under-procurement (DSM / real-time) rate: Rs 12.0 / kWh on shortfall
    - Over-procurement stranded rate: Rs 2.5 / kWh on unused surplus
    - Unit conversion: 1 MW for 1 hour = 1 MWh = 1,000 kWh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Marked assumptions from Requirements.pdf Section 6
DEFAULT_PARAMS: dict[str, Any] = {
    "contracted_rate_rs_per_kwh": 5.0,
    "dsm_rate_rs_per_kwh": 12.0,
    "stranded_rate_rs_per_kwh": 2.5,
    "hours_per_step": 1.0,
    "mw_to_kwh": 1000.0,  # 1 MW * 1 h = 1,000 kWh
    "model_name": None,   # Optional filter when forecast_df has multiple models
    "fold": None,         # Optional filter when forecast_df has multiple folds
    "week_selection": "peak_week",  # "peak_week" | "last_week" | "first_week"
}

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def _merge_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Merge caller params onto DEFAULT_PARAMS without mutating caller dict."""
    merged = dict(DEFAULT_PARAMS)
    if params:
        merged.update(params)
    return merged


def _extract_timestamp_series(df: pd.DataFrame) -> pd.Series:
    """Return a Series preserving timezone-aware IST timestamps if present."""
    if "ts" in df.columns:
        return pd.to_datetime(df["ts"]).reset_index(drop=True)
    if isinstance(df.index, pd.DatetimeIndex) or df.index.name == "ts":
        return pd.Series(pd.to_datetime(df.index)).reset_index(drop=True)
    raise ValueError("forecast_df must contain a 'ts' column or a DatetimeIndex.")


def select_sample_week(
    forecast_df: pd.DataFrame,
    model_name: str | None = None,
    fold: int | str | None = None,
    week_selection: str = "peak_week",
) -> pd.DataFrame:
    """
    Normalize forecast_df and select a clean 168-hour (7 x 24h) sample week.

    If forecast_df already has <= 168 rows, sorts by ts and returns it directly.
    If forecast_df spans multiple models/folds/months, filters to one model and
    selects a contiguous 168-hour block starting at hour 00:00.
    """
    if forecast_df.empty:
        raise ValueError("forecast_df is empty.")

    df = forecast_df.copy().reset_index(drop=True)
    df["ts"] = _extract_timestamp_series(forecast_df)

    # Filter to a single model if multiple models exist
    if "model" in df.columns and df["model"].nunique() > 1:
        if model_name is not None:
            df = df[df["model"] == model_name].copy()
        else:
            # Prefer non-baseline LightGBM/main model if present
            models = list(df["model"].dropna().unique())
            preferred = [
                m for m in models
                if "naive" not in str(m).lower() and "linear" not in str(m).lower()
            ]
            chosen = preferred[0] if preferred else models[0]
            df = df[df["model"] == chosen].copy()

    # Filter to a single fold if requested
    if fold is not None and "fold" in df.columns:
        df = df[df["fold"] == fold].copy()

    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    required_cols = {"p10", "p50", "p90", "actual"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"forecast_df missing required columns: {sorted(missing)}")

    if len(df) <= 168:
        return df

    # Build aligned 7-day (168h) blocks starting at 00:00
    df["_date"] = df["ts"].dt.date
    daily_counts = df.groupby("_date").size()
    full_days = sorted(daily_counts[daily_counts == 24].index)

    if len(full_days) < 7:
        return df.tail(168).drop(columns=["_date"]).reset_index(drop=True)

    candidate_blocks: list[tuple[tuple[float, float], pd.DataFrame]] = []
    for i in range(0, len(full_days) - 6):
        window_days = full_days[i : i + 7]
        # Verify calendar continuity (7 consecutive days)
        if (window_days[-1] - window_days[0]).days != 6:
            continue
        block = df[df["_date"].isin(window_days)].copy()
        if len(block) == 168:
            peak_val = float(block["actual"].max())
            total_val = float(block["actual"].sum())
            candidate_blocks.append(((peak_val, total_val), block))

    if not candidate_blocks:
        return df.tail(168).drop(columns=["_date"]).reset_index(drop=True)

    if week_selection == "first_week":
        chosen_block = candidate_blocks[0][1]
    elif week_selection == "last_week":
        chosen_block = candidate_blocks[-1][1]
    else:
        # Default "peak_week": pick the 7-day block with the highest peak load
        chosen_block = max(candidate_blocks, key=lambda item: item[0])[1]

    return chosen_block.drop(columns=["_date"]).reset_index(drop=True)


def evaluate_schedule_cost(
    schedule_mw: np.ndarray,
    actual_mw: np.ndarray,
    params: dict[str, Any],
) -> dict[str, float]:
    """
    Evaluate hourly procurement cost and reliability metrics for a buy schedule.

    Cost_h = (schedule_h * contracted_rate)
           + max(0, actual_h - schedule_h) * dsm_rate
           + max(0, schedule_h - actual_h) * stranded_rate
    """
    kwh_factor = float(params["hours_per_step"]) * float(params["mw_to_kwh"])
    c_rate = float(params["contracted_rate_rs_per_kwh"])
    dsm_rate = float(params["dsm_rate_rs_per_kwh"])
    str_rate = float(params["stranded_rate_rs_per_kwh"])

    shortfall_mw = np.maximum(0.0, actual_mw - schedule_mw)
    surplus_mw = np.maximum(0.0, schedule_mw - actual_mw)

    contracted_rs = float(np.sum(schedule_mw * kwh_factor * c_rate))
    dsm_rs = float(np.sum(shortfall_mw * kwh_factor * dsm_rate))
    stranded_rs = float(np.sum(surplus_mw * kwh_factor * str_rate))
    total_rs = contracted_rs + dsm_rs + stranded_rs

    return {
        "contracted_cost_rs": round(contracted_rs, 2),
        "dsm_shortfall_cost_rs": round(dsm_rs, 2),
        "stranded_surplus_cost_rs": round(stranded_rs, 2),
        "total_cost_rs": round(total_rs, 2),
        "contracted_mwh": round(float(np.sum(schedule_mw * float(params["hours_per_step"]))), 2),
        "shortfall_mwh": round(float(np.sum(shortfall_mw * float(params["hours_per_step"]))), 2),
        "surplus_mwh": round(float(np.sum(surplus_mw * float(params["hours_per_step"]))), 2),
        "max_shortfall_mw": round(float(np.max(shortfall_mw)), 2),
        "shortfall_hours": int(np.sum(shortfall_mw > 1e-6)),
    }


def _reshape_schedule_24x7(week_df: pd.DataFrame, schedule_mw: np.ndarray) -> tuple[list[list[float]], list[str]]:
    """
    Reshape a 168-hour schedule into a 24x7 matrix (24 hours x 7 days).

    If week_df is not an exact 24x7 grid, pads or falls back cleanly while
    preserving chronological hourly values.
    """
    dates = week_df["ts"].dt.strftime("%Y-%m-%d").tolist()
    unique_days = list(dict.fromkeys(dates))

    if len(week_df) == 168 and len(unique_days) == 7:
        # Build explicit (hour 0..23) x (day 0..6) grid
        grid = np.zeros((24, 7), dtype=float)
        day_to_col = {d: idx for idx, d in enumerate(unique_days)}
        hours = week_df["ts"].dt.hour.to_numpy()
        for val, d, h in zip(schedule_mw, dates, hours):
            grid[int(h), day_to_col[d]] = round(float(val), 2)
        return grid.tolist(), unique_days

    # Fallback for non-168-row test inputs
    n_cols = max(1, int(np.ceil(len(schedule_mw) / 24)))
    padded = np.full(24 * n_cols, np.nan, dtype=float)
    padded[: len(schedule_mw)] = np.round(schedule_mw, 2)
    grid_list = [
        [None if np.isnan(x) else float(x) for x in row]
        for row in padded.reshape((n_cols, 24)).T
    ]
    return grid_list, unique_days


def run_sensitivity(
    forecast_df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    dsm_rates: tuple[float, ...] = (8.0, 10.0, 12.0, 15.0, 18.0),
    stranded_rates: tuple[float, ...] = (1.5, 2.5, 3.5),
) -> list[dict[str, Any]]:
    """
    Compute cost and savings sensitivity across assumed DSM and stranded rates.

    Directly supports judge Q&A on why cost savings hold under varying tariffs.
    """
    base_params = _merge_params(params)
    week_df = select_sample_week(
        forecast_df,
        model_name=base_params.get("model_name"),
        fold=base_params.get("fold"),
        week_selection=base_params.get("week_selection", "peak_week"),
    )
    p90 = week_df["p90"].to_numpy(dtype=float)
    p50 = week_df["p50"].to_numpy(dtype=float)
    actual = week_df["actual"].to_numpy(dtype=float)
    flat = np.full_like(p90, fill_value=float(np.mean(p90)))

    rows: list[dict[str, Any]] = []
    for dsm_rate in dsm_rates:
        for stranded_rate in stranded_rates:
            p = dict(base_params)
            p["dsm_rate_rs_per_kwh"] = float(dsm_rate)
            p["stranded_rate_rs_per_kwh"] = float(stranded_rate)
            m_p90 = evaluate_schedule_cost(p90, actual, p)
            m_p50 = evaluate_schedule_cost(p50, actual, p)
            m_flat = evaluate_schedule_cost(flat, actual, p)
            saving = round(m_flat["total_cost_rs"] - m_p90["total_cost_rs"], 2)
            saving_pct = round(100.0 * saving / m_flat["total_cost_rs"], 2) if m_flat["total_cost_rs"] else 0.0
            rows.append({
                "dsm_rate_rs_per_kwh": float(dsm_rate),
                "stranded_rate_rs_per_kwh": float(stranded_rate),
                "cost_model_rs": m_p90["total_cost_rs"],
                "cost_p50_rs": m_p50["total_cost_rs"],
                "cost_flat_rs": m_flat["total_cost_rs"],
                "saving_vs_flat_rs": saving,
                "saving_vs_flat_pct": saving_pct,
            })
    return rows


def simulate_week(forecast_df: pd.DataFrame, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Simulate one week of DISCOM power procurement using P90 vs P50 and flat schedules.

    Args:
        forecast_df: DataFrame with columns ts (or DatetimeIndex), p10, p50, p90, actual.
        params: Optional dictionary overriding DEFAULT_PARAMS cost assumptions.

    Returns:
        Dictionary matching Contract #4 (results/procurement_week.json) plus
        P50 comparison, cost breakdown, reliability metrics, and sensitivity.
    """
    cfg = _merge_params(params)
    week_df = select_sample_week(
        forecast_df,
        model_name=cfg.get("model_name"),
        fold=cfg.get("fold"),
        week_selection=cfg.get("week_selection", "peak_week"),
    )

    p10 = week_df["p10"].to_numpy(dtype=float)
    p50 = week_df["p50"].to_numpy(dtype=float)
    p90 = week_df["p90"].to_numpy(dtype=float)
    actual = week_df["actual"].to_numpy(dtype=float)

    # Risk-averse schedule buys P90 each hour; naive baseline buys flat weekly mean(P90)
    schedule_p90 = p90
    schedule_p50 = p50
    flat_level_mw = float(np.mean(p90))
    schedule_flat = np.full_like(p90, fill_value=flat_level_mw)

    p90_metrics = evaluate_schedule_cost(schedule_p90, actual, cfg)
    p50_metrics = evaluate_schedule_cost(schedule_p50, actual, cfg)
    flat_metrics = evaluate_schedule_cost(schedule_flat, actual, cfg)

    cost_model = p90_metrics["total_cost_rs"]
    cost_flat = flat_metrics["total_cost_rs"]
    cost_p50 = p50_metrics["total_cost_rs"]
    saving = round(cost_flat - cost_model, 2)
    saving_pct = round(100.0 * saving / cost_flat, 2) if cost_flat > 0 else 0.0

    # P90 risk exposure: peak uncertainty buffer above point forecast (MW)
    tail_spread_mw = np.maximum(0.0, p90 - p50)
    p90_exposure_mw = round(float(np.max(tail_spread_mw)), 2)

    schedule_24x7, schedule_days = _reshape_schedule_24x7(week_df, schedule_p90)
    sensitivity = run_sensitivity(week_df, cfg)

    return {
        # Required Contract #4 top-level keys
        "schedule": schedule_24x7,
        "p90_exposure_mw": p90_exposure_mw,
        "cost_model": cost_model,
        "cost_flat": cost_flat,
        "saving": saving,
        # Supplementary keys for dashboard & procurement memo
        "saving_pct_vs_flat": saving_pct,
        "cost_p50": cost_p50,
        "saving_vs_p50": round(cost_p50 - cost_model, 2),
        "flat_schedule_mw": round(flat_level_mw, 2),
        "week_start": str(week_df["ts"].iloc[0]),
        "week_end": str(week_df["ts"].iloc[-1]),
        "n_hours": int(len(week_df)),
        "schedule_days": schedule_days,
        "schedule_hours": list(range(24)),
        "schedule_hourly": [round(float(x), 2) for x in schedule_p90],
        "p10_hourly": [round(float(x), 2) for x in p10],
        "p50_hourly": [round(float(x), 2) for x in p50],
        "actual_hourly": [round(float(x), 2) for x in actual],
        "timestamps": [ts.isoformat() for ts in week_df["ts"]],
        "exposure_details": {
            "max_p90_minus_p50_mw": p90_exposure_mw,
            "mean_p90_minus_p50_mw": round(float(np.mean(tail_spread_mw)), 2),
            "max_p90_mw": round(float(np.max(p90)), 2),
            "max_actual_mw": round(float(np.max(actual)), 2),
            "p90_shortfall_hours": p90_metrics["shortfall_hours"],
            "p50_shortfall_hours": p50_metrics["shortfall_hours"],
            "flat_shortfall_hours": flat_metrics["shortfall_hours"],
            "p90_shortfall_mwh": p90_metrics["shortfall_mwh"],
            "p50_shortfall_mwh": p50_metrics["shortfall_mwh"],
            "flat_shortfall_mwh": flat_metrics["shortfall_mwh"],
        },
        "cost_breakdown": {
            "p90_model": p90_metrics,
            "p50_point": p50_metrics,
            "flat_p90_avg": flat_metrics,
        },
        "sensitivity": sensitivity,
        "assumptions": {
            "note": "All tariff rates are explicit hackathon assumptions, not live DISCOM PPA/DSM tariffs.",
            "contracted_rate_rs_per_kwh": float(cfg["contracted_rate_rs_per_kwh"]),
            "dsm_rate_rs_per_kwh": float(cfg["dsm_rate_rs_per_kwh"]),
            "stranded_rate_rs_per_kwh": float(cfg["stranded_rate_rs_per_kwh"]),
            "mw_to_kwh": float(cfg["mw_to_kwh"]),
            "schedule_rule": "Risk-averse hourly buy = P90_h; compared against P50_h and flat mean(P90).",
        },
    }


def build_dummy_week_forecast(
    start_ts: str = "2024-06-10 00:00:00+05:30",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic 168-hour synthetic Delhi summer week forecast stub.

    Used ONLY for standalone testing / pre-H6 dashboard wiring before Modeling
    and Features produce results/forecasts_week.parquet.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start_ts, periods=168, freq="h")
    hours = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()

    # Delhi June profile: morning ramp ~06-10, afternoon high, evening peak ~19-23
    diurnal = (
        5300.0
        + 650.0 * np.exp(-0.5 * ((hours - 15.0) / 3.2) ** 2)
        + 1450.0 * np.exp(-0.5 * ((hours - 22.0) / 2.4) ** 2)
        - 750.0 * np.exp(-0.5 * ((hours - 5.0) / 2.2) ** 2)
    )
    weekend_adj = np.where(dow >= 5, -220.0, 0.0)
    heatwave_drift = np.linspace(0.0, 380.0, 168)

    p50 = diurnal + weekend_adj + heatwave_drift + rng.normal(0.0, 45.0, size=168)
    spread = 210.0 + 110.0 * np.exp(-0.5 * ((hours - 21.5) / 3.0) ** 2)
    p10 = p50 - spread
    p90 = p50 + spread
    actual = p50 + rng.normal(0.0, spread / 1.45, size=168)

    # Ensure quantile monotonicity p10 <= p50 <= p90
    sorted_q = np.sort(np.vstack([p10, p50, p90]), axis=0)
    return pd.DataFrame({
        "ts": ts,
        "fold": 11,
        "horizon": "week",
        "model": "stub_lightgbm_quantile",
        "p10": np.round(sorted_q[0], 2),
        "p50": np.round(sorted_q[1], 2),
        "p90": np.round(sorted_q[2], 2),
        "actual": np.round(actual, 2),
    })


def _load_forecast_input(input_path: Path | None, use_dummy: bool) -> tuple[pd.DataFrame, bool]:
    """Load real results parquet if available, or return dummy stub when requested."""
    if use_dummy:
        return build_dummy_week_forecast(), True

    candidates = [input_path] if input_path else [
        RESULTS_DIR / "forecasts_week.parquet",
        RESULTS_DIR / "forecasts_day.parquet",
    ]
    for path in candidates:
        if path is not None and path.exists():
            return pd.read_parquet(path), False

    raise FileNotFoundError(
        "No forecast results found in results/forecasts_week.parquet or "
        "results/forecasts_day.parquet. Run with --dummy for stub testing."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DISCOM weekly procurement simulation.")
    parser.add_argument("--input", type=Path, default=None, help="Path to forecasts parquet.")
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "procurement_week.json",
        help="Output JSON path (Contract #4).",
    )
    parser.add_argument("--dummy", action="store_true", help="Use dummy stub forecast for testing.")
    parser.add_argument("--model", type=str, default=None, help="Filter to specific model name.")
    parser.add_argument(
        "--week-selection",
        choices=["peak_week", "last_week", "first_week"],
        default="peak_week",
        help="How to choose the 168h sample week when input spans multiple weeks.",
    )
    args = parser.parse_args()

    forecast_df, is_dummy = _load_forecast_input(args.input, args.dummy)
    result = simulate_week(
        forecast_df,
        params={
            "model_name": args.model,
            "week_selection": args.week_selection,
        },
    )
    result["is_dummy_data"] = is_dummy

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Wrote procurement simulation -> {args.out}")
    print(
        f"Mode: {'DUMMY STUB' if is_dummy else 'REAL FORECASTS'} | "
        f"Window: {result['week_start']} to {result['week_end']} ({result['n_hours']}h)"
    )
    print(
        f"Cost P90 Model: Rs {result['cost_model']:,.2f} | "
        f"Cost Flat: Rs {result['cost_flat']:,.2f} | "
        f"Saving: Rs {result['saving']:,.2f} ({result['saving_pct_vs_flat']}%)"
    )
    print(f"P90 Exposure (max P90-P50): {result['p90_exposure_mw']:.2f} MW")


if __name__ == "__main__":
    main()
