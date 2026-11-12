"""
procurement.py  –  Probabilistic Procurement Cost Simulation
=============================================================
Compares three procurement scheduling strategies for a one-week
(168 h) window using quantile forecasts:

  Strategy        | How schedule_h is set
  ----------------+--------------------------------------
  P90  (cautious) | schedule_h = P90_h  (risk-averse)
  P50  (median)   | schedule_h = P50_h
  Flat (naive)    | schedule_h = mean(P90) for every hour

Cost model (per hour, h):
    cost_h = schedule_h * CONTRACTED
           + max(0, actual_h - schedule_h) * DSM
           + max(0, schedule_h - actual_h) * STRANDED

    Total cost = sum over 168 hours of cost_h
    (rates in Rs/kWh;  1 MW × 1 h = 1 MWh = 1000 kWh)

ASSUMPTIONS (placeholder rates — label clearly in every output):
    CONTRACTED  = Rs 5.00 / kWh   (day-ahead / bilateral)
    DSM         = Rs 12.00 / kWh  (real-time penalty for under-procurement)
    STRANDED    = Rs 2.50 / kWh   (waste cost for over-procurement)

Only pandas + numpy required.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


# ── ASSUMPTION block (single source of truth for all defaults) ────────
DEFAULT_PARAMS: dict[str, Any] = {
    # --- ASSUMPTIONS: placeholder rates, not actual DISCOM tariffs ---
    "rate_contracted_per_kwh": 5.00,   # Rs/kWh – day-ahead / PPA
    "rate_dsm_per_kwh":       12.00,   # Rs/kWh – DSM / real-time penalty
    "rate_stranded_per_kwh":  2.50,    # Rs/kWh – over-procurement waste
}

# Sensitivity grid for the memo / pitch
SENSITIVITY_DSM   = [8, 10, 12, 14, 16]       # Rs/kWh
SENSITIVITY_STRAND = [1.0, 2.0, 2.5, 3.0, 4.0] # Rs/kWh


# ── Helpers ───────────────────────────────────────────────────────────

def _kwh_to_mwh(rate_kwh: float) -> float:
    """Convert Rs/kWh to Rs/MWh (×1000)."""
    return rate_kwh * 1000.0


def _hour_cost(
    schedule_mw: float,
    actual_mw: float,
    r_contracted: float,   # Rs/MWh
    r_dsm: float,          # Rs/MWh
    r_stranded: float,     # Rs/MWh
) -> dict[str, float]:
    """Cost for one hour under the given schedule.

    Returns dict with individual components and total (all in Rs).
    """
    contracted_cost = schedule_mw * r_contracted
    under = max(0.0, actual_mw - schedule_mw)
    over  = max(0.0, schedule_mw - actual_mw)
    dsm_cost      = under * r_dsm
    stranded_cost = over  * r_stranded

    return {
        "schedule_mw":     round(schedule_mw, 2),
        "actual_mw":       round(actual_mw, 2),
        "under_mw":        round(under, 2),
        "over_mw":         round(over, 2),
        "contracted_cost": round(contracted_cost, 2),
        "dsm_cost":        round(dsm_cost, 2),
        "stranded_cost":   round(stranded_cost, 2),
        "total_cost":      round(contracted_cost + dsm_cost + stranded_cost, 2),
    }


def _strategy_cost(
    schedule: np.ndarray,
    actual: np.ndarray,
    r_contracted: float,
    r_dsm: float,
    r_stranded: float,
) -> tuple[float, list[dict]]:
    """Total cost and per-hour detail for a 168 h schedule."""
    details = []
    for s, a in zip(schedule, actual):
        details.append(_hour_cost(s, a, r_contracted, r_dsm, r_stranded))
    total = sum(d["total_cost"] for d in details)
    return round(total, 2), details


# ── Public API ────────────────────────────────────────────────────────

def simulate_week(
    forecast_df: pd.DataFrame,
    params: dict | None = None,
) -> dict[str, Any]:
    """Simulate one week of procurement under three strategies.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        168 rows with columns: ts, p10, p50, p90, actual
    params : dict, optional
        Override any key from DEFAULT_PARAMS.

    Returns
    -------
    dict  (matches results/procurement_week.json schema)
        Keys: schedule_p90, schedule_p50, schedule_flat,
              p90_exposure_mw, cost_p90, cost_p50, cost_flat,
              saving_p90_vs_flat, saving_pct,
              sensitivity_table, assumptions, hourly_detail_p90
    """
    # ── validate ──────────────────────────────────────────────────
    required = {"ts", "p10", "p50", "p90", "actual"}
    missing = required - set(forecast_df.columns)
    if missing:
        raise ValueError(f"forecast_df missing columns: {missing}")
    if len(forecast_df) < 168:
        raise ValueError(f"Need 168 rows, got {len(forecast_df)}")
    # Take exactly 168 rows (last complete week)
    df = forecast_df.tail(168).reset_index(drop=True)

    # ── merge params ──────────────────────────────────────────────
    p = {**DEFAULT_PARAMS, **(params or {})}
    r_con = _kwh_to_mwh(p["rate_contracted_per_kwh"])
    r_dsm = _kwh_to_mwh(p["rate_dsm_per_kwh"])
    r_str = _kwh_to_mwh(p["rate_stranded_per_kwh"])

    actual   = df["actual"].values.astype(float)
    p50_vals = df["p50"].values.astype(float)
    p90_vals = df["p90"].values.astype(float)

    # ── three schedules ───────────────────────────────────────────
    sched_p90  = p90_vals.copy()                              # risk-averse
    sched_p50  = p50_vals.copy()                              # median
    sched_flat = np.full(168, float(np.mean(p90_vals)))       # flat = avg(P90)

    # ── costs ─────────────────────────────────────────────────────
    cost_p90, detail_p90   = _strategy_cost(sched_p90,  actual, r_con, r_dsm, r_str)
    cost_p50, _            = _strategy_cost(sched_p50,  actual, r_con, r_dsm, r_str)
    cost_flat, _           = _strategy_cost(sched_flat, actual, r_con, r_dsm, r_str)

    saving_p90_vs_flat = round(cost_flat - cost_p90, 2)
    saving_pct = round(
        (saving_p90_vs_flat / cost_flat * 100) if cost_flat else 0.0, 2
    )

    # ── P90 risk exposure: max shortfall when using P90 schedule ──
    p90_exposure = round(float(np.max(np.maximum(actual - sched_p90, 0))), 2)

    # ── sensitivity table ─────────────────────────────────────────
    sensitivity = []
    for dsm_rate in SENSITIVITY_DSM:
        for strand_rate in SENSITIVITY_STRAND:
            rd = _kwh_to_mwh(dsm_rate)
            rs = _kwh_to_mwh(strand_rate)
            c90, _ = _strategy_cost(sched_p90,  actual, r_con, rd, rs)
            cfl, _ = _strategy_cost(sched_flat, actual, r_con, rd, rs)
            sensitivity.append({
                "dsm_rs_kwh":      dsm_rate,
                "stranded_rs_kwh": strand_rate,
                "cost_p90":        c90,
                "cost_flat":       cfl,
                "saving":          round(cfl - c90, 2),
                "saving_pct":      round((cfl - c90) / cfl * 100 if cfl else 0, 2),
            })

    # ── assumptions declaration ───────────────────────────────────
    assumptions = {
        "note": "PLACEHOLDER RATES – not actual DISCOM tariffs",
        "rate_contracted_rs_kwh": p["rate_contracted_per_kwh"],
        "rate_dsm_rs_kwh":       p["rate_dsm_per_kwh"],
        "rate_stranded_rs_kwh":  p["rate_stranded_per_kwh"],
        "flat_schedule_rule":    "constant = mean(P90) across all 168 hours",
        "p90_schedule_rule":     "schedule_h = P90_h (risk-averse)",
        "p50_schedule_rule":     "schedule_h = P50_h (median)",
        "cost_formula":          "cost_h = schedule*contracted + max(0,actual-schedule)*dsm + max(0,schedule-actual)*stranded",
    }

    # ── attach timestamps to hourly detail ────────────────────────
    for i, d in enumerate(detail_p90):
        d["ts"] = str(df.iloc[i]["ts"])

    return {
        "schedule_p90":        [round(float(x), 2) for x in sched_p90],
        "schedule_p50":        [round(float(x), 2) for x in sched_p50],
        "schedule_flat":       [round(float(x), 2) for x in sched_flat],
        "p90_exposure_mw":     p90_exposure,
        "cost_p90":            cost_p90,
        "cost_p50":            cost_p50,
        "cost_flat":           cost_flat,
        "saving_p90_vs_flat":  saving_p90_vs_flat,
        "saving_pct":          saving_pct,
        "sensitivity_table":   sensitivity,
        "assumptions":         assumptions,
        "hourly_detail_p90":   detail_p90,
    }
