#!/usr/bin/env python
"""
run_procurement.py  –  Read forecasts, run simulation, write JSON
==================================================================
Usage:
    python -m procurement.run_procurement
    python procurement/run_procurement.py

Reads results/forecasts_week.parquet (or the _DUMMY variant),
picks the last fold's last 168 hours, calls simulate_week(),
writes results/procurement_week.json, and prints a summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS   = ROOT / "results"
REAL_PATH = RESULTS / "forecasts_week.parquet"
DUMMY_PATH = RESULTS / "forecasts_week_DUMMY.parquet"
OUTPUT     = RESULTS / "procurement_week.json"

from procurement.procurement import simulate_week


def run_procurement() -> dict | None:
    """Load forecasts, simulate, save JSON, and print summary."""
    # Prefer real file; fall back to dummy
    if REAL_PATH.exists():
        src = REAL_PATH
    elif DUMMY_PATH.exists():
        src = DUMMY_PATH
        print("[procurement] Using DUMMY forecasts (real file not yet available)")
    else:
        print(
            "[procurement] No forecast file found.\n"
            f"  Expected: {REAL_PATH}\n"
            f"  Or dummy: {DUMMY_PATH}\n"
            "  Run: python stubs/generate_dummy_forecasts.py"
        )
        return None

    print(f"[procurement] Reading {src.name} ...")
    df = pd.read_parquet(src)
    print(f"  Loaded {len(df)} rows, cols: {list(df.columns)}")

    # Use last fold's last 168 hours
    if "fold" in df.columns:
        last_fold = df["fold"].max()
        df = df[df["fold"] == last_fold]
        print(f"  Filtered to fold {last_fold} ({len(df)} rows)")

    if "ts" in df.columns:
        df = df.sort_values("ts")

    if len(df) < 168:
        print(f"  [ERROR] Need 168 rows, only {len(df)} available")
        return None

    week = df.tail(168).reset_index(drop=True)
    print(f"  Week: {week['ts'].iloc[0]}  to  {week['ts'].iloc[-1]}")

    # Run simulation
    print("[procurement] Running simulation ...")
    result = simulate_week(week)

    # Write JSON
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[procurement] Written -> {OUTPUT.name}")

    # Print summary
    print()
    print("=" * 60)
    print("  PROCUREMENT COST COMPARISON  (ASSUMPTION: placeholder rates)")
    print("=" * 60)
    print(f"  {'Strategy':<20} {'Total Cost (Rs)':>20} {'vs Flat':>12}")
    print("-" * 60)
    for label, key in [("P90 (risk-averse)", "cost_p90"),
                       ("P50 (median)",      "cost_p50"),
                       ("Flat (naive)",       "cost_flat")]:
        cost = result[key]
        diff = result["cost_flat"] - cost
        pct  = diff / result["cost_flat"] * 100 if result["cost_flat"] else 0
        sign = "+" if diff >= 0 else ""
        print(f"  {label:<20} {cost:>20,.0f} {sign}{pct:>10.1f}%")
    print("-" * 60)
    print(f"  P90 exposure (max shortfall): {result['p90_exposure_mw']:,.1f} MW")
    print(f"  Saving (P90 vs Flat):         Rs {result['saving_p90_vs_flat']:,.0f}")
    print("=" * 60)

    # Print sensitivity snippet
    print("\n  Sensitivity (P90 saving vs Flat at different DSM/Stranded rates):")
    sens = result["sensitivity_table"]
    print(f"  {'DSM(Rs/kWh)':<14} {'Stranded(Rs/kWh)':<18} {'Saving(Rs)':>16} {'%':>8}")
    for s in sens:
        if s["stranded_rs_kwh"] in [1.0, 2.5, 4.0]:  # show subset
            print(f"  {s['dsm_rs_kwh']:<14} {s['stranded_rs_kwh']:<18} {s['saving']:>16,.0f} {s['saving_pct']:>7.1f}%")
    print()

    return result


if __name__ == "__main__":
    res = run_procurement()
    sys.exit(0 if res else 1)
