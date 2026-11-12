#!/usr/bin/env python
"""
run_all.py  –  Master integration script (Lead owns this)
==========================================================
Runs the full pipeline from a clean clone:
    data -> features -> backtest -> procurement -> summary

If a teammate's module is missing, it reports what's missing and
continues with whatever is available.  The dummy generator fills in
contract-compatible placeholder files so the pipeline is always runnable.

Usage:
    python run_all.py            # run full pipeline
    python run_all.py --dummy    # generate dummy files first, then run
"""
from __future__ import annotations

import os
import sys
import glob
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
DATA_PATH = os.path.join(ROOT, "data", "clean_hourly.parquet")


# ── Helpers ───────────────────────────────────────────────────────────

def _print(tag: str, msg: str):
    print(f"  [{tag}] {msg}")

def ok(msg: str):   _print("PASS", msg)
def fail(msg: str): _print("FAIL", msg)
def warn(msg: str): _print("WARN", msg)
def info(msg: str): _print("INFO", msg)

results: list[tuple[str, bool]] = []


def step(name: str):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")


# ── Step 0: Parse args & optional dummy generation ────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dummy", action="store_true",
                    help="Generate dummy contract files before running")
args, _ = parser.parse_known_args()

if args.dummy:
    step("Step 0: Generate dummy contract files")
    try:
        from stubs.generate_dummy_forecasts import main as gen_dummy
        gen_dummy()
        ok("Dummy files generated in results/")
    except Exception as e:
        fail(f"Dummy generation failed: {e}")


# ── Step 1: Check data contract ──────────────────────────────────────

step("Step 1: Data contract  (data/clean_hourly.parquet)")
df_clean = None
if os.path.isfile(DATA_PATH):
    try:
        import pandas as pd
        df_clean = pd.read_parquet(DATA_PATH)
        required = {"ts","load_mw","load_mw_imputed_flag","temp_c","rh_pct",
                     "apparent_temp_c","is_holiday","festival_name","is_oddeven","is_lockdown"}
        missing = required - set(df_clean.columns)
        if missing:
            warn(f"Parquet found but missing columns: {sorted(missing)}")
            results.append(("Data", False))
        else:
            ok(f"Validated  ({len(df_clean)} rows, {len(df_clean.columns)} cols)")
            results.append(("Data", True))
    except Exception as e:
        fail(f"Could not read: {e}")
        results.append(("Data", False))
else:
    fail("data/clean_hourly.parquet not found (P2 in progress)")
    results.append(("Data", False))


# ── Step 2: Feature engineering ──────────────────────────────────────

step("Step 2: Feature engineering  (features/build_features.py)")
try:
    from features.build_features import build_features
    if df_clean is not None:
        for h in ("day", "week"):
            feat = build_features(df=df_clean, horizon=h)
            ok(f"horizon='{h}' -> {feat.shape}")
        results.append(("Features", True))
    else:
        warn("Skipped (no clean data)")
        results.append(("Features", False))
except ImportError:
    fail("features.build_features not available (P3 in progress)")
    results.append(("Features", False))
except Exception as e:
    fail(f"Error: {e}")
    results.append(("Features", False))


# ── Step 3: Model contract ───────────────────────────────────────────

step("Step 3: Model contract  (models/train.py or stubs/)")
fit_predict_fn = None
try:
    from models.train import fit_predict
    fit_predict_fn = fit_predict
    ok("Loaded models.train.fit_predict (P4)")
    results.append(("Model", True))
except ImportError:
    warn("models.train not ready; trying stub")
    try:
        from stubs.stub_fit_predict import fit_predict
        fit_predict_fn = fit_predict
        ok("Using stubs.stub_fit_predict")
        results.append(("Model (stub)", True))
    except ImportError:
        fail("No model or stub available")
        results.append(("Model", False))


# ── Step 4: Walk-forward backtest ────────────────────────────────────

step("Step 4: Backtest  (eval/backtest.py)")
try:
    from eval.backtest import run_backtest
    if fit_predict_fn:
        for h in ("day", "week"):
            run_backtest(fit_predict=fit_predict_fn, horizon=h)
            ok(f"Backtest done for horizon='{h}'")
        results.append(("Backtest", True))
    else:
        fail("No fit_predict available")
        results.append(("Backtest", False))
except ImportError:
    fail("eval.backtest not available (P3 in progress)")
    results.append(("Backtest", False))
except Exception as e:
    fail(f"Backtest error: {e}")
    results.append(("Backtest", False))


# ── Step 5: Procurement simulation ───────────────────────────────────

step("Step 5: Procurement simulation  (procurement/)")
try:
    from procurement.run_procurement import run_procurement
    res = run_procurement()
    if res:
        ok("procurement_week.json written")
        results.append(("Procurement", True))
    else:
        warn("Procurement skipped (no forecast parquet)")
        results.append(("Procurement", False))
except Exception as e:
    fail(f"Procurement error: {e}")
    results.append(("Procurement", False))


# ── Step 6: Results inventory ────────────────────────────────────────

step("Step 6: Results file inventory")
expected = ["forecasts_day.parquet", "forecasts_week.parquet",
            "peak_flags.parquet", "metrics.csv", "missed_peaks.csv",
            "procurement_week.json"]
for f in expected:
    p = os.path.join(RESULTS, f)
    if os.path.isfile(p):
        kb = os.path.getsize(p) / 1024
        ok(f"results/{f}  ({kb:.1f} KB)")
    else:
        info(f"results/{f}  [pending]")


# ── Final report ─────────────────────────────────────────────────────

step("Pipeline Report Card")
for name, passed in results:
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}]  {name}")

n_pass = sum(p for _, p in results)
print(f"\n  {n_pass}/{len(results)} steps passed.")
print(f"  Dashboard: streamlit run app/app.py")
print()
