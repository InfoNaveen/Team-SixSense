"""
CodeSpark 2026 — Contract Checker
==================================
Validates that all agreed-upon contract files exist and expose the
correct columns / function signatures according to FIXED CONTRACTS.

Run:  python scripts/check_contracts.py

Checks:
  1. data/clean_hourly.parquet    — exists, has required columns (ts, load_mw, etc.)
  2. features/build_features.py   — exists, has build_features(df, horizon)
  3. eval/backtest.py             — exists, has get_folds(df), run_backtest(fit_predict, horizon)
  4. models/ (or stubs/)          — has fit_predict(X_train, y_train, X_test)
  5. procurement/procurement.py   — exists, has simulate_week(forecast_df, params)
  6. results/ files               — checks status of all 6 contract files
  7. app/app.py                   — exists, does not import features or models
"""

from __future__ import annotations

import os
import sys
import importlib.util

# Handle Windows cp1252 console safely
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ASCII / terminal-safe status badges
PASS_ICON = "[PASS]"
FAIL_ICON = "[FAIL]"
INFO_ICON = "[INFO]"

checks: list[tuple[str, bool, str]] = []


def check(label: str, passed: bool, detail: str = "") -> None:
    """Register a single contract check."""
    checks.append((label, passed, detail))
    icon = PASS_ICON if passed else FAIL_ICON
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon}  {label}{suffix}")


def info(msg: str) -> None:
    print(f"  {INFO_ICON}  {msg}")


# ---------------------------------------------------------------------------
# 1. data/clean_hourly.parquet
# ---------------------------------------------------------------------------
print("\n--- 1. Data Contract (data/clean_hourly.parquet) ---")
parquet_path = os.path.join(ROOT, "data", "clean_hourly.parquet")
if os.path.isfile(parquet_path):
    try:
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        required_cols = {
            "ts",
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
        actual_cols = set(df.columns)
        missing = required_cols - actual_cols
        if missing:
            check(
                "clean_hourly.parquet schema",
                False,
                f"Missing required columns: {sorted(missing)}",
            )
        else:
            check(
                "clean_hourly.parquet schema",
                True,
                f"{len(df)} rows, all 10 required columns present",
            )
    except Exception as exc:
        check("clean_hourly.parquet readable", False, str(exc))
else:
    check("clean_hourly.parquet exists", False, "File not yet produced by Data (P2)")


# ---------------------------------------------------------------------------
# 2. features/build_features.py -> build_features(df, horizon)
# ---------------------------------------------------------------------------
print("\n--- 2. Features Contract (features/build_features.py) ---")
feat_path = os.path.join(ROOT, "features", "build_features.py")
if os.path.isfile(feat_path):
    check("features/build_features.py exists", True)
    try:
        spec = importlib.util.spec_from_file_location("build_features", feat_path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        if hasattr(mod, "build_features") and callable(mod.build_features):
            check("build_features(df, horizon) callable", True)
        else:
            check("build_features() callable", False, "Function not found in module")
    except Exception as exc:
        check("features/build_features.py importable", False, str(exc))
else:
    check("features/build_features.py exists", False, "Not yet produced by Features (P3)")


# ---------------------------------------------------------------------------
# 3. eval/backtest.py -> get_folds(df), run_backtest(fit_predict, horizon)
# ---------------------------------------------------------------------------
print("\n--- 3. Eval Contract (eval/backtest.py) ---")
bt_path = os.path.join(ROOT, "eval", "backtest.py")
if os.path.isfile(bt_path):
    check("eval/backtest.py exists", True)
    try:
        spec = importlib.util.spec_from_file_location("backtest", bt_path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        for fn_name in ("get_folds", "run_backtest"):
            if hasattr(mod, fn_name) and callable(getattr(mod, fn_name)):
                check(f"{fn_name}() callable", True)
            else:
                check(f"{fn_name}() callable", False, f"Function {fn_name} not found")
    except Exception as exc:
        check("eval/backtest.py importable", False, str(exc))
else:
    check("eval/backtest.py exists", False, "Not yet produced by Features (P3)")


# ---------------------------------------------------------------------------
# 4. models/ (Modeling supplies fit_predict)
# ---------------------------------------------------------------------------
print("\n--- 4. Modeling Contract (models/train.py or stubs/stub_fit_predict.py) ---")
model_path = os.path.join(ROOT, "models", "train.py")
stub_path = os.path.join(ROOT, "stubs", "stub_fit_predict.py")
if os.path.isfile(model_path):
    check("models/train.py exists", True)
    try:
        spec = importlib.util.spec_from_file_location("models_train", model_path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        if hasattr(mod, "fit_predict") and callable(mod.fit_predict):
            check("models/train.py has fit_predict()", True)
        else:
            check("models/train.py has fit_predict()", False, "fit_predict not found")
    except Exception as exc:
        check("models/train.py importable", False, str(exc))
else:
    check("models/train.py exists", False, "Modeling (P4) in progress; checking stub...")
    if os.path.isfile(stub_path):
        check("stubs/stub_fit_predict.py fallback available", True)


# ---------------------------------------------------------------------------
# 5. procurement/procurement.py -> simulate_week(forecast_df, params)
# ---------------------------------------------------------------------------
print("\n--- 5. Procurement Contract (procurement/procurement.py) ---")
proc_path = os.path.join(ROOT, "procurement", "procurement.py")
if os.path.isfile(proc_path):
    check("procurement/procurement.py exists", True)
    try:
        spec = importlib.util.spec_from_file_location("procurement", proc_path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        if hasattr(mod, "simulate_week") and callable(mod.simulate_week):
            check("simulate_week(forecast_df, params) callable", True)
        else:
            check("simulate_week() callable", False, "simulate_week not found")
    except Exception as exc:
        check("procurement/procurement.py importable", False, str(exc))
else:
    check("procurement/procurement.py exists", False, "Lead owns procurement/")


# ---------------------------------------------------------------------------
# 6. results/ files
# ---------------------------------------------------------------------------
print("\n--- 6. Results Files Contract (results/) ---")
results_dir = os.path.join(ROOT, "results")
contract_results = [
    "forecasts_day.parquet",
    "forecasts_week.parquet",
    "peak_flags.parquet",
    "metrics.csv",
    "missed_peaks.csv",
    "procurement_week.json",
]
if os.path.isdir(results_dir):
    check("results/ directory exists", True)
    for fname in contract_results:
        fpath = os.path.join(results_dir, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            check(f"results/{fname}", True, f"{size_kb:.1f} KB")
        else:
            info(f"results/{fname} [pending pipeline run]")
else:
    check("results/ directory", False, "results/ directory missing")


# ---------------------------------------------------------------------------
# 7. app/app.py -> Data produces, reads results/ only
# ---------------------------------------------------------------------------
print("\n--- 7. App Contract (app/app.py) ---")
app_path = os.path.join(ROOT, "app", "app.py")
if os.path.isfile(app_path):
    check("app/app.py exists", True)
    with open(app_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    bad_imports = []
    for line in content.splitlines():
        line_clean = line.strip()
        if line_clean.startswith("import") or line_clean.startswith("from"):
            if "models" in line_clean:
                bad_imports.append("models")
            if "features" in line_clean:
                bad_imports.append("features")
    if bad_imports:
        check("app/app.py isolation", False, f"Illegal imports: {set(bad_imports)}")
    else:
        check("app/app.py isolation (reads results/ only)", True)
else:
    info("app/app.py [pending Data/Dashboard deliverable]")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
total_checked = len(checks)
passed_checked = sum(1 for _, ok, _ in checks if ok)
print(f"  Contract Status Summary: {passed_checked}/{total_checked} checks passed")
print("=" * 55 + "\n")
