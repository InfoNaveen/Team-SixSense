"""
Standalone smoke-test: validates backtest harness + feature builder
using only pandas + numpy (no sklearn, no pytest).

Run:
    python tests/smoke_no_deps.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

# ── Minimal stubs so imports work without sklearn ───────────────────────────
import types

_sklearn_stub = types.ModuleType("sklearn")
_lr_stub = types.ModuleType("sklearn.linear_model")

class _LinearRegressionStub:
    def fit(self, X, y):
        self._coef = np.linalg.lstsq(
            np.hstack([X, np.ones((len(X),1))]), y, rcond=None
        )[0]
        return self
    def predict(self, X):
        return np.hstack([X, np.ones((len(X),1))]) @ self._coef

_lr_stub.LinearRegression = _LinearRegressionStub
_sklearn_stub.linear_model = _lr_stub
sys.modules.setdefault("sklearn", _sklearn_stub)
sys.modules.setdefault("sklearn.linear_model", _lr_stub)

# ── Now import our code ──────────────────────────────────────────────────────
from eval.backtest import (
    get_folds, run_backtest,
    seasonal_naive_fit_predict, linear_regression_fit_predict,
)
from features.build_features import build_features, assert_no_leakage
from eval.metrics import (
    mape, rmse, pinball_loss, picp,
    peak_timing_error_minutes, peak_magnitude_error_mw,
    peak_precision_recall_f1, improvement_pct,
)

PASS = "✅"
FAIL = "❌"
results = []

def check(name, fn):
    try:
        fn()
        print(f"{PASS}  {name}")
        results.append(True)
    except Exception as e:
        print(f"{FAIL}  {name}  →  {e}")
        results.append(False)

# ── Synthetic data ───────────────────────────────────────────────────────────
rng   = np.random.default_rng(42)
N     = 24 * 7 * 60          # ~14 months (> 12 required by get_folds)
index = pd.date_range("2024-01-01", periods=N, freq="h")
df = pd.DataFrame({
    "actual":          2000 + 500*np.sin(2*np.pi*np.arange(N)/24) + 200*rng.standard_normal(N),
    "temp_c":          25 + 10*np.sin(2*np.pi*np.arange(N)/(24*365)) + 5*rng.standard_normal(N),
    "humidity_pct":    np.clip(60 + 20*rng.standard_normal(N), 20, 100),
    "apparent_temp_c": 23 + 10*np.sin(2*np.pi*np.arange(N)/(24*365)),
    "is_holiday":      (rng.random(N) < 0.03).astype(int),
    "is_lockdown":     np.zeros(N, dtype=int),
    "is_oddeven":      np.zeros(N, dtype=int),
    "festival_name":   ["Diwali" if i == 500 else "" for i in range(N)],
}, index=index)

# ── 1. get_folds ─────────────────────────────────────────────────────────────
def _test_folds_count():
    folds = get_folds(df)
    assert len(folds) == 12, f"Expected 12 folds, got {len(folds)}"

def _test_folds_no_overlap():
    folds = get_folds(df)
    for f in folds:
        assert f["train"].index.max() < f["test"].index.min()

def _test_folds_expanding():
    folds = get_folds(df)
    sizes = [len(f["train"]) for f in folds]
    for i in range(1, len(sizes)):
        assert sizes[i] >= sizes[i-1]

check("get_folds: returns 12 folds",          _test_folds_count)
check("get_folds: train strictly before test", _test_folds_no_overlap)
check("get_folds: expanding train window",     _test_folds_expanding)

# ── 2. Baselines ─────────────────────────────────────────────────────────────
folds = get_folds(df)
fold  = folds[-1]   # last fold always valid regardless of data length

def _test_sn_shape():
    p = seasonal_naive_fit_predict(fold["train"], fold["test"], "day_ahead")
    assert p.shape[1] == 3 and len(p) == len(fold["test"])

def _test_sn_monotone():
    p = seasonal_naive_fit_predict(fold["train"], fold["test"], "day_ahead")
    assert (p["p10"] <= p["p50"]).all()
    assert (p["p50"] <= p["p90"]).all()

def _test_lr_shape():
    p = linear_regression_fit_predict(fold["train"], fold["test"], "day_ahead")
    assert p.shape[1] == 3 and len(p) == len(fold["test"])

def _test_lr_monotone():
    p = linear_regression_fit_predict(fold["train"], fold["test"], "day_ahead")
    assert (p["p10"] <= p["p90"]).all()

check("seasonal_naive: correct shape",    _test_sn_shape)
check("seasonal_naive: P10 ≤ P50 ≤ P90", _test_sn_monotone)
check("linear_regression: correct shape", _test_lr_shape)
check("linear_regression: P10 ≤ P90",    _test_lr_monotone)

# ── 3. build_features ────────────────────────────────────────────────────────
origin   = fold["origin"]
combined = pd.concat([fold["train"], fold["test"]]).sort_index()

def _test_features_no_actual():
    f = build_features(combined, "day_ahead", origin)
    assert "actual" not in f.columns

def _test_features_day_ahead_lags():
    f = build_features(combined, "day_ahead", origin)
    assert "load_lag_48h"  in f.columns
    assert "load_lag_168h" in f.columns
    assert "load_lag_336h" not in f.columns

def _test_features_week_ahead_lags():
    f = build_features(combined, "week_ahead", origin)
    assert "load_lag_168h" in f.columns
    assert "load_lag_336h" in f.columns
    assert "load_lag_48h"  not in f.columns

def _test_cdh_non_negative():
    f = build_features(combined, "day_ahead", origin)
    assert (f["cdh"] >= 0).all()

def _test_calendar_cols():
    f = build_features(combined, "day_ahead", origin)
    for c in ["hour","day_of_week","month","is_weekend","day_index","days_to_festival"]:
        assert c in f.columns, f"Missing: {c}"

def _test_leakage_guard():
    import features.build_features as bf
    orig_assert = bf.assert_no_leakage
    bf.assert_no_leakage = lambda *a, **kw: None   # patch out temporarily
    try:
        f = build_features(combined, "day_ahead", origin)
    finally:
        bf.assert_no_leakage = orig_assert
    # Inject a leaky value
    future_mask = combined.index > origin
    f.loc[future_mask, "load_lag_48h"] = combined.loc[future_mask, "actual"] * 1.5
    try:
        assert_no_leakage(f, origin, combined)
        raise AssertionError("Leakage NOT detected — guard is broken!")
    except AssertionError as e:
        if "LEAKAGE" not in str(e):
            raise

check("build_features: no actual column",           _test_features_no_actual)
check("build_features: day-ahead load lags",         _test_features_day_ahead_lags)
check("build_features: week-ahead load lags",        _test_features_week_ahead_lags)
check("build_features: CDH ≥ 0",                     _test_cdh_non_negative)
check("build_features: all calendar columns present",_test_calendar_cols)
check("leakage guard: catches injected leakage",     _test_leakage_guard)

# ── 4. Metrics ───────────────────────────────────────────────────────────────
actual = np.array([100., 200., 300., 400., 500.])
pred   = np.array([110., 190., 310., 390., 510.])

def _test_mape():
    v = mape(actual, pred)
    assert 0 < v < 10, f"MAPE={v}"

def _test_rmse():
    v = rmse(actual, pred)
    assert v > 0, f"RMSE={v}"
    assert abs(v - np.sqrt(np.mean((actual-pred)**2))) < 1e-6

def _test_pinball_q05_eq_mae_half():
    # pinball at q=0.5 should equal MAE/2
    pb  = pinball_loss(actual, pred, q=0.5)
    mae = np.mean(np.abs(actual - pred))
    assert abs(pb - mae/2) < 1e-6, f"pinball(0.5)={pb}, MAE/2={mae/2}"

def _test_picp_full():
    p10 = actual - 20
    p90 = actual + 20
    assert abs(picp(actual, p10, p90) - 1.0) < 1e-6

def _test_picp_zero():
    assert picp(actual, actual + 100, actual + 200) == 0.0

def _test_improvement():
    assert improvement_pct(5.0, 10.0) == 50.0
    assert improvement_pct(10.0, 10.0) == 0.0

def _test_prf_perfect():
    a = np.array([True, False, True, False])
    p = np.array([True, False, True, False])
    r = peak_precision_recall_f1(a, p)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0

check("mape: plausible value",             _test_mape)
check("rmse: matches formula",             _test_rmse)
check("pinball q=0.5: equals MAE/2",       _test_pinball_q05_eq_mae_half)
check("picp: 1.0 when all covered",        _test_picp_full)
check("picp: 0.0 when none covered",       _test_picp_zero)
check("improvement_pct: correct formula",  _test_improvement)
check("peak P/R/F1: perfect classifier",   _test_prf_perfect)

# ── 5. run_backtest end-to-end ────────────────────────────────────────────────
def _test_run_backtest_writes_parquet():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        run_backtest(
            seasonal_naive_fit_predict, "day_ahead",
            df, results_dir=tmp, model_name="snaive_test",
        )
        out = pathlib.Path(tmp) / "forecasts_day_ahead.parquet"
        assert out.exists()
        result = pd.read_parquet(out)
        assert set(result.columns) >= {"timestamp","actual","p10","p50","p90","fold_id","model"}
        assert len(result) > 0

check("run_backtest: writes forecasts parquet", _test_run_backtest_writes_parquet)

# ── Summary ──────────────────────────────────────────────────────────────────
n_pass = sum(results)
n_fail = len(results) - n_pass
print(f"\n{'='*50}")
print(f"Results: {n_pass}/{len(results)} passed", ("🎉" if n_fail == 0 else "⚠️"))
if n_fail:
    sys.exit(1)
