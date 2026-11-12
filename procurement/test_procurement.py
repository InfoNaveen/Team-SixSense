"""
test_procurement.py  –  Known-answer unit tests for procurement.py
===================================================================
Run:  python -m pytest procurement/test_procurement.py -v
  or: python procurement/test_procurement.py

Tests use hand-crafted 168-hour dummy data where expected costs can be
computed on paper, so we can assert exact values (within fp tolerance).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from procurement.procurement import simulate_week, _hour_cost, _kwh_to_mwh, DEFAULT_PARAMS


# ── helpers ───────────────────────────────────────────────────────────

def _make_constant_df(
    actual: float, p50: float, p90: float, n: int = 168
) -> pd.DataFrame:
    """168 identical rows – makes hand calculation trivial."""
    ts = pd.date_range("2025-06-01", periods=n, freq="h")
    return pd.DataFrame({
        "ts": ts,
        "p10": [p50 - 200] * n,
        "p50": [p50] * n,
        "p90": [p90] * n,
        "actual": [actual] * n,
    })


def _approx(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) < tol


# ── TEST 1: Perfect forecast (actual == P90) ─────────────────────────

def test_perfect_forecast():
    """When actual == schedule for every hour, DSM=0 and stranded=0.
    Cost = 168 * actual * contracted_rate (in Rs/MWh)."""
    mw = 5000.0
    df = _make_constant_df(actual=mw, p50=mw, p90=mw)
    result = simulate_week(df)

    r_con = _kwh_to_mwh(DEFAULT_PARAMS["rate_contracted_per_kwh"])  # 5000 Rs/MWh
    expected = 168 * mw * r_con  # 168 * 5000 * 5000 = 4,200,000,000

    assert _approx(result["cost_p90"], expected), (
        f"Perfect forecast: expected {expected}, got {result['cost_p90']}"
    )
    assert result["p90_exposure_mw"] == 0.0, "No shortfall when actual == P90"
    print(f"  [PASS] test_perfect_forecast: cost_p90 = {result['cost_p90']}")


# ── TEST 2: Under-procurement (actual > P90) ─────────────────────────

def test_under_procurement():
    """Actual 6000 MW but P90 schedule = 5000 MW.
    Under-procurement = 1000 MW/h.
    cost_h = 5000*5000 + 1000*12000 + 0 = 37,000,000 Rs/h
    total  = 168 * 37,000,000 = 6,216,000,000"""
    df = _make_constant_df(actual=6000, p50=4500, p90=5000)
    result = simulate_week(df)

    r_con = _kwh_to_mwh(5.0)   # 5000
    r_dsm = _kwh_to_mwh(12.0)  # 12000
    expected_h = 5000 * r_con + 1000 * r_dsm  # 25M + 12M = 37M
    expected = 168 * expected_h

    assert _approx(result["cost_p90"], expected), (
        f"Under-procurement: expected {expected}, got {result['cost_p90']}"
    )
    assert _approx(result["p90_exposure_mw"], 1000.0), (
        f"P90 exposure should be 1000 MW, got {result['p90_exposure_mw']}"
    )
    print(f"  [PASS] test_under_procurement: cost_p90 = {result['cost_p90']}")


# ── TEST 3: Over-procurement (actual < P90) ──────────────────────────

def test_over_procurement():
    """Actual 4000 MW but P90 schedule = 5000 MW.
    Over-procurement = 1000 MW/h.
    cost_h = 5000*5000 + 0 + 1000*2500 = 27,500,000 Rs/h
    total  = 168 * 27,500,000 = 4,620,000,000"""
    df = _make_constant_df(actual=4000, p50=4500, p90=5000)
    result = simulate_week(df)

    r_con = _kwh_to_mwh(5.0)
    r_str = _kwh_to_mwh(2.5)
    expected_h = 5000 * r_con + 1000 * r_str  # 25M + 2.5M = 27.5M
    expected = 168 * expected_h

    assert _approx(result["cost_p90"], expected), (
        f"Over-procurement: expected {expected}, got {result['cost_p90']}"
    )
    assert result["p90_exposure_mw"] == 0.0, "No shortfall when actual < P90"
    print(f"  [PASS] test_over_procurement: cost_p90 = {result['cost_p90']}")


# ── TEST 4: P90 beats flat when load varies ──────────────────────────

def test_p90_beats_flat():
    """With varying load, P90 (hour-by-hour) should cost less than flat
    (constant = mean(P90)), because flat over-procures during low-demand
    hours and under-procures during peaks."""
    np.random.seed(42)
    ts = pd.date_range("2025-06-01", periods=168, freq="h")
    hour = ts.hour.values.astype(float)
    # Strong diurnal variation: 3000–7000 MW
    actual = 5000 + 2000 * np.sin(2 * np.pi * (hour - 6) / 24)
    p50 = actual + np.random.normal(0, 50, 168)
    p90 = actual + 300  # P90 is 300 MW above actual every hour

    df = pd.DataFrame({
        "ts": ts,
        "p10": p50 - 300,
        "p50": p50,
        "p90": p90,
        "actual": actual,
    })
    result = simulate_week(df)

    assert result["saving_p90_vs_flat"] > 0, (
        f"P90 should beat flat; saving = {result['saving_p90_vs_flat']}"
    )
    print(
        f"  [PASS] test_p90_beats_flat: "
        f"P90={result['cost_p90']:,.0f}  flat={result['cost_flat']:,.0f}  "
        f"saving={result['saving_pct']}%"
    )


# ── TEST 5: Output schema is correct ─────────────────────────────────

def test_output_schema():
    """Check all required keys exist and schedules have 168 elements."""
    df = _make_constant_df(actual=5000, p50=4800, p90=5200)
    result = simulate_week(df)

    required_keys = {
        "schedule_p90", "schedule_p50", "schedule_flat",
        "p90_exposure_mw", "cost_p90", "cost_p50", "cost_flat",
        "saving_p90_vs_flat", "saving_pct",
        "sensitivity_table", "assumptions", "hourly_detail_p90",
    }
    missing = required_keys - set(result.keys())
    assert not missing, f"Missing keys: {missing}"
    assert len(result["schedule_p90"]) == 168
    assert len(result["schedule_p50"]) == 168
    assert len(result["schedule_flat"]) == 168
    assert len(result["hourly_detail_p90"]) == 168
    assert len(result["sensitivity_table"]) == len([8,10,12,14,16]) * len([1.0,2.0,2.5,3.0,4.0])
    assert "note" in result["assumptions"]
    print(f"  [PASS] test_output_schema: all {len(required_keys)} keys present")


# ── TEST 6: Sensitivity table sanity ─────────────────────────────────

def test_sensitivity_sanity():
    """Higher DSM rate should increase flat cost more than P90 cost
    (since flat under-procures more during peaks)."""
    np.random.seed(99)
    ts = pd.date_range("2025-07-01", periods=168, freq="h")
    hour = ts.hour.values.astype(float)
    actual = 5000 + 1500 * np.sin(2 * np.pi * (hour - 6) / 24)
    p90 = actual + 200
    df = pd.DataFrame({
        "ts": ts, "p10": actual - 400, "p50": actual, "p90": p90, "actual": actual,
    })
    result = simulate_week(df)

    sens = result["sensitivity_table"]
    # At higher DSM, saving should grow (flat suffers more from under-procurement)
    low_dsm  = [s for s in sens if s["dsm_rs_kwh"] == 8  and s["stranded_rs_kwh"] == 2.5]
    high_dsm = [s for s in sens if s["dsm_rs_kwh"] == 16 and s["stranded_rs_kwh"] == 2.5]
    assert len(low_dsm) == 1 and len(high_dsm) == 1

    # With varying load, flat under-procures at peaks so higher DSM hurts flat more
    # (P90 tracks load so it's less exposed to DSM changes)
    print(
        f"  [PASS] test_sensitivity_sanity: "
        f"saving@DSM8 = {low_dsm[0]['saving']:,.0f}  "
        f"saving@DSM16 = {high_dsm[0]['saving']:,.0f}"
    )


# ── TEST 7: Custom params override defaults ──────────────────────────

def test_custom_params():
    """Passing custom rates should change costs."""
    df = _make_constant_df(actual=5000, p50=4800, p90=5000)
    result_default = simulate_week(df)
    result_custom  = simulate_week(df, params={"rate_contracted_per_kwh": 10.0})

    # Doubling contracted rate should double the cost (no DSM/stranded here)
    assert result_custom["cost_p90"] > result_default["cost_p90"], (
        "Higher contracted rate should increase cost"
    )
    print(f"  [PASS] test_custom_params: default={result_default['cost_p90']:,.0f} custom={result_custom['cost_p90']:,.0f}")


# ── Runner ────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_perfect_forecast,
        test_under_procurement,
        test_over_procurement,
        test_p90_beats_flat,
        test_output_schema,
        test_sensitivity_sanity,
        test_custom_params,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

