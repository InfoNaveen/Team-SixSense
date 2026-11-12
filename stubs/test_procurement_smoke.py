"""
Smoke test for procurement/procurement.py using synthetic dummy forecasts.
Validates the procurement simulation math and contract 4 JSON output schema.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Adjust path to find procurement module
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from procurement.procurement import simulate_week, _default_params


def test_procurement_simulation():
    print("Testing procurement simulation on 168h synthetic dummy forecast...")

    # Generate 168 hours of synthetic forecast data mimicking Delhi summer load
    np.random.seed(42)
    timestamps = pd.date_range("2025-06-01 00:00:00", periods=168, freq="h")
    # Base daily cycle + noise
    hour_of_day = timestamps.hour
    base_cycle = 4500 + 1500 * np.sin((hour_of_day - 6) * np.pi / 12)
    actual = base_cycle + np.random.normal(0, 150, size=168)
    p50 = base_cycle + np.random.normal(0, 50, size=168)
    p10 = p50 - 300
    p90 = p50 + 400

    dummy_df = pd.DataFrame({
        "ts": timestamps,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "actual": actual,
    })

    result = simulate_week(dummy_df)

    # Validate output schema
    required_keys = {
        "schedule",
        "p90_exposure_mw",
        "cost_model",
        "cost_flat",
        "cost_probabilistic",
        "saving",
        "saving_pct",
        "hourly_detail",
    }
    missing_keys = required_keys - set(result.keys())
    assert not missing_keys, f"Missing output keys: {missing_keys}"
    assert len(result["schedule"]) == 168, f"Schedule length != 168 (got {len(result['schedule'])})"
    assert len(result["hourly_detail"]) == 168, f"Hourly detail != 168 (got {len(result['hourly_detail'])})"
    assert result["cost_flat"] > 0, "Flat cost must be positive"
    assert result["cost_probabilistic"] > 0, "Probabilistic cost must be positive"
    assert result["saving"] > 0, f"Expected savings > 0, got {result['saving']}"
    assert result["p90_exposure_mw"] >= 0, "P90 exposure cannot be negative"

    print(f"  [PASS] Schedule length: {len(result['schedule'])} hours")
    print(f"  [PASS] Cost flat: Rs. {result['cost_flat']:,.2f}")
    print(f"  [PASS] Cost probabilistic: Rs. {result['cost_probabilistic']:,.2f}")
    print(f"  [PASS] Savings: Rs. {result['saving']:,.2f} ({result['saving_pct']}%)")
    print(f"  [PASS] P90 exposure: {result['p90_exposure_mw']:,.2f} MW")
    print("\nProcurement smoke test passed successfully!")


if __name__ == "__main__":
    test_procurement_simulation()

