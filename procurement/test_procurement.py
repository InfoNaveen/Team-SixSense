"""Unit tests for procurement/procurement.py (Lead / P1)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from procurement.procurement import (
    DEFAULT_PARAMS,
    build_dummy_week_forecast,
    evaluate_schedule_cost,
    simulate_week,
)


class TestProcurementSimulation(unittest.TestCase):
    def test_exact_cost_formula_hand_calculated(self) -> None:
        """Verify cost equation on a 2-hour deterministic input."""
        # Hour 0: schedule=100 MW, actual=120 MW -> shortfall 20 MW
        # Hour 1: schedule=100 MW, actual=90 MW  -> surplus 10 MW
        # contracted = (100+100)*1000 * 5 = 1,000,000 Rs
        # dsm        = 20*1000 * 12       =   240,000 Rs
        # stranded   = 10*1000 * 2.5      =    25,000 Rs
        # total      = 1,265,000 Rs
        schedule = np.array([100.0, 100.0])
        actual = np.array([120.0, 90.0])
        metrics = evaluate_schedule_cost(schedule, actual, DEFAULT_PARAMS)

        self.assertAlmostEqual(metrics["contracted_cost_rs"], 1_000_000.0)
        self.assertAlmostEqual(metrics["dsm_shortfall_cost_rs"], 240_000.0)
        self.assertAlmostEqual(metrics["stranded_surplus_cost_rs"], 25_000.0)
        self.assertAlmostEqual(metrics["total_cost_rs"], 1_265_000.0)
        self.assertEqual(metrics["shortfall_hours"], 1)

    def test_simulate_week_contract_keys_and_24x7_shape(self) -> None:
        """Verify Contract #4 and Contract #5 output structure on 168h stub."""
        df = build_dummy_week_forecast(seed=7)
        out = simulate_week(df, params={})

        for key in ("schedule", "p90_exposure_mw", "cost_model", "cost_flat", "saving"):
            self.assertIn(key, out)

        self.assertEqual(len(out["schedule"]), 24)
        self.assertTrue(all(len(row) == 7 for row in out["schedule"]))
        self.assertGreater(out["p90_exposure_mw"], 0.0)
        self.assertAlmostEqual(out["saving"], round(out["cost_flat"] - out["cost_model"], 2))
        self.assertGreater(out["saving"], 0.0)
        self.assertGreater(len(out["sensitivity"]), 0)

    def test_multi_model_and_multi_week_selection(self) -> None:
        """Verify simulate_week selects main model and a clean 168h block."""
        w1 = build_dummy_week_forecast(start_ts="2024-05-01 00:00:00+05:30", seed=1)
        w2 = build_dummy_week_forecast(start_ts="2024-05-08 00:00:00+05:30", seed=2)
        w2["actual"] = w2["actual"] + 500.0
        w2["p50"] = w2["p50"] + 500.0
        w2["p90"] = w2["p90"] + 500.0

        combined = pd.concat([w1, w2], ignore_index=True)
        naive = combined.copy()
        naive["model"] = "seasonal_naive"
        combined_models = pd.concat([naive, combined], ignore_index=True)

        out = simulate_week(combined_models, params={"week_selection": "peak_week"})
        self.assertEqual(out["n_hours"], 168)
        self.assertTrue(out["week_start"].startswith("2024-05-08"))

    def test_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=24, freq="h"), "p50": [100.0] * 24})
        with self.assertRaises(ValueError):
            simulate_week(df, params={})


if __name__ == "__main__":
    unittest.main()

