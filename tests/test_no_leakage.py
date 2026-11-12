"""
tests/test_no_leakage.py
========================
Automated leakage and sanity tests for the features + evaluation harness.

Run with:
    pytest tests/test_no_leakage.py -v

Or as a standalone smoke-test (no pytest required):
    python tests/test_no_leakage.py --smoke
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
import pandas as pd
import pytest

# Allow running from project root without installing
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from features.build_features import build_features, assert_no_leakage
from eval.backtest import (
    get_folds,
    seasonal_naive_fit_predict,
    linear_regression_fit_predict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_df(n_hours: int = 24 * 7 * 20, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic hourly DataFrame spanning *n_hours*.
    Has all required raw columns with plausible values.
    """
    rng   = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_hours, freq="h")

    temp_c       = 25 + 10 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 365)) \
                   + 5 * rng.standard_normal(n_hours)
    humidity_pct = np.clip(60 + 20 * rng.standard_normal(n_hours), 20, 100)
    actual       = (
        2000
        + 500 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
        + 200 * rng.standard_normal(n_hours)
    )

    return pd.DataFrame(
        {
            "actual":          actual,
            "temp_c":          temp_c,
            "humidity_pct":    humidity_pct,
            "apparent_temp_c": temp_c - 2,
            "is_holiday":      (rng.random(n_hours) < 0.03).astype(int),
            "is_lockdown":     np.zeros(n_hours, dtype=int),
            "is_oddeven":      np.zeros(n_hours, dtype=int),
            "festival_name":   ["Diwali" if i == 500 else "" for i in range(n_hours)],
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# 1. get_folds
# ---------------------------------------------------------------------------

class TestGetFolds:
    def test_returns_12_folds(self):
        df    = _make_synthetic_df()
        folds = get_folds(df)
        assert len(folds) == 12, f"Expected 12 folds, got {len(folds)}"

    def test_train_strictly_before_test(self):
        df    = _make_synthetic_df()
        folds = get_folds(df)
        for fold in folds:
            assert fold["train"].index.max() < fold["test"].index.min(), (
                f"Fold {fold['fold_id']}: train overlaps test"
            )

    def test_no_gaps_between_folds(self):
        """Train end + 1 h should equal test start (continuous timeline)."""
        df    = _make_synthetic_df()
        folds = get_folds(df)
        for fold in folds:
            train_end  = fold["train"].index.max()
            test_start = fold["test"].index.min()
            gap = (test_start - train_end).total_seconds() / 3600
            assert gap <= 2, (  # allow up to 1 h gap at month boundary
                f"Fold {fold['fold_id']}: gap between train and test = {gap} h"
            )

    def test_expanding_train_window(self):
        df    = _make_synthetic_df()
        folds = get_folds(df)
        train_sizes = [len(f["train"]) for f in folds]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1], (
                f"Fold {i}: train shrank from {train_sizes[i-1]} to {train_sizes[i]}"
            )

    def test_every_month_is_different(self):
        df    = _make_synthetic_df()
        folds = get_folds(df)
        months = [f["month_label"] for f in folds]
        assert len(set(months)) == len(months), "Duplicate month labels in folds"

    def test_requires_datetimeindex(self):
        df = _make_synthetic_df().reset_index()
        with pytest.raises(TypeError):
            get_folds(df)

    def test_requires_actual_column(self):
        df = _make_synthetic_df().drop(columns=["actual"])
        with pytest.raises(ValueError):
            get_folds(df)


# ---------------------------------------------------------------------------
# 2. build_features — leakage
# ---------------------------------------------------------------------------

class TestLeakage:
    def _base_df(self):
        return _make_synthetic_df(n_hours=24 * 7 * 8)

    def test_no_leakage_day_ahead(self):
        df     = self._base_df()
        origin = df.index[-24 * 7]   # one week before the end
        feats  = build_features(df, horizon="day_ahead",  origin=origin)
        # assert_no_leakage is called inside build_features; if no exception → pass

    def test_no_leakage_week_ahead(self):
        df     = self._base_df()
        origin = df.index[-24 * 14]  # two weeks before the end
        feats  = build_features(df, horizon="week_ahead", origin=origin)

    def test_leakage_detected_when_introduced(self):
        """
        Manually create a leaky feature and confirm assert_no_leakage catches it.
        """
        df     = self._base_df()
        origin = df.index[-24 * 7]

        # Build clean features first (bypassing internal assert)
        import features.build_features as bf_module
        original_assert = bf_module.assert_no_leakage
        bf_module.assert_no_leakage = lambda *a, **kw: None   # patch out
        try:
            feats = build_features(df, horizon="day_ahead", origin=origin)
        finally:
            bf_module.assert_no_leakage = original_assert

        # Inject a leaky column: use future actual directly
        future_mask = df.index > origin
        feats.loc[future_mask, "load_lag_48h"] = df.loc[future_mask, "actual"] * 1.5

        with pytest.raises(AssertionError, match="LEAKAGE"):
            assert_no_leakage(feats, origin, df)


# ---------------------------------------------------------------------------
# 3. build_features — shape and content
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def _df_and_origin(self):
        df     = _make_synthetic_df()
        origin = df.index[-24 * 7]
        return df, origin

    def test_output_has_no_actual_column(self):
        df, origin = self._df_and_origin()
        feats = build_features(df, "day_ahead", origin)
        assert "actual" not in feats.columns

    def test_day_ahead_load_lags(self):
        df, origin = self._df_and_origin()
        feats = build_features(df, "day_ahead", origin)
        assert "load_lag_48h"  in feats.columns
        assert "load_lag_168h" in feats.columns
        assert "load_lag_336h" not in feats.columns  # week-ahead only

    def test_week_ahead_load_lags(self):
        df, origin = self._df_and_origin()
        feats = build_features(df, "week_ahead", origin)
        assert "load_lag_168h" in feats.columns
        assert "load_lag_336h" in feats.columns
        assert "load_lag_48h"  not in feats.columns  # day-ahead only

    def test_calendar_columns_present(self):
        df, origin = self._df_and_origin()
        feats = build_features(df, "day_ahead", origin)
        for col in ["hour", "day_of_week", "month", "is_weekend", "is_holiday",
                    "days_to_festival", "days_since_festival", "day_index"]:
            assert col in feats.columns, f"Missing calendar column: {col}"

    def test_cdh_non_negative(self):
        df, origin = self._df_and_origin()
        feats = build_features(df, "day_ahead", origin)
        assert (feats["cdh"] >= 0).all(), "CDH contains negative values"

    def test_heat_index_plausible(self):
        df, origin = self._df_and_origin()
        feats = build_features(df, "day_ahead", origin)
        assert feats["heat_index"].between(-20, 80).all(), \
            "Heat index out of plausible range"


# ---------------------------------------------------------------------------
# 4. Baselines smoke-test
# ---------------------------------------------------------------------------

class TestBaselines:
    def _folds(self):
        df    = _make_synthetic_df()
        return get_folds(df), df

    def test_seasonal_naive_returns_correct_columns(self):
        folds, df = self._folds()
        fold = folds[6]
        preds = seasonal_naive_fit_predict(fold["train"], fold["test"], "day_ahead")
        assert set(preds.columns) >= {"p10", "p50", "p90"}
        assert len(preds) == len(fold["test"])

    def test_seasonal_naive_p10_le_p50_le_p90(self):
        folds, df = self._folds()
        fold  = folds[6]
        preds = seasonal_naive_fit_predict(fold["train"], fold["test"], "day_ahead")
        assert (preds["p10"] <= preds["p50"]).all(), "P10 > P50 detected"
        assert (preds["p50"] <= preds["p90"]).all(), "P50 > P90 detected"

    def test_linear_regression_returns_correct_shape(self):
        folds, df = self._folds()
        fold  = folds[6]
        preds = linear_regression_fit_predict(fold["train"], fold["test"], "day_ahead")
        assert preds.shape == (len(fold["test"]), 3)

    def test_linear_regression_p10_le_p90(self):
        folds, df = self._folds()
        fold  = folds[6]
        preds = linear_regression_fit_predict(fold["train"], fold["test"], "day_ahead")
        assert (preds["p10"] <= preds["p90"]).all()


# ---------------------------------------------------------------------------
# Standalone smoke-test runner (no pytest)
# ---------------------------------------------------------------------------

def _smoke_test():
    print("=== Smoke Test ===")

    df = _make_synthetic_df()
    print(f"Synthetic df: {len(df)} rows, {df.index[0]} → {df.index[-1]}")

    folds = get_folds(df)
    print(f"Folds: {len(folds)}")

    fold = folds[6]
    preds_sn = seasonal_naive_fit_predict(fold["train"], fold["test"], "day_ahead")
    preds_lr = linear_regression_fit_predict(fold["train"], fold["test"], "day_ahead")
    print(f"Seasonal naïve predictions shape: {preds_sn.shape}")
    print(f"Linear regression predictions shape: {preds_lr.shape}")

    origin = fold["origin"]
    combined = pd.concat([fold["train"], fold["test"]])
    feats = build_features(combined, horizon="day_ahead", origin=origin)
    print(f"Features shape: {feats.shape}")
    print(f"Feature columns: {list(feats.columns)}")

    print("\n✅ All smoke tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args, _ = parser.parse_known_args()

    if args.smoke:
        _smoke_test()
    else:
        # Run pytest programmatically
        import pytest as _pytest
        sys.exit(_pytest.main([__file__, "-v"]))

