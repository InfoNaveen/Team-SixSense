"""
Stub implementation of the fit_predict contract (models/train.py).

This lets the Lead test the full pipeline end-to-end before the
Modeling teammate (P4) delivers the real LightGBM model.

Contract
--------
fit_predict(X_train, y_train, X_test) -> pd.DataFrame
    Returns a DataFrame aligned with X_test's index containing columns:
        p10  — 10th-percentile forecast (lower bound)
        p50  — 50th-percentile forecast (point estimate)
        p90  — 90th-percentile forecast (upper bound)
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def fit_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Stub: returns p50 = mean(y_train), p10 = p50*0.85, p90 = p50*1.15.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target (load in MW).
    X_test : pd.DataFrame
        Test feature matrix (used for index and length).

    Returns
    -------
    pd.DataFrame
        Columns ``p10``, ``p50``, ``p90`` matching X_test index.
    """
    p50 = float(np.mean(y_train)) if len(y_train) > 0 else 4000.0
    n = len(X_test)

    preds = pd.DataFrame(
        {
            "p10": np.full(n, p50 * 0.85),
            "p50": np.full(n, p50),
            "p90": np.full(n, p50 * 1.15),
        },
        index=X_test.index,
    )
    return preds
