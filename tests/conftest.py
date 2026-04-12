"""Pytest configuration: make the project root importable and provide fixtures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tiny_csv(tmp_path):
    """Write a 5-row CSV to a temp file and return its Path."""
    df = pd.DataFrame(
        {
            "msno": ["a", "b", "c", "d", "e"],
            "is_churn": [0, 1, 0, 0, 1],
            "total_secs": [1200.0, 0.0, 3600.5, 450.0, 0.0],
        }
    )
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def binary_probs():
    """Synthetic binary classification predictions with known structure.

    Returns y_true (balanced 0/1) and y_prob where positives have higher
    scores than negatives, so ROC-AUC should be clearly > 0.5.
    """
    rng = np.random.default_rng(seed=42)
    n = 200
    y_true = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
    y_prob = np.concatenate(
        [
            rng.beta(2, 5, size=n // 2),
            rng.beta(5, 2, size=n // 2),
        ]
    )
    return y_true, y_prob
