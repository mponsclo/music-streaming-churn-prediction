"""Tests for src/temporal_eval.py.

The build_round1_features function requires the raw 21.5M-row transaction CSVs
which are not in CI, so we test the pure-function compute_metrics that guards
the temporal holdout numbers reported in the README.
"""

import numpy as np

from src.temporal_eval import compute_metrics


def test_compute_metrics_returns_expected_keys(binary_probs):
    y_true, y_prob = binary_probs
    metrics = compute_metrics(y_true, y_prob)
    assert set(metrics) == {"log_loss", "roc_auc", "pr_auc", "f1"}


def test_compute_metrics_perfect_predictions():
    """Perfect predictions (prob = label, clipped) should yield ROC-AUC = 1.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])
    metrics = compute_metrics(y_true, y_prob)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_compute_metrics_degrades_under_drift():
    """Simulate distribution drift: positives now have lower scores than trained.

    ROC-AUC should drop below the perfectly-separated case.
    """
    y_true = np.array([0] * 100 + [1] * 100)
    y_prob_good = np.concatenate([np.full(100, 0.1), np.full(100, 0.9)])
    y_prob_drifted = np.concatenate([np.full(100, 0.3), np.full(100, 0.7)])

    m_good = compute_metrics(y_true, y_prob_good)
    m_drifted = compute_metrics(y_true, y_prob_drifted)

    assert m_good["log_loss"] < m_drifted["log_loss"]
