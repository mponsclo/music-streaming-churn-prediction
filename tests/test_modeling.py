"""Tests for src/modeling.py core functions."""

import numpy as np

from src.modeling import compute_metrics


def test_compute_metrics_returns_expected_keys(binary_probs):
    y_true, y_prob = binary_probs
    metrics = compute_metrics(y_true, y_prob)
    assert set(metrics) == {"log_loss", "roc_auc", "pr_auc", "f1", "f1_threshold"}


def test_compute_metrics_values_in_valid_ranges(binary_probs):
    y_true, y_prob = binary_probs
    metrics = compute_metrics(y_true, y_prob)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["log_loss"] > 0.0
    assert 0.05 <= metrics["f1_threshold"] <= 0.95


def test_compute_metrics_separates_signal_from_noise(binary_probs):
    """Our synthetic data has real signal (positives drawn from Beta(5,2)).

    Perfect predictions would give ROC-AUC 1.0, random would give 0.5.
    Our fixture should land comfortably above 0.75.
    """
    y_true, y_prob = binary_probs
    metrics = compute_metrics(y_true, y_prob)
    assert metrics["roc_auc"] > 0.75


def test_compute_metrics_random_prob_near_half():
    """Random predictions on balanced data should give ROC-AUC near 0.5."""
    rng = np.random.default_rng(seed=0)
    y_true = rng.integers(0, 2, size=2000)
    y_prob = rng.uniform(0, 1, size=2000)
    metrics = compute_metrics(y_true, y_prob)
    assert abs(metrics["roc_auc"] - 0.5) < 0.05
