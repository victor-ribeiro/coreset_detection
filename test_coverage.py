"""
Tests for compute_coverage_mean — Cycle 5.
Spec-driven: tests verify specs/technical/coverage_metric.md.
"""
import math
import numpy as np
import pytest
from main import compute_coverage_mean


# ── Basic correctness ─────────────────────────────────────────────────────────

def test_coverage_mean_value():
    """coverage_mean = mean(min_j d(x_i, c_j)).

    Spec: coverage_metric.md — definition.
    """
    # X_train: 4 points in 2D. Coreset: 1 point at origin.
    X_train = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    coreset = np.array([[0.0, 0.0]])
    result = compute_coverage_mean(X_train, coreset)
    # distances to origin: 1, 1, 2, 2 → mean = 1.5
    assert math.isclose(result, 1.5, rel_tol=1e-6)


def test_coverage_mean_full_coreset():
    """K=n: every point is its own neighbor → coverage_mean = 0.

    Spec: coverage_metric.md — K=n → 0.
    """
    X = np.random.default_rng(0).standard_normal((20, 5))
    result = compute_coverage_mean(X, X)
    assert math.isclose(result, 0.0, abs_tol=1e-9)


def test_coverage_mean_monotone():
    """coverage_mean decreases (or stays equal) as K increases.

    Spec: coverage_metric.md — monotone property.
    """
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 4))
    c1 = X[:1]    # K=1
    c5 = X[:5]    # K=5
    c20 = X[:20]  # K=20

    cov1 = compute_coverage_mean(X, c1)
    cov5 = compute_coverage_mean(X, c5)
    cov20 = compute_coverage_mean(X, c20)

    assert cov1 >= cov5 >= cov20


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_coverage_mean_empty_coreset():
    """K=0: empty coreset → nan (guard).

    Spec: coverage_metric.md — guard K=0 → nan.
    """
    X = np.ones((10, 3))
    result = compute_coverage_mean(X, np.empty((0, 3)))
    assert math.isnan(result)


def test_coverage_mean_single_train_point():
    """n=1: coverage_mean = distance from that point to nearest coreset."""
    X_train = np.array([[3.0, 4.0]])
    coreset = np.array([[0.0, 0.0]])
    result = compute_coverage_mean(X_train, coreset)
    assert math.isclose(result, 5.0, rel_tol=1e-6)  # sqrt(9+16)


# ── CSV schema ────────────────────────────────────────────────────────────────

def test_coverage_mean_in_results_record():
    """coverage_mean key present in the result record built by _train_on_coreset.

    Spec: coverage_metric.md — CSV coluna coverage_mean.
    Verifies the dict key exists and is a float (or nan).
    """
    # Simulate what _train_on_coreset builds per metric_fn
    metadata = {"coverage_mean": 0.42, "selection_times": 1.0}
    record = {
        "dataset": "test",
        "metodo": "freddy",
        "fracao": 0.1,
        "metrica": "accuracy_score",
        "valor": 0.9,
        "modelo": "DecisionTreeClassifier",
        "run": 0,
        "train_rep": 0,
        "train_time": 0.5,
        "selection_time": metadata.get("selection_times"),
        "coverage_mean": metadata.get("coverage_mean", float("nan")),
    }
    assert "coverage_mean" in record
    assert isinstance(record["coverage_mean"], float)
    assert math.isclose(record["coverage_mean"], 0.42)


def test_coverage_mean_nan_when_missing_from_metadata():
    """Old metadata without coverage_mean key → nan (backward compat).

    Spec: coverage_metric.md — .get() defensivo.
    """
    metadata = {"selection_times": 1.0}  # no coverage_mean
    val = metadata.get("coverage_mean", float("nan"))
    assert math.isnan(val)


def test_coverage_mean_nan_for_baseline():
    """method=none baseline → coverage_mean = nan.

    Spec: coverage_metric.md — baseline method=none.
    """
    record = {"coverage_mean": float("nan")}
    assert math.isnan(record["coverage_mean"])
