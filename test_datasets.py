"""
Tests for utiils/datasets.py — Cycle 4 preprocessing fixes.
Spec-driven: tests verify specs/technical/dataset_preprocessing.md.
"""
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from utiils.datasets import get_pipeline


# ── get_pipeline: covtype ─────────────────────────────────────────────────────

def test_covtype_pipeline_steps():
    """covtype pipeline must have normalize + pca steps (spec: dataset_preprocessing.md)."""
    p = get_pipeline("covtype")
    assert isinstance(p, Pipeline)
    step_names = [s[0] for s in p.steps]
    assert "normalize" in step_names
    assert "pca" in step_names


def test_covtype_pipeline_no_leakage():
    """Pipeline fit only on train — test transform uses train statistics (no leakage)."""
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((200, 54)).astype(np.float32)
    X_test = rng.standard_normal((50, 54)).astype(np.float32)

    p = get_pipeline("covtype")
    X_train_t = p.fit_transform(X_train)
    X_test_t = p.transform(X_test)

    # PCA(15) → output has 15 components
    assert X_train_t.shape == (200, 15)
    assert X_test_t.shape == (50, 15)


def test_covtype_pipeline_pca_random_state():
    """PCA in covtype pipeline must use random_state=42 for reproducibility."""
    p = get_pipeline("covtype")
    pca_step = dict(p.steps)["pca"]
    assert pca_step.random_state == 42


# ── get_pipeline: bike_share ──────────────────────────────────────────────────

def test_bike_share_pipeline_steps():
    """bike_share pipeline must contain ColumnTransformer."""
    p = get_pipeline("bike_share")
    assert isinstance(p, Pipeline)
    assert len(p.steps) == 1
    ct = p.steps[0][1]
    assert isinstance(ct, ColumnTransformer)


def test_bike_share_pipeline_scales_target_columns():
    """casual(col10) and registered(col11) must be scaled to [0,1] after pipeline.

    Spec: dataset_preprocessing.md — bike_share | MinMaxScaler on [casual, registered].
    """
    rng = np.random.default_rng(1)
    n = 100
    # Build array with 15 columns matching bike_share post-loader shape
    # cols 10,11 = casual, registered
    X = rng.standard_normal((n, 15))
    X[:, 10] = rng.integers(0, 500, size=n).astype(float)   # casual
    X[:, 11] = rng.integers(0, 5000, size=n).astype(float)  # registered

    p = get_pipeline("bike_share")
    X_t = p.fit_transform(X)

    # ColumnTransformer puts transformed cols first (indices 0 and 1 in output)
    assert X_t[:, 0].min() >= 0.0 - 1e-9
    assert X_t[:, 0].max() <= 1.0 + 1e-9
    assert X_t[:, 1].min() >= 0.0 - 1e-9
    assert X_t[:, 1].max() <= 1.0 + 1e-9


# ── get_pipeline: other datasets return Pipeline([]) ─────────────────────────

@pytest.mark.parametrize("name", ["adult", "sgemm", "hepmass", "predictmds", "storage_perf", "higgs"])
def test_empty_pipeline_for_no_transform_datasets(name):
    """Datasets without learned transforms return a passthrough pipeline (no-op)."""
    p = get_pipeline(name)
    assert isinstance(p, Pipeline)
    # Must be a no-op: fit_transform returns same values as input
    X = np.random.default_rng(3).standard_normal((20, 5))
    X_t = p.fit_transform(X)
    np.testing.assert_array_almost_equal(X_t, X)


def test_empty_pipeline_is_noop():
    """Passthrough pipeline must be a no-op: fit_transform(X) == X."""
    X = np.random.default_rng(2).standard_normal((50, 10))
    p = get_pipeline("adult")
    X_t = p.fit_transform(X)
    np.testing.assert_array_almost_equal(X_t, X)


# ── Negative: pipeline NOT called inside load_dataset ────────────────────────

def test_get_pipeline_is_separate_from_load_dataset():
    """get_pipeline must be a separate function — load_dataset must NOT apply transforms.

    This verifies the architecture decision: no leakage by design.
    Spec: dataset_preprocessing.md — Interface Contract.
    """
    import inspect
    from utiils import datasets as ds_module

    src = inspect.getsource(ds_module.load_dataset)
    # load_dataset must not call get_pipeline or Pipeline internally
    assert "get_pipeline" not in src
    assert "fit_transform" not in src
    assert "Pipeline" not in src


# ── sgemm target shape ────────────────────────────────────────────────────────

def test_sgemm_target_is_scalar():
    """sgemm target must be 1D (mean of 4 runs), not (n,4).
    Spec: dataset_preprocessing.md — sgemm | mean(Run1..Run4).
    """
    import pandas as pd
    from io import StringIO
    from utiils.datasets import load_sgemm_dataset

    # Minimal synthetic CSV matching sgemm format (14 features + 4 run cols)
    header = "idx,MWG,NWG,KWG,MDIMC,NDIMC,MDIMA,NDIMB,KWI,VWM,VWN,STRM,STRN,SA,SB,Run1 (ms),Run2 (ms),Run3 (ms),Run4 (ms)"
    rows = "\n".join(f"0,{i},{i},{i},{i},{i},{i},{i},{i},{i},{i},0,0,0,0,{i*1.0},{i*2.0},{i*3.0},{i*4.0}" for i in range(1, 6))
    csv_text = f"{header}\n{rows}"

    from pathlib import Path
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_text)
        tmp_path = f.name

    try:
        X, y = load_sgemm_dataset({"root": tmp_path})
        assert y.ndim == 1, f"Expected 1D target, got shape {y.shape}"
        assert X.shape[1] == 14
        # mean of [1,2,3,4] = 2.5 for row i=1
        expected_mean = (1.0 + 2.0 + 3.0 + 4.0) / 4
        assert np.isclose(y[0], expected_mean), f"y[0]={y[0]}, expected {expected_mean}"
    finally:
        os.unlink(tmp_path)
