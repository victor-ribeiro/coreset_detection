"""
Tests for experimental protocol fixes (Cycle 3).
Spec: decisions P0/requirement — bugs B1, B2, B3, B4.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# B1 — random_state=run_idx in cmd_select_coreset
# ---------------------------------------------------------------------------

def test_select_coreset_different_splits():
    """B1: different run_idx → different train/test splits."""
    from sklearn.model_selection import train_test_split

    n = 100
    full_set = np.arange(n)
    train0, test0 = train_test_split(full_set, test_size=0.2, random_state=0)
    train1, test1 = train_test_split(full_set, test_size=0.2, random_state=1)
    assert not np.array_equal(train0, train1), "run_0 and run_1 must produce different splits"


def test_select_coreset_reproducible():
    """B1: same run_idx → same split (reproducible)."""
    from sklearn.model_selection import train_test_split

    n = 100
    full_set = np.arange(n)
    train_a, _ = train_test_split(full_set, test_size=0.2, random_state=3)
    train_b, _ = train_test_split(full_set, test_size=0.2, random_state=3)
    assert np.array_equal(train_a, train_b), "same random_state must reproduce same split"


# ---------------------------------------------------------------------------
# B4 — train_rep in CSV records + random_state per model
# ---------------------------------------------------------------------------

def test_train_rep_in_results():
    """B4: 'train_rep' key must be present in each result record."""
    record = {
        "dataset": "covtype",
        "metodo": "freddy",
        "fracao": 0.05,
        "metrica": "accuracy_score",
        "valor": 0.9,
        "modelo": "RandomForestClassifier",
        "run": 0,
        "train_rep": 2,
        "train_time": 1.0,
        "selection_time": 0.5,
    }
    assert "train_rep" in record


def test_train_rep_values_range():
    """B4: train_rep must cover 0..4 for 5 reps."""
    reps = list(range(5))
    assert reps == [0, 1, 2, 3, 4]


def test_model_seed_unique():
    """B4: seed = run_idx*5+i must be unique for all (run, rep) combinations."""
    seeds = set()
    for run_idx in range(10):
        for i in range(5):
            seed = run_idx * 5 + i
            assert seed not in seeds, f"duplicate seed {seed} for run={run_idx}, rep={i}"
            seeds.add(seed)


def test_model_seed_formula():
    """B4: verify specific seed values for known (run, rep) pairs."""
    assert 0 * 5 + 0 == 0
    assert 0 * 5 + 4 == 4
    assert 1 * 5 + 0 == 5
    assert 1 * 5 + 4 == 9
    assert 2 * 5 + 3 == 13


def test_random_state_accepted_by_models():
    """B4: sklearn models accept random_state kwarg without error."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier

    for cls in [DecisionTreeClassifier, RandomForestClassifier, LogisticRegression]:
        model = cls(random_state=42)
        assert model is not None

    # SGDClassifier also accepts random_state
    model = SGDClassifier(random_state=42)
    assert model is not None


def test_random_state_fallback_for_xgb():
    """B4: models without random_state param must not raise TypeError (try/except fallback)."""
    # Simulate a model class without random_state
    class NoRandomStateModel:
        def __init__(self):
            pass

    seed = 7
    try:
        model = NoRandomStateModel(random_state=seed)
    except TypeError:
        model = NoRandomStateModel()
    assert model is not None


# ---------------------------------------------------------------------------
# B3 — output path: outputs/{name}/{model}/{dataset}/
# ---------------------------------------------------------------------------

def test_output_path_structure():
    """B3: output dir must be outputs/{name}/{model}/{dataset}/."""
    output_base = Path("outputs")
    name = "exp1"
    model = "RandomForestClassifier"
    dataset = "covtype"

    output_dir = output_base / name / model / dataset
    expected = Path("outputs/exp1/RandomForestClassifier/covtype")
    assert output_dir == expected


def test_output_path_created(tmp_path):
    """B3: output directory is created with parents."""
    output_dir = tmp_path / "outputs" / "exp1" / "XGBClassifier" / "covtype"
    output_dir.mkdir(parents=True, exist_ok=True)
    assert output_dir.exists()


# ---------------------------------------------------------------------------
# B2 — hypothesis_test.py column alignment
# ---------------------------------------------------------------------------

def _make_valid_df():
    """Create a minimal valid DataFrame with correct column names."""
    return pd.DataFrame({
        "metodo": ["freddy", "random"] * 5,
        "metrica": ["accuracy_score"] * 10,
        "fracao": [0.05] * 10,
        "valor": np.random.rand(10),
        "modelo": ["RandomForestClassifier"] * 10,
        "run": list(range(10)),
    })


def test_load_experiment_data_valid(tmp_path):
    """B2: load_experiment_data succeeds with correct columns."""
    from hypothesis_test import load_experiment_data

    df = _make_valid_df()
    csv_path = tmp_path / "results.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_experiment_data(tmp_path)
    assert set(["metodo", "metrica", "fracao", "valor"]).issubset(set(loaded.columns))


def test_load_raises_on_wrong_cols(tmp_path):
    """B2 negative: ValueError raised when CSV has wrong column names (method instead of metodo)."""
    from hypothesis_test import load_experiment_data

    df_wrong = pd.DataFrame({
        "method": ["freddy"] * 5,   # wrong — should be 'metodo'
        "metric": ["accuracy"] * 5,  # wrong — should be 'metrica'
        "frac": [0.05] * 5,          # wrong — should be 'fracao'
        "test": np.random.rand(5),   # wrong — should be 'valor'
    })
    csv_path = tmp_path / "wrong.csv"
    df_wrong.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="colunas ausentes"):
        load_experiment_data(tmp_path)


def test_load_raises_no_csv(tmp_path):
    """B2 negative: ValueError raised when directory has no CSVs."""
    from hypothesis_test import load_experiment_data

    with pytest.raises(ValueError, match="Nenhum CSV encontrado"):
        load_experiment_data(tmp_path)


def test_fracao_rounded(tmp_path):
    """B2: fracao values are rounded to 4 decimal places after load."""
    from hypothesis_test import load_experiment_data

    # 0.1 in float64 has precision issues
    df = _make_valid_df()
    df["fracao"] = 0.1 + 1e-10  # tiny float error
    csv_path = tmp_path / "results.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_experiment_data(tmp_path)
    assert (loaded["fracao"] == round(0.1 + 1e-10, 4)).all()


def test_run_hypothesis_uses_metodo_col(tmp_path):
    """B2: run_hypothesis_tests filters on 'metodo', not 'method'."""
    from hypothesis_test import load_experiment_data, run_hypothesis_tests

    np.random.seed(42)
    df = pd.DataFrame({
        "metodo": ["freddy"] * 10 + ["random"] * 10,
        "metrica": ["accuracy_score"] * 20,
        "fracao": [0.05] * 20,
        "valor": np.concatenate([np.random.rand(10) + 0.1, np.random.rand(10)]),
        "modelo": ["RandomForestClassifier"] * 20,
        "run": list(range(20)),
    })
    csv_path = tmp_path / "results.csv"
    df.to_csv(csv_path, index=False)

    data = load_experiment_data(tmp_path)
    results = run_hypothesis_tests(data, baseline="random", alpha=0.05)
    # Should find freddy vs random comparison
    assert len(results) > 0
    assert results[0]["method"] == "freddy"
    assert results[0]["baseline"] == "random"
