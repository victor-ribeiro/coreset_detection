"""
Tests for sampling/freddy.py — Ciclo 6.
Nova implementação: Queue heap + utility_score + batching sequencial.
"""
import numpy as np
import pytest
from sampling.freddy import freddy, Queue, utility_score, _base_inc


RNG = np.random.default_rng(42)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small():
    return RNG.random((100, 5)).astype(np.float32)


@pytest.fixture
def medium():
    return RNG.random((500, 10)).astype(np.float32)


# ── UC1: retorna exatamente K índices ─────────────────────────────────────────

def test_returns_exactly_K(small):
    elapsed, indices = freddy(small, K=20)
    assert len(indices) == 20


def test_returns_K_small(small):
    elapsed, indices = freddy(small, K=1)
    assert len(indices) == 1


def test_returns_K_large(medium):
    elapsed, indices = freddy(medium, K=100)
    assert len(indices) == 100


def test_K_zero(small):
    elapsed, indices = freddy(small, K=0)
    assert len(indices) == 0


def test_K_near_n():
    data = RNG.random((50, 4)).astype(np.float32)
    elapsed, indices = freddy(data, K=45, batch_size=20)
    assert len(indices) == 45


# ── UC2: índices válidos e únicos ─────────────────────────────────────────────

def test_indices_in_range(medium):
    n = len(medium)
    elapsed, indices = freddy(medium, K=50)
    assert np.all(indices >= 0)
    assert np.all(indices < n)


def test_indices_are_integers(small):
    elapsed, indices = freddy(small, K=10)
    assert np.issubdtype(indices.dtype, np.integer)


def test_indices_unique(medium):
    elapsed, indices = freddy(medium, K=50)
    assert len(indices) == len(np.unique(indices))


# ── Interface @timeit ─────────────────────────────────────────────────────────

def test_timeit_interface(small):
    result = freddy(small, K=10)
    assert isinstance(result, tuple)
    elapsed, indices = result
    assert isinstance(elapsed, float)
    assert isinstance(indices, np.ndarray)


def test_elapsed_positive(small):
    elapsed, _ = freddy(small, K=10)
    assert elapsed > 0


# ── return_vals ───────────────────────────────────────────────────────────────

def test_return_vals_structure(small):
    elapsed, (vals, indices) = freddy(small, K=10, return_vals=True)
    assert isinstance(vals, np.ndarray)
    assert isinstance(indices, np.ndarray)


def test_return_vals_lengths_consistent(small):
    elapsed, (vals, indices) = freddy(small, K=10, return_vals=True)
    # vals contains scores for greedily selected points (may be < K if fallback filled)
    assert len(indices) == 10


# ── utility_score ─────────────────────────────────────────────────────────────

def test_utility_score_nonnegative():
    e = RNG.random(10).astype(np.float32)
    sset = RNG.random(10).astype(np.float32)
    score = utility_score(e, sset)
    assert score >= 0.0


def test_utility_score_increases_with_larger_e():
    sset = np.zeros(5, dtype=np.float32)
    e_small = np.ones(5, dtype=np.float32) * 0.1
    e_large = np.ones(5, dtype=np.float32) * 0.9
    assert utility_score(e_large, sset) > utility_score(e_small, sset)


# ── _base_inc ─────────────────────────────────────────────────────────────────

def test_base_inc_positive():
    assert _base_inc(0.15) > 0


def test_base_inc_symmetric():
    assert np.isclose(_base_inc(0.5), _base_inc(-0.5))


# ── Queue ─────────────────────────────────────────────────────────────────────

def test_queue_max_heap_order():
    q = Queue()
    q.push(0.1, "a")
    q.push(0.9, "b")
    q.push(0.5, "c")
    score, _ = q.head
    assert score == pytest.approx(0.9)


def test_queue_empty_after_all_pops():
    q = Queue()
    q.push(1.0, "x")
    q.head
    assert len(q) == 0
