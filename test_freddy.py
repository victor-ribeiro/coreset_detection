"""
Tests for sampling/freddy.py — FREDDY Algorithm 1 (Ribeiro 2026).
Spec-driven: tests verify specs, not implementation.
"""
import numpy as np
import pytest
from sampling.freddy import freddy, _estimate_marginal_gain, _update_coverage


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
    """Algorithm 1 must return exactly K elements (outer while |S| < K)."""
    elapsed, indices = freddy(small, K=20)
    assert len(indices) == 20


def test_returns_K_small(small):
    elapsed, indices = freddy(small, K=1)
    assert len(indices) == 1


def test_returns_K_large(medium):
    elapsed, indices = freddy(medium, K=100)
    assert len(indices) == 100


def test_K_zero(small):
    """K=0 → empty selection."""
    elapsed, indices = freddy(small, K=0)
    assert len(indices) == 0


def test_K_near_n():
    """K close to n — fallback fills remainder randomly if max_outer reached."""
    data = RNG.random((50, 4)).astype(np.float32)
    elapsed, indices = freddy(data, K=45, batch_size=20)
    assert len(indices) == 45


# ── UC2: índices válidos e únicos ─────────────────────────────────────────────

def test_indices_in_range(medium):
    """All selected indices must be in [0, n)."""
    n = len(medium)
    elapsed, indices = freddy(medium, K=50)
    assert np.all(indices >= 0)
    assert np.all(indices < n)


def test_indices_are_integers(small):
    """Indices must be integer dtype."""
    elapsed, indices = freddy(small, K=10)
    assert np.issubdtype(indices.dtype, np.integer)


def test_indices_unique(medium):
    """Selected indices must be unique — no element selected twice."""
    elapsed, indices = freddy(medium, K=50)
    assert len(indices) == len(np.unique(indices))


# ── Interface @timeit ─────────────────────────────────────────────────────────

def test_timeit_interface(small):
    """@timeit wraps freddy: must return (elapsed: float, indices: ndarray)."""
    result = freddy(small, K=10)
    assert isinstance(result, tuple)
    elapsed, indices = result
    assert isinstance(elapsed, float)
    assert isinstance(indices, np.ndarray)


def test_elapsed_positive(small):
    elapsed, _ = freddy(small, K=10)
    assert elapsed > 0


# ── Eq.5: _estimate_marginal_gain ────────────────────────────────────────────

def test_marginal_gain_scaling():
    """Eq.5 + log-modulation: gain = log(1 + (n/b) * Σmax(0, s-m_i))."""
    sim_row = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    m_i_batch = np.zeros(3, dtype=np.float64)
    n, b = 300, 3
    gain = _estimate_marginal_gain(sim_row, m_i_batch, n, b)
    expected = np.log(1 + (n / b) * sim_row.sum())  # m_i=0 → max(0, s-0) = s
    assert np.isclose(gain, expected)


def test_marginal_gain_zero_when_covered():
    """Gain = 0 when m_i already covers the candidate (Eq.5 max(0, s-m_i))."""
    sim_row = np.array([0.5, 0.3], dtype=np.float64)
    m_i_batch = np.array([0.9, 0.9], dtype=np.float64)  # already covered
    gain = _estimate_marginal_gain(sim_row, m_i_batch, n=100, b=2)
    assert gain == 0.0


def test_marginal_gain_nonnegative():
    """Marginal gain must always be >= 0."""
    sim_row = RNG.random(10).astype(np.float64)
    m_i_batch = RNG.random(10).astype(np.float64)
    gain = _estimate_marginal_gain(sim_row, m_i_batch, n=500, b=10)
    assert gain >= 0.0


# ── Eq.7: _update_coverage ───────────────────────────────────────────────────

def test_coverage_update_inplace():
    """Eq.7: m_i[batch_idx] = max(m_i, sim_row) — in-place update."""
    m_i = np.zeros(10, dtype=np.float64)
    batch_idx = np.array([0, 3, 7])
    sim_row = np.array([0.6, 0.4, 0.8])
    _update_coverage(m_i, batch_idx, sim_row)
    assert np.isclose(m_i[0], 0.6)
    assert np.isclose(m_i[3], 0.4)
    assert np.isclose(m_i[7], 0.8)
    # Non-batch points unchanged
    assert m_i[1] == 0.0
    assert m_i[5] == 0.0


def test_coverage_update_monotone():
    """Eq.7: coverage state is monotone non-decreasing."""
    m_i = np.array([0.5, 0.3, 0.1], dtype=np.float64)
    batch_idx = np.array([0, 1, 2])
    sim_row = np.array([0.3, 0.3, 0.9])  # first two are lower → no decrease
    _update_coverage(m_i, batch_idx, sim_row)
    assert m_i[0] == 0.5   # 0.5 > 0.3 → unchanged
    assert m_i[1] == 0.3   # equal
    assert m_i[2] == 0.9   # 0.9 > 0.1 → updated


# ── return_vals flag ──────────────────────────────────────────────────────────

def test_return_vals(small):
    """return_vals=True returns (vals, indices) — both as ndarray."""
    elapsed, (vals, indices) = freddy(small, K=10, return_vals=True)
    assert isinstance(vals, np.ndarray)
    assert isinstance(indices, np.ndarray)
    assert len(vals) == len(indices) == 10
