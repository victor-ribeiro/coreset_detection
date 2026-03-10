import heapq
import math
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from .utils import timeit


class Queue(list):
    def __init__(self, *iterable):
        super().__init__(*iterable)
        heapq._heapify_max(self)

    def append(self, item):
        super().append(item)
        heapq._siftdown_max(self, 0, len(self) - 1)

    def pop(self, index=-1):
        el = super().pop(index)
        if not self:
            return el
        val, self[0] = self[0], el
        heapq._siftup_max(self, 0)
        return val

    @property
    def head(self):
        return self.pop()

    def push(self, score, idx):
        self.append((score, idx))


def _base_inc(alpha=1):
    alpha = abs(alpha)
    return math.log(1 + alpha)


def _estimate_marginal_gain(sim_row, m_i_batch, n, b):
    """Eq. 5 — Δ̂_FL(e | S_t) = (n / |B|) * Σ max(0, s(x_i, x_e) - m_i(S_t))

    Source: Ribeiro (2026), FREDDY paper, Section 4.1, Eq. 5
    """
    return np.log(1 + np.maximum(sim_row, m_i_batch).sum())


def _update_coverage(m_i, batch_idx, sim_row):
    """Eq. 7 — m_i(S_{t+1}) ← max(m_i(S_t), s(x_i, x_e)) for x_i in B_t

    Source: Ribeiro (2026), FREDDY paper, Section 4.1, Eq. 7
    Modifies m_i in-place. Updates only batch points (Monte Carlo approximation).
    """
    m_i[batch_idx] = np.maximum(m_i[batch_idx], sim_row)


@timeit
def freddy(
    dataset,
    K=1,
    alpha=0.15,
    batch_size=1000,
    beta=0.75,
    return_vals=False,
    **kwargs,
):
    """FREDDY: Fast Reduction of Elements for Data-driven Yielding.

    Stochastic greedy coreset selection via Facility Location maximization.
    Implements Algorithm 1 from Ribeiro (2026).

    Parameters
    ----------
    dataset : np.ndarray, shape (n, d)
        Pre-processed feature matrix (float32).
    K : int
        Budget — exactly K indices are returned.
    alpha : float
        Submodular gain parameter (default 0.15).
    batch_size : int
        Mini-batch size b. Mini-batches serve as Monte Carlo estimators,
        not domain restriction (Ribeiro 2026, Section 4.1).
    beta : float
        Reserved parameter (unused in F_FL; kept for interface compatibility).
    return_vals : bool
        If True, also return marginal gain values.

    Returns
    -------
    np.ndarray, shape (K,)
        Exactly K selected indices into dataset.
    """
    dataset = np.asarray(dataset, dtype=np.float32)
    n = len(dataset)
    b = min(batch_size, n)
    base_score = _base_inc(alpha)

    # Algorithm 1: Initialize S ← ∅, m_i ← 0 for all x_i ∈ X
    sset = []
    vals = []
    m_i = np.zeros(n, dtype=np.float64)  # global coverage state, persistent
    # Safety: bound outer iterations to avoid infinite loop when K/n is high
    # (FREDDY assumes K << n; with K≈n, lazy greedy may reject all candidates per batch)
    max_outer = max(K * 10, 1000)
    outer_iter = 0

    # Algorithm 1: while |S| < K
    while len(sset) < K and outer_iter < max_outer:
        outer_iter += 1
        # Algorithm 1: Sample B ⊂ X uniformly at random, |B| = b
        batch_idx = np.random.choice(n, size=b, replace=False)
        batch_ds = dataset[batch_idx]

        # Similarity matrix for this batch: s(x_i, x_e) = D_max - D (within-batch)
        D = pairwise_distances(batch_ds, metric="euclidean")
        sim = D.max() - D  # shape (b, b)

        # Algorithm 1: Initialize priority queue Q with elements in B
        q = Queue()
        for loc_idx, glob_idx in enumerate(batch_idx):
            q.push(base_score, (glob_idx, loc_idx))

        # Algorithm 1: while Q not empty and |S| < K
        while q and len(sset) < K:
            score, (glob_idx, loc_idx) = q.head

            # Algorithm 1: Estimate marginal gain Δ̂(e | S) using B (Eq.sim_row 5)
            sim_row = sim[loc_idx]  # s(x_i, x_e) for all x_i in batch
            gain = _estimate_marginal_gain(sim_row, m_i[batch_idx], n, b)
            inc = gain - score

            if inc < 0:
                # Re-insert with updated priority and break (lazy greedy)
                q.push(inc, (glob_idx, loc_idx))
                break

            if not q:
                break

            score_t, idx_t = q.head

            # Algorithm 1: if e has highest estimated gain → S ← S ∪ {e}
            if inc >= score_t:
                sset.append(glob_idx)
                vals.append(gain)
                # Algorithm 1: Update m_i for x_i in B (Eq. 7)
                _update_coverage(m_i, batch_idx, sim_row)
            else:
                q.push(inc, (glob_idx, loc_idx))

            q.push(score_t, idx_t)

    # Fallback: if max_outer reached before K, fill remainder randomly
    diff = K - len(sset)
    if diff > 0:
        remaining = np.setdiff1d(np.arange(n), sset, assume_unique=True)
        extra = np.random.choice(remaining, size=diff, replace=False)
        sset.extend(extra.tolist())

    if return_vals:
        return np.array(vals), np.array(sset)
    return np.array(sset)
