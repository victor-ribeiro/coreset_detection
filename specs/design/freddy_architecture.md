# freddy.py — Architecture Document

## Pattern: Modular Monolith | KISS+YAGNI | SRP | Single-threaded

## Modules & Responsibilities

| Module | File | Responsibility |
|--------|------|---------------|
| Queue | freddy.py | Max-heap priority queue for lazy greedy selection |
| _base_inc | freddy.py | Compute log(1 + alpha) normalization constant |
| _estimate_marginal_gain | freddy.py | Estimate Δ̂_FL(e|S) via mini-batch B (Algorithm 1, Eq.5) |
| _update_coverage | freddy.py | Update global m_i ← max(m_i, s(x_i, x_e)) for x_i in B (Eq.7) |
| freddy | freddy.py | Main entry: while |S|<K, sample random B, run lazy greedy, return K indices |
| craig_baseline | lazzy_greddy.py | CRAIG coreset selection (unchanged) |

## Interfaces (Signatures & I/O)

```python
# Public API (UNCHANGED)
@timeit
def freddy(
    dataset: np.ndarray,   # shape (n, d), float32
    K: int,                # budget — exactly K indices returned
    alpha: float = 0.15,   # submodular gain parameter
    batch_size: int = 1000,# mini-batch size b
    beta: float = 0.75,    # reserved (not used in F_FL)
    return_vals: bool = False,
) -> np.ndarray            # shape (K,), dtype int — selected indices

# Internal helpers
def _estimate_marginal_gain(
    e_sim_row: np.ndarray,  # similarity row for candidate e (shape: batch_size,)
    m_i_batch: np.ndarray,  # current m_i for batch points (shape: batch_size,)
    n: int,                 # total dataset size (scaling factor n/|B|)
    b: int,                 # batch size |B|
) -> float                  # estimated marginal gain Δ̂_FL

def _update_coverage(
    m_i: np.ndarray,        # global coverage state (shape: n,), modified IN-PLACE
    batch_idx: np.ndarray,  # indices of batch points in dataset
    e_sim_row: np.ndarray,  # similarity s(x_i, x_e) for batch points
) -> None                   # modifies m_i in-place
```

## Dependencies

```
freddy.py
├── numpy (array ops, random.choice for mini-batch sampling)
├── sklearn.metrics.pairwise.pairwise_distances (euclidean distances)
├── heapq (internal Queue implementation)
├── math (log)
└── .utils.timeit (decorator — unchanged)
```

## Assumptions

1. `dataset` is pre-processed numpy array, dtype float32, shape (n, d)
2. Similarity s(x_i, x_e) = D.max() - D (inverted euclidean distance within batch)
3. `batch_size` << n (mini-batch is small relative to dataset)
4. m_i global state starts at 0.0 (no prior coverage)
5. Marginal gain scaling: n/|B| (unbiased estimator per Eq.5)
6. @timeit decorator wraps the function — returns (elapsed, result)

## Negative Scope (what freddy.py does NOT do)

- ICEL term (λ * u_e) — not implemented
- Parallelization
- Clustering or pseudo-labels
- Model training or gradient computation
- Modifying lazzy_greddy.py or __init__.py

## Algorithm 1 → Code Mapping

| Algorithm 1 step | Code |
|-----------------|------|
| Initialize S ← ∅ | `sset = []` |
| Initialize m_i ← 0 | `m_i = np.zeros(n)` |
| while |S| < K | `while len(sset) < K:` |
| Sample B ⊂ X randomly | `batch_idx = np.random.choice(n, size=b, replace=False)` |
| Init queue Q with B | `q = Queue(); for v in batch_idx: q.push(base_inc, ...)` |
| Extract e with highest priority | `score, idx_s = q.head` |
| Estimate Δ̂(e\|S) | `_estimate_marginal_gain(...)` |
| If highest gain → S ← S∪{e} | `sset.append(idx_s[0])` |
| Update m_i for B | `_update_coverage(m_i, batch_idx, ...)` |
| Else → update priority in Q | `q.push(inc, idx_s)` |
