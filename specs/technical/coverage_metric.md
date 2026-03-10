# Coverage Metric — Technical Specification

## Definition

coverage_mean(S, X_train) = (1/n) * Σ_{x_i ∈ X_train} min_{c_j ∈ S} d(x_i, c_j)

Where:
- S = coreset (selected subset), |S| = K
- X_train = full training set, |X_train| = n
- d(·,·) = Euclidean distance (L2)

Interpretation: average distance from each training point to its nearest coreset point.
Lower = better coverage (coreset is more representative of training space).

## Reference

Classical k-center coverage metric. Related to:
- Har-Peled, S. & Mazumdar, S. (2004). "On coresets for k-means and k-median clustering."
  STOC 2004. https://doi.org/10.1145/1007352.1007400
- Feldman, D. & Langberg, M. (2011). "A unified framework for approximating and clustering data."
  STOC 2011.

## Implementation

```python
from sklearn.neighbors import NearestNeighbors

def coverage_mean(X_train: np.ndarray, coreset_feat: np.ndarray) -> float:
    """Mean distance from each train point to its nearest coreset neighbor.

    Parameters
    ----------
    X_train : np.ndarray, shape (n, d)
        Full training set (pipeline-transformed).
    coreset_feat : np.ndarray, shape (K, d)
        Selected coreset points (pipeline-transformed).

    Returns
    -------
    float
        Mean coverage radius. Lower = better.
    """
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(coreset_feat)
    distances, _ = nn.kneighbors(X_train)
    return float(distances.mean())
```

Complexity: O(K·d) fit + O(n·log(K)·d) query (ball_tree/kd_tree for d ≤ ~20; brute for high-d).

## Integration in main.py

Computed in `cmd_select_coreset` after coreset selection:
```python
coreset_feat = train_feat[indices]
cov = coverage_mean(train_feat, coreset_feat)
metadata["coverage_mean"] = cov
```

Propagated in `_train_on_coreset` as a per-run column in CSV:
```python
"coverage_mean": metadata.get("coverage_mean", float("nan"))
```

Baseline (method="none"): `coverage_mean = float("nan")` — no coreset selected.

## Expected Behavior

- Monotone: coverage_mean(S_K) ≥ coverage_mean(S_{K'}) when K < K' (more points → smaller radius)
- coverage_mean = 0 when K = n (full dataset selected as coreset)
- coverage_mean > 0 when K < n
