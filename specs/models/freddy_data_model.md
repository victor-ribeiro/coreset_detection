# FREDDY — Data Model

## Inputs

| Variable | Type | Shape | Description |
|----------|------|-------|-------------|
| dataset | np.ndarray float32 | (n, d) | Training features, pre-processed |
| K | int | — | Budget: number of elements to select |
| alpha | float | — | Submodular gain parameter (default 0.15) |
| batch_size | int | — | Mini-batch size b (default 1000) |

## Internal State

| Variable | Type | Shape | Description |
|----------|------|-------|-------------|
| m_i | np.ndarray float64 | (n,) | Global coverage state, initialized to 0 |
| sset | list[int] | (K,) | Selected indices, grows until len==K |
| q | Queue | — | Max-heap priority queue for lazy greedy |

## Output

| Variable | Type | Shape | Description |
|----------|------|-------|-------------|
| indices | np.ndarray int | (K,) | Exactly K selected indices into dataset |

## Similarity Function

s(x_i, x_e) = D_max - euclidean_distance(x_i, x_e)

where D_max = max pairwise distance within the current mini-batch.
This is a within-batch normalization (approximation of global similarity).
