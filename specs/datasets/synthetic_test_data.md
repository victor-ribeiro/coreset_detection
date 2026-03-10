# Synthetic Test Data — FREDDY

Generated for Phase 6 unit tests. No real dataset required for correctness tests.

## Dataset specs

| Name | n | d | dtype | Purpose |
|------|---|---|-------|---------|
| small | 100 | 5 | float32 | Basic correctness |
| medium | 500 | 10 | float32 | K variety |
| single | 1 | 3 | float32 | Edge case K=1 |
| K_equals_n | 50 | 4 | float32 | K=n edge case |

## Generation

```python
np.random.seed(42)
small = np.random.rand(100, 5).astype(np.float32)
medium = np.random.rand(500, 10).astype(np.float32)
```
