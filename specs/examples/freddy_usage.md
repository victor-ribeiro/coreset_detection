# FREDDY — Usage Examples

## Basic usage (via SAMPLERS registry)

```python
from sampling import SAMPLERS

sampler_fn, defaults = SAMPLERS["freddy"]
# defaults: {"alpha": 0.15, "batch_size": 1000, "beta": 0.75}

elapsed, indices = sampler_fn(train_feat, K=500)
# indices: np.ndarray shape (500,), dtype int
# elapsed: float, seconds
```

## Direct call

```python
from sampling.freddy import freddy

elapsed, indices = freddy(dataset, K=1000, alpha=0.15, batch_size=512)
assert len(indices) == 1000  # guaranteed post-fix
```

## Expected behavior (Algorithm 1)

- K=1000, n=50000, batch_size=512:
  - Outer loop runs until exactly 1000 elements selected
  - Each iteration: random sample 512 from 50000
  - m_i[50000] updated stochastically across iterations
  - Result: exactly 1000 indices (no more, no less)
