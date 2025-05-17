import numpy as np
from .utils import timeit


@timeit
def random_sampler(data, K):
    size = len(data)
    rng = np.random.default_rng(42)
    sset = rng.integers(0, size, size=K, dtype=int)
    return sset
