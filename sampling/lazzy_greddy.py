import heapq
import math
import numpy as np
import multiprocessing as mp
from itertools import batched
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import BisectingKMeans

from .craig.lazy_greedy import FacilityLocation, lazy_greedy_heap
from .utils import timeit

# N_JOBS = mp.cpu_count() - 1
N_JOBS = 4


@timeit
def craig_baseline(data, K, b_size=4000):
    features = data.astype(np.single)
    idx = np.arange(len(features), dtype=int)
    start = 0
    end = start + b_size
    sset = []
    ds = batched(features, b_size)
    ds = map(np.array, ds)
    D = map(lambda x: pairwise_distances(x, features, n_jobs=N_JOBS), ds)
    D = map(lambda x: np.max(x) - x, D)
    V = batched(idx, b_size)
    locator = map(
        lambda d, v: FacilityLocation(D=d, V=np.array(v).reshape(-1, 1)), D, V
    )
    V = batched(idx, b_size)
    sset = map(
        lambda loc, v: lazy_greedy_heap(
            F=loc, V=np.array(v), B=int(len(v) * (K / len(features)))
        ),
        locator,
        batched(idx, b_size),
    )
    sset = (np.array(s) for s, _ in sset)
    sset = np.hstack([*sset])
    # sset = np.fromiter(iter(sset), dtype=int)
    return sset
