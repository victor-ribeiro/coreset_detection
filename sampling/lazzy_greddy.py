import heapq
import math
import numpy as np
import multiprocessing as mp
from itertools import batched
from sklearn.metrics.pairwise import pairwise_distances

from .craig.lazy_greedy import FacilityLocation, lazy_greedy_heap
from .utils import timeit

N_JOBS = mp.cpu_count() - 1

# b_sise = 1024
# Run 1/10 - Sample 1/5
# Select time: 15.88 seconds
# Training time: 0.06 seconds
# Run 2/10 - Sample 1/5
# Select time: 16.46 seconds
# Training time: 0.05 seconds
# Run 3/10 - Sample 1/5
# Select time: 13.79 seconds
# Training time: 0.04 seconds
# Run 4/10 - Sample 1/5
# Select time: 14.50 seconds
# Training time: 0.04 seconds
# Run 5/10 - Sample 1/5
# Select time: 13.05 seconds
# Training time: 0.06 seconds
# Run 6/10 - Sample 1/5
# Select time: 16.07 seconds
# Training time: 0.04 seconds
# Run 7/10 - Sample 1/5
# Select time: 14.26 seconds
# Training time: 0.04 seconds
# Run 8/10 - Sample 1/5
# Select time: 15.38 seconds
# Training time: 0.04 second
####################################
# b_sise = 32


@timeit
def craig_baseline(data, K, b_size=4000):
    features = data.astype(np.single)
    V = np.arange(len(features), dtype=int).reshape(-1, 1)
    start = 0
    end = start + b_size
    sset = []
    for ds in batched(features, b_size):
        ds = np.array(ds)
        D = pairwise_distances(features, ds, metric="euclidean", n_jobs=int(N_JOBS / 2))
        # D = pairwise_distances(ds, features, metric="euclidean", n_jobs=2)
        v = V[start:end]
        D = D.max() - D
        B = int(len(D) * (K / len(features)))
        locator = FacilityLocation(D=D, V=v)
        sset_idx, *_ = lazy_greedy_heap(F=locator, V=v, B=B)
        sset_idx = np.array(sset_idx, dtype=int).reshape(1, -1)[0]
        sset.append(sset_idx)
        start += b_size
        end += b_size
    sset = np.hstack(sset)
    return sset


REDUCE = {"mean": np.mean, "sum": np.sum}


class Queue(list):
    def __init__(self, *iterable):
        super().__init__(*iterable)
        heapq._heapify_max(self)

    def append(self, item: "Any"):
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

    def push(self, idx, score):
        item = (idx, score)
        self.append(item)


def _base_inc(alpha=1):
    alpha = abs(alpha)
    return math.log(1 + alpha)


def utility_score(e, sset, /, acc=0, alpha=0.1, beta=1.1):
    gamma = (alpha + beta) / 2
    norm = 1 / _base_inc(alpha)
    argmax = np.maximum(e, sset)
    f_norm = alpha / (sset.sum() + acc + 1)
    util = norm * math.log(1 + (argmax.sum() + acc) * f_norm)
    return util + (math.log(1 + ((sset.sum() + acc) ** gamma)) * beta)


@timeit
def freddy(
    dataset,
    base_inc=_base_inc,
    alpha=0.15,
    metric="similarity",
    K=1,
    batch_size=256,
    beta=0.75,
    return_vals=False,
):
    # basic config
    base_inc = _base_inc(alpha)
    idx = np.arange(len(dataset))
    # idx = np.random.permutation(idx)
    q = Queue()
    sset = []
    vals = []
    argmax = 0
    inc = 0
    for ds, V in zip(
        batched(dataset, batch_size),
        batched(idx, batch_size),
    ):
        D = pairwise_distances(ds, metric="euclidean")
        D = D.max(axis=1) - D
        size = len(D)
        # localmax = np.median(D, axis=0)
        localmax = np.amax(D, axis=1)
        argmax += localmax.sum()
        _ = [q.push(base_inc, i) for i in zip(V, range(size))]
        while q and len(sset) < K:
            score, idx_s = q.head
            s = D[:, idx_s[1]]
            score_s = utility_score(s, localmax, acc=argmax, alpha=alpha, beta=beta)
            inc = score_s - score
            if (inc < 0) or (not q):
                break
            score_t, idx_t = q.head
            if inc > score_t:
                score = utility_score(s, localmax, acc=argmax, alpha=alpha, beta=beta)
                localmax = np.maximum(localmax, s)
                sset.append(idx_s[0])
                vals.append(score)
            else:
                q.push(inc, idx_s)
            q.push(score_t, idx_t)
    np.random.shuffle(sset)
    if return_vals:
        return np.array(vals), sset
    return sset
