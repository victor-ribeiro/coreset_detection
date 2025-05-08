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


def entropy(x):
    x = np.abs(x)
    total = x.sum()
    p = x / total
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


# def utility_score(e, sset, /, acc=0, alpha=0.1, beta=1.1):
#     norm = 1 / _base_inc(alpha)
#     argmax = np.maximum(e, sset)
#     f_norm = alpha / (sset.sum() + acc + 1)
#     util = norm * math.log(1 + (argmax.sum()) * f_norm)
#     return util
def utility_score(e, sset, /, alpha=0.1, beta=1.1):
    norm = 1 / _base_inc(alpha)
    argmax = np.maximum(e, sset)
    f_norm = alpha / (sset.sum(axis=0) + 1)
    util = norm * np.log(1 + (argmax.sum(axis=0)) * f_norm)
    return util


@timeit
def freddy(
    dataset,
    base_inc=_base_inc,
    alpha=0.15,
    metric="similarity",
    K=1,
    batch_size=1000,
    beta=0.75,
    return_vals=False,
):
    # basic config
    base_inc = _base_inc(alpha)
    idx = np.arange(len(dataset))
    # dataset = dataset[idx]
    q = Queue()
    sset = []
    vals = []
    _ = [q.push(base_inc, (V, V % batch_size)) for V in range(len(dataset))]
    h_ = entropy(dataset)
    for ds, V in zip(
        batched(dataset, batch_size),
        batched(idx, batch_size),
    ):
        ds = np.array(ds)
        D = pairwise_distances(ds)
        D = D.max() - D * (h_ - entropy(dataset[sset]))
        localmax = np.amax(D, axis=1)
        score_s = utility_score(D, localmax, alpha=alpha, beta=beta)
        while q and len(sset) < K:

            score, idx_s = q.head
            inc = score_s - score
            if np.all(inc < 0) or (not q):
                # break
                continue
            score_t, idx_t = q.head
            if inc[idx_s[1]] > score_t:
                vals.append(score_s[idx_s[1]])
                sset.append(idx_s[0])
            else:
                q.push(inc[idx_s[1]], idx_s)
            q.push(score_t, idx_t)
        else:
            break

    np.random.shuffle(sset)
    if return_vals:
        return np.array(vals), sset
    return np.array(sset)
