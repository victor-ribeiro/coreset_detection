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
    if x.ndim == 1:
        total = x.sum()
        if total == 0:
            return 0.0
        p = x / total
        p = p[p > 0]
        return -(p * np.log2(p)).sum()
    else:
        # Vetorizado: entropia por coluna (feature)
        totals = x.sum(axis=0)
        totals[totals == 0] = 1  # Evitar divisão por zero
        p = x / totals
        h = np.where(p > 0, -p * np.log2(p + 1e-10), 0).sum(axis=0)
        return h.mean()


def utility_score(e, sset, /, acc=0, alpha=0.1):
    norm = 1 / _base_inc(alpha)
    argmax = np.maximum(e, sset)
    f_norm = alpha / (sset.sum() + 1)
    util = norm * math.log(1 + (argmax.sum()) * f_norm)
    return util


# Garantia teórica para funções submodulares: (1 - 1/e) ≈ 0.632
SUBMODULAR_GUARANTEE = 1 - 1 / math.e


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
    opt_threshold=SUBMODULAR_GUARANTEE,
):
    # basic config
    base_inc = _base_inc(alpha)
    idx = np.arange(len(dataset))
    dataset = dataset[idx].astype(np.float32)
    q = Queue()
    sset = []
    vals = []

    # Variáveis para early stopping por fração do ótimo
    f_current = 0.0
    f_opt_estimate = 0.0

    argmax = 0
    for ds, V in zip(
        batched(dataset, batch_size),
        batched(idx, batch_size),
    ):
        for v in V:
            q.push(base_inc, (v, v % batch_size))

        ds = np.asarray(ds)
        D = pairwise_distances(ds, metric="euclidean")
        D = D.max(axis=1, keepdims=True) - D
        localmax = np.amax(D, axis=1)
        argmax += localmax.sum()

        # Estimar upper bound do ótimo
        f_opt_estimate += localmax.sum()

        while q and len(sset) < K:
            score, idx_s = q.head
            s = D[idx_s[1]]
            score_s = utility_score(s, localmax, acc=argmax, alpha=alpha)
            inc = score_s - score
            if inc < 0:
                # q.push(inc, idx_s)
                continue
            if not q:
                break
            score_t, idx_t = q.head
            if inc > score_t:
                vals.append(score_s)
                sset.append(idx_s[0])
                f_current += inc

                # Early stopping: atingiu fração do ótimo
                if f_opt_estimate > 0 and f_current >= opt_threshold * f_opt_estimate:
                    break
            else:
                q.push(inc, idx_s)
            q.push(score_t, idx_t)
        else:
            break

        if f_opt_estimate > 0 and f_current >= opt_threshold * f_opt_estimate:
            break

    if return_vals:
        return np.array(vals), sset
    return np.array(sset)
