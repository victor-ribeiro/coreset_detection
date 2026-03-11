import heapq
import math
import numpy as np
from itertools import batched
from sklearn.metrics.pairwise import pairwise_distances
from .utils import timeit


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


def utility_score(e, sset, /, acc=0, alpha=0.1):
    norm = 1 / _base_inc(alpha)
    argmax = np.maximum(e, sset)
    util = norm * math.log(1 + (argmax.sum()))
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
    dataset = dataset[idx].astype(np.float32)
    sset = []
    vals = []

    argmax = 0
    q = Queue()
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

        while q and len(sset) < K:
            score, idx_s = q.head
            s = D[idx_s[1]]
            score_s = utility_score(s, localmax, acc=argmax, alpha=alpha)
            inc = score_s - score
            if inc < 0:
                q.push(inc, idx_s)
                break
            if not q:
                break
            score_t, idx_t = q.head
            if inc > score_t:
                vals.append(score_s)
                sset.append(idx_s[0])
            else:
                q.push(inc, idx_s)
            q.push(score_t, idx_t)

    if return_vals:
        return np.array(vals), sset
    return np.array(sset)
