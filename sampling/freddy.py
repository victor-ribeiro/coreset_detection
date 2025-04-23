import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import BisectingKMeans

from .utils import timeit


def entropy(x):
    x = np.abs(x)
    total = x.sum()
    p = x / total
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


def _n_cluster(dataset, alpha=1, max_iter=100, tol=10e-2):
    val = np.zeros(max_iter)
    base = np.log(1 + alpha)
    for idx, n in enumerate(range(max_iter)):
        # print(val)
        sampler = BisectingKMeans(n_clusters=n + 2)
        sampler.fit(dataset)
        if val[:idx].sum() == 0:

            val[idx] = np.log(1 + sampler.inertia_ * alpha / base)
            continue

        val[idx] = np.log(1 + sampler.inertia_ * alpha / val[val > 0].max() / base)

        if abs(val[:idx].min() - val[idx]) < tol:
            return sampler.inertia_, sampler.cluster_centers_
    # return sampler.cluster_centers_
    return ValueError("Does not converge")


@timeit
def kmeans_sampler(dataset, K, alpha=1, tol=10e-3, max_iter=500):
    kmeans = BisectingKMeans(n_clusters=10)
    kmeans.fit(dataset)
    clusters = kmeans.cluster_centers_
    base = np.log(1 + alpha)
    print(f"Found {len(clusters)} clusters, tol: {tol}")
    dist = pairwise_distances(dataset, clusters)
    dist -= np.amax(dist, axis=0)
    dist = np.abs(dist).sum(axis=1)
    sset = np.argsort(dist, kind="heapsort")[::-1]
    return sset[:K]


@timeit
def pmi_kmeans_sampler(
    dataset,
    K,
    alpha=1,
    tol=1,
    max_iter=500,
):
    _, clusters = _n_cluster(dataset, alpha=alpha, max_iter=max_iter, tol=tol)
    print(f"Found {len(clusters)} clusters, tol: {tol}")
    dist = pairwise_distances(clusters, dataset)
    softmax = np.exp(dist - dist.max())
    softmax /= dist.sum()
    h_c = entropy(clusters)
    h_p = entropy(dataset)
    pmi = np.log2(K)

    pmi = ((1 - softmax) / (dist * (h_p - h_c))).sum(axis=0)
    # sset = np.argsort(pmi, kind="heapsort")
    sset = np.argsort(pmi, kind="heapsort")[::-1]

    return sset[:K]
