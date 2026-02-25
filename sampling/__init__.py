from .freddy import kmeans_sampler, pmi_kmeans_sampler
from .random import random_sampler
from .lazzy_greddy import freddy, craig_baseline
from .gradmatch import gradmatch

SAMPLERS = {
    "random": (random_sampler, {}),
    "kmeans": (kmeans_sampler, {"alpha": 1, "tol": 10e-3, "max_iter": 500}),
    "pmi_kmeans": (pmi_kmeans_sampler, {"alpha": 1, "tol": 1, "max_iter": 500}),
    "freddy": (freddy, {"alpha": 0.15, "batch_size": 1000, "beta": 0.75}),
    "craig": (craig_baseline, {"b_size": 4000}),
    "gradmatch": (gradmatch, {"tol": 1e-4, "batch_size": 1024}),
}
