from .freddy import freddy
from .random import random_sampler
from .lazzy_greddy import craig_baseline
from .gradmatch import gradmatch

SAMPLERS = {
    "random": (random_sampler, {}),
    "freddy": (freddy, {"alpha": 0.15, "batch_size": 1000, "beta": 0.75}),
    "craig": (craig_baseline, {"b_size": 4000}),
    "gradmatch": (gradmatch, {"tol": 1e-4, "batch_size": 1024}),
}
