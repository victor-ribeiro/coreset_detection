from time import perf_counter
from functools import wraps


def timeit(f_):
    @wraps(f_)
    def inner(*args, **kwargs):
        init_time = perf_counter()
        out = f_(*args, **kwargs)
        end_time = perf_counter()
        return end_time - init_time, out

    return inner
