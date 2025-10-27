# %%
# [0, 1, 2, 3, 4, ...]


# [
#     [[0, 1, 2], [3]],
#     [[1, 2, 3], [4]],
#     ...,
# ]

import numpy as np

idx = np.arange(100)
idx


# %%
def calc_idx(datetime_col):
    pass


def get_contiguos():
    # calc_idx
    #
    pass


def janelament():
    pass


idx = np.random.randint(1, 10, 100)
idx.sort()
# get_contiguos() -> janelament()


# %%
import numpy as np


def diff(col, i) -> bool:
    if i:
        return col[i] - col[i - 1]
    return 0


timestamp = [1, 1, 1, 4, 4, 4, 4, 4, 6, 6, 10, 12, 22, 22]
# timestamp = np.array(timestamp)
cont = map(lambda i: diff(timestamp, i[0]), enumerate(timestamp))
# indices = list(map(lambda x: x[0], filter(lambda x: x[1] != 0, enumerate(cont))))
# indices = map(lambda x: x[0], filter(lambda x: x[1] != 0, enumerate(cont)))
# BAGARITO [0, 3, 8, 10, 11, 12]
indices = map(
    lambda x: x[0],
    filter(lambda x: x[1] != 0 if x[0] else True, enumerate(cont)),
)
from functools import reduce

pairs, _ = reduce(
    lambda acc, curr: (acc[0] + [(acc[1], curr)], curr), indices, ([], next(indices))
)
blocks = map(lambda x: slice(*x), pairs)
blocks = map(lambda x: timestamp[x], blocks)

unique = reduce(lambda x, y: len(x), blocks)
unique
# %%
