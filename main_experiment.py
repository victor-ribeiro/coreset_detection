from pathlib import Path

from sampling import freddy
from utiils.datasets import *
from utiils.sampling import collect

CONFIG = Path(".config")
CONFIG = {file.stem: file.resolve() for file in CONFIG.rglob("*.json")}
# name = "adult"
name = "covtype"
args = {
    "alpha": 0.1,
    "batch_size": 32,
    "K": 0.1,
}

collect(
    freddy,
    load_covtype_dataset,
    # load_adult_dataset,
    load_config(CONFIG[name]),
    runs=100,
    name=name,
    **args,
)
# print(out)
