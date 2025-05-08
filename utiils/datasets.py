from pathlib import Path
from sklearn.preprocessing import (
    OrdinalEncoder,
    minmax_scale,
    normalize,
    LabelBinarizer,
    StandardScaler,
)

import numpy as np
import pandas as pd
import json

ROOT = Path(__file__).resolve(strict=True).parent.parent
DATA_ROOT = ROOT / "data"

DATASETS = {}


def register(f_):
    DATASETS[f_.__name__] = f_
    return f_


def load_config(path):
    """
    Load the configuration file from the specified path.
    """
    with open(path, "r") as file:
        config = json.load(file)
    return config


@register
def load_adult_dataset(config):
    path = config["root"]
    path = DATA_ROOT / Path(path)
    dataset = pd.read_csv(path)
    dataset[config["target"]] = dataset[config["target"]].map({">50K": 1, "<=50K": 0})
    dataset.replace(" ?", np.nan, inplace=True)
    dataset = pd.get_dummies(
        dataset,
        columns=[
            "gender",
            "education",
            "race",
            "relationship",
            "workclass",
            "marital-status",
            "occupation",
            "native-country",
        ],
        drop_first=False,
        dtype=int,
    )
    target = dataset.pop(config["target"])
    return dataset.values, target.values


@register
def load_bike_share_dataset(config):
    path = config["root"]
    path = DATA_ROOT / Path(path)
    names = [
        "dteday",
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "casual",
        "registered",
        "cnt",
    ]

    dataset = pd.read_csv(path, names=names, engine="pyarrow", skiprows=1, index_col=0)
    ################################ PREPROCESSING  ##########################################
    dataset["dteday"] = (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        .fit_transform(dataset[["dteday"]])
        .reshape(-1, 1)
    )
    dataset = pd.get_dummies(
        dataset,
        columns=["workingday", "holiday", "weathersit"],
        drop_first=False,
        dtype=int,
    )
    dataset[["casual", "registered"]] = minmax_scale(dataset[["casual", "registered"]])
    ##########################################################################################
    target = dataset.pop(config["target"])
    return dataset.values, target.values


@register
def load_covtype_dataset(config):
    names = [
        "elevation",
        "aspect",
        "slope",
        "horizontal_distance_to_hydrology",
        "vertical_distance_to_hydrology",
        "horizontal_distance_to_roadways",
        "hillshade_9am",
        "hillshade_noon",
        "hillshade_3pm",
        "horizontal_distance_to_fire_points",
        "wilderness_area_0",
        "wilderness_area_1",
        "wilderness_area_2",
        "wilderness_area_3",
        "soil_type_0",
        "soil_type_1",
        "soil_type_2",
        "soil_type_3",
        "soil_type_4",
        "soil_type_5",
        "soil_type_6",
        "soil_type_7",
        "soil_type_8",
        "soil_type_9",
        "soil_type_10",
        "soil_type_11",
        "soil_type_12",
        "soil_type_13",
        "soil_type_14",
        "soil_type_15",
        "soil_type_16",
        "soil_type_17",
        "soil_type_18",
        "soil_type_19",
        "soil_type_20",
        "soil_type_21",
        "soil_type_22",
        "soil_type_23",
        "soil_type_24",
        "soil_type_25",
        "soil_type_26",
        "soil_type_27",
        "soil_type_28",
        "soil_type_29",
        "soil_type_30",
        "soil_type_31",
        "soil_type_32",
        "soil_type_33",
        "soil_type_34",
        "soil_type_35",
        "soil_type_36",
        "soil_type_37",
        "soil_type_38",
        "soil_type_39",
        "cover_type",
    ]
    path = DATA_ROOT / Path(config["root"])

    dataset = pd.read_csv(path, engine="pyarrow", names=names)
    dataset[config["target"]] -= 1
    target = dataset.pop(config["target"])
    return StandardScaler().fit_transform(dataset), LabelBinarizer().fit_transform(
        target
    )


@register
def load_sgemm_dataset(config):
    path = DATA_ROOT / Path(config["root"])
    names = [
        "MWG",
        "NWG",
        "KWG",
        "MDIMC",
        "NDIMC",
        "MDIMA",
        "NDIMB",
        "KWI",
        "VWM",
        "VWN",
        "STRM",
        "STRN",
        "SA",
        "SB",
        "Run1 (ms)",
        "Run2 (ms)",
        "Run3 (ms)",
        "Run4 (ms)",
    ]
    dataset = pd.read_csv(path, engine="pyarrow", index_col=0, skiprows=1, names=names)
    dataset[config["target"]] = dataset[
        ["Run1 (ms)", "Run2 (ms)", "Run3 (ms)", "Run4 (ms)"]
    ].mean(axis=1)
    dataset = dataset.drop(columns=["Run1 (ms)", "Run2 (ms)", "Run3 (ms)", "Run4 (ms)"])
    target = dataset.pop(config["target"])
    dataset = normalize(
        dataset,
        axis=0,
        norm="max",
    )
    return dataset.astype(np.float64), target.values.astype(np.float64)


@register
def load_hepmass_dataset(config):
    path = DATA_ROOT / Path(config["train"])
    dataset = pd.read_csv(path, engine="pyarrow")
    dataset.drop(columns=["mass"], inplace=True)
    # dataset["mass"] = (
    #     OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    #     .fit_transform(dataset[["mass"]])
    #     .reshape(-1, 1)
    # )
    target = dataset.pop(config["target"])
    return dataset.values.astype(np.float32), target.values.astype(np.float32)
