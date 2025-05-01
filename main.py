from pathlib import Path

import pandas as pd

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    root_mean_squared_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from xgboost import XGBClassifier, XGBRegressor

from utiils.arguments import get_args
from utiils.datasets import (
    load_config,
    load_adult_dataset,
    load_bike_share_dataset,
    load_covtype_dataset,
    load_hepmass_dataset,
    load_sgemm_dataset,
)
from utiils.trainer import experiment, _parse_expperiment_name

CONFIG = Path(".config")
CONFIG = {file.stem: file.resolve() for file in CONFIG.rglob("*.json")}

if __name__ == "__main__":
    args = get_args()
    if not hasattr(args, "method"):
        args.method = "none"
    if not hasattr(args, "train_frac"):
        args.train_frac = 1
    print(args.__dict__)
    spln_args = {}
    match args.dataset:
        case "hepmass":
            config = load_config(CONFIG["hepmass"])
            dataset, target = load_hepmass_dataset(config)
            metrics = [accuracy_score, precision_score]
            print(f"Loaded {args.dataset} dataset with {dataset.shape[0]} samples.")
        case "sgemm":
            config = load_config(CONFIG["sgemm"])
            dataset, target = load_sgemm_dataset(config)
            metrics = [root_mean_squared_error]
            print(f"Loaded {args.dataset} dataset with {dataset.shape[0]} samples.")
        case "covtype":
            config = load_config(CONFIG["covtype"])
            dataset, target = load_covtype_dataset(config)
            metrics = [accuracy_score, precision_score]
            print(f"Loaded {args.dataset} dataset with {dataset.shape[0]} samples.")
        case "adult":
            config = load_config(CONFIG["adult"])
            dataset, target = load_adult_dataset(config)
            metrics = [accuracy_score, precision_score]
            print(f"Loaded {args.dataset} dataset with {dataset.shape[0]} samples.")
        case "bike_share":
            config = load_config(CONFIG["bike_share"])
            dataset, target = load_bike_share_dataset(config)
            metrics = [root_mean_squared_error]
            print(f"Loaded {args.dataset} dataset with {dataset.shape[0]} samples.")
        case _:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    print(f"Loaded {args.dataset} dataset with {dataset.shape[0]} samples.")
    match args.model:
        case "DecisionTreeClassifier":
            model = DecisionTreeClassifier
        case "DecisionTreeRegressor":
            model = DecisionTreeRegressor
        case "RandomForestClassifier":
            model = RandomForestClassifier
        case "RandomForestRegressor":
            model = RandomForestRegressor
        case "XGBClassifier":
            model = XGBClassifier
        case "XGBRegressor":
            model = XGBRegressor
        case "LogisticRegression":
            model = LogisticRegression
        case "LinearRegression":
            model = LinearRegression
        case _:
            raise ValueError(f"Unknown model: {args.model}")

    # spln_args = None
    match args.method:
        case "none":
            sampler = None
        case "kmeans":
            from sampling import kmeans_sampler

            spln_args["alpha"] = args.alpha
            spln_args["max_iter"] = args.max_iter
            spln_args["tol"] = args.tol
            sampler = kmeans_sampler
        case "pmi_kmeans":
            from sampling import pmi_kmeans_sampler

            spln_args["alpha"] = args.alpha
            spln_args["max_iter"] = args.max_iter
            spln_args["tol"] = args.tol

            sampler = pmi_kmeans_sampler
        case "random":
            from sampling import random_sampler

            sampler = random_sampler
        case "craig":
            from sampling import craig_baseline

            spln_args["b_size"] = args.batch_size
            sampler = craig_baseline
        case "freddy":
            from sampling import freddy

            spln_args = {
                "alpha": args.alpha,
                "beta": args.beta,
                "batch_size": args.batch_size,
            }
            sampler = freddy
        case "gradmatch":
            from sampling import gradmatch

            spln_args = {"tol": args.tol, "batch_size": args.batch_size}
            sampler = gradmatch
        case _:
            raise ValueError(f"Unknown sampling method: {args.method}")

    result = experiment(
        learner=model,
        features=dataset,
        target=target,
        frac=args.train_frac,
        runs=args.run,
        resample=args.resample,
        sampler=sampler,
        sampling_args=spln_args,
        metrics=metrics,
        # model_args={"n_jobs": 2},
    )
    name = f"{_parse_expperiment_name(args)}.csv"
    output_dir = Path("outputs") / args.name / args.model / args.dataset
    output = output_dir / name

    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)

    result = pd.DataFrame.from_records(result)
    result.to_csv(output, index=False)
