import json
from pathlib import Path
from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    SGDClassifier,
    SGDRegressor,
)
from xgboost import XGBClassifier, XGBRegressor

from utiils.arguments import get_args
from utiils.datasets import load_dataset, get_metric_functions, get_pipeline
from sampling import SAMPLERS

MODELS = {
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "XGBClassifier": XGBClassifier,
    "XGBRegressor": XGBRegressor,
    "LogisticRegression": LogisticRegression,
    "LinearRegression": LinearRegression,
    "SGDClassifier": SGDClassifier,
    "SGDRegressor": SGDRegressor,
}

SAMPLING_HYPERPARAMS = ["alpha", "tol", "max_iter", "batch_size", "beta", "b_size"]


def compute_coverage_mean(X_train: np.ndarray, coreset_feat: np.ndarray) -> float:
    """Mean distance from each train point to its nearest coreset neighbor.

    coverage_mean = (1/n) * Σ_i min_j d(x_i, c_j)

    Lower = better coverage (coreset more representative of training space).
    Spec: specs/technical/coverage_metric.md
    Ref: Har-Peled & Mazumdar (2004), STOC.
    """
    if len(coreset_feat) == 0:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(coreset_feat)
    distances, _ = nn.kneighbors(X_train)
    return float(distances.mean())


def cmd_select_coreset(args):
    features, target = load_dataset(args.dataset)
    print(
        f"Dataset {args.dataset}: {features.shape[0]} amostras, {features.shape[1]} features"
    )

    sampler_fn, defaults = SAMPLERS[args.method]
    sampling_args = dict(defaults)
    for key in SAMPLING_HYPERPARAMS:
        val = getattr(args, key, None)
        if val is not None:
            sampling_args[key] = val

    n_samples = len(features)
    full_set = np.arange(n_samples)
    pipeline = get_pipeline(args.dataset)

    out_dir = Path(args.output_dir) / args.dataset / args.method / str(args.train_frac)
    out_dir.mkdir(parents=True, exist_ok=True)

    for run_idx in range(args.runs):
        train_idx, test_idx = train_test_split(
            full_set, test_size=args.test_size, random_state=run_idx
        )
        K = (
            int(len(train_idx) * args.train_frac)
            if args.train_frac < 1
            else int(args.train_frac)
        )
        train_feat = pipeline.fit_transform(features[train_idx])
        metadata = {
            "dataset": args.dataset,
            "method": args.method,
            "train_frac": args.train_frac,
            "K": K,
            "train_size": len(train_feat),
            "test_size": args.test_size,
            "sampling_args": {
                k: v
                for k, v in sampling_args.items()
                if isinstance(v, (int, float, str, bool))
            },
            "runs": args.runs,
            "created_at": datetime.now().isoformat(),
        }

        print(f"  Run {run_idx + 1}/{args.runs} - K={K} ({args.train_frac*100:.1f}%)")
        elapsed, indices = sampler_fn(train_feat, K=K, **sampling_args)
        metadata["selection_times"] = elapsed
        metadata["coverage_mean"] = compute_coverage_mean(train_feat, train_feat[indices])
        np.save(out_dir / f"train_{run_idx}.npy", train_idx[indices])
        np.save(out_dir / f"test_{run_idx}.npy", test_idx)

        print(
            f"    Tempo de selecao: {elapsed:.2f}s, {len(indices)} indices selecionados"
        )

        with open(out_dir / f"metadata_{run_idx}.json", "w") as f_:
            json.dump(metadata, f_, indent=2)

        print(f"  Salvo em {out_dir}")


def cmd_model_train(args):
    model_cls = MODELS[args.model]

    if args.coreset_dir is None:
        raise ValueError("--coreset_dir e obrigatorio")

    if args.method == "none":
        results = _train_full_dataset(model_cls, args)
    else:
        results = _train_on_coreset(model_cls, args)

    df = pd.DataFrame.from_records(results)

    dataset_name = df["dataset"].iloc[0]
    if args.method == "none":
        fname = f"{dataset_name}_{args.model}_none_1.0.csv"
    else:
        metodo = df["metodo"].iloc[0]
        fracao = df["fracao"].iloc[0]
        fname = f"{dataset_name}_{args.model}_{metodo}_{fracao}.csv"

    output_dir = Path(args.output) / args.name / args.model / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / fname
    df.to_csv(output_path, index=False)
    print(f"Resultados salvos em {output_path}")
    print(df.to_string(index=False))


def _train_on_coreset(model_cls, args):
    coreset_dir = Path(args.coreset_dir)

    train_sort = sorted(coreset_dir.glob("train_*.npy"))
    test_sort = sorted(coreset_dir.glob("test_*.npy"))
    _meta_path = sorted(coreset_dir.glob("metadata_*.json"))
    results = []
    for train_file, test_file, meta_path in zip(train_sort, test_sort, _meta_path):
        print(f"loading: {train_file.name}, {test_file.name}, {meta_path.name}")
        # meta_path = coreset_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json nao encontrado em {coreset_dir}")

        with open(meta_path) as f:
            metadata = json.load(f)

        dataset_name = metadata["dataset"]
        method = metadata["method"]
        frac = metadata["train_frac"]
        # selection_times = metadata.get("selection_times")

        features, target = load_dataset(dataset_name)
        metrics = get_metric_functions(dataset_name)
        pipeline = get_pipeline(dataset_name)

        # Ler indices de treino/teste salvos pelo select-coreset

        train_idx = np.load(train_file)
        test_idx = np.load(test_file)

        train_raw = features[train_idx]
        test_raw = features[test_idx]
        train_feat = pipeline.fit_transform(train_raw)
        test_feat = pipeline.transform(test_raw)
        train_target, test_target = target[train_idx], target[test_idx]
        run_idx = int(train_file.stem.split("_")[1])
        # coreset_idx = np.load(run_file)
        for i in range(5):
            seed = run_idx * 5 + i
            try:
                model = model_cls(random_state=seed)
            except TypeError:
                model = model_cls()
            t0 = perf_counter()
            model.fit(train_feat, train_target)
            train_time = perf_counter() - t0

            test_pred = model.predict(test_feat)
            # sel_time = selection_times[run_idx] if run_idx < len(selection_times) else 0

            for metric_fn in metrics:
                try:
                    value = metric_fn(test_target, test_pred)
                except Exception:
                    value = metric_fn(test_target, test_pred, average="macro")

                results.append(
                    {
                        "dataset": dataset_name,
                        "metodo": method,
                        "fracao": frac,
                        "metrica": metric_fn.__name__,
                        "valor": value,
                        "modelo": model_cls.__name__,
                        "run": run_idx,
                        "train_rep": i,
                        "train_time": train_time,
                        "selection_time": metadata.get("selection_times"),
                        "coverage_mean": metadata.get("coverage_mean", float("nan")),
                    }
                )

            del model
            print(f"  run_{run_idx} rep_{i}: train_time={train_time:.2f}s")

    return results


def _train_full_dataset(model_cls, args):
    features, target = load_dataset(args.dataset)
    metrics = get_metric_functions(args.dataset)
    pipeline = get_pipeline(args.dataset)

    train_raw, test_raw, train_target, test_target = train_test_split(
        features, target, test_size=args.test_size, random_state=42
    )
    train_feat = pipeline.fit_transform(train_raw)
    test_feat = pipeline.transform(test_raw)

    model = model_cls()
    t0 = perf_counter()
    model.fit(train_feat, train_target)
    train_time = perf_counter() - t0

    test_pred = model.predict(test_feat)

    results = []
    for metric_fn in metrics:
        try:
            value = metric_fn(test_target, test_pred)
        except Exception:
            value = metric_fn(test_target, test_pred, average="macro")

        results.append(
            {
                "dataset": args.dataset,
                "metodo": "none",
                "fracao": 1.0,
                "metrica": metric_fn.__name__,
                "valor": value,
                "modelo": model_cls.__name__,
                "run": 0,
                "train_time": train_time,
                "selection_time": 0,
                "coverage_mean": float("nan"),
            }
        )

    del model
    print(f"  full dataset: train_time={train_time:.2f}s")

    return results


if __name__ == "__main__":
    args = get_args()
    print(args)

    if args.command == "select-coreset":
        cmd_select_coreset(args)
    elif args.command == "model-train":
        cmd_model_train(args)
