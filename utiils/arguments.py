from argparse import ArgumentParser
from dataclasses import dataclass, field


@dataclass
class Args:
    default: str = None
    choices: list = field(
        default_factory=list,
    )
    required: bool = False


MODELS = Args(
    default="DecisionTreeClassifier",
    choices=[
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "RandomForestClassifier",
        "RandomForestRegressor",
        "XGBClassifier",
        "XGBRegressor",
        "LogisticRegression",
        "LinearRegression",
    ],
)
METHODS = Args(
    default="pmi_kmeans_sampler",
    choices=["none", "kmeans", "pmi_kmeans", "random", "craig", "freddy", "gradmatch"],
)
DATASETS = Args(
    default="bike_share",
    choices=[
        "sgemm",
        "covtype",
        "adult",
        "bike_share",
        "hepmass",
        "predictmds",
        "storage_perf",
        "higgs",
    ],
)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=MODELS.required,
        choices=MODELS.choices,
        default=MODELS.default,
    )
    parser.add_argument(
        "--run",
        "-r",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--resample",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--method",
        type=str,
        default=METHODS.default,
        choices=METHODS.choices,
    )
    parser.add_argument("--name", "-n", default="default_experiment")
    #######################################################

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=DATASETS.default,
        choices=DATASETS.choices,
        required=DATASETS.required,
    )

    parser.add_argument("--alpha", "-a", type=float, default=0.15, required=False)
    parser.add_argument("--tol", "-t", type=float, default=10e-3, required=False)
    parser.add_argument("--max_iter", "-i", type=int, default=100, required=False)
    parser.add_argument("--train_frac", type=float, default=0.1, required=False)
    parser.add_argument("--random_seed", type=int, default=42, required=False)
    parser.add_argument("--batch_size", type=int, default=1024, required=False)
    parser.add_argument("--beta", "-b", type=float, default=0.75, required=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
    print(args.dataset)
    print(args.model)
