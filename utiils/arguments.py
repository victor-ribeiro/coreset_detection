from argparse import ArgumentParser


MODELS = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "XGBClassifier",
    "XGBRegressor",
    "LogisticRegression",
    "LinearRegression",
    "SGDClassifier",
    "SGDRegressor",
]

METHODS = ["random", "freddy", "craig", "gradmatch", "none"]

DATASET_NAMES = [
    "sgemm",
    "covtype",
    "adult",
    "bike_share",
    "hepmass",
    "predictmds",
    "storage_perf",
    "higgs",
]


def get_args():
    parser = ArgumentParser(description="Coreset Detection Experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # === select-coreset ===
    sel = subparsers.add_parser(
        "select-coreset", help="Selecionar coreset e salvar indices"
    )
    sel.add_argument("--dataset", "-d", required=True, choices=DATASET_NAMES)
    sel.add_argument("--method", required=True, choices=METHODS)
    sel.add_argument("--train_frac", type=float, required=True)
    sel.add_argument("--runs", "-r", type=int, default=1)
    sel.add_argument("--test_size", type=float, default=0.2)
    sel.add_argument("--output_dir", type=str, default="coreset")
    # Hiperparametros de metodo (opcionais, defaults vem do registro SAMPLERS)
    sel.add_argument("--alpha", "-a", type=float, default=None)
    sel.add_argument("--tol", "-t", type=float, default=None)
    sel.add_argument("--max_iter", "-i", type=int, default=None)
    sel.add_argument("--batch_size", type=int, default=None)
    sel.add_argument("--beta", "-b", type=float, default=None)
    sel.add_argument("--b_size", type=int, default=None)

    # === model-train ===
    trn = subparsers.add_parser(
        "model-train", help="Treinar modelo usando coreset salvo"
    )
    trn.add_argument("--model", "-m", required=True, choices=MODELS)
    trn.add_argument(
        "--coreset_dir",
        type=str,
        required=True,
        help="Caminho para diretorio do coreset (ex: coreset/covtype/freddy/0.05)",
    )
    trn.add_argument(
        "--method",
        type=str,
        default=None,
        help="Usar 'none' para treinar no dataset completo",
    )
    trn.add_argument("--output", "-o", type=str, default="outputs")
    trn.add_argument("--name", "-n", type=str, default="default_experiment")
    trn.add_argument("--dataset", "-d", required=True, choices=DATASET_NAMES)

    # === compute-coverage ===
    cov = subparsers.add_parser(
        "compute-coverage", help="Calcular coverage_mean post-hoc para coresets salvos"
    )
    cov.add_argument("--coreset_dir", type=str, required=True,
                     help="Diretorio com train_*.npy e metadata_*.json")
    cov.add_argument("--dataset", "-d", required=True, choices=DATASET_NAMES)

    return parser.parse_args()
