from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import numpy as np
from scipy import stats


def get_args():
    parser = ArgumentParser(description="Testes de hipótese para experimentos de coreset")
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default="default_experiment",
        help="Nome do experimento (pasta em outputs/)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="XGBClassifier",
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
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="covtype",
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
    parser.add_argument(
        "--baseline",
        "-b",
        type=str,
        default="random_sampler",
        choices=["random", "random_sampler", "freddy", "gradmatch", "craig", "craig_baseline"],
        help="Método baseline para comparação",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.05,
        help="Nível de significância (default: 0.05)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Métrica específica para analisar (default: todas)",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=None,
        help="Fração específica para analisar (default: todas)",
    )
    return parser.parse_args()


def load_experiment_data(experiment_dir: Path) -> pd.DataFrame:
    """Carrega todos os CSVs de um diretório de experimento."""
    all_data = []
    for csv_file in experiment_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        all_data.append(df)

    if not all_data:
        raise ValueError(f"Nenhum CSV encontrado em {experiment_dir}")

    return pd.concat(all_data, ignore_index=True)


def paired_ttest(group1: np.ndarray, group2: np.ndarray, alpha: float = 0.05):
    """
    Executa t-test pareado entre dois grupos.

    Returns:
        dict com estatística t, p-valor, e se rejeita H0
    """
    # Garantir mesmo tamanho
    min_len = min(len(group1), len(group2))
    g1 = group1[:min_len]
    g2 = group2[:min_len]

    t_stat, p_value = stats.ttest_rel(g1, g2)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "reject_h0": p_value < alpha,
        "mean_diff": np.mean(g1) - np.mean(g2),
        "std_diff": np.std(g1 - g2),
        "n_samples": min_len,
    }


def run_hypothesis_tests(
    data: pd.DataFrame,
    baseline: str,
    alpha: float = 0.05,
    metric_filter: str = None,
    frac_filter: float = None,
) -> list[dict]:
    """
    Executa testes de hipótese comparando métodos contra o baseline.

    H0: Não há diferença significativa entre o método e o baseline
    H1: Há diferença significativa
    """
    results = []

    # Filtrar por métrica se especificado
    if metric_filter:
        data = data[data["metric"] == metric_filter]

    # Filtrar por fração se especificado
    if frac_filter:
        data = data[data["frac"] == frac_filter]

    # Obter métodos únicos (excluindo baseline)
    methods = data["method"].unique()
    methods = [m for m in methods if m != baseline and m is not None]

    # Obter métricas e frações únicas
    metrics = data["metric"].unique()
    fracs = data["frac"].dropna().unique()

    baseline_data = data[data["method"] == baseline]

    for method in methods:
        method_data = data[data["method"] == method]

        for metric in metrics:
            for frac in fracs:
                # Filtrar dados
                b_values = baseline_data[
                    (baseline_data["metric"] == metric) &
                    (baseline_data["frac"] == frac)
                ]["test"].values

                m_values = method_data[
                    (method_data["metric"] == metric) &
                    (method_data["frac"] == frac)
                ]["test"].values

                if len(b_values) < 2 or len(m_values) < 2:
                    continue

                # Executar t-test pareado
                test_result = paired_ttest(m_values, b_values, alpha)

                results.append({
                    "method": method,
                    "baseline": baseline,
                    "metric": metric,
                    "frac": frac,
                    "method_mean": np.mean(m_values),
                    "method_std": np.std(m_values),
                    "baseline_mean": np.mean(b_values),
                    "baseline_std": np.std(b_values),
                    **test_result,
                })

    return results


def print_results(results: list[dict], alpha: float):
    """Imprime resultados formatados."""
    if not results:
        print("Nenhum resultado encontrado.")
        return

    print("\n" + "=" * 80)
    print(f"TESTES DE HIPÓTESE (t-test pareado, α = {alpha})")
    print("=" * 80)
    print(f"H0: Não há diferença significativa entre método e baseline")
    print(f"H1: Há diferença significativa")
    print("-" * 80)

    df = pd.DataFrame(results)

    for metric in df["metric"].unique():
        print(f"\n### Métrica: {metric}")
        print("-" * 60)

        metric_df = df[df["metric"] == metric]

        for _, row in metric_df.iterrows():
            significance = "***" if row["reject_h0"] else ""
            direction = ">" if row["mean_diff"] > 0 else "<"

            print(f"  {row['method']:12} vs {row['baseline']:8} | "
                  f"frac={row['frac']:.2f} | "
                  f"Δ={row['mean_diff']:+.4f} ({row['method']:>8} {direction} {row['baseline']}) | "
                  f"p={row['p_value']:.4f} {significance}")

    print("\n" + "-" * 80)
    print("*** = Rejeita H0 (diferença significativa)")
    print("=" * 80)


if __name__ == "__main__":
    args = get_args()

    # Construir caminho do experimento
    output_dir = Path("outputs") / args.experiment / args.model / args.dataset

    if not output_dir.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {output_dir}")

    print(f"Carregando dados de: {output_dir}")
    data = load_experiment_data(output_dir)
    print(f"Total de registros: {len(data)}")
    print(f"Métodos encontrados: {data['method'].unique()}")
    print(f"Métricas: {data['metric'].unique()}")
    print(f"Frações: {sorted(data['frac'].dropna().unique())}")

    # Executar testes
    results = run_hypothesis_tests(
        data=data,
        baseline=args.baseline,
        alpha=args.alpha,
        metric_filter=args.metric,
        frac_filter=args.frac,
    )

    # Imprimir resultados
    print_results(results, args.alpha)

    # Salvar resultados
    if results:
        results_df = pd.DataFrame(results)
        output_file = output_dir / f"hypothesis_test_{args.baseline}_alpha{args.alpha}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResultados salvos em: {output_file}")
