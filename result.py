# %%
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# === Carregar dados ===
files = Path("outputs/default_experiment")
df = pd.concat(
    [pd.read_csv(f_) for f_ in files.rglob("*.csv")],
    ignore_index=True,
)
df["elapsed"] = df["selection_time"] + df["train_time"]

datasets = df["dataset"].unique()
modelos = df["modelo"].unique()
metricas = df["metrica"].unique()
metodos = df["metodo"].unique()
fracoes = sorted(df["fracao"].unique())

METHOD_LABELS = {
    "freddy": "Freddy",
    "random": "Random",
    "gradmatch": "GradMatch",
}
METRIC_LABELS = {
    "accuracy_score": "Accuracy",
    "precision_score": "Precision",
}

df["metodo_label"] = df["metodo"].map(METHOD_LABELS)

# %% === Grafico 1: Comparativo metodo x fracao, facetado por modelo (um grafico por dataset e metrica) ===
Path("figs").mkdir(parents=True, exist_ok=True)
for dataset in datasets:
    df_ds = df[df.dataset == dataset]
    modelos_ds = df_ds["modelo"].unique()

    for metrica in metricas:
        df_m = df_ds[df_ds.metrica == metrica]
        n_modelos = len(modelos_ds)
        ncols = min(3, n_modelos)
        nrows = int(np.ceil(n_modelos / ncols))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False
        )
        fig.suptitle(
            f"{METRIC_LABELS.get(metrica, metrica)} -- {dataset}",
            fontsize=16, fontweight="bold", y=1.02,
        )

        for idx, modelo in enumerate(modelos_ds):
            ax = axes[idx // ncols, idx % ncols]
            subset = df_m[df_m.modelo == modelo]

            sns.boxplot(
                data=subset,
                x="fracao",
                y="valor",
                hue="metodo_label",
                ax=ax,
                palette="Set2",
                fliersize=2,
            )
            ax.set_title(modelo, fontsize=12)
            ax.set_xlabel("Fração de seleção")
            ax.set_ylabel(METRIC_LABELS.get(metrica, metrica))

            if idx == 0:
                ax.legend(title="Método", fontsize=8, title_fontsize=9)
            else:
                ax.get_legend().remove()

        # Remover eixos vazios
        for idx in range(len(modelos_ds), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.tight_layout()
        fig.savefig(f"figs/{dataset}_{metrica}_por_modelo.pdf", bbox_inches="tight", dpi=150)
        fig.savefig(f"figs/{dataset}_{metrica}_por_modelo.png", bbox_inches="tight", dpi=150)

plt.show()

# %% === Grafico 2: Tempo de selecao por metodo e fracao ===
for dataset in datasets:
    df_ds = df[df.dataset == dataset]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df_ds.drop_duplicates(subset=["metodo", "fracao", "run"]),
        x="fracao",
        y="selection_time",
        hue="metodo_label",
        ax=ax,
        palette="Set2",
        errorbar="sd",
    )
    ax.set_title(f"Tempo de seleção -- {dataset}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Fração de seleção")
    ax.set_ylabel("Tempo (s)")
    ax.legend(title="Método")

    fig.tight_layout()
    fig.savefig(f"figs/{dataset}_selection_time.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(f"figs/{dataset}_selection_time.png", bbox_inches="tight", dpi=150)
plt.show()

# %% === Tabelas LaTeX: uma por (dataset, metrica, modelo) ===
tex_dir = Path("tables/tables")
tex_dir.mkdir(parents=True, exist_ok=True)

for dataset in datasets:
    df_ds = df[df.dataset == dataset]
    modelos_ds = df_ds["modelo"].unique()

    for metrica in metricas:
        for modelo in modelos_ds:
            subset = df_ds[(df_ds.metrica == metrica) & (df_ds.modelo == modelo)]
            if subset.empty:
                continue

            # Pivot: media +- desvio padrao
            stats = subset.groupby(["fracao", "metodo"])["valor"].agg(["mean", "std"])
            stats["cell"] = stats.apply(
                lambda r: f"{r['mean']:.4f} $\\pm$ {r['std']:.4f}", axis=1
            )
            table = stats["cell"].unstack("metodo")
            table = table.reindex(columns=sorted(table.columns))
            table.columns = [METHOD_LABELS.get(c, c) for c in table.columns]
            table.index.name = "Fração"

            # Destacar melhor valor por linha (bold)
            means = subset.groupby(["fracao", "metodo"])["valor"].mean().unstack("metodo")
            means = means.reindex(columns=sorted(means.columns))

            for frac in table.index:
                best_method = means.loc[frac].idxmax()
                best_label = METHOD_LABELS.get(best_method, best_method)
                table.loc[frac, best_label] = (
                    "\\textbf{" + table.loc[frac, best_label] + "}"
                )

            latex_str = table.to_latex(
                escape=False,
                caption=f"{METRIC_LABELS.get(metrica, metrica)} -- {modelo} ({dataset})",
                label=f"tab:{dataset}_{metrica}_{modelo}",
                position="htbp",
            )

            fname = tex_dir / f"{dataset}_{metrica}_{modelo}.tex"
            with open(fname, "w") as f:
                f.write(latex_str)
            print(f"Tabela salva: {fname}")

# %% === Tabela consolidada: todos os modelos por (dataset, metrica) ===
for dataset in datasets:
    df_ds = df[df.dataset == dataset]

    for metrica in metricas:
        subset = df_ds[df_ds.metrica == metrica]

        stats = subset.groupby(["modelo", "fracao", "metodo"])["valor"].agg(["mean", "std"])
        stats["cell"] = stats.apply(
            lambda r: f"{r['mean']:.4f} $\\pm$ {r['std']:.4f}", axis=1
        )
        table = stats["cell"].unstack("metodo")
        table = table.reindex(columns=sorted(table.columns))
        table.columns = [METHOD_LABELS.get(c, c) for c in table.columns]
        table.index.names = ["Modelo", "Fração"]

        # Bold no melhor por (modelo, fracao)
        means = (
            subset.groupby(["modelo", "fracao", "metodo"])["valor"].mean().unstack("metodo")
        )
        means = means.reindex(columns=sorted(means.columns))

        for idx in table.index:
            best_method = means.loc[idx].idxmax()
            best_label = METHOD_LABELS.get(best_method, best_method)
            table.loc[idx, best_label] = "\\textbf{" + table.loc[idx, best_label] + "}"

        latex_str = table.to_latex(
            escape=False,
            caption=f"{METRIC_LABELS.get(metrica, metrica)} -- Todos os modelos ({dataset})",
            label=f"tab:{dataset}_{metrica}_all",
            position="htbp",
            multirow=True,
        )

        fname = tex_dir / f"{dataset}_{metrica}_consolidado.tex"
        with open(fname, "w") as f:
            f.write(latex_str)
        print(f"Tabela consolidada salva: {fname}")

# %% === Tabela de tempo de selecao por dataset ===
for dataset in datasets:
    df_ds = df[df.dataset == dataset]
    time_df = df_ds.drop_duplicates(subset=["metodo", "fracao", "run"])
    stats = time_df.groupby(["fracao", "metodo"])["selection_time"].agg(["mean", "std"])
    stats["cell"] = stats.apply(lambda r: f"{r['mean']:.2f} $\\pm$ {r['std']:.2f}", axis=1)
    table = stats["cell"].unstack("metodo")
    table = table.reindex(columns=sorted(table.columns))
    table.columns = [METHOD_LABELS.get(c, c) for c in table.columns]
    table.index.name = "Fração"

    latex_str = table.to_latex(
        escape=False,
        caption=f"Tempo de seleção (segundos) -- {dataset}",
        label=f"tab:{dataset}_selection_time",
        position="htbp",
    )

    fname = tex_dir / f"{dataset}_selection_time.tex"
    with open(fname, "w") as f:
        f.write(latex_str)
    print(f"Tabela salva: {fname}")

# %%
