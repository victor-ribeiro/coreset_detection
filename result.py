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
    "craig": "craig",
    "none": "none",
}
METRIC_LABELS = {
    "accuracy_score": "Accuracy",
    "precision_score": "Precision",
}

df["metodo_label"] = df["metodo"].map(METHOD_LABELS)

# Paleta global fixa — garante coerência de cores entre todos os gráficos
_set2 = sns.color_palette("Set2")
COLOR_MAP = {
    "Freddy": _set2[0],
    "Random": _set2[1],
    "GradMatch": _set2[2],
    "craig": _set2[3],
    "none": "#b0b0b0",
}
HUE_ORDER = [k for k in COLOR_MAP if k in df["metodo_label"].values]

# %% === Grafico 1: Comparativo metodo x fracao, facetado por modelo (um grafico por dataset e metrica) ===
Path("figs").mkdir(parents=True, exist_ok=True)
for dataset in datasets:
    df_ds = df[df.dataset == dataset]
    modelos_ds = df_ds["modelo"].unique()

    for metrica in metricas:
        df_m = df_ds[df_ds.metrica == metrica]
        n_modelos = len(modelos_ds)
        if n_modelos == 0:
            continue
        ncols = min(3, n_modelos)

        nrows = int(np.ceil(n_modelos / ncols))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False
        )
        fig.suptitle(
            f"{METRIC_LABELS.get(metrica, metrica)} -- {dataset}",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        for idx, modelo in enumerate(modelos_ds):
            ax = axes[idx // ncols, idx % ncols]
            subset = df_m[df_m.modelo == modelo]

            sns.boxplot(
                data=subset,
                x="fracao",
                y="valor",
                hue="metodo_label",
                hue_order=[h for h in HUE_ORDER if h in subset["metodo_label"].values],
                ax=ax,
                palette=COLOR_MAP,
                # fliersize=2,
            )
            ax.set_title(modelo, fontsize=12)
            ax.set_xlabel("Selection fraction")
            ax.set_ylabel(METRIC_LABELS.get(metrica, metrica))

            if idx == 0:
                ax.legend(title="Method", fontsize=8, title_fontsize=9)
            else:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

        # Remover eixos vazios
        for idx in range(len(modelos_ds), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)
        fig.tight_layout()
        fig.savefig(
            f"figs/{dataset}_{metrica}_por_modelo.pdf", bbox_inches="tight", dpi=150
        )
        fig.savefig(
            f"figs/{dataset}_{metrica}_por_modelo.png", bbox_inches="tight", dpi=150
        )

plt.show()

# %% === Grafico 2: Tempo de selecao por metodo e fracao ===
for dataset in datasets:
    df_ds = df[df.dataset == dataset]
    fig, ax = plt.subplots(figsize=(10, 5))
    _ds_bar = df_ds.drop_duplicates(subset=["metodo", "fracao", "run"])
    sns.barplot(
        data=_ds_bar,
        x="fracao",
        y="selection_time",
        hue="metodo_label",
        hue_order=[h for h in HUE_ORDER if h in _ds_bar["metodo_label"].values],
        ax=ax,
        palette=COLOR_MAP,
        errorbar="sd",
    )
    ax.set_title(f"Selection time -- {dataset}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Selection fraction")
    ax.set_ylabel("Time (s)")
    ax.legend(title="Method")

    fig.tight_layout()
    fig.savefig(f"figs/{dataset}_selection_time.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(f"figs/{dataset}_selection_time.png", bbox_inches="tight", dpi=150)
plt.show()

# %% === Correlação Spearman: coverage_mean x metrica de qualidade ===
from scipy.stats import spearmanr

df_corr = df[df["metodo"] != "none"].dropna(subset=["coverage_mean", "valor"])

records = []
for (dataset, modelo, fracao, metrica), group in df_corr.groupby(
    ["dataset", "modelo", "fracao", "metrica"]
):
    if len(group) < 5:
        continue
    r, p = spearmanr(group["coverage_mean"], group["valor"])
    records.append(
        {
            "dataset": dataset,
            "modelo": modelo,
            "fracao": fracao,
            "metrica": metrica,
            "spearman_r": r,
            "p_value": p,
            "n": len(group),
        }
    )

df_spearman = pd.DataFrame(records)
df_spearman["significativo"] = df_spearman["p_value"] < 0.05
print(df_spearman.to_string(index=False))

# %% === Grafico 3: Dispersão coverage_mean x metrica de qualidade (com tendência) ===
df_scatter = (
    df[df["metodo"] != "none"]
    .dropna(subset=["coverage_mean", "valor"])
    .groupby(
        ["dataset", "metodo", "metodo_label", "fracao", "run", "modelo", "metrica"],
        as_index=False,
    )[["valor", "coverage_mean"]]
    .mean()
)

if df_scatter.empty:
    print("Aviso: sem dados de coverage_mean disponíveis — Grafico 3 ignorado.")
else:
    color_map = COLOR_MAP

    for dataset in df_scatter["dataset"].unique():
        df_ds = df_scatter[df_scatter["dataset"] == dataset]
        for metrica in df_ds["metrica"].unique():
            df_m = df_ds[df_ds["metrica"] == metrica]
            modelos_ds = df_m["modelo"].unique()
            ncols = min(3, len(modelos_ds))
            nrows = int(np.ceil(len(modelos_ds) / ncols))

            fig, axes = plt.subplots(
                nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False
            )
            fig.suptitle(
                f"Coverage vs {METRIC_LABELS.get(metrica, metrica)} — {dataset}",
                fontsize=16,
                fontweight="bold",
                y=1.02,
            )

            for idx, modelo in enumerate(modelos_ds):
                ax = axes[idx // ncols, idx % ncols]
                subset = df_m[df_m["modelo"] == modelo]

                for metodo_label, grp in subset.groupby("metodo_label"):
                    color = color_map[metodo_label]
                    ax.scatter(
                        grp["coverage_mean"],
                        grp["valor"],
                        label=metodo_label,
                        color=color,
                        alpha=0.7,
                        s=40,
                    )
                    if len(grp) >= 3:
                        sns.regplot(
                            data=grp,
                            x="coverage_mean",
                            y="valor",
                            ax=ax,
                            scatter=False,
                            color=color,
                            line_kws={"linewidth": 1.5},
                            ci=None,
                        )

                ax.set_title(modelo, fontsize=12)
                ax.set_xlabel("coverage_mean")
                ax.set_ylabel(METRIC_LABELS.get(metrica, metrica))
                if idx == 0:
                    ax.legend(title="Método", fontsize=8, title_fontsize=9)

            for idx in range(len(modelos_ds), nrows * ncols):
                axes[idx // ncols, idx % ncols].set_visible(False)

            fig.tight_layout()
            fig.savefig(
                f"figs/scatter_coverage_{dataset}_{metrica}.pdf",
                bbox_inches="tight",
                dpi=150,
            )
            fig.savefig(
                f"figs/scatter_coverage_{dataset}_{metrica}.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.show()

# %% === Grafico 4: Dual-axis — fracao x (boxplot valor + coverage_mean) por metodo ===
df_dual_box = df.dropna(subset=["valor"]).copy()
df_dual_box["fracao_str"] = df_dual_box["fracao"].astype(str)

df_dual_line = (
    df.dropna(subset=["coverage_mean"])
    .groupby(
        ["dataset", "metodo", "metodo_label", "fracao", "modelo", "metrica"],
        as_index=False,
    )["coverage_mean"]
    .mean()
)

_box_methods = df_dual_box["metodo_label"].dropna().unique()
if len(_box_methods) == 0:
    print("Aviso: sem dados disponíveis — Grafico 4 ignorado.")
else:
    color_map = COLOR_MAP

    for dataset in df_dual_box["dataset"].unique():
        db_ds = df_dual_box[df_dual_box["dataset"] == dataset]
        dl_ds = df_dual_line[df_dual_line["dataset"] == dataset]

        for metrica in db_ds["metrica"].unique():
            db_m = db_ds[db_ds["metrica"] == metrica]
            dl_m = dl_ds[dl_ds["metrica"] == metrica]
            modelos_ds = db_m["modelo"].unique()
            ncols = min(3, len(modelos_ds))
            nrows = int(np.ceil(len(modelos_ds) / ncols))

            fracoes_str = sorted(db_m["fracao_str"].unique(), key=float)

            fig, axes = plt.subplots(
                nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False
            )
            fig.suptitle(
                f"{METRIC_LABELS.get(metrica, metrica)} & Coverage vs Fração — {dataset}",
                fontsize=16,
                fontweight="bold",
                y=1.02,
            )

            for idx, modelo in enumerate(modelos_ds):
                ax = axes[idx // ncols, idx % ncols]
                ax2 = ax.twinx()

                db_sub = db_m[db_m["modelo"] == modelo]
                dl_sub = dl_m[dl_m["modelo"] == modelo].sort_values("fracao")

                sns.boxplot(
                    data=db_sub,
                    x="fracao_str",
                    y="valor",
                    hue="metodo_label",
                    hue_order=[
                        h for h in HUE_ORDER if h in db_sub["metodo_label"].values
                    ],
                    order=fracoes_str,
                    ax=ax,
                    palette=COLOR_MAP,
                )

                for metodo_label, grp in dl_sub.groupby("metodo_label"):
                    color = color_map.get(metodo_label)
                    cov = grp.dropna(subset=["coverage_mean"])
                    if cov.empty:
                        continue
                    xs = [fracoes_str.index(str(f)) for f in cov["fracao"]]
                    ax2.plot(
                        xs,
                        cov["coverage_mean"],
                        color=color,
                        linestyle="--",
                        marker="s",
                        markersize=4,
                    )

                ax.set_title(modelo, fontsize=12)
                ax.set_xlabel("Fração de seleção")
                ax.set_ylabel(METRIC_LABELS.get(metrica, metrica))
                ax2.set_ylabel("coverage_mean", color="gray")
                ax2.tick_params(axis="y", labelcolor="gray")

                if idx == 0:
                    h1, l1 = ax.get_legend_handles_labels()
                    ax.legend(h1, l1, title="Método", fontsize=7, title_fontsize=8)
                else:
                    ax.get_legend().remove()

            for idx in range(len(modelos_ds), nrows * ncols):
                axes[idx // ncols, idx % ncols].set_visible(False)

            fig.tight_layout()
            fig.savefig(
                f"figs/dual_axis_{dataset}_{metrica}.pdf",
                bbox_inches="tight",
                dpi=150,
            )
            fig.savefig(
                f"figs/dual_axis_{dataset}_{metrica}.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.show()

# # %% === Tabelas LaTeX: uma por (dataset, metrica, modelo) ===
# tex_dir = Path("tables/tables")
# tex_dir.mkdir(parents=True, exist_ok=True)

# for dataset in datasets:
#     df_ds = df[df.dataset == dataset]
#     modelos_ds = df_ds["modelo"].unique()

#     for metrica in metricas:
#         for modelo in modelos_ds:
#             subset = df_ds[(df_ds.metrica == metrica) & (df_ds.modelo == modelo)]
#             if subset.empty:
#                 continue

#             # Pivot: media +- desvio padrao
#             stats = subset.groupby(["fracao", "metodo"])["valor"].agg(["mean", "std"])
#             stats["cell"] = stats.apply(
#                 lambda r: f"{r['mean']:.4f} $\\pm$ {r['std']:.4f}", axis=1
#             )
#             table = stats["cell"].unstack("metodo")
#             table = table.reindex(columns=sorted(table.columns))
#             table.columns = [METHOD_LABELS.get(c, c) for c in table.columns]
#             table.index.name = "Fração"

#             # Destacar melhor valor por linha (bold)
#             means = (
#                 subset.groupby(["fracao", "metodo"])["valor"].mean().unstack("metodo")
#             )
#             means = means.reindex(columns=sorted(means.columns))

#             for frac in table.index:
#                 best_method = means.loc[frac].idxmax()
#                 best_label = METHOD_LABELS.get(best_method, best_method)
#                 table.loc[frac, best_label] = (
#                     "\\textbf{" + table.loc[frac, best_label] + "}"
#                 )

#             latex_str = table.to_latex(
#                 escape=False,
#                 caption=f"{METRIC_LABELS.get(metrica, metrica)} -- {modelo} ({dataset})",
#                 label=f"tab:{dataset}_{metrica}_{modelo}",
#                 position="htbp",
#             )

#             fname = tex_dir / f"{dataset}_{metrica}_{modelo}.tex"
#             with open(fname, "w") as f:
#                 f.write(latex_str)
#             print(f"Tabela salva: {fname}")

# # %% === Tabela consolidada: todos os modelos por (dataset, metrica) ===
# for dataset in datasets:
#     df_ds = df[df.dataset == dataset]

#     for metrica in metricas:
#         subset = df_ds[df_ds.metrica == metrica]

#         stats = subset.groupby(["modelo", "fracao", "metodo"])["valor"].agg(
#             ["mean", "std"]
#         )
#         stats["cell"] = stats.apply(
#             lambda r: f"{r['mean']:.4f} $\\pm$ {r['std']:.4f}", axis=1
#         )
#         table = stats["cell"].unstack("metodo")
#         table = table.reindex(columns=sorted(table.columns))
#         table.columns = [METHOD_LABELS.get(c, c) for c in table.columns]
#         table.index.names = ["Modelo", "Fração"]

#         # Bold no melhor por (modelo, fracao)
#         means = (
#             subset.groupby(["modelo", "fracao", "metodo"])["valor"]
#             .mean()
#             .unstack("metodo")
#         )
#         means = means.reindex(columns=sorted(means.columns))

#         for idx in table.index:
#             best_method = means.loc[idx].idxmax()
#             best_label = METHOD_LABELS.get(best_method, best_method)
#             table.loc[idx, best_label] = "\\textbf{" + table.loc[idx, best_label] + "}"

#         latex_str = table.to_latex(
#             escape=False,
#             caption=f"{METRIC_LABELS.get(metrica, metrica)} -- Todos os modelos ({dataset})",
#             label=f"tab:{dataset}_{metrica}_all",
#             position="htbp",
#             multirow=True,
#         )

#         fname = tex_dir / f"{dataset}_{metrica}_consolidado.tex"
#         with open(fname, "w") as f:
#             f.write(latex_str)
#         print(f"Tabela consolidada salva: {fname}")

# # %% === Tabela de tempo de selecao por dataset ===
# for dataset in datasets:
#     df_ds = df[df.dataset == dataset]
#     time_df = df_ds.drop_duplicates(subset=["metodo", "fracao", "run"])
#     stats = time_df.groupby(["fracao", "metodo"])["selection_time"].agg(["mean", "std"])
#     stats["cell"] = stats.apply(
#         lambda r: f"{r['mean']:.2f} $\\pm$ {r['std']:.2f}", axis=1
#     )
#     table = stats["cell"].unstack("metodo")
#     table = table.reindex(columns=sorted(table.columns))
#     table.columns = [METHOD_LABELS.get(c, c) for c in table.columns]
#     table.index.name = "Fração"

#     latex_str = table.to_latex(
#         escape=False,
#         caption=f"Tempo de seleção (segundos) -- {dataset}",
#         label=f"tab:{dataset}_selection_time",
#         position="htbp",
#     )

#     fname = tex_dir / f"{dataset}_selection_time.tex"
#     with open(fname, "w") as f:
#         f.write(latex_str)
#     print(f"Tabela salva: {fname}")


# %%
# %%
