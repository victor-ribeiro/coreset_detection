# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

result = Path().rglob("*.csv")
result = map(pd.read_csv, result)
result = pd.concat(result)
result.index = range(len(result))
result.method = result.method.fillna("full dataset")
result.loc[result.method == "full dataset", "frac"] = 1

result.selection_time = result.selection_time.fillna(0)

result["total_time"] = result.train_time + result.selection_time

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# resul = result[(result.metric == "recall_score")]
fig.suptitle("bike share DATASET", fontsize=16)
sns.barplot(result, x="frac", y="test", hue="method", ax=ax[0])
ax[0].set_title("Test RMSE")
ax[0].set_xlabel("Fraction of data", fontsize=13)
ax[0].set_ylabel("RMSE", fontsize=13)
# ax[0].set_ylim(0.5, 1)
ax[0].grid(axis="y")
sns.barplot(result, x="frac", y="selection_time", hue="method", ax=ax[1])
ax[1].set_title("Selection Time")
ax[1].set_xlabel("Fraction of data")
ax[1].set_ylabel("Time (s)")
ax[1].set_yscale("log")
ax[1].grid(axis="y")

plt.show()


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("COVTYPE DATASET", fontsize=16)
sns.barplot(result, x="frac", y="val", hue="method", ax=ax[0])
ax[0].set_title("Validation Accuracy")
ax[0].set_xlabel("Fraction of data", fontsize=13)
ax[0].set_ylabel("Accuracy", fontsize=13)
# ax[0].set_ylim(0.5, 1)
ax[0].grid(axis="y")
sns.barplot(result, x="frac", y="total_time", hue="method", ax=ax[1])
ax[1].set_title("Selection Time")
ax[1].set_xlabel("Fraction of data", fontsize=13)
ax[1].set_ylabel("Time (s)", fontsize=13)
ax[1].set_yscale("log")
ax[1].grid(axis="y")
plt.show()
# %%
