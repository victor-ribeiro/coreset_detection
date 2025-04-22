# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

result = Path().rglob("*.csv")
result = map(pd.read_csv, result)
result = pd.concat(result)
result.index = range(len(result))
result.method = result.method.fillna("full dataset")
result.selection_time = result.selection_time.fillna(0)
result = result[(result.frac < 0.5) | (result.frac == 1)]

result["total_time"] = result.train_time + result.selection_time

import seaborn as sns

sns.pointplot(result, x="frac", y="test", hue="method")
plt.show()

sns.pointplot(result, x="frac", y="val", hue="method")
plt.show()


# sns.pointplot(result, x="frac", y="total_time", hue="method")
# plt.show()


# sns.pointplot(result, x="frac", y="selection_time", hue="method")
# plt.show()
# %%
