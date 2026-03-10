# Dataset Preprocessing — Technical Reference

## Problem: Data Leakage via Preprocessing

### Definition
Data leakage occurs when information from the test set influences the training pipeline.
A transformer that calls `.fit()` on the full dataset (before train/test split) leaks test statistics into training.

### Affected loaders (before fix)

| Loader | Leaking transform | Fix |
|--------|------------------|-----|
| `load_covtype_dataset` | `normalize(dataset.values)` + `PCA(15).fit_transform(dataset)` | PCA + normalize → Pipeline |
| `load_bike_share_dataset` | `minmax_scale(dataset[["casual","registered"]])` | MinMaxScaler → Pipeline |
| `load_sgemm_dataset` | target = (n,4) — incompatible with sklearn estimators | target = mean(Run1..Run4) |
| `load_adult_dataset` | NaN from `replace(" ?", np.nan)` not handled | dropna() before return |

### Correct pattern (sklearn Pipeline)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.decomposition import PCA

# In datasets.py — loader returns raw data + pipeline
def load_covtype_dataset(config):
    # ... read CSV, fixed encoding (no fit) ...
    pipeline = Pipeline([("normalize", FunctionTransformer(normalize)), ("pca", PCA(n_components=15))])
    return X_raw, y, pipeline

# In main.py — pipeline fit ONLY on train split
train_feat = pipeline.fit_transform(features[train_idx])
test_feat = pipeline.transform(features[test_idx])
```

### Reference
- Scikit-learn documentation: "Pipelines and composite estimators"
  https://scikit-learn.org/stable/modules/compose.html
- Kaufman et al. (2012) "Leakage in Data Mining: Formulation, Detection, and Avoidance"
  ACM Transactions on Knowledge Discovery from Data, 6(4).

## Interface Contract

```python
def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    """
    Returns:
        X : np.ndarray, shape (n, d) — raw features (no learned transforms applied)
        y : np.ndarray, shape (n,)   — target (univariate, dtype appropriate for task)
        pipeline : sklearn Pipeline  — transforms to fit on train, apply to train+test
                   Pipeline([]) for datasets with no learned transforms
    """
```

## Per-dataset pipeline specification

| Dataset | Pipeline contents | Target |
|---------|------------------|--------|
| covtype | FunctionTransformer(normalize) → PCA(n_components=15) | cover_type - 1 (int, classes 0-6) |
| bike_share | MinMaxScaler on [casual, registered] columns | cnt (float) |
| adult | none (dropna applied at load time) | income >50K (binary int) |
| sgemm | none | mean(Run1, Run2, Run3, Run4) (float, ms) |
| hepmass | none | label (binary int) |
| predictmds | none | year (float) |
| storage_perf | none | throughput (float) |
| higgs | none | label (binary int) |

## main.py integration

```python
# cmd_select_coreset and _train_on_coreset:
features, target, pipeline = load_dataset(dataset_name)

# After split:
train_feat = pipeline.fit_transform(features[train_idx])
test_feat = pipeline.transform(features[test_idx])

# For coreset selection, pass train_feat (already transformed):
elapsed, indices = sampler_fn(train_feat, K=K, **sampling_args)
```
