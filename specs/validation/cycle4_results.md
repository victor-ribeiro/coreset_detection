# Ciclo 4 — Revisão de Datasets: Resultados

## Problemas Corrigidos

| Bug | Arquivo | Correção | Status |
|-----|---------|---------|--------|
| D1: data leakage covtype | utiils/datasets.py | PCA + normalize removidos do loader → get_pipeline("covtype") | ✅ |
| D2: data leakage bike_share | utiils/datasets.py | minmax_scale removido do loader → get_pipeline("bike_share") ColumnTransformer cols 10,11 | ✅ |
| D3: sgemm target (n,4) | utiils/datasets.py | target = mean(Run1..Run4), 1D scalar | ✅ |
| D4: adult NaN não tratado | utiils/datasets.py | dropna() após replace(" ?", nan) | ✅ |

## Nova Função

```python
get_pipeline(name: str) -> sklearn.Pipeline
```

- Preserva interface `load_dataset()` sem breaking change
- `covtype`: `FunctionTransformer(normalize) → PCA(n_components=15, random_state=42)`
- `bike_share`: `ColumnTransformer(MinMaxScaler(), cols=[10, 11])` — casual=col10, registered=col11
- Demais datasets: `Pipeline([("passthrough", FunctionTransformer())])` — no-op

## Integração main.py

Três funções atualizadas para aplicar pipeline após o split:

```python
pipeline = get_pipeline(dataset_name)
train_feat = pipeline.fit_transform(features[train_idx])   # fit APENAS no treino
test_feat  = pipeline.transform(features[test_idx])        # sem fit no teste
```

Funções: `cmd_select_coreset`, `_train_on_coreset`, `_train_full_dataset`

## Decisão de Modelagem (freddy)

`_estimate_marginal_gain` usa `log(1 + (n/b) * Σmax(0, s - m_i))` — fator logarítmico modela ganhos marginais decrescentes. Não está na equação do paper; é decisão do usuário. Documentado em `specs/technical/freddy_algorithm.md`.

## Testes

- 30/30 testes automatizados passando (`test_datasets.py`: 14, `test_freddy.py`: 16)
- Teste manual confirmado pelo usuário: pipeline funcional em bike_share e covtype

## Limitação Conhecida

- `bike_share`: índices de coluna `casual=10, registered=11` são hardcoded em `get_pipeline`. Se a ordem das colunas mudar no loader, o pipeline quebrará silenciosamente. Solução futura: retornar índices do loader.
