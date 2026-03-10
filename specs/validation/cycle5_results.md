# Ciclo 5 — Coverage Metric: Resultados

## Entregável

Nova métrica `coverage_mean` adicionada ao pipeline experimental.

## Implementação

| Módulo | Mudança |
|--------|---------|
| `main.py::compute_coverage_mean(X_train, coreset_feat)` | Nova função pura — NearestNeighbors(n=1), guard K=0→nan |
| `main.py::cmd_select_coreset` | Chama `compute_coverage_mean` após seleção, salva em `metadata["coverage_mean"]` |
| `main.py::_train_on_coreset` | Lê `metadata.get("coverage_mean", nan)`, propaga para CSV |
| `main.py::_train_full_dataset` | `coverage_mean = float("nan")` — sem coreset |

## Definição

```
coverage_mean(S, X_train) = (1/n) * Σ_i min_j d(x_i, c_j)
```

- S = coreset (K pontos), X_train = conjunto de treino completo
- d = distância Euclidiana
- Menor = melhor cobertura geométrica

Ref: Har-Peled & Mazumdar (2004), STOC.

## Schema CSV Final

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| ... (colunas anteriores) | | |
| coverage_mean | float\|nan | Raio médio de cobertura do coreset sobre X_train. nan para method=none. |

## Testes

- 53/53 testes automatizados passando (`test_coverage.py`: 8, `test_datasets.py`: 14, `test_freddy.py`: 16, `test_protocol.py`: 15)
- Manual testing confirmado: coverage_mean no metadata JSON e CSV

## Limitação Conhecida

- Para datasets grandes (higgs: n=11M), `kneighbors(X_train)` pode ser lento — aceito como limitação documentada
- CSVs gerados antes do Ciclo 5 não terão a coluna `coverage_mean` (breaking change documentado, backward compat via `.get()`)
