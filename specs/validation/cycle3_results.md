# Ciclo 3 — Protocolo Experimental: Resultados

## Bugs Corrigidos

| Bug | Arquivo | Linha | Correção | Status |
|-----|---------|-------|---------|--------|
| B1: random_state fixo | main.py | 64 | `random_state=42` → `random_state=run_idx` | ✅ |
| B2: colunas erradas | hypothesis_test.py | 134-164 | `method/metric/frac/test` → `metodo/metrica/fracao/valor` | ✅ |
| B3: path sem subpastas | main.py | 125 | `outputs/{name}/` → `outputs/{name}/{model}/{dataset}/` | ✅ |
| B4: train_rep ausente | main.py | 190 | Adicionado `"train_rep": i` no registro CSV | ✅ |

## Melhorias Adicionais (solicitadas pelo usuário)

| Melhoria | Arquivo | Descrição |
|---------|---------|-----------|
| Reprodutibilidade | main.py:167-171 | `random_state = run_idx * 5 + i` nos modelos de treino |
| Validação de colunas | hypothesis_test.py:82-92 | `ValueError` explícito se CSV tem colunas erradas |
| Float precision | hypothesis_test.py:98 | `fracao.round(4)` após `pd.concat` |

## Schema CSV Final

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| dataset | str | Nome do dataset |
| metodo | str | Método de seleção de coreset |
| fracao | float (4 dec) | Fração do dataset selecionada |
| metrica | str | Nome da função de métrica |
| valor | float | Valor da métrica |
| modelo | str | Classe do modelo sklearn/xgb |
| run | int | Índice do run de coreset (0..runs-1) |
| train_rep | int | Índice da repetição de treino (0..4) |
| train_time | float | Tempo de treino em segundos |
| selection_time | float | Tempo de seleção do coreset em segundos |

## Path de Saída

```
outputs/{name}/{model}/{dataset}/{dataset}_{model}_{metodo}_{fracao}.csv
```

hypothesis_test.py espera: `outputs/{experiment}/{model}/{dataset}/`

## Testes

- 15/15 testes automatizados passando (test_protocol.py, 0.61s)
- Teste manual: índices diferentes entre runs confirmados (train_0 ≠ train_1 ≠ train_2)
- Gap: sem teste de integração end-to-end — a incluir em ciclo futuro

## Limitação Conhecida

- dados em `coreset/` gerados antes do fix (random_state=42) ficam "órfãos" — re-rodar `select-coreset` para produzir dados com a correção completa
- `paired_ttest` trunca séries de tamanhos diferentes como guard — não é o procedimento estatístico correto; se len(b) ≠ len(m), indica bug upstream nos dados
