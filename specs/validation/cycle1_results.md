# Cycle 1 — Validation Results

## Deliverables vs Results

| Deliverable | Expected | Obtained | Status |
|-------------|---------|---------|--------|
| freddy() retorna K | len(result) == K sempre | 16/16 testes, manual OK | ✅ |
| Loop externo while |S|<K | Iteração até atingir K | Implementado com fallback | ✅ |
| m_i global persistente | zeros(n), persiste entre iters | Confirmado por test_coverage_update_* | ✅ |
| Eq.5 scaling n/b | (n/b)*sum(max(0, s-m_i)) | Verificado test_marginal_gain_scaling | ✅ |
| Eq.7 update in-place | m_i[B] = max(m_i[B], sim) | Verificado test_coverage_update_inplace | ✅ |
| Interface pública mantida | freddy(dataset,K,...)->ndarray | __init__.py importa corretamente | ✅ |

## Bug descoberto em P6 (edge case)

- K/n alto (K≈n) causa loop infinito no outer while
- Solução: max_outer = max(K*10, 1000) + fallback aleatório
- Comportamento esperado: FREDDY assume K<<n (paper Section 4.1)

## Scope out confirmed

- kmeans_sampler, pmi_kmeans_sampler: removidos (não eram Algorithm 1)
- hypothesis_test.py: não modificado (escopo futuro)
- run.sh: não modificado (escopo futuro)
