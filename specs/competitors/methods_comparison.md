# Competitor Methods

Source: Table 1, Ribeiro (2026); Mirzasoleiman et al. (2020); Killamsetty et al. (2021a, 2021b)

## Methods compared in experiments

| Method | Category | Key Idea | Reference |
|--------|----------|----------|-----------|
| Random | Training-Free | Uniform random sampling (baseline) | — |
| CRAIG | Training-Oriented (A Posteriori) | Facility Location + lazy greedy over weighted gradients | Mirzasoleiman et al., ICML 2020 |
| GradMatch | Training-Oriented (A Posteriori) | Orthogonal Matching Pursuit on gradients | Killamsetty et al., ICML 2021a |
| FREDDY | Training-Free (A Priori) | Stochastic greedy on similarity (Facility Location + ICEL), mini-batch Monte Carlo | Ribeiro 2026 |
| k-Means | Training-Free (A Priori) | Cluster centroids | — |

## FREDDY vs CRAIG

FREDDY is positioned as a variation of CRAIG (Mirzasoleiman et al. 2020):
- CRAIG: pairwise distances over full dataset → O(n²) — expensive preprocessing
- FREDDY: mini-batch random sampling → O(n × b) — scalable; 21× speedup reported vs CRAIG
- Both maximize Facility Location submodular function
- FREDDY adds ICEL regularization for geometric diversity
