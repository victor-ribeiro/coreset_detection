# FREDDY Algorithm — Technical Specification

Source: Ribeiro, V. "FREDDY: Fast Reduction of Elements for Data-driven Yielding" (2026)
File: specs/references/FREDDY__Fast_Reduction_of_Elements_for_Data_driven_Yielding.pdf

## Objective Function (Eq. 1)

F(S) = F_FL(S) + λ * F_ICEL(S)

- F_FL: Facility Location — global representativeness
  F_FL(S) = Σ_{x_i ∈ X} max_{x_j ∈ S} s(x_i, x_j)
- F_ICEL: Intrinsic Coverage Estimation Loss — geometric diversity
  F_ICEL(S) = Σ_{x_j ∈ S} u_j  (u_j = expected distance to other points, geometry-only)
- λ ≥ 0: trade-off parameter

## Marginal Gain Estimation (Eq. 5, 6)

For candidate e ∉ S_t, estimated using mini-batch B_t:

  Δ̂_FL(e | S_t) = (n / |B_t|) * Σ_{x_i ∈ B_t} max(0, s(x_i, x_e) - m_i(S_t))

  Δ̂(e | S_t) = Δ̂_FL(e | S_t) + λ * u_e

## Implementation variant (freddy.py — log-modulated gain)

The implementation applies a logarithmic modulation to the marginal gain:

  gain = log(1 + Δ̂_FL(e | S_t))

Rationale: log(1+x) is monotone increasing, so greedy ordering is preserved.
Effect: compresses large gains, reduces sensitivity to outlier similarities.
This is a modeling decision (not in the paper) — introduced to better model
diminishing returns in the facility location objective.

## Coverage State Update (Eq. 7)

After selecting e:
  m_i(S_{t+1}) ← max(m_i(S_t), s(x_i, x_e))  ∀x_i ∈ B_t

## Algorithm 1 — Pseudocode

Input: X = {x_1,...,x_n}, budget K, batch size b, λ
Output: Selected subset S

Initialize S ← ∅
Initialize coverage state m_i ← 0 for all x_i ∈ X

while |S| < K do:
    Sample mini-batch B ⊂ X uniformly at random, |B| = b
    Initialize priority queue Q with elements in B
    while Q not empty AND |S| < K do:
        Extract candidate e with highest stored priority from Q
        Estimate marginal gain Δ̂(e | S) using B (Eq. 5+6)
        if e has highest estimated gain then:
            S ← S ∪ {e}
            Update m_i ← max(m_i, s(x_i, x_e)) for all x_i ∈ B
        else:
            Update priority of e in Q

return S

## Key Design Principles

1. Mini-batches = Monte Carlo estimators (NOT domain restriction)
2. Coverage state m_i is GLOBAL and persists across all iterations
3. Mini-batches are sampled RANDOMLY at each outer iteration
4. Lazy greedy: re-evaluate candidate only when at top of queue

## Implementation Divergences in freddy.py (current)

| Aspect | Algorithm 1 | freddy.py current |
|--------|-------------|-------------------|
| Outer loop | while |S| < K | for batch in sequential_batches |
| Mini-batch | random sample each iteration | deterministic sequential partition |
| Coverage state | m_i global, persistent | localmax reset per batch |
| Result | always returns exactly K | may return << K |
