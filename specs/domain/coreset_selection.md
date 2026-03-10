# Coreset Selection — Domain Overview

Source: Ribeiro (2026), Moser et al. (2025), Weinreich et al. (2025)

## Taxonomy (Moser et al. 2025)

| Paradigm | Description |
|----------|-------------|
| Training-Free | Use intrinsic dataset properties (geometry/distribution). No labels needed. |
| Training-Oriented | Rely on information from a trained auxiliary model (loss values, gradients). |
| Blind Coreset Selection | Complement training-oriented; use pseudo-labels for importance estimation. |

## Classification (Weinreich et al. 2025)

- A Priori: selection before model inference (data-driven or model-oriented)
- A Posteriori: selection after model inference (creates model dependency)

## Selection Strategies

- Error-Based: difference between prediction and true label
- Confidence-Based: model uncertainty
- Information-Theoretic: information content per parameter
- Similarity-Based: distance/correlation metrics (FREDDY falls here)
- Distribution-Based: distribution characteristics of full dataset
- Learning-Based: adaptive via auxiliary models

## FREDDY Classification

- Paradigm: Training-Free
- Classification: Data-Driven (A Priori) — Similarity-Based
- Does NOT require labels, model training, or clustering
