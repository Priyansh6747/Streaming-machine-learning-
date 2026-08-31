# ROSE: Robust Online Self-Adjusting Ensemble for Continual Learning on Imbalanced Drifting Data Streams

**Full citation:** Cano, A., & Krawczyk, B. (2022). ROSE: Robust online self-adjusting ensemble for continual learning on imbalanced drifting data streams. *Machine Learning*, 111(7), 2561–2599. https://doi.org/10.1007/s10994-022-06168-x

**Status:** ✅ Verified (full text read from Material/)

---

## Why this paper matters

ROSE is a strong reference for **how modern adaptive ensembles handle drift +
imbalance**, and a useful comparator/design source for our Ch2 system chapter and
the adaptation side of Ch3. It is not a delayed-label paper, but its
background-ensemble + per-subspace-drift-detection pattern is exactly the kind of
adaptation machinery our Ch2 service would run.

## Method (4 features)

1. **Variable-size random feature subspaces** per base classifier (normal
   distribution centered on μ·f, μ=0.7), giving diversity + noise robustness.
2. **ADWIN drift detection per subspace + background ensemble**: on an ADWIN
   warning, a background ensemble is trained on the most recent per-class
   windows; after 1000 instances the top-k classifiers (by accuracy×Kappa) from
   the union replace the current ensemble.
3. **Sliding window per class** (total budget 1000 instances, so
   1000/num_classes each) — counters imbalance and, notably, *latency of class
   arrival*.
4. **Self-adjusting λ for bagging**: `λ = λ_min + log10(#majority/#instance
   class) × λ_min`, λ_min=4; balances exposure of minority instances.

Complexity: O(2·k·λ·|S|) time; memory O((2·k·r·v·l·c) + (|w|·f)).

## Key numbers

Static class imbalance (Kappa, avg over 6 generators):

| IR | ROSE | KUE | ARF | LB | SRP |
|---|---|---|---|---|---|
| 5 | 79.80 | 73.87 | 73.73 | 73.01 | 63.71 |
| 100 | 47.07 | 40.52 | 33.87 | 37.40 | 29.02 |
| Avg | 63.65 | 57.85 | 54.20 | 55.12 | 43.77 |
| Rank | 2.00 | 4.84 | 5.46 | 3.86 | 5.81 |

- Drifting IR (Kappa): ROSE 59.95 (rank 2.23) vs KUE 53.24, ARF 49.08, LB 54.45.
- Noise (Kappa, 0→40%): ROSE 59.95→36.54 (rank 1.70) — most robust.
- Instance-level difficulty (Kappa avg): ROSE 71.47 (rank 2.72), SRP 65.02.
- 24 real-world imbalanced streams: ROSE best on Kappa; SRP best on plain
  accuracy (metric matters!).
- Ablations: buffer 50–500 stable; epochs 1→10 raises F1 0.965→0.983; K-means vs
  DBSCAN 0.973 vs 0.965.

## How it differs from Ch3 / our lemmas

- **Immediate labels assumed** throughout; no delay, no partial labels.
- Drift handling is implicit+explicit but *not* the object of study — no
  detector-level delay/FP/FN metrics.
- The per-class sliding window is motivated partly by "latency of class
  arrival" in real streams — an interesting hook connecting imbalance to label
  availability, worth citing in our selective-labeling discussion.
- Strong methodological template: 30 ensembles, 24 real datasets, Friedman ranks,
  ablation study — a model for how we should present Ch1/Ch3 results.

## Recommended use

- Ch2: reference architecture for drift-triggered ensemble adaptation in a
  service.
- Ch3 (adaptation side): a state-of-the-art adaptive baseline to include when we
  measure how delayed labels degrade *adaptation* (not just detection).
- Methodology: emulate their experimental rigor (Friedman + ablation).
