# A Survey on Semi-supervised Learning for Delayed Partially Labelled Data Streams

**Full citation:** Gomes, H. M., Grzenda, M., Mello, R., Read, J., Le Nguyen, M. H., & Bifet, A. (2022). A Survey on Semi-supervised Learning for Delayed Partially Labelled Data Streams. *ACM Computing Surveys*, 55(4), Article 75, pp. 1–42. https://doi.org/10.1145/3523055

**Status:** ✅ Verified (full text read via ACM DL)

---

## Why this paper matters for The Delay Tax

This is **the anchor survey** for Ch3 and the single best "closest prior work" citation.
It formalizes the exact setting our project studies (delayed + partially labelled
streams), organizes the method landscape, and — crucially for us — includes an
illustrative experiment showing what delayed labels do to drift detection.

## Core contribution

1. **A unified problem setting.** Defines a data stream as tuples
   `{(x_k, ?)}` (unlabelled) and `{(·, y_k)}` (labelled), a temporal-mapping
   function `T(·)`, a *verification latency* `V(x_k, y_k) = T(y_k) − T(x_k)`, and four
   label-arrival regimes:
   - (i) Immediate + fully labelled (`T(y)−T(x) = 1`)
   - (ii) Delayed + fully labelled (`D` random, bounded)
   - (iii) Immediate + partially labelled (some labels never arrive)
   - (iv) **Delayed + partially labelled** (both) — the regime we target.
2. **Taxonomy of SSL methods for streams**: intrinsically SSL, self-training,
   learning by disagreement, representation learning, and unsupervised/SSL drift
   detection.
3. **Learning guarantees** (SLT-based) for SSL, with the explicit caveat that a
   fixed JPD assumption breaks under drift.
4. **Evaluation review**: test-then-train, prequential, continuous re-evaluation
   (Grzenda et al. 2020), Kappa+ / no-change baselines, and the need to report
   memory/time. They propose generating reduced partially-labelled streams by
   removing labels with probability `p_u`.

## Key numbers / empirical nugget (their small experiment)

Using AGRAWAL (100k instances, 3 abrupt drifts at 25k/50k/75k), a 10-learner SRP
ensemble with ADWIN per learner, they compare no-delay vs 10,000-instance label delay:

| Setting | Effect |
|---|---|
| No delay | High detection rate immediately after each drift |
| Delay = 10,000 | Detections still occur but **shifted by ~10,000 instances** |
| Prequential accuracy (delayed) | **Prolonged periods below 50%** on a balanced binary problem |

This is a direct empirical demonstration of our **Lemma 1 (reaction floor)**:
the detector cannot alarm before the delayed labels arrive. They do not prove a
bound, do not isolate the detector's power loss vs label rate (our Lemma 2), and
do not study batch-arrival autocorrelation (our Lemma 3).

## How it differs from what Ch3 proposes

- It is a **survey**; the experiment is illustrative, not a systematic detector
  benchmark under delay.
- No formal bounds (no reaction-floor theorem, no `1/√p` power law, no
  batch-correlation factor, no identifiability proof beyond citing Žliobaitė).
- No hybrid detector architecture (label-free early warning + delayed
  confirmation) is built or evaluated.
- Their stated open direction — "forget outdated labels once a drift has been
  detected" — is adjacent to our adaptation-side work.

## Formulas / bounds restated

- Verification latency: `V(x_k, y_k) = T(y_k) − T(x_k)`.
- They rely on Vapnik ERMP bounds with `n` = number of *labelled* instances:
  `P(sup|R_emp − R| > ε) ≤ 2N(F,2n)exp(−2nε²)`; the key qualitative point for us:
  **the effective sample size for any guarantee is the labelled count**, which is
  the same intuition that drives our Lemma 2 (power loss ∝ 1/√p).

## Connections to our project

- Cite as the canonical formalization of the setting (our scope doc already does).
- Their Table 1 (Žliobaitė's observability conditions) is the seed of our Lemma 4.
- Their evaluation recommendations (report memory/time; continuous re-evaluation;
  Kappa+) feed directly into our Ch1 and Ch3 methodology.
