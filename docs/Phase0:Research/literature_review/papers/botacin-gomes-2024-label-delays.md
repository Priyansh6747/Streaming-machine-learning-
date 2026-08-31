# Towards More Realistic Evaluations: The Impact of Label Delays in Malware Detection Pipelines

**Full citation:** Botacin, M., & Gomes, H. (2024). Towards more realistic evaluations: The impact of label delays in malware detection pipelines. *Computers & Security*, 147 (per ScienceDirect pii S0167404824004279). Author-hosted PDF: https://marcusbotacin.github.io/files/label-delays.pdf · Code: https://github.com/marcusbotacin/ML.delay.experiments

**Status:** ✅ Verified (full text read; COSE 2024 confirmed via GitHub + ScienceDirect)

---

## Why this paper matters

This is the **closest published empirical treatment of our exact Ch3 problem**:
it measures, with real malware data and real drift detectors (DDM/EDDM), what label
delay does to a drift-retraining pipeline — and it proposes the closest existing
thing to our hybrid (a pseudo-label secondary classifier as an early signal).

## What they did

- **Setting:** malware detection modeled as a *delayed and partially labelled
  evolving data stream* (explicitly citing the Gomes 2022 survey).
- **Pipelines:** Naive Triage (no retraining), Triage-Update (drift detector
  triggers retrain), Triage-Oracle-Update (sandbox = oracle), and
  Triage-Oracle-**Eager**-Update (secondary classifier produces **pseudo-labels**
  to bridge the delay before the oracle's ground truth arrives).
- **Data:** DREBIN (Android malware, temporally ordered by first-seen), 250 epochs
  × 500 samples; replicated on Androzoo (~284k samples, ~270 epochs).
- **Detectors:** DDM and EDDM, epoch-level drift checks.
- **Classifier:** Random Forest, FPR < 1% enforced, 10 executions averaged.
- **New metric:** the *exposure* metric — cumulative sum of False Negatives over
  samples and days: `Exp(samples; days) = Σ_day Σ_sample FN(s, d)`. Non-decreasing;
  captures how long a user is exposed to a missed sample.

## Key numbers (all from the paper)

| Finding | Number |
|---|---|
| Drift mitigation reduces exposure | ~50% |
| Delayed labels increase exposure | ~50% (up to 75% detection decrease) |
| Delay at which exposure +50% vs immediate (DDM) | 81 epochs (~⅓ of stream) |
| Delay at which exposure +50% vs immediate (EDDM) | 61 epochs |
| Exposure increase per 10 epochs of delay | ~50k points (DDM), ~60k (EDDM) |
| Pseudo-labels reduce detection loss | 30% loss instead of 70% loss |
| 10% queue capacity ≈ equivalent delay | 100+ epochs (oracle-only) |
| Queue capacity with same exposure using pseudo-labels | down to 70% (vs 100%) |
| Metrics comparison (Table 4) | AUT-F1 0.52→0.18 as delay 0→80; Exposure 0.49→0.90 |

Pseudo-label detail: with 21 epochs delay, mitigation strong; effect weakens to
nil at 45 epochs, then reappears for larger delays (secondary model's own
predictions become relevant again). Updating the secondary model on drift points
matters (un-updated secondary degrades over time).

## How it differs from Ch3 / our lemmas

- **Empirical, not formal.** It demonstrates the reaction floor (Lemma 1) but
  proves no bound, and does not touch power loss (Lemma 2), batch autocorrelation
  (Lemma 3), or identifiability (Lemma 4).
- **Application-specific.** DREBIN/Androzoo + DDM/EDDM only; no synthetic
  ground-truth drift positions, so no clean detection-delay curve vs label
  latency.
- **Pseudo-labels, not a two-stage hybrid.** Their eager-update secondary
  classifier is *a* label-free early signal, but there is no explicit
  confirmation protocol, no warning-within-tolerance-window machinery, and no
  calibration of false-alarm cost of unconfirmed warnings.
- Their **exposure metric** is a strong candidate for how we should frame Ch2/Ch3
  "reality tax" results (time-aware, cumulative) — and we must cite it when
  introducing any time-aware metric.

## What Ch3 must NOT claim as novel

- "Delayed labels measurably degrade drift-based retraining" — already quantified
  here.
- "A secondary/label-free signal mitigates delay" — already shown (pseudo-labels).
- "You need time-aware metrics" — already argued (exposure vs AUT).

**What remains open (our space):** formal reaction-floor bound; power-loss law vs
label rate on detectors in isolation; false-alarm inflation from batch arrival;
identifiability ceiling; a *confirmation protocol* (early warning → delayed
supervised confirmation → calibrated action) evaluated against both naive
delayed-supervised and label-free-only baselines.
