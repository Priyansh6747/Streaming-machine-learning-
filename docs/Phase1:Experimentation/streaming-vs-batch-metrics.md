# Streaming vs Batch: Real Metrics Where Online Learning Wins (and Where It Doesn't)

**Purpose:** Evidence dossier + experimental protocol for the teacher's plan — "take some models that are not streaming and beat them on real metrics." Produced for the Delay Tax major project.

**Status:** ✅ Evidence verified (citations below); protocol to be run in Ch1.

---

## 1. The metrics that prove streaming learning works

Prequential (test-then-train) evaluation is the accepted protocol: for each sample, `predict_one(x) → metric.update(y, y_pred) → learn_one(x, y)` (Gama, Sebastião & Rodrigues, 2013, *Machine Learning* 90:317–346). All metrics below are computed inside that loop.

### Predictive quality metrics (per sample, over the stream)

| Metric | What it proves | River class |
|---|---|---|
| Prequential Accuracy | overall correctness on the live stream | `metrics.Accuracy` |
| **Kappa (Cohen)** | chance-corrected agreement — the headline metric; robust to class imbalance | `metrics.CohenKappa` |
| Rolling/windowed Kappa | recency-biased view — shows recovery after drift | `metrics.RollingCohenKappa` |
| AUC-ROC / AUC-PR | ranking quality; PR for rare classes (fraud) | `metrics.ROCAUC`, `metrics.PrecisionRecallAUC` |
| F1 / Recall / Precision | per-class utility (e.g., recall of the fraud class) | `metrics.F1`, `metrics.Recall` |

### Drift-response metrics (what batch models structurally can't report)

| Metric | What it proves |
|---|---|
| Detection delay (samples from drift onset to alarm) | speed of adaptation |
| False-alarm rate / mean time to false alarm (MTFA) | alarm cost |
| Missed-detection ratio (MDR) | reliability |
| **Recovery time** (samples from drift onset until rolling Kappa returns to pre-drift level) | the money metric — batch models *never* recover until the next scheduled retrain |

Metric definitions follow Cerqueira et al., KDD '26 (F1-detection score, normalized detection time; DOI 10.1145/3770855.3819070 — already noted in `literature_review/papers/kdd2026-drift-benchmark.md`).

### Resource metrics (the three streaming constraints from `notes/01`)

| Metric | Constraint |
|---|---|
| Memory footprint (bytes / model) | bounded memory |
| Throughput (samples/sec), latency per sample | bounded time |
| Number of passes over the data | single pass |

### Label-efficiency metric

| Metric | What it proves |
|---|---|
| Performance vs. fraction of labels used (l_access sweep) | streaming keeps working when labels are scarce (STUDD's headline) |

## 2. Evidence that streaming learning works — with numbers

Verified from the vault's literature index (✅ in `literature_review/index.md`):

- **ROSE** (Cano & Krawczyk, 2022, *Machine Learning* 111:2561–2599): streaming self-adjusting ensemble, **Kappa 63.65 vs next-best 57.85** across 24 imbalanced real datasets, 30 ensembles compared. Streaming wins on real, imbalanced, drifting data.
- **ARF** (Gomes et al., 2017, *Machine Learning* 106:1469–1495): adaptive operators + resampling cope with abrupt, gradual, and recurring drift "without complex optimizations" — the flagship streaming learner.
- **HAT** (Bifet & Gavaldà, 2009): ADWIN-per-node drift handling; superior adaptability on Adult and Poker-Hand (the origin of "adaptive beats static").
- **STUDD** (Cerqueira et al., 2023, *Machine Learning* 112:4351–4378): label-free early warning; **best label efficiency among non-baseline methods** on 19 benchmark streams; MDR ~19%. Batch retraining starves without labels; streaming detectors don't.
- **KDD '26 drift benchmark** (Cerqueira et al., 2026): 14 detectors × 7 real datasets; gives us the protocol and metrics to make our comparisons rigorous.
- **Botacin & Gomes** (2024, *Computers & Security* 147): delayed labels (the Delay Tax) raise user exposure ~50% and cut detection up to 75% — the production cost that adaptive streaming systems mitigate.
- **Gomes et al.** (2022, ACM Computing Surveys 55(4):75): a 10,000-instance label delay on AGRAWAL shifts detections by ~10,000 instances and keeps prequential accuracy below 50% — the reaction-floor effect our project formalizes.

## 3. Non-streaming models we can beat (and the honest caveat)

### The counter-evidence first (do not cherry-pick)

**Amekoe et al. 2024** (arXiv:2409.10111) — the most direct batch-vs-streaming study:
- No-delay synthetic benchmark (5 streams, 1M samples, Optuna-tuned): retrained XGBoost `r_XGBoost` **Avg AUC-ROC 92.81** vs ARF **91.02**; batch wins, slightly.
- Real fraud data (6.5M IBANs, 0.1% fraud, delayed labels): `propagate_XGBoost` **AUCPR 23.41** vs ARF **14.23**; batch wins clearly under delay + extreme imbalance.
- ARF runtime ~**2.2×** r_XGBoost (3,702 s vs 1,689 s).

**Implication for our claims:** we cannot promise "streaming beats tuned, frequently-retrained batch" — the literature says otherwise under delay and rare-class conditions. That is *fine*: it is exactly why our project's gap (delay-aware adaptation) exists. Our defensible wins are below.

### Batch models we beat, and on what metric

| Batch model (non-streaming) | Setting we beat it in | Metric where we win |
|---|---|---|
| Frozen RF / XGBoost / LR / SVM / DT (train once on warm-up window, deploy, never update) | Any drifting stream | Accuracy/Kappa after drift onset; **recovery time**; AUC |
| Scheduled-retrain batch (refit every k samples, no drift detection) | Between retrain points, and drift occurring right after a retrain | Rolling Kappa dip depth + recovery time; detection delay |
| Batch models needing all labels before retraining | Label-scarce streams | Label efficiency (STUDD-style l_access sweep) |
| Any batch model on unbounded streams | Memory | Bounded-memory footprint (batch stores data; streaming doesn't) |

### Why these wins are structural, not cherry-picked

- A **frozen** model has no mechanism to react to drift; a streaming model with a detector has a bounded reaction time. This is the Delay Tax's Lemma 1 (reaction floor) in reverse: streaming's floor is `T ≥ τ + L_τ`; a frozen batch model's "recovery" is undefined (never).
- A **scheduled** retrain reacts on a fixed clock, not on evidence; streaming reacts to detected change, so it recovers faster and spends less compute when nothing changes.

## 4. Protocol to prove it (Ch1 experiments)

1. **Pin the stack:** Python ≥3.11, pin River version (per `notes/03-river-library.md`), scikit-learn for batch baselines.
2. **Datasets:** real streams — Elec2, Phishing, Forest Covertype, Airlines, CreditCard fraud (report Elec2 Kappa+ caveats); synthetic with known drift timing — SEA, Agrawal, Hyperplane, RandomRBF, STAGGER, LED (for ground-truth detection-delay/recovery measurement).
3. **Streaming models:** Hoeffding Tree, HAT, ARF, SRP, Leveraging Bagging, kNN(+ADWIN), online LR.
4. **Batch baselines:** (a) *frozen* — fit on first 20–30%, predict-only afterwards; (b) *scheduled* — refit every k on a sliding window, no drift detection; (c) *oracle-tuned* (XGBoost/EBM with Optuna) — reported as the honest upper bound (Amekoe-style).
5. **Seeds & statistics:** ≥10 seeds on synthetic; real ordered streams use bootstrap windows; Friedman + Nemenyi across datasets; Wilcoxon signed-rank pairwise; report effect sizes (per `notes/04-evaluation.md`).
6. **Outputs:** per-metric tables; accuracy-over-time / rolling-Kappa-over-time curves with drift onsets marked (Amekoe Fig. 1 style); recovery-time and detection-delay tables; memory/throughput columns. The "win card": one claim per metric, with the number and the dataset it came from.

## 5. Expected claims (the win card, preview)

- Streaming models recover rolling Kappa to pre-drift level within a bounded number of samples on every synthetic drift; frozen batch models never do before the stream ends.
- On real streams with drift, HAT/ARF rolling Kappa ≥ frozen RF/XGBoost on the post-drift window (supported by ROSE 63.65 vs 57.85 Kappa and HAT's ADWIN results).
- Streaming matches scheduled-retrain batch accuracy while using 1 pass and bounded memory; and with drift detectors, reacts between retrain points where batch cannot.
- Under label scarcity, streaming + detectors hold performance where batch retraining starves (STUDD label-efficiency result).

## 6. Risks / what could sink the claim (and mitigations)

- **Amekoe-style tuned batch beats streaming on delay/rare-class** → our headline is *frozen* and *scheduled* batch, reported alongside oracle-tuned batch as an upper bound; never overclaim.
- **ARF is compute-expensive** → use HT/HAT/LB as the cheap streaming proof; report ARF as the accuracy ceiling with its cost stated.
- **Elec2 Kappa+ collapse** → prefer rolling Cohen Kappa + AUC-PR; treat Elec2 as secondary.
- **Synthetic-only results** → real datasets carry the headline; synthetic only provides ground-truth drift timing for delay/recovery metrics.
- **Teacher requires novelty** → the note is evidence for Ch1 (baseline chapter); the novelty stays in Ch3 (delay tax). The comparison harness itself is a deliverable.

## Conclusion — groundwork update

**What we're proving:** streaming learning beats non-streaming models on real, drifting, label-scarce data.

- **Models — streaming:** Hoeffding Tree, HAT, ARF, SRP, Leveraging Bagging, kNN(+ADWIN), online LR (River). **Batch to beat:** frozen (RF/XGBoost/LR/SVM/DT, train-once) and scheduled-retrain (refit every k, no drift detector); oracle-tuned XGBoost/EBM reported as honest upper bound.
- **Metrics:** prequential accuracy, Cohen Kappa, rolling Kappa, AUC-ROC/PR, F1/recall, detection delay, MTFA, MDR, recovery time, memory, throughput, label efficiency.
- **How:** prequential test-then-train loop; ≥10 seeds, Friedman + Nemenyi, Wilcoxon; real datasets (Elec2, Phishing, Covertype, Airlines, CreditCard) + synthetic with known drift (SEA, Agrawal, Hyperplane, RBF, STAGGER, LED); win card = one claim per metric, with dataset + number.
- **Where we win:** frozen batch → decisively after drift; scheduled batch → recovery time + compute between retrains. **Caveat:** tuned frequent-retrain batch beats streaming under delay + rare classes — that is the Delay Tax gap (Ch3).

## Sources

- Amekoe, K. M., Lebbah, M., Jaffre, G., Azzag, H., & Dagdia, Z. C. (2024). Evaluating the Efficacy of Instance Incremental vs. Batch Learning in Delayed Label Environments. arXiv:2409.10111. https://arxiv.org/abs/2409.10111
- Cano, A., & Krawczyk, B. (2022). ROSE. *Machine Learning* 111:2561–2599. https://doi.org/10.1007/s10994-022-06168-x
- Cerqueira, V., Gomes, H. M., Heyden, M., Pfahringer, B., & Bifet, A. (2026). A Framework for Evaluating and Benchmarking Concept Drift Detection Methods. KDD '26. https://doi.org/10.1145/3770855.3819070
- Cerqueira, V., Gomes, H. M., Bifet, A., & Torgo, L. (2023). STUDD. *Machine Learning* 112:4351–4378. https://doi.org/10.1007/s10994-022-06188-7
- Bifet, A., & Gavaldà, R. (2009). Learning from Time-Changing Data with Adaptive Windowing / HAT. IDA. https://doi.org/10.1007/978-3-642-03915-7_22
- Gomes, H. M., et al. (2017). Adaptive random forests for evolving data stream classification. *Machine Learning* 106:1469–1495. https://doi.org/10.1007/s10994-017-5642-8
- Botacin, M., & Gomes, H. (2024). Towards More Realistic Evaluations: The Impact of Label Delays in Malware Detection Pipelines. *Computers & Security* 147. https://www.sciencedirect.com/science/article/abs/pii/S0167404824004279
- Gomes, H. M., et al. (2022). A Survey on Semi-supervised Learning for Delayed Partially Labelled Data Streams. ACM Computing Surveys 55(4):75. https://doi.org/10.1145/3523055
- Gama, J., Sebastião, R., & Rodrigues, P. P. (2013). On evaluating stream learning algorithms. *Machine Learning* 90:317–346. https://doi.org/10.1007/s10994-012-9334-7
