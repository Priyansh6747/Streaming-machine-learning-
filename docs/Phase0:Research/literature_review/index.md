# Literature Review Index — The Delay Tax

Systematic catalog for the "Delay Tax" project (streaming ML drift detection under
delayed/partial/selective labels). Sorted newest-first within each section.

**Verified** = ✅ (URL resolves, authors/venue/year cross-checked) or ⚠️ (unverified — flagged why).

---

## Section 1 — Ch1: Detector baselines, prequential evaluation, River/MOA tooling

### Streaming vs batch / retraining-policy comparisons (Ch1 thesis evidence — added Sep 2026)

Direct answer to "has anyone beaten non-streaming models with streaming ones, or compared them on real metrics?" — yes, repeatedly; the honest summary is *context-dependent, no universal winner, streaming wins where it has a structural mechanism (per-sample updates), not by detector alone.*

### [When to Retrain: An Empirical Study of Retraining Policies for Streaming ML Under Concept Drift, Budget, and Latency Constraints](https://arxiv.org/abs/2608.19488)
- **Authors:** Sawan Dasari
- **Year / Venue:** 2026, arXiv:2608.19488 (cs.LG); code public
- **Chapter relevance:** Ch1 (primary — retraining-policy evidence for the thesis), Ch2 (secondary — latency/budget system model)
- **Core claim:** Under explicit budgets and training-plus-deployment latency, the retraining *policy* (periodic vs error-threshold vs ADWIN-triggered vs none) matters far less than whether the model learns incrementally per sample; without incremental updates, simple periodic retraining beats both reactive policies on abrupt and gradual drift, reactive policies winning only under recurring drift.
- **Key numbers:** 3,933 runs; 54 paired comparisons; without incremental updates policy choice separates post-drift accuracy by 15–55 pp; latency–budget queueing interaction silently halves effective budgets; immediate labels, linear learner.
- **Method / dataset:** Simulator with stale-model-serves-during-latency; 3 drift regimes × 3 budgets × ≤5 latency levels × 3 datasets × 2 learning modes.
- **Gap vs. our lemmas:** Scope-defining for Ch1: our claim must be about *streaming learners* (bounded recovery via per-sample updates), not drift detectors as batch-refit triggers. Immediate labels only — Delay Tax (Ch3) untouched. **Deep note:** `papers/dasari-2026-when-to-retrain.md`.
- **Verified:** ✅

### [Online versus batch retraining in one-step-ahead urban water demand forecasting under concept drift](https://doi.org/10.2166/ws.2026.134)
- **Authors:** Lukas Hans, Dietlind Zühlke
- **Year / Venue:** 2026, Water Supply 26(4):255–268
- **Chapter relevance:** Ch1 (primary — deployment-faithful online-vs-batch-retraining evidence), Ch2 (secondary — runtime/latency in ops)
- **Core claim:** Deployment-faithful prequential evaluation of incremental updating vs weekly batch retraining (rolling + expanding windows) on 6 years of hourly demand; an online averaging ensemble of three incremental learners achieves the lowest MAE/RMSE and mitigates high-error episodes; several expanding-window batch models are statistically indistinguishable in the milder-drift area, but the online ensemble is consistently faster (seconds vs minutes/hours) and keeps accuracy where batch baselines degrade.
- **Key numbers:** Graphical abstract: avg MAE 391.2, runtime 27.6 s; Diebold–Mariano tests; two drift areas (variance/level drift vs distributional/shape drift).
- **Method / dataset:** German utility, 6 years hourly consumption + meteorology; MAE/RMSE, DM tests, end-to-end runtime; online ensemble vs weekly batch retrain.
- **Gap vs. our lemmas:** Applied confirmation that online wins under stronger drift and on cost — but regression forecasting, not classification/detector metrics; no recovery-time or damage-area measurement (our Ch1 metrics).
- **Verified:** ✅

### [Digital twin-based anomaly detection under concept drift: A comparison between iterative batch learning and incremental learning approaches](https://doi.org/10.1016/j.eswa.2025.130062)
- **Authors:** Farah Abdoune, Maroua Nouiri, Olivier Cardin
- **Year / Venue:** 2026, Expert Systems with Applications 299:130062
- **Chapter relevance:** Ch1 (secondary — applied batch-vs-incremental comparison), Ch2 (secondary — digital-twin/ops context)
- **Core claim:** Within a manufacturing digital-twin anomaly-detection framework, compares (i) a statistical threshold baseline, (ii) iterative batch learning (Isolation Forests periodically retrained with operator feedback), and (iii) incremental learning (Half-Space Trees updating in real time) under concept drift.
- **Key numbers:** Headline numbers not yet extracted (full text not in vault).
- **Method / dataset:** Manufacturing digital twin; threshold baseline vs periodic-batch iForest vs incremental HST.
- **Gap vs. our lemmas:** Same paradigm axis as Ch1 but anomaly detection in a DT setting; supports the "drift favours incremental" applied story; no detector-level or delayed-label metrics.
- **Verified:** ✅ (Crossref + publisher record)

### [An Experimental Comparison of Batch and Online Machine Learning Algorithms](https://doi.org/10.1007/978-981-99-7007-0_9)
- **Authors:** Thomas Bartz-Beielstein, Lukas Hans
- **Year / Venue:** 2024, chapter 9 in *Online Machine Learning* (Springer, MLFA series), pp. 105–124
- **Chapter relevance:** Ch1 (secondary — methodology of BML-vs-OML experiments), Ch2 (secondary)
- **Core claim:** Book-chapter experimental studies comparing batch and online ML: study 1 predicts bike-sharing demand; study 2 compares on very large datasets in the presence of drift.
- **Key numbers:** n/a at abstract level (chapter).
- **Method / dataset:** Bike-sharing demand regression; large drifting datasets.
- **Gap vs. our lemmas:** Same-lab companion to the Water Supply 2026 study (Hans co-author); classification/metric framing and recovery-time measurement still ours.
- **Verified:** ✅

### [All Versus One: An Empirical Comparison on Retrained and Incremental Machine Learning for Modeling Performance of Adaptable Software](https://dl.acm.org/doi/10.1109/SEAMS.2019.00029)
- **Authors:** Tao Chen
- **Year / Venue:** 2019, SEAMS (14th Intl Symposium on Software Engineering for Adaptive and Self-Managing Systems)
- **Chapter relevance:** Ch1 (secondary — "retrained is not always more accurate" evidence), Ch2 (secondary — software/cloud domain)
- **Core claim:** First large study comparing full-retrain ("all data, discard old model") vs truly incremental ("tune on each new sample") versions of the *same* algorithms; refutes the general belief that retrained ⇒ better accuracy: incremental wins on highly fluctuated data (and MLP), retrained wins on stable data (and SVM); incremental is always faster (15% to 3 orders of magnitude) and can dominate on both axes for 3 of 5 performance indicators.
- **Key numbers:** 8 learning algorithms (LR/DT/SVM/MLP + bagging/boosting), 5 performance indicators, 3 real adaptable software systems, 1,360 conditions, 10 runs; Wilcoxon + effect sizes.
- **Method / dataset:** WEKA/MOA implementations; S-RUBiS, ASOS, C-VARD; MAE + mean training time; drift counted via ADWIN, extent via mRSD.
- **Gap vs. our lemmas:** Regression performance modeling (software domain), not classification streams; no detectors/recovery metrics; complements Read 2012 and reinforces our per-metric/per-context claim structure.
- **Verified:** ✅

### [Batch-Incremental versus Instance-Incremental Learning in Dynamic and Evolving Data](https://doi.org/10.1007/978-3-642-34156-4_29)
- **Authors:** Jesse Read, Albert Bifet, Bernhard Pfahringer, Geoff Holmes
- **Year / Venue:** 2012, IDA (LNCS 7619), pp. 313–323
- **Chapter relevance:** Ch1 (primary — the canonical instance- vs batch-incremental comparison)
- **Core claim:** First in-depth comparison of instance-incremental (learn per example) vs batch-incremental (gather batches, retrain) classifiers under drift; papers in both camps typically cite no evidence for the choice — this supplies it, revealing respective advantages/disadvantages with no universal winner.
- **Key numbers:** 104 citations (Springer, Sep 2026); headline empirical numbers to be extracted on full read.
- **Method / dataset:** MOA; multiple algorithms of each paradigm, drift adaptation focus.
- **Gap vs. our lemmas:** Establishes context-dependence (hence our per-metric win card), not reactivity metrics; immediate labels only. **Deep note:** `papers/read-2012-batch-vs-instance.md`.
- **Verified:** ✅

### [A Framework for Evaluating and Benchmarking Concept Drift Detection Methods](https://dl.acm.org/doi/10.1145/3770855.3819070)
- **Authors:** Vitor Cerqueira, Heitor Murilo Gomes, Marco Heyden, Bernhard Pfahringer, Albert Bifet
- **Year / Venue:** 2026, KDD '26 (Proceedings of the 32nd ACM SIGKDD, V.2), pp. 8671–8682
- **Chapter relevance:** Ch1 (primary), Ch2 (secondary — reproducible benchmark infra)
- **Core claim:** Evaluation of drift detectors is inconsistent (oversimplified synthetic generators, incompatible metrics, hidden hyperparameter choice); proposes a Monte-Carlo drift-injection framework on real data, timing-aware metrics (F1-detection score, normalized detection time), and a leave-one-dataset-out hyperparameter protocol.
- **Key numbers:** Benchmarks 14 detectors on 7 real datasets × 4 drift types (class prior, label swap, feature permutation, feature filtering) × abrupt/gradual; KDD'26 acceptance rate 21%; open access, code public.
- **Method / dataset:** Drift injected into real streams via Monte Carlo trials.
- **Gap vs. our lemmas:** Immediate-label setting; no delayed/partial labels. Confirms our Ch1 methodology gap (no accepted protocol) and is the strongest candidate baseline harness to reproduce for the Phase-0 "reproduce a published baseline" task.
- **Verified:** ✅

### [Practical Machine Learning for Streaming Data (KDD'24 tutorial)](https://dl.acm.org/doi/10.1145/3637528.3671442)
- **Authors:** Heitor Murilo Gomes, Albert Bifet
- **Year / Venue:** 2024, KDD '24 (tutorial abstract, 2 pp.)
- **Chapter relevance:** Ch1 (tooling), Ch3 (secondary — delayed/partial label section)
- **Core claim:** Tutorial bridging stream-ML research and practice via CapyMOA (Python API over MOA); includes drift detection/evaluation and a section on partially and delayed labeled streams using cluster-and-label and incremental self-training.
- **Key numbers:** n/a (tutorial).
- **Method / dataset:** CapyMOA/MOA tooling; ADWIN, STUDD, Hoeffding Trees, SGBT, SOKNL, CluStream, StreamKM++, Half-Space Trees, Online Isolation Forest.
- **Gap vs. our lemmas:** none — tooling reference; confirms CapyMOA as a practical alternative/supplement to River.
- **Verified:** ✅ (ACM DL page + full text in Material/)

### [DeepRiver: A Deep Learning Library for Data Streams](https://doi.org/10.21105/joss.07226)
- **Authors:** Cedric Kulbach, Lucas Cazzonelli, Hoang-Anh Ngo, Max Halford, Saulo Martiello Mastelini
- **Year / Venue:** 2025, Journal of Open Source Software 10(105):7226
- **Chapter relevance:** Ch1 (tooling, secondary), Ch2 (secondary)
- **Core claim:** DeepRiver adds deep learning (PyTorch) to River's API for online/streaming scenarios: class-incremental learning, stream-specific optimizers, anomaly detection (autoencoders).
- **Key numbers:** n/a (software paper).
- **Method / dataset:** River + PyTorch; Autoencoder, RollingClassifier, etc.
- **Gap vs. our lemmas:** none — out of our scope (we exclude deep learning), but useful to know River's ecosystem.
- **Verified:** ✅ (JOSS page + full text in Material/)

### [River: Machine Learning for Streaming Data in Python (JMLR 2021)](https://arxiv.org/abs/2012.04740)
- **Authors:** Jacob Montiel, Max Halford, Saulo Martiello Mastelini, Geoffrey Bolmier, Raphael Sourty, Robin Vaysse, Adil Zouitine, Heitor M. Gomes, Jesse Read, TaleL Abdessalem, Albert Bifet
- **Year / Venue:** 2021, JMLR 22(110):1–8
- **Chapter relevance:** Ch1 (primary — our chosen tool), Ch2 (secondary — deployment APIs)
- **Core claim:** River is a Python library for online/streaming ML (merge of creme + scikit-multiflow) providing incremental models, drift detectors, prequential metrics, and synthetic generators in a unified dict-based API.
- **Key numbers:** n/a (tooling paper); implements ADWIN, DDM, EDDM, Page-Hinkley, KSWIN, HDDM, Hoeffding trees, ARF, SRP, prequential metrics.
- **Method / dataset:** Library.
- **Gap vs. our lemmas:** none — the instrument, not the subject. Needed to pin the River version for the Phase-0 baseline reproduction.
- **Verified:** ✅

### [Adaptive Learning from Evolving Data Streams (HAT)](https://link.springer.com/chapter/10.1007/978-3-642-03915-7_22)
- **Authors:** Albert Bifet, Ricard Gavaldà
- **Year / Venue:** 2009, IDA (Springer)
- **Chapter relevance:** Ch1 (adaptive model baseline)
- **Core claim:** Introduces Hoeffding Adaptive Tree (HAT), which uses ADWIN at tree nodes to detect and adapt to drift; a core adaptive baseline.
- **Key numbers:** Reports superior adaptability on synthetic and real datasets (Adult, Poker-Hand); HAT-INC, HAT-EWMA, HAT-ADWIN variants.
- **Method / dataset:** Incremental decision trees with ADWIN per node.
- **Gap vs. our lemmas:** none — immediate labels; a Ch1 baseline model.
- **Verified:** ✅

### [On Evaluating Stream Learning Algorithms](https://link.springer.com/article/10.1007/s10994-012-9334-7)
- **Authors:** João Gama, Raquel Sebastião, Pedro Pereira Rodrigues
- **Year / Venue:** 2013, Machine Learning 90(3):317–346
- **Chapter relevance:** Ch1 (prequential evaluation protocol)
- **Core claim:** Prequential (test-then-train) evaluation with forgetting mechanisms is the correct protocol for ordered streams; random splits leak temporal information.
- **Key numbers:** n/a (methodology).
- **Method / dataset:** Formalizes prequential evaluation and sliding-window/fading-factor variants.
- **Gap vs. our lemmas:** none — foundational evaluation protocol for Ch1; breaks under delayed labels (our Ch3 motivation).
- **Verified:** ✅

### [A Comparative Study on Concept Drift Detectors](https://doi.org/10.1016/j.eswa.2014.07.019)
- **Authors:** Paulo M. Gonçalves Jr, Silas G. T. de Carvalho Santos, Roberto S. M. Barros, Davi C. L. Vieira
- **Year / Venue:** 2014, Expert Systems with Applications 41(18):8144–8156
- **Chapter relevance:** Ch1 (detector comparison baseline)
- **Core claim:** Systematic comparison of DDM, EDDM, Page-Hinkley, STEPD, DoF, ADWIN, Paired Learners, ECDD under abrupt/gradual drift, noise, and irrelevant attributes; no detector dominates.
- **Key numbers:** n/a reported in abstract; the canonical "no single detector wins" empirical reference.
- **Method / dataset:** Synthetic (SEA) and real streams; detector configs matter.
- **Gap vs. our lemmas:** none — immediate labels; reusable as Ch1 comparator and to justify our sweep design.
- **Verified:** ⚠️ (DOI cross-checked in KDD'26 reference list; full-text not re-fetched)

### [Learning from Time-Changing Data with Adaptive Windowing (ADWIN)](https://doi.org/10.1137/1.9781611972771.42)
- **Authors:** Albert Bifet, Ricard Gavaldà
- **Year / Venue:** 2007, SIAM SDM (foundational)
- **Chapter relevance:** Ch1 (detector baseline, foundational)
- **Core claim:** ADWIN maintains an adaptive window and shrinks it when two subwindows differ significantly; provides false-positive guarantees.
- **Key numbers:** n/a (foundational).
- **Method / dataset:** Adaptive windowing / change detection on error or feature streams.
- **Gap vs. our lemmas:** none — immediate labels; the canonical detector for Ch1 and the natural subject of our Lemma 2 (power loss) experiment.
- **Verified:** ✅

### [Learning with Drift Detection (DDM)](https://doi.org/10.1007/978-3-540-28645-5_29)
- **Authors:** João Gama, Pedro Medas, Gladys Castillo, Pedro Rodrigues
- **Year / Venue:** 2004, SBIA (foundational)
- **Chapter relevance:** Ch1 (detector baseline, foundational)
- **Core claim:** DDM tracks error rate and its std, warns/alarms when error significantly exceeds its minimum; cheap, standard supervised detector.
- **Key numbers:** n/a (foundational).
- **Method / dataset:** Statistical process control on error stream.
- **Gap vs. our lemmas:** none — immediate labels; subject of our Lemma 3 (batch-arrival false-alarm inflation) experiment.
- **Verified:** ✅

### [Early Drift Detection Method (EDDM)](https://dl.acm.org/doi/10.1145/347090.347107) (Baena-García et al., 2006, IWKDDS)
- **Authors:** Manuel Baena-García, José del Campo-Ávila, Raúl Fidalgo, Albert Bifet, Ricard Gavaldà, Rafael Morales-Bueno
- **Year / Venue:** 2006, 4th Intl Workshop on Knowledge Discovery from Data Streams (foundational)
- **Chapter relevance:** Ch1 (detector baseline, foundational)
- **Core claim:** EDDM monitors distance between errors to improve detection of gradual drift.
- **Key numbers:** n/a (foundational).
- **Method / dataset:** Error-distance monitoring.
- **Gap vs. our lemmas:** none — immediate labels.
- **Verified:** ⚠️ (citation cross-checked in KDD'26 and STUDD reference lists; not separately fetched)

### [Mining High-Speed Data Streams (VFDT / Hoeffding Tree)](https://dl.acm.org/doi/10.1145/347090.347107)
- **Authors:** Pedro Domingos, Geoff Hulten
- **Year / Venue:** 2000, KDD (foundational)
- **Chapter relevance:** Ch1 (static model baseline, foundational)
- **Core claim:** Hoeffding Tree learns a decision tree in one pass with constant memory per leaf using the Hoeffding bound; the canonical static stream classifier.
- **Key numbers:** Reported constant-time-per-example and high throughput; 1,598 citations on ACM DL.
- **Method / dataset:** VFDT on synthetic/large data.
- **Gap vs. our lemmas:** none — immediate labels; Ch1 static baseline.
- **Verified:** ✅

### [Mining Time-Changing Data Streams (CVFDT)](https://dl.acm.org/doi/10.1145/502512.502529)
- **Authors:** Geoff Hulten, Laurie Spencer, Pedro Domingos
- **Year / Venue:** 2001, KDD (foundational)
- **Chapter relevance:** Ch1 (static/adaptive model baseline, foundational)
- **Core claim:** Extends VFDT with a sliding window and alternate subtrees so the tree stays current under drift while still exploiting old data; the original demonstration that an *adaptive* stream learner tracks evolving concepts better than trees rebuilt periodically on recent data — the ancestor of our frozen-vs-adaptive claim.
- **Key numbers:** Constant (amortised) time per example; experiments on synthetic + real evolving streams (e.g., network/data-stream settings).
- **Method / dataset:** Hoeffding-tree variant with windowed statistics + alternate-subtree replacement.
- **Gap vs. our lemmas:** Immediate labels; recovery-time/recovery-metrics framing not present — but it is the correct historical citation for "adaptive beats periodically-rebuilt batch," alongside VFDT (2000).
- **Verified:** ✅

### [Learning in the Presence of Concept Drift and Hidden Contexts](https://doi.org/10.1007/bf00116900)
- **Authors:** Gerhard Widmer, Miroslav Kubat
- **Year / Venue:** 1996, Machine Learning 23(1):69–101 (foundational)
- **Chapter relevance:** Ch1 (concept drift foundations)
- **Core claim:** Early formal treatment of concept drift and hidden contexts; introduced the notion of context-dependent concepts.
- **Key numbers:** n/a (foundational).
- **Method / dataset:** FLORA2 algorithm on synthetic streams.
- **Gap vs. our lemmas:** none — immediate labels; contextualizes Ch1.
- **Verified:** ⚠️ (citation cross-checked in multiple reference lists; not separately fetched)

---

## Section 2 — Ch2: Systems / operational cost of continual learning

### [A High-Throughput Architecture for Anomaly Detection in Streaming Data Using Machine Learning Algorithms](https://link.springer.com/article/10.1007/s41870-023-01585-0)
- **Authors:** Chellammal Surianarayanan, Saranya Kunasekaran, Pethuru Raj Chelliah
- **Year / Venue:** 2023 (published online 04 Nov 2023; issue 2024), International Journal of Information Technology 16:493–506
- **Chapter relevance:** Ch2 (primary — streaming architecture, Kafka, throughput)
- **Core claim:** Proposes a high-throughput Kafka publish-subscribe architecture for continuous ML anomaly detection, with two alternating subscribers to hide ML computation time.
- **Key numbers:** Single-node: avg accuracy 96.7%, computation 37.5 ms; distributed: avg accuracy 98.6%, computation 38.5 ms; Random Forest with threshold-based point anomaly detection.
- **Method / dataset:** Apache Kafka pub/sub, RF; single-node vs distributed configs.
- **Gap vs. our lemmas:** none — no delayed labels; a concrete reference for our Ch2 event-source/backpressure design (Redis Streams vs Kafka decision).
- **Verified:** ✅ (Springer full text fetched)

### [Optimizing Machine Learning Inference Serving Systems: A Survey](https://www.semanticscholar.org/paper/Optimizing-Machine-Learning-Inference-Serving-A-Ebadi-Keller/b418ad9a70c626133b9b0b8d8b566739f54cff85)
- **Authors:** (survey) Ebadi & Keller (per Semantic Scholar)
- **Year / Venue:** 2025 (recent; last-3-years survey)
- **Chapter relevance:** Ch2 (serving latency/throughput optimization)
- **Core claim:** Surveys techniques for resource optimization and latency reduction in ML inference serving.
- **Key numbers:** n/a (survey).
- **Method / dataset:** Survey of serving systems literature.
- **Gap vs. our lemmas:** none — no delayed labels; informs Ch2's latency/throughput framing, not Ch3.
- **Verified:** ⚠️ (Semantic Scholar abstract page; full author list not independently cross-checked)

### [Machine Learning (In) Security: A Stream of Problems](https://marcusbotacin.github.io/files/2023-ceschin-insecurity.pdf)
- **Authors:** F. Ceschin, H. M. Gomes, M. Botacin, A. Bifet, B. Pfahringer, L. S. Oliveira, A. Grégio
- **Year / Venue:** 2023, Expert Systems with Applications 212:118590
- **Chapter relevance:** Ch2 (primary), Ch1 (secondary)
- **Core claim:** Argues malware detection must be modeled as an evolving data stream; surveys pipeline realism gaps (drift, delayed labels, imbalance).
- **Key numbers:** n/a (survey/positioning); frames the delayed-label production story.
- **Method / dataset:** Survey of stream-ML-for-security pitfalls.
- **Gap vs. our lemmas:** none directly — but a strong production-motivation citation for Ch2 and Ch3.
- **Verified:** ⚠️ (PDF link from author site; year/venue cross-checked in Botacin & Gomes references)

### [TESSERACT: Eliminating Experimental Bias in Malware Classification Across Space and Time](https://www.usenix.org/conference/usenixsecurity19/presentation/pendlebury)
- **Authors:** Feargus Pendlebury, Fabio Pierazzi, Roberto Jordaney, Johannes Kinder, Lorenzo Cavallaro
- **Year / Venue:** 2019, USENIX Security (foundational for realistic evaluation)
- **Chapter relevance:** Ch2 (evaluation bias), Ch3 (delayed-label motivation)
- **Core claim:** Temporal (and spatial) bias in malware ML evaluation inflates results; proposes temporal-aware evaluation and the AUT (area under time) metric.
- **Key numbers:** Reports evaluation bias reduction with time-aware splits (details in paper).
- **Method / dataset:** Android malware (DREBIN) with temporal splits; AUT metric.
- **Gap vs. our lemmas:** none — no delayed labels, but AUT is a temporal metric Ch2/Ch3 should contrast with the exposure metric of Botacin & Gomes.
- **Verified:** ⚠️ (USENIX page; content cross-checked via Botacin & Gomes reference list)

### [Challenges in Benchmarking Stream Learning Algorithms with Real-world Data](https://link.springer.com/article/10.1007/s10618-020-00698-5)
- **Authors:** Vinícius M. A. Souza, Denis M. dos Reis, André G. Maletzke, Gustavo E. A. P. A. Batista
- **Year / Venue:** 2020, Data Mining and Knowledge Discovery 34:1805–1858
- **Chapter relevance:** Ch2 (benchmarking pitfalls), Ch1 (secondary)
- **Core claim:** Real-world stream benchmarks have undocumented preprocessing and ordering issues that confound results; proposes a benchmark repository and methodological fixes.
- **Key numbers:** n/a (methodology).
- **Method / dataset:** Stream repository (the one used by STUDD).
- **Gap vs. our lemmas:** none — but reinforces our Elec2/Kappa+ caution for Ch1.
- **Verified:** ✅

### [MOA: A Real-Time Analytics Open Source Framework](https://link.springer.com/chapter/10.1007/978-3-642-24958-7_58)
- **Authors:** Albert Bifet, Geoff Holmes, Bernhard Pfahringer, Jesse Read, Philipp Kranen, Hardy Kremer, Timm Jansen, Thomas Seidl
- **Year / Venue:** 2011, ECML-PKDD (tooling)
- **Chapter relevance:** Ch2 (infrastructure/tooling contrast), Ch1 (secondary)
- **Core claim:** MOA is the reference Java framework for stream mining; relevant as the ecosystem River/CapyMOA interoperate with.
- **Key numbers:** n/a (tooling).
- **Method / dataset:** Framework.
- **Gap vs. our lemmas:** none.
- **Verified:** ⚠️ (Springer chapter page; cross-checked in Grzenda references)

---

## Section 3 — Ch3: Delayed / partial / selective labels + drift detection (the core gap)

### [Concept Drift Detection in Delayed and Partially Labeled Data Streams: An Experimental Survey](https://www.sciencedirect.com/science/article/pii/S1051200426004434)
- **Authors:** Brenno de Mello Alencar, Guilherme Weigert Cassales, Heitor Murilo Gomes, Cássio V. S. Prazeres, Tatiane Rios, Albert Bifet, Ricardo Araújo Rios
- **Year / Venue:** 2026, Digital Signal Processing 182:106325
- **Chapter relevance:** Ch3 (primary — the closest existing survey)
- **Core claim:** First recent experimental survey organizing drift detectors under delayed and partially labeled streams; provides a taxonomy of detectors and evaluation approaches under label scarcity.
- **Key numbers:** DSP 182, article 106325 (verified via dblp).
- **Method / dataset:** Experimental survey of detectors under delayed/partial labels.
- **Gap vs. our lemmas:** Directly adjacent to the whole of Ch3. It likely partially covers our Lemma 2 (label-scarcity degradation) empirically; the gap is the formal reaction-floor/identifiability treatment and the hybrid confirmation protocol. **Potential partial scoop — must read in full before finalizing Ch3 novelty claims.**
- **Verified:** ✅ (dblp + ScienceDirect + Semantic Scholar)

### [Towards More Realistic Evaluations: The Impact of Label Delays in Malware Detection Pipelines](https://www.sciencedirect.com/science/article/abs/pii/S0167404824004279)
- **Authors:** Marcus Botacin, Heitor Gomes
- **Year / Venue:** 2024, Computers & Security (COSE), vol. 147 (per ScienceDirect pii)
- **Chapter relevance:** Ch3 (primary — direct empirical measurement of label-delay cost), Ch2 (secondary — queue/pipeline modeling)
- **Core claim:** Models malware detection as delayed+partially-labeled evolving stream; quantifies how label delay (from analysis-queue limits) degrades drift-based retraining, proposes the exposure metric, and shows pseudo-labels mitigate delay.
- **Key numbers:** Delayed labels increase user exposure ~50% (up to 75% detection decrease); 81 epochs delay → exposure +50% vs immediate (DDM), 61 epochs (EDDM); pseudo-labels reduce loss to ~30% of original; 10% queue ≈ 100+ epochs delay; 10 executions averaged; DREBIN 250×500 samples, Androzoo replication.
- **Method / dataset:** DREBIN + Androzoo, RF, DDM/EDDM, epoch-based drift detection, exposure metric, pseudo-label (secondary classifier) mitigation.
- **Gap vs. our lemmas:** Empirically confirms the reaction-floor phenomenon (Lemma 1) and the value of a label-free/early signal (hybrid), but does not derive the formal bounds or the statistical power/identifiability lemmas; its pseudo-label approach is the closest existing thing to our hybrid's Stage-1. **Must-cite; do not let Ch3 claim the exposure-style quantification as novel without contrasting.**
- **Verified:** ✅ (PDF fetched + ScienceDirect record; COSE 2024 confirmed via GitHub repo)

### [A Survey on Semi-supervised Learning for Delayed Partially Labelled Data Streams](https://dl.acm.org/doi/10.1145/3523055)
- **Authors:** Heitor Murilo Gomes, Maciej Grzenda, Rodrigo Mello, Jesse Read, Minh Huong Le Nguyen, Albert Bifet
- **Year / Venue:** 2022 (published online 21 Nov 2022; issue 2023), ACM Computing Surveys 55(4), Article 75
- **Chapter relevance:** Ch3 (primary — anchor survey), Ch1 (secondary)
- **Core claim:** Formalizes "delayed partially labelled data streams" (immediate/delayed/infinite labels × supervised/semi-supervised); surveys SSL methods, learning guarantees, and evaluation practice; proposes a unified problem setting.
- **Key numbers:** 63 citations / 6,779 downloads (ACM DL, Aug 2026). Included a small experiment: with ADWIN + SRP on AGRAWAL (100k instances, 3 drifts), a 10,000-instance label delay shifts detections by ~10,000 instances and causes prolonged prequential accuracy below 50%.
- **Method / dataset:** Survey + illustrative delayed-label AGRAWAL experiment.
- **Gap vs. our lemmas:** Confirms the reaction-floor phenomenon empirically (Lemma 1) but does not prove it or bound power loss (Lemma 2) or batch-arrival inflation (Lemma 3). Identifiability (Lemma 4) is discussed via Žliobaitė. **The reference to cite as "closest prior work."**
- **Verified:** ✅

### [STUDD: A Student–Teacher Method for Unsupervised Concept Drift Detection](https://link.springer.com/article/10.1007/s10994-022-06188-7)
- **Authors:** Vitor Cerqueira, Heitor Murilo Gomes, Albert Bifet, Luis Torgo
- **Year / Venue:** 2022 (online 21 Jun 2022; issue 2023), Machine Learning 112:4351–4378
- **Chapter relevance:** Ch3 (label-free baseline for hybrid Stage-1), Ch1 (secondary)
- **Core claim:** Student model mimics teacher; drift is detected by monitoring the student's mimicking loss (label-free) — a strong candidate for the label-free early-warning stage of our hybrid.
- **Key numbers:** 19 benchmark streams (truncated to 150k each); MDR ~19% (3/16), FAR/MTD/MTFA reported; label efficiency best among non-baselines; Bayesian sign test ROPE 5%.
- **Method / dataset:** Student-teacher with Page-Hinkley; comparison vs supervised (SS/DSS/WS/DWS) and unsupervised (OS/OF/FF) variants.
- **Gap vs. our lemmas:** Unsupervised only — explicitly notes it cannot catch real (posterior) drift without labels, which is exactly our Lemma 4 and motivates the hybrid. **Direct Stage-1 baseline for Ch3.**
- **Verified:** ✅

### [ROSE: Robust Online Self-Adjusting Ensemble for Continual Learning on Imbalanced Drifting Data Streams](https://doi.org/10.1007/s10994-022-06168-x)
- **Authors:** Alberto Cano, Bartosz Krawczyk
- **Year / Venue:** 2022, Machine Learning 111:2561–2599
- **Chapter relevance:** Ch3 (secondary — adaptation method under drift+imbalance), Ch2 (secondary)
- **Core claim:** Online ensemble with variable-size random feature subspaces, ADWIN-per-subspace drift detection + background ensemble, per-class sliding windows, and self-adjusting λ bagging for imbalance.
- **Key numbers:** 30 ensembles compared; 24 imbalanced real datasets; 36 imbalanced generators; 12 drifting-imbalance generators; Kappa avg 63.65 vs next-best 57.85 (static IR); Wilcoxon/Bayesian significance; 384 GB RAM Xeon runs.
- **Method / dataset:** MOA; HoeffdingTree base; prequential Kappa/AUC; Friedman ranks.
- **Gap vs. our lemmas:** Immediate labels; contributes the background-ensemble + drift-detector adaptation pattern relevant to Ch2/Ch3 but not delayed-label detection.
- **Verified:** ✅ (full text in Material/)

### [Stream-Based Active Learning with Verification Latency in Non-stationary Environments](https://arxiv.org/abs/2204.06822)
- **Authors:** Rodrigo Mello, Heitor Murilo Gomes, Albert Bifet, Bernhard Pfahringer
- **Year / Venue:** 2022 (arXiv:2204.06822; IJCNN 2022)
- **Chapter relevance:** Ch3 (selective labeling / active learning under latency), Ch2 (secondary)
- **Core claim:** Proposes PRopagate: estimates query utility under unknown label delay and predicts requested-but-unavailable labels; drift-dependent labeling budget.
- **Key numbers:** Reports improvements over active-learning baselines under constant and variable latency (from abstract; per-paper details to be read).
- **Method / dataset:** Stream active learning with verification latency.
- **Gap vs. our lemmas:** Addresses label acquisition under latency rather than detector isolation; the "only flagged cases labeled" endogenous-bias scenario connects to our selective-labeling regime. **Partially overlaps our selective-labeling chapter.**
- **Verified:** ✅ (arXiv abstract page resolves)

### [Statistical Drift Detection with Parametric Gamma–Weibull and Adaptive Deep Learning for Malware Detection in IoT Data Streams](https://doi.org/10.1007/s44443-026-01091-w)
- **Authors:** Friday I. Agu, Robert Andok, Jaromír Klarák, Jean Rosemond Dora
- **Year / Venue:** 2026, Journal of King Saud University — Computer and Information Sciences 38:673
- **Chapter relevance:** Ch3 (secondary — hybrid statistical detector + pseudo-label adaptation), Ch2 (secondary — edge/runtime profiling)
- **Core claim:** A Gamma–Weibull log-likelihood-ratio CUSUM detector (ARL₀-calibrated) gates lightweight LSTM/CNN adaptation; post-drift buffer, pseudo-labeling + clustering labels, no human labels.
- **Key numbers:** Detection delay 6 samples (abrupt), 80 (gradual), 35 (mixed), zero false alarms (noise-free); ADWIN delay 1.1±1.2 but ~1270 false alarms; PHT 125.2±106 delay, 0 FA; DDM 425.9±401 delay, 625 FA; LSTM macro-F1 0.977→0.981, CNN 0.457→0.576; full pipeline 1.43 s CPU; monitoring <150 ms; memory <50 KB.
- **Method / dataset:** Simulated Gamma→Weibull streams (T=10k, MC=30); IoT-23 (benign n=451; malware n=14,784; capture 33-1 and 1-1).
- **Gap vs. our lemmas:** Uses a parametric distribution model (not error-stream detectors) and pseudo-labels (not delayed ground truth); does not treat the four lemmas, but its gatekeeper adaptation is a good contrast for our hybrid architecture. **Include as a 2026 "adjacent hybrid" citation.**
- **Verified:** ✅ (full text in Material/)

### [Delayed Labelling Evaluation for Data Streams](https://link.springer.com/article/10.1007/s10618-019-00654-y)
- **Authors:** Maciej Grzenda, Heitor Murilo Gomes, Albert Bifet
- **Year / Venue:** 2020, Data Mining and Knowledge Discovery 34:1237–1266
- **Chapter relevance:** Ch3 (evaluation protocol under delay), Ch1 (secondary)
- **Core claim:** Introduces continuous re-evaluation: buffer predictions, re-poll the evolving model before the delayed label arrives, and score binned predictions (initial / periodic / test-then-train). The standard evaluation answer for delayed-label streams.
- **Key numbers:** Hyperplane (delay 100), LED (delay 1000), AGRAWAL (delay 1000, 4 gradual drifts), Airlines (natural latency up to 13h), Electricity (~1–2 day latency), Forest cover (delay 1000); K=10 re-polls, B=50 bins recommended. Open-source MOA implementation.
- **Method / dataset:** MOA; HT/HAT/ARF/kNN/NB/NC/MC; binned performance λ(T,b).
- **Gap vs. our lemmas:** Evaluation methodology, not detector analysis; it shows models can refine predictions while labels are pending — directly relevant to our hybrid's confirmation stage and to how we evaluate the delayed-label regimes. **Use their evaluation protocol in Ch3.**
- **Verified:** ✅

### [Credit Card Fraud Detection and Concept-Drift Adaptation with Delayed Supervised Information](https://www.semanticscholar.org/paper/Credit-card-fraud-detection-and-concept-drift-with-Pozzolo-Boracchi/b769ec8f3be051b7c5f9a0ba5f85a38594950df9)
- **Authors:** Andrea Dal Pozzolo, Giacomo Boracchi, Olivier Caelen, Cesare Alippi, Gianluca Bontempi
- **Year / Venue:** 2015, IEEE IJCNN
- **Chapter relevance:** Ch3 (delayed labels in fraud — production story), Ch2 (secondary)
- **Core claim:** In credit-card fraud, immediate investigator feedback and delayed chargeback labels carry different information; training separate models on each and aggregating outputs beats treating them identically.
- **Key numbers:** Reports improved performance by separating immediate vs delayed supervision (per abstract; details in paper).
- **Method / dataset:** Fraud detection with delayed supervised information.
- **Gap vs. our lemmas:** An application-specific adaptation strategy, not a general detector benchmark; supports our premise that delayed labels break standard practice, and echoes the two-signal structure of our hybrid. **Empirical anchor for the fraud dataset story.**
- **Verified:** ⚠️ (Semantic Scholar hit a bot-wall on fetch; citation cross-checked in Gomes survey reference list)

### [Online Learning under Delayed Feedback](https://proceedings.mlr.press/v28/joulani13.html)
- **Authors:** Pooria Joulani, András György, Csaba Szepesvári
- **Year / Venue:** 2013, ICML (PMLR 28(3):1453–1461)
- **Chapter relevance:** Ch3 (theory — delayed-feedback regret), Ch1 (secondary)
- **Core claim:** Delay increases regret multiplicatively in adversarial online learning and additively in stochastic settings; gives black-box meta-algorithms to handle delay.
- **Key numbers:** n/a (theoretical).
- **Method / dataset:** Regret bounds for full-information and bandit feedback under delay.
- **Gap vs. our lemmas:** This is the regret-theoretic cousin of our delay tax, but for learning (not detection); our Lemma 1 (reaction floor) is the detection-side analogue. **Cite for the theoretical framing of "delay costs."**
- **Verified:** ✅

### [Change with Delayed Labeling: When is it Detectable?](https://dl.acm.org/doi/10.1109/ICDMW.2010.49)
- **Authors:** Indrė Žliobaitė
- **Year / Venue:** 2010, IEEE ICDM Workshops (foundational)
- **Chapter relevance:** Ch3 (identifiability — Lemma 4), Ch1 (secondary)
- **Core claim:** Characterizes which drift types are detectable without new labels; pure posterior drift (P(Y|X) changes, P(X) fixed) is not detectable from inputs — the formal basis of our identifiability ceiling.
- **Key numbers:** n/a (theoretical/analytical; two drift types empirically evaluated).
- **Method / dataset:** Analytical observability conditions + experiments.
- **Gap vs. our lemmas:** Directly establishes Lemma 4's identifiability ceiling; does not propose the hybrid confirmation architecture. **The theoretical root of our hybrid's justification.**
- **Verified:** ✅

### [A Predictive and User-Centric Approach to Machine Learning in Data Streaming Scenarios](https://doi.org/10.1016/j.neucom.2021.05.049)
- **Authors:** Davide Carneiro, Miguel Guimarães, Fábio Silva, Paulo Novais
- **Year / Venue:** 2021 (preprint May 2021), Neurocomputing (per Material filename)
- **Chapter relevance:** Ch3 (secondary — retraining decision support), Ch2 (secondary)
- **Core claim:** Meta-learning approach that predicts whether retraining a model on new data will improve performance, to avoid wasteful retraining in streaming scenarios; pairs it with an explainable interface for human-in-the-loop validation.
- **Key numbers:** Meta-model trained on 673 meta-instances from 47 datasets; 10-fold CV RMSE 0.000355, r² 0.98, MAE 0.007; streaming meta-model updated every 2000 instances; 53 datasets total.
- **Method / dataset:** pymfe meta-features (57), Random Forest meta-model; regression (binary classification transformed to 0/1); tax fraud case study.
- **Gap vs. our lemmas:** Addresses "when to retrain" via predicted performance rather than drift detection; relevant to Ch2's adaptation-cost framing and Ch3's selective-adaptation angle, but not delayed-label detection per se.
- **Verified:** ⚠️ (full text in Material/; DOI inferred from filename — verify before citing)

### [Stream-based Active Learning for Sliding Windows under the Influence of Verification Latency](https://dl.acm.org/doi/10.1007/s10994-021-06099-z)
- **Authors:** Rodrigo Mello, Heitor Murilo Gomes, Albert Bifet, Bernhard Pfahringer
- **Year / Venue:** 2021, Machine Learning (online 2021)
- **Chapter relevance:** Ch3 (verification latency in active learning), Ch2 (secondary)
- **Core claim:** Forgetting and Simulating: simulates the learner's future state when a queried label arrives, improving utility estimates under latency.
- **Key numbers:** Reports improvements under constant and variable latency (per abstract).
- **Method / dataset:** Stream active learning with sliding windows.
- **Gap vs. our lemmas:** Adjacent — label acquisition under latency, not detector isolation; complements the selective-labeling regime.
- **Verified:** ⚠️ (DOI cross-checked in Gomes survey reference list; not separately fetched)

---

## Section 4 — Ch3: Hybrid / label-free + confirmation methods

### [STUDD: A Student–Teacher Method for Unsupervised Concept Drift Detection](https://link.springer.com/article/10.1007/s10994-022-06188-7)
- (listed in Section 3; here for hybrid relevance) — label-free early warning baseline.
- **Verified:** ✅

### [Statistical Drift Detection with Parametric Gamma–Weibull and Adaptive Deep Learning for Malware Detection in IoT Data Streams](https://doi.org/10.1007/s44443-026-01091-w)
- (listed in Section 3; here for hybrid relevance) — detector-gated pseudo-label adaptation, 2026.
- **Verified:** ✅

### [Unsupervised Concept Drift Detectors: A Survey](https://link.springer.com/chapter/10.1007/978-3-031-20738-9_121)
- **Authors:** (Shen, Ming, Li, Gao, Zhang — per Gomes survey; verify full list)
- **Year / Venue:** 2023, Springer (survey chapter)
- **Chapter relevance:** Ch3 (label-free detector families), Ch1 (secondary)
- **Core claim:** Surveys unsupervised (label-free) drift detectors, classifying them separately from supervised error-based methods.
- **Key numbers:** n/a (survey).
- **Method / dataset:** Survey.
- **Gap vs. our lemmas:** Catalogues the Stage-1 label-free options for our hybrid; does not treat delayed supervised confirmation.
- **Verified:** ⚠️ (citation cross-checked in research notes; not separately fetched)

### [Automatic Model Monitoring for Data Streams](https://arxiv.org/abs/1908.04240)
- **Authors:** Fábio Pinto, Marco O. P. Sampaio, Pedro Bizarro
- **Year / Venue:** 2019, arXiv:1908.04240 (Feedzai)
- **Chapter relevance:** Ch3 (hybrid-style monitoring), Ch2 (secondary)
- **Core claim:** Window-based monitoring with Jensen–Shannon divergence for drift; explains alarms via an auxiliary classifier.
- **Key numbers:** n/a (from abstract).
- **Method / dataset:** Windowed JSD drift detection + explanation model.
- **Gap vs. our lemmas:** Unsupervised monitoring (no delayed confirmation); useful as a label-free Stage-1 candidate.
- **Verified:** ⚠️ (arXiv abstract page; cross-checked in STUDD references)

---

## Foundational quick-reference (no per-paper note)

| Paper | Year | Role |
|---|---|---|
| Page, *Continuous Inspection Schemes*, Biometrika | 1954 | Root of Page-Hinkley / quickest detection (Lemma 1 framing) |
| Lorden, *Procedures for Reacting to a Change in Distribution*, Ann. Math. Statist. | 1971 | Minimax quickest detection (detection-delay vs false-alarm trade-off) |
| Kuncheva & Sánchez, *Nearest Neighbour Classifiers for Streaming Data with Delayed Labelling*, IEEE ICDM | 2008 | Early delayed-label classification work |
| Masud et al., *Classification and Novel Class Detection...*, IEEE TKDE | 2011 | Delayed labels + concept evolution |
| Ditzler, Roveri, Alippi, Polikar, *Learning in Nonstationary Environments: A Survey*, IEEE CIM | 2015 | Nonstationary learning survey |

---

## How to use this index

- `papers/<slug>.md` contains deep notes for the load-bearing papers (Ch3 anchors + closest prior work + the KDD'26 benchmark).
- `gap-analysis.md` maps every entry to the four lemmas and flags scoop risk.
- ⚠️ entries need a final verification pass before going into the report bibliography.
