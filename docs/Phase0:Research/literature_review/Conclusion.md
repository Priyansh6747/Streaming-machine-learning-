# Literature Review Conclusion — The Delay Tax

**Papers reviewed:** 38 unique papers — 33 catalogued in the four index sections (the index has 35 entries; STUDD and the Gamma–Weibull paper are each cross-listed in two sections) plus 5 foundational quick-reference items.

**Verified:** 21 of the 33 catalogued papers are ✅ verified; 12 are ⚠️ flagged unverified; the 5 foundational items carry no individual flag in the index.

**Deep-dive notes:** 6 papers received full per-paper notes in `papers/*.md` (Gomes 2022, Botacin & Gomes 2024, Cerqueira KDD'26, STUDD 2023, Agu 2026, ROSE 2022).

**Date range:** 1954 (Page, foundational) – 2026; the 33 catalogued papers span 1996–2026.

---

## Conclusions

### Is the gap real?

**Verdict: Yes — delayed/partial-label drift detection is confirmed fragmented and unbenchmarked, but the immediate-label side of the field now has a standard benchmark, so the claim must be sharpened.**

Of the 33 catalogued papers, roughly 14 address delayed or partial labels in some form, but only 4 study drift *detectors* under delay at all — Alencar 2026 (results unread), Gomes 2022 (one illustrative experiment), Botacin & Gomes 2024 (DDM/EDDM inside a pipeline), Žliobaitė 2010 (theory) — and none provides a detector-level benchmark across label-observation regimes, none states a formal detection bound under label latency, and none builds and evaluates a two-stage early-warning + delayed-confirmation hybrid. Fragmentation is visible in the terminology alone: verification latency, delayed labelling, label latency, extreme verification latency, and delayed partially labelled streams all name the same area. The important update: Cerqueira et al. (KDD'26) has standardized the *immediate-label* detector benchmark (14 detectors, 7 real datasets, 4 drift types, abrupt+gradual), so our gap statement becomes "a standard benchmark now exists for immediate labels, but not for delayed, partial, or selective labels."

### Lemma 1 — Reaction floor (detection cannot precede label arrival)

**Verdict: empirically demonstrated by two papers, never stated as a theorem — novel as a formal statement, trivial to prove, and must cite the empirics.**

Gomes 2022's AGRAWAL experiment (100k instances, 3 drifts) shows a 10,000-instance label delay shifts ADWIN+SRP detections by ~10,000 instances and holds prequential accuracy below 50%. Botacin & Gomes 2024 shows delay maps near-linearly to harm: exposure reaches +50% at 81 epochs of delay (DDM) and 61 (EDDM), exposure grows ~50k points (DDM) / ~60k (EDDM) per 10 epochs of delay, and beyond 81–91 epochs drift mitigation is fully nullified. Joulani 2013 provides the learning-side analogue (delay multiplies regret in adversarial settings, adds it in stochastic ones). No logged paper states the pointwise bound T ≥ τ + L_τ for error-stream detectors under verification latency; the mathematical root is classical quickest detection (Page 1954, Lorden 1971), which nobody in this literature has applied to delayed-error detectors.

### Lemma 2 — Power loss (1/√p sensitivity degradation under partial labels)

**Verdict: the closed-form 1/√p bound for windowed error-stream detectors appears genuinely novel; adjacent treatments are intuition-level only, and one unread paper may hold the empirics.**

Gomes 2022 formalizes only the general SLT intuition that effective sample size equals the labelled count. STUDD 2023 sweeps label access over {1, 10, 25, 50}% and shows relative method ordering shifts — qualitative, no law. The one real risk is Alencar 2026 (DSP): an experimental survey that likely measures detector degradation as label availability decreases; its results are unread (paywalled), and if it already reports a 1/√p-like curve for ADWIN/DDM/KSWIN, our contribution shrinks to formalizing the factor rather than measuring it.

### Lemma 3 — False-alarm inflation under batch arrival (design effect 1+(b−1)ρ)

**Verdict: the most clearly novel lemma — no direct treatment found anywhere in the logged literature.**

Nothing logged studies batched or bursty label arrival injecting autocorrelation into the observed error stream and inflating detector false alarms. Botacin & Gomes 2024 is the closest conceptual neighbor but models a different mechanism — queue size → delay (their mapping: 10% queue capacity ≈ 100+ epochs of delay, and queue capacity can drop to 70% with pseudo-labels at the same exposure), not batch correlation → spurious alarms. The clustered-data variance factor 1+(b−1)ρ is classical statistics, but its application to drift detectors under batch-reconciled labels is open territory.

### Lemma 4 — Identifiability ceiling (pure posterior drift invisible to label-free detection)

**Verdict: already established — not novel as a claim; the novelty must be the hybrid's use of it.**

Žliobaitė 2010 proves the detectability conditions (pure posterior drift with fixed P(X) is undetectable from inputs); Gomes 2022 reproduces the observability table; STUDD 2023 demonstrates the limitation empirically (missed-detection ratio ~19%, 3/16 datasets) and states it explicitly. Ch3 should present Lemma 4 as a properly credited restatement of Žliobaitė, and the actual contribution is the conclusion that follows from it: label-free signals can only be an early warning, so the hybrid confirmation architecture is the information-maximal response.

### Scoop risk

**Verdict: one high-risk unread paper, one medium-high already-published overlap; the combination we propose remains unclaimed.**

- **Alencar et al. 2026 (DSP 182:106325) — HIGH.** An experimental survey of exactly our Ch3 setting. Verification status: ✅ that it exists (dblp + ScienceDirect + Semantic Scholar), but **results unread — paywalled**. If it contains delay-vs-latency and label-coverage-vs-false-alarm curves for standard detectors, part of our "measure the degradation" chapter is scooped as measurement; our remaining novelty would be formal lemmas 1–3 and the hybrid protocol. Must be read in full before finalizing any novelty claim.
- **Botacin & Gomes 2024 (Computers & Security) — MEDIUM-HIGH.** Already quantified label-delay cost on real malware pipelines (~50% exposure increase, up to 75% detection decrease; 30% loss with pseudo-labels instead of 70%) and shown pseudo-label mitigation. We must not claim "we quantify that delayed labels hurt."
- **Gomes 2022 (CSUR) — MEDIUM.** Covers our formal framing at survey level and explicitly flags the under-explored delayed+partial intersection as open — cite them as naming our gap.
- **Agu 2026 (JKSU-CIS) — LOW-MEDIUM.** Architecturally close hybrid (detector-gated pseudo-label adaptation, detection delay 6 samples abrupt / 80 gradual / 35 mixed) but sidesteps delayed ground truth rather than measuring it — must be distinguished.
- **Mello 2021/2022 — LOW.** Verification latency in active learning overlaps our selective-labeling regime, not detector isolation.

### Closest prior work

**Verdict: Botacin & Gomes 2024 is the single closest paper; Ch3 must add detector-level isolation, formal bounds, and a confirmation protocol to be a genuine contribution.**

Their key numbers: delayed labels increase user exposure ~50% (up to 75% detection decrease); exposure +50% at 81 epochs delay (DDM) and 61 (EDDM); ~50k/60k exposure-point increase per 10 epochs of delay; 10% queue capacity ≈ 100+ epochs of delay; a pseudo-label secondary classifier reduces the detection loss from ~70% to ~30%; exposure metric goes 0.49→0.90 while AUT-F1 goes 0.52→0.18 as delay goes 0→80; results averaged over 10 executions on DREBIN (250 epochs × 500 samples) and replicated on Androzoo (~284k samples, ~270 epochs). What Ch3 must add beyond this: (1) detector-level isolation — clean curves of detection delay vs label latency and false-alarm rate vs label coverage, on synthetic streams with known drift positions; (2) the formal bounds of lemmas 1–3; (3) a confirmation protocol (label-free early warning within a tolerance window → delayed supervised confirmation → calibrated action) evaluated against naive-delayed, label-free-only, and immediate-label controls; and (4) no claim of a novel time-aware metric without contrasting with the exposure metric and Pendlebury 2019's AUT metric.

### Recommended baseline to reproduce

**Verdict: reproduce Cerqueira et al. (KDD'26); fallbacks are Gonçalves 2014 and Gomes 2022's AGRAWAL experiment.**

Chosen over the alternatives because it is the most recent (2026), venue-strong (KDD), open-access (CC-BY) with public code, uses immediate labels (the correct control condition for Phase 0), and is naturally extendable — we can take the field's new standard benchmark and break it with delayed-label regimes in Ch3. The reproduction target: one detector comparison from it, e.g., ADWIN vs DDM vs KSWIN on one injected-drift real dataset under both abrupt and gradual transitions, using their F1-detection score and normalized detection time; River and CapyMOA both implement all three detectors, so a 2–3 detector, 1–2 dataset subset is realistic. Stretch pilot: reproduce Gomes 2022's delayed-AGRAWAL shift (~10,000 instances) as the first delayed-label sanity check.

### Overall go/no-go read

Proceed with the locked Ch1/Ch2/Ch3 scope, with two conditions before implementation starts: reword the gap claim (immediate-label benchmarks now exist; delayed-label ones do not) and read Alencar 2026 in full before finalizing Ch3 novelty. Ch2 — replica divergence, backpressure, restart/recovery cost — is untouched by the entire delayed-label literature and remains our most defensible novelty with zero scoop risk. Ch3 rests on a combination no single paper has: formal lemmas 1–3 (novel as detector-level statements), lemma 4 (Žliobaitė's result, credited and operationalized), detector-level delay/coverage/batch curves on synthetic ground truth, and a hybrid confirmation protocol that Gomes 2022 and STUDD 2023 explicitly name as open and that no logged paper builds or evaluates. The literature supports proceeding; it does not support claiming the delayed-label measurement itself as new.

---

## Papers reviewed

### Ch1 — Detector baselines, prequential evaluation, tooling

**[Cerqueira, Gomes, Heyden, Pfahringer & Bifet 2026] — A Framework for Evaluating and Benchmarking Concept Drift Detection Methods.** KDD '26, pp. 8671–8682. Benchmarks drift detection by injecting controlled distributional changes into real datasets via Monte Carlo trials: 14 detectors, 7 real datasets, 4 drift types (class prior, label swap, feature permutation, feature filtering), each abrupt and gradual, with new timing-aware metrics (F1 detection score, normalized detection time) and a leave-one-dataset-out hyperparameter protocol. KDD'26 acceptance rate 21% (1,157 of 5,395 submissions). Bears on Ch1 as the reference protocol and the Phase-0 reproduction target; immediate-label only, which is precisely the gap Ch3 fills. Verified: ✅

**[Kulbach, Cazzonelli, Ngo, Halford & Mastelini 2025] — DeepRiver: A Deep Learning Library for Data Streams.** Journal of Open Source Software 10(105):7226. Adds PyTorch deep learning to River's streaming API — class-incremental learning, stream-specific optimizers, autoencoder anomaly detection. No quantitative results in the paper (software paper); tooling/background only, since the project excludes deep learning by scope. Verified: ✅

**[Gomes & Bifet 2024] — Practical Machine Learning for Streaming Data.** KDD '24 tutorial (2 pp.). Demonstrates stream-ML practice via CapyMOA (Python API over MOA): drift detection and evaluation (ADWIN, STUDD), clustering/anomaly methods, and a section on partially and delayed labeled streams using cluster-and-label and incremental self-training. No numbers (tutorial); confirms CapyMOA as a viable supplement to River. Verified: ✅

**[Montiel, Halford, Mastelini, Bolmier, Sourty, Vaysse, Zouitine, Gomes, Read, Abdessalem & Bifet 2021] — River: Machine Learning for Streaming Data in Python.** JMLR 22(110):1–8. The tooling paper for our chosen framework: dict-based API, incremental models, and the exact detector set Ch1 needs — ADWIN, DDM, EDDM, Page-Hinkley, KSWIN, HDDM — plus Hoeffding trees, ARF, SRP, and prequential metrics. No numbers (library); the version to pin for Phase 0. Verified: ✅

**[Gonçalves Jr, de Carvalho Santos, Barros & Vieira 2014] — A Comparative Study on Concept Drift Detectors.** Expert Systems with Applications 41(18):8144–8156. Systematic comparison of DDM, EDDM, Page-Hinkley, STEPD, DoF, ADWIN, Paired Learners, and ECDD under abrupt/gradual drift, noise, and irrelevant attributes; the logged conclusion is the canonical "no single detector wins" — qualitative-only in our notes (no accuracy table captured). Ch1 comparator and justification for our sweep design. Verified: ⚠️ (DOI cross-checked in the KDD'26 reference list; full text not re-fetched)

**[Gama, Sebastião & Rodrigues 2013] — On Evaluating Stream Learning Algorithms.** Machine Learning 90(3):317–346. Establishes prequential (test-then-train) evaluation with sliding-window or fading-factor forgetting as the correct protocol for ordered streams, since random splits leak temporal information. No numbers (methodology). Foundational for Ch1 — and the protocol that breaks under delayed labels, which is our Ch3 motivation. Verified: ✅

**[Bifet & Gavaldà 2009] — Adaptive Learning from Evolving Data Streams (HAT).** IDA. Introduces the Hoeffding Adaptive Tree, which puts ADWIN at tree nodes to detect and adapt to drift; logged claim is superior adaptability on Adult and Poker-Hand across HAT-INC/EWMA/ADWIN variants — qualitative-only in our notes (no numeric scores captured). Ch1 adaptive-model baseline. Verified: ✅

**[Bifet & Gavaldà 2007] — Learning from Time-Changing Data with Adaptive Windowing (ADWIN).** SIAM SDM. The canonical adaptive-window detector with false-positive guarantees; foundational, no numbers logged. Ch1 detector baseline and the natural subject of our Lemma-2 power-loss experiment. Verified: ✅

**[Baena-García, del Campo-Ávila, Fidalgo, Bifet, Gavaldà & Morales-Bueno 2006] — Early Drift Detection Method (EDDM).** 4th Intl Workshop on Knowledge Discovery from Data Streams. Detects drift from distances between errors to better catch gradual drift; foundational, no numbers logged. Ch1 detector baseline, and one of the two detectors used in the Botacin & Gomes pipeline. Verified: ⚠️ (cross-checked in KDD'26 and STUDD reference lists; not separately fetched)

**[Gama, Medas, Castillo & Rodrigues 2004] — Learning with Drift Detection (DDM).** SBIA. Applies statistical process control to the error stream with warn/alarm thresholds; foundational, no numbers logged. Ch1 detector baseline and the subject of our Lemma-3 batch-arrival experiment. Verified: ✅

**[Domingos & Hulten 2000] — Mining High-Speed Data Streams (VFDT / Hoeffding Tree).** KDD, pp. 71–80. One-pass decision tree using the Hoeffding bound with constant memory per leaf; reported constant-time-per-example and high throughput, 1,598 citations on ACM DL. Ch1 static-model baseline. Verified: ✅

**[Widmer & Kubat 1996] — Learning in the Presence of Concept Drift and Hidden Contexts.** Machine Learning 23(1):69–101. Early formalization of drift and hidden contexts via the FLORA2 algorithm; foundational, no numbers logged. Background for Ch1's drift framing. Verified: ⚠️ (cross-checked in multiple reference lists; not separately fetched)

**[Ditzler, Roveri, Alippi & Polikar 2015] — Learning in Nonstationary Environments: A Survey.** IEEE Computational Intelligence Magazine. Foundational quick-reference item: survey of nonstationary learning; no numbers logged. Background only. Verified: — (foundational quick-reference; unflagged in index)

**[Lorden 1971] — Procedures for Reacting to a Change in Distribution.** Annals of Mathematical Statistics. Foundational quick-reference item: minimax quickest detection, i.e., the detection-delay vs false-alarm trade-off that frames our Lemma 1. No numbers logged. Verified: — (foundational quick-reference; unflagged in index)

**[Page 1954] — Continuous Inspection Schemes.** Biometrika. Foundational quick-reference item: root of Page-Hinkley/CUSUM and of the quickest-detection framing behind our Lemma-1 reaction floor. No numbers logged. Verified: — (foundational quick-reference; unflagged in index)

### Ch2 — Systems / operational cost of continual learning

**[Ebadi & Keller 2025] — Optimizing Machine Learning Inference Serving Systems: A Survey.** (Semantic Scholar). Surveys resource-usage and latency optimization for ML inference serving over the last three years. No numbers logged (survey). Informs Ch2's latency/throughput framing; no delayed-label content. Verified: ⚠️ (author list not independently cross-checked)

**[Surianarayanan, Kunasekaran & Chelliah 2023/2024] — A High-Throughput Architecture for Anomaly Detection in Streaming Data Using Machine Learning Algorithms.** International Journal of Information Technology 16:493–506 (published online Nov 2023). Kafka publish-subscribe architecture with two alternating subscribers to hide ML computation time, Random Forest threshold-based point anomaly detection. Reported 96.7% accuracy at 37.5 ms computation (single node) and 98.6% at 38.5 ms (distributed). Ch2 reference for our event-source and backpressure design (Redis Streams vs Kafka decision). Verified: ✅

**[Ceschin, Gomes, Botacin, Bifet, Pfahringer, Oliveira & Grégio 2023] — Machine Learning (In) Security: A Stream of Problems.** Expert Systems with Applications 212:118590. Positions malware detection as an evolving data-stream problem and catalogs pipeline-realism gaps (drift, delayed labels, imbalance). No numbers (survey/positioning, qualitative-only). Production-motivation citation for Ch2 and Ch3. Verified: ⚠️ (author-hosted PDF; venue cross-checked in Botacin & Gomes references)

**[Souza, dos Reis, Maletzke & Batista 2020] — Challenges in Benchmarking Stream Learning Algorithms with Real-world Data.** Data Mining and Knowledge Discovery 34:1805–1858. Shows real-world stream benchmarks suffer undocumented preprocessing and ordering that confound results, and supplies the benchmark repository STUDD used. No numbers (methodology). Reinforces our Elec2/Kappa+ caution for Ch1. Verified: ✅

**[Pendlebury, Pierazzi, Jordaney, Kinder & Cavallaro 2019] — TESSERACT: Eliminating Experimental Bias in Malware Classification Across Space and Time.** USENIX Security. Argues temporal and spatial bias inflates malware ML evaluation and proposes the AUT (area under time) metric with time-aware splits; our notes record the bias-reduction claim qualitatively, without specific numbers. Ch2/Ch3 metric to contrast against the exposure metric. Verified: ⚠️ (USENIX page; content cross-checked via Botacin & Gomes references)

**[Bifet, Holmes, Pfahringer, Read, Kranen, Kremer, Jansen & Seidl 2011] — MOA: A Real-Time Analytics Open Source Framework.** ECML-PKDD. The reference Java stream-mining framework that River and CapyMOA interoperate with. No numbers (tooling). Ch2 infrastructure/tooling contrast. Verified: ⚠️ (Springer chapter page; cross-checked in Grzenda references)

### Ch3 — Delayed / partial / selective labels + drift detection

**[Alencar, Cassales, Gomes, Prazeres, Rios, Bifet & Rios 2026] — Concept Drift Detection in Delayed and Partially Labeled Data Streams: An Experimental Survey.** Digital Signal Processing 182:106325. The closest survey to all of Ch3: organizes detectors under delayed and partially labeled streams, provides a detector taxonomy, and runs label-scarcity experiments. Existence verified; **results unread (paywalled)** — if it already reports delay-vs-latency and label-coverage-vs-false-alarm curves for standard detectors, part of our measurement chapter is taken. Highest scoop risk in the review. Verified: ✅ (existence via dblp + ScienceDirect + Semantic Scholar); content unread.

**[Agu, Andok, Klarák & Dora 2026] — Statistical Drift Detection with Parametric Gamma–Weibull and Adaptive Deep Learning for Malware Detection in IoT Data Streams.** Journal of King Saud University — Computer and Information Sciences 38:673. A Gamma–Weibull log-likelihood-ratio CUSUM detector (ARL₀ ≈ 2000) gates LSTM/CNN fine-tuning on a post-drift buffer labeled by pseudo-labels (τ=0.80) plus K-means. Reported detection delays of 6 samples (abrupt), 80 (gradual), and 35 (mixed) at zero false alarms in noise-free simulation (T=10,000, MC=30); in a 20-replication comparison, ADWIN was fastest (1.1±1.2) but fired ~1,270 false alarms, Page-Hinkley 125.2±106 with 0, KSWIN 176.3±272.9 with ~126, DDM 425.9±401 with ~625; adaptation raised LSTM macro-F1 0.977→0.981 and CNN 0.457→0.576; full pipeline 1.43 s CPU, monitoring <150 ms, pseudo-labeling 2.4 ms, model memory <50 KB. Real data: IoT-23 benign (n=451) and malware (n=14,784) duration streams, first alarm at t=31 vs ADWIN t=7, PHT t=75, KSWIN t=251, DDM none. Adjacent hybrid — must be distinguished from Ch3, which measures delayed ground truth rather than sidestepping it with pseudo-labels. Verified: ✅

**[Botacin & Gomes 2024] — Towards More Realistic Evaluations: The Impact of Label Delays in Malware Detection Pipelines.** Computers & Security, vol. 147. Models malware detection as a delayed+partially-labeled evolving stream and measures how analysis-queue-driven label delay degrades DDM/EDDM drift retraining on DREBIN (250 epochs × 500 samples) and Androzoo (~284k samples, ~270 epochs), introducing the exposure metric. Delayed labels raised user exposure ~50% (up to 75% detection decrease); exposure hit +50% at 81 epochs of delay (DDM) and 61 (EDDM); exposure grows ~50k points (DDM) / ~60k (EDDM) per 10 epochs of delay; a 10% queue capacity maps to 100+ epochs of delay (queue capacity can drop to 70% at the same exposure with pseudo-labels); a pseudo-label secondary classifier cut the detection loss from ~70% to ~30%; the exposure metric runs 0.49→0.90 while AUT-F1 runs 0.52→0.18 as delay goes 0→80; averages over 10 executions with FPR < 1% enforced. The closest prior work to Ch3 — see the Closest prior work section for what must be added. Verified: ✅

**[Cerqueira, Gomes, Bifet & Torgo 2023] — STUDD: A Student–Teacher Method for Unsupervised Concept Drift Detection.** Machine Learning 112(11):4351–4378. A student model mimics the deployed teacher and Page-Hinkley (δ=0.001) monitors the mimicking loss — fully label-free. On 19 benchmark streams (truncated to 150k each; W=2000, 500 for small sets), missed-detection ratio ~19% (3/16 datasets, tied with DWS), best label efficiency among non-baselines, and — directly relevant to our Lemma 4 — it explicitly cannot catch posterior drift that leaves P(X) unchanged; sensitivity sweeps over label access {1, 10, 25, 50}% (delay {250..4000}) shift relative method ordering, with STUDD relatively better as label access drops and worst at 50% access (qualitative). Candidate Stage-1 component for our hybrid. Verified: ✅

**[Gomes, Grzenda, Mello, Read, Le Nguyen & Bifet 2022] — A Survey on Semi-supervised Learning for Delayed Partially Labelled Data Streams.** ACM Computing Surveys 55(4), Article 75. The anchor survey: formalizes the setting (immediate/delayed/infinite labels × supervised/semi-supervised), surveys SSL methods and evaluation practice, and includes an AGRAWAL experiment (100k instances, 3 drifts) where a 10,000-instance label delay shifts ADWIN+SRP detections by ~10,000 instances and holds prequential accuracy below 50%. 63 citations / 6,779 downloads on ACM DL. Names our gap and the under-explored delayed+partial intersection as open. Verified: ✅

**[Mello, Gomes, Bifet & Pfahringer 2022] — Stream-Based Active Learning with Verification Latency in Non-stationary Environments (PRopagate).** IJCNN (arXiv:2204.06822). Estimates query utility under unknown label delay and predicts requested-but-unavailable labels, with a drift-dependent labeling budget. Our notes hold only the qualitative claim of improvement over active-learning baselines under constant and variable latency — no numeric results logged yet. Bears on the selective-labeling regime ("only flagged cases are labeled"). Verified: ✅ (arXiv resolves); results qualitative-only in our notes.

**[Cano & Krawczyk 2022] — ROSE: Robust Online Self-Adjusting Ensemble for Continual Learning on Imbalanced Drifting Data Streams.** Machine Learning 111:2561–2599. Online ensemble with variable-size random feature subspaces (μ=0.7), ADWIN-per-subspace drift detection with a background ensemble (replacement after 1000 instances), per-class sliding windows (1000-instance total budget), and self-adjusting λ bagging (λ_min=4; λ = λ_min + log10(#majority/#instance)×λ_min). Compared against 30 ensembles on 24 imbalanced real datasets, 36 imbalanced generators, and 12 drifting-imbalance generators; best mean Kappa 63.65 vs next-best 57.85 (KUE) at static imbalance, 59.95 vs 54.45 (LB) under drifting imbalance, and most noise-robust (Kappa 59.95→36.54 at 0→40% noise, rank 1.70); ablation: 1→10 adaptation epochs raises F1 0.965→0.983; runs on a 384 GB Xeon. Immediate labels only; its background-ensemble adaptation pattern informs Ch2 and the adaptation side of Ch3. Verified: ✅

**[Carneiro, Guimarães, Silva & Novais 2021] — A Predictive and User-Centric Approach to Machine Learning in Data Streaming Scenarios.** Neurocomputing (preprint May 2021). Meta-learning predicts whether retraining a model on new data will improve performance before spending the compute, paired with an explainable interface; meta-model trained on 673 meta-instances from 47 datasets (53 total), 10-fold CV RMSE 0.000355, r² 0.98, MAE 0.007, re-trained every 2,000 streamed instances; batch-test RMSE predictions 0.111 vs 0.102 observed (wine), 0.321 vs 0.400 (diabetes), 0.093 vs 0.161 (breast cancer). Relevant to Ch2's adaptation-cost framing and selective adaptation, not delayed-label detection. Verified: ⚠️ (full text in Material; DOI inferred from filename — verify before citing)

**[Mello, Gomes, Bifet & Pfahringer 2021] — Stream-Based Active Learning for Sliding Windows under the Influence of Verification Latency.** Machine Learning (online 2021). Forgetting and Simulating: simulates the learner's future state at label-arrival time to improve utility estimates. Our notes log only the qualitative improvement claim under constant and variable latency. Adjacent to our selective-labeling regime; not detector isolation. Verified: ⚠️ (DOI cross-checked in Gomes survey reference list; not separately fetched)

**[Grzenda, Gomes & Bifet 2020] — Delayed Labelling Evaluation for Data Streams.** Data Mining and Knowledge Discovery 34:1237–1266. Continuous re-evaluation: buffer predictions and re-poll the evolving model before the delayed label arrives, scoring binned predictions (initial / periodic / test-then-train). Applied to Hyperplane (delay 100), LED (1000), AGRAWAL (1000, 4 gradual drifts), Airlines (natural latency up to 13 h), Electricity (~1–2 days), Forest cover (1000); recommends K=10 re-polls and B=50 bins (50–100 validated); open-source MOA implementation. The standard evaluation protocol for delayed-label streams — Ch3 should use it, not reinvent it. Verified: ✅

**[Dal Pozzolo, Boracchi, Caelen, Alippi & Bontempi 2015] — Credit Card Fraud Detection and Concept-Drift Adaptation with Delayed Supervised Information.** IEEE IJCNN. Shows immediate investigator feedback and delayed chargeback labels carry different information, and that separate models with output aggregation beat identical treatment. Our notes hold only the qualitative improvement claim. Empirical anchor for the fraud dataset story and the two-signal structure of our hybrid. Verified: ⚠️ (citation cross-checked in Gomes survey reference list; Semantic Scholar fetch bot-walled)

**[Joulani, György & Szepesvári 2013] — Online Learning under Delayed Feedback.** ICML, PMLR 28(3):1453–1461. Theoretical: delay multiplies regret in adversarial online learning and adds it in stochastic settings, with black-box meta-algorithms for delayed feedback. No numeric results (theory). The regret-side cousin of our reaction floor — cite for the theoretical framing of delay costs. Verified: ✅

**[Masud, Gao, Khan, Han & Thuraisingham 2011] — Classification and Novel Class Detection in Concept-Drifting Data Streams under Time Constraints.** IEEE TKDE. Foundational quick-reference item: delayed labels plus concept evolution; no numbers logged. Verified: — (foundational quick-reference; unflagged in index)

**[Žliobaitė 2010] — Change with Delayed Labeling: When is it Detectable?** IEEE ICDM Workshops. Analytical conditions for which drift types are detectable without new labels — pure posterior drift with fixed P(X) is not — which is the direct basis of our Lemma 4; two drift types empirically evaluated, no numeric results in our notes. Does not propose a hybrid architecture, which is why the hybrid remains open. Verified: ✅

**[Kuncheva & Sánchez 2008] — Nearest Neighbour Classifiers for Streaming Data with Delayed Labelling.** IEEE ICDM. Foundational quick-reference item: early delayed-label classification work; no numbers logged. Verified: — (foundational quick-reference; unflagged in index)

### Ch3 — Hybrid / label-free + confirmation methods

**[Shen, Ming, Li, Gao & Zhang 2023] — Unsupervised Concept Drift Detectors: A Survey.** Springer (survey chapter). Surveys label-free detector families separately from supervised error-based methods. No numbers (survey). Catalogs the Stage-1 options for our hybrid; no delayed-confirmation treatment. Verified: ⚠️ (author list per Gomes survey; not separately fetched)

**[Pinto, Sampaio & Bizarro 2019] — Automatic Model Monitoring for Data Streams.** arXiv:1908.04240 (Feedzai). Windowed Jensen–Shannon divergence drift monitoring with alarm explanations from an auxiliary classifier. No numbers logged (qualitative-only from abstract). A label-free Stage-1 candidate for the hybrid. Verified: ⚠️ (arXiv abstract; cross-checked in STUDD references)
