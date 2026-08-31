The problem is **not solved**, but it is also not untouched. There is a recognizable research thread spanning roughly 2008–2026, including a dedicated 2022 survey and a 2026 experimental survey focused specifically on drift detection with delayed and partial labels. However, the literature is fragmented: delayed labels, verification latency, partial labels, extreme verification latency, semi-supervised streams, active learning, and positive–unlabelled learning are usually studied separately rather than as one unified deployment problem.

The strongest conclusion is that **there is still a substantial research gap in detection, adaptation, evaluation, and benchmarking under realistic label-observation mechanisms**—especially when latency is stochastic, labels are selection-biased, some labels never arrive, and the data-generating process itself drifts.

## 1. How much work exists?

### Overall volume and maturity

The literature falls into several partially overlapping lines:

| Line of work | Approximate development | Main concern |
|---|---:|---|
| Delayed-label stream classification | 2008–present | Updating classifiers when labels arrive after the instance |
| Detecting drift without current labels | 2010–present | Inferring change from \(X\), predictions, uncertainty, or representations |
| Delayed-label evaluation | 2016–present | Measuring performance without immediate test-then-train feedback |
| Semi-supervised streams | 2009–present | Exploiting unlabeled examples during drift |
| Stream-based active learning with latency | 2021–2022 onward | Querying labels whose utility changes before the answer arrives |
| Extreme or infinite verification latency | 2023–2025 onward | Learning and adapting when labels are very rare or never available |
| Dedicated delayed/partial drift surveys | 2026 | Consolidating detector methods and experiments |

The most important synthesis is **Gomes, Grzenda, Mello, Read, Le Nguyen, and Bifet, “A Survey on Semi-supervised Learning for Delayed Partially Labelled Data Streams,” ACM Computing Surveys, 2022**. It defines a unified setting containing both unlabeled observations and delayed or permanently unavailable labels, reviews semi-supervised and active-learning methods, discusses learning guarantees, and proposes changes to benchmarking practice. [dl.acm](https://dl.acm.org/doi/fullHtml/10.1145/3523055)

A second important synthesis is **Alencar, Cassales, Gomes, Prazeres, Rios, Bifet, and Rios, “Concept drift detection in delayed and partially labeled data streams: An experimental survey,” Digital Signal Processing, vol. 182, 2026, article 106325**. This is particularly relevant to your proposed gap because it organizes drift detectors under delayed and partially labeled conditions and experimentally examines detector performance as label availability decreases. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1051200426004434)

So the area is active, but the volume is modest compared with ordinary concept-drift research. There is a large literature on concept drift assuming immediate labels; a much smaller literature studies the consequences of violating that assumption.

### Terminology

The following terms are substantially overlapping but not perfectly interchangeable:

- **Immediate labeling** or **immediate-label setting**: the label is available immediately after prediction, usually before the next instance is processed.
- **Delayed labeling** or **delayed labelling**: a label arrives after a finite delay.
- **Label delay**: the time or number of instances between feature arrival and label arrival.
- **Label latency**: a newer, broader term for the same delay; often defined as \(d=t_Y-t_X\).
- **Verification latency**: probably the most established term in concept-drift and active-learning papers. It emphasizes that the label is obtained only after an external verification process.
- **Delayed supervised information**: common in fraud-detection work, where investigator feedback or chargeback information arrives later.
- **Partially labeled data streams** or **partially labelled data streams**: only a subset of observations receives labels.
- **Delayed partially labeled data streams**: the unified term used by Gomes et al. for streams that combine finite delays with labels that may never arrive.
- **Weakly supervised streams**: labels are incomplete, noisy, coarse, or indirect; this can include but is not limited to delayed labels.
- **Infinite verification latency**: a label is effectively never received.
- **Extreme verification latency (EVL)**: recent papers use this for very long or indefinite delays, particularly when labels are unavailable for extended periods. AiGAS-dEVL explicitly uses this terminology for streams where labels are delayed or may remain unavailable indefinitely. [arxiv](https://arxiv.org/html/2504.05761v1)
- **Delayed feedback**: broader than delayed labels; it can include rewards, outcomes, user responses, fraud confirmations, or other feedback signals.
- **Delayed supervision**: another broad term, often used in continual learning and online learning.
- **Delayed ground truth** or **delayed outcomes**: common in applied papers, especially healthcare, finance, and forecasting.
- **Missing labels**, **label scarcity**, and **label sparsity**: usually refer to partial availability, although they do not necessarily imply a time-based delay.
- **Selective labeling**, **verification bias**, or **selective labels**: labels are observed conditional on an action, such as reviewing only transactions that were flagged.
- **Positive–unlabelled learning (PU learning)**: a related but distinct setting in which positive examples are observed and the remaining examples are unlabeled; the unlabeled set is not automatically equivalent to delayed or missing labels.

Historically, **Kuncheva and Sánchez used “label delay,” while Marrs and colleagues used “verification latency”**; later papers use both terms. The terminology inconsistency is real and can hide relevant work. The 2022 survey is useful partly because it tries to distinguish these settings formally. [link.springer](https://link.springer.com/article/10.1007/s10994-021-06099-z)

## 2. Detection without timely labels

### What happens to supervised detectors?

Classical detectors such as DDM, EDDM, Page-Hinkley, CUSUM, ADWIN, and accuracy-based ensemble detectors usually monitor a loss or error sequence:

\[
e_t = \mathbb{1}[\hat y_t \ne y_t].
\]

With verification latency, \(e_t\) cannot be computed at time \(t\). It becomes available only when the corresponding label arrives. Therefore:

1. The detector cannot react earlier than the label-observation process allows.
2. The errors it receives correspond to historical predictions, not necessarily the current model or current concept.
3. Labels may arrive out of order.
4. A long delay can produce stale alarms.
5. Batch or bursty label arrivals can produce artificial bursts in the error stream.
6. Partial labeling reduces the effective sample size and changes the missingness mechanism.

The theoretical issue was analyzed by **Žliobaitė, “Change with Delayed Labeling: When is it Detectable?” IEEE ICDM Workshops, 2010**. The paper characterizes which types of changes can be detected without observing new labels and proposes monitoring classifier outputs as a computationally practical alternative. It shows that some changes are identifiable from unlabeled inputs or prediction behavior, while other changes are fundamentally indistinguishable without labels. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/5693384)

This last point is crucial: a detector using only \(X\) cannot generally detect a pure conditional drift such as

\[
P_t(Y\mid X)\ne P_{t+\Delta}(Y\mid X)
\]

when \(P_t(X)\) remains unchanged. Conversely, input-distribution drift can occur without any degradation in predictive accuracy.

### Has degradation been quantified?

**Only partially and inconsistently.**

There are many papers showing that delayed labels harm adaptation or active learning, but relatively few provide a clean detector-level curve of:

- detection delay versus label latency;
- false-alarm rate versus label coverage;
- missed-drift rate versus partial-label mechanism;
- performance under constant versus stochastic latency;
- performance under missing-not-at-random labels.

The 2026 experimental survey appears to be the first recent work aimed directly at organizing and experimentally comparing concept-drift detection under delayed and partially labeled streams. Its existence is itself evidence that this evaluation problem was not previously standardized. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1051200426004434)

Earlier studies establish the mechanisms of degradation, but often study classifiers or active-learning systems rather than isolating the detector. For example, **Plasse and Adams, “Handling Delayed Labels in Temporally Evolving Data Streams,” IEEE Big Data, 2016**, provides a taxonomy of delay mechanisms and discusses how delayed labels affect stream learning. [computer](https://www.computer.org/csdl/proceedings-article/big-data/2016/07840877/12OmNC8MsCY)

In fraud detection, **Dal Pozzolo, Boracchi, Caelen, Alippi, and Bontempi, “Credit Card Fraud Detection and Concept-Drift Adaptation with Delayed Supervised Information,” IEEE IJCNN, 2015**, separates immediate investigator feedback from delayed labels and finds that training separate models on the two information sources and aggregating their outputs is better than treating them identically. [semanticscholar](https://www.semanticscholar.org/paper/Credit-card-fraud-detection-and-concept-drift-with-Pozzolo-Boracchi/b769ec8f3be051b7c5f9a0ba5f85a38594950df9)

That is relevant evidence, but it is not a general detector benchmark. It is an application-specific adaptation strategy.

### Label-free detectors

Label-free detectors can be grouped into four broad families:

| Family | What it monitors | Main advantage | Main limitation |
|---|---|---|---|
| Input-distribution tests | Changes in \(P(X)\) | Works immediately | Misses pure posterior drift |
| Representation or embedding tests | Changes in learned feature distributions | More useful for high-dimensional data | Depends on representation stability |
| Prediction-distribution tests | Changes in \(P(\hat Y)\), confidence, or output histograms | Uses model information without ground truth | Can confuse prior shift, uncertainty, and model bias |
| Model/uncertainty monitors | Confidence, entropy, disagreement, reconstruction error, residual proxies | May provide early warning before labels | Proxy can change without actual error increase |

Examples include statistical tests, window-based divergence measures, clustering, density monitoring, reconstruction-error monitoring, classifier disagreement, and prediction-uncertainty methods.

**Shen, Ming, Li, Gao, and Zhang, “Unsupervised Concept Drift Detectors: A Survey,” Springer, 2023**, surveys unsupervised detectors and classifies them separately from supervised error-based methods. [link.springer](https://link.springer.com/chapter/10.1007/978-3-031-20738-9_121)

**Hinder, Vaquet, and Hammer, “One or Two Things We Know about Concept Drift—A Survey on Monitoring Evolving Environments,” 2023**, focuses specifically on unsupervised data streams and drift localization. The authors note that much of the prior survey literature assumes supervision, despite the importance of unsupervised monitoring for anomaly detection and engineering systems. [arxiv](https://arxiv.org/abs/2310.15826)

A later benchmark, **“A Benchmark and Survey of Fully Unsupervised Concept Drift Detection,” International Journal of Data Science and Analytics, 2024**, evaluates fully unsupervised detectors on real-world streams and reports that performance depends strongly on the metric and data set; no detector dominates universally. [d-nb](https://d-nb.info/1348118334/34)

Recent representation-based examples include **DriftLens**, which detects distributional changes in deep-model embeddings without requiring labels, and prediction-uncertainty approaches such as **PUDD**, which use changes in uncertainty rather than error rate as an early-warning signal. [arxiv](https://arxiv.org/html/2406.17813v2)

### Comparison with supervised detectors

There is no accepted universal ranking, because the methods detect different phenomena:

- A supervised detector is usually more semantically aligned with **real performance degradation**, but its earliest possible alarm is bounded by label latency.
- An input-based detector can alarm immediately, but it may detect harmless covariate shift.
- An output or uncertainty detector may provide a useful compromise, but its relationship to true error depends on calibration and class balance.
- In high-dimensional data, simple distribution tests may have low power unless applied to a suitable representation.
- Under selective labeling, supervised detectors can be biased because the observed errors are not representative of the stream.

Thus, comparing “detection delay” alone is misleading. A label-free detector may alarm earlier but with more false positives or with respect to a different definition of drift.

### Hybrid early warning and confirmation

There are several partial examples, but not yet a mature, generally accepted framework.

The clearest early work is Žliobaitė’s prediction-output monitoring: use a classifier trained on older labeled data, monitor its predictions on current unlabeled observations, and use later labels to confirm or refine the alarm. [research.tue](https://research.tue.nl/en/publications/change-with-delayed-labeling-when-is-it-detectable/)

Fraud-detection systems also use a practical hybrid pattern:

- immediate investigator feedback;
- delayed outcome labels;
- unsupervised or anomaly-based signals for unlabeled transactions;
- separate models or memory windows for immediate and delayed information.

Dal Pozzolo et al. found that separating immediate feedback from delayed labels was beneficial in a real fraud setting. [semanticscholar](https://www.semanticscholar.org/paper/Credit-card-fraud-detection-and-concept-drift-with-Pozzolo-Boracchi/b769ec8f3be051b7c5f9a0ba5f85a38594950df9)

The 2022 active-learning work by **Mello et al., “Stream-Based Active Learning with Verification Latency in Non-stationary Environments,” 2022** proposes PRopagate, which estimates the utility of a query despite unknown label delay and predicts requested but unavailable labels. It also proposes a drift-dependent labeling-budget strategy and reports improvements over existing active-learning methods. [arxiv](https://arxiv.org/abs/2204.06822)

The related **Mello et al., “Stream-based Active Learning for Sliding Windows under the Influence of Verification Latency,” Machine Learning, 2021** proposes Forgetting and Simulating. It simulates the learner’s future state when a queried label will arrive, rather than estimating utility using the present state. Experiments report improvements under both constant and variable latency. [dl.acm](https://dl.acm.org/doi/10.1007/s10994-021-06099-z)

These are hybrid in spirit, but they primarily address **label acquisition and adaptation**, not a standardized two-stage detector of the form:

1. label-free alarm;
2. delayed supervised confirmation;
3. calibrated action under confirmation uncertainty.

That architecture remains a promising research direction.

## 3. Evaluation under delayed labels

### Why ordinary prequential evaluation fails

Standard test-then-train evaluation assumes:

1. observe \(x_t\);
2. predict \(\hat y_t\);
3. immediately observe \(y_t\);
4. score the prediction;
5. train on \((x_t,y_t)\).

Under delay, the actual sequence is more like:

1. observe \(x_t\);
2. predict \(\hat y_t\);
3. continue processing many future instances;
4. receive labels for older instances, possibly out of order;
5. update the model using only the labels that have arrived.

If a model is updated with \(y_{t-k}\) at time \(t\), the learner’s state is different from that under ordinary prequential evaluation. Directly replaying the data with immediate labels therefore gives optimistic results.

### Main evaluation proposals

The main dedicated evaluation proposal is **Grzenda, Gomes, and Bifet, “Delayed Labelling Evaluation for Data Streams,” Data Mining and Knowledge Discovery, 34, 1237–1266, 2020**. It introduces **continuous re-evaluation**. [link.springer](https://link.springer.com/article/10.1007/s10618-019-00654-y)

The method:

- makes an initial prediction when \(x_i\) arrives;
- stores the instance while its label is pending;
- periodically predicts again as the model evolves;
- evaluates predictions in time bins between instance arrival and label arrival;
- computes performance both immediately after arrival and just before the label becomes available;
- supports fixed and variable delays.

This is more informative than scoring only the first prediction because it measures whether the system can refine a decision while waiting for ground truth. The authors provide an open-source implementation for MOA. [github](https://github.com/mgrzenda/ContinuousReEvaluation)

A related paper is **Grzenda, Gomes, and Bifet, “Performance Measures for Evolving Predictions under Delayed Labelling Classification,” IEEE IJCNN, 2020**. It formalizes binned performance measures for evolving predictions and treats the final pre-label prediction as a distinct evaluation point. [dl.acm](https://dl.acm.org/doi/10.1145/3523055)

Other approaches include:

- replaying the stream with fixed artificial delays;
- delaying model updates but scoring at the original prediction time;
- evaluating labels only when they arrive;
- using chronological train/validation/test periods;
- reporting performance against the age of available supervision;
- comparing immediate-label and delayed-label versions of the same stream;
- separate evaluation of initial predictions and revised predictions.

### Is there an accepted benchmark?

**No. There is no universally accepted benchmark or protocol.**

The field has useful ingredients:

- MOA and related stream-learning frameworks;
- synthetic generators with injected fixed or stochastic delays;
- real temporal data such as flight delays, electricity, fraud, and software-defect streams;
- continuous re-evaluation;
- fully unsupervised drift benchmarks;
- semi-supervised streaming protocols.

But papers still differ in:

- whether delay is measured in time or instance count;
- whether labels arrive in order;
- whether latency is constant, random, heavy-tailed, or feature-dependent;
- whether missing labels are missing completely at random;
- whether the model may use unlabeled instances for representation learning;
- whether pseudo-labels count as training data;
- whether drift detectors are evaluated for statistical alarm quality or downstream accuracy;
- whether performance is measured when predictions are made or when labels arrive.

The 2022 survey explicitly reviews current benchmarking practices and proposes adaptations, which suggests that protocol consensus remains incomplete. [dl.acm](https://dl.acm.org/doi/fullHtml/10.1145/3523055)

A proper benchmark should specify at least:

\[
\mathcal{O}_t = \{x_t,\; \text{labels revealed by }t,\; \text{query responses by }t\}.
\]

It should then define the learner’s information set at every time \(t\), rather than merely attaching a delay to a conventional labeled data set.

## 4. Adjacent fields

### Semi-supervised streaming

Semi-supervised stream learning is the closest neighboring field. It uses unlabeled \(x_t\) for clustering, density estimation, representation learning, pseudo-labeling, or model regularization, while updating from the labels that eventually arrive.

Important earlier work includes:

- **Abdulsalam, Skillicorn, and Martin, work on intermittent labeling and streaming random forests, 2010**, where labeled and unlabeled blocks alternate and the system decides whether enough labeled data exists to update or trust the model. [link.springer](https://link.springer.com/article/10.1007/s10618-019-00654-y)
- **Masud et al., “Classification and Novel Class Detection in Concept-Drifting Data Streams under Time Constraints,” IEEE TKDE, 2011**, which detects novel classes before their labels become available. [dl.acm](https://dl.acm.org/doi/10.1109/TKDE.2010.61)
- **Larranaga et al., “Classifying Evolving Data Streams with Partially Labeled Data,” 2011**, which uses unlabeled data for drift detection and an EM-based classifier. The method monitors feature, conditional, and dual drift. [oa.upm](https://oa.upm.es/72870/3/LARRANAGA_2011_05_D.pdf)
- **COMPOSE: A Semisupervised Learning Framework for Initially Labeled Nonstationary Streaming Data, IEEE TNNLS, 2013**, which uses unlabeled data in initially labeled nonstationary streams. [dl.acm](https://dl.acm.org/doi/fullHtml/10.1145/3523055)
- **Mello et al., “Online Reliable Semi-Supervised Learning on Evolving Data Streams,” Information Sciences, 2020**, which addresses reliability of semi-supervised updates under evolving distributions. [dl.acm](https://dl.acm.org/doi/fullHtml/10.1145/3523055)

Semi-supervised learning contributes mechanisms for exploiting \(X\) before labels arrive, but it does not automatically solve selective-label bias or guarantee that pseudo-labels remain reliable after drift.

### Active learning on streams

Active learning addresses partial labeling by deciding which instances should be sent to an oracle. It contributes:

- query-budget allocation;
- uncertainty and diversity sampling;
- drift-triggered queries;
- sampling-bias correction;
- label-value estimation;
- explicit treatment of oracle delay.

The important limitation is that many active-learning algorithms assume a queried label is immediately available. The 2021 and 2022 verification-latency papers show that this assumption can make utility estimates stale or even cause active-learning strategies to fail. [arxiv](https://arxiv.org/abs/2204.06822)

The 2023 survey **“Active Learning for Data Streams: A Survey,” Machine Learning, 2023** reviews online active-learning methods and identifies annotation cost, stream evolution, and delayed responses as continuing challenges. [dl.acm](https://dl.acm.org/doi/10.1007/s10994-023-06454-2)

Active learning is particularly relevant to your biased-label scenario, because “only flagged cases are reviewed” is not merely label scarcity: it is an **endogenous observation policy**. The selection mechanism must be modeled, logged, or corrected.

### Self-training and pseudo-labeling

Self-training can increase the effective training rate by assigning labels to high-confidence unlabeled examples. In a drifting stream it is useful for:

- pretraining representations before ground truth arrives;
- maintaining a model during long label gaps;
- expanding class coverage;
- providing an early estimate of current class proportions.

But error propagation is especially dangerous under drift. A wrong pseudo-label can become part of the detector’s reference distribution and cause confirmation bias. Strong confidence calibration, temporal consistency, disagreement-based filtering, and delayed correction are needed.

The literature contains many semi-supervised stream algorithms, but comparatively little work systematically evaluates pseudo-labeling under the combination of:

- concept drift;
- verification latency;
- selection-biased labels;
- class imbalance;
- non-random missingness.

### Positive–unlabelled learning

PU learning is transferable, but only under specific assumptions.

The useful analogy is:

- confirmed positive = observed positive;
- unlabeled = not-yet-confirmed or never-confirmed cases.

However, the analogy breaks when:

- negatives are also eventually observed;
- labels arrive after a known delay;
- confirmation probability changes over time;
- the probability of being labeled depends on the model’s prediction;
- the positive class itself drifts;
- labels arrive through delayed events rather than random sampling.

There is older work specifically titled **“Positive Unlabeled Learning for Data Stream Classification,” SDM, 2009**, which already recognized that concept drift makes traditional PU methods inadequate. [personal.ntu.edu](https://personal.ntu.edu.sg/xlli/publication/SDM09.pdf)

PU methods can contribute:

- class-prior estimation;
- correction for positive-selection bias;
- risk estimators using positive and unlabeled data;
- delayed-positive modeling;
- propensity-weighted learning.

The key research opportunity is to combine PU risk estimation with explicit temporal processes for label arrival. In other words, treat label availability as a stochastic observation process:

\[
P(R_t=1\mid X_t,Y_t,A_t,t),
\]

where \(R_t\) indicates that the true label is observed and \(A_t\) may represent the system’s action, such as flagging or manual review.

## 5. What remains unsolved?

### The central unsolved problem

The missing general solution is a learner that jointly models:

\[
P_t(X,Y,R,L),
\]

where:

- \(X\) is the incoming data;
- \(Y\) is the true target;
- \(R\) indicates whether the label is observed;
- \(L\) is the random label-arrival time;
- all of these may change under concept drift.

Most existing papers simplify this substantially:

- fixed delay;
- labels eventually arrive;
- labels arrive independently of the prediction;
- labels arrive in order;
- missing labels are random;
- the drift type is known or synthetically injected;
- one particular classifier or detector is evaluated.

That is why the problem remains open despite many individual papers.

### Specific open problems

#### 1. Identifiability

A label-free detector cannot distinguish every type of concept drift. Pure changes in \(P(Y\mid X)\) can be invisible when \(P(X)\) does not change. Future work needs explicit guarantees stating which drift classes are identifiable under which observation assumptions.

#### 2. Joint drift and label-process modeling

Most work treats latency as an experimental nuisance rather than a drifting process. In real systems, review capacity, fraud-reporting behavior, medical follow-up, and chargeback timing may all change over time.

The arrival process itself can drift:

\[
P(L_t \mid X_t,Y_t,t) \ne P(L_{t+\Delta}\mid X_{t+\Delta},Y_{t+\Delta},t+\Delta).
\]

This can create apparent concept drift or conceal real drift.

#### 3. Selection bias and verification bias

“Only flagged cases are labeled” is a missing-not-at-random problem. A detector trained on reviewed cases may estimate the performance of the review policy rather than the performance of the deployed model on the full population.

This remains insufficiently integrated into stream concept-drift benchmarks.

#### 4. Detector calibration under delayed errors

A detector should know whether its alarm probability is calibrated when:

- errors arrive in bursts;
- errors correspond to old model versions;
- only a subset of errors is observed;
- the delay distribution is heavy-tailed;
- the stream is strongly imbalanced.

Existing detectors generally do not expose these assumptions.

#### 5. Early warning versus actionable drift

An early warning is not necessarily useful. A label-free detector may identify a change in \(P(X)\) that does not affect decisions, generating expensive or harmful retraining.

Future systems should optimize a cost-sensitive objective involving:

- time-to-action;
- false alarms;
- missed drifts;
- retraining cost;
- delayed confirmation;
- downstream business loss.

#### 6. Confirmation protocols

There is no standard protocol for combining:

1. an unsupervised warning;
2. a temporary safe response;
3. delayed labels;
4. confirmation or rejection of the warning;
5. retrospective correction.

This is perhaps the clearest opportunity for a focused thesis or research program.

#### 7. Evaluation with multiple predictions per sample

Continuous re-evaluation is a strong start, but open issues remain:

- How frequently should a pending sample be re-predicted?
- Should predictions be weighted by time, exposure, or business cost?
- How should predictions be scored when features also evolve after initial arrival?
- How should several model versions be credited for one eventual outcome?
- How should delayed labels arriving out of order be handled?

#### 8. Realistic benchmarks

The field needs benchmark generators and real data sets with explicit metadata for:

- label-arrival timestamps;
- missing-label indicators;
- review or intervention actions;
- model scores that triggered review;
- delayed outcome processes;
- drift annotations;
- label-order violations;
- class imbalance;
- recurrent and gradual drift.

Current practice often creates a synthetic delay by shifting labels by \(k\) instances. That is useful for controlled experiments but does not reproduce fraud, healthcare, or moderation workflows.

#### 9. Learning from label absence

A missing label is not always uninformative. In fraud, no chargeback after a sufficiently long period may be weak negative evidence. In healthcare, lack of follow-up may encode access, severity, or demographic bias.

The literature needs models that distinguish:

- not labeled yet;
- unlikely ever to be labeled;
- labeled negative;
- censored outcome;
- label suppressed by a review policy.

#### 10. Safety of pseudo-label adaptation

A model that self-trains during a long label blackout may adapt quickly—or reinforce its own mistakes. Drift-aware confidence thresholds, ensemble disagreement, abstention, and later label-based correction remain open design problems.

### Recent activity: active or dormant?

The area is **active but fragmented**, not dormant.

Evidence of recent activity includes:

- the 2022 dedicated survey on delayed partially labeled streams; [dl.acm](https://dl.acm.org/doi/fullHtml/10.1145/3523055)
- 2021–2022 work on active learning with verification latency; [arxiv](https://arxiv.org/abs/2204.06822)
- 2023 surveys of unsupervised drift detection and stream active learning; [link.springer](https://link.springer.com/chapter/10.1007/978-3-031-20738-9_121)
- the 2024 benchmark of fully unsupervised drift detection; [d-nb](https://d-nb.info/1348118334/34)
- 2024–2025 work on extreme verification latency and label-free adaptation, including AiGAS-dEVL; [arxiv](https://arxiv.org/html/2407.05379v1)
- the 2026 experimental survey dedicated to delayed and partially labeled drift detection. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1051200426004434)

At the same time, the publication pattern suggests that the field has not yet converged on a canonical problem formulation, benchmark, or dominant algorithm. The newest papers often introduce a method for one specific combination of delay, drift, data type, and application rather than resolving the general problem.

## Bottom-line assessment

You would not be entering an empty field, but you would also not be duplicating a solved problem.

A defensible research contribution would need to go beyond “apply ADWIN without labels” or “use pseudo-labeling under delay.” The strongest gap is:

> **A unified, benchmarked framework for concept-drift detection and adaptation when labels arrive through a delayed, partial, and selection-biased observation process, with label-free early warnings and delayed supervised confirmation.**

A particularly strong project could contribute:

1. a formal observation model for delayed and selectively observed labels;
2. a taxonomy of immediate, finite-delay, stochastic-delay, partial, and infinite-latency settings;
3. a hybrid detector combining input/representation/uncertainty signals with delayed error confirmation;
4. detection-delay, false-alarm, missed-drift, and downstream-cost metrics;
5. synthetic generators plus at least one real temporal benchmark;
6. ablations separating covariate drift, posterior drift, label-process drift, and review-policy drift.

The literature already supplies the foundations—Žliobaitė’s detectability analysis, Plasse and Adams’s delay taxonomy, Grzenda et al.’s continuous re-evaluation, the Gomes et al. survey, active-learning-with-latency methods, and recent unsupervised-detector benchmarks—but **no widely accepted framework currently combines all of these pieces**.
