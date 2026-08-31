# Literature Survey: Online Learning and Concept Drift Adaptation

## 1. Foundations and taxonomy

### 1.1 Origins

Online learning predates modern stream mining. Its roots include stochastic approximation, the perceptron, adaptive filters, sequential Bayesian inference, and incremental statistics. The modern data-stream-mining field emerged in the late 1990s and early 2000s because databases and sensor systems began producing data that were effectively unbounded, arrived continuously, and could not be stored or repeatedly scanned.

The central constraints are:

- Each example may be observed only once.
- The stream may be too large to store.
- Predictions must be available while learning continues.
- Computation and memory per instance should be bounded.
- The data-generating process may be non-stationary.

Domingos and Hulten’s VFDT paper is generally regarded as a foundational stream-classification work: it showed that a decision tree could be learned in one pass, with constant memory per leaf, using the Hoeffding bound to approximate the tree that would have been learned from a much larger batch.  The MOA framework later consolidated stream generators, online classifiers, ensembles, drift detectors, and evaluation protocols into a common research platform. [oamonitor.ireland.openaire](https://oamonitor.ireland.openaire.eu/rpo/rcsi/search/publication?pid=10.1145/347090.347107)

### 1.2 Important distinctions

Let a stream be

\[
S=\{(x_t,y_t)\}_{t=1}^{\infty},
\]

where \(x_t\) is observed at time \(t\), and \(y_t\) may arrive immediately or after a delay.

| Term | Meaning | Typical constraint |
|---|---|---|
| Batch learning | Train on a fixed dataset, usually with multiple passes and random access | Data are assumed to be available before training |
| Incremental learning | Update an existing model with new observations or mini-batches | May support repeated passes over stored data |
| Online learning | Update sequentially, usually one example at a time, while making predictions continuously | Often single-pass and bounded-memory |
| Stream mining | Online/incremental learning under potentially unbounded arrival, resource, and non-stationarity constraints | Emphasizes throughput, memory, latency, and adaptation |
| Continual learning | Learning sequential tasks or experiences while controlling forgetting | Usually broader than classical stream mining |
| Test-then-train | Predict on \(x_t\), receive \(y_t\), evaluate, then update | Standard prequential protocol |

“Online” and “incremental” are therefore not exact synonyms. A system that periodically retrains on a growing database is incremental but not necessarily online. Conversely, an online algorithm may use mini-batches or replay buffers and still be considered stream-oriented if it processes data sequentially.

### 1.3 Concept drift and related shifts

A concept is often represented by the joint distribution \(P_t(X,Y)\), or, in supervised prediction, by the conditional relationship \(P_t(Y\mid X)\). Concept drift occurs when the data-generating process changes over time:

\[
P_t(X,Y)\neq P_{t+\Delta}(X,Y).
\]

The practically important decomposition is:

\[
P(X,Y)=P(X)P(Y\mid X).
\]

This gives several cases:

- **Real concept drift:** \(P(Y\mid X)\) changes. The relationship being learned changes, so predictive performance can deteriorate even if the input distribution is stable.
- **Virtual drift / covariate shift:** \(P(X)\) changes while \(P(Y\mid X)\) remains approximately stable. The model may remain valid, although regions of the feature space may become more or less important.
- **Label shift:** \(P(Y)\) changes while \(P(X\mid Y)\) is approximately stable.
- **Prior-probability or class-ratio drift:** Often used informally for changes in \(P(Y)\).
- **Feature or sensor drift:** Changes in measurement properties, calibration, missingness, or feature semantics.
- **Novel-class drift:** New target classes appear that were absent during initial training.

The literature does not use these labels perfectly consistently. Some papers call any change in \(P(X,Y)\) “concept drift,” whereas others reserve the term for changes in \(P(Y\mid X)\). Gama et al.’s survey remains the standard conceptual reference, defining drift primarily around changes in the relation between inputs and targets. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11294200/)

### 1.4 Formal drift types

The widely used four-way taxonomy concerns the *temporal transition* between concepts rather than only the statistical distance between distributions.

| Drift type | Formal intuition | Operational consequence |
|---|---|---|
| Sudden / abrupt | A transition from \(C_a\) to \(C_b\) occurs over a very short interval | Old model becomes obsolete quickly |
| Gradual | Examples from \(C_a\) and \(C_b\) coexist during a transition, with the probability of \(C_b\) increasing | Detection is difficult because both concepts are temporarily present |
| Incremental | The target concept moves through a sequence of small intermediate concepts | Short-term change may be almost invisible |
| Recurring / reappearing | A previous concept returns after an interval; recurrence may be periodic or irregular | Reusing historical models can be beneficial |

Bose et al.’s *Characterizing Concept Drift* is the work most commonly associated with formalizing this four-category characterization, including transition duration and the distinction between recurring concepts and genuinely new concepts.  The earlier literature used related but less standardized terminology: abrupt and gradual change were already common, while incremental and recurring drift became more systematically distinguished in later taxonomies. [arxiv](https://arxiv.org/pdf/1511.03816.pdf)

Two additional dimensions are important:

- **Magnitude/severity:** How different \(C_a\) and \(C_b\) are.
- **Duration:** How long the transition lasts.
- **Frequency:** How often drift occurs.
- **Predictability:** Whether drift is known, scheduled, seasonal, or unexpected.
- **Location:** Whether all classes/features are affected or only a subregion.
- **Blip drift:** A short-lived excursion followed by a return to the previous concept.

A recurring concept is not necessarily gradual. A stream may switch abruptly between two previously seen regimes, or transition gradually between them.

***

## 2. Algorithms and adaptation

### 2.1 Hoeffding Trees and their successors

#### VFDT / Hoeffding Tree

**Domingos, P., and Hulten, G. (2000). “Mining High-Speed Data Streams.” Proceedings of the 6th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 71–80.**

VFDT incrementally grows a decision tree using sufficient statistics at leaves. It applies the Hoeffding bound to decide when the observed difference between the best and second-best split attributes is statistically large enough to justify a split. The algorithm provides a probabilistic guarantee that, after enough examples, its selected split is close to the split selected by an ideal batch learner. The paper reports constant-time-per-example behavior and very high throughput. [oamonitor.ireland.openaire](https://oamonitor.ireland.openaire.eu/rpo/rcsi/search/publication?pid=10.1145/347090.347107)

**Influence:** Very high; canonical, hundreds-to-thousands citation tier.

Important limitations are that standard VFDT is effectively stationary: once a split is made, it is not naturally revised when the distribution changes. It can also become biased by stale sufficient statistics, delayed labels, class imbalance, and high-cardinality attributes.

#### CVFDT

**Hulten, G., Spencer, L., and Domingos, P. (2001). “Mining Time-Changing Data Streams.” Proceedings of the 7th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 97–106.**

CVFDT extends VFDT to changing streams. It maintains statistics over a sliding window, grows alternative subtrees when an existing subtree becomes questionable, and replaces the old subtree when the alternative becomes more accurate. Its goal is to approximate repeatedly rebuilding VFDT on a moving window while retaining near-constant processing cost. [bibbase](https://bibbase.org/network/publication/hulten-spencer-domingos-miningtimechangingdatastreams-2001)

**Influence:** Very high; canonical, hundreds-to-thousands citation tier.

#### Important successors

- **Hoeffding Adaptive Tree (HAT):** Adds ADWIN-based monitoring at tree nodes and replaces unreliable subtrees with alternatives. It is more local and adaptive than a single global detector.
- **Hoeffding Option Tree:** Represents multiple possible paths and can improve predictive flexibility at additional memory and computation cost.
- **Hoeffding Adaptive Tree with Naive Bayes leaves:** Often strong on streams where local class boundaries are approximately linear.
- **Extremely Fast Decision Tree:** Reduces the number of examples needed before splitting and improves efficiency in some settings, but has less broad adoption than VFDT/HAT. [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/1802.08780)
- **Very Fast Decision Tree variants:** Add numeric-attribute handling, improved split criteria, memory management, and better treatment of imbalanced or evolving streams.
- **Adaptive model trees and neural stream learners:** More recent systems increasingly combine adaptive trees with linear models, neural networks, replay, or continual-learning mechanisms.

The major design pattern introduced by CVFDT—maintaining an incumbent model and testing candidate replacements—reappears in adaptive ensembles, drift-triggered retraining, and model repositories.

### 2.2 Online ensembles

#### Online Bagging and Boosting

**Oza, N., and Russell, S. (2001). “Online Bagging and Boosting.” Proceedings of the 2nd International Workshop on Artificial Intelligence and Statistics, pp. 105–112.**

Oza and Russell derive online approximations to bootstrap bagging and boosting. Each arriving example is assigned a Poisson-distributed multiplicity for each base learner, allowing a batch-style ensemble to be simulated in one pass. [proceedings.mlr](https://proceedings.mlr.press/r3/oza01a.html)

**Influence:** High; foundational ensemble paper.

#### Leveraging Bagging

**Bifet, A., Holmes, G., and Pfahringer, B. (2010). “Leveraging Bagging for Evolving Data Streams.” In Machine Learning and Knowledge Discovery in Databases, ECML PKDD 2010.**

Leveraging Bagging strengthens online bagging using larger Poisson multiplicities, output codes, and randomization mechanisms. The method was designed particularly for noisy and drifting streams.

**Influence:** High within stream mining; hundreds citation tier.

#### Adaptive Random Forest

**Gomes, H. M., Bifet, A., Read, J., Barddal, J. P., Enembreck, F., Pfahringer, B., Holmes, G., and Abdessalem, T. (2017). “Adaptive Random Forests for Evolving Data Stream Classification.” Machine Learning, 106, 1469–1495.**

Adaptive Random Forest (ARF) combines online random forests, random feature subspaces, Poisson resampling, background trees, and drift/warning detectors. A warning signal can initiate a background tree, while a confirmed drift can replace the incumbent tree. ARF is designed to diversify errors and adapt locally rather than rebuild the entire forest. [link.springer](https://link.springer.com/article/10.1007/s10994-017-5642-8)

**Influence:** Very high for modern stream classification; hundreds citation tier and a frequent baseline.

#### Streaming Random Patches

**Gomes, H. M., et al. (2019/2021). “Learning from Evolving Data Streams through Ensembles of Random Patches.” Knowledge and Information Systems.**

Streaming Random Patches (SRP) combines online bagging with random subspaces and random instance-feature patches. It often provides a strong accuracy–diversity–adaptation trade-off and is now a standard benchmark model in River, MOA-derived systems, and CapyMOA. [dl.acm](https://dl.acm.org/doi/10.1007/s10115-021-01579-z)

**Influence:** High and growing; tens-to-hundreds citation tier.

#### Current practical ranking

There is no universally accepted “state-of-the-art” classifier for all streams. The strongest repeatedly competitive classical baselines are typically:

1. **ARF**, especially when drift detection and background learners are configured properly.
2. **SRP**, often competitive with or better than ARF on heterogeneous streams.
3. **Leveraging Bagging** and **OzaBag/OzaBoost**, particularly when computational simplicity matters.
4. **HAT**, when a single adaptive tree is preferred for interpretability or memory constraints.
5. **Adaptive linear, neural, and continual-learning models**, when the feature space is high-dimensional, temporal, or representation learning is necessary.

The field has moved since the 2017–2020 wave in four directions:

- From proposing another detector or ensemble to measuring compute, memory, energy, and latency.
- From purely supervised, immediately labeled streams toward delayed-label, semi-supervised, and label-free monitoring.
- From small synthetic streams and Elec2 toward more realistic temporal, transactional, sensor, and industrial data.
- From single-model adaptation toward model repositories, replay, meta-learning, online continual learning, and adaptive representations.

However, most post-2020 methods do not yet have the same independent replication base as VFDT, ADWIN, ARF, or MOA. Claims of state-of-the-art performance are often benchmark-dependent.

### 2.3 Drift detectors

Most classical detectors monitor a sequence of binary prediction outcomes:

\[
e_t=\mathbf{1}[\hat y_t\neq y_t].
\]

This means they detect changes in *error behavior*, not necessarily changes in the input distribution. A detector can miss covariate drift that does not affect accuracy, and it can signal because of noise, class imbalance, or a temporary difficult region.

| Detector | Mechanism | Assumptions and strengths | Main weaknesses |
|---|---|---|---|
| DDM | Monitors online error rate and its estimated standard deviation; signals when the current error is sufficiently larger than the historical minimum | Very simple and cheap; designed for abrupt degradation | Sensitive to noise, class imbalance, early unstable estimates, and gradual drift |
| EDDM | Monitors the distance between consecutive errors rather than error rate | Better than DDM for gradual drift and changes occurring after a long period of high accuracy | Still error-stream dependent; can be unstable with sparse errors or noisy labels |
| ADWIN | Maintains an adaptive window and repeatedly tests whether older and newer subwindows have significantly different means | Nonparametric-ish adaptive memory; automatically shrinks after change and expands during stability | Computational overhead; sensitivity depends on confidence, compression, and error distribution; may detect performance fluctuations rather than semantic drift |
| Page-Hinkley | CUSUM-like cumulative test of deviations from a running mean | Cheap and useful for persistent mean shifts | Primarily suited to sustained changes; parameter-sensitive; not inherently robust to autocorrelation |
| HDDM-A | Uses Hoeffding bounds on differences between historical and recent error means | Statistical guarantee under bounded independent observations | Independence assumptions are often violated; can be slow or conservative |
| HDDM-W | Exponentially weighted moving averages plus Hoeffding-style tests | More responsive to recent observations and often useful for gradual drift | Fading parameter and threshold influence false alarms and delay |
| KSWIN | Compares recent and reference portions of an error window using a two-sample Kolmogorov–Smirnov test | Can detect distributional changes in the monitored sequence; often strong on abrupt changes | Sensitive to window size, dependence, continuous-value assumptions, and repeated testing; can be expensive |
| FHDDM | Uses a sliding window and Hoeffding bounds to compare the maximum observed correct-prediction rate with the recent rate | Fast and designed to reduce false alarms relative to simple error-rate schemes | Requires labeled predictions and can be less effective when drift is gradual or errors are sparse |

DDM was introduced by **Gama, Medas, Castillo, and Rodrigues (2004)**; EDDM by **Baena-García et al. (2006)**; ADWIN by **Bifet and Gavalda (2007)**; HDDM by **Frías-Blanco et al. (2015)**; KSWIN by **Raab, Heusinger, and Schleif (2020)**; and FHDDM by **Pesaranghader and Viktor (2016)**. The exact detector configuration matters at least as much as the detector name.

### 2.4 Label-free and unsupervised detection

When labels are delayed or unavailable, the system cannot directly monitor \(e_t\). Label-free methods instead compare the current input distribution with a reference distribution.

Common families include:

- **Univariate statistical tests:** Kolmogorov–Smirnov, Mann–Whitney, chi-square, population-stability index, and Jensen–Shannon divergence.
- **Multivariate two-sample tests:** Maximum Mean Discrepancy, energy distance, classifier-based two-sample tests, and kernel change-point detection.
- **Density and likelihood monitoring:** Compare current likelihoods, reconstruction errors, or density estimates with a reference period.
- **Representation-space monitoring:** Embed observations using an encoder and test drift in the latent space.
- **Prediction-distribution monitoring:** Track changes in \(P(\hat Y)\), confidence, entropy, calibration, or disagreement among an ensemble.
- **Model-based detectors:** Train a discriminator to distinguish reference from current examples.
- **Change-point methods:** Bayesian online change-point detection, Page-Hinkley-like tests on features, and sequential likelihood-ratio tests.

The crucial limitation is that input drift is not equivalent to harmful predictive drift. A statistically significant change in \(P(X)\) may be irrelevant to \(P(Y\mid X)\); conversely, real concept drift may occur with little detectable change in \(P(X)\). Recent surveys explicitly distinguish “label-free at monitoring time” from fully unsupervised learning because many systems still require a labeled reference set or a pretrained model. [arxiv](https://arxiv.org/html/2508.00042v2)

### 2.5 Comparative evidence

Comparative studies generally reach three conclusions:

1. **No detector dominates across drift types.** Detector rankings change with drift speed, severity, noise, class imbalance, base learner, and threshold settings.
2. **Synthetic benchmarks favor detectors whose assumptions match the generator.** A detector designed for error-rate shifts can look excellent when the generator directly changes class boundaries.
3. **Detection accuracy is not enough.** False alarms, detection delay, computational cost, recovery time, and downstream predictive accuracy must be reported.

A 2014 comparative study evaluated DDM, EDDM, Page-Hinkley, STEPD, DoF, ADWIN, Paired Learners, and ECDD under abrupt and gradual changes, with noise and irrelevant attributes.  A later benchmark of DDM, EDDM, HDDM-A, and HDDM-W found HDDM-W to offer a strong overall trade-off, especially for abrupt and gradual drift, but weaker performance on incremental drift.  A 2024 controlled energy-and-accuracy study reported KSWIN as accuracy-oriented but more energy-intensive, HDDM-W and ADWIN as balanced, and DDM, EDDM, Page-Hinkley, and HDDM-A as computationally cheap but often practically weak in that experimental setting. [2024.sci-hub](https://2024.sci-hub.se/7317/f2a2d64f7341fe9b99167d9408e9a733/10.1016@j.eswa.2014.07.019.pdf)

These findings are useful but not definitive. Detector comparisons are often single-study or small-family results, while the robust conclusion—“detector choice is data- and drift-dependent”—is well replicated.

***

## 3. Evaluation methodology and benchmarks

### 3.1 Prequential evaluation

The standard protocol is **test-then-train**, also called predictive sequential or prequential evaluation:

1. Receive \(x_t\).
2. Predict \(\hat y_t\) using the current model.
3. Receive \(y_t\).
4. Record the loss \(L(\hat y_t,y_t)\).
5. Update the model with \((x_t,y_t)\).

The cumulative prequential error is

\[
\hat L_n=\frac{1}{n}\sum_{t=1}^{n}L(\hat y_t,y_t).
\]

For drifting streams, recent-window or fading-factor error is usually more informative:

\[
\hat L_t^{(\alpha)}
=
\frac{\sum_{i=1}^{t}\alpha^{t-i}L_i}
{\sum_{i=1}^{t}\alpha^{t-i}},
\qquad 0<\alpha<1.
\]

Dawid introduced the broader prequential perspective in statistical theory.  For stream learning, Gomes et al. argue that prequential error with forgetting mechanisms is reliable: under stationary data and consistent algorithms, holdout, ordinary prequential, sliding-window prequential, and fading-factor estimates converge to the Bayes error. [cs.ubc](https://www.cs.ubc.ca/~murphyk/MLRG/dawid84Prequential.pdf)

### 3.2 Metrics

A serious survey or experiment should report more than mean accuracy:

- Prequential accuracy and error.
- Balanced accuracy, macro-F1, geometric mean, or Matthews correlation for imbalance.
- Kappa and **Kappa-plus**, especially when temporal persistence creates a strong no-change baseline.
- Prequential AUC or log loss when probabilistic predictions matter.
- Detection delay, false-positive rate, missed-drift rate, and warning-to-drift interval.
- Recovery time after drift.
- Memory, throughput, CPU time, energy, and model size.
- Statistical uncertainty across seeds and stream orderings.

### 3.3 Problems with standard practice

The main criticisms are methodological:

- **Temporal leakage:** Random train/test splits destroy the temporal structure and may expose future regimes during training.
- **Unreported ordering:** Some “real” datasets are shuffled, partially sorted, or distributed in ways that change the drift pattern.
- **One-pass claims with multiple passes:** Hyperparameter tuning or repeated evaluation can violate the intended stream constraint.
- **Immediate-label assumption:** Many applications have delayed, missing, selective, or noisy labels.
- **Single metric reporting:** Accuracy may reward a trivial persistence predictor.
- **Detector-adaptation confounding:** A detector can improve a classifier’s accuracy while producing poor drift timestamps, or vice versa.
- **Weak statistical testing:** Many papers compare many algorithms on a few streams without confidence intervals, paired tests, or correction for multiple comparisons.
- **Synthetic simplicity:** Standard generators often change only one controlled factor and do not reproduce real temporal dependence, missingness, or heterogeneous drift.
- **Configuration sensitivity:** Results depend heavily on window sizes, confidence levels, grace periods, ensemble size, and random seeds.

### 3.4 Standard datasets and generators

#### Synthetic generators

Common generators include:

- **SEA concepts:** Abrupt threshold changes with controllable noise.
- **Hyperplane:** Gradual or abrupt movement of a separating hyperplane.
- **Random RBF:** Moving class centroids and varying drift speed.
- **LED:** Irrelevant features and noise.
- **AGRAWAL:** Rule-based concepts and controlled changes.
- **STAGGER:** Small categorical concept space.
- **Sine and mixed generators:** Boundary movement and feature/class changes.
- **Random tree and rotating-hyperplane generators:** More complex evolving decision surfaces.

Generators are valuable because drift location, severity, speed, and recurrence can be known exactly. They are insufficient as the sole evidence because algorithms may be tuned to their simple mechanisms.

#### Real datasets

Frequently used real streams include:

- Electricity / Elec2.
- Forest Covertype.
- Poker-Hand.
- Airlines.
- NOAA weather.
- Electricity pricing and demand.
- Sensor, network intrusion, fraud, click, and industrial monitoring streams.
- Image and text streams in newer continual-learning work.

#### The Elec2 problem

The Electricity dataset is widely used but should not be treated as a neutral benchmark. The data are from the Australian New South Wales electricity market; the common version contains 27,552 complete records sampled from a larger set, with observations every 30 minutes and a binary label related to price movement.  Documentation and preprocessing histories are inconsistent across implementations. [search.r-project](https://search.r-project.org/CRAN/refmans/dynaTree/html/elec2.html)

More importantly, the sequence exhibits strong temporal dependence. A classifier that predicts the previous label or “no change” can be surprisingly competitive. Bifet and colleagues’ benchmarking critique argues that ordinary accuracy can overstate learning ability on Electricity and proposes temporally aware baselines and Kappa-plus-like measures.  A later discussion also emphasizes that poorly documented “real-world” datasets can harm reproducibility and make apparent algorithmic improvements difficult to interpret. [users.ics.aalto](https://users.ics.aalto.fi/jesse/papers/Pitfalls%20in%20Benchmarking%20Data%20Stream%20Classification%20and%20How%20to%20Avoid%20Them.pdf)

Therefore, a defensible study should:

- State the exact Elec2 file, preprocessing, ordering, missing-value treatment, and label definition.
- Include a previous-label or no-change baseline.
- Report balanced and temporal-aware metrics.
- Avoid presenting Elec2 as evidence of general superiority.
- Include multiple real streams with different dependence structures.

***

## 4. Tooling and research practice

| Framework | Language and status | Strengths | Limitations | Research role |
|---|---|---|---|---|
| MOA | Java; mature reference platform | Large collection of stream learners, generators, detectors, ensembles, and evaluators; historically dominant in academic stream mining | Java integration overhead; less convenient for modern Python ML and deep learning | The most established academic benchmark platform |
| scikit-multiflow | Python; largely legacy compared with River | Accessible scikit-learn-like API, stream generators, multi-output learning, detectors | Development and ecosystem momentum have shifted toward River and newer systems; slower in some benchmarks | Important historical Python baseline |
| River | Python; actively used and broadly accessible | Clean one-instance API, preprocessing, metrics, drift detectors, linear models, trees, ensembles, anomaly detection, and practical deployment support | Pure-Python implementations can be slower than Java/C++; not every MOA algorithm is available | Best general-purpose Python choice for prototyping and reproducible applied research |
| CapyMOA | Python interface over efficient MOA components, with integrations for PyTorch and scikit-learn | Combines Python usability with MOA performance; supports classification, regression, clustering, anomaly detection, drift detection, and online continual learning | Newer; smaller independent adoption and citation base; some evidence is currently preprint or project-reported | Promising research platform for efficient Python experiments |
| creme | Python; predecessor to River | Introduced practical online-learning APIs | Superseded by River | Historical rather than new-project choice |
| Vowpal Wabbit | C++ with Python bindings | Highly efficient online linear learning and contextual prediction | Narrower algorithmic scope; less focused on drift benchmarks | Strong for large sparse linear streams |
| scikit-learn partial-fit | Python | Familiar models and ecosystem | Not a full stream-mining framework; weak support for drift-specific evaluation and adaptation | Useful component inside custom pipelines |
| PyTorch/JAX continual-learning stacks | Python | Representation learning, deep models, replay, adapters, and custom research | Require substantial engineering for strict online constraints and fair evaluation | Increasingly important for high-dimensional streams |

MOA’s foundational paper is **Bifet, Holmes, Kirkby, and Pfahringer (2010), “MOA: Massive Online Analysis,” Journal of Machine Learning Research, 11, 1601–1604**. It remains the most influential stream-mining software citation. [semanticscholar](https://www.semanticscholar.org/paper/MOA:-Massive-Online-Analysis-Bifet-Holmes/daedbcd23ed448f815694f7032f62ce482cd669f)

**Montiel et al. (2018), “scikit-multiflow: A Multi-output Streaming Framework,” Journal of Machine Learning Research, 19, 1–5**, provided a Python framework for stream learning, generators, evaluators, and multi-output tasks. [jmlr](https://www.jmlr.org/papers/volume19/18-251/18-251.pdf)

**Montiel et al. (2021), “River: Machine Learning for Streaming Data in Python,” arXiv:2012.04740**, describes River as the merger of creme and scikit-multiflow, with a redesigned architecture for dynamic streams and continual learning. [arxiv](https://arxiv.org/abs/2012.04740)

**Gomes et al. (2025), “CapyMOA: Efficient Machine Learning for Data Streams in Python,” arXiv:2502.07432**, presents CapyMOA as a Python layer integrating MOA, scikit-learn, PyTorch, and stream-learning components. Its reported SRP benchmark was close to MOA and substantially faster than the Python alternatives tested, although these results are from the authors’ benchmark rather than a broad independent replication. [arxiv](https://arxiv.org/pdf/2502.07432.pdf)

### Recommended tooling strategy

For a one-year Master’s project:

- Use **River** for rapid prototyping, custom detectors, delayed-label experiments, and deployment-oriented pipelines.
- Use **MOA or CapyMOA** for comparison against canonical literature and efficient large-scale experiments.
- Reimplement or cross-check the key baselines in at least two frameworks when possible.
- Record exact versions, random seeds, parameters, stream order, label delays, and resource measurements.
- Benchmark ARF, SRP, Leveraging Bagging, HAT, a non-adaptive Hoeffding Tree, and at least one online linear model.
- Treat scikit-multiflow mainly as a compatibility or historical baseline rather than the primary new platform.

The community most consistently benchmarks on MOA-compatible algorithms and datasets. River is more common in Python-centric applied work, while CapyMOA is a promising bridge between the two ecosystems. Adoption of CapyMOA should not yet be described as equivalent to MOA’s long-established maturity.

***

## 5. Synthesis and references

### 5.1 What is well established

**Strongly replicated findings**

- Single-pass Hoeffding-style trees are effective, interpretable, and computationally attractive baselines.
- Ensembles generally outperform isolated trees on heterogeneous and drifting streams.
- ARF, SRP, and Leveraging Bagging are strong classical baselines.
- Prequential evaluation is more appropriate than random train/test splitting for temporally ordered streams.
- No drift detector is uniformly best.
- Detection performance depends on drift type, noise, class imbalance, window parameters, and the base learner.
- Realistic evaluation must include resource usage and delayed-label conditions.

**Less settled findings**

- Whether detector-triggered adaptation consistently beats passive adaptation.
- Whether explicit drift detection is necessary for strong ensembles.
- Whether deep continual-learning models outperform well-tuned adaptive trees under strict one-pass and bounded-memory constraints.
- Whether label-free input-distribution drift predicts degradation in task performance.
- Which detector is best under gradual, recurring, multivariate, or high-dimensional drift.
- Whether recent neural and meta-learning methods generalize beyond their authors’ datasets.

### 5.2 Main disagreements

1. **Active versus passive adaptation:** Active detectors can save computation and provide alerts, but false alarms and missed detections can damage the learner. Passive forgetting, sliding windows, fading factors, or continuously adaptive ensembles often perform competitively without explicit drift decisions.

2. **Error drift versus input drift:** Error-based detectors are directly connected to predictive degradation but require labels. Input-based detectors work earlier in deployment but may detect harmless covariate shift.

3. **Synthetic versus real benchmarks:** Synthetic streams provide ground-truth drift times and controlled comparisons; real streams provide realism but often have undocumented preprocessing, unknown drift locations, and confounded temporal dependence.

4. **Accuracy versus efficiency:** KSWIN-like and multivariate detectors may improve detection in some settings while consuming considerably more time or energy. The best detector depends on the application’s cost model, not solely predictive accuracy. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10805468/)

### 5.3 Proposed research gap

A defensible Master’s project could focus on:

> **Resource-aware adaptation under delayed or unavailable labels, comparing error-based, input-distribution, and hybrid drift detectors with ARF/SRP/HAT across controlled and real temporal streams.**

A strong experimental design would measure:

- Predictive performance.
- Detection delay and false alarms.
- Recovery time.
- Memory and throughput.
- Energy or CPU cost.
- Robustness to label delay, noise, class imbalance, recurrence, and temporal dependence.

This would address a genuine gap between classical detector benchmarks—which usually assume immediate labels—and deployment-oriented drift monitoring—which often lacks ground-truth labels.

### 5.4 Full reference list

1. Baena-García, M., del Campo-Ávila, J., Fidalgo, R., Bifet, A., Gavalda, R., and Morales-Bueno, R. (2006). “Early Drift Detection Method.” Proceedings of the 4th International Workshop on Knowledge Discovery from Data Streams.

2. Bifet, A., and Gavaldà, R. (2007). “Learning from Time-Changing Data with Adaptive Windowing.” Proceedings of the 2007 SIAM International Conference on Data Mining.

3. Bifet, A., Holmes, G., Kirkby, R., and Pfahringer, B. (2010). “MOA: Massive Online Analysis.” *Journal of Machine Learning Research*, 11, 1601–1604. [semanticscholar](https://www.semanticscholar.org/paper/MOA:-Massive-Online-Analysis-Bifet-Holmes/daedbcd23ed448f815694f7032f62ce482cd669f)

4. Bifet, A., Holmes, G., and Pfahringer, B. (2010). “Leveraging Bagging for Evolving Data Streams.” In *Machine Learning and Knowledge Discovery in Databases*, ECML PKDD.

5. Bose, R. P. J. C., van der Aalst, W. M. P., Žliobaitė, I., and Pechenizkiy, M. (2014/2017). “Dealing with Concept Drifts in Process Mining.” Related drift-characterization work formalizes abrupt, gradual, incremental, and recurring changes; terminology is consolidated in later work such as *Characterizing Concept Drift*. [arxiv](https://arxiv.org/pdf/1511.03816.pdf)

6. Dawid, A. P. (1984). “Present Position and Potential Developments: Some Personal Views: Statistical Theory. The Prequential Approach.” *Journal of the Royal Statistical Society: Series A*, 147(2), 278–292. [cs.ubc](https://www.cs.ubc.ca/~murphyk/MLRG/dawid84Prequential.pdf)

7. Frías-Blanco, I., del Campo-Ávila, J., Ramos-Jimenez, G., Morales-Bueno, R., Ortiz-Díaz, A., and Caballero-Murillo, J. (2015). “Online and Non-Parametric Drift Detection Methods Based on Hoeffding’s Bounds.” *IEEE Transactions on Knowledge and Data Engineering*, 27(3), 810–823.

8. Gama, J., Medas, P., Castillo, G., and Rodrigues, P. (2004). “Learning with Drift Detection.” In *Advances in Artificial Intelligence—SBIA 2004*, 286–295.

9. Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., and Bouchachia, A. (2014). “A Survey on Concept Drift Adaptation.” *ACM Computing Surveys*, 46(4), Article 44. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11294200/)

10. Gomes, H. M., Bifet, A., Read, J., Barddal, J. P., Enembreck, F., Pfahringer, B., Holmes, G., and Abdessalem, T. (2017). “Adaptive Random Forests for Evolving Data Stream Classification.” *Machine Learning*, 106, 1469–1495. [link.springer](https://link.springer.com/article/10.1007/s10994-017-5642-8)

11. Gomes, H. M., et al. (2021). “Learning from Evolving Data Streams through Ensembles of Random Patches.” *Knowledge and Information Systems*. [dl.acm](https://dl.acm.org/doi/10.1007/s10115-021-01579-z)

12. Gomes, H. M., et al. (2025). “CapyMOA: Efficient Machine Learning for Data Streams in Python.” arXiv:2502.07432. [arxiv](https://arxiv.org/pdf/2502.07432.pdf)

13. Grzenda, M., Gomes, H. M., and Bifet, A. (2020). “Delayed Labelling Evaluation for Data Streams.” *Data Mining and Knowledge Discovery*, 34, 1237–1266. [moa.cms.waikato.ac](https://moa.cms.waikato.ac.nz/publications/)

14. Hulten, G., Spencer, L., and Domingos, P. (2001). “Mining Time-Changing Data Streams.” Proceedings of the 7th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 97–106. [bibbase](https://bibbase.org/network/publication/hulten-spencer-domingos-miningtimechangingdatastreams-2001)

15. Kuncheva, L. I., and Plumpton, C. O. (2008). “Adaptive Learning Rate for Online Linear Discriminant Classifiers.” In *Structural, Syntactic, and Statistical Pattern Recognition*, LNCS 5342, 510–519. [search.r-project](https://search.r-project.org/CRAN/refmans/dynaTree/html/elec2.html)

16. Montiel, J., Read, J., Minku, L. L., Silva, D. F., and Bifet, A. (2018). “Scikit-Multiflow: A Multi-output Streaming Framework.” *Journal of Machine Learning Research*, 19, 1–5. [jmlr](https://www.jmlr.org/papers/volume19/18-251/18-251.pdf)

17. Montiel, J., Halford, M., Mastelini, S. M., Bolmier, G., Sourty, R., Vaysse, R., Zouitine, A., Gomes, H. M., Read, J., Abdessalem, T., and Bifet, A. (2021). “River: Machine Learning for Streaming Data in Python.” arXiv:2012.04740. [arxiv](https://arxiv.org/abs/2012.04740)

18. Oza, N. C., and Russell, S. (2001). “Online Bagging and Boosting.” Proceedings of the 2nd International Workshop on Artificial Intelligence and Statistics, 105–112. [proceedings.mlr](https://proceedings.mlr.press/r3/oza01a.html)

19. Pesaranghader, A., and Viktor, H. L. (2016). “Fast Hoeffding Drift Detection Method for Evolving Data Streams.” Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases Workshops.

20. Raab, C., Heusinger, M., and Schleif, F.-M. (2020). “Reactive Soft Prototype Computing for Concept Drift Streams.” Related work introduced and popularized KSWIN-style two-window Kolmogorov–Smirnov monitoring in stream-learning implementations. [arxiv](https://arxiv.org/html/2606.20216)

21. Rabanser, S., Günnemann, S., and Lipton, Z. C. (2019). “Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.” *Advances in Neural Information Processing Systems*, 32.

22. Žliobaitė, I. (2010). “Learning under Concept Drift: An Overview.” Technical report and widely cited conceptual survey.

23. van der Aalst, W. M. P., et al. (2013). “Drift Characterization and Detection in Process Mining.” Related process-mining work contributed formal language for gradual, incremental, and recurring changes.
