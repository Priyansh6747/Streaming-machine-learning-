# 06 — Literature and landscape

Reading list and context. Citations are from memory — **verify every one on Google
Scholar before it goes in your bibliography.**

## Read these first (surveys)

Start here. Surveys give you the vocabulary, the taxonomy, and a map of who did what.

- **Gama, Žliobaitė, Bifet, Pechenizkiy & Bouchachia (2014).**
  *A Survey on Concept Drift Adaptation.* ACM Computing Surveys.
  → **The** reference. The drift taxonomy in [02](02-concept-drift.md) is essentially
  from here. If you read one paper, read this one.

- **Lu, Liu, Dong, Gu, Gama & Zhang (2019).**
  *Learning under Concept Drift: A Review.* IEEE TKDE.
  → More recent, more detector-focused. Good complement to Gama 2014.

- **Bifet, Gavaldà, Holmes & Pfahringer (2018).**
  *Machine Learning for Data Streams, with Practical Examples in MOA.* MIT Press.
  → Textbook by the people who built the field. Available free online.

## Foundational algorithms

- **Domingos & Hulten (2000).** *Mining High-Speed Data Streams.* KDD.
  → **VFDT / Hoeffding Tree.** The origin of streaming decision trees. Understand
  the Hoeffding bound argument (see [03](03-river-library.md)).

- **Hulten, Spencer & Domingos (2001).** *Mining Time-Changing Data Streams.* KDD.
  → CVFDT — VFDT extended to handle drift.

- **Bifet & Gavaldà (2007).** *Learning from Time-Changing Data with Adaptive
  Windowing.* SDM.
  → **ADWIN.** The most-used drift detector; has actual theoretical guarantees.
  Essential if you pick Option A or C.

- **Bifet & Gavaldà (2009).** *Adaptive Learning from Evolving Data Streams.* IDA.
  → **Hoeffding Adaptive Tree (HAT).**

- **Gama, Medas, Castillo & Rodrigues (2004).** *Learning with Drift Detection.* SBIA.
  → **DDM**, the first widely-used detector. Simple enough to reimplement yourself.

- **Baena-García et al. (2006).** *Early Drift Detection Method.* → **EDDM.**

- **Page (1954).** *Continuous Inspection Schemes.* Biometrika.
  → The origin of Page-Hinkley / CUSUM. Nice to cite — shows the field's roots in
  1950s sequential analysis, not 2010s ML.

- **Raab, Heusinger & Schleif (2020).** *Reactive Soft Prototype Computing for
  Concept Drift Streams.* Neurocomputing. → **KSWIN.**

## Ensembles (the strong baselines)

- **Gomes et al. (2017).** *Adaptive Random Forests for Evolving Data Stream
  Classification.* Machine Learning. → **ARF.** Your main baseline for Option B/C.
- **Gomes, Read & Bifet (2019).** *Streaming Random Patches for Evolving Data Stream
  Classification.* ICDM. → **SRP.**
- **Bifet, Holmes & Pfahringer (2010).** *Leveraging Bagging for Evolving Data
  Streams.* ECML-PKDD.
- **Oza & Russell (2001).** *Online Bagging and Boosting.* → the Poisson(1) trick
  that makes bagging work in a stream.

## Tools and evaluation

- **Montiel, Halford, Mastelini, et al. (2021).** *River: Machine Learning for
  Streaming Data in Python.* JMLR 22(110).
  → **Cite this for River itself.** Non-negotiable.
- **Bifet, Holmes, Kirkby & Pfahringer (2010).** *MOA: Massive Online Analysis.* JMLR.
  → The Java framework. Still the reference implementation for most algorithms.
- **Montiel, Read, Bifet & Abdessalem (2018).** *Scikit-Multiflow.* JMLR.
  → River's predecessor. Cite when explaining River's lineage.
- **Gama, Sebastião & Rodrigues (2013).** *On Evaluating Stream Learning Algorithms.*
  Machine Learning.
  → **Read this for your methodology chapter.** Justifies prequential evaluation and
  fading factors. Cite it when you explain why you didn't use a train/test split.
- **Demšar (2006).** *Statistical Comparisons of Classifiers over Multiple Data Sets.*
  JMLR. → Friedman + Nemenyi, the convention for comparing many methods. Cite it when
  you do your statistical testing.

## Related but distinct — one paragraph each, no more

Mentioning these shows breadth; going deep on them derails the project.

- **Continual / lifelong learning** (deep learning). Same underlying worry —
  catastrophic forgetting — different community, different methods (EWC, replay
  buffers). Streaming ML cares about bounded memory and single-pass; continual
  learning usually doesn't.
- **Transfer learning / domain adaptation.** Handles distribution shift between two
  *fixed* domains, not a continuous stream.
- **Online learning theory.** Regret bounds, adversarial settings, online convex
  optimisation. Theoretically adjacent, practically separate.
- **Time-series forecasting.** Also sequential, but the task is extrapolating a
  signal, not classifying incoming samples under a shifting concept.

## The tool landscape

| Tool | Language | Notes |
|---|---|---|
| **River** | Python | What you're using. Merger of `creme` + `scikit-multiflow`. |
| **MOA** | Java | The original, from Waikato (the WEKA group). Most complete algorithm set; most papers benchmark here. |
| **scikit-multiflow** | Python | Deprecated — merged into River. Many tutorials still use it; don't. |
| **deep-river** | Python | River + PyTorch, for online neural nets. |
| **Apache Flink / Spark Streaming** | JVM | Production stream *processing*, not ML research frameworks. Mention as deployment context. |

## Standard benchmark datasets

Knowing these by name is useful — they appear in nearly every paper.

**Real, with genuine drift**
- **Elec2 / Electricity** — Australian electricity prices, 45k samples. The most-used
  real drift benchmark. *Caveat:* known to have autocorrelation issues that inflate
  naive-classifier results — worth mentioning, it shows you read critically.
- **Airlines**, **Covertype**, **Poker Hand**, **KDDCup99**, **CreditCard** (severe imbalance).
- River ships several: `datasets.Phishing`, `Elec2`, `CreditCard`, `SMSSpam`, `Bananas`.

**Synthetic (ground-truth drift — your main tool)**
- **SEA** — sudden drift via a shifting threshold. The classic.
- **Agrawal** — 10 predefined functions; switching function = drift.
- **Hyperplane** — a rotating hyperplane; the standard *incremental* drift generator.
- **RandomRBF** — drifting cluster centroids.
- **STAGGER** — small, abrupt, very interpretable.
- **LED** — 7-segment display digits with noise.

All available under `river.datasets.synth`, and composable with
`synth.ConceptDriftStream` to place drift exactly where you want it.

---

Next: [07 — Open questions](07-open-questions.md)
