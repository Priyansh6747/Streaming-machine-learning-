# 03 — River: the library your teacher linked

Docs: <https://riverml.xyz/> · API overview: <https://riverml.xyz/latest/api/overview/>
Source: <https://github.com/online-ml/river> · Requires **Python 3.11+**

## What it is

River is, in one line, **"scikit-learn for data streams."** It's the standard Python
library for online machine learning. It was created in 2020 by merging two earlier
projects — **`creme`** and **`scikit-multiflow`** — which is worth knowing because a
lot of tutorials and papers still reference the older names.

Important framing for your report: River is your **instrument**, not your subject.
The project is about streaming learning; River is how you run the experiments. Cite
it (Montiel et al., JMLR 2021), don't make it the topic.

## The core API — four methods

Everything in River is built on a tiny interface:

```python
model.learn_one(x, y)        # update the model with ONE labelled sample
model.predict_one(x)         # predict a label for ONE sample
model.predict_proba_one(x)   # predict class probabilities (classifiers)
metric.update(y, y_pred)     # update a running score — TRUE label first
```

The key detail: **`x` is a Python dict**, not a numpy array.

```python
x = {"age": 31, "country": "IN", "amount": 240.5}
```

This is a deliberate design choice, not laziness. In a stream you handle one sample
at a time, so vectorisation buys you nothing — and dicts let features appear,
disappear, or be sparse without reshaping anything. It also means **categorical
features are just strings**, no fixed-width one-hot encoding needed up front.

## The canonical loop

```python
from river import compose, datasets, linear_model, metrics, preprocessing

model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
metric = metrics.Accuracy()

for x, y in datasets.Phishing():
    y_pred = model.predict_one(x)   # 1. TEST  (model hasn't seen this sample)
    metric.update(y, y_pred)        # 2. SCORE
    model.learn_one(x, y)           # 3. TRAIN (now it has)

print(metric)
```

Three things to notice:

- The `|` operator builds a **pipeline**, same idea as sklearn's `Pipeline`.
- `StandardScaler` **also learns incrementally** — it maintains a running mean and
  variance. It never sees a whole column. This is what "the whole pipeline is
  online" means in practice.
- Predict-then-learn is not stylistic. It is the evaluation protocol. See
  [04 — Evaluation](04-evaluation.md).

River also wraps this loop for you:

```python
from river import evaluate
evaluate.progressive_val_score(dataset, model, metric)
```

Use the manual loop when you need per-step logging (which you will, for
accuracy-over-time plots); use the wrapper for quick baselines.

## Drift detection API

```python
from river import drift

detector = drift.ADWIN()

for i, val in enumerate(stream):
    detector.update(val)              # feed it a number — often the error (0/1)
    if detector.drift_detected:       # boolean property, checked after each update
        print(f"drift at index {i}")
```

That's the whole interface: `update(value)` then check `.drift_detected`.
The same shape works for `drift.DDM()`, `drift.PageHinkley()`, `drift.KSWIN()`,
`drift.binary.DDM()`, etc.

## Module map

Grouped by what you'd actually reach for. Full list at the API overview link above.

**Getting data in**
- `datasets` — ready-made real streams: `Phishing`, `Elec2`, `CreditCard`, `Bananas`,
  `SMSSpam`, `TrumpApproval`, and more.
- `datasets.synth` — **synthetic generators**, the important ones for research:
  `ConceptDriftStream` (injects drift at a position *you* choose), `SEA`, `Agrawal`,
  `Hyperplane`, `RandomRBF`, `STAGGER`, `LED`.
- `stream` — helpers to iterate CSVs, SQL queries, arrays, and to shuffle/simulate
  delays.

**Reacting to change**
- `drift` — `ADWIN`, `DDM`, `EDDM`, `PageHinkley`, `KSWIN`, `HDDM_A`, `HDDM_W`.

**Models**
- `tree` — `HoeffdingTreeClassifier` (the classic VFDT), `HoeffdingAdaptiveTreeClassifier`
  (self-resetting subtrees under drift), regression variants.
- `forest` — `ARFClassifier` / `ARFRegressor` (Adaptive Random Forest), the strong baseline.
- `ensemble` — bagging/boosting variants: `ADWINBaggingClassifier`, `SRPClassifier`
  (Streaming Random Patches), `LeveragingBaggingClassifier`.
- `linear_model` — `LogisticRegression`, `LinearRegression`, `PAClassifier`.
- `naive_bayes` — `GaussianNB`, `MultinomialNB`. Cheap, surprisingly strong baseline.
- `neighbors` — `KNNClassifier` over a sliding window.
- `neural_net` — small MLPs. (Also see the separate `deep-river` project for PyTorch.)

**Data handling**
- `preprocessing` — running scalers, encoders.
- `feature_extraction` — aggregations, TF-IDF, target encoding.
- `feature_selection`, `imblearn` (class imbalance), `compose` (pipelines, unions).

**Other tasks**
- `anomaly` — `HalfSpaceTrees`, `OneClassSVM`.
- `cluster` — `KMeans`, `DBSTREAM`, `CluStream`.
- `time_series` — `SNARIMAX`, `HoltWinters`.
- `reco` — recommender models. `bandit` — multi-armed bandits. `active` — active learning.

**Measuring**
- `metrics` — incrementally-updated `Accuracy`, `F1`, `ROCAUC`, `MAE`, `Rolling(...)`
  windowed wrappers, `ClassificationReport`.
- `evaluate` — `progressive_val_score`.
- `stats` — running mean, variance, quantiles, entropy.
- `optim` — SGD variants, learning-rate schedules.
- `base` — the abstract interfaces everything inherits from. Read this if you want
  to write your own estimator, which is a legitimate way to add novelty.

## The one algorithm to understand deeply

**Hoeffding Tree / VFDT** (Domingos & Hulten, 2000) is the foundational streaming
algorithm and the one most likely to be asked about.

The problem: a normal decision tree picks a split by scanning the entire dataset.
You can't do that in a stream.

The insight: you don't need the whole dataset — you need enough samples to be
*statistically confident* that the currently-best attribute is genuinely best. The
**Hoeffding bound** tells you how many that is. It states that after $n$ independent
observations of a variable with range $R$, the true mean is within $\epsilon$ of the
observed mean with probability $1 - \delta$, where

$$\epsilon = \sqrt{\frac{R^2 \ln(1/\delta)}{2n}}$$

So: accumulate statistics at a leaf; when the gap between the best and second-best
split attribute exceeds $\epsilon$, you can split *now* and never look back. The
tree is built in one pass, in bounded memory, with a probabilistic guarantee that it
matches what a batch tree would have chosen.

**Hoeffding Adaptive Tree (HAT)** adds ADWIN to each internal node: when a subtree's
error stream shifts, it grows a replacement subtree in the background and swaps it
in. That's informed adaptation ([02](02-concept-drift.md)) built into the model.

## Practical gotchas

- `metric.update(y_true, y_pred)` — **ground truth first.** Easy to silently reverse.
- `learn_one` returns `None` in current versions (older tutorials chain `.learn_one(...)`
  expecting `self` — that pattern is dead).
- Dict keys are the feature names. Inconsistent keys between samples are *allowed*,
  which is powerful but means typos fail silently rather than crashing.
- Everything is pure Python + some Cython — it is **slow per-sample compared to a
  vectorised batch fit**. That's expected and is itself worth measuring honestly.
- Set random seeds on synthetic generators (`seed=`) or your experiments aren't
  reproducible.

---

Next: [04 — Evaluation](04-evaluation.md)
