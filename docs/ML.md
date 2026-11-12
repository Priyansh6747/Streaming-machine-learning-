# ML.md — the machine-learning reference

**What this is:** everything ML about this project in one self-contained file — the
problem framing, every model used and planned, *why* each was chosen and what the
alternatives were, the pseudocode behind every number, the metric definitions, and the
honest defects. Written so it can answer a traditional ML question without any other
context.

**Status of claims:** ✅ = measured in this project · 📋 = planned / spec'd · 📚 = from
the literature.

---

## 1. The ML problem, framed

### 1.1 The task

**Supervised binary classification on a non-stationary data stream.**

Not regression, not clustering, not recommendation — **classification only**, locked
deliberately as a scope guard. Multi-class is allowed later; everything else is out.

### 1.2 The regime: online / streaming learning

Standard (batch) ML assumes the dataset is a *finished object*: finite, available all
at once, drawn i.i.d. from a fixed distribution. Streaming ML drops all three.
An algorithm is "streaming" only if it respects three hard constraints:

| Constraint | Meaning |
|---|---|
| **Single pass** | Each sample is seen exactly once. You cannot revisit sample #5. |
| **Bounded memory** | Memory must not grow with the number of samples seen. |
| **Bounded time per sample** | Processing must keep up with arrival rate, or you fall behind forever. |

A useful test: *if this ran for a year without restarting, would it still work?* If the
answer involves "well, memory would…", it is not a streaming algorithm.

### 1.3 The formal setting

The model estimates `P(X, y)`, usually decomposed as `P(y|X) · P(X)`.

**Concept drift** is when that distribution changes over time:

```
P_t(X, y) ≠ P_{t+Δ}(X, y)
```

Two flavours that matter, and they drive everything downstream:

| Flavour | What changes | Effect |
|---|---|---|
| **Real drift** | `P(y|X)` — the input→output *rule* | Genuinely breaks the model. Accuracy suffers. |
| **Virtual drift / covariate shift** | `P(X)` only — the inputs move, the rule is intact | May not cost any accuracy, but **input-watching detectors still fire** on it. |

Drift *shapes* (the standard taxonomy):

| Shape | Behaviour | Detectability |
|---|---|---|
| **Sudden / abrupt** | Concept A flips to B instantly | Easiest to detect, most damaging |
| **Gradual** | A and B interleave with shifting probability over a window | Hard — error rises slowly and noisily |
| **Incremental** | A deforms continuously through intermediate states | Hard |
| **Recurring** | Old concepts return (seasons, day/night) | Forget-and-restore territory |

There is also the **blip** — a one-off anomaly that is *not* drift. A good detector must
not fire on it. This sensitivity/false-alarm tension is the **stability–plasticity
trade-off** and it is measurable.

### 1.4 Detection vs adaptation — two separate things

A very common confusion, worth having straight:

- **Detection** = noticing the change. Watches a signal and raises an alarm. This is a
  *statistical hypothesis test* over a stream.
- **Adaptation** = doing something about it. Reset, retrain, reweight, or just keep
  learning with a forgetting factor.

They are independently designable: you can adapt **blindly** (no detector at all —
sliding windows, fading factors, a learning rate that never reaches zero) or
**informed** (explicit detector triggers a reaction). The project's Exp-01/02 ship an
adaptation with **no detector at all** — see §4.7, this is the single most attackable
design choice in the work.

### 1.5 How models adapt — the three strategies

| Strategy | Mechanism | Strength | Weakness |
|---|---|---|---|
| **Blind / passive** | Keep learning; recent data outweighs old | Always on, simple, no tuning | Slow on sudden drift; permanently discards still-valid old knowledge |
| **Informed** | Detector fires → reset / retrain / swap | Sharp, fast on the drift it detects | A false alarm throws away a good model |
| **Ensemble** | Pool of models, weighted by recent performance; drop worst, add fresh | Drift handling is emergent; strongest general baseline | More compute, more memory |

This project's streaming model uses **blind adaptation with a bounded memory window** —
which is why it recovers without any detector.

---

## 2. Models USED (measured)

Two models, one race, same data, same order. ✅

### 2.1 Streaming model — DeepRiver `RollingClassifier`

| | |
|---|---|
| **Library** | `deep-river` 0.3.4 (River + PyTorch) |
| **Architecture** | RNN `3 → 16 → Linear(16→2)`, logits |
| **Parameters** | ~2.3k |
| **Update rule** | one optimiser step per sample: `predict_one` → `metric.update` → `learn_one` |
| **Forgetting mechanism** | **rolling window of 500 samples** — old samples fall out |
| **Optimiser / loss / lr** | Adam / cross-entropy / 1e-3 |
| **Warm-up** | learns from sample 1 onward |
| **Measured cost** | 99.1 s for the 48k-loop on an RTX 3050 = **484 samples/s**; 1,089 s / **44 samples/s** on CPU (~11× slower); peak process RSS 1,074 MB |

**Why this model:** it is a *deep* streaming learner — it demonstrates that the
thesis mechanism (bounded recovery via continuous refit) holds for a neural model, not
just for the classical trees the literature usually tests. The rolling window *is* the
drift-adaptation mechanism: it needs no detector, and its memory is bounded by design.

**Why the window is the mechanism:** a window of 500 means the model is always trained
on roughly the last 500 samples. After a drift, the old-concept samples age out within
~500 samples, and the net is refit on the new concept automatically. That is why the
recovery is bounded and why no detector is required.

**Alternatives considered:**

| Alternative | Why not chosen |
|---|---|
| DeepRiver per-sample `Classifier` (no rolling window) | This is the interesting *comparison*, not the baseline — per-sample SGD with no explicit window is a different forgetting story. Deferred to a follow-up experiment. |
| River `HoeffdingTree` | The canonical *static* streaming learner — a deliberate control, belongs in Exp-03's matrix |
| River `HAT` (Hoeffding Adaptive Tree) | The canonical *adaptive* streaming learner; stronger literature pedigree. Should be added as a classical arm in Exp-03 so the claim isn't deep-net-specific |
| River `ARF` (Adaptive Random Forest) | Strongest general streaming baseline, but ~2.2× the runtime of a retrained XGBoost 📚 and heavier to run; reserved as an accuracy ceiling |
| River `SRP`, `LeveragingBagging`, online `LogisticRegression` | In the planned Ch1 matrix; not needed for the pilot |

**Known deviation from spec:** the specified "MLP 2 → 16 → 2" is **not expressible** in
DeepRiver's `RollingClassifier` — its feature-incremental layer handling requires the
module's first layer to take `n_features` units, and it feeds the net a temporal window.
The RNN head is the architecture DeepRiver documents for rolling models. Capacity (16)
and learning rate are unchanged. ✅ documented in the Exp-01 write-up.

### 2.2 Frozen batch model — scikit-learn `RandomForestClassifier`

| | |
|---|---|
| **Library** | `scikit-learn` 1.9.0 |
| **Config** | `n_estimators=100`, `random_state=seed` |
| **Training** | `fit` **once** on the first 12,000 samples (20% warm-up) |
| **Thereafter** | `predict_one` only — **never updated** |
| **Measured cost** | 0.17 s to fit; near-zero per-sample prediction cost |

**Why this model:** it is the *frozen* baseline — the canonical "train once, deploy,
never touch" deployment. It is the weakest possible opponent, and that is the point:
Exp-01 was designed as the **weakest form of the claim**. If the thesis mechanism can't
show up against this, the bigger experiments are pointless.

**Why RandomForest specifically** (rather than another frozen model):

| Candidate | Why not the primary |
|---|---|
| **XGBoost** | Reserved as the **oracle-tuned upper bound** (the Amekoe-style honest opponent). Spending it as the frozen strawman wastes the strongest baseline. |
| **Logistic Regression** | A linear model can't express the SEA rule (`x₀+x₁ > threshold` needs only a linear boundary — but with 10% label noise and a rolling comparison it underfits relative to a tree ensemble); weaker baseline makes the win less credible |
| **SVM** | Poor fit for a streaming/one-shot comparison: no incremental story, scaling sensitivity, and cost grows with training set |
| **Decision Tree (single)** | Too high-variance — a single tree's post-drift collapse would be confounded with its own instability rather than the drift |
| **Random Forest** | ✔ Robust to feature scaling, needs no tuning, handles the non-linear threshold rule, deterministic given a seed, cheap to fit, and the field's default tabular baseline |

### 2.3 Summary of the measured race

| | Streaming (DeepRiver rolling) | Frozen (RF) |
|---|---|---|
| Pre-drift baseline κ₀ | 0.751 ± 0.017 | **0.767 ± 0.018** |
| Dip depth Δκ | **0.045 ± 0.015** | 0.173 ± 0.018 (~3.8× deeper) |
| Min rolling κ after τ | 0.706 ± 0.015 | 0.594 ± 0.013 |
| Recovery time | median **0**, mean 319 ± 765 | **never, 11/11** |
| Damage area | **13 ± 31** (max 103) | **3,710 ± 592** (min 2,439) |
| Full-run accuracy | **0.8830 ± 0.0021** | 0.8539 ± 0.0017 |
| Full-run Cohen Kappa | **0.7530 ± 0.0044** | 0.6850 ± 0.0035 |

(Frozen wins the stationary phase slightly; streaming wins everything after the drift.)

---

## 3. Models PLANNED

📋 From the locked Ch1 protocol. None of these is measured yet.

### 3.1 Streaming learners (River)

| Model | What it is | Role |
|---|---|---|
| **Hoeffding Tree** | One-pass decision tree; constant memory per leaf via the Hoeffding bound | Canonical **static** streaming learner — the control |
| **HAT** (Hoeffding Adaptive Tree) | Hoeffding tree with ADWIN at each node | Canonical **adaptive** learner |
| **ARF** (Adaptive Random Forest) | Ensemble with adaptive operators + resampling | Strongest general baseline / accuracy ceiling |
| **SRP** (Streaming Random Patches) | Ensemble with random subspaces + patches | Diversity in the ensemble arm |
| **Leveraging Bagging** | Online bagging with λ-boosting | Classic ensemble baseline |
| **kNN + ADWIN** | Lazy learner gated by a detector | Non-parametric counterpoint |
| **Logistic Regression** | Online linear model | Deliberately **blind/passive** — the no-detector control |

### 3.2 Batch baselines (the ladder)

| Baseline | Definition | What it tests |
|---|---|---|
| **(a) Frozen** | Fit on warm-up, predict-only forever | No adaptation at all — done ✅ |
| **(b) Scheduled-retrain** | Refit every *k* samples on a sliding window, **no** detector | Clock-driven adaptation — the honest industrial baseline |
| **(c) Oracle-tuned** | XGBoost / EBM tuned with Optuna, frequently retrained | The **honest upper bound**; reported so no one can accuse us of an easy opponent |

### 3.3 Drift detectors

| Detector | Mechanism | Notes |
|---|---|---|
| **ADWIN** (2007) | Adaptive window; when two sub-windows differ statistically, drop the older | Rigorous guarantees, no window-size parameter; de-facto standard |
| **DDM** (2004) | Statistical process control on the error rate; warn at 2σ, alarm at 3σ | Simple, cheap; weak on gradual drift; assumes independent Bernoulli errors |
| **EDDM** (2006) | Tracks the *distance between* errors | Fixes DDM's gradual-drift weakness |
| **Page-Hinkley** (1954) | CUSUM-family: accumulates deviation from the running mean | Classical sequential analysis; threshold-sensitive |
| **KSWIN** (2020) | Kolmogorov–Smirnov test between a sliding window and a reference | Non-parametric, distribution-based |
| **HDDM** | Hoeffding-bound drift detection | Also available in River; optional |

**The guaranteed finding:** no detector wins on delay, false alarms and misses
simultaneously. Sensitivity trades against false alarms. Reporting that trade-off with
real numbers *is* a legitimate result — and it becomes the baseline the hybrid is
judged against. 📚

---

## 4. "Why A and not B" — the decision table

This is the section that answers the traditional viva question. Every entry is a
design choice, its justification, and the alternative.

### 4.1 Why streaming/online at all, instead of batch?

Because on a drifting, unbounded, label-scarce stream, a batch model's update schedule
does not depend on what the data is doing. **Batch models update on a clock; streaming
models update on evidence.**

The quantitative content of the whole project is one ratio:

```
Streaming damage        ≈ (d + a) × Δe          # d = detection delay, a = adaptation time
Scheduled-batch damage  ≈ (r_next − τ) × Δe     # r_next = next retrain tick
Frozen-batch damage     = unbounded (never recovers on its own)
```

Streaming wins whenever `d + a < r_next − τ` — the generic case, since `d + a` is tens
to low hundreds of samples while `r_next − τ` spans a whole retrain interval.

**The honest counter:** a *tuned, frequently retrained* batch model can beat streaming
under delayed labels and rare classes 📚 — retrained XGBoost AUC-ROC **92.81** vs ARF
**91.02** (no delay); delayed fraud AUCPR **23.41** vs **14.23**. That is exactly why
the Delay Tax is the project's open problem rather than a solved one.

### 4.2 Why prequential evaluation instead of a train/test split?

**Random shuffling a temporal stream is invalid** — it leaks future information into
training, and on a drifting stream it estimates a quantity that never existed. The
accepted protocol is prequential (test-then-train): score each sample **before** it
trains the model. Every sample is used for testing then training, in that order;
nothing wasted, nothing leaked. 📚 Gama, Sebastião & Rodrigues (2013).

### 4.3 Why Cohen's Kappa instead of accuracy?

- Kappa is **chance-corrected** — it subtracts the agreement you'd get by guessing.
- It is **robust to class imbalance** (critical for the fraud/CreditCard case).
- **Accuracy alone overstates learning** on temporal streams, because temporal
  persistence creates a strong "predict the previous label" baseline that a naive model
  can exploit. This is the known Kappa+/Elec2 trap.

Accuracy is still *reported* (§6), just never as the headline.

### 4.4 Why a rolling (windowed) metric as well as a cumulative one?

**Cumulative metrics converge and hide drift.** Over 48,000 samples, a permanent
collapse spreading over thousands of samples is averaged down to a couple of
percentage points. The rolling metric (w = 1,000) makes the *shape* visible — dips and
recoveries. In this field, **the accuracy-over-time curve is itself the result**;
almost every good figure plots something against "number of samples seen".

### 4.5 Why SEA as the stream?

- Synthetic streams are the only way to get **ground-truth drift positions**. Without a
  known τ, detection delay and recovery time are *uncomputable*.
- SEA is the standard abrupt-drift benchmark — 3 features, only 2 relevant, a linear
  threshold rule with constructive noise.
- **But it is deliberately simple**: a linear boundary in 2 features. That is a
  weakness (see §11), not a strength.

*Alternatives:* Agrawal (different rule families), Hyperplane (incremental drift),
RandomRBF (gradual), STAGGER, LED, and `ConceptDriftStream` compositions for recurring
drift. All planned for the Ch1 matrix; SEA serves the pilot.

### 4.6 Why 60,000 samples, τ at 50%, 20% warm-up, noise 0.1?

| Choice | Value | Why |
|---|---|---|
| Stream length | 60,000 | Long enough that the drift window (5,000) is a small slice; short enough to run 11 seeds on a laptop GPU |
| Drift position τ | 30,000 (50%) | Symmetric: equal pre- and post-drift room, so κ₀ is well estimated and there's ample post-drift region |
| Warm-up | 12,000 (20%) | Enough to fit the RF properly; a standard 20–30% warm-up in the protocol |
| Label noise | 0.1 (10%) | Makes the problem non-trivial — without noise, the frozen RF could memorise the SEA rule perfectly and the drift would be the only error source. Balanced classes by construction. |

### 4.7 Why is there **no drift detector** in Exp-01/02? *(the attackable bit)*

**Straight answer:** the recovery came entirely from the rolling window — blind
adaptation, no detector. It was a deliberate minimal design (Exp-01 documents itself as
"no detector, no delay, no oracle tuning, no statistics").

**Defence available:** it is consistent with the literature that the *learner* matters
more than the *trigger* — with per-sample incremental updates, the retraining policy
becomes irrelevant 📚 (54 paired comparisons, no practical difference vs no-retrain).

**Honest weakness:** the project is fundamentally about **detectors**, and neither
concluded experiment runs one. The detector matrix (RQ1) is genuinely unrun. Do not
oversell this as "intentional".

### 4.8 Why 11 seeds and not more? Why not the full statistical protocol?

The pre-registered bar was **≥10 seeds**, and 11 were run (42 plus 0–9). That bar is
met. But the accepted protocol for comparing many methods over many datasets is
**Friedman + Nemenyi** with ≥10 seeds across **multiple datasets and multiple learners**
📚 Demšar (2006). Eleven seeds on **one** stream is a within-stream distribution — call
it a distribution, **not** statistical significance.

### 4.9 The metric constants — why 0.95, 500, 5,000?

| Constant | Value | Justification |
|---|---|---|
| Recovery band | `κ ≥ 0.95 · κ₀` | "Effectively back to baseline" — 5% tolerance |
| Sustained stretch | 500 consecutive samples | Prevents a single lucky sample counting as recovery |
| Lookahead budget | 5,000 samples from τ | The **pre-registered falsification budget** — set before the run, so recovery couldn't be judged against a moving target |

All three were fixed **before** the run. That is the defence. The gap is that **no
sensitivity analysis** over these constants has been done — a known next step.

---

## 5. Pseudocode

### 5.1 The prequential loop (the core protocol)

```python
for x, y in stream:
    y_pred = model.predict_one(x)      # model has NEVER seen this sample
    metric.update(y, y_pred)           # so this score is honest
    model.learn_one(x, y)              # now let it learn  (frozen model: skipped)
```

The ordering is the whole point. Note the river API convention: `metric.update` takes
the **true label first**.

### 5.2 Stream generation (SEA, one abrupt drift)

```python
# One continuous feature draw; only the LABEL RULE switches at tau.
# (river 0.25's synth.SEA reseeds its RNG per generator, so concatenating two
#  variants REPLAYS identical feature vectors -> the frozen model would have
#  already seen its "test" chunk during warm-up. Hence: draw once, switch the rule.)
X = uniform(0, 10, size=(60000, 3), seed=seed)
y = where(X[:, 0] + X[:, 1] > 8, 1, 0)          # concept 0  for t < tau
y[tau:] = where(X[tau:, 0] + X[tau:, 1] > 9, 1, 0)  # concept 1  for t >= tau
y = flip_labels(y, p=0.10, seed=seed)             # 10% label noise
```

### 5.3 Drift-window metrics

```python
kappa0 = mean(rolling_kappa[tau-1000 : tau])                 # pre-drift baseline
dip    = min(rolling_kappa[tau : tau+5000])                   # worst after drift
dk     = kappa0 - dip                                         # dip depth

# recovery: first sample from which kappa stays >= 0.95*kappa0 for 500 consecutive
recovery = None
run = 0
for i in range(tau, tau + 5000):
    if rolling_kappa[i] >= 0.95 * kappa0:
        run += 1
        if run >= 500:
            recovery = i - 500 + 1
            break
    else:
        run = 0

# damage area: skill lost x time spent below baseline
if recovery is not None:
    window = range(tau, recovery + 1)
else:
    window = range(tau, len(stream))                          # never recovered
damage = sum(max(kappa0 - rolling_kappa[t], 0) for t in window)
```

> ⚠️ **This exact code has a defect** — see §11.1. The `recovery` truncation
> collapses `damage` to ~zero whenever recovery fires at τ.

### 5.4 Blind adaptation (what the streaming model actually does)

```python
class RollingAdaptation:
    def __init__(self, window=500):
        self.window = deque(maxlen=window)     # bounded memory: old samples fall out

    def learn_one(self, x, y):
        self.window.append((x, y))
        for _ in range(1):                     # one optimiser step per sample
            loss = cross_entropy(self.net(sequence(self.window)), y)
            loss.backward(); self.optimizer.step(); self.optimizer.zero_grad()
```

The `deque(maxlen=window)` **is** the drift mechanism: after a change, old-concept
samples age out within ~`window` samples and the net refits on the new concept. No
detector needed — that is why recovery is bounded.

### 5.5 Detector integration (planned)

```python
detector = ADWIN()
for x, y in stream:
    y_pred = model.predict_one(x)
    error = int(y_pred != y)
    detector.update(error)
    if detector.drift_detected:
        model.reset()                          # or retrain on a recent buffer
    model.learn_one(x, y)
```

### 5.6 The hybrid detector (planned — the Ch3 contribution)

```python
# Stage 1: label-free early warning on P(X). Fires FAST but also on virtual drift.
# Stage 2: supervised confirmation on the DELAYED error stream. Reliable but LATE.
for t, x in enumerate(stream):
    if stage1_warns(t, x):                     # KS / MMD-style two-sample test
        pending.append(t)                      # unconfirmed warning
    for y_true, t0 in labels_arriving_at(t):   # label for sample t0 lands now
        err = int(pred[t0] != y_true)
        if t0 in pending and stage2_confirms(err):
            adapt()                            # confirmed -> act
        elif t0 in pending:
            revoke(t0)                         # unconfirmed -> revoke, cost it
```

The design is justified by the lemmas: **L4 says stage 1 can never confirm** (pure
posterior drift is invisible to `P(X)`), and **L1 says stage 2 can never be early** (it
cannot fire before the labels land). So the hybrid is the only architecture that is
*both* early and reliable given the information available.

---

## 6. Metric reference

### 6.1 Predictive quality (per-sample, inside the prequential loop)

| Metric | What it proves | River class |
|---|---|---|
| Prequential accuracy | overall correctness on the live stream | `metrics.Accuracy` |
| **Cohen Kappa** | chance-corrected agreement — the headline | `metrics.CohenKappa` |
| **Rolling Kappa** (w=1,000) | recency-biased view; reveals recovery | `metrics.RollingCohenKappa` |
| AUC-ROC / AUC-PR | ranking quality; PR for rare classes | `metrics.ROCAUC`, `metrics.PrecisionRecallAUC` |
| F1 / recall / precision | per-class utility (e.g. fraud-class recall) | `metrics.F1`, `metrics.Recall` |

### 6.2 Drift-response metrics (what batch models structurally cannot report)

| Metric | Definition | Good |
|---|---|---|
| **Detection delay** | samples from true τ to first alarm | ↓ |
| **False positives** | alarms with no drift nearby | ↓ |
| **Missed detections (MDR)** | drift points with no alarm inside the tolerance window | ↓ |
| **MTFA / MTD** | mean time between false alarms / mean time to detect | MTFA ↑, MTD ↓ |
| **Recovery time** | samples from τ until rolling Kappa returns to the pre-drift band | ↓ |
| **Damage area** | `Σ (κ₀ − κ(t))` over the drift window, in samples·κ | ↓ |

**Why damage area is the "money metric":** recovery time tells you *how long* the model
was wrong; damage area tells you *how much skill × time* it lost — it prices the whole
drift in one number. It is also the metric where streaming's advantage is structural
rather than incidental.

### 6.3 Resource metrics (the three streaming constraints)

| Metric | Constraint |
|---|---|
| Memory footprint (bytes/model) | bounded memory |
| Throughput (samples/s), per-sample latency | bounded time |
| Number of passes over the data | single pass |

### 6.4 Label-efficiency metric

| Metric | What it proves |
|---|---|
| Performance vs fraction of labels used (`l_access` sweep) | streaming keeps working when labels are scarce |

### 6.5 The tolerance window — state it explicitly

A drift at index 5000 with an alarm at 5040 is a *successful detection with delay 40*.
An alarm at 9000 is a *false positive plus a missed detection*. Papers vary; **stating
and justifying the window is a mark of rigour**, not a detail. 📚

---

## 7. Datasets

### 7.1 Synthetic (ground-truth drift — required for detector metrics)

| Generator | Drift type it gives |
|---|---|
| **SEA** | Abrupt (used in Exp-01/02) |
| **Agrawal** | Abrupt, different rule families |
| **Hyperplane** | Incremental drift |
| **RandomRBF** | Gradual |
| **STAGGER** | Abrupt over discrete concepts |
| **LED** | Abrupt, irrelevant-feature control |
| **ConceptDriftStream** | Compositions — recurrence, multi-drift |

**Why synthetic is non-negotiable for the detector chapter:** without a known τ,
detection delay is uncomputable. Synthetic gives ground truth; real streams give
external validity.

### 7.2 Real streams

| Dataset | Role | Caveat |
|---|---|---|
| **Elec2** | Classic electricity-price drift stream | Known temporal autocorrelation — must report a "predict-the-previous-label" no-change baseline and Kappa+, never accuracy alone. Treat as **secondary**. |
| **Phishing** | Real drift, moderate size | — |
| **Forest Covertype** | Larger, multi-class | — |
| **Airlines** | Arrival-delay stream | — |
| **CreditCard** | Extreme imbalance (~0.17% fraud) | Accuracy is meaningless → AUC-PR. Provides the natural **delayed-label** story (chargebacks). |

**Protocol difference on real data:** there is no ground-truth τ, so drift is *inferred*
from rolling-metric dips and detector alarms, and the comparison targets **post-dip
behaviour** — dip depth, recovery slope, and the fraction of the post-drift window spent
below each model's own baseline.

**Scope note:** the locked scope says **≤3** real datasets (Elec2, Phishing,
CreditCard); the README roadmap lists 5. That inconsistency should be resolved before
the viva.

---

## 8. Statistical protocol

| Requirement | Value |
|---|---|
| Seeds | **≥10** (synthetic). Exp-02 ran 11. ✅ |
| Reporting | mean ± SD, **plus median and counts** — a heavy-tailed, zero-floored distribution makes the mean an unstable centre |
| Multi-dataset comparison | **Friedman test + Nemenyi post-hoc**, critical-difference diagram |
| Pairwise | **Wilcoxon signed-rank** |
| Effect sizes | reported, not just p-values |
| Budgets | fixed stream lengths, fixed drift positions, exact versions recorded |

📚 Demšar (2006), *JMLR* 7:1–30 — single runs are noise; this is the field's convention.

**Where the project stands:** ≥10 seeds ✅, mean±SD ✅, Friedman+Nemenyi ❌ (needs
multiple datasets — Exp-04/05), Wilcoxon ❌, effect sizes ❌.

---

## 9. The four lemmas (the Ch3 theory)

Formal setting: the concept is `C_t ∈ {C₀, C₁}` with real drift at τ; instantaneous
error `e_t = 1[ŷ_t ≠ y_t]`. Under immediate labelling the detector seeks a stopping time
`T` — classical **quickest detection** (Page 1954; Lorden 1971; Shiryaev). Under label
delay, each error becomes observable only at `t + L_t`, so the detector's filtration is

```
F_t = σ( e_s : s + L_s ≤ t )
```

— quickest detection **with delayed observations**, a variant the literature has barely
touched.

| # | Lemma | Statement | Novelty |
|---|---|---|---|
| **1** | Reaction floor | `T ≥ τ + L_τ` pointwise. No error-stream detector can alarm before the drift's labels arrive. | Novel **as a theorem** (trivial to prove — that's the point); the phenomenon is already known empirically 📚 |
| **2** | Power loss | Sensitivity degrades by `1/√p`: `Δ_min ∝ 1/√(pW)` instead of `1/√W`. | Appears **genuinely novel** — but check Alencar 2026 first (scoop risk) |
| **3** | False-alarm inflation | Batch arrival of size `b` inflates error-rate variance by the design effect `1 + (b−1)ρ`, so DDM fires above its nominal α. | **Most clearly novel** — no prior treatment found |
| **4** | Identifiability ceiling | Pure posterior drift (fixed `P(X)`, changed `P(y\|X)`) is invisible to any label-free detector: the likelihood ratio is 1, KL = 0. | **NOT novel** — Žliobaitė 2010. Ours is only the *use* (hybrid justification) |

**Scope guard:** no new change-detection algorithms, no regret bounds, no online convex
optimisation, no statistical learning theory, nothing beyond these four. If a lemma
resists a clean proof, it degrades to an empirical claim and the project survives.

**Known scoop risks to state proactively:** Alencar et al. 2026 (DSP) is an unread
experimental survey of exactly the Ch3 setting — HIGH risk. Botacin & Gomes 2024 already
quantified label-delay cost on a real pipeline (~50% exposure increase), so **never
claim "we show delayed labels hurt."**

---

## 10. Why this is not just a tutorial — the gap

The field lacks three things:

1. A **detector-level benchmark under label-observation regimes** (delayed, partial,
   selective).
2. **Formal bounds** connecting label latency/coverage to detection quality.
3. A standard **confirmation protocol** for early-warning + delayed-confirmation hybrids.

**Sharpened in 2026:** a standard benchmark now exists for **immediate**-label drift
detection (Cerqueira et al., KDD'26 — 14 detectors × 7 real datasets, 4 drift types,
each abrupt and gradual). It does **not** exist for delayed/partial/selective labels.
That refinement is the honest gap statement.

Of 33 catalogued papers in the project's review, only **4** study drift detectors under
delay at all, and **none** provides a detector-level benchmark, a formal detection bound,
or a built hybrid.

---

## 11. Known defects and honest caveats

Read this before claiming anything.

### 11.1 ⚠️ The damage-area metric is truncated — the headline ratio is inflated

**The defect.** `damage_area` is summed from τ to the **recovery point**. But recovery
is defined as the *first* 500-long in-band run starting at τ — so if the model is
already in-band at τ and never leaves for 500 samples, `recovery = τ`, and the
integration window is a **single sample**. Damage collapses to ≈ 0.

**Proof from the committed artifact** (`experiments/exp02/results.json`):

| Seed | Dip depth Δκ | Recorded damage |
|---|---|---|
| 1 | 0.027 | **0.0** |
| 3 | **0.071** | **0.0** |
| 7 | 0.053 | **0.0** |

A 0.071 excursion below baseline cannot integrate to zero over any non-degenerate
window. **8 of 11 streaming seeds sit on this floor**, while the frozen model — which
never recovers — always integrates the full window.

**Consequences (resolved by E01):**

1. The comparison was **asymmetric**: truncated integral (streaming) vs complete integral
   (frozen). Both the **294×** mean ratio and the **24×** worst-vs-best ratio were
   **upper bounds**; both are **retired**.
2. E01 corrected it with a fixed 5,000-sample window and a panel-shared `κ_ref`:
   `damage_ref` **70.6 ± 66.5** vs **504.4 ± 104.1** (**≈7.1×**); `damage_full`
   **293.9 vs 3,408.1** (**≈11.6×**); worst-vs-best **1.27×**. See
   [`Reports/E01-metric-v2.md`](Reports/E01-metric-v2.md).
3. The claim that "damage area and dip depth are the trustworthy metrics" is **too
   generous** — the legacy damage area was *defined in terms of* the same τ-anchored
   recovery rule. `damage_ref` and `damage_full` are the trustworthy ones now;
   recovery time is meaningful only conditional on an exit.
4. **Untouched:** frozen recovered on **0 of 11** seeds, and the streaming dip is
   consistently ~3.8× shallower. Those stand.

**The fix is done:** E01 integrates over a fixed window regardless of recovery and
keeps the old artifacts for auditability. Do not quote 294×/24× except as a retired
upper bound.

### 11.2 "Recovery = 0" does not mean "instant recovery"

It means the model **never left** its pre-drift band. 0 is the metric **floor**, not a
speed. 8 of 11 seeds sit on it. Reading it as a latency is the easiest way to
misinterpret the results.

Related: the recovery rule is anchored at τ, so a model that stays in band for the first
500 samples and dips *afterwards* is scored 0. Seven of eleven seeds did leave the band
after τ, but only three are credited with a recovery time. **Recovery time is therefore
not monotone in dip depth** (seed 3 dips deeper than seed 9 yet scores 0).

### 11.3 The frozen opponent is a strawman by construction

"A model that never updates never recovers" is true by definition ✅ measured, 0/11.
It is the weakest possible form of the claim and it is documented as such. The
informative comparison is a **scheduled-retrain** model (Exp-03), where streaming's
`d + a` is measured against the batch model's `r_next − τ` — and where the batch
recovery time should scatter uniformly over `[0, k]` as pure phase noise.

### 11.4 No detector, one synthetic stream, one drift type, one drift position

Both concluded experiments: SEA only, abrupt only, τ at 50% only, two models, no
scheduled baseline, no label delay, no detector. This is **existence evidence and a
first distribution — not a field-wide claim.**

### 11.5 Cost is unfavourable and reported as such

484 samples/s on an RTX 3050 for a 2.3k-parameter net (44/s on CPU) versus 0.17 s to fit
the RF. A streaming model that is slower than the batch one is a real tension — and per
the literature, ARF is ~2.2× slower than a retrained XGBoost 📚. The cost column stands
even though it is unflattering. At this window size the GPU is effectively required.

### 11.6 H2 was refuted, then restored

On seed 42 the two models were level pre-drift (0.755 vs 0.759) → "batch wins steady
state" was refuted. On 11 seeds the frozen model *does* lead (0.767 vs 0.751) → H2
supported. Not a flip-flop: **n=1 was noise**. The honest reading is that the frozen
model wins the stationary phase slightly and still loses the run because it collapses
permanently at τ. Streaming pays a small steady-state tax and earns it back on drift.

### 11.7 The scope guard says no deep learning — the pilot uses it

The locked scope lists "❌ Deep learning / `deep-river`" as out of scope, yet DeepRiver
is the subject of both experiments. Acknowledged in the Exp-01 write-up as a deliberate
extension, but it is **not formally amended in the scope document** — and an
undocumented scope reversal reads as drift, which is unfortunate in a drift project.

### 11.8 Duplicated metric code

The recovery/damage definitions exist twice (once per runner), copied by hand. Any
definition change must be made in both or Exp-01/Exp-02 stop being comparable. There is
no shared config-driven harness — that is the main missing development artifact.

---

## 12. Quick-fire ML Q&A

| Question | Answer |
|---|---|
| What is concept drift? | `P_t(X,y) ≠ P_{t+Δ}(X,y)` — the joint distribution changes over time. |
| Real vs virtual drift? | Real changes `P(y\|X)` (the rule) and breaks the model. Virtual/covariate shift changes only `P(X)` — the rule is intact, but input-watching detectors still fire on it. |
| Why is prequential evaluation correct? | Random splits leak future information on ordered data. Test-then-train scores each sample before it trains, so the score is honest. |
| Why not cross-validation? | There is no "end" to hold out, the order is meaningful, and folds would leak across time. |
| Why Kappa over accuracy? | Chance-corrected and imbalance-robust; accuracy overstates learning when temporal persistence gives a strong no-change baseline. |
| What is a drift detector, mechanically? | Mostly a 1-D change detector on the **model's error stream** — it asks "has this Bernoulli mean changed?" |
| Detection vs adaptation? | Detection notices; adaptation responds. Independently designable — blind adaptation needs no detector. |
| What is damage area? | `Σ (κ₀ − κ(t))` over the drift window in samples·κ — skill lost × time wrong. The one number that prices a whole drift. (Mind §11.1.) |
| Why does a rolling window adapt? | Old samples age out, so after a drift the model is refit on recent (new-concept) data within ~window samples. Memory stays bounded. |
| Why does a frozen model never recover? | It has no mechanism that reads the data. Recovery requires an external retrain decision. |
| Why is label delay a problem? | Every detector reads the error stream; late labels mean a late error stream, so drift is detected late — and meanwhile the model visibly reinforces the pre-drift concept. Prequential evaluation itself breaks: you cannot even measure your own accuracy in real time. |
| What is the reaction floor? | `T ≥ τ + L_τ` — no error-stream detector can alarm before the drift's labels arrive. A hard lower bound on detection delay, independent of detector quality. |
| What is partial vs selective labelling? | Partial: each label arrives with probability `p`. Selective: only samples the model *flagged* get labelled — an endogenous policy, so the detector learns the review policy, not the deployed model. |
| Why can't label-free detection confirm drift? | Pure posterior drift leaves `P(X)` identical, so any `P(X)`-based test has zero power (KL = 0). Label-free signals can only ever be an early *warning*. |
| What is the hybrid and why is it maximal? | Label-free early warning + delayed supervised confirmation. L4 says stage 1 can't confirm; L1 says stage 2 can't be early — so the hybrid is the information-maximal architecture given the available information. |
| Why SEA / synthetic at all? | Only synthetic streams have known drift positions, which makes detection delay and recovery time computable. Real streams validate external validity. |
| Why is Elec2 treated as secondary? | Known temporal autocorrelation; a "no-change" baseline and Kappa+ must be reported, and accuracy alone overstates learning. |
| Why binary classification only? | Scope guard — focus. Imbalance is a secondary axis, not a separate task. |
| Why is single-run evidence meaningless here? | Synthetic streams are random (draws + label noise), so one replica can't separate a model property from a draw property. ≥10 seeds, mean±SD is the field's convention (Demšar 2006). |
| What would falsify the thesis? | Streaming recovery comparable to a sane retrain interval (`d + a ≥ r_next − τ`); no dip at τ (dataset problem, not a model win); no significant difference under Friedman/Nemenyi; or the reactivity win not worth the resource cost. |
| What if the hybrid loses? | A complete negative result: "we quantified how delayed labels break existing methods, and why the obvious fix is insufficient." Lemmas 1–3 hold regardless — Ch3 cannot fail. |

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Prequential** | Test-then-train evaluation: score each sample, then learn from it |
| **Concept drift** | Change over time in `P(X,y)` |
| **Real drift** | Change in `P(y\|X)` — the rule itself |
| **Virtual drift / covariate shift** | Change in `P(X)` only; rule intact |
| **Abrupt / gradual / incremental / recurring** | The four drift shapes over time |
| **κ₀** | Pre-drift baseline rolling Kappa |
| **Δκ** | Dip depth — how far below κ₀ the model falls |
| **Damage area** | Skill lost × time below baseline, in samples·κ |
| **Reaction floor** | `T ≥ τ + L_τ` — detection cannot precede label arrival |
| **τ** | The true drift onset time |
| **L_t** | Label latency process (fixed, stochastic, or policy-driven) |
| **d, a** | Detection delay, adaptation time (`d + a` = streaming recovery) |
| **r_next, k** | Next scheduled retrain; retrain interval |
| **Tolerance window W** | How close an alarm must be to a true drift to count as a detection |
| **MTFA / MDR** | Mean time between false alarms / missed-detection ratio |
| **Rolling Kappa** | Kappa over a sliding recent window (w = 1,000) — reveals drift |
| **Blind vs informed adaptation** | Adapt without a detector vs triggered by one |
| **Delay Tax** | The project's problem: the cost of learning from late, partial, selective labels |
| **Hybrid detector** | Label-free early warning + delayed supervised confirmation |

---

## 14. One-screen recap

**Problem.** Supervised binary classification on a non-stationary stream — single pass,
bounded memory, bounded per-sample time, with the distribution allowed to change.

**Models.**

- *Used (measured):* DeepRiver `RollingClassifier` (RNN 3→16→2, rolling window 500,
  Adam 1e-3, 1 step/sample) as the streaming learner; sklearn `RandomForest(100)` fit
  once on the first 20% as the frozen batch baseline.
- *Planned:* Hoeffding Tree / HAT / ARF / SRP / Leveraging Bagging / kNN+ADWIN / online
  LR (streaming); frozen / scheduled-retrain / oracle-tuned XGBoost (batch ladder);
  ADWIN / DDM / EDDM / Page-Hinkley / KSWIN (detectors).

**Why those.** The streaming model was chosen to test the thesis mechanism with a deep
learner; its rolling window *is* the drift mechanism, so no detector is needed. The
frozen RF is deliberately the weakest possible opponent — the pilot is the weakest form
of the claim. XGBoost is held back as the honest oracle-tuned upper bound.

**Metrics.** Cohen Kappa (headline, chance-corrected), rolling Kappa w=1,000
(recency), κ₀ / Δκ / **recovery time** / **damage area** (drift-window family),
accuracy and AUC-PR (secondary), memory / throughput / passes (resource), label
efficiency (planned). Detector metrics: delay, FP, MDR, MTFA.

**Method.** Prequential test-then-train on SEA with one abrupt drift at τ = 30,000
(60k samples, 20% warm-up, 10% label noise), 11 seeds, rolling window w = 1,000,
recovery = first 500-sample sustained return to 0.95·κ₀ within a pre-registered
5,000-sample budget; damage area = the integral of skill lost over the drift window.

**Result.** Frozen recovered **0/11**; streaming 11/11 inside budget (median 0 — never
left its band). Dip 0.045 vs 0.173 (~3.8×). Damage 13 ± 31 vs 3,710 ± 592. Full run
0.883/0.753 vs 0.854/0.685. Seed 42 reproduced exactly (delta 0).

**Three things to say before being asked.** (1) The damage ratio is an **upper bound** —
the metric is truncated by the recovery rule on 8 of 11 seeds. (2) The frozen baseline is
a **strawman by construction**; scheduled-retrain is the real opponent and it is next.
(3) These are **one synthetic stream, one abrupt drift, no detector, no label delay** —
existence evidence, not a field-wide claim.
