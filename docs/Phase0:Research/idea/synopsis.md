# The Delay Tax

**Drift detection and adaptation when labels are late, partial, and biased**

*Project synopsis — streaming (online) machine learning with a systems engineering
spine*

---

## 1. Summary

Almost all machine learning assumes the dataset is a finished object: it sits on
disk, you load it, you split it, you fit a model, and you are done. **Streaming
machine learning** drops that assumption. Data arrives one sample at a time,
potentially forever, and is too large or too fast to store. The model must learn
incrementally, in a single pass, using bounded memory — and because the world that
generates the data keeps changing, it must also *notice when it becomes wrong* and
adapt. That last part, **concept drift**, is the research problem.

This project targets a specific, under-studied failure of the field's core
assumption. Nearly every paper on drift detection and adaptation assumes labels
arrive **instantly, completely, and without bias** alongside each sample. In
production they do not: fraud is confirmed 30–90 days after the fact, ad
conversions arrive a week later, analyst verdicts sometimes never come. This
project asks what happens when that assumption breaks — formally, measurably, and
end to end — and whether a practical remedy exists.

The project has three parts, in increasing order of novelty:

1. **Baseline.** A rigorous detector-and-adaptation benchmark under controlled
   synthetic drift (the field's "what happens" reference numbers).
2. **System.** A real, containerised continual-learning service — the same model
   that serves predictions is the model that learns — with live drift monitoring,
   measured for the operational costs the literature never reports: adaptation
   latency, memory growth, replica divergence, recovery cost.
3. **Contribution.** A formal model of **delayed, partial, and selective labelling**
   as a *delay tax* — four provable lemmas describing exactly how detection
   degrades — plus a **hybrid detector** that pairs a label-free early warning with
   delayed supervised confirmation, evaluated against the baseline.

The mathematical contribution is deliberately applied: concentration bounds,
quickest-detection theory, and hypothesis testing, all of which can be verified
empirically. The systems contribution is measurement the field lacks because the
field is populated by ML researchers rather than infrastructure engineers. Every
chapter has a survivable failure mode; no single bet can sink the project.

---

## 2. Background: what streaming machine learning is

### 2.1 The three constraints

Standard batch learning rests on an assumption so common it is usually invisible:
the data is finite, available all at once, and drawn i.i.d. from a fixed
distribution that will not change. Streaming ML drops all three parts. Any
algorithm that claims to be a streaming algorithm must respect three hard
constraints:

1. **Single pass.** Each sample is seen exactly once; you cannot go back and
   revisit sample #5.
2. **Bounded memory.** Memory usage must not grow with the number of samples seen.
   A model that processed ten billion rows must occupy roughly the same space as
   one that processed ten thousand.
3. **Bounded time per sample.** Processing must keep up with the arrival rate, or
   the learner falls behind forever.

A useful test: *if this ran for a year without restarting, would it still work?*
If the answer involves "well, memory would…", it is not a streaming algorithm.

### 2.2 Why it matters

There are two distinct motivations, often confused:

- **The engineering motivation — scale and latency.** The data does not fit in
  RAM, arrives too fast, or needs a model updated *now* rather than after
  tonight's retraining job. Clickstreams, sensors, network packets, financial
  ticks.
- **The statistical motivation — the world changes.** Even if the data *did* fit
  in memory, a model trained on last year's data may be wrong about this year,
  because the process generating the data has changed. Batch learning has no
  mechanism to notice this. Streaming learning does.

The statistical motivation is the spine of this project. In streaming ML, the
central question is not "how accurate is my model?" — a single number — but *"how
does accuracy behave over time?"* — a curve. Accuracy becomes a time series;
drops, recoveries, and recovery speed are the interesting phenomena. Almost every
good figure in this field plots something against "number of samples seen."

### 2.3 The River framework

River is "scikit-learn for data streams" — the standard Python library for online
machine learning (Montiel et al., JMLR 2021; merger of *creme* and
*scikit-multiflow*). It is the instrument for this project, not the subject. Its
core API is four methods:

```python
model.learn_one(x, y)        # update the model with ONE labelled sample
model.predict_one(x)         # predict a label for ONE sample
model.predict_proba_one(x)   # predict class probabilities
metric.update(y, y_pred)     # update a running score — TRUE label first
```

where `x` is a Python dict — a deliberate design choice that lets features appear,
disappear, or stay sparse without reshaping. River ships the models this project
needs (Hoeffding trees, adaptive forests, linear models), the drift detectors
(ADWIN, DDM, EDDM, Page-Hinkley, KSWIN), synthetic generators with ground-truth
drift, real datasets, and incrementally-updated metrics.

---

## 3. Background: concept drift

### 3.1 Definition

A model estimates a joint distribution $P(X, y)$, usually decomposed as
$P(y \mid X) \cdot P(X)$. **Concept drift** is when that distribution changes over
time:

$$P_t(X, y) \neq P_{t+\Delta}(X, y).$$

The world that produced the training data is not the world being predicted in.

### 3.2 Two flavours that matter

- **Real drift** — $P(y \mid X)$ changes. The input–output relationship itself
  changed, which genuinely breaks the model.
- **Virtual drift / covariate shift** — $P(X)$ changes while $P(y \mid X)$ holds.
  The inputs have shifted but the rule is intact; accuracy may not suffer.

The distinction drives detector design: detectors that watch the *input
distribution* fire on virtual drift (false alarms from the model's point of view);
detectors that watch the *error stream* only fire on drift that actually costs
accuracy. This tension is central to this project.

### 3.3 Types by shape over time

The standard taxonomy, in terms of which concept is active:

```
SUDDEN (abrupt)         GRADUAL                 INCREMENTAL             RECURRING

 ────┐                   ──┐ ┌┐ ┌──┐             ──╲                 ──┐   ┌──┐   ┌──
     │                     └─┘└─┘  └───            ╲___                └───┘  └───┘
     └────────                                        ╲──

concept A flips        A and B interleave,      A morphs smoothly     A and B alternate
to concept B           B wins over time         into B                (e.g. seasonal)
instantly
```

- **Sudden / abrupt** — an instantaneous switch; easiest to detect, most damaging.
- **Gradual** — a transition window during which samples come from A *or* B with
  shifting probability; hard to detect, because error rises slowly and noisily.
- **Incremental** — the concept deforms continuously through intermediate states.
- **Recurring** — old concepts return (day/night, seasons). Forgetting the old
  concept is actively wasteful, motivating archive-and-restore approaches.

There is also the **blip** — a one-off anomaly that is *not* drift. A good detector
must not fire on it. This is the stability–plasticity trade-off: react fast enough
to catch real drift, but not so fast that noise triggers a reset.

### 3.4 How detectors work

Most detectors do not look at the features at all. They watch a one-dimensional
stream of the model's errors ($1$ = wrong, $0$ = right) and ask: *has this
stream's mean changed?*

| Detector | Idea | Notes |
|---|---|---|
| **DDM** (2004) | Track error rate $p$ and its std $s$; warn at $p + s > p_{min} + 2s_{min}$, alarm at $3s$ | Simple, cheap; weak on gradual drift |
| **EDDM** (2006) | Track the *distance between* errors instead | Designed to fix DDM's gradual-drift weakness |
| **ADWIN** (2007) | Adaptive window; when two sub-windows have statistically different means, drop the older part | Rigorous guarantees, no window-size parameter; the de-facto standard |
| **Page-Hinkley** (1954) | CUSUM-family: accumulates deviation from the running mean | Classical sequential analysis; threshold-sensitive |
| **KSWIN** (2020) | Kolmogorov–Smirnov test between a sliding window and a reference | Non-parametric, distribution-based |

The trade-off is always the same: **sensitivity vs. false alarms.** Tune a
detector to catch drift faster and it fires on noise. That trade-off curve is
measurable, and measuring it is part of this project's baseline chapter.

### 3.5 How models adapt

Three responses to drift:

1. **Blind adaptation (no detector).** Keep learning and let recent data outweigh
   old data — sliding windows, fading factors, a learning rate that never reaches
   zero. Always-on, simple, but adapts slowly to sudden drift and permanently
   discards old-but-still-valid knowledge.
2. **Informed adaptation (detector + reaction).** Run an explicit detector; when it
   fires, reset the model, retrain on a recent buffer, or swap in a background
   model trained in parallel. Reacts sharply, but a false alarm throws away a good
   model.
3. **Ensembles.** Keep a pool of models, weight them by recent performance, drop
   the worst, add fresh ones. Drift handling is emergent. This is what **Adaptive
   Random Forest (ARF)** does, and it is the strongest general baseline.

The deep question — whether explicit detection actually *beats* passive
adaptation — is contested in the literature, and this project measures it directly
in its baseline chapter.

### 3.6 How a streaming model is evaluated

Random train/test splitting is invalid for ordered streams: order is meaningful,
there is no "end" to hold out, and a random split leaks future information into
training. The standard protocol is **prequential evaluation** (test-then-train):

```python
for x, y in stream:
    y_pred = model.predict_one(x)   # model has NEVER seen this sample
    metric.update(y, y_pred)        # so this score is honest
    model.learn_one(x, y)           # now let it learn
```

Every sample is used for testing and training, in that order — nothing wasted,
nothing leaked. Cumulative metrics converge and hide drift; **rolling** metrics
over a recent window reveal dips and recoveries. The accuracy-over-time curve is a
result in itself: dip depth and recovery slope are what get compared.

Two evaluation facts anchor this project:

- **Memory and per-sample time are first-class results** in streaming ML, not
  afterthoughts — the field's whole claim is bounded resources.
- **Statistical rigour is expected**: multiple seeds reported as mean ± std,
  fixed experimental budgets, and the **Friedman test + Nemenyi post-hoc** with a
  critical-difference diagram (Demšar, 2006) as the accepted convention for
  comparing many methods over many datasets.

---

## 4. The problem: labels are late, partial, and biased

### 4.1 The hidden assumption

Every paper in the field's canonical evaluation loop assumes the label `y` arrives
in the same tick as the sample `x`. The detector watches the error stream $e_t$ in
real time; the evaluator scores each prediction immediately; the learner updates
immediately. In production, none of this holds:

- A fraud model approves a transaction *now* and learns the truth **30–90 days
  later**, when the chargeback lands.
- An ad model serves an impression *now* and learns the conversion **next week**.
- An intrusion model flags a login *now* and receives the analyst's verdict
  **later — or never**.
- Labels arrive **in batches** (overnight reconciliation), **out of order**, and
  **only for flagged cases** — which makes the observed errors a biased sample of
  true errors.

### 4.2 The cascade of consequences

Follow the consequence of a late label; it is not obvious and it is nasty:

1. **Detection is late.** A drift detector works by watching the error stream. If
   the error stream is $d$ days late, drift is detected $d$ days after it
   happened — and the model has been silently wrong for a month.
2. **Evaluation breaks first.** Prequential evaluation cannot score a prediction
   whose label has not arrived. The instrument the entire field uses to measure
   success does not work in the setting the field claims to serve: **you cannot
   even measure your own accuracy in real time**.
3. **Adaptation compounds the error.** While labels are pending, the model keeps
   learning from stale samples. After a drift, it is not just slow to adapt — it
   is actively *reinforcing the pre-drift concept* with outdated data.
4. **The label process itself misbehaves.** Batch arrival injects autocorrelation
   into the error stream; selective labelling injects bias. Detectors calibrated
   for i.i.d. errors misread both — false alarms, or missed drift entirely.

### 4.3 Why this is a research gap, not a tutorial topic

The research phase of this project (a deep literature survey; see the reference
list, §12) established three things:

- **The area is active but fragmented.** The delayed/partial-label literature
  spans roughly 2008–2026, but terminology alone is split across *verification
  latency, delayed labelling, label latency, extreme verification latency, and
  partially labelled streams* — and the same problem is studied separately in
  semi-supervised streaming, stream active learning, and positive-unlabelled
  learning. No widely accepted benchmark or evaluation protocol exists.
- **The specific degradation is unquantified.** Papers show that delayed labels
  *harm* classifiers or active-learning systems, but almost none isolates the
  **detector** and produces a clean curve of detection delay vs. label latency, or
  false-alarm rate vs. label coverage.
- **The confirmation protocol is unbuilt.** Research explicitly names "a
  two-stage detector: label-free alarm, delayed supervised confirmation, calibrated
  action" as a promising, unbuilt architecture. This project builds exactly that.

The most relevant prior syntheses: Gomes et al. (2022), the *ACM Computing Surveys*
survey on semi-supervised learning for delayed partially labelled streams; and a
2026 experimental survey on concept-drift detection in delayed and partially
labelled streams (*Digital Signal Processing*). Both confirm the gap this project
targets.

---

## 5. Research questions

Three questions, one spine. Each is falsifiable and measurable.

- **RQ1 — Baseline (guaranteed results).** Under controlled synthetic drift —
  sudden, gradual, incremental, recurring — how do the standard detectors (ADWIN,
  DDM, EDDM, Page-Hinkley, KSWIN) compare on detection delay, false-alarm rate, and
  missed detections; and how do adaptive models (Hoeffding Adaptive Tree, Adaptive
  Random Forest) recover relative to static ones (plain Hoeffding Tree, Naive
  Bayes)? *This is the control condition: the idealised numbers the field's papers
  assume, measured with our own rigour.*

- **RQ2 — System (novel by measurement).** What does running continual learning as
  a real service cost — p50/p99 prediction latency during adaptation, throughput
  ceiling, memory growth, checkpoint/restart cost, and **how fast do scaled-out
  replicas diverge** from each other? Where does it break first?

- **RQ3 — Contribution (the gap).** By how much — exactly — does delayed, partial,
  and selective labelling degrade drift detection and adaptation? Can a **hybrid
  detector** (label-free early warning + delayed supervised confirmation) recover
  the immediate-label baseline, or come provably close? Do the four lemmas of §6
  hold empirically?

---

## 6. The mathematical angle: the delay tax

The project's theory. It formalises "delayed labels are bad" into **four provable
claims** about *how* and *by what factor* detection degrades. The mathematics is
deliberately applied — probability, concentration bounds, hypothesis testing,
classical quickest detection — every lemma has an attached experiment, and none
requires novel analysis.

### 6.1 The formal setting

Let the concept be $C_t \in \{C_0, C_1\}$ with a real drift at time $\tau$. The
model predicts $\hat{y}_t$; the instantaneous error is

$$e_t = \mathbb{1}[\hat{y}_t \neq y_t].$$

**Immediate labelling.** The detector observes $(e_t)$ in real time and raises an
alarm at a stopping time $T$. This is the classical **quickest detection** problem
(Page 1954; Lorden 1971; Shiryaev). The field's "detection delay vs. false-alarm
trade-off" is the empirical image of Lorden's minimax criterion: minimise
worst-case detection delay subject to a false-alarm constraint (average run
length between false alarms).

**Delayed labelling.** Each error $e_t$ becomes observable only at time
$t + L_t$, where $L_t$ is a latency process — fixed ($L_t = d$), stochastic
(e.g. geometric with mean $d$), or driven by a review policy (only flagged samples
are ever labelled). The detector's information set at time $t$ is the filtration

$$\mathcal{F}_t = \sigma\big( e_s : s + L_s \le t \big).$$

This is **quickest detection with delayed observations** — a variant the
literature has barely touched. The four lemmas follow from this setting.

### 6.2 Lemma 1 — The reaction floor

> Any detector that observes only the error stream cannot alarm before the
> drift's labels arrive: $T \ge \tau + L_\tau$ pointwise, hence
> $\mathbb{E}[T] \ge \tau + \mathbb{E}[L]$.

*Proof idea.* The first error containing information about the new concept is
$e_\tau$, and it enters $\mathcal{F}_t$ only at $t = \tau + L_\tau$. The alarm is
a stopping time with respect to $\mathcal{F}_t$, so it cannot precede that
information. Trivial — and that triviality is the point: it is a **hard lower
bound that no algorithm can beat**, independent of detector quality. It converts
"you detect drift a month late" from a complaint into a theorem.

*Experiment.* For each delay regime, measure every detector's detection delay and
confirm it never dips below $d$ (fixed) or $\mathbb{E}[L]$ (stochastic).

### 6.3 Lemma 2 — The power loss

> With only a fraction $p$ of labels arriving, a windowed detector's sensitivity
> degrades by the factor $1/\sqrt{p}$: its minimum detectable mean shift grows
> from $\Delta_{\min} \propto 1/\sqrt{W}$ to
> $\Delta_{\min} \propto 1/\sqrt{pW}$.

*Proof idea.* The observed error stream is a randomly thinned version of the true
one: a window of $W$ arrivals yields only ~$pW$ distinct fresh errors. Any
mean-estimate test has standard error $\sigma / \sqrt{n}$ with $n = pW$
(Hoeffding / CLT-based; ADWIN's threshold is a constant times
$\sqrt{\ln(1/\delta)/W}$). Thinning inflates the standard error by $1/\sqrt{p}$.

*Experiment.* For label coverage $p \in \{1, .5, .25, .1\}$ and a fixed drift
magnitude, measure ADWIN's detection delay and verify it tracks the predicted
$1/\sqrt{p}$ inflation.

### 6.4 Lemma 3 — False-alarm inflation under batch arrival

> If labels arrive in batches of size $b$, the observed error stream inherits
> intra-batch correlation $\rho > 0$, inflating the variance of the estimated
> error rate by the clustered-data factor $1 + (b-1)\rho$. Detectors calibrated
> for i.i.d. errors then fire spuriously: DDM's alarm rate rises above its
> nominal level $\alpha$.

*Proof idea.* The variance of a clustered sample mean carries the standard design
effect $1 + (b-1)\rho$. DDM's thresholds assume independent Bernoulli errors.
Batch arrival — exactly what overnight or weekly label reconciliation does in
production — violates that assumption, so the detector sees "drift" in pure noise.

*Experiment.* Feed detectors a delayed-error stream under batch arrival with *no*
actual drift; measure false-alarm rate vs. batch size $b$.

### 6.5 Lemma 4 — The identifiability ceiling

> If $P(X)$ is unchanged but $P(y \mid X)$ differs between concepts (pure
> posterior drift), no label-free detector has any power: the likelihood ratio of
> the observed $X$-stream is identically 1, so the KL divergence between the two
> hypotheses is 0.

*Proof idea.* The marginal $P(X)$ cannot distinguish between two joints that share
it. Formally, the two candidate distributions over $X$-observations are identical,
so no test can have power above its size. This is the formal content of
Žliobaitė's detectability analysis (2010) — and it **justifies the hybrid by
construction**: label-free signals can only ever be an early warning; confirmation
requires labels. The hybrid of §8 is the information-maximal architecture
available, which is the strongest design statement a scoping document can make.

*Experiment.* Construct a generator with pure posterior drift (flip $P(y \mid X)$
in a region while keeping the marginal fixed); confirm label-free detectors miss
it while the hybrid's delayed confirmation catches it.

### 6.6 What the mathematics is for

The lemmas give the report a theory section that is genuinely provable — each with
an empirical counterpart — without betting the project on novel analysis. They
convert Chapter 3's headline from a table of numbers into a narrative:

> *Detection under delay is bounded below by the reaction floor, weakened by
> $1/\sqrt{p}$, distorted by batch autocorrelation, and capped by
> identifiability — and the hybrid is the best possible response.*

Explicitly **out of mathematical scope**: new change-detection algorithms, regret
bounds, online convex optimisation, statistical learning theory, and any theorem
beyond the four lemmas. If a lemma resists a clean proof, it degrades to an
empirical claim with a simulation study — the project survives either way.

---

## 7. Chapter 1 — The baseline (the control condition)

### 7.1 Purpose

Everything later is measured against "what the literature says happens." Chapter 1
produces those idealised numbers ourselves, with documented rigour, so that
Chapter 3 can show reality is worse.

### 7.2 The run matrix

| Axis | Values |
|---|---|
| Drift type | sudden (SEA), incremental (Hyperplane), gradual (RandomRBF), recurring (ConceptDriftStream composition) |
| Detector | ADWIN, DDM, EDDM, Page-Hinkley, KSWIN |
| Model | Hoeffding Tree (static), Hoeffding Adaptive Tree (adaptive), ARF (ensemble), Logistic Regression (blind/passive) |
| Detector metrics | detection delay, false-positive rate, missed detections, MTFA / MTD |
| Model metrics | dip depth, recovery time, rolling accuracy, memory, throughput |
| Rigour | ≥ 10 seeds, fixed stream length (~100k), fixed drift positions, mean ± std, Friedman + Nemenyi |

### 7.3 Detector metrics — precise definitions

Detector metrics require **known ground-truth drift points**, which is why
synthetic streams are non-negotiable for this part:

| Metric | Definition | Good value |
|---|---|---|
| **Detection delay** | samples between true drift and first alarm | low |
| **False positives (FP)** | alarms with no drift nearby | low |
| **Missed detections (FN)** | drift points with no alarm within a tolerance window | low |
| **MTFA** | mean time between false alarms | high |
| **MTD** | mean time to detection | low |
| **Recovery time** | samples until rolling accuracy returns to pre-drift level | low |

A **tolerance window** must be defined explicitly: a drift at index 5000 with an
alarm at 5040 is a successful detection with delay 40; an alarm at 9000 is a false
positive plus a missed detection. Papers vary; stating and justifying the window is
a mark of rigour.

The central finding to expect: **no detector wins on all three of delay, FP, and
FN.** Sensitivity trades against false alarms. Showing that trade-off with real
numbers is a legitimate result — and it becomes the baseline against which the
hybrid is judged.

### 7.4 Deliverable

A reusable, config-driven, seeded experiment harness (one runner serves both
Chapter 1 and Chapter 3), plus the first real results: trade-off tables,
accuracy-over-time curves, and statistical comparisons.

---

## 8. Chapter 3 — The delayed-label contribution (the research)

*Chapter 2, the system, is described in §9; Chapter 3 is presented here because it
is the intellectual heart of the project.*

### 8.1 Step 1 — The delay regimes

Labels are withheld from the baseline streams according to explicit observation
processes:

| Regime | Process | Production story |
|---|---|---|
| **Fixed delay** | $L_t = d$, for $d \in \{10, 100, 1000\}$ samples | Batch reporting on a fixed schedule |
| **Stochastic delay** | $L_t \sim$ Geometric(mean $d$) | Variable human/queue latency |
| **Partial labelling** | label arrives with probability $p \in \{1, .5, .1\}$ | Only some events ever get confirmed |
| **Selective labelling** | only samples the model *flagged* get labelled | "Only flagged cases are reviewed" — the biased case |

The selective regime is the subtle one: it is not mere label scarcity but an
**endogenous observation policy**, because the probability of being labelled
depends on the model's own prediction. Any detector trained on those labels
estimates the review policy, not the deployed model.

### 8.2 Step 2 — Measure the degradation

For every regime, re-run the Chapter 1 matrix and compare against the $d = 0$
control, at two levels:

- **Detector level:** how do detection delay, FP rate, and missed-detection rate
  change with $d$ and $p$? This is the clean, largely-unpublished curve the field
  lacks — and it is exactly where the lemmas live.
- **Downstream level:** how does the *model's* accuracy-over-time curve degrade?
  Dip depth, recovery time, and whether the model ever returns to baseline.

### 8.3 Step 3 — Build the hybrid

The hybrid is a two-stage architecture:

```
input stream x_t ──► label-free early warning (monitors P(X))
                        │
                        ▼
                   "possible drift"   (fast, but fires on virtual drift too)
                        │
                        ▼
        delayed error stream e_{t-d} ──► supervised confirmation
                        │
                        ▼
              confirmed alarm → adaptation (reset / retrain / swap)
```

- **Stage 1 (label-free, immediate):** monitors the input distribution $P(X)$
  (windowed two-sample tests, e.g. KS / MMD-style statistics). Fires early —
  possibly on harmless virtual drift — but provides a *warning within a tolerance
  window*.
- **Stage 2 (supervised, delayed):** confirmation from the error stream once the
  delayed labels arrive. A confirmed alarm triggers adaptation; an
  unconfirmed warning is revoked (with the false-alarm cost explicitly
  recorded).

The design is not ad hoc: **Lemma 4 shows stage 1 cannot confirm, and Lemma 1
shows stage 2 cannot be early.** The hybrid is the only architecture that is both
early *and* reliable, given the information available.

### 8.4 Step 4 — Evaluate

Compare four systems on the full delay matrix: (a) naive delayed-supervised
detector, (b) label-free only, (c) the hybrid, (d) the immediate-label control
from Chapter 1. Test the lemmas empirically; verify the $1/\sqrt{p}$ factor and
the batch-correlation inflation with real measurements.

### 8.5 The failsafe

If the hybrid does not beat the naive baselines, that is a complete, publishable
**negative result**: "we quantified exactly how delayed labels break existing
methods, and showed why the obvious fix is insufficient." Lemmas 1–3 hold
regardless and remain the contribution. Chapter 3 **cannot fail** — which is
precisely why it is the bet.

---

## 9. Chapter 2 — The system (the infrastructure spine)

### 9.1 What is built

A real, containerised, reproducible continual-learning service — the part that
makes this a major project rather than a notebook:

```
events ──► event stream (Redis Streams / Kafka) ──► learning consumer
                                                        │
                                                        ▼
                     POST /predict ◄── model service (serves AND learns, one pass)
                                                        │
                                                        ▼
                          live drift monitor (a real service, not a script)
                                                        │
                                                        ▼
                     Prometheus metrics ──► Grafana dashboard
                     load generator ──► pushes the system until something breaks
```

The defining property: **the same model that serves predictions is the model that
learns** — not a served snapshot and a separate offline trainer. That is what makes
the operational questions real.

### 9.2 What is measured (the "reality tax")

- **Adaptation cost.** What do model resets, background retraining, and drift
  reactions do to p50/p99 prediction latency and throughput? What happens when a
  reset coincides with peak load?
- **Replica divergence.** Scale the service to 3 replicas. Each learns from the
  events it happens to consume; they diverge. How fast? Which one is "the model"?
  Nobody has a clean answer to horizontal scaling of a stateful learner —
  measuring the divergence is itself a publishable observation.
- **Backpressure.** When consumers fall behind, do we drop learning or drop
  serving? What is the cost of each?
- **Recovery.** How expensive is checkpointing and restarting model state? A batch
  model is an immutable artifact you can roll back; a streaming model has no
  artifact. What does recovery even mean?

### 9.3 Why it is a research chapter, not a demo

The field evaluates by replaying a CSV in a single Python process: labels arrive
instantly and always, no network, no queue, no concurrency, no backpressure, and
success is measured almost entirely as accuracy. Production violates every one of
those assumptions. The system chapter measures the gap — the operational cost the
literature never reports — and it exists because the field is populated by ML
researchers, not infrastructure people. This is the author's home turf, and it is
deliberately scoped to a single node (docker-compose), which is sufficient for the
measurements and honest about their limits.

---

## 10. Datasets

| Kind | Data | Role |
|---|---|---|
| Synthetic | SEA, Hyperplane, RandomRBF, STAGGER, ConceptDriftStream compositions | Ground-truth drift positions → detector metrics, lemmas 1–4 |
| Real | Elec2, Phishing, CreditCard (≤ 3) | External validity; natural delayed-label story (fraud) |
| Baseline honesty | Elec2 | Known temporal autocorrelation: report a no-change baseline and Kappa+, never accuracy alone |

Synthetic streams are the main experimental tool because drift position is known
exactly — detection delay would otherwise be uncomputable. Real streams validate
that the findings are not artefacts of generators. Elec2's known flaws are handled
by reporting a "predict the previous label" baseline and temporal-aware metrics,
rather than treating it as neutral ground truth.

---

## 11. Scope guards

The most common way student projects fail is scope creep, not difficulty. Written
down and defended:

| ❌ Out of scope | ✅ Instead |
|---|---|
| Deep learning / online neural nets | Classical models + ensembles (River) |
| Regression, clustering, recommendation | **Classification only** |
| Distributed *training* (parameter servers, federated learning) | Single-node service; measuring replica divergence is in scope |
| Building a streaming framework | River; we build the harness and system *around* it |
| Kubernetes / multi-cloud production deployment | docker-compose on one machine |
| More than ~3 real datasets | Exactly 3, chosen and justified |
| Novel adaptation algorithms | Hybrid architecture + formal delay model + measurement |
| Theorem-proving beyond the four lemmas | The lemmas; anything harder degrades to empirical |

---

## 12. References (key works)

Citations from the research phase — **each must be verified on Google Scholar or
Semantic Scholar before entering the report bibliography.**

**Foundational**
- Domingos & Hulten (2000). *Mining High-Speed Data Streams* (VFDT / Hoeffding Tree). KDD.
- Bifet & Gavaldà (2007). *Learning from Time-Changing Data with Adaptive Windowing* (ADWIN). SDM.
- Gama et al. (2004). *Learning with Drift Detection* (DDM). SBIA.
- Baena-García et al. (2006). *Early Drift Detection Method* (EDDM).
- Raab, Heusinger & Schleif (2020). *Reactive Soft Prototype Computing* (KSWIN). Neurocomputing.
- Page (1954). *Continuous Inspection Schemes*. Biometrika. (Roots of Page-Hinkley / CUSUM.)

**Surveys and methodology**
- Gama, Žliobaitė, Bifet, Pechenizkiy & Bouchachia (2014). *A Survey on Concept Drift Adaptation*. ACM Computing Surveys. **The reference.**
- Lu et al. (2019). *Learning under Concept Drift: A Review*. IEEE TKDE.
- Gama, Sebastião & Rodrigues (2013). *On Evaluating Stream Learning Algorithms*. Machine Learning. (Prequential justification.)
- Demšar (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets*. JMLR. (Friedman + Nemenyi.)

**Delayed and partial labels (the gap)**
- Žliobaitė (2010). *Change with Delayed Labeling: When is it Detectable?* IEEE ICDM Workshops. (Identifiability — Lemma 4.)
- Grzenda, Gomes & Bifet (2020). *Delayed Labelling Evaluation for Data Streams*. Data Mining and Knowledge Discovery.
- Gomes et al. (2022). *A Survey on Semi-supervised Learning for Delayed Partially Labelled Data Streams*. ACM Computing Surveys.
- Alencar et al. (2026). *Concept drift detection in delayed and partially labeled data streams: An experimental survey*. Digital Signal Processing.

**Models and tools**
- Gomes et al. (2017). *Adaptive Random Forests for Evolving Data Stream Classification*. Machine Learning.
- Oza & Russell (2001). *Online Bagging and Boosting*.
- Montiel et al. (2021). *River: Machine Learning for Streaming Data in Python*. JMLR.

---

## 13. Timeline

Roughly 10–15 hours/week alongside coursework, deliberate slack built in:

| Phase | Weeks | Work | Exit criterion |
|---|---|---|---|
| **0 — Orientation** | 1–4 | Verify citations; get River running; reproduce one published baseline number | A baseline number reproduced; bibliography skeleton |
| **1 — Baseline** | 5–12 | Harness + Chapter 1 run matrix | Reusable harness; idealised numbers done |
| **2 — System** | 13–24 | Event stream, model service, live monitor, metrics, load generator | Running docker-compose stack |
| **3 — Break it** | 25–32 | Latency / throughput / memory / divergence / recovery measurements | "Reality tax" data |
| **4 — Delay tax** | 33–44 | Delay regimes → degradation curves → hybrid → lemma verification | The contribution (or the negative result) |
| **5 — Write-up** | 45–52 | Report, figures, statistics, README, demo | Submittable project |

Phases 2–3 can shrink if the year tightens; Phase 4 keeps its budget. Phase 1 goes
first because everything depends on the control condition.

---

## 14. Deliverables

1. **Experiment harness** — config-driven, seeded, reusable; one runner for
   Chapters 1 and 3.
2. **The running system** — docker-compose stack: event stream, model service,
   live drift monitor, Prometheus/Grafana, load generator.
3. **Results** — accuracy-over-time curves; delay/FP/FN trade-off tables; the
   delay-tax measurements (lemmas 1–3); system cost curves (latency, throughput,
   memory, divergence).
4. **The report** — verified literature review, methodology chapter, the four
   lemmas with proofs, three results chapters, statistical testing.
5. **Repo hygiene** — README, pinned versions, seed policy, one-command
   reproduction.
6. **Demo** — a live drift-and-recovery dashboard, as a viva prop.

---

## 15. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Hybrid does not beat naive baselines | Complete, publishable negative result; lemmas 1–3 still hold and are the contribution |
| A lemma's proof does not cleanly go through | Degrade to an empirical claim + simulation study; report honestly |
| System chapter overruns | Shrink Phase 3; Chapter 2 remains a complete chapter |
| Elec2 results contested | Pre-empt with no-change baseline + Kappa+ + exact preprocessing statement |
| Scope creep (the real killer) | §11 guards; anything new must pass "which research question does this serve?" |
| Teacher requires novelty beyond the hybrid | The hybrid *is* the novelty; if novelty is mandatory, its weight increases, not the project's size |

Every chapter has a survivable failure mode. The project cannot fail as a whole
unless it is abandoned — and the safeguards above exist to prevent exactly that.

---

*This synopsis is self-contained. A more detailed working document
(`idea/scope.md`) tracks open decisions, the full literature landscape, and
implementation notes.*
