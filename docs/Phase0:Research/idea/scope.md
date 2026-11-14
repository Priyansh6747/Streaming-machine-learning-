# Scope — The Delay Tax

**Working title:** *The Delay Tax: drift detection and adaptation when labels are
late, partial, and biased.*

**Status:** scope locked — pending only the teacher-dependent questions in §12.
This document supersedes `notes/08-scope.md` (kept as the draft it was). It
consolidates the research phase (now complete) into a single reference.

> **Thesis spine:**
> *Streaming ML works in simulation. Does it work in production?
> Build the system, find where it breaks, fix one thing properly.*
>
> **The one thing we fix:** production labels arrive **late, partially, and
> selectively** — and the field's detectors, metrics, and evaluation protocol
> quietly assume they arrive **instantly, completely, and unbiased**.

---

## 1. The problem in one page

Streaming ML rests on a hidden assumption:

> Sample `x` arrives, and its true label `y` arrives **with it**. The model
> predicts, is immediately told whether it was right, and learns.

That assumption is false in every real deployment:

- Fraud: a transaction is approved *now*; the chargeback lands **30–90 days later**.
- Ads: an impression is served *now*; the conversion arrives **next week**.
- Intrusion: a login is flagged *now*; the analyst verdict comes **later, or never**.

Follow the consequences, because they cascade:

1. **Detection is late.** Drift detectors watch the *error stream*. If labels are
   late, the error stream is late, so **drift is detected only after the labels
   catch up** — meanwhile the model is silently wrong.
2. **Evaluation breaks first.** Prequential scoring (`predict → score → learn`)
   requires the label in the same tick. Under delay you **cannot even measure your
   own accuracy in real time**, so the field's own instrument for success fails in
   the setting the field claims to serve.
3. **Adaptation compounds the error.** While labels are pending, the model keeps
   learning *stale* samples — after a drift, it is actively reinforcing the
   pre-drift concept.
4. **The label process itself misbehaves.** Labels arrive in bursts, out of order,
   or only for flagged cases. That injects autocorrelation and selection bias into
   the error stream — which detectors misread as drift (false alarms) or miss
   entirely.

The research phase (done — §2) confirms this is a **real, under-populated gap**,
not a tutorial topic.

---

## 2. What the research established

Findings from `research/01-landscape.md` and `research/02-delayed-labels.md`.

**The gap is real and open.** The delayed/partial-label literature is fragmented —
terminology alone spans *verification latency, delayed labelling, label latency,
extreme verification latency, partially labelled streams* — and no widely accepted
benchmark or protocol exists. The most relevant syntheses are Gomes et al. (2022,
*ACM CSUR* survey on delayed partially labelled streams) and a dedicated 2026
experimental survey in *Digital Signal Processing*. The area is **active but not
converged**: no canonical problem formulation, no dominant method.

**Well-established (safe to build on):**

- Prequential evaluation is the right protocol for ordered streams; random
  train/test splits are invalid (temporal leakage).
- No drift detector dominates. Delay / false-alarm / missed-detection trade-offs
  depend on drift type, noise, imbalance, and configuration.
- ARF, SRP, Leveraging Bagging, HAT are the strong adaptive baselines;
  Hoeffding Tree is the canonical static one.
- Synthetic generators give ground-truth drift → detection delay is computable;
  real streams give external validity but unknown drift positions.
- Elec2 has known temporal-autocorrelation problems (a "no change" baseline and
  Kappa+ must be reported; accuracy alone overstates learning).

**The specific holes this project targets:**

- **Detector-level degradation under delay is unquantified.** There is no clean
  curve of *detection delay vs label latency*, or *false-alarm rate vs label
  coverage*. Existing work studies classifiers or active-learning systems, not the
  detector in isolation.
- **No standard confirmation protocol** combining a label-free early warning with
  delayed supervised confirmation. Research names this architecture explicitly as
  a promising, unbuilt direction.
- **The operational cost of continual learning is under-measured.** Papers rarely
  report the latency, memory, and divergence costs of adaptation — the field is
  populated by ML researchers, not infrastructure people. This is our home turf.

**The bet, confirmed:** the two strongest gaps (delayed labels, operational cost)
both lie exactly where the field's standard assumptions diverge most sharply from
production reality — and neither requires deriving novel algorithms.

---

## 3. Decisions locked

| Decision | Choice | Why |
|---|---|---|
| Direction | **Systems + delayed labels** (Notes 05 Option C → 08) | The bigger question; 70% infrastructure; novel |
| Data | **Synthetic + real**, in that order | Ground truth first, external validity second |
| Task | **Binary + multiclass classification only** | Focus; imbalance is a secondary axis, not a task |
| Tool | **River**, plus our own harness | It is the instrument, not the subject |
| Deployment | **docker-compose, single node** | Sufficient; k8s is scope creep |
| Adaptation strategy | Detector-triggered **and** passive (both measured) | The active-vs-passive question is a real finding |
| Headline contribution | **The hybrid detector + the delay-tax lemmas (§6)** | Novel, measurable, failsafe (negative result is a result) |
| Real datasets | **≤ 3** (Elec2, Phishing, + CreditCard) | Elec2 caveats reported; enough breadth, no more |
| Reproducibility | ≥ 10 seeds, fixed budgets, exact versions recorded | Non-negotiable for a defensible project |

---

## 4. Research questions

Three questions, one spine. Each is falsifiable and measurable.

- **RQ1 (baseline, guaranteed).** Under controlled synthetic drift — sudden,
  gradual, incremental, recurring — how do ADWIN, DDM, EDDM, Page-Hinkley, KSWIN
  compare on detection delay, false-alarm rate, and missed detections, and how do
  adaptive models (HAT, ARF) recover compared to static ones (Hoeffding Tree,
  Naive Bayes)? *This is the control condition: "what the literature says happens."*

- **RQ2 (system, novel-by-measurement).** What does running continual learning as
  a real service cost — p50/p99 prediction latency during adaptation, throughput
  ceiling, memory growth, checkpoint/restart cost, and **how fast do scaled-out
  replicas diverge**? Where does it break first?

- **RQ3 (contribution, the gap).** By how much does delayed, partial, and
  selective labelling degrade drift detection and adaptation — exactly — and can a
  **hybrid detector** (label-free early warning + delayed supervised confirmation)
  recover the immediate-label baseline, or come provably close?

The math (§6) turns RQ3 from "degrades a lot" into **four specific, provable
claims** about *how* and *by what factor* it degrades.

---

## 5. The three chapters

### Chapter 1 — The baseline (safe)

The condensed Options A+B from `notes/05`. One experiment harness, one run matrix:

| Axis | Values |
|---|---|
| Drift type | sudden (SEA), incremental (Hyperplane), gradual (RandomRBF), recurring (ConceptDriftStream composition) |
| Detector | ADWIN, DDM, EDDM, Page-Hinkley, KSWIN |
| Model | Hoeffding Tree (static), HAT (adaptive), ARF (ensemble), Logistic Regression (blind/passive) |
| Metric | detection delay, FP rate, missed detections, MTFA/MTD, dip depth, recovery time, rolling accuracy |
| Rigour | ≥ 10 seeds, fixed stream length (~100k), fixed drift positions, mean ± std, Friedman + Nemenyi |

**Deliverable:** the reusable harness + the idealised numbers every later chapter
measures reality against.

### Chapter 2 — The system (infrastructure spine)

A real, containerised, reproducible continual-learning service:

- Event source: **Redis Streams** (zero-JVM friction) or Kafka/Redpanda — decided
  in Phase 2, pinned early.
- Model service: `POST /predict` + a learning consumer; **the same model that
  serves is the model that learns**, one pass.
- Live drift monitor (a real service, not an offline script).
- Prometheus metrics + Grafana dashboard; load generator to push it until it breaks.
- Versioned, seeded, reproducible end to end (docker-compose).

**Deliverable:** the running system + the "reality tax" chapter — the operational
cost the literature never reports: adaptation latency cost, replica divergence,
backpressure behaviour (drop learning or drop serving?), restart/recovery cost.

### Chapter 3 — The delayed-label contribution (the research)

The core novel work, built on Chapter 1's harness:

1. **Delay regimes** — fixed (`d = 10/100/1000`), stochastic (geometric mean `d`),
   partial (label probability `p ∈ {1, .5, .1}`), selective (only "flagged" samples
   labelled — the endogenous-bias case).
2. **Measure the degradation** — detector delay/FP/FN and model recovery vs. the
   `d = 0` control, at detector level and downstream accuracy level.
3. **Build the hybrid** — label-free early warning on the input stream `P(X)`,
   supervised confirmation on the delayed error stream, with an explicit
   confirmation protocol (warning within tolerance window, confirm on labels).
4. **Evaluate** — hybrid vs. naive-delayed-supervised vs. label-free-only vs.
   immediate-label control; test the §6 lemmas empirically.

**Fallback (safe):** if the hybrid does not beat the naive baselines, that is a
publishable negative result — "we quantified how much delayed labels break
existing methods and why the obvious fix is insufficient." Chapter 3 **cannot
fail**, which is why it is the bet.

---

## 6. The mathematical angle — *the delay tax*

The project's theory. Deliberately **applied** math — probability, concentration
bounds, hypothesis testing — squarely within reach, and every claim is
**empirically verifiable** in Chapter 3. This is not a bet on novel theory; it is
a bet on formalising the obvious correctly and measuring the constants.

### 6.1 The formal setting

Let the concept be `C_t ∈ {C₀, C₁}`, with a real drift at time `τ`. The model
predicts `ŷ_t`; the instantaneous error is

```
e_t = 1[ŷ_t ≠ y_t].
```

Under immediate labelling a detector observes `(e_t)` in real time and alarms at a
stopping time `T` — the classical **quickest detection** problem (Page 1954;
Lorden 1971 minimax; Shiryaev). The field's "detection delay vs false-alarm
trade-off" is the empirical image of Lorden's criterion: minimise worst-case delay
subject to a false-alarm constraint (average run length).

Under label delay, each error `e_t` becomes observable only at `t + L_t`, for a
latency process `L_t` (fixed, stochastic, or driven by a review policy). The
detector's filtration is

```
F_t = σ( e_s : s + L_s ≤ t ).
```

This is *quickest detection with delayed observations* — a variant the literature
has barely touched. Four claims follow, each provable with undergraduate/early-grad
probability and each with a Phase-3 experiment attached.

### 6.2 Lemma 1 — The reaction floor

> Any detector that observes only the error stream cannot alarm before the drift's
> labels arrive: `T ≥ τ + L_τ` pointwise, hence `E[T] ≥ τ + E[L]`.

*Proof idea:* `e_τ ∉ F_t` for `t < τ + L_τ`; the alarm is a stopping time w.r.t.
`F_t`. Trivial — and that triviality is the point: it is a **hard lower bound on
detection delay that no algorithm can beat**, independent of how good the detector
is. It converts "you detect drift a month late" from a complaint into a theorem.

**Experiment:** for each delay regime, measure detection delay of every detector
and confirm it never dips below `d` (fixed case) / `E[L]` (stochastic case).

### 6.3 Lemma 2 — The power loss (effective sample size)

> Under stochastic per-sample delay, a windowed detector's confidence band widens
> by the factor `1/√p`, where `p` is the fraction of the window's labels that have
> arrived — equivalently, its minimum detectable mean shift grows from
> `Δ_min ∝ 1/√W` to `Δ_min ∝ 1/√(pW)`.

*Proof idea:* the observed error stream is a randomly thinned version of the true
one; a window of `W` arrivals contains only ~`pW` distinct fresh errors, and any
mean-estimate test (Hoeffding/CLT based — ADWIN's threshold is a constant times
`√(ln(1/δ)/W)`) has standard error `σ/√n` with `n = pW`. Delay degrades
sensitivity by the factor `1/√p` — a clean, checkable constant.

**Experiment:** for `p ∈ {1, .5, .25, .1}` and a fixed drift magnitude, measure
ADWIN's detection delay; verify it tracks the predicted `1/√p` inflation.

### 6.4 Lemma 3 — False-alarm inflation under batch arrival

> If labels arrive in batches of size `b`, the observed error stream inherits
> intra-batch correlation `ρ > 0`, inflating the variance of the estimated error
> rate by `1 + (b−1)ρ` (the standard clustered-data factor). Detectors calibrated
> for i.i.d. errors then fire spuriously: DDM's alarm rate rises above its nominal
> `α`.

*Proof idea:* the variance of a clustered sample mean; DDM's thresholds assume
`e_t` independent Bernoulli. Batch arrival is exactly what daily/weekly label
reconciliation does in production.

**Experiment:** feed detectors a delayed-error stream under batch arrival with no
actual drift; measure FP rate vs. `b`.

### 6.5 Lemma 4 — The identifiability ceiling (why the hybrid is maximal)

> If `P(X)` is unchanged but `P(Y|X)` differs between concepts (pure posterior
> drift), no label-free detector has any power: the likelihood ratio of the
> `X`-observations is identically 1, so the KL divergence between the two
> hypotheses is 0.

*Proof idea:* the `X`-marginal cannot distinguish joints that share it. This is
the formal content of Žliobaitė's detectability result (2010) — and it **justifies
the hybrid by construction**: label-free signals can only be an early warning;
confirmation requires labels. The hybrid is the information-maximal architecture
available, which is the strongest design statement a scoping document can make.

**Experiment:** construct a generator with pure posterior drift (e.g. flip
`P(y|x)` in a region while keeping the marginal fixed); confirm label-free
detectors miss it while the hybrid's delayed confirmation catches it.

### 6.6 What the math is *for*

- It gives the report a theory section with **four lemmas it can actually prove**,
  each with an empirical counterpart — rigorous, but no dependence on novel
  analysis (a systems-strength fit).
- It converts Chapter 3's headline from a table of numbers into a narrative:
  *detection is bounded below by the reaction floor, weakened by `1/√p`, distorted
  by batch autocorrelation, and capped by identifiability — and the hybrid is the
  best possible response.*

### 6.7 Explicitly *not* in scope (math)

- No new change-detection algorithms, no regret bounds, no online convex
  optimisation, no statistical learning theory, no proofs of novel theorems
  beyond the four lemmas above. If a lemma resists a clean proof, it degrades to
  an empirical claim — the project survives.

---

## 7. Datasets

| Kind | Data | Role |
|---|---|---|
| Synthetic | SEA, Hyperplane, RandomRBF, STAGGER, ConceptDriftStream compositions | Ground-truth drift positions → detector metrics, lemmas 1–4 |
| Real | Elec2, Phishing, CreditCard (≤3) | External validity; Elec2 with no-change baseline + Kappa+; CreditCard gives a natural delayed-label (fraud) story |
| Real, optional | A second fraud/intrusion stream if CreditCard proves too small | Only if time allows — not planned |

---

## 8. Deliverables

1. **Experiment harness** — config-driven, seeded, reusable; one runner for
   Chapters 1 and 3.
2. **The running system** — docker-compose stack: stream, model service, live
   drift monitor, Prometheus/Grafana, load generator.
3. **Results** — accuracy-over-time curves; delay/FP/FN trade-off tables;
   the delay-tax measurements (lemmas 1–3); system cost curves (latency,
   throughput, memory, divergence).
4. **The report** — literature review (citations verified), methodology chapter,
   the four lemmas with proofs, three results chapters, statistical testing.
5. **Repo hygiene** — README, pinned versions, seed policy, one-command
   reproduction.
6. **Demo** — the live drift-and-recovery dashboard as a viva prop.

---

## 9. Scope guards

| ❌ Out of scope | ✅ Instead |
|---|---|
| Deep learning as the benchmark classifier | Deep models are permitted **only as label-free signal extractors** for the hybrid detector (DF-H); the benchmark host stays classical (Hoeffding Tree / ARF / HAT) |
| Regression, clustering, recsys | **Classification only** |
| Distributed *training* (parameter servers, federated) | Single-node service; replica *divergence measurement* is in scope |
| Building a streaming framework | Use River; we build the *harness and system around it* |
| k8s / multi-cloud production deploy | docker-compose on one machine |
| > 3 real datasets | Exactly 3, chosen and justified |
| Novel adaptation algorithms | Hybrid *architecture* + formalised delay model + measurement |
| Novel theorem-proving beyond the four lemmas | Lemmas; anything harder degrades to empirical |

The single most common way student projects fail is scope creep, not difficulty.

> **Scope amendment (2026-11-14):** DF-H uses a deep encoder as a **label-free
> feature monitor**, not as the benchmark classifier. The classical host remains
> the measured learner. This implements the hybrid architecture in §6.5 and
> keeps the classical baseline comparison intact. If the deep monitor is
> disallowed, the same fusion runs on a classical PCA/KS feature monitor.

---

## 10. Milestones (~10–15 h/week, alongside coursework)

| Phase | Weeks | Work | Exit criterion |
|---|---|---|---|
| 0 — Orientation | 1–4 | *Mostly done.* Verify the 10–15 citations we'll actually cite; get River running; reproduce one published baseline | A baseline number reproduced, bibliography skeleton |
| 1 — Baseline | 5–12 | Harness + Chapter 1 run matrix | Harness reusable; idealised numbers done |
| 2 — System | 13–24 | Stream, model service, live monitor, metrics, load gen | Running docker-compose stack |
| 3 — Break it | 25–32 | Latency/throughput/memory/divergence/restart measurements | "Reality tax" chapter data |
| 4 — Delay tax | 33–44 | Delay regimes → degradation curves → hybrid → lemma verification | The contribution (or the negative result) |
| 5 — Write-up | 45–52 | Report, figures, stats, README, demo | Submittable project |

Phase 2 and 3 can shrink if the year tightens; Phase 4 is the bet and keeps its
budget. Phase 1 is the control condition everything depends on — it goes first.

---

## 11. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Hybrid doesn't beat naive baselines (Ch 3) | Complete, publishable negative result; lemmas 1–3 still hold and are still the contribution |
| Lemma 2 or 3 proof doesn't cleanly go through | Degrade to empirical claim + simulation study; report honestly |
| Phase 2 (system) overruns | Shrink Phase 3; Chapter 2 is still a complete chapter |
| Elec2 results are contested | Pre-empt with no-change baseline + Kappa+ + exact preprocessing statement |
| Scope creep (the real killer) | §9 guards; anything new must pass the "which research question does this serve?" test |

Every chapter has a survivable failure mode; no single bet can sink the project.

---

## 12. Open decisions (teacher-dependent — must be asked)

- Does the teacher require **novelty**, or is rigorous measurement/comparison
  acceptable? (This decides how hard Chapter 3 has to land.)
- **Report only, or report + working system?** (This scope assumes both; a
  report-only variant drops Phase 2/3 to optional.)
- Required **format/length** (IEEE two-column?), interim milestones, individual
  or group, deadline.

These change *packaging*, not *substance*. The spine survives any answer.

---

## 13. Still to do before implementation

- [ ] Verify the citations we will actually use (Google Scholar / Semantic Scholar —
      every one; the research files are explicitly unverified).
- [ ] Ask the teacher the §12 questions.
- [ ] Decide the event-source choice (Redis Streams vs Kafka/Redpanda) in Phase 2.
- [ ] Pin River version; reproduce one published baseline number (Phase 0 exit).
- [ ] Optionally run research prompts 3–5 from
      `research/perplexity-prompts.md` to sharpen Chapters 2 and the benchmark
      methodology — useful, not blocking (the notes already cover the essentials).
