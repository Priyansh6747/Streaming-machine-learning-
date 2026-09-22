# DriftDeck — drift monitoring that keeps working when your labels are late

**Product doc — turning the Delay Tax research into a product.**

> **One line:** DriftDeck is an ML observability platform that detects and *prices*
> model drift in production — and keeps doing so when the truth arrives **late,
> partially, or selectively**. Its engine is a hybrid detector (label-free early
> warning + delayed supervised confirmation), and its headline number is the
> **drift cost in business terms**, not an abstract accuracy drop.

*(Working name — the file and title rename in one edit if you want a different one.)*

---

## 1. TL;DR

| | |
|---|---|
| **What** | ML drift monitoring for production models whose labels are delayed, partial, or selective |
| **Who** | MLOps / ML platform teams running fraud, lending, ads, intrusion, or content models |
| **The gap** | Every drift tool assumes labels arrive with the prediction. In production they don't (chargebacks take 30–90 days). |
| **The wedge** | DriftDeck is **delayed-label-native** — it models the delay, not papers over it |
| **The engine** | Hybrid detector = label-free early warning + delayed supervised confirmation |
| **The metric** | **Damage area** — skill lost × time wrong, converted to money |
| **The moat** | Four formal lemmas (the Delay Tax) that let the product state *honest limits*, not marketing promises |
| **The build order** | Validate the lemmas → MVP on fraud → hosted |

---

## 2. The problem

A model is deployed. The world changes. The model silently gets worse. Two facts make
this expensive, and they compound:

**Fact 1 — drift is invisible to the people who own the model.** You can't see accuracy
in production without labels, and accuracy is exactly what's degrading. By the time a
quarterly report catches it, the loss has already happened.

**Fact 2 — the labels that would reveal drift arrive late, or never.** This is the Delay
Tax, and it is measured:

| Cost of late labels | Number | Source |
|---|---|---|
| Delayed labels raise user exposure | **+50%** at 81 (DDM) / 61 (EDDM) epochs of delay | Botacin & Gomes 2024 |
| Delayed labels cut detection | up to **−75%** | Botacin & Gomes 2024 |
| A 10,000-instance label delay shifts drift detection | by **~10,000 instances** (prequential accuracy pinned below 50%) | Gomes et al. 2022 |
| Batch retraining starves without labels | tuned batch wins under delay+rare-class (AUCPR **23.41 vs 14.23**) | Amekoe et al. 2024 |

The production stories are concrete: a fraud model approves a transaction *now* and
learns the truth when the chargeback lands in 30–90 days. An ad model serves now and
learns the conversion next week. An intrusion model flags now and gets the analyst
verdict later, or never. Labels arrive in overnight **batches**, **out of order**, and
**only for flagged cases** — which makes the observed errors a biased sample of true
errors.

**Why this is a product and not a feature:** the market's existing tools solve "detect
drift when labels are here." The harder problem — "keep watching when labels are *not*
here, and be honest about what you can and can't know yet" — is unsolved. That is the
product.

---

## 3. The wedge: what everyone else assumes away

The research's gap analysis (26-page review, 38 papers, `notes.pdf`) established the
wedge precisely. The delayed/partial-label literature is **fragmented** (five names for
one thing: verification latency, delayed labelling, label latency, extreme verification
latency, delayed partially labelled streams) and **unbenchmarked**: a standard benchmark
now exists for *immediate*-label drift detection (Cerqueira et al., KDD'26), but **none
exists for delayed, partial, or selective labels**.

Concretely, the competition falls into two camps:

| Camp | Examples | What they assume | The gap DriftDeck fills |
|---|---|---|---|
| Data/target drift monitors | Evidently, Arize, Fiddler, WhyLabs | Labels eventually available, or ignore the delay | Don't model *when* the truth arrives |
| Label-free performance estimators | **NannyML** (CBPE), Seldon Alibi Detect | Estimate accuracy *without* labels | Great — but they treat "no labels yet" as a state, not a *costed, bounded* timeline |
| **DriftDeck** | — | Labels are a **stochastic observation process** with latency | Models the delay, gives the earliest provable answer, prices the wait |

The differentiation is narrow but real, and it is exactly what the four lemmas give.

---

## 4. From research to product — the mapping

Every piece of the research becomes a product feature. This table is the conversion.

| Research artifact | Product feature |
|---|---|
| **Lemma 1 — reaction floor** (`T ≥ τ + L_τ`) | "Earliest you can know" — the product tells the customer, honestly, the fastest possible confirmation given their median label latency, and shows the early warning meanwhile |
| **Lemma 2 — power loss** (`1/√p`) | **Coverage dial** — "you label 10% of traffic, so my confidence is ~3.2× noisier" — a per-model sensitivity/confidence number |
| **Lemma 3 — batch-arrival inflation** (`1+(b−1)ρ`) | **False-alarm suppression** on batch-reconciled labels (overnight jobs) — fewer spurious pages |
| **Lemma 4 — identifiability ceiling** | **Honest limits** — "this drift type is invisible without labels; here's what I can and can't see" |
| **Hybrid detector** (Ch3) | The **core engine** — early warning (fast, may be false) + delayed confirmation (slow, reliable) |
| **Damage area / recovery time / dip depth** (Exp-01/02) | The **money metric** — drift priced as `Σ (baseline − actual) × cost-per-error` |
| **Prequential harness** (Ch1) | The **backtest/trust engine** — prove the detector on *your* data before you deploy it |
| **The system chapter** (Ch2) | The **deployment/scale story** — replica divergence, backpressure, restart cost |
| **Exp-01/02** (measured) | The **proof-of-concept demo** — a model that keeps learning heals itself; a frozen one doesn't |

---

## 5. The engine — the hybrid detector

The product's core. It is justified by construction, not by tuning:

- **Stage 1 — label-free early warning** (on `P(X)`, live): KS / MMD / Jensen–Shannon
  two-sample tests over a sliding window. Fast, catches drift immediately, **but also
  fires on harmless virtual drift**.
- **Stage 2 — supervised confirmation** (on the delayed error stream): join the late
  labels to the buffered predictions, form the error stream, run ADWIN / DDM / KSWIN.
  Reliable, **but bounded below by the reaction floor** (can't fire before labels land).
- **The hybrid logic**: a warning inside a tolerance window, then a confirmation → alarm.
  A warning that never confirms → revoked, with its false-alarm cost recorded.

**Why this is the information-maximal architecture:** Lemma 4 proves Stage 1 *cannot*
confirm (pure posterior drift is invisible to `P(X)`), and Lemma 1 proves Stage 2
*cannot* be early. So the hybrid is the only design that is both early and reliable
given the available information. That is a defensible engineering claim, not a slogan.

**Cheap to run:** the detectors are O(1) per sample — literature puts monitoring at
**<150 ms** and **<50 KB** of state per detector (Agu et al. 2026). DriftDeck can watch
thousands of models on one node.

---

## 6. The money metric — damage area → drift cost

Recovery time tells you *how long* the model was wrong. Damage area tells you *how much
skill × time* you lost:

```
damage = Σ (κ₀ − κ(t))   for t = drift → recovery      # in samples · Kappa
drift_cost = damage × cost_per_error(currency)        # the number the business sees
```

This is the product's "so what." A PM doesn't care about Cohen's Kappa. They care that
"the model was 0.173 κ below baseline for 12,000 transactions, costing ~$X." DriftDeck
reports the dollar number, with the Kappa behind it.

*(Honest note from the research: the damage-area metric itself has a truncation defect
— see `ML.md` §11.1 — that must be fixed before it ships. It is a known, bounded issue,
not a mystery.)*

---

## 7. Positioning vs competitors

| Tool | Drift detection | Label delay handled | Prices drift in money | Honest limits |
|---|---|---|---|---|
| Evidently | ✔ | ✖ (assumes labels) | partial | ✖ |
| Arize / Fiddler / WhyLabs | ✔ | ✖ | partial | ✖ |
| NannyML | ✔ (CBPE, label-free) | ✔ implicitly | partial | ✖ |
| Seldon Alibi Detect | ✔ | ✖ | ✖ | ✖ |
| **DriftDeck** | ✔ hybrid | ✔ **native** | ✔ **headline** | ✔ **lemmas** |

The honest position: DriftDeck does not win "who has the most charts." It wins the one
narrow, expensive case — **"my labels are late and I need to know how wrong I am right
now, and what it's costing me."**

---

## 8. Who it's for — and the killer demo

**Ideal customer profile:** a team with production ML models where ground truth is
intrinsically delayed or partial — fraud/payments (chargebacks 30–90 days), lending
(defaults), ad conversion (weeks), cybersecurity (analyst triage), insurance (claims).

**The killer demo is fraud**, because it is the canonical Delay Tax story and it is
*visible*: a fraud model approved a transaction; the chargeback lands 45 days later. A
fraud team already understands the lag. Show them, on their own stream, that DriftDeck
(a) warned within hours using input drift, (b) confirmed when the chargebacks landed,
(c) priced the accumulated loss — and that their current monitoring said nothing for 45
days. That demo closes.

---

## 9. Architecture

The product is an observability **sidecar/layer**, not a serving framework — it attaches
to any model (batch or streaming, yours or theirs). This is deliberately the Ch2
"systems" DNA: the hard part is not the math, it is the **two-stream join** and the
operational reality.

```
                        ┌──────────────────────────────┐
   your serving app ──► │  driftdeck-sdk               │  wraps model.predict()
                        │  emits (id, features, ŷ)     │
                        └──────────────┬───────────────┘
                                       │
                              ┌────────▼─────────┐
                              │  event backbone   │   Kafka / Redpanda  (Redis Streams for single-node)
                              │  ── inference ──  │   topic 1: live, every sample
                              │  ── labels ─────  │   topic 2: late, partial, selective
                              └────────┬─────────┘
                                       │
              ┌────────────────────────┼─────────────────────────┐
              │                        │                         │
        ┌─────▼─────┐           ┌──────▼──────┐           ┌──────▼──────┐
        │ stage-1    │           │ label-join   │           │ stage-2      │
        │ label-free │           │ + buffer     │           │ supervised   │
        │ early-warn │           │ (retain ≥    │           │ confirmation │
        │ on P(X)    │           │  max delay)  │           │ ADWIN/DDM/…  │
        └─────┬─────┘           └──────┬──────┘           └──────┬──────┘
              │                        │                         │
              └────────────►  hybrid detector  ◄──────────────────┘
                                       │
                          drift event: {type, confidence, delay, lemma-bounds}
                                       │
                     ┌─────────────────▼─────────────────┐
                     │  damage estimator                 │  rolling κ, dip depth,
                     │  → drift_cost = damage × $/error  │  recovery time
                     └─────────────────┬─────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
        ┌─────▼─────┐          ┌───────▼──────┐          ┌──────▼──────┐
        │  store     │          │  UI ("deck") │          │  alerts      │
        │ Prometheus │          │ live drift,  │          │ webhook/PD/  │
        │ + ClickHs  │          │ damage, lemmas│         │ retrain hook │
        └───────────┘          └──────────────┘          └─────────────┘

        ┌──────────────────────────────────────────────────────────────┐
        │  driftdeck-backtest  — offline replay harness (the trust engine)│
        └──────────────────────────────────────────────────────────────┘
```

### Components

| Component | Responsibility | Notes |
|---|---|---|
| `driftdeck-sdk` | Python SDK wrapping `predict()`; emits `(id, features, ŷ)`; also a client to submit late labels | Drop-in for sklearn/River/PyTorch serving |
| **Event backbone** | Kafka/Redpanda (scale) or Redis Streams (single-node) | Two topics: `inference`, `labels`. The delay lives in the label topic |
| **Label-join + buffer** | Joins late labels to predictions; retains predictions ≥ the max label delay | The two-stream join is the hard engineering, not the math |
| **Stage-1 detector** | Label-free early warning on `P(X)` | KS/MMD/Jensen–Shannon windows |
| **Stage-2 detector** | Supervised confirmation on the delayed error stream | ADWIN/DDM/EDDM/KSWIN/Page-Hinkley |
| **Hybrid logic** | Correlates warnings and confirmations; alarms, revokes, records false-alarm cost | Encodes Lemmas 1 & 4 |
| **Damage estimator** | Rolling κ, dip depth, recovery time, damage area → `$` | Configurable `cost_per_error` |
| **Store** | Prometheus (alerts) + ClickHouse/Timescale (per-model drift history, high cardinality) | Two-tier: real-time vs analytic |
| **UI — "the deck"** | Live drift timeline, damage in `$`, lemma bounds, per-model health | The demo dashboard |
| **Alerts** | Webhooks, PagerDuty, Slack; a **retrain/rollback trigger** (CI or model-registry webhook) | Closes the loop from *detect* to *act* |
| **Backtest harness** | Offline prequential replay of the customer's own stream | The trust engine + sales demo |

### The response half (optional, and where the research's adaptation side ships)

Detection is the core. But the research also measured **recovery** (Exp-01/02: a model
that keeps learning heals itself; a frozen one doesn't). So DriftDeck ships a
"respond" tier: on a confirmed drift, trigger retrain / rollback / model-registry
swap — or, for customers who want it, serve a **continually-learning model** that
self-adapts within a bounded window. That second tier is what Exp-01/02 actually
demonstrated.

---

## 10. Deployment model

| Tier | Shape | When |
|---|---|---|
| **MVP** | docker-compose, single node, Redis Streams | validate + first demo |
| **Scale** | k8s + Helm, Kafka/Redpanda, horizontal detector workers | multi-model, high QPS |
| **On-prem / edge** | single binary or sidecar per model | fraud/security customers with data-residency constraints |

Detector cost is the enabler: O(1) per sample, <50 KB state per detector, so a small
fleet of detector workers watches a large number of models. The expensive part is the
label-join retention buffer — sized by the customer's max label delay, not by traffic.

---

## 11. The build order (roadmap)

Ordered so the product de-risks itself *and* the academic claims de-risk themselves —
they are now the same thing.

| Phase | Build | Exit criterion | De-risks |
|---|---|---|---|
| **A — Validate** | Run the delayed-label experiments (the project's Exp-05): the four lemmas, on synthetic ground truth | The `1/√p`, reaction-floor, and batch-inflation curves hold empirically | The core engine is true, not assumed — and Alencar 2026 is read |
| **B — MVP** | SDK + detector core + label-join + damage estimator + minimal UI, on **one real dataset (CreditCard fraud)** | End-to-end: a live fraud stream, a late chargeback, a confirmed drift, a dollar figure | The killer demo |
| **C — Harden** | Alerts, retrain trigger, model-registry integrations, auth/tenancy, multi-model | A pilot customer runs it in shadow mode | Product–market fit signal |
| **D — Distribute** | Hosted SaaS + on-prem enterprise, OSS the SDK + detectors | Paying customer | Business |

The research's own roadmap (Exp-03 baseline ladder, Exp-04 real streams, Exp-05 delay
tax) *is* Phase A–B of the product. There is no wasted work — the academic experiments
become the product's proof and its backtest engine.

---

## 12. Business model (brief)

- **OSS core** — SDK + detectors + the hybrid logic. Builds trust and distribution in
  the MLOps community, and it is cheap to give away (the detectors are the research).
- **Paid** — hosted (multi-model, retention, alerting, SSO, compliance, support) and
  on-prem. **Per-model or per-event** pricing, not per-seat — the value scales with
  drift cost, not headcount.
- **The wedge in pricing:** the dollar "drift cost" the product reports *is* the ROI
  number that justifies the subscription. "You lost $X to drift last quarter; DriftDeck
  costs $Y."

---

## 13. Risks — honest

| Risk | Severity | Mitigation |
|---|---|---|
| **Crowded space** (Evidently, NannyML, Arize, Fiddler, WhyLabs, Seldon) | High | Differentiation is narrow-but-real (delay); go deep on fraud/security, not wide |
| **NannyML already does label-free performance estimation** | Medium-High | Differentiate on the *timeline* (earliest provable answer) + the dollar metric + honest limits, not on "we also estimate accuracy" |
| **Lemmas are unvalidated** — the core claim is unproven until Exp-05 | High | Phase A runs first; the lemmas are cheap to test and can degrade to empirical claims if a proof resists |
| **Alencar 2026 (DSP) may scoop the measurement** | Medium-High | Read it now; if it already has delay-vs-latency curves, the novelty shifts to the *formal bounds* + the *hybrid protocol* + the *product* |
| **Thin proof-of-concept** (one synthetic stream, no detector actually run) | Medium | Position Exp-01/02 as the mechanism demo; the product's proof is Phase A–B, not the two experiments |
| **Buyer is MLOps, not infra** | Medium | The author's infra background helps the build, not the sale — needs an MLOps-minded co-founder/early customer |
| **Data residency / compliance** (fraud, security) | Medium | On-prem/edge tier first-class, not an afterthought |

---

## 14. What the research already gives you

This is not starting from zero. The project hands the product:

1. **A pre-registered, reproducible proof of the mechanism** — Exp-01/02: a model that
   keeps learning recovers from drift in bounded time; a frozen one never does (0/11
   seeds). That is the "adaptation" half of the story, measured, with artifacts.
2. **The metric family** — damage area, recovery time, dip depth — the product's money
   metric is already defined and (partially) measured.
3. **The four lemmas** — a genuine, citable theory that becomes the "honest limits"
   feature and the marketing moat.
4. **The hybrid detector design** — justified by construction, ready to build.
5. **The prequential harness** — the future backtest/trust engine.
6. **The systems knowledge** — replica divergence, backpressure, restart cost — the
   deployment/scale story no ML competitor has.

The honest gap: the lemmas and the hybrid are **specified but not yet measured**. The
product's first move is to close that gap — and that move *is* the next experiment.

---

## 15. One-screen recap

**Product.** DriftDeck — ML observability that detects and prices model drift when
labels are late, partial, or selective.

**Why it exists.** Existing drift tools assume labels arrive with predictions; in
production they don't (chargebacks take 30–90 days), and the loss is invisible until it
is large.

**The wedge.** Delayed-label-native monitoring — model the delay, give the earliest
provable answer, price the wait.

**The engine.** Hybrid detector: label-free early warning (fast, may be false) +
delayed supervised confirmation (reliable, floor-bounded).

**The metric.** Damage area → drift cost in dollars.

**The moat.** Four lemmas (reaction floor, `1/√p`, batch inflation, identifiability)
that let the product state honest limits instead of marketing promises.

**The build order.** Validate the lemmas (Exp-05) → MVP on fraud → harden → distribute.

**The evidence.** Exp-01/02 prove the adaptation mechanism; the harness is the future
backtest engine; the research roadmap *is* the product's phase A–B.
