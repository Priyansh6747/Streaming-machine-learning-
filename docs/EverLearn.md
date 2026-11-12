# EverLearn — models that never stop learning

**Product doc — turning the streaming-learning research into a product.**

> **One line:** EverLearn is a model-serving layer where the model is never stale,
> because it learns from every prediction it makes — one sample at a time, in
> bounded memory, with no retraining pipeline.

*(Working name — rename in one edit if you want a different one.)*

---

## 1. TL;DR

| | |
|---|---|
| **What** | A self-learning serving layer: predict now, learn on the next label, forever |
| **Who** | Teams running high-velocity ML — fraud, ads, personalisation, recommendations, moderation |
| **The gap** | Every production model is frozen between retrains. The staleness window is where money leaks |
| **The wedge** | Real-time ML made the *features* real-time; the *model* never got there. EverLearn does |
| **The engine** | River / DeepRiver streaming learners — O(1) time and bounded memory per sample |
| **The metric** | **Staleness saved** = damage area avoided × cost-per-error, in dollars |
| **The moat** | A measured mechanism (Exp-01/02) plus governance that makes self-modifying models safe to deploy |
| **The build order** | Shadow mode on fraud → live on one stream → governance → platform |

---

## 2. The idea

Almost every ML product runs the same ritual: train a model, freeze it, deploy it,
watch it silently rot until someone reruns the pipeline. The model is a *finished
object* that is periodically rebuilt.

Streaming learning inverts that. The model is **never finished** — it is permanently
mid-training. Every sample that flows through it follows the same three steps:
**predict, score, learn, discard.** The label arrives (now, or later), the model takes
one bounded-memory update, and the world's most recent truth is already inside it.
There is no retrain event because there is nothing to retrain: the model is always
as current as the last sample it saw.

The product is that inversion, packaged. **EverLearn wraps any model and makes it
self-updating** — the customer stops maintaining a retraining pipeline and instead
owns a model that cannot go stale by construction.

This is deliberately *not* a drift-detection product. DriftDeck answers "is my model
wrong?"; EverLearn answers "make it stop being wrong — automatically." Detection and
adaptation are the two halves of the research; DriftDeck is the watch, EverLearn is
the act.

---

## 3. The problem — the staleness window

Between one retrain and the next, the world moves and the model doesn't. The model is
silently wrong for the whole window, and two costs compound inside it:

**Cost 1 — the decay itself.** Accuracy leaks continuously from the moment of
deployment. A weekly retrain means the model is, on average, *half a week stale* —
and that staleness is invisible to the team that owns it until someone measures it.

**Cost 2 — the retrain machinery.** The pipeline that rebuilds the model is itself a
product-scale investment: feature stores, training jobs, validation, registry, rollout,
rollback. MLOps teams spend a large share of their effort feeding a cycle that is,
structurally, always behind the data.

The research already priced cost 1. On a drifting stream, the frozen model never
recovered — its corrected `damage_ref` was **504.4 ± 104.1** samples·κ across 11
seeds, versus **70.6 ± 66.5** for the streaming model (**≈7.1×**), and a model that
retrains on a schedule can only heal at the tick, staying wrong for up to a full
interval by pure luck of phase. The staleness window is not an accident of bad ops;
it is the defining property of batch-trained models.

**Why this is a product and not a feature:** every "retrain faster" fix reduces the
window but keeps it — and keeps the machinery. Only a model that learns from the stream
itself eliminates both at once. That is the offer.

---

## 4. The wedge — features are real-time, models are frozen

The real-time ML market already proved the demand for lower staleness. Tecton, Chalk,
Fennel and friends made **features** real-time: sub-second freshness for the inputs to
the model. But the model those features feed is still batch-trained — retrained
nightly or weekly, deployed through a registry.

So the industry is currently real-time *until the last mile*, then frozen. A fraud
model consumes a feature computed 100 ms ago, but its weights were decided three weeks
ago. That last mile — **the model's own knowledge** — is the entire staleness that
remains, and nobody sells it as a product.

| Camp | Examples | What they made real-time | What EverLearn changes |
|---|---|---|---|
| Real-time feature platforms | Tecton, Chalk, Fennel, Featureform | Features (the inputs) | Nothing — EverLearn sits downstream of them |
| Serving + retrain platforms | Seldon, Vertex, SageMaker | Deployment, not learning | The retrain job disappears |
| Online-learning libraries | Vowpal Wabbit, River | The algorithm, as OSS | The operational product around it |
| **EverLearn** | — | **The model itself** | The model's knowledge updates with every sample |

The differentiation is one sentence: **everyone else made the inputs fresh and left
the model frozen. EverLearn makes the model fresh.**

---

## 5. What the customer actually buys

Not "streaming ML" — nobody buys a mechanism. They buy **the end of the staleness
window**, delivered as three things:

1. **The SDK.** Wrap any model: `EverLearn(model=your_model)` preserves the `.predict()`
   contract; the learning happens invisibly when each label lands. Drop-in for
   sklearn / River / PyTorch serving paths.
2. **The engine.** River / DeepRiver learners underneath — O(1) time and bounded memory
   per sample, so one node learns continuously at line rate instead of running batch
   jobs.
3. **The governance.** The honest reason teams don't ship self-modifying models is
   fear. EverLearn's controls exist to make that fear manageable: **shadow mode**
   (learn silently, measure, deploy nothing), **snapshots** (the model is a sequence
   of versions, rollback to any of them), **learning dials** (per-model learning rate,
   sample cap, what-not-to-learn rules), and an **audit log** (what the model learned,
   when, from what).

The headline metric the product reports: **staleness saved** — the damage area the
self-learning model avoided versus the customer's scheduled-retrain baseline, times
their cost-per-error. The research's damage-area metric becomes the customer's
dollar figure.

---

## 6. The evidence — the mechanism is already measured

The product's founding claim — *a model that keeps learning recovers; a frozen one
never does* — is not a pitch, it is Exp-01/02 plus the E01 metric correction:

| Fact | Number | Source |
|---|---|---|
| Frozen batch model recovered after drift | **0 / 11 seeds** | E01 |
| Streaming model recovered (real excursion) | **7 / 11 seeds**; the other 4 never left the band | E01 |
| Damage paid while wrong, corrected `damage_ref` | **70.6 ± 66.5** vs **504.4 ± 104.1** samples·κ — ≈ **7.1×** | E01 |
| Damage over the rest of the stream (`damage_full`) | **293.9** vs **3,408.1** samples·κ — ≈ **11.6×** | E01 |
| Worst streaming seed vs best frozen seed | **1.27×**, not 24× | E01 |
| Streaming self-heal on the pilot stream | back inside the pre-drift band **834 samples** after drift on seed 42; frozen fell 0.75 → 0.59 κ and never returned | Exp-01 |

The legacy **294×** mean and **24×** worst-vs-best figures were upper bounds from a
truncated integral and are retired. The durable claim is the 0/11 frozen recovery
and the corrected ≈7× mean damage gap.

The honest caveat, carried over from the research: this is one synthetic stream, two
models, no delayed labels, no real data. It proves the *mechanism*, not the market.
The product's first engineering phase exists to close exactly that gap — on the
customer's own stream, in shadow mode, where the proof doubles as the sales demo.

---

## 7. The two killer use cases

Streaming learning wins in two different places, and it is worth keeping them
distinct because they sell differently:

**A. Self-healing serving — the drop-in upgrade.** Fraud detection, ads, content
moderation: the model consumes real-time features (already built), but its weights go
stale. EverLearn sits behind the existing serving API; predictions are unchanged; the
model simply keeps learning from each confirmed label (chargeback, conversion, verdict)
as it arrives. The sale is "your current model, minus the decay between retrains."
The killer demo: run the customer's own stream through a frozen model and the
self-learning model side by side — the frozen one drifts off and stays off, the
EverLearn model dips and heals itself live (this demo *is* Exp-01, rerun on their data).

**B. Per-entity models — where streaming is the only option.** One lightweight model
per user, session, device, or seller, updated on every interaction, on-device or
server-side. Batch systems cannot build per-user models at all — they fake it with
embeddings and global retrains. Streaming learning makes per-entity models *trivial*:
a few hundred bytes of state per entity, one update per event, personalisation that
reacts to the user's last ten actions, not last month's global retrain. This use case
has no batch competitor, which makes it the cleanest market entry.

---

## 8. How it works mechanically

The loop is the prequential harness from the research, productised:

```
   request ──► predict_one(x) ──► serve ŷ            (unchanged, fast path)
                                      │
                         (label arrives — now or days later)
                                      │
             join (prediction ↔ label) via the event backbone
                                      │
             learn_one(x, y)          (O(1), bounded memory)
             metric.update(y, ŷ)      (rolling κ, damage area, staleness saved)
             snapshot policy          (version the model when skill moves)
```

Every sample is scored *before* it is learned — that is what makes the reported skill
honest — and nothing is stored. The infrastructure mirrors DriftDeck's: an SDK that
emits `(id, features, ŷ)`, an event backbone (Kafka at scale, Redis Streams for one
node) with a predictions topic and a labels topic, and a join buffer sized by the
customer's maximum label delay. The engineering hard part is the same two-stream join;
the learning itself is the cheap part.

---

## 9. The hard problems — honest

The idea is strong; these are the things that kill it if ignored:

| Problem | Severity | Honest status & mitigation |
|---|---|---|
| **The Delay Tax on the learner** — labels arrive late, so the model learns yesterday's truth, and drift is exactly when it matters most | High | This is the project's open problem. The product's first experiment is a delayed-label learner; until then EverLearn can only promise bounded staleness, not zero lag |
| **Trust & governance** — teams will not deploy a model that changes itself | High | Shadow mode first (measure, don't deploy), snapshots + rollback, audit log, per-model learning dials. Governance *is* the product, not an add-on |
| **Catastrophic forgetting** — bounded-memory continual learning degrades on long streams | Medium | Hybrid design: streaming core + periodic offline consolidation. Known solutions; needs engineering, not research |
| **Poisoned labels** — a model that learns from everything is exposed to bad labels | Medium | Learning dials (blocklists, confidence gates, anomaly fences) at the ingestion point |
| **Regulation** — some industries require frozen, validated models | Medium | Target the ones that don't (fraud, ads, personalisation, moderation); on-prem tier for the rest |
| **It is not a category** — nobody has budget for "streaming ML" | High | Sell the outcome: "never stale," "no retrain pipeline." The mechanism stays under the hood |

---

## 10. Positioning

| | Detects staleness | Removes staleness | Model keeps learning | Per-entity models |
|---|---|---|---|---|
| Retrain pipelines (the status quo) | ✖ | ✖ (only at the tick) | ✖ | ✖ |
| Real-time feature platforms | ✖ | ✖ | ✖ | ✖ |
| Drift monitors (incl. DriftDeck) | ✔ | ✖ (tells you to retrain) | ✖ | ✖ |
| Vowpal Wabbit / River (OSS) | — | ✔ | ✔ | ✔ |
| **EverLearn** | ✔ (built-in, as a by-product) | ✔ **by construction** | ✔ | ✔ **native** |

The honest position: EverLearn does not compete with drift monitors — it is the thing
they tell you to do, done automatically. The research's own pairing is the strategy:
**DriftDeck prices the decay, EverLearn deletes it.** A customer that buys one is one
sentence away from buying the other.

---

## 11. The build order

Ordered so the product de-risks itself *and* the research's next experiments become
the product's first milestones:

| Phase | Build | Exit criterion | De-risks |
|---|---|---|---|
| **A — Shadow** | SDK + engine + label-join + damage accounting, on one real stream (CreditCard fraud), shadow mode only | The self-learning model measurably beats the customer's scheduled-retrain baseline on staleness saved — without being deployed | The mechanism survives real data; the killer demo exists |
| **B — Live** | Snapshots, rollback, one pilot model learning in production | The pilot runs live with an owner who can revert it in one action | Governance under fire; first real dollars attached to staleness saved |
| **C — Platform** | Multi-model, per-entity models, alerts, integrations (registry, feature stores), auth/tenancy | Several models, one customer, zero retrain jobs left for them | Product–market fit signal |
| **D — Distribute** | Hosted + on-prem; OSS the SDK and engine | Paying customers | Business |

The research roadmap slots straight in: Exp-05 (label delay) is Phase A's core
engineering risk; Exp-03/04 (baseline ladder, real streams) are Phase A's shadow-mode
validation written in academic form. No wasted work — the experiments *are* the
product's first build phases.

---

## 12. Business model (brief)

- **OSS core** — the SDK and the engine. Cheap to give away (the algorithms are the
  research; River is already OSS) and it builds distribution in the MLOps community.
- **Paid** — the operational layer: the label-join infrastructure, snapshots and
  rollback, the audit log, multi-model management, SSO, support. Hosted or on-prem.
- **Pricing** — per model or per event, not per seat: the value scales with the
  staleness the customer avoids, and the "staleness saved" dollar figure the product
  reports is the ROI number that justifies the subscription.

---

## 13. One-screen recap

**Product.** EverLearn — a self-learning serving layer. Models that learn from every
prediction, in bounded memory, with no retraining pipeline.

**Why it exists.** Every production model is frozen between retrains, silently leaking
money through the staleness window, and the retrain machinery that fights it is itself
expensive and always behind.

**The wedge.** Real-time ML made features real-time; EverLearn makes the model's own
knowledge real-time. The last mile nobody sells.

**The offer.** Drop-in SDK, River/DeepRiver engine, and governance (shadow mode,
snapshots, rollback, audit) that makes self-modifying models safe to deploy.

**The metric.** Staleness saved — damage area avoided × cost-per-error, in dollars.

**The evidence.** E01: a streaming model self-heals after drift (7/11 seeds with a real
excursion; the other 4 never left the band); a frozen model never does (0/11), paying
**≈7.1×** more corrected `damage_ref`. The mechanism is measured, not promised.

**The honest gaps.** Delayed labels weaken the learner exactly when it matters (the
Delay Tax — the project's open problem), and self-modifying models need governance
before any team will ship them. Both are Phase A.

**The build order.** Shadow mode on fraud → live pilot → platform → distribute — with
the research's own experiments as the first milestones.
