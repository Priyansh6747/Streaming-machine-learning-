# EverRank — per-user personalization as a drop-in reranking layer

**Product doc — EverLearn's flagship application: a personalization layer that sits
between any recommendation system's ranking and its serving.**

> **One line:** EverRank is a drop-in reranking layer. Your existing recsys keeps
> generating candidates; EverRank reorders them with one tiny streaming model per
> user — and per session — so the feed reacts to *you*, not your demographic.

*(Working name — rename in one edit if you want a different one.)*

---

## 1. TL;DR

| | |
|---|---|
| **What** | A reranking layer on top of any existing recommendation system |
| **Who** | E-commerce, feeds, streaming, ads — anyone with a ranked list and a user |
| **The gap** | Personalization stops at the global model: one model serves everyone, retrained monthly |
| **The wedge** | **Zero-risk adoption** — the layer is a pass-through at worst, so a customer can try it without betting their pipeline |
| **The engine** | Per-user + per-session streaming adaptors (KB-scale), learned one interaction at a time, IPS-corrected |
| **The metric** | Per-user lift in CTR/conversion, and **drift recovery time** — how many interactions until the feed notices your taste changed |
| **The moat** | The measured self-healing mechanism (Exp-01/02) applied at the level of a single user |
| **The build order** | Shadow rerank on one feed → live A/B → session layer → platform |

---

## 2. The idea

Modern recommendation systems are already staged: **retrieval → ranking → reranking**.
Retrieval finds a few hundred candidates, ranking scores them with the global model,
reranking does a final polish before serving.

EverRank inserts itself at that last seam — the one every real system already has. The
customer's retrieval and ranking keep working untouched; EverRank takes their top-K
and reorders it:

```
existing system:  retrieval ──► global ranking ──► [EverRank] ──► feed
                    (theirs)      (theirs)          (ours)
```

The layer shape is not a compromise — it is the product. Three properties fall out of
it, each of which is a hard-won answer to a classic objection:

- **Zero-risk adoption.** At worst the layer reproduces the incumbent's ranking, so
  the customer's downside is "exactly what I have today." No data migration, no
  rip-and-replace, no trust leap. Integration is one API call between their ranking
  and their serving.
- **The echo chamber is solved upstream.** The candidate list still comes from the
  global system — trending items, new inventory, collaborative "people like you"
  — because EverRank can only reorder what it is given. Explore and exploit map
  cleanly onto the two layers: *upstream explores, EverRank exploits.* The feed
  cannot become a filter bubble because the layer cannot narrow the candidate pool.
- **Cold start degrades gracefully.** A new user has no data, the adaptor sits at its
  prior, and the layer reproduces the global ranking. EverRank only diverges from the
  incumbent when the user gives it evidence — a *provable* no-harm property, and the
  sentence that closes a sale.

---

## 3. The problem — personalization stopped at the last mile

Every mainstream recsys is one global model plus per-user *state*: your embedding,
your last-fifty-items history, your bandit posteriors. The state is personal; the
function mapping state → score is shared by everyone. The consequences:

- **You get the feed of "people like you," never yours.** The global model learns the
  average of its users. Quirks, niche tastes, and the one weird thing you love are
  noise to it.
- **One user's taste drift is invisible.** When your life changes — new city, new baby,
  new hobby — that shift is a drop in the aggregate. A monthly retrain cannot see it,
  and the feed quietly rots for weeks.
- **The retrain cycle is always behind.** Between retrains the model is stale by
  construction, and "retrain more often" only shrinks the window — it keeps the
  machinery and the delay.

The customer's feed therefore reacts to the *segment*, not the person. EverRank's
answer is the obvious one: stop learning one model for everyone and start learning one
tiny model per person, online, all the time.

---

## 4. The architecture — global, user, session

Three levels, each with a different learning clock:

```
global model          (not ours)   items, trends, cold start — retrained monthly
  └─ per-user adaptor (ours)       slow taste — weeks — KB-scale, one step per interaction
       └─ per-session adaptor (ours)  current intent — seconds — ephemeral
```

**Per-user adaptor.** A linear probe on the incumbent's features or embeddings —
a vector of a few thousand numbers, a few KB. Each interaction is one gradient step.
With 10M users × ~5 KB the whole fleet is tens of GB: object storage plus an LRU hot
cache. Small capacity is not a compromise — a user hands you hundreds of samples, and
a model bigger than this would have nothing to learn from. The tininess is forced by
the data, and it is the right answer.

**Per-session adaptor.** The strongest predictor of your next action is your last five.
Within a session the feedback is dense and fast (clicks in seconds, not purchases in
weeks), so a per-session layer — conditioned on context, reset when the session ends —
is where the "current intent" lives. This is the level where engagement platforms
actually win, and it is pure streaming: you cannot retrain a batch model mid-session.

**On "every mood."** Users are multi-intent (browsing for self vs. gifts, work vs.
leisure, 1am doomscroll vs. morning utility), but *detecting* mood is the wrong frame
— it fragments an already-sparse signal. The right version is conditioning on
observable context: time of day, device, current category, query terms, session
length. The model learns "this user at 1am on a phone" as an implicit intent state,
no mood labels required.

**The loop, with the honesty built in:**

```
serve reorder ──► observe (click: seconds · purchase: days · churn: weeks)
                        │
                        └─► learn_one(x, y), IPS-weighted
```

The delayed, partial labels here *are* the research's Delay Tax, per user — and the
IPS weighting is non-negotiable: a reranker only sees clicks on what it chose to show,
so its own policy biases its labels. An uncorrected online learner amplifies its own
mistakes; this is how reranking layers silently fail in production. EverRank learns
counterfactually (inverse propensity scoring, maintained online) from day one.

---

## 5. Why the research is the engine

This product is the project's findings applied at the level of a single user:

| Research result | EverRank feature |
|---|---|
| **E01 — streaming self-heals, frozen never does** (frozen 0/11; real streaming recovery 7/11; ≈7.1× less corrected `damage_ref`) | The per-user adaptor recovers within a bounded number of interactions when the user's taste changes; the incumbent's monthly retrain literally cannot see one user's drift in the aggregate |
| **`damage_ref` / recovery rate** | The product's headline metric: interactions × skill lost *for this user*, while their feed was stale |
| **The Delay Tax** | Clicks land in seconds, purchases in days, churn in weeks — the adaptor learns on yesterday's truth, and the honest promise is "bounded staleness," not zero lag |
| **Selective/biased labels literature** | The IPS-corrected learner — feedback is conditioned on what EverRank itself chose to show |
| **Prequential harness** | The shadow-rerank engine: replay the customer's own interactions, prove per-user lift offline before a live A/B |

The Exp-02 story, told as a product: the frozen model is the customer's feed today;
the streaming model is EverRank; the drift is a user whose life changed. One heals,
the other never notices.

---

## 6. The honest gotchas

| Problem | Severity | Status & mitigation |
|---|---|---|
| **Position bias / feedback loop** — the reranker shapes its own labels | High | IPS online from day one; treated as load-bearing, not an add-on |
| **Lift concentrates on active users** — the long tail stays on the global model | Medium | By design: active users drive the revenue. The pitch says it plainly |
| **Per-user evaluation & governance** — global NDCG hides that 3% of users got worse | Medium | Per-user metrics, per-user rollback, versioned snapshots — EverLearn's governance, scaled to millions of models |
| **Per-mood fragmentation** — one model per intent state shatters the data | Medium | Context conditioning, not mood detection; one adaptor per user, context as features |
| **Depth ceiling** — with ~100 samples per user the adaptor learns coarse taste, not semantics | Low | The global tower carries the semantics; EverRank is the last mile |

---

## 7. Positioning

| | Reranks your list | Per-user model | Learns online | Notices per-user taste drift |
|---|---|---|---|---|
| The incumbent (your customer's system) | — | ✖ (per-user *state*, one model) | ✖ | ✖ |
| Algolia / Constructor / Recombee / Amazon Personalize | ✔ | ✖ (one global LTR model) | ✖ (periodic retrain) | ✖ |
| TikTok's engagement engine | ✔ (in-house) | ✖ (global, ultra-fast retrain) | partial | partial |
| **EverRank** | ✔ | ✔ | ✔ **every interaction** | ✔ **headline feature** |

One-liner for the sales deck: **their reranker is one model for all your users,
retrained monthly; ours is one model per user, updated every interaction — and it
notices when that user changed.**

And the ecosystem note: EverLearn is the engine and the governance, EverRank is the
flagship application, DriftDeck can watch the *global* model upstream (and, one day,
the fleet of per-user adaptors). Three docs, one story: detect the decay, price it,
delete it — per user.

---

## 8. The killer demo

Two scenes, both runnable on the customer's own logged data:

**Scene 1 — the drift.** Find a user whose behavior changed on a real date (moved
city, new baby, new category of purchases). Replay: the incumbent's monthly-retrained
feed stays stale for weeks after that date; EverRank's adaptor pivots within a bounded
number of interactions. Show the damage area as engagement lost, priced in churn risk.
This is Exp-02, rerun on one real user.

**Scene 2 — the session.** Replay a single session. Watch the feed pivot on the last
five actions: the user was shopping for themselves, then searched a gift category, and
the incumbent's session-blind ranking keeps serving the old intent for the rest of the
evening while EverRank's session layer has already turned. No drift needed — just
speed. This is the demo that makes product teams feel the gap.

---

## 9. The build order

| Phase | Build | Exit criterion | De-risks |
|---|---|---|---|
| **A — Shadow rerank** | Offline replay of a real feed: global ranking vs EverRank's per-user adaptor, IPS-corrected, on held-out interactions | Per-user lift vs incumbent on real data — without a single live request | The mechanism survives real data; the demo exists |
| **B — Live A/B** | The layer in production, same candidates, two orderings; per-user rollback and snapshots | Statistically honest lift in CTR/conversion; rollback proven in anger | Governance under fire; first dollar value attached |
| **C — Session layer** | Per-session adaptor on top, context conditioning | Measurable mid-session pivot lift | The second, defensible moat (speed, not just drift) |
| **D — Platform** | Multi-customer, on-prem tier, integrations with the major recsys stacks | Paying customers | Business |

The research roadmap slots straight in: Exp-05 (delayed labels) is Phase A's core
learning risk, and the selective-labels literature already in the review is Phase A's
IPS design. The experiments *are* the product's first milestones.

---

## 10. Business model (brief)

- **Pricing per request or per lift** — the layer's value scales with traffic and
  with the engagement it recovers, not with headcount.
- **The ROI number is built in** — per-user lift × traffic = recovered engagement,
  reported by the product itself, and that figure is the subscription justification.
- **Land-and-expand** — the zero-risk layer is the wedge into an account that already
  runs a recsys; EverLearn's governance platform and DriftDeck's monitoring are the
  expansion.

---

## 11. One-screen recap

**Product.** EverRank — a drop-in reranking layer. The customer's recsys generates the
candidates; EverRank reorders them with one tiny streaming model per user, and per
session.

**Why it exists.** Personalization stops at the global model: one model for everyone,
retrained monthly, so you get the feed of "people like you" — and it can't notice when
*your* taste changed.

**The wedge.** Zero-risk adoption: worst case the layer is a pass-through, integration
is one API call, and with no user data it provably reproduces the incumbent's ranking.

**The architecture.** Global (theirs, explores) → per-user adaptor (slow taste, KB-scale,
learns online) → per-session adaptor (current intent, seconds). Echo chamber solved by
construction: the layer can only reorder, never narrow.

**The metric.** Per-user lift in CTR/conversion, plus drift recovery time — how many
interactions until the feed notices your life changed.

**The honest parts.** IPS-corrected learning is mandatory (the reranker shapes its own
labels); lift concentrates on active users; per-user governance at millions-of-models
scale is the real engineering.

**The evidence.** Exp-01/02's self-healing mechanism — streaming recovers in bounded
time, frozen never does — applied at the level of one user, whose taste drift is
exactly the concept drift the project measures.

**The build order.** Shadow rerank on one feed → live A/B → session layer → platform.
