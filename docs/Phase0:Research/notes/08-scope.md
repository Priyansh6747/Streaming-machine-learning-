# 08 — Scoping a one-year project

## The problem with the old scope

The detector benchmark from [05](05-project-directions.md) is good work, but it's
**6–8 weeks**, not a year. Stretching it to fill twelve months would mean adding
more detectors and more datasets — more rows in a table, not more insight. Examiners
notice that. A bigger table is not a bigger contribution.

So we need a **bigger question**, not more runs of the same one.

## The move: make it a systems question

Here's the thing nobody told you yet. Look at how concept drift research is almost
always evaluated:

- A CSV is replayed in a single Python process
- Labels arrive **instantly and always** — `learn_one(x, y)` gets `y` in the same tick as `x`
- No network, no queue, no concurrency, no backpressure
- Success is measured almost entirely as accuracy
- Nothing is said about versioning, rollback, audit, or reproducibility

Now compare that to how a model actually runs in production. Every single one of
those assumptions is false. **That gap is your project.** And it's a gap that exists
precisely because the field is populated by ML researchers, not infrastructure
people. You're walking in with the exact background that's missing.

> **Thesis spine:**
> *Streaming ML works in simulation. Does it work in production?
> Build the system, find where it breaks, fix one thing properly.*

That's a year. It's genuinely novel. And roughly 70% of it is systems engineering,
which is your day job.

---

## The specific gap worth attacking: delayed labels

This is the sharpest version of the problem, and worth understanding fully because
it's likely your core contribution.

**In every paper:** sample `x` arrives, and its true label `y` arrives with it.
The model predicts, is immediately told whether it was right, and learns.

**In reality:** you approve a transaction *now*. You find out it was fraudulent
**30–90 days later**, when the chargeback lands. You show an ad *now*. You learn
whether it converted next week. You flag a login as suspicious *now*. You find out
when the user complains, or never.

Follow the consequence, because it's not obvious and it's nasty:

> A drift detector works by watching the model's **error stream**.
> If labels are 30 days late, the error stream is 30 days late.
> **So you detect drift 30 days after it happened** — and you've been silently
> wrong for a month.

It gets worse. The standard evaluation protocol itself breaks: prequential
evaluation ([04](04-evaluation.md)) assumes you can score each prediction as it
happens. Under delayed labels **you cannot even measure your own accuracy in real
time**, let alone react to it. The instrument the entire field uses to measure
success doesn't work in the setting the field claims to serve.

That's a real research gap. And the way out is interesting:

- **Unsupervised drift detection** — watch the *input* distribution `P(X)` instead of
  the error stream. No labels needed, so no delay. But it fires on harmless changes
  too (virtual drift, [02](02-concept-drift.md)).
- **Hybrid** — use the label-free signal for an early warning, then confirm with the
  delayed supervised signal when it eventually arrives.

Designing, building, and evaluating that hybrid is a legitimate contribution. It's
also — conveniently — a queueing and state-management problem, which is your turf.

## Secondary gap: operating a model that never stops training

A batch model is an artifact. `model_v3.pkl` sits in blob storage. It's immutable,
versioned, hashable, rollback-able. Standard deployment.

A streaming model **has no artifact**. It's a mutating in-memory process. So every
operational practice you rely on quietly breaks:

| You normally do | With a streaming model |
|---|---|
| Roll back to the last good version | Roll back to *what*? There are no versions. |
| Pin a build for reproducibility | State depends on every event ever seen, in order |
| A/B test v2 against v1 | Both keep changing while you test |
| Audit "why was I denied in March?" | That model state no longer exists |
| Restart a pod | You just lost all learned state |
| Scale to 3 replicas | Three models now diverge — which one is *the* model? |

That last row is worth sitting with. **Horizontally scaling a stateful learner is an
open problem.** Nobody has a clean answer. You're allowed to just measure how bad the
divergence gets — that alone is a publishable observation.

This chapter is nearly pure infra, and it's underexplored because ML researchers
don't run production systems.

---

## The year, in phases

Assume this runs alongside coursework — call it **10–15 hours/week**, not full time.
Deliberate slack is built in.

### Phase 0 — Orientation (Weeks 1–4)
Understand the field well enough to know what's already done.
- Deep literature survey (see [research/perplexity-prompts.md](../research/perplexity-prompts.md))
- Get River running; reproduce a published baseline number
- **Deliverable:** annotated bibliography + a working hello-world stream
- **Checkpoint:** can you explain concept drift to a classmate in 5 minutes?

### Phase 1 — The offline baseline (Weeks 5–12)
The original Option C, now correctly sized as **one chapter**, not the project.
- Detector benchmark under controlled synthetic drift
- Adaptive vs static models: dip depth, recovery time, memory, throughput
- **Deliverable:** a reusable experiment harness + your first real results
- **Why it matters:** this is your control condition. Everything later is measured
  against "what the literature says happens." You need the idealised numbers before
  you can show reality is worse.

### Phase 2 — Build the real system (Weeks 13–24)
The infrastructure spine. Your home turf, and the part that makes this a *major*
project rather than a notebook.
- Event source (Kafka / Redpanda / Redis Streams — your call)
- A model service exposing `POST /predict` and consuming a learning stream
- Metrics → Prometheus, dashboards → Grafana
- Drift detection running as a live monitor, not an offline script
- Load generation so you can push it until something breaks
- **Deliverable:** a running system, containerised, reproducible
- **Chapter:** "an architecture for continual learning in production"

### Phase 3 — Break it, measure it (Weeks 25–32)
Now the research gets interesting, because you can ask questions nobody has data on.
- What does drift adaptation *cost* — p50/p99 prediction latency, throughput ceiling?
- What happens when a model reset coincides with peak load?
- How do replicas diverge when you scale out? How fast?
- What happens under backpressure — do you drop learning or drop serving?
- Restart/recovery: how expensive is checkpointing model state?
- **Deliverable:** the "reality tax" chapter — the operational cost the literature
  never reports

### Phase 4 — The delayed-label contribution (Weeks 33–44)
The core novel work.
- Simulate realistic label delay (fixed, variable, partial, never-arriving)
- Show that standard detectors degrade badly — quantify it
- Build the hybrid: unsupervised early-warning + delayed supervised confirmation
- Evaluate against the Phase 1 baselines
- **Deliverable:** your actual contribution
- **Fallback:** if the hybrid doesn't beat the baselines, *that's still a result.*
  "We quantified how much delayed labels break existing methods, and show why the
  obvious fix is insufficient" is a perfectly good chapter. **You cannot fail this
  phase**, which is exactly why it's the one to bet on.

### Phase 5 — Write up (Weeks 45–52)
- Full report, figures, statistical testing
- Clean the repo, write a proper README, make it reproducible
- Prepare the demo (a live drift-and-recovery dashboard is a genuinely good viva prop)

---

## What this gets you

Three chapters that each stand alone, in increasing order of novelty:

1. **Baseline** — "here's what the literature says happens" *(safe, guaranteed)*
2. **System** — "here's what happens when you actually deploy it" *(novel, infra-heavy)*
3. **Contribution** — "here's a gap, and here's my attempt at it" *(the research)*

If Phase 4 goes badly, you still have 1 and 2 and a complete project. If Phase 2
runs long, Phase 3 can shrink. **The failure modes are all survivable**, which is
what you want when you've never done this before.

## Scope guards — write these down and defend them

Things it will be tempting to add. Don't.

- ❌ Deep learning / online neural nets — different field, huge time sink
- ❌ Regression, clustering, recommendation — **classification only**
- ❌ Distributed *training* (parameter servers, federated learning) — a PhD, not this
- ❌ Building your own streaming framework — use River
- ❌ Multi-cloud / k8s production-grade deploy — docker-compose is sufficient and
  nobody will mark you down for it
- ❌ More than ~3 real datasets

The single most common way student projects fail is scope creep, not difficulty.

## Open decisions

- [ ] Does your teacher require novelty, or is rigorous comparison enough?
- [ ] Report only, or report + working system? (This scope assumes both)
- [ ] Is there a required format (IEEE two-column?) and length?
- [ ] Interim milestones / review presentations during the year?
- [ ] Individual or group?

---

See [research/perplexity-prompts.md](../research/perplexity-prompts.md) for the
literature survey prompts. **Phase 0 starts there** — you need to confirm the gap is
real before betting a year on it.
