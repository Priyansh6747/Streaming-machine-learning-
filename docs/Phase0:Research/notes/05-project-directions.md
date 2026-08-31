# 05 — Candidate project directions

## The trap to avoid first

> "I applied River to dataset X and got 92% accuracy."

That is a **tutorial**, not research. It has no question, so it can have no finding.
Nothing about it could have come out the other way.

A research project needs a question **whose answer you don't know in advance** and
**can measure**. The template:

> *Under condition C, does method A outperform method B on measure M — and why?*

Every option below fits that template.

---

## Option A — Drift detector benchmark under controlled drift

**Question.** How do ADWIN, DDM, EDDM, Page-Hinkley and KSWIN compare on detection
delay, false-alarm rate and missed detections, across sudden / gradual / incremental
/ recurring drift, and how does that change with noise level and drift magnitude?

**Method.** `datasets.synth.ConceptDriftStream` — you set the drift position, so you
have ground truth. Sweep drift type × noise × detector × seed. Report mean ± std.

- **Rigour:** ★★★★★ — real ground truth, real numbers
- **Effort:** ★★☆☆☆ — cheap to run, laptop-scale
- **Novelty:** ★★☆☆☆ — this comparison exists in the literature; your angle has to be
  the specific conditions you sweep
- **Risk:** low. Guaranteed to produce results.

---

## Option B — Does adaptation actually pay off?

**Question.** Do drift-aware models (Hoeffding Adaptive Tree, ARF, ADWIN-Bagging)
actually recover better than static ones (Naive Bayes, plain Hoeffding Tree) — and
what does adaptation *cost* in time and memory?

**Method.** Same streams, adaptive vs static models. Measure the accuracy-over-time
curve: **dip depth** and **recovery time** after each injected drift, plus
throughput and memory.

**Why it's good.** Produces the single best figure in this whole area — the "V"
where a static model drops and never recovers while the adaptive one climbs back.

- **Rigour:** ★★★★☆ · **Effort:** ★★☆☆☆ · **Novelty:** ★★☆☆☆ · **Risk:** low

---

## Option C — A + B combined ⭐ recommended

One coherent story in two acts:

> **Act 1:** How well can we *detect* drift? (Option A)
> **Act 2:** Does detecting it actually *help*? (Option B)

**Why this is the right pick.** It's a complete narrative rather than a lone
experiment. Both halves share the same infrastructure — one stream generator, one
experiment runner — so the second act is nearly free once the first works. Every
result is measurable against ground truth, it runs on a laptop, and it validates on
real datasets (Elec2, Phishing) at the end for credibility. It cannot fail to
produce results, which matters when there's a deadline.

- **Rigour:** ★★★★★ · **Effort:** ★★★☆☆ · **Novelty:** ★★★☆☆ · **Risk:** low

---

## Option D — Online vs. batch: the real trade-off

**Question.** Against a batch model (scikit-learn) periodically retrained on a
sliding window, how does an online model compare on accuracy, memory, latency, and
total compute — and where is the crossover point?

**Method.** Same data, two regimes. Sweep the retraining interval for the batch side.
Plot accuracy against cost.

**Strength.** The practical/engineering question a lot of people actually care about,
and the resource measurements are concrete.
**Weakness.** Easy to make an unfair comparison. You must be scrupulous about the
batch model's retraining budget, or a reviewer will (correctly) dismiss it.

- **Rigour:** ★★★☆☆ · **Effort:** ★★★☆☆ · **Novelty:** ★★☆☆☆ · **Risk:** medium

---

## Option E — Applied: streaming fraud / intrusion detection

**Question.** How do streaming models perform on a **severely imbalanced** real
stream, and do streaming resampling methods (`river.imblearn`) help?

**Data.** `datasets.CreditCard` (0.17% fraud), or a network intrusion dataset
(NSL-KDD, CICIDS2017) fed through `river.stream.iter_csv`.

**Strength.** Concrete application, easy to motivate, imbalance is a genuinely open
problem in streams.
**Weakness.** Ground-truth drift points are unknown in real data, so you can't
measure detection delay — you lose the cleanest metrics from A. Also risks becoming
a benchmark run rather than a question.

- **Rigour:** ★★★☆☆ · **Effort:** ★★★☆☆ · **Novelty:** ★★★☆☆ · **Risk:** medium

---

## Option F — Build something: a hybrid or an ensemble strategy

**Question.** Can a meta-detector (e.g. requiring 2 of 3 detectors to agree before
resetting) beat any single detector on the delay/false-alarm trade-off?

**Method.** Implement it against `river.base` interfaces, benchmark it with Option A's
harness.

**Strength.** Highest novelty — you produce an artefact, not just a comparison.
**Weakness.** Highest risk. It might simply not work, and you need Option A's
infrastructure built first anyway.

- **Rigour:** ★★★★☆ · **Effort:** ★★★★★ · **Novelty:** ★★★★★ · **Risk:** high

**Sensible move:** treat F as a *stretch chapter* on top of C. If it works, it's your
headline contribution. If it doesn't, C is already a complete project and the failed
attempt is an honest, reportable negative result.

---

## Summary

| | Rigour | Effort | Novelty | Risk |
|---|---|---|---|---|
| A — detector benchmark | ★★★★★ | ★★ | ★★ | low |
| B — adaptation payoff | ★★★★ | ★★ | ★★ | low |
| **C — A + B** ⭐ | ★★★★★ | ★★★ | ★★★ | low |
| D — online vs batch | ★★★ | ★★★ | ★★ | medium |
| E — imbalanced application | ★★★ | ★★★ | ★★★ | medium |
| F — novel method | ★★★★ | ★★★★★ | ★★★★★ | high |

**Recommendation: C, with F as an optional stretch chapter.**

---

Next: [06 — Literature & landscape](06-literature-landscape.md)
