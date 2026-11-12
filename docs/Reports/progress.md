# Progress log — from a question to a measured answer

**What this page is:** a plain-language record of what we set out to do, how we
are going about it, and what we have actually finished — written so that someone
who understands “the world changes under a model” but has no machine-learning
background can follow it.

**Last updated:** 2026-11-12.

---

## 1. The aim

We want to answer one question honestly:

> **When the world changes, does a decision system that keeps learning do
> meaningfully better than one that was trained once — and what does it cost when
> the correct answers arrive late?**

That is the whole project. Everything else is there to make the answer fair,
repeatable, and hard to attack.

The final deliverable is not a chatbot. It is a **small decision system** that
learns one example at a time, notices when it has become wrong, adapts, and
reports how confident it is. Around it sits a **measuring lab** that anyone can
re-run.

---

## 2. What “drift” means here

A decision system learns from examples. If the world that produced those
examples changes, yesterday’s lessons can become today’s mistakes.

Examples:

- Fraud patterns change, so a fraud detector trained last year misses this
  year’s tricks.
- A spam filter trained on old spam starts letting new spam through.
- A shop’s customers change, so a recommendation rule trained on last season’s
  buyers becomes stale.

That is **drift**: the relationship between the situation and the right answer
has moved.

A second problem makes it harder: **the correct answer often arrives late**.
A loan’s default is known months later; a fraud confirmation may take weeks; a
customer’s real preference shows up only after several interactions. The system
has to keep working while it is still waiting for the truth.

---

## 3. What we measure, in plain words

We do not want a single “average accuracy” number, because averages hide the
moment that matters. A system can look fine on average and still be badly wrong
for a long stretch after a change.

So we measure four things:

| What we measure | Plain meaning |
|---|---|
| **Recovery rate** | After a real change, how often does the system get back to its old standard? |
| **Recovery time** | When it does recover, how many new examples does it need? |
| **Damage score** | How wrong was it, and for how long? Think of the area of the “wrongness × time” region. |
| **Cost** | How much computing time and memory did that recovery take? |

The key comparison is always:

- a **frozen** system — trained once, then never updated; and
- a **learning** system — updates itself after every example.

Both see exactly the same examples in exactly the same order. The only
difference is whether the system is allowed to keep learning.

---

## 4. How we are proceeding

The work is a sequence of gates, not a pile of experiments. Each step is only
“done” when the step before it is trustworthy.

1. **Make the ruler honest.** Build a repeatable lab and fix the way damage is
   measured, before trusting any comparison.
2. **Test the early-warning signals.** Different kinds of change need different
   alarms. Find out which alarm is reliable, how fast it rings, and how often it
   cries wolf.
3. **Compare against the honest competition.** Not just a frozen system. Also a
   system that is retrained on a calendar, and a system that is allowed to peek
   at when the change happened.
4. **Build the learning decision system.** A compact model that keeps learning,
   remembers earlier situations, and stays confident and calibrated.
5. **Measure the cost of late answers.** Delay the correct answers on purpose
   and see what breaks first: the warning, the learning, or the score.
6. **Test on real-world records.** Move beyond computer-generated streams.
7. **Write the final report.** Only from numbers that can be re-run.

This order matters. A fair ruler before a comparison; a fair comparison before a
new model; a measured model before a claim.

---

## 5. Progress, date by date

| Date | What we did | What it gave us | Why it helped |
|---|---|---|---|
| **2026-08-31** | Started the project, built the documentation site, began the reading and the literature map. | A shared vocabulary, a map of the field, and a list of open gaps. | It stopped us from re-inventing known ideas and gave us a place to record every claim. |
| **2026-09-09** | Ran the first controlled comparison: a learning system vs a frozen one, one change, one repeat. | The first measured evidence of the shape: the learning system dipped and came back; the frozen one fell and stayed down. | It turned the argument into a measurement. |
| **2026-09-12** | Repeated the same comparison on 11 different random runs. | The claim was not a one-off: the frozen system never recovered in any of the 11 runs. The repeat also exposed a flaw in how we measured damage. | It showed the result was not a one-off and told us where to look for our own mistakes. |
| **2026-09-22** | Wrote the viva preparation brief: what is done, what is weak, what to say if challenged. | A written list of weaknesses, including the damage-measurement flaw, before an examiner could find them. | It turned a hidden defect into a known, fixable item. |
| **2026-09-23 → 2026-11-11** | Planning and design for the next round; no new measured result was published in this window. | A concrete plan for a reusable experiment lab and a fairer metric. | It kept the project from adding more results on top of a ruler we did not trust. |
| **2026-11-12** | Built the reusable experiment lab; fixed the damage ruler; re-ran the 11-repeat comparison as **E01**; wrote the correction into the documentation. | The corrected headline: frozen recovered **0/11**; the learning system had a real recovery on **7/11**; damage was **≈7.1×** lower over a fixed stretch after the change, **≈11.6×** lower over the rest of the stream. | It replaced an inflated **294×** with a defensible **≈7×**, and made every number re-runnable from a saved file. |

### What changed because of E01

- The old **294×** damage gap was a side effect of a ruler that cut off the
  learning system’s damage. It is **retired**.
- The old **24×** “worst run vs best run” claim did not survive. The corrected
  worst-vs-best gap is **1.27×**, not 24×.
- The old **“11/11 recovered”** counted “never left its normal band” as a
  recovery. The corrected count is **7/11 real recoveries**.
- The old **835-example recovery** was one lucky run. It was never the result;
  the pattern across runs is the result.

### What survived

- The frozen system recovered on **0 of 11** runs. This is the strongest and
  most stable finding.
- The learning system’s damage was lower on **11 of 11** runs.
- The mean corrected gap is **≈7×** in the fixed window and **≈12×** over the
  rest of the stream.
- The learning system still wins the overall averages: **88.3% vs 85.4%**
  correct, with a better agreement score that corrects for chance.

### The no-drift control (E02, first pass)

We also ran the opposite test: **no change at all**. Two stationary streams,
five repeats each, every model and every early-warning signal. Nothing should
ring. This is the “cry wolf” test.

| Signal | Raw mistake signal | Smoothed mistake signal | What it means |
|---|---:|---:|---|
| PageHinkley | **0.000** per 1,000 | **0.055** per 1,000 | Stays calm; meets the strict target. |
| ADWIN | **0.008** per 1,000 | **0.522** per 1,000 | Calm on the raw signal; overreacts when smoothed. |
| KSWIN | **0.000** per 1,000 | **7.67** per 1,000 | Calm on the raw signal; very noisy when smoothed. |

**Plain reading:** smoothing the mistake signal is not free. It makes two of the
three alarms fire on a world that has not changed. PageHinkley is the robust
default in this setting. This is a useful result for the detector map: the
“smoothed” version must be treated as a diagnostic, not as the default alarm.

The models themselves stayed stable on both stationary streams, which is the
sanity check the control was built for.

### The detector map (E04, first pass)

We then ran the same four **kinds of change** — sudden, gradual, incremental and
recurring — across four different computer-generated worlds, five repeats each,
with two different base learners and eight early-warning signals.

After fixing a scoring bug (a signal that never rang was being left out of the
average, which made it look perfect), one alarm came out ahead on **all four**
kinds of change: **HDDM-A**, using the raw mistake signal.

| Kind of change | Best signal | Balanced score | Changes caught | False alarms per 1,000 |
|---|---|---:|---:|---:|
| Sudden | HDDM-A, raw | **0.68** | 75% | 0.008 |
| Gradual | HDDM-A, raw | **0.84** | 95% | 0.008 |
| Incremental | HDDM-A, raw | **0.88** | 90% | 0.003 |
| Recurring | HDDM-A, raw | **0.50** | 73% | 0.031 |

**Plain reading:**

- One alarm — HDDM-A on the raw signal — is a sensible default across all four
  kinds of change.
- **Recurring change is the hardest** (balanced score 0.50); incremental change
  is the easiest (0.88).
- Smoothing the mistake signal does **not** help: it raises false alarms without
  buying a better ranking. The raw signal stays the default; the smoothed one is
  a diagnostic.
- The ranking of alarms is mostly the same whether the base learner is static or
  adaptive; KSWIN is the exception.
- The scoring bug mattered: before the fix, a signal that almost never rang
  looked perfect. After the fix, it did not. This is exactly the kind of error
  the lab is meant to catch.

Full technical write-up: [E04 — Detector map](E04-detector-map.md).

---

## 6. How the project has benefited

| What we did | How it helped |
|---|---|
| Built a reusable experiment lab | Every future result uses the same rules, so a new experiment cannot quietly use easier conditions. |
| Fixed the damage ruler | We removed the biggest known attack on our own result before anyone else used it. |
| Added progress logs and saved files after every run | A long run no longer has to be repeated from zero; we can see where time goes. |
| Wrote the correction into the live documentation | There is one place for the current numbers; old pages carry a warning, not a contradiction. |
| Kept the old results | The change is auditable: anyone can see what we believed, why it was wrong, and what replaced it. |

---

## 7. What we are not claiming

Honesty is part of the method. We are **not** claiming:

- that the learning system always wins;
- that **294×** is the true size of the effect (it is not);
- that the result holds on real-world records yet;
- that the late-answer problem is solved;
- that this is a chatbot or a large language model.

We are claiming a smaller, measured, repeatable result: **a system that keeps
learning recovers from change, a frozen one does not, and the corrected damage
gap is around seven times in the fixed window.**

---

## 8. What is next

In order:

1. **Which early-warning signal works for which kind of change.** A map of
   alarms and change types, with false-alarm rates.
2. **The honest competition.** Add a system retrained on a fixed calendar and a
   system allowed to peek at when the change happened.
3. **The learning decision system.** A compact model that remembers earlier
   situations and stays calibrated.
4. **The cost of late answers.** Delay the correct answers and measure what
   breaks first.
5. **Real-world records.** Move from computer-generated streams to real data.
6. **The final report.** Only from saved, re-runnable numbers.

---

## 9. Plain-word glossary

| Word | What it means here |
|---|---|
| **Drift** | The world changes, so yesterday’s lessons become wrong. |
| **Learner / decision system** | A rule that is learned from examples and used to make a decision. |
| **Frozen** | Trained once, then never updated. |
| **Learning / streaming** | Updates itself after every new example. |
| **Correct answer (label)** | The truth for one example; often arrives late. |
| **Early-warning signal (detector)** | A smoke alarm for change: it rings when the stream starts to look different. |
| **Damage score** | How wrong the system was, multiplied by how long it stayed wrong. |
| **Repeat (seed)** | One random re-run of the same experiment, to separate a real effect from luck. |
| **Comparison that peeks** | A baseline that is allowed to know when the change happened; an upper bound, not a fair opponent. |
| **Calibration** | When the system says “70% sure,” it is right about 70% of the time. |

---

## 10. Where to read more

- **The corrected result:** [E01 — Metric v2 correction](E01-metric-v2.md)
- **All measured verdicts:** [Results](Results.md)
- **The argument and the metric family:** [Thesis](../Phase1:Experimentation/thesis.md)
- **The first experiment, with its correction warning:** [Exp-01](../Phase1:Experimentation/exp01-deepriver-vs-frozen-sea.md)
- **The 11-repeat sweep, with its correction warning:** [Exp-02](../Phase1:Experimentation/exp02-seed-sweep-sea.md)
