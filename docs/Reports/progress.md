# Progress log — from a question to a measured answer

**What this page is:** a plain-language record of what we set out to do, how we
are going about it, and what we have actually finished — written so that someone
who understands “the world changes under a model” but has no machine-learning
background can follow it.

**Last updated:** 2026-11-14.

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
| **2026-11-12** | Built the reusable experiment lab; fixed the damage ruler; re-ran the 11-repeat comparison as **E01** on a clean three-model panel; wrote the correction into the documentation. | The corrected headline: frozen recovered **0/11**; the learning system had a real recovery on **8/11**; damage was **≈6.9×** lower over a fixed stretch after the change, **≈10.8×** lower over the rest of the stream. | It replaced an inflated **294×** with a defensible **≈7×**, and made every number re-runnable from a saved file. |
| **2026-11-13** | Re-ran the detector map on real Elec2 records (**E17**); built an isolated, parallel reproduction of the KDD'26 14-detector benchmark. | On the static host HDDM-A wins all four shapes again (0.93/0.96/0.96/0.79); on the adaptive host ADWIN wins sudden (1.00 vs 0.40) and recurring (0.77 vs 0.59). The 10-trial and 50-trial KDD runs agree: SEED/STEPD 2.12, HDDM-W 3.56, HDDM-A 3.87; the F1 values match the paper's tables. The ABCD/ABCD(X) arm is skipped and cited from the paper. | It shows the synthetic default is host-conditional, and gives us a real-data ranking where SEED and STEPD beat HDDM-A. |
| **2026-11-14** | Documented the real-stream model check (**E14**) with acceptance rules; reconciled the no-change control (**E02**) false-alarm numbers to the committed artifact; finished the **E03** 11-seed ladder. | E14 passes provisionally: ARF leads both real datasets, and the deep model is confirmed as a signal extractor rather than the host. E02 now quotes one set of worst-host numbers. E03 has 220 runs: every streaming adapter beats frozen and every retrain clock on `damage_ref` (best LB 455.7 vs scheduled-1k 733.5 and frozen 871.7). | It closes the Phase 0 measurement gaps; the phase-noise test remains inconclusive because the refit buffer dominates the 1k/5k clocks. |
| **2026-11-14** *(detector pilots)* | Ran the **E18** deep-signal gate and the **E19** DF-H v1 fusion pilot. | E18: best deep AUC **0.616** vs error **0.598** — gate failed; at 0.1/1k, feature permutation is detected in **72–414** samples vs error 184–1,033, and feature filtering is detected where error is blind; `feature_ks` is competitive. E19: DFH ranks **3rd** (7.09) behind STEPD 5.22 and SEED 5.44; FAR 0.04 vs ADWIN 0.11; no primary feature-cell win; ablations own feature cells; no full E20. | It closes the detector track honestly: the feature monitor adds timing value but not top-2, so **E21 Delay Tax becomes the headline contribution**. |
| **2026-11-14** *(delay tax)* | Measured the **E21** fixed-delay Lemma-1 headline from the E19 trial artifacts; added the fixed/geometric/partial/selective/batch label scheduler and tests. | At `L = 2,000`, Stage 1 warns in **70–513** samples after onset vs **SEED/STEPD 2,120–2,710** and **ADWIN 2,472–3,269**; the Stage-1 line is flat in `L`, every error-only line shifts by `L`. | It gives the project its surviving headline: the label-free warning is not floor-bound, even though the detector lost the KDD rank. The full regime matrix and Lemmas 2–3 remain scaffolded, not run. |
| **2026-11-14** *(shootout)* | Ran **E22** on the 80-cell Covtype + Electricity pilot: DF-H vs ADWIN, HDDM-A, HDDM-W and PageHinkley, with SEED/STEPD as reference. | DF-H is **3rd (7.34)**, ahead of HDDM-W 7.47, ADWIN 9.94, HDDM-A 11.34 and PageHinkley 12.53; behind STEPD 5.44/SEED 5.59. Feature permutation **5.8/5.5** vs the panel's 8.8–16.0; FAR 0.04; HDDM-W wins Covtype 6.1 vs 7.9. | It gives the direct capability proof against the requested panel, with the honest limits: not top-2, and feature filtering is better handled by the Stage-1-only ablation. |

### What changed because of E01

- The old **294×** damage gap was a side effect of a ruler that cut off the
  learning system’s damage. It is **retired**.
- The old **24×** “worst run vs best run” claim did not survive. On the clean
  panel the worst learning-system run still pays **0.74×** the damage of the
  best frozen run, so the comparison is categorical the other way.
- The old **“11/11 recovered”** counted “never left its normal band” as a
  recovery. The corrected count is **8/11 real recoveries**.
- The old **835-example recovery** was one lucky run. It was never the result;
  the pattern across runs is the result.

### What survived

- The frozen system recovered on **0 of 11** runs. This is the strongest and
  most stable finding.
- The learning system’s damage was lower on **11 of 11** runs.
- The mean corrected gap is **≈6.9×** in the fixed window and **≈10.8×** over
  the rest of the stream.
- The learning system still wins the overall averages: **88.3% vs 85.4%**
  correct, with a better agreement score that corrects for chance.

### The no-drift control (E02, first pass)

We also ran the opposite test: **no change at all**. Two stationary streams,
five repeats each, every model and every early-warning signal. Nothing should
ring. This is the “cry wolf” test.

| Signal | Raw mistake signal | Smoothed mistake signal | What it means |
|---|---:|---:|---|
| PageHinkley | **0.055** per 1,000 | **0.050** per 1,000 | Stays calm; meets the strict target. |
| ADWIN | **0.012** per 1,000 | **0.548** per 1,000 | Calm on the raw signal; overreacts when smoothed. |
| KSWIN | **0.002** per 1,000 | **7.667** per 1,000 | Calm on the raw signal; very noisy when smoothed. |

*Values are the worst host (static HT or adaptive HAT) in the E02 artifact; the
acceptance block uses the same worst-host rule.*

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

### The real-stream detector check (E17)

We re-ran the same eight alarms on real electricity-market records (Elec2,
45,312 rows), shuffled, with a known label-swap change injected halfway through.
Five repeats, the same two hosts.

| Kind of change | HDDM-A, static host | ADWIN, static host | HDDM-A, adaptive host | ADWIN, adaptive host |
|---|---:|---:|---:|---:|
| Sudden | **0.93** | 0.45 | 0.40 | **1.00** |
| Gradual | **0.96** | 0.50 | 1.00 | 1.00 |
| Incremental | **0.96** | 0.84 | **1.00** | 0.97 |
| Recurring | **0.79** | 0.51 | 0.59 | **0.77** |

**Plain reading:** on the static learner, HDDM-A keeps its four-shape win. On
the adaptive learner, ADWIN wins sudden and recurring change. The adaptive
learner can repair the mistake rate quickly enough that HDDM-A's one-sided test
misses the change; ADWIN compares windows two-sidedly and still rings. The
“HDDM-A is the default” claim is therefore true for the static host, not
universally. Five repeats and one real dataset: a warning, not a final ranking.
Full write-up: [E17 — Real-stream detector map](E17-real-detector-map.md).

### The real-stream model check (E14)

We also replayed the model comparison on real features and labels: Elec2
(45,312 rows) and Phishing (1,250 rows), with a known change injected halfway
through. Five repeats, five models.

| Model | Agreement (κ) | Damage paid (`damage_ref`) | Recovered? |
|---|---:|---:|---:|
| **ARF** | **0.724** | **198.0** | 20% of runs |
| HAT | 0.583 | 245.7 | 15% |
| DriftFormer | 0.524 | 284.9 | 0% |
| Frozen RF | 0.395 | 297.4 | 0% |

**Plain reading:** ARF leads on both real datasets; the deep model does not beat
it. That is expected and useful: for DF-H the deep model is a **sensor**, not the
main classifier. The slice is provisional — two datasets, five repeats — so it is
a host check, not a field-wide ranking. The acceptance rules pass provisionally
(pooled damage ratio 0.666×, ARF lower in 18/20 cells, frozen recovered 0/20).
Full write-up: [E14 — real-stream model check](E14-real-streams.md).

### The scheduled-retrain ladder (E03)

We ran the full 11-repeat ladder on two computer-generated streams: a frozen
model, three retrain clocks (1k, 5k, 20k), static and adaptive trees, ARF,
Leveraging Bagging, ADWIN Bagging, and the timing oracle. 220 runs.

| Model | Damage paid (`damage_ref`) | Recovered? |
|---|---:|---:|
| **Leveraging Bagging** | **455.7** | 100% |
| HAT | 489.2 | 96% |
| ARF | 575.4 | 59% |
| ADWIN Bagging | 614.1 | 100% |
| Hoeffding Tree | 700.8 | 96% |
| Scheduled RF, 1k | 733.5 | 100% |
| Scheduled RF, 5k | 824.1 | 100% |
| Scheduled RF, 20k | 849.4 | 68% |
| Frozen RF | 871.7 | 0% |
| Oracle refit | 864.3 | 0% |

**Plain reading:** every learning-from-the-stream model pays less damage than the
frozen model and than every retrain clock. The best is Leveraging Bagging; even
the fastest retrain clock (1k) pays **1.6×** more damage. The scheduled models
“recover” by construction at each tick, so recovery rate is not the comparison —
damage is. The phase-noise test is **inconclusive**: the 12k refit buffer is
longer than the 1k and 5k clocks, so the lag is buffer-dominated; only the 20k
cell is valid and it has one observation. Artifact:
`experiments/results/e03/summary.json`.

### The KDD'26 benchmark fast pilot

We reproduced the paper's Monte Carlo protocol on seven real datasets at both
ten and fifty trials, with the six fastest detectors. The 50-trial run
confirms the pilot: **SEED and STEPD tie for first** (average rank 2.12; mean
F1 0.300 and 0.303), **HDDM-W is third** (3.56; 0.163), and **HDDM-A is fourth**
(3.87; 0.155). DDM (4.54) and ADWIN (4.80) follow. HDDM-A is clearly not the
best detector on real data.

The F1 values also reproduce the paper's tables — for example abrupt feature
permutation is 0.572/0.526/0.313 for SEED/STEPD/HDDM-A here vs 0.61/0.53/0.36
published. The ABCD/ABCD(X) arm is **not re-run**: it is the expensive part of
the matrix, it overheats the laptop CPU, and the paper's published result is
used for that cell instead.

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

The detector track is now the project's ultimate goal. The question is no longer
"which single alarm is best" — it is "can a label-free deep monitor plus a
supervised confirmation beat the detector benchmark, and what happens when the
labels arrive late?"

**Ultimate goal:** build **DF-H** — a two-stage detector that combines a
label-free deep representation monitor with supervised confirmation — and show
that it (a) is top-2 on the KDD'26 benchmark with wins on the feature and
recurring cells, and (b) keeps the most detection power under delayed, partial
and selective labels.

In order:

1. **Signal gate (E18).** ✅ ran — **negative**: best deep AUC 0.616 vs error
   0.598. The timing diagnostic is positive on feature drift, but the classical
   KS control is competitive.
2. **DF-H v1 (E19).** ✅ ran — **gate failed**: DFH 3rd overall (7.09) behind
   STEPD 5.22 and SEED 5.44; FAR passes, top-2 and primary feature-cell checks
   fail.
3. **DF-H v2 (E20).** **Not run.** The pilot-first rule says the full matrix is
   not justified; a learned gate would need more datasets than the pilot has.
4. **Delay Tax benchmark (E21).** ✅ **fixed-delay headline measured**: at
   `L = 2,000`, Stage 1 warns in **70–513** samples vs SEED/STEPD
   **2,120–2,710** and ADWIN **2,472–3,269**; Stage 1 is flat in `L`. The
   stochastic/partial/selective/batch matrix and Lemmas 2–3 are scaffolded but
   not run.
5. **Final report.** Only from saved, re-runnable numbers.

The older breadth-first experiments (E05, E08–E13) are deferred; E15 is replaced
by E21. See [Roadmap v2](Roadmap-v2.md) for the full plan and gates.

**Detector-pilot status (2026-11-14).** The E18 instrumentation is complete
(`signal_snapshot()`, `DeepSignalMonitor`, `signals.npz`, pure-NumPy
`build_trial_stream`, `experiments.deep_kdd` precompute/report/detect). E18's
AUC gate failed, but at 0.1/1k FAR the deep signals detect feature permutation
2–5× earlier than error and detect feature filtering when error is blind. E19's
DF-H v1 pilot ranked 3rd, passed FAR, and beat the no-feature control, but
failed top-2, the primary feature-cell test and Covtype robustness. The full
E20 matrix is not justified. **E21 now carries the headline**: at `L = 2,000`,
Stage 1 warns in 70–513 samples while the error-only panel takes 2,120–3,269;
the label-free warning is not floor-bound. **E22 then shows the capability
against the requested classical panel**: DF-H is 3rd (7.34), ahead of HDDM-W
(7.47), ADWIN (9.94), HDDM-A (11.34) and PageHinkley (12.53); only SEED/STEPD
lead. The detector remains a supporting warning/confirmation result, with
feature permutation as its clear strong cell.

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
- **The detector track:** [E18 — deep signal gate](E18-deep-signals.md) · [E19 — DF-H v1 pilot](E19-hybrid-detector.md) · [E22 — detector shootout](E22-detector-shootout.md)
- **The surviving headline:** [E21 — Delay Tax detector](E21-delay-detector.md)
