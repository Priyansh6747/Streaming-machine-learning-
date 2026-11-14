# Streaming Machine Learning — the Delay Tax

**Models that never stop learning, on data that never stops changing — and the
price of learning from yesterday's truth.**

A research project (major-project / final-year scale) on **machine learning from
data streams**: models that learn one sample at a time, forever, in bounded memory,
and that can tell when the world has quietly changed under them — **concept
drift**. Built on the [River](https://riverml.xyz/) framework, with PyTorch deep
learners (DeepRiver) as the experimental subject.

| Status | |
|---|---|
| Phase 0 — research & literature | ✅ complete |
| Phase 1 — experimentation | 🔬 concluded: single-seed pilot, 11-seed sweep, **E01 corrected metric**, **E04 detector map**, **E17 real-stream detector check**, **KDD'26 six-detector reproduction**; **DF-H detector track (E18–E21) is the new ultimate goal** |

---

## The vision

Almost all machine learning you were taught treats the dataset as a *finished
object*: it sits on disk, you split it, you fit a model, you are done. Streaming
machine learning drops that assumption. Data arrives one sample at a time,
potentially forever, and is too large — or too fast — to store. The model is never
"finished"; it is permanently mid-training:

```mermaid
flowchart LR
    S["Data stream<br/>x1, x2, x3, ... (never ends)"] --> P["predict_one(x)<br/>1. TEST"]
    P --> M["metric.update(y, y_pred)<br/>2. SCORE"]
    M --> L["learn_one(x, y)<br/>3. TRAIN"]
    L --> D{"Drift<br/>detector"}
    D -- "no drift" --> S
    D -- "DRIFT!" --> R["Adapt:<br/>reset / retrain / reweight"]
    R --> S
```

Each sample is used **for testing before it is used for training** — that is what
makes the score honest — and the world generating the data keeps changing, so the
model must also *notice when it becomes wrong* and adapt.

The project's own angle is the **Delay Tax**: in production, the label for today's
decision often arrives hours or days late. A model is then being judged — and
updated — on yesterday's truth, and every adaptive trick we know is weakened by
that lag. The full idea is in the
[synopsis](docs/Phase0:Research/idea/synopsis.md); the underlying problem
(types of drift, why detection is hard) in the
[concept-drift note](docs/Phase0:Research/notes/02-concept-drift.md).

---

## The core argument

On a drifting stream, **every model is equally good before the change — the
difference is what happens after**:

- **A frozen batch model** (trained once, deployed, never updated) has no mechanism
  that reacts to data. After a drift it stays wrong until an *external* decision to
  retrain. Its recovery time is undefined.
- **A scheduled-retrain model** watches its clock, not the data. It recovers only at
  the next retrain tick — on average it stays wrong for half a retrain interval too
  long, by pure luck of phase.
- **A streaming model** watches the error stream itself. It detects the change,
  adapts immediately, and its performance returns to baseline within a bounded,
  data-driven number of samples.

The cost of a drift is the *area of skill lost* while a model is wrong — and it is
exactly that damage area, not global accuracy, where streaming wins structurally.
Schematic of the three behaviours (illustrative, pre-registered):

![Recovery shapes: streaming self-heals, scheduled heals on the retrain tick, frozen never does](docs/Phase1:Experimentation/figures/thesis-recovery-schematic.png)

The full mechanism and the metric family are argued in the
[thesis](docs/Phase1:Experimentation/thesis.md), and priced against the
literature in the [metrics dossier](docs/Phase1:Experimentation/streaming-vs-batch-metrics.md)
(including the honest counter-evidence: a *tuned, frequently retrained* batch model
can beat streaming under delayed labels and rare classes — which is exactly why the
Delay Tax is the open problem).

---

## First evidence: the pilot, then the sweep

Exp-01 put the weakest form of the claim on the bench: one synthetic drifting
stream, one abrupt change, a **DeepRiver rolling deep classifier** vs a **frozen
RandomForest**. On seed 42 the streaming model dipped briefly and was back inside
its pre-drift band **835 samples** after the drift, while the frozen model fell from
≈ 0.75 to ≈ 0.59 Kappa — **and never came back**.

Exp-02 then asked the obvious question — was 835 a fluke of that one seed? — by
rerunning the identical experiment on **11 seeds**. The frozen model recovered on
**none of the eleven**. The original damage-area metric reported **294×**, but E01
showed that was an **upper bound**: the integral was truncated on seeds where the
recovery rule collapsed. The corrected metric uses a fixed 5,000-sample window and
a shared reference level: **74.8 ± 61.8** vs **515.0 ± 97.1** samples·κ — a
**≈ 6.9×** gap (**≈ 10.8×** over the rest of the stream). On the clean panel the
worst streaming seed (217.9) is **0.74×** the best frozen seed (295.2), so the old
**24×** worst-case claim is retired too. Recovery is **8/11** seeds with a real
excursion (285–4,279 samples); the old 835 was seed luck. The claim generalised;
the magnitude did not — which is the honest version of the result.

![E01 corrected metric: damage_ref and AUE per model](docs/Phase1:Experimentation/figures/e01-metric-v2-model-comparison.png)

*E01 (11 seeds, corrected metric): `damage_ref` is the primary damage number; AUE
is secondary because the timing oracle knows when the drift happened but not what
the new concept is.*

![Exp-02 seed sweep (legacy metric): recovery time and damage area per seed](docs/Phase1:Experimentation/figures/exp02-seed-sweep-recovery.png)

![The legacy damage gap as a distribution, one dot per seed](docs/Phase1:Experimentation/figures/exp02-damage-distribution.png)

![How hard the drift hit each model, per seed](docs/Phase1:Experimentation/figures/exp02-dip-depth-per-seed.png)

![Exp-01 measured rolling Kappa over the stream: DeepRiver dips and heals, frozen RF stays down](docs/Phase1:Experimentation/figures/exp01-rolling-kappa-deepriver-vs-frozen.png)

![Exp-01 measured recovery time and legacy damage area](docs/Phase1:Experimentation/figures/exp01-damage-area-vs-recovery.png)

The detector map (E04, 5-seed first pass) then asked which early-warning signal
catches which kind of change. Across four synthetic generators, four drift shapes,
two hosts and eight detectors, **HDDM-A on the raw mistake stream is the best
default on the static host**; recurring drift is the hard case (balanced score
0.50) and incremental is the easiest (0.88). The real-stream check (E17, Elec2,
five seeds) repeats that four-shape win on the static host
(0.93/0.96/0.96/0.79), but on the adaptive HAT host **ADWIN wins abrupt (1.00 vs
0.40) and recurring (0.77 vs 0.59)**; the best-host mean is ADWIN 0.935 vs
HDDM-A 0.910. The default is therefore **host-conditional**, not universal.
Smoothing the mistake signal raises false alarms without improving the ranking.
The 7-dataset KDD'26 reproduction is no longer hypothetical: a 50-trial,
six-detector run ranks **SEED/STEPD at 2.12** and **HDDM-A at 3.87**,
reproducing the paper's tables. The ABCD/ABCD(X) arm is cited from the paper
rather than re-run; the full 11-seed matrix and detector tuning are still next.

Honest caveat: the main recovery experiments use one deliberately simple synthetic
stream family; the detector check adds one real dataset (E17) but still has no
scheduled-retrain baseline and no label delay — existence evidence and a first map,
not yet a field-wide claim. Real datasets beyond E17, the full baseline ladder and
the Delay Tax are the next step (below).

---

## Where this is going

1. **DF-H — the ultimate goal.** A two-stage detector that combines a label-free
   deep monitor (SSL loss, embedding shift, head disagreement) with supervised
   confirmation (SEED/STEPD/ADWIN/HDDM-A). The target is **top-2 on the KDD'26
   benchmark**, with wins on feature-permutation/filtering and recurring drift,
   and no worse false alarms than ADWIN.
2. **The signal gate.** Before any benchmark run, measure whether the deep
   signals actually beat the raw error-rate baseline on held-out data. If they
   do not, the deep track falls back to a classical feature monitor.
3. **The Delay Tax benchmark.** Fixed, stochastic, partial, selective and batch
   labels; DF-H vs the strong scalar detectors; lemmas 1–3 measured. This is the
   novel contribution and the part no published benchmark covers.
4. **The final report.** Re-centred on the detector: measurement lab → DF-H
   benchmark → delay-tax results → minimal system/demo.

The older breadth-first experiments (E05, E08–E13) are deferred; E15 is replaced
by E21. The full plan is [Roadmap v2](docs/Reports/Roadmap-v2.md).

---

## Explore the vault

Everything lives in `docs/`, organised in two phases:

| | What you'll find |
|---|---|
| **Phase 0 — Research** | [Background notes](docs/Phase0:Research/notes/01-streaming-learning.md) (read in order), [literature review](docs/Phase0:Research/literature_review/index.md) with per-paper notes, and the [idea & scope](docs/Phase0:Research/idea/synopsis.md) |
| **Phase 1 — Experimentation** | The [thesis](docs/Phase1:Experimentation/thesis.md) (the argument), [Exp-01 write-up](docs/Phase1:Experimentation/exp01-deepriver-vs-frozen-sea.md) (the pilot), [Exp-02 write-up](docs/Phase1:Experimentation/exp02-seed-sweep-sea.md) (the 11-seed sweep), and the [Ch1 protocol](docs/Phase1:Experimentation/streaming-vs-batch-metrics.md) |
| **Results** | The [global results page](docs/Reports/Results.md) — the consolidated bottom line across every concluded experiment, with the [milestone reports](docs/Reports/Sept9.md) behind it |

Code lives in `experiments/` (one folder per experiment + a `scripts/` and
`figures/` home for runners and chart generators); data artifacts (`results.json`)
and figures are checked into the repo, so every claim in the docs traces to a
number in a file. The documentation site (mkdocs) mirrors this content at
[docs/index.md](docs/index.md).
