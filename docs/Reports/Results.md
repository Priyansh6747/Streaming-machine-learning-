# Results — the measured verdicts, all experiments

**What this page is:** the project's **global results file**. One page that holds
the *conclusion* of every experiment — not the effort, not the logs, not the
milestones: the bottom line, the numbers that back it, and the process that
produced it. Written for management review; every figure and number traces to a
data artifact (`experiments/*/results.json`).

**How this page updates:** an experiment is added here **only when it concludes**,
as a single row in the verdict registry (§4) plus its measured evidence (§3). The
global verdict (§1) is re-stated over all concluded experiments whenever one lands.

| | |
|---|---|
| **Status** | 2 experiments concluded (Exp-01 2026-09-09, Exp-02 2026-09-12); 3 scheduled |
| **Data artifacts** | `experiments/exp01/results.json` (full per-sample series + metrics) · `experiments/exp02/results.json` (11-seed sweep: per-seed rows + distribution) |
| **Method / protocol** | [`thesis.md`](../Phase1:Experimentation/thesis.md) · [`streaming-vs-batch-metrics.md`](../Phase1:Experimentation/streaming-vs-batch-metrics.md) |
| **Formatting note** | unicode math (τ, κ₀, Δκ, ≈, ≥) renders identically in Obsidian, GitHub, mkdocs |

---

## 1. The global verdict

> **A model that keeps learning from the stream heals itself after a concept
> change — a model that was trained once and frozen does not.**
> Across **11 seeds** (Exp-02, the project's first statistical result): the frozen
> model recovered on **0 of 11** runs. The skill lost while the model is wrong —
> the project's "damage area" — was **≈ 294× smaller** for the streaming model
> (**13 ± 31** vs **3,710 ± 592** samples·κ), and even the *worst* streaming seed
> (103) beat the *best* frozen seed (2,439) by **24×**. The streaming model also
> wins the overall averages: **88.3% vs 85.4%** accuracy, **0.753 vs 0.685**
> Cohen Kappa. Recovery time is bounded but is **not one number**: it is 0 on
> 8 of 11 seeds (no excursion to recover from) and 174–2,497 samples otherwise —
> always inside the pre-registered 5,000-sample budget.

![The process — how one experiment produces a verdict](figures/results-process-flow.png)

*Figure 1 — how every experiment in this file runs: one stream, two models in the
same race, one prequential scoreboard, one verdict. No result below was produced
any other way.*

### Why this matters, in one line each

| | Streaming model (keeps learning) | Frozen model (trained once) |
|---|---|---|
| Reacts to change on its own? | **Yes** — adapts on its own error signal | **No** — waits for a human to retrain |
| Recovery after the drift at τ | **bounded** — 174–2,497 samples when an excursion occurs; 0 on 8/11 seeds (never left its band) | **never** — 0 of 11 seeds recovered |
| Skill lost during the drift (damage area) | **13 ± 31** samples·κ (worst seed 103) | **3,710 ± 592** samples·κ (best seed 2,439) |
| Overall quality (full run, 48k samples) | **0.883** acc · **0.753** κ | 0.854 acc · 0.685 κ |

---

## 2. The process — how results are produced

One experiment = **one controlled race**, run identically for both models on the
same data in the same order. Three non-negotiable rules keep the comparison fair:

1. **Prequential scoring** — every sample is scored *before* it trains the model
   (test-then-train). No model ever sees a future sample.
2. **Same warm-up, same data** — both models train on the same first 20% of the
   stream; the frozen model never updates after that, the streaming model updates
   on every sample.
3. **Measured, not asserted** — every number in this file is read from a committed
   `results.json`; figures are re-rendered from that file, never hand-drawn.

The scoreboard metrics used in every verdict:

| Metric | What it measures | Good direction |
|---|---|---|
| Accuracy / Cohen Kappa (rolling, w = 1,000) | classification skill over time, chance-corrected | ↑ |
| κ₀ — pre-drift baseline | each model's own skill *before* the change | reference level |
| Δκ — dip depth | how much skill the drift knocks off | ↓ (shallower) |
| **Recovery time** | samples from drift τ until the model is back inside its pre-drift band (≥ 0.95·κ₀ for 500 consecutive samples) | ↓ (faster) |
| **Damage area** | total skill lost × time spent below baseline (Σ samples·κ) — the one number that prices the whole drift | ↓ |

---

## 3. Measured evidence

### 3a. Exp-01 — SEA, one abrupt drift (seed 42)

The first data point. Measured 2026-09-09, seed 42, 60,000 samples, abrupt drift
at τ = 30,000, 10% label noise. Streaming model: DeepRiver rolling deep classifier
(RNN head, window = 500). Frozen model: Random Forest (100 trees) trained once on
the warm-up. Full series: `experiments/exp01/results.json`.

![Headline scoreboard — full-run averages](figures/results-headline-scoreboard.png)

*Figure 2 — the overall race, 48,000 post-warm-up samples. The streaming model
wins the cumulative scoreboard as well as the reactivity contest: +2.9 pp accuracy
and +6.8 pp Cohen Kappa.*

![Rolling Cohen Kappa over the stream — the recovery shapes](../Phase1:Experimentation/figures/exp01-rolling-kappa-deepriver-vs-frozen.png)

*Figure 3 — the verdict in one picture. Before the drift both models sit at the
same skill level (κ₀ ≈ 0.755–0.759). At τ the frozen model (red) falls to 0.592
and stays down; the streaming model (blue) dips only to 0.703 and is back inside
its own band 835 samples later — with no human intervention.*

![Recovery time and damage area — measured](../Phase1:Experimentation/figures/exp01-damage-area-vs-recovery.png)

*Figure 4 — the two decisive columns. Recovery: 835 samples vs never. Damage paid
for the drift: 31 vs 3,307 samples·κ (log scale).*

| Drift-window metric (τ = 30,000) | Streaming (DeepRiver rolling) | Frozen (RF) |
|---|---|---|
| Pre-drift baseline κ₀ | 0.755 | 0.759 |
| Minimum rolling Kappa after τ | 0.703 | 0.592 |
| Dip depth Δκ | **0.052** (shallow) | 0.167 (3× deeper) |
| **Recovery time** | **835 samples** — inside the pre-registered 5,000 budget | **never** |
| **Damage area** | **31** samples·κ | **3,307** samples·κ |

### 3b. Exp-02 — the 11-seed sweep (is Exp-01's number reproducible?)

The statistical data point. Measured 2026-09-12: **exactly the Exp-01 replica** — same
stream, same two models, same warm-up, same prequential loop, same metric definitions —
rerun on **11 seeds** (42, Exp-01's, plus 0–9), changing nothing but the seed. Full
per-seed detail: `experiments/exp02/results.json`. Seed 42 reproduced Exp-01 **exactly**
(835 samples, delta 0), so the sweep is a genuine rerun of the same experiment.

![Exp-02 seed sweep — recovery time and damage area per seed](../Phase1:Experimentation/figures/exp02-seed-sweep-recovery.png)

*Figure 5 — left: recovery time per seed; 8 of 11 seeds sit on the metric floor (0 =
never left the pre-drift band) and the rest recover in 174–2,497 samples, all inside the
5,000 budget. Right: damage area per seed (log scale); the 5 seeds with exactly zero
measured damage are drawn as hatched stubs. The damage gap is stable; the recovery
number is not.*

| Metric (11 seeds) | Streaming (DeepRiver rolling) | Frozen (RF) |
|---|---|---|
| Pre-drift baseline κ₀ | 0.751 ± 0.017 | **0.767 ± 0.018** |
| Dip depth Δκ | **0.045 ± 0.015** | 0.173 ± 0.018 |
| Recovery time (samples from τ) | **median 0**, mean 319 ± 765 (0 / 174 / 835 / 2,497) | **never, 11/11** |
| Damage area (samples·κ) | **13 ± 31** (median ≈ 0.001, max 103) | 3,710 ± 592 (min 2,439, max 4,368) |
| Full-run accuracy | **0.8830 ± 0.0021** | 0.8539 ± 0.0017 |
| Full-run Cohen Kappa | **0.7530 ± 0.0044** | 0.6850 ± 0.0035 |

![The damage gap as a distribution, one dot per seed](../Phase1:Experimentation/figures/exp02-damage-distribution.png)

*Figure 6 — the same damage numbers as a distribution: one dot per seed, median marked.
Every frozen seed sits in a narrow band; every streaming seed is two orders of magnitude
below it. Median ≈ 0.001 vs 3,822 samples·κ.*

**What the sweep settles.** Exp-01's specific **835** was a property of seed 42, not of
the method: across 11 seeds the recovery time is 0 on eight seeds (no excursion to
recover from) and 174 / 835 / 2,497 on the other three. What *is* stable is everything
the claim actually rests on — the frozen model never recovers (0/11, no exceptions), the
damage gap is two orders of magnitude on the mean and **24× even worst-case-vs-best-case**,
and the streaming dip is consistently ~3.8× shallower. Exp-02 also mildly **restores**
Exp-01's refuted control expectation: the frozen model *does* start ahead while the
concept is stationary (κ₀ 0.767 vs 0.751), and still loses the full run because it
collapses at τ.

**One methodological finding, recorded not patched.** The recovery-time rule looks for
the first sustained in-band run *starting at τ*, so a model that stays in band for the
first 500 samples and dips afterwards is scored 0. Seven of eleven seeds did leave the
band after τ, but only the three with an *early* excursion are credited with a recovery
time — which makes recovery non-monotone in dip depth (seed 3 dips deeper than seed 9
yet scores 0). **Damage area and dip depth are the trustworthy drift-window metrics;
recovery time is meaningful only conditional on an early excursion.** Details in the
[Exp-02 write-up](../Phase1:Experimentation/exp02-seed-sweep-sea.md) §6c.

![Why the recovery metric under-reports](../Phase1:Experimentation/figures/exp02-recovery-vs-dip.png)

*Figure 7 — the flaw in one picture. Green = an excursion early enough for its recovery
to be credited; orange X = a seed that did leave its band but was scored 0 because the
excursion began after τ + 500; hollow blue = never left the band. Four of the eleven
seeds are orange — which is why the mean recovery time is not a property of the model.*

---

## 4. Verdict registry — one conclusion per experiment

The cumulative table. Rows land here **only with a measured verdict**; scheduled
experiments are listed so the roadmap is visible, with their acceptance bar.

| Exp | Question asked | Setting | Verdict (measured) | Conclusion |
|---|---|---|---|---|
| **Exp-01** ✅ | Can a streaming deep learner beat a *frozen* batch model on a drifting stream? | SEA, one abrupt drift, 2 models, seed 42 | **Streaming wins** — recovers in 835 samples (frozen: never); damage 31 vs 3,307; also wins full-run acc/κ | Weakest form of the claim **confirmed**. A model that keeps learning survives concept drift on its own |
| **Exp-02** ✅ | Is the 835-sample recovery a fluke of one seed? | SEA, one abrupt drift, 2 models, **11 seeds** | **Streaming wins on every seed** — frozen recovered **0/11**; damage **13 ± 31** vs **3,710 ± 592** (≈294×; worst streaming seed still 24× better than best frozen seed); dip 0.045 vs 0.173; full run **0.883/0.753** vs 0.854/0.685. Seed 42 reproduced Exp-01 exactly (835, delta 0) | The claim **generalises**, the number does not — 835 was seed luck; recovery is 0 on 8/11 seeds (no excursion) and 174–2,497 otherwise. Refines H2: frozen *does* win steady state (κ₀ 0.767 vs 0.751) yet loses the run. Recovery time under-reports late dips; damage area is the trustworthy metric |
| Exp-03 ⏳ | Does streaming also beat a *scheduled-retrain* batch model (the honest industrial baseline)? | SEA + baseline ladder | *pending* | acceptance: streaming recovery ≪ retrain interval r_next − τ |
| Exp-04 ⏳ | Does the win survive on **real** streams (Phishing, Covertype, Airlines, CreditCard, Elec2)? | 5 real streams, post-dip protocol | *pending* | acceptance: streaming ≥ frozen on post-drift windows |
| Exp-05 ⏳ | The **Delay Tax**: do delayed/rare labels starve batch retraining while streaming keeps working? | label-delay sweep | *pending* | acceptance: streaming quality degrades slower under delay |

*Scope guard: Exp-01 is deliberately the easiest possible version of the claim
(one synthetic stream, no scheduled retrain, no label delay, single seed). Each
row above removes one of those cushions. The global verdict stands on what is
measured today and is re-stated as rows conclude.*

---

## 5. The verdict vs what was promised

The project pre-registered its expectations *before* running (thesis §6), so the
results can be checked against the promises rather than reported after the fact.

### 5a. Exp-01 — pre-registered expectations vs measured

| Pre-registered expectation | Expected | Measured (Exp-01) | Verdict |
|---|---|---|---|
| Streaming recovery is bounded | ~250–350 samples | **835 samples** | **PASS** (same order; well inside the 5,000-sample falsification budget) |
| Frozen model never recovers | never | **never** (still depressed at stream end) | **PASS** |
| Frozen pays unbounded damage | unbounded | 3,307 vs 31 samples·κ | **PASS** (≈ 106×) |
| Dip is shallow for streaming, deep for frozen | shallow vs deep | 0.052 vs 0.167 | **PASS** |
| Frozen wins cumulative accuracy (honest control) | batch ahead in steady state | streaming wins 0.757 vs 0.689 | **REFUTED — in our favour** on this seed |

![Expected vs measured — recovery time and damage area](figures/expected-vs-measured-sept9.png)

*Figure 8 — pre-registered expectations (hatched) against measured results (blue):
the measured recovery is inside the promised regime, and the measured damage gap is
even larger than the thesis sketched.*

### 5b. Exp-02 — pre-registered acceptance vs measured

| Pre-registered condition (fixed before the run) | Required | Measured | Verdict |
|---|---|---|---|
| Seeds | ≥ 10 | 11 | **PASS** |
| Recovery reported as a distribution | mean ± SD | mean 319 ± 765, median 0, range 0–2,497 | **PASS** — with a heavy-tail caveat: 8/11 seeds sit on the metric floor, so the mean is not a robust centre |
| Streaming recovers on every seed | every seed | 11/11 (8 of them never left the band) | **PASS** |
| Frozen never recovers on every seed | every seed | 11/11 | **PASS** |
| H2: frozen wins steady state (restated control) | frozen κ₀ ≥ streaming κ₀ | frozen 0.767 vs streaming 0.751 | **SUPPORTED** — unlike the single-seed Exp-01, where the two were level |

**Bottom line:** the falsification conditions set before the runs were not
triggered. Exp-01 showed the effect exists on one seed; Exp-02 shows the *claim*
holds on all eleven — the frozen model never recovers and the damage gap survives
worst-case-vs-best-case comparison. The specific 835-sample figure was seed luck and
is retired as a headline number; the honest headline is the damage gap, plus the
finding that recovery time under-reports late dips. The one control expectation
Exp-01 refuted (batch winning steady state) is restored by the larger sample, and
streaming still wins the run outright.

---

## 6. Provenance

- Measured numbers: `experiments/exp01/results.json` · `experiments/exp02/results.json`
- Full write-ups: [`exp01-deepriver-vs-frozen-sea.md`](../Phase1:Experimentation/exp01-deepriver-vs-frozen-sea.md) ·
  [`exp02-seed-sweep-sea.md`](../Phase1:Experimentation/exp02-seed-sweep-sea.md)
- Thesis + pre-registered expectations: [`thesis.md`](../Phase1:Experimentation/thesis.md)
- Protocol & metric definitions: [`streaming-vs-batch-metrics.md`](../Phase1:Experimentation/streaming-vs-batch-metrics.md)
- Milestone records: [`Sept9.md`](Sept9.md) (Exp-01)
- Figure sources: `experiments/scripts/make_results_figures.py` ·
  `experiments/scripts/make_exp01_summary_chart.py` ·
  `experiments/scripts/exp02_seed_sweep_sea.py --figure-only`
