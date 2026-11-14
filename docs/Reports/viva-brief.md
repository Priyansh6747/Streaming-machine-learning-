# Viva Brief — Streaming ML / the Delay Tax

**Purpose:** preparation for a progress viva — *what is done, what is next, and why
this is a Type 2b project*. Every number here is traceable to a committed artifact.

**Date prepared:** 2026-09-22 · **Scope as executed:** Phase 0 complete, Phase 1
started (Exp-01, Exp-02 concluded).

> This page is excluded from the published mkdocs site (`exclude_docs`). It lives in
> the vault for your own use and does not appear on GitHub Pages.

---

## 0. If you remember only five things

1. **The claim:** on a drifting stream, all models are equally good *before* the
   change — the difference is what happens *after*. A model that keeps learning
   heals itself; a model trained once and frozen does not.
2. **The evidence:** 11 seeds. Frozen recovered **0/11**. Streaming had a **real
   recovery on 8/11**; on the other three it never left its pre-drift band. The old
   “11/11 recovered” counted the metric floor as a recovery.
3. **The honest headline is not 294×.** It is: *frozen never recovers, and the
   streaming model pays **≈6.9×** less corrected `damage_ref` on the mean
   (74.8 vs 515.0)*. The legacy 294× and the 24× worst-vs-best claim are **retired**;
   on the clean panel the worst streaming seed is **0.74×** the best frozen seed.
   E01 is the authority (see §7.1).
4. **The project type is 2b.** Development = the instrumented experiment pipeline
   (built) **+** the containerised continual-learning service (next). Say the second
   half out loud, unprompted.
5. **The Delay Tax — the headline contribution — is not yet run.** Zero experiments.
   It is the roadmap, and you should present it as such rather than letting an
   examiner discover it.

---

## 1. The 60-second pitch

Almost all ML treats the dataset as a finished object. Streaming ML drops that:
data arrives one sample at a time, forever, too large to store, and the model must
learn incrementally in a single pass in bounded memory. Because the world generating
the data keeps changing, the model must also notice when it becomes wrong and adapt
— that is **concept drift**, and it is the project's problem.

The project has three parts, in increasing novelty:

1. **Baseline (Ch1)** — the rigorous comparison the field lacks: streaming vs
   non-streaming, on *reactivity* metrics rather than average accuracy.
2. **System (Ch2)** — a real containerised continual-learning service, measuring the
   operational costs the literature never reports (adaptation latency, replica
   divergence, backpressure, restart cost).
3. **Contribution (Ch3)** — the **Delay Tax**: production labels arrive late,
   partially, and selectively; the field assumes they arrive instantly and completely.
   Formalised as four lemmas plus a hybrid detector, and measured.

**The thesis mechanism in one line:** batch models update on a **clock**, streaming
models update on **evidence**. Streaming's recovery time is `d + a` (bounded,
data-driven); a scheduled batch model's is `r_next − τ` (luck of phase); a frozen
model's is undefined. Streaming wins whenever `d + a < r_next − τ`, and that ratio is
the quantitative content of the project.

---

## 2. Type compliance — why this is 2b, not a pure research project

> **The rule:** the Major Project may be 2a (full development) or 2b (research +
> development). **Pure research with no development is not allowed.**

This is the first thing an examiner will test, so lead with it.

### 2a. Development that exists today

| Artifact | What it is | Evidence |
|---|---|---|
| Experiment runners | `experiments/scripts/exp01_deepriver_vs_frozen_sea.py`, `exp02_seed_sweep_sea.py` (flags: `--device`, `--require-cuda`, `--limit`, `--figure-only`, `--figures-dir`) | repo |
| Machine-readable results | `experiments/exp01/results.json`, `experiments/exp02/results.json` | repo |
| Figure generators | `make_exp01_summary_chart.py`, `make_results_figures.py`, `make_sept9_report_graphs.py`; a 9-figure suite regenerable from the artifact with no rerun | repo |
| Pinned, reproducible environment | `pyproject.toml` + `uv.lock`, Python pinned `>=3.12,<3.13`, platform-guarded resolution (deep-river's torch `<2.6` has no cp313 wheels) | repo |
| Documentation system | MkDocs Material, 38 pages, MathJax + Mermaid, per-page **and** aggregated PDF export via mkdocs-exporter + headless Chromium | `mkdocs.yml` |
| CI/CD | `.github/workflows/ci.yml` — builds and deploys on push to `main`, PDF export enabled | repo |

**The engineering property to stress:** every claim in the write-ups traces to a
number in a committed `results.json`, and every figure re-renders from that file.
The research is *instrumented* — that is designed, not incidental.

### 2b. Development that is planned (the spine — say this unprompted)

| Component | Status |
|---|---|
| Config-driven, reusable experiment harness (one runner serves Ch1 **and** Ch3) | next — Ch1 exit criterion |
| Event source (Redis Streams or Kafka/Redpanda) | planned, Ch2 |
| Model service: `POST /predict` + learning consumer (the same model serves and learns, one pass) | planned, Ch2 |
| Live drift monitor **as a service**, not an offline script | planned, Ch2 |
| Prometheus metrics + Grafana dashboard | planned, Ch2 |
| Load generator — push the system until it breaks | planned, Ch2 |
| Live drift-and-recovery dashboard as the demo prop | planned, Ch2 |

### 2c. The honest risk, and the line to say

**The risk:** to date the *delivered* artifacts are predominantly research +
simulation. The development that exists is infrastructure *around* research
(pipeline, packaging, docs, CI) — there is no deployed system yet, and `main.py` is
still a stub.

**Say this, don't wait to be asked:**

> "Chapter 1 is the measurement half and it is instrumented — code, artifact,
> figure, one command to reproduce. Chapter 2 is the development half and it is
> next: a containerised continual-learning service where the model that serves is
> the model that learns, with a live drift monitor, Prometheus/Grafana and a load
> generator. The delay-tax experiments are built on the same harness. That is the
> research-plus-development structure, and Chapter 2 is where the development
> weight sits."

**Then be concrete about the near-term development deliverable**, because "planned"
is weak and "built" is strong. If there is time before the viva, the single highest-value
development artifact to have in hand is the **config-driven harness** — one runner
that expresses Exp-01/02 as config rather than as a bespoke script (see §7.2).

---

## 3. Progress to date — grounded in git

| Commit | Date | What landed |
|---|---|---|
| `46d27a3` | 2026-08-31 | first commit |
| `5a3df3a` | 2026-08-31 | MkDocs GitHub Pages deploy workflow |
| `e7a3325` | 2026-08-31 | docs restructured into Phase 0 / Phase 1 |
| `b7ab207` | 2026-08-31 | agent-tooling folder policy |
| `01601fd` | 2026-09-09 | first experiment + thesis + literature review |
| `3c6d09b` | 2026-09-09 | **Exp-01** — DeepRiver rolling vs frozen batch on SEA |
| `7ca63c3` | 2026-09-09 | global Results page |
| `c5e0070` | 2026-09-12 | **Exp-02** — 11-seed sweep + 9-figure suite |

**Two concluded experiments in three days of execution** (2026-09-09 → 09-12), on top
of a complete Phase 0 research base.

| | |
|---|---|
| Phase 0 — research & literature review | ✅ complete |
| Phase 1 — experimentation | 🔬 2 of 5 concluded |
| Exp-01, Exp-02 | ✅ concluded |
| Exp-03 (baseline ladder), Exp-04 (real streams), Exp-05 (Delay Tax) | ⏳ pending |

---

## 4. The numbers board — memorise these

### 4a. Setup (identical in both experiments)

| | |
|---|---|
| Stream | SEA synthetic, one continuous `U(0,10)³` draw, binary labels |
| Length | 60,000 samples · warm-up 12,000 (20%) · **drift at τ = 30,000** |
| Drift | abrupt: rule changes from `x₀+x₁ > 8` to `x₀+x₁ > 9` |
| Noise | 0.1 (10% label noise) |
| Evaluation | prequential (test-then-train), post-warm-up n = 48,000 |
| Rolling metric | Cohen Kappa, window w = 1,000 |
| Recovery rule | κ ≥ 0.95·κ₀ for 500 consecutive samples, within 5,000 samples of τ |
| Streaming model | DeepRiver `RollingClassifier`, RNN 3→16→2, window 500, Adam, lr 1e-3 |
| Frozen model | `RandomForestClassifier(100)`, fit once on the 12,000-sample warm-up |

### 4b. Exp-01 — seed 42 (2026-09-09, RTX 3050)

| Metric | Streaming | Frozen |
|---|---|---|
| Full-run accuracy | **0.885** | 0.856 |
| Full-run Cohen Kappa | **0.757** | 0.689 |
| κ₀ (pre-drift baseline) | 0.755 | 0.759 |
| Min rolling κ after τ | 0.703 | 0.592 |
| Dip depth Δκ | **0.052** | 0.167 |
| Recovery time | **835 samples** | **never** |
| Damage area | **31** samples·κ | **3,307** samples·κ |
| Cost | 99.1 s (484 samples/s) GPU · 1,089 s (44/s) CPU · peak RSS 1,074 MB | 0.17 s to fit |

### 4c. Exp-02 — 11 seeds, 42 + 0–9 (2026-09-12, 1,127 s total ≈ 102 s/seed) *(legacy metric)*

| Metric | Streaming | Frozen |
|---|---|---|
| κ₀ | 0.751 ± 0.017 | **0.767 ± 0.018** |
| Min rolling κ after τ | 0.706 ± 0.015 | 0.594 ± 0.013 |
| Dip depth Δκ | **0.045 ± 0.015** | 0.173 ± 0.018 (~3.8× deeper) |
| Recovery time | median **0**, mean 319 ± 765; values 0 / 174 / 835 / 2,497 | **never, 11/11** |
| Damage area | **13 ± 31** (median ≈ 0.001, max 103) | **3,710 ± 592** (min 2,439, max 4,368) |
| Full-run accuracy | **0.8830 ± 0.0021** | 0.8539 ± 0.0017 |
| Full-run Cohen Kappa | **0.7530 ± 0.0044** | 0.6850 ± 0.0035 |

**Derived claims (legacy metric):**

- Frozen recovered on **0 of 11** seeds. Streaming **11 of 11** only because 8 never
  left the band; under the corrected rule the real-recovery count is **8/11**.
- Mean damage ratio **≈ 294×** (`3,710 / 13`); **worst-vs-best 24×** (103 vs 2,439).
  **Both are upper bounds and are retired by E01.**
- Reproducibility cross-check: seed 42 reproduced Exp-01 **exactly** — 835 samples,
  delta 0, same κ₀ (0.755), same dip (0.052), same damage (31.2).
- **H2 restored:** on 11 seeds the frozen model *does* lead in steady state
  (0.767 vs 0.751) — and still loses the full run because it collapses at τ.

**Corrected by E01 (2026-11-12):** `damage_ref` **74.8 ± 61.8** vs **515.0 ± 97.1**
(**≈6.9×**); `damage_full` **321.4 ± 203.8** vs **3,472.9 ± 559.7** (**≈10.8×**);
worst-vs-best **0.74×**. The frozen-never-recovers result is unchanged. See
[`E01-metric-v2.md`](E01-metric-v2.md).

### 4d. Literature anchors (the fence and the precedent)

| Fact | Number | Source |
|---|---|---|
| Adaptive streaming beats static on real imbalanced drifting data | Kappa **63.65 vs 57.85** over 24 datasets | ROSE, Cano & Krawczyk 2022 |
| Tuned frequent-retrain batch **beats** streaming under delay + rare class | AUC-ROC **92.81 vs 91.02**; delayed fraud AUCPR **23.41 vs 14.23**; ARF **2.2×** slower | Amekoe et al. 2024 |
| Per-sample incremental updates make the retraining policy irrelevant | 54 paired comparisons, no practical difference vs no-retrain | Dasari 2026 |
| Label delay shifts detections by the delay itself | ~10,000-instance delay → ~10,000-instance shift; prequential accuracy < 50% | Gomes et al. 2022 |
| Delayed labels in production malware pipelines | user exposure **+50%**, detection **−75%** | Botacin & Gomes 2024 |
| Comparative statistics convention | ≥10 seeds, mean ± SD, Friedman + Nemenyi, Wilcoxon | Demšar 2006 |

### 4e. The four lemmas (Ch3 — the theory, none of it yet empirically tested)

| # | Lemma | Statement | Novelty |
|---|---|---|---|
| L1 | Reaction floor | `T ≥ τ + L_τ` pointwise — no error-stream detector can alarm before the drift's labels arrive | Novel *as stated*; phenomenon known empirically |
| L2 | Power loss | Sensitivity degrades by `1/√p`; `Δ_min ∝ 1/√(pW)` | Appears genuinely novel — **check Alencar 2026 first** |
| L3 | False-alarm inflation | Batch arrival inflates error-rate variance by `1 + (b−1)ρ` | Most clearly novel — no prior treatment found |
| L4 | Identifiability ceiling | Pure posterior drift is invisible to any label-free detector (KL = 0) | **Not novel** — Žliobaitė 2010. Ours is only the *use* (hybrid motivation) |

**Scoop risk to acknowledge proactively:** Alencar et al. 2026 (DSP) is an
experimental survey of exactly the Ch3 setting; it must be read in full before any
novelty claim. Botacin & Gomes 2024 already quantified label-delay cost on a real
pipeline — so never claim "we show delayed labels hurt".

---

## 5. What's next — the roadmap with acceptance criteria

| # | Experiment | Question | Acceptance bar | Status |
|---|---|---|---|---|
| Exp-03 | **Baseline ladder** | Does streaming also beat a *scheduled-retrain* batch model — the honest industrial baseline? | streaming recovery ≪ `r_next − τ` | next |
| Exp-04 | **Real streams** | Does the win survive on Elec2, Phishing, Covertype, Airlines, CreditCard? | streaming ≥ frozen on post-drift windows | pending |
| Exp-05 | **The Delay Tax** | Do delayed/rare labels starve batch retraining while streaming keeps working? | streaming degrades slower under delay | pending |
| E01 | **Recovery-metric revision** | Fix the τ-anchored rule (§7.1) | fixed-window, panel-shared `damage_ref`; recovery conditional on exit | ✅ **concluded 2026-11-12** |
| E04 | **Detector map** | Which detector works for which drift shape? | balanced score, episode recall, FAR, NDT; 4 shapes × 4 generators × 2 hosts | ✅ **first pass 2026-11-12** (5 seeds; full matrix pending) |
| E17 | **Real-stream detector check** | Does the E04 default survive on real data? | Elec2, label-swap, 4 shapes × 8 detectors × 2 hosts × 5 seeds | ✅ **concluded 2026-11-13** (one dataset; one mechanism) |
| KDD'26 | **External detector benchmark** | Do SEED/STEPD/ABCD beat HDDM-A on seven real datasets? | paper protocol (4 drift types × 2 transitions × 50 trials); compare ranks | 50-trial six-detector run: SEED/STEPD 2.12, HDDM-W 3.56, HDDM-A 3.87; F1 matches the paper; ABCD/ABCD(X) cited from the paper, not re-run |
| Ch2 | **The system** | What does continual learning cost as a real service? | running docker-compose stack + cost curves | planned |

**Order matters for the viva narrative:** Exp-03 removes the "your opponent is a
strawman" objection *before* real streams. The metric revision is **done** (E01,
2026-11-12), so the corrected damage numbers propagate into everything downstream.

**The development half (Ch2) in one line:** event source → model service that serves
*and* learns → live drift monitor → Prometheus/Grafana → load generator → demo
dashboard. Deliverable: the "reality tax" chapter — adaptation latency, replica
divergence, backpressure policy, restart cost.

---

## 6. Attacks, ranked, with pre-empted answers

### Attack 1 — "A frozen model never recovering is trivially true. You built a strawman."

**Correct, and lead with it.** A model with no update mechanism *cannot* recover by
construction — that was the point of Exp-01 as the *weakest* form of the claim, and it
is documented as such. The informative comparison is a **scheduled-retrain** batch
model, and Exp-03 is exactly that: streaming recovery `d + a` against the batch
model's `r_next − τ`, plus the pre-registered prediction that the batch recovery time
scatters uniformly over `[0, k]` as pure phase noise.

### Attack 2 — "Your headline 294× damage gap is inflated." ← *resolved, but know the history*

**Correct — and E01 already fixed it.** The legacy `damage_area` integrated from τ
to the *recovery* point; on 8/11 seeds recovery fired at τ, truncating the streaming
integral. The **294×** and **24×** are upper bounds and are **retired**. E01 reran
the pilot with a fixed-window, panel-shared `damage_ref`:

- `damage_ref`: **74.8 ± 61.8** vs **515.0 ± 97.1** samples·κ → **≈6.9×**
- `damage_full`: **321.4 ± 203.8** vs **3,472.9 ± 559.7** → **≈10.8×**
- worst-vs-best: **0.74×**, not 24×
- frozen recovered **0/11** — unchanged

The corrected claim is smaller and defensible. Do not quote 294× or 24× except as a
retired upper bound. See §7.1 and [`E01-metric-v2.md`](E01-metric-v2.md).

### Attack 3 — "There is no drift detector anywhere in your detector project."

**Was fair; now partly addressed.** E01/02 measured recovery from the rolling window
alone. E04 (2026-11-12, first pass) benchmarks **eight detectors** — ADWIN, KSWIN,
PageHinkley, DDM, EDDM, FHDDM, HDDM-A, HDDM-W — across four generators, four drift
shapes, two hosts, raw and smoothed mistake streams. E17 (2026-11-13) replays the
same eight on real Elec2 records with an injected label swap. **HDDM-A on the raw
signal is the best default on the static host** (balanced score 0.93/0.96/0.96/0.79
for abrupt/gradual/incremental/recurring), but on the adaptive HAT host **ADWIN wins
abrupt (1.00 vs 0.40) and recurring (0.77 vs 0.59)**; the best-host mean is ADWIN
0.935 vs HDDM-A 0.910. So the default is **host-conditional**, not universal.
Recurring remains the hard case; smoothing raises false alarms. Both are 5-seed
first passes, not the full 11-seed tuned matrix. Say that, not “we benchmarked all
detectors”. The design response still stands: per-sample incremental updates make the
retraining *policy* less important (Dasari 2026, 54 paired comparisons). See
[`E04-detector-map.md`](E04-detector-map.md) and
[`E17-real-detector-map.md`](E17-real-detector-map.md).

### Attack 4 — "Your own scope says no deep learning." ← *known inconsistency*

`idea/scope.md` §9 lists "❌ Deep learning / `deep-river`" as out of scope, yet
DeepRiver **is** the subject of both concluded experiments. The Exp-01 write-up
acknowledges this as "a deliberate, teacher-relevant extension". **Fix it properly
before the viva** (§7.2) — an undocumented scope reversal reads as drift, which is
unfortunate for a concept-drift project.

### Attack 5 — "11 seeds on one stream is not Demšar."

**Correct.** The pre-registered bar was ≥10 seeds and it is met, but Friedman +
Nemenyi requires *many datasets × many learners*. What exists is a within-stream
distribution; cross-dataset significance is Exp-04/05. Say "distribution", not
"statistical significance".

### Attack 6 — "The metric definitions are arbitrary." (0.95 band, 500 stretch, 5,000 lookahead)

They were **pre-registered before the run** with an explicit falsification budget —
that is the defence, and it is a good one. The gap is a missing **sensitivity
analysis** over those constants: admit it as a known next step.

### Attack 7 — "You promised the Delay Tax and ran none of it."

**True — zero experiments on the headline contribution.** Frame it as sequencing, not
slippage: Ch1 is the control condition everything is measured against, and it goes
first by design. The lemmas are cheap to verify empirically, and L1's experiment is a
delay sweep on the harness that already exists.

### Attack 8 — "Your streaming model is slower than the batch one."

**True and reported.** 484 samples/s on an RTX 3050 for a 2.3k-parameter net
(44/s on CPU) versus 0.17 s to fit the RF. The cost column stands unfavourable and is
not hidden; Amekoe 2024 (ARF 2.2× slower) shows this is a field-wide tension, not an
implementation accident.

### Attack 9 — "H2 was refuted, then restored. Which is it?"

Not a flip-flop — one seed versus eleven. On seed 42 the two were level
(0.755 vs 0.759), so H2 was recorded as refuted; on 11 seeds the frozen model leads
(0.767 vs 0.751) and H2 is supported. The honest reading: **the frozen model wins the
stationary phase slightly and still loses the run** because it collapses permanently
at τ. The n=1 estimate was noise, which is exactly why Exp-02 exists.

### Attack 10 — "Where is the development? This looks like research only."

§2. Have the development inventory ready, name Chapter 2 concretely, and be honest
that the system is not built yet.

---

## 7. What changed, and what remains

### 7.1 The damage-area defect — resolved by E01 (2026-11-12)

**The defect (history).** The legacy `damage_area` integrated from τ to the
*recovery* point. On 8/11 seeds the recovery rule collapsed to τ (the model never
left its band), so the streaming integral covered a single sample while frozen — which
never recovered — integrated the full window. Both **294×** and **24×** were upper
bounds.

**The fix (E01).** Fixed 5,000-sample window; `κ_ref` shared across the model panel;
recovery only counted after a real band exit. The old artifacts are kept; E01 is a
new artifact.

**Corrected result (11 seeds):**

| Metric | DeepRiver | Frozen RF |
|---|---:|---:|
| `damage_ref` (primary) | **74.8 ± 61.8** (worst 217.9) | **515.0 ± 97.1** (best 295.2) |
| `damage_full` | **321.4 ± 203.8** | **3,472.9 ± 559.7** |
| `recovery_rate` | **8/11** | **0/11** |
| Worst-vs-best | **0.74×** | — |

**What to say:**

1. **294× and 24× are retired.** Corrected mean gap: **≈6.9×** (`damage_ref`) or
   **≈10.8×** (`damage_full`). On the clean panel the worst streaming seed is
   **0.74×** the best frozen seed, not 24×.
2. **Frozen never recovers (0/11)** — unchanged and still the strongest claim.
3. **Streaming had a real recovery on 8/11**, not 11/11; the other three never left
   their band. “Recovery = 0” is the metric floor, not instant recovery.
4. **AUE is secondary.** The timing oracle knows *when* the drift happened but is refit
   on pre-drift data; its `damage_ref` (505) is almost frozen. AUE measures excess over
   a weak oracle, not distance from an upper bound.
5. **Panel composition.** The artifact is the clean DeepRiver + frozen + oracle panel.
   The pre-fix four-model artifact is archived at `results/_archive/e01_5model/`; the
   numbers above supersede it.

**What survives untouched:** frozen recovered on **0/11**; DeepRiver's `damage_ref`
was lower on **11/11** seeds; full-run accuracy/Kappa are **0.883/0.753** vs
**0.854/0.685**. The corrected claim is smaller and defensible.

### 7.2 Close the two consistency gaps

- **Amend the scope properly.** Add a dated note to `idea/scope.md` §9 recording that
  deep learners were deliberately brought in for the Ch1 pilot, with the reason. Also
  consider adding a classical arm (HAT / ARF) to Exp-03 so the thesis claim is not
  deep-net-specific.
- **Build the reusable harness.** Today `exp01` and `exp02` duplicate `drift_metrics`
  "verbatim" by hand. A config-driven runner that expresses both as configuration is
  the Ch1 exit criterion *and* the single most convincing piece of 2b development you
  could have in hand for this viva. It also makes the metric fix a config change.

---

## 8. Likely questions — short answers

| Question | Answer |
|---|---|
| What is concept drift? | `P_t(X,y) ≠ P_{t+Δ}(X,y)` — the joint distribution changes over time. Real drift changes `P(y|X)`; virtual drift changes only `P(X)`. |
| Why is prequential evaluation right? | Its the accepted protocol for ordered streams (Gama 2013). Random splits leak future information; test-then-train scores each sample before it trains. |
| Why Kappa and not accuracy? | Chance-corrected and robust to class imbalance. Accuracy alone overstates learning when temporal persistence creates a strong no-change baseline. |
| Why SEA? | Synthetic streams give **ground-truth drift positions** — without a known τ, detection delay and recovery time are uncomputable. |
| Why a rolling window of 500? | It is the bounded-memory mechanism: old samples fall out, which is what makes the model drift-adaptive. It was a pre-registered default, not fitted. |
| What is `damage_ref`? | The skill lost while the model is wrong, integrated over a **fixed** drift window against a **panel-shared** `κ_ref`, in samples·κ. Current headline: **74.8 vs 515.0** samples·κ (**≈6.9×**). The legacy "damage area" is retired (§7.1). |
| What would falsify the thesis? | Streaming recovery comparable to a sane retrain interval (`d + a ≥ r_next − τ`); no dip at τ (dataset problem); no significant difference under Friedman/Nemenyi; reactivity win not worth the resource cost. |
| Why is this not just a tutorial? | The gap is specific and open: a standard benchmark now exists for *immediate-label* drift detection (KDD'26), but none for delayed/partial/selective labels, and no formal bounds connect label latency to detection quality. |
| What is the single novel claim? | The **confirmation protocol** — label-free early warning + delayed supervised confirmation. L4 proves label-free cannot confirm; L1 proves supervised cannot be early; so the hybrid is the information-maximal architecture. |
| What if the hybrid fails? | A complete negative result: "we quantified how delayed labels break existing methods, and why the obvious fix is insufficient." Lemmas 1–3 hold regardless. Ch3 cannot fail — which is why it is the bet. |
| Why is it 2b and not 2a? | It is both halves: measurement (Ch1/Ch3) *and* a deployed system (Ch2) — plus the instrumented pipeline already built. It is explicitly not research-only. |
| What is the biggest threat to the project? | Scope creep, not difficulty. The scope guards exist for that reason, and every chapter has a survivable failure mode. |
