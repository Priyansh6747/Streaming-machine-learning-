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
2. **The evidence:** 11 seeds. Frozen recovered **0/11**. Streaming recovered 11/11
   inside the pre-registered 5,000-sample budget.
3. **The honest headline is not 294×.** It is: *frozen never recovers on any seed,
   and the worst streaming seed still beats the best frozen seed by ~24×* — and both
   figures are **upper bounds** on the gap (see §7.1).
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

### 4c. Exp-02 — 11 seeds, 42 + 0–9 (2026-09-12, 1,127 s total ≈ 102 s/seed)

| Metric | Streaming | Frozen |
|---|---|---|
| κ₀ | 0.751 ± 0.017 | **0.767 ± 0.018** |
| Min rolling κ after τ | 0.706 ± 0.015 | 0.594 ± 0.013 |
| Dip depth Δκ | **0.045 ± 0.015** | 0.173 ± 0.018 (~3.8× deeper) |
| Recovery time | median **0**, mean 319 ± 765; values 0 / 174 / 835 / 2,497 | **never, 11/11** |
| Damage area | **13 ± 31** (median ≈ 0.001, max 103) | **3,710 ± 592** (min 2,439, max 4,368) |
| Full-run accuracy | **0.8830 ± 0.0021** | 0.8539 ± 0.0017 |
| Full-run Cohen Kappa | **0.7530 ± 0.0044** | 0.6850 ± 0.0035 |

**Derived claims:**

- Frozen recovered on **0 of 11** seeds. Streaming **11 of 11** inside budget (of
  which 8 never left the band — see §7.1 on what "0" means).
- Mean damage ratio **≈ 294×** (`3,710 / 13`); **worst-vs-best 24×** (103 vs 2,439).
- Reproducibility cross-check: seed 42 reproduced Exp-01 **exactly** — 835 samples,
  delta 0, same κ₀ (0.755), same dip (0.052), same damage (31.2).
- Acceptance scorecard: **4/4 PASS** (≥10 seeds; distribution reported; streaming
  recovers every seed; frozen never recovers).
- **H2 restored:** on 11 seeds the frozen model *does* lead in steady state
  (0.767 vs 0.751) — and still loses the full run because it collapses at τ.

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
| — | **Recovery-metric revision** | Fix the τ-anchored rule (§7.1) | re-report with a fixed-window definition | pending |
| Ch2 | **The system** | What does continual learning cost as a real service? | running docker-compose stack + cost curves | planned |

**Order matters for the viva narrative:** Exp-03 removes the "your opponent is a
strawman" objection *before* real streams, and the metric revision happens first so
the corrected numbers propagate into everything downstream.

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

### Attack 2 — "Your headline 294× damage gap is inflated." ← *the one to expect*

**Yes. See §7.1 — this is a real defect, and you should raise it yourself.** The
short version: `damage_area` is integrated from τ to the *recovery* point, and on 8 of
11 seeds the recovery rule fires at τ (the metric floor), which collapses the integral
to a single sample. So streaming damage is **truncated** while frozen damage (which
never recovers) is **integrated over the full window**. The 294× and the 24× are both
**upper bounds**. The qualitative claim — frozen never recovers, 0/11 — is untouched.

### Attack 3 — "There is no drift detector anywhere in your detector project."

**Fair.** Exp-01/02 measured recovery from the rolling window alone — no ADWIN, DDM,
EDDM, Page-Hinkley or KSWIN. Two responses: (a) it is a legitimate design (Dasari 2026:
with per-sample incremental updates the retraining *policy* is irrelevant — the learner
does the work, 54 paired comparisons); (b) the detector matrix is RQ1 and is genuinely
not yet run. Don't oversell the second as done.

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

## 7. Two things to fix before you walk in

### 7.1 The damage-area artefact — know this cold

**The defect.** In `experiments/scripts/exp02_seed_sweep_sea.py`:

- `recovery` = the first sample of the first 500-long in-band run starting at τ.
- `damage_area` = `Σ max(κ₀ − κ(t), 0)` for `t` from τ **to `recovery`**.

If the model is already in-band at τ and stays there for 500 samples, `recovery = τ`,
so the integration range is a **single sample** and damage ≈ 0.

**Proof from the committed artifact** (`experiments/exp02/results.json`): seed 1 has a
dip of 0.027 and damage 0.0; seed 3 dips **0.071** and damage 0.0; seed 7 dips 0.053
and damage 0.0. A 0.071 excursion below baseline cannot integrate to zero over any
non-degenerate window. **8 of 11 streaming seeds sit on this floor**, while the frozen
model — which never recovers — always integrates the full window.

**Consequences to state yourself:**

1. The comparison is **asymmetric**: truncated integral (streaming) vs complete
   integral (frozen). Both 294× and 24× are **upper bounds** on the true gap.
2. Correcting the metric will **shrink** the gap. Direction unknown in magnitude —
   the correction needs a rerun, because Exp-02's artifact does not retain the
   rolling series.
3. The docs' claim that "damage area and dip depth are the trustworthy metrics" is
   **too generous** — damage area is *defined in terms of* the same τ-anchored
   recovery rule, so the two are not independent evidence. **Say this before an
   examiner does.**
4. What survives untouched: **frozen recovered on 0 of 11 seeds**, and the streaming
   dip is consistently ~3.8× shallower. Those are the defensible claims.

**The fix** (a few lines, then a rerun at ~102 s/seed ≈ 19 min on the GPU): integrate
damage over a fixed window `[τ, τ+W)` independently of recovery, or from the first
band-exit to recovery. Re-report; do not overwrite the old artifact — keep both so the
change is auditable.

**Why "recovery = 0" is not instant recovery:** it means the model *never left* its
pre-drift band. 0 is the metric floor, not a speed. 8/11 seeds sit on it.

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
| What is the damage area? | The skill lost while the model is wrong: `Σ (κ₀ − κ(t))` over the drift window, in samples·κ. The one number that prices a whole drift. (Mind §7.1.) |
| What would falsify the thesis? | Streaming recovery comparable to a sane retrain interval (`d + a ≥ r_next − τ`); no dip at τ (dataset problem); no significant difference under Friedman/Nemenyi; reactivity win not worth the resource cost. |
| Why is this not just a tutorial? | The gap is specific and open: a standard benchmark now exists for *immediate-label* drift detection (KDD'26), but none for delayed/partial/selective labels, and no formal bounds connect label latency to detection quality. |
| What is the single novel claim? | The **confirmation protocol** — label-free early warning + delayed supervised confirmation. L4 proves label-free cannot confirm; L1 proves supervised cannot be early; so the hybrid is the information-maximal architecture. |
| What if the hybrid fails? | A complete negative result: "we quantified how delayed labels break existing methods, and why the obvious fix is insufficient." Lemmas 1–3 hold regardless. Ch3 cannot fail — which is why it is the bet. |
| Why is it 2b and not 2a? | It is both halves: measurement (Ch1/Ch3) *and* a deployed system (Ch2) — plus the instrumented pipeline already built. It is explicitly not research-only. |
| What is the biggest threat to the project? | Scope creep, not difficulty. The scope guards exist for that reason, and every chapter has a survivable failure mode. |
