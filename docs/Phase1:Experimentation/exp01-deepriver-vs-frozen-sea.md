# Exp-01 — DeepRiver rolling deep classifier vs frozen batch on SEA

**Series:** Ch1 / thesis pilot ("streaming vs non-streaming on real metrics"), first
atomic experiment.

**Aim → libs → method → calculations → results → conclusions** as requested; §5–6
were pre-registered placeholders and are now filled from the first run
(`experiments/exp01/results.json`).

**Status:** ✅ Run on 2026-09-09 (seed 42, RTX 3050 Laptop GPU). Code:
`experiments/scripts/exp01_deepriver_vs_frozen_sea.py`; data artifact:
`experiments/exp01/results.json`; figure: `figures/exp01-rolling-kappa-deepriver-vs-frozen.png`.

!!! warning "Metric correction (E01, 2026-11-12)"
    The damage-area numbers on this page use the legacy τ→recovery integral, which
    truncated the streaming arm. The corrected 11-seed result is `damage_ref`
    **74.8 vs 515.0** samples·κ (**≈6.9×**) and `damage_full` **321.4 vs 3,472.9**
    (**≈10.8×**). The legacy 294×/24× are retired. See
    [E01 — Metric v2 correction](../Reports/E01-metric-v2.md).

**Formatting note:** math is written in Unicode (τ, κ₀, Δκ, ≈, ≥) so it renders
identically in Obsidian, GitHub, and the mkdocs site — no LaTeX pipeline required.

---

## 1. Aim

First empirical check of the thesis (see [`thesis.md`](thesis.md) §1) using a
**deep streaming learner**:

> Can a DeepRiver rolling classifier — a bounded-memory model that retrains on a
> recent window of data — beat a **frozen** batch model (trained once, never
> updated) on a drifting stream, when measured prequentially?

This is deliberately the *weakest* form of the claim: no scheduled-retrain batch, no
delay, no oracle tuning, no statistics. If DeepRiver cannot show the expected
recovery behaviour here, the bigger experiments are pointless. Small and atomic by
design: **one synthetic stream, one injected drift, two models, three metrics.**

### Hypotheses (pre-registered)

- **H1 — Recovery:** after the drift at τ, DeepRiver's rolling Kappa re-enters its
  pre-drift band within a bounded number of samples; the frozen RF's does not recover
  before the stream ends.
- **H2 — Steady state (expected loss):** before τ, frozen RF (trained on the whole
  warm-up) has higher Kappa than the online net. This is expected and *not* the
  claim — it is the honest control half of the picture.
- **Falsification:** if DeepRiver does not recover within a generous window
  (pre-registered: rolling Kappa back to ≥ 0.95 · κ₀ within 5,000 samples of τ), the
  pilot is inconclusive and the result is recorded as such rather than tuned away
  silently.

---

## 2. Libraries

Focus: **DeepRiver** (deep learning on data streams, River + PyTorch).

| Package | Role | Notes |
|---|---|---|
| `deep-river` 0.3.4 | streaming deep models (`RollingClassifier`) | the subject of the experiment |
| `river` 0.25.0 | metrics (`Accuracy`, `CohenKappa`, `Rolling`), prequential loop | pinned by deep-river (`<0.26`); see `notes/03-river-library.md` |
| `torch` 2.5.1 | neural net backend | CUDA build; run on the RTX 3050 (Laptop) GPU |
| `scikit-learn` 1.9.0 | frozen batch baseline (`RandomForestClassifier`) | the opponent |
| `numpy` | stream generation, numerics | |
| `pytest` | not used by us | required at import time by deep-river 0.3.4 (its `utils.estimator_checks` imports it) — packaging bug, so it's a project dep |

Environment: repo-root uv project, **Python 3.12** (deep-river's torch `<2.6` has no
cp313 wheels, so the repo pins `>=3.12,<3.13`). Run:
`uv run experiments/scripts/exp01_deepriver_vs_frozen_sea.py` — the script
auto-selects CUDA if present (`--device cuda|cpu` to force). Exact versions in
`pyproject.toml` / `uv.lock`.

---

## 3. Method

### 3a. Stream (SEA, one abrupt drift)

- Generator: SEA concepts — 3 numerical features drawn once, continuously, from
  U(0, 10)³; only features 0+1 matter (`x₀ + x₁` vs a threshold); binary labels.
- Length: **60,000 samples**. Label rule: `x₀ + x₁ > 8` (SEA concept 0) for
  t < 30,000, then `x₀ + x₁ > 9` (SEA concept 1) → **abrupt drift at τ = 30,000**.
- `noise = 0.1` (10% label noise, so the problem is non-trivial); balanced classes
  by construction; fixed seed 42 (single seed for Exp-01; the ≥10-seed sweep is a
  later, separate experiment).
- Implementation note: river 0.25's `synth.SEA` reseeds its RNG per generator, so
  concatenating two variants *replays identical feature vectors* under the new
  label rule (train-set memorisation confound). The script therefore draws one
  uninterrupted feature stream and switches only the label rule at τ — identical
  semantics to the intended SEA abrupt drift.

### 3b. Models

| Role | Model | Warm-up | Update rule |
|---|---|---|---|
| Streaming | DeepRiver `RollingClassifier` over a small **RNN head** (RNN 3 → 16 hidden → Linear → 2 logits; Adam, lr 1e-3, cross-entropy; one optimiser step per sample) | learns from sample 1 onwards | per sample: predict → score → `learn_one` on a **rolling window of 500** (bounded memory; old samples fall out → drift adaptivity) |
| Frozen batch | sklearn `RandomForestClassifier(n_estimators=100)` | fit once on first **12,000 samples (20%)** | **never** updated after warm-up |

- **Architecture note:** the spec'd "MLP 2 → 16 → 2" cannot be expressed in
  DeepRiver's `RollingClassifier` — its feature-incremental layer handling requires
  the module's first layer to take `n_features` units, and it feeds the net a
  temporal window. The RNN head is the architecture DeepRiver documents for
  rolling models; capacity (16 hidden) and lr are unchanged.
- DeepRiver 0.3.4 bug worked around in the script: `_learn` runs the forward
  before `module.train()`, so a predict-first (eval-mode) cuDNN RNN backward
  crashes; the wrapper forces train mode before `learn_one`.
- Window size and learning rate are the two tuning knobs; pre-registered defaults
  (window_size = 500, lr = 1e-3) may be swept in a follow-up, *not* silently
  fitted to make H1 pass.

### 3c. Evaluation protocol

Standard prequential (test-then-train) loop over the **post-warm-up** region
(t ∈ [12,000, 60,000)) for both models, on identical samples:

```
for x, y in stream[12k:60k]:
    y_pred = model.predict_one(x)     # frozen RF: predict only, no learning
    metric.update(y, y_pred)
    if model is streaming:
        model.learn_one(x, y)         # DeepRiver: windowed update
```

Metrics recorded **per sample**: accuracy, Cohen Kappa, rolling Cohen Kappa
(w = 1,000). Runtime (seconds) and peak process RSS measured around the whole loop.

---

## 4. Calculations

Let κ(t) be rolling Cohen Kappa at sample t (window w = 1,000).

- **Pre-drift baseline:** κ₀ = mean of κ(t) over the 1,000 samples immediately
  before τ (i.e. t ∈ [29,000, 30,000)).
- **Dip depth:** Δκ = κ₀ − min κ(t) over t ∈ [τ, τ + 5,000).
- **Recovery time:** the smallest r such that κ(t) ≥ 0.95 · κ₀ for 500 consecutive
  samples within [τ, τ + 5,000]; reported as samples from τ. Frozen model: "never"
  if the stream ends before the band is re-entered.
- **Damage area** (the thesis's money metric, thesis.md §3):
  `area = Σ (κ₀ − κ(t))` summed over t = τ … t_recover, in units of samples × Kappa;
  frozen area measured to stream end when it never recovers.
- **Full-run metrics:** prequential accuracy and Cohen Kappa over the whole
  post-warm-up region (cumulative).
- **Cost:** wall-clock seconds, samples/second, and memory (max RSS − baseline RSS).

---

## 5. Results

Measured, seed 42, first run. Full per-sample series and metrics:
`experiments/exp01/results.json`. Pre-registered expectations are in *italics*
after each table; deviations are stated, not tuned away.

### 5a. Full-run metrics (post-warm-up, cumulative, n = 48,000)

| Model | Accuracy | Cohen Kappa | Runtime (s) | Samples/s | Peak RSS (MB) |
|---|---|---|---|---|---|
| DeepRiver rolling | **0.885** | **0.757** | 99.1 (CUDA) | 484 | 1074 |
| Frozen RF | 0.856 | 0.689 | 0.17 (fit only) | — | — |

*Expectation was frozen RF winning cumulative accuracy/Kappa (H2).* **Deviated:**
DeepRiver wins both — its pre-drift steady state is *not* below the frozen RF
(κ_dr ≈ 0.755 vs κ_fr ≈ 0.759), and it keeps that level after the drift while the
frozen model drops. The per-metric win is therefore even cleaner than
pre-registered. Runtime is for the DeepRiver loop on the RTX 3050 (CUDA); the
same 48k loop takes ~1,089 s (44 samples/s) on CPU, so GPU is effectively
required at this window size. RSS is peak *process* RSS (both models + CUDA
context in one process), not a per-model delta.

![Rolling Cohen Kappa over the stream — DeepRiver dips and recovers, frozen RF dips and stays down](figures/exp01-rolling-kappa-deepriver-vs-frozen.png)

*Figure 1 — rolling Cohen Kappa (w = 1,000), drift at τ = 30,000 marked. The
blue (DeepRiver) dip is shallow and self-healing; the red (frozen RF) dip is
deeper and never heals before the stream ends.*

### 5b. Drift-window metrics (τ = 30,000)

| Model | κ₀ | Min κ | Dip depth Δκ | Recovery time (samples from τ) | Damage area (samples·κ) |
|---|---|---|---|---|---|
| DeepRiver rolling | 0.755 | 0.703 | 0.052 | **835** (bounded) | 31 |
| Frozen RF | 0.759 | 0.592 | 0.167 | **never** (≥ 5,000) | 3,307 |

*Expectation (H1): DeepRiver dips then recovers within a few thousand samples;
frozen RF dips and stays depressed.* **Confirmed** — DeepRiver is back inside its
pre-drift band (≥ 0.95 · κ₀ for 500 consecutive samples) 835 samples after τ,
within the pre-registered 5,000-sample budget; the frozen RF never re-enters the
band in the window and is still depressed at stream end. Damage area differs by
two orders of magnitude (31 vs 3,307 samples·κ) — the recovery-time and
damage-area columns are the win card, as pre-registered.

### 5c. Headline verdict

- H1 (bounded recovery): **pass** — DeepRiver re-enters its pre-drift band 835
  samples after τ; frozen RF does not recover before the stream ends.
- H2 (batch wins steady state): **refuted** — DeepRiver matches frozen RF
  pre-drift (κ 0.755 vs 0.759) and wins cumulative Kappa 0.757 vs 0.689. The
  honest-control half of the pre-registration did not materialise; the streaming
  model is not penalised in steady state on this problem.

---

## 6. Conclusions

Written after the run (seed 42, single replicate — see caveats below).

- **First DeepRiver-backed data point for thesis.md §6b:** a streaming deep
  learner (RollingClassifier, window 500) shows **bounded recovery** (835 samples)
  on an abrupt drift where a frozen batch model never recovers within the stream.
  The win is per-metric (recovery time 835 vs never, damage area 31 vs 3,307
  samples·κ), and — beyond the pre-registration — DeepRiver also wins cumulative
  accuracy/Kappa because its steady state matches the frozen RF pre-drift instead
  of trailing it.
- **Interpretation against the literature** (`literature_review/index.md` §1):
  consistent with Read 2012 / Chen 2019 in that there is no universal winner —
  context matters — and with Dasari 2026's "the learner matters more than the
  trigger": the rolling window's continuous refit was what recovered, no detector
  needed. The deep net *did not* pay a steady-state tax on this simple problem,
  which is the part most likely to change on harder streams.
- **Falsification path honoured:** the pre-registered recovery budget was
  0.95 · κ₀ within 5,000 samples; H1 passed at 835. The result was *not* tuned —
  window (500), lr (1e-3) and architecture (16 hidden) are the pre-registered
  defaults; the two model changes from spec (single continuous stream; RNN head)
  were forced by river/DeepRiver mechanics and are documented in §3a–3b.
- **Caveats, recorded honestly:**
  - Single seed, single replicate — the ≥10-seed sweep (with the confidence
    interval on recovery time) is the follow-up; this pilot is existence
    evidence, not statistics.
  - Cost is real: 484 samples/s on an RTX 3050 (1,089 s / 44 samples/s on CPU)
    for a 2.3k-parameter net — per-sample windowed refits are the slow path, and
    the cost column stands even though it is unfavourable. At this scale the GPU
    is not optional.
  - Peak RSS 1,074 MB includes the CUDA context; the pure model footprint is far
    smaller. A CPU-only measurement belongs to the cost-focused experiment.
- **Next knobs, as separate atomic experiments:** seed sweep; harder drift
  (SEA concept 3 threshold 9.5, or a real stream); window-size/lr sweep; DeepRiver
  per-sample `Classifier` vs this rolling one (the windowed-refit vs per-sample-SGD
  question the literature flags); a scheduled-retrain batch baseline.

---

## 7. Cross-references

- Thesis + expected-result sketches: [`thesis.md`](thesis.md) (§6a–6c, §6f falsifiers).
- Ch1 protocol (models/datasets/statistics): [`streaming-vs-batch-metrics.md`](streaming-vs-batch-metrics.md).
- Literature (direct comparisons, scope-defining evidence): [`Phase0:Research/literature_review/index.md`](../Phase0:Research/literature_review/index.md) §1 subgroup.
- Scope guard note: the old `notes/08-scope.md` excludes deep learning — this
  experiment series is a deliberate, teacher-relevant extension of it.
