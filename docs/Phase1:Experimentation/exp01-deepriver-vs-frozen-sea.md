# Exp-01 — DeepRiver rolling deep classifier vs frozen batch on SEA

**Series:** Ch1 / thesis pilot ("streaming vs non-streaming on real metrics"), first
atomic experiment.

**Aim → libs → method → calculations → results → conclusions** as requested; results
and conclusions are **pre-registered placeholders** — this doc is written *before*
the first run so the expectations are on record.

**Status:** 📝 Spec'd, not yet run. Code target: `experiments/exp01/` (repo root, uv
project).

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
| `deep-river` | streaming deep models (`RollingClassifier`) | the subject of the experiment |
| `river` | stream generator (SEA), metrics, prequential loop | DeepRiver's host framework |
| `torch` | neural net backend | CPU build sufficient for this scale |
| `scikit-learn` | frozen batch baseline (`RandomForestClassifier`) | the opponent |
| `numpy` | numerics | |

Environment: the repo-root uv project (Python ≥ 3.13). Install (first run):
`uv add deep-river scikit-learn` — pulls `torch` as a dependency (switch to the CPU
index `https://download.pytorch.org/whl/cpu` if the default wheel is the large CUDA
build). Exact versions pinned in `pyproject.toml` / `uv.lock` after first successful
install (River must be pinned per `notes/03-river-library.md`).

---

## 3. Method

### 3a. Stream (SEA, one abrupt drift)

- Generator: `river.synth.SEA` — 2 numerical features, binary classification, three
  separable concepts.
- Length: **60,000 samples**. Concept: `classification_function=0` for t < 30,000,
  then `classification_function=1` → **abrupt drift at τ = 30,000**.
- `noise = 0.1` (10% label noise, so the problem is non-trivial); balanced classes;
  fixed `random_state` seed (single seed for Exp-01; the ≥10-seed sweep is a later,
  separate experiment).

### 3b. Models

| Role | Model | Warm-up | Update rule |
|---|---|---|---|
| Streaming | DeepRiver `RollingClassifier` (PyTorch MLP: 2 → 16 → 2, ReLU hidden, softmax out; Adam; binary cross-entropy) | learns from sample 1 onwards | per sample: predict → score → `learn_one` on a **rolling window** (bounded memory; old samples fall out → drift adaptivity) |
| Frozen batch | sklearn `RandomForestClassifier(n_estimators=100)` | fit once on first **12,000 samples (20%)** | **never** updated after warm-up |

Window size and learning rate are the two tuning knobs; pre-registered defaults
(window_size ≈ 500, lr ≈ 1e-3) may be swept in a follow-up, *not* silently fitted
to make H1 pass.

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
(w = 1,000). Runtime (seconds) and process-RSS delta measured around the whole loop.

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

Pre-registered expectations are in *italics*; numbers to be filled from the run
(`experiments/exp01/results.json`). No cherry-picking: if H1 fails, that cell says
so.

### 5a. Full-run metrics (post-warm-up, cumulative)

| Model | Accuracy | Cohen Kappa | Runtime (s) | Samples/s | Mem Δ (MB) |
|---|---|---|---|---|---|
| DeepRiver rolling | _*_ | _*_ | _*_ | _*_ | _*_ |
| Frozen RF | _*_ | _*_ | _*_ | _*_ | _*_ |

*Expectation: frozen RF wins cumulative accuracy/Kappa (H2) because it dominates the
30k pre-drift samples; that is fine — the claim is not global accuracy.*

### 5b. Drift-window metrics (τ = 30,000)

| Model | κ₀ | Min κ | Dip depth Δκ | Recovery time (samples) | Damage area (samples·κ) |
|---|---|---|---|---|---|
| DeepRiver rolling | _*_ | _*_ | _*_ | _*bounded*_ | _*small*_ |
| Frozen RF | _*_ | _*_ | _*_ | _*never*_ | _*large/unbounded*_ |

*Expectation (H1): DeepRiver dips then recovers within a few thousand samples; frozen
RF dips and stays depressed — the recovery-time and damage-area columns are the win
card for this experiment.*

### 5c. Headline verdict

- H1 (bounded recovery): **pass / fail / inconclusive** —
- H2 (batch wins steady state): **confirmed / refuted** —

---

## 6. Conclusions

*To be written after the run. Expected shape:*

- If H1 holds: first DeepRiver-backed data point for thesis.md §6b — a streaming
  deep learner shows bounded recovery on a drift where a frozen batch model never
  recovers; per-metric win (recovery time, damage area), *not* global accuracy.
- Interpret against the literature logged in `literature_review/index.md` §1: Read
  2012 / Chen 2019 (no universal winner — context matters), Dasari 2026 (per-sample
  learning dominates policy; expect the *learner* to matter more than the retrain
  trigger). A DeepRiver rolling window is closer to "continuous windowed refit" than
  "per-sample SGD", so how strongly it wins (or whether it does at all) is a real,
  informative question.
- If H1 fails or is inconclusive: recorded honestly; next knobs are window size,
  learning rate, net capacity, or switching to DeepRiver's per-sample `Classifier` —
  as separate atomic experiments, not silent re-tuning.
- Cost column reported even if unfavourable (slow windowed refits are expected and
  are part of the picture).

---

## 7. Cross-references

- Thesis + expected-result sketches: [`thesis.md`](thesis.md) (§6a–6c, §6f falsifiers).
- Ch1 protocol (models/datasets/statistics): [`streaming-vs-batch-metrics.md`](streaming-vs-batch-metrics.md).
- Literature (direct comparisons, scope-defining evidence): [`Phase0:Research/literature_review/index.md`](../Phase0:Research/literature_review/index.md) §1 subgroup.
- Scope guard note: the old `notes/08-scope.md` excludes deep learning — this
  experiment series is a deliberate, teacher-relevant extension of it.
