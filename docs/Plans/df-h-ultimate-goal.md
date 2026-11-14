# DF-H: deep hybrid drift detector — ultimate-goal roadmap

**Status:** approved 2026-11-14
**Supersedes:** the breadth-first E05–E15 detector roadmap. E01–E17 remain as
controls and evidence; the detector track is re-centred on DF-H.
**Related:** `docs/Phase0:Research/idea/scope.md` §5.3 and §6.5,
`docs/Reports/Nov14.md`, `docs/Reports/Results.md`,
`docs/Reports/E17-real-detector-map.md`.

> Readable summary: [Roadmap v2](../Reports/Roadmap-v2.md). Evidence snapshot:
> [Nov 14 post-mid-term report](../Reports/Nov14.md).

> **Status update (2026-11-14):** E18 and E19 ran and failed their
> pre-registered gates (best deep AUC 0.616 vs error 0.598; DF-H 3rd overall,
> 7.09, behind STEPD/SEED). E20 is not run under the pilot-first rule. E21 now
> carries the headline: the fixed-delay Lemma-1 synthesis shows label-free
> Stage 1 warning at **70–513** samples vs **2,120–3,269** for the error-only
> panel at `L = 2,000`; the full regimes remain scaffolded. **E22** adds the
> focused capability check: DF-H is **3rd overall (7.34)**, ahead of HDDM-W,
> ADWIN, HDDM-A and PageHinkley, behind only SEED/STEPD. See
> [E18](../Reports/E18-deep-signals.md),
> [E19](../Reports/E19-hybrid-detector.md),
> [E21](../Reports/E21-delay-detector.md) and
> [E22](../Reports/E22-detector-shootout.md).

---

## 0. The decision

### 0.1 Ultimate goal

Build **DF-H**, a two-stage drift detector:

1. **Stage 1 — label-free deep monitor.** A deep representation model
   (DriftFormer's encoder, with DeepRiver/an autoencoder as baselines) watches
   the input stream and emits drift signals without labels: masked-feature SSL
   loss, embedding shift, head-bank disagreement, predictive entropy.
2. **Stage 2 — supervised confirmation.** Classical error-stream detectors
   (SEED, STEPD, ADWIN, HDDM-A) run on the host's mistakes. A Stage-1 warning is
   confirmed only when a Stage-2 detector fires inside a short window.
3. **Fusion.** A calibrated, pre-registered rule reports both the **warning**
   time (Stage 1) and the **confirmed** detection time (Stage 2).

### 0.2 Success criteria

**Primary — KDD'26 immediate-label benchmark.** Same protocol as the current
reproduction: 7 datasets × 4 drift types × 2 transitions × 50 trials,
Hoeffding Tree host, same signal/anchor convention.

| Criterion | Bar |
|---|---|
| Overall average rank | **top-2** (currently SEED/STEPD at 2.12) |
| Weak cells | **best on at least two** of `feature_permutation`, `feature_filtering`, `recurring` |
| False alarms | **FAR no worse than ADWIN** |
| Per-dataset robustness | **no dataset rank worse than 4** |
| Stretch | **#1 overall**, or a clear win on all three weak cells |

**Secondary — the Delay Tax benchmark (the novel contribution).**

| Criterion | Bar |
|---|---|
| Fixed delay `L` | DF-H warning time grows by **≤ L + c**, while error-only detectors are floor-bound at `τ + L` |
| Partial labels | DF-H keeps higher recall than the best error-only detector at `p = 0.5` and `p = 0.1` |
| Selective labels | DF-H warns earlier on unlabelled drift; confirmation still needs labels |
| Lemmas | Lemma 1 (floor), Lemma 2 (`1/√p`), Lemma 3 (batch FAR) verified empirically |

**Negative result is acceptable.** If DF-H does not beat the baselines, the
delay-tax measurement and Lemmas 1–3 still stand — the scope already declares
this a publishable negative result. The detector track cannot sink the project.

### 0.3 Non-goals

- Do **not** optimise DriftFormer as a classifier. E14 shows it is weaker than
  ARF on real streams (κ 0.524 vs 0.724; `damage_ref` 284.9 vs 198.0). It is a
  **signal extractor**, not the benchmark model.
- Do **not** replace the KDD Hoeffding Tree host. A deep host would make the
  comparison a different benchmark.
- Do **not** train a deep model per KDD trial. Use a **frozen per-dataset
  encoder** and precompute signals.
- Do **not** expand the E05/E08–E13 matrix. Those are deferred.
- Do **not** build Chapter 2 (the service) before the detector result.

---

## 1. Why this is the right goal

The current evidence points to exactly one missing artefact: a detector that
uses information the scalar detectors cannot see.

- **No scalar detector wins everywhere.** E04: HDDM-A is best on all four
  synthetic shapes on the static host, but recurring is only F1 **0.500**.
  E17: on real Elec2 the default is host-conditional — HDDM-A wins all four
  shapes on the static HT, but ADWIN wins abrupt **1.000 vs 0.400** and
  recurring **0.770 vs 0.593** on the adaptive HAT. The best-host mean is
  ADWIN **0.935** vs HDDM-A **0.910**.
- **The external benchmark has headroom but a low ceiling.** KDD'26:
  SEED/STEPD rank **2.12** (mean F1 **0.300/0.303**); HDDM-W **3.56**;
  HDDM-A **3.87**. Absolute F1 is low, so a better signal matters — but naive
  combinations raise false alarms (E02: EWMA makes ADWIN **0.548/1k** and KSWIN
  **7.667/1k** vs a 0.1 target).
- **Feature-space detection is the strongest arm on feature drift.** ABCD(X) is
  cited as rank **1.0 abrupt / 3.3 gradual** on feature permutation. A deep
  representation monitor is a nonlinear version of that idea.
- **The mechanism already exists in DriftFormer.** `_step()` computes a
  masked-feature SSL loss and feeds an internal ADWIN; `forward()` returns the
  four head logits and the `cls_out` embedding. The detector is an instrumentation
  and fusion problem, not a new model.
- **The hybrid is already the declared contribution.** `scope.md` §5.3 names
  "label-free early warning + delayed supervised confirmation" as the headline;
  §6.5 argues it is the information-maximal architecture. This plan makes it the
  centre instead of a late chapter.

---

## 2. What "based on DeepRiver" means, and the scope amendment

DriftFormer/DeepRiver supplies the **representation**, not the classifier.

- **KDD benchmark host:** Hoeffding Tree (same as the current reproduction).
  The deep monitor is an additional, label-free detector channel.
- **Synthetic/real validation host:** Hoeffding Tree and HAT/ARF, to test
  host-conditional behaviour.
- **Scope amendment (dated note in `scope.md` §9):** deep models are permitted
  as **label-free signal extractors** for the hybrid detector. They are not the
  benchmark classifier and do not replace the classical baseline. This is the
  implementation of §6.5, not a reversal of the classical scope.
- **Fallback if deep is disallowed:** replace Stage 1 with a classical
  feature-space monitor (PCA/KS/linear MMD, ABCD(X)-style). The fusion, gates,
  and delay benchmark are unchanged.

---

## 3. Architecture

### 3.1 Stage 1 — label-free deep signals

All signals are computed per window (default **500 samples**, stride **100**)
from a frozen encoder, and normalised against the first 20% of the stream.

| Signal | Definition | Why it should help |
|---|---|---|
| `ssl_loss` | Masked-feature cross-entropy from the SSL head on the window | The encoder's own reconstruction/feature model becomes surprised |
| `embedding_shift` | Standardised L2 distance between window mean/covariance of `cls_out` and a reference window; optional linear MMD | Direct `P(X)` change; strong on feature permutation/filtering |
| `head_disagreement` | Mean pairwise JS/TV across the four head softmaxes | Concept uncertainty without labels |
| `entropy` | Predictive entropy of the active head | Confidence collapse |
| `internal_adwin` | Existing ADWIN on the SSL loss | Cheapest deep signal; baseline |
| `feature_ks` | Max per-feature KS statistic vs the reference (classical control) | Shows how much the deep monitor adds over a simple test |

**Encoder training.** Per dataset, train DriftFormer's encoder with the SSL
objective on a fixed un-drifted reference sample (20k rows, or all rows for small
datasets). Freeze it. No labels are used for Stage 1. For synthetic streams,
train on the first 12k warm-up samples or a separate reference seed.

### 3.2 Stage 2 — supervised confirmation

Use a fixed panel, not a test-time best-detector oracle:

- **SEED** and **STEPD** — the KDD'26 top two.
- **ADWIN** — best on the adaptive host in E17.
- **HDDM-A** — best on the static synthetic host in E04.
- Raw mistake channel only; E02 rules out EWMA.

### 3.3 Fusion

Pre-register the rule; report all variants as ablations:

| Variant | Rule |
|---|---|
| `S1` | Stage 1 only (label-free) |
| `S2` | Best fixed scalar detector only |
| `OR` | Stage 1 or Stage 2 |
| `AND` | Stage 1 and Stage 2 within `W_confirm = 2,000` |
| `2of3` | Stage 1 + SEED + ADWIN, two of three |
| `2of3_nodeep` | SEED + ADWIN + HDDM-A, two of three (control) |

**v1** uses a fixed weighted z-score with a threshold calibrated to
**FAR ≤ 0.1/1k** on E02 and the pre-drift 20% of each trial. **v2** replaces the
fixed weights with a small logistic regression / gradient-boosted model trained
on window features.

**Leakage rule.** Any learned fusion is trained with **leave-one-dataset-out**
on KDD and never tuned on the test trials. Synthetic streams are training-only.

### 3.4 Warning vs confirmed alarm

For the delay benchmark, report two times separately:

- `T_warn`: first Stage-1 alarm (no labels needed).
- `T_confirm`: first Stage-2 confirmation after the label arrives.

This is the honest version of the hybrid: Stage 1 can beat the reaction floor,
Stage 2 cannot. Lemma 4 (pure posterior drift) is exactly the case where Stage 1
has no power and Stage 2 must do the work.

---

## 4. Benchmark strategy

### 4.1 KDD'26 immediate-label benchmark

- **Protocol:** unchanged from `experiments/kdd_bench.py` — 7 datasets, 4 drift
  types, abrupt/gradual, 50 trials, 100k row cap, shuffled per trial, Hoeffding
  Tree host.
- **Convention:** headline uses `--signal correct --gradual-anchor code`
  (reproduces the published F1 tables); the paper-protocol defaults are a
  robustness check.
- **Deep signal path:** precompute in the main environment; `kdd_env` loads
  `.npz` signals and runs pure-NumPy fusion. No torch in `kdd_env`.
- **Baselines:** SEED, STEPD, HDDM-A, HDDM-W, ADWIN, DDM, `S1`, `S2`,
  `2of3_nodeep`, and the current best `2of3`.
- **Metrics:** KDD F1, FAR/1k, NDT, average rank over 56 cells, per-drift-type
  rank, per-dataset rank, Friedman/Nemenyi, bootstrap CIs over trials.
- **ABCD/ABCD(X):** keep cited from the paper; do not run the expensive arm.

### 4.2 Delay Tax benchmark (replaces E15)

- **Regimes:** fixed `L ∈ {0, 10, 100, 1000}`; geometric mean `L`; partial
  `p ∈ {1, 0.5, 0.1}`; selective (only flagged samples labelled); batch arrival
  `b ∈ {1, 16, 128}`.
- **Hosts:** Hoeffding Tree and ARF.
- **Detectors:** SEED, STEPD, ADWIN, HDDM-A, DF-H.
- **Metrics:** `T_warn`, `T_confirm`, FAR, recall, downstream damage/recovery.
- **Lemmas:** L1 floor (`T ≥ τ + L`), L2 power loss (`1/√p`), L3 batch FAR
  inflation, L4 identifiability (pure posterior drift).
- **E06** stays as the Lemma-1 pilot; E21 is the full benchmark.

### 4.3 Controls

- **No drift:** E02, for FAR calibration.
- **Pure posterior drift:** `label_swap` on unchanged `P(X)` — Stage 1 must be
  blind; Stage 2 must catch it.
- **Feature drift:** `feature_permutation`/`feature_filtering` — Stage 1 should
  win; Stage 2 may add little.
- **Host swap:** HT vs HAT — the E17 host-conditional result must hold.

---

## 5. Phased roadmap and gates

### Phase 0 — close the in-flight work (time-boxed)

1. Commit the working tree in logical units; keep the 99 MB ARFF and the JVM
   crash log out of git.
2. Finish E03's 11-seed run (20/22 units done); regenerate the summary and
   schedule report.
3. Document E14 with acceptance rules; render E17 figures.
4. Reconcile the E01 74.8/515.0 and E02 FAR numbers across live docs.
5. Add the new `Roadmap v2` page and nav.

**Exit:** a clean enough tree to start the detector without mixing unrelated
changes into its commits.

### Phase 1 — instrumentation (E18 scaffold)

- `DriftFormerLearner.signal_snapshot()` returns per-block `ssl_loss`,
  `head_disagreement`, `entropy`, and `cls_out` statistics.
- New `DeepSignalMonitor` owns the windowing, reference statistics, and
  normalisation.
- Artifact writer emits `signals.npz` beside the existing `series.npz`.
- Refactor `kdd_bench.py` to expose a pure-NumPy `build_trial_stream(task, data)`
  used by both the KDD runner and the deep precompute.
- Tests: shapes, finiteness, determinism, stream parity.

**Exit:** a 1,500-sample SEA smoke run and a 1-seed E17 run produce finite,
deterministic signal arrays.

### Phase 2 — signal-validation gate (E18)

- Extract signals on SEA, Hyperplane, Elec2, and a KDD pilot:
  **2 datasets × 4 drift types × 2 transitions × 5 trials = 80 trials**.
- Measure windowed AUC for "drift within the next W samples" on held-out
  datasets; compare every deep signal against the raw error-rate baseline.
- **Go/no-go gate:** at least one deep signal has **AUC ≥ 0.75** on held-out
  data and beats the error-rate AUC by **≥ 0.10**. If not, stop or fall back to
  the classical feature monitor.

**Exit artifact:** `experiments/results/e18/` + `docs/Reports/E18-deep-signals.md`.

### Phase 3 — DF-H v1, rule-based fusion (E19)

- Implement the fusion variants; evaluate on synthetic + E17 + the KDD pilot
  (2 datasets, 5 trials).
- **Gate:** top-2 on the pilot; best on at least one feature cell; FAR ≤ ADWIN;
  no pilot dataset rank worse than 3.
- **Exit:** artifact + report; decide whether v2 is worth the full KDD run.

### Phase 4 — DF-H v2, learned fusion + full KDD (E20)

- Train the fusion on synthetic + a subset of KDD datasets; leave-one-dataset-out.
- Full **7 × 4 × 2 × 50** KDD run with precomputed deep signals.
- **Gate:** top-2 overall; best on at least two weak cells; FAR ≤ ADWIN; no
  dataset rank > 4. Stretch: #1 overall.
- **Exit:** final KDD artifact + `docs/Reports/E20-deep-kdd.md`; Results row.

### Phase 5 — Delay Tax benchmark (E21)

- Add label regimes to the harness: fixed, geometric, partial, selective, batch.
- Run DF-H vs SEED/STEPD/ADWIN/HDDM-A on the delay regimes.
- **Gate:** `T_warn ≤ L + c`; `T_confirm` matches the best error-only detector
  plus the expected label delay; partial/selective recall improvement; Lemma 1–3
  slopes measured.
- **Exit:** delay artifact + `docs/Reports/E21-delay-detector.md`; headline
  contribution.

### Phase 6 — write-up and minimal system

- Restructure the final report around the detector:
  1. Measurement lab + baseline detector map (E01/E02/E04/E17/KDD).
  2. DF-H architecture + immediate-label benchmark (E18–E20).
  3. Delay Tax benchmark + lemmas (E06/E21).
  4. Minimal system/demo or a clearly scoped Chapter 2.
- The mid-term report stays frozen; corrected numbers go into the final report.

---

## 6. File-level implementation

| File | Change |
|---|---|
| `experiments/driftlab/deep/train.py` | `signal_snapshot()`; record `ssl_loss`, `head_disagreement`, `entropy`, optional `cls_out` stats; keep the internal ADWIN |
| `experiments/driftlab/deep/model.py` | `encode()` / `head_probs()` helpers; no architecture change |
| `experiments/driftlab/deep/signals.py` (new) | `SignalConfig`, `DeepSignalMonitor`, windowing, z-score/MAD normalisation, fusion helpers |
| `experiments/driftlab/detectors.py` | `HybridDetector`, `run_hybrid`; raw-only defaults; warning/confirmed alarm records |
| `experiments/driftlab/config.py` | `deep_signals` and `hybrid` config blocks; defaults; config hash includes them |
| `experiments/driftlab/harness.py` | Emit `signals.npz`; run the hybrid; add `detector="dfh"` rows; keep the existing detector replay intact |
| `experiments/driftlab/artifacts.py` | No schema change needed for series; add a `signals` helper only if the key convention needs it |
| `experiments/kdd_bench.py` | Extract `build_trial_stream`; add `--deep-signals DIR`; run pure-NumPy fusion variants; write `detector="DFH"` rows |
| `experiments/deep_kdd.py` (new) | `pretrain` per dataset; `prepare-signals` per trial; cache `.npz`; replicate the KDD trial RNG exactly |
| `experiments/registry/e18_deep_signals.yaml` | Signal-extraction config on SEA/Hyperplane/Elec2 |
| `experiments/registry/e19_hybrid_detector.yaml` | Rule-based fusion on synthetic + E17 + KDD pilot |
| `experiments/registry/e20_deep_kdd.yaml` | Full KDD DF-H run (may be CLI-driven rather than harness-driven) |
| `experiments/registry/e21_delay_detector.yaml` | Delay regimes; DF-H vs scalar detectors |
| `experiments/tests/test_deep_signals.py` | Shape/finite/deterministic signal tests |
| `experiments/tests/test_hybrid_detector.py` | Fusion thresholds, confirmation window, FAR calibration |
| `experiments/tests/test_deep_kdd.py` | Trial-stream parity between the KDD runner and the precompute |
| `experiments/tests/test_label_regimes.py` | Fixed/geometric/partial/selective/batch scheduling |
| `docs/Phase1:Experimentation/exp18-deep-signals.md` | E18 write-up |
| `docs/Reports/E19-hybrid-detector.md` | E19 write-up |
| `docs/Reports/E20-deep-kdd.md` | E20 write-up |
| `docs/Reports/E21-delay-detector.md` | E21 write-up |
| `docs/Reports/Roadmap-v2.md` | This plan in user-facing form; linked from Results/progress |
| `docs/Phase0:Research/idea/scope.md` | Dated deep-monitor scope amendment |
| `mkdocs.yml` | Nav entries for the new pages |

---

## 7. What we cut, defer, or keep

| Item | Decision | Reason |
|---|---|---|
| **E03 baseline ladder** | **Finish, no expansion** | Already 20/22 units; a control, not the goal |
| **E05 severity × width** | **Defer** | Interesting but not on the DF-H critical path |
| **E06 delay shift** | **Keep** | Lemma-1 pilot and E21 precursor |
| **E07 block gate** | **Keep** | Prerequisite for DriftFormer speed/reproducibility |
| **E08 DriftFormer vs panel** | **Cut to one host-quality check** | We are not optimising the classifier |
| **E09 pretrain transfer** | **Defer** | Only relevant if the encoder transfer is a bottleneck |
| **E10–E12 ablations/mechanisms/architecture** | **Defer; keep E10-lite and E12-lite** | Only signal and encoder-size ablations are needed |
| **E13 calibration** | **Keep only if v2 uses probabilities** | Otherwise defer |
| **E14 real streams** | **Keep as host validation; no 5-dataset expansion now** | Already shows DriftFormer is weaker than ARF |
| **E15 Delay Tax** | **Replace with E21** | E21 is the detector benchmark, not a model panel |
| **ABCD/ABCD(X)** | **Cited only** | Expensive/thermal; not needed to prove DF-H |
| **Chapter 2 service** | **Defer until after E20/E21** | The detector is the development half; the service is a separate chapter |
| **Product docs** | **Defer** | Not on the critical path |

---

## 8. Risks and fallbacks

| Risk | Mitigation / fallback |
|---|---|
| Deep signal has no discriminative power | Phase-2 gate stops early; fall back to a classical feature monitor (PCA/KS/linear MMD) with the same fusion |
| KDD host mismatch | Keep the Hoeffding Tree host; deep model is only a signal channel |
| Leakage in the learned fusion | Leave-one-dataset-out; pre-register the rule; no test-trial tuning |
| Deep compute cost | Frozen per-dataset encoder; batched GPU inference; cached signals; pilot-first |
| DriftFormer is a weak classifier | Never use it as the host; use its encoder only; compare against an autoencoder/DeepRiver representation |
| False alarms | Raw channel only; E02-calibrated threshold; confirmation window; FAR target |
| Scope says deep is out | Dated scope amendment: deep is a label-free signal extractor, not the benchmark classifier; classical fallback ready |
| Delay floor cannot be beaten | Report warning vs confirmed separately; Lemma 1 is a theorem, not a failure |
| KDD protocol quirks | Use the public-code convention for the headline; run the paper convention as a robustness check |
| Laptop thermal ceiling | Reduced worker counts / `taskset`; hand heavy runs to the user |

---

## 9. Verification plan

### 9.1 Static and unit checks

```bash
uv run pytest experiments/tests/test_deep_signals.py \
    experiments/tests/test_hybrid_detector.py \
    experiments/tests/test_deep_kdd.py \
    experiments/tests/test_label_regimes.py -q
```

### 9.2 Smoke runs (small, quick)

```bash
# signal extraction, 1 seed / 1,500 samples
uv run python -m experiments.run --exp e18 --seeds 0 --limit 1500 --no-proba \
    --out experiments/results/_smoke/e18

# E17 signal extraction, 1 seed
uv run python -m experiments.run --exp e18 --seeds 0 --out experiments/results/_smoke/e18-e17
```

### 9.3 Signal gate

```bash
# 80-trial KDD pilot: 2 datasets x 4 drift types x 2 transitions x 5 trials
uv run python -m experiments.deep_kdd prepare-signals \
    --datasets Covtype,Electricity --trials 5 \
    --out experiments/data/deep_signals/pilot

uv run python -m experiments.deep_kdd report \
    --signals experiments/data/deep_signals/pilot \
    --out experiments/results/e18
```

Expected: an AUC table per signal, drift type, and dataset; the gate is written
into `summary.json` as `pass = true/false`.

### 9.4 DF-H pilot and full KDD

```bash
# pilot: scalar + DF-H on the same 80 trials
./experiments/kdd.sh run --datasets Covtype,Electricity --trials 5 --workers 8 \
    --signal correct --gradual-anchor code \
    --detectors SEED,STEPD,ADWIN,HDDMAverage,HDDMWeighted \
    --deep-signals experiments/data/deep_signals/pilot \
    --out experiments/results/e19_kdd_pilot

# full: only after the pilot gates pass
./experiments/kdd.sh run --datasets all --trials 50 --workers 14 \
    --signal correct --gradual-anchor code \
    --detectors SEED,STEPD,ADWIN,HDDMAverage,HDDMWeighted \
    --deep-signals experiments/data/deep_signals/full \
    --out experiments/results/e20_kdd_full
```

### 9.5 Delay benchmark

```bash
uv run python -m experiments.run --exp e21 --seeds 0-4 \
    --out experiments/results/e21
```

Expected: `T_warn`, `T_confirm`, FAR, recall, and the lemma slopes in
`summary.json`; a figure per regime.

### 9.6 Site

```bash
uv run mkdocs serve
```

The new pages must be reachable from the nav; a page without a nav entry is
treated as missing.

---

## 10. Assumptions and open decisions

1. **"Beat the benchmark" = top-2 overall + best on at least two weak cells.**
   #1 overall is the stretch goal. If #1 is mandatory, tighten Phase-4 gates and
   add an ABCD-style feature ensemble, but do not promise it.
2. **Deep is allowed as a label-free signal extractor.** The benchmark host stays
   classical. If the teacher forbids deep, use the classical feature-monitor
   fallback.
3. **Frozen per-dataset encoder, not per-trial training.** This keeps the KDD
   benchmark feasible and the comparison fair.
4. **Pilot-first compute.** No full 50-trial deep signal run until the 80-trial
   pilot passes.
5. **Cut list is accepted.** E05, E08–E13 (except the lite ablations) and the
   full Chapter 2 are deferred.
6. **Heavy commands are handed to the user.** Per the project rule, the agent
   writes and statically checks the scripts; the user runs the long jobs.

If any of these is wrong, the affected phase changes; the architecture and
gates do not.

---

## 11. Definition of done

- `DF-H` implemented as a reusable detector adapter with tests.
- E18 signal report with a clear go/no-go verdict.
- E19 pilot result; E20 full KDD result with a DF-H row beside SEED/STEPD.
- E21 delay benchmark with Lemmas 1–3 measured (or a documented negative
  result).
- New docs pages wired into `mkdocs.yml`; Results/progress/scope updated.
- The mid-term report remains frozen; the final report is written from saved
  artifacts.
- One-command reproduction for every result.

---

## Appendix — experiment map

| Experiment | Role after this plan |
|---|---|
| Exp-01 / Exp-02 | Legacy pilot; superseded by E01 |
| E01 | Corrected metric / control |
| E02 | No-drift FAR calibration |
| E03 | Baseline ladder; finish, no expansion |
| E04 | Synthetic detector map / baseline |
| E06 | Lemma-1 delay pilot |
| E07 | DriftFormer block gate |
| E14 | Real-stream host validation |
| E17 | Real-stream detector map / host-conditional baseline |
| KDD'26 | External benchmark / baseline ranking |
| **E18** | **Deep signal extraction + validation gate** |
| **E19** | **DF-H v1, rule-based fusion, pilot** |
| **E20** | **DF-H v2, learned fusion, full KDD** |
| **E21** | **Delay Tax detector benchmark + lemmas** |
| E05, E08–E13, E15 | Deferred/cut except the lite ablations |

**The single sentence to remember:** *a deep label-free monitor provides the
signal no error-stream detector can see; a supervised confirmation stage keeps
the cells where SEED/STEPD are strong; DF-H is the detector that owns both.*
