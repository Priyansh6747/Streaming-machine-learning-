# E17 — The detector map on a real stream (Elec2)

**Status:** first real-stream pass, 5 seeds · **Artifact:** `experiments/results/e17/` ·
**Command:** `uv run python -m experiments.run --exp e17 --seeds 0-4`

---

## 1. Question

E04 found **HDDM-A on the raw mistake signal** to be the best cell for every one
of the four synthetic drift shapes. That was a synthetic-only result: the streams
came from SEA, Hyperplane, RandomRBF and Agrawal. Does the default survive when
the error stream comes from real records?

This is a detector check, not the E14 model check. It asks whether the eight
alarms still rank the same way when the features and labels are real.

## 2. Design

| | |
|---|---|
| Stream | River `Elec2` — 45,312 real electricity-market records, 8 features, binary label |
| Drift | label swap `0 → 1` injected at a known position |
| Shapes | abrupt, gradual, incremental, recurring |
| Length | 45,312 · warm-up 9,062 · change at `τ = 22,656` |
| Widths | abrupt 0 · gradual 2,000 · incremental 10,000 · recurring period 8,000, 2 episodes |
| Seeds | 5: `0–4` |
| Hosts | static Hoeffding Tree (`ht`) and adaptive HAT (`hat`) |
| Detectors | ADWIN, KSWIN, PageHinkley, DDM, EDDM, FHDDM, HDDM-A, HDDM-W |
| Channels | raw mistake signal, and EWMA-smoothed mistake signal |
| `δ_max` | abrupt 1,000 · gradual 2,000 · incremental 4,000 · recurring 2,000 |

Rows are shuffled per seed, then the label swap is injected. Shuffling removes
Elec2's natural temporal drift, so the only known change is the injected one.
This is the semi-synthetic protocol of the KDD'26 benchmark, not a natural-drift
test: it preserves real feature complexity and class overlap while keeping exact
detection ground truth.

An alarm inside `[τ, τ_end + δ_max]` is a true positive; an alarm outside any
window is a false alarm. `f1` is the project's balanced score, the harmonic mean
of alarm precision and episode recall. Zero-alarm runs count as F1 = 0. `FAR` is
false alarms per 1,000 samples. `NDT` is the delay normalized by the per-shape
`δ_max`, measured from the drift end.

## 3. Result

### 3a. Static host (`ht`) — mean over 5 seeds, raw channel

| Drift shape | Detector | F1 | Recall | Precision | FAR /1k | NDT |
|---|---|---:|---:|---:|---:|---:|
| Abrupt | **HDDM-A** | **0.933** | 1.000 | 0.900 | 0.0044 | 0.088 |
| Abrupt | ADWIN | 0.452 | 1.000 | 0.300 | 0.0839 | 0.069 |
| Gradual | **HDDM-A** | **0.960** | 1.000 | 0.933 | 0.0044 | 0.000 |
| Gradual | ADWIN | 0.495 | 1.000 | 0.333 | 0.0794 | 0.000 |
| Incremental | **HDDM-A** | **0.960** | 1.000 | 0.933 | 0.0044 | 0.000 |
| Incremental | ADWIN | 0.842 | 1.000 | 0.730 | 0.0265 | 0.000 |
| Recurring | **HDDM-A** | **0.787** | 1.000 | 0.667 | 0.0265 | 0.044 |
| Recurring | ADWIN | 0.508 | 1.000 | 0.346 | 0.1280 | 0.030 |

### 3b. Adaptive host (`hat`) — mean over 5 seeds, raw channel

| Drift shape | Detector | F1 | Recall | Precision | FAR /1k | NDT |
|---|---|---:|---:|---:|---:|---:|
| Abrupt | HDDM-A | 0.400 | 0.400 | 0.400 | 0.0000 | 0.026 |
| Abrupt | **ADWIN** | **1.000** | 1.000 | 1.000 | 0.0000 | 0.165 |
| Gradual | HDDM-A | 1.000 | 1.000 | 1.000 | 0.0000 | 0.000 |
| Gradual | ADWIN | 1.000 | 1.000 | 1.000 | 0.0000 | 0.000 |
| Incremental | **HDDM-A** | **1.000** | 1.000 | 1.000 | 0.0000 | 0.000 |
| Incremental | ADWIN | 0.971 | 1.000 | 0.950 | 0.0044 | 0.000 |
| Recurring | HDDM-A | 0.593 | 0.700 | 0.533 | 0.0265 | 0.007 |
| Recurring | **ADWIN** | **0.770** | 1.000 | 0.633 | 0.0309 | 0.051 |

### 3c. Best cell per shape (all detectors, hosts and channels)

| Drift shape | Best default | Host | Channel | F1 | Recall | FAR /1k |
|---|---|---|---:|---:|---:|---:|
| Abrupt | ADWIN | `hat` | raw | 1.000 | 1.000 | 0.0000 |
| Gradual | ADWIN (tied with HDDM-A) | `hat` | raw | 1.000 | 1.000 | 0.0000 |
| Incremental | HDDM-A | `hat` | raw | 1.000 | 1.000 | 0.0000 |
| Recurring | HDDM-A | `ht` | raw | 0.787 | 1.000 | 0.0265 |

### 3d. Overall, best host/channel per detector

This is a descriptive summary, not a pre-registered ranking: the host and
channel for each detector are chosen after seeing the means.

| Detector | Best host / channel | Mean F1 over the four shapes |
|---|---|---:|
| ADWIN | `hat` / raw | **0.935** |
| HDDM-A | `ht` / raw | 0.910 |
| PageHinkley | `hat` / raw | 0.788 |
| FHDDM | `hat` / raw | 0.715 |
| HDDM-W | `hat` / raw | 0.580 |
| DDM | `ht` / raw | 0.573 |
| KSWIN | `ht` / raw | 0.506 |
| EDDM | `ht` / raw | 0.413 |

## 4. Findings

1. **On the static host, HDDM-A repeats the synthetic result.** It has the best
   mean F1 in all four shapes: 0.933 abrupt, 0.960 gradual, 0.960 incremental,
   0.787 recurring. Its false-alarm rate stays at 0.0044 per 1,000 except for
   recurring, where it is 0.0265.
2. **On the adaptive host, the default breaks.** ADWIN wins abrupt (1.000 vs
   0.400) and recurring (0.770 vs 0.593), ties gradual at 1.000, and loses only
   incremental (0.971 vs 1.000).
3. **The failure is the one-sided test meeting a self-repairing learner.**
   HDDM-A only alarms when the error rate rises significantly above its frozen
   low-error reference. HAT adapts on its own, so an abrupt label swap can be
   repaired before the cumulative test separates. ADWIN compares two windows
   two-sidedly and still sees the distribution change. This is the concrete
   version of E04's caution that the host changes what a detector sees.
4. **Recurring change remains the hard case.** HDDM-A's precision on the static
   host is 0.667; on the adaptive host its recall falls to 0.70. ADWIN catches
   all recurring episodes on both hosts but pays 0.031–0.128 false alarms per
   1,000.
5. **Raw still beats EWMA.** Every best cell in §3c uses the raw channel; no
   smoothed cell wins a shape. This agrees with E04 and the E02 no-drift control.
6. **Do not claim a new overall winner.** The best-host mean is ADWIN 0.935 vs
   HDDM-A 0.910. The honest statement is: *HDDM-A is the best default on the
   static host; ADWIN matches or beats it on the adaptive host.*

## 5. What this does not say

- It is **one real dataset**, **one injected mechanism** (label swap `0 → 1`),
  and **5 seeds**. It is not the KDD'26 seven-dataset, four-mechanism,
  fifty-trial benchmark.
- The rows are shuffled, so **natural temporal drift is removed**. This tests
  detector response to a known injected change, not natural drift in Elec2.
- Elec2 is binary and comparatively easy; several cells are near-perfect, which
  compresses the differences between detectors.
- The label-swap direction interacts with one-sided detectors. A drift that
  lowers the error rate would be invisible to HDDM-A by construction.

## 6. Reproduce

```bash
uv run python -m experiments.run --exp e17 --seeds 0-4
uv run python -m experiments.detector_report --exp e17
uv run python -m experiments.run --exp e17 --figure-only
```

**Artifacts:** `experiments/results/e17/summary.json`,
`experiments/results/e17/detections.jsonl` (raw counts),
`experiments/results/e17/detector_report.json` (corrected report),
`experiments/results/e17/figures/`.
**Config:** `experiments/registry/e17_real_detector_map.yaml`.

## 7. Next — the KDD'26 benchmark reproduction

The KDD'26 paper benchmarks 14 detectors on 7 real datasets and reports SEED,
STEPD and ABCD as the strongest overall; HDDM-A is mid-tier. A first real-stream
check on one dataset cannot settle that. The isolated parallel runner
`experiments/kdd_bench.py` reproduces the paper's Monte Carlo protocol:

- one task = one `(dataset, drift type, transition, trial)` run;
- the Hoeffding Tree is trained once and all detectors are updated in the same
  pass, avoiding the public code's per-detector learner re-runs;
- 13 detector configurations: 11 scalar detectors, ABCD on the error stream,
  and ABCD(X) on the feature vector;
- 50 trials per cell by default, with `--resume`;
- four drift mechanisms and abrupt/gradual transitions.

**Status (2026-11-13):** the isolated environment resolves, `check` constructs
all 13 configurations in both modes, and the six-detector benchmark has now run
at both 10 and 50 trials on all 7 datasets, with the public-code-exact settings
(`--signal correct --gradual-anchor code`).

| Detector | 10-trial rank | 50-trial rank | 50-trial mean F1 |
|---|---:|---:|---:|
| **SEED** | 2.23 | **2.12** | 0.300 |
| **STEPD** | 2.30 | **2.12** | 0.303 |
| HDDM-W | 3.56 | 3.56 | 0.163 |
| HDDM-A | 3.94 | 3.87 | 0.155 |
| DDM | 4.52 | 4.54 | 0.101 |
| ADWIN | 4.45 | 4.80 | 0.122 |

The 50-trial run confirms the 10-trial ranking. **SEED and STEPD tie at 2.12;
HDDM-A is fourth at 3.87, behind HDDM-W (3.56).** The paper-protocol defaults
(`--signal error`, `--gradual-anchor paper`) gave the same top two and the same
mid-tier HDDM-A rank (3.91), so the conclusion is robust to the convention.

The absolute F1 values reproduce the paper's Tables 5–6 for the shared
detectors. Abrupt feature permutation: SEED/STEPD/HDDM-A/HDDM-W =
0.572/0.526/0.313/0.237 here vs 0.61/0.53/0.36/0.25 published. Gradual feature
permutation: 0.424/0.442/0.089/0.122 vs 0.46/0.45/0.10/0.14. DDM's abrupt
feature-permutation F1 is 0.021 vs 0.02 published. Not every cell matches
(ADWIN's class-prior and filtering F1 are lower here), but the six-detector
ranking is clear. The published tables therefore appear to use the public code's
correctness signal and onset-anchored gradual window, not the paper's Algorithm
1 error signal.

The ABCD/ABCD(X) arm was **not re-run**. It is the most expensive part of the
matrix, it overheats the 16-thread laptop CPU, and it is not needed for the
HDDM-A conclusion. The paper's published result is used for that cell: ABCD(X)
is the strongest detector on feature permutation (average rank 1.0 abrupt / 3.3
gradual). Treat the ABCD row as **cited, not reproduced**.

```bash
uv sync --project experiments/kdd_env
./experiments/kdd.sh prepare --download-usp
./experiments/kdd.sh check

# headline six-detector comparison (50 trials, all 7 datasets)
./experiments/kdd.sh run --datasets all --trials 50 --workers 14 \
    --signal correct --gradual-anchor code \
    --detectors ADWIN,DDM,HDDMAverage,HDDMWeighted,SEED,STEPD \
    --out experiments/results/kdd_bench_full

# ABCD/ABCD(X) is not re-run: it is the expensive part of the matrix and
# overheats the laptop CPU. The paper's published values are used for that cell.
```

The runner is deliberately explicit about two public-code discrepancies: the
released code feeds correctness (`int(true == pred)`) into MOA detectors that
document `1 = error`, and its gradual window is `(onset, onset)` rather than the
paper's end-anchored window. The script defaults follow the paper
(`--signal error`, `--gradual-anchor paper`); the headline run above uses
`--signal correct --gradual-anchor code` because that configuration reproduces
the published F1 tables.
