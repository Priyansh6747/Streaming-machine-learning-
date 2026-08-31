# A Framework for Evaluating and Benchmarking Concept Drift Detection Methods

**Full citation:** Cerqueira, V., Gomes, H. M., Heyden, M., Pfahringer, B., & Bifet, A. (2026). A Framework for Evaluating and Benchmarking Concept Drift Detection Methods. *Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26), V.2*, pp. 8671–8682. https://doi.org/10.1145/3770855.3819070

**Status:** ✅ Verified (ACM DL full page fetched; open access, CC-BY-4.0)

---

## Why this paper matters

This is the **state-of-the-art answer to our Ch1 methodology problem**: how to
benchmark drift detectors fairly. It is the strongest candidate harness to
reproduce for the "pin River version and reproduce one published baseline"
pre-implementation task, and its metrics should be adopted in Ch3.

## Core contributions

1. **Real-data drift injection via Monte Carlo.** Instead of oversimplified
   synthetic generators, they inject controlled distributional changes into
   *real-world datasets* so drift position is known (supervised evaluation) while
   preserving real data complexity.
2. **Timing-aware evaluation protocol** with new metrics:
   - **F1 detection score** (combines precision/recall of alarms vs true drift
     events),
   - **normalized detection time** (comparable across streams of different
     lengths).
3. **Leave-one-dataset-out hyperparameter optimization**: tune on all-but-one
   dataset, test on the held-out one, to measure configuration robustness across
   heterogeneous stream dynamics.

## Experimental design

| Axis | Value |
|---|---|
| Detectors | 14 widely used drift detection methods |
| Datasets | 7 real-world datasets |
| Drift types | class prior, label swap, feature permutation, feature filtering |
| Transitions | abrupt and gradual |
| Metric emphasis | F1-detection score, normalized detection time |
| Availability | all code and experiments public |

## Key numbers

- KDD '26 acceptance rate 21% (1,157 of 5,395 submissions).
- 0 citations / 315 downloads (as of Aug 2026 — brand new).
- The paper's reference list is itself a useful Ch1 bibliography (DDM, EDDM,
  ADWIN, RDDM, HDDM, KSWIN, EWMA charts, Page 1954, Gama 2013, etc.).

## How it differs from Ch3 / our lemmas

- **Immediate labels only.** No delayed/partial/selective label regimes — which
  is exactly the gap Ch3 fills.
- No formal bounds (no reaction floor, no power law).
- It gives us the **metric definitions** (F1-detection, normalized delay) we
  should use to report our Lemma 1–4 experiments, and a **baseline protocol**
  (leave-one-dataset-out HP search) we should adopt for Ch1 and Ch3.

## Recommended use in the project

- **Phase 0 baseline reproduction:** reproduce a headline result from this paper
  with River/CapyMOA (e.g., ADWIN vs DDM vs KSWIN on one injected-drift real
  dataset) — a concrete, recent, citable published baseline.
- **Ch1:** adopt their protocol (or explicitly justify deviations).
- **Ch3:** extend their drift-injection framework with delay/label-coverage
  regimes; their normalized metrics make our delay curves comparable across
  streams.
