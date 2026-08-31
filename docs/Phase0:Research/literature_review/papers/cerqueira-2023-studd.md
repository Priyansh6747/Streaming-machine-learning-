# STUDD: A Student–Teacher Method for Unsupervised Concept Drift Detection

**Full citation:** Cerqueira, V., Gomes, H. M., Bifet, A., & Torgo, L. (2023). STUDD: A student–teacher method for unsupervised concept drift detection. *Machine Learning*, 112(11), 4351–4378. (Online 21 Jun 2022.) https://doi.org/10.1007/s10994-022-06188-7

**Status:** ✅ Verified (Springer full text read)

---

## Why this paper matters

STUDD is the strongest candidate for the **label-free early-warning stage
(Stage 1) of our hybrid detector**. It detects drift without any labels after
deployment, and it explicitly documents the exact limitation (inability to catch
posterior drift) that our Lemma 4 formalizes and that motivates the hybrid's
supervised confirmation stage.

## Method

- **Teacher** = the deployed classifier. **Student** = a model trained to mimic
  the teacher's predictions on the training set (model-compression style).
- At run time, drift is detected by feeding the **student's mimicking loss**
  `L(ŷ_T, ŷ_S)` — the discrepancy between teacher and student predictions — into a
  standard detector (Page–Hinkley with δ=0.001).
- On alarm, the classifier is retrained on the most recent window W.

## Experimental setup

- 19 benchmark streams (Electricity, Forest Covertype, Poker, Gas, Ozone, NOAA,
  Insects, etc.), truncated to 150k observations; W=2000 (500 for small sets).
- Baselines: BL-st (static), BL-ret (periodic retrain); unsupervised OS/OF/FF
  (Žliobaitė-style output sliding/fixed, feature-fixed dos Reis et al.);
  supervised SS/DSS/WS/DWS (strong / delayed-strong / weak / delayed-weak).
- Label-access parameters: l_access ∈ {1,10,25,50}%, l_delay ∈ {250..4000}.

## Key numbers

| Metric | Value |
|---|---|
| Missed-detection ratio (MDR) | ~19% (3/16 datasets; tied with DWS) |
| Label efficiency | best among non-baseline methods |
| Predictive performance vs supervised | often equivalent, or loses significantly |
| vs unsupervised (OS/OF/FF) | equivalent; ~25% prob of losing to FF |
| Labeling cost vs all others | wins significantly (Bayes sign test, ROPE 5%) |
| Sensitivity to l_access | STUDD relatively better as label availability drops; at l_access=50% it is worst |

STUDD is **conservative** (few alarms) — good when false alarms / label
acquisition are costly, which matches our Ch3 framing.

## How it differs from Ch3 / our lemmas

- **Unsupervised only.** It explicitly cannot detect real (posterior) drift that
  leaves P(X) unchanged — the identifiability ceiling (our Lemma 4). The authors
  say combining unsupervised detection with delayed/limited supervised feedback
  "may be worthwhile" — i.e., they name our hybrid direction as open work.
- Its Page–Hinkley-on-mimicking-loss design is a candidate Stage-1 component, but
  there is no delayed confirmation, no tolerance-window protocol, no
  false-alarm-cost calibration for unconfirmed warnings.
- It uses batch retraining (not streaming models), so it is not directly a
  Ch1-style detector benchmark.

## Recommended use

- Stage-1 baseline in Ch3 (label-free early warning), compared against the
  delayed-supervised baseline and the hybrid.
- Cite when justifying the hybrid: STUDD's own discussion supports the need for
  supervised confirmation.
