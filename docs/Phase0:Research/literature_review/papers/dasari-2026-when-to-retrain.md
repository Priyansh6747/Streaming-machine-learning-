# When to Retrain: An Empirical Study of Retraining Policies for Streaming ML Under Concept Drift, Budget, and Latency Constraints

**Full citation:** Dasari, S. (2026). When to retrain: An empirical study of retraining
policies for streaming ML under concept drift, budget, and latency constraints.
arXiv:2608.19488. https://arxiv.org/abs/2608.19488

**Status:** ✅ Verified (arXiv record + full abstract; code public at
github.com/sa1dasari/Study-of-Drift-Triggered-Retraining-Policies-Under-Budget-and-Latency-Constraints)

---

## Why this paper matters

This is the closest published answer to the teacher's Ch1 question — *"take some
models that are not streaming and beat them on real metrics"* — posed at the level
of **retraining policy** (clock-driven vs event-driven), not just learner type. It
compares periodic retraining, error-threshold triggering, and ADWIN (statistical
drift)-triggered retraining against a no-retrain baseline, under a unified system
model that makes budgets and training-plus-deployment latency explicit.

Its headline result is a **scope-defining caution for our thesis**: the retraining
*policy* is not the main lever — whether the deployed model learns **incrementally
per sample** is. This forces our Ch1 claim to be stated about *streaming learners*,
not about drift detectors bolted onto batch models.

## Method

- Simulator-based controlled study; retraining consumes budget while a **stale model
  keeps serving** during training-plus-deployment latency (a queueing interaction).
- Policies: periodic, error-threshold, ADWIN-triggered, and a no-retrain floor.
- Grid: 3 drift regimes × 3 budget levels × up to 5 latency levels × 3 datasets ×
  2 learning modes (per-sample `partial_fit` incremental vs refit-only) → **3,933
  runs**; 54 paired comparisons for the incremental-mode analysis.
- Learner: linear online learner; immediate labels (no delay regime).

## Key findings

1. **Incremental learning dominates policy.** With per-sample incremental updates,
   no retraining policy differed from the no-retrain baseline by a practically
   significant margin in any of the 54 paired comparisons — even at extreme latency.
2. **Without incremental updates**, policy choice separates outcomes by **15–55
   percentage points of post-drift accuracy**, and **simple periodic retraining
   significantly outperforms both reactive policies** under abrupt and gradual
   drift.
3. Reactive (drift-triggered / error-threshold) policies retain an advantage over
   periodic **only under recurring drift**.
4. Systematic failure modes of reactive policies are documented, including a
   latency–budget queueing interaction that **silently halves effective retraining
   budgets**.

## How it differs from our project / Ch1 thesis

- **Confirms our mechanism, redirects the claim.** "Recovery is bounded because the
  model updates per sample" is the right statement; "recovery is bounded because a
  detector fires" is not — a detector firing at a *batch refit* cadence inherits all
  the clock's weaknesses (this paper: periodic beats reactive refits on abrupt and
  gradual drift).
- **Immediate labels only.** No delayed/partial labels; leaves the Delay Tax (Ch3)
  untouched and supports our "reaction floor T ≥ τ + L_τ" framing as the extension.
- **Linear learner, three datasets.** Narrower model/dataset coverage than our
  planned Ch1 sweep (trees/ensembles; real + synthetic with ground-truth drift).

## Recommended use

- Ch1 / Phase-1 thesis §1–3: cite as the empirical boundary of the "event-driven
  beats clock-driven" claim — the win requires a per-sample learner, and detectors
  alone (as refit triggers) do not beat a schedule except under recurring drift.
- Ch1 protocol: adopt its "stale model serves during latency" simulator idea when
  justifying our scheduled-retrain baseline (the phase-noise argument) and when
  reporting throughput/latency.
- Reproducibility: its public simulator is a candidate sanity-check harness before
  we build our own River-based one.
