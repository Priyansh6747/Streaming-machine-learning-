# Batch-Incremental versus Instance-Incremental Learning in Dynamic and Evolving Data

**Full citation:** Read, J., Bifet, A., Pfahringer, B., & Holmes, G. (2012).
Batch-incremental versus instance-incremental learning in dynamic and evolving data.
In *Advances in Intelligent Data Analysis XI* (IDA 2012), LNCS 7619, pp. 313–323.
https://doi.org/10.1007/978-3-642-34156-4_29

**Status:** ✅ Verified (Springer chapter page + abstract; 104 citations on Springer,
Sep 2026). Full text not yet in vault — headline numbers to be extracted on full
read.

---

## Why this paper matters

The canonical, pre-Amekoe treatment of exactly our Ch1 axis: **instance-incremental
(streaming, one sample at a time) vs batch-incremental (gather examples, retrain on
the accumulated batch) learning under concept drift**. Written by the MOA/River core
authors, it is the paper the field cites for "there is no blanket winner — the two
paradigms have different advantages" — which is precisely the nuance our per-metric
win card (thesis.md §1) is built on.

## Method

- First in-depth analysis comparing both paradigms, including how each adapts to
  concept drift, with an extensive empirical study over several versions of each
  approach (per abstract).
- MOA-based classifiers; batch-incremental variants retrain on batches of
  accumulated/windowed data; instance-incremental variants update per example
  (detail in full text).

## Key findings (abstract-level; details to read)

- Both paradigms are dominant in the literature, but papers rarely justify the
  choice with evidence — this study supplies that evidence.
- The results reveal **respective advantages and disadvantages** of each approach:
  no universal winner, consistent with the later, larger-scale studies (Chen 2019;
  Amekoe 2024) that report winner-by-algorithm, winner-by-drift-character
  interactions.

## How it differs from our project / Ch1 thesis

- **No reactivity metrics.** Comparison is framed around accuracy/adaptation
  behaviour, not recovery time, detection delay, or damage area — our metric family
  is not measured there.
- **Immediate labels** assumed; no delay regime (Ch3 untouched).
- **2012-era learner zoo** (Hoeffding trees, ensembles) — our Ch1 adds the modern
  baselines (XGBoost scheduled-retrain, oracle-tuned) and per-metric statistics
  (rolling Kappa, Friedman + Nemenyi).

## Recommended use

- Ch1 / Phase-1 thesis §4 (Facts): the citation that "instance-incremental vs
  batch-incremental has been compared before and the result is context-dependent" —
  motivates why our claims must be **per-metric and per-dataset**, and why the win
  card structure (one claim per metric with dataset + number) is the honest
  presentation.
- Do not cite it as "streaming beats batch" — it doesn't claim that.
