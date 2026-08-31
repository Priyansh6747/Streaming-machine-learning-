# 01 — What streaming (online) machine learning is

## The assumption that gets dropped

Standard "batch" ML rests on an assumption so common it's usually invisible:

> The dataset is finite, available all at once, and drawn i.i.d. from a fixed
> distribution that will not change.

Streaming ML drops all three parts of that. Data arrives **continuously**, you see
each sample **once**, and the underlying distribution is **allowed to change over
time**.

## Side by side

| | Batch learning (scikit-learn) | Online / streaming learning (River) |
|---|---|---|
| Data availability | Whole dataset up front | One sample at a time |
| Passes over data | Many (epochs) | **Exactly one** |
| Memory | Grows with dataset size | **Constant / bounded** |
| Training | `fit(X, y)` — from scratch | `learn_one(x, y)` — incremental |
| Prediction | `predict(X)` on a matrix | `predict_one(x)` on one sample |
| Data representation | NumPy array / DataFrame | Python **dict** `{"feature": value}` |
| Evaluation | train/test split, cross-validation | **Prequential** (test-then-train) |
| Handling new data | Retrain the whole model | Model was never not-training |
| Distribution changes | Model silently rots | Model detects and adapts |

## The three hard constraints

Any algorithm that claims to be a streaming algorithm must respect these:

1. **Single pass** — you cannot go back and look at sample #5 again. It's gone.
2. **Bounded memory** — memory usage must not grow with the number of samples
   seen. A model that has processed 10 billion rows must occupy roughly the same
   space as one that has processed 10 thousand.
3. **Bounded time per sample** — processing must keep up with arrival rate. If
   samples arrive faster than you can learn from them, you fall behind forever.

A useful test: *if this ran for a year without restarting, would it still work?*
If the answer involves "well, memory would…", it isn't a streaming algorithm.

## Why anyone bothers

Two separate motivations, often confused:

**(a) The engineering motivation — scale and latency.**
The data doesn't fit in RAM, or it arrives too fast, or you need the model updated
*now* rather than after tonight's retraining job. Think clickstreams, sensor
networks, network packets, financial ticks, log pipelines.

**(b) The statistical motivation — the world changes.**
This one is more interesting and more researchable. Even if your data *did* fit in
memory, a model trained on last year's data may be wrong about this year, because
the process generating the data has changed. Batch learning has no mechanism to
notice this. Streaming learning does. See [02 — Concept drift](02-concept-drift.md).

Motivation (b) is the better spine for a research project. (a) is an engineering
benchmark; (b) is a scientific question.

## What "online" does *not* mean

Some clarifications worth having straight, because they come up in vivas:

- **Online ≠ real-time.** Online means incremental updates. Real-time means latency
  guarantees. They usually co-occur but they're different properties.
- **Online ≠ mini-batch SGD.** Training a neural net with mini-batches over multiple
  epochs is still batch learning — you're storing the dataset and revisiting it.
  Online learning sees each sample once.
- **`partial_fit` in scikit-learn is not the same thing.** It supports incremental
  fitting, but the surrounding ecosystem (scalers, encoders, metrics, evaluation)
  still assumes batch. River makes the *whole pipeline* incremental — including
  the scaler, which maintains a running mean and variance and never sees a column.
- **Online learning ≠ the online learning of learning theory** (regret bounds,
  adversarial settings). Related field, different emphasis. Worth one sentence in
  a related-work section, not more.

## The mental shift

In batch ML you ask: *how accurate is my model?* — a single number.

In streaming ML you ask: *how does my model's accuracy behave over time?* — a curve.
Accuracy becomes a time series. Drops, recoveries, and how fast recovery happens are
the interesting phenomena. Almost every good figure in this field is
**something plotted against "number of samples seen."**

---

Next: [02 — Concept drift](02-concept-drift.md)
