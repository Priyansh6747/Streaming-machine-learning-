# 04 — How you evaluate a streaming model

This is where students most often go wrong, and where a careful project scores marks.

## The problem with train/test split

A train/test split assumes the data is exchangeable — that it doesn't matter which
rows land in which half. In a stream that's false twice over:

1. **There is no "end" of the data** to hold out from.
2. **Order is meaningful.** If the concept drifts, a random split leaks future
   information into training. Your model gets to see post-drift samples while
   "training on the past," and the reported accuracy is fiction.

Random shuffling a temporal stream is the streaming-ML equivalent of using future
prices to predict past ones. If you do it, your numbers are meaningless.

## Prequential evaluation (test-then-train)

The standard protocol, also called **progressive validation** or
**interleaved test-then-train**:

> For each arriving sample: **predict first, score the prediction, then learn from it.**

```python
for x, y in stream:
    y_pred = model.predict_one(x)   # model has NEVER seen this sample
    metric.update(y, y_pred)        # so this score is honest
    model.learn_one(x, y)           # now let it learn
```

Every sample is used for both testing and training, in that order — so no data is
wasted and no data leaks. The measured accuracy is genuinely out-of-sample at every
step.

## Cumulative vs. rolling metrics — pick deliberately

**Cumulative** (`metrics.Accuracy()`) averages over *everything seen so far*.
It converges and becomes sluggish: after 100,000 samples, one bad stretch barely
moves it. It **hides drift.**

**Rolling** (`metrics.Rolling(metrics.Accuracy(), window_size=1000)`) averages over
the last *w* samples only. It reveals dips and recoveries.

```
cumulative accuracy          rolling accuracy (w=1000)
                                        
1.0 ┤                        1.0 ┤    ╭──────╮      ╭────
    │  ─────────────╲___         │────╯      ╲     ╱
0.5 ┤                            0.5 ┤            ╲___╱
    └──────────────────          └────────────────────────
         ↑ drift                       ↑ drift
    barely visible               clearly visible + recovery
```

**Use rolling metrics for your plots.** Report cumulative for headline single
numbers. Saying *why* you did both is the kind of thing that earns credit.

## Metrics for the model

Standard classification/regression metrics apply, but note:

- **Accuracy is misleading under class imbalance**, which is common in streaming
  applications (fraud, intrusion detection). Report **F1 / precision / recall /
  ROC-AUC** as well.
- The **accuracy-over-time curve** is a result in itself, not just a picture.
  Its dip depth and recovery slope are what you're actually comparing.
- **Memory** (`model._memory_usage` or `sys.getsizeof` / `pympler`) and
  **wall-clock time per sample** are first-class results in this field, not
  afterthoughts. Streaming ML's whole claim is bounded resources — measure it.

## Metrics for the *drift detector*

Different question, different metrics. These require **known ground-truth drift
points**, which is why synthetic streams are non-negotiable for this part.

| Metric | Definition | Good value |
|---|---|---|
| **Detection delay** | samples between true drift and first alarm | low |
| **False positives (FP)** | alarms with no drift nearby | low |
| **Missed detections (FN)** | drift points with no alarm within a tolerance window | low |
| **MTFA** | mean time between false alarms | high |
| **MTD** | mean time to detection | low |
| **Recovery time** | samples until rolling accuracy returns to pre-drift level | low |

You need a **tolerance window** — a drift at index 5000 with an alarm at 5040 is a
successful detection with delay 40; an alarm at 9000 is a false positive plus a
missed detection. Define the window explicitly and justify it; papers vary, and
being explicit about it is a mark of rigour.

The central finding to expect: **no detector wins on all three of delay, FP, and
FN.** Sensitivity trades against false alarms. Showing that trade-off with real
numbers is a legitimate result.

## Experimental hygiene (what separates a project from a demo)

- **Multiple seeds.** Synthetic streams are random. Run ≥10 seeds and report
  **mean ± standard deviation**, never a single run.
- **Fixed budget.** Same stream length, same drift positions, same seeds across all
  compared methods. Otherwise you're comparing noise.
- **Statistical testing.** For comparing many models over many streams, the standard
  is the **Friedman test + Nemenyi post-hoc**, visualised as a *critical difference
  diagram*. This is the accepted convention in the streaming-ML literature and using
  it signals you read the actual papers (Demšar, 2006).
- **A null baseline.** Include something trivial — majority-class, or a model that
  never adapts. If your clever method doesn't clearly beat "do nothing," that's an
  important finding and you should report it rather than hide it.
- **Report the losses.** If the adaptive model is slower or heavier, say so. A
  results section where the proposed method wins on every axis reads as
  unconvincing, not impressive.

---

Next: [05 — Project directions](05-project-directions.md)
