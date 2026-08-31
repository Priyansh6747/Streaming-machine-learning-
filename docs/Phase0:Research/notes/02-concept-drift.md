# 02 — Concept drift (the actual research problem)

If this project has a thesis, it lives here.

## Definition

A model learns a mapping from inputs to outputs. Formally it's estimating a joint
distribution $P(X, y)$, usually decomposed as $P(y \mid X) \cdot P(X)$.

**Concept drift** is when that distribution changes between time $t$ and time $t+\Delta$:

$$P_t(X, y) \neq P_{t+\Delta}(X, y)$$

The world that produced your training data is not the world you're now predicting in.

## Two flavours — and the distinction matters

**Real concept drift** — $P(y \mid X)$ changes.
The *relationship itself* changed. The same input now warrants a different output.
This genuinely breaks the model. Example: an email with a particular set of words
used to be spam; spammers adapted and now legitimate marketing uses those words too.

**Virtual drift** (a.k.a. covariate shift / data drift) — $P(X)$ changes but
$P(y \mid X)$ stays the same.
The inputs you see have shifted, but the underlying rule is intact. Example: your
user base expands into a new age bracket — different feature distribution, but the
rule "high transaction velocity + new device = fraud" still holds.

Why it matters: **virtual drift doesn't necessarily hurt accuracy; real drift does.**
Many "drift detectors" watch the input distribution and therefore fire on virtual
drift — a false alarm from the model's point of view. Detectors that watch the
*error stream* only fire on drift that actually costs you accuracy. This tension is
itself a legitimate thing to investigate.

## Types, by shape over time

This is the taxonomy every paper uses. Assume the y-axis is "which concept is active":

```
SUDDEN (abrupt)         GRADUAL                 INCREMENTAL             RECURRING
                                                                    
 ────┐                   ──┐ ┌┐ ┌──┐             ──╲                 ──┐   ┌──┐   ┌──
     │                     └─┘└─┘  └───            ╲___                └───┘  └───┘
     └────────                                        ╲──
                                                     
concept A flips        A and B interleave,      A morphs smoothly     A and B alternate
to concept B           B wins over time         into B                (e.g. seasonal)
instantly
```

- **Sudden / abrupt** — instantaneous switch. Easiest to detect, most damaging on impact.
- **Gradual** — during a transition window, samples come from A *or* B with shifting
  probability. Hard to detect: the error rises slowly and noisily.
- **Incremental** — the concept deforms continuously through intermediate states.
- **Recurring** — old concepts come back (day/night, weekday/weekend, seasons).
  Special because *forgetting* the old concept is actively wasteful — this motivates
  approaches that archive and restore old models rather than discarding them.

There's also **blips / outliers** — a one-off anomaly that is *not* drift. A good
detector must not fire on these. This is precisely the
stability–plasticity trade-off: react fast enough to catch real drift, but not so
fast that noise triggers a reset.

## Three responses to drift

**1. Blind adaptation (no detector).**
Just keep learning and let recent data outweigh old data — sliding windows,
exponential decay / fading factors, or a learning rate that never goes to zero.
Simple, always-on, but adapts slowly to sudden drift and permanently discards
old-but-still-valid knowledge.

**2. Informed adaptation (detector + reaction).**
Run an explicit drift detector. When it fires, do something drastic: reset the
model, retrain on a recent buffer, swap in a background model that's been training
in parallel. Reacts sharply, but is only as good as the detector — a false alarm
throws away a perfectly good model.

**3. Ensembles.**
Keep a pool of models, weight them by recent performance, drop the worst, add fresh
ones. Drift handling is emergent rather than explicit. This is what
**Adaptive Random Forest (ARF)** does, and it's the strongest general baseline.
Internally it uses ADWIN per-tree — so it's arguably a hybrid of 2 and 3.

## How detectors work (the intuition)

Most detectors don't look at your features at all. They watch a **1-D stream of the
model's errors** (1 = wrong, 0 = right) and ask: *has this stream's mean changed?*

| Detector | Idea | Notes |
|---|---|---|
| **DDM** (Gama et al., 2004) | Track error rate $p$ and its std $s$; warn at $p + s > p_{min} + 2s_{min}$, alarm at $3s$. | Assumes error should *decrease* as learning proceeds. Weak on gradual drift. |
| **EDDM** | Like DDM but tracks the *distance between* errors. | Designed to fix DDM's gradual-drift weakness. |
| **ADWIN** (Bifet & Gavaldà, 2007) | Keep an adaptive window; whenever two sub-windows have statistically different means, drop the older part. | Rigorous guarantees, no window-size parameter, the de-facto standard. |
| **Page-Hinkley** | Sequential change detection (CUSUM family); accumulates deviation from the running mean. | Classical statistics, sensitive to its threshold. |
| **KSWIN** (Raab et al., 2020) | Kolmogorov–Smirnov test between a sliding window and a reference sample. | Non-parametric, distribution-based. |

The trade-off is always the same: **sensitivity vs. false alarms.** Tune a detector
to catch drift faster and it starts firing on noise. This trade-off curve is a
perfectly good research object — you can *measure* it.

## Why this makes a good project

Because with **synthetic streams you know the ground truth.** You choose the drift
position. That means you can compute things you can never compute on real data:

- **Detection delay** — samples between true drift and the alarm.
- **False positives** — alarms where no drift occurred.
- **Missed detections** — drift that was never flagged.
- **Recovery time** — samples until accuracy returns to pre-drift level.

Real datasets give you realism; synthetic streams give you measurable truth. A
serious project uses **both**, and says why.

---

Next: [03 — The River library](03-river-library.md)
