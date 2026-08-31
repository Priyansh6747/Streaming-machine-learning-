# 03 — Bringing the two together: low-precision streaming models

> **One-line version:** streaming ML's *bounded-memory, constrained-device,
> always-learning* profile is exactly the profile quantization was built for —
> and the interesting research is not "quantize a transformer," it's
> "quantize the *online loop* and see what breaks."

---

## Why these two fit so well

Streaming ML and quantization have matching constraints:

| Streaming constraint | What it needs | What quantization gives |
|---|---|---|
| Bounded memory, forever | Small model, small state | 2–8× smaller weights & activations |
| Runs on constrained devices | Cheap compute | Integer math, less memory traffic |
| Always learning | Adaptation without retraining cost | QAT-style robustness at train time |
| Drift detection on error stats | Stable, accurate error signal | A lever (and a risk) on that signal |

The fit is structural, not incidental. A streaming model that must run forever
on a small device under a fixed budget is the *canonical* quantization use
case. The LLM world (4-bit models) is the loud version; streaming is the quiet
one where it's arguably more natural.

---

## Concrete directions (what we could actually do)

### Direction A — Quantize the streaming model, measure the drift behavior

Take a River model (e.g., `linear.LogisticRegression`, `tree.HoeffdingTree`,
`neural.MLP`), quantize it (PTQ or QAT), and run it through the standard
streaming loop with a drift detector. Measure:

- **Accuracy cost** of quantization at 8-bit / 4-bit / 2-bit.
- **Memory saved** — model size plus activations.
- **Drift detection behavior** — does the detector fire at the same points?
  Does low precision make the error signal noisier (more false alarms) or
  smoother?

This is a clean, empirical, publishable-shaped result: "quantizing streaming
models to X bits costs Y accuracy and changes drift detection in Z way."

### Direction B — Quantize the drift detector itself

Drift detectors (ADWIN, DDM, Page-Hinkley) track running statistics over the
error stream. Quantize *those statistics* and ask: how much precision does a
detector actually need before it stops detecting? This is a genuinely
underexplored question — most quantization research targets models, not
detectors.

### Direction C — QAT-style streaming: learn to be robust to rounding

The insight: a streaming model is *always training*. Use quantization-aware
training online — the model learns under simulated rounding, so it stays
accurate at low precision while adapting to drift. The research question is
whether "rounding-robust" learning also makes the model more *stable* under
drift (fewer catastrophic resets, smoother adaptation).

### Direction D — Streaming-aware quantization: re-quantize as the data drifts

Standard quantization fixes the scale/range once. But in a stream, the data
distribution moves — so the *optimal* scale moves too. A scheme that
re-calibrates ranges as drift is detected is a novel angle: quantization and
drift detection collaborating instead of just coexisting.

---

## What's genuinely researchable (our honest filter)

For each direction, the question is: *would a competent practitioner be able
to answer this from existing docs, or is it open?*

- **A is mostly implementation** — a well-built experiment on known tools, with
  a measured result. Good for a first project; the contribution is the numbers
  and the analysis, not a new method.
- **B and C are genuinely open** — no standard answers exist, and the setup
  (streaming + quantization) is niche enough that the results would be new.
- **D is speculative but high-value** — needs the most care, and might be a
  stretch goal rather than a first target.

---

## The honest risk: quantization is mostly irrelevant to *small* models

The uncomfortable truth from the LLM world: quantization shines when models are
large. A small streaming model (a few thousand parameters) is already tiny —
cutting it from 40 KB to 10 KB may save nothing that matters, while the
accuracy loss is *relatively* bigger (small models have less redundancy to
absorb rounding).

So the research case for quantization in streaming is **not** "make it
smaller." It's:

1. **The device reality** — streaming models run on edge hardware (sensors,
   phones) where every bit and every watt counts, even for small models.
2. **The drift interaction** — quantized models may drift-detect differently,
   which is a *scientific* question regardless of size.
3. **The always-training angle** — QAT-in-the-loop is genuinely different from
   quantize-once-and-deploy.

If we want a project that's both honest and interesting: **lead with the
drift interaction (B/C), use the memory savings as the motivation, not the
goal.**

---

## How this connects to the existing project

- The project (see `notes/` and `research/`) is about **delayed labels and
  drift** — how detectors degrade when ground truth arrives late.
- Quantization adds a *second* axis of degradation: precision loss on top of
  label delay. "How do delayed labels + low precision interact?" is a question
  nobody has answered, and it's a natural extension of the current direction.
- The River framework gives us the streaming loop; PyTorch gives us
  quantization; the intersection is where our contribution would live.

---

## Open questions (for the next exploration session)

- Which River models are quantizable with existing tools, and which break
  (trees with histograms, non-differentiable learners)?
- How much precision does ADWIN/DDM actually need? (Direction B — quickest
  win, most novel per unit effort.)
- Does QAT-online change the *speed* of adaptation to drift, not just the
  steady-state accuracy?
- Is there a streaming library (or a way to hack River) that exposes the
  online update in a quantizable form?
