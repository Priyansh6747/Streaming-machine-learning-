# 02 — Quantization fundamentals, then streaming angles

> **One-line version:** quantization is trading a little accuracy for a lot of
> memory and speed by storing weights and activations in fewer bits — and the
> streaming world wants exactly that, because streaming models run on
> constrained devices under a permanent memory budget.

---

## Why quantization exists

Neural networks are trained in **float32** (4 bytes per value). Most of that
precision is wasted: a trained model's weights are robust to small rounding.
Quantization stores each value in **fewer bits** — typically 8, 4, or even 2 —
and the model gets smaller, faster, and cheaper to run.

The three things you buy:

- **Memory.** A 1B-parameter model at float32 is 4 GB. At int8 it's 1 GB. At
  int4 it's 500 MB. For a device (or a streaming budget) this is the whole game.
- **Speed.** Integer math is faster than float math, and 8-bit values move
  through memory faster than 32-bit. On CPU especially, int8 is a big win.
- **Energy.** Less memory traffic + cheaper arithmetic = less power. Relevant
  for edge devices, which is where streaming models live.

---

## The core idea: scale and zero-point

A float value `x` in a known range `[min, max]` maps to an integer `q` via:

```
q = round(x / scale) + zero_point
```

- **scale** is the step size — how much one integer step is worth.
- **zero_point** is the integer that represents the float 0.0.

So instead of storing the float, you store the integer plus the (scale,
zero_point) pair for the whole tensor. Two tensors with different ranges get
different scales — that's why per-channel and per-tensor quantization differ.

The two questions that define a quantization scheme:

1. **What range do we assume?** Fixed-range (per tensor/channel) vs dynamic.
2. **What precision?** 8-bit (safe), 4-bit (tight), 2-bit (extreme).

---

## The three quantization families

### 1. Post-Training Quantization (PTQ)

Quantize an already-trained model. No retraining. Calibrate the ranges by
running a small amount of data through it and recording activation ranges.

- **Pros:** trivial to do, no training cost.
- **Cons:** accuracy loss can be significant at low bit-widths, especially for
  small models (the model can't adapt to the rounding).

### 2. Quantization-Aware Training (QAT)

Simulate the quantization *during* training — the forward pass uses fake
quantized values (rounding injected), the backward pass uses the straight-
through estimator (STEs around the rounding) so gradients still flow. The
model learns weights that are robust to being rounded.

- **Pros:** much better accuracy at low bit-widths.
- **Cons:** needs the training pipeline, which streaming models (which learn
  online) *already have* — a natural fit.

### 3. Dynamic vs static quantization

- **Static:** ranges are fixed after calibration (from PTQ or QAT).
- **Dynamic:** activations are quantized on the fly per layer using the actual
  observed range. More accurate, slightly more compute at inference.

---

## The memory math (useful to internalize)

| Precision | Bytes/value | 1M params | 1B params |
|---|---|---|---|
| float32 | 4 | 4 MB | 4 GB |
| float16 / bfloat16 | 2 | 2 MB | 2 GB |
| int8 | 1 | 1 MB | 1 GB |
| int4 | 0.5 | 0.5 MB | 500 MB |
| int2 / 1.58-bit | ~0.25 | ~0.25 MB | ~250 MB |

Note: weights aren't the only cost. **Activations** and **optimizer states**
(Adam keeps two float32 moments per parameter) also consume memory. For
streaming models, the model is usually small; the *bounded-memory* constraint
is about the whole system over infinite time.

---

## When quantization costs you

- **Small models are more fragile.** A large model has redundancy; a small
  streaming model has less slack, so the same rounding hurts more.
- **Outliers kill the range.** If one weight is far outside the pack, the
  scale stretches and everything else loses precision. This is the reason
  "per-channel" and "outlier-aware" schemes exist.
- **The last layers hurt most.** Output layers and softmax are sensitive to
  precision — often kept at higher precision.

---

## Streaming-specific angles (the part we care about)

This is where quantization meets the project:

- **Streaming models are small.** They run on constrained devices with a
  permanent memory budget — exactly the profile quantization is built for.
- **The bounded-memory constraint is *the* constraint.** Quantization is a
  direct lever on it: a 2-bit model is a fraction of the size of a float32
  one, forever, not just at inference.
- **Online learning + QAT is a natural pair.** QAT needs training-time
  simulation; streaming models are *always* training. A streaming model that
  keeps adapting at low precision is a coherent research direction.
- **Drift detection is numerical.** Detectors watch error statistics over
  time. If quantization changes the error distribution, it changes what the
  detector sees — a real interaction worth studying.

---

## Tooling to know

- **River** — the streaming framework; supports `linear`, `tree`, `neural`
  models but quantization is not its focus.
- **PyTorch** — `torch.quantization` (PTQ/QAT), `torchao` for low-bit work.
- **Hugging Face** — `bitsandbytes`, GPTQ, AWQ for LLMs (the industry is
  pushing to 4-bit and below).
- **ONNX Runtime** — int8 quantization for deployment.
- **keras / TF** — `tf.lite` quantization for edge.

---

## Quick mental model

```
float32  --train-->  weights (full precision)
                          |
             PTQ: quantize after      QAT: simulate during
                          |
                     int8 / int4 / int2  weights + scale/zero-point
                          |
              smaller, faster, cheaper -- with a little accuracy loss
```

---

## Open questions for us

- Which models in River are worth quantizing — and does quantizing them break
  their *online update* rules (e.g., trees that store histograms)?
- Does low-precision training change *drift detection* — either the detector's
  sensitivity or the model's adaptation speed?
- Can QAT-style "robust to rounding" training make a streaming model *more*
  stable under drift, not just smaller?
- Is there a streaming-specific quantization scheme — one that re-quantizes as
  the data distribution drifts?
