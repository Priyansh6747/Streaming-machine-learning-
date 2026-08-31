# 01 — Mapping transformer knowledge onto quant trading

> **One-line version:** your transformer knowledge transfers to quant trading
> through the *habits* (learned representations, attention over context, the
> eval mindset) — but the *problem shape* is different: financial data is
> non-stationary, low signal-to-noise, and the ground truth (future returns)
> is only knowable in hindsight.

---

## The good news: the fit is real

Finance is a data-stream problem at its core. Prices tick continuously, regimes
change, and a model that doesn't adapt dies. The streaming ML skills you've
been building are *directly* market-shaped. And your transformer background
gives you a head start on the representation side. This is not a stretch — the
two fields genuinely overlap.

---

## What your transformer knowledge already covers

| Transformer muscle | Quant trading equivalent | Transfers? |
|---|---|---|
| Learned embeddings of tokens | Learned feature representations of market state | Yes, core skill |
| Attention over a context window | Attention over recent price/order-flow context | Yes, with care |
| The train / val / test split ritual | Walk-forward / backtest with no leakage | Concept, re-tooled |
| The eval harness (accuracy, F1) | Sharpe, drawdown, hit rate, per-trade stats | Concept, re-tooled |
| Positional encodings | Time features — time-of-day, session, calendar effects | Yes |
| Sequence modeling (LSTM/attention) | Modeling price paths as sequences | Yes, but overfit-prone |
| The loss as a training signal | The P&L / alpha as the real objective | Re-framed |
| Batching, epochs, i.i.d. assumption | Non-stationary, ordered, drifting data | **No** — must unlearn |

---

## Where attention genuinely helps in markets

Attention is not useless in finance — it's just *differently* useful than in
NLP. Three places it earns its keep:

1. **Cross-sectional attention** (over assets, not time): which of 500 stocks
   matter most *right now* for predicting a target? Cross-sectional attention
   is a natural, defensible use — the "context" is the market, and which
   assets matter shifts with regime.
2. **Feature selection over windows**: attention over a rolling window of
   features (not raw prices) to decide *which recent context* matters — e.g.,
   "the last 3 days of volatility matter more than the last 20."
3. **Regime detection**: attention weights as a soft indicator of *which
   regime* the model thinks it's in — a bridge to the drift-detection side of
   the project.

The trap to avoid: **attention over raw prices as a universal sequence model**
is where overfitting lives. Markets are not language; the "next token" is
noise-dominated, and attention's power to memorize long patterns becomes a
liability, not an asset.

---

## The honest differences (what's new)

- **Non-stationarity.** NLP data is mostly stationary. Markets are
  non-stationary by construction — the distribution that generated yesterday's
  prices is not today's. This is the single biggest shift.
- **Low signal-to-noise.** A good predictive edge is a tiny edge over noise.
  Transformers are built to extract strong structure; markets have weak
  structure. The failure mode is fitting noise.
- **Hindsight ground truth.** In NLP, labels are in the data. In trading,
  the label (future return) is a *realized* value you can only know after the
  fact — the delayed-label problem in its purest form.
- **Survivorship and leakage.** The dataset itself lies: delisted stocks
  vanish from history, and look-ahead bias is everywhere. Your transformer
  instinct (clean split, careful features) is the *right* instinct here.
- **The eval metric is money, not accuracy.** Accuracy doesn't compound.
  Sharpe, drawdown, and win rate do. The objective function changes shape.

---

## How to use transformer knowledge *as leverage*

Concrete moves that work:

1. **Steal the eval mindset, upgrade it to walk-forward.** The habit of
   "never evaluate on training data" becomes "never let future data leak into
   the past." Walk-forward / expanding-window evaluation is the market
   version of your hold-out discipline.
2. **Reuse the representation layer.** Whatever feature engineering you did
   (embeddings, normalized columns, time features) maps to the market-input
   pipeline. The model changes; the input hygiene survives.
3. **Think of drift detection as regime monitoring.** You know what a rising
   validation loss means. Drift detectors (ADWIN, DDM) are the streaming
   version: watch the error/P&L distribution, react when the regime shifts.
4. **Port the optimizer discipline, not the architecture.** Start with linear
   models and gradient-boosted trees on *good features* before attention. The
   edge in quant is usually in the features and the risk model, not the
   architecture.

---

## The bridge summary

| | Transformer world | Quant trading |
|---|---|---|
| Data | Fixed dataset, reusable | Ticking, non-stationary, non-reusable |
| Samples | i.i.d., shuffleable | Ordered, drifting, regime-dependent |
| Ground truth | In the data | Realized later (delayed labels) |
| Signal | Strong structure | Weak structure, noise-dominated |
| Eval | Accuracy / loss | Sharpe, drawdown, P&L |
| The research problem | Scale, generalization | Drift, adaptation, edge decay, risk |

**What transfers:** the eval mindset, representation learning, attention as a
concept, time-feature discipline, the toolkit.
**What doesn't:** the i.i.d. assumptions, the "one model fits all" instinct,
training to accuracy.
**What's new (and where the project is):** non-stationarity, delayed labels,
edge decay, regime detection, risk-aware evaluation.

---

## Open questions for us

- Is cross-sectional attention (over assets) a *defensible* use of
  transformers in a market model, or does it overfit as badly as temporal
  attention?
- Can drift detection double as *regime detection* — giving a trading signal
  ("the regime changed, adapt the model") rather than just a warning?
- How does the delayed-label problem (see `notes/02-concept-drift.md`,
  `research/02-delayed-labels.md`) bite quant models specifically — and can
  the project's delayed-label research *be* a market research project?
- What does a *minimal honest* backtest look like — one that can't be accused
  of look-ahead bias — for a streaming market model?
