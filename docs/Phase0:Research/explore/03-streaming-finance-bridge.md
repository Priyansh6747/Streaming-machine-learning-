# 03 — Why finance IS a streaming problem (the bridge)

> **One-line version:** live markets are the *purest* streaming ML problem that
> exists — data arrives one bar at a time, forever, the distribution drifts
> (regimes), and the ground truth arrives late (future returns). Everything
> the project has been studying applies to markets *directly*, with zero
> translation.

---

## The mapping is almost embarrassingly direct

| Streaming ML concept | Market equivalent | How direct? |
|---|---|---|
| Data stream, one sample at a time | Price bars, one per interval | Identical |
| Concept drift | Regime shifts (bull/bear, vol expansion) | Identical mechanism |
| Delayed labels | Future returns realized after the decision | Identical, worse |
| Bounded memory | Can't store the full tick history | Identical constraint |
| Prequential eval (test-then-train) | Walk-forward, out-of-sample discipline | Identical idea |
| Model adaptation / reset | Regime-aware retraining, model re-weighting | Identical need |
| Forgetting management | A model stuck in the old regime trades badly | Identical failure |

The whole point of the project — "the model must notice when it becomes wrong
and adapt" — is the *entire job* of a live trading model. A fund that doesn't
adapt to regime is a fund that blows up. Markets are not an *application* of
streaming ML; they are its natural habitat.

---

## The three-way intersection that makes this project special

The project has three threads. In markets, all three are *live* simultaneously:

1. **Streaming learning** — the online loop (`predict → score → learn`) is
   exactly how a live model trades: predict next bar's move, act, learn from
   what actually happened.
2. **Concept drift / regime detection** — drift detectors (ADWIN, DDM,
   Page-Hinkley) are regime detectors wearing a different name. A drift alarm
   in a market model *is* a regime-change signal — actionable, not just
   diagnostic.
3. **Delayed labels** — in trading, the label is the future return, and it is
   *inherently* delayed: you decide now, the outcome arrives later. The
   project's core research problem is not an edge case in trading — it's the
   *definition* of the trading problem.

This is the strongest possible fit: the research question the project is
already chasing (how do delayed labels break streaming models?) is the exact
question a live trading system faces every single bar.

---

## Where the research value is

### Direction A — Delayed labels in trading: the honest framing

A trading model's label (future return) arrives after the trade. The project's
delayed-label work applies *verbatim*: quantify how label delay degrades a
streaming trading model's decisions, and whether drift detectors can still
catch regime change when the error signal is stale.

### Direction B — Drift detection as regime detection

Drift detectors as a *trading signal*, not just a warning: when ADWIN fires,
the regime changed — what should the model *do* (reset, reweight recent data,
shift to a different strategy)? This is the "adapt" side of the streaming loop
given a concrete, financial meaning.

### Direction C — The quant + streaming + transformer triangle

A model that combines: streaming adaptation (regime-aware learning), delayed
labels (the honest trading constraint), and transformer-style representation
(cross-sectional attention over assets, carefully used). The contribution is
the *combination*, not any single piece — and it's a combination nobody has
cleanly built.

### Direction D — The degenerate case nobody checks

What happens to a streaming trading model when the label delay is *longer than
the regime*? (E.g., labels arrive after the regime has already flipped.) If
detectors can't keep up, the model systematically trades on stale truth — a
clean, publishable negative result, in the spirit of the project's existing
scope (`notes/08-scope.md`, `idea/synopsis.md`).

---

## The honest risks (keep these visible)

- **Backtest overfitting is the #1 killer.** Any strategy can be made to look
  great in hindsight. The disciplines in `02-quant-trading.md` (walk-forward,
  costs, out-of-sample, multiple-testing honesty) are not optional.
- **Data quality is half the battle.** Adjusted prices, survivorship-free
  universes, point-in-time fundamentals. Garbage data = confident garbage
  results.
- **The edge is small and decays.** Even a good project's results will be
  modest. That's normal; the research value is in the *mechanism* (how delay
  and drift interact), not in claiming a money machine.
- **Costs decide everything.** A backtest without realistic spread/slippage
  is fiction. Especially at bar-level frequencies.
- **"Quant" tempts shortcuts.** A fake backtest with a great curve is the
  quant equivalent of a p-hacked result. Our hygiene is our credibility.

---

## The practical starting recipe

A minimal, honest, research-grade setup:

1. **Synthetic or free data first.** Either generate a stream with known
   regimes (clean science: we control the ground truth), or use free daily
   bars (yfinance) / free crypto (ccxt). Synthetic first isolates the
   *method*; real data validates it.
2. **A streaming baseline.** A River model (e.g., `linear.LogisticRegression`
   on engineered features) through the prequential loop, with a drift
   detector attached.
3. **A delayed-label regime.** Feed labels with delay τ; measure how accuracy,
   drift detection, and *simulated P&L* degrade as τ grows.
4. **The bridge result.** Show where the model breaks and whether drift
   detection can rescue it — the same shape as the project's existing delayed-
   label research, but on a market-flavored stream.

---

## Open questions (next exploration session)

- Synthetic regime stream vs real market data first — which gives a cleaner
  *research* answer?
- Which strategy family (momentum / mean reversion / stat-arb) maps most
  cleanly onto a streaming model with delayed labels?
- Do drift detectors meaningfully *predict* regime changes in markets, or
  only confirm them after the fact?
- What's the minimal honest cost model (spread + slippage + fees) for a
  bar-frequency backtest?
