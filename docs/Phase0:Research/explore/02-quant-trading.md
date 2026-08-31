# 02 — Financial quant fundamentals

> **One-line version:** quant trading is using math and data to make
> *statistically disciplined* bets on markets — the edge is small, the data is
> noisy and non-stationary, and the difference between a good quant and a
> gambler is the evaluation, risk, and leakage hygiene around the model.

---

## What "quant" actually means

Quantitative trading = making trading decisions from *numerical models and
data* rather than gut feel. The spectrum:

- **Statistical arbitrage** — find and exploit temporary mispricings between
  related assets (pairs, baskets).
- **Momentum / trend following** — the market's tendency to continue in a
  direction for a while; classic systematic strategy.
- **Mean reversion** — the market's tendency to snap back to an average;
  the statistical opposite of momentum.
- **Market making** — profit from the bid/ask spread by providing liquidity.
- **High-frequency (HFT)** — microstructure, sub-second edges. Not our lane
  with streaming ML at sample rates like one bar per minute.
- **Factor investing** — long/short portfolios tilted toward known factors
  (value, momentum, size, volatility).

For a research project, the accessible lanes are **statistical arbitrage,
momentum, and mean reversion** on moderate-frequency data (minutes to days).
HFT is out — it's a latency/microstructure game, not an ML problem.

---

## The quant pipeline (the mental model to keep)

```
raw data → features → model → position sizing → execution → P&L
                ↑                                   ↓
                └────── risk & evaluation loop ←─────┘
```

Every stage is a discipline:

1. **Raw data** — prices, volumes, order flow, fundamentals, alternative data.
   Hygiene here is everything: adjusted prices (splits/dividends), no
   look-ahead, survivorship-free universes.
2. **Features** — returns, volatility, momentum, ratios, spreads, time-of-day,
   regime indicators. *This is where edges live.* Most quant alpha is feature
   engineering, not architecture.
3. **Model** — predicts a *target*: direction, return magnitude, volatility,
   or a mispricing signal. Linear models, trees, and (carefully) neural nets.
   The model is usually the *smallest* contributor to edge.
4. **Position sizing** — how much to bet, given the edge and the uncertainty
   (Kelly, fractional Kelly, vol targeting). Model confidence alone is not
   enough.
5. **Execution** — actually trading without eating the edge in costs
   (spread, fees, slippage).
6. **Risk & evaluation** — the loop that keeps you honest: Sharpe, drawdown,
   exposure, leakage checks, out-of-sample discipline.

---

## Key concepts, in plain terms

- **Alpha** — the part of a return that isn't explained by the market; the
  edge your model adds. "Alpha decay" = the edge fading as others find it.
- **Beta** — the market's general direction; a long-only portfolio has market
  beta. Quant strategies try to isolate alpha *net of* beta.
- **Sharpe ratio** — (excess return) / (volatility). The standard scorecard.
  A Sharpe above 1 is good; above 2 is great. It's the *risk-adjusted* return.
- **Drawdown** — peak-to-trough loss. A strategy with great Sharpe but a
  40% drawdown is psychologically and practically hard to run. Drawdown is
  what actually kills funds.
- **P&L / returns** — log returns (continuous compounding) are the natural
  scale for models; simple returns for portfolio math. Log returns are
  roughly additive over time, simple returns aren't.
- **Volatility** — the standard deviation of returns; the thing that scales
  with risk. Vol is itself forecastable (GARCH, realized vol) and mean-reverts.
- **Regime** — a persistent market state (bull/bear, high/low vol). Regime
  shifts are concept drift wearing a suit.
- **Bid/ask spread** — the cost of trading. If your edge per trade is smaller
  than the spread, you lose regardless of model quality.
- **Slippage** — the difference between your intended and actual fill price.
  Backtests that ignore slippage are fiction.

---

## Data types (and their character)

| Type | Frequency | Character | Notes |
|---|---|---|---|
| Ticks | ms | Noisy, huge | HFT lane; microstructure |
| OHLCV bars | 1m–1d | Standard for ML | The workhorse: open/high/low/close/volume |
| Fundamentals | quarterly | Sparse, lagged, survivorship-heavy | Point-in-time data is the only honest version |
| Order book | ms | Microstructure | Depth, imbalance; HFT-ish |
| Alternative | mixed | Novel, sparse | Sentiment, satellite, search trends |
| Options | mixed | Derivative structure | Implied vol is a rich feature source |

**The two data traps that end projects:**

1. **Survivorship bias** — backtesting only stocks that still exist today.
   Delisted stocks (which often failed) vanish from the historical dataset,
   inflating returns. You must include delisted names and mark them dead.
2. **Look-ahead bias** — using information not available at the time (future
   prices, restated fundamentals, same-day close to predict same-day return).
   The fix is strict time-ordering: features computed *before* the prediction
   point, labels *after* it.

---

## Backtesting: the honest version

A backtest is a *simulation* of a strategy on history. It is not proof — it's
a sanity check that's easy to fool. The disciplines:

- **Walk-forward evaluation** — train on a window, test on the next, roll
  forward. Never train and test on overlapping periods.
- **Expanding vs rolling window** — expanding = use all history so far (more
  data, adapts slower to regime); rolling = fixed recent window (less data,
  adapts faster). Streaming ML maps to this directly.
- **Transaction costs** — model spread + slippage + fees explicitly. A
  strategy profitable in zero-cost backtest often dies with realistic costs.
- **Out-of-sample discipline** — final evaluation on data never touched during
  development. This is the quant version of your hold-out test set.
- **Multiple-testing honesty** — trying 500 features/strategies and reporting
  the best one is *p-hacking by another name*. Report the distribution, not
  the winner.

---

## The honest math (what wins and what doesn't)

- **The edge is small.** A great strategy wins 52–55% of the time, or has a
  Sharpe of 1–2. Anyone promising much more is selling something.
- **Volatility is the enemy of compounding.** Deep drawdowns need larger
  recoveries: a 50% loss needs a 100% gain to break even. Risk management is
  not optional — it's the strategy.
- **More complexity = more overfitting risk.** Linear models on good features
  routinely beat deep models on mediocre features, in production.
- **Transaction costs are a tax on frequency.** High-frequency strategies die
  first when costs are realistic. Lower frequency (bars/days) tolerates costs
  better.
- **The market is adaptive.** Whatever edge you find decays as it's
  exploited — alpha decay is the reason streaming adaptation is *required*,
  not optional.

---

## Tooling landscape

- **Python** — the ecosystem: `pandas` (data), `numpy` (math), `scikit-learn`
  / `lightgbm` / `xgboost` (models), `statsmodels` (econometrics).
- **Backtesting** — `backtrader`, `vectorbt`, `zipline`, `bt`.
- **Data** — `yfinance` (free, fine for learning), `ccxt` (crypto), paid
  feeds for production (Polygon, EOD, Bloomberg/Refinitiv).
- **Streaming ML** — `River` (our framework) — a natural fit for the online
  loop that live markets *are*.
- **Research infra** — Jupyter, `mlflow` (experiment tracking), `wandb`.

---

## Open questions for us

- Momentum, mean reversion, or stat-arb as the first strategy family? (Each
  maps differently onto streaming models.)
- Which data source: free daily bars (yfinance), free crypto (ccxt), or a
  synthetic stream (generate regimes ourselves) for a clean research setup?
- How do we measure a *streaming* trading strategy — Sharpe computed
  prequentially, per-bar, with realistic costs?
- Does the delayed-label angle (labels arrive after the trade decision) make
  our project *uniquely* suited to markets, or uniquely unsuited? (I suspect
  suited — see 03.)
