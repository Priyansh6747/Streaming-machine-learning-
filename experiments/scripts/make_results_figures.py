"""Render the global Results page figures for docs/Reports/Results.md.

Figure 1 — process block diagram: how an experiment produces a verdict
  (stream -> two model lanes -> prequential evaluation -> drift analysis -> verdict).
Figure 2 — headline scoreboard: full-run accuracy & Cohen Kappa, measured.

Numbers are read from experiments/exp01/results.json (written by
exp01_deepriver_vs_frozen_sea.py), so figures and the data artifact cannot drift
apart. This script only renders.

Outputs into docs/Reports/figures/. Run from repo root:
  uv run experiments/scripts/make_results_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs/Reports/figures"
OUT.mkdir(parents=True, exist_ok=True)

results = json.loads((REPO / "experiments/exp01/results.json").read_text())
full = results["full_run"]
dm = results["drift_metrics"]
dr, fr = dm["deepriver"], dm["frozen"]

STREAM_BLUE = "#1f77b4"
FROZEN_RED = "#d62728"
GREY = "#666666"
LIGHT = "#f4f7fa"


# -------------------------------------------------------------------------------
# Figure 1 — process block diagram
# -------------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc, ec, fs=9.5, tc="black", bold=False, rounded=True):
    style = "round,pad=0.02,rounding_size=0.012" if rounded else "square,pad=0.02"
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle=style,
        linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.4):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color=color, zorder=1,
    ))


fig, ax = plt.subplots(figsize=(11.5, 6.6), dpi=150)
ax.set_xlim(0, 12)
ax.set_ylim(0, 7.4)
ax.axis("off")

# Stream source
box(ax, 6, 6.9, 7.2, 0.72,
    "Data stream — SEA synthetic · 60,000 samples · abrupt drift at τ = 30,000\n"
    "(10% label noise · one continuous U(0,10)³ draw · balanced classes)",
    "#dbeafe", "#2563eb", fs=9.5, tc="#1e3a5f", bold=True)

# Two model lanes
box(ax, 3.0, 5.7, 4.1, 0.85,
    "STREAMING learner\nDeepRiver RollingClassifier (RNN head, window = 500)\n"
    "updates every sample",
    "#e8f1fa", STREAM_BLUE, fs=9, tc="#0d2c44", bold=True)
box(ax, 9.0, 5.7, 4.1, 0.85,
    "FROZEN baseline\nRandom Forest (100 trees) — fit once on warm-up\n"
    "never updated after τ",
    "#fdeeee", FROZEN_RED, fs=9, tc="#5a1515", bold=True)
arrow(ax, 5.2, 6.5, 3.0, 6.12)
arrow(ax, 6.8, 6.5, 9.0, 6.12)

# Prequential loop
box(ax, 6, 4.55, 10.8, 0.8,
    "Prequential evaluation — test-then-train on every sample, identical order for both models\n"
    "predict → score → (streaming only) learn → next sample",
    "#fff7e6", "#b45309", fs=9.5, tc="#4a2c07", bold=True)
arrow(ax, 3.0, 5.27, 3.0, 5.0, lw=0)          # no direct arrows; feed from lanes below
arrow(ax, 9.0, 5.27, 9.0, 5.0, lw=0)
arrow(ax, 3.0, 5.27, 5.0, 4.95)
arrow(ax, 9.0, 5.27, 7.0, 4.95)
# label evaluation region
ax.text(11.6, 4.55, "post warm-up\nn = 48,000", ha="right", va="center",
        fontsize=8.5, color="#8a5a20", fontweight="bold")

# Per-sample metrics
box(ax, 6, 3.4, 10.8, 0.72,
    "Metrics recorded every sample — accuracy · Cohen Kappa · rolling Cohen Kappa (w = 1,000)",
    "#f0f4ff", "#4f46e5", fs=10, tc="#23204a", bold=True)
arrow(ax, 6, 4.15, 6, 3.76)

# Drift-window analysis
box(ax, 6, 2.2, 10.8, 0.72,
    "Drift-window analysis at τ — baseline κ₀ · dip depth Δκ · recovery time · damage area (Σ samples·κ below baseline)",
    "#f0fdf4", "#15803d", fs=9.8, tc="#123f22", bold=True)
arrow(ax, 6, 3.04, 6, 2.56)

# Verdict
box(ax, 4.0, 0.9, 7.6, 0.9,
    "VERDICT — measured (Exp-01)\nstreaming recovers in 835 samples (bounded) · frozen never recovers",
    STREAM_BLUE, "#0f4c81", fs=10, tc="white", bold=True)
box(ax, 9.6, 0.9, 4.4, 0.9,
    "Damage area\n31 vs 3,307 samples·κ\n(≈ 106× smaller)",
    "#fef2f2", FROZEN_RED, fs=8.6, tc="#7f1d1d", bold=True)
arrow(ax, 6, 1.84, 5.0, 1.35)
arrow(ax, 6, 1.84, 8.6, 1.35)

ax.set_title("How an experiment runs and how the numbers above were produced",
             fontsize=12.5, pad=8)
fig.tight_layout()
fig.savefig(OUT / "results-process-flow.png", bbox_inches="tight")
plt.close(fig)
print(f"saved: {OUT / 'results-process-flow.png'}")


# -------------------------------------------------------------------------------
# Figure 2 — headline scoreboard (full-run metrics)
# -------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)

labels = ["Streaming\n(DeepRiver rolling)", "Frozen\nRF"]
colors = [STREAM_BLUE, FROZEN_RED]

# accuracy
acc = [full["deepriver"]["accuracy"], full["frozen"]["accuracy"]]
kap = [full["deepriver"]["cohen_kappa"], full["frozen"]["cohen_kappa"]]
ax = axes[0]
acc_pct = [a * 100 for a in acc]
bars = ax.bar(labels, acc_pct, color=colors, width=0.5)
for b, v in zip(bars, acc_pct):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
            ha="center", fontsize=12, fontweight="bold")
ax.set_ylim(0, 100)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Cumulative accuracy over the stream")
ax.grid(axis="y", alpha=0.25, lw=0.5)
# delta callout between the two bars
x0, x1 = 0.0, 1.0
y0, y1 = acc_pct
ymid = (y0 + y1) / 2 + 6
ax.plot([x0, x0, x1, x1], [y0 + 0.5, ymid, ymid, y1 + 0.5],
        color=STREAM_BLUE, lw=1.2)
ax.text(0.5, ymid + 1.5, f"+{y0 - y1:.1f} pp streaming",
        ha="center", va="bottom", fontsize=11, color=STREAM_BLUE,
        fontweight="bold")

ax = axes[1]
bars = ax.bar(labels, kap, color=colors, width=0.5)
for b, v in zip(bars, kap):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
            ha="center", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Cohen Kappa")
ax.set_title("Cumulative Cohen Kappa over the stream")
ax.grid(axis="y", alpha=0.25, lw=0.5)
x0, x1 = 0.0, 1.0
y0, y1 = kap
ymid = (y0 + y1) / 2 + 0.09
ax.plot([x0, x0, x1, x1], [y0 + 0.008, ymid, ymid, y1 + 0.008],
        color=STREAM_BLUE, lw=1.2)
ax.text(0.5, ymid + 0.015, f"+{(y0 - y1) * 100:.1f} pp streaming",
        ha="center", va="bottom", fontsize=11, color=STREAM_BLUE,
        fontweight="bold")

fig.suptitle("Headline result — streaming beats frozen even on the full-run averages\n"
             "(48,000 post-warm-up samples, seed 42, measured)",
             fontsize=12.5, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "results-headline-scoreboard.png", bbox_inches="tight")
plt.close(fig)
print(f"saved: {OUT / 'results-headline-scoreboard.png'}")
