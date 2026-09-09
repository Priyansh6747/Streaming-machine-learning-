"""Generate the thesis recovery-time schematic (replaces the ASCII sketch in thesis.md §6a).

Output: docs/Phase1:Experimentation/figures/thesis-recovery-schematic.png
Run from repo root: uv run experiments/figures/make_thesis_schematic.py

The curves are illustrative (pre-registered expectations), not measured data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# --- Scenario (illustrative, matches thesis.md §6) ---------------------------
TAU = 25_000          # drift onset (samples)
D_PLUS_A = 250        # streaming recovery: d + a (samples)
KAPPA0 = 0.85         # pre-drift rolling Kappa baseline
XMAX = 30_000

x = np.linspace(0, XMAX, 4000)

def steady(t):
    """Gently rising plateau toward kappa0 (small warm-up noise)."""
    return KAPPA0 - 0.02 * np.exp(-t / 4000)


def dip(x_, depth, width, onset):
    """Smooth downward dip starting at onset, returning over `width` samples."""
    out = np.ones_like(x_) * KAPPA0
    post = x_ >= onset
    z = (x_[post] - onset) / width
    out[post] = KAPPA0 - depth * np.exp(-((z - 0.25) ** 2) / (2 * 0.18**2)) * (z < 1.0)
    return out

# 1. Streaming (HAT + ADWIN): shallow dip, bounded recovery
stream = steady(x).copy()
post = x >= TAU
z = (x[post] - TAU) / D_PLUS_A
stream[post] = KAPPA0 - 0.28 * np.exp(-((z - 0.3) ** 2) / (2 * 0.16**2))

# 2. Scheduled batch (refit k = 5000, next retrain at ~27,500): dip until retrain
R_NEXT = 27_500
sched = steady(x).copy()
mid = (x >= TAU) & (x < R_NEXT)
sched[mid] = KAPPA0 - 0.30 - 0.03 * np.sin((x[mid] - TAU) / 800)  # flat low + noise
sched[x >= R_NEXT] = KAPPA0 - 0.01  # recovers at the retrain tick

# 3. Frozen batch: permanent step down
frozen = steady(x).copy()
frozen[x >= TAU] = 0.42 + 0.02 * np.sin(x[x >= TAU] / 900)

# --- Plot ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)

ax.axhspan(KAPPA0 - 0.02, KAPPA0 + 0.02, color="green", alpha=0.07, zorder=0)
ax.axvline(TAU, color="crimson", ls="--", lw=1.2, zorder=1)
ax.text(TAU, 0.30, "drift $\\tau$", color="crimson", ha="center", fontsize=10, zorder=5)

ax.plot(x, stream, color="tab:blue", lw=2.2, label="Streaming (HAT + ADWIN)")
ax.plot(x, sched, color="tab:orange", lw=2.0, label="Scheduled batch (refit k = 5,000)")
ax.plot(x, frozen, color="tab:red", lw=2.0, ls=":", label="Frozen RF (never retrains)")

# baseline annotation
ax.annotate("$\\kappa_0$ band", xy=(1500, KAPPA0 + 0.035), fontsize=10, color="green")

# streaming damage area (small) + recovery arrow
xs = x[(x >= TAU) & (x < TAU + D_PLUS_A)]
ax.fill_between(xs, KAPPA0, stream[(x >= TAU) & (x < TAU + D_PLUS_A)],
                color="tab:blue", alpha=0.18, zorder=2)
ax.annotate("", xy=(TAU + D_PLUS_A, KAPPA0 - 0.30),
            xytext=(TAU + 40, KAPPA0 - 0.30),
            arrowprops=dict(arrowstyle="<->", color="tab:blue", lw=1.4))
ax.text(TAU + D_PLUS_A / 2, KAPPA0 - 0.38, "recovery $d + a \\approx 250$ samples",
        ha="center", fontsize=9, color="tab:blue")

# scheduled-batch damage area (medium): tau .. next retrain
xs2 = x[(x >= TAU) & (x < R_NEXT)]
ax.fill_between(xs2, KAPPA0, sched[(x >= TAU) & (x < R_NEXT)],
                color="tab:orange", alpha=0.15, zorder=2)
ax.annotate("", xy=(R_NEXT, 0.20), xytext=(TAU, 0.20),
            arrowprops=dict(arrowstyle="<->", color="tab:orange", lw=1.4))
ax.text((TAU + R_NEXT) / 2, 0.14, "wrong until next retrain ($\\sim k$ samples)",
        ha="center", fontsize=9, color="tab:orange")
ax.axvline(R_NEXT, color="tab:orange", ls=":", lw=1.0, alpha=0.8)

# frozen damage: shade to the end + unbounded arrow
xs3 = x[x >= TAU]
ax.fill_between(xs3, KAPPA0, frozen[x >= TAU], color="tab:red", alpha=0.10, zorder=2)
ax.annotate("never recovers (unbounded damage $\\rightarrow$)",
            xy=(XMAX - 150, 0.33), fontsize=9, color="tab:red", ha="right")

ax.set_xlim(0, XMAX)
ax.set_ylim(0.0, 1.02)
ax.set_xlabel("Stream time (samples)")
ax.set_ylabel("Rolling Cohen Kappa ($w = 1{,}000$)")
ax.set_title("Expected shapes after drift at $\\tau = 25{,}000$ — streaming vs batch "
             "(illustrative, not measured)")
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
ax.grid(alpha=0.25, lw=0.5)
ax.set_xticks([0, 10_000, 20_000, TAU, R_NEXT, XMAX])
ax.set_xticklabels(["0", "10k", "20k", "$\\tau$=25k", "retrain", "30k"])

out = Path(__file__).resolve().parents[2] / "docs/Phase1:Experimentation/figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "thesis-recovery-schematic.png", bbox_inches="tight")
print(f"saved: {out / 'thesis-recovery-schematic.png'}")
