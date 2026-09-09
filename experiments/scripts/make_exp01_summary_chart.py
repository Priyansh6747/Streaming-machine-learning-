"""Render the Exp-01 summary chart: recovery time and damage area, measured.

Source of truth is experiments/exp01/results.json (written by
exp01_deepriver_vs_frozen_sea.py) — this script only RENDERS the headline numbers
so the figure and the data artifact cannot drift apart. Values below mirror the
2026-09-09 seed-42 run (see results.json -> drift_metrics).

Output: docs/Phase1:Experimentation/figures/exp01-damage-area-vs-recovery.png
Run from repo root: uv run experiments/scripts/make_exp01_summary_chart.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]

# --- measured numbers, read from the data artifact -----------------------------
results = json.loads((REPO / "experiments/exp01/results.json").read_text())
dr = results["drift_metrics"]["deepriver"]
fr = results["drift_metrics"]["frozen"]

recovery = {"DeepRiver\nrolling": dr["recovery_samples"], "Frozen RF": fr["recovery_samples"]}
damage = {"DeepRiver\nrolling": dr["damage_area"], "Frozen RF": fr["damage_area"]}
never = fr["recovery_samples"] is None

# --- plot -----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.4), dpi=150)
colors = {"DeepRiver\nrolling": "#1f77b4", "Frozen RF": "#d62728"}

# left: recovery time after drift (samples)
bars = ax1.bar(recovery.keys(), [0 if v is None else v for v in recovery.values()],
               color=[colors[k] for k in recovery])
for bar, (k, v) in zip(bars, recovery.items()):
    label = "never (stream ends\nbefore it recovers)" if v is None else f"{v:,} samples"
    ax1.text(bar.get_x() + bar.get_width() / 2, 80, label, ha="center", va="bottom",
             fontsize=9, color="#444")
ax1.set_title("Recovery time after the drift\n(fewer samples = better)")
ax1.set_ylabel("Samples from drift onset (rolling Kappa back in band)")
ax1.set_ylim(0, 900)
ax1.grid(axis="y", alpha=0.25, lw=0.5)

# right: damage area (samples * Kappa), log scale — frozen is ~100x larger
ax2.bar(damage.keys(), [max(v, 1.0) for v in damage.values()],
        color=[colors[k] for k in damage])
for bar, (k, v) in zip(ax2.patches, damage.items()):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
             f"{v:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.set_title("Damage area (skill lost after the drift)\n(log scale)")
ax2.set_ylabel("samples × Kappa below baseline")
ax2.set_yscale("log")
ax2.set_ylim(1, 60_000)
ax2.grid(axis="y", alpha=0.25, lw=0.5)

fig.suptitle("Exp-01 measured, seed 42: the streaming model heals, the frozen model never does",
             fontsize=12, y=1.02)

out = REPO / "docs/Phase1:Experimentation/figures"
out.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out / "exp01-damage-area-vs-recovery.png", bbox_inches="tight")
print(f"saved: {out / 'exp01-damage-area-vs-recovery.png'}")
