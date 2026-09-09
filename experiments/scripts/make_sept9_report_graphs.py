"""Render milestone-report graphs for docs/Reports/Sept9.md.

Graph 1 — "state-of-the-project" phased roadmap (concept: what exists -> what runs).
Graph 2 — expected (thesis §6b, pre-registered) vs measured (exp01 results.json):
          2a recovery time, 2b damage area.

Numbers for "expected" come from docs/Phase1:Experimentation/thesis.md §6b
(illustrative values, HAT+ADWIN line ~ the streaming claim); numbers for
"measured" are read from experiments/exp01/results.json (written by
exp01_deepriver_vs_frozen_sea.py). This script only renders.

Outputs into docs/Reports/figures/. Run from repo root:
  uv run experiments/scripts/make_sept9_report_graphs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs/Reports/figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- measured (source: experiments/exp01/results.json -> drift_metrics) ---------
results = json.loads((REPO / "experiments/exp01/results.json").read_text())
dm = results["drift_metrics"]
dr_meas, fr_meas = dm["deepriver"], dm["frozen"]

# expected (thesis §6b, streaming line HAT+ADWIN as the representative claim)
dr_rec_exp = 250      # streaming recovery ~250 samples
fr_rec_exp = None     # frozen: never
dr_area_exp = 50      # damage area ~50
fr_area_exp = None    # frozen: unbounded

# -------------------------------------------------------------------------------
# Graph 1 — phased roadmap (three phases: done / running / next)
# -------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 2.6), dpi=150)
phases = [
    ("Phase 0 · Research", "argument + evidence base", "done"),
    ("Phase 1 · Exp-01", "first measured pilot (SEA)", "running"),
    ("Phase 1 · Exp-02…", "seed sweep · real streams · delay tax", "next"),
]
colors = {"done": "#2ca02c", "running": "#1f77b4", "next": "#bbbbbb"}
for i, (name, sub, state) in enumerate(phases):
    ax.barh(0, 1, left=i, height=0.55, color=colors[state], alpha=0.85)
    ax.text(i + 0.5, 0.22, name, ha="center", va="center", fontsize=11,
            fontweight="bold", color="white")
    ax.text(i + 0.5, -0.26, sub, ha="center", va="center", fontsize=8.5, color="#333")
ax.set_xlim(0, 3)
ax.set_ylim(-0.55, 0.5)
ax.axis("off")
ax.set_title("The project — where we are (Sept 9)", fontsize=12, pad=14)
fig.tight_layout()
fig.savefig(OUT / "project-roadmap-sept9.png", bbox_inches="tight")
plt.close(fig)
print(f"saved: {OUT / 'project-roadmap-sept9.png'}")

# -------------------------------------------------------------------------------
# Graph 2 — expected vs measured
# -------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=150)

# 2a — recovery time (samples after drift)
models = ["Streaming\n(DeepRiver rolling)", "Frozen\nRF"]
measured = [dr_meas["recovery_samples"], fr_meas["recovery_samples"]]  # 835 / never
expected = [dr_rec_exp, fr_rec_exp]                                    # ~250 / never
x = range(len(models))
for xi, (m, e) in zip(x, zip(measured, expected)):
    # expected bar
    if e is not None:
        ax1.bar(xi - 0.18, e, width=0.36, color="#fdae6b", hatch="//",
                label="expected (thesis)" if xi == 0 else None)
    else:
        ax1.text(xi - 0.18, 400, "never\n(expected)", ha="center", va="center",
                 fontsize=8.5, color="#8a5a20")
    # measured bar
    if m is not None:
        ax1.bar(xi + 0.18, m, width=0.36, color="#1f77b4",
                label="measured (Exp-01)" if xi == 0 else None)
    else:
        ax1.text(xi + 0.18, 400, "never\n(measured)", ha="center", va="center",
                 fontsize=8.5, color="#d62728")
if dr_rec_exp is not None:
    ax1.text(0 - 0.18, dr_rec_exp + 30, "~250", ha="center", fontsize=9,
             color="#8a5a20")
ax1.text(0 + 0.18, dr_meas["recovery_samples"] + 30, "835", ha="center", fontsize=9,
         color="#1f77b4", fontweight="bold")
ax1.set_xticks(list(x))
ax1.set_xticklabels(models)
ax1.set_ylabel("Samples from drift onset to recovery")
ax1.set_title("Recovery time — expected vs measured")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_ylim(0, 1100)
ax1.grid(axis="y", alpha=0.25)

# 2b — damage area (samples*kappa, log)
x = list(range(len(models)))
m_damage = [dr_meas["damage_area"], fr_meas["damage_area"]]
e_damage = [dr_area_exp, fr_area_exp]
for xi in x:
    if e_damage[xi] is not None:
        ax2.bar(xi - 0.18, e_damage[xi], width=0.36, color="#fdae6b", hatch="//",
                label="expected (thesis)" if xi == 0 else None)
    else:
        ax2.text(xi - 0.18, 300, "unbounded\n(expected)", ha="center", va="center",
                 fontsize=8.5, color="#8a5a20")
    ax2.bar(xi + 0.18, m_damage[xi], width=0.36, color="#1f77b4",
            label="measured (Exp-01)" if xi == 0 else None)
ax2.text(0.18, dr_meas["damage_area"] * 1.25, "31", ha="center", fontsize=9,
         color="#1f77b4", fontweight="bold")
ax2.text(1.18, fr_meas["damage_area"] * 1.25, "3,307", ha="center", fontsize=9,
         color="#1f77b4", fontweight="bold")
ax2.set_xticks(list(x))
ax2.set_xticklabels(models)
ax2.set_ylabel("Damage area (samples·κ, log scale)")
ax2.set_title("Damage area — expected vs measured")
ax2.set_yscale("log")
ax2.set_ylim(1, 100_000)
ax2.grid(axis="y", alpha=0.25)

fig.suptitle("Exp-01 measured vs the thesis's pre-registered expectations (§6b)",
             fontsize=12, y=1.0)
fig.tight_layout()
fig.savefig(OUT / "expected-vs-measured-sept9.png", bbox_inches="tight")
plt.close(fig)
print(f"saved: {OUT / 'expected-vs-measured-sept9.png'}")
