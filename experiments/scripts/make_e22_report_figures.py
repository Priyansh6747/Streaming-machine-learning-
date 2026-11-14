"""Render the E22 detector-shootout figures from committed artifacts only."""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results" / "e22_detector_shootout"
OUT = ROOT / "docs" / "Reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FOCUS = [
    "DFH",
    "S1_deep",
    "S1_both",
    "ADWIN",
    "HDDMAverage",
    "HDDMWeighted",
    "PageHinkley",
    "SEED",
    "STEPD",
]
COLORS = {
    "DFH": "#7c3aed",
    "S1_deep": "#2563eb",
    "S1_both": "#0ea5e9",
    "ADWIN": "#dc2626",
    "HDDMAverage": "#f59e0b",
    "HDDMWeighted": "#6b7280",
    "PageHinkley": "#111827",
    "SEED": "#0f766e",
    "STEPD": "#14b8a6",
}
LABELS = {
    "DFH": "DF-H (ours)",
    "S1_deep": "Stage 1 deep",
    "S1_both": "Stage 1 both",
    "ADWIN": "ADWIN",
    "HDDMAverage": "HDDM-A",
    "HDDMWeighted": "HDDM-W",
    "PageHinkley": "PageHinkley",
    "SEED": "SEED (ref)",
    "STEPD": "STEPD (ref)",
}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


def load():
    ranking = json.loads((RESULTS / "ranking.json").read_text())
    with (RESULTS / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    agg = collections.defaultdict(lambda: {"f1": [], "far": []})
    for row in rows:
        if row["detector"] not in FOCUS:
            continue
        for key, source in (("f1", "f1_mean"), ("far", "far_mean")):
            try:
                value = float(row[source])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                agg[row["detector"]][key].append(value)
    stats = {
        detector: {
            "f1": float(np.mean(values["f1"])) if values["f1"] else np.nan,
            "far": float(np.mean(values["far"])) if values["far"] else np.nan,
        }
        for detector, values in agg.items()
    }
    return ranking, rows, stats


def per_dataset_ranks(rows):
    result = {}
    for dataset in ("Covtype", "Electricity"):
        subset = [
            row for row in rows
            if row["dataset"] == dataset and row["f1_mean"] not in {"", "nan"}
        ]
        cells = collections.defaultdict(list)
        for row in subset:
            cells[(row["drift_type"], row["transition"])].append(row)
        ranks = collections.defaultdict(list)
        for items in cells.values():
            items = sorted(items, key=lambda row: row["detector"])
            scores = rankdata([-float(row["f1_mean"]) for row in items], method="average")
            for row, score in zip(items, scores):
                ranks[row["detector"]].append(float(score))
        result[dataset] = {
            detector: float(np.mean(values)) for detector, values in ranks.items()
        }
    return result


def overall(ranking, stats):
    detectors = sorted(FOCUS, key=lambda name: ranking["overall"][name])
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.15]}, constrained_layout=True
    )
    ax = axes[0]
    y = np.arange(len(detectors))
    values = [ranking["overall"][name] for name in detectors]
    ax.barh(y, values, color=[COLORS[name] for name in detectors])
    ax.set_yticks(y, [LABELS[name] for name in detectors])
    ax.invert_yaxis()
    ax.axvline(2.0, color="#111827", ls="--", lw=1.1)
    ax.set_xlabel("average rank (lower is better)")
    ax.set_title("overall rank over 80 cells")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(values):
        ax.text(value + 0.12, i, f"{value:.2f}", va="center", fontsize=8)
    handles = [
        Patch(color="#7c3aed", label="ours"),
        Patch(color="#2563eb", label="Stage 1"),
        Patch(color="#0f766e", label="SEED/STEPD reference"),
        Patch(color="#dc2626", label="requested classical"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8)

    ax = axes[1]
    for name in FOCUS:
        values = stats[name]
        ax.scatter(values["far"], values["f1"], s=55, color=COLORS[name], zorder=3)
        ax.annotate(
            LABELS[name],
            (values["far"], values["f1"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.5,
            color=COLORS[name],
        )
    ax.set_xlabel("mean FAR per 1k (lower is better)")
    ax.set_ylabel("mean F1 (higher is better)")
    ax.set_title("F1 vs false alarms")
    ax.grid(alpha=0.25)
    fig.suptitle(
        "E22 — our detector beats the requested classical panel; SEED/STEPD still lead",
        fontsize=11,
    )
    path = OUT / "e22-overall.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(path.relative_to(ROOT))


def profile(ranking, rows):
    drift_types = ["class_prior", "label_swap", "feature_permutation", "feature_filtering"]
    columns = [
        ("abrupt", "feature_permutation"),
        ("gradual", "feature_permutation"),
        ("abrupt", "feature_filtering"),
        ("gradual", "feature_filtering"),
    ]
    feature = np.full((len(FOCUS), 4), np.nan)
    for i, detector in enumerate(FOCUS):
        for j, (transition, drift) in enumerate(columns):
            value = ranking["by_transition"].get(transition, {}).get(drift, {}).get(detector)
            if value is not None:
                feature[i, j] = float(value)

    f1 = collections.defaultdict(list)
    for row in rows:
        if row["detector"] in FOCUS and row["f1_mean"] not in {"", "nan"}:
            f1[(row["detector"], row["drift_type"])].append(float(row["f1_mean"]))
    f1_matrix = np.full((len(FOCUS), len(drift_types)), np.nan)
    for i, detector in enumerate(FOCUS):
        for j, drift in enumerate(drift_types):
            values = f1.get((detector, drift))
            if values:
                f1_matrix[i, j] = float(np.mean(values))

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4.9), gridspec_kw={"width_ratios": [1, 1]}, constrained_layout=True
    )
    ax = axes[0]
    image = ax.imshow(feature, vmin=1, vmax=16, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(
        [f"{transition}\n{drift.replace('_', ' ')}" for transition, drift in columns],
        fontsize=7.5,
    )
    ax.set_yticks(range(len(FOCUS)))
    ax.set_yticklabels([LABELS[name] for name in FOCUS], fontsize=8)
    for tick, name in zip(ax.get_yticklabels(), FOCUS):
        tick.set_color(COLORS[name])
    for i in range(len(FOCUS)):
        for j in range(4):
            if np.isfinite(feature[i, j]):
                ax.text(
                    j,
                    i,
                    f"{feature[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="white" if feature[i, j] < 8 else "black",
                    fontsize=7.5,
                )
    ax.add_patch(Rectangle((-0.5, -0.5), 4, 1, fill=False, edgecolor="#7c3aed", lw=2))
    ax.set_title("rank in the feature cells (1 = best)")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="average rank")

    ax = axes[1]
    image = ax.imshow(f1_matrix, vmin=0.0, vmax=0.5, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(drift_types)))
    ax.set_xticklabels([name.replace("_", "\n") for name in drift_types], fontsize=8)
    ax.set_yticks(range(len(FOCUS)))
    ax.set_yticklabels([LABELS[name] for name in FOCUS], fontsize=8)
    for tick, name in zip(ax.get_yticklabels(), FOCUS):
        tick.set_color(COLORS[name])
    for i in range(len(FOCUS)):
        for j in range(len(drift_types)):
            if np.isfinite(f1_matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{f1_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if f1_matrix[i, j] < 0.25 else "black",
                    fontsize=7.5,
                )
    ax.set_title("mean F1 by drift type")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="mean F1")
    fig.suptitle(
        "E22 — capability profile: feature drift is the strong cell, label-only drift is not",
        fontsize=11,
    )
    path = OUT / "e22-profile.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(path.relative_to(ROOT))


def main() -> None:
    ranking, rows, stats = load()
    overall(ranking, stats)
    profile(ranking, rows)
    per_dataset = per_dataset_ranks(rows)
    print("\nper-dataset ordinal rank (ties averaged):")
    for dataset, ranks in per_dataset.items():
        print(
            f"  {dataset:12s} "
            + "  ".join(f"{name}:{ranks[name]:.1f}" for name in FOCUS)
        )


if __name__ == "__main__":
    main()
