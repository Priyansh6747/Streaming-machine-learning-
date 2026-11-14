"""Render the E18/E19 report figures from committed artifacts only.

    uv run python experiments/scripts/make_e18_e19_figures.py
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "docs" / "Reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DEEP = "#2563eb"
DEEP2 = "#0f766e"
CLASSICAL = "#6b7280"
ERROR = "#dc2626"
FUSION = "#7c3aed"
SCALAR = "#94a3b8"
LIGHT = "#cbd5e1"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})


def load(exp: str) -> dict:
    return json.loads((RESULTS / exp / "summary.json").read_text())


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(path.relative_to(ROOT))


def _signal_color(name: str) -> str:
    if name == "error_rate":
        return ERROR
    if name == "feature_ks":
        return CLASSICAL
    return DEEP


def e18_signal_gate() -> None:
    summary = load("e18")
    overall = summary["overall"]
    deep_order = [
        "feature_ks",
        "embedding_shift",
        "embedding_var_shift",
        "error_rate",
        "ssl_loss",
        "head_disagreement",
        "internal_adwin",
        "entropy",
    ]
    signals = [name for name in deep_order if name in overall]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 4.8),
        gridspec_kw={"width_ratios": [1, 1.35]},
        constrained_layout=True,
    )
    ax = axes[0]
    values = [overall[name] for name in signals]
    colors = [_signal_color(name) for name in signals]
    y = np.arange(len(signals))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y, [name.replace("_", " ") for name in signals])
    ax.invert_yaxis()
    ax.set_xlim(0.4, 0.85)
    ax.axvline(0.75, color=ERROR, ls="--", lw=1.2, label="gate: AUC 0.75")
    ax.set_xlabel("AUC (higher is better)")
    ax.set_title("overall AUC, 80 cells")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(values):
        ax.text(value + 0.004, i, f"{value:.3f}", va="center", fontsize=8)
    handles = [
        Patch(color=DEEP, label="deep signal"),
        Patch(color=CLASSICAL, label="classical control"),
        Patch(color=ERROR, label="error-rate baseline"),
        Line2D([0], [0], color=ERROR, ls="--", label="gate: AUC 0.75"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=8)

    ax = axes[1]
    drift_types = ["class_prior", "feature_permutation", "feature_filtering", "label_swap"]
    matrix = np.full((len(signals), len(drift_types)), np.nan)
    for i, name in enumerate(signals):
        for j, drift in enumerate(drift_types):
            value = summary["by_drift_type"].get(drift, {}).get(name)
            if value is not None:
                matrix[i, j] = float(value)
    image = ax.imshow(matrix, vmin=0.40, vmax=0.90, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(drift_types)))
    ax.set_xticklabels([name.replace("_", "\n") for name in drift_types], fontsize=8)
    ax.set_yticks(range(len(signals)))
    ax.set_yticklabels([name.replace("_", " ") for name in signals])
    for tick, name in zip(ax.get_yticklabels(), signals):
        tick.set_color(_signal_color(name))
    for i in range(len(signals)):
        for j in range(len(drift_types)):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] < 0.65 else "black",
                    fontsize=7,
                )
    ax.set_title("AUC by drift type")
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02, label="AUC")
    fig.suptitle(
        "E18 — the pre-registered AUC gate fails: best deep 0.616 vs error 0.598",
        fontsize=11,
    )
    save(fig, "e18-signal-gate.png")


def e18_timing() -> None:
    detection = json.loads((RESULTS / "e18" / "detection.json").read_text())
    rows = {
        (row["dataset"], row["drift_type"], row["transition"], row["signal"]): row
        for row in detection["rows"]
    }
    cells = [
        ("Covtype", "abrupt"),
        ("Covtype", "gradual"),
        ("Electricity", "abrupt"),
        ("Electricity", "gradual"),
    ]
    signals = ["ssl_loss", "embedding_var_shift", "feature_ks", "error_rate"]
    colors = {"ssl_loss": DEEP, "embedding_var_shift": DEEP2, "feature_ks": CLASSICAL, "error_rate": ERROR}
    labels = {
        "ssl_loss": "SSL loss",
        "embedding_var_shift": "embedding var shift",
        "feature_ks": "feature KS",
        "error_rate": "error rate",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, constrained_layout=True)
    for ax, drift in zip(axes, ("feature_permutation", "feature_filtering")):
        x = np.arange(len(cells))
        width = 0.2
        for offset, signal in zip((-1.5, -0.5, 0.5, 1.5), signals):
            heights = []
            misses = []
            for dataset, transition in cells:
                row = rows.get((dataset, drift, transition, signal))
                if row is None or not row.get("median_delay"):
                    heights.append(0.0)
                    misses.append(True)
                else:
                    heights.append(float(row["median_delay"]))
                    misses.append(False)
            positions = x + offset * width
            ax.bar(positions, heights, width * 0.9, color=colors[signal], label=labels[signal])
            for position, height, miss, (dataset, transition) in zip(
                positions, heights, misses, cells
            ):
                row = rows.get((dataset, drift, transition, signal))
                if miss:
                    ax.text(position, 40, "miss", ha="center", va="bottom", fontsize=7, rotation=90, color=colors[signal])
                elif row and row.get("hit_rate", 1.0) < 1.0:
                    ax.text(position, height * 1.04, f"{row['hit_rate']*100:.0f}%", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{dataset}\n{transition}" for dataset, transition in cells], fontsize=8)
        ax.set_yscale("symlog", linthresh=100)
        ax.set_ylim(0, 3_200)
        ax.set_ylabel("median detection delay (samples, lower is better)")
        ax.set_title(drift.replace("_", " "))
        ax.grid(axis="y", alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "E18 — feature permutation is detected earlier; filtering is detected where error is blind",
        fontsize=11,
    )
    save(fig, "e18-timing.png")


def _e19_tables():
    ranking = json.loads((RESULTS / "e19_kdd_pilot" / "ranking.json").read_text())
    with (RESULTS / "e19_kdd_pilot" / "summary.csv").open() as handle:
        summary_rows = list(csv.DictReader(handle))
    aggregate = collections.defaultdict(lambda: {"f1": [], "far": []})
    for row in summary_rows:
        try:
            f1 = float(row["f1_mean"])
        except (TypeError, ValueError):
            continue
        try:
            far = float(row["far_mean"])
        except (TypeError, ValueError):
            continue
        aggregate[row["detector"]]["f1"].append(f1)
        aggregate[row["detector"]]["far"].append(far)
    stats = {
        detector: {
            "f1": float(np.mean(values["f1"])),
            "far": float(np.mean(values["far"])),
        }
        for detector, values in aggregate.items()
    }
    return ranking, summary_rows, stats


def _per_dataset_ranks(summary_rows):
    result = {}
    for dataset in ("Covtype", "Electricity"):
        subset = [
            row for row in summary_rows
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


def _dataset_positions(per_dataset):
    positions = {}
    for dataset, ranks in per_dataset.items():
        items = sorted(ranks.items(), key=lambda item: item[1])
        values = np.asarray([value for _, value in items], dtype=float)
        scores = rankdata(values, method="average")
        positions[dataset] = {
            name: float(score) for (name, _), score in zip(items, scores)
        }
    return positions


def e19_overall() -> None:
    ranking, _, stats = _e19_tables()
    overall = ranking["overall"]
    detectors = [
        name for name in sorted(overall, key=lambda name: overall[name])
        if name != "2of3_both"
    ]
    colors = []
    for name in detectors:
        if name in {"DFH", "2of3_both"}:
            colors.append(FUSION)
        elif name in {"SEED", "STEPD"}:
            colors.append(DEEP)
        elif name == "2of3_nodeep":
            colors.append(CLASSICAL)
        else:
            colors.append(LIGHT)

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1, 1.1]}, constrained_layout=True
    )
    ax = axes[0]
    y = np.arange(len(detectors))
    values = [overall[name] for name in detectors]
    ax.barh(y, values, color=colors)
    ax.set_yticks(y, detectors)
    ax.invert_yaxis()
    ax.axvline(2.0, color=ERROR, ls="--", lw=1.2, label="gate: top-2")
    ax.set_xlabel("average rank (lower is better)")
    ax.set_title("overall rank over 80 cells")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(values):
        ax.text(value + 0.08, i, f"{value:.2f}", va="center", fontsize=8)
    handles = [
        Patch(color=DEEP, label="STEPD / SEED"),
        Patch(color=FUSION, label="DFH"),
        Patch(color=CLASSICAL, label="no-feature control"),
        Patch(color=LIGHT, label="other"),
        Line2D([0], [0], color=ERROR, ls="--", label="gate: top-2"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8)

    ax = axes[1]
    key = {"DFH", "SEED", "STEPD", "ADWIN", "2of3_nodeep", "AND_deep", "S1_both"}
    for name, values in stats.items():
        if name == "2of3_both":
            continue
        color = FUSION if name == "DFH" else (
            DEEP if name in {"SEED", "STEPD"} else (
                CLASSICAL if name == "2of3_nodeep" else SCALAR
            )
        )
        ax.scatter(values["far"], values["f1"], s=46, color=color, zorder=3)
        if name in key:
            ax.annotate(
                name,
                (values["far"], values["f1"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
    adwin_far = stats.get("ADWIN", {}).get("far")
    if adwin_far is not None:
        ax.axvline(adwin_far, color=ERROR, ls=":", lw=1.0, label="ADWIN FAR")
    ax.set_xlabel("mean FAR per 1k (lower is better)")
    ax.set_ylabel("mean F1 (higher is better)")
    ax.set_title("F1 vs false alarms")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "E19 — DF-H v1 is third overall; it wins on false alarms, not rank",
        fontsize=11,
    )
    save(fig, "e19-overall-rank.png")


def e19_cells() -> None:
    ranking, summary_rows, _ = _e19_tables()
    drift_types = ["feature_permutation", "feature_filtering"]
    transitions = ["abrupt", "gradual"]
    detectors = [
        "DFH",
        "2of3_deep",
        "2of3_classical",
        "2of3_nodeep",
        "AND_deep",
        "S1_both",
        "OR_deep",
        "SEED",
        "STEPD",
    ]
    matrix = np.full((len(detectors), 4), np.nan)
    columns = []
    for transition in transitions:
        for drift in drift_types:
            columns.append((transition, drift))
    for i, detector in enumerate(detectors):
        for j, (transition, drift) in enumerate(columns):
            value = ranking["by_transition"].get(transition, {}).get(drift, {}).get(detector)
            if value is not None:
                matrix[i, j] = float(value)

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.15, 1]}, constrained_layout=True
    )
    ax = axes[0]
    image = ax.imshow(matrix, vmin=1, vmax=12, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(
        [f"{transition}\n{drift.replace('_', ' ')}" for transition, drift in columns],
        fontsize=8,
    )
    ax.set_yticks(range(len(detectors)))
    ax.set_yticklabels(detectors)
    for tick, name in zip(ax.get_yticklabels(), detectors):
        tick.set_color(FUSION if name == "DFH" else (
            CLASSICAL if name == "2of3_nodeep" else (
                DEEP if name in {"SEED", "STEPD"} else "black"
            )
        ))
    for i in range(len(detectors)):
        for j in range(len(columns)):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] < 7 else "black",
                    fontsize=8,
                )
    ax.add_patch(
        Rectangle((-0.5, -0.5), len(columns), 1, fill=False, edgecolor=FUSION, lw=2)
    )
    ax.set_title("rank in the four feature cells (1 = best)")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="average rank")

    ax = axes[1]
    per_dataset = _per_dataset_ranks(summary_rows)
    positions = _dataset_positions(per_dataset)
    selected = ["DFH", "2of3_deep", "2of3_classical", "2of3_nodeep", "SEED", "STEPD", "ADWIN"]
    y = np.arange(len(selected))
    cov = [positions["Covtype"].get(name, np.nan) for name in selected]
    elec = [positions["Electricity"].get(name, np.nan) for name in selected]
    ax.barh(y - 0.2, cov, 0.4, color=DEEP, label="Covtype")
    ax.barh(y + 0.2, elec, 0.4, color=DEEP2, label="Electricity")
    ax.axvline(3.0, color=ERROR, ls="--", lw=1.1, label="gate: position ≤ 3")
    ax.set_yticks(y, selected)
    ax.invert_yaxis()
    ax.set_xlabel("ordinal rank (1 = best; ties averaged)")
    ax.set_title("per-dataset robustness")
    ax.grid(axis="x", alpha=0.25)
    for i, (left, right) in enumerate(zip(cov, elec)):
        if np.isfinite(left):
            ax.text(left + 0.08, i - 0.2, f"{left:.1f}", va="center", fontsize=7)
        if np.isfinite(right):
            ax.text(right + 0.08, i + 0.2, f"{right:.1f}", va="center", fontsize=7)
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    fig.suptitle(
        "E19 — the feature monitor wins cells, but not Covtype-wide robustness",
        fontsize=11,
    )
    save(fig, "e19-feature-cells.png")


def main() -> None:
    e18_signal_gate()
    e18_timing()
    e19_overall()
    e19_cells()


if __name__ == "__main__":
    main()
