"""E21 Lemma-1 synthesis: label-free warning vs the error-only label tax.

Reads committed E19 artifacts only. Stage 1 is label-free, so its detection
delay does not change with the label delay L. An error-only detector cannot
alarm before its label arrives, so its delay is the immediate-label delay plus
L (Lemma 1).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "docs" / "Reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
E21 = RESULTS / "e21"
E21.mkdir(parents=True, exist_ok=True)

DEEP = "#2563eb"
CLASSICAL = "#6b7280"
SEED = "#dc2626"
STEPD = "#f59e0b"
ADWIN = "#7c3aed"

L_VALUES = [0, 100, 500, 2_000, 10_000]
SCALARS = ["SEED", "STEPD", "ADWIN", "HDDMAverage"]
STAGE1 = ["S1_deep", "S1_classical"]


def _float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _stage1_stats(trials, dataset, transition, detector):
    values = []
    total = 0
    for row in trials:
        if (
            row["dataset"] == dataset
            and row["drift_type"] == "feature_permutation"
            and row["transition"] == transition
            and row["detector"] == detector
        ):
            total += 1
            value = _float(row["mdt"])
            if value is not None:
                values.append(value)
    return {
        "base_delay": float(np.mean(values)) if values else None,
        "hit_rate": len(values) / total if total else 0.0,
        "n": total,
    }


def _scalar_stats(summary, dataset, transition):
    result = {}
    for row in summary:
        if (
            row["dataset"] == dataset
            and row["drift_type"] == "feature_permutation"
            and row["transition"] == transition
            and row["detector"] in SCALARS
        ):
            result[row["detector"]] = {"base_delay": _float(row["mdt_mean"])}
    return result


def main() -> None:
    with (RESULTS / "e19_kdd_pilot" / "trials.csv").open() as handle:
        trials = list(csv.DictReader(handle))
    with (RESULTS / "e19_kdd_pilot" / "summary.csv").open() as handle:
        summary = list(csv.DictReader(handle))

    cells = []
    for dataset in ("Covtype", "Electricity"):
        for transition in ("abrupt", "gradual"):
            stage1 = {
                detector: _stage1_stats(trials, dataset, transition, detector)
                for detector in STAGE1
            }
            error_only = _scalar_stats(summary, dataset, transition)
            cells.append(
                {
                    "dataset": dataset,
                    "transition": transition,
                    "stage1": stage1,
                    "error_only": error_only,
                }
            )

    payload = {
        "experiment": "e21",
        "lemma": "Lemma 1 — fixed label delay",
        "label": "Stage 1 is label-free; error-only delay is base + L",
        "source": "experiments/results/e19_kdd_pilot/trials.csv + summary.csv",
        "l_values": L_VALUES,
        "cells": cells,
    }
    (E21 / "delay_floor.json").write_text(json.dumps(payload, indent=2))

    rows = []
    for cell in cells:
        row = {"dataset": cell["dataset"], "transition": cell["transition"]}
        for detector in STAGE1 + SCALARS:
            base = cell["stage1"].get(detector, {}).get("base_delay")
            if base is None:
                base = cell["error_only"].get(detector, {}).get("base_delay")
            row[detector] = base
            for delay in L_VALUES:
                if base is None:
                    row[f"{detector}@L={delay}"] = None
                elif detector in STAGE1:
                    row[f"{detector}@L={delay}"] = base
                else:
                    row[f"{detector}@L={delay}"] = base + delay
        rows.append(row)
    fieldnames = list(rows[0])
    with (E21 / "delay_floor.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False, constrained_layout=True)
    colors = {
        "S1_deep": DEEP,
        "S1_classical": CLASSICAL,
        "SEED": SEED,
        "STEPD": STEPD,
        "ADWIN": ADWIN,
    }
    labels = {
        "S1_deep": "Stage 1 · deep (label-free)",
        "S1_classical": "Stage 1 · classical (label-free)",
        "SEED": "SEED (error-only)",
        "STEPD": "STEPD (error-only)",
        "ADWIN": "ADWIN (error-only)",
    }
    for ax, cell in zip(axes.flat, cells):
        for detector in STAGE1:
            stats = cell["stage1"].get(detector, {})
            base = stats.get("base_delay")
            if base is None:
                continue
            ax.plot(
                L_VALUES,
                [base] * len(L_VALUES),
                color=colors[detector],
                marker="o",
                lw=2.2,
                label=labels[detector],
            )
        for detector in ("SEED", "STEPD", "ADWIN"):
            base = cell["error_only"].get(detector, {}).get("base_delay")
            if base is None:
                continue
            ax.plot(
                L_VALUES,
                [base + delay for delay in L_VALUES],
                color=colors[detector],
                marker="s",
                ls="--",
                lw=1.8,
                label=labels[detector],
            )
        ax.set_title(f"{cell['dataset']} — {cell['transition']} feature permutation")
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlim(-10, 12_000)
        ax.set_ylim(0, 13_000)
        ax.set_xticks(L_VALUES)
        ax.set_xticklabels([str(value) for value in L_VALUES], fontsize=8)
        ax.set_xlabel("label delay L (samples, symlog)")
        ax.set_ylabel("detection delay after onset (samples)")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(loc="upper left", frameon=False, fontsize=7.5)
    fig.suptitle(
        "E21 — the label tax: error-only detection shifts by L; label-free warning does not",
        fontsize=11,
    )
    path = OUT / "e21-delay-floor.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(path.relative_to(ROOT))

    print(f"\nAt L=2000 samples (detection delay after onset):")
    print(f"{'cell':24s}{'S1_deep':>10s}{'S1_class':>10s}{'SEED':>10s}{'STEPD':>10s}{'ADWIN':>10s}")
    for cell in cells:
        values = []
        for detector in ("S1_deep", "S1_classical", "SEED", "STEPD", "ADWIN"):
            base = cell["stage1"].get(detector, {}).get("base_delay")
            if base is None:
                base = cell["error_only"].get(detector, {}).get("base_delay")
            values.append("—" if base is None else f"{base + (0 if detector in STAGE1 else 2000):.0f}")
        print(
            f"{cell['dataset'] + '/' + cell['transition']:24s}"
            + "".join(f"{value:>10s}" for value in values)
        )
    print(f"\nwrote {E21 / 'delay_floor.json'}")


if __name__ == "__main__":
    main()
