"""Render the post-mid-term report figures from saved artifacts."""

from __future__ import annotations

import collections
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

BATCH = "#6b7280"
STREAM = "#2563eb"
ACCENT = "#dc2626"


def load(exp: str) -> dict:
    return json.loads((RESULTS / exp / "summary.json").read_text())


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(path.relative_to(ROOT))


def e01_correction() -> None:
    summary = load("e01")
    rows = summary["per_seed"]
    seeds = sorted({r["seed"] for r in rows}, key=lambda s: (s == 42, s))
    series = {
        model: {r["seed"]: r["damage_ref"] for r in rows if r["model"] == model}
        for model in ("deepriver_rolling", "frozen_rf")
    }
    x = np.arange(len(seeds))
    fig, ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    ax.bar(x - 0.2, [series["deepriver_rolling"][s] for s in seeds], 0.4,
           label="DeepRiver (streaming)", color=STREAM)
    ax.bar(x + 0.2, [series["frozen_rf"][s] for s in seeds], 0.4,
           label="RandomForest (frozen)", color=BATCH)
    ax.set_xticks(x, [str(s) for s in seeds])
    ax.set_xlabel("seed")
    ax.set_ylabel("damage_ref (samples·κ, 5k window)")
    ax.set_title("E01 — corrected damage per seed: the streaming arm is lower on 11/11")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "post-midterm-e01-correction.png")


def e02_control() -> None:
    counts = collections.defaultdict(lambda: [0, 0])
    for line in (RESULTS / "e02" / "detections.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["host"] != "ht":
            continue
        key = (r["detector"], r["channel"])
        counts[key][0] += r["fp_alarms"]
        counts[key][1] += r["n_samples"]
    detectors = ["adwin", "kswin", "page_hinkley"]
    raw = [1000 * counts[(d, "raw")][0] / counts[(d, "raw")][1] for d in detectors]
    ewma = [1000 * counts[(d, "ewma")][0] / counts[(d, "ewma")][1] for d in detectors]
    x = np.arange(len(detectors))
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(x - 0.2, raw, 0.4, label="raw mistake signal", color=STREAM)
    ax.bar(x + 0.2, ewma, 0.4, label="EWMA-smoothed signal", color=ACCENT)
    for i, (a, b) in enumerate(zip(raw, ewma)):
        ax.text(i - 0.2, a + 0.08, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.08, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["ADWIN", "KSWIN", "PageHinkley"])
    ax.set_ylabel("false alarms per 1,000 samples")
    ax.set_title("E02 — no-drift control: smoothing creates false alarms")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "post-midterm-e02-control.png")


def e03_ladder() -> None:
    summary = load("e03")
    order = [
        "frozen_rf", "scheduled_rf_1k", "scheduled_rf_5k", "scheduled_rf_20k",
        "ht", "hat", "arf", "lb", "adwin_bagging", "oracle_ref",
    ]
    labels = {
        "frozen_rf": "frozen RF",
        "scheduled_rf_1k": "scheduled RF (1k)",
        "scheduled_rf_5k": "scheduled RF (5k)",
        "scheduled_rf_20k": "scheduled RF (20k)",
        "ht": "Hoeffding Tree",
        "hat": "HAT",
        "arf": "ARF",
        "lb": "Leveraging Bagging",
        "adwin_bagging": "ADWIN Bagging",
        "oracle_ref": "oracle refit",
    }
    models = [m for m in order if m in summary["models"]]
    colors = [BATCH if m.startswith(("frozen", "scheduled", "oracle")) else STREAM for m in models]
    y = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, metric, title in (
        (axes[0], "damage_ref", "damage_ref — 5k post-drift window"),
        (axes[1], "damage_full", "damage_full — rest of stream"),
    ):
        values = [summary["models"][m][metric]["mean"] for m in models]
        ax.barh(y, values, color=colors)
        ax.set_yticks(y, [labels[m] for m in models])
        ax.invert_yaxis()
        ax.set_xlabel("samples·κ (lower is better)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        for i, v in enumerate(values):
            ax.text(v * 1.01, i, f"{v:,.0f}", va="center", fontsize=8)
    fig.suptitle("E03 — baseline ladder, 5-seed slice (SEA + Hyperplane, abrupt drift)")
    save(fig, "post-midterm-e03-ladder.png")


def e14_real_streams() -> None:
    summary = load("e14")
    models = ["frozen_rf", "arf", "hat", "driftformer", "oracle_ref"]
    labels = ["frozen RF", "ARF", "HAT", "DriftFormer", "oracle refit"]
    generators = ["elec2", "phishing"]
    values = {g: [] for g in generators}
    kappas = {g: [] for g in generators}
    for model in models:
        for g in generators:
            rows = [r for r in summary["per_seed"] if r["model"] == model and r["generator"] == g]
            values[g].append(float(np.mean([r["damage_ref"] for r in rows])))
            kappas[g].append(float(np.mean([r["kappa"] for r in rows])))
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    ax = axes[0]
    ax.bar(x - 0.2, values["elec2"], 0.4, label="Elec2", color=STREAM)
    ax.bar(x + 0.2, values["phishing"], 0.4, label="Phishing", color="#0f766e")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("damage_ref (samples·κ)")
    ax.set_title("damage after the injected drift")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax = axes[1]
    ax.bar(x - 0.2, kappas["elec2"], 0.4, color=STREAM)
    ax.bar(x + 0.2, kappas["phishing"], 0.4, color="#0f766e")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("full-run Cohen κ")
    ax.set_title("full-run agreement")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("E14 — real streams, 5 seeds: ARF leads on both datasets")
    save(fig, "post-midterm-e14-real-streams.png")


def e17_detector_map() -> None:
    report = json.loads((RESULTS / "e17" / "detector_report.json").read_text())
    shapes = report["shapes"]
    table = report["table"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True, sharey=True)
    for ax, host, title in (
        (axes[0], "ht", "static Hoeffding Tree"),
        (axes[1], "hat", "adaptive HAT"),
    ):
        for detector, color, offset in (("hddm_a", STREAM, -0.2), ("adwin", ACCENT, 0.2)):
            vals = []
            for shape in shapes:
                row = next(
                    r for r in table
                    if r["host"] == host and r["detector"] == detector
                    and r["channel"] == "raw" and r["shape"] == shape
                )
                vals.append(row["f1"])
            ax.bar(np.arange(len(shapes)) + offset, vals, 0.4,
                   label=detector.replace("_", "-").upper(), color=color)
        ax.set_xticks(np.arange(len(shapes)), shapes)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"E17 — {title}")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("balanced F1 (5 seeds)")
    axes[0].legend(frameon=False)
    fig.suptitle("E17 — the synthetic detector default is host-conditional on real Elec2")
    save(fig, "post-midterm-e17-detector-map.png")


def kdd_ranking() -> None:
    ranking = json.loads((RESULTS / "kdd_bench_full" / "ranking.json").read_text())["overall"]
    f1 = collections.defaultdict(list)
    with (RESULTS / "kdd_bench_full" / "summary.csv").open() as fh:
        for row in csv.DictReader(fh):
            if row["f1_mean"]:
                f1[row["detector"]].append(float(row["f1_mean"]))
    detectors = sorted(ranking, key=lambda d: ranking[d])
    ranks = [ranking[d] for d in detectors]
    mean_f1 = [float(np.mean(f1[d])) for d in detectors]
    y = np.arange(len(detectors))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].barh(y, ranks, color=[STREAM if d in ("SEED", "STEPD") else BATCH for d in detectors])
    axes[0].set_yticks(y, detectors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("average rank (lower is better)")
    axes[0].set_title("average rank over 56 dataset × drift cells")
    axes[0].grid(axis="x", alpha=0.25)
    for i, v in enumerate(ranks):
        axes[0].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
    axes[1].barh(y, mean_f1, color=[STREAM if d in ("SEED", "STEPD") else BATCH for d in detectors])
    axes[1].set_yticks(y, detectors)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("mean F1")
    axes[1].set_title("mean F1 over the same cells")
    axes[1].grid(axis="x", alpha=0.25)
    for i, v in enumerate(mean_f1):
        axes[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    fig.suptitle("KDD'26 reproduction — 50 trials, 7 real datasets, 6 detectors")
    save(fig, "post-midterm-kdd-ranking.png")


if __name__ == "__main__":
    e01_correction()
    e02_control()
    e03_ladder()
    e14_real_streams()
    e17_detector_map()
    kdd_ranking()
