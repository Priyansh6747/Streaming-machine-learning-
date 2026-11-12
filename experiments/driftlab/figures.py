"""Figure rendering from committed artifacts only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def render_experiment(exp_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    exp_dir = Path(exp_dir)
    summary = json.loads((exp_dir / "summary.json").read_text())
    out = exp_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    rows = summary.get("detectors_by_shape") or []
    report_path = exp_dir / "detector_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        rows = report.get("by_generator") or rows
    if rows:
        written.extend(_detector_heatmaps(rows, out))
    models = summary.get("models") or {}
    if models:
        written.append(_model_bars(models, out))
    return written


def _detector_heatmaps(rows: list[dict], out: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    written: list[Path] = []
    hosts = sorted({row["host"] for row in rows})
    channels = sorted({row["channel"] for row in rows})
    for host in hosts:
        channel = "raw" if "raw" in channels else channels[0]
        subset = [r for r in rows if r["host"] == host and r["channel"] == channel]
        if not subset:
            continue
        detectors = sorted({r["detector"] for r in subset})
        columns = sorted(
            {(r["generator"], r["shape"], int(r.get("delay", 0))) for r in subset}
        )
        index = {
            (r["detector"], r["generator"], r["shape"], int(r.get("delay", 0))): r
            for r in subset
        }
        matrix = np.full((len(detectors), len(columns)), np.nan)
        for i, detector in enumerate(detectors):
            for j, (generator, shape, delay) in enumerate(columns):
                row = index.get((detector, generator, shape, delay))
                if row and row.get("f1") is not None and np.isfinite(row["f1"]):
                    matrix[i, j] = float(row["f1"])

        fig, ax = plt.subplots(figsize=(max(8.5, 0.62 * len(columns) + 3), 0.52 * len(detectors) + 2.2))
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(
            [
                f"{g}\n{s}" if delay == 0 else f"{g}\n{s}\nL={delay}"
                for g, s, delay in columns
            ],
            fontsize=7,
            rotation=45,
            ha="right",
        )
        ax.set_yticks(range(len(detectors)))
        ax.set_yticklabels(detectors, fontsize=8)
        for i in range(len(detectors)):
            for j in range(len(columns)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                            color="white" if matrix[i, j] < 0.55 else "black", fontsize=7)
        ax.set_title(f"Detector F1 by drift shape — host={host}, channel={channel}")
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="F1")
        fig.tight_layout()
        path = out / f"detector_f1_{host}_{channel}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def _model_bars(models: dict, out: Path) -> Path:
    import matplotlib.pyplot as plt

    names = sorted(models)
    panels = []
    for metric, label in (
        ("damage_ref", "damage_ref (samples·κ)"),
        ("aue", "AUE (excess error vs timing oracle)"),
    ):
        values = [models[n][metric]["mean"] for n in names]
        if any(v is not None and np.isfinite(v) for v in values):
            panels.append((metric, label, values, [models[n][metric]["sd"] for n in names]))
    if not panels:
        panels = [("damage_ref", "damage_ref", [0.0] * len(names), [0.0] * len(names))]

    fig, axes = plt.subplots(
        1, len(panels), figsize=(max(7, 0.8 * len(names) + 3) * len(panels), 4.8), squeeze=False
    )
    positions = np.arange(len(names))
    for ax, (_, label, values, errors) in zip(axes[0], panels):
        clean_values = [0.0 if v is None else float(v) for v in values]
        clean_errors = [0.0 if v is None else float(v) for v in errors]
        ax.bar(positions, clean_values, yerr=clean_errors, capsize=3, color="tab:blue", alpha=0.85)
        ax.set_xticks(positions)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0][0].set_title("Model comparison")
    fig.tight_layout()
    path = out / "model_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
