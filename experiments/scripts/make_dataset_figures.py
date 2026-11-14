"""Render the dataset reference figures from cached KDD artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KDD = ROOT / "experiments" / "data" / "kdd"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

KDD_NAMES = ["Asfault", "Covtype", "Electricity", "GasSensorArray", "NOAA", "Posture", "Rialto"]
RIVER = [("Elec2 (E14/E17)", 45_312, 8, 2), ("Phishing (E14)", 1_250, 9, 2)]
CAP = 100_000
BLUE = "#2563eb"
TEAL = "#0f766e"
GREY = "#6b7280"


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(path.relative_to(ROOT))


def load(name: str):
    with np.load(KDD / f"{name}.npz") as z:
        X, y = z["X"], z["y"]
        return X.shape[0], X.shape[1], y


def overview() -> None:
    rows = []
    for name in KDD_NAMES:
        n, d, _ = load(name)
        rows.append((name, n, d, "KDD'26 reproduction"))
    for name, n, d, _ in RIVER:
        rows.append((name, n, d, "model / detector experiments"))
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    feats = [r[2] for r in rows]
    colors = [BLUE if r[3].startswith("KDD") else TEAL for r in rows]
    y = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True, sharey=True)
    axes[0].barh(y, values, color=colors, log=True)
    axes[0].axvline(CAP, color=GREY, linestyle="--", linewidth=1)
    axes[0].text(CAP * 1.05, y[-1], "100k trial cap", color=GREY, fontsize=8, va="center")
    axes[0].set_xlabel("rows (log scale)")
    axes[0].set_title("rows available")
    for i, v in enumerate(values):
        axes[0].text(v * 1.08, i, f"{v:,}", va="center", fontsize=8)
    axes[1].barh(y, feats, color=colors)
    axes[1].set_xlabel("features")
    axes[1].set_title("feature count")
    for i, v in enumerate(feats):
        axes[1].text(v + 1, i, str(v), va="center", fontsize=8)
    axes[0].set_yticks(y, labels)
    fig.suptitle("Datasets used — real streams and the KDD'26 reproduction sets")
    save(fig, "dataset-overview.png")


def class_balance() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("tab20")
    y = np.arange(len(KDD_NAMES))
    for i, name in enumerate(KDD_NAMES):
        _, _, yv = load(name)
        labels, counts = np.unique(yv, return_counts=True)
        shares = counts / counts.sum()
        left = 0.0
        for j, (label, share) in enumerate(zip(labels, shares)):
            ax.barh(i, share, left=left, color=cmap(j % 20), edgecolor="white", linewidth=0.4)
            left += share
        ax.text(1.01, i, f"{len(labels)} classes", va="center", fontsize=9)
    ax.set_yticks(y, KDD_NAMES)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("share of rows")
    ax.set_title("KDD'26 class balance (full cached data, before the per-trial 100k cap)")
    ax.grid(axis="x", alpha=0.2)
    save(fig, "dataset-class-balance.png")


def drift_shapes() -> None:
    t = np.arange(60_000)
    pos, width, period = 30_000, 2_000, 8_000
    shapes = {
        "abrupt": (t >= pos).astype(float),
        "gradual": np.clip((t - pos) / width, 0, 1),
        "incremental": np.floor(np.clip((t - pos) / 10_000, 0, 1) * 10) / 10,
        "recurring": np.where((t >= pos) & (((t - pos) // period) % 2 == 0), 1.0, 0.0),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.2), constrained_layout=True, sharex=True)
    for ax, (name, p) in zip(axes.ravel(), shapes.items()):
        ax.axvspan(0, pos, color="#e5e7eb", alpha=0.6)
        ax.plot(t, p, color=BLUE, linewidth=1.4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(name)
        ax.grid(alpha=0.2)
        ax.set_ylabel("weight on new concept")
    for ax in axes[-1]:
        ax.set_xlabel("sample index")
    axes[0, 0].annotate("warm-up", (pos / 2, 0.85), ha="center", color=GREY, fontsize=9)
    fig.suptitle("Drift schedules: τ = 30,000, width = 2,000 (incremental 10,000), period = 8,000")
    save(fig, "dataset-drift-shapes.png")


if __name__ == "__main__":
    overview()
    class_balance()
    drift_shapes()
