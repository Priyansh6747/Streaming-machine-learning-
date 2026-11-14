"""E03 analysis: scheduled-retrain phase noise."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    exp_dir = REPO / "experiments/results/e03"
    summary = json.loads((exp_dir / "summary.json").read_text())
    meta = json.loads((exp_dir / "meta.json").read_text())
    config = meta.get("config") or {}
    buffer_by_model = {
        str(spec.get("id")): int((spec.get("params") or {}).get("window") or 0)
        for spec in (config.get("models") or [])
        if str(spec.get("id", "")).startswith("scheduled_rf_")
    }
    recovery_horizon = (config.get("metrics") or {}).get("recovery_horizon")
    episodes = [
        json.loads(line)
        for line in (exp_dir / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]

    rows = []
    for model in ("scheduled_rf_1k", "scheduled_rf_5k", "scheduled_rf_20k"):
        every = int(model.rsplit("_", 1)[-1].replace("k", "000"))
        values = [
            int(row["recovery_samples"])
            for row in episodes
            if row["model"] == model and row.get("recovery_samples") is not None
        ]
        if not values:
            continue
        result = stats.kstest(values, "uniform", args=(0, every))
        rows.append(
            {
                "model": model,
                "every": every,
                "buffer_window": buffer_by_model.get(model),
                "n": len(values),
                "mean_recovery": float(np.mean(values)),
                "min_recovery": float(np.min(values)),
                "max_recovery": float(np.max(values)),
                "expected_uniform_mean": every / 2,
                "ks_statistic": float(result.statistic),
                "ks_p_value": float(result.pvalue),
                "uniform_not_rejected": bool(result.pvalue > 0.05),
                "note": (
                    "uniform phase not rejected"
                    if result.pvalue > 0.05
                    else "not uniform over the retrain interval; the refit buffer dominates the lag"
                ),
            }
        )

    payload = {
        "recovery_horizon": recovery_horizon,
        "buffer_note": (
            "Recovery lag is measured to the configured recovery horizon while "
            "damage_ref stays on the damage window. With a refit buffer longer "
            "than the retrain interval, lag is buffer-dominated and is not "
            "expected to be uniform over [0, every]."
        ),
        "scheduled_phase_noise": rows,
        "model_aue": {
            name: values.get("aue", {}).get("mean")
            for name, values in (summary.get("models") or {}).items()
        },
    }
    (exp_dir / "schedule_test.json").write_text(json.dumps(payload, indent=2))
    figure = _phase_figure(episodes, rows, exp_dir)
    for row in rows:
        print(
            f"{row['model']:<18} n={row['n']:<3} "
            f"buffer={row.get('buffer_window')} "
            f"rec={row['min_recovery']:.0f}-{row['max_recovery']:.0f} "
            f"mean={row['mean_recovery']:.0f} "
            f"KS p={row['ks_p_value']:.3f} uniform={row['uniform_not_rejected']}"
        )
    if figure:
        print(f"figure {figure}")
    return 0


def _phase_figure(episodes: list[dict], rows: list[dict], exp_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: dict[str, list[float]] = {}
    for row in rows:
        model = row["model"]
        values = [
            float(item["recovery_samples"])
            for item in episodes
            if item["model"] == model and item.get("recovery_samples") is not None
        ]
        if values:
            series[model] = values
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 0.9 * len(series) + 2.4))
    rng = np.random.default_rng(0)
    labels = []
    for index, (model, values) in enumerate(sorted(series.items())):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(
            values,
            np.full(len(values), index) + jitter,
            s=28,
            alpha=0.8,
            color="tab:blue",
        )
        ax.scatter(
            [float(np.mean(values))],
            [index],
            marker="D",
            s=45,
            color="tab:red",
            zorder=3,
        )
        row = next((item for item in rows if item["model"] == model), None)
        label = f"{model}  (n={len(values)})"
        if row and row.get("buffer_window"):
            label += f", buffer={row['buffer_window']}"
        labels.append(label)

    ax.set_yticks(range(len(series)))
    ax.set_yticklabels(labels, fontsize=8)
    buffers = {
        row.get("buffer_window") for row in rows if row.get("buffer_window")
    }
    if len(buffers) == 1:
        buffer = buffers.pop()
        ax.axvline(
            buffer,
            color="tab:orange",
            linestyle=":",
            linewidth=1.2,
            label="refit buffer",
        )
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("recovery time after drift (samples)")
    ax.set_title("Scheduled retraining: measured recovery lag")
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    path = exp_dir / "figures" / "schedule_phase.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
