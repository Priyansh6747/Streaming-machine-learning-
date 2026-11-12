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
                "n": len(values),
                "mean_recovery": float(np.mean(values)),
                "expected_uniform_mean": every / 2,
                "ks_statistic": float(result.statistic),
                "ks_p_value": float(result.pvalue),
                "uniform_not_rejected": bool(result.pvalue > 0.05),
            }
        )

    payload = {
        "scheduled_phase_noise": rows,
        "model_aue": {
            name: values.get("aue", {}).get("mean")
            for name, values in (summary.get("models") or {}).items()
        },
    }
    (exp_dir / "schedule_test.json").write_text(json.dumps(payload, indent=2))
    for row in rows:
        print(
            f"{row['model']:<18} n={row['n']:<3} mean={row['mean_recovery']:.0f} "
            f"expected={row['expected_uniform_mean']:.0f} "
            f"KS p={row['ks_p_value']:.3f} uniform={row['uniform_not_rejected']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
