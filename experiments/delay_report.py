"""E06 analysis: detection delay shifts by exactly L (Lemma 1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    summary_path = REPO / "experiments" / "results" / "e06" / "summary.json"
    summary = json.loads(summary_path.read_text())
    rows = summary.get("detectors_by_shape") or []
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["host"], row["detector"], row["channel"])
        groups.setdefault(key, []).append(row)

    results = []
    for (host, detector, channel), items in sorted(groups.items()):
        by_delay: dict[int, list[float]] = {}
        by_ndt: dict[int, list[float]] = {}
        for item in items:
            delay = int(item.get("delay", 0))
            if item.get("mean_delay_from_ds") is not None:
                by_delay.setdefault(delay, []).append(float(item["mean_delay_from_ds"]))
            if item.get("ndt_ds") is not None:
                by_ndt.setdefault(delay, []).append(float(item["ndt_ds"]))
        delays = sorted(by_delay)
        if len(delays) < 2:
            continue
        y = np.asarray([np.mean(by_delay[d]) for d in delays])
        slope, intercept = np.polyfit(delays, y, 1)
        fitted = slope * np.asarray(delays) + intercept
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
        ndt = [np.mean(by_ndt[d]) for d in delays if d in by_ndt]
        results.append(
            {
                "host": host,
                "detector": detector,
                "channel": channel,
                "delays": delays,
                "mean_delay_from_ds": [float(v) for v in y],
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": 1.0 - ss_res / ss_tot,
                "ndt_ds_by_delay": [float(v) for v in ndt],
            }
        )

    out = REPO / "experiments" / "results" / "e06" / "delay_slope.json"
    out.write_text(json.dumps({"rows": results}, indent=2))
    for row in results:
        print(
            f"{row['host']:>3} {row['detector']:<14} {row['channel']:<5} "
            f"slope={row['slope']:.3f} r2={row['r2']:.3f} "
            f"delays={row['mean_delay_from_ds']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
