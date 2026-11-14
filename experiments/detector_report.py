"""E04 analysis: detector effectiveness by drift shape and host.

The stored `f1` in an old artifact can be None when a detector made zero alarms,
which makes the F1 mean exclude exactly the runs that should score zero. This
report recomputes precision, recall and F1 from the raw TP/FP/episode counts in
`detections.jsonl`, so no-alarm runs count as F1 = 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]


def _recompute(row: dict) -> tuple[float, float | None, float | None]:
    tp = int(row.get("tp_alarms", 0))
    fp = int(row.get("fp_alarms", 0))
    n_episodes = int(row.get("n_episodes", 0))
    detected = int(row.get("detected_episodes", 0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = detected / n_episodes if n_episodes else None
    if n_episodes == 0:
        f1 = None
    elif (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


def _mean(items: list[dict], key: str) -> float | None:
    values = [
        float(row[key])
        for row in items
        if row.get(key) is not None and np.isfinite(row[key])
    ]
    return float(np.mean(values)) if values else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute detector metrics from detections.jsonl")
    parser.add_argument("--exp", default="e04", help="experiment id (default: e04)")
    args = parser.parse_args(argv)
    exp_dir = REPO / "experiments" / "results" / args.exp
    rows = [
        json.loads(line)
        for line in (exp_dir / "detections.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if int(row.get("delay", 0)) == 0]
    for row in rows:
        precision, recall, f1 = _recompute(row)
        row["alarm_precision"] = precision
        row["episode_recall"] = recall
        row["f1"] = f1

    shapes = sorted({row["shape"] for row in rows})
    hosts = sorted({row["host"] for row in rows})
    channels = sorted({row["channel"] for row in rows})

    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    by_generator: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["host"], row["detector"], row["channel"], row["shape"])
        grouped.setdefault(key, []).append(row)
        generator_key = (
            row["host"],
            row["detector"],
            row["channel"],
            row["generator"],
            row["shape"],
        )
        by_generator.setdefault(generator_key, []).append(row)

    table = []
    for (host, detector, channel, shape), items in sorted(grouped.items()):
        table.append(
            {
                "host": host,
                "detector": detector,
                "channel": channel,
                "shape": shape,
                "f1": _mean(items, "f1"),
                "episode_recall": _mean(items, "episode_recall"),
                "alarm_precision": _mean(items, "alarm_precision"),
                "ndt_de": _mean(items, "ndt_de"),
                "far_per_1k": _mean(items, "far_per_1k"),
                "n_generators": len({row["generator"] for row in items}),
            }
        )

    generator_table = []
    for (host, detector, channel, generator, shape), items in sorted(by_generator.items()):
        generator_table.append(
            {
                "host": host,
                "detector": detector,
                "channel": channel,
                "generator": generator,
                "shape": shape,
                "f1": _mean(items, "f1"),
                "episode_recall": _mean(items, "episode_recall"),
                "ndt_de": _mean(items, "ndt_de"),
                "far_per_1k": _mean(items, "far_per_1k"),
            }
        )

    best = {}
    for shape in shapes:
        candidates = [
            row for row in table if row["shape"] == shape and row["f1"] is not None
        ]
        best[shape] = max(candidates, key=lambda row: row["f1"]) if candidates else None

    abrupt_vs_gradual = []
    for host in hosts:
        for detector in sorted({row["detector"] for row in rows}):
            for channel in channels:
                abrupt = _mean(
                    [
                        row
                        for row in rows
                        if row["host"] == host
                        and row["detector"] == detector
                        and row["channel"] == channel
                        and row["shape"] == "abrupt"
                    ],
                    "ndt_de",
                )
                gradual = _mean(
                    [
                        row
                        for row in rows
                        if row["host"] == host
                        and row["detector"] == detector
                        and row["channel"] == channel
                        and row["shape"] == "gradual"
                    ],
                    "ndt_de",
                )
                if abrupt is not None and gradual is not None:
                    abrupt_vs_gradual.append(
                        {
                            "host": host,
                            "detector": detector,
                            "channel": channel,
                            "ndt_abrupt": abrupt,
                            "ndt_gradual": gradual,
                            "gradual_minus_abrupt": gradual - abrupt,
                        }
                    )

    host_correlation = []
    if len(hosts) >= 2:
        for detector in sorted({row["detector"] for row in rows}):
            for channel in channels:
                left, right = [], []
                for shape in shapes:
                    left.append(
                        next(
                            (
                                row["f1"]
                                for row in table
                                if row["host"] == hosts[0]
                                and row["detector"] == detector
                                and row["channel"] == channel
                                and row["shape"] == shape
                            ),
                            np.nan,
                        )
                    )
                    right.append(
                        next(
                            (
                                row["f1"]
                                for row in table
                                if row["host"] == hosts[1]
                                and row["detector"] == detector
                                and row["channel"] == channel
                                and row["shape"] == shape
                            ),
                            np.nan,
                        )
                    )
                if (
                    len(left) >= 2
                    and np.isfinite(left).all()
                    and np.isfinite(right).all()
                ):
                    rho, p_value = stats.spearmanr(left, right)
                    host_correlation.append(
                        {
                            "detector": detector,
                            "channel": channel,
                            "spearman_rho": float(rho),
                            "p": float(p_value),
                        }
                    )

    payload = {
        "metric_note": "F1 recomputed from TP/FP/detected/n_episodes; zero alarms = F1 0",
        "shapes": shapes,
        "hosts": hosts,
        "channels": channels,
        "table": table,
        "by_generator": generator_table,
        "best_by_shape": best,
        "abrupt_vs_gradual": abrupt_vs_gradual,
        "host_rank_correlation": host_correlation,
    }
    out = exp_dir / "detector_report.json"
    out.write_text(json.dumps(payload, indent=2))
    for shape, row in best.items():
        if row:
            print(
                f"{shape:<12} best={row['detector']:<14} host={row['host']:<3} "
                f"channel={row['channel']:<5} f1={row['f1']:.3f} "
                f"recall={row['episode_recall']:.3f} "
                f"far/1k={row['far_per_1k']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
