"""E16: consolidate measured cost from existing artifacts (no new runs)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    results_dir = REPO / "experiments" / "results"
    rows: list[dict] = []
    for summary_path in sorted(results_dir.glob("*/summary.json")):
        name = summary_path.parent.name
        relative = summary_path.relative_to(results_dir)
        if (
            name == "e16_cost"
            or name.endswith("_smoke")
            or any(part.startswith("_") for part in relative.parts)
        ):
            continue
        summary = json.loads(summary_path.read_text())
        exp_id = summary.get("exp_id") or name
        for model, values in (summary.get("models") or {}).items():
            rows.append(
                {
                    "exp_id": exp_id,
                    "model": model,
                    "n": values.get("n"),
                    "samples_per_s": values.get("samples_per_s", {}).get("mean"),
                    "model_bytes": values.get("model_bytes"),
                    "accuracy": values.get("accuracy", {}).get("mean"),
                    "kappa": values.get("kappa", {}).get("mean"),
                    "aue": values.get("aue", {}).get("mean"),
                    "damage_ref": values.get("damage_ref", {}).get("mean"),
                    "recovery_rate": values.get("recovery_rate", {}).get("mean"),
                }
            )

    out_dir = results_dir / "e16_cost"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"exp_id": "e16", "rows": rows}, indent=2)
    )
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    fieldnames = fieldnames or ["exp_id", "model"]
    with (out_dir / "cost.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['exp_id']:>5} {row['model']:<28} "
            f"n={row['n']} samples/s={row['samples_per_s']} "
            f"bytes={row['model_bytes']} aue={row['aue']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
