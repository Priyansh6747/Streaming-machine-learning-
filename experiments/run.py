"""CLI for DriftLab experiments.

Examples:
    uv run python -m experiments.run --exp e01 --seeds 42 --limit 5000
    uv run python -m experiments.run --config experiments/registry/e04_detectors.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .driftlab.config import load_config, parse_seeds
from .driftlab.harness import run_experiment


def _resolve_config(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    registry = Path(__file__).resolve().parent / "registry"
    matches = sorted(registry.glob(f"{value}*.yaml"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        options = ", ".join(m.name for m in matches)
        raise SystemExit(f"ambiguous experiment {value!r}: {options}")
    raise SystemExit(f"no config found for {value!r} in {registry}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a DriftLab experiment")
    parser.add_argument("--exp", help="registry id or prefix (e.g. e01)")
    parser.add_argument("--config", help="explicit YAML config path")
    parser.add_argument("--seeds", help="override seeds: '42', '0-10', or '0,1,2'")
    parser.add_argument("--limit", type=int, default=None, help="stream length override (smoke test)")
    parser.add_argument("--out", help="output directory override")
    parser.add_argument("--need-proba", dest="need_proba", action="store_true", default=None)
    parser.add_argument("--no-proba", dest="need_proba", action="store_false")
    parser.add_argument("--summary-only", action="store_true", help="print the summary and exit")
    parser.add_argument("--figure-only", action="store_true", help="re-render figures from a committed artifact")
    parser.add_argument("--report", action="store_true", help="print a compact report from the artifact")
    args = parser.parse_args(argv)

    if not args.exp and not args.config:
        parser.error("one of --exp or --config is required")
    config_path = Path(args.config) if args.config else _resolve_config(args.exp)
    config = load_config(config_path)
    seeds = parse_seeds(args.seeds) if args.seeds else None
    repo = Path(__file__).resolve().parents[1]
    out = Path(args.out) if args.out else repo / "experiments" / "results" / config.exp_id

    if args.figure_only:
        from .driftlab.figures import render_experiment

        written = render_experiment(out)
        for path in written:
            print(path)
        return 0

    if args.report:
        summary = json.loads((out / "summary.json").read_text())
        print(json.dumps(summary.get("acceptance", {}), indent=2, default=str))
        for model, values in sorted((summary.get("models") or {}).items()):
            print(
                f"{model:24s} aue={values['aue']['mean']} "
                f"damage_ref={values['damage_ref']['mean']} "
                f"recovery={values['recovery_rate']['mean']} "
                f"kappa={values['kappa']['mean']} "
                f"samples/s={values['samples_per_s']['mean']}"
            )
        return 0

    print(
        f"{config.exp_id}: {len(config.streams)} streams × {len(seeds or config.seeds)} seeds "
        f"× {len(config.models)} models, detectors={config.detectors}",
        flush=True,
    )
    summary = run_experiment(
        config,
        repo=repo,
        out_dir=out,
        limit=args.limit,
        seeds=seeds,
        need_proba=args.need_proba,
    )
    print(json.dumps(summary.get("acceptance", {}), indent=2, default=str))
    if args.summary_only:
        print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
