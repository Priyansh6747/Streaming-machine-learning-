"""CLI for DriftFormer pretraining.

Example:
    uv run python -m experiments.pretrain --generators sea,hyperplane,agrawal \
        --out experiments/checkpoints/df_logo_rbf.pt --tasks 8 --samples 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .driftlab.deep.pretrain import pretrain
from .driftlab.deep.train import DriftFormerConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pretrain DriftFormer on synthetic drift tasks")
    parser.add_argument("--generators", default="sea,hyperplane,randomrbf,agrawal")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args(argv)

    config = DriftFormerConfig(block_size=args.block_size, seed=args.seed)
    metadata = pretrain(
        generators=[g.strip() for g in args.generators.split(",") if g.strip()],
        out_path=Path(args.out),
        n_tasks=args.tasks,
        n_samples=args.samples,
        seed=args.seed,
        max_features=args.max_features,
        config=config,
    )
    print(json.dumps(metadata, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
