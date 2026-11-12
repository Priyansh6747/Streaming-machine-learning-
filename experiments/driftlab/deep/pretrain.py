"""Leave-one-generator-out pretraining corpus for DriftFormer."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import StreamConfig
from ..streams import build_stream
from .train import DriftFormerConfig, DriftFormerLearner

SHAPES = ["abrupt", "gradual", "incremental", "recurring"]
CONCEPT_PAIRS = [(0, 1), (1, 2), (2, 3), (0, 3)]


def make_tasks(
    generators: Sequence[str],
    n_tasks: int = 8,
    n_samples: int = 20_000,
    seed: int = 0,
) -> list:
    tasks = []
    for index in range(n_tasks):
        generator = generators[index % len(generators)]
        shape = SHAPES[index % len(SHAPES)]
        concept_a, concept_b = CONCEPT_PAIRS[index % len(CONCEPT_PAIRS)]
        cfg = StreamConfig(
            generator=generator,
            shape=shape,
            n_samples=n_samples,
            warm_up=max(1, int(0.2 * n_samples)),
            noise=0.1,
            position=int(0.5 * n_samples),
            width=max(1, int(0.1 * n_samples)),
            period=max(1, int(0.2 * n_samples)),
            n_episodes=2,
            concept_a=concept_a,
            concept_b=concept_b,
            seed=seed + index,
        )
        tasks.append(build_stream(cfg))
    return tasks


def pretrain(
    generators: Sequence[str],
    out_path: str | Path,
    n_tasks: int = 8,
    n_samples: int = 20_000,
    seed: int = 0,
    max_features: int = 10,
    config: DriftFormerConfig | None = None,
) -> dict:
    tasks = make_tasks(generators, n_tasks=n_tasks, n_samples=n_samples, seed=seed)
    learner = DriftFormerLearner(
        model_id="pretrain",
        feature_keys=list(range(max_features)),
        n_classes=2,
        config=config or DriftFormerConfig(),
        n_features=max_features,
        seed=seed,
    )
    for task in tasks:
        learner.set_feature_keys(task.feature_keys)
        learner.warm_up(task.X, task.y)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    learner.save(path)
    metadata = {
        "generators": list(generators),
        "n_tasks": n_tasks,
        "n_samples": n_samples,
        "max_features": max_features,
        "config": asdict(learner.config),
        "checkpoint": str(path),
        "task_meta": [task.meta for task in tasks],
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str))
    return metadata
