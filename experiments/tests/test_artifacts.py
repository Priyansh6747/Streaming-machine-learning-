from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.driftlab.artifacts import ArtifactWriter
from experiments.driftlab.config import canonical_hash


def test_canonical_hash_is_stable():
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


def test_artifact_writer_roundtrip(tmp_path: Path):
    repo = tmp_path
    out = repo / "experiments" / "results" / "x01"
    writer = ArtifactWriter(
        exp_id="x01",
        out_dir=out,
        repo=repo,
        config={"exp_id": "x01"},
        config_hash="abc",
    )
    writer.write_meta()
    writer.add_episode({"model": "m", "aue": 0.1})
    writer.add_detection({"detector": "adwin", "f1": 0.5})
    writer.add_series("y", np.asarray([0, 1, 1], dtype=np.int8))
    writer.finalize({"ok": True})
    assert (out / "meta.json").exists()
    assert (out / "episodes.jsonl").exists()
    assert (out / "detections.jsonl").exists()
    assert (out / "series.npz").exists()
    assert (out / "summary.json").exists()
    assert (out / "results.json").exists()


def test_resume_preserves_existing_rows(tmp_path: Path):
    repo = tmp_path
    out = repo / "experiments" / "results" / "x01"
    first = ArtifactWriter(
        exp_id="x01",
        out_dir=out,
        repo=repo,
        config={"exp_id": "x01"},
        config_hash="abc",
    )
    first.add_episode({"model": "m"})
    first.add_model_run({"model": "m", "seed": 0})
    assert (out / "episodes.jsonl").exists()

    fresh = ArtifactWriter(
        exp_id="x01",
        out_dir=out,
        repo=repo,
        config={"exp_id": "x01"},
        config_hash="abc",
        resume=False,
    )
    assert not (out / "episodes.jsonl").exists()
    assert not (out / "models.jsonl").exists()
    assert fresh.completed_units() == set()

    first.add_episode({"model": "m2"})
    resumed = ArtifactWriter(
        exp_id="x01",
        out_dir=out,
        repo=repo,
        config={"exp_id": "x01"},
        config_hash="abc",
        resume=True,
    )
    assert (out / "episodes.jsonl").exists()
    assert resumed.completed_units() == set()


def test_seed_cache_roundtrip(tmp_path: Path):
    repo = tmp_path
    out = repo / "experiments" / "results" / "x01"
    writer = ArtifactWriter(
        exp_id="x01",
        out_dir=out,
        repo=repo,
        config={"exp_id": "x01"},
        config_hash="abc",
    )
    writer.add_series("s0_seed0_err", np.asarray([1, 0, 1], dtype=np.int8))
    writer.add_series("s0_seed1_err", np.asarray([0, 0], dtype=np.int8))
    writer.save_seed_cache(
        "s0_seed0",
        {"models": [{"seed": 0}], "episodes": [{"seed": 0}], "detections": []},
        "s0_seed0_",
    )
    assert writer.completed_units() == {"s0_seed0"}

    resumed = ArtifactWriter(
        exp_id="x01",
        out_dir=out,
        repo=repo,
        config={"exp_id": "x01"},
        config_hash="abc",
        resume=True,
    )
    models, episodes, detections = resumed.load_seed_caches({"s0_seed0"})
    assert models == [{"seed": 0}]
    assert episodes == [{"seed": 0}]
    assert detections == []
    assert "s0_seed0_err" in resumed.series_items()
    assert "s0_seed1_err" not in resumed.series_items()
