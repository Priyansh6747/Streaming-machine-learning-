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
