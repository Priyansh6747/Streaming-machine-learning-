"""Artifact writers for DriftLab experiments."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def git_sha(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("river", "torch", "numpy", "sklearn", "scipy", "deep_river"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "missing"
    return versions


@dataclass
class ArtifactWriter:
    exp_id: str
    out_dir: Path
    repo: Path
    config: dict[str, Any]
    config_hash: str
    resume: bool = False

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "figures").mkdir(exist_ok=True)
        self._cache_dir = self.out_dir / "cache"
        self._cache_dir.mkdir(exist_ok=True)
        self._episodes_path = self.out_dir / "episodes.jsonl"
        self._detections_path = self.out_dir / "detections.jsonl"
        self._models_path = self.out_dir / "models.jsonl"
        if not self.resume:
            for path in (self._episodes_path, self._detections_path, self._models_path):
                if path.exists():
                    path.unlink()
            for path in self._cache_dir.glob("*"):
                if path.is_file():
                    path.unlink()
        self._series: dict[str, np.ndarray] = {}
        self._start = time.time()

    def write_meta(self) -> None:
        payload = {
            "schema": "driftlab/2",
            "exp_id": self.exp_id,
            "config_hash": self.config_hash,
            "git_sha": git_sha(self.repo),
            "versions": package_versions(),
            "started_unix": self._start,
            "resumed": self.resume,
            "platform": platform.platform(),
            "argv": sys.argv,
            "config": self.config,
        }
        self._write_json("meta.json", payload)

    def add_episode(self, row: dict[str, Any]) -> None:
        with self._episodes_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def add_detection(self, row: dict[str, Any]) -> None:
        with self._detections_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def add_model_run(self, row: dict[str, Any]) -> None:
        with self._models_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def write_partial(self, payload: dict[str, Any]) -> None:
        self._write_json("partial.json", payload)

    def add_series(self, key: str, values: np.ndarray) -> None:
        self._series[key] = np.asarray(values)

    def series_items(self) -> dict[str, np.ndarray]:
        return dict(self._series)

    def completed_units(self) -> set[str]:
        return {path.stem for path in self._cache_dir.glob("*.json")}

    def save_seed_cache(
        self,
        unit: str,
        rows: dict[str, list[dict[str, Any]]],
        series_prefix: str,
    ) -> None:
        payload = {
            "models": rows.get("models") or [],
            "episodes": rows.get("episodes") or [],
            "detections": rows.get("detections") or [],
        }
        (self._cache_dir / f"{unit}.json").write_text(json.dumps(payload, indent=2))
        series = {
            key: value
            for key, value in self._series.items()
            if key.startswith(series_prefix)
        }
        if series:
            np.savez_compressed(self._cache_dir / f"{unit}.npz", **series)

    def load_seed_caches(
        self, allowed: set[str] | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        models: list[dict[str, Any]] = []
        episodes: list[dict[str, Any]] = []
        detections: list[dict[str, Any]] = []
        for path in sorted(self._cache_dir.glob("*.json")):
            if allowed is not None and path.stem not in allowed:
                continue
            payload = json.loads(path.read_text())
            models.extend(payload.get("models") or [])
            episodes.extend(payload.get("episodes") or [])
            detections.extend(payload.get("detections") or [])
        for path in sorted(self._cache_dir.glob("*.npz")):
            if allowed is not None and path.stem not in allowed:
                continue
            with np.load(path) as data:
                for key in data.files:
                    self._series[key] = data[key]
        return models, episodes, detections

    def rewrite_rows(
        self,
        models: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
        detections: list[dict[str, Any]],
    ) -> None:
        for path, rows in (
            (self._models_path, models),
            (self._episodes_path, episodes),
            (self._detections_path, detections),
        ):
            if rows:
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def finalize(self, summary: dict[str, Any]) -> None:
        if self._series:
            np.savez_compressed(self.out_dir / "series.npz", **self._series)
        summary = {
            **summary,
            "exp_id": self.exp_id,
            "config_hash": self.config_hash,
            "runtime_s": time.time() - self._start,
        }
        self._write_json("summary.json", summary)
        self._write_json("results.json", summary)

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        (self.out_dir / name).write_text(json.dumps(payload, indent=2, default=str))

    def write_index_entry(self) -> None:
        index = self.repo / "experiments" / "results" / "runs.index.jsonl"
        index.parent.mkdir(parents=True, exist_ok=True)
        try:
            out_dir = str(self.out_dir.relative_to(self.repo))
        except ValueError:
            out_dir = str(self.out_dir)
        entry = {
            "exp_id": self.exp_id,
            "config_hash": self.config_hash,
            "finished_unix": time.time(),
            "out_dir": out_dir,
        }
        with index.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
