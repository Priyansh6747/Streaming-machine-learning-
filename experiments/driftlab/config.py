"""Experiment configuration and canonical hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_METRICS: dict[str, Any] = {
    "rolling_window": 1_000,
    "band": 0.95,
    "stretch": 500,
    "window": 5_000,
    "recovery_horizon": None,
    "delta_max": {
        "abrupt": 1_000,
        "gradual": 2_000,
        "incremental": 4_000,
        "recurring": 2_000,
    },
    "ewma_alpha": 0.01,
}


@dataclass(frozen=True)
class EpisodeSpec:
    start: int
    end: int
    kind: str

    def as_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "kind": self.kind}


@dataclass
class StreamConfig:
    generator: str
    shape: str = "abrupt"
    n_samples: int = 60_000
    warm_up: int = 12_000
    noise: float = 0.1
    seed: int = 42
    position: int = 30_000
    position_jitter: int = 0
    width: int = 0
    period: int = 8_000
    n_episodes: int = 2
    concept_a: int | str | None = None
    concept_b: int | str | None = None
    mechanism: str = "rule"
    params: dict[str, Any] = field(default_factory=dict)

    def episodes(self) -> list[EpisodeSpec]:
        if self.shape == "stationary":
            return []
        if self.shape == "abrupt":
            return [EpisodeSpec(self.position, self.position, "abrupt")]
        if self.shape in {"gradual", "incremental"}:
            return [EpisodeSpec(self.position, self.position + self.width, self.shape)]
        if self.shape == "recurring":
            return [
                EpisodeSpec(
                    self.position + i * self.period,
                    self.position + i * self.period,
                    "recurring",
                )
                for i in range(self.n_episodes)
            ]
        raise ValueError(f"unknown drift shape: {self.shape!r}")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["episodes"] = [ep.as_dict() for ep in self.episodes()]
        return payload


@dataclass
class ExperimentConfig:
    exp_id: str
    title: str
    description: str
    streams: list[StreamConfig]
    models: list[dict[str, Any]]
    detectors: list[str]
    detector_hosts: list[str]
    detector_channels: list[str]
    metrics: dict[str, Any]
    seeds: list[int]
    acceptance: dict[str, Any]
    detector_delays: list[int] = field(default_factory=lambda: [0])
    model_delays: list[int] = field(default_factory=lambda: [0])
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "exp_id": self.exp_id,
            "title": self.title,
            "description": self.description,
            "streams": [s.as_dict() for s in self.streams],
            "models": self.models,
            "detectors": self.detectors,
            "detector_hosts": self.detector_hosts,
            "detector_channels": self.detector_channels,
            "detector_delays": self.detector_delays,
            "model_delays": self.model_delays,
            "metrics": self.metrics,
            "seeds": self.seeds,
            "acceptance": self.acceptance,
        }

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.as_dict())


def canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_seeds(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "-" in text and "," not in text:
            lo, hi = text.split("-", 1)
            return list(range(int(lo), int(hi) + 1))
        return [int(part) for part in text.split(",") if part.strip()]
    raise TypeError(f"cannot parse seeds from {value!r}")


def _stream_from_dict(payload: dict[str, Any], seed: int) -> StreamConfig:
    body = {k: v for k, v in payload.items() if k not in {"seed", "params"}}
    params = dict(payload.get("params") or {})
    return StreamConfig(seed=int(payload.get("seed", seed)), params=params, **body)


def expand_stream_grid(grid: dict[str, Any]) -> list[StreamConfig]:
    generators = grid["generators"]
    shapes = grid["shapes"]
    defaults = dict(grid.get("defaults") or {})
    per_shape = dict(grid.get("per_shape") or {})
    streams: list[StreamConfig] = []
    for generator in generators:
        for shape in shapes:
            merged = {**defaults, **dict(per_shape.get(shape) or {})}
            merged.pop("generator", None)
            merged.pop("shape", None)
            if shape == "abrupt":
                merged.setdefault("width", 0)
            elif shape in {"gradual", "incremental"}:
                merged.setdefault("width", 2_000)
            elif shape == "recurring":
                merged.setdefault("period", 8_000)
                merged.setdefault("n_episodes", 2)
            streams.append(StreamConfig(generator=generator, shape=shape, **merged))
    return streams


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: expected a YAML mapping")

    seeds = parse_seeds(raw.get("seeds", "0-10"))
    if "streams" in raw:
        streams = [
            _stream_from_dict(item, seeds[0] if seeds else 42)
            for item in raw["streams"]
        ]
    elif "stream_grid" in raw:
        streams = expand_stream_grid(raw["stream_grid"])
    else:
        raise ValueError(f"{path}: expected 'streams' or 'stream_grid'")

    metrics = {**DEFAULT_METRICS, **dict(raw.get("metrics") or {})}
    metrics["delta_max"] = {
        **DEFAULT_METRICS["delta_max"],
        **dict((raw.get("metrics") or {}).get("delta_max") or {}),
    }

    detector_hosts = (
        list(raw["detector_hosts"]) if "detector_hosts" in raw else ["ht", "hat"]
    )
    detector_channels = (
        list(raw["detector_channels"]) if "detector_channels" in raw else ["raw", "ewma"]
    )
    return ExperimentConfig(
        exp_id=str(raw["exp_id"]),
        title=str(raw.get("title", "")),
        description=str(raw.get("description", "")),
        streams=streams,
        models=list(raw.get("models") or []),
        detectors=list(raw.get("detectors") or []),
        detector_hosts=detector_hosts,
        detector_channels=detector_channels,
        metrics=metrics,
        seeds=seeds,
        acceptance=dict(raw.get("acceptance") or {}),
        detector_delays=[int(v) for v in (raw.get("detector_delays") or [0])],
        model_delays=[int(v) for v in (raw.get("model_delays") or [0])],
        raw=raw,
    )


def window_for(shape: str, width: int, metrics: dict[str, Any]) -> int:
    base = int(metrics.get("window", 5_000))
    if shape in {"gradual", "incremental"}:
        return 2 * int(width) + base
    return base


def delta_max_for(shape: str, metrics: dict[str, Any]) -> int:
    return int(metrics["delta_max"].get(shape, 2_000))
