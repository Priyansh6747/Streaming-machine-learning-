"""Ground-truth drift streams for the DriftLab harness.

The feature process is drawn once per stream and the concept is mixed with a
weight ``p(t)``.  That keeps the drift schedule exact and avoids the River
generator RNG-replay confound documented in Exp-02.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from river.datasets import synth

from .config import EpisodeSpec, StreamConfig


@dataclass
class StreamData:
    X: list[dict[Any, float]]
    y: list[int]
    episodes: list[EpisodeSpec]
    feature_keys: list[Any]
    meta: dict[str, Any]

    def __len__(self) -> int:
        return len(self.y)


def mixing_weight(t: int, cfg: StreamConfig) -> float:
    """Probability that sample ``t`` comes from concept B."""
    if cfg.shape == "stationary" or t < cfg.position:
        return 0.0
    if cfg.shape == "abrupt":
        return 1.0
    if cfg.shape == "recurring":
        if cfg.period <= 0:
            raise ValueError("recurring drift needs a positive period")
        block = (t - cfg.position) // cfg.period
        return 1.0 if block % 2 == 0 else 0.0
    if cfg.shape in {"gradual", "incremental"}:
        if cfg.width <= 0:
            return 1.0
        p = min(1.0, max(0.0, (t - cfg.position) / cfg.width))
        if cfg.shape == "incremental":
            steps = int(cfg.params.get("n_steps", 10))
            if steps > 1:
                p = math.floor(p * steps) / steps
        return p
    raise ValueError(f"unknown drift shape: {cfg.shape!r}")


def _maybe_flip(label: int, rng: np.random.Generator, noise: float) -> int:
    if noise > 0 and rng.random() < noise:
        return 1 - label
    return label


def _sea(cfg: StreamConfig, rng: np.random.Generator) -> tuple[list[dict], list[int], list]:
    thresholds = {0: 8.0, 1: 9.0, 2: 7.0, 3: 9.5}
    a = int(cfg.concept_a if cfg.concept_a is not None else 0)
    b = int(cfg.concept_b if cfg.concept_b is not None else 1)
    if a not in thresholds or b not in thresholds:
        raise ValueError("SEA concepts must be variants 0..3")
    X: list[dict] = []
    y: list[int] = []
    for t in range(cfg.n_samples):
        x = {i: float(rng.uniform(0.0, 10.0)) for i in range(3)}
        p = mixing_weight(t, cfg)
        threshold = (1.0 - p) * thresholds[a] + p * thresholds[b]
        label = int(x[0] + x[1] > threshold)
        X.append(x)
        y.append(_maybe_flip(label, rng, cfg.noise))
    return X, y, [0, 1, 2]


def _hyperplane(cfg: StreamConfig, rng: np.random.Generator) -> tuple[list[dict], list[int], list]:
    n_features = int(cfg.params.get("n_features", 10))
    w_a = rng.random(n_features)
    w_b = rng.random(n_features)
    X: list[dict] = []
    y: list[int] = []
    for t in range(cfg.n_samples):
        x = {i: float(rng.random()) for i in range(n_features)}
        p = mixing_weight(t, cfg)
        w = (1.0 - p) * w_a + p * w_b
        score = sum(w[i] * x[i] for i in range(n_features))
        label = int(score >= 0.5 * float(w.sum()))
        X.append(x)
        y.append(_maybe_flip(label, rng, cfg.noise))
    return X, y, list(range(n_features))


def _rbf_centroids(rng: np.random.Generator, n_features: int, n_centroids: int, n_classes: int):
    centers = rng.random((n_centroids, n_features))
    labels = rng.integers(0, n_classes, size=n_centroids)
    stds = rng.random(n_centroids)
    weights = rng.random(n_centroids)
    return centers, labels, stds, weights


def _sample_rbf(
    rng: np.random.Generator,
    centers: np.ndarray,
    labels: np.ndarray,
    stds: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict, int]:
    idx = int(rng.choice(len(weights), p=weights / weights.sum()))
    direction = rng.normal(size=centers.shape[1])
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        norm = 1.0
    offset = direction / norm * rng.normal(0.0, stds[idx])
    x = {i: float(centers[idx, i] + offset[i]) for i in range(centers.shape[1])}
    return x, int(labels[idx])


def _random_rbf(cfg: StreamConfig, rng: np.random.Generator) -> tuple[list[dict], list[int], list]:
    n_features = int(cfg.params.get("n_features", 10))
    n_centroids = int(cfg.params.get("n_centroids", 20))
    n_classes = int(cfg.params.get("n_classes", 2))
    a = _rbf_centroids(rng, n_features, n_centroids, n_classes)
    b = _rbf_centroids(rng, n_features, n_centroids, n_classes)
    X: list[dict] = []
    y: list[int] = []
    for t in range(cfg.n_samples):
        p = mixing_weight(t, cfg)
        source = a if rng.random() >= p else b
        x, label = _sample_rbf(rng, *source)
        X.append(x)
        y.append(_maybe_flip(label, rng, cfg.noise))
    return X, y, list(range(n_features))


def _agrawal(cfg: StreamConfig, rng: np.random.Generator) -> tuple[list[dict], list[int], list]:
    a = int(cfg.concept_a if cfg.concept_a is not None else 0)
    b = int(cfg.concept_b if cfg.concept_b is not None else 1)
    gen_a = synth.Agrawal(classification_function=a, seed=cfg.seed, balance_classes=False)
    gen_b = synth.Agrawal(classification_function=b, seed=cfg.seed, balance_classes=False)
    it_a, it_b = iter(gen_a), iter(gen_b)
    X: list[dict] = []
    y: list[int] = []
    for t in range(cfg.n_samples):
        xa, ya = next(it_a)
        xb, yb = next(it_b)
        p = mixing_weight(t, cfg)
        x, label = (xa, ya) if rng.random() >= p else (xb, yb)
        X.append(x)
        y.append(_maybe_flip(int(label), rng, cfg.noise))
    keys = list(X[0].keys()) if X else []
    return X, y, keys


def _river_dataset(cfg: StreamConfig, rng: np.random.Generator) -> tuple[list[dict], list[int], list]:
    from river.datasets import Elec2, Phishing

    factories = {"elec2": Elec2, "phishing": Phishing}
    try:
        dataset = factories[cfg.generator]()
    except KeyError as exc:
        raise ValueError(f"unknown River dataset: {cfg.generator!r}") from exc
    rows = list(dataset)
    rng.shuffle(rows)
    rows = rows[: cfg.n_samples]
    X = [dict(x) for x, _ in rows]
    y = [int(value) for _, value in rows]
    keys = list(X[0].keys()) if X else []
    return X, y, keys


_BUILDERS = {
    "sea": _sea,
    "hyperplane": _hyperplane,
    "randomrbf": _random_rbf,
    "agrawal": _agrawal,
    "elec2": _river_dataset,
    "phishing": _river_dataset,
}


def apply_mechanism(
    X: list[dict], y: list[int], cfg: StreamConfig, rng: np.random.Generator
) -> tuple[list[dict], list[int]]:
    if cfg.mechanism in {"", "rule"}:
        return X, y

    if cfg.mechanism == "label_swap":
        source = int(cfg.params.get("swap_from", 0))
        target = int(cfg.params.get("swap_to", 1))
        for t in range(cfg.position, len(y)):
            if y[t] != source:
                continue
            weight = mixing_weight(t, cfg)
            if weight >= 1.0 or rng.random() < weight:
                y[t] = target
        return X, y

    if cfg.mechanism == "feature_permutation":
        keys = list(X[0].keys())
        order = rng.permutation(len(keys))
        for t in range(cfg.position, len(X)):
            weight = mixing_weight(t, cfg)
            if weight < 1.0 and rng.random() >= weight:
                continue
            original = X[t]
            X[t] = {keys[i]: original[keys[int(order[i])]] for i in range(len(keys))}
        return X, y

    if cfg.mechanism == "class_prior":
        source = int(cfg.params.get("class_label", 0))
        skip = float(cfg.params.get("skip_prob", 0.8))
        keep = [
            i
            for i in range(len(X))
            if not (
                i >= cfg.position
                and y[i] == source
                and rng.random() < skip * mixing_weight(i, cfg)
            )
        ]
        return [X[i] for i in keep], [y[i] for i in keep]

    if cfg.mechanism == "feature_filtering":
        key = cfg.params.get("filter_key", list(X[0].keys())[0])
        if isinstance(key, str) and key.isdigit():
            key = int(key)
        pre = [float(row[key]) for row in X[cfg.warm_up : cfg.position]]
        threshold = float(cfg.params.get("threshold", np.median(pre) if pre else 0.0))
        keep = [
            i
            for i in range(len(X))
            if i < cfg.position or float(X[i][key]) <= threshold
        ]
        return [X[i] for i in keep], [y[i] for i in keep]

    raise ValueError(f"unknown drift mechanism: {cfg.mechanism!r}")


def build_stream(cfg: StreamConfig) -> StreamData:
    if cfg.shape == "stationary":
        if cfg.n_samples <= 0:
            raise ValueError("n_samples must be positive")
    if cfg.n_samples <= cfg.warm_up:
        raise ValueError("n_samples must exceed warm_up")
    if not 0.0 <= cfg.noise <= 1.0:
        raise ValueError("noise must be in [0, 1]")
    try:
        builder = _BUILDERS[cfg.generator]
    except KeyError as exc:
        raise ValueError(f"unknown generator: {cfg.generator!r}") from exc

    rng = np.random.default_rng(cfg.seed)
    X, y, feature_keys = builder(cfg, rng)
    X, y = apply_mechanism(X, y, cfg, rng)
    episodes = cfg.episodes()
    meta = {
        "generator": cfg.generator,
        "shape": cfg.shape,
        "n_samples": len(y),
        "source_n_samples": cfg.n_samples,
        "mechanism": cfg.mechanism,
        "warm_up": cfg.warm_up,
        "noise": cfg.noise,
        "seed": cfg.seed,
        "position": cfg.position,
        "position_jitter": cfg.position_jitter,
        "width": cfg.width,
        "period": cfg.period,
        "n_episodes": cfg.n_episodes,
        "concept_a": cfg.concept_a,
        "concept_b": cfg.concept_b,
    }
    return StreamData(X=X, y=y, episodes=episodes, feature_keys=feature_keys, meta=meta)


def stream_from_dict(payload: dict[str, Any]) -> StreamConfig:
    return StreamConfig(**payload)
