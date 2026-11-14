from __future__ import annotations

import numpy as np
import pytest

from experiments.driftlab.config import StreamConfig
from experiments.driftlab.streams import apply_mechanism, build_stream, mixing_weight


def test_sea_abrupt_schedule():
    cfg = StreamConfig(
        generator="sea",
        shape="abrupt",
        n_samples=1_000,
        warm_up=200,
        position=500,
        concept_a=0,
        concept_b=1,
        noise=0.0,
    )
    stream = build_stream(cfg)
    assert len(stream) == 1_000
    assert stream.episodes[0].start == 500
    assert mixing_weight(499, cfg) == 0.0
    assert mixing_weight(500, cfg) == 1.0
    assert stream.feature_keys == [0, 1, 2]


def test_sea_features_are_not_replayed():
    cfg = StreamConfig(
        generator="sea",
        shape="abrupt",
        n_samples=200,
        warm_up=20,
        position=100,
        noise=0.0,
    )
    stream = build_stream(cfg)
    assert stream.X[0] != stream.X[100]


def test_gradual_mixing_weight():
    cfg = StreamConfig(
        generator="sea",
        shape="gradual",
        n_samples=1_000,
        warm_up=200,
        position=400,
        width=200,
        noise=0.0,
    )
    assert mixing_weight(399, cfg) == 0.0
    assert mixing_weight(500, cfg) == pytest.approx(0.5)
    assert mixing_weight(600, cfg) == 1.0


def test_incremental_is_stepped():
    cfg = StreamConfig(
        generator="sea",
        shape="incremental",
        n_samples=1_000,
        warm_up=200,
        position=400,
        width=200,
        noise=0.0,
        params={"n_steps": 4},
    )
    weights = {mixing_weight(t, cfg) for t in range(400, 600, 10)}
    assert len(weights) <= 5
    assert mixing_weight(600, cfg) == 1.0


def test_recurring_episodes_alternate():
    cfg = StreamConfig(
        generator="sea",
        shape="recurring",
        n_samples=2_000,
        warm_up=200,
        position=500,
        period=500,
        n_episodes=2,
        noise=0.0,
    )
    stream = build_stream(cfg)
    assert [ep.start for ep in stream.episodes] == [500, 1_000]
    assert mixing_weight(500, cfg) == 1.0
    assert mixing_weight(999, cfg) == 1.0
    assert mixing_weight(1_000, cfg) == 0.0


def test_hyperplane_has_expected_width():
    cfg = StreamConfig(
        generator="hyperplane",
        shape="abrupt",
        n_samples=500,
        warm_up=100,
        position=250,
        noise=0.0,
        params={"n_features": 5},
    )
    stream = build_stream(cfg)
    assert len(stream.X[0]) == 5


def test_agrawal_feature_keys_are_stable():
    cfg = StreamConfig(
        generator="agrawal",
        shape="abrupt",
        n_samples=300,
        warm_up=50,
        position=150,
        concept_a=0,
        concept_b=1,
        noise=0.0,
    )
    stream = build_stream(cfg)
    assert len(stream.feature_keys) == 9
    assert set(stream.X[0]) == set(stream.feature_keys)


def _label_swap_cfg(**overrides) -> StreamConfig:
    body = {
        "generator": "elec2",
        "n_samples": 100,
        "warm_up": 10,
        "position": 20,
        "noise": 0.0,
        "mechanism": "label_swap",
        "params": {"swap_from": 0, "swap_to": 1},
    }
    body.update(overrides)
    return StreamConfig(**body)


def _labels(n: int = 100) -> list[int]:
    return [0] * n


def _features(n: int = 100) -> list[dict]:
    return [{k: float(t) + k for k in range(6)} for t in range(n)]


def test_label_swap_abrupt_swaps_all_post_position_source_labels():
    cfg = _label_swap_cfg(shape="abrupt")
    _, labels = apply_mechanism(_features(), _labels(), cfg, np.random.default_rng(0))
    assert labels[:20] == [0] * 20
    assert labels[20:] == [1] * 80


def test_label_swap_recurring_alternates_blocks():
    cfg = _label_swap_cfg(shape="recurring", period=20, n_episodes=2)
    _, labels = apply_mechanism(_features(), _labels(), cfg, np.random.default_rng(0))
    assert labels[20:40] == [1] * 20
    assert labels[40:60] == [0] * 20
    assert labels[60:80] == [1] * 20
    assert labels[80:] == [0] * 20


def test_label_swap_gradual_starts_soft_and_ends_full():
    cfg = _label_swap_cfg(shape="gradual", width=20)
    _, labels = apply_mechanism(_features(), _labels(), cfg, np.random.default_rng(0))
    assert labels[19] == 0
    assert labels[20] == 0
    assert 0 < sum(labels[20:40]) < 20
    assert labels[40:] == [1] * 60


def test_label_swap_incremental_is_stepped():
    cfg = _label_swap_cfg(shape="incremental", width=20, params={"swap_from": 0, "swap_to": 1, "n_steps": 4})
    _, labels = apply_mechanism(_features(), _labels(), cfg, np.random.default_rng(0))
    assert labels[40:] == [1] * 60
    ramp = labels[20:40]
    assert 0 < sum(ramp) < 20


def test_feature_permutation_respects_recurring_schedule():
    cfg = StreamConfig(
        generator="elec2",
        shape="recurring",
        n_samples=100,
        warm_up=10,
        position=20,
        period=20,
        n_episodes=2,
        noise=0.0,
        mechanism="feature_permutation",
    )
    X = _features()
    original = [dict(row) for row in X]
    out, _ = apply_mechanism(X, _labels(), cfg, np.random.default_rng(0))
    assert out[40:60] == original[40:60]
    assert out[80:] == original[80:]
    assert out[20:40] != original[20:40]
    assert out[60:80] != original[60:80]
    for offset, row in enumerate(out[20:40]):
        assert sorted(row.values()) == sorted(original[20 + offset].values())
