from __future__ import annotations

import pytest

from experiments.driftlab.config import StreamConfig
from experiments.driftlab.streams import build_stream, mixing_weight


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
