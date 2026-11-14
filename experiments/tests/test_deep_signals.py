from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.driftlab.deep.signals import (
    SIGNAL_NAMES,
    DeepSignalMonitor,
    SignalConfig,
)
from experiments.driftlab.deep.train import DriftFormerConfig, DriftFormerLearner


def make_learner() -> DriftFormerLearner:
    config = DriftFormerConfig(
        d_model=8,
        n_heads=2,
        n_layers=1,
        ffn=16,
        n_heads_bank=2,
        n_bins=4,
        block_size=8,
        batch_size=8,
        device="cpu",
        seed=7,
    )
    return DriftFormerLearner("df", [0, 1, 2, 3], n_classes=2, config=config)


def test_pretrain_ssl_leaves_supervised_heads_untouched():
    learner = make_learner()
    before = [parameter.detach().clone() for parameter in learner.model.heads.parameters()]
    X = np.random.default_rng(0).normal(size=(32, 4))
    learner.pretrain_ssl(X, epochs=1, batch_size=8)
    after = [parameter.detach().clone() for parameter in learner.model.heads.parameters()]
    for old, new in zip(before, after):
        assert torch.equal(old, new)


def test_signal_snapshot_is_finite_and_deterministic():
    learner = make_learner()
    X = np.random.default_rng(1).normal(size=(16, 4))
    first = learner.signal_snapshot(X, window_index=3)
    second = learner.signal_snapshot(X, window_index=3)
    for key in ("ssl_loss", "head_disagreement", "entropy"):
        assert np.isfinite(first[key])
        assert first[key] == pytest.approx(second[key])
    assert first["cls_mean"].shape == (8,)
    assert first["cls_var"].shape == (8,)
    assert np.all(np.isfinite(first["cls_mean"]))
    assert np.all(np.isfinite(first["cls_var"]))
    assert first["n_samples"] == 16


def test_monitor_extracts_every_signal():
    learner = make_learner()
    X = np.random.default_rng(2).normal(size=(120, 4))
    config = SignalConfig(
        window_size=16,
        stride=8,
        reference_fraction=0.25,
        ks_reference_size=64,
    )
    series = DeepSignalMonitor(config).extract(learner, X)
    assert len(series.starts) == 14
    assert series.reference_windows == 4
    assert set(series.values) == set(SIGNAL_NAMES)
    for name in SIGNAL_NAMES:
        assert series.values[name].shape == (14,)
        assert np.all(np.isfinite(series.values[name]))
        assert series.zvalues[name].shape == (14,)
        assert np.all(np.isfinite(series.zvalues[name]))
    arrays = series.as_arrays("s0_")
    assert arrays["s0_window_start"].shape == (14,)
    assert "s0_ssl_loss_z" in arrays


def test_monitor_rejects_short_streams():
    learner = make_learner()
    with pytest.raises(ValueError, match="at least"):
        DeepSignalMonitor(SignalConfig(window_size=64, stride=16)).extract(
            learner, np.zeros((32, 4))
        )
