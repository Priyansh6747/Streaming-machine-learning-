from __future__ import annotations

import numpy as np
import pytest

from experiments.driftlab.config import EpisodeSpec, StreamConfig
from experiments.driftlab.metrics import evaluate_window, rolling_kappa
from experiments.driftlab.streams import build_stream


def test_arrival_time_shift_reports_prediction_time_recovery():
    kappa = np.concatenate([np.full(15, 0.8), np.full(5, 0.5), np.full(10, 0.8)])
    zeros = np.zeros_like(kappa)
    result = evaluate_window(
        kappa,
        zeros,
        zeros,
        EpisodeSpec(10, 10, "abrupt"),
        kappa_ref=0.8,
        window=10,
        band=0.9,
        stretch=3,
        shift=5,
    )
    assert result.exit_sample == 15
    assert result.exit_prediction_sample == 10
    assert result.recovery_sample == 20
    assert result.recovery_samples == 5
    assert result.recovery_prediction_samples == 0
    assert result.damage_ref > 0


def test_rolling_kappa_skips_missing_labels():
    kappa = rolling_kappa(
        [0, None, 1, 1],
        [0, None, 1, 0],
        window=2,
        warm_up=0,
    )
    assert np.isnan(kappa[1])
    assert np.isfinite(kappa[2])


def test_label_swap_changes_only_post_drift_labels():
    cfg = StreamConfig(
        generator="sea",
        shape="abrupt",
        n_samples=1_000,
        warm_up=200,
        position=500,
        noise=0.0,
        concept_a=0,
        concept_b=0,
        mechanism="label_swap",
        params={"swap_from": 0, "swap_to": 1},
    )
    stream = build_stream(cfg)
    pre = stream.y[:500]
    post = stream.y[500:]
    assert all(v in {0, 1} for v in pre + post)
    assert all(v == 1 for v in post)


def test_feature_permutation_keeps_feature_keys():
    cfg = StreamConfig(
        generator="sea",
        shape="abrupt",
        n_samples=1_000,
        warm_up=200,
        position=500,
        noise=0.0,
        mechanism="feature_permutation",
    )
    stream = build_stream(cfg)
    assert set(stream.X[0]) == set(stream.X[600])
    assert len(stream.X[0]) == 3
