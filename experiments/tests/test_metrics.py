from __future__ import annotations

import numpy as np
import pytest

from experiments.driftlab.config import EpisodeSpec
from experiments.driftlab.metrics import (
    evaluate_detections,
    evaluate_window,
    panel_kappa_ref,
)


def test_detection_tp_fp_and_ndt():
    episodes = [EpisodeSpec(100, 100, "abrupt")]
    result = evaluate_detections([110, 200], episodes, n_samples=1_000, delta_max=50)
    assert result.tp_alarms == 1
    assert result.fp_alarms == 1
    assert result.detected_episodes == 1
    assert result.fn_episodes == 0
    assert result.alarm_precision == pytest.approx(0.5)
    assert result.episode_recall == pytest.approx(1.0)
    assert result.ndt_ds == pytest.approx(10 / 50)
    assert result.ndt_de == pytest.approx(10 / 50)
    assert result.far_per_1k == pytest.approx(1.0)


def test_detection_missed_episode():
    episodes = [EpisodeSpec(100, 100, "abrupt")]
    result = evaluate_detections([10], episodes, n_samples=1_000, delta_max=50)
    assert result.tp_alarms == 0
    assert result.fp_alarms == 1
    assert result.detected_episodes == 0
    assert result.fn_episodes == 1
    assert result.episode_recall == pytest.approx(0.0)
    assert result.ndt_ds is None


def test_detection_gradual_anchors():
    episodes = [EpisodeSpec(100, 300, "gradual")]
    result = evaluate_detections([350], episodes, n_samples=1_000, delta_max=100)
    assert result.detected_episodes == 1
    assert result.ndt_ds == pytest.approx(1.0)
    assert result.ndt_de == pytest.approx(0.5)


def test_detection_no_drift_control():
    result = evaluate_detections([10, 20, 30], [], n_samples=1_000, delta_max=50)
    assert result.fp_alarms == 3
    assert result.far_per_1k == pytest.approx(3.0)
    assert result.episode_recall is None


def test_no_alarms_with_episode_is_zero_f1():
    result = evaluate_detections([], [EpisodeSpec(100, 100, "abrupt")], n_samples=1_000, delta_max=50)
    assert result.alarm_precision == 0.0
    assert result.episode_recall == 0.0
    assert result.f1 == 0.0


def test_window_no_exit_has_zero_damage():
    kappa = np.full(100, 0.8)
    err = np.zeros(100)
    oracle = np.zeros(100)
    result = evaluate_window(
        kappa, err, oracle, EpisodeSpec(50, 50, "abrupt"), kappa_ref=0.8, window=20
    )
    assert result.exit_sample is None
    assert result.recovery_sample is None
    assert result.recovery_rate == 0.0
    assert result.damage_ref == pytest.approx(0.0)
    assert result.aue == pytest.approx(0.0)


def test_window_exit_and_recovery():
    kappa = np.concatenate([np.full(10, 0.8), np.full(5, 0.5), np.full(10, 0.8)])
    err = np.zeros(25)
    oracle = np.zeros(25)
    result = evaluate_window(
        kappa,
        err,
        oracle,
        EpisodeSpec(10, 10, "abrupt"),
        kappa_ref=0.8,
        window=15,
        band=0.9,
        stretch=3,
    )
    assert result.exit_sample == 10
    assert result.recovery_sample == 15
    assert result.recovery_samples == 5
    assert result.recovery_rate == 1.0
    assert result.dip == pytest.approx(0.3)
    assert result.damage_ref > 0


def test_aue_excess_error():
    kappa = np.full(50, 0.8)
    err = np.zeros(50)
    err[10:20] = 0.25
    oracle = np.zeros(50)
    oracle[10:20] = 0.05
    result = evaluate_window(
        kappa, err, oracle, EpisodeSpec(10, 10, "abrupt"), kappa_ref=0.8, window=10
    )
    assert result.aue == pytest.approx(0.2)


def test_panel_reference_is_shared():
    left = np.full(100, 0.8)
    right = np.full(100, 0.6)
    ref = panel_kappa_ref({"left": left, "right": right}, EpisodeSpec(50, 50, "abrupt"))
    assert ref == pytest.approx(0.7)
