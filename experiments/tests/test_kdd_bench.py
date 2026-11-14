from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.kdd_bench import (
    DRIFT_TYPES,
    MAX_DELAY,
    TRANSITIONS,
    DriftEpisode,
    _fit_episode,
    build_trial_stream,
    evaluate_alarms,
    should_apply_drift,
    transform_instance,
)


class StubRng:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def episode(**overrides) -> DriftEpisode:
    body = {
        "onset": 50,
        "width": 0,
        "selected_label": 0,
        "swap_label": 1,
        "permutation": (2, 0, 1),
        "filter_index": 0,
        "filter_threshold": 0.5,
    }
    body.update(overrides)
    return DriftEpisode(**body)


def test_abrupt_drift_starts_at_onset():
    ep = episode()
    assert not should_apply_drift(49, ep, StubRng(0.0))
    assert should_apply_drift(50, ep, StubRng(0.0))
    assert should_apply_drift(500, ep, StubRng(0.0))


def test_gradual_drift_is_probabilistic_across_width():
    ep = episode(onset=50, width=40)
    assert not should_apply_drift(49, ep, StubRng(0.99))
    assert not should_apply_drift(50, ep, StubRng(0.0))
    assert should_apply_drift(70, ep, StubRng(0.49))
    assert not should_apply_drift(70, ep, StubRng(0.51))
    assert should_apply_drift(90, ep, StubRng(0.99))


def test_label_swap_only_touches_selected_label():
    ep = episode(selected_label=0, swap_label=1)
    x = np.array([1.0, 2.0, 3.0])
    x_out, y_out = transform_instance(x, 0, "label_swap", ep, StubRng(0.5))
    assert y_out == 1
    assert np.array_equal(x_out, x)
    _, y_other = transform_instance(x, 2, "label_swap", ep, StubRng(0.5))
    assert y_other == 2


def test_class_prior_skips_selected_label_with_probability():
    ep = episode(selected_label=0)
    x = np.array([1.0, 2.0, 3.0])
    assert transform_instance(x, 0, "class_prior", ep, StubRng(0.0)) is None
    kept = transform_instance(x, 0, "class_prior", ep, StubRng(0.99))
    assert kept is not None and kept[1] == 0
    assert transform_instance(x, 1, "class_prior", ep, StubRng(0.0)) is not None


def test_feature_permutation_applies_fixed_permutation():
    ep = episode(permutation=(2, 0, 1))
    x = np.array([1.0, 2.0, 3.0])
    x_out, y_out = transform_instance(x, 1, "feature_permutation", ep, StubRng(0.5))
    assert np.array_equal(x_out, np.array([3.0, 1.0, 2.0]))
    assert y_out == 1


def test_feature_filtering_skips_values_above_threshold():
    ep = episode(filter_index=0, filter_threshold=0.5)
    assert transform_instance(np.array([0.6, 0.0]), 1, "feature_filtering", ep, StubRng(0.5)) is None
    kept = transform_instance(np.array([0.4, 0.0]), 1, "feature_filtering", ep, StubRng(0.5))
    assert kept is not None


def test_metrics_perfect_detection():
    metrics = evaluate_alarms([100], onset=100, end=100, max_delay=50, n_samples=1_000)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["mdt"] == 0.0
    assert metrics["ndt"] == 0.0
    assert metrics["far"] == 0.0
    assert metrics["ar"] == pytest.approx(1.0)


def test_metrics_count_false_alarms():
    metrics = evaluate_alarms([10, 100], onset=100, end=100, max_delay=50, n_samples=1_000)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["far"] == pytest.approx(1.0)
    assert metrics["ar"] == pytest.approx(2.0)


def test_metrics_missed_detection():
    metrics = evaluate_alarms([10], onset=100, end=100, max_delay=50, n_samples=1_000)
    assert metrics["tp"] == 0
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert math.isnan(metrics["mdt"])


def test_metrics_gradual_window_anchored_at_end():
    inside = evaluate_alarms([190], onset=100, end=150, max_delay=50, n_samples=1_000)
    outside = evaluate_alarms([201], onset=100, end=150, max_delay=50, n_samples=1_000)
    assert inside["tp"] == 1
    assert outside["tp"] == 0
    assert outside["fp"] == 1


def _data(n: int = 240, n_features: int = 3):
    rng = np.random.default_rng(11)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.int32)
    return X, y, [f"f{index}" for index in range(n_features)]


def _task(
    drift_type: str,
    transition: str = "abrupt",
    trial: int = 0,
    seed: int = 3,
    max_instances: int = 1_000,
) -> dict:
    return {
        "dataset": "Electricity",
        "drift_type": drift_type,
        "transition": transition,
        "trial": trial,
        "seed": seed,
        "max_instances": max_instances,
    }


@pytest.mark.parametrize("drift_type", DRIFT_TYPES)
@pytest.mark.parametrize("transition", TRANSITIONS)
def test_build_trial_stream_matches_reference_loop(drift_type: str, transition: str):
    data = _data()
    task = _task(drift_type, transition)
    stream = build_trial_stream(task, data)

    X_all, y_all, _ = data
    rng = np.random.default_rng(
        [
            task["seed"],
            task["trial"],
            DRIFT_TYPES.index(drift_type),
            TRANSITIONS.index(transition),
        ]
    )
    order = rng.permutation(len(y_all))
    X = X_all[order]
    y = y_all[order]
    episode = _fit_episode(
        X,
        y,
        drift_type,
        transition,
        MAX_DELAY["Electricity"],
        rng,
    )
    kept_x = []
    kept_y = []
    kept_index = []
    for index in range(len(y)):
        x = X[index]
        label = int(y[index])
        if should_apply_drift(index, episode, rng):
            transformed = transform_instance(x, label, drift_type, episode, rng)
            if transformed is None:
                continue
            x, label = transformed
        kept_x.append(x)
        kept_y.append(label)
        kept_index.append(index)

    assert stream.episode == episode
    assert stream.n_samples == len(y)
    assert np.array_equal(stream.index, np.asarray(kept_index))
    assert np.array_equal(stream.y, np.asarray(kept_y))
    assert np.array_equal(stream.X, np.asarray(kept_x, dtype=X_all.dtype))


def test_build_trial_stream_is_deterministic():
    data = _data()
    first = build_trial_stream(_task("feature_filtering"), data)
    second = build_trial_stream(_task("feature_filtering"), data)
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.y, second.y)
    assert np.array_equal(first.index, second.index)
    assert first.episode == second.episode


def test_build_trial_stream_respects_max_instances():
    stream = build_trial_stream(_task("label_swap", max_instances=120), _data(240))
    assert stream.n_samples == 120
    assert int(stream.index.max()) < 120


@pytest.mark.parametrize("drift_type", ["class_prior", "feature_filtering"])
def test_build_trial_stream_drops_rows(drift_type: str):
    stream = build_trial_stream(_task(drift_type), _data(400))
    assert len(stream.y) < stream.n_samples
