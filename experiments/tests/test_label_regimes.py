from __future__ import annotations

import numpy as np
import pytest

from experiments.driftlab.labels import (
    batch,
    build_schedule,
    fixed,
    geometric,
    partial,
    selective,
)


def test_fixed_schedule_shifts_every_label():
    schedule = fixed(10, delay=3)
    assert schedule.arrival[0] == 3
    assert schedule.arrival[-1] == 12
    assert schedule.available_at(2).size == 0
    assert schedule.available_at(3).tolist() == [0]
    assert schedule.hit_rate() == 1.0
    assert schedule.mean_delay() == 3.0


def test_partial_schedule_endpoints():
    all_labels = partial(50, p=1.0, seed=0)
    none = partial(50, p=0.0, seed=0)
    assert all_labels.hit_rate() == 1.0
    assert none.hit_rate() == 0.0
    assert np.array_equal(all_labels.arrival, np.arange(50))
    assert np.all(none.arrival == -1)


def test_batch_schedule_arrives_at_batch_end():
    schedule = batch(25, batch_size=10)
    assert schedule.arrival[0] == 9
    assert schedule.arrival[9] == 9
    assert schedule.arrival[10] == 19
    assert schedule.arrival[24] == 24


def test_batch_schedule_with_delay():
    schedule = batch(20, batch_size=5, delay=7)
    assert schedule.arrival[0] == 11
    assert schedule.arrival[19] == 26


def test_selective_schedule_only_labels_flagged():
    flags = np.zeros(20, dtype=bool)
    flags[[2, 7, 11]] = True
    schedule = selective(20, flags)
    assert schedule.available_at(20).tolist() == [2, 7, 11]
    assert schedule.hit_rate() == pytest.approx(3 / 20)


def test_geometric_schedule_is_deterministic_and_reproducible():
    first = geometric(200, mean_delay=20, seed=7)
    second = geometric(200, mean_delay=20, seed=7)
    other = geometric(200, mean_delay=20, seed=8)
    assert np.array_equal(first.arrival, second.arrival)
    assert not np.array_equal(first.arrival, other.arrival)
    assert first.hit_rate() == 1.0
    assert 5.0 < first.mean_delay() < 60.0


def test_build_schedule_dispatch():
    assert build_schedule("fixed", 5, delay=2).regime == "fixed"
    assert build_schedule("partial", 5, p=0.5, seed=1).regime == "partial"
    with pytest.raises(ValueError, match="unknown label regime"):
        build_schedule("nope", 5)
