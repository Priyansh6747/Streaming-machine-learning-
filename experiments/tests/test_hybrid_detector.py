from __future__ import annotations

import numpy as np

from experiments.driftlab.fusion import (
    alarms_from_windows,
    build_hybrid_runs,
    calibrate_threshold,
    fuse_and,
    fuse_or,
    fuse_two_of_three,
)


def test_calibrate_threshold_respects_far():
    values = np.arange(1_000, dtype=float)
    threshold = calibrate_threshold(values, far_per_1k=1.0, samples_per_window=1)
    assert int((values >= threshold).sum()) == 1


def test_alarms_from_windows_uses_first_crossing_and_refractory():
    times = [100, 200, 3_000, 3_100]
    scores = [1.0, 1.0, 1.0, 0.1]
    assert alarms_from_windows(
        times, scores, threshold=0.5, refractory=2_000
    ) == [100, 3_000]


def test_fuse_and_confirms_inside_window():
    alarms, warnings = fuse_and(
        [100, 5_000],
        [500, 6_000],
        window=1_000,
        refractory=2_000,
    )
    assert alarms == [500, 6_000]
    assert warnings == [100, 5_000]


def test_fuse_and_drops_unconfirmed_warning():
    alarms, warnings = fuse_and([100], [5_000], window=1_000)
    assert alarms == []
    assert warnings == []


def test_fuse_two_of_three_needs_two_distinct_sources():
    alarms, _ = fuse_two_of_three([[100], [200], [3_000]], window=500)
    assert alarms == [200]
    alarms, _ = fuse_two_of_three([[100], [100], [5_000]], window=500)
    assert alarms == [100]
    alarms, _ = fuse_two_of_three([[100], [2_000], [4_000]], window=500)
    assert alarms == []


def test_fuse_or_is_refractory_suppressed():
    assert fuse_or([100], [150], [3_000], refractory=2_000) == [100, 3_000]


def test_build_hybrid_runs_covers_all_variants():
    runs = build_hybrid_runs(
        s1_deep=[100],
        s1_classical=[300],
        s1_both=[100],
        scalar_alarms={"SEED": [400], "ADWIN": [900], "HDDMAverage": [5_000]},
        window=2_000,
    )
    names = {run.name for run in runs}
    assert {
        "S1_deep",
        "S1_classical",
        "S1_both",
        "OR_deep",
        "AND_deep",
        "2of3_deep",
        "2of3_classical",
        "2of3_both",
        "2of3_nodeep",
        "DFH",
    } == names
    dfh = next(run for run in runs if run.name == "DFH")
    two_both = next(run for run in runs if run.name == "2of3_both")
    assert dfh.alarms == two_both.alarms
