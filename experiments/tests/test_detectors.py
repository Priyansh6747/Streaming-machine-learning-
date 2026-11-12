from __future__ import annotations

import numpy as np
import pytest

from experiments.driftlab.detectors import (
    DETECTOR_NAMES,
    ewma,
    replay_all,
    run_detector,
    supports_channel,
)


def test_ewma_tracks_direction():
    values = [0.0] * 100 + [1.0] * 100
    smoothed = ewma(values, alpha=0.1)
    assert smoothed[0] == pytest.approx(0.0)
    assert smoothed[-1] > smoothed[99]
    assert 0.0 <= smoothed[-1] <= 1.0


def test_every_detector_runs_without_error():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.integers(0, 2, 300), rng.integers(0, 2, 300)])
    for name in DETECTOR_NAMES:
        run = run_detector(name, values, channel="raw")
        assert isinstance(run.alarms, list)
        assert all(isinstance(a, int) for a in run.alarms)


def test_binary_detectors_reject_ewma_channel():
    for name in ["eddm", "fhddm", "hddm_a", "hddm_w", "ddm"]:
        assert not supports_channel(name, "ewma")
        with pytest.raises(ValueError):
            run_detector(name, [0, 1, 0, 1], channel="ewma")


def test_replay_all_skips_unsupported_channels():
    runs = replay_all([0, 1] * 100, ["adwin", "eddm"], ["raw", "ewma"])
    keys = {(r.detector, r.channel) for r in runs}
    assert ("adwin", "raw") in keys
    assert ("adwin", "ewma") in keys
    assert ("eddm", "raw") in keys
    assert ("eddm", "ewma") not in keys
