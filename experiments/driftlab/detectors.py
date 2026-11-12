"""Drift-detector adapters and replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from river.drift import ADWIN, KSWIN, PageHinkley
from river.drift.binary import DDM, EDDM, FHDDM, HDDMA, HDDMW

DETECTOR_FACTORIES: dict[str, Any] = {
    "adwin": lambda: ADWIN(delta=0.002),
    "kswin": lambda: KSWIN(alpha=0.005, window_size=100, stat_size=30, seed=42),
    "page_hinkley": lambda: PageHinkley(
        min_instances=30, delta=0.005, threshold=50.0, alpha=1 - 0.0001, mode="both"
    ),
    "ddm": lambda: DDM(warm_start=30, warning_threshold=2.0, drift_threshold=3.0),
    "eddm": lambda: EDDM(warm_start=30, alpha=0.95, beta=0.9),
    "fhddm": lambda: FHDDM(sliding_window_size=100, confidence_level=0.000001),
    "hddm_a": lambda: HDDMA(
        drift_confidence=0.001, warning_confidence=0.005, two_sided_test=False
    ),
    "hddm_w": lambda: HDDMW(
        drift_confidence=0.001,
        warning_confidence=0.005,
        lambda_val=0.05,
        two_sided_test=False,
    ),
}

CONTINUOUS_DETECTORS = {"adwin", "kswin", "page_hinkley"}
BINARY_DETECTORS = {"ddm", "eddm", "fhddm", "hddm_a", "hddm_w"}
DETECTOR_NAMES = sorted(DETECTOR_FACTORIES)


@dataclass
class DetectorRun:
    detector: str
    channel: str
    alarms: list[int] = field(default_factory=list)
    warnings: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "channel": self.channel,
            "alarms": self.alarms,
            "warnings": self.warnings,
        }


def make_detector(name: str):
    try:
        return DETECTOR_FACTORIES[name]()
    except KeyError as exc:
        raise ValueError(
            f"unknown detector {name!r}; available: {', '.join(DETECTOR_NAMES)}"
        ) from exc


def ewma(values: Sequence[float], alpha: float = 0.01) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.empty_like(arr)
    if len(arr) == 0:
        return out
    prev = float(arr[0])
    for i, value in enumerate(arr):
        prev = alpha * float(value) + (1.0 - alpha) * prev
        out[i] = prev
    return out


def channel_values(values: Sequence[float], channel: str, alpha: float = 0.01) -> np.ndarray:
    if channel == "raw":
        return np.asarray(values, dtype=float)
    if channel == "ewma":
        return ewma(values, alpha=alpha)
    raise ValueError(f"unknown error channel: {channel!r}")


def supports_channel(detector: str, channel: str) -> bool:
    if channel == "raw":
        return True
    if channel == "ewma":
        return detector in CONTINUOUS_DETECTORS
    return False


def run_detector(
    name: str,
    values: Sequence[float],
    channel: str = "raw",
    alpha: float = 0.01,
) -> DetectorRun:
    if not supports_channel(name, channel):
        raise ValueError(f"{name!r} does not support the {channel!r} channel")
    series = channel_values(values, channel, alpha=alpha)
    detector = make_detector(name)
    run = DetectorRun(detector=name, channel=channel)
    for t, value in enumerate(series):
        detector.update(float(value))
        if detector.drift_detected:
            run.alarms.append(t)
        if getattr(detector, "warning_detected", False):
            run.warnings.append(t)
    return run


def replay_all(
    values: Sequence[float],
    detectors: Sequence[str],
    channels: Sequence[str],
    alpha: float = 0.01,
) -> list[DetectorRun]:
    runs: list[DetectorRun] = []
    for name in detectors:
        for channel in channels:
            if supports_channel(name, channel):
                runs.append(run_detector(name, values, channel=channel, alpha=alpha))
    return runs
