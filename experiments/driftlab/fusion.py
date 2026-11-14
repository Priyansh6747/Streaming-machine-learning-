"""Pure-NumPy alarm fusion for DF-H.

This module deliberately has no River, torch or CapyMOA imports: the KDD
runner executes it inside the isolated ``kdd_env``.

Stage 1 turns windowed deep/classical scores into alarms at a calibrated
false-alarm rate. Stage 2 is the scalar detector panel. The fusion variants
report the confirmation time (the second vote) for AND/2-of-3 rules, and the
earliest alarm for OR rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass
class FusionRun:
    name: str
    alarms: list[int]
    warnings: list[int] = field(default_factory=list)
    family: str = "fusion"


def calibrate_threshold(
    values: Sequence[float],
    far_per_1k: float,
    samples_per_window: int = 100,
) -> float:
    """Threshold whose pre-drift alarm rate is at most ``far_per_1k``.

    ``values`` are pre-drift window scores. ``samples_per_window`` is the
    stride, because a new decision is made every stride samples.
    """
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(array) == 0:
        return float("inf")
    total_samples = len(array) * max(1, int(samples_per_window))
    allowed = max(1, int(round(total_samples * float(far_per_1k) / 1_000.0)))
    ordered = np.sort(array)[::-1]
    index = min(allowed, len(ordered) - 1)
    return float(np.nextafter(ordered[index], np.inf))


def alarms_from_windows(
    times: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    refractory: int = 2_000,
) -> list[int]:
    """Emit the first threshold crossing and suppress repeats for a refractory."""
    out: list[int] = []
    for time, score in zip(times, scores):
        if not np.isfinite(score) or float(score) < float(threshold):
            continue
        stamp = int(time)
        if out and stamp < out[-1] + int(refractory):
            continue
        out.append(stamp)
    return out


def _dedupe(times: Iterable[int], refractory: int) -> list[int]:
    out: list[int] = []
    for stamp in sorted(int(value) for value in times):
        if out and stamp < out[-1] + int(refractory):
            continue
        out.append(stamp)
    return out


def fuse_or(
    *sources: Sequence[int],
    refractory: int = 2_000,
) -> list[int]:
    return _dedupe((stamp for source in sources for stamp in source), refractory)


def fuse_and(
    warnings: Sequence[int],
    confirmations: Sequence[int],
    window: int = 2_000,
    refractory: int = 2_000,
) -> tuple[list[int], list[int]]:
    """Confirm each warning with the first Stage-2 alarm inside ``window``."""
    alarms: list[int] = []
    used: list[int] = []
    ordered = sorted(int(value) for value in confirmations)
    for warning in sorted(set(int(value) for value in warnings)):
        candidates = [
            stamp for stamp in ordered if warning <= stamp <= warning + int(window)
        ]
        if not candidates:
            continue
        confirmation = candidates[0]
        if alarms and confirmation < alarms[-1] + int(refractory):
            continue
        alarms.append(confirmation)
        used.append(warning)
    return alarms, used


def fuse_two_of_three(
    sources: Sequence[Sequence[int]],
    window: int = 2_000,
    refractory: int = 2_000,
) -> tuple[list[int], list[int]]:
    """Emit the second distinct source vote inside a confirmation window."""
    events = sorted(
        (int(stamp), index)
        for index, source in enumerate(sources)
        for stamp in source
    )
    alarms: list[int] = []
    warnings: list[int] = []
    position = 0
    while position < len(events):
        start = events[position][0]
        seen: list[int] = []
        confirmation: int | None = None
        for stamp, index in events:
            if stamp < start or stamp > start + int(window):
                continue
            if index not in seen:
                seen.append(index)
                if len(seen) == 2:
                    confirmation = stamp
                    break
        if confirmation is None:
            position += 1
            continue
        if not alarms or confirmation >= alarms[-1] + int(refractory):
            alarms.append(confirmation)
            warnings.append(start)
        next_position = position
        while next_position < len(events) and events[next_position][0] <= confirmation:
            next_position += 1
        position = max(position + 1, next_position)
    return alarms, warnings


def build_hybrid_runs(
    s1_deep: Sequence[int],
    s1_classical: Sequence[int],
    s1_both: Sequence[int],
    scalar_alarms: dict[str, Sequence[int]],
    window: int = 2_000,
    refractory: int = 2_000,
) -> list[FusionRun]:
    """Build every pre-registered Stage-1, OR, AND and 2-of-3 variant."""
    seed = list(scalar_alarms.get("SEED", []))
    adwin = list(scalar_alarms.get("ADWIN", []))
    hddm = list(scalar_alarms.get("HDDMAverage", []))

    and_alarms, and_warnings = fuse_and(
        s1_deep, seed, window=window, refractory=refractory
    )
    two_deep, two_deep_warn = fuse_two_of_three(
        [s1_deep, seed, adwin], window=window, refractory=refractory
    )
    two_classical, two_classical_warn = fuse_two_of_three(
        [s1_classical, seed, adwin], window=window, refractory=refractory
    )
    two_both, two_both_warn = fuse_two_of_three(
        [s1_both, seed, adwin], window=window, refractory=refractory
    )
    two_nodeep, two_nodeep_warn = fuse_two_of_three(
        [seed, adwin, hddm], window=window, refractory=refractory
    )
    runs = [
        FusionRun("S1_deep", list(s1_deep), list(s1_deep)),
        FusionRun("S1_classical", list(s1_classical), list(s1_classical)),
        FusionRun("S1_both", list(s1_both), list(s1_both)),
        FusionRun(
            "OR_deep",
            fuse_or(s1_deep, seed, refractory=refractory),
            list(s1_deep),
        ),
        FusionRun("AND_deep", and_alarms, and_warnings),
        FusionRun("2of3_deep", two_deep, two_deep_warn),
        FusionRun("2of3_classical", two_classical, two_classical_warn),
        FusionRun("2of3_both", two_both, two_both_warn),
        FusionRun("2of3_nodeep", two_nodeep, two_nodeep_warn),
        FusionRun("DFH", two_both, two_both_warn),
    ]
    return runs
