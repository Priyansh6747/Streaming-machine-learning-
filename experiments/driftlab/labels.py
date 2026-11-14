"""Label-arrival schedules for the Delay Tax benchmark.

A schedule maps each stream index ``t`` to the index at which its label becomes
available. ``-1`` means the label never arrives. The fixed schedule is the
Lemma-1 case; the other regimes are the E21 benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LabelSchedule:
    arrival: np.ndarray
    regime: str
    params: dict[str, Any]

    def __post_init__(self) -> None:
        arrival = np.asarray(self.arrival, dtype=np.int64)
        if arrival.ndim != 1:
            raise ValueError("arrival must be one-dimensional")
        object.__setattr__(self, "arrival", arrival)

    def __len__(self) -> int:
        return len(self.arrival)

    def available_at(self, t: int) -> np.ndarray:
        """Indices whose labels have arrived by time ``t``."""
        return np.flatnonzero((self.arrival >= 0) & (self.arrival <= int(t)))

    def is_available(self, index: int, t: int) -> bool:
        arrival = int(self.arrival[int(index)])
        return arrival >= 0 and arrival <= int(t)

    def hit_rate(self) -> float:
        return float(np.mean(self.arrival >= 0)) if len(self.arrival) else 0.0

    def mean_delay(self) -> float:
        observed = self.arrival[self.arrival >= 0]
        if len(observed) == 0:
            return float("inf")
        indices = np.flatnonzero(self.arrival >= 0)
        return float(np.mean(observed - indices))


def fixed(n: int, delay: int = 0) -> LabelSchedule:
    arrival = np.arange(n, dtype=np.int64) + int(delay)
    return LabelSchedule(arrival, "fixed", {"delay": int(delay)})


def geometric(n: int, mean_delay: int = 100, seed: int = 0) -> LabelSchedule:
    """Geometric label delays with the requested mean."""
    mean_delay = max(0, int(mean_delay))
    if mean_delay == 0:
        return fixed(n, 0)
    rng = np.random.default_rng(seed)
    delays = rng.geometric(1.0 / (mean_delay + 1.0), size=n) - 1
    arrival = np.arange(n, dtype=np.int64) + delays.astype(np.int64)
    return LabelSchedule(arrival, "geometric", {"mean_delay": mean_delay, "seed": seed})


def partial(n: int, p: float = 0.5, seed: int = 0) -> LabelSchedule:
    """Each label arrives immediately with probability ``p``, else never."""
    p = float(np.clip(p, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    mask = rng.random(n) < p
    arrival = np.where(mask, np.arange(n, dtype=np.int64), -1)
    return LabelSchedule(arrival, "partial", {"p": p, "seed": seed})


def selective(n: int, flags: np.ndarray) -> LabelSchedule:
    """Only the flagged samples receive an immediate label."""
    flags = np.asarray(flags, dtype=bool)
    if flags.shape != (n,):
        raise ValueError(f"expected {n} flags, got {flags.shape}")
    arrival = np.where(flags, np.arange(n, dtype=np.int64), -1)
    return LabelSchedule(arrival, "selective", {"selected": int(flags.sum())})


def batch(n: int, batch_size: int = 16, delay: int = 0) -> LabelSchedule:
    """Labels arrive at the end of each batch, plus an optional delay."""
    batch_size = max(1, int(batch_size))
    indices = np.arange(n, dtype=np.int64)
    batch_end = ((indices // batch_size) + 1) * batch_size - 1
    arrival = np.minimum(batch_end, n - 1) + int(delay)
    return LabelSchedule(
        arrival,
        "batch",
        {"batch_size": batch_size, "delay": int(delay)},
    )


def build_schedule(regime: str, n: int, **params: Any) -> LabelSchedule:
    if regime == "fixed":
        return fixed(n, **params)
    if regime == "geometric":
        return geometric(n, **params)
    if regime == "partial":
        return partial(n, **params)
    if regime == "selective":
        return selective(n, **params)
    if regime == "batch":
        return batch(n, **params)
    raise ValueError(f"unknown label regime: {regime!r}")
