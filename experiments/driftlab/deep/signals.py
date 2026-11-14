"""Label-free deep drift signals for DF-H Stage 1.

The monitor windows a raw feature stream, asks the frozen encoder for a
deterministic snapshot per window, and turns the snapshots into the signal
family described in the DF-H plan:

* ``ssl_loss`` — masked-feature cross-entropy of the encoder
* ``embedding_shift`` — standardised shift of the ``cls`` embedding mean
* ``embedding_var_shift`` — shift of the ``cls`` embedding variance
* ``head_disagreement`` — mean pairwise Jensen-Shannon divergence across heads
* ``entropy`` — normalised predictive entropy of the active head
* ``internal_adwin`` — ADWIN on the windowed SSL loss (baseline alarm)
* ``feature_ks`` — max per-feature KS statistic against the reference window

All values are normalised against the first ``reference_fraction`` of windows
with a robust median/MAD z-score. No labels are used anywhere in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from river.drift import ADWIN
from scipy.stats import ks_2samp

SIGNAL_NAMES = [
    "ssl_loss",
    "embedding_shift",
    "embedding_var_shift",
    "head_disagreement",
    "entropy",
    "internal_adwin",
    "feature_ks",
]


@dataclass
class SignalConfig:
    window_size: int = 500
    stride: int = 100
    reference_fraction: float = 0.2
    mask_fraction: float = 0.15
    adwin_delta: float = 0.002
    ks_reference_size: int = 2_000
    eps: float = 1e-6

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SignalSeries:
    starts: np.ndarray
    ends: np.ndarray
    values: dict[str, np.ndarray]
    zvalues: dict[str, np.ndarray]
    reference_windows: int
    config: SignalConfig

    def as_arrays(self, prefix: str = "") -> dict[str, np.ndarray]:
        payload = {
            f"{prefix}window_start": self.starts.astype(np.int64),
            f"{prefix}window_end": self.ends.astype(np.int64),
        }
        for name in SIGNAL_NAMES:
            payload[f"{prefix}{name}"] = self.values[name]
            payload[f"{prefix}{name}_z"] = self.zvalues[name]
        return payload


class DeepSignalMonitor:
    def __init__(self, config: SignalConfig | None = None):
        self.config = config or SignalConfig()

    def extract(self, learner: Any, X: Sequence[dict] | np.ndarray) -> SignalSeries:
        array = _as_array(learner, X)
        n_samples, n_features = array.shape
        window = int(self.config.window_size)
        stride = int(self.config.stride)
        if window <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive")
        if n_samples < window:
            raise ValueError(
                f"need at least {window} samples for one window, got {n_samples}"
            )
        starts = np.arange(0, n_samples - window + 1, stride, dtype=np.int64)
        ends = starts + window
        snapshots = [
            learner.signal_snapshot(array[start:end], window_index=index)
            for index, (start, end) in enumerate(zip(starts, ends))
        ]

        n_windows = len(starts)
        ref_count = min(
            n_windows,
            max(1, int(np.ceil(n_windows * float(self.config.reference_fraction)))),
        )
        ssl_loss = np.asarray([item["ssl_loss"] for item in snapshots], dtype=np.float64)
        disagreement = np.asarray(
            [item["head_disagreement"] for item in snapshots], dtype=np.float64
        )
        entropy = np.asarray([item["entropy"] for item in snapshots], dtype=np.float64)
        means = np.vstack([item["cls_mean"] for item in snapshots])
        variances = np.vstack([item["cls_var"] for item in snapshots])
        ref_mean = means[:ref_count].mean(axis=0)
        ref_var = variances[:ref_count].mean(axis=0) + self.config.eps
        scale = np.sqrt(ref_var)
        embedding_shift = np.sqrt(
            np.mean(((means - ref_mean) / np.maximum(scale, self.config.eps)) ** 2, axis=1)
        )
        embedding_var_shift = np.mean(
            np.abs(np.log((variances + self.config.eps) / (ref_var + self.config.eps))),
            axis=1,
        )

        internal_adwin = np.zeros(n_windows, dtype=np.float64)
        adwin = ADWIN(delta=float(self.config.adwin_delta))
        for index, value in enumerate(ssl_loss):
            adwin.update(float(value))
            if adwin.drift_detected:
                internal_adwin[index] = 1.0

        feature_ks = _feature_ks(
            array,
            starts,
            ends,
            ref_count=ref_count,
            n_features=n_features,
            reference_size=int(self.config.ks_reference_size),
        )

        values = {
            "ssl_loss": ssl_loss,
            "embedding_shift": embedding_shift,
            "embedding_var_shift": embedding_var_shift,
            "head_disagreement": disagreement,
            "entropy": entropy,
            "internal_adwin": internal_adwin,
            "feature_ks": feature_ks,
        }
        values = {
            name: np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
            for name, series in values.items()
        }
        zvalues = {
            name: _robust_z(series, ref_count, self.config.eps)
            for name, series in values.items()
        }
        return SignalSeries(
            starts=starts,
            ends=ends,
            values=values,
            zvalues=zvalues,
            reference_windows=ref_count,
            config=self.config,
        )


def _as_array(learner: Any, X: Sequence[dict] | np.ndarray) -> np.ndarray:
    if isinstance(X, np.ndarray):
        array = np.asarray(X, dtype=np.float64)
    else:
        keys = list(getattr(learner, "feature_keys", []) or [])
        if not keys:
            raise ValueError("learner has no feature_keys; pass a 2D array instead")
        rows = []
        for item in X:
            if isinstance(item, dict):
                rows.append([float(item.get(key, 0.0)) for key in keys])
            else:
                rows.append([float(value) for value in item])
        array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D feature array, got shape {array.shape}")
    return array


def _feature_ks(
    array: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    ref_count: int,
    n_features: int,
    reference_size: int,
) -> np.ndarray:
    reference = array[: int(ends[ref_count - 1])]
    if len(reference) > reference_size:
        step = int(np.ceil(len(reference) / max(1, reference_size)))
        reference = reference[::step][:reference_size]
    scores = np.zeros(len(starts), dtype=np.float64)
    for index, (start, end) in enumerate(zip(starts, ends)):
        window = array[int(start) : int(end)]
        best = 0.0
        for feature in range(n_features):
            statistic = ks_2samp(
                reference[:, feature],
                window[:, feature],
                method="asymp",
            ).statistic
            if np.isfinite(statistic) and statistic > best:
                best = float(statistic)
        scores[index] = best
    return scores


def _robust_z(values: np.ndarray, ref_count: int, eps: float) -> np.ndarray:
    reference = values[:ref_count]
    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median)))
    scale = 1.4826 * mad
    if scale < eps:
        scale = float(np.std(reference))
    return (values - median) / max(scale, eps)
