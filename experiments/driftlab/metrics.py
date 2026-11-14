"""Metric v2: oracle-referenced excess error and KDD-2026 detection metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from .config import EpisodeSpec

try:  # River is a hard dependency, but keep the module importable for tests.
    from river.metrics import CohenKappa
    from river.utils import Rolling
except Exception:  # pragma: no cover
    CohenKappa = None
    Rolling = None


@dataclass
class DetectionMetrics:
    n_episodes: int
    tp_alarms: int
    fp_alarms: int
    fn_episodes: int
    detected_episodes: int
    alarm_precision: float | None
    episode_recall: float | None
    f1: float | None
    ndt_ds: float | None
    ndt_de: float | None
    mean_delay_from_ds: float | None
    mean_delay_from_de: float | None
    alarm_rate_per_1k: float
    far_per_1k: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WindowMetrics:
    start: int
    end: int
    kind: str
    window: int
    shift: int
    kappa_ref: float
    kappa0: float | None
    min_kappa: float | None
    dip: float | None
    exit_sample: int | None
    exit_prediction_sample: int | None
    recovery_sample: int | None
    recovery_samples: int | None
    recovery_prediction_samples: int | None
    recovery_rate: float
    damage_ref: float | None
    damage_full: float | None
    attain_abs: float | None
    aue: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def rolling_kappa(
    y_true: Sequence[int],
    y_pred: Sequence[int | None],
    window: int,
    warm_up: int = 0,
) -> np.ndarray:
    if Rolling is None or CohenKappa is None:
        raise RuntimeError("river is required for rolling_kappa")
    roll = Rolling(CohenKappa(), window_size=window)
    out = np.full(len(y_true), np.nan, dtype=float)
    for t, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yp is None or yt is None:
            continue
        roll.update(int(yt), int(yp))
        if t >= warm_up:
            try:
                value = roll.get()
            except Exception:
                value = None
            if value is not None and np.isfinite(value):
                out[t] = float(value)
    return out


def panel_kappa_ref(
    kappa_by_model: dict[str, np.ndarray],
    episode: EpisodeSpec,
    pre_window: int = 1_000,
) -> float:
    lo = max(0, episode.start - pre_window)
    hi = max(lo + 1, episode.start)
    values = []
    for series in kappa_by_model.values():
        segment = np.asarray(series[lo:hi], dtype=float)
        if np.isfinite(segment).any():
            values.append(float(np.nanmean(segment)))
    return float(np.mean(values)) if values else float("nan")


def evaluate_window(
    kappa: np.ndarray,
    err: np.ndarray,
    oracle_err: np.ndarray,
    episode: EpisodeSpec,
    kappa_ref: float,
    window: int,
    band: float = 0.95,
    stretch: int = 500,
    shift: int = 0,
    recovery_horizon: int | None = None,
) -> WindowMetrics:
    n = len(kappa)
    analysis_lo = max(0, episode.start + shift)
    hi = min(n, analysis_lo + max(1, window))
    horizon = max(1, window, int(recovery_horizon or 0))
    recovery_hi = min(n, analysis_lo + horizon)
    segment = np.asarray(kappa[analysis_lo:hi], dtype=float)
    finite = np.isfinite(segment)

    kappa0 = None
    pre_hi = max(0, episode.start)
    pre_lo = max(0, pre_hi - 1_000)
    pre = np.asarray(kappa[pre_lo:pre_hi], dtype=float)
    if np.isfinite(pre).any():
        kappa0 = float(np.nanmean(pre))

    if not finite.any() or not np.isfinite(kappa_ref):
        return WindowMetrics(
            start=episode.start,
            end=episode.end,
            kind=episode.kind,
            window=window,
            shift=shift,
            kappa_ref=float(kappa_ref),
            kappa0=kappa0,
            min_kappa=None,
            dip=None,
            exit_sample=None,
            exit_prediction_sample=None,
            recovery_sample=None,
            recovery_samples=None,
            recovery_prediction_samples=None,
            recovery_rate=0.0,
            damage_ref=None,
            damage_full=None,
            attain_abs=None,
            aue=None,
        )

    min_kappa = float(np.nanmin(segment))
    dip = max(0.0, float(kappa_ref) - min_kappa)
    threshold = band * float(kappa_ref)

    below = np.where(segment < threshold)[0]
    exit_sample = int(analysis_lo + below[0]) if len(below) else None

    recovery_sample = None
    if exit_sample is not None:
        run = 0
        for t in range(exit_sample, recovery_hi):
            value = kappa[t]
            if np.isfinite(value) and value >= threshold:
                run += 1
                if run >= stretch:
                    recovery_sample = t - stretch + 1
                    break
            else:
                run = 0

    damage_ref = float(np.nansum(np.maximum(float(kappa_ref) - segment, 0.0)))
    full_segment = np.asarray(kappa[analysis_lo:], dtype=float)
    damage_full = float(np.nansum(np.maximum(float(kappa_ref) - full_segment, 0.0)))

    second_half = slice(analysis_lo + (hi - analysis_lo) // 2, hi)
    attain_abs = float(np.nanmean(kappa[second_half])) if np.isfinite(kappa[second_half]).any() else None

    err_seg = np.asarray(err[analysis_lo:hi], dtype=float)
    oracle_seg = np.asarray(oracle_err[analysis_lo:hi], dtype=float)
    mask = np.isfinite(err_seg) & np.isfinite(oracle_seg)
    aue = float(np.mean(err_seg[mask] - oracle_seg[mask])) if mask.any() else None

    recovery_samples = (recovery_sample - analysis_lo) if recovery_sample is not None else None
    return WindowMetrics(
        start=episode.start,
        end=episode.end,
        kind=episode.kind,
        window=window,
        shift=shift,
        kappa_ref=float(kappa_ref),
        kappa0=kappa0,
        min_kappa=min_kappa,
        dip=dip,
        exit_sample=exit_sample,
        exit_prediction_sample=(exit_sample - shift) if exit_sample is not None else None,
        recovery_sample=recovery_sample,
        recovery_samples=recovery_samples,
        recovery_prediction_samples=(
            recovery_samples - shift if recovery_samples is not None else None
        ),
        recovery_rate=1.0 if recovery_sample is not None else 0.0,
        damage_ref=damage_ref,
        damage_full=damage_full,
        attain_abs=attain_abs,
        aue=aue,
    )


def evaluate_detections(
    alarms: Sequence[int],
    episodes: Sequence[EpisodeSpec],
    n_samples: int,
    delta_max: int,
    delta_pre: int = 0,
    shift: int = 0,
) -> DetectionMetrics:
    clean = sorted({int(a) for a in alarms if 0 <= int(a) < n_samples})
    windows = [
        (
            max(0, ep.start + shift - delta_pre),
            min(n_samples - 1, ep.end + shift + delta_max),
            ep,
        )
        for ep in episodes
    ]
    first_alarm: dict[int, int] = {}
    tp_alarms = 0
    fp_alarms = 0
    for alarm in clean:
        matched = None
        for idx, (lo, hi, _) in enumerate(windows):
            if lo <= alarm <= hi:
                matched = idx
                break
        if matched is None:
            fp_alarms += 1
        else:
            tp_alarms += 1
            first_alarm.setdefault(matched, alarm)

    n_episodes = len(episodes)
    detected = len(first_alarm)
    fn_episodes = n_episodes - detected
    alarm_precision = (
        tp_alarms / (tp_alarms + fp_alarms) if (tp_alarms + fp_alarms) else 0.0
    )
    episode_recall = detected / n_episodes if n_episodes else None
    if n_episodes == 0:
        f1 = None
    elif (alarm_precision + episode_recall) > 0:
        f1 = (
            2
            * alarm_precision
            * episode_recall
            / (alarm_precision + episode_recall)
        )
    else:
        f1 = 0.0

    def _ndt(anchor: str) -> float | None:
        if not first_alarm or delta_max <= 0:
            return None
        values = []
        for idx, alarm in first_alarm.items():
            ep = windows[idx][2]
            base = (ep.start if anchor == "start" else ep.end) + shift
            values.append(min(1.0, max(0.0, (alarm - base) / delta_max)))
        return float(np.mean(values)) if values else None

    def _raw_delay(anchor: str) -> float | None:
        if not first_alarm:
            return None
        values = []
        for idx, alarm in first_alarm.items():
            ep = windows[idx][2]
            base = ep.start if anchor == "start" else ep.end
            values.append(alarm - base)
        return float(np.mean(values)) if values else None

    return DetectionMetrics(
        n_episodes=n_episodes,
        tp_alarms=tp_alarms,
        fp_alarms=fp_alarms,
        fn_episodes=fn_episodes,
        detected_episodes=detected,
        alarm_precision=alarm_precision,
        episode_recall=episode_recall,
        f1=f1,
        ndt_ds=_ndt("start"),
        ndt_de=_ndt("end"),
        mean_delay_from_ds=_raw_delay("start"),
        mean_delay_from_de=_raw_delay("end"),
        alarm_rate_per_1k=len(clean) / n_samples * 1_000 if n_samples else 0.0,
        far_per_1k=fp_alarms / n_samples * 1_000 if n_samples else 0.0,
    )
