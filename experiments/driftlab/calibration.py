"""Calibration metrics and online temperature scaling."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def ece(y_true: Sequence[int], proba_pos: Sequence[float], n_bins: int = 15) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba_pos, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return float("nan")
    confidence = np.maximum(p, 1.0 - p)
    pred = (p >= 0.5).astype(float)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        members = bins == b
        if members.any():
            total += members.mean() * abs(correct[members].mean() - confidence[members].mean())
    return float(total)


def ece_multiclass(y_true: Sequence[int], proba: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or len(y) == 0:
        return float("nan")
    confidence = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        members = bins == b
        if members.any():
            total += members.mean() * abs(correct[members].mean() - confidence[members].mean())
    return float(total)


def brier(y_true: Sequence[int], proba_pos: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba_pos, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    if not mask.any():
        return float("nan")
    return float(np.mean((p[mask] - y[mask]) ** 2))


def reliability_curve(
    y_true: Sequence[int], proba_pos: Sequence[float], n_bins: int = 15
) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba_pos, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    y, p = y[mask], p[mask]
    confidence = np.maximum(p, 1.0 - p)
    correct = ((p >= 0.5).astype(float) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges) - 1, 0, n_bins - 1)
    centers, accuracy, mean_confidence, counts = [], [], [], []
    for b in range(n_bins):
        members = bins == b
        centers.append(float((edges[b] + edges[b + 1]) / 2))
        counts.append(int(members.sum()))
        accuracy.append(float(correct[members].mean()) if members.any() else float("nan"))
        mean_confidence.append(float(confidence[members].mean()) if members.any() else float("nan"))
    return {
        "centers": centers,
        "accuracy": accuracy,
        "confidence": mean_confidence,
        "counts": counts,
    }


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(
    logits: Sequence[float],
    y_true: Sequence[int],
    grid: Sequence[float] | None = None,
) -> float:
    z = np.asarray(logits, dtype=float)
    y = np.asarray(y_true, dtype=float)
    mask = np.isfinite(z) & np.isfinite(y)
    z, y = z[mask], y[mask]
    if len(z) == 0:
        return 1.0
    grid = np.asarray(grid if grid is not None else np.linspace(0.25, 4.0, 76), dtype=float)
    best_t, best_nll = 1.0, float("inf")
    for temperature in grid:
        p = _sigmoid(z / temperature)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_t, best_nll = float(temperature), float(nll)
    return best_t
