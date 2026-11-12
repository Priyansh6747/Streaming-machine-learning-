"""Paired statistics and Nemenyi critical differences."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats

# q_alpha=0.05 for k=2..10, two-tailed Nemenyi test.
_Q05 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}


def _clean_pair(a: Sequence[float], b: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask], right[mask]


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict:
    left, right = _clean_pair(a, b)
    if len(left) < 2 or np.allclose(left, right):
        return {"n": int(len(left)), "statistic": None, "p_value": None, "effect": None}
    result = stats.wilcoxon(left, right, zero_method="wilcox", alternative="two-sided")
    diff = left - right
    ranks = stats.rankdata(np.abs(diff))
    positive = float(ranks[diff > 0].sum())
    negative = float(ranks[diff < 0].sum())
    total = positive + negative
    effect = (positive - negative) / total if total else 0.0
    return {
        "n": int(len(left)),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "effect": float(effect),
    }


def friedman(matrix: Sequence[Sequence[float]]) -> dict:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 3:
        return {"n_problems": int(values.shape[0]) if values.ndim == 2 else 0, "statistic": None, "p_value": None}
    result = stats.friedmanchisquare(*[values[:, j] for j in range(values.shape[1])])
    return {
        "n_problems": int(values.shape[0]),
        "n_methods": int(values.shape[1]),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def average_ranks(matrix: Sequence[Sequence[float]], higher_is_better: bool = False) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if not higher_is_better:
        values = -values
    ranks = np.vstack([stats.rankdata(row, method="average") for row in values])
    return ranks.mean(axis=0)


def nemenyi_cd(n_methods: int, n_problems: int, alpha: float = 0.05) -> float | None:
    if alpha != 0.05 or n_methods not in _Q05 or n_problems <= 0:
        return None
    return _Q05[n_methods] * np.sqrt(n_methods * (n_methods + 1) / (6 * n_problems))


def nemenyi_pairs(
    ranks: Sequence[float], n_problems: int, alpha: float = 0.05
) -> dict:
    ranks = np.asarray(ranks, dtype=float)
    cd = nemenyi_cd(len(ranks), n_problems, alpha=alpha)
    pairs = []
    if cd is not None:
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                if abs(ranks[i] - ranks[j]) > cd:
                    pairs.append({"i": i, "j": j, "diff": float(abs(ranks[i] - ranks[j]))})
    return {"cd": cd, "significant_pairs": pairs}


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    clean = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(clean) < 2:
        return {"mean": float(clean[0]) if len(clean) else None, "low": None, "high": None, "n": len(clean)}
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(n_boot, len(clean)), replace=True)
    stats_boot = np.apply_along_axis(statistic, 1, samples)
    return {
        "mean": float(statistic(clean)),
        "low": float(np.quantile(stats_boot, alpha / 2)),
        "high": float(np.quantile(stats_boot, 1 - alpha / 2)),
        "n": int(len(clean)),
    }
