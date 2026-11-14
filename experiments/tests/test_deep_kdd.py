from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.deep_kdd import cmd_report
from experiments.driftlab.deep.signals import SIGNAL_NAMES


def _write_cell(
    root: Path,
    name: str,
    values: dict[str, np.ndarray],
    error_rate: np.ndarray,
    onset: int = 500,
    window_size: int = 500,
) -> None:
    path = root / "Electricity" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    ends = np.asarray([100, 200, 300, 400, 500, 600, 700, 800, 900, 1_000])
    payload = {
        "signal_names": np.asarray(SIGNAL_NAMES, dtype=str),
        "onset": np.asarray(onset, dtype=np.int64),
        "width": np.asarray(0, dtype=np.int64),
        "n_samples": np.asarray(1_200, dtype=np.int64),
        "window_size": np.asarray(window_size, dtype=np.int64),
        "stride": np.asarray(100, dtype=np.int64),
        "window_start_index": ends - 100,
        "window_end_index": ends,
        "error_rate": error_rate,
    }
    payload.update(values)
    np.savez_compressed(path, **payload)


def _namespace(signals: Path, out: Path) -> argparse.Namespace:
    return argparse.Namespace(
        signals=str(signals),
        out=str(out),
        horizon=500,
        min_auc=0.75,
        min_gain=0.10,
    )


def test_report_gate_passes_on_a_separating_signal(tmp_path: Path):
    signals = tmp_path / "signals"
    out = tmp_path / "report"
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float64)
    values = {
        name: np.full(10, 0.5, dtype=np.float64) for name in SIGNAL_NAMES
    }
    values["ssl_loss"] = labels
    _write_cell(
        signals,
        "label_swap__abrupt__t00",
        values,
        error_rate=np.full(10, 0.5, dtype=np.float64),
    )
    assert cmd_report(_namespace(signals, out)) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["gate"]["pass"] is True
    assert summary["gate"]["best_signal"] == "ssl_loss"
    assert summary["gate"]["best_auc"] == 1.0
    assert summary["gate"]["error_rate_auc"] == 0.5


def test_report_gate_fails_when_every_signal_is_flat(tmp_path: Path):
    signals = tmp_path / "signals"
    out = tmp_path / "report"
    values = {
        name: np.full(10, 0.5, dtype=np.float64) for name in SIGNAL_NAMES
    }
    _write_cell(
        signals,
        "feature_filtering__gradual__t00",
        values,
        error_rate=np.full(10, 0.5, dtype=np.float64),
    )
    assert cmd_report(_namespace(signals, out)) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["gate"]["pass"] is False
    assert summary["gate"]["best_auc"] == 0.5
