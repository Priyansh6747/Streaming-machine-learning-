"""Precompute DF-H Stage-1 signals for the KDD'26 benchmark.

The deep encoder runs in the main environment (torch); ``kdd_env`` only ever
loads the resulting ``.npz`` files with NumPy. One ``.npz`` is written per
(dataset, drift type, transition, trial) cell.

Usage
-----
    uv run python -m experiments.deep_kdd pretrain \
        --datasets Covtype,Electricity

    uv run python -m experiments.deep_kdd prepare-signals \
        --datasets Covtype,Electricity --trials 5 \
        --out experiments/data/deep_signals/pilot

    uv run python -m experiments.deep_kdd report \
        --signals experiments/data/deep_signals/pilot \
        --out experiments/results/e18
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from .driftlab.deep.signals import SIGNAL_NAMES, DeepSignalMonitor, SignalConfig
from .driftlab.fusion import calibrate_threshold
from .kdd_bench import (
    DEFAULT_DATA_DIR,
    DRIFT_TYPES,
    REPO,
    TRANSITIONS,
    _load_npz,
    _resolve_datasets,
    build_trial_stream,
)

DEFAULT_SIGNALS_DIR = REPO / "experiments" / "data" / "deep_signals"
DEFAULT_CHECKPOINT_DIR = REPO / "experiments" / "checkpoints" / "deep"
DEFAULT_REPORT_DIR = REPO / "experiments" / "results" / "e18"

ENCODER_CONFIG: dict[str, Any] = {
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "ffn": 128,
    "n_heads_bank": 4,
    "n_bins": 8,
    "dropout": 0.0,
    "block_size": 32,
    "batch_size": 128,
    "ring_size": 4_096,
    "reservoir_size": 8_192,
    "mask_fraction": 0.15,
    "device": "auto",
}


def _dataset(name: str, data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    return _load_npz(name, data_dir)


def _encoder(
    dataset: str,
    X: np.ndarray,
    y: np.ndarray,
    checkpoint_dir: Path,
    device: str,
    epochs: int,
    reference_size: int,
    force: bool,
):
    from .driftlab.deep.train import DriftFormerConfig, DriftFormerLearner

    path = checkpoint_dir / f"{dataset}.pt"
    config = DriftFormerConfig.from_dict({**ENCODER_CONFIG, "device": device})
    learner = DriftFormerLearner(
        f"{dataset}_encoder",
        list(range(X.shape[1])),
        n_classes=int(y.max()) + 1,
        config=config,
        seed=12,
    )
    if path.exists() and not force:
        learner.load(path)
        return learner

    rng = np.random.default_rng(12)
    size = min(int(reference_size), len(X))
    if size < len(X):
        indices = rng.choice(len(X), size=size, replace=False)
        reference = X[indices]
    else:
        reference = X
    print(f"  pretraining {dataset} encoder on {len(reference)} rows", flush=True)
    learner.pretrain_ssl(reference, epochs=int(epochs))
    path.parent.mkdir(parents=True, exist_ok=True)
    learner.save(path)
    print(f"  saved {path}", flush=True)
    return learner


def _task(
    dataset: str,
    drift_type: str,
    transition: str,
    trial: int,
    seed: int,
    max_instances: int,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "drift_type": drift_type,
        "transition": transition,
        "trial": trial,
        "seed": seed,
        "max_instances": max_instances,
    }


def _window_error_rate(
    trial,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    from river.tree import HoeffdingTreeClassifier

    learner = HoeffdingTreeClassifier()
    errors = np.zeros(len(trial.y), dtype=np.float64)
    for index, (row, label) in enumerate(zip(trial.X, trial.y)):
        x = {feature: float(value) for feature, value in enumerate(row)}
        prediction = learner.predict_one(x)
        errors[index] = 0.0 if prediction is None else float(int(prediction) != int(label))
        learner.learn_one(x, int(label))
    return np.asarray(
        [
            float(np.mean(errors[int(start) : int(end)]))
            for start, end in zip(starts, ends)
        ],
        dtype=np.float64,
    )


def _signal_path(
    out_dir: Path,
    dataset: str,
    drift_type: str,
    transition: str,
    trial: int,
) -> Path:
    return out_dir / dataset / f"{drift_type}__{transition}__t{trial:02d}.npz"


def _write_signals(
    path: Path,
    trial,
    series,
    error_rate: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = series.as_arrays()
    payload.update(
        {
            "signal_names": np.asarray(SIGNAL_NAMES, dtype=str),
            "onset": np.asarray(trial.episode.onset, dtype=np.int64),
            "width": np.asarray(trial.episode.width, dtype=np.int64),
            "n_samples": np.asarray(trial.n_samples, dtype=np.int64),
            "kept_index": trial.index,
            "window_start_index": trial.index[series.starts],
            "window_end_index": trial.index[series.ends - 1],
            "error_rate": error_rate,
            "window_size": np.asarray(series.config.window_size, dtype=np.int64),
            "stride": np.asarray(series.config.stride, dtype=np.int64),
        }
    )
    np.savez_compressed(path, **payload)


def cmd_pretrain(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoints)
    names = _resolve_datasets(args.datasets)
    ok = 0
    for dataset in names:
        path = checkpoint_dir / f"{dataset}.pt"
        if path.exists() and not args.force:
            print(f"{dataset:16s} cached {path}")
            ok += 1
            continue
        try:
            X, y, _ = _dataset(dataset, data_dir)
        except FileNotFoundError:
            print(f"{dataset:16s} missing {data_dir / f'{dataset}.npz'}", file=sys.stderr)
            continue
        _encoder(
            dataset,
            X,
            y,
            checkpoint_dir,
            args.device,
            args.epochs,
            args.reference_size,
            force=True,
        )
        ok += 1
    print(f"ready: {ok}/{len(names)} encoders in {checkpoint_dir}")
    return 0 if ok else 1


def cmd_prepare_signals(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    checkpoint_dir = Path(args.checkpoints)
    names = _resolve_datasets(args.datasets)
    drift_types = [part.strip() for part in args.drift_types.split(",") if part.strip()]
    transitions = [part.strip() for part in args.transitions.split(",") if part.strip()]
    unknown = [name for name in drift_types if name not in DRIFT_TYPES]
    unknown += [name for name in transitions if name not in TRANSITIONS]
    if unknown:
        raise SystemExit(f"unknown drift type or transition: {', '.join(unknown)}")

    signal_config = SignalConfig(
        window_size=args.window_size,
        stride=args.stride,
        reference_fraction=args.reference_fraction,
        mask_fraction=args.mask_fraction,
        adwin_delta=args.adwin_delta,
        ks_reference_size=args.ks_reference_size,
    )
    written = 0
    skipped = 0
    started = time.time()
    for dataset in names:
        try:
            X, y, feature_names = _dataset(dataset, data_dir)
        except FileNotFoundError:
            print(f"{dataset:16s} missing {data_dir / f'{dataset}.npz'}", file=sys.stderr)
            continue
        learner = _encoder(
            dataset,
            X,
            y,
            checkpoint_dir,
            args.device,
            args.epochs,
            args.reference_size,
            force=args.retrain,
        )
        monitor = DeepSignalMonitor(signal_config)
        for drift_type in drift_types:
            for transition in transitions:
                for trial in range(args.trials):
                    path = _signal_path(out_dir, dataset, drift_type, transition, trial)
                    if path.exists() and not args.force:
                        skipped += 1
                        continue
                    task = _task(
                        dataset,
                        drift_type,
                        transition,
                        trial,
                        args.seed,
                        args.max_instances,
                    )
                    trial_stream = build_trial_stream(task, (X, y, feature_names))
                    series = monitor.extract(learner, trial_stream.X)
                    error_rate = _window_error_rate(
                        trial_stream,
                        series.starts,
                        series.ends,
                    )
                    _write_signals(path, trial_stream, series, error_rate)
                    written += 1
                    if written % 5 == 0:
                        elapsed = time.time() - started
                        print(
                            f"[{written}] {dataset} {drift_type}/{transition} "
                            f"trial {trial} ({elapsed:.0f}s)",
                            flush=True,
                        )
    meta = {
        "protocol": "DF-H Stage-1 signal precompute",
        "datasets": names,
        "drift_types": drift_types,
        "transitions": transitions,
        "trials": args.trials,
        "max_instances": args.max_instances,
        "seed": args.seed,
        "signal_config": signal_config.as_dict(),
        "encoder_config": ENCODER_CONFIG,
        "epochs": args.epochs,
        "reference_size": args.reference_size,
        "written": written,
        "skipped": skipped,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": " ".join(sys.argv),
    }
    try:
        meta["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        meta["git_sha"] = None
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"wrote {written} signal files, skipped {skipped}, under {out_dir}")
    return 0


def _auc(labels: np.ndarray, values: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    if not np.all(np.isfinite(values)):
        return None
    try:
        return float(roc_auc_score(labels, values))
    except ValueError:
        return None


def cmd_report(args: argparse.Namespace) -> int:
    signals_dir = Path(args.signals)
    out_dir = Path(args.out)
    paths = sorted(signals_dir.rglob("*.npz"))
    if not paths:
        raise SystemExit(f"no signal files under {signals_dir}")

    per_cell: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    by_dataset: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_drift: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_transition: dict[tuple[str, str], list[float]] = defaultdict(list)
    overall: dict[str, list[float]] = defaultdict(list)
    skipped = 0

    for path in paths:
        dataset = path.parent.name
        pieces = path.stem.split("__")
        if len(pieces) != 3:
            skipped += 1
            continue
        drift_type, transition, _trial = pieces
        with np.load(path) as payload:
            horizon = int(args.horizon or int(payload["window_size"]))
            onset = int(payload["onset"])
            width = int(payload["width"])
            window_start = payload["window_start_index"]
            window_end = payload["window_end_index"]
            # Causal detection label: the window contains post-onset data or ends
            # within `horizon` of the drift end. A monitor cannot see a future
            # change, so a "next W" label would test precognition, not detection.
            labels = (window_end >= onset) & (
                window_start <= onset + width + horizon
            )
            labels = labels.astype(np.int64)
            if len(np.unique(labels)) < 2:
                skipped += 1
                continue
            names = [str(name) for name in payload["signal_names"]]
            values = {name: payload[name] for name in names}
            values["error_rate"] = payload["error_rate"]
        for name, series in values.items():
            auc = _auc(labels, series)
            if auc is None:
                continue
            per_cell[(dataset, drift_type, transition, name)].append(auc)
            by_dataset[(dataset, name)].append(auc)
            by_drift[(drift_type, name)].append(auc)
            by_transition[(transition, name)].append(auc)
            overall[name].append(auc)

    if not overall:
        raise SystemExit("no signal had both label classes; nothing to report")

    def aggregate(groups: dict[tuple, list[float]]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for key, values in sorted(groups.items()):
            if len(key) == 2:
                left, right = key
                result.setdefault(str(left), {})[str(right)] = float(np.mean(values))
            else:
                dataset, drift_type, transition, name = key
                cell = f"{dataset}/{drift_type}/{transition}"
                result.setdefault(cell, {})[str(name)] = float(np.mean(values))
        return result

    overall_means = {name: float(np.mean(values)) for name, values in overall.items()}
    deep_names = [
        name
        for name in SIGNAL_NAMES
        if name != "feature_ks" and name in overall_means
    ]
    baseline = overall_means.get("error_rate")
    best = max(deep_names, key=lambda name: overall_means[name]) if deep_names else None
    best_auc = overall_means.get(best) if best else None
    passed = (
        best_auc is not None
        and baseline is not None
        and best_auc >= float(args.min_auc)
        and (best_auc - baseline) >= float(args.min_gain)
    )
    summary = {
        "gate": {
            "pass": bool(passed),
            "best_signal": best,
            "best_auc": best_auc,
            "error_rate_auc": baseline,
            "min_auc": float(args.min_auc),
            "min_gain": float(args.min_gain),
            "horizon": int(args.horizon) if args.horizon else None,
            "label": (
                "causal detection window: positive if the window overlaps the "
                "drift or ends within horizon of its end"
            ),
            "note": (
                "fixed signals, no tuning; trials are held out, but the encoder is "
                "pretrained on a fixed unlabeled reference sample from each dataset. "
                "v2 uses leave-one-dataset-out."
            ),
        },
        "overall": overall_means,
        "by_dataset": aggregate(by_dataset),
        "by_drift_type": aggregate(by_drift),
        "by_transition": aggregate(by_transition),
        "by_cell": aggregate(per_cell),
        "n_files": len(paths),
        "n_skipped": skipped,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"{'signal':24s}{'AUC':>8s}{'vs error':>10s}")
    for name, value in sorted(overall_means.items(), key=lambda item: item[1], reverse=True):
        delta = value - baseline if baseline is not None else float("nan")
        print(f"{name:24s}{value:8.3f}{delta:+10.3f}")
    print(
        f"\ngate: {'PASS' if passed else 'FAIL'} "
        f"(best={best} {best_auc:.3f}, error={baseline:.3f})"
        if best_auc is not None and baseline is not None
        else f"\ngate: FAIL (incomplete AUC table)"
    )
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    """FAR-calibrated detection delay, using the saved pre-drift windows."""
    signals_dir = Path(args.signals)
    out_dir = Path(args.out)
    paths = sorted(signals_dir.rglob("*.npz"))
    if not paths:
        raise SystemExit(f"no signal files under {signals_dir}")

    trials: list[dict[str, Any]] = []
    pooled: dict[tuple[str, str], list[float]] = defaultdict(list)
    strides: dict[str, int] = {}
    for path in paths:
        dataset = path.parent.name
        pieces = path.stem.split("__")
        if len(pieces) != 3:
            continue
        drift_type, transition, _trial = pieces
        with np.load(path) as payload:
            if "ssl_loss_z" not in payload.files:
                continue
            onset = int(payload["onset"])
            ends = payload["window_end_index"].astype(float)
            stride = int(payload["stride"]) if "stride" in payload.files else 100
            names = [str(name) for name in payload["signal_names"]]
            scores = {name: payload[f"{name}_z"].astype(float) for name in names}
            error_rate = payload["error_rate"].astype(float)
            reference = max(1, int(np.ceil(len(ends) * 0.2)))
            ref_values = error_rate[:reference]
            median = float(np.median(ref_values))
            mad = float(np.median(np.abs(ref_values - median))) * 1.4826
            if mad < 1e-6:
                mad = float(np.std(ref_values))
            scores["error_rate"] = (error_rate - median) / max(mad, 1e-6)
            first = int(np.where(ends >= onset)[0][0]) if np.any(ends >= onset) else None
            if first is None:
                continue
            for name, series in scores.items():
                pooled[(dataset, name)].extend(series[:first].tolist())
            trials.append(
                {
                    "dataset": dataset,
                    "drift_type": drift_type,
                    "transition": transition,
                    "onset": onset,
                    "width": int(payload["width"]),
                    "ends": ends,
                    "stride": stride,
                    "scores": scores,
                    "first": first,
                }
            )
            strides[dataset] = stride

    thresholds = {
        key: calibrate_threshold(
            values,
            args.far,
            samples_per_window=strides.get(key[0], 100),
        )
        for key, values in pooled.items()
    }
    grouped: dict[tuple[str, str, str, str], list[dict[str, float]]] = defaultdict(list)
    for trial in trials:
        dataset = trial["dataset"]
        ends = trial["ends"]
        onset = trial["onset"]
        first = trial["first"]
        stride = trial["stride"]
        for name, series in trial["scores"].items():
            threshold = thresholds[(dataset, name)]
            false_alarms = int((series[:first] >= threshold).sum())
            hits = np.where(
                (np.arange(len(series)) >= first) & (series >= threshold)
            )[0]
            delay = None
            if len(hits):
                value = float(ends[hits[0]] - onset)
                if value <= args.budget:
                    delay = value
            grouped[(dataset, trial["drift_type"], trial["transition"], name)].append(
                {
                    "delay": delay,
                    "far": false_alarms / max(1, first * stride) * 1000.0,
                }
            )

    rows = []
    for key, values in sorted(grouped.items()):
        dataset, drift_type, transition, name = key
        delays = [value["delay"] for value in values if value["delay"] is not None]
        rows.append(
            {
                "dataset": dataset,
                "drift_type": drift_type,
                "transition": transition,
                "signal": name,
                "threshold": thresholds[(dataset, name)],
                "n": len(values),
                "hits": len(delays),
                "hit_rate": len(delays) / max(1, len(values)),
                "median_delay": float(np.median(delays)) if delays else None,
                "p90_delay": float(np.percentile(delays, 90)) if delays else None,
                "far_per_1k": float(np.mean([value["far"] for value in values])),
            }
        )

    payload = {
        "far_target_per_1k": args.far,
        "delay_budget": args.budget,
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "detection.json").write_text(json.dumps(payload, indent=2, default=str))
    with (out_dir / "detection.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(
        f"{'dataset':14s}{'drift/transition':30s}{'signal':22s}"
        f"{'hit%':>7s}{'med':>8s}{'p90':>8s}{'FAR':>8s}"
    )
    for row in rows:
        if row["drift_type"] not in {"feature_permutation", "feature_filtering"}:
            continue
        if row["signal"] == "internal_adwin":
            continue
        delay = f"{row['median_delay']:.0f}" if row["median_delay"] is not None else "-"
        p90 = f"{row['p90_delay']:.0f}" if row["p90_delay"] is not None else "-"
        print(
            f"{row['dataset']:14s}{row['drift_type'] + '/' + row['transition']:30s}"
            f"{row['signal']:22s}{row['hit_rate'] * 100:7.1f}"
            f"{delay:>8s}{p90:>8s}{row['far_per_1k']:8.3f}"
        )
    print(f"wrote {out_dir / 'detection.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DF-H deep-signal KDD precompute")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain", help="cache one SSL encoder per dataset")
    pretrain.add_argument("--datasets", default="all")
    pretrain.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    pretrain.add_argument("--checkpoints", default=str(DEFAULT_CHECKPOINT_DIR))
    pretrain.add_argument("--reference-size", type=int, default=20_000)
    pretrain.add_argument("--epochs", type=int, default=1)
    pretrain.add_argument("--device", default="auto")
    pretrain.add_argument("--force", action="store_true")
    pretrain.set_defaults(func=cmd_pretrain)

    prepare = subparsers.add_parser("prepare-signals", help="extract signals per KDD trial")
    prepare.add_argument("--datasets", default="Covtype,Electricity")
    prepare.add_argument("--drift-types", default=",".join(DRIFT_TYPES))
    prepare.add_argument("--transitions", default=",".join(TRANSITIONS))
    prepare.add_argument("--trials", type=int, default=5)
    prepare.add_argument("--max-instances", type=int, default=100_000)
    prepare.add_argument("--seed", type=int, default=12)
    prepare.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    prepare.add_argument("--checkpoints", default=str(DEFAULT_CHECKPOINT_DIR))
    prepare.add_argument("--out", default=str(DEFAULT_SIGNALS_DIR / "pilot"))
    prepare.add_argument("--reference-size", type=int, default=20_000)
    prepare.add_argument("--epochs", type=int, default=1)
    prepare.add_argument("--device", default="auto")
    prepare.add_argument("--window-size", type=int, default=500)
    prepare.add_argument("--stride", type=int, default=100)
    prepare.add_argument("--reference-fraction", type=float, default=0.2)
    prepare.add_argument("--mask-fraction", type=float, default=0.15)
    prepare.add_argument("--adwin-delta", type=float, default=0.002)
    prepare.add_argument("--ks-reference-size", type=int, default=2_000)
    prepare.add_argument("--retrain", action="store_true", help="ignore cached encoders")
    prepare.add_argument("--force", action="store_true", help="overwrite signal files")
    prepare.set_defaults(func=cmd_prepare_signals)

    report = subparsers.add_parser("report", help="AUC table and go/no-go gate")
    report.add_argument("--signals", default=str(DEFAULT_SIGNALS_DIR / "pilot"))
    report.add_argument("--out", default=str(DEFAULT_REPORT_DIR))
    report.add_argument("--horizon", type=int, default=None)
    report.add_argument("--min-auc", type=float, default=0.75)
    report.add_argument("--min-gain", type=float, default=0.10)
    report.set_defaults(func=cmd_report)

    detect = subparsers.add_parser(
        "detect", help="FAR-calibrated detection delay from saved signals"
    )
    detect.add_argument("--signals", default=str(DEFAULT_SIGNALS_DIR / "pilot"))
    detect.add_argument("--out", default=str(DEFAULT_REPORT_DIR))
    detect.add_argument("--far", type=float, default=0.1)
    detect.add_argument("--budget", type=int, default=2_500)
    detect.set_defaults(func=cmd_detect)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
