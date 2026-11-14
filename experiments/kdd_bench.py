"""Parallel reproduction of the KDD'26 drift-detection benchmark.

Reference
---------
Cerqueira, Gomes, Heyden, Pfahringer & Bifet (2026), "A Framework for
Evaluating and Benchmarking Concept Drift Detection Methods", KDD '26,
pp. 8671-8682.

Protocol (paper + public code)
------------------------------
* 7 real datasets, shuffled per Monte Carlo trial, capped at 100k rows.
* 4 injected drift types: class prior, label swap, feature permutation,
  feature filtering.
* abrupt and gradual transitions; 50 trials per (dataset, type, transition).
* a Hoeffding Tree learner and the paper's tuned detector configurations.

Parallelism
-----------
One task is a whole (dataset, drift type, transition, trial) run. The
learner is trained once per task and every detector is updated in the same
pass, so the stream is traversed once instead of once per detector.
Workers are forked and each starts its own CapyMOA/MOA JVM lazily.

Deliberate differences from the public code
-------------------------------------------
* The paper's Algorithm 1 feeds the *error* bit. The public
  ``prequential_workflow.py`` feeds ``int(true == pred)`` (correctness),
  which contradicts MOA's documented ``input`` contract (1 = error).
  ``--signal error`` is the default; ``--signal correct`` reproduces the
  public code exactly.
* Gradual ground truth is anchored at the drift end by default
  (``--gradual-anchor paper``); ``--gradual-anchor code`` uses the public
  code's ``(onset, onset)`` window.
* ``ABCD(X)`` is included by feeding feature vectors to ABCD; the public
  ``3_real.py`` leaves that path commented out.
* STUDD is omitted: the released final config comments it out, and it is a
  label-free meta-detector rather than an error-stream detector.
* Feature-filtering thresholds are the median of the pre-drift rows (the
  paper's definition); the public code hard-codes one full-dataset median
  table.

Usage
-----
    uv sync --extra kdd
    uv run python -m experiments.kdd_bench prepare --download-usp
    uv run python -m experiments.kdd_bench check
    uv run python -m experiments.kdd_bench run --datasets all --trials 50 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO / "experiments" / "data" / "kdd"
DEFAULT_OUT_DIR = REPO / "experiments" / "results" / "kdd_bench"
USP_ZIP_ID = "1JERZnbGGToAEz_3LRV7n2Vz79LiDAEY-"
USP_ZIP_URL = f"https://drive.google.com/uc?id={USP_ZIP_ID}"

DATASETS = {
    "Asfault": {"capymoa": None, "csv": "Asfault.csv"},
    "Covtype": {"capymoa": "CovtypeNorm", "csv": "Covtype.csv"},
    "Electricity": {"capymoa": "Electricity", "csv": "Electricity.csv"},
    "GasSensorArray": {"capymoa": None, "csv": "GasSensorArray.csv"},
    "NOAA": {"capymoa": None, "csv": "NOAA.csv"},
    "Posture": {"capymoa": None, "csv": "Posture.csv"},
    "Rialto": {"capymoa": None, "csv": "Rialto.csv"},
}

MAX_DELAY = {
    "Asfault": 500,
    "Covtype": 2500,
    "Electricity": 2500,
    "GasSensorArray": 1500,
    "NOAA": 1500,
    "Posture": 2500,
    "Rialto": 2500,
}

DRIFT_TYPES = ["class_prior", "label_swap", "feature_permutation", "feature_filtering"]
TRANSITIONS = ["abrupt", "gradual"]

DETECTOR_NAMES = [
    "ABCD",
    "ABCDx",
    "ADWIN",
    "CUSUM",
    "DDM",
    "EWMAChart",
    "GeometricMovingAverage",
    "HDDMAverage",
    "HDDMWeighted",
    "PageHinkley",
    "RDDM",
    "SEED",
    "STEPD",
]

TUNED_PARAMS: dict[str, dict[str, dict[str, Any]]] = {
    "abrupt": {
        "ABCD": {
            "delta_drift": 0.02,
            "delta_warn": 0.01,
            "encoding_factor": 0.7,
            "model_id": "kpca",
            "num_splits": 100,
        },
        "ADWIN": {"delta": 0.01},
        "CUSUM": {"delta": 0.01, "lambda_": 50, "min_n_instances": 2000},
        "DDM": {"min_n_instances": 2000, "out_control_level": 2.25},
        "EWMAChart": {"lambda_": 0.001, "min_n_instances": 300},
        "GeometricMovingAverage": {
            "alpha": 0.5,
            "lambda_": 0.1,
            "min_n_instances": 300,
        },
        "HDDMAverage": {"drift_confidence": 0.005, "test_type": "Two-sided"},
        "HDDMWeighted": {
            "drift_confidence": 0.001,
            "lambda_": 0.01,
            "test_type": "One-sided",
        },
        "PageHinkley": {
            "alpha": 0.995,
            "delta": 0.005,
            "lambda_": 30,
            "min_n_instances": 1000,
        },
        "RDDM": {"drift_level": 1.9, "min_n_instances": 2000},
        "SEED": {
            "alpha": 0.6,
            "block_size": 32,
            "delta": 0.01,
            "epsilon_prime": 0.01,
        },
        "STEPD": {"alpha_drift": 0.001, "window_size": 500},
    },
    "gradual": {
        "ABCD": {
            "delta_drift": 0.02,
            "delta_warn": 0.01,
            "encoding_factor": 0.7,
            "model_id": "kpca",
            "num_splits": 100,
        },
        "ADWIN": {"delta": 0.01},
        "CUSUM": {"delta": 0.001, "lambda_": 100, "min_n_instances": 30},
        "DDM": {"min_n_instances": 100, "out_control_level": 1.75},
        "EWMAChart": {"lambda_": 0.002, "min_n_instances": 500},
        "GeometricMovingAverage": {
            "alpha": 0.1,
            "lambda_": 0.1,
            "min_n_instances": 100,
        },
        "HDDMAverage": {"drift_confidence": 0.002, "test_type": "Two-sided"},
        "HDDMWeighted": {
            "drift_confidence": 0.005,
            "lambda_": 0.1,
            "test_type": "Two-sided",
        },
        "PageHinkley": {
            "alpha": 0.999,
            "delta": 0.002,
            "lambda_": 50,
            "min_n_instances": 1000,
        },
        "RDDM": {"drift_level": 1.5, "min_n_instances": 30},
        "SEED": {
            "alpha": 0.7,
            "block_size": 50,
            "delta": 0.05,
            "epsilon_prime": 0.0075,
        },
        "STEPD": {"alpha_drift": 0.001, "window_size": 500},
    },
}

_DATA: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}


@dataclass(frozen=True)
class DriftEpisode:
    onset: int
    width: int
    selected_label: int
    swap_label: int
    permutation: tuple[int, ...]
    filter_index: int
    filter_threshold: float


def should_apply_drift(index: int, episode: DriftEpisode, rng: Any) -> bool:
    if index >= episode.onset + episode.width:
        return True
    if index < episode.onset:
        return False
    if episode.width <= 0:
        return True
    progress = (index - episode.onset) / episode.width
    return bool(rng.random() < progress)


def transform_instance(
    x: np.ndarray,
    y: int,
    drift_type: str,
    episode: DriftEpisode,
    rng: Any,
    skip_probability: float = 0.75,
) -> tuple[np.ndarray, int] | None:
    if drift_type == "class_prior":
        if y == episode.selected_label and rng.random() < skip_probability:
            return None
        return x, y
    if drift_type == "label_swap":
        if y == episode.selected_label:
            return x, episode.swap_label
        return x, y
    if drift_type == "feature_permutation":
        return x[np.asarray(episode.permutation)], y
    if drift_type == "feature_filtering":
        if x[episode.filter_index] > episode.filter_threshold:
            return None
        return x, y
    raise ValueError(f"unknown drift type: {drift_type!r}")


def evaluate_alarms(
    alarms: Iterable[int],
    onset: int,
    end: int,
    max_delay: int,
    n_samples: int,
) -> dict[str, Any]:
    clean = sorted(int(a) for a in alarms)
    tp = sum(1 for a in clean if onset <= a <= end + max_delay)
    fp = len(clean) - tp
    detected = tp > 0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = 1.0 if detected else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    first = min((a for a in clean if onset <= a <= end + max_delay), default=None)
    mdt = float(first - onset) if first is not None else float("nan")
    ndt = mdt / max_delay if max_delay > 0 and math.isfinite(mdt) else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": 0 if detected else 1,
        "precision": precision,
        "recall": recall,
        "episode_recall": recall,
        "f1": f1,
        "mdt": mdt,
        "ndt": ndt,
        "far": fp / n_samples * 1000.0,
        "ar": len(clean) / n_samples * 1000.0,
        "n_alarms": len(clean),
        "n_episodes": 1,
    }


def _load_npz(name: str, data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(data_dir / f"{name}.npz", allow_pickle=False) as payload:
        X = payload["X"].astype(np.float32, copy=False)
        y = payload["y"].astype(np.int32, copy=False)
        if "feature_names" in payload:
            names = [str(value) for value in payload["feature_names"]]
        else:
            names = [f"f{index}" for index in range(X.shape[1])]
    return X, y, names


def _load_data(datasets: Iterable[str], data_dir: Path) -> None:
    for name in datasets:
        if name not in _DATA:
            _DATA[name] = _load_npz(name, data_dir)


def _fit_episode(
    X: np.ndarray,
    y: np.ndarray,
    drift_type: str,
    transition: str,
    max_delay: int,
    rng: np.random.Generator,
) -> DriftEpisode:
    onset_fraction = float(rng.uniform(0.5, 0.8))
    onset = int(X.shape[0] * onset_fraction)
    width = max_delay if transition == "gradual" else 0
    labels = np.unique(y)
    selected = int(rng.choice(labels))
    alternatives = labels[labels != selected]
    swap = int(rng.choice(alternatives)) if len(alternatives) else selected
    permutation = tuple(int(i) for i in rng.permutation(X.shape[1]))
    filter_index = int(rng.integers(0, X.shape[1]))
    filter_threshold = float(np.median(X[: max(1, onset), filter_index]))
    return DriftEpisode(
        onset=onset,
        width=width,
        selected_label=selected,
        swap_label=swap,
        permutation=permutation,
        filter_index=filter_index,
        filter_threshold=filter_threshold,
    )


def _worker_init(data_dir: str, datasets: list[str], jvm_xmx: str) -> None:
    os.environ["JAVA_TOOL_OPTIONS"] = f"-Xmx{jvm_xmx}"
    import capymoa  # noqa: F401

    if not _DATA:
        _load_data(datasets, Path(data_dir))


def _build_detectors(
    transition: str,
    params_mode: str,
    detector_names: list[str],
):
    from capymoa.drift import detectors as library

    built = []
    for name in detector_names:
        source = "ABCD" if name == "ABCDx" else name
        params = TUNED_PARAMS[transition][source] if params_mode == "hypertuned" else {}
        detector = getattr(library, source)(**params)
        built.append((name, detector, name == "ABCDx"))
    return built


def _run_trial(task: dict[str, Any]) -> list[dict[str, Any]]:
    from capymoa.classifier import HoeffdingTree
    from capymoa.core import LabeledInstance
    from capymoa.stream import NumpyStream

    dataset = task["dataset"]
    drift_type = task["drift_type"]
    transition = task["transition"]
    trial = task["trial"]
    signal = task["signal"]
    gradual_anchor = task["gradual_anchor"]
    params_mode = task["params_mode"]
    max_instances = task["max_instances"]
    detector_names = task["detectors"]

    X, y, feature_names = _DATA[dataset]
    rng = np.random.default_rng([task["seed"], trial, DRIFT_TYPES.index(drift_type), TRANSITIONS.index(transition)])
    order = rng.permutation(len(y))
    X = X[order]
    y = y[order]
    if len(y) > max_instances:
        X = X[:max_instances]
        y = y[:max_instances]
    n_samples = len(y)

    episode = _fit_episode(X, y, drift_type, transition, MAX_DELAY[dataset], rng)
    stream = NumpyStream(
        X=X,
        y=y,
        dataset_name=dataset,
        feature_names=feature_names,
    )
    schema = stream.get_schema()
    learner = HoeffdingTree(schema=schema)
    detectors = _build_detectors(transition, params_mode, detector_names)
    alarms: dict[str, list[int]] = {name: [] for name, _, _ in detectors}
    errors: dict[str, str] = {}
    min_training = int(n_samples * 0.25)

    index = 0
    while stream.has_more_instances() and index < n_samples:
        instance = stream.next_instance()
        if should_apply_drift(index, episode, rng):
            transformed = transform_instance(instance.x, int(instance.y_index), drift_type, episode, rng)
            if transformed is None:
                index += 1
                continue
            x_new, y_new = transformed
            if y_new != int(instance.y_index):
                instance = LabeledInstance.from_array(schema, x_new, y_new)
            elif not np.array_equal(x_new, instance.x):
                instance = LabeledInstance.from_array(schema, x_new, y_new)

        if index > min_training:
            prediction = learner.predict(instance)
            if signal == "error":
                score = float(prediction != instance.y_index)
            else:
                score = float(prediction == instance.y_index)
            for name, detector, monitor_instance in detectors:
                if name in errors:
                    continue
                try:
                    detector.add_element(instance if monitor_instance else score)
                    if detector.detected_change():
                        alarms[name].append(index)
                except Exception as exc:  # noqa: BLE001 - report, do not kill the trial
                    errors[name] = f"{type(exc).__name__}: {exc}"

        learner.train(instance)
        index += 1

    if gradual_anchor == "paper":
        end = episode.onset + episode.width
    else:
        end = episode.onset
    rows = []
    for name, _, _ in detectors:
        metrics = evaluate_alarms(
            alarms[name],
            onset=episode.onset,
            end=end,
            max_delay=MAX_DELAY[dataset],
            n_samples=n_samples,
        )
        rows.append(
            {
                "dataset": dataset,
                "drift_type": drift_type,
                "transition": transition,
                "trial": trial,
                "detector": name,
                "onset": episode.onset,
                "width": episode.width,
                "error": errors.get(name, ""),
                **metrics,
            }
        )
    return rows


def _read_csv_dataset(name: str, path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    header = "infer" if name in {"Covtype", "Electricity"} else None
    frame = pd.read_csv(path, header=header, on_bad_lines="skip")
    if "target" not in frame.columns:
        frame = frame.rename(columns={frame.columns[-1]: "target"})
    if name not in {"Covtype", "Electricity"}:
        frame.columns = [f"F{i}" for i in range(1, frame.shape[1])] + ["target"]
    X = frame.drop(columns="target").to_numpy(dtype=np.float32)
    y = LabelEncoder().fit_transform(frame["target"]).astype(np.int32)
    names = [str(column) for column in frame.columns[:-1]]
    return X, y, names


def _read_capymoa_dataset(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from capymoa.datasets import CovtypeNorm, Electricity

    factory = {"Electricity": Electricity, "Covtype": CovtypeNorm}[name]
    stream = factory()
    schema = stream.get_schema()
    names = [
        str(schema._moa_header.attribute(i).name())
        for i in range(schema.get_num_attributes())
    ]
    X: list[np.ndarray] = []
    y: list[int] = []
    while stream.has_more_instances():
        instance = stream.next_instance()
        X.append(np.asarray(instance.x, dtype=np.float32))
        y.append(int(instance.y_index))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32), names


def _download_usp(data_dir: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("gdown is required: uv sync --extra kdd") from exc
    zip_path = data_dir / "usp_ds.zip"
    if not zip_path.exists():
        print(f"downloading {USP_ZIP_URL}", flush=True)
        gdown.download(USP_ZIP_URL, str(zip_path), quiet=False)
    if not zip_path.exists():
        raise SystemExit(f"download failed; fetch it manually: {USP_ZIP_URL}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)
    _canonicalize_csvs(data_dir)


def _canonicalize_csvs(data_dir: Path) -> None:
    for path in sorted(data_dir.rglob("*.csv")):
        stem = path.stem.lower().replace("_", "").replace("-", "")
        for name in DATASETS:
            normalized = name.lower().replace("_", "").replace("-", "")
            if stem == normalized or stem.startswith(normalized):
                target = data_dir / f"{name}.csv"
                if path.resolve() != target.resolve():
                    path.replace(target)
                break


def _prepare_one(name: str, data_dir: Path, force: bool) -> bool:
    output = data_dir / f"{name}.npz"
    if output.exists() and not force:
        print(f"{name:16s} cached")
        return True
    csv_path = data_dir / f"{name}.csv"
    if csv_path.exists():
        X, y, names = _read_csv_dataset(name, csv_path)
    elif DATASETS[name]["capymoa"]:
        print(f"{name:16s} exporting from CapyMOA", flush=True)
        X, y, names = _read_capymoa_dataset(name)
    else:
        print(f"{name:16s} missing {csv_path.name}; run prepare --download-usp", file=sys.stderr)
        return False
    np.savez_compressed(
        output,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int32),
        feature_names=np.asarray(names, dtype=str),
    )
    print(f"{name:16s} saved {X.shape[0]} rows x {X.shape[1]} features")
    return True


def cmd_prepare(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.download_usp:
        _download_usp(data_dir)
    names = _resolve_datasets(args.datasets)
    ok = 0
    for name in names:
        ok += int(_prepare_one(name, data_dir, args.force))
    print(f"ready: {ok}/{len(names)} datasets in {data_dir}")
    return 0 if ok else 1


def cmd_list(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    for name in DATASETS:
        npz = (data_dir / f"{name}.npz").exists()
        csv_path = (data_dir / f"{name}.csv").exists()
        auto = "yes" if DATASETS[name]["capymoa"] else "no"
        print(
            f"{name:16s} npz={'yes' if npz else 'no ':3s} "
            f"csv={'yes' if csv_path else 'no ':3s} capymoa={auto:3s}"
        )
    return 0


def _resolve_datasets(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(DATASETS)
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown datasets: {', '.join(unknown)}")
    return names


def cmd_check(args: argparse.Namespace) -> int:
    os.environ["JAVA_TOOL_OPTIONS"] = f"-Xmx{args.jvm_xmx}"
    import capymoa
    from capymoa.classifier import HoeffdingTree  # noqa: F401
    from capymoa.drift import detectors as library
    from importlib.metadata import version

    print(f"capymoa {version('capymoa')}")
    print(f"java {os.environ.get('JAVA_TOOL_OPTIONS', '')}")
    failures = 0
    for transition in TRANSITIONS:
        for name in DETECTOR_NAMES:
            source = "ABCD" if name == "ABCDx" else name
            params = TUNED_PARAMS[transition][source] if args.params == "hypertuned" else {}
            try:
                getattr(library, source)(**params)
                print(f"ok   {transition:7s} {name}")
            except Exception as exc:  # noqa: BLE001 - this command is a probe
                failures += 1
                print(f"FAIL {transition:7s} {name}: {type(exc).__name__}: {exc}")
    return 1 if failures else 0


def _read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _task_key(task: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(task["dataset"]),
        str(task["drift_type"]),
        str(task["transition"]),
        int(task["trial"]),
    )


def _write_meta(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["drift_type"], row["transition"], row["detector"])].append(row)
    metrics = ["precision", "recall", "episode_recall", "f1", "mdt", "ndt", "far", "ar"]
    summary = []
    for (dataset, drift_type, transition, detector), items in sorted(grouped.items()):
        record: dict[str, Any] = {
            "dataset": dataset,
            "drift_type": drift_type,
            "transition": transition,
            "detector": detector,
            "n": len(items),
        }
        for metric in metrics:
            values = [
                float(item[metric])
                for item in items
                if item.get(metric) not in (None, "") and math.isfinite(float(item[metric]))
            ]
            record[f"{metric}_mean"] = float(np.mean(values)) if values else None
            record[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(record)
    return summary


def _average_ranks(summary: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy.stats import rankdata

    payload: dict[str, Any] = {"by_transition": {}, "overall": {}}
    for transition in TRANSITIONS:
        by_drift: dict[str, dict[str, float]] = {}
        for drift_type in DRIFT_TYPES:
            subset = [
                row
                for row in summary
                if row["transition"] == transition
                and row["drift_type"] == drift_type
                and row["f1_mean"] is not None
            ]
            ranks: dict[str, list[float]] = defaultdict(list)
            for dataset in sorted({row["dataset"] for row in subset}):
                per_dataset = sorted(
                    [row for row in subset if row["dataset"] == dataset],
                    key=lambda row: row["detector"],
                )
                values = [row["f1_mean"] for row in per_dataset]
                if not values:
                    continue
                scores = rankdata([-value for value in values], method="average")
                for row, score in zip(per_dataset, scores):
                    ranks[row["detector"]].append(float(score))
            by_drift[drift_type] = {
                detector: float(np.mean(values)) for detector, values in ranks.items()
            }
        payload["by_transition"][transition] = by_drift
    overall: dict[str, list[float]] = defaultdict(list)
    for drift_map in payload["by_transition"].values():
        for ranks in drift_map.values():
            for detector, value in ranks.items():
                overall[detector].append(value)
    payload["overall"] = {
        detector: float(np.mean(values)) for detector, values in overall.items()
    }
    return payload


def _print_ranking(ranking: dict[str, Any]) -> None:
    for transition in TRANSITIONS:
        print(f"\nAverage rank, {transition} (lower is better)")
        print(f"{'Detector':24s}" + "".join(f"{drift:>20s}" for drift in DRIFT_TYPES))
        table = ranking["by_transition"][transition]
        detectors = sorted(
            {detector for drift in DRIFT_TYPES for detector in table.get(drift, {})},
            key=lambda detector: ranking["overall"].get(detector, float("inf")),
        )
        for detector in detectors:
            values = [table.get(drift, {}).get(detector) for drift in DRIFT_TYPES]
            cells = "".join(
                f"{value:>20.2f}" if value is not None else f"{'-':>20s}"
                for value in values
            )
            print(f"{detector:24s}{cells}")
    print("\nOverall rank")
    for detector, rank in sorted(ranking["overall"].items(), key=lambda item: item[1]):
        print(f"{detector:24s}{rank:6.2f}")


def cmd_run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = _resolve_datasets(args.datasets)
    available = [name for name in names if (data_dir / f"{name}.npz").exists()]
    missing = [name for name in names if name not in available]
    if missing:
        print(
            f"missing cached datasets: {', '.join(missing)}; "
            f"run: python -m experiments.kdd_bench prepare --download-usp",
            file=sys.stderr,
        )
    if not available:
        return 1

    detectors = [part.strip() for part in args.detectors.split(",") if part.strip()]
    unknown = [name for name in detectors if name not in DETECTOR_NAMES]
    if unknown:
        raise SystemExit(f"unknown detectors: {', '.join(unknown)}")

    drift_types = [part.strip() for part in args.drift_types.split(",") if part.strip()]
    unknown_drift = [name for name in drift_types if name not in DRIFT_TYPES]
    if unknown_drift:
        raise SystemExit(f"unknown drift types: {', '.join(unknown_drift)}")
    transitions = [part.strip() for part in args.transitions.split(",") if part.strip()]
    unknown_transitions = [name for name in transitions if name not in TRANSITIONS]
    if unknown_transitions:
        raise SystemExit(f"unknown transitions: {', '.join(unknown_transitions)}")

    tasks = [
        {
            "dataset": dataset,
            "drift_type": drift_type,
            "transition": transition,
            "trial": trial,
            "signal": args.signal,
            "gradual_anchor": args.gradual_anchor,
            "params_mode": args.params,
            "max_instances": args.max_instances,
            "detectors": detectors,
            "seed": args.seed,
        }
        for dataset in available
        for drift_type in drift_types
        for transition in transitions
        for trial in range(args.trials)
    ]
    if args.limit:
        tasks = tasks[: args.limit]

    trials_path = out_dir / "trials.csv"
    existing = _read_existing(trials_path) if args.resume else []
    completed = {_task_key(row) for row in existing} if args.resume else set()
    pending = [task for task in tasks if _task_key(task) not in completed]
    print(
        f"{len(available)} datasets, {len(tasks)} tasks, {len(pending)} pending, "
        f"detectors={len(detectors)}, workers={args.workers}",
        flush=True,
    )
    if args.dry_run:
        return 0

    _load_data(available, data_dir)
    meta = {
        "protocol": "KDD-2026 drift detection benchmark (reproduction)",
        "datasets": available,
        "missing": missing,
        "drift_types": drift_types,
        "transitions": transitions,
        "trials": args.trials,
        "detectors": detectors,
        "signal": args.signal,
        "gradual_anchor": args.gradual_anchor,
        "params": args.params,
        "max_instances": args.max_instances,
        "seed": args.seed,
        "workers": args.workers,
        "start_method": args.start_method,
        "jvm_xmx": args.jvm_xmx,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": " ".join(sys.argv),
    }
    try:
        meta["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        meta["git_sha"] = None
    try:
        from importlib.metadata import version

        meta["capymoa_version"] = version("capymoa")
    except Exception:
        meta["capymoa_version"] = None
    meta["numpy_version"] = np.__version__
    _write_meta(out_dir / "meta.json", meta)

    fieldnames = [
        "dataset",
        "drift_type",
        "transition",
        "trial",
        "detector",
        "onset",
        "width",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "episode_recall",
        "f1",
        "mdt",
        "ndt",
        "far",
        "ar",
        "n_alarms",
        "n_episodes",
        "error",
    ]
    rows = list(existing)
    mode = "a" if args.resume and trials_path.exists() else "w"
    context = mp.get_context(args.start_method)
    started = time.time()
    done = 0
    failures = 0
    with trials_path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
            handle.flush()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=_worker_init,
            initargs=(str(data_dir), available, args.jvm_xmx),
        ) as executor:
            futures = {executor.submit(_run_trial, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    produced = future.result()
                except Exception as exc:  # noqa: BLE001 - keep the sweep alive
                    failures += 1
                    print(
                        f"task failed {_task_key(task)}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                writer.writerows(produced)
                handle.flush()
                rows.extend(produced)
                done += 1
                if done % 10 == 0 or done == len(pending):
                    elapsed = time.time() - started
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (len(pending) - done) / rate if rate > 0 else float("inf")
                    print(
                        f"[{done}/{len(pending)}] {rate:.2f} trials/s, "
                        f"ETA {eta / 60:.1f} min",
                        flush=True,
                    )

    summary = _aggregate(rows)
    if summary:
        with (out_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        ranking = _average_ranks(summary)
        (out_dir / "ranking.json").write_text(json.dumps(ranking, indent=2))
        _print_ranking(ranking)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["n_rows"] = len(rows)
    meta["n_tasks_completed"] = done
    meta["n_task_failures"] = failures
    _write_meta(out_dir / "meta.json", meta)
    print(f"\nwrote {out_dir}")
    return 0 if failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel KDD'26 drift-detector benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="cache datasets as .npz")
    prepare.add_argument("--datasets", default="all")
    prepare.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    prepare.add_argument("--download-usp", action="store_true")
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=cmd_prepare)

    listing = subparsers.add_parser("list", help="show cached dataset status")
    listing.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    listing.set_defaults(func=cmd_list)

    check = subparsers.add_parser("check", help="construct every detector and report failures")
    check.add_argument("--params", choices=["hypertuned", "default"], default="hypertuned")
    check.add_argument("--jvm-xmx", default="512m")
    check.set_defaults(func=cmd_check)

    run = subparsers.add_parser("run", help="run the Monte Carlo benchmark")
    run.add_argument("--datasets", default="all")
    run.add_argument("--drift-types", default=",".join(DRIFT_TYPES))
    run.add_argument("--transitions", default=",".join(TRANSITIONS))
    run.add_argument("--trials", type=int, default=50)
    run.add_argument("--detectors", default=",".join(DETECTOR_NAMES))
    run.add_argument("--signal", choices=["error", "correct"], default="error")
    run.add_argument("--gradual-anchor", choices=["paper", "code"], default="paper")
    run.add_argument("--params", choices=["hypertuned", "default"], default="hypertuned")
    run.add_argument("--max-instances", type=int, default=100_000)
    run.add_argument("--seed", type=int, default=12)
    run.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    run.add_argument("--start-method", choices=["fork", "spawn"], default="fork")
    run.add_argument("--jvm-xmx", default="512m")
    run.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    run.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--limit", type=int, default=None)
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
