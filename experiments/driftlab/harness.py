"""Config-driven experiment runner."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score

from .artifacts import ArtifactWriter
from .calibration import brier as brier_score
from .calibration import ece as ece_score
from .config import ExperimentConfig, StreamConfig, delta_max_for, window_for
from .detectors import replay_all
from .learners import BaseLearner, build_learner
from .metrics import evaluate_detections, evaluate_window, panel_kappa_ref, rolling_kappa
from .streams import StreamData, build_stream

ORACLE_ID = "oracle_ref"
DEFAULT_HOSTS = {
    "ht": "HoeffdingTreeClassifier",
    "hat": "HoeffdingAdaptiveTreeClassifier",
    "arf": "ARFClassifier",
}


@dataclass
class ModelRun:
    model_id: str
    predictions: np.ndarray
    proba_pos: np.ndarray
    proba_max: np.ndarray
    errors: np.ndarray
    kappa: np.ndarray
    accuracy: float
    full_kappa: float
    ece: float
    brier: float
    runtime_s: float
    model_bytes: int
    samples_per_s: float


def _sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def scale_stream_config(cfg: StreamConfig, limit: int | None) -> StreamConfig:
    if not limit or limit >= cfg.n_samples:
        return cfg
    warm = max(1, int(0.2 * limit))
    position = max(warm + 2, int(0.5 * limit))
    width = max(1, int(cfg.width * limit / cfg.n_samples)) if cfg.width else 0
    period = max(1, int(cfg.period * limit / cfg.n_samples)) if cfg.period else cfg.period
    episodes = cfg.n_episodes
    if cfg.shape == "recurring" and period > 0:
        episodes = max(1, min(cfg.n_episodes, (limit - position) // period))
    return replace(
        cfg,
        n_samples=limit,
        warm_up=warm,
        position=position,
        width=width,
        period=period,
        n_episodes=episodes,
    )


def _finish_run(
    spec: dict[str, Any],
    stream: StreamData,
    learner: BaseLearner,
    pred: np.ndarray,
    proba_pos: np.ndarray,
    proba_max: np.ndarray,
    errors: np.ndarray,
    warm: int,
    rolling_window: int,
    need_proba: bool,
    delay: int,
    t0: float,
) -> ModelRun:
    n = len(stream)
    runtime = time.perf_counter() - t0
    valid = pred[warm:] >= 0
    y_true_post = np.asarray(stream.y[warm:])
    if valid.any():
        accuracy = float(np.mean(pred[warm:][valid] == y_true_post[valid]))
        try:
            full_kappa = float(cohen_kappa_score(y_true_post[valid], pred[warm:][valid]))
        except Exception:
            full_kappa = float("nan")
    else:
        accuracy = float("nan")
        full_kappa = float("nan")

    if delay > 0:
        arrival_y = [stream.y[t - delay] if t >= delay else None for t in range(n)]
        arrival_pred = [
            int(pred[t - delay]) if t >= delay and pred[t - delay] >= 0 else None
            for t in range(n)
        ]
        kappa = rolling_kappa(
            arrival_y, arrival_pred, window=rolling_window, warm_up=warm + delay
        )
    else:
        kappa = rolling_kappa(
            stream.y,
            [None if p < 0 else int(p) for p in pred],
            window=rolling_window,
            warm_up=warm,
        )
    if need_proba:
        proba_mask = np.isfinite(proba_pos[warm:]) & (pred[warm:] >= 0)
        if proba_mask.any():
            ece_value = ece_score(y_true_post[proba_mask], proba_pos[warm:][proba_mask])
            brier_value = brier_score(y_true_post[proba_mask], proba_pos[warm:][proba_mask])
        else:
            ece_value = float("nan")
            brier_value = float("nan")
    else:
        ece_value = float("nan")
        brier_value = float("nan")

    evaluated = max(1, n - warm)
    return ModelRun(
        model_id=str(spec.get("id") or spec.get("type")),
        predictions=pred,
        proba_pos=proba_pos,
        proba_max=proba_max,
        errors=errors,
        kappa=kappa,
        accuracy=accuracy,
        full_kappa=full_kappa,
        ece=float(ece_value),
        brier=float(brier_value),
        runtime_s=runtime,
        model_bytes=learner.model_bytes(),
        samples_per_s=evaluated / runtime if runtime > 0 else float("nan"),
    )


def run_model(
    spec: dict[str, Any],
    stream: StreamData,
    seed: int,
    rolling_window: int,
    need_proba: bool = True,
    delay: int = 0,
) -> ModelRun:
    learner: BaseLearner = build_learner(spec, stream, seed)
    n = len(stream)
    warm = int(stream.meta["warm_up"])
    delay = max(0, int(delay))
    pred = np.full(n, -1, dtype=np.int16)
    proba_pos = np.full(n, np.nan, dtype=np.float32)
    proba_max = np.full(n, np.nan, dtype=np.float32)
    errors = np.full(n, np.nan, dtype=np.float32)

    def record(t: int, y_pred: int | None, proba: dict[int, float] | None) -> None:
        if y_pred is not None:
            pred[t] = int(y_pred)
        if need_proba and proba:
            proba_pos[t] = float(proba.get(1, np.nan))
            proba_max[t] = float(max(proba.values()))

    t0 = time.perf_counter()
    if learner.kind in {"frozen", "scheduled", "oracle"} and delay <= 0:
        learner.fit_initial(stream.X[:warm], stream.y[:warm])
        if learner.kind == "scheduled":
            every = int(getattr(learner, "every", 0))
            boundaries = [t for t in range(warm, n) if every > 0 and t % every == 0]
        elif learner.kind == "oracle":
            boundaries = [ep.start for ep in stream.episodes if warm <= ep.start < n]
        else:
            boundaries = []
        segments: list[tuple[int, int]] = []
        start = warm
        for boundary in boundaries:
            if boundary >= start:
                segments.append((start, boundary))
                start = boundary + 1
        if start < n:
            segments.append((start, n - 1))
        for lo, hi in segments:
            if hi < lo:
                continue
            batch_preds, batch_probas = learner.predict_batch(
                stream.X[lo : hi + 1], return_proba=need_proba
            )
            for offset, t in enumerate(range(lo, hi + 1)):
                record(
                    t,
                    batch_preds[offset],
                    batch_probas[offset] if batch_probas is not None else None,
                )
                y_pred = batch_preds[offset]
                if y_pred is not None:
                    errors[t] = float(int(y_pred) != int(stream.y[t]))
            for t in range(lo, hi + 1):
                learner.learn_one(stream.X[t], stream.y[t], t)
        return _finish_run(
            spec,
            stream,
            learner,
            pred,
            proba_pos,
            proba_max,
            errors,
            warm,
            rolling_window,
            need_proba,
            delay,
            t0,
        )
    if delay <= 0:
        if learner.kind == "driftformer":
            learner.warm_up(stream.X[:warm], stream.y[:warm])
            block = max(1, int(getattr(learner.config, "block_size", 1)))
            for start in range(warm, n, block):
                end = min(n, start + block)
                preds, probas = learner.predict_batch(
                    stream.X[start:end], return_proba=need_proba
                )
                for offset, t in enumerate(range(start, end)):
                    y_true = int(stream.y[t])
                    y_pred = preds[offset] if offset < len(preds) else None
                    record(t, y_pred, probas[offset] if probas is not None else None)
                    if y_pred is not None:
                        errors[t] = float(int(y_pred) != y_true)
                learner.learn_batch(stream.X[start:end], stream.y[start:end], t_start=start)
        else:
            if learner.kind in {"sklearn", "frozen", "scheduled", "oracle"}:
                learner.fit_initial(stream.X[:warm], stream.y[:warm])
            else:
                for t in range(warm):
                    learner.learn_one(stream.X[t], stream.y[t], t)

            for t in range(warm, n):
                x = stream.X[t]
                y_true = int(stream.y[t])
                y_pred = learner.predict_one(x)
                proba = learner.predict_proba_one(x) if need_proba else None
                record(t, y_pred, proba)
                if y_pred is not None:
                    errors[t] = float(int(y_pred) != y_true)
                learner.learn_one(x, y_true, t)
    else:
        if learner.kind == "driftformer":
            learner.warm_up(stream.X[:warm], stream.y[:warm])
            block = max(1, int(getattr(learner.config, "block_size", 1)))
            for start in range(warm, n, block):
                end = min(n, start + block)
                preds, probas = learner.predict_batch(
                    stream.X[start:end], return_proba=need_proba
                )
                for offset, t in enumerate(range(start, end)):
                    record(t, preds[offset], probas[offset] if probas is not None else None)
                for t in range(max(start, warm + delay), end):
                    source = t - delay
                    if source >= warm and pred[source] >= 0:
                        errors[t] = float(pred[source] != stream.y[source])
                train_start = start - delay
                if train_start >= warm:
                    train_end = min(end - delay, n)
                    if train_end > train_start:
                        learner.learn_batch(
                            stream.X[train_start:train_end],
                            stream.y[train_start:train_end],
                            t_start=train_start,
                        )
        else:
            if learner.kind in {"sklearn", "frozen", "scheduled", "oracle"}:
                learner.fit_initial(stream.X[:warm], stream.y[:warm])
            else:
                for t in range(warm):
                    learner.learn_one(stream.X[t], stream.y[t], t)

            for t in range(warm, n):
                x = stream.X[t]
                y_pred = learner.predict_one(x)
                proba = learner.predict_proba_one(x) if need_proba else None
                record(t, y_pred, proba)
                source = t - delay
                if source >= warm:
                    if pred[source] >= 0:
                        errors[t] = float(pred[source] != stream.y[source])
                    learner.learn_one(stream.X[source], stream.y[source], t)

    return _finish_run(
        spec,
        stream,
        learner,
        pred,
        proba_pos,
        proba_max,
        errors,
        warm,
        rolling_window,
        need_proba,
        delay,
        t0,
    )


def _ensure_host_specs(config: ExperimentConfig) -> list[dict[str, Any]]:
    specs = list(config.models)
    if not config.detectors:
        return specs
    existing = {str(spec.get("id")) for spec in specs}
    for host_id in config.detector_hosts:
        if host_id in existing:
            continue
        class_name = DEFAULT_HOSTS.get(host_id)
        if class_name is None:
            raise ValueError(f"unknown detector host {host_id!r}")
        specs.append(
            {
                "id": host_id,
                "type": "river",
                "params": {"class": class_name},
            }
        )
    return specs


def _oracle_spec(config: ExperimentConfig, stream: StreamData, delay: int = 0) -> dict[str, Any]:
    return {
        "id": ORACLE_ID,
        "type": "oracle_rf",
        "params": {"window": int(stream.meta["warm_up"]), "n_jobs": 1, "delay": int(delay)},
    }


def _mean(values: Iterable[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(clean)) if clean else None


def _sd(values: Iterable[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.std(clean, ddof=1)) if len(clean) >= 2 else None


def _summarize_models(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in per_seed:
        grouped.setdefault((row["model"], int(row.get("delay", 0))), []).append(row)
    summary: dict[str, Any] = {}
    for (model_id, delay), rows in grouped.items():
        name = model_id if delay == 0 else f"{model_id}|L={delay}"
        summary[name] = {
            "n": len(rows),
            "accuracy": {"mean": _mean(r["accuracy"] for r in rows), "sd": _sd(r["accuracy"] for r in rows)},
            "kappa": {"mean": _mean(r["kappa"] for r in rows), "sd": _sd(r["kappa"] for r in rows)},
            "ece": {"mean": _mean(r["ece"] for r in rows), "sd": _sd(r["ece"] for r in rows)},
            "brier": {"mean": _mean(r["brier"] for r in rows), "sd": _sd(r["brier"] for r in rows)},
            "aue": {"mean": _mean(r["aue"] for r in rows), "sd": _sd(r["aue"] for r in rows)},
            "damage_ref": {
                "mean": _mean(r["damage_ref"] for r in rows),
                "sd": _sd(r["damage_ref"] for r in rows),
            },
            "damage_full": {
                "mean": _mean(r["damage_full"] for r in rows),
                "sd": _sd(r["damage_full"] for r in rows),
            },
            "recovery_rate": {
                "mean": _mean(r["recovery_rate"] for r in rows),
                "sd": _sd(r["recovery_rate"] for r in rows),
            },
            "recovery_samples": {
                "mean": _mean(r["recovery_samples"] for r in rows),
                "sd": _sd(r["recovery_samples"] for r in rows),
            },
            "recovery_prediction_samples": {
                "mean": _mean(r.get("recovery_prediction_samples") for r in rows),
                "sd": _sd(r.get("recovery_prediction_samples") for r in rows),
            },
            "dip": {"mean": _mean(r["dip"] for r in rows), "sd": _sd(r["dip"] for r in rows)},
            "samples_per_s": {
                "mean": _mean(r["samples_per_s"] for r in rows),
                "sd": _sd(r["samples_per_s"] for r in rows),
            },
            "model_bytes": _mean(r["model_bytes"] for r in rows),
        }
    return summary


def _summarize_detections(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    by_shape: dict[tuple[str, str, str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["host"], row["detector"], row["channel"], int(row.get("delay", 0)))
        grouped.setdefault(key, []).append(row)
        shape_key = (
            row["generator"],
            row["shape"],
            row["host"],
            row["detector"],
            row["channel"],
            int(row.get("delay", 0)),
        )
        by_shape.setdefault(shape_key, []).append(row)

    overall: dict[str, Any] = {}
    for (host, detector, channel, delay), items in grouped.items():
        name = f"{host}|{detector}|{channel}|delay={delay}"
        overall[name] = {
            "n": len(items),
            "f1": {"mean": _mean(r["f1"] for r in items), "sd": _sd(r["f1"] for r in items)},
            "episode_recall": {
                "mean": _mean(r["episode_recall"] for r in items),
                "sd": _sd(r["episode_recall"] for r in items),
            },
            "alarm_precision": {
                "mean": _mean(r["alarm_precision"] for r in items),
                "sd": _sd(r["alarm_precision"] for r in items),
            },
            "ndt_ds": {"mean": _mean(r["ndt_ds"] for r in items), "sd": _sd(r["ndt_ds"] for r in items)},
            "ndt_de": {"mean": _mean(r["ndt_de"] for r in items), "sd": _sd(r["ndt_de"] for r in items)},
            "mean_delay_from_ds": {
                "mean": _mean(r["mean_delay_from_ds"] for r in items),
                "sd": _sd(r["mean_delay_from_ds"] for r in items),
            },
            "mean_delay_from_de": {
                "mean": _mean(r["mean_delay_from_de"] for r in items),
                "sd": _sd(r["mean_delay_from_de"] for r in items),
            },
            "far_per_1k": {
                "mean": _mean(r["far_per_1k"] for r in items),
                "sd": _sd(r["far_per_1k"] for r in items),
            },
            "alarm_rate_per_1k": {
                "mean": _mean(r["alarm_rate_per_1k"] for r in items),
                "sd": _sd(r["alarm_rate_per_1k"] for r in items),
            },
        }

    shape_rows: list[dict[str, Any]] = []
    for key, items in by_shape.items():
        generator, shape, host, detector, channel, delay = key
        shape_rows.append(
            {
                "generator": generator,
                "shape": shape,
                "host": host,
                "detector": detector,
                "channel": channel,
                "delay": delay,
                "n": len(items),
                "f1": _mean(r["f1"] for r in items),
                "episode_recall": _mean(r["episode_recall"] for r in items),
                "ndt_ds": _mean(r["ndt_ds"] for r in items),
                "ndt_de": _mean(r["ndt_de"] for r in items),
                "mean_delay_from_ds": _mean(r["mean_delay_from_ds"] for r in items),
                "mean_delay_from_de": _mean(r["mean_delay_from_de"] for r in items),
                "far_per_1k": _mean(r["far_per_1k"] for r in items),
                "alarm_rate_per_1k": _mean(r["alarm_rate_per_1k"] for r in items),
            }
        )
    return overall, shape_rows


def _acceptance_verdicts(
    config: ExperimentConfig,
    model_summary: dict[str, Any],
    detector_summary: dict[str, Any],
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    rules = config.acceptance or {}
    verdicts: dict[str, Any] = {}

    baseline = rules.get("baseline_model")
    treatment = rules.get("treatment_model")
    if baseline and treatment and baseline in model_summary and treatment in model_summary:
        baseline_aue = model_summary[baseline]["aue"]["mean"]
        treatment_aue = model_summary[treatment]["aue"]["mean"]
        if baseline_aue is not None and treatment_aue is not None:
            threshold = float(rules.get("min_aue_ratio", 0.0))
            gain = baseline_aue - treatment_aue
            if treatment_aue <= 0:
                ratio = None
                passed = True
                note = "treatment is at or better than the oracle, ratio not meaningful"
            else:
                ratio = baseline_aue / treatment_aue
                passed = ratio >= threshold
                note = None
            verdicts["aue_ratio"] = {
                "baseline": baseline,
                "treatment": treatment,
                "baseline_aue": baseline_aue,
                "treatment_aue": treatment_aue,
                "aue_gain": gain,
                "ratio": ratio,
                "threshold": threshold,
                "note": note,
                "pass": passed,
            }

    if rules.get("frozen_never_recovers") and baseline:
        rows = [r for r in per_seed if r["model"] == baseline]
        never = sum(1 for r in rows if (r["recovery_rate"] or 0.0) == 0.0)
        verdicts["frozen_never_recovers"] = {
            "model": baseline,
            "never": never,
            "of": len(rows),
            "pass": never == len(rows),
        }

    if rules.get("treatment_damage_lower_seeds") and baseline and treatment:
        by_key: dict[tuple[int, int], dict[str, float]] = {}
        for row in per_seed:
            if row["model"] in {baseline, treatment} and row["damage_ref"] is not None:
                by_key.setdefault((row["stream"], row["seed"]), {})[row["model"]] = row["damage_ref"]
        wins = sum(
            1
            for values in by_key.values()
            if baseline in values and treatment in values and values[treatment] < values[baseline]
        )
        required = int(rules["treatment_damage_lower_seeds"])
        available_seeds = len({row["seed"] for row in per_seed})
        enough = available_seeds >= required
        verdict = {
            "wins": wins,
            "of": len(by_key),
            "available_seeds": available_seeds,
            "required": required,
            "pass": (wins >= required) if enough else None,
        }
        if not enough:
            verdict["note"] = (
                f"only {available_seeds} seeds; the pre-registered bar needs {required}"
            )
        verdicts["treatment_damage_lower_seeds"] = verdict

    far_rules = rules.get("max_far_per_1k") or {}
    for detector, threshold in far_rules.items():
        values = [
            row["far_per_1k"]["mean"]
            for name, row in detector_summary.items()
            if name.split("|")[1] == detector and row["far_per_1k"]["mean"] is not None
        ]
        worst = max(values) if values else None
        verdicts[f"far_{detector}"] = {
            "worst_mean_far_per_1k": worst,
            "threshold": float(threshold),
            "pass": worst is not None and worst <= float(threshold),
        }

    checks = [
        value["pass"]
        for value in verdicts.values()
        if isinstance(value, dict) and "pass" in value
    ]
    if any(check is None for check in checks):
        verdicts["pass"] = None
    elif checks:
        verdicts["pass"] = all(checks)
    else:
        verdicts["pass"] = None
        verdicts["note"] = "no acceptance rules defined; nothing was asserted"
    return verdicts


def _extract_deep_signals(
    writer: ArtifactWriter,
    config: ExperimentConfig,
    stream: StreamData,
    stream_idx: int,
    seed: int,
    warm: int,
    host_runs: dict[str, ModelRun],
) -> None:
    from .deep.signals import DeepSignalMonitor, SignalConfig
    from .deep.train import DriftFormerConfig, DriftFormerLearner

    settings = config.deep_signals
    encoder_params = dict(settings.get("encoder") or {})
    encoder_params.setdefault("device", "cpu")
    learner = DriftFormerLearner(
        "dfh_encoder",
        stream.feature_keys,
        n_classes=max(stream.y) + 1,
        config=DriftFormerConfig.from_dict(encoder_params),
        seed=seed,
    )
    learner.pretrain_ssl(
        stream.X[:warm],
        epochs=int(settings.get("epochs", 1)),
        batch_size=settings.get("batch_size"),
    )
    signal_config = SignalConfig(
        window_size=int(settings.get("window_size", 500)),
        stride=int(settings.get("stride", 100)),
        reference_fraction=float(settings.get("reference_fraction", 0.2)),
        mask_fraction=float(
            settings.get("mask_fraction", encoder_params.get("mask_fraction", 0.15))
        ),
        adwin_delta=float(
            settings.get("adwin_delta", encoder_params.get("adwin_delta", 0.002))
        ),
        ks_reference_size=int(settings.get("ks_reference_size", 2_000)),
    )
    try:
        series = DeepSignalMonitor(signal_config).extract(learner, stream.X[warm:])
    except ValueError as exc:
        print(f"  deep signals skipped: {exc}", flush=True)
        return

    prefix = f"s{stream_idx}_seed{seed}_signals_"
    arrays = series.as_arrays(prefix)
    baseline_id = str(settings.get("baseline_model", "ht"))
    host = host_runs.get(baseline_id)
    if host is not None:
        errors = np.asarray(host.errors[warm:], dtype=np.float64)
        error_rate = np.asarray(
            [
                float(np.nanmean(errors[int(start) : int(end)]))
                if np.isfinite(errors[int(start) : int(end)]).any()
                else 0.0
                for start, end in zip(series.starts, series.ends)
            ],
            dtype=np.float64,
        )
        arrays[f"{prefix}error_rate"] = np.nan_to_num(error_rate)
    for key, values in arrays.items():
        writer.add_signal(key, values)
    print(
        f"  deep signals: {len(series.starts)} windows, "
        f"{len(series.values)} signals, reference={series.reference_windows}",
        flush=True,
    )


def run_experiment(
    config: ExperimentConfig,
    repo: Path,
    out_dir: Path | None = None,
    limit: int | None = None,
    seeds: Sequence[int] | None = None,
    need_proba: bool | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir) if out_dir else repo / "experiments" / "results" / config.exp_id
    if resume and (out / "meta.json").exists():
        existing = json.loads((out / "meta.json").read_text())
        if existing.get("config_hash") != config.config_hash:
            raise SystemExit(
                f"{out}: config hash changed; rerun without --resume or use --out"
            )
    writer = ArtifactWriter(
        exp_id=config.exp_id,
        out_dir=out,
        repo=repo,
        config=config.as_dict(),
        config_hash=config.config_hash,
        resume=resume,
    )
    writer.write_meta()
    seeds = list(seeds) if seeds else list(config.seeds)
    need_proba = bool(config.raw.get("need_proba", True)) if need_proba is None else need_proba
    model_specs = _ensure_host_specs(config)
    allowed_units = {
        f"s{stream_idx}_seed{seed}"
        for stream_idx in range(len(config.streams))
        for seed in seeds
    }

    per_seed: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    if resume:
        cached_models, cached_episodes, cached_detections = writer.load_seed_caches(
            allowed_units
        )
        per_seed.extend(cached_models)
        episode_rows.extend(cached_episodes)
        detection_rows.extend(cached_detections)
    completed = writer.completed_units() & allowed_units if resume else set()
    done_units: list[str] = sorted(completed)

    for stream_idx, stream_cfg in enumerate(config.streams):
        for seed in seeds:
            unit = f"s{stream_idx}_seed{seed}"
            if unit in completed:
                print(f"[{config.exp_id}] skip {unit} (cached)", flush=True)
                continue
            model_start = len(per_seed)
            episode_start = len(episode_rows)
            detection_start = len(detection_rows)
            print(
                f"[{config.exp_id}] stream {stream_idx + 1}/{len(config.streams)} "
                f"{stream_cfg.generator}/{stream_cfg.shape} seed {seed}",
                flush=True,
            )
            cfg = replace(stream_cfg, seed=seed)
            if cfg.position_jitter > 0:
                cfg = replace(
                    cfg,
                    position=cfg.position + (seed * 997) % cfg.position_jitter,
                )
            cfg = scale_stream_config(cfg, limit)
            stream = build_stream(cfg)
            warm = int(stream.meta["warm_up"])
            host_runs_delay0: dict[str, ModelRun] = {}
            for delay in config.model_delays:
                runs: list[ModelRun] = []
                for spec in model_specs:
                    model_label = str(spec.get("id") or spec.get("type"))
                    print(f"  model {model_label} delay={delay}", flush=True)
                    run = run_model(
                        spec,
                        stream,
                        seed,
                        config.metrics["rolling_window"],
                        need_proba,
                        delay=delay,
                    )
                    print(
                        f"  done {run.model_id}: {run.runtime_s:.1f}s "
                        f"acc={run.accuracy:.3f} kappa={run.full_kappa:.3f}",
                        flush=True,
                    )
                    runs.append(run)
                    prefix = f"s{stream_idx}_seed{seed}_L{delay}_{_sanitize(run.model_id)}"
                    writer.add_series(f"{prefix}_pred", run.predictions)
                    writer.add_series(f"{prefix}_err", run.errors.astype(np.float16))
                    writer.add_series(f"{prefix}_kappa", run.kappa.astype(np.float16))
                    if need_proba:
                        writer.add_series(f"{prefix}_proba_pos", run.proba_pos.astype(np.float16))
                        writer.add_series(f"{prefix}_proba_max", run.proba_max.astype(np.float16))

                oracle = run_model(
                    _oracle_spec(config, stream, delay=delay),
                    stream,
                    seed,
                    config.metrics["rolling_window"],
                    need_proba=False,
                    delay=delay,
                )
                runs.append(oracle)
                writer.add_series(
                    f"s{stream_idx}_seed{seed}_L{delay}_y",
                    np.asarray(stream.y, dtype=np.int8),
                )
                writer.add_series(
                    f"s{stream_idx}_seed{seed}_L{delay}_{_sanitize(oracle.model_id)}_err",
                    oracle.errors.astype(np.float16),
                )
                if delay == 0:
                    host_runs_delay0 = {r.model_id: r for r in runs}

                kappa_by_model = {
                    r.model_id: r.kappa for r in runs if r.model_id not in {ORACLE_ID}
                }
                model_episode_rows: dict[tuple[str, int], list[dict[str, Any]]] = {
                    (r.model_id, delay): [] for r in runs
                }
                for episode in stream.episodes:
                    ref = panel_kappa_ref(kappa_by_model, episode)
                    width = window_for(episode.kind, episode.end - episode.start, config.metrics)
                    for run in runs:
                        result = evaluate_window(
                            run.kappa,
                            run.errors,
                            oracle.errors,
                            episode,
                            ref,
                            width,
                            band=float(config.metrics["band"]),
                            stretch=int(config.metrics["stretch"]),
                            shift=delay,
                            recovery_horizon=(
                                int(config.metrics["recovery_horizon"])
                                if config.metrics.get("recovery_horizon")
                                else None
                            ),
                        )
                        row = {
                            "stream": stream_idx,
                            "generator": stream.meta["generator"],
                            "shape": stream.meta["shape"],
                            "seed": seed,
                            "model": run.model_id,
                            "delay": delay,
                            "episode": episode.as_dict(),
                            **result.as_dict(),
                        }
                        writer.add_episode(row)
                        episode_rows.append(row)
                        model_episode_rows[(run.model_id, delay)].append(row)

                for run in runs:
                    rows = model_episode_rows[(run.model_id, delay)]
                    per_seed.append(
                        {
                            "stream": stream_idx,
                            "generator": stream.meta["generator"],
                            "shape": stream.meta["shape"],
                            "seed": seed,
                            "model": run.model_id,
                            "delay": delay,
                            "accuracy": run.accuracy,
                            "kappa": run.full_kappa,
                            "ece": run.ece,
                            "brier": run.brier,
                            "aue": _mean(r["aue"] for r in rows),
                            "damage_ref": _mean(r["damage_ref"] for r in rows),
                            "damage_full": _mean(r["damage_full"] for r in rows),
                            "recovery_rate": _mean(r["recovery_rate"] for r in rows),
                            "recovery_samples": _mean(r["recovery_samples"] for r in rows),
                            "recovery_prediction_samples": _mean(
                                r["recovery_prediction_samples"] for r in rows
                            ),
                            "dip": _mean(r["dip"] for r in rows),
                            "samples_per_s": run.samples_per_s,
                            "model_bytes": run.model_bytes,
                        }
                    )
                    writer.add_model_run(per_seed[-1])

            if config.deep_signals.get("enabled", False):
                _extract_deep_signals(
                    writer,
                    config,
                    stream,
                    stream_idx,
                    seed,
                    warm,
                    host_runs_delay0,
                )

            if config.detectors:
                host_runs = host_runs_delay0
                delta_max = delta_max_for(stream.meta["shape"], config.metrics)
                for host_id in config.detector_hosts:
                    host = host_runs.get(host_id)
                    if host is None:
                        continue
                    base_values = host.errors[warm:]
                    for det_delay in config.detector_delays:
                        if det_delay < 0 or det_delay >= len(base_values):
                            continue
                        values = (
                            base_values[: len(base_values) - det_delay]
                            if det_delay
                            else base_values
                        )
                        offset = warm + det_delay
                        for detector_run in replay_all(
                            values,
                            config.detectors,
                            config.detector_channels,
                            alpha=float(config.metrics["ewma_alpha"]),
                        ):
                            alarms = [a + offset for a in detector_run.alarms]
                            metrics = evaluate_detections(
                                alarms,
                                stream.episodes,
                                n_samples=len(stream),
                                delta_max=delta_max,
                                shift=det_delay,
                            )
                            row = {
                                "stream": stream_idx,
                                "generator": stream.meta["generator"],
                                "shape": stream.meta["shape"],
                                "seed": seed,
                                "host": host_id,
                                "detector": detector_run.detector,
                                "channel": detector_run.channel,
                                "delay": det_delay,
                                "n_samples": len(stream),
                                "warm_up": warm,
                                "delta_max": delta_max,
                                **metrics.as_dict(),
                            }
                            writer.add_detection(row)
                            detection_rows.append(row)

            writer.save_seed_cache(
                unit,
                {
                    "models": per_seed[model_start:],
                    "episodes": episode_rows[episode_start:],
                    "detections": detection_rows[detection_start:],
                },
                f"{unit}_",
            )
            done_units.append(unit)
            writer.write_partial(
                {
                    "status": "in progress",
                    "stream": stream_idx,
                    "streams_total": len(config.streams),
                    "seed": seed,
                    "units_done": list(done_units),
                    "models_done": len(per_seed),
                    "episodes": len(episode_rows),
                    "detections": len(detection_rows),
                }
            )
            print(f"[{config.exp_id}] {unit} done", flush=True)

    model_summary = _summarize_models(per_seed)
    detector_summary, detector_by_shape = _summarize_detections(detection_rows)
    verdicts = (
        _acceptance_verdicts(config, model_summary, detector_summary, per_seed)
        if not limit
        else {
            "smoke_slice": True,
            "limit": limit,
            "pass": None,
            "note": "acceptance is not evaluated on a truncated stream",
        }
    )
    summary = {
        "title": config.title,
        "description": config.description,
        "n_streams": len(config.streams),
        "seeds": seeds,
        "n_model_runs": len(per_seed),
        "n_episodes": len(episode_rows),
        "n_detections": len(detection_rows),
        "models": model_summary,
        "detectors": detector_summary,
        "detectors_by_shape": detector_by_shape,
        "acceptance": verdicts,
        "per_seed": per_seed,
        "limits": {"limit": limit, "need_proba": need_proba},
    }
    if config.deep_signals:
        summary["deep_signals"] = {
            "enabled": bool(config.deep_signals.get("enabled", False)),
            "config": config.deep_signals,
            "n_arrays": len(writer.signal_items()),
        }
    writer.rewrite_rows(per_seed, episode_rows, detection_rows)
    writer.finalize(summary)
    writer.write_index_entry()
    return summary
