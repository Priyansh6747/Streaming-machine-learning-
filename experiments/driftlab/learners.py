"""Uniform learner protocol for classical, batch and River models."""

from __future__ import annotations

import inspect
import pickle
import random
from collections import deque
from typing import Any, Iterable, Sequence

import numpy as np
from river import ensemble, forest, tree
from sklearn.ensemble import RandomForestClassifier

RIVER_MODELS: dict[str, Any] = {
    "HoeffdingTreeClassifier": tree.HoeffdingTreeClassifier,
    "HoeffdingAdaptiveTreeClassifier": tree.HoeffdingAdaptiveTreeClassifier,
    "ARFClassifier": forest.ARFClassifier,
    "LeveragingBaggingClassifier": ensemble.LeveragingBaggingClassifier,
    "ADWINBaggingClassifier": ensemble.ADWINBaggingClassifier,
}


class BaseLearner:
    id: str
    kind: str

    def fit_initial(self, X: Sequence[dict], y: Sequence[int]) -> None:
        return None

    def predict_one(self, x: dict) -> int | None:
        raise NotImplementedError

    def predict_proba_one(self, x: dict) -> dict[int, float] | None:
        return None

    def learn_one(self, x: dict, y: int, t: int | None = None) -> None:
        raise NotImplementedError

    def model_bytes(self) -> int:
        try:
            return len(pickle.dumps(self))
        except Exception:
            return 0


class RiverLearner(BaseLearner):
    kind = "river"

    def __init__(self, model_id: str, model: Any):
        self.id = model_id
        self.model = model

    def predict_one(self, x: dict) -> int | None:
        try:
            return self.model.predict_one(x)
        except Exception:
            return None

    def predict_proba_one(self, x: dict) -> dict[int, float] | None:
        try:
            proba = self.model.predict_proba_one(x)
        except Exception:
            return None
        return proba or None

    def learn_one(self, x: dict, y: int, t: int | None = None) -> None:
        self.model.learn_one(x, y)

    def model_bytes(self) -> int:
        try:
            return len(pickle.dumps(self.model))
        except Exception:
            return 0


class SklearnRF(BaseLearner):
    kind = "sklearn"

    def __init__(
        self,
        model_id: str,
        feature_keys: Sequence[Any],
        n_estimators: int = 100,
        n_jobs: int = 1,
        seed: int = 42,
    ):
        self.id = model_id
        self.feature_keys = list(feature_keys)
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.seed = seed
        self.model: RandomForestClassifier | None = None
        self.buffer: deque[tuple[np.ndarray, int]] = deque()

    def _to_array(self, x: dict) -> np.ndarray:
        return np.asarray([float(x[k]) for k in self.feature_keys], dtype=np.float64)

    def fit_initial(self, X: Sequence[dict], y: Sequence[int]) -> None:
        for xi, yi in zip(X, y):
            self.buffer.append((self._to_array(xi), int(yi)))
        self._refit()

    def _refit(self) -> None:
        if len(self.buffer) < 2:
            return
        X = np.vstack([row for row, _ in self.buffer])
        y = np.asarray([label for _, label in self.buffer], dtype=np.int64)
        if len(np.unique(y)) < 2:
            return
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X, y)

    def predict_one(self, x: dict) -> int | None:
        if self.model is None:
            return None
        return int(self.model.predict(self._to_array(x).reshape(1, -1))[0])

    def predict_proba_one(self, x: dict) -> dict[int, float] | None:
        if self.model is None:
            return None
        proba = self.model.predict_proba(self._to_array(x).reshape(1, -1))[0]
        return {int(c): float(p) for c, p in zip(self.model.classes_, proba)}

    def predict_batch(
        self, X: Sequence[dict], return_proba: bool = False
    ) -> tuple[list[int | None], list[dict[int, float]] | None]:
        if not X:
            return [], [] if return_proba else None
        if self.model is None:
            return [None] * len(X), None
        array = np.vstack([self._to_array(x) for x in X])
        predictions = [int(value) for value in self.model.predict(array)]
        probabilities = None
        if return_proba:
            matrix = self.model.predict_proba(array)
            classes = self.model.classes_
            probabilities = [
                {int(c): float(p) for c, p in zip(classes, row)} for row in matrix
            ]
        return predictions, probabilities

    def learn_one(self, x: dict, y: int, t: int | None = None) -> None:
        return None

    def model_bytes(self) -> int:
        if self.model is None:
            return 0
        try:
            return len(pickle.dumps(self.model))
        except Exception:
            return 0


class FrozenRF(SklearnRF):
    kind = "frozen"

    def learn_one(self, x: dict, y: int, t: int | None = None) -> None:
        return None


class ScheduledRF(SklearnRF):
    kind = "scheduled"

    def __init__(self, model_id: str, feature_keys: Sequence[Any], every: int = 5_000,
                 window: int = 12_000, **kwargs: Any):
        super().__init__(model_id, feature_keys, **kwargs)
        self.every = int(every)
        self.window = int(window)
        self.maxlen = self.window
        self.buffer = deque(maxlen=self.maxlen)
        self._seen = 0

    def fit_initial(self, X: Sequence[dict], y: Sequence[int]) -> None:
        rows = list(zip(X, y))[-self.window:]
        for xi, yi in rows:
            self.buffer.append((self._to_array(xi), int(yi)))
        self._seen = len(rows)
        self._refit()

    def learn_one(self, x: dict, y: int, t: int | None = None) -> None:
        self.buffer.append((self._to_array(x), int(y)))
        self._seen += 1
        due = (t is not None and self.every > 0 and int(t) % self.every == 0) or (
            t is None and self.every > 0 and self._seen % self.every == 0
        )
        if due:
            self._refit()


class OracleRF(SklearnRF):
    kind = "oracle"

    def __init__(self, model_id: str, feature_keys: Sequence[Any], episode_starts: Iterable[int],
                 window: int = 12_000, delay: int = 0, **kwargs: Any):
        super().__init__(model_id, feature_keys, **kwargs)
        self.episode_starts = {int(v) for v in episode_starts}
        self.window = int(window)
        self.delay = int(delay)
        self.buffer = deque(maxlen=self.window)

    def fit_initial(self, X: Sequence[dict], y: Sequence[int]) -> None:
        rows = list(zip(X, y))[-self.window:]
        for xi, yi in rows:
            self.buffer.append((self._to_array(xi), int(yi)))
        self._refit()

    def learn_one(self, x: dict, y: int, t: int | None = None) -> None:
        self.buffer.append((self._to_array(x), int(y)))
        if t is not None:
            now = int(t)
            if now in self.episode_starts or (now - self.delay) in self.episode_starts:
                self._refit()


def _instantiate_river(class_name: str, params: dict[str, Any], seed: int) -> Any:
    try:
        cls = RIVER_MODELS[class_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown River model {class_name!r}; available: {', '.join(sorted(RIVER_MODELS))}"
        ) from exc
    signature = inspect.signature(cls.__init__)
    kwargs = dict(params)
    if class_name in {
        "BaggingClassifier",
        "ADWINBaggingClassifier",
        "LeveragingBaggingClassifier",
    }:
        kwargs.setdefault("model", tree.HoeffdingTreeClassifier())
    if "seed" in signature.parameters and "seed" not in kwargs:
        kwargs["seed"] = seed
    if "rng" in signature.parameters and "rng" not in kwargs:
        kwargs["rng"] = random.Random(seed)
    return cls(**kwargs)


def build_learner(spec: dict[str, Any], stream, seed: int) -> BaseLearner:
    model_id = str(spec.get("id") or spec.get("type") or "model")
    kind = str(spec.get("type"))
    params = dict(spec.get("params") or {})
    if kind == "river":
        class_name = str(params.pop("class"))
        return RiverLearner(model_id, _instantiate_river(class_name, params, seed))
    if kind in {"frozen_rf", "scheduled_rf", "oracle_rf"}:
        rf_params = {
            "n_estimators": int(params.get("n_estimators", 100)),
            "n_jobs": int(params.get("n_jobs", 1)),
        }
        if kind == "frozen_rf":
            return FrozenRF(model_id, stream.feature_keys, seed=seed, **rf_params)
        if kind == "scheduled_rf":
            return ScheduledRF(
                model_id,
                stream.feature_keys,
                every=int(params.get("every", 5_000)),
                window=int(params.get("window", stream.meta["warm_up"])),
                seed=seed,
                **rf_params,
            )
        return OracleRF(
            model_id,
            stream.feature_keys,
            episode_starts=[ep.start for ep in stream.episodes],
            window=int(params.get("window", stream.meta["warm_up"])),
            delay=int(params.get("delay", 0)),
            seed=seed,
            **rf_params,
        )
    if kind == "deepriver_rolling":
        from .deep.deepriver import DeepRiverRolling

        return DeepRiverRolling(
            model_id,
            n_features=len(stream.feature_keys),
            seed=seed,
            **params,
        )
    if kind == "driftformer":
        from .deep.train import DriftFormerConfig, DriftFormerLearner

        n_features = params.pop("n_features", None)
        checkpoint = params.pop("checkpoint", None)
        config = DriftFormerConfig.from_dict(params)
        learner = DriftFormerLearner(
            model_id,
            stream.feature_keys,
            n_classes=max(stream.y) + 1,
            config=config,
            n_features=n_features,
            seed=seed,
        )
        if checkpoint:
            learner.load(checkpoint)
        return learner
    raise ValueError(f"unknown learner type: {kind!r}")
