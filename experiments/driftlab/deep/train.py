"""Block-prequential training loop for DriftFormer."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from river.drift import ADWIN

from ..calibration import fit_temperature
from .model import DriftFormer, DriftFormerShape


@dataclass
class DriftFormerConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ffn: int = 128
    n_heads_bank: int = 4
    n_bins: int = 8
    dropout: float = 0.0
    block_size: int = 32
    batch_size: int = 128
    ring_size: int = 4_096
    reservoir_size: int = 8_192
    lr_encoder: float = 1e-4
    lr_head: float = 3e-4
    weight_decay: float = 1e-4
    ssl_weight: float = 0.3
    label_smoothing: float = 0.05
    mask_fraction: float = 0.15
    temperature_every: int = 1_000
    head_update_every: int = 32
    head_decay: float = 0.99
    reset_on_alarm: bool = True
    refractory: int = 2_000
    two_strike: int = 2
    adwin_delta: float = 0.002
    device: str = "cpu"
    seed: int = 42

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DriftFormerConfig":
        known = {item.name for item in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})


class RunningStandardizer:
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.count > 1:
            std = np.sqrt(self.m2 / (self.count - 1))
        else:
            std = np.ones_like(self.mean)
        return (x - self.mean) / np.maximum(std, 1e-6)

    def reset(self) -> None:
        self.mean.fill(0.0)
        self.m2.fill(0.0)
        self.count = 0


def _mean_pairwise_js(probs: torch.Tensor) -> float:
    """Mean pairwise Jensen-Shannon divergence across a head bank, in [0, 1]."""
    n_heads = int(probs.shape[1])
    if n_heads < 2:
        return 0.0
    eps = 1e-12
    total = 0.0
    pairs = 0
    for left in range(n_heads):
        for right in range(left + 1, n_heads):
            p = probs[:, left, :].clamp_min(eps)
            q = probs[:, right, :].clamp_min(eps)
            middle = 0.5 * (p + q)
            kl_p = (p * (p.log() - middle.log())).sum(dim=-1)
            kl_q = (q * (q.log() - middle.log())).sum(dim=-1)
            total += float((0.5 * (kl_p + kl_q)).mean())
            pairs += 1
    return total / max(1, pairs) / math.log(2.0)


class DriftFormerLearner:
    kind = "driftformer"

    def __init__(
        self,
        model_id: str,
        feature_keys: Sequence[Any],
        n_classes: int,
        config: DriftFormerConfig | None = None,
        n_features: int | None = None,
        **kwargs: Any,
    ):
        self.id = model_id
        self.feature_keys = list(feature_keys)
        self.n_features = int(n_features) if n_features else len(self.feature_keys)
        self.n_classes = int(n_classes)
        base = config or DriftFormerConfig()
        overrides = {k: v for k, v in kwargs.items() if k in {item.name for item in fields(DriftFormerConfig)}}
        self.config = DriftFormerConfig(**{**asdict(base), **overrides})
        if self.config.device == "auto":
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.config.device)

        torch.manual_seed(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)
        shape = DriftFormerShape(
            n_features=self.n_features,
            n_classes=self.n_classes,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            ffn=self.config.ffn,
            n_heads_bank=self.config.n_heads_bank,
            n_bins=self.config.n_bins,
            dropout=self.config.dropout,
        )
        self.model = DriftFormer(shape).to(self.device)
        token_params = [
            self.model.feature_weight,
            self.model.feature_bias,
            self.model.position,
            self.model.cls,
        ]
        encoder_params = list(self.model.encoder.parameters()) + token_params
        head_params = [p for p in self.model.heads.parameters()] + list(self.model.ssl_head.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.config.lr_encoder},
                {"params": head_params, "lr": self.config.lr_head},
            ],
            weight_decay=self.config.weight_decay,
        )

        self.standardizer = RunningStandardizer(self.n_features)
        self.ring: deque[tuple[np.ndarray, int]] = deque(maxlen=self.config.ring_size)
        self.reservoir: list[tuple[np.ndarray, int]] = []
        self.seen = 0
        self.head_scores = np.zeros(self.config.n_heads_bank, dtype=float)
        self.active_head = 0
        self.temperature = 1.0
        self.last_reset = -10**12
        self.strikes = 0
        self.warming = True
        self.detector = ADWIN(delta=self.config.adwin_delta)
        self.last_loss: float | None = None
        self.loss_history: deque[float] = deque(maxlen=1_000)
        self._next_temperature = self.config.temperature_every
        self._last_head_update = 0
        self.force_active_until = 0

    def set_feature_keys(self, feature_keys: Sequence[Any]) -> None:
        self.feature_keys = list(feature_keys)

    def _to_array(self, x: dict) -> np.ndarray:
        array = np.zeros(self.n_features, dtype=np.float64)
        for i, key in enumerate(self.feature_keys[: self.n_features]):
            if key in x:
                array[i] = float(x[key])
        return array

    def _batch_arrays(self, X: Sequence[dict], standardize: bool = True) -> np.ndarray:
        raw = np.vstack([self._to_array(x) for x in X])
        if standardize:
            raw = np.vstack([self.standardizer.transform(row) for row in raw])
        return raw.astype(np.float32)

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def predict_batch(
        self, X: Sequence[dict], return_proba: bool = True
    ) -> tuple[list[int], list[dict[int, float]] | None]:
        if not X:
            return [], [] if return_proba else None
        array = self._batch_arrays(X, standardize=True)
        self.model.eval()
        with torch.no_grad():
            logits, _, _ = self.model(self._tensor(array))
            head_logits = logits[:, self.active_head, :] / max(self.temperature, 1e-3)
            probs = F.softmax(head_logits, dim=-1)
        predictions = probs.argmax(dim=-1).detach().cpu().numpy().astype(int)
        if not return_proba:
            return predictions.tolist(), None
        probas = []
        for row in probs.detach().cpu().numpy():
            probas.append({int(i): float(p) for i, p in enumerate(row)})
        return predictions.tolist(), probas

    def predict_one(self, x: dict) -> int | None:
        predictions, _ = self.predict_batch([x], return_proba=False)
        return predictions[0] if predictions else None

    def predict_proba_one(self, x: dict) -> dict[int, float] | None:
        _, probas = self.predict_batch([x], return_proba=True)
        return probas[0] if probas else None

    def _reservoir_add(self, array: np.ndarray, label: int) -> None:
        if len(self.reservoir) < self.config.reservoir_size:
            self.reservoir.append((array, int(label)))
            return
        index = int(self.rng.integers(0, self.seen))
        if index < self.config.reservoir_size:
            self.reservoir[index] = (array, int(label))

    def _update_head_scores(self) -> None:
        if len(self.ring) < 8:
            return
        items = list(self.ring)[-256:]
        X = np.vstack([item[0] for item in items])
        y = np.asarray([item[1] for item in items], dtype=np.int64)
        self.model.eval()
        with torch.no_grad():
            logits, _, _ = self.model(self._tensor(X))
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
        for head in range(self.config.n_heads_bank):
            correct = float(np.mean(preds[:, head] == y))
            self.head_scores[head] = (
                self.config.head_decay * self.head_scores[head]
                + (1.0 - self.config.head_decay) * correct
            )
        if self.seen >= self.force_active_until:
            self.active_head = int(np.argmax(self.head_scores))

    def _sample_batch(self) -> tuple[np.ndarray, np.ndarray]:
        batch = self.config.batch_size
        recent_n = max(1, batch // 2)
        recent = list(self.ring)
        if len(recent) >= recent_n:
            indices = self.rng.choice(len(recent), size=recent_n, replace=False)
            items = [recent[int(i)] for i in indices]
        else:
            items = recent
        need = batch - len(items)
        if need > 0 and self.reservoir:
            indices = self.rng.choice(len(self.reservoir), size=need, replace=len(self.reservoir) < need)
            items.extend(self.reservoir[int(i)] for i in indices)
        while len(items) < batch:
            items.append(items[self.rng.integers(0, len(items))])
        X = np.vstack([item[0] for item in items])
        y = np.asarray([item[1] for item in items], dtype=np.int64)
        return X, y

    def _feature_bins(self, X: np.ndarray) -> torch.Tensor:
        clipped = np.clip(X, -3.0, 3.0)
        bins = ((clipped + 3.0) / 6.0 * self.config.n_bins).astype(int)
        bins = np.clip(bins, 0, self.config.n_bins - 1)
        return torch.as_tensor(bins, dtype=torch.long, device=self.device)

    def _step(self) -> None:
        if len(self.ring) < 8:
            return
        X, y = self._sample_batch()
        x_tensor = self._tensor(X)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=self.device)
        self.model.train()
        logits, ssl_logits, _ = self.model(x_tensor)
        supervised = F.cross_entropy(
            logits[:, self.active_head, :],
            y_tensor,
            label_smoothing=self.config.label_smoothing,
        )
        mask = torch.rand_like(x_tensor) < self.config.mask_fraction
        empty = mask.sum(dim=1) == 0
        if empty.any():
            mask[empty, 0] = True
        bins = self._feature_bins(X)
        self_supervised = F.cross_entropy(ssl_logits[mask], bins[mask])
        loss = supervised + self.config.ssl_weight * self_supervised
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.last_loss = float(supervised.detach())
        self.loss_history.append(float(self_supervised.detach()))
        if not self.warming:
            self.detector.update(float(self_supervised.detach()))
            if self.detector.drift_detected:
                self.on_alarm(self.seen)

    def _fit_temperature(self) -> None:
        if self.n_classes != 2 or len(self.ring) < 32:
            return
        items = list(self.ring)[-512:]
        X = np.vstack([item[0] for item in items])
        y = np.asarray([item[1] for item in items], dtype=np.int64)
        self.model.eval()
        with torch.no_grad():
            logits, _, _ = self.model(self._tensor(X))
            active = logits[:, self.active_head, :].detach().cpu().numpy()
        self.temperature = fit_temperature(active[:, 1] - active[:, 0], y)

    def on_alarm(self, t: int) -> None:
        if not self.config.reset_on_alarm or self.warming:
            return
        if t - self.last_reset < self.config.refractory:
            return
        self.strikes += 1
        if self.strikes < self.config.two_strike:
            return
        worst = int(np.argmin(self.head_scores))
        head = self.model.heads[worst]
        torch.nn.init.xavier_uniform_(head.weight)
        torch.nn.init.zeros_(head.bias)
        self.head_scores[worst] = float(np.max(self.head_scores)) + 1e-6
        self.active_head = worst
        self.force_active_until = self.seen + 10 * max(1, self.config.block_size)
        self.last_reset = t
        self.strikes = 0

    def learn_batch(self, X: Sequence[dict], y: Sequence[int], t_start: int | None = None) -> None:
        for x, label in zip(X, y):
            array = self._to_array(x)
            self.standardizer.update(array)
            standardized = self.standardizer.transform(array).astype(np.float32)
            self.ring.append((standardized, int(label)))
            self._reservoir_add(standardized, int(label))
            self.seen += 1
        if self.seen - self._last_head_update >= self.config.head_update_every:
            self._update_head_scores()
            self._last_head_update = self.seen
        self._step()
        if self.seen >= self._next_temperature:
            self._fit_temperature()
            self._next_temperature += self.config.temperature_every

    def learn_one(self, x: dict, y: int, t: int | None = None) -> None:
        self.learn_batch([x], [y], t_start=t)

    def warm_up(self, X: Sequence[dict], y: Sequence[int]) -> None:
        self.warming = True
        block = self.config.block_size
        for start in range(0, len(X), block):
            self.learn_batch(X[start : start + block], y[start : start + block], t_start=start)
        self.warming = False
        self.detector = ADWIN(delta=self.config.adwin_delta)

    def _standardize_array(self, array: np.ndarray) -> np.ndarray:
        if self.standardizer.count > 1:
            std = np.sqrt(self.standardizer.m2 / (self.standardizer.count - 1))
        else:
            std = np.ones_like(self.standardizer.mean)
        return (array - self.standardizer.mean) / np.maximum(std, 1e-6)

    def pretrain_ssl(
        self,
        X: Sequence[dict] | np.ndarray,
        epochs: int = 1,
        batch_size: int | None = None,
    ) -> None:
        """Train the encoder with the masked-feature objective only.

        No labels are used: the supervised heads keep their initial weights, so
        Stage 1 of DF-H stays label-free.
        """
        if isinstance(X, np.ndarray):
            array = np.asarray(X, dtype=np.float64)
        else:
            array = np.vstack([self._to_array(x) for x in X])
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if len(array) == 0:
            return
        for row in array:
            self.standardizer.update(row)
        standardized = self._standardize_array(array).astype(np.float32)
        batch = max(2, int(batch_size or self.config.batch_size))
        rng = np.random.default_rng(self.config.seed)
        head_params = [p for head in self.model.heads for p in head.parameters()]
        previous = [p.requires_grad for p in head_params]
        for p in head_params:
            p.requires_grad_(False)
        was_training = self.model.training
        try:
            for _ in range(max(1, int(epochs))):
                self.model.train()
                order = rng.permutation(len(standardized))
                for start in range(0, len(order), batch):
                    chunk = standardized[order[start : start + batch]]
                    if len(chunk) < 2:
                        continue
                    mask_np = rng.random(chunk.shape) < self.config.mask_fraction
                    empty = ~mask_np.any(axis=1)
                    if empty.any():
                        mask_np[empty, 0] = True
                    mask = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
                    bins = self._feature_bins(chunk)
                    _, ssl_logits, _ = self.model(self._tensor(chunk))
                    loss = F.cross_entropy(ssl_logits[mask], bins[mask])
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.last_loss = float(loss.detach())
                    self.loss_history.append(float(loss.detach()))
        finally:
            for p, flag in zip(head_params, previous):
                p.requires_grad_(flag)
            self.model.train(was_training)

    def signal_snapshot(
        self,
        X: Sequence[dict] | np.ndarray,
        window_index: int = 0,
    ) -> dict[str, Any]:
        """Deterministic label-free signals for one window of raw features."""
        if isinstance(X, np.ndarray):
            raw = np.asarray(X, dtype=np.float64)
        else:
            raw = np.vstack([self._to_array(x) for x in X])
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        standardized = self._standardize_array(raw).astype(np.float32)
        x_tensor = self._tensor(standardized)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                logits, ssl_logits, cls_out = self.model(x_tensor)
                probs = F.softmax(logits, dim=-1)
                rng = np.random.default_rng([self.config.seed, int(window_index)])
                mask_np = rng.random(standardized.shape) < self.config.mask_fraction
                empty = ~mask_np.any(axis=1)
                if empty.any():
                    mask_np[empty, 0] = True
                mask = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
                bins = self._feature_bins(standardized)
                ssl_loss = F.cross_entropy(ssl_logits[mask], bins[mask])
                active = probs[:, self.active_head, :].clamp_min(1e-12)
                entropy = -(
                    (active * active.log()).sum(dim=-1)
                    / math.log(max(2, self.n_classes))
                ).mean()
                cls = cls_out.detach().cpu().numpy().astype(np.float64)
        finally:
            self.model.train(was_training)
        return {
            "ssl_loss": float(ssl_loss.detach()),
            "head_disagreement": _mean_pairwise_js(probs),
            "entropy": float(entropy),
            "cls_mean": cls.mean(axis=0),
            "cls_var": cls.var(axis=0),
            "n_samples": int(len(raw)),
        }

    def model_bytes(self) -> int:
        return int(sum(p.numel() * p.element_size() for p in self.model.parameters()))

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "config": asdict(self.config),
            "shape": asdict(self.model.shape),
            "standardizer": {
                "mean": self.standardizer.mean,
                "m2": self.standardizer.m2,
                "count": self.standardizer.count,
            },
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        stats = payload.get("standardizer") or {}
        if stats:
            self.standardizer.mean = np.asarray(stats["mean"], dtype=np.float64)
            self.standardizer.m2 = np.asarray(stats["m2"], dtype=np.float64)
            self.standardizer.count = int(stats["count"])
