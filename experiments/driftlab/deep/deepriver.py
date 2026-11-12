"""DeepRiver rolling classifier wrapper (Exp-01/02 replica)."""

from __future__ import annotations

import pickle
from typing import Any

import torch


class _RNNHead(torch.nn.Module):
    def __init__(self, n_features: int, n_hidden: int = 16) -> None:
        super().__init__()
        self.rnn = torch.nn.RNN(
            input_size=n_features, hidden_size=n_hidden, batch_first=True
        )
        self.head = torch.nn.Linear(n_hidden, 2)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(X.permute(1, 0, 2))
        return self.head(out[:, -1, :])


class DeepRiverRolling:
    """Same architecture and workaround as experiments/scripts/exp02_seed_sweep_sea.py."""

    kind = "deepriver"

    def __init__(
        self,
        model_id: str,
        n_features: int,
        seed: int = 42,
        window_size: int = 500,
        lr: float = 1e-3,
        n_hidden: int = 16,
        device: str = "cpu",
        **_: Any,
    ):
        from deep_river.classification import RollingClassifier

        self.id = model_id
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.n_features = n_features

        class _TrainableRollingClassifier(RollingClassifier):
            def learn_one(self, x, y):  # noqa: D102
                self.module.train()
                return super().learn_one(x, y)

        torch.manual_seed(seed)
        self.model = _TrainableRollingClassifier(
            module=_RNNHead(n_features=n_features, n_hidden=n_hidden),
            loss_fn="cross_entropy",
            optimizer_fn="adam",
            lr=lr,
            device=device,
            seed=seed,
            window_size=window_size,
            is_class_incremental=False,
            is_feature_incremental=False,
        )

    def predict_one(self, x: dict) -> int | None:
        try:
            return self.model.predict_one(x)
        except Exception:
            return None

    def predict_proba_one(self, x: dict) -> dict[int, float] | None:
        try:
            return self.model.predict_proba_one(x)
        except Exception:
            return None

    def learn_one(self, x: dict, y: int, t: int | None = None) -> None:
        self.model.learn_one(x, y)

    def model_bytes(self) -> int:
        try:
            return len(pickle.dumps(self.model))
        except Exception:
            total = 0
            for param in self.model.module.parameters():
                total += param.numel() * param.element_size()
            return total
