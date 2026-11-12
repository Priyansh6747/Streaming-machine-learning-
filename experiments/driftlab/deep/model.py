"""DriftFormer: a compact feature-token transformer with a head bank."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DriftFormerShape:
    n_features: int
    n_classes: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ffn: int = 128
    n_heads_bank: int = 4
    n_bins: int = 8
    dropout: float = 0.0


class DriftFormer(nn.Module):
    def __init__(self, shape: DriftFormerShape):
        super().__init__()
        self.shape = shape
        d = shape.d_model
        self.feature_weight = nn.Parameter(torch.randn(shape.n_features, d) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(shape.n_features, d))
        self.position = nn.Parameter(torch.randn(shape.n_features + 1, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=shape.n_heads,
            dim_feedforward=shape.ffn,
            dropout=shape.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=shape.n_layers)
        self.heads = nn.ModuleList(
            [nn.Linear(d, shape.n_classes) for _ in range(shape.n_heads_bank)]
        )
        self.ssl_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, shape.n_bins),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, n_features = x.shape
        if n_features != self.shape.n_features:
            raise ValueError(f"expected {self.shape.n_features} features, got {n_features}")
        tokens = x.unsqueeze(-1) * self.feature_weight + self.feature_bias
        cls = self.cls.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.position.unsqueeze(0)
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0]
        feature_out = encoded[:, 1:]
        logits = torch.stack([head(cls_out) for head in self.heads], dim=1)
        ssl_logits = self.ssl_head(feature_out)
        return logits, ssl_logits, cls_out

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
