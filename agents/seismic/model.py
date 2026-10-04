"""The deep sequence model for the seismic agent.

Architecture: an LSTM encodes the 60-day feature sequence step by step
(giving every position an order-aware representation), a Transformer
encoder layer then lets each day attend to the whole window (useful for
e.g. relating a quiet recent week to an active week 40 days back), and a
linear+sigmoid head turns the most recent day's representation into a
single M5+/7-day probability.
"""

from __future__ import annotations

import torch
from torch import nn


class SeismicForecastModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int = 60,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        transformer_heads: int = 4,
        transformer_layers: int = 1,
        transformer_ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        # Learned positional embedding: the LSTM already encodes order via
        # recurrence, but giving the Transformer layer an explicit position
        # signal as well is cheap and standard practice.
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, lstm_hidden))
        nn.init.normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) -> logits: (batch,)."""
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out + self.pos_embedding[:, : lstm_out.size(1), :]
        attended = self.transformer(lstm_out)
        most_recent_day = attended[:, -1, :]
        return self.head(most_recent_day).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x))


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017), for severe class imbalance.

    alpha weights the positive class; gamma down-weights easy (already
    well-classified) examples so the loss focuses on hard ones -- useful
    here since the vast majority of cell-days are "quiet" and trivially
    correct to classify as negative.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class WeightedBCELoss(nn.Module):
    """Plain weighted binary cross-entropy, as a simpler alternative to focal loss."""

    def __init__(self, positive_weight: float) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(positive_weight))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
