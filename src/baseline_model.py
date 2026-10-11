"""Standard unconstrained CNN-RNN baseline for Mario 2D.

Serves as the controlled scientific baseline for evaluating whether the
MaleCNS connectome constraint provides beneficial inductive bias.
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
from src.connectome_model import MarioCNNEncoder


class BaselineMarioPolicy(nn.Module):
    """Unconstrained standard CNN-GRU baseline for Mario 2D."""

    def __init__(
        self,
        num_actions: int = 5,
        in_channels: int = 1,
        cnn_dim: int = 512,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.encoder = MarioCNNEncoder(in_channels=in_channels, out_dim=cnn_dim)
        self.gru = nn.GRUCell(cnn_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32, device=device)

    def forward(
        self,
        obs: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if h is None:
            h = self.init_hidden(obs.shape[0], obs.device)

        # 1. Visual encoding
        features = self.encoder(obs)  # (B, 512)

        # 2. Standard unconstrained GRU recurrence
        h_next = self.gru(features, h)  # (B, hidden_dim)

        # 3. Action logits & value estimation
        logits = self.actor(h_next)
        value = self.critic(h_next)

        return logits, value, h_next

