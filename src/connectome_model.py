"""MaleCNS Connectome-Constrained Recurrent Neural Network for Mario 2D.

Architecture:
1. MarioCNNEncoder: Visual feature extractor (1x84x84 -> 512)
2. ConnectomeInputAdapter: Projects visual features into sensory (Visual Projection) neurons
3. ConnectomeRecurrentLayer: Fixed sparse directed MaleCNS topology (30,000 neurons)
4. ActionDecoder: Decodes Descending Neuron premotor activity into 5 discrete actions
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarioCNNEncoder(nn.Module):
    """Nature-style CNN visual encoder for 84x84 grayscale frames."""

    def __init__(self, in_channels: int = 1, out_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 84, 84)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.reshape(h.size(0), -1)
        return F.relu(self.fc(h))


class ConnectomeRecurrentLayer(nn.Module):
    """Recurrent neural layer constrained by fixed sparse MaleCNS topology.

    Message passing convention:
        A[i, j] = connection strength from presynaptic neuron i to postsynaptic neuron j.
        Postsynaptic drive into neuron j = sum_i h_i * A[i, j].
        In PyTorch: W = A.T, incoming = (W @ h.T).T.
    """

    def __init__(
        self,
        sparse_adj: sp.csr_matrix,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.num_nodes = sparse_adj.shape[0]
        self.alpha = alpha

        # Transpose so that W @ x computes the incoming postsynaptic sum
        adj_t = sparse_adj.T.tocoo()
        indices = torch.tensor(np.vstack([adj_t.row, adj_t.col]), dtype=torch.long)
        values = torch.tensor(adj_t.data, dtype=torch.float32)

        # Register fixed sparse topology as non-trainable buffer
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, (self.num_nodes, self.num_nodes)
        ).coalesce()
        self.register_buffer("W_rec", sparse_tensor)

        # Trainable per-neuron intrinsic excitability / bias
        self.bias = nn.Parameter(torch.zeros(self.num_nodes, dtype=torch.float32))

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_nodes, dtype=torch.float32, device=device)

    def forward(
        self,
        u_in: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """One recurrent step.

        Args:
            u_in: (B, N) input drive injected into neurons
            h: (B, N) previous recurrent state
        Returns:
            h_next: (B, N) updated recurrent state
        """
        # Recurrent message passing: (N, N) @ (N, B) -> (N, B) -> transpose to (B, N)
        # Note: torch.sparse.mm on MPS or CPU
        incoming = torch.sparse.mm(self.W_rec, h.T).T

        # Continuous-rate leaky-integrator recurrent dynamics
        total_drive = incoming + u_in + self.bias
        h_next = (1.0 - self.alpha) * h + self.alpha * torch.tanh(total_drive)
        return h_next


class ConnectomeMarioPolicy(nn.Module):
    """End-to-end connectome-constrained policy model for Mario."""

    def __init__(
        self,
        subgraph_npz_path: Path | str,
        node_mapping_csv_path: Path | str,
        node_metadata_csv_path: Path | str,
        num_actions: int = 5,
        in_channels: int = 1,
        cnn_dim: int = 512,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        # Load sparse graph
        sub_csr = sp.load_npz(str(subgraph_npz_path))
        self.num_nodes = sub_csr.shape[0]

        # Load node mappings and metadata to identify sensory and motor neurons
        node_map = pd.read_csv(str(node_mapping_csv_path))
        metadata = pd.read_csv(str(node_metadata_csv_path))

        merged = node_map.merge(metadata, on="body_id", how="left")
        vp_mask = (merged["superclass"] == "visual_projection").to_numpy()
        dn_mask = (merged["superclass"] == "descending_neuron").to_numpy()

        vp_indices = np.flatnonzero(vp_mask)
        dn_indices = np.flatnonzero(dn_mask)

        # Fallback if specific classes are not isolated: use first K / last K nodes
        if len(vp_indices) == 0:
            vp_indices = np.arange(min(1000, self.num_nodes))
        if len(dn_indices) == 0:
            dn_indices = np.arange(max(0, self.num_nodes - 1000), self.num_nodes)

        self.register_buffer("sensory_indices", torch.tensor(vp_indices, dtype=torch.long))
        self.register_buffer("motor_indices", torch.tensor(dn_indices, dtype=torch.long))

        self.num_sensory = len(vp_indices)
        self.num_motor = len(dn_indices)

        # Trainable modules
        self.encoder = MarioCNNEncoder(in_channels=in_channels, out_dim=cnn_dim)
        self.adapter = nn.Linear(cnn_dim, self.num_sensory)
        self.rnn = ConnectomeRecurrentLayer(sub_csr, alpha=alpha)
        self.actor = nn.Linear(self.num_motor, num_actions)
        self.critic = nn.Linear(self.num_motor, 1)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.rnn.init_hidden(batch_size, device)

    def forward(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        num_ticks: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward step.

        Args:
            obs: (B, 1, 84, 84) observation tensor
            h: (B, N) previous recurrent state
            num_ticks: number of internal connectome recurrent message passing ticks per frame
        Returns:
            logits: (B, 5) action logits
            value: (B, 1) state value estimate
            h_next: (B, N) updated recurrent state
        """
        B = obs.size(0)
        # 1. CNN visual encoding
        features = self.encoder(obs)  # (B, 512)

        # 2. Input adaptation into sensory neurons
        sensory_drive = self.adapter(features)  # (B, N_sensory)

        # Embed into full (B, N) injection vector
        u_in = torch.zeros(B, self.num_nodes, device=obs.device, dtype=torch.float32)
        u_in[:, self.sensory_indices] = sensory_drive

        # 3. Fixed MaleCNS recurrent propagation across internal ticks
        if h is None:
            h = self.init_hidden(B, obs.device)
        h_current = h
        for _ in range(num_ticks):
            h_current = self.rnn(u_in, h_current)

        # 4. Motor readout from descending neurons
        motor_activity = h_current[:, self.motor_indices]
        logits = self.actor(motor_activity)
        value = self.critic(motor_activity)

        return logits, value, h_current
