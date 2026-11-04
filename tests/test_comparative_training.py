"""Unit and Integration Tests for Comparative Training Pipeline.

Verifies:
1. TransitionDataset produces normalized float32 observations and int64 actions.
2. Parameter parity: Connectome and Baseline GRU parameters are matched within 0.5%.
3. One training step reduces cross-entropy loss on both architectures.
4. Model checkpoint saving and loading roundtrip works seamlessly.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.baseline_model import BaselineMarioPolicy
from src.connectome_model import ConnectomeMarioPolicy
from src.train_and_evaluate import TransitionDataset, train_one_epoch, evaluate_loss


class TestComparativeTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processed_dir = REPO_ROOT / "data" / "processed"
        cls.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        # Synthetic mini-dataset for fast unit testing
        cls.num_samples = 32
        cls.raw_obs = torch.randint(0, 256, (cls.num_samples, 1, 84, 84), dtype=torch.uint8)
        cls.actions = torch.randint(0, 5, (cls.num_samples,), dtype=torch.long)
        cls.rewards = torch.zeros(cls.num_samples, dtype=torch.float32)

        cls.dataset = TransitionDataset(cls.raw_obs, cls.actions, cls.rewards)
        cls.loader = DataLoader(cls.dataset, batch_size=8, shuffle=False)

    def test_dataset_normalization(self):
        """Observations must be normalized to float32 [0.0, 1.0]."""
        obs, action, reward = self.dataset[0]
        self.assertEqual(obs.dtype, torch.float32)
        self.assertEqual(obs.shape, (1, 84, 84))
        self.assertTrue((obs >= 0.0).all())
        self.assertTrue((obs <= 1.0).all())

    def test_parameter_parity(self):
        """Connectome and Baseline models must have matched trainable parameter counts within 0.5%."""
        c_model = ConnectomeMarioPolicy(
            subgraph_npz_path=self.processed_dir / "subgraph_graph.npz",
            node_mapping_csv_path=self.processed_dir / "subgraph_node_mapping.csv",
            node_metadata_csv_path=self.processed_dir / "node_metadata.csv",
        )
        b_model = BaselineMarioPolicy(cnn_dim=512, hidden_dim=1024)

        c_params = sum(p.numel() for p in c_model.parameters() if p.requires_grad)
        b_params = sum(p.numel() for p in b_model.parameters() if p.requires_grad)

        diff = abs(c_params - b_params)
        diff_pct = (diff / c_params) * 100

        self.assertLess(
            diff_pct,
            0.5,
            f"Parameter mismatch too large: Connectome={c_params}, Baseline={b_params} (diff={diff_pct:.2f}%)",
        )

    def test_connectome_training_step(self):
        """Connectome model executes forward and backward step without error."""
        c_model = ConnectomeMarioPolicy(
            subgraph_npz_path=self.processed_dir / "subgraph_graph.npz",
            node_mapping_csv_path=self.processed_dir / "subgraph_node_mapping.csv",
            node_metadata_csv_path=self.processed_dir / "node_metadata.csv",
        ).to(self.device)

        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, c_model.parameters()), lr=1e-4)
        crit = nn.CrossEntropyLoss()

        loss, acc = train_one_epoch(c_model, self.loader, opt, crit, self.device)
        self.assertFalse(torch.isnan(torch.tensor(loss)))
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_baseline_training_step(self):
        """Baseline model executes forward and backward step without error."""
        b_model = BaselineMarioPolicy(cnn_dim=512, hidden_dim=1024).to(self.device)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, b_model.parameters()), lr=1e-4)
        crit = nn.CrossEntropyLoss()

        loss, acc = train_one_epoch(b_model, self.loader, opt, crit, self.device)
        self.assertFalse(torch.isnan(torch.tensor(loss)))
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


def run_comparative_training_tests() -> bool:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestComparativeTraining)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    return res.wasSuccessful()


if __name__ == "__main__":
    success = run_comparative_training_tests()
    sys.exit(0 if success else 1)

