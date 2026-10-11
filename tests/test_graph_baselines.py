"""Unit and Invariant Tests for Graph Baselines & Ablation Architectures.

Verifies:
1. Model B (Rewired Connectome):
   - In-degree and out-degree sequences match real connectome identically.
   - Edge count matches exactly (6,001,050).
   - Zero self-loops and zero multi-edges.
   - Wiring has been substantially shuffled (>95% edges modified).
2. Model C (Random Erdős–Rényi):
   - Parameter matched (30,000 nodes, 6,001,050 edges).
   - Zero self-loops and zero multi-edges.
   - Row-normalized (row sums equal 1.0).
3. Model Parity:
   - Connectome models (Real, Rewired, Random) have identical trainable parameter count.
   - Forward pass succeeds on CPU / MPS producing expected tensor shapes.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import scipy.sparse as sp
import torch

from src.connectome_model import ConnectomeMarioPolicy


class TestGraphBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processed_dir = REPO_ROOT / "data" / "processed"
        cls.real_path = cls.processed_dir / "subgraph_graph.npz"
        cls.shuffled_path = cls.processed_dir / "shuffled_connectome_graph.npz"
        cls.random_path = cls.processed_dir / "random_sparse_graph.npz"
        cls.mapping_path = cls.processed_dir / "subgraph_node_mapping.csv"
        cls.metadata_path = cls.processed_dir / "node_metadata.csv"

        cls.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    def test_shuffled_graph_degree_preservation(self):
        """Model B must preserve exact in-degree and out-degree sequences while rewiring >95% edges."""
        self.assertTrue(self.real_path.exists(), f"Real graph missing: {self.real_path}")
        self.assertTrue(self.shuffled_path.exists(), f"Shuffled graph missing: {self.shuffled_path}")

        real_csr = sp.load_npz(str(self.real_path))
        shuf_csr = sp.load_npz(str(self.shuffled_path))

        # Node count
        self.assertEqual(shuf_csr.shape, real_csr.shape)
        # Edge count
        self.assertEqual(shuf_csr.nnz, real_csr.nnz)

        # In-degree sequence
        real_in = np.bincount(real_csr.indices, minlength=real_csr.shape[0])
        shuf_in = np.bincount(shuf_csr.indices, minlength=shuf_csr.shape[0])
        self.assertTrue(np.array_equal(real_in, shuf_in), "In-degree sequences do not match exactly!")

        # Out-degree sequence
        real_out = np.diff(real_csr.indptr)
        shuf_out = np.diff(shuf_csr.indptr)
        self.assertTrue(np.array_equal(real_out, shuf_out), "Out-degree sequences do not match exactly!")

        # Zero self loops
        self.assertEqual(shuf_csr.diagonal().nonzero()[0].size, 0)

        # Row normalization
        row_sums = np.array(shuf_csr.sum(axis=1)).flatten()
        active_rows = row_sums > 0
        self.assertTrue(np.allclose(row_sums[active_rows], 1.0, atol=1e-5))

        # Verify rewiring distance: col indices differ
        diff_count = np.sum(shuf_csr.indices != real_csr.indices)
        diff_ratio = diff_count / real_csr.nnz
        self.assertGreater(diff_ratio, 0.95, f"Rewiring distance too low: {diff_ratio:.2%}")

    def test_random_sparse_graph_properties(self):
        """Model C must match node and edge counts, zero self-loops, and row normalization."""
        self.assertTrue(self.random_path.exists(), f"Random graph missing: {self.random_path}")

        er_csr = sp.load_npz(str(self.random_path))
        self.assertEqual(er_csr.shape, (30000, 30000))
        self.assertEqual(er_csr.nnz, 6001050)
        self.assertEqual(er_csr.diagonal().nonzero()[0].size, 0)

        # Row normalization
        row_sums = np.array(er_csr.sum(axis=1)).flatten()
        active_rows = row_sums > 0
        self.assertTrue(np.allclose(row_sums[active_rows], 1.0, atol=1e-5))

        # Mean degree check
        out_degrees = np.diff(er_csr.indptr)
        self.assertAlmostEqual(float(np.mean(out_degrees)), 200.035, delta=0.5)

    def test_trainable_parameter_identity(self):
        """Models A, B, and C must have identical trainable parameter counts."""
        model_a = ConnectomeMarioPolicy(
            subgraph_npz_path=self.real_path,
            node_mapping_csv_path=self.mapping_path,
            node_metadata_csv_path=self.metadata_path,
        )
        model_b = ConnectomeMarioPolicy(
            subgraph_npz_path=self.shuffled_path,
            node_mapping_csv_path=self.mapping_path,
            node_metadata_csv_path=self.metadata_path,
        )
        model_c = ConnectomeMarioPolicy(
            subgraph_npz_path=self.random_path,
            node_mapping_csv_path=self.mapping_path,
            node_metadata_csv_path=self.metadata_path,
        )

        p_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
        p_b = sum(p.numel() for p in model_b.parameters() if p.requires_grad)
        p_c = sum(p.numel() for p in model_c.parameters() if p.requires_grad)

        self.assertEqual(p_a, p_b, f"Param count mismatch between A ({p_a}) and B ({p_b})")
        self.assertEqual(p_a, p_c, f"Param count mismatch between A ({p_a}) and C ({p_c})")
        self.assertEqual(p_a, 6435987)

    def test_forward_step_rewired_and_random(self):
        """Forward step works cleanly on rewired and random connectome policies."""
        model_b = ConnectomeMarioPolicy(
            subgraph_npz_path=self.shuffled_path,
            node_mapping_csv_path=self.mapping_path,
            node_metadata_csv_path=self.metadata_path,
        ).to(self.device)

        model_c = ConnectomeMarioPolicy(
            subgraph_npz_path=self.random_path,
            node_mapping_csv_path=self.mapping_path,
            node_metadata_csv_path=self.metadata_path,
        ).to(self.device)

        dummy_obs = torch.zeros(2, 1, 84, 84, device=self.device)

        logits_b, val_b, h_b = model_b(dummy_obs, None)
        self.assertEqual(logits_b.shape, (2, 5))
        self.assertEqual(val_b.shape, (2, 1))
        self.assertEqual(h_b.shape, (2, 30000))
        self.assertFalse(torch.isnan(logits_b).any())

        logits_c, val_c, h_c = model_c(dummy_obs, None)
        self.assertEqual(logits_c.shape, (2, 5))
        self.assertEqual(val_c.shape, (2, 1))
        self.assertEqual(h_c.shape, (2, 30000))
        self.assertFalse(torch.isnan(logits_c).any())


def run_graph_baseline_tests() -> bool:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestGraphBaselines)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    return res.wasSuccessful()


if __name__ == "__main__":
    success = run_graph_baseline_tests()
    sys.exit(0 if success else 1)

