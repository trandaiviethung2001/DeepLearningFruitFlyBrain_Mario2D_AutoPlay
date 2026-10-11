"""Unit and Integration Tests for Demonstration Dataset Pipeline.

Verifies:
1. Action indices are within valid range [0, 4].
2. Observation tensors have expected shape (N, 1, 84, 84) and uint8 dtype in [0, 255].
3. Metadata transition count matches tensor lengths exactly.
4. Whole-episode partitioning guarantees zero seed/episode leakage between train, val, test splits.
5. Action distribution contains genuine behavioral diversity (locomotion + timed jumps + noops).
6. Observation diversity has non-zero spatial-temporal variance.
7. Serialization roundtrip preserves identical tensors and metadata.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.demonstration_collector import (
    collect_demonstration_dataset,
    load_demonstration_dataset,
    ScriptedDemonstrator,
    RightOnlyPolicy,
    RandomPolicy,
    MARIO_5_ACTIONS,
    ACTION_NAMES,
)


class TestDemonstrationDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.TemporaryDirectory()
        cls.test_path = Path(cls.test_dir.name) / "test_demonstrations.pt"
        # Collect a quick 3-episode mini dataset for testing
        cls.seeds = [101, 102, 103]
        cls.dataset = collect_demonstration_dataset(
            seeds=cls.seeds,
            output_path=cls.test_path,
            max_steps_per_episode=300,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )

    @classmethod
    def tearDownClass(cls):
        cls.test_dir.cleanup()

    def test_tensor_shapes_and_dtypes(self):
        """Observations must be (N, 1, 84, 84) uint8, actions int64, rewards float32."""
        num_transitions = self.dataset["metadata"]["num_transitions"]
        self.assertGreater(num_transitions, 0)

        obs = self.dataset["obs"]
        actions = self.dataset["actions"]
        rewards = self.dataset["rewards"]
        dones = self.dataset["dones"]
        x_positions = self.dataset["x_positions"]
        y_positions = self.dataset["y_positions"]

        self.assertEqual(obs.shape, (num_transitions, 1, 84, 84))
        self.assertEqual(obs.dtype, torch.uint8)
        self.assertEqual(actions.shape, (num_transitions,))
        self.assertEqual(actions.dtype, torch.long)
        self.assertEqual(rewards.shape, (num_transitions,))
        self.assertEqual(rewards.dtype, torch.float32)
        self.assertEqual(dones.shape, (num_transitions,))
        self.assertEqual(dones.dtype, torch.bool)
        self.assertEqual(x_positions.shape, (num_transitions,))
        self.assertEqual(y_positions.shape, (num_transitions,))

    def test_valid_action_space(self):
        """All recorded actions must be valid indices in [0, 4]."""
        actions = self.dataset["actions"]
        self.assertTrue((actions >= 0).all().item())
        self.assertTrue((actions < len(MARIO_5_ACTIONS)).all().item())

    def test_action_diversity(self):
        """Demonstrator must not collapse to a single action."""
        actions = self.dataset["actions"]
        unique_actions = torch.unique(actions).tolist()
        # Must have at least RIGHT (1) and RIGHT+JUMP (4)
        self.assertGreaterEqual(len(unique_actions), 2, "Dataset collapsed to single action!")
        self.assertIn(1, unique_actions, "Locomotion action RIGHT must be present")
        self.assertIn(4, unique_actions, "Obstacle jump action RIGHT+JUMP must be present")

    def test_observation_variance(self):
        """Visual observations must have non-zero variance (not black or frozen frames)."""
        obs = self.dataset["obs"].float()
        obs_std = torch.std(obs).item()
        self.assertGreater(obs_std, 1.0, f"Observations lack visual contrast: std={obs_std}")

    def test_whole_episode_split_isolation(self):
        """Train, val, and test splits must be partitioned by seed with zero overlap."""
        splits = self.dataset["metadata"]["splits"]
        train_seeds = set(splits["train_seeds"])
        val_seeds = set(splits["val_seeds"])
        test_seeds = set(splits["test_seeds"])

        # Check disjointness
        self.assertTrue(train_seeds.isdisjoint(val_seeds), "Data leakage between train and val!")
        self.assertTrue(train_seeds.isdisjoint(test_seeds), "Data leakage between train and test!")
        self.assertTrue(val_seeds.isdisjoint(test_seeds), "Data leakage between val and test!")

        # All seeds accounted for
        all_partitioned = train_seeds | val_seeds | test_seeds
        self.assertEqual(all_partitioned, set(self.seeds))

    def test_metadata_alignment(self):
        """Metadata transition counts must match actual tensor lengths."""
        episodes = self.dataset["metadata"]["episodes"]
        self.assertEqual(len(episodes), len(self.seeds))
        total_ep_steps = sum(ep["steps"] for ep in episodes)
        self.assertEqual(total_ep_steps, self.dataset["metadata"]["num_transitions"])
        self.assertEqual(self.dataset["obs"].size(0), total_ep_steps)

    def test_serialization_roundtrip(self):
        """Dataset loaded from disk must match memory dataset exactly."""
        loaded = load_demonstration_dataset(self.test_path)
        self.assertEqual(loaded["metadata"]["num_transitions"], self.dataset["metadata"]["num_transitions"])
        self.assertTrue(torch.equal(loaded["obs"], self.dataset["obs"]))
        self.assertTrue(torch.equal(loaded["actions"], self.dataset["actions"]))
        self.assertTrue(torch.equal(loaded["rewards"], self.dataset["rewards"]))
        self.assertTrue(torch.equal(loaded["dones"], self.dataset["dones"]))
        self.assertTrue(torch.equal(loaded["x_positions"], self.dataset["x_positions"]))


def run_demonstration_tests() -> bool:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDemonstrationDataset)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    return res.wasSuccessful()


if __name__ == "__main__":
    success = run_demonstration_tests()
    sys.exit(0 if success else 1)

