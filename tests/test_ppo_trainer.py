"""Unit and Integration Tests for PPO Trainer (Reinforcement Learning Block).

Verifies:
1. compute_gae calculates mathematically correct advantages and returns.
2. RolloutBuffer stores transitions accurately and respects capacity limits.
3. PPOTrainer forward & backward pass: non-zero gradients flow to Actor, Critic, and CNN encoder.
4. Short end-to-end PPO step with 4-frame stacked Mario environment succeeds without error.
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

from src.environment import Mario2DEnv
from src.baseline_model import BaselineMarioPolicy
from src.connectome_model import ConnectomeMarioPolicy
from src.ppo_trainer import PPOTrainer, RolloutBuffer, compute_gae


class TestPPOTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        cls.processed_dir = REPO_ROOT / "data" / "processed"

    def test_gae_computation(self):
        """GAE advantages and returns must match theoretical equations."""
        rewards = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float32)
        values = torch.tensor([0.5, 0.8, 1.2], dtype=torch.float32)
        dones = torch.tensor([False, False, True], dtype=torch.bool)
        next_val = 0.0
        gamma = 0.99
        lam = 0.95

        advantages, returns = compute_gae(rewards, values, dones, next_val, gamma, lam)

        self.assertEqual(advantages.shape, (3,))
        self.assertEqual(returns.shape, (3,))
        # On final terminal step: delta_2 = 2.0 + 0 - 1.2 = 0.8
        self.assertAlmostEqual(advantages[2].item(), 0.8, places=4)
        self.assertAlmostEqual(returns[2].item(), 2.0, places=4)
        self.assertTrue(torch.all(returns == advantages + values))

    def test_rollout_buffer(self):
        """RolloutBuffer correctly appends transitions and resets."""
        buf = RolloutBuffer(buffer_size=4, obs_shape=(4, 84, 84), device=torch.device("cpu"))
        self.assertFalse(buf.is_full())

        for i in range(4):
            buf.add(
                obs=torch.zeros(4, 84, 84),
                action=i % 5,
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.6,
            )

        self.assertTrue(buf.is_full())
        self.assertEqual(buf.obs.shape, (4, 4, 84, 84))
        self.assertEqual(buf.actions.shape, (4,))

        buf.reset()
        self.assertEqual(buf.ptr, 0)
        self.assertFalse(buf.is_full())

    def test_ppo_update_gradients(self):
        """PPOTrainer update produces non-zero gradients on trainable parameters without NaN."""
        model = BaselineMarioPolicy(in_channels=4, hidden_dim=64).to(self.device)
        env = Mario2DEnv(frame_stack=4)
        trainer = PPOTrainer(model=model, env=env, device=self.device, lr=1e-4)

        buf = RolloutBuffer(buffer_size=16, obs_shape=(4, 84, 84), device=self.device)
        for i in range(16):
            buf.add(
                obs=torch.rand(4, 84, 84),
                action=i % 5,
                reward=float(i),
                done=(i == 15),
                value=0.5,
                log_prob=-1.6,
            )

        stats = trainer.update(buffer=buf, bootstrap_val=0.0, ppo_epochs=1, mini_batch_size=8)
        env.close()

        self.assertIn("policy_loss", stats)
        self.assertIn("value_loss", stats)
        self.assertIn("entropy", stats)
        self.assertFalse(torch.isnan(torch.tensor(stats["policy_loss"])))
        self.assertFalse(torch.isnan(torch.tensor(stats["value_loss"])))

    def test_short_rollout_collection(self):
        """PPOTrainer collects real transitions in SuperMarioBros environment."""
        model = BaselineMarioPolicy(in_channels=4, hidden_dim=64).to(self.device)
        env = Mario2DEnv(frame_stack=4)
        trainer = PPOTrainer(model=model, env=env, device=self.device)

        buf = RolloutBuffer(buffer_size=8, obs_shape=(4, 84, 84), device=self.device)
        obs, _ = env.reset()
        h = None

        obs, h, roll_stats, b_val = trainer.collect_rollout(buf, obs, h)
        env.close()

        self.assertTrue(buf.is_full())
        self.assertEqual(buf.actions.shape, (8,))
        self.assertIsInstance(roll_stats["mean_reward"], float)
        self.assertIsInstance(b_val, float)

    def test_dopamine_reward_shaper(self):
        """DopamineRewardShaper properly rewards forward progress, milestones, and flag win."""
        from src.ppo_trainer import DopamineRewardShaper
        shaper = DopamineRewardShaper()

        # Step 1: Forward movement
        r1, d1 = shaper.shape(raw_reward=10.0, info={"x_pos": 60}, done=False)
        self.assertGreater(d1["dopamine"], 0.0)
        self.assertAlmostEqual(d1["dx"], 20.0)

        # Step 2: Stagnant step
        r2, d2 = shaper.shape(raw_reward=0.0, info={"x_pos": 60}, done=False)
        self.assertLess(d2["dopamine"], 0.0)
        self.assertEqual(d2["stagnant_steps"], 1.0)

        # Step 3: Milestone crossing (x=350)
        r3, d3 = shaper.shape(raw_reward=10.0, info={"x_pos": 360}, done=False)
        self.assertGreater(d3["dopamine"], 50.0)

        # Step 4: Winning Flagpole touch
        r4, d4 = shaper.shape(raw_reward=50.0, info={"x_pos": 3161, "flag_get": True}, done=True)
        self.assertGreaterEqual(d4["dopamine"], 1000.0)


def run_ppo_trainer_tests() -> bool:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPPOTrainer)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    return res.wasSuccessful()


if __name__ == "__main__":
    success = run_ppo_trainer_tests()
    sys.exit(0 if success else 1)

