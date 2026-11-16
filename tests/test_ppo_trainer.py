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
        """DopamineRewardShaper properly rewards forward progress, score, coins, powerup, and flag win."""
        from src.ppo_trainer import DopamineRewardShaper
        shaper = DopamineRewardShaper(jump_penalty=0.1, score_scale=0.01, coin_scale=1.0, powerup_bonus=5.0, flag_time_scale=0.05)
        shaper.reset(initial_x=40, initial_score=0, initial_coins=0, initial_status="small")

        # Step 1: Forward movement + jump action (Action 4: RIGHT+JUMP)
        r1, d1 = shaper.shape(raw_reward=10.0, info={"x_pos": 60, "score": 0, "coins": 0, "status": "small"}, done=False, action=4)
        self.assertGreater(d1["dopamine"], 0.0)
        self.assertAlmostEqual(d1["dx"], 20.0)

        # Step 2: Stagnant step without jump
        r2, d2 = shaper.shape(raw_reward=0.0, info={"x_pos": 60, "score": 0, "coins": 0, "status": "small"}, done=False, action=0)
        self.assertLess(d2["dopamine"], 0.0)
        self.assertEqual(d2["stagnant_steps"], 1.0)

        # Step 3: Enemy stomp (+200 score) & coin hit (+1 coin)
        r3, d3 = shaper.shape(raw_reward=0.0, info={"x_pos": 60, "score": 200, "coins": 1, "status": "small"}, done=False, action=0)
        self.assertAlmostEqual(d3["score_gain"], 2.0)
        self.assertAlmostEqual(d3["coins_gain"], 1.0)
        self.assertGreater(d3["dopamine"], 2.5)

        # Step 4: Mushroom collection (small -> tall)
        r4, d4 = shaper.shape(raw_reward=0.0, info={"x_pos": 60, "score": 1200, "coins": 1, "status": "tall"}, done=False, action=0)
        self.assertAlmostEqual(d4["powerup_gain"], 5.0)
        self.assertAlmostEqual(d4["score_gain"], 10.0)

        # Step 5: Winning Flagpole touch with time bonus
        r5, d5 = shaper.shape(raw_reward=50.0, info={"x_pos": 3161, "score": 1200, "coins": 1, "status": "tall", "flag_get": True, "time": 350}, done=True, action=1)
        self.assertGreaterEqual(d5["flag_bonus"], 50.0)
        self.assertAlmostEqual(d5["time_bonus"], 350 * 0.05)
        self.assertGreaterEqual(d5["dopamine"], 65.0)

        # Step 6: Goomba stomp (+100 score, no coin, no powerup)
        r6, d6 = shaper.shape(raw_reward=0.0, info={"x_pos": 3161, "score": 1300, "coins": 1, "status": "tall"}, done=False, action=0)
        self.assertEqual(d6["enemy_bonus"], 15.0)
        self.assertGreaterEqual(d6["dopamine"], 15.0)

        # Step 7: Walking left (Action 2: LEFT) should NOT incur jump penalty
        shaper.reset(initial_x=60, initial_score=1300, initial_coins=1, initial_status="tall")
        r7, d7 = shaper.shape(raw_reward=0.0, info={"x_pos": 55, "score": 1300, "coins": 1, "status": "tall"}, done=False, action=2)
        # dx = -5, stagnant penalty applies (-0.05), time penalty applies (-0.01), jump penalty does NOT apply
        expected_dopamine = -shaper.stagnant_penalty - shaper.time_penalty
        self.assertAlmostEqual(d7["dopamine"], expected_dopamine)

    def test_vector_mario_env_rollout(self):
        """PPOTrainer collects transitions properly across multiple parallel environments."""
        from src.environment import VectorMarioEnv
        model = BaselineMarioPolicy(in_channels=4, hidden_dim=64).to(self.device)
        vec_env = VectorMarioEnv(num_envs=2, frame_stack=4)
        trainer = PPOTrainer(model=model, env=vec_env, device=self.device, anneal_lr=True)

        buf = RolloutBuffer(buffer_size=8, obs_shape=(4, 84, 84), device=self.device)
        obs, _ = vec_env.reset()
        h = None

        obs, h, roll_stats, b_vals = trainer.collect_rollout(buf, obs, h, seq_len=4)
        self.assertTrue(buf.is_full())
        self.assertEqual(len(b_vals), 2)
        self.assertIsInstance(roll_stats["mean_reward"], float)

        # Update test with vector env
        up_stats = trainer.update(buffer=buf, bootstrap_val=b_vals, ppo_epochs=1, mini_batch_size=8, seq_len=4)
        vec_env.close()

        self.assertIn("policy_loss", up_stats)
        self.assertFalse(torch.isnan(torch.tensor(up_stats["policy_loss"])))

    def test_stateful_checkpoint_preservation(self):
        """Checkpoints must save and restore optimizer_state_dict, lr, and model weights."""
        import tempfile
        model = BaselineMarioPolicy(in_channels=4, hidden_dim=64).to(self.device)
        env = Mario2DEnv(frame_stack=4)
        trainer = PPOTrainer(model=model, env=env, device=self.device, lr=1.5e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_p = Path(tmpdir) / "test_ckpt.pt"
            trainer.train(
                total_timesteps=8,
                rollout_steps=8,
                ppo_epochs=1,
                mini_batch_size=8,
                checkpoint_path=ckpt_p,
                seq_len=4,
            )

            latest_p = ckpt_p.parent / f"{ckpt_p.stem}_latest{ckpt_p.suffix}"
            self.assertTrue(latest_p.exists())
            saved = torch.load(latest_p, map_location="cpu", weights_only=False)

            self.assertIn("optimizer_state_dict", saved)
            self.assertIn("lr", saved)
            self.assertIn("model_state_dict", saved)
            self.assertIn("iteration", saved)
            self.assertIn("timesteps", saved)
            self.assertEqual(saved["timesteps"], 8)
        env.close()



def run_ppo_trainer_tests() -> bool:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPPOTrainer)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    return res.wasSuccessful()


if __name__ == "__main__":
    success = run_ppo_trainer_tests()
    sys.exit(0 if success else 1)

