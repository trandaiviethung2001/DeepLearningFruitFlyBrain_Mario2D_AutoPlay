"""Deterministic tests for action mapping, jumping, termination, reward and reset semantics.

Run: .venv/bin/python -m pytest tests/test_env_semantics.py -q
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.environment import Mario2DEnv

NOOP, RIGHT, LEFT, JUMP, RIGHT_JUMP = range(5)


def _run(env, actions, seed=None):
    obs, info = env.reset(seed=seed)
    xs, ys, last = [], [], (False, False, info)
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        xs.append(info["x_pos"]); ys.append(info["y_pos"])
        last = (term, trunc, info)
        if term or trunc:
            break
    return xs, ys, last


def _takeoffs(ys):
    ground, n, air = min(ys[:3]), 0, False
    for y in ys:
        if not air and y > ground + 2:
            n, air = n + 1, True
        elif air and y <= ground:
            air = False
    return n


class TestEnvSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = Mario2DEnv(frame_stack=4)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_action_mapping(self):
        """RIGHT increases x, LEFT does not, JUMP raises y."""
        xs, _, _ = _run(self.env, [RIGHT] * 60)
        self.assertGreater(xs[-1], xs[0] + 20)
        xs, _, _ = _run(self.env, [LEFT] * 60)
        self.assertLessEqual(xs[-1], 45)
        _, ys, _ = _run(self.env, [NOOP] * 10 + [JUMP] * 20 + [NOOP] * 20)
        self.assertGreater(max(ys), ys[5] + 20)

    def test_right_jump_moves_right_while_airborne(self):
        xs, ys, _ = _run(self.env, [NOOP] * 10 + [RIGHT_JUMP] * 30)
        self.assertGreater(max(ys), ys[5] + 30)
        self.assertGreater(xs[-1], xs[9] + 10)

    def test_held_A_jumps_once_release_rejumps(self):
        """Holding A never re-triggers a jump; releasing for one frame does."""
        _, ys_hold, _ = _run(self.env, [RIGHT_JUMP] * 140)
        self.assertEqual(_takeoffs(ys_hold), 1)
        _, ys_pulse, _ = _run(self.env, ([RIGHT_JUMP] * 30 + [RIGHT]) * 4)
        self.assertGreaterEqual(_takeoffs(ys_pulse), 2)

    def test_jump_height_increases_with_hold(self):
        h = {}
        for k in (1, 8, 32):
            _, ys, _ = _run(self.env, [NOOP] * 30 + [JUMP] * k + [NOOP] * (60 - k))
            h[k] = max(ys) - ys[29]
        self.assertLess(h[1], h[8])
        self.assertLess(h[8], h[32])
        self.assertLess(h[1], 32, "a 1-frame tap must be lower than a 2-tile (32 px) pipe")

    def test_death_is_termination_with_penalty(self):
        xs, _, (term, trunc, info) = _run(self.env, [RIGHT] * 1000)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertTrue(info["death"])
        self.assertEqual(max(xs), 296)
        self.assertLess(info["reward_components_sum"]["death"], 0)

    def test_seed_without_noops_is_deterministic(self):
        a, _, _ = _run(self.env, [RIGHT] * 50, seed=1)
        b, _, _ = _run(self.env, [RIGHT] * 50, seed=2)
        self.assertEqual(a, b)

    def test_noop_starts_make_seeds_matter_and_reproducible(self):
        env = Mario2DEnv(frame_stack=4, noop_max=120)
        times = set()
        for s in range(10):
            _, info = env.reset(seed=s)
            times.add(info.get("time"))
        self.assertGreater(len(times), 1, "different seeds should produce different starting clocks")
        # Determinism given the same seed
        _, i1 = env.reset(seed=3)
        _, i2 = env.reset(seed=3)
        self.assertEqual(i1.get("time"), i2.get("time"))
        env.close()

    def test_frame_skip_sums_rewards_and_components(self):
        env = Mario2DEnv(frame_stack=4, frame_skip=4)
        obs, info = env.reset()
        for _ in range(20):
            obs, r, term, trunc, info = env.step(RIGHT)
            comp = info["reward_components_sum"]
            # env clips each frame's total to [-15, 15]; components are unclipped
            self.assertAlmostEqual(r, sum(comp.values()), delta=1e-5) if abs(sum(comp.values())) < 15 else None
            self.assertEqual(info["frames"], 4)
        self.assertEqual(obs.shape, (4, 84, 84))
        # last stacked channel == current screen
        cur = env.preprocess_frame(env.env.unwrapped.screen.copy())
        self.assertTrue(torch.allclose(obs[-1:], cur))
        env.close()

    def test_held_A_across_skipped_steps(self):
        """With frame_skip, consecutive RIGHT+JUMP steps keep A held (one take-off)."""
        env = Mario2DEnv(frame_stack=4, frame_skip=4)
        _, ys, _ = _run(env, [RIGHT_JUMP] * 35)
        self.assertEqual(_takeoffs(ys), 1)
        _, ys, _ = _run(env, ([RIGHT_JUMP] * 8 + [RIGHT]) * 4)
        self.assertGreaterEqual(_takeoffs(ys), 2)
        env.close()

    def test_stall_truncation_is_truncation_not_termination(self):
        env = Mario2DEnv(frame_stack=1, stall_frames=120)
        _, _, (term, trunc, info) = _run(env, [NOOP] * 500)
        env.close()
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(info["truncation_reason"], "stall")

    def test_snapshot_start_state(self):
        env = Mario2DEnv(frame_stack=4)
        env.reset()
        for _ in range(100):
            _, _, _, _, info = env.step(RIGHT)
        x_snap = info["x_pos"]
        env.set_start_states([env.snapshot()])
        xs0 = set()
        for s in range(8):
            _, info = env.reset(seed=s)
            xs0.add(info["x_pos"])
            _, r, term, trunc, info2 = env.step(NOOP)
            self.assertFalse(term)
            self.assertLessEqual(abs(r), 15)  # no spurious progress reward from the jump in x
        self.assertTrue(any(abs(x - x_snap) <= 3 for x in xs0))
        _, info = env.reset(seed=0, use_start_states=False)
        self.assertLess(info["x_pos"], 60)
        env.close()


def run_env_semantics_tests() -> bool:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnvSemantics)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


if __name__ == "__main__":
    sys.exit(0 if run_env_semantics_tests() else 1)
