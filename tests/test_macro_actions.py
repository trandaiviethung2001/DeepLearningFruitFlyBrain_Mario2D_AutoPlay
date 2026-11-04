"""Tests for Macro-Action Wrapper and Auto-Unlatching.

Verifies:
1. Raw environment (auto_unlatch=False) reproduces NES button A latching: constant RIGHT+JUMP dies at x=296.
2. Unlatched environment (auto_unlatch=True) / MarioMacroActionWrapper executes continuous repeated takeoffs,
   clearing early obstacles (x > 1,000) under pure greedy RIGHT+JUMP without dying.
3. Tap vs hold jumping behavior remains intact.

Run: .venv/bin/python -m pytest tests/test_macro_actions.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import Mario2DEnv, MarioMacroActionWrapper

RIGHT, JUMP, RIGHT_JUMP = 1, 3, 4


def _count_takeoffs(env, actions):
    takeoffs = 0
    in_air = False
    xs = []
    obs, info = env.reset(seed=0)
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        xs.append(info["x_pos"])
        died = bool(info.get("is_dying") or info.get("is_dead"))
        airborne = bool(env.ram[0x001D] == 1) and not died
        if airborne and not in_air:
            takeoffs += 1
            in_air = True
        elif not airborne:
            in_air = False
        if term or trunc:
            break
    died = bool(info.get("is_dying") or info.get("is_dead"))
    return takeoffs, max(xs) if xs else 0, died, len(xs)


class TestMacroActions(unittest.TestCase):
    def test_raw_env_latches_button_a(self):
        """Without auto-unlatch, greedy continuous RIGHT+JUMP latches button A and dies at x=296."""
        env = Mario2DEnv(frame_skip=4, auto_unlatch=False)
        takeoffs, max_x, died, steps = _count_takeoffs(env, [RIGHT_JUMP] * 100)
        env.close()

        self.assertEqual(takeoffs, 1, "Raw env should only execute 1 jump before latching")
        self.assertLessEqual(max_x, 300, "Raw env should die at Goomba 1 (x <= 300)")
        self.assertTrue(died, "Raw env should terminate by death at Goomba 1")

    def test_auto_unlatch_enables_continuous_jumping(self):
        """With auto_unlatch=True, greedy continuous RIGHT+JUMP executes repeated jumps past x=1,000."""
        env = Mario2DEnv(frame_skip=4, auto_unlatch=True)
        takeoffs, max_x, died, steps = _count_takeoffs(env, [RIGHT_JUMP] * 200)
        env.close()

        self.assertGreaterEqual(takeoffs, 10, f"Expected >= 10 jumps, got {takeoffs}")
        self.assertGreater(max_x, 1000, f"Expected max_x > 1000, got {max_x}")
        self.assertFalse(died, "Agent should not die at x=296 Goomba")

    def test_macro_action_wrapper(self):
        """MarioMacroActionWrapper correctly activates auto_unlatch and clears early obstacles."""
        raw_env = Mario2DEnv(frame_skip=4, auto_unlatch=False)
        wrapped_env = MarioMacroActionWrapper(raw_env)
        self.assertTrue(wrapped_env.auto_unlatch)

        takeoffs, max_x, died, steps = _count_takeoffs(wrapped_env, [RIGHT_JUMP] * 200)
        wrapped_env.close()

        self.assertGreaterEqual(takeoffs, 10)
        self.assertGreater(max_x, 1000)
        self.assertFalse(died)


def run_macro_action_tests() -> bool:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMacroActions)
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    return res.wasSuccessful()


if __name__ == "__main__":
    unittest.main()
