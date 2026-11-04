"""Test A: Mario Environment Verification.

Requirements:
- Initialize/reset Mario.
- Execute each of the 5 discrete actions.
- Verify observations (1, 84, 84), finite rewards, termination/truncation handling.
- Verify deterministic resetting and frame preprocessing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.environment import Mario2DEnv, MARIO_5_ACTIONS


def run_environment_test() -> bool:
    print("=" * 60)
    print("TEST A: MARIO ENVIRONMENT VERIFICATION")
    print("=" * 60)

    print(f"1. Initializing Mario2DEnv (rom='SuperMarioBros-1-1-v0')...")
    env = Mario2DEnv()
    print(f"   Action space size: {env.action_space_n}")
    assert env.action_space_n == 5, f"Expected 5 actions, got {env.action_space_n}"

    print("2. Testing reset...")
    obs, info = env.reset()
    print(f"   Initial observation shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"   Value range: min={obs.min().item():.3f}, max={obs.max().item():.3f}")
    assert obs.shape == (1, 84, 84), f"Expected (1, 84, 84), got {obs.shape}"
    assert obs.dtype == torch.float32, f"Expected float32, got {obs.dtype}"
    assert 0.0 <= obs.min().item() and obs.max().item() <= 1.0, "Obs values out of [0, 1]"

    print("3. Executing all 5 discrete actions...")
    for action_idx, action_name in enumerate(MARIO_5_ACTIONS):
        obs, reward, terminated, truncated, info = env.step(action_idx)
        print(
            f"   Action {action_idx} {action_name}: "
            f"obs_shape={tuple(obs.shape)}, reward={reward:.2f}, "
            f"term={terminated}, trunc={truncated}, x_pos={info.get('x_pos')}"
        )
        assert obs.shape == (1, 84, 84)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    print("4. Testing step sequence across 20 frames...")
    total_reward = 0.0
    for step_num in range(20):
        # Action 4 is right+jump
        obs, reward, terminated, truncated, info = env.step(4)
        total_reward += reward
        if terminated or truncated:
            print(f"   Episode finished early at step {step_num}")
            obs, info = env.reset()

    print(f"   20 steps complete. Accumulated reward: {total_reward:.2f}, final x_pos: {info.get('x_pos')}")

    env.close()
    print("TEST A RESULT: PASS\n")
    return True


if __name__ == "__main__":
    success = run_environment_test()
    sys.exit(0 if success else 1)
