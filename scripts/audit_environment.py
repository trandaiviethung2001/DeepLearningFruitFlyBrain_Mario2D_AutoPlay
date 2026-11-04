"""Phase-1 audit: environment determinism, action semantics, observation alignment, termination.

Run: .venv/bin/python scripts/audit_environment.py
Writes logs/audit_environment.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.environment import Mario2DEnv

RIGHT, JUMP, RIGHT_JUMP, NOOP = 1, 3, 4, 0


def rollout(env, actions, seed=None):
    obs, info = env.reset(seed=seed)
    xs, ys, dones = [], [], []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        xs.append(info["x_pos"]); ys.append(info["y_pos"])
        if term or trunc:
            dones.append(True)
            break
    return xs, ys, info, len(xs)


def count_jumps(ys):
    """Count take-offs: y rising from a ground plateau."""
    ground = min(ys[:5]) if ys else 0
    jumps, airborne = 0, False
    for y in ys:
        if not airborne and y > ground + 2:
            jumps += 1; airborne = True
        elif airborne and y <= ground:
            airborne = False
    return jumps


def main():
    out = {}
    env = Mario2DEnv(frame_stack=1)

    # 1. Determinism across "seeds"
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 5, size=300).tolist()
    xa, _, _, _ = rollout(env, seq, seed=1)
    xb, _, _, _ = rollout(env, seq, seed=999)
    out["seed_changes_dynamics"] = xa != xb
    print("1. Same actions, different seeds -> identical trajectory:", xa == xb)

    # 2. Held RIGHT+JUMP (A never released)
    xs, ys, info, n = rollout(env, [RIGHT_JUMP] * 1500)
    out["hold_right_jump"] = {"max_x": max(xs), "takeoffs": count_jumps(ys), "steps": n,
                              "died": n < 1500, "max_y": max(ys)}
    print("2. Hold RIGHT+JUMP 1500 frames:", out["hold_right_jump"])

    # 3. RIGHT only
    xs, ys, info, n = rollout(env, [RIGHT] * 1500)
    out["right_only"] = {"max_x": max(xs), "steps": n, "terminated": n < 1500, "life": info.get("life")}
    print("3. RIGHT only:", out["right_only"])

    # 4. Jump height vs hold duration (from standstill after 30 NOOP frames)
    heights = {}
    for hold in (1, 2, 4, 8, 16, 32):
        acts = [NOOP] * 30 + [JUMP] * hold + [NOOP] * (60 - hold)
        _, ys, _, _ = rollout(env, acts)
        heights[hold] = int(max(ys) - ys[29])
    out["jump_height_by_hold_frames"] = heights
    print("4. Jump apex height (px) by A-hold frames:", heights)

    # 5. Re-jump requires release: alternate 20 frames RIGHT+JUMP / 1 frame RIGHT
    acts = ([RIGHT_JUMP] * 20 + [RIGHT] * 1) * 70
    xs, ys, _, n = rollout(env, acts)
    out["pulse_right_jump_20on_1off"] = {"max_x": max(xs), "takeoffs": count_jumps(ys), "steps": n}
    print("5. RIGHT+JUMP pulsed (20 on / 1 off):", out["pulse_right_jump_20on_1off"])

    # 6. Observation alignment: last stacked frame == preprocess(current screen)
    env4 = Mario2DEnv(frame_stack=4)
    obs, _ = env4.reset()
    for _ in range(50):
        obs, *_ = env4.step(RIGHT)
    cur = env4.preprocess_frame(env4.env.unwrapped.screen.copy())
    out["obs_last_channel_matches_screen"] = bool(torch.allclose(obs[-1:], cur))
    out["obs_shape"], out["obs_dtype"] = list(obs.shape), str(obs.dtype)
    out["obs_range"] = [float(obs.min()), float(obs.max())]
    out["adjacent_stack_frames_identical_frac"] = float((obs[-1] == obs[-2]).float().mean())
    print("6. Obs last channel == current screen:", out["obs_last_channel_matches_screen"],
          "shape", obs.shape, "range", out["obs_range"],
          "| pixel-identical fraction between consecutive stacked frames:",
          round(out["adjacent_stack_frames_identical_frac"], 3))
    env4.close()

    # 7. Raw reward decomposition when stuck at a pipe (RIGHT only after it stops progressing)
    obs, info = env.reset()
    total, stuck_r, last_x, stuck_steps = 0.0, 0.0, 0, 0
    for t in range(1500):
        obs, r, term, trunc, info = env.step(RIGHT)
        total += r
        if info["x_pos"] == last_x:
            stuck_r += r; stuck_steps += 1
        last_x = info["x_pos"]
        if term or trunc:
            break
    out["right_only_reward"] = {"total": total, "reward_while_not_progressing": stuck_r,
                                "non_progress_steps": stuck_steps, "final_x": info["x_pos"],
                                "terminated": term}
    print("7. RIGHT-only raw reward:", out["right_only_reward"])
    env.close()

    p = REPO_ROOT / "logs" / "audit_environment.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print("Saved", p)


if __name__ == "__main__":
    main()

