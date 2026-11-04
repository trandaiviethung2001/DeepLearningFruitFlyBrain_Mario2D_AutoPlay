"""Generates World 1-1 simulation data for Baseline GRU-1024 Policy.

Captures all 488 macro steps / 1,950 60 FPS lossless frames:
- Native 256x240 WebP frames
- GRU-1024 hidden unit states
- GRU activation statistics & Hoyer sparsity
- Softmax action probabilities and value estimates V(s)
"""

from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline_model import BaselineMarioPolicy
from src.environment import Mario2DEnv
from src.visualization import ACTION_NAMES, compute_sparsity_ratio


def main():
    device = torch.device("cpu")
    ckpt_path = REPO_ROOT / "checkpoints" / "baseline_ppo_v3.pt"
    output_path = REPO_ROOT / "data" / "baseline_simulation_data.json"

    print("1. Loading BaselineMarioPolicy (GRU-1024)...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_channels = ckpt["model_state_dict"]["encoder.conv1.weight"].shape[1]

    policy = BaselineMarioPolicy(in_channels=in_channels, cnn_dim=512, hidden_dim=1024)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.to(device)
    policy.eval()

    print("2. Running Mario World 1-1 with Baseline GRU...")
    env = Mario2DEnv(
        rom_id="SuperMarioBros-1-1-v0",
        frame_stack=in_channels,
        frame_skip=4,
        auto_unlatch=True,
        render_mode="rgb_array",
        max_episode_frames=3000,
        stall_frames=300,
    )

    obs, info = env.reset(seed=1000)
    h = None

    steps_data: List[Dict[str, Any]] = []
    macro_step = 0
    frame_count = 0
    t0 = time.time()

    while macro_step < 500:
        obs_t = obs.unsqueeze(0).float() if isinstance(obs, torch.Tensor) else torch.from_numpy(obs).unsqueeze(0).float()

        with torch.no_grad():
            logits, val, h_next = policy(obs_t, h)
            h_gru = h_next.squeeze(0).cpu().numpy()  # (1024,)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            action = int(np.argmax(probs))
            val_num = round(float(val.item()), 2)

        h = h_next
        macro_step += 1

        sparsity = round(float(compute_sparsity_ratio(h_gru)), 1)
        mean_abs_gru = round(float(np.mean(np.abs(h_gru))), 3)

        # Record subframes
        for f in range(env.frame_skip):
            act = action
            if env.auto_unlatch:
                if act in (3, 4):
                    on_ground = (env.ram[0x001D] == 0)
                    if on_ground and env._last_held_a and f == 0:
                        act = 1 if action == 4 else 0
                        env._last_held_a = False
                    else:
                        env._last_held_a = True
                else:
                    env._last_held_a = False

            obs_raw, reward, term, trunc, sub_info = env._raw_step(act)
            frame_count += 1

            img = Image.fromarray(obs_raw)
            buf = io.BytesIO()
            img.save(buf, format="WEBP", lossless=True)
            frame_b64 = "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

            sub_x = int(sub_info.get("x_pos", 0))
            sub_y = int(sub_info.get("y_pos", 0))
            sub_flag = bool(sub_info.get("flag_get", False))

            milestone = ""
            if frame_count == 1:
                milestone = "Game Spawn & Sensory Intake"
            elif 290 <= sub_x <= 315 and act in [2, 4]:
                milestone = "Goomba 1 Escape Leap"
            elif 390 <= sub_x <= 420:
                milestone = "Pipe 1 Jump"
            elif 540 <= sub_x <= 570:
                milestone = "Pipe 2 Jump"
            elif 730 <= sub_x <= 760:
                milestone = "Pipe 3 Jump"
            elif 940 <= sub_x <= 970:
                milestone = "Pipe 4 Jump"
            elif 1380 <= sub_x <= 1430:
                milestone = "Pit 1 Abyss Leap"
            elif sub_flag or sub_x >= 3160:
                milestone = "Flagpole Capture Win"

            step_entry = {
                "step": frame_count,
                "macro_step": macro_step,
                "subframe": f,
                "x": sub_x,
                "y": sub_y,
                "a": act,
                "a_name": ACTION_NAMES[act],
                "p": [round(float(p), 3) for p in probs],
                "v": val_num,
                "sp": sparsity,
                "act": [round(mean_abs_gru, 3), round(mean_abs_gru, 3), round(mean_abs_gru, 3)],
                "gf": 0.0,
                "m": milestone,
                "f": frame_b64,
                "dns": [],
                "vps": [],
                "cbs": []
            }
            steps_data.append(step_entry)

            if term or trunc:
                break
        if term or trunc:
            break

    env.close()
    print(f"Captured {len(steps_data)} frames in {time.time()-t0:.2f}s! Final X={sub_x}, Flag={sub_flag}")

    out_obj = {
        "metadata": {
            "model_type": "baseline_gru",
            "model_name": "Dense GRU-1024 Baseline",
            "checkpoint": "checkpoints/baseline_ppo_v3.pt",
            "hidden_units": 1024,
            "architecture": "Nature CNN (Conv 8x8, 4x4, 3x3) -> Dense GRU (1024) -> Actor/Critic",
            "max_x": sub_x,
            "flag_captured": sub_flag,
            "total_frames": len(steps_data),
            "macro_steps": macro_step,
        },
        "steps": steps_data
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f)
    print(f"Saved {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    main()
