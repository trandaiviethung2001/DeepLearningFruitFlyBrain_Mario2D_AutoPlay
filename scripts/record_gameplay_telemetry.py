"""CLI script to record Mario gameplay with synchronized neural telemetry.

Runs an agent episode, logs NES frame + neural state, and exports:
- High-definition MP4 video (with H.264 encoding via ffmpeg)
- Compact animated GIF for artifact documentation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline_model import BaselineMarioPolicy
from src.connectome_model import ConnectomeMarioPolicy
from src.environment import Mario2DEnv
from src.visualization import (
    ACTION_NAMES,
    MarioTelemetryFrame,
    export_telemetry_gif,
    export_telemetry_mp4,
    render_telemetry_canvas,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Record Mario telemetry and neural visualization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--model-type", type=str, choices=["baseline", "connectome"], default="baseline")
    parser.add_argument("--seed", type=int, default=1000, help="RNG seed")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps to record")
    parser.add_argument("--auto-unlatch", action="store_true", default=True, help="Enable auto unlatch")
    parser.add_argument("--greedy", action="store_true", default=True, help="Use argmax greedy actions")
    parser.add_argument("--out-gif", type=str, default="", help="Path for output GIF")
    parser.add_argument("--out-mp4", type=str, default="", help="Path for output MP4")
    parser.add_argument("--fps", type=int, default=15, help="Frame rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"), help="Device (cuda, mps or cpu)")
    return parser.parse_args()


def load_policy(model_type: str, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    in_channels = ckpt.get("frame_stack", 4)
    if "model_state_dict" in ckpt and "encoder.conv1.weight" in ckpt["model_state_dict"]:
        in_channels = ckpt["model_state_dict"]["encoder.conv1.weight"].shape[1]

    if model_type == "baseline":
        policy = BaselineMarioPolicy(in_channels=in_channels, cnn_dim=512, hidden_dim=1024)
        policy.load_state_dict(ckpt["model_state_dict"])
    else:
        processed_dir = REPO_ROOT / "data" / "processed"
        policy = ConnectomeMarioPolicy(
            subgraph_npz_path=processed_dir / "subgraph_graph.npz",
            node_mapping_csv_path=processed_dir / "subgraph_node_mapping.csv",
            node_metadata_csv_path=processed_dir / "node_metadata.csv",
            in_channels=in_channels,
            cnn_dim=512,
        )
        policy.load_state_dict(ckpt["model_state_dict"])

    policy.to(device)
    policy.eval()
    return policy, in_channels


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading {args.model_type} policy from: {args.checkpoint}")
    policy, in_channels = load_policy(args.model_type, args.checkpoint, device)

    env = Mario2DEnv(
        frame_stack=in_channels,
        frame_skip=4,
        auto_unlatch=args.auto_unlatch,
        render_mode="rgb_array",
        max_episode_frames=args.max_steps * 4 + 100,
        stall_frames=600,
    )

    obs, info = env.reset(seed=args.seed)
    h = None

    telemetry_frames: list[MarioTelemetryFrame] = []
    composite_images: list[np.ndarray] = []

    print(f"Recording episode (seed={args.seed}, greedy={args.greedy}, max_steps={args.max_steps})...")
    t0 = time.time()
    total_reward = 0.0
    step = 0

    while step < args.max_steps:
        # Get current high-res emulator RGB frame
        rgb_frame = env.render()
        if rgb_frame is None or not isinstance(rgb_frame, np.ndarray):
            rgb_frame = np.zeros((240, 256, 3), dtype=np.uint8)

        if isinstance(obs, torch.Tensor):
            obs_t = obs.unsqueeze(0).float().to(device)
        else:
            obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)
        with torch.no_grad():
            if args.model_type == "connectome":
                logits, value, h_next, tele = policy(obs_t, h, num_ticks=2, return_telemetry=True)
                sensory_act = tele["sensory_drive"].cpu().numpy().squeeze(0)
                core_act = tele["h_core"].cpu().numpy().squeeze(0)
                motor_act = tele["motor_activity"].cpu().numpy().squeeze(0)
            else:
                logits, value, h_next, tele = policy(obs_t, h, return_telemetry=True)
                sensory_act = None
                core_act = tele["h_core"].cpu().numpy().squeeze(0)
                motor_act = None

            probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
            val = float(value.item())

            if args.greedy:
                action = int(np.argmax(probs))
            else:
                action = int(np.random.choice(len(probs), p=probs))

        h = h_next

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        total_reward += reward

        x_pos = int(next_info.get("x_pos", 0))
        y_pos = int(next_info.get("y_pos", 0))

        t_frame = MarioTelemetryFrame(
            step=step,
            rgb_frame=rgb_frame,
            x_pos=x_pos,
            y_pos=y_pos,
            action_idx=action,
            action_probs=probs,
            value=val,
            reward=float(reward),
            sensory_act=sensory_act,
            core_act=core_act,
            motor_act=motor_act,
            info=next_info,
        )
        telemetry_frames.append(t_frame)

        # Render composite frame every N steps or every step
        rendered_canvas = render_telemetry_canvas(
            t_frame,
            dpi=80,
            model_name="MaleCNS Biological Connectome" if args.model_type == "connectome" else "Dense GRU-1024 Baseline",
        )
        composite_images.append(rendered_canvas)

        obs = next_obs
        step += 1

        if terminated or truncated or next_info.get("flag_get", False):
            print(f"Episode ended at step {step}: X={x_pos}, Flag={next_info.get('flag_get', False)}")
            break

    env.close()
    elapsed = time.time() - t0
    print(f"Finished episode recording: {len(composite_images)} frames in {elapsed:.1f}s.")
    print(f"Max X reached: {t_frame.x_pos}, Total reward: {total_reward:.1f}")

    # Export GIF
    if args.out_gif:
        gif_path = export_telemetry_gif(composite_images, args.out_gif, fps=args.fps)
        print(f"Exported GIF to: {gif_path} ({gif_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Export MP4
    if args.out_mp4:
        mp4_path = export_telemetry_mp4(composite_images, args.out_mp4, fps=max(args.fps, 20))
        print(f"Exported MP4 to: {mp4_path} ({mp4_path.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
