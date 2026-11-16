"""Interactive Mario Gameplay Player using Trained Connectome or Baseline Policy.

Opens a real-time macOS game window so you can watch the agent play Super Mario Bros live!

Usage:
    # Play live with the 30,000-neuron Drosophila MaleCNS Connectome policy:
    .venv/bin/python scripts/play.py

    # Play live with the Baseline GRU policy:
    .venv/bin/python scripts/play.py --model baseline

    # Play headless with live terminal telemetry:
    .venv/bin/python scripts/play.py --no-gui
"""

from __future__ import annotations

import argparse
import sys
import io
import time
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline_model import BaselineMarioPolicy
from src.connectome_model import ConnectomeMarioPolicy
from src.environment import Mario2DEnv
from src.visualization import ACTION_NAMES, compute_sparsity_ratio


def parse_args():
    parser = argparse.ArgumentParser(description="Watch Mario play live with trained Connectome or Baseline policy")
    parser.add_argument(
        "--model",
        type=str,
        choices=["connectome", "baseline"],
        default="connectome",
        help="Model architecture to run (default: connectome)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint (defaults to best v3 checkpoint)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Game RNG seed (default: 1000)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0 = authentic 1.0x NES real-time speed; 0.5 = slow motion; 2.0 = fast forward)",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        default=True,
        help="Render all 60 frames/sec smoothly without skipping frames (default: True)",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_false",
        dest="smooth",
        help="Only render once per agent action step",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="Open live graphical NES window (default: True)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_false",
        dest="gui",
        help="Disable GUI window and run in terminal mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Inference device (cuda, mps, or cpu)",
    )
    parser.add_argument(
        "--auto-unlatch",
        action="store_true",
        default=False,
        help="Enable auto-unlatching of button A on ground (default: False)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        default=False,
        help="Sample actions probabilistically from policy distribution (like training) instead of greedy argmax",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        default=False,
        help="Use deterministic greedy argmax action selection (default mode)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0; < 1.0 is more confident)",
    )
    return parser.parse_args()


def load_agent(model_type: str, checkpoint_path: str, device: torch.device):
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    print(f"Loading checkpoint: {ckpt_file} onto {device}...")
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
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

    # Set default checkpoint if not provided
    if not args.checkpoint:
        v2_path = REPO_ROOT / "checkpoints" / f"{args.model}_ppo_v2.pt"
        v3_path = REPO_ROOT / "checkpoints" / f"{args.model}_ppo_v3.pt"
        if v2_path.exists():
            args.checkpoint = str(v2_path)
        elif v3_path.exists():
            args.checkpoint = str(v3_path)
        else:
            args.checkpoint = str(v2_path)

    policy, in_channels = load_agent(args.model, args.checkpoint, device)

    render_mode = "human" if args.gui else "rgb_array"
    smooth_render = bool(args.gui and args.smooth)
    print("\n" + "=" * 65)
    print(f"STARTING LIVE GAMEPLAY: {args.model.upper()} POLICY")
    print(f"Device: {device} | GUI Window: {args.gui} | Speed Multiplier: {args.speed}x")
    print(f"Seed: {args.seed} | Checkpoint: {Path(args.checkpoint).name} | Smooth 60 FPS: {smooth_render}")
    print("=" * 65)

    env = Mario2DEnv(
        frame_stack=in_channels,
        frame_skip=4,
        auto_unlatch=args.auto_unlatch,
        render_mode=render_mode,
        smooth_render=smooth_render,
        speed=args.speed,
        max_episode_frames=5000,
        stall_frames=600,
    )

    obs, info = env.reset(seed=args.seed)
    h = None
    step = 0
    total_reward = 0.0
    # 4 NES frames per step at 60 FPS = 4/60s per step for 1.0x real-time speed
    step_delay = (4.0 / 60.0) / max(0.05, args.speed)

    print("\n[Mario Spawned] Running game loop (Press Ctrl+C to stop)...")
    print("-" * 65)

    try:
        while True:
            t0 = time.time()

            # If not in smooth_render mode but GUI is on, render once per step
            if args.gui and not smooth_render:
                env.render()

            # Inference
            if isinstance(obs, torch.Tensor):
                obs_t = obs.unsqueeze(0).float().to(device)
            else:
                obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)

            with torch.no_grad():
                if args.model == "connectome":
                    logits, val, h_next, tele = policy(obs_t, h, num_ticks=2, return_telemetry=True)
                    core_act = tele["h_core"].cpu().numpy().squeeze(0)
                    sparsity = compute_sparsity_ratio(core_act)
                else:
                    logits, val, h_next = policy(obs_t, h)
                    sparsity = 0.0

                probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
                if args.sample:
                    dist = torch.distributions.Categorical(logits=logits / max(1e-3, args.temperature))
                    action = dist.sample().item()
                else:
                    action = int(np.argmax(probs))

            h = h_next

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            total_reward += reward
            step += 1

            x_pos = int(next_info.get("x_pos", 0))
            y_pos = int(next_info.get("y_pos", 0))
            action_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action)
            flag = bool(next_info.get("flag_get", False))

            # Live terminal telemetry
            telemetry_str = (
                f"\rStep: {step:4d} | X: {x_pos:4d} | Y: {y_pos:3d} | Act: {action_name:<10} | "
                f"P(Act): {probs[action]:.2f} | Rew: {total_reward:6.1f}"
            )
            if args.model == "connectome":
                telemetry_str += f" | Sparsity L1/L2: {sparsity:5.1f}"
            sys.stdout.write(telemetry_str)
            sys.stdout.flush()

            obs = next_obs

            # Control playback speed when not using smooth_render
            if not smooth_render:
                elapsed = time.time() - t0
                sleep_time = step_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if flag:
                print(f"\n\n[FLAGPOLE REACHED] Mario won the game at step {step} (X={x_pos})!")
                break
            if terminated:
                print(f"\n\n[DIED] Episode ended at step {step} (X={x_pos}).")
                break
            if truncated:
                print(f"\n\n[TIMEOUT] Episode truncated at step {step} (X={x_pos}).")
                break

    except KeyboardInterrupt:
        print("\n\nGameplay paused by user (Ctrl+C).")

    finally:
        env.close()

    print("-" * 65)
    print(f"Final Statistics: Steps = {step} | Max X = {x_pos} | Total Reward = {total_reward:.1f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

