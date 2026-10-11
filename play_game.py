"""Visual Gameplay Player & Replay Generator.

Enables viewing live gameplay of the Drosophila MaleCNS connectome policy,
parameter-matched baseline GRU, or scripted demonstrator controlling Mario in 2D.

Features:
- Live GUI window rendering on screen (--render).
- High-quality animated GIF recording (--save-gif replay.gif).
- Terminal progress bar and real-time state telemetry.
- Action distribution breakdown and performance summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image

from src.baseline_model import BaselineMarioPolicy
from src.connectome_model import ConnectomeMarioPolicy
from src.demonstration_collector import ScriptedDemonstrator, ACTION_NAMES
from src.environment import Mario2DEnv


WORLD_1_1_LENGTH = 3161  # Flagpole position in pixels


def render_terminal_bar(x_pos: int, action_name: str, step: int, reward: float) -> str:
    """Render a visual ASCII progress bar of World 1-1."""
    width = 30
    progress = min(1.0, max(0.0, (x_pos - 40) / (WORLD_1_1_LENGTH - 40)))
    filled = int(width * progress)
    bar = "=" * filled + (">" if filled < width else "=") + " " * (width - filled - 1)
    pct = progress * 100
    return f"Step {step:4d} | X: {x_pos:4d}m [{bar}] {pct:4.1f}% | Act: {action_name:<10} | Reward: {reward:6.1f}"


def run_visual_episode(
    model_type: str = "connectome",
    checkpoint_path: Optional[str] = None,
    render_gui: bool = False,
    save_gif_path: Optional[str] = "gameplay_replay.gif",
    frame_stack: int = 4,
    max_steps: int = 2000,
    seed: int = 1,
    fps: int = 25,
    temperature: float = 0.5,
) -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 70)
    print("🎮 MALECNS CONNECTOME MARIO 2D GAMEPLAY VIEWER")
    print("=" * 70)
    print(f"Model Type:      {model_type.upper()}")
    print(f"Execution Device:{device}")
    print(f"Frame Stacking:  {frame_stack} frames")
    print(f"GUI Rendering:   {'ENABLED (Live Screen)' if render_gui else 'DISABLED (Use --render to enable)'}")
    print(f"GIF Export:      {save_gif_path if save_gif_path else 'None'}")
    print("=" * 70)

    # Initialize environment
    env = Mario2DEnv(frame_stack=frame_stack, render_mode="human" if render_gui else None)

    # Load Model
    model = None
    teacher = None
    processed_dir = REPO_ROOT / "data" / "processed"

    if model_type == "connectome":
        ckpt_p = Path(checkpoint_path) if checkpoint_path else Path("checkpoints/connectome_ppo_best.pt")
        if not ckpt_p.exists():
            ckpt_p = Path("checkpoints/connectome_model_best.pt")
        print(f"Loading Connectome weights from: {ckpt_p}...")
        model = ConnectomeMarioPolicy(
            subgraph_npz_path=processed_dir / "subgraph_graph.npz",
            node_mapping_csv_path=processed_dir / "subgraph_node_mapping.csv",
            node_metadata_csv_path=processed_dir / "node_metadata.csv",
            in_channels=frame_stack,
        ).to(device)
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()

    elif model_type == "baseline":
        ckpt_p = Path(checkpoint_path) if checkpoint_path else Path("checkpoints/baseline_ppo_best.pt")
        if not ckpt_p.exists():
            ckpt_p = Path("checkpoints/baseline_gru_best.pt")
        print(f"Loading Baseline GRU weights from: {ckpt_p}...")
        model = BaselineMarioPolicy(in_channels=frame_stack, hidden_dim=1024).to(device)
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()

    elif model_type == "demonstrator":
        print("Using Scripted RAM-Aware Expert Demonstrator...")
        teacher = ScriptedDemonstrator()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Reset environment
    obs, info = env.reset(seed=seed)
    h: Optional[torch.Tensor] = None

    frames_for_gif: List[Image.Image] = []
    action_counts = {name: 0 for name in ACTION_NAMES}
    total_reward = 0.0
    max_x = info.get("x_pos", 40)
    flag_get = False

    print("\nStarting gameplay execution...\n")
    start_time = time.time()

    for step in range(1, max_steps + 1):
        step_t0 = time.time()

        # Decide Action
        if teacher is not None:
            action = teacher.get_action(obs, info, env.ram)
        else:
            inp = obs.unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _, h = model(inp, h)
                if temperature > 0.0:
                    dist = torch.distributions.Categorical(logits=logits / temperature)
                    action = dist.sample().item()
                else:
                    action = torch.argmax(logits, dim=-1).item()

        action_name = ACTION_NAMES[action]
        action_counts[action_name] += 1

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        cur_x = info.get("x_pos", 40)
        max_x = max(max_x, cur_x)
        flag_get = bool(info.get("flag_get", False))

        # Record visual frame for GIF
        if save_gif_path:
            # Capture raw NES RGB screen every frame (or alternate frame for speed/size)
            if step % 2 == 0:
                raw_screen = env.env.unwrapped.screen
                if raw_screen is not None:
                    # Convert to PIL Image
                    img = Image.fromarray(raw_screen)
                    frames_for_gif.append(img)

        # Optional live screen rendering
        if render_gui:
            env.env.render()
            # Pacing
            elapsed = time.time() - step_t0
            sleep_time = max(0.0, (1.0 / fps) - elapsed)
            time.sleep(sleep_time)

        # Print terminal progress update every 10 steps or at landmark events
        if step % 10 == 0 or terminated or truncated or flag_get:
            bar_text = render_terminal_bar(cur_x, action_name, step, total_reward)
            sys.stdout.write(f"\r{bar_text}")
            sys.stdout.flush()

        obs = next_obs
        if terminated or truncated or flag_get:
            break

    total_duration = time.time() - start_time
    env.close()

    print("\n\n" + "=" * 70)
    print("🏁 GAMEPLAY RUN SUMMARY")
    print("=" * 70)
    print(f"Total Steps:        {step:,} steps")
    print(f"Elapsed Time:       {total_duration:.2f} seconds ({step / total_duration:.1f} FPS)")
    print(f"Final Mario X:      {max_x} / {WORLD_1_1_LENGTH} pixels ({(max_x/WORLD_1_1_LENGTH)*100:.1f}% World 1-1)")
    print(f"Accumulated Reward: {total_reward:.1f}")
    print(f"First Obstacle:     {'CLEARED (x > 296)' if max_x > 296 else 'FAILED'}")
    print(f"Flagpole Touch:     {'VICTORY (Touched Flagpole!)' if flag_get else 'No'}")
    print(f"Action Breakdown:   {action_counts}")
    print("=" * 70)

    # Save animated GIF if requested
    if save_gif_path and len(frames_for_gif) > 0:
        gif_file = Path(save_gif_path)
        gif_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nEncoding animated GIF ({len(frames_for_gif)} frames at 15 FPS)...")
        # Save GIF with Pillow
        frames_for_gif[0].save(
            str(gif_file),
            save_all=True,
            append_images=frames_for_gif[1:],
            duration=66,  # ~15 FPS
            loop=0,
            optimize=True,
        )
        print(f"✨ Successfully saved gameplay replay to: {gif_file.resolve()}")
        print(f"   Size: {gif_file.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"   👉 You can open this file in VS Code or double-click to view in Finder!")


def main():
    parser = argparse.ArgumentParser(description="Mario 2D Connectome Gameplay Player")
    parser.add_argument("--model", type=str, default="connectome", choices=["connectome", "baseline", "demonstrator"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--render", action="store_true", help="Enable live GUI window rendering on screen")
    parser.add_argument("--save-gif", type=str, default="gameplay_replay.gif", help="Path to save animated GIF")
    parser.add_argument("--seed", type=int, default=1, help="Environment seed")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--frame-stack", type=int, default=4, help="Frame stack count (1 or 4)")
    parser.add_argument("--fps", type=int, default=30, help="Target playback FPS when rendering")
    parser.add_argument("--temperature", type=float, default=0.5, help="Action sampling temperature (0 for argmax)")
    args = parser.parse_args()

    run_visual_episode(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        render_gui=args.render,
        save_gif_path=args.save_gif,
        frame_stack=args.frame_stack,
        max_steps=args.max_steps,
        seed=args.seed,
        fps=args.fps,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
