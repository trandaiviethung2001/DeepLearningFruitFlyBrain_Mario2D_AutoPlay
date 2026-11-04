"""Telemetry and Visuomotor Visualization Engine for Mario Connectome Agents.

Provides:
1. Multi-panel real-time neural and behavioral telemetry logging.
2. Synchronized composite frame rendering (NES gameplay + neural activity).
3. Export engines for high-definition MP4 (via ffmpeg) and animated GIF (via PIL).
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ACTION_NAMES = ["NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B"]


class MarioTelemetryFrame:
    """Stores one step of environment and neural telemetry."""

    def __init__(
        self,
        step: int,
        rgb_frame: np.ndarray,
        x_pos: int,
        y_pos: int,
        action_idx: int,
        action_probs: np.ndarray,
        value: float,
        reward: float,
        sensory_act: Optional[np.ndarray] = None,
        core_act: Optional[np.ndarray] = None,
        motor_act: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
    ):
        self.step = step
        self.rgb_frame = rgb_frame  # (H, W, 3) uint8
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.action_idx = action_idx
        self.action_probs = action_probs  # (5,)
        self.value = value
        self.reward = reward
        self.sensory_act = sensory_act  # (9201,) or None
        self.core_act = core_act  # (30000,) or None
        self.motor_act = motor_act  # (1314,) or None
        self.info = info or {}


def compute_sparsity_ratio(act: np.ndarray) -> float:
    """Compute L1 / L2 sparsity measure (Hoyer's metric component).

    Lower value indicates higher sparsity (fewer active neurons).
    """
    l1 = float(np.sum(np.abs(act)))
    l2 = float(np.sqrt(np.sum(act**2)) + 1e-8)
    return l1 / l2


def render_telemetry_canvas(
    t_frame: MarioTelemetryFrame,
    dpi: int = 100,
    model_name: str = "MaleCNS Connectome",
) -> np.ndarray:
    """Render a 4-panel synchronized telemetry image.

    Layout:
    - Left column: NES Gameplay Screen with HUD overlay
    - Right column top: Action Probabilities Bar Chart
    - Right column mid: Sensory (VP 9.2k) Activation Heatmap / Top Neurons
    - Right column bottom: 30,000 Recurrent Core Firing Distribution & Sparsity
    """
    fig = plt.figure(figsize=(12, 7), dpi=dpi)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

    # 1. Left Panel (spans all 3 rows): Game Screen
    ax_game = fig.add_subplot(gs[:, 0])
    ax_game.imshow(t_frame.rgb_frame)
    ax_game.set_title(
        f"NES Mario 2D Gameplay | {model_name}\n"
        f"Step: {t_frame.step:4d} | X: {t_frame.x_pos:4d} | Y: {t_frame.y_pos:3d} | V(s): {t_frame.value:+.2f}",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax_game.axis("off")

    # Overlay current action label on game screen
    action_text = ACTION_NAMES[t_frame.action_idx] if 0 <= t_frame.action_idx < len(ACTION_NAMES) else f"ACT {t_frame.action_idx}"
    ax_game.text(
        10,
        25,
        f"Action: {action_text}",
        color="yellow",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
    )

    # 2. Right Top: Action Distribution Bar Chart
    ax_act = fig.add_subplot(gs[0, 1])
    colors = ["#2b5c8f"] * len(ACTION_NAMES)
    if 0 <= t_frame.action_idx < len(colors):
        colors[t_frame.action_idx] = "#e63946"  # Highlight chosen action in red
    bars = ax_act.bar(ACTION_NAMES, t_frame.action_probs, color=colors, edgecolor="black", alpha=0.85)
    ax_act.set_ylim(0, 1.05)
    ax_act.set_ylabel("Probability", fontsize=9)
    ax_act.set_title("Motor Readout: Action Distribution", fontsize=10, fontweight="bold")
    ax_act.tick_params(axis="x", rotation=15, labelsize=8)
    for bar in bars:
        h = bar.get_height()
        if h > 0.05:
            ax_act.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    # 3. Right Mid: Sensory Projection Neurons (VP 9,201)
    ax_sen = fig.add_subplot(gs[1, 1])
    if t_frame.sensory_act is not None and len(t_frame.sensory_act) > 0:
        sen_data = np.abs(t_frame.sensory_act)
        # Reshape or sample into a 2D sensory grid (e.g. 96 x 96 approx 9216)
        n_sen = len(sen_data)
        grid_dim = int(np.ceil(np.sqrt(n_sen)))
        padded = np.zeros(grid_dim * grid_dim, dtype=np.float32)
        padded[:n_sen] = sen_data
        grid = padded.reshape(grid_dim, grid_dim)
        im_sen = ax_sen.imshow(grid, cmap="inferno", aspect="auto")
        plt.colorbar(im_sen, ax=ax_sen, fraction=0.046, pad=0.04)
        mean_vp = float(np.mean(sen_data))
        ax_sen.set_title(f"Sensory VPs (N={n_sen:,}) | Mean Act: {mean_vp:.3f}", fontsize=10, fontweight="bold")
    else:
        ax_sen.text(0.5, 0.5, "Sensory VPs: N/A (Standard Baseline)", ha="center", va="center", fontsize=9)
        ax_sen.set_title("Sensory Visual Projections", fontsize=10, fontweight="bold")
    ax_sen.axis("off")

    # 4. Right Bottom: 30,000 Recurrent Core Activity & Sparsity
    ax_core = fig.add_subplot(gs[2, 1])
    if t_frame.core_act is not None and len(t_frame.core_act) > 0:
        c_act = t_frame.core_act
        n_core = len(c_act)
        sparsity = compute_sparsity_ratio(c_act)
        active_ratio = float(np.mean(np.abs(c_act) > 0.05))
        ax_core.hist(c_act, bins=40, range=(-1.0, 1.0), color="#2a9d8f", edgecolor="black", alpha=0.75, density=True)
        ax_core.set_xlim(-1.05, 1.05)
        ax_core.set_xlabel("Neuron Activation State tanh(h)", fontsize=8)
        ax_core.set_ylabel("Density", fontsize=8)
        ax_core.set_title(
            f"Recurrent Core (N={n_core:,}) | L1/L2: {sparsity:.1f} | Active: {active_ratio*100:.1f}%",
            fontsize=10,
            fontweight="bold",
        )
    else:
        ax_core.text(0.5, 0.5, "Core Recurrent State: N/A", ha="center", va="center", fontsize=9)
        ax_core.set_title("Recurrent Core Activity", fontsize=10, fontweight="bold")

    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.25, hspace=0.35)

    # Render figure to numpy RGB array
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.array(img)


def export_telemetry_gif(
    frames: List[np.ndarray],
    output_path: Path | str,
    fps: int = 15,
) -> Path:
    """Save an array of RGB frames as an animated GIF using PIL."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    pil_images = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / max(1, fps))

    # Optimize and save
    pil_images[0].save(
        str(out_file),
        save_all=True,
        append_images=pil_images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out_file


def export_telemetry_mp4(
    frames: List[np.ndarray],
    output_path: Path | str,
    fps: int = 30,
) -> Path:
    """Save an array of RGB frames as an MP4 video using ffmpeg."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if len(frames) == 0:
        raise ValueError("Cannot export empty frame list to MP4")

    h, w = frames[0].shape[:2]
    # Ensure even dimensions for libx264
    w = w - (w % 2)
    h = h - (h % 2)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        str(out_file),
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames:
        cropped = f[:h, :w]
        process.stdin.write(cropped.tobytes())
    process.stdin.close()
    process.wait()

    if process.returncode != 0:
        err = process.stderr.read().decode()
        raise RuntimeError(f"ffmpeg export failed with code {process.returncode}: {err}")

    return out_file
