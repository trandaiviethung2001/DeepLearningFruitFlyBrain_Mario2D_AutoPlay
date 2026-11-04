"""Mario 2D Gymnasium Environment Wrapper.

Provides standardized interface for Super Mario Bros World 1-1:
- 5 discrete actions: NOOP, RIGHT, LEFT, JUMP, RIGHT+JUMP
- Observation: (frame_stack, 84, 84) normalized float32 tensor
- Proper reset/step API handling across gym and gymnasium wrappers

Optional control/evaluation features (all default to the original behaviour):
- frame_skip:      repeat each agent action for k NES frames (rewards summed). Consecutive
                   agent steps that both press A keep the button held, so a full-height jump
                   requires several consecutive A-steps and a *new* jump requires releasing A.
- noop_max:        random 0..noop_max NOOP frames after reset, drawn from the reset seed.
                   NES emulation is deterministic, so without this the reset seed has no effect.
- max_episode_frames / stall_frames: truncation (not termination) limits in NES frames.
- start_states:    emulator snapshots used as alternative reset points (curriculum training).
"""

from __future__ import annotations

from collections import deque
import time
from typing import Any, List, Optional, Tuple

import gym_super_mario_bros
import numpy as np
import torch
import torchvision.transforms as T
from nes_py.wrappers import JoypadSpace

MARIO_5_ACTIONS = [
    ["NOOP"],          # Action 0: No operation
    ["right"],         # Action 1: Walk right
    ["left"],          # Action 2: Walk left
    ["A"],             # Action 3: Jump
    ["right", "A"],    # Action 4: Run / jump right
]

# Level-geometry landmarks for World 1-1 (pixel x of the obstacle's far edge; tile = 16 px).
# Goomba-1 is the first lethal obstacle (RIGHT-only dies at x=296).
WORLD_1_1_LANDMARKS = {
    "goomba1": 300,
    "pipe1": 481,
    "pipe2": 641,
    "pipe3": 769,
    "pipe4": 945,
    "pit1": 1137,
}
WORLD_1_1_FLAG_X = 3161


class Mario2DEnv:
    """Standardized wrapper around SuperMarioBros-1-1-v0."""

    def __init__(
        self,
        rom_id: str = "SuperMarioBros-1-1-v0",
        render_mode: str | None = None,
        frame_stack: int = 1,
        frame_skip: int = 1,
        noop_max: int = 0,
        max_episode_frames: Optional[int] = None,
        stall_frames: Optional[int] = None,
        auto_unlatch: bool = False,
        smooth_render: bool = False,
        speed: float = 1.0,
    ):
        self.rom_id = rom_id
        self.render_mode = render_mode
        self.frame_stack = max(1, frame_stack)
        self.frame_skip = max(1, frame_skip)
        self.noop_max = max(0, noop_max)
        self.max_episode_frames = max_episode_frames
        self.stall_frames = stall_frames
        self.auto_unlatch = auto_unlatch
        self.smooth_render = smooth_render
        self.speed = speed
        self._last_held_a = False
        if render_mode is not None:
            raw_env = gym_super_mario_bros.make(rom_id, render_mode=render_mode)
        else:
            raw_env = gym_super_mario_bros.make(rom_id)
        self.env = JoypadSpace(raw_env, MARIO_5_ACTIONS)
        self.action_space_n = len(MARIO_5_ACTIONS)
        self.frame_buffer: deque[torch.Tensor] = deque(maxlen=self.frame_stack)
        self.start_states: List[Any] = []
        self._rng = np.random.default_rng()
        self._episode_frames = 0
        self._best_x = 0
        self._frames_since_progress = 0

        # Standard observation transformation: RGB -> Grayscale -> 84x84 -> float32 [0, 1]
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 84)),
            T.ToTensor(),  # yields torch.Tensor (1, 84, 84) in [0.0, 1.0]
        ])

    def render(self) -> Any:
        """Render the environment frame (opens GUI window if render_mode='human')."""
        return self.env.render()

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert raw (240, 256, 3) uint8 numpy array to (1, 84, 84) float32 tensor."""
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        tensor = self.transform(frame)
        return tensor

    def _get_stacked_obs(self) -> torch.Tensor:
        """Return stacked observation tensor of shape (frame_stack, 84, 84)."""
        return torch.cat(list(self.frame_buffer), dim=0)

    @property
    def ram(self) -> np.ndarray:
        """Direct access to NES 2048-byte RAM array."""
        return self.env.unwrapped.ram

    # ------------------------------------------------------------------ snapshots
    def snapshot(self) -> Any:
        """Opaque emulator snapshot of the current state (for curriculum start points)."""
        return self.env.unwrapped.dump_state()

    def set_start_states(self, states: List[Any]) -> None:
        """Use these snapshots (sampled uniformly together with the true level start) at reset."""
        self.start_states = list(states)

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None, use_start_states: bool = True) -> Tuple[torch.Tensor, dict[str, Any]]:
        """Reset environment and return processed observation tensor and info dict."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            try:
                res = self.env.reset(seed=seed)
            except TypeError:
                res = self.env.reset()
        else:
            res = self.env.reset()
        if isinstance(res, tuple):
            obs, info = res[0], res[1]
        else:
            obs, info = res, {}

        # Curriculum: optionally start from a stored snapshot (index 0 == true level start).
        if use_start_states and self.start_states:
            k = int(self._rng.integers(0, len(self.start_states) + 1))
            if k > 0:
                u = self.env.unwrapped
                u.load_state(self.start_states[k - 1])
                u._did_reset()  # re-sync reward bookkeeping (x max, clock, score) to the snapshot
                obs, _, _, _, info = self.env.step(0)

        # Random no-op starts (deterministic given the seed)
        if self.noop_max > 0:
            for _ in range(int(self._rng.integers(0, self.noop_max + 1))):
                obs, _, term, trunc, info = self.env.step(0)
                if term or trunc:
                    res = self.env.reset()
                    obs, info = (res[0], res[1]) if isinstance(res, tuple) else (res, {})

        proc_obs = self.preprocess_frame(obs)
        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(proc_obs)

        self._episode_frames = 0
        self._best_x = int(info.get("x_pos", 0))
        self._frames_since_progress = 0
        self._last_held_a = False
        return self._get_stacked_obs(), info

    def _raw_step(self, action: int):
        step_res = self.env.step(action)
        if len(step_res) == 5:
            return step_res
        if len(step_res) == 4:
            obs, reward, done, info = step_res
            return obs, reward, done, False, info
        raise RuntimeError(f"Unexpected step return format of length {len(step_res)}")

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        """Take a step and return (obs, reward, terminated, truncated, info).

        With frame_skip > 1 the action is repeated; rewards (and reward components) are summed
        over the executed frames and the loop stops early at termination.
        When auto_unlatch=True, button A is automatically unlatched on ground takeoffs to
        prevent NES controller latching deadlock during repeated jumps.
        """
        total_reward = 0.0
        terminated = truncated = False
        components: dict[str, float] = {}
        info: dict[str, Any] = {}
        obs = None
        frames = 0
        for f in range(self.frame_skip):
            act = action
            if self.auto_unlatch:
                if act in (3, 4):
                    on_ground = (self.ram[0x001D] == 0)
                    if on_ground and self._last_held_a and f == 0:
                        act = 1 if action == 4 else 0
                        self._last_held_a = False
                    else:
                        self._last_held_a = True
                else:
                    self._last_held_a = False

            obs, reward, terminated, truncated, info = self._raw_step(act)
            frames += 1
            if self.render_mode == "human" and self.smooth_render:
                self.env.render()
                if self.speed > 0:
                    time.sleep(1.0 / (60.0 * self.speed))
            total_reward += float(reward)
            for k, v in (info.get("reward_components") or {}).items():
                components[k] = components.get(k, 0.0) + float(v)
            if terminated or truncated:
                break

        self._episode_frames += frames
        x = int(info.get("x_pos", 0))
        if x > self._best_x:
            self._best_x = x
            self._frames_since_progress = 0
        else:
            self._frames_since_progress += frames

        # External truncation limits (time-limit style; the episode is NOT a failure state)
        trunc_reason = None
        if not terminated:
            if self.max_episode_frames is not None and self._episode_frames >= self.max_episode_frames:
                truncated, trunc_reason = True, "max_frames"
            elif self.stall_frames is not None and self._frames_since_progress >= self.stall_frames:
                truncated, trunc_reason = True, "stall"

        info = dict(info)
        info["reward_components_sum"] = components
        info["frames"] = frames
        info["episode_frames"] = self._episode_frames
        info["death"] = bool(info.get("is_dying", False) or info.get("is_dead", False))
        info["truncation_reason"] = trunc_reason

        proc_obs = self.preprocess_frame(obs)
        self.frame_buffer.append(proc_obs)
        return self._get_stacked_obs(), total_reward, bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()


class MarioMacroActionWrapper:
    """Macro-action wrapper that automatically unlatches NES button A on ground takeoffs."""

    def __init__(self, env: Mario2DEnv):
        self.env = env
        self.env.auto_unlatch = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action: int):
        return self.env.step(action)

    def close(self):
        self.env.close()

