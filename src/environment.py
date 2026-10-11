"""Mario 2D Gymnasium Environment Wrapper.

Provides standardized interface for Super Mario Bros World 1-1:
- 5 discrete actions: NOOP, RIGHT, LEFT, JUMP, RIGHT+JUMP
- Observation: (1, 84, 84) normalized float32 tensor
- Proper reset/step API handling across gym and gymnasium wrappers
"""

from __future__ import annotations

from typing import Any, Tuple
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import torchvision.transforms as T
import torch

from collections import deque

MARIO_5_ACTIONS = [
    ["NOOP"],          # Action 0: No operation
    ["right"],         # Action 1: Walk right
    ["left"],          # Action 2: Walk left
    ["A"],             # Action 3: Jump
    ["right", "A"],    # Action 4: Run / jump right
]


class Mario2DEnv:
    """Standardized wrapper around SuperMarioBros-1-1-v0."""

    def __init__(
        self,
        rom_id: str = "SuperMarioBros-1-1-v0",
        render_mode: str | None = None,
        frame_stack: int = 1,
    ):
        self.rom_id = rom_id
        self.frame_stack = max(1, frame_stack)
        raw_env = gym_super_mario_bros.make(rom_id)
        self.env = JoypadSpace(raw_env, MARIO_5_ACTIONS)
        self.action_space_n = len(MARIO_5_ACTIONS)
        self.frame_buffer: deque[torch.Tensor] = deque(maxlen=self.frame_stack)

        # Standard observation transformation: RGB -> Grayscale -> 84x84 -> float32 [0, 1]
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 84)),
            T.ToTensor(),  # yields torch.Tensor (1, 84, 84) in [0.0, 1.0]
        ])

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

    def reset(self, seed: int | None = None) -> Tuple[torch.Tensor, dict[str, Any]]:
        """Reset environment and return processed observation tensor and info dict."""
        if seed is not None:
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
        proc_obs = self.preprocess_frame(obs)

        # Clear buffer and fill with initial frame
        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(proc_obs)

        return self._get_stacked_obs(), info

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        """Take a step and return (obs, reward, terminated, truncated, info)."""
        step_res = self.env.step(action)
        if len(step_res) == 5:
            obs, reward, terminated, truncated, info = step_res
        elif len(step_res) == 4:
            obs, reward, done, info = step_res
            terminated = done
            truncated = False
        else:
            raise RuntimeError(f"Unexpected step return format of length {len(step_res)}")

        proc_obs = self.preprocess_frame(obs)
        self.frame_buffer.append(proc_obs)
        return self._get_stacked_obs(), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()

