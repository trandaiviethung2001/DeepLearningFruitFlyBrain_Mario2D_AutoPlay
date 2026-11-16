"""Proximal Policy Optimization (PPO) Trainer for Connectome and Baseline Mario Policies.

Key fixes & features:
- Proper distinction between termination (death/flag, V=0) and truncation (time/stall limit, V=bootstrap).
- GAE advantage estimation with per-episode boundary isolation (no advantage leakage across episodes).
- Principled reward shaping: raw progress + small time penalty - substantial death penalty + flag bonus.
  Does NOT penalize zero velocity during airborne jumping frames.
- Configurable frame_skip (default 4 for RL, matching standard Atari/NES RL literature).
- Preserves raw task reward in logs alongside shaped reward.
- Saves self-describing checkpoints containing architecture and environment metadata.
- Pretrained weight transfer support from behavioral cloning checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# NumPy 2.0+ compatibility shim for legacy Gym
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.baseline_model import BaselineMarioPolicy
from src.connectome_model import ConnectomeMarioPolicy
from src.environment import Mario2DEnv, VectorMarioEnv, WORLD_1_1_FLAG_X

ACTION_NAMES = ["NOOP", "RIGHT", "LEFT", "JUMP", "RIGHT+JUMP"]


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminateds: Optional[torch.Tensor] = None,
    truncateds: Any = None,
    next_value: float = 0.0,
    gamma: float = 0.99,
    lam: float = 0.95,
    dones: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).

    Supports both:
      compute_gae(rewards, values, dones, next_val, gamma, lam)
      compute_gae(rewards, values, terminateds=..., truncateds=..., next_value=...)
    """
    T = len(rewards)
    terms = terminateds if terminateds is not None else dones

    if isinstance(truncateds, (int, float)):
        # Legacy positional call: 4th arg is next_val
        actual_next_val = float(truncateds)
        actual_gamma = next_value if (isinstance(next_value, float) and next_value > 0.0) else gamma
        actual_lam = lam
        truncs = torch.zeros_like(terms)
    elif truncateds is None:
        actual_next_val = next_value
        actual_gamma, actual_lam = gamma, lam
        truncs = torch.zeros_like(terms)
    else:
        actual_next_val = next_value
        actual_gamma, actual_lam = gamma, lam
        truncs = truncateds

    advantages = torch.zeros(T, dtype=torch.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = actual_next_val
        else:
            next_val = values[t + 1].item()

        is_term = bool(terms[t])
        is_trunc = bool(truncs[t])

        if is_term:
            delta = rewards[t].item() - values[t].item()
            last_gae = delta
        elif is_trunc:
            delta = rewards[t].item() + actual_gamma * next_val - values[t].item()
            last_gae = delta
        else:
            delta = rewards[t].item() + actual_gamma * next_val - values[t].item()
            last_gae = delta + actual_gamma * actual_lam * last_gae

        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class MarioRewardShaper:
    """Principled reward shaper for Super Mario Bros."""

    def __init__(
        self,
        progress_scale: float = 0.1,
        time_penalty: float = 0.01,
        death_penalty: float = 25.0,
        flag_bonus: float = 50.0,
        jump_penalty: float = 0.05,
        stagnant_penalty: float = 0.05,
        score_scale: float = 0.01,
        coin_scale: float = 1.0,
        powerup_bonus: float = 5.0,
        flag_time_scale: float = 0.05,
        enemy_kill_bonus: float = 15.0,
    ):
        self.progress_scale = progress_scale
        self.time_penalty = time_penalty
        self.death_penalty = death_penalty
        self.flag_bonus = flag_bonus
        self.jump_penalty = jump_penalty
        self.stagnant_penalty = stagnant_penalty
        self.score_scale = score_scale
        self.coin_scale = coin_scale
        self.powerup_bonus = powerup_bonus
        self.flag_time_scale = flag_time_scale
        self.enemy_kill_bonus = enemy_kill_bonus

        self.prev_x = 40
        self.prev_score = 0
        self.prev_coins = 0
        self.prev_status = "small"

    def reset(
        self,
        initial_x: int = 40,
        initial_score: int = 0,
        initial_coins: int = 0,
        initial_status: str = "small",
    ) -> None:
        self.prev_x = initial_x
        self.prev_score = initial_score
        self.prev_coins = initial_coins
        self.prev_status = initial_status

    def shape(
        self,
        raw_reward: float,
        info: Dict[str, Any],
        done: bool = False,
        terminated: Optional[bool] = None,
        truncated: bool = False,
        action: Optional[int] = None,
    ) -> Tuple[float, Dict[str, float]]:
        cur_x = int(info.get("x_pos", self.prev_x))
        dx = cur_x - self.prev_x
        self.prev_x = cur_x

        cur_score = int(info.get("score", self.prev_score))
        d_score = max(0, cur_score - self.prev_score)
        self.prev_score = cur_score

        cur_coins = int(info.get("coins", self.prev_coins))
        d_coins = max(0, cur_coins - self.prev_coins)
        self.prev_coins = cur_coins

        cur_status = str(info.get("status", self.prev_status))
        powerup_gain = 0.0
        status_rank = {"small": 0, "tall": 1, "fireball": 2}
        prev_rank = status_rank.get(self.prev_status, 0)
        cur_rank = status_rank.get(cur_status, 0)
        if cur_rank > prev_rank:
            powerup_gain = self.powerup_bonus * float(cur_rank - prev_rank)
        self.prev_status = cur_status

        is_term = terminated if terminated is not None else done
        shaping = 0.0
        if dx > 0:
            shaping += float(dx) * self.progress_scale
        else:
            shaping -= self.stagnant_penalty
        shaping -= self.time_penalty

        # Jump energy penalty to prevent continuous blind hopping (Action 3: JUMP, Action 4: RIGHT+JUMP)
        if action is not None and action in (3, 4):
            shaping -= self.jump_penalty

        # Score & coin dopamine
        score_gain = float(d_score) * self.score_scale
        shaping += score_gain

        coin_gain = float(d_coins) * self.coin_scale
        shaping += coin_gain

        shaping += powerup_gain

        is_flag = bool(info.get("flag_get", False))

        # Enemy kill / stomp dopamine bonus
        enemy_bonus = 0.0
        # In SMB, stomping Goomba = +100, stomping Koopa = +100, shell kicks = +400..+1000
        # Excludes coin collections (d_coins > 0), powerups (cur_rank > prev_rank), and flag
        if not is_flag and cur_rank == prev_rank and d_coins == 0:
            if d_score in (100, 200, 400, 500, 800, 1000):
                enemy_bonus = self.enemy_kill_bonus
            elif d_score > 1000 and (d_score % 100 == 0):
                enemy_bonus = self.enemy_kill_bonus * 1.5
        shaping += enemy_bonus

        time_bonus = 0.0
        if is_term and not is_flag:
            shaping -= self.death_penalty
        if is_flag:
            shaping += self.flag_bonus
            cur_time = float(info.get("time", 0.0))
            time_bonus = cur_time * self.flag_time_scale
            shaping += time_bonus

        total_r = float(raw_reward * 0.1 + shaping)
        diag = {
            "dopamine": shaping,
            "shaped_reward": total_r,
            "raw_reward": raw_reward,
            "dx": float(dx),
            "stagnant_steps": 0.0 if dx > 0 else 1.0,
            "stagnant_penalty": -self.stagnant_penalty if dx <= 0 else 0.0,
            "score_gain": score_gain,
            "coins_gain": coin_gain,
            "powerup_gain": powerup_gain,
            "enemy_bonus": enemy_bonus,
            "time_bonus": time_bonus,
            "flag_bonus": self.flag_bonus if is_flag else 0.0,
            "death_penalty": -self.death_penalty if (is_term and not is_flag) else 0.0,
        }
        return total_r, diag


# Alias for backward compatibility
DopamineRewardShaper = MarioRewardShaper


class RolloutBuffer:
    """Stores on-policy rollout transitions for PPO optimization."""

    def __init__(self, buffer_size: int, obs_shape: Tuple[int, ...], device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.obs = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32)
        self.actions = torch.zeros(buffer_size, dtype=torch.long)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.raw_rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.terminateds = torch.zeros(buffer_size, dtype=torch.bool)
        self.truncateds = torch.zeros(buffer_size, dtype=torch.bool)
        self.values = torch.zeros(buffer_size, dtype=torch.float32)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.initial_hiddens: Dict[int, Optional[torch.Tensor]] = {}
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        done: Optional[bool] = None,
        value: float = 0.0,
        log_prob: float = 0.0,
        raw_reward: Optional[float] = None,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None,
    ) -> None:
        if self.ptr >= self.buffer_size:
            raise IndexError("RolloutBuffer overflow")
        self.obs[self.ptr] = obs.cpu()
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = float(reward)
        self.raw_rewards[self.ptr] = float(raw_reward if raw_reward is not None else reward)
        self.terminateds[self.ptr] = bool(terminated if terminated is not None else (done or False))
        self.truncateds[self.ptr] = bool(truncated if truncated is not None else False)
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = float(log_prob)
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.buffer_size

    def reset(self) -> None:
        self.ptr = 0
        self.initial_hiddens.clear()


class PPOTrainer:
    """PPO RL Trainer for Mario policies."""

    def __init__(
        self,
        model: nn.Module,
        env: Any,
        device: torch.device,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.02,
        max_grad_norm: float = 0.5,
        use_shaping: bool = True,
        use_dopamine: bool = True,
        jump_penalty: float = 0.05,
        stagnant_penalty: float = 0.05,
        death_penalty: float = 25.0,
        score_scale: float = 0.01,
        coin_scale: float = 1.0,
        powerup_bonus: float = 5.0,
        flag_time_scale: float = 0.05,
        enemy_kill_bonus: float = 15.0,
        render: bool = False,
        anneal_lr: bool = True,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.env = env
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.base_lr = lr
        self.anneal_lr = anneal_lr
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.render = render

        self.is_vector_env = hasattr(env, "num_envs") and env.num_envs > 1
        self.num_envs = env.num_envs if self.is_vector_env else 1

        if use_shaping and use_dopamine:
            self.shapers = [
                MarioRewardShaper(
                    jump_penalty=jump_penalty,
                    stagnant_penalty=stagnant_penalty,
                    death_penalty=death_penalty,
                    score_scale=score_scale,
                    coin_scale=coin_scale,
                    powerup_bonus=powerup_bonus,
                    flag_time_scale=flag_time_scale,
                    enemy_kill_bonus=enemy_kill_bonus,
                )
                for _ in range(self.num_envs)
            ]
            self.shaper = self.shapers[0]
        else:
            self.shapers = []
            self.shaper = None
        self.dopamine_shaper = self.shaper

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            eps=1e-5,
        )

    @torch.no_grad()
    def collect_rollout(
        self,
        buffer: RolloutBuffer,
        current_obs: torch.Tensor,
        current_h: Optional[torch.Tensor],
        seq_len: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any], Any]:
        """Collect transitions to fill the buffer."""
        self.model.eval()
        buffer.reset()

        if self.is_vector_env:
            steps_per_env = buffer.buffer_size // self.num_envs
            if buffer.buffer_size % self.num_envs != 0:
                raise ValueError(
                    f"Buffer size {buffer.buffer_size} must be divisible by num_envs {self.num_envs}"
                )
            if seq_len > 1 and steps_per_env % seq_len != 0:
                raise ValueError(
                    f"steps_per_env {steps_per_env} must be divisible by seq_len {seq_len}"
                )

            ep_shaped_rewards: List[float] = []
            ep_raw_rewards: List[float] = []
            ep_lens: List[int] = []
            ep_max_xs: List[int] = []

            cur_shaped_rewards = [0.0] * self.num_envs
            cur_raw_rewards = [0.0] * self.num_envs
            cur_lens = [0] * self.num_envs
            cur_max_xs = [40] * self.num_envs

            env_records: List[List[Dict[str, Any]]] = [[] for _ in range(self.num_envs)]
            obs = current_obs
            h = current_h

            for s in range(steps_per_env):
                if seq_len > 1 and (s % seq_len == 0):
                    for e in range(self.num_envs):
                        buf_idx = e * steps_per_env + s
                        buffer.initial_hiddens[buf_idx] = (
                            h[e : e + 1].detach().clone().cpu() if h is not None else None
                        )

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits, val, next_h = self.model(obs.to(self.device), h)

                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                vals = val.squeeze(-1)

                next_obs, raw_rs, terms, truncs, infos = self.env.step(actions.cpu().numpy())

                if self.render:
                    self.env.render()

                for e in range(self.num_envs):
                    term = bool(terms[e])
                    trunc = bool(truncs[e])
                    done = term or trunc
                    raw_r = float(raw_rs[e])
                    info = infos[e]
                    act = int(actions[e].item())
                    lp = float(log_probs[e].item())
                    v = float(vals[e].item())

                    if self.shapers:
                        step_r, _ = self.shapers[e].shape(
                            raw_r, info, done=done, terminated=term, truncated=trunc, action=act
                        )
                    else:
                        step_r = raw_r

                    cur_shaped_rewards[e] += step_r
                    cur_raw_rewards[e] += raw_r
                    cur_lens[e] += 1
                    cur_max_xs[e] = max(cur_max_xs[e], int(info.get("x_pos", 40)))

                    env_records[e].append({
                        "obs": obs[e],
                        "action": act,
                        "reward": step_r,
                        "raw_reward": raw_r,
                        "terminated": term,
                        "truncated": trunc,
                        "value": v,
                        "log_prob": lp,
                    })

                    if done:
                        ep_shaped_rewards.append(cur_shaped_rewards[e])
                        ep_raw_rewards.append(cur_raw_rewards[e])
                        ep_lens.append(cur_lens[e])
                        ep_max_xs.append(cur_max_xs[e])

                        cur_shaped_rewards[e] = 0.0
                        cur_raw_rewards[e] = 0.0
                        cur_lens[e] = 0
                        cur_max_xs[e] = 40

                        if self.shapers:
                            self.shapers[e].reset(
                                initial_x=int(info.get("x_pos", 40)),
                                initial_score=int(info.get("score", 0)),
                                initial_coins=int(info.get("coins", 0)),
                                initial_status=str(info.get("status", "small")),
                            )
                        next_h[e] = self.model.init_hidden(1, self.device)[0]

                obs = next_obs
                h = next_h

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                _, final_val, _ = self.model(obs.to(self.device), h)
            bootstrap_vals = [float(v) for v in final_val.squeeze(-1)]
            for e in range(self.num_envs):
                if env_records[e][-1]["terminated"]:
                    bootstrap_vals[e] = 0.0

            for e in range(self.num_envs):
                for rec in env_records[e]:
                    buffer.add(
                        obs=rec["obs"],
                        action=rec["action"],
                        reward=rec["reward"],
                        raw_reward=rec["raw_reward"],
                        terminated=rec["terminated"],
                        truncated=rec["truncated"],
                        value=rec["value"],
                        log_prob=rec["log_prob"],
                    )

            mean_sh_r = float(np.mean(ep_shaped_rewards)) if ep_shaped_rewards else float(np.mean(cur_shaped_rewards))
            stats = {
                "num_episodes": len(ep_shaped_rewards),
                "mean_reward": mean_sh_r,
                "mean_shaped_reward": mean_sh_r,
                "mean_raw_reward": float(np.mean(ep_raw_rewards)) if ep_raw_rewards else float(np.mean(cur_raw_rewards)),
                "mean_len": float(np.mean(ep_lens)) if ep_lens else float(np.mean(cur_lens)),
                "max_x": int(np.max(ep_max_xs)) if ep_max_xs else int(np.max(cur_max_xs)),
            }
            return obs, h, stats, bootstrap_vals

        # Single environment path
        ep_shaped_rewards: List[float] = []
        ep_raw_rewards: List[float] = []
        ep_lens: List[int] = []
        ep_max_xs: List[int] = []

        cur_shaped_reward = 0.0
        cur_raw_reward = 0.0
        cur_len = 0
        cur_max_x = 40

        obs = current_obs
        h = current_h
        done = False

        while not buffer.is_full():
            if seq_len > 1 and (buffer.ptr % seq_len == 0):
                buffer.initial_hiddens[buffer.ptr] = h.detach().clone().cpu() if h is not None else None
            inp = obs.unsqueeze(0).to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits, val, h = self.model(inp, h)

            dist = Categorical(logits=logits)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
            value = val.squeeze().item()

            next_obs, raw_r, term, trunc, info = self.env.step(action)
            done = term or trunc

            if self.render:
                self.env.render()

            if self.shaper is not None:
                step_reward, _ = self.shaper.shape(raw_r, info, done=done, terminated=term, truncated=trunc, action=action)
            else:
                step_reward = raw_r

            cur_shaped_reward += step_reward
            cur_raw_reward += raw_r
            cur_len += 1
            cur_max_x = max(cur_max_x, int(info.get("x_pos", 40)))

            buffer.add(
                obs=obs,
                action=action,
                reward=step_reward,
                raw_reward=raw_r,
                terminated=term,
                truncated=trunc,
                value=value,
                log_prob=log_prob,
            )

            if done:
                ep_shaped_rewards.append(cur_shaped_reward)
                ep_raw_rewards.append(cur_raw_reward)
                ep_lens.append(cur_len)
                ep_max_xs.append(cur_max_x)

                cur_shaped_reward = 0.0
                cur_raw_reward = 0.0
                cur_len = 0
                cur_max_x = 40

                obs, info = self.env.reset()
                if self.shaper is not None:
                    self.shaper.reset(
                        initial_x=int(info.get("x_pos", 40)),
                        initial_score=int(info.get("score", 0)),
                        initial_coins=int(info.get("coins", 0)),
                        initial_status=str(info.get("status", "small")),
                    )
                h = None
            else:
                obs = next_obs

        # Bootstrap value for final state
        inp = obs.unsqueeze(0).to(self.device)
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            _, final_val, _ = self.model(inp, h)
        bootstrap_val = final_val.squeeze().item() if not done else 0.0

        mean_sh_r = float(np.mean(ep_shaped_rewards)) if ep_shaped_rewards else cur_shaped_reward
        stats = {
            "num_episodes": len(ep_shaped_rewards),
            "mean_reward": mean_sh_r,
            "mean_shaped_reward": mean_sh_r,
            "mean_raw_reward": float(np.mean(ep_raw_rewards)) if ep_raw_rewards else cur_raw_reward,
            "mean_len": float(np.mean(ep_lens)) if ep_lens else cur_len,
            "max_x": int(np.max(ep_max_xs)) if ep_max_xs else cur_max_x,
        }
        return obs, h, stats, bootstrap_val


    def update(
        self,
        buffer: RolloutBuffer,
        bootstrap_val: Any,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        seq_len: int = 1,
    ) -> Dict[str, float]:
        """Perform PPO clipped surrogate update."""
        self.model.train()

        if self.is_vector_env and isinstance(bootstrap_val, (list, tuple, np.ndarray, torch.Tensor)):
            adv_list = []
            ret_list = []
            T = buffer.buffer_size // self.num_envs
            for e in range(self.num_envs):
                start_i = e * T
                end_i = (e + 1) * T
                b_val = float(bootstrap_val[e])
                adv_e, ret_e = compute_gae(
                    rewards=buffer.rewards[start_i:end_i],
                    values=buffer.values[start_i:end_i],
                    terminateds=buffer.terminateds[start_i:end_i],
                    truncateds=buffer.truncateds[start_i:end_i],
                    next_value=b_val,
                    gamma=self.gamma,
                    lam=self.gae_lambda,
                )
                adv_list.append(adv_e)
                ret_list.append(ret_e)
            advantages = torch.cat(adv_list, dim=0)
            returns = torch.cat(ret_list, dim=0)
        else:
            single_b_val = float(bootstrap_val[0]) if isinstance(bootstrap_val, (list, tuple)) else float(bootstrap_val)
            advantages, returns = compute_gae(
                rewards=buffer.rewards,
                values=buffer.values,
                terminateds=buffer.terminateds,
                truncateds=buffer.truncateds,
                next_value=single_b_val,
                gamma=self.gamma,
                lam=self.gae_lambda,
            )

        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std

        b_obs = buffer.obs.to(self.device)
        b_actions = buffer.actions.to(self.device)
        b_old_log_probs = buffer.log_probs.to(self.device)
        b_advantages = norm_advantages.to(self.device)
        b_returns = returns.to(self.device)

        N = buffer.buffer_size
        total_ploss = 0.0
        total_vloss = 0.0
        total_ent = 0.0
        total_updates = 0

        # Recurrent BPTT sequence chunking path
        if seq_len > 1 and N % seq_len == 0:
            num_chunks = N // seq_len
            chunk_indices = np.arange(num_chunks)
            chunks_per_batch = max(1, mini_batch_size // seq_len)
            b_dones = (buffer.terminateds | buffer.truncateds).to(self.device)

            for _ in range(ppo_epochs):
                np.random.shuffle(chunk_indices)
                for c_start in range(0, num_chunks, chunks_per_batch):
                    c_end = min(c_start + chunks_per_batch, num_chunks)
                    b_chunk_ids = chunk_indices[c_start:c_end]
                    B = len(b_chunk_ids)

                    step_indices = np.array([np.arange(cid * seq_len, (cid + 1) * seq_len) for cid in b_chunk_ids])
                    flat_step_idx = step_indices.reshape(-1)

                    mb_obs = b_obs[step_indices]  # (B, seq_len, frame_stack, 84, 84)
                    mb_acts = b_actions[flat_step_idx]
                    mb_old_lp = b_old_log_probs[flat_step_idx]
                    mb_adv = b_advantages[flat_step_idx]
                    mb_ret = b_returns[flat_step_idx]
                    mb_dones = b_dones[step_indices]  # (B, seq_len)

                    h_inits = []
                    for cid in b_chunk_ids:
                        init_h = buffer.initial_hiddens.get(int(cid * seq_len))
                        if init_h is not None:
                            h_inits.append(init_h.to(self.device))
                        else:
                            h_inits.append(self.model.init_hidden(1, self.device))
                    h_curr = torch.cat(h_inits, dim=0)

                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        logits_list = []
                        values_list = []
                        for t in range(seq_len):
                            if t > 0:
                                done_mask = (1.0 - mb_dones[:, t - 1].float()).unsqueeze(-1)
                                h_curr = h_curr * done_mask
                            l_t, v_t, h_curr = self.model(mb_obs[:, t], h_curr)
                            logits_list.append(l_t)
                            values_list.append(v_t.squeeze(-1))

                        mb_logits = torch.stack(logits_list, dim=1).reshape(-1, logits_list[0].shape[-1])
                        mb_logits = torch.clamp(mb_logits, -20.0, 20.0)
                        mb_values = torch.stack(values_list, dim=1).reshape(-1)

                        dist = Categorical(logits=mb_logits)
                        new_log_probs = dist.log_prob(mb_acts)
                        entropy = dist.entropy().mean()

                        ratio = torch.exp(new_log_probs - mb_old_lp)
                        surr1 = ratio * mb_adv
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_loss = 0.5 * F.mse_loss(mb_values, mb_ret)
                        # Adaptive entropy bonus scaling: prevents policy saturation & vanishing gradients when entropy collapses
                        eff_ent_coef = min(0.5, self.ent_coef * max(1.0, 0.2 / max(1e-4, entropy.item()))) if self.ent_coef > 0 else 0.0
                        loss = policy_loss + self.vf_coef * value_loss - eff_ent_coef * entropy

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        self.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    total_ploss += policy_loss.item()
                    total_vloss += value_loss.item()
                    total_ent += entropy.item()
                    total_updates += 1
        else:
            # Single-frame feedforward path
            indices = np.arange(N)
            for _ in range(ppo_epochs):
                np.random.shuffle(indices)
                for start in range(0, N, mini_batch_size):
                    end = min(start + mini_batch_size, N)
                    mb_idx = indices[start:end]

                    mb_obs = b_obs[mb_idx]
                    mb_acts = b_actions[mb_idx]
                    mb_old_lp = b_old_log_probs[mb_idx]
                    mb_adv = b_advantages[mb_idx]
                    mb_ret = b_returns[mb_idx]

                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        logits, values, _ = self.model(mb_obs, None)
                        logits = torch.clamp(logits, -20.0, 20.0)
                        dist = Categorical(logits=logits)
                        new_log_probs = dist.log_prob(mb_acts)
                        entropy = dist.entropy().mean()

                        ratio = torch.exp(new_log_probs - mb_old_lp)
                        surr1 = ratio * mb_adv
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_loss = 0.5 * F.mse_loss(values.squeeze(-1), mb_ret)
                        # Adaptive entropy bonus scaling: prevents policy saturation & vanishing gradients when entropy collapses
                        eff_ent_coef = min(0.5, self.ent_coef * max(1.0, 0.2 / max(1e-4, entropy.item()))) if self.ent_coef > 0 else 0.0
                        loss = policy_loss + self.vf_coef * value_loss - eff_ent_coef * entropy

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        self.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    total_ploss += policy_loss.item()
                    total_vloss += value_loss.item()
                    total_ent += entropy.item()
                    total_updates += 1

        return {
            "policy_loss": round(total_ploss / max(1, total_updates), 4),
            "value_loss": round(total_vloss / max(1, total_updates), 4),
            "entropy": round(total_ent / max(1, total_updates), 4),
        }

    def train(
        self,
        total_timesteps: int,
        rollout_steps: int = 512,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        checkpoint_path: Optional[Path] = None,
        csv_log_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        seq_len: int = 16,
        initial_timesteps: int = 0,
        initial_iteration: int = 0,
        initial_best_reward: float = -float("inf"),
        start_lr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run complete PPO training loop."""
        if start_lr is not None:
            self.base_lr = start_lr
        obs_shape = (self.env.frame_stack, 84, 84)
        buffer = RolloutBuffer(buffer_size=rollout_steps, obs_shape=obs_shape, device=self.device)

        if csv_log_path is not None:
            csv_log_path.parent.mkdir(parents=True, exist_ok=True)
            if not (initial_timesteps > 0 and csv_log_path.exists()):
                with open(csv_log_path, "w") as f:
                    f.write(
                        "iteration,total_timesteps,lr,mean_raw_reward,mean_shaped_reward,max_x,"
                        "policy_loss,value_loss,entropy,duration_sec,timestamp\n"
                    )

        timesteps = initial_timesteps
        target_timesteps = initial_timesteps + total_timesteps
        iteration = initial_iteration

        if self.is_vector_env:
            obs, infos = self.env.reset()
            if self.shapers:
                for e, sh in enumerate(self.shapers):
                    info = infos[e]
                    sh.reset(
                        initial_x=int(info.get("x_pos", 40)),
                        initial_score=int(info.get("score", 0)),
                        initial_coins=int(info.get("coins", 0)),
                        initial_status=str(info.get("status", "small")),
                    )
        else:
            obs, info = self.env.reset()
            if self.shaper is not None:
                self.shaper.reset(
                    initial_x=int(info.get("x_pos", 40)),
                    initial_score=int(info.get("score", 0)),
                    initial_coins=int(info.get("coins", 0)),
                    initial_status=str(info.get("status", "small")),
                )

        h: Optional[torch.Tensor] = None
        best_reward = initial_best_reward

        t_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"PPO TRAINING: Steps = {timesteps:,} -> {target_timesteps:,} (+{total_timesteps:,}) | Rollout = {rollout_steps} | Device = {self.device}")
        print(f"Envs = {self.num_envs} | Frame Stack = {self.env.frame_stack} | Frame Skip = {self.env.frame_skip} | Seq Len = {seq_len}")
        print(f"AMP = {self.use_amp} | Anneal LR = {self.anneal_lr} (Base: {self.base_lr:.2e})")
        if initial_timesteps > 0:
            print(f"Resumed from Step: {initial_timesteps:,} | Best Shaped Reward: {best_reward:.1f}")
        print(f"{'=' * 60}")

        while timesteps < target_timesteps:
            it_t0 = time.time()
            iteration += 1

            if self.anneal_lr:
                frac = max(0.0, 1.0 - ((timesteps - initial_timesteps) / max(1, total_timesteps)))
                lr_now = max(1e-7, self.base_lr * frac)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr_now
            else:
                lr_now = self.base_lr

            obs, h, roll_stats, bootstrap_val = self.collect_rollout(buffer, obs, h, seq_len=seq_len)
            timesteps += rollout_steps

            up_stats = self.update(
                buffer=buffer,
                bootstrap_val=bootstrap_val,
                ppo_epochs=ppo_epochs,
                mini_batch_size=mini_batch_size,
                seq_len=seq_len,
            )

            it_duration = round(time.time() - it_t0, 2)
            raw_r = roll_stats["mean_raw_reward"]
            shaped_r = roll_stats["mean_shaped_reward"]
            max_x = roll_stats["max_x"]

            is_best = shaped_r > best_reward
            ckpt_data = {
                "iteration": iteration,
                "timesteps": timesteps,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
                "lr": lr_now,
                "mean_raw_reward": raw_r,
                "mean_shaped_reward": shaped_r,
                "max_x": max_x,
                "config": config or {},
            }

            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                # Save latest checkpoint on every iteration
                latest_path = checkpoint_path.parent / f"{checkpoint_path.stem}_latest{checkpoint_path.suffix}"
                torch.save(ckpt_data, latest_path)

                if is_best:
                    best_reward = shaped_r
                    torch.save(ckpt_data, checkpoint_path)

            if csv_log_path is not None:
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                with open(csv_log_path, "a") as f:
                    f.write(
                        f"{iteration},{timesteps},{lr_now:.6e},{raw_r:.2f},{shaped_r:.2f},{max_x},"
                        f"{up_stats['policy_loss']:.4f},{up_stats['value_loss']:.4f},"
                        f"{up_stats['entropy']:.4f},{it_duration:.2f},{ts}\n"
                    )

            best_mark = " [*BEST*]" if is_best else ""
            print(
                f"Iter {iteration:3d} | Steps: {timesteps:6,d}/{target_timesteps:,d} "
                f"({it_duration:4.1f}s) | LR: {lr_now:.2e} | RawR: {raw_r:6.1f} | ShapedR: {shaped_r:6.1f} | "
                f"Max X: {max_x:4d} | P-Loss: {up_stats['policy_loss']:6.3f} | "
                f"V-Loss: {up_stats['value_loss']:6.3f} | Ent: {up_stats['entropy']:5.3f}{best_mark}"
            )

        total_time = round(time.time() - t_start, 2)
        print(f"\nTraining Complete in {total_time}s! Best Shaped Reward: {best_reward:.1f}")
        return {
            "total_timesteps": timesteps,
            "total_iterations": iteration,
            "best_shaped_reward": best_reward,
            "total_time_seconds": total_time,
        }



def main():
    parser = argparse.ArgumentParser(description="PPO Trainer for Mario Policies")
    parser.add_argument("--model-type", type=str, default="connectome", choices=["connectome", "baseline"])
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments (default: 4)")
    parser.add_argument("--total-timesteps", type=int, default=10000, help="Total environment steps")
    parser.add_argument("--rollout-steps", type=int, default=512, help="Rollout steps per iteration")
    parser.add_argument("--frame-stack", type=int, default=4, help="Frame stack count (default: 4)")
    parser.add_argument("--frame-skip", type=int, default=4, help="Frame skip count (default: 4)")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True, help="Anneal learning rate linearly to 0 (default: True)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Automatic Mixed Precision FP16 on CUDA (default: True)")
    parser.add_argument("--ent-coef", type=float, default=0.02, help="Entropy coefficient")
    parser.add_argument("--checkpoint-in", type=str, default=None, help="Input pretrained checkpoint")
    parser.add_argument("--checkpoint-out", type=str, default=None, help="Output checkpoint save path")
    parser.add_argument("--log-csv", type=str, default=None, help="Path for CSV metrics log")
    parser.add_argument("--render", action="store_true", help="Render GUI during training")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length for recurrent BPTT (default: 16)")
    parser.add_argument("--auto-unlatch", action=argparse.BooleanOptionalAction, default=False, help="Auto-unlatch button A on ground (default: False)")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training steps counter and append logs from checkpoint")
    parser.add_argument("--death-penalty", type=float, default=25.0, help="Penalty applied when Mario dies (default: 25.0)")
    parser.add_argument("--jump-penalty", type=float, default=0.05, help="Jump metabolic penalty to prevent blind hopping (default: 0.05)")
    parser.add_argument("--stagnant-penalty", type=float, default=0.05, help="Penalty applied when Mario does not make forward progress (default: 0.05)")
    parser.add_argument("--score-scale", type=float, default=0.01, help="Reward scaling for score gain (default: 0.01)")
    parser.add_argument("--coin-scale", type=float, default=1.0, help="Reward bonus for coin collected (default: 1.0)")
    parser.add_argument("--powerup-bonus", type=float, default=5.0, help="Reward bonus for powerup collected (default: 5.0)")
    parser.add_argument("--flag-time-scale", type=float, default=0.05, help="Reward scaling for remaining time on flag clearance (default: 0.05)")
    parser.add_argument("--enemy-kill-bonus", type=float, default=15.0, help="Dopamine reward bonus for stomping or killing an enemy (default: 15.0)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    if args.num_envs > 1:
        env = VectorMarioEnv(
            num_envs=args.num_envs,
            frame_stack=args.frame_stack,
            frame_skip=args.frame_skip,
            noop_max=30,
            auto_unlatch=args.auto_unlatch,
            render_mode="human" if args.render else None,
        )
    else:
        env = Mario2DEnv(
            frame_stack=args.frame_stack,
            frame_skip=args.frame_skip,
            noop_max=30,
            auto_unlatch=args.auto_unlatch,
            render_mode="human" if args.render else None,
        )

    processed_dir = Path("data/processed")
    if args.model_type == "connectome":
        model = ConnectomeMarioPolicy(
            subgraph_npz_path=processed_dir / "subgraph_graph.npz",
            node_mapping_csv_path=processed_dir / "subgraph_node_mapping.csv",
            node_metadata_csv_path=processed_dir / "node_metadata.csv",
            in_channels=args.frame_stack,
        ).to(device)
    else:
        model = BaselineMarioPolicy(in_channels=args.frame_stack, hidden_dim=1024).to(device)

    init_timesteps = 0
    init_iteration = 0
    init_best_reward = -float("inf")
    saved_lr = None
    saved_optim_state = None
    saved_scaler_state = None

    ckpt_in_path = args.checkpoint_in
    if args.resume and not ckpt_in_path:
        candidates = [
            Path(f"checkpoints/{args.model_type}_ppo_v2_latest.pt"),
            Path(f"checkpoints/{args.model_type}_ppo_v2.pt"),
        ]
        best_cand = None
        best_iter = -1
        for cand in candidates:
            if cand.exists():
                try:
                    c_data = torch.load(cand, map_location="cpu", weights_only=False)
                    it = int(c_data.get("iteration", 0))
                    if it > best_iter:
                        best_iter = it
                        best_cand = cand
                except Exception:
                    if best_cand is None:
                        best_cand = cand
        if best_cand is not None:
            ckpt_in_path = str(best_cand)

    # Optional weight transfer / resume
    if ckpt_in_path and Path(ckpt_in_path).exists():
        print(f"Loading weights from {ckpt_in_path}...")
        ckpt = torch.load(ckpt_in_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        msd = model.state_dict()
        compat = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        msd.update(compat)
        model.load_state_dict(msd)
        print(f"Loaded {len(compat)}/{len(msd)} parameter tensors.")

        if args.resume:
            init_timesteps = int(ckpt.get("timesteps", 0))
            init_iteration = int(ckpt.get("iteration", 0))
            init_best_reward = float(ckpt.get("mean_shaped_reward", -float("inf")))
            saved_lr = ckpt.get("lr")
            saved_optim_state = ckpt.get("optimizer_state_dict")
            saved_scaler_state = ckpt.get("scaler_state_dict")
            print(f"Resuming: Iteration {init_iteration:,} | Steps {init_timesteps:,} | Best Reward {init_best_reward:.1f}")
            if saved_lr is not None:
                print(f"Preserved Learning Rate: {saved_lr:.2e}")

    ckpt_out = Path(args.checkpoint_out) if args.checkpoint_out else Path(f"checkpoints/{args.model_type}_ppo_v2.pt")
    csv_log = Path(args.log_csv) if args.log_csv else Path(f"logs/{args.model_type}_ppo_v2_metrics.csv")

    config = {
        "model_type": args.model_type,
        "num_envs": args.num_envs,
        "frame_stack": args.frame_stack,
        "frame_skip": args.frame_skip,
        "seq_len": args.seq_len,
        "auto_unlatch": args.auto_unlatch,
        "lr": args.lr,
        "anneal_lr": args.anneal_lr,
        "amp": args.amp,
        "ent_coef": args.ent_coef,
        "death_penalty": args.death_penalty,
        "jump_penalty": args.jump_penalty,
        "stagnant_penalty": args.stagnant_penalty,
        "enemy_kill_bonus": args.enemy_kill_bonus,
        "seed": args.seed,
    }

    effective_lr = saved_lr if (args.resume and saved_lr is not None and ("--lr" not in sys.argv)) else args.lr

    trainer = PPOTrainer(
        model=model,
        env=env,
        device=device,
        lr=effective_lr,
        ent_coef=args.ent_coef,
        death_penalty=args.death_penalty,
        jump_penalty=args.jump_penalty,
        stagnant_penalty=args.stagnant_penalty,
        score_scale=args.score_scale,
        coin_scale=args.coin_scale,
        powerup_bonus=args.powerup_bonus,
        flag_time_scale=args.flag_time_scale,
        enemy_kill_bonus=args.enemy_kill_bonus,
        render=args.render,
        anneal_lr=args.anneal_lr,
        use_amp=args.amp,
    )

    if args.resume:
        if saved_optim_state is not None:
            try:
                trainer.optimizer.load_state_dict(saved_optim_state)
                print("Restored optimizer state dict (Adam momentums).")
            except Exception as e:
                print(f"Note: Could not restore optimizer state dict: {e}")
        if saved_scaler_state is not None and trainer.use_amp:
            try:
                trainer.scaler.load_state_dict(saved_scaler_state)
                print("Restored GradScaler state dict.")
            except Exception as e:
                print(f"Note: Could not restore GradScaler state dict: {e}")

    trainer.train(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        checkpoint_path=ckpt_out,
        csv_log_path=csv_log,
        config=config,
        seq_len=args.seq_len,
        initial_timesteps=init_timesteps,
        initial_iteration=init_iteration,
        initial_best_reward=init_best_reward,
        start_lr=effective_lr,
    )
    env.close()


if __name__ == "__main__":
    main()
