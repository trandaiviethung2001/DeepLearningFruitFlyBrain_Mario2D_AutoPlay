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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.baseline_model import BaselineMarioPolicy
from src.connectome_model import ConnectomeMarioPolicy
from src.environment import Mario2DEnv, WORLD_1_1_FLAG_X

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
        death_penalty: float = 15.0,
        flag_bonus: float = 50.0,
        jump_penalty: float = 0.0,
        score_scale: float = 0.01,
        coin_scale: float = 1.0,
        powerup_bonus: float = 5.0,
        flag_time_scale: float = 0.05,
    ):
        self.progress_scale = progress_scale
        self.time_penalty = time_penalty
        self.death_penalty = death_penalty
        self.flag_bonus = flag_bonus
        self.jump_penalty = jump_penalty
        self.score_scale = score_scale
        self.coin_scale = coin_scale
        self.powerup_bonus = powerup_bonus
        self.flag_time_scale = flag_time_scale

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
        shaping -= self.time_penalty

        # Jump energy penalty to prevent continuous blind hopping
        if action is not None and action in (2, 3, 4):
            shaping -= self.jump_penalty

        # Score & coin dopamine
        score_gain = float(d_score) * self.score_scale
        shaping += score_gain

        coin_gain = float(d_coins) * self.coin_scale
        shaping += coin_gain

        shaping += powerup_gain

        time_bonus = 0.0
        is_flag = bool(info.get("flag_get", False))
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
            "score_gain": score_gain,
            "coins_gain": coin_gain,
            "powerup_gain": powerup_gain,
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
        env: Mario2DEnv,
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
        jump_penalty: float = 0.0,
        score_scale: float = 0.01,
        coin_scale: float = 1.0,
        powerup_bonus: float = 5.0,
        flag_time_scale: float = 0.05,
        render: bool = False,
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
        self.shaper = (
            MarioRewardShaper(
                jump_penalty=jump_penalty,
                score_scale=score_scale,
                coin_scale=coin_scale,
                powerup_bonus=powerup_bonus,
                flag_time_scale=flag_time_scale,
            )
            if (use_shaping and use_dopamine)
            else None
        )
        self.dopamine_shaper = self.shaper
        self.render = render

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any], float]:
        """Collect transitions to fill the buffer."""
        self.model.eval()
        buffer.reset()

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
        bootstrap_val: float,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        seq_len: int = 1,
    ) -> Dict[str, float]:
        """Perform PPO clipped surrogate update."""
        self.model.train()

        advantages, returns = compute_gae(
            rewards=buffer.rewards,
            values=buffer.values,
            terminateds=buffer.terminateds,
            truncateds=buffer.truncateds,
            next_value=bootstrap_val,
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
                    mb_values = torch.stack(values_list, dim=1).reshape(-1)

                    dist = Categorical(logits=mb_logits)
                    new_log_probs = dist.log_prob(mb_acts)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - mb_old_lp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = 0.5 * F.mse_loss(mb_values, mb_ret)
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()

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

                    logits, values, _ = self.model(mb_obs, None)
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(mb_acts)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - mb_old_lp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = 0.5 * F.mse_loss(values.squeeze(-1), mb_ret)
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()

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
    ) -> Dict[str, Any]:
        """Run complete PPO training loop."""
        obs_shape = (self.env.frame_stack, 84, 84)
        buffer = RolloutBuffer(buffer_size=rollout_steps, obs_shape=obs_shape, device=self.device)

        if csv_log_path is not None:
            csv_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_log_path, "w") as f:
                f.write(
                    "iteration,total_timesteps,mean_raw_reward,mean_shaped_reward,max_x,"
                    "policy_loss,value_loss,entropy,duration_sec,timestamp\n"
                )

        timesteps = 0
        iteration = 0
        obs, info = self.env.reset()
        if self.shaper is not None:
            self.shaper.reset(
                initial_x=int(info.get("x_pos", 40)),
                initial_score=int(info.get("score", 0)),
                initial_coins=int(info.get("coins", 0)),
                initial_status=str(info.get("status", "small")),
            )
        h: Optional[torch.Tensor] = None
        best_reward = -float("inf")

        t_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"PPO TRAINING: Steps = {total_timesteps:,} | Rollout = {rollout_steps} | Device = {self.device}")
        print(f"Frame Stack = {self.env.frame_stack} | Frame Skip = {self.env.frame_skip} | Seq Len = {seq_len}")
        print(f"{'=' * 60}")

        while timesteps < total_timesteps:
            it_t0 = time.time()
            iteration += 1

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
            if is_best and checkpoint_path is not None:
                best_reward = shaped_r
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                ckpt_data = {
                    "iteration": iteration,
                    "timesteps": timesteps,
                    "model_state_dict": self.model.state_dict(),
                    "mean_raw_reward": raw_r,
                    "mean_shaped_reward": shaped_r,
                    "max_x": max_x,
                    "config": config or {},
                }
                torch.save(ckpt_data, checkpoint_path)

            if csv_log_path is not None:
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                with open(csv_log_path, "a") as f:
                    f.write(
                        f"{iteration},{timesteps},{raw_r:.2f},{shaped_r:.2f},{max_x},"
                        f"{up_stats['policy_loss']:.4f},{up_stats['value_loss']:.4f},"
                        f"{up_stats['entropy']:.4f},{it_duration:.2f},{ts}\n"
                    )

            best_mark = " [*BEST*]" if is_best else ""
            print(
                f"Iter {iteration:3d} | Steps: {timesteps:6,d}/{total_timesteps:,d} "
                f"({it_duration:4.1f}s) | RawR: {raw_r:6.1f} | ShapedR: {shaped_r:6.1f} | "
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
    parser.add_argument("--total-timesteps", type=int, default=10000, help="Total environment steps")
    parser.add_argument("--rollout-steps", type=int, default=512, help="Rollout steps per iteration")
    parser.add_argument("--frame-stack", type=int, default=4, help="Frame stack count (default: 4)")
    parser.add_argument("--frame-skip", type=int, default=4, help="Frame skip count (default: 4)")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.02, help="Entropy coefficient")
    parser.add_argument("--checkpoint-in", type=str, default=None, help="Input pretrained checkpoint")
    parser.add_argument("--checkpoint-out", type=str, default=None, help="Output checkpoint save path")
    parser.add_argument("--log-csv", type=str, default=None, help="Path for CSV metrics log")
    parser.add_argument("--render", action="store_true", help="Render GUI during training")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length for recurrent BPTT (default: 16)")
    parser.add_argument("--auto-unlatch", action=argparse.BooleanOptionalAction, default=False, help="Auto-unlatch button A on ground (default: False)")
    parser.add_argument("--jump-penalty", type=float, default=0.05, help="Jump metabolic penalty to prevent blind hopping (default: 0.05)")
    parser.add_argument("--score-scale", type=float, default=0.01, help="Reward scaling for score gain (default: 0.01)")
    parser.add_argument("--coin-scale", type=float, default=1.0, help="Reward bonus for coin collected (default: 1.0)")
    parser.add_argument("--powerup-bonus", type=float, default=5.0, help="Reward bonus for powerup collected (default: 5.0)")
    parser.add_argument("--flag-time-scale", type=float, default=0.05, help="Reward scaling for remaining time on flag clearance (default: 0.05)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

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

    # Optional weight transfer
    if args.checkpoint_in and Path(args.checkpoint_in).exists():
        print(f"Loading weights from {args.checkpoint_in}...")
        ckpt = torch.load(args.checkpoint_in, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        msd = model.state_dict()
        compat = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        msd.update(compat)
        model.load_state_dict(msd)
        print(f"Loaded {len(compat)}/{len(msd)} parameter tensors.")

    ckpt_out = Path(args.checkpoint_out) if args.checkpoint_out else Path(f"checkpoints/{args.model_type}_ppo_v2.pt")
    csv_log = Path(args.log_csv) if args.log_csv else Path(f"logs/{args.model_type}_ppo_v2_metrics.csv")

    config = {
        "model_type": args.model_type,
        "frame_stack": args.frame_stack,
        "frame_skip": args.frame_skip,
        "seq_len": args.seq_len,
        "auto_unlatch": args.auto_unlatch,
        "lr": args.lr,
        "ent_coef": args.ent_coef,
        "seed": args.seed,
    }

    trainer = PPOTrainer(
        model=model,
        env=env,
        device=device,
        lr=args.lr,
        ent_coef=args.ent_coef,
        jump_penalty=args.jump_penalty,
        score_scale=args.score_scale,
        coin_scale=args.coin_scale,
        powerup_bonus=args.powerup_bonus,
        flag_time_scale=args.flag_time_scale,
        render=args.render,
    )

    trainer.train(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        checkpoint_path=ckpt_out,
        csv_log_path=csv_log,
        config=config,
        seq_len=args.seq_len,
    )
    env.close()


if __name__ == "__main__":
    main()
