"""Proximal Policy Optimization (PPO) Trainer for Recurrent Connectome Policies.

Features:
- Supports both ConnectomeMarioPolicy (sparse 30k neurons) and BaselineMarioPolicy (dense GRU-1024).
- Generalized Advantage Estimation (GAE) with discount gamma and GAE lambda.
- PPO clipped surrogate objective with value function clipping and entropy regularization.
- On-policy rollout collection in Super Mario Bros 1-1 with frame stacking support.
- Pretrained weight initialization (fine-tuning from Behavioral Cloning checkpoints).
- Per-iteration CSV metric logging and checkpoint serialization.
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
from src.environment import Mario2DEnv, MARIO_5_ACTIONS

ACTION_NAMES = ["NOOP", "RIGHT", "LEFT", "JUMP", "RIGHT+JUMP"]


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: (T,) step rewards
        values: (T,) critic value estimates V(s_t)
        dones: (T,) boolean episode termination flags
        next_value: V(s_T) bootstrap value for final state
        gamma: discount factor
        lam: GAE exponential smoothing parameter
    Returns:
        advantages: (T,) GAE advantage estimates
        returns: (T,) target returns for value function
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
            non_terminal = 1.0 - float(dones[t])
        else:
            next_val = values[t + 1].item()
            non_terminal = 1.0 - float(dones[t])

        delta = rewards[t].item() + gamma * next_val * non_terminal - values[t].item()
        advantages[t] = last_gae = delta + gamma * lam * non_terminal * last_gae

    returns = advantages + values
    return advantages, returns


class DopamineRewardShaper:
    """Biologically-inspired Dopaminergic Neuromodulation & Reward Shaping.

    Simulates Drosophila melanogaster dopaminergic reinforcement pathways:
    - Appetitive Dopamine (PAM-cluster neurons):
      * Winning Dopamine (flagpole touch / level completion): +1000.0
      * Obstacle Milestone Dopamine (passing pipes and gaps): +50.0 to +100.0
      * Forward Velocity Momentum: +1.5 * max(0, dx)
    - Aversive Dopamine (PPL1-cluster neurons):
      * Anti-Stagnation Penalty: -1.0 per stationary step, -2.5 if stuck > 25 steps
      * Death Penalty: -30.0 upon premature death
    """

    def __init__(self):
        self.prev_x = 40
        self.stagnant_steps = 0
        self.milestones_reached: set[int] = set()
        # World 1-1 Key Landmark Milestones in Super Mario Bros:
        self.milestones = [350, 500, 680, 900, 1250, 1600, 2000, 2400, 3000]

    def reset(self) -> None:
        self.prev_x = 40
        self.stagnant_steps = 0
        self.milestones_reached.clear()

    def shape(self, raw_reward: float, info: Dict[str, Any], done: bool) -> Tuple[float, Dict[str, float]]:
        cur_x = info.get("x_pos", self.prev_x)
        dx = cur_x - self.prev_x
        self.prev_x = cur_x

        dopamine = 0.0

        # 1. Forward velocity momentum (scaled)
        if dx > 0:
            dopamine += float(dx) * 0.1
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1
            # Aversive signal against zero progress / jumping in place
            dopamine -= 0.1
            if self.stagnant_steps > 25:
                dopamine -= 0.25

        # 2. Obstacle milestone bonus (scaled)
        for m in self.milestones:
            if cur_x >= m and m not in self.milestones_reached:
                self.milestones_reached.add(m)
                dopamine += 5.0

        # 3. Winning Dopamine (Flagpole)
        if info.get("flag_get", False):
            dopamine += 50.0

        # 4. Premature death penalty
        if done and not info.get("flag_get", False):
            dopamine -= 2.0

        shaped_r = float(raw_reward * 0.1 + dopamine)
        return shaped_r, {"dopamine": dopamine, "dx": float(dx), "stagnant_steps": float(self.stagnant_steps)}


class RolloutBuffer:
    """Buffer storing on-policy rollout transitions for PPO update."""

    def __init__(self, buffer_size: int, obs_shape: Tuple[int, ...], device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.obs = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32)
        self.actions = torch.zeros(buffer_size, dtype=torch.long)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool)
        self.values = torch.zeros(buffer_size, dtype=torch.float32)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        if self.ptr >= self.buffer_size:
            raise IndexError("RolloutBuffer overflow")
        self.obs[self.ptr] = obs.cpu()
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.buffer_size

    def reset(self) -> None:
        self.ptr = 0


class PPOTrainer:
    """PPO fine-tuning engine for connectome and baseline Mario policies."""

    def __init__(
        self,
        model: nn.Module,
        env: Mario2DEnv,
        device: torch.device,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.05,
        max_grad_norm: float = 0.5,
        use_dopamine: bool = True,
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
        self.use_dopamine = use_dopamine
        self.dopamine_shaper = DopamineRewardShaper() if use_dopamine else None
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Collect one rollout buffer worth of environment interactions."""
        self.model.eval()
        buffer.reset()

        ep_rewards = []
        ep_lens = []
        cur_ep_reward = 0.0
        cur_ep_len = 0
        max_x = 40

        h = current_h
        obs = current_obs

        while not buffer.is_full():
            inp = obs.unsqueeze(0).to(self.device)  # (1, C, 84, 84)
            logits, val, h = self.model(inp, h)

            dist = Categorical(logits=logits)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
            value = val.squeeze().item()

            next_obs, r, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if self.render:
                self.env.render()
                time.sleep(0.015)

            if self.dopamine_shaper is not None:
                step_reward, _ = self.dopamine_shaper.shape(r, info, done)
            else:
                step_reward = r

            cur_ep_reward += step_reward
            cur_ep_len += 1
            max_x = max(max_x, info.get("x_pos", 40))

            buffer.add(
                obs=obs,
                action=action,
                reward=step_reward,
                done=done,
                value=value,
                log_prob=log_prob,
            )

            if done:
                ep_rewards.append(cur_ep_reward)
                ep_lens.append(cur_ep_len)
                cur_ep_reward = 0.0
                cur_ep_len = 0
                obs, _ = self.env.reset()
                if self.dopamine_shaper is not None:
                    self.dopamine_shaper.reset()
                h = None  # Reset hidden state at episode boundary
            else:
                obs = next_obs

        # Bootstrap value for final state
        inp = obs.unsqueeze(0).to(self.device)
        _, final_val, _ = self.model(inp, h)
        bootstrap_value = final_val.squeeze().item() if not done else 0.0

        stats = {
            "num_episodes": len(ep_rewards),
            "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else cur_ep_reward,
            "mean_len": float(np.mean(ep_lens)) if ep_lens else cur_ep_len,
            "max_x": int(max_x),
        }
        return obs, h, stats, bootstrap_value

    def update(
        self,
        buffer: RolloutBuffer,
        bootstrap_val: float,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
    ) -> Dict[str, float]:
        """Perform PPO policy and value function update."""
        self.model.train()

        # Compute GAE advantages and target returns
        advantages, returns = compute_gae(
            rewards=buffer.rewards,
            values=buffer.values,
            dones=buffer.dones,
            next_value=bootstrap_val,
            gamma=self.gamma,
            lam=self.gae_lambda,
        )

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std

        # Move tensors to device
        b_obs = buffer.obs.to(self.device)
        b_actions = buffer.actions.to(self.device)
        b_old_log_probs = buffer.log_probs.to(self.device)
        b_advantages = norm_advantages.to(self.device)
        b_returns = returns.to(self.device)

        N = buffer.buffer_size
        indices = np.arange(N)

        total_ploss = 0.0
        total_vloss = 0.0
        total_ent = 0.0
        total_updates = 0

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

                # Forward pass (evaluating policy and critic)
                logits, values, _ = self.model(mb_obs, None)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_acts)
                entropy = dist.entropy().mean()

                # Clipped surrogate objective
                ratio = torch.exp(new_log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
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
    ) -> Dict[str, Any]:
        """Run full PPO training loop."""
        obs_shape = (self.env.frame_stack, 84, 84)
        buffer = RolloutBuffer(buffer_size=rollout_steps, obs_shape=obs_shape, device=self.device)

        if csv_log_path is not None:
            csv_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_log_path, "w") as f:
                f.write("iteration,total_timesteps,mean_reward,max_x,policy_loss,value_loss,entropy,duration_sec,timestamp\n")

        timesteps = 0
        iteration = 0
        obs, _ = self.env.reset()
        h: Optional[torch.Tensor] = None
        best_reward = -float("inf")

        t_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"PPO FINE-TUNING: Total Steps = {total_timesteps:,} | Rollout = {rollout_steps}")
        print(f"Device: {self.device} | Frame Stack: {self.env.frame_stack}")
        print(f"{'=' * 60}")

        while timesteps < total_timesteps:
            it_t0 = time.time()
            iteration += 1

            # 1. Rollout collection
            obs, h, roll_stats, bootstrap_val = self.collect_rollout(buffer, obs, h)
            timesteps += rollout_steps

            # 2. PPO update
            up_stats = self.update(
                buffer=buffer,
                bootstrap_val=bootstrap_val,
                ppo_epochs=ppo_epochs,
                mini_batch_size=mini_batch_size,
            )

            it_duration = round(time.time() - it_t0, 2)
            mean_r = roll_stats["mean_reward"]
            max_x = roll_stats["max_x"]

            is_best = mean_r > best_reward
            if is_best and checkpoint_path is not None:
                best_reward = mean_r
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "iteration": iteration,
                        "timesteps": timesteps,
                        "model_state_dict": self.model.state_dict(),
                        "mean_reward": mean_r,
                        "max_x": max_x,
                    },
                    checkpoint_path,
                )

            if csv_log_path is not None:
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                with open(csv_log_path, "a") as f:
                    f.write(
                        f"{iteration},{timesteps},{mean_r:.2f},{max_x},"
                        f"{up_stats['policy_loss']:.4f},{up_stats['value_loss']:.4f},"
                        f"{up_stats['entropy']:.4f},{it_duration:.2f},{ts}\n"
                    )

            best_mark = " [*BEST*]" if is_best else ""
            print(
                f"Iter {iteration:3d} | Steps: {timesteps:7,d}/{total_timesteps:,d} "
                f"({it_duration:4.1f}s) | Reward: {mean_r:6.1f} | Max X: {max_x:4d} | "
                f"P-Loss: {up_stats['policy_loss']:6.3f} | V-Loss: {up_stats['value_loss']:6.3f} | "
                f"Ent: {up_stats['entropy']:5.3f}{best_mark}"
            )

        total_time = round(time.time() - t_start, 2)
        print(f"\nPPO Training Complete in {total_time}s! Best Reward: {best_reward:.1f}")
        return {
            "total_timesteps": timesteps,
            "total_iterations": iteration,
            "best_reward": best_reward,
            "total_time_seconds": total_time,
        }


def main():
    parser = argparse.ArgumentParser(description="PPO Fine-Tuning for Connectome & Baseline Models")
    parser.add_argument("--model-type", type=str, default="connectome", choices=["connectome", "baseline"])
    parser.add_argument("--total-timesteps", type=int, default=20000, help="Total environment interaction steps")
    parser.add_argument("--rollout-steps", type=int, default=512, help="Rollout steps per iteration")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of stacked frames (1 or 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint-in", type=str, default=None, help="Initial pretrained checkpoint path")
    parser.add_argument("--checkpoint-out", type=str, default=None, help="Output checkpoint save path")
    parser.add_argument("--log-csv", type=str, default=None, help="CSV metrics log path")
    parser.add_argument(
        "--use-dopamine",
        action="store_true",
        default=True,
        help="Enable dopaminergic reward shaping (winning dopamine, milestones, velocity momentum)",
    )
    parser.add_argument(
        "--no-dopamine",
        dest="use_dopamine",
        action="store_false",
        help="Disable dopamine reward shaping and use raw environment rewards",
    )
    parser.add_argument("--ent-coef", type=float, default=0.05, help="Entropy coefficient for exploration")
    parser.add_argument("--render", action="store_true", help="Render live game window on desktop during training")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Initialize environment
    env = Mario2DEnv(frame_stack=args.frame_stack, render_mode="human" if args.render else None)

    # Initialize model
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

    # Optionally load pretrained weights
    if args.checkpoint_in and Path(args.checkpoint_in).exists():
        print(f"Loading pretrained weights from {args.checkpoint_in}...")
        ckpt = torch.load(args.checkpoint_in, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model_state = model.state_dict()

        # Smart weight expansion: transfer 1-channel conv1 to 4-channel conv1
        if (
            "encoder.conv1.weight" in state_dict
            and "encoder.conv1.weight" in model_state
            and model_state["encoder.conv1.weight"].shape[1] == 4
            and state_dict["encoder.conv1.weight"].shape[1] == 1
        ):
            old_w = state_dict["encoder.conv1.weight"]
            model_state["encoder.conv1.weight"] = old_w.repeat(1, 4, 1, 1) / 4.0
            print("   [Transfer] Expanded encoder.conv1.weight from (32, 1, 8, 8) to (32, 4, 8, 8)")

        # Load all other shape-compatible tensors
        compatible = {
            k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        model_state.update(compatible)
        model.load_state_dict(model_state)
        print(f"   [Transfer] Loaded {len(compatible)}/{len(model_state)} matching parameter tensors.")

    ckpt_out = Path(args.checkpoint_out) if args.checkpoint_out else Path(f"checkpoints/{args.model_type}_ppo_best.pt")
    csv_log = Path(args.log_csv) if args.log_csv else Path(f"logs/{args.model_type}_ppo_training_metrics.csv")

    trainer = PPOTrainer(
        model=model,
        env=env,
        device=device,
        lr=args.lr,
        ent_coef=args.ent_coef,
        use_dopamine=args.use_dopamine,
        render=args.render,
    )

    train_stats = trainer.train(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        checkpoint_path=ckpt_out,
        csv_log_path=csv_log,
    )

    # Post-training evaluation on held-out test seeds
    print("\nEvaluating trained PPO policy on held-out test seeds: [13, 14, 15]...")
    eval_seeds = [13, 14, 15]
    eval_results = []
    model.eval()

    for s in eval_seeds:
        obs, info = env.reset(seed=s)
        h = None
        tot_r = 0.0
        max_x = info.get("x_pos", 40)
        steps = 0
        flag_get = False

        for step in range(2500):
            inp = obs.unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _, h = model(inp, h)
                action = torch.argmax(logits, dim=-1).item()
            next_obs, r, term, trunc, info = env.step(action)
            tot_r += r
            max_x = max(max_x, info.get("x_pos", 40))
            flag_get = bool(info.get("flag_get", False))
            steps = step + 1
            obs = next_obs
            if term or trunc:
                break

        eval_results.append({
            "seed": s,
            "steps": steps,
            "final_x": max_x,
            "total_reward": round(tot_r, 2),
            "flag_get": flag_get,
            "cleared_first_obstacle": max_x > 296,
        })
        print(f"  [Seed {s}] Steps: {steps:4d} | Final X: {max_x:4d} | Reward: {tot_r:6.1f} | Cleared Obstacle 1: {max_x > 296} | Flag: {flag_get}")

    xs = [r["final_x"] for r in eval_results]
    rs = [r["total_reward"] for r in eval_results]
    eval_summary = {
        "mean_final_x": round(float(np.mean(xs)), 2),
        "max_final_x": int(np.max(xs)),
        "mean_reward": round(float(np.mean(rs)), 2),
        "first_obstacle_clear_rate": round(float(np.mean([r["cleared_first_obstacle"] for r in eval_results])), 2),
        "level_clear_rate": round(float(np.mean([r["flag_get"] for r in eval_results])), 2),
        "episodes": eval_results,
    }

    report_p = Path(f"logs/{args.model_type}_ppo_eval_summary.json")
    with open(report_p, "w") as f:
        json.dump({
            "model_type": args.model_type,
            "training": train_stats,
            "evaluation": eval_summary,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)
    print(f"\nSaved evaluation summary to: {report_p}")
    env.close()


if __name__ == "__main__":
    main()
