"""Mario 2D Demonstration Collector & Demonstrator Benchmark.

Provides:
- ScriptedDemonstrator: RAM-aware teacher policy with obstacle, enemy, and pit navigation.
- RightOnlyPolicy & RandomPolicy: Baselines for fair evaluation.
- Dataset collection pipeline storing compact (uint8) transition tensors with episode-level splits.
- Automated benchmarking comparing demonstrator against naive baselines.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.environment import Mario2DEnv, MARIO_5_ACTIONS

ACTION_NAMES = [
    "NOOP",         # 0
    "RIGHT",        # 1
    "LEFT",         # 2
    "JUMP",         # 3
    "RIGHT+JUMP",   # 4
]


class ScriptedDemonstrator:
    """Privileged RAM-aware teacher policy for Super Mario Bros World 1-1.
    
    Uses NES RAM state to detect:
    - Active ground enemies (Goombas, Koopas) approaching Mario.
    - Level obstacles (pipes, pits) via geometric waypoints.
    - Mario airborne status (RAM 0x001D) to only initiate jumps from the ground.
    - Horizontal progress stalls (stuck detection) to clear unforeseen obstacles.
    """

    DEFAULT_OBSTACLES: List[Tuple[int, int, str]] = [
        (420, 435, "pipe1"),
        (600, 615, "pipe2"),
        (700, 715, "pipe3"),
        (890, 905, "pipe4"),
        (1075, 1095, "pit1"),
        (1345, 1368, "pit2"),
        (2430, 2445, "pit3"),
    ]

    def __init__(
        self,
        jump_hold_min: int = 20,
        jump_hold_max: int = 24,
        stuck_threshold: int = 5,
        enemy_lookahead_min: int = 0,
        enemy_lookahead_max: int = 52,
        enemy_y_tolerance: int = 35,
        start_noop_min: int = 2,
        start_noop_max: int = 15,
        obstacles: Optional[List[Tuple[int, int, str]]] = None,
    ):
        self.jump_hold_min = jump_hold_min
        self.jump_hold_max = jump_hold_max
        self.stuck_threshold = stuck_threshold
        self.enemy_lookahead_min = enemy_lookahead_min
        self.enemy_lookahead_max = enemy_lookahead_max
        self.enemy_y_tolerance = enemy_y_tolerance
        self.start_noop_min = start_noop_min
        self.start_noop_max = start_noop_max
        self.obstacles = obstacles or self.DEFAULT_OBSTACLES

        self.jump_timer = 0
        self.stuck_timer = 0
        self.prev_x = 0
        self.noop_remaining = 0
        self.rng = random.Random(42)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset demonstrator state for a new episode."""
        if seed is not None:
            self.rng = random.Random(seed)
        self.jump_timer = 0
        self.stuck_timer = 0
        self.prev_x = 0
        self.noop_remaining = self.rng.randint(self.start_noop_min, self.start_noop_max)

    def get_action(
        self,
        obs: torch.Tensor,
        info: Dict[str, Any],
        ram: np.ndarray,
    ) -> int:
        """Compute the expert action for the current step."""
        # Initial jitter / delay
        if self.noop_remaining > 0:
            self.noop_remaining -= 1
            return 0  # NOOP

        # Mario state from RAM
        mx = int(ram[0x006D]) * 256 + int(ram[0x0086])
        my = int(ram[0x00CE])
        on_ground = int(ram[0x001D]) == 0

        # Stuck check
        if mx == self.prev_x:
            self.stuck_timer += 1
        else:
            self.stuck_timer = 0
        self.prev_x = mx

        # 1. Waypoint obstacle check
        near_obstacle = any(s <= mx <= e for s, e, _ in self.obstacles)

        # 2. Dynamic enemy check in RAM (5 slots)
        enemy_near = False
        for slot in range(5):
            if int(ram[0x000F + slot]) != 0:
                ex = int(ram[0x006E + slot]) * 256 + int(ram[0x0087 + slot])
                ey = int(ram[0x00CF + slot])
                dx = ex - mx
                dy = abs(ey - my)
                if (self.enemy_lookahead_min < dx < self.enemy_lookahead_max) and (dy < self.enemy_y_tolerance):
                    enemy_near = True
                    break

        # Initiate jump ONLY when on ground
        if on_ground and self.jump_timer == 0:
            if near_obstacle or enemy_near or (self.stuck_timer >= self.stuck_threshold):
                self.jump_timer = self.rng.randint(self.jump_hold_min, self.jump_hold_max)
                self.stuck_timer = 0

        # Execute action
        if self.jump_timer > 0:
            self.jump_timer -= 1
            return 4  # RIGHT + JUMP
        else:
            return 1  # RIGHT


class RightOnlyPolicy:
    """Trivial baseline: always press RIGHT (Action 1)."""

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def get_action(self, obs: torch.Tensor, info: Dict[str, Any], ram: np.ndarray) -> int:
        return 1


class RandomPolicy:
    """Random baseline: uniformly sample from {0, 1, 2, 3, 4}."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)

    def get_action(self, obs: torch.Tensor, info: Dict[str, Any], ram: np.ndarray) -> int:
        return self.rng.randint(0, 4)


def collect_trajectory(
    env: Mario2DEnv,
    policy: Any,
    seed: int,
    max_steps: int = 2500,
) -> Dict[str, Any]:
    """Execute a single episode and record all transition data.
    
    Returns a dict with transition tensors (observations stored as uint8).
    """
    if hasattr(policy, "reset"):
        policy.reset(seed)

    obs_tensor, info = env.reset(seed=seed)
    ram = env.ram

    obs_list: List[torch.Tensor] = []
    actions_list: List[int] = []
    rewards_list: List[float] = []
    dones_list: List[bool] = []
    x_pos_list: List[int] = []
    y_pos_list: List[int] = []

    total_reward = 0.0
    final_x = info.get("x_pos", 40)
    flag_get = False
    terminated = False
    step_count = 0

    for step in range(max_steps):
        # Convert float32 [0.0, 1.0] to uint8 [0, 255] for 4x memory savings
        uint8_obs = (obs_tensor * 255.0).clamp(0, 255).to(torch.uint8)
        obs_list.append(uint8_obs)

        action = policy.get_action(obs_tensor, info, ram)
        actions_list.append(action)

        next_obs, reward, term, trunc, info = env.step(action)
        ram = env.ram
        done = term or trunc

        rewards_list.append(reward)
        dones_list.append(done)
        current_x = info.get("x_pos", 40)
        current_y = info.get("y_pos", 79)
        x_pos_list.append(current_x)
        y_pos_list.append(current_y)

        total_reward += reward
        final_x = max(final_x, current_x)
        flag_get = bool(info.get("flag_get", False))
        terminated = done
        step_count = step + 1

        obs_tensor = next_obs
        if done:
            break

    # Action distribution summary
    action_counts = {name: 0 for name in ACTION_NAMES}
    for a in actions_list:
        action_counts[ACTION_NAMES[a]] += 1

    return {
        "seed": seed,
        "steps": step_count,
        "final_x": final_x,
        "total_reward": round(total_reward, 2),
        "flag_get": flag_get,
        "terminated": terminated,
        "action_counts": action_counts,
        "obs": torch.stack(obs_list, dim=0),  # (T, 1, 84, 84) uint8
        "actions": torch.tensor(actions_list, dtype=torch.long),
        "rewards": torch.tensor(rewards_list, dtype=torch.float32),
        "dones": torch.tensor(dones_list, dtype=torch.bool),
        "x_positions": torch.tensor(x_pos_list, dtype=torch.long),
        "y_positions": torch.tensor(y_pos_list, dtype=torch.long),
    }


def collect_demonstration_dataset(
    seeds: List[int],
    output_path: Optional[str | Path] = None,
    max_steps_per_episode: int = 2500,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Dict[str, Any]:
    """Collect demonstration trajectories across seeds and split by episode."""
    env = Mario2DEnv()
    demonstrator = ScriptedDemonstrator()

    episodes_data: List[Dict[str, Any]] = []
    print(f"Collecting demonstrations across {len(seeds)} seeds: {seeds}...")

    t0 = time.time()
    for i, s in enumerate(seeds):
        ep_t0 = time.time()
        traj = collect_trajectory(env, demonstrator, seed=s, max_steps=max_steps_per_episode)
        ep_duration = round(time.time() - ep_t0, 2)
        episodes_data.append(traj)
        print(
            f"  [Seed {s:>4d}] Steps: {traj['steps']:>4d} | Max X: {traj['final_x']:>4d} | "
            f"Reward: {traj['total_reward']:>7.1f} | Flag: {traj['flag_get']} | Time: {ep_duration}s"
        )
    env.close()
    collection_time = round(time.time() - t0, 2)

    # Episode-level partitioning
    n_episodes = len(seeds)
    n_train = max(1, int(round(n_episodes * train_ratio)))
    n_val = max(1, int(round(n_episodes * val_ratio))) if n_episodes >= 3 else 0
    n_test = max(0, n_episodes - n_train - n_val)
    if n_train + n_val + n_test > n_episodes:
        n_train = n_episodes - n_val - n_test

    train_seeds = seeds[:n_train]
    val_seeds = seeds[n_train : n_train + n_val]
    test_seeds = seeds[n_train + n_val :]

    # Concatenate transitions
    all_obs: List[torch.Tensor] = []
    all_actions: List[torch.Tensor] = []
    all_rewards: List[torch.Tensor] = []
    all_dones: List[torch.Tensor] = []
    all_x: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []
    all_ep_ids: List[torch.Tensor] = []
    all_seeds: List[torch.Tensor] = []
    all_steps: List[torch.Tensor] = []

    episode_summaries = []
    overall_action_counts = {name: 0 for name in ACTION_NAMES}

    for ep_idx, ep in enumerate(episodes_data):
        t_len = ep["steps"]
        all_obs.append(ep["obs"])
        all_actions.append(ep["actions"])
        all_rewards.append(ep["rewards"])
        all_dones.append(ep["dones"])
        all_x.append(ep["x_positions"])
        all_y.append(ep["y_positions"])
        all_ep_ids.append(torch.full((t_len,), ep_idx, dtype=torch.long))
        all_seeds.append(torch.full((t_len,), ep["seed"], dtype=torch.long))
        all_steps.append(torch.arange(t_len, dtype=torch.long))

        for k, v in ep["action_counts"].items():
            overall_action_counts[k] += v

        # Determine split assignment
        s_val = ep["seed"]
        if s_val in train_seeds:
            split_tag = "train"
        elif s_val in val_seeds:
            split_tag = "val"
        else:
            split_tag = "test"

        episode_summaries.append({
            "episode_id": ep_idx,
            "seed": ep["seed"],
            "split": split_tag,
            "steps": ep["steps"],
            "final_x": ep["final_x"],
            "total_reward": ep["total_reward"],
            "flag_get": ep["flag_get"],
            "action_counts": ep["action_counts"],
        })

    dataset = {
        "metadata": {
            "num_transitions": sum(ep["steps"] for ep in episodes_data),
            "num_episodes": n_episodes,
            "collection_time_seconds": collection_time,
            "action_space": MARIO_5_ACTIONS,
            "action_names": ACTION_NAMES,
            "splits": {
                "train_seeds": train_seeds,
                "val_seeds": val_seeds,
                "test_seeds": test_seeds,
            },
            "overall_action_counts": overall_action_counts,
            "episodes": episode_summaries,
        },
        "obs": torch.cat(all_obs, dim=0),  # uint8 tensor
        "actions": torch.cat(all_actions, dim=0),
        "rewards": torch.cat(all_rewards, dim=0),
        "dones": torch.cat(all_dones, dim=0),
        "x_positions": torch.cat(all_x, dim=0),
        "y_positions": torch.cat(all_y, dim=0),
        "episode_ids": torch.cat(all_ep_ids, dim=0),
        "seeds": torch.cat(all_seeds, dim=0),
        "step_indices": torch.cat(all_steps, dim=0),
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, out_p)
        file_size_mb = round(out_p.stat().st_size / (1024 * 1024), 2)
        print(f"Saved demonstration dataset to {out_p} ({file_size_mb} MB, {dataset['metadata']['num_transitions']} transitions)")

    return dataset


def load_demonstration_dataset(dataset_path: str | Path) -> Dict[str, Any]:
    """Load demonstration dataset from disk."""
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Demonstration dataset not found at {p}")
    return torch.load(p, weights_only=False)


def benchmark_policies(
    seeds: List[int],
    max_steps: int = 2500,
) -> Dict[str, Any]:
    """Benchmark Demonstrator vs RIGHT-only vs Random policy under identical seeds."""
    env = Mario2DEnv()

    policies = [
        ("Scripted Demonstrator", lambda: ScriptedDemonstrator()),
        ("RIGHT-Only Policy", lambda: RightOnlyPolicy()),
        ("Random Policy", lambda: RandomPolicy()),
    ]

    results: Dict[str, Any] = {}

    for name, ctor in policies:
        print(f"\nEvaluating policy: {name} across {len(seeds)} seeds...")
        ep_stats = []
        action_totals = {aname: 0 for aname in ACTION_NAMES}

        for s in seeds:
            pol = ctor()
            traj = collect_trajectory(env, pol, seed=s, max_steps=max_steps)
            for aname, cnt in traj["action_counts"].items():
                action_totals[aname] += cnt

            ep_stats.append({
                "seed": s,
                "steps": traj["steps"],
                "final_x": traj["final_x"],
                "total_reward": traj["total_reward"],
                "flag_get": traj["flag_get"],
                "terminated": traj["terminated"],
            })

        xs = [e["final_x"] for e in ep_stats]
        rewards = [e["total_reward"] for e in ep_stats]
        steps = [e["steps"] for e in ep_stats]
        flags = [e["flag_get"] for e in ep_stats]

        total_actions = sum(action_totals.values())
        action_props = {k: round(v / total_actions, 4) for k, v in action_totals.items()} if total_actions > 0 else {}

        results[name] = {
            "mean_final_x": round(float(np.mean(xs)), 2),
            "std_final_x": round(float(np.std(xs)), 2),
            "max_final_x": int(np.max(xs)),
            "min_final_x": int(np.min(xs)),
            "mean_reward": round(float(np.mean(rewards)), 2),
            "std_reward": round(float(np.std(rewards)), 2),
            "mean_steps": round(float(np.mean(steps)), 2),
            "level_clear_rate": round(float(np.mean(flags)), 2),
            "action_counts": action_totals,
            "action_proportions": action_props,
            "episodes": ep_stats,
        }

    env.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Mario Demonstration Collector")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Episode seeds")
    parser.add_argument("--output", type=str, default="data/demonstrations/pilot_demonstrations.pt", help="Output path")
    parser.add_argument("--max-steps", type=int, default=2500, help="Max steps per episode")
    parser.add_argument("--benchmark", action="store_true", help="Run policy benchmark")
    args = parser.parse_args()

    if args.benchmark:
        bench_results = benchmark_policies(args.seeds, max_steps=args.max_steps)
        print("\n" + "=" * 70)
        print("BENCHMARK COMPARISON SUMMARY")
        print("=" * 70)
        for pol_name, stats in bench_results.items():
            print(f"Policy: {pol_name}")
            print(f"  Mean Final X:   {stats['mean_final_x']} +/- {stats['std_final_x']} (max: {stats['max_final_x']})")
            print(f"  Mean Reward:    {stats['mean_reward']} +/- {stats['std_reward']}")
            print(f"  Mean Steps:     {stats['mean_steps']}")
            print(f"  Level Clear %:  {stats['level_clear_rate'] * 100:.1f}%")
            print(f"  Actions:        {stats['action_proportions']}")
            print()

    collect_demonstration_dataset(
        seeds=args.seeds,
        output_path=args.output,
        max_steps_per_episode=args.max_steps,
    )


if __name__ == "__main__":
    main()
