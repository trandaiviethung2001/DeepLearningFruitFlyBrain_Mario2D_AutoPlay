"""Phase 4: Scientific Connectome Ablation Study & Hypothesis Testing.

Compares 4 experimental architectures on identical environments and evaluation seeds:
- Model A: Biological MaleCNS Connectome (30,000 nodes, 6,001,050 edges)
- Model B: Degree-Preserving Shuffled Null Model (Matched N, E, in-degree & out-degree sequences)
- Model C: Erdős–Rényi Random Sparse Null Model (Matched N, E, uniform random wiring)
- Model D: Unconstrained Dense GRU-1024 Baseline (Checkpointed at 100% flagpole completion)

Evaluates:
1. Behavioral performance: Flagpole Win Rate, Max X progress, Task Return.
2. Obstacle traversal: Goomba 1, Pipes 1–4, Pit 1.
3. Neurocomputational metrics: L1/L2 Sparsity ratio, Firing rate, Policy Entropy.
4. Statistical hypothesis testing: Welch's t-test and Cohen's d effect sizes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline_model import BaselineMarioPolicy
from src.connectome_model import ConnectomeMarioPolicy
from src.environment import Mario2DEnv, WORLD_1_1_LANDMARKS
from src.visualization import compute_sparsity_ratio

DEFAULT_EVAL_SEEDS = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]


def parse_args():
    parser = argparse.ArgumentParser(description="Run scientific connectome ablation study")
    default_conn = "checkpoints/connectome_ppo_v3.pt"
    if not Path(default_conn).exists():
        if Path("checkpoints/connectome_ppo_v2_latest.pt").exists():
            default_conn = "checkpoints/connectome_ppo_v2_latest.pt"
        elif Path("checkpoints/connectome_ppo_v2.pt").exists():
            default_conn = "checkpoints/connectome_ppo_v2.pt"
    parser.add_argument("--connectome-ckpt", type=str, default=default_conn, help="Connectome checkpoint")
    parser.add_argument("--baseline-ckpt", type=str, default="checkpoints/baseline_ppo_v3.pt", help="Baseline checkpoint")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS, help="Evaluation seeds")
    parser.add_argument("--max-steps", type=int, default=1250, help="Max steps per episode (x4 frames = 5000)")
    parser.add_argument("--mode", type=str, choices=["greedy", "sample"], default="greedy", help="Action selection mode")
    parser.add_argument("--output-json", type=str, default="logs/scientific_ablation_results.json", help="Path to output JSON")
    return parser.parse_args()


def load_model_family(
    family: str,
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, int]:
    """Load model for specific ablation family."""
    processed_dir = REPO_ROOT / "data" / "processed"

    if family == "Model_D_Baseline":
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt and "encoder.conv1.weight" in ckpt["model_state_dict"]:
            in_channels = ckpt["model_state_dict"]["encoder.conv1.weight"].shape[1]
        else:
            in_channels = ckpt.get("frame_stack", 4)
        model = BaselineMarioPolicy(in_channels=in_channels, cnn_dim=512, hidden_dim=1024)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # Load graph topology based on model family
        if family == "Model_A_Biological":
            graph_path = processed_dir / "subgraph_graph.npz"
        elif family == "Model_B_Shuffled_Null":
            graph_path = processed_dir / "shuffled_connectome_graph.npz"
        elif family == "Model_C_Random_Sparse":
            graph_path = processed_dir / "random_sparse_graph.npz"
        else:
            raise ValueError(f"Unknown family: {family}")

        # Check if checkpoint exists
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt and "encoder.conv1.weight" in ckpt["model_state_dict"]:
                in_channels = ckpt["model_state_dict"]["encoder.conv1.weight"].shape[1]
            else:
                in_channels = ckpt.get("frame_stack", 4)
            model = ConnectomeMarioPolicy(
                subgraph_npz_path=graph_path,
                node_mapping_csv_path=processed_dir / "subgraph_node_mapping.csv",
                node_metadata_csv_path=processed_dir / "node_metadata.csv",
                in_channels=in_channels,
                cnn_dim=512,
            )
            # Load compatible weights (encoder, adapter, actor, critic, rnn.bias)
            state_dict = ckpt["model_state_dict"]
            model_dict = model.state_dict()
            # Filter out fixed topology W_rec
            pretrained_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape and "W_rec" not in k
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print(f"  [WARNING] Checkpoint not found at '{ckpt_path}'. Evaluating UNTRAINED {family} with random initialization!")
            in_channels = 4
            model = ConnectomeMarioPolicy(
                subgraph_npz_path=graph_path,
                node_mapping_csv_path=processed_dir / "subgraph_node_mapping.csv",
                node_metadata_csv_path=processed_dir / "node_metadata.csv",
                in_channels=in_channels,
                cnn_dim=512,
            )

    model.to(device)
    model.eval()
    return model, in_channels


def evaluate_single_seed(
    model: torch.nn.Module,
    is_connectome: bool,
    seed: int,
    in_channels: int,
    max_steps: int,
    mode: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Run one evaluation episode on a fixed seed and collect metrics."""
    env = Mario2DEnv(
        frame_stack=in_channels,
        frame_skip=4,
        auto_unlatch=True,
        max_episode_frames=max_steps * 4,
        stall_frames=600,
    )

    obs, info = env.reset(seed=seed)
    h = None

    total_reward = 0.0
    max_x = int(info.get("x_pos", 0))
    action_counts = np.zeros(5, dtype=np.int64)
    entropies: list[float] = []
    sparsities: list[float] = []
    firing_rates: list[float] = []
    steps = 0

    while steps < max_steps:
        if isinstance(obs, torch.Tensor):
            obs_t = obs.unsqueeze(0).float().to(device)
        else:
            obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)

        with torch.no_grad():
            if is_connectome:
                logits, value, h_next, tele = model(obs_t, h, num_ticks=2, return_telemetry=True)
                core_act = tele["h_core"].cpu().numpy().squeeze(0)
                sparsities.append(compute_sparsity_ratio(core_act))
                firing_rates.append(float(np.mean(np.abs(core_act) > 0.05)))
            else:
                logits, value, h_next, tele = model(obs_t, h, return_telemetry=True)
                core_act = tele["h_core"].cpu().numpy().squeeze(0)
                sparsities.append(compute_sparsity_ratio(core_act))
                firing_rates.append(float(np.mean(np.abs(core_act) > 0.05)))

            probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
            ent = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
            entropies.append(ent)

            if mode == "greedy":
                action = int(np.argmax(probs))
            else:
                action = int(np.random.choice(len(probs), p=probs))

        action_counts[action] += 1
        h = h_next

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        total_reward += reward
        steps += 1
        cur_x = int(next_info.get("x_pos", 0))
        if cur_x > max_x:
            max_x = cur_x

        obs = next_obs
        if terminated or truncated or next_info.get("flag_get", False):
            info = next_info
            break

    env.close()

    total_actions = max(1, int(np.sum(action_counts)))
    action_dist = (action_counts / total_actions).tolist()

    return {
        "seed": seed,
        "max_x": max_x,
        "total_reward": round(float(total_reward), 2),
        "steps": steps,
        "flag_get": bool(info.get("flag_get", False)),
        "cleared_goomba1": bool(max_x >= WORLD_1_1_LANDMARKS["goomba1"]),
        "cleared_pipe1": bool(max_x >= WORLD_1_1_LANDMARKS["pipe1"]),
        "cleared_pipe2": bool(max_x >= WORLD_1_1_LANDMARKS["pipe2"]),
        "cleared_pipe3": bool(max_x >= WORLD_1_1_LANDMARKS["pipe3"]),
        "cleared_pipe4": bool(max_x >= WORLD_1_1_LANDMARKS["pipe4"]),
        "cleared_pit1": bool(max_x >= WORLD_1_1_LANDMARKS["pit1"]),
        "mean_entropy": round(float(np.mean(entropies)) if entropies else 0.0, 4),
        "mean_sparsity_l1_l2": round(float(np.mean(sparsities)) if sparsities else 0.0, 2),
        "mean_firing_rate": round(float(np.mean(firing_rates)) if firing_rates else 0.0, 4),
        "action_distribution": [round(p, 4) for p in action_dist],
    }


def compute_welch_t_test(samples_a: list[float], samples_b: list[float]) -> Dict[str, float]:
    """Compute Welch's two-sample t-test and Cohen's d effect size."""
    n1, n2 = len(samples_a), len(samples_b)
    m1, m2 = float(np.mean(samples_a)), float(np.mean(samples_b))
    s1, s2 = float(np.std(samples_a, ddof=1)), float(np.std(samples_b, ddof=1))

    # Welch's t-test
    res = stats.ttest_ind(samples_a, samples_b, equal_var=False)
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)

    # Pooled standard deviation for Cohen's d
    s_pooled = math.sqrt(((n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)) / max(1, (n1 + n2 - 2)))
    cohens_d = (m1 - m2) / max(1e-8, s_pooled)

    return {
        "mean_a": round(m1, 2),
        "mean_b": round(m2, 2),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_val, 6),
        "significant_p05": bool(p_val < 0.05),
        "cohens_d": round(cohens_d, 4),
    }


def run_ablation_study(args) -> Dict[str, Any]:
    device = torch.device("cpu")
    families = [
        ("Model_A_Biological", Path(args.connectome_ckpt), True),
        ("Model_B_Shuffled_Null", Path(args.connectome_ckpt), True),
        ("Model_C_Random_Sparse", Path(args.connectome_ckpt), True),
        ("Model_D_Baseline", Path(args.baseline_ckpt), False),
    ]

    results: Dict[str, Any] = {}

    print("=" * 70)
    print("PHASE 4: SCIENTIFIC CONNECTOME ABLATION STUDY")
    print(f"Seeds: {args.seeds} | Mode: {args.mode} | Max Steps: {args.max_steps}")
    print("=" * 70)

    for family_name, ckpt_path, is_connectome in families:
        print(f"\nEvaluating {family_name}...")
        t0 = time.time()
        model, in_channels = load_model_family(family_name, ckpt_path, device)

        seed_results = []
        for s in args.seeds:
            res = evaluate_single_seed(
                model=model,
                is_connectome=is_connectome,
                seed=s,
                in_channels=in_channels,
                max_steps=args.max_steps,
                mode=args.mode,
                device=device,
            )
            seed_results.append(res)

        elapsed = time.time() - t0
        xs = [r["max_x"] for r in seed_results]
        returns = [r["total_reward"] for r in seed_results]
        wins = [1 if r["flag_get"] else 0 for r in seed_results]
        entropies = [r["mean_entropy"] for r in seed_results]
        sparsities = [r["mean_sparsity_l1_l2"] for r in seed_results]
        firing_rates = [r["mean_firing_rate"] for r in seed_results]

        agg = {
            "mean_max_x": round(float(np.mean(xs)), 2),
            "std_max_x": round(float(np.std(xs)), 2),
            "mean_return": round(float(np.mean(returns)), 2),
            "std_return": round(float(np.std(returns)), 2),
            "flagpole_win_rate": round(float(np.mean(wins)), 2),
            "goomba1_clear_rate": round(float(np.mean([1 if r["cleared_goomba1"] else 0 for r in seed_results])), 2),
            "pipe1_clear_rate": round(float(np.mean([1 if r["cleared_pipe1"] else 0 for r in seed_results])), 2),
            "pipe2_clear_rate": round(float(np.mean([1 if r["cleared_pipe2"] else 0 for r in seed_results])), 2),
            "pipe3_clear_rate": round(float(np.mean([1 if r["cleared_pipe3"] else 0 for r in seed_results])), 2),
            "pipe4_clear_rate": round(float(np.mean([1 if r["cleared_pipe4"] else 0 for r in seed_results])), 2),
            "pit1_clear_rate": round(float(np.mean([1 if r["cleared_pit1"] else 0 for r in seed_results])), 2),
            "mean_entropy": round(float(np.mean(entropies)), 4),
            "mean_sparsity_l1_l2": round(float(np.mean(sparsities)), 2),
            "mean_firing_rate": round(float(np.mean(firing_rates)), 4),
            "elapsed_seconds": round(elapsed, 2),
            "seeds": seed_results,
        }
        results[family_name] = agg
        print(f"  -> Mean X: {agg['mean_max_x']} ± {agg['std_max_x']} | Win Rate: {agg['flagpole_win_rate']*100:.0f}% | L1/L2 Sparsity: {agg['mean_sparsity_l1_l2']}")

    # Perform statistical tests
    xs_a = [r["max_x"] for r in results["Model_A_Biological"]["seeds"]]
    xs_b = [r["max_x"] for r in results["Model_B_Shuffled_Null"]["seeds"]]
    xs_c = [r["max_x"] for r in results["Model_C_Random_Sparse"]["seeds"]]
    xs_d = [r["max_x"] for r in results["Model_D_Baseline"]["seeds"]]

    hyp_tests = {
        "A_vs_B_Shuffled_Null_Progress": compute_welch_t_test(xs_a, xs_b),
        "A_vs_C_Random_Sparse_Progress": compute_welch_t_test(xs_a, xs_c),
        "A_vs_D_Dense_Baseline_Progress": compute_welch_t_test(xs_a, xs_d),
    }
    results["hypothesis_testing"] = hyp_tests

    # Save to JSON
    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved scientific ablation results to: {out_file}")

    return results


if __name__ == "__main__":
    cli_args = parse_args()
    run_ablation_study(cli_args)
