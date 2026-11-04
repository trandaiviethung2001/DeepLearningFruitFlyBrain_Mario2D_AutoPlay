"""Test D: End-to-End Environment Integration Verification.

Requirements:
- Run at least 100 environment steps through the complete observation-to-action pipeline:
  Mario frame -> CNN encoder -> input adapter -> MaleCNS connectome recurrent layer -> action decoder -> Mario environment.
- Record exceptions, shapes, inference latency, and peak memory.
"""

from __future__ import annotations

import sys
import time
import resource
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.environment import Mario2DEnv
from src.connectome_model import ConnectomeMarioPolicy


def run_environment_integration_test(num_steps: int = 100) -> bool:
    print("=" * 60)
    print(f"TEST D: END-TO-END ENVIRONMENT INTEGRATION ({num_steps} STEPS)")
    print("=" * 60)

    processed_dir = REPO_ROOT / "data" / "processed"
    sub_graph_path = processed_dir / "subgraph_graph.npz"
    node_map_path = processed_dir / "subgraph_node_mapping.csv"
    node_meta_path = processed_dir / "node_metadata.csv"

    print("1. Initializing environment and model...")
    env = Mario2DEnv()
    device = torch.device("cpu")

    model = ConnectomeMarioPolicy(
        subgraph_npz_path=sub_graph_path,
        node_mapping_csv_path=node_map_path,
        node_metadata_csv_path=node_meta_path,
    ).to(device)
    model.eval()

    print(f"2. Resetting environment and initializing recurrent state (N={model.num_nodes})...")
    obs, info = env.reset()
    h = model.init_hidden(batch_size=1, device=device)

    print(f"3. Executing {num_steps} end-to-end observation-action cycles...")
    step_latencies = []
    total_reward = 0.0
    action_counts = {a: 0 for a in range(5)}
    t_start = time.time()

    for step_idx in range(num_steps):
        t0 = time.time()
        # Add batch dimension: (1, 1, 84, 84)
        obs_tensor = obs.unsqueeze(0).to(device)

        with torch.no_grad():
            logits, value, h = model(obs_tensor, h, num_ticks=1)
            # Sample or take greedy action
            action = torch.argmax(logits, dim=-1).item()

        action_counts[action] += 1

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        dt = (time.time() - t0) * 1000.0  # ms
        step_latencies.append(dt)

        if terminated or truncated:
            print(f"   Episode boundary at step {step_idx}; resetting state.")
            obs, info = env.reset()
            h = model.init_hidden(batch_size=1, device=device)

    total_time = time.time() - t_start
    env.close()

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    avg_latency = sum(step_latencies) / len(step_latencies)
    fps = num_steps / total_time

    print("\n--- Integration Performance Metrics ---")
    print(f"Total steps executed: {num_steps}")
    print(f"Total elapsed time:   {total_time:.2f}s")
    print(f"Average step latency: {avg_latency:.2f} ms")
    print(f"Throughput:           {fps:.2f} FPS")
    print(f"Peak memory (RSS):    {peak_rss_mb:.2f} MB")
    print(f"Accumulated reward:   {total_reward:.2f}")
    print(f"Final Mario x_pos:    {info.get('x_pos')}")
    print(f"Action distribution:  {action_counts}")

    assert len(step_latencies) == num_steps
    assert peak_rss_mb < 4000.0, f"Memory footprint exceeded 4GB budget: {peak_rss_mb:.2f} MB"
    print("TEST D RESULT: PASS\n")
    return True


if __name__ == "__main__":
    success = run_environment_integration_test(100)
    sys.exit(0 if success else 1)

