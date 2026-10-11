"""Test F: Local Inference Benchmark on Apple Silicon (MPS / CPU Fallback).

Requirements:
- Test batch-size-one inference on Apple Silicon using MPS when supported, with CPU fallback.
- Measure memory and latency across 50 consecutive inference steps.
- Provide detailed statistics (mean, median, p95, min, max latency, FPS, peak RAM).
"""

from __future__ import annotations

import sys
import time
import resource
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from src.connectome_model import ConnectomeMarioPolicy


def run_local_inference_test(num_steps: int = 50) -> bool:
    print("=" * 60)
    print("TEST F: LOCAL INFERENCE BENCHMARK (APPLE SILICON / MPS)")
    print("=" * 60)

    # 1. Device detection & validation
    mps_available = torch.backends.mps.is_available()
    device = torch.device("mps" if mps_available else "cpu")
    print(f"1. Target inference device: {device} (MPS available: {mps_available})")

    processed_dir = REPO_ROOT / "data" / "processed"
    sub_graph_path = processed_dir / "subgraph_graph.npz"
    node_map_path = processed_dir / "subgraph_node_mapping.csv"
    node_meta_path = processed_dir / "node_metadata.csv"

    print("2. Loading ConnectomeMarioPolicy onto device...")
    t_load0 = time.time()
    try:
        model = ConnectomeMarioPolicy(
            subgraph_npz_path=sub_graph_path,
            node_mapping_csv_path=node_map_path,
            node_metadata_csv_path=node_meta_path,
        ).to(device)
    except Exception as e:
        print(f"   Device {device} initialization encountered error: {e}. Falling back to CPU...")
        device = torch.device("cpu")
        model = ConnectomeMarioPolicy(
            subgraph_npz_path=sub_graph_path,
            node_mapping_csv_path=node_map_path,
            node_metadata_csv_path=node_meta_path,
        ).to(device)

    model.eval()
    t_load = time.time() - t_load0
    print(f"   Model initialized in {t_load:.2f}s on {device}.")

    # 3. Warm-up
    print("3. Warming up inference engine (5 steps)...")
    h = model.init_hidden(batch_size=1, device=device)
    dummy_obs = torch.randn(1, 1, 84, 84, device=device)
    for _ in range(5):
        with torch.no_grad():
            _, _, h = model(dummy_obs, h, num_ticks=1)
    if device.type == "mps":
        torch.mps.synchronize()

    # 4. Benchmark loop
    print(f"4. Benchmarking {num_steps} sequential batch-1 steps...")
    latencies = []
    h = model.init_hidden(batch_size=1, device=device)

    for i in range(num_steps):
        obs = torch.randn(1, 1, 84, 84, device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, value, h = model(obs, h, num_ticks=1)
            action = torch.argmax(logits, dim=-1).item()
        if device.type == "mps":
            torch.mps.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(dt)

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    l_arr = np.array(latencies)

    print("\n--- Latency and Memory Profile ---")
    print(f"Device:            {device}")
    print(f"Total steps:       {num_steps}")
    print(f"Mean latency:      {np.mean(l_arr):.2f} ms")
    print(f"Median latency:    {np.median(l_arr):.2f} ms")
    print(f"P95 latency:       {np.percentile(l_arr, 95):.2f} ms")
    print(f"Min latency:       {np.min(l_arr):.2f} ms")
    print(f"Max latency:       {np.max(l_arr):.2f} ms")
    print(f"Effective FPS:     {1000.0 / np.mean(l_arr):.2f} FPS")
    print(f"Peak Host RAM:     {peak_rss_mb:.2f} MB")

    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
    assert np.mean(l_arr) < 250.0, f"Latency too high: {np.mean(l_arr):.2f} ms"
    print("TEST F RESULT: PASS\n")
    return True


if __name__ == "__main__":
    success = run_local_inference_test()
    sys.exit(0 if success else 1)

