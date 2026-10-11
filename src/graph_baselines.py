"""Graph Baseline Generators for Connectome Ablation Study.

Generates:
1. Rewired MaleCNS Graph: Degree-preserving directed shuffled graph (Model B).
   - Preserves exact in-degree and out-degree sequences for all 30,000 nodes.
   - Exact edge count (E = 6,001,050).
   - Zero self-loops, zero multi-edges.
   - Row-normalized so sum of outgoing edge weights for every non-empty node is 1.0.

2. Random Sparse Graph: Directed Erdős–Rényi graph (Model C).
   - Parameter matched: N = 30,000, E = 6,001,050.
   - Zero self-loops, zero multi-edges.
   - Row-normalized so sum of outgoing edge weights is 1.0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def generate_degree_preserving_shuffled_graph(
    input_csr_path: Path | str,
    output_npz_path: Path | str,
    swaps_per_edge: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a degree-preserving directed shuffled graph via double-edge swaps.

    Conditions guaranteed:
    1. Exact node count (N = 30,000).
    2. Exact edge count (E = 6,001,050).
    3. Exact in-degree sequence identical to input connectome.
    4. Exact out-degree sequence identical to input connectome.
    5. Zero self-loops (u != v).
    6. Zero multi-edges (no duplicates).
    7. Row-normalized weights (outgoing rows sum to 1.0).
    """
    t0 = time.time()
    input_path = Path(input_csr_path)
    output_path = Path(output_npz_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input graph not found: {input_path}")

    adj = sp.load_npz(str(input_path))
    num_nodes = adj.shape[0]
    total_edges = adj.nnz

    # Compute original degree sequences for verification
    orig_out_degrees = np.diff(adj.indptr)
    orig_in_degrees = np.bincount(adj.indices, minlength=num_nodes)

    coo = adj.tocoo()
    row = coo.row.copy()
    col = coo.col.copy()
    orig_col = coo.col.copy()

    # Build initial edge set for fast O(1) collision checking
    edge_hashes = row.astype(np.int64) * num_nodes + col.astype(np.int64)
    edge_set = set(edge_hashes)

    target_swaps = int(total_edges * swaps_per_edge)
    rng = np.random.default_rng(seed)

    # Eliminate any initial autapses (self-loops) from the biological graph
    init_autapses = np.flatnonzero(row == col)
    for i1 in init_autapses:
        cur_u1, cur_v1 = row[i1], col[i1]
        for _ in range(1000):
            i2 = rng.integers(0, total_edges)
            cur_u2, cur_v2 = row[i2], col[i2]
            if cur_u1 == cur_u2 or cur_v1 == cur_v2 or cur_u1 == cur_v2 or cur_u2 == cur_v1:
                continue
            new_h1 = cur_u1 * num_nodes + cur_v2
            new_h2 = cur_u2 * num_nodes + cur_v1
            if new_h1 not in edge_set and new_h2 not in edge_set:
                edge_set.remove(cur_u1 * num_nodes + cur_v1)
                edge_set.remove(cur_u2 * num_nodes + cur_v2)
                edge_set.add(new_h1)
                edge_set.add(new_h2)
                col[i1] = cur_v2
                col[i2] = cur_v1
                break

    num_successful_swaps = 0
    batch_size = min(500000, total_edges)

    while num_successful_swaps < target_swaps:
        idx1 = rng.integers(0, total_edges, size=batch_size)
        idx2 = rng.integers(0, total_edges, size=batch_size)

        u1, v1 = row[idx1], col[idx1]
        u2, v2 = row[idx2], col[idx2]

        # Fast filter on indices: endpoints must be different
        candidate_mask = (u1 != u2) & (v1 != v2) & (u1 != v2) & (u2 != v1)
        cand1 = idx1[candidate_mask]
        cand2 = idx2[candidate_mask]

        for i in range(len(cand1)):
            i1, i2 = cand1[i], cand2[i]
            cur_u1, cur_v1 = row[i1], col[i1]
            cur_u2, cur_v2 = row[i2], col[i2]

            # Re-check in case either edge was mutated earlier in this batch
            if cur_u1 == cur_u2 or cur_v1 == cur_v2 or cur_u1 == cur_v2 or cur_u2 == cur_v1:
                continue

            new_h1 = cur_u1 * num_nodes + cur_v2
            new_h2 = cur_u2 * num_nodes + cur_v1

            # Verify neither proposed edge exists in the graph
            if new_h1 not in edge_set and new_h2 not in edge_set:
                old_h1 = cur_u1 * num_nodes + cur_v1
                old_h2 = cur_u2 * num_nodes + cur_v2

                edge_set.remove(old_h1)
                edge_set.remove(old_h2)
                edge_set.add(new_h1)
                edge_set.add(new_h2)

                col[i1] = cur_v2
                col[i2] = cur_v1
                num_successful_swaps += 1

                if num_successful_swaps >= target_swaps:
                    break

    # Build new CSR matrix preserving row weights / row normalization
    # Construct unweighted adjacency first, then row-normalize
    data = np.ones(total_edges, dtype=np.float32)
    shuffled_csr = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)

    # Row normalize
    row_sums = np.array(shuffled_csr.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # Avoid divide by zero for sink nodes
    inv_row_sums = sp.diags(1.0 / row_sums)
    normalized_csr = inv_row_sums.dot(shuffled_csr).tocsr().astype(np.float32)

    # Invariant Verification
    new_out_degrees = np.diff(normalized_csr.indptr)
    new_in_degrees = np.bincount(normalized_csr.indices, minlength=num_nodes)

    out_match = bool(np.array_equal(orig_out_degrees, new_out_degrees))
    in_match = bool(np.array_equal(orig_in_degrees, new_in_degrees))
    edge_count_match = bool(normalized_csr.nnz == total_edges)
    num_self_loops = int(normalized_csr.diagonal().nonzero()[0].size)
    modified_edges = int(np.sum(col != orig_col))
    modified_ratio = float(modified_edges / total_edges)

    if not out_match or not in_match or not edge_count_match or num_self_loops > 0:
        raise ValueError(
            f"Degree-preserving invariants violated: out_match={out_match}, "
            f"in_match={in_match}, edges={normalized_csr.nnz} (expected {total_edges}), "
            f"self_loops={num_self_loops}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(str(output_path), normalized_csr)
    duration = round(time.time() - t0, 2)

    stats = {
        "generator": "degree_preserving_shuffled_graph",
        "input_csr_path": str(input_path),
        "output_npz_path": str(output_path),
        "num_nodes": int(num_nodes),
        "num_edges": int(normalized_csr.nnz),
        "target_swaps": int(target_swaps),
        "executed_swaps": int(num_successful_swaps),
        "modified_edges": modified_edges,
        "modified_edge_ratio": round(modified_ratio, 6),
        "out_degree_preserved": out_match,
        "in_degree_preserved": in_match,
        "self_loops": num_self_loops,
        "duration_seconds": duration,
        "seed": seed,
    }
    return stats


def generate_random_sparse_graph(
    output_npz_path: Path | str,
    num_nodes: int = 30000,
    num_edges: int = 6001050,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a uniform directed Erdős–Rényi random sparse graph.

    Conditions guaranteed:
    1. Exact node count (N = 30,000).
    2. Exact edge count (E = 6,001,050).
    3. Zero self-loops (u != v).
    4. Zero multi-edges (no duplicates).
    5. Row-normalized weights (outgoing rows sum to 1.0).
    """
    t0 = time.time()
    output_path = Path(output_npz_path)
    rng = np.random.default_rng(seed)

    edge_set = set()
    while len(edge_set) < num_edges:
        needed = num_edges - len(edge_set)
        sample_size = int(needed * 1.05) + 1000
        u = rng.integers(0, num_nodes, size=sample_size, dtype=np.int64)
        v = rng.integers(0, num_nodes, size=sample_size, dtype=np.int64)
        valid = u != v
        u, v = u[valid], v[valid]
        hashes = u * num_nodes + v
        edge_set.update(hashes)

    # Retain exactly num_edges
    all_hashes = np.array(list(edge_set)[:num_edges], dtype=np.int64)
    row = all_hashes // num_nodes
    col = all_hashes % num_nodes

    data = np.ones(num_edges, dtype=np.float32)
    er_csr = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)

    # Row normalize
    row_sums = np.array(er_csr.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    inv_row_sums = sp.diags(1.0 / row_sums)
    normalized_csr = inv_row_sums.dot(er_csr).tocsr().astype(np.float32)

    # Invariant Verification
    edge_count_match = bool(normalized_csr.nnz == num_edges)
    num_self_loops = int(normalized_csr.diagonal().nonzero()[0].size)
    out_degrees = np.diff(normalized_csr.indptr)
    in_degrees = np.bincount(normalized_csr.indices, minlength=num_nodes)

    if not edge_count_match or num_self_loops > 0:
        raise ValueError(
            f"Erdős-Rényi invariants violated: edges={normalized_csr.nnz} (expected {num_edges}), "
            f"self_loops={num_self_loops}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(str(output_path), normalized_csr)
    duration = round(time.time() - t0, 2)

    stats = {
        "generator": "random_sparse_erdos_renyi",
        "output_npz_path": str(output_path),
        "num_nodes": int(num_nodes),
        "num_edges": int(normalized_csr.nnz),
        "mean_out_degree": round(float(np.mean(out_degrees)), 3),
        "min_out_degree": int(np.min(out_degrees)),
        "max_out_degree": int(np.max(out_degrees)),
        "mean_in_degree": round(float(np.mean(in_degrees)), 3),
        "min_in_degree": int(np.min(in_degrees)),
        "max_in_degree": int(np.max(in_degrees)),
        "self_loops": num_self_loops,
        "duration_seconds": duration,
        "seed": seed,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate Baseline Graphs for MaleCNS Ablation Study")
    parser.add_argument(
        "--input-connectome",
        type=str,
        default="data/processed/subgraph_graph.npz",
        help="Path to biological connectome CSR graph",
    )
    parser.add_argument(
        "--output-shuffled",
        type=str,
        default="data/processed/shuffled_connectome_graph.npz",
        help="Output path for degree-preserving rewired connectome",
    )
    parser.add_argument(
        "--output-random",
        type=str,
        default="data/processed/random_sparse_graph.npz",
        help="Output path for random Erdős–Rényi graph",
    )
    parser.add_argument(
        "--swaps-per-edge",
        type=float,
        default=1.0,
        help="Number of successful double-edge swaps per edge",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log-summary",
        type=str,
        default="logs/graph_baselines_summary.json",
        help="Path to save execution summary JSON",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("GENERATING GRAPH BASELINES FOR ABLATION STUDY")
    print("=" * 70)

    print(f"\n1. Generating Degree-Preserving Shuffled Graph (Model B)...")
    shuffled_stats = generate_degree_preserving_shuffled_graph(
        input_csr_path=args.input_connectome,
        output_npz_path=args.output_shuffled,
        swaps_per_edge=args.swaps_per_edge,
        seed=args.seed,
    )
    print(f"   [DONE] Generated in {shuffled_stats['duration_seconds']}s")
    print(f"   Nodes: {shuffled_stats['num_nodes']:,} | Edges: {shuffled_stats['num_edges']:,}")
    print(f"   Swaps: {shuffled_stats['executed_swaps']:,} | Modified Edges: {shuffled_stats['modified_edges']:,} ({shuffled_stats['modified_edge_ratio']*100:.2f}%)")
    print(f"   Degree sequences preserved: In={shuffled_stats['in_degree_preserved']}, Out={shuffled_stats['out_degree_preserved']}")

    print(f"\n2. Generating Random Sparse Erdős–Rényi Graph (Model C)...")
    random_stats = generate_random_sparse_graph(
        output_npz_path=args.output_random,
        num_nodes=shuffled_stats["num_nodes"],
        num_edges=shuffled_stats["num_edges"],
        seed=args.seed,
    )
    print(f"   [DONE] Generated in {random_stats['duration_seconds']}s")
    print(f"   Nodes: {random_stats['num_nodes']:,} | Edges: {random_stats['num_edges']:,}")
    print(f"   Out-degree range: [{random_stats['min_out_degree']}, {random_stats['max_out_degree']}] (Mean: {random_stats['mean_out_degree']})")
    print(f"   In-degree range: [{random_stats['min_in_degree']}, {random_stats['max_in_degree']}] (Mean: {random_stats['mean_in_degree']})")

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rewired_model_b": shuffled_stats,
        "random_model_c": random_stats,
    }
    log_p = Path(args.log_summary)
    log_p.parent.mkdir(parents=True, exist_ok=True)
    with open(log_p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {log_p}")
    print("=" * 70)


if __name__ == "__main__":
    main()
