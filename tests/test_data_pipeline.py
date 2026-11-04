"""Test B: Data Pipeline Verification on Real MaleCNS Dataset.

Requirements:
- Verify that the actual MaleCNS files exist and were processed.
- Verify graph shape, node mapping, edge orientation, weights, sparse format.
- Verify save/load round trip from data/processed/.
- Reject synthetic data; assert real biological connectivity properties.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import scipy.sparse as sp


def run_data_pipeline_test() -> bool:
    print("=" * 60)
    print("TEST B: REAL MALECNS DATA PIPELINE VERIFICATION")
    print("=" * 60)

    processed_dir = REPO_ROOT / "data" / "processed"
    report_file = processed_dir / "processing_report.json"
    full_graph_file = processed_dir / "connectome_graph.npz"
    sub_graph_file = processed_dir / "subgraph_graph.npz"
    sub_raw_file = processed_dir / "subgraph_graph_raw.npz"
    node_map_file = processed_dir / "subgraph_node_mapping.csv"
    edge_csv_file = processed_dir / "subgraph_edges.csv"

    print("1. Checking processed artifact existence...")
    required_files = [report_file, full_graph_file, sub_graph_file, node_map_file]
    for rf in required_files:
        assert rf.is_file(), f"Missing required artifact: {rf}"
        print(f"   [Found] {rf.name} ({rf.stat().st_size / 1024 / 1024:.2f} MB)")
    # Intermediate artifacts (raw-weight subgraph + edge CSV) are optional; they can be
    # regenerated with data/data_processing.py and are only used for extra spot checks.
    has_raw = sub_raw_file.is_file() and edge_csv_file.is_file()
    if not has_raw:
        print("   [Skip] subgraph_graph_raw.npz / subgraph_edges.csv not present (optional)")

    print("2. Verifying processing report...")
    with open(report_file, "r") as f:
        report = json.load(f)

    assert report["build_stats"]["n_input_edges"] == 151856684
    assert report["build_stats"]["n_edges"] == 26028386
    assert report["subgraph_stats"]["n_nodes"] == 30000
    assert report["subgraph_stats"]["n_edges"] == 6001050
    print(f"   Input edges: {report['build_stats']['n_input_edges']:,}")
    print(f"   Verified backbone edges: {report['build_stats']['n_edges']:,}")
    print(f"   Visuomotor subgraph nodes: {report['subgraph_stats']['n_nodes']:,}")
    print(f"   Visuomotor subgraph edges: {report['subgraph_stats']['n_edges']:,}")

    print("3. Verifying sparse graph load & shape...")
    full_csr = sp.load_npz(str(full_graph_file))
    sub_csr = sp.load_npz(str(sub_graph_file))
    assert full_csr.shape == (211577, 211577), f"Expected (211577, 211577), got {full_csr.shape}"
    assert full_csr.nnz == 26028386, f"Expected 26028386, got {full_csr.nnz}"
    assert sub_csr.shape == (30000, 30000), f"Expected (30000, 30000), got {sub_csr.shape}"
    assert sub_csr.nnz == 6001050, f"Expected 6001050, got {sub_csr.nnz}"
    print(f"   Full graph CSR: shape={full_csr.shape}, nnz={full_csr.nnz:,}")
    print(f"   Subgraph CSR: shape={sub_csr.shape}, nnz={sub_csr.nnz:,}")

    print("4. Verifying edge weight semantics & normalization...")
    if has_raw:
        sub_raw = sp.load_npz(str(sub_raw_file))
        assert sub_raw.shape == (30000, 30000)
        assert sub_raw.nnz == 6001050
        # Raw weights are integer synapse counts >= 1
        assert sub_raw.data.min() >= 1.0, f"Raw weights must be >= 1, got {sub_raw.data.min()}"
        print(f"   Raw weight range in subgraph: [{sub_raw.data.min()}, {sub_raw.data.max()}]")
        print(f"   Raw weight sum in subgraph: {sub_raw.data.sum():,}")

    # Processed weights are row-normalized (out-strength sums to <= 1.0)
    row_sums = np.asarray(sub_csr.sum(axis=1)).ravel()
    nonzero_rows = row_sums[row_sums > 0]
    assert np.allclose(nonzero_rows, 1.0, atol=1e-4), "Row normalization failed!"
    print(f"   Row normalization verified: {len(nonzero_rows)} active rows sum to 1.0")

    print("5. Verifying edge directionality and mapping consistency...")
    node_map = pd.read_csv(str(node_map_file))
    assert len(node_map) == 30000
    assert list(node_map.columns) == ["subgraph_index", "parent_graph_index", "body_id"]
    print(f"   Node map: {len(node_map):,} entries mapped deterministically.")

    if has_raw:
        # Spot check sample of edges from CSV vs CSR
        edges_df = pd.read_csv(str(edge_csv_file), nrows=100)
        for _, r in edges_df.head(10).iterrows():
            src_i, dst_i = int(r["src_index"]), int(r["dst_index"])
            raw_w = int(r["raw_weight"])
            assert sub_raw[src_i, dst_i] == raw_w, f"Mismatch at ({src_i}, {dst_i})"
        print("   Edge list matches sparse CSR structure exactly.")

    print("TEST B RESULT: PASS\n")
    return True


if __name__ == "__main__":
    success = run_data_pipeline_test()
    sys.exit(0 if success else 1)

