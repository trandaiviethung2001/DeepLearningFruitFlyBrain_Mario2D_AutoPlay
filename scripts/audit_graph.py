"""Verify connectome graph statistics used by ConnectomeMarioPolicy.

Run: .venv/bin/python scripts/audit_graph.py   -> logs/audit_graph.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order


def main():
    p = REPO_ROOT / "data/processed"
    A = sp.load_npz(p / "subgraph_graph.npz").tocsr()
    nm = pd.read_csv(p / "subgraph_node_mapping.csv")
    meta = pd.read_csv(p / "node_metadata.csv", low_memory=False)
    merged = nm.merge(meta, on="body_id", how="left")
    vp = np.flatnonzero((merged["superclass"] == "visual_projection").to_numpy())
    dn = np.flatnonzero((merged["superclass"] == "descending_neuron").to_numpy())
    out = {"n_nodes": A.shape[0], "n_edges": int(A.nnz), "n_visual_projection": len(vp), "n_descending": len(dn),
           "self_loops": int(A.diagonal().astype(bool).sum())}
    row_sums = np.asarray(A.sum(1)).ravel()
    out["row_sum_stats"] = {"min_nonzero": float(row_sums[row_sums > 0].min()), "max": float(row_sums.max()),
                            "zero_rows": int((row_sums == 0).sum())}

    # Multi-source BFS from all VP neurons along A[i,j] (i presynaptic -> j postsynaptic)
    super_src = A.shape[0]
    B = sp.vstack([sp.hstack([A, sp.csr_matrix((A.shape[0], 1))]),
                   sp.csr_matrix((np.ones(len(vp)), (np.zeros(len(vp)), vp)), shape=(1, A.shape[0] + 1))]).tocsr()
    reach = breadth_first_order(B, super_src, directed=True, return_predecessors=False)
    reach_set = np.zeros(A.shape[0] + 1, bool); reach_set[reach] = True
    out["dn_reachable_from_any_vp"] = int(reach_set[dn].sum())

    # VPs with a path to at least one DN: BFS on reversed graph from all DNs
    AT = A.T.tocsr()
    C = sp.vstack([sp.hstack([AT, sp.csr_matrix((A.shape[0], 1))]),
                   sp.csr_matrix((np.ones(len(dn)), (np.zeros(len(dn)), dn)), shape=(1, A.shape[0] + 1))]).tocsr()
    back = breadth_first_order(C, super_src, directed=True, return_predecessors=False)
    bset = np.zeros(A.shape[0] + 1, bool); bset[back] = True
    out["vp_with_path_to_some_dn"] = int(bset[vp].sum())

    # Shortest hop distance VP -> DN (relevant: the model runs only num_ticks=2 recurrent steps per frame)
    dist = np.full(A.shape[0], -1); frontier = vp.copy(); dist[vp] = 0; d = 0
    Ab = (A > 0).astype(np.int8).tocsr()
    while frontier.size and d < 20:
        d += 1
        nxt = np.unique(Ab[frontier].indices)
        nxt = nxt[dist[nxt] < 0]
        dist[nxt] = d; frontier = nxt
    dn_d = dist[dn]
    out["dn_hop_distance_from_vp"] = {str(k): int((dn_d == k).sum()) for k in sorted(set(dn_d.tolist()))}
    out["dn_reachable_within_2_hops"] = int(((dn_d >= 0) & (dn_d <= 2)).sum())
    out["dn_reachable_within_1_hop"] = int(((dn_d >= 0) & (dn_d <= 1)).sum())
    print(json.dumps(out, indent=1))
    (REPO_ROOT / "logs/audit_graph.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

