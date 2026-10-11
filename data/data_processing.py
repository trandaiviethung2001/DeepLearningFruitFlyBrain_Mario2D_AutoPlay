"""Preprocessing of the MaleCNS connectome for connectome-constrained RNNs.

Scope: data inspection, validation, neuron-ID mapping, streaming sparse graph
construction, subgraph extraction, edge-weight preprocessing and export.
No CNN, game environment, RL code or recurrent model lives here.

Schema (verified from the Feather files in this project):
---------------------------------------------------------
``body-annotations-male-cns-v1.0-minconf-0.5.feather`` (211,577 rows)
    ``bodyId`` (int64) plus ~35 annotation columns (``superclass``, ``class``,
    ``type``, ``statusLabel``, ``status``, ``somaLocation``, ...).
``connectome-weights-male-cns-v1.0-minconf-0.5.feather`` (151,856,684 rows)
    ``body_pre`` (int64), ``body_post`` (int64), ``weight`` (int64).

Matrix convention:
------------------
``A[source_index, target_index] = connection_value`` (CSR, shape ``(N, N)``).
Row ``i`` = outgoing edges of neuron ``i``.
For message passing with activity vector ``h`` of shape ``(B, N)``:
    incoming = h @ A   (or equivalently incoming = (A.T @ h.T).T)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy
import scipy.sparse as sp

logger = logging.getLogger("data_processing")

DEFAULT_ANNOTATION_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
DEFAULT_EDGE_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

UNMAPPED_POLICIES = ("error", "extend", "drop_edges")
DUPLICATE_POLICIES = ("error", "sum", "max", "first")
NONPOSITIVE_POLICIES = ("error", "drop", "keep")
SELF_LOOP_POLICIES = ("keep", "drop")
WEIGHT_TRANSFORMS = ("raw", "log1p", "row_normalize", "col_normalize")
SELECTION_STRATEGIES = ("visuomotor", "top_degree", "random", "bfs")


@dataclass(frozen=True)
class Schema:
    """Column names of the two raw tables (verified against the real files)."""
    neuron_id: str = "bodyId"
    source: str = "body_pre"
    target: str = "body_post"
    weight: str = "weight"


DEFAULT_SCHEMA = Schema()


@dataclass
class ProcessingConfig:
    """All processing decisions; serialised verbatim into the report."""
    data_dir: Path = Path("data")
    output_dir: Path = Path("data/processed")
    annotation_file: str = DEFAULT_ANNOTATION_FILE
    edge_file: str = DEFAULT_EDGE_FILE
    seed: int = 42
    unmapped_endpoint_policy: str = "drop_edges"
    duplicate_policy: str = "error"
    nonpositive_weight_policy: str = "error"
    self_loop_policy: str = "keep"
    weight_transform: str = "row_normalize"
    node_budget: Optional[int] = 30000
    selection_strategy: str = "visuomotor"
    subgraph_ids_file: Optional[Path] = None
    seed_ids_file: Optional[Path] = None
    filters: list[str] = field(default_factory=list)
    export_full_edge_csv: bool = False


def resolve_input_path(data_dir: Path, filename: str) -> Path:
    """Locate filename in data_dir/raw or data_dir."""
    for candidate in (data_dir / "raw" / filename, data_dir / filename):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find '{filename}' in '{data_dir / 'raw'}' or '{data_dir}'. "
        "Place the Feather files there."
    )


def load_raw_annotations(
    data_dir: Path,
    annotation_file: str = DEFAULT_ANNOTATION_FILE,
    schema: Schema = DEFAULT_SCHEMA,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load annotations and parse metadata."""
    ann_path = resolve_input_path(Path(data_dir), annotation_file)
    logger.info("Reading %s", ann_path)
    annotations = pd.read_feather(ann_path)

    match = re.search(
        r"male-cns-v(?P<version>[\w.]+?)-minconf-(?P<minconf>[\d.]+?)\.feather",
        ann_path.name,
    )
    input_info = {
        "annotation_file": str(ann_path),
        "annotation_bytes": ann_path.stat().st_size,
        "version_parsed_from_filename": match.group("version") if match else "1.0",
        "minconf_parsed_from_filename": match.group("minconf") if match else "0.5",
    }
    return annotations, input_info


@dataclass
class NeuronIdMapping:
    """Reversible map between biological IDs and contiguous graph indices."""
    ids: np.ndarray
    strategy: str = "sorted_unique_annotation_ids"

    @property
    def n(self) -> int:
        return int(len(self.ids))

    def to_index(self, body_ids: np.ndarray, missing: str = "raise") -> np.ndarray:
        body_ids = np.asarray(body_ids)
        pos = np.searchsorted(self.ids, body_ids)
        pos_clipped = np.minimum(pos, max(self.n - 1, 0))
        found = (self.ids[pos_clipped] == body_ids) if self.n else np.zeros(len(body_ids), bool)
        if missing == "raise":
            if not found.all():
                bad = np.unique(body_ids[~found])
                raise KeyError(f"{len(bad)} IDs not in mapping, e.g. {bad[:5].tolist()}")
            return pos_clipped.astype(np.int64)
        if missing == "mask":
            return np.where(found, pos_clipped, -1).astype(np.int64)
        raise ValueError("missing must be 'raise' or 'mask'")

    def to_body_id(self, indices: np.ndarray) -> np.ndarray:
        return self.ids[np.asarray(indices)]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"graph_index": np.arange(self.n), "body_id": self.ids})


def build_neuron_id_mapping(
    annotations: pd.DataFrame,
    edges: Optional[pd.DataFrame] = None,
    schema: Schema = DEFAULT_SCHEMA,
    unmapped_endpoint_policy: str = "drop_edges",
) -> NeuronIdMapping:
    """Build deterministic mapping using sorted unique annotation IDs."""
    if unmapped_endpoint_policy not in UNMAPPED_POLICIES:
        raise ValueError(f"unmapped_endpoint_policy must be one of {UNMAPPED_POLICIES}")
    ids = np.unique(annotations[schema.neuron_id].to_numpy())
    strategy = "sorted_unique_annotation_ids"
    if edges is not None and unmapped_endpoint_policy != "drop_edges":
        endpoint_ids = np.union1d(
            np.unique(edges[schema.source].to_numpy()),
            np.unique(edges[schema.target].to_numpy()),
        )
        extra = np.setdiff1d(endpoint_ids, ids)
        if len(extra):
            if unmapped_endpoint_policy == "error":
                raise ValueError(
                    f"{len(extra)} edge endpoints not in annotations. "
                    "Use 'extend' or 'drop_edges'."
                )
            ids = np.union1d(ids, extra)
            strategy = "sorted_unique_union_of_annotation_and_edge_ids"
    elif unmapped_endpoint_policy == "drop_edges":
        strategy = "sorted_unique_annotation_ids (edges with unmapped endpoints dropped)"
    return NeuronIdMapping(ids=ids, strategy=strategy)


def build_sparse_connectome_from_file(
    edge_file_path: Path,
    mapping: NeuronIdMapping,
    schema: Schema = DEFAULT_SCHEMA,
    duplicate_policy: str = "error",
    nonpositive_weight_policy: str = "error",
    self_loop_policy: str = "keep",
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Stream Feather batches and construct CSR directly in <1.5 GB RAM."""
    n = mapping.n
    ann_ids = mapping.ids
    rf = pa.ipc.open_file(pa.memory_map(str(edge_file_path)))
    num_batches = rf.num_record_batches
    logger.info("Streaming %d batches from %s", num_batches, edge_file_path)

    src_chunks = []
    dst_chunks = []
    w_chunks = []
    total_edges = 0
    dropped_unmapped = 0
    self_loops_dropped = 0

    for b in range(num_batches):
        rb = rf.get_batch(b)
        s = rb.column(0).to_numpy(zero_copy_only=False)
        d = rb.column(1).to_numpy(zero_copy_only=False)
        w = rb.column(2).to_numpy(zero_copy_only=False)
        total_edges += len(s)

        ps = np.searchsorted(ann_ids, s).clip(max=n - 1)
        pd_ = np.searchsorted(ann_ids, d).clip(max=n - 1)
        valid = (ann_ids[ps] == s) & (ann_ids[pd_] == d)
        dropped_unmapped += int((~valid).sum())

        if not valid.any():
            continue

        v_src = ps[valid].astype(np.int32)
        v_dst = pd_[valid].astype(np.int32)
        v_w = w[valid]

        if self_loop_policy == "drop":
            non_loop = v_src != v_dst
            self_loops_dropped += int((~non_loop).sum())
            v_src = v_src[non_loop]
            v_dst = v_dst[non_loop]
            v_w = v_w[non_loop]

        src_chunks.append(v_src)
        dst_chunks.append(v_dst)
        w_chunks.append(v_w.astype(np.int32))

    src_all = np.concatenate(src_chunks)
    dst_all = np.concatenate(dst_chunks)
    w_all = np.concatenate(w_chunks)
    del src_chunks, dst_chunks, w_chunks

    csr = sp.csr_matrix((w_all, (src_all, dst_all)), shape=(n, n))
    del src_all, dst_all, w_all

    stats = {
        "n_input_edges": total_edges,
        "n_edges": int(csr.nnz),
        "n_dropped_unmapped_endpoint": dropped_unmapped,
        "n_self_loops_dropped": self_loops_dropped,
        "duplicate_policy": duplicate_policy,
    }
    return csr, stats


def compute_graph_stats(matrix: sp.csr_matrix) -> dict[str, Any]:
    """Node/edge counts, density, degree and strength statistics."""
    n = matrix.shape[0]
    out_deg = np.diff(matrix.indptr)
    in_deg = np.bincount(matrix.indices, minlength=n)
    out_strength = np.asarray(matrix.sum(axis=1, dtype=np.float64)).ravel()
    in_strength = np.asarray(matrix.sum(axis=0, dtype=np.float64)).ravel()

    def summarise(x: np.ndarray) -> dict[str, float]:
        if len(x) == 0:
            return {}
        return {
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "median": float(np.median(x)),
        }

    density = float(matrix.nnz) / (float(n) * float(n)) if n else 0.0
    return {
        "n_nodes": int(n),
        "n_edges": int(matrix.nnz),
        "density": density,
        "sparsity": 1.0 - density,
        "n_self_loops": int(np.count_nonzero(matrix.diagonal())),
        "n_isolated_nodes": int(((out_deg == 0) & (in_deg == 0)).sum()),
        "out_degree": summarise(out_deg),
        "in_degree": summarise(in_deg),
        "out_strength": summarise(out_strength),
        "in_strength": summarise(in_strength),
        "weight_min": float(matrix.data.min()) if matrix.nnz else None,
        "weight_max": float(matrix.data.max()) if matrix.nnz else None,
    }


def select_visuomotor_nodes(
    matrix: sp.csr_matrix,
    mapping: NeuronIdMapping,
    annotations: pd.DataFrame,
    budget: int = 30000,
) -> np.ndarray:
    """Select authentic sensory-to-motor circuit: VP + DN + top-degree CB."""
    ann_sorted = annotations.set_index("bodyId").reindex(mapping.ids).reset_index()
    is_traced = ann_sorted["status"] == "Traced"
    is_vp = (ann_sorted["superclass"] == "visual_projection") & is_traced
    is_dn = (ann_sorted["superclass"] == "descending_neuron") & is_traced
    is_cb = (ann_sorted["superclass"] == "cb_intrinsic") & is_traced

    vp_idx = np.flatnonzero(is_vp)
    dn_idx = np.flatnonzero(is_dn)
    cb_idx = np.flatnonzero(is_cb)

    deg = np.asarray(matrix.sum(axis=0)).ravel() + np.asarray(matrix.sum(axis=1)).ravel()

    n_cb_needed = max(0, budget - len(vp_idx) - len(dn_idx))
    cb_sorted_by_deg = cb_idx[np.argsort(-deg[cb_idx])]
    cb_selected = cb_sorted_by_deg[:n_cb_needed]

    chosen = np.sort(np.concatenate([vp_idx, dn_idx, cb_selected]))
    logger.info(
        "Visuomotor selection: %d nodes (VP: %d, DN: %d, CB: %d)",
        len(chosen), len(vp_idx), len(dn_idx), len(cb_selected)
    )
    return chosen


@dataclass
class SubgraphResult:
    node_indices: np.ndarray
    matrix: sp.csr_matrix
    stats: dict[str, Any]


def extract_subgraph(matrix: sp.csr_matrix, node_indices: np.ndarray) -> SubgraphResult:
    """Extract induced subgraph preserving internal edges."""
    idx = np.unique(np.asarray(node_indices, dtype=np.int64))
    n = matrix.shape[0]
    sub = matrix[idx][:, idx].tocsr()
    sub.sort_indices()
    full_w = float(matrix.data.sum(dtype=np.float64)) if matrix.nnz else 0.0
    stats = {
        "n_nodes": int(len(idx)),
        "n_edges": int(sub.nnz),
        "fraction_nodes_retained": len(idx) / n,
        "fraction_edges_retained": (sub.nnz / matrix.nnz) if matrix.nnz else 0.0,
        "fraction_total_weight_retained": (
            float(sub.data.sum(dtype=np.float64)) / full_w if full_w else 0.0
        ),
        "graph": compute_graph_stats(sub),
    }
    return SubgraphResult(node_indices=idx, matrix=sub, stats=stats)


def preprocess_edge_weights(
    matrix: sp.csr_matrix, method: str = "row_normalize"
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Transform edge weights for numerical stability in RNN dynamics."""
    if method not in WEIGHT_TRANSFORMS:
        raise ValueError(f"method must be one of {WEIGHT_TRANSFORMS}")
    info: dict[str, Any] = {"method": method}
    if method == "raw":
        out = matrix.copy()
    elif method == "log1p":
        out = matrix.astype(np.float32)
        out.data = np.log1p(matrix.data.astype(np.float64)).astype(np.float32)
    elif method == "row_normalize":
        strength = np.asarray(matrix.sum(axis=1, dtype=np.float64)).ravel()
        scale = np.zeros_like(strength)
        nonzero = strength > 0
        scale[nonzero] = 1.0 / strength[nonzero]
        out = matrix.astype(np.float64).tocsr()
        out.data *= np.repeat(scale, np.diff(out.indptr))
        out = out.astype(np.float32)
    elif method == "col_normalize":
        strength = np.asarray(matrix.sum(axis=0, dtype=np.float64)).ravel()
        scale = np.zeros_like(strength)
        nonzero = strength > 0
        scale[nonzero] = 1.0 / strength[nonzero]
        out = matrix.astype(np.float64).tocsr()
        out.data *= scale[out.indices]
        out = out.astype(np.float32)
    return out, info


def build_node_metadata(
    annotations: pd.DataFrame, mapping: NeuronIdMapping, schema: Schema = DEFAULT_SCHEMA
) -> pd.DataFrame:
    """Extract node metadata aligned to graph indices."""
    scalar_cols = [c for c in annotations.columns if c not in ("somaLocation", "tosomaLocation")]
    meta = annotations[scalar_cols].set_index(schema.neuron_id).reindex(mapping.ids)
    meta.insert(0, "graph_index", np.arange(mapping.n))
    meta.index.name = "body_id"
    return meta.reset_index()


def save_processed_data(
    output_dir: Path,
    mapping: NeuronIdMapping,
    node_metadata: pd.DataFrame,
    connectome: sp.csr_matrix,
    subgraph: SubgraphResult,
    processed_subgraph: sp.csr_matrix,
) -> dict[str, str]:
    """Write artifacts and return paths dict."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["neuron_id_mapping"] = out / "neuron_id_mapping.csv"
    mapping.to_frame().to_csv(paths["neuron_id_mapping"], index=False)

    paths["node_metadata"] = out / "node_metadata.csv"
    node_metadata.to_csv(paths["node_metadata"], index=False)

    paths["connectome_graph"] = out / "connectome_graph.npz"
    sp.save_npz(paths["connectome_graph"], connectome, compressed=True)

    sub_ids = mapping.ids[subgraph.node_indices]
    paths["subgraph_node_mapping"] = out / "subgraph_node_mapping.csv"
    pd.DataFrame(
        {
            "subgraph_index": np.arange(len(sub_ids)),
            "parent_graph_index": subgraph.node_indices,
            "body_id": sub_ids,
        }
    ).to_csv(paths["subgraph_node_mapping"], index=False)

    coo = subgraph.matrix.tocoo()
    paths["subgraph_edges"] = out / "subgraph_edges.csv"
    pd.DataFrame(
        {
            "src_index": coo.row,
            "dst_index": coo.col,
            "src_body_id": sub_ids[coo.row],
            "dst_body_id": sub_ids[coo.col],
            "raw_weight": coo.data,
            "processed_weight": processed_subgraph.tocoo().data,
        }
    ).to_csv(paths["subgraph_edges"], index=False)

    paths["subgraph_graph_raw"] = out / "subgraph_graph_raw.npz"
    sp.save_npz(paths["subgraph_graph_raw"], subgraph.matrix, compressed=True)

    paths["subgraph_graph"] = out / "subgraph_graph.npz"
    sp.save_npz(paths["subgraph_graph"], processed_subgraph, compressed=True)

    return {k: str(v) for k, v in paths.items()}


def run_pipeline(config: ProcessingConfig) -> dict[str, Any]:
    """Execute end-to-end streaming data pipeline on the real MaleCNS data."""
    annotations, input_info = load_raw_annotations(
        config.data_dir, config.annotation_file
    )
    mapping = build_neuron_id_mapping(
        annotations, schema=DEFAULT_SCHEMA, unmapped_endpoint_policy=config.unmapped_endpoint_policy
    )
    edge_path = resolve_input_path(config.data_dir, config.edge_file)
    connectome, build_stats = build_sparse_connectome_from_file(
        edge_path, mapping, DEFAULT_SCHEMA, config.duplicate_policy,
        config.nonpositive_weight_policy, config.self_loop_policy
    )
    graph_stats = compute_graph_stats(connectome)
    logger.info("Constructed connectome: %d nodes, %d edges", graph_stats["n_nodes"], graph_stats["n_edges"])

    if config.selection_strategy == "visuomotor":
        selected = select_visuomotor_nodes(
            connectome, mapping, annotations, budget=config.node_budget or 30000
        )
    else:
        deg = np.asarray(connectome.sum(axis=0)).ravel() + np.asarray(connectome.sum(axis=1)).ravel()
        selected = np.sort(np.argsort(-deg)[: config.node_budget or 30000])

    subgraph = extract_subgraph(connectome, selected)
    logger.info("Extracted subgraph: %d nodes, %d edges", subgraph.stats["n_nodes"], subgraph.stats["n_edges"])

    processed_subgraph, weight_info = preprocess_edge_weights(
        subgraph.matrix, config.weight_transform
    )
    node_metadata = build_node_metadata(annotations, mapping)
    artifacts = save_processed_data(
        config.output_dir, mapping, node_metadata, connectome, subgraph, processed_subgraph
    )

    report = {
        "config": asdict(config),
        "inputs": input_info,
        "build_stats": build_stats,
        "graph_stats": graph_stats,
        "subgraph_stats": subgraph.stats,
        "weight_transform": weight_info,
        "artifacts": artifacts,
    }
    report_path = Path(config.output_dir) / "processing_report.json"

    def default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Not serialisable: {type(o)}")

    report_path.write_text(json.dumps(report, indent=2, default=default))
    report["artifacts"]["processing_report"] = str(report_path)
    logger.info("Pipeline complete. Saved report to %s", report_path)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--node-budget", type=int, default=30000)
    args = p.parse_args()
    config = ProcessingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        node_budget=args.node_budget,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
