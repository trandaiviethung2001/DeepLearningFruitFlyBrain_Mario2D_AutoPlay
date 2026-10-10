"""Preprocessing of the MaleCNS connectome for connectome-constrained RNNs.

Scope: data inspection, validation, neuron-ID mapping, sparse graph
construction, subgraph extraction, edge-weight preprocessing and export.
No CNN, game environment, RL code or recurrent model lives here.

Schema (verified from the Feather files in this project, 2026-10)
----------------------------------------------------------------
``body-annotations-male-cns-v1.0-minconf-0.5.feather`` (211,577 rows)
    ``bodyId`` (int64) plus ~35 annotation columns (``superclass``, ``class``,
    ``type``, ``statusLabel``, ``somaLocation`` (list<int64>), ...).
``connectome-weights-male-cns-v1.0-minconf-0.5.feather`` (151,856,684 rows)
    ``body_pre`` (int64), ``body_post`` (int64), ``weight`` (int64).

Interpretation that is *assumed* and must be confirmed against
https://male-cns.janelia.org/ (it is NOT verifiable from the files alone):
    * ``body_pre`` -> ``body_post`` is the direction of signal flow
      (presynaptic -> postsynaptic).
    * ``weight`` is a non-negative integer connection strength (believed to be
      a synapse count). It is NOT a calibrated physiological synaptic strength.
    * ``bodyId`` in the annotations and ``body_pre``/``body_post`` share one ID
      space. This is *validated* here, never assumed.
    * Whether the confidence threshold ("minconf-0.5") is applied to synapses,
      to neuron identity, or both is unknown. Version/threshold strings are
      only parsed from file names and are labelled as such in the report.

Matrix convention
-----------------
``A[source_index, target_index] = connection_value`` (CSR, shape ``(N, N)``).
Row ``i`` = outgoing edges of neuron ``i``. For message passing with
activity vector ``h`` of shape ``(N,)`` or ``(B, N)``::

    incoming = A.T @ h        # (N,)      or equivalently
    incoming = h @ A          # (B, N) @ (N, N) -> (B, N)

Never use ``A @ h`` for forward signal propagation: that reverses direction.

Memory: the full graph has ~1.5e8 edges. Expect roughly 10-15 GB peak RAM when
processing the real files (sorting for duplicate detection dominates).
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
SELECTION_STRATEGIES = ("top_degree", "random", "bfs")


# --------------------------------------------------------------------------- #
# Configuration objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Schema:
    """Column names of the two raw tables (verified against the real files)."""

    neuron_id: str = "bodyId"  # in the annotation table
    source: str = "body_pre"  # in the edge table
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
    # Graph construction policies (conservative defaults: fail loudly).
    unmapped_endpoint_policy: str = "error"
    duplicate_policy: str = "error"
    nonpositive_weight_policy: str = "error"
    self_loop_policy: str = "keep"
    # Weight preprocessing is applied to the *subgraph* only (see docs).
    weight_transform: str = "raw"
    # Subgraph selection.
    node_budget: Optional[int] = 30000
    selection_strategy: str = "top_degree"
    subgraph_ids_file: Optional[Path] = None
    seed_ids_file: Optional[Path] = None
    filters: list[str] = field(default_factory=list)  # "column=v1,v2"
    export_full_edge_csv: bool = False


# --------------------------------------------------------------------------- #
# Loading and inspection
# --------------------------------------------------------------------------- #
def resolve_input_path(data_dir: Path, filename: str) -> Path:
    """Locate ``filename`` in ``data_dir/raw`` or, failing that, ``data_dir``."""
    for candidate in (data_dir / "raw" / filename, data_dir / filename):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find '{filename}' in '{data_dir / 'raw'}' or '{data_dir}'. "
        "This module never downloads data; place the Feather files there."
    )


def load_raw_data(
    data_dir: Path,
    annotation_file: str = DEFAULT_ANNOTATION_FILE,
    edge_file: str = DEFAULT_EDGE_FILE,
    schema: Schema = DEFAULT_SCHEMA,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load both Feather files.

    Only the three edge columns required by ``schema`` are read from the edge
    table. Returns ``(annotations, edges, input_info)``.
    """
    ann_path = resolve_input_path(Path(data_dir), annotation_file)
    edge_path = resolve_input_path(Path(data_dir), edge_file)
    logger.info("Reading %s", ann_path)
    annotations = pd.read_feather(ann_path)
    logger.info("Reading %s", edge_path)
    try:
        edges = pd.read_feather(
            edge_path, columns=[schema.source, schema.target, schema.weight]
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Edge file {edge_path} lacks columns "
            f"{[schema.source, schema.target, schema.weight]}: {exc}"
        ) from exc

    match = re.search(
        r"male-cns-v(?P<version>[\w.]+?)-minconf-(?P<minconf>[\d.]+?)\.feather",
        edge_path.name,
    )
    input_info = {
        "annotation_file": str(ann_path),
        "edge_file": str(edge_path),
        "annotation_bytes": ann_path.stat().st_size,
        "edge_bytes": edge_path.stat().st_size,
        "version_parsed_from_filename": match.group("version") if match else None,
        "minconf_parsed_from_filename": match.group("minconf") if match else None,
        "version_note": (
            "Parsed from the file name only; not verified against dataset "
            "metadata. What 'minconf' thresholds is not verified."
        ),
    }
    return annotations, edges, input_info


def _hashable_columns(df: pd.DataFrame) -> list[str]:
    """Columns whose cells are scalars (list/array columns are excluded)."""
    cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().head(100)
            if any(isinstance(v, (list, tuple, np.ndarray)) for v in sample):
                continue
        cols.append(col)
    return cols


def inspect_data(
    annotations: pd.DataFrame, edges: pd.DataFrame, n_examples: int = 5
) -> dict[str, Any]:
    """Log shape, columns, dtypes, missing values, duplicates and examples.

    Duplicate *edge rows* are not computed here (too expensive at 1.5e8 rows);
    duplicate (source, target) pairs are reported by
    :func:`build_sparse_connectome`.
    """
    summary: dict[str, Any] = {}
    for name, df in (("annotations", annotations), ("edges", edges)):
        scalar_cols = _hashable_columns(df)
        missing = df.isna().sum()
        missing = missing[missing > 0]
        entry: dict[str, Any] = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "missing_values": {c: int(v) for c, v in missing.items()},
        }
        if name == "annotations":
            entry["duplicate_rows_scalar_columns"] = int(
                df.duplicated(subset=scalar_cols).sum()
            )
            entry["list_valued_columns"] = [c for c in df.columns if c not in scalar_cols]
        summary[name] = entry
        logger.info("--- %s: shape=%s", name, df.shape)
        logger.info("columns/dtypes: %s", entry["dtypes"])
        logger.info("columns with missing values: %s", entry["missing_values"] or "none")
        if "duplicate_rows_scalar_columns" in entry:
            logger.info("duplicate rows: %d", entry["duplicate_rows_scalar_columns"])
        logger.info("head:\n%s", df.head(n_examples).to_string())
    return summary


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_connectome(
    annotations: pd.DataFrame,
    edges: pd.DataFrame,
    schema: Schema = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Validate IDs, endpoints and weights *without modifying* any data.

    Raises ``ValueError`` only for structural problems that make the data
    unusable (missing columns, null/duplicate annotation IDs, null endpoints,
    non-integer-valued IDs, non-numeric weights). Policy-dependent issues
    (unmapped endpoints, non-positive weights) are reported here and enforced
    in :func:`build_neuron_id_mapping` / :func:`build_sparse_connectome`.
    """
    for df, cols, label in (
        (annotations, [schema.neuron_id], "annotations"),
        (edges, [schema.source, schema.target, schema.weight], "edges"),
    ):
        absent = [c for c in cols if c not in df.columns]
        if absent:
            raise ValueError(
                f"{label} table is missing required columns {absent}; "
                f"available: {list(df.columns)[:20]}..."
            )

    ids = annotations[schema.neuron_id]
    if ids.isna().any():
        raise ValueError(f"{int(ids.isna().sum())} annotation rows have null {schema.neuron_id}")
    if not pd.api.types.is_integer_dtype(ids):
        raise ValueError(f"{schema.neuron_id} must be an integer column, got {ids.dtype}")
    n_dup_ids = int(ids.duplicated().sum())
    if n_dup_ids:
        raise ValueError(
            f"{n_dup_ids} duplicated {schema.neuron_id} values in annotations; "
            "the authoritative neuron inventory must have unique IDs."
        )

    for col in (schema.source, schema.target):
        series = edges[col]
        if series.isna().any():
            raise ValueError(f"{int(series.isna().sum())} edges have a null '{col}' endpoint")
        if not pd.api.types.is_integer_dtype(series):
            raise ValueError(f"Edge column '{col}' must be integer typed, got {series.dtype}")
    w = edges[schema.weight]
    if not pd.api.types.is_numeric_dtype(w):
        raise ValueError(f"Weight column '{schema.weight}' must be numeric, got {w.dtype}")
    if w.isna().any():
        raise ValueError(f"{int(w.isna().sum())} edges have a null weight")

    ann_ids = np.unique(ids.to_numpy())
    src_ids = np.unique(edges[schema.source].to_numpy())
    dst_ids = np.unique(edges[schema.target].to_numpy())
    edge_union = np.union1d(src_ids, dst_ids)
    all_union = np.union1d(ann_ids, edge_union)
    unmapped_src = np.setdiff1d(src_ids, ann_ids)
    unmapped_dst = np.setdiff1d(dst_ids, ann_ids)
    unmapped_any = np.union1d(unmapped_src, unmapped_dst)
    w_values = w.to_numpy()
    report = {
        "n_edges": int(len(edges)),
        "unique_ids": {
            "annotations": int(len(ann_ids)),
            "edge_sources": int(len(src_ids)),
            "edge_targets": int(len(dst_ids)),
            "edge_union_sources_targets": int(len(edge_union)),
            "edge_intersection_sources_targets": int(len(np.intersect1d(src_ids, dst_ids))),
            "annotations_union_edges": int(len(all_union)),
            "annotations_intersection_edges": int(len(np.intersect1d(ann_ids, edge_union))),
        },
        "endpoints_not_in_annotations": {
            "unique_sources": int(len(unmapped_src)),
            "unique_targets": int(len(unmapped_dst)),
            "unique_any": int(len(unmapped_any)),
            "example_ids": [int(x) for x in unmapped_any[:10]],
        },
        "annotated_neurons_without_edges": int(len(np.setdiff1d(ann_ids, edge_union))),
        "weights": {
            "dtype": str(w.dtype),
            "min": w_values.min().item() if len(w_values) else None,
            "max": w_values.max().item() if len(w_values) else None,
            "n_zero": int((w_values == 0).sum()),
            "n_negative": int((w_values < 0).sum()),
            "all_integer_valued": bool(
                pd.api.types.is_integer_dtype(w) or np.all(np.mod(w_values, 1) == 0)
            ),
        },
        "n_self_loops_in_raw_edges": int(
            (edges[schema.source].to_numpy() == edges[schema.target].to_numpy()).sum()
        ),
    }
    if report["endpoints_not_in_annotations"]["unique_any"]:
        logger.warning(
            "%d edge endpoint IDs are absent from the annotation table",
            report["endpoints_not_in_annotations"]["unique_any"],
        )
    if report["weights"]["n_zero"] or report["weights"]["n_negative"]:
        logger.warning(
            "Non-positive weights: %d zero, %d negative",
            report["weights"]["n_zero"],
            report["weights"]["n_negative"],
        )
    return report


# --------------------------------------------------------------------------- #
# Neuron ID mapping
# --------------------------------------------------------------------------- #
@dataclass
class NeuronIdMapping:
    """Reversible map between biological IDs and contiguous graph indices.

    ``ids`` is sorted ascending and unique, so ``index = position in ids`` is
    deterministic regardless of row order in the raw files.
    """

    ids: np.ndarray
    strategy: str = "sorted_unique_annotation_ids"

    @property
    def n(self) -> int:
        return int(len(self.ids))

    def to_index(self, body_ids: np.ndarray, missing: str = "raise") -> np.ndarray:
        """Map biological IDs to indices; ``missing`` is 'raise' or 'mask' (-1)."""
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
    unmapped_endpoint_policy: str = "error",
) -> NeuronIdMapping:
    """Build the ID mapping. The annotation table is the authoritative inventory.

    ``unmapped_endpoint_policy`` controls edge endpoints absent from it:
      * ``'error'``      - raise (default; nothing silently dropped).
      * ``'extend'``     - add those IDs to the inventory (they get no metadata).
      * ``'drop_edges'`` - keep the inventory; such edges are dropped (and
        counted) later in :func:`build_sparse_connectome`.
    """
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
                    f"{len(extra)} edge endpoints are not in the annotation table "
                    f"(e.g. {extra[:5].tolist()}). Choose unmapped_endpoint_policy "
                    "'extend' or 'drop_edges' after deciding how to treat them."
                )
            ids = np.union1d(ids, extra)
            strategy = "sorted_unique_union_of_annotation_and_edge_ids"
    elif unmapped_endpoint_policy == "drop_edges":
        strategy = "sorted_unique_annotation_ids (edges with unmapped endpoints dropped)"
    return NeuronIdMapping(ids=ids, strategy=strategy)


# --------------------------------------------------------------------------- #
# Sparse graph
# --------------------------------------------------------------------------- #
def build_sparse_connectome(
    edges: pd.DataFrame,
    mapping: NeuronIdMapping,
    schema: Schema = DEFAULT_SCHEMA,
    duplicate_policy: str = "error",
    nonpositive_weight_policy: str = "error",
    self_loop_policy: str = "keep",
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Build ``A[source_index, target_index] = weight`` as a CSR matrix.

    Policies (all counted in the returned stats):
      * duplicate (source, target) pairs: ``error`` | ``sum`` | ``max`` | ``first``.
        Which is correct depends on dataset semantics (are duplicates separate
        measurements, or split by e.g. neuropil?). Verify before using ``sum``.
      * non-positive weights: ``error`` | ``drop`` | ``keep``.
      * self loops: ``keep`` (never added; only retained if present) | ``drop``.
      * edges with endpoints outside the mapping are dropped and counted
        (only reachable with ``unmapped_endpoint_policy='drop_edges'``).

    No dense N x N matrix is created.
    """
    for name, allowed, value in (
        ("duplicate_policy", DUPLICATE_POLICIES, duplicate_policy),
        ("nonpositive_weight_policy", NONPOSITIVE_POLICIES, nonpositive_weight_policy),
        ("self_loop_policy", SELF_LOOP_POLICIES, self_loop_policy),
    ):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got '{value}'")

    n = mapping.n
    src = mapping.to_index(edges[schema.source].to_numpy(), missing="mask")
    dst = mapping.to_index(edges[schema.target].to_numpy(), missing="mask")
    w = edges[schema.weight].to_numpy()
    stats: dict[str, Any] = {"n_input_edges": int(len(edges))}

    keep = (src >= 0) & (dst >= 0)
    stats["n_dropped_unmapped_endpoint"] = int((~keep).sum())
    if stats["n_dropped_unmapped_endpoint"]:
        src, dst, w = src[keep], dst[keep], w[keep]
        logger.warning("Dropped %d edges with unmapped endpoints", stats["n_dropped_unmapped_endpoint"])
    del keep

    nonpos = w <= 0
    stats["n_nonpositive_weight"] = int(nonpos.sum())
    if stats["n_nonpositive_weight"]:
        if nonpositive_weight_policy == "error":
            raise ValueError(
                f"{stats['n_nonpositive_weight']} edges have weight <= 0. Their meaning "
                "must be confirmed; set nonpositive_weight_policy to 'drop' or 'keep'."
            )
        if nonpositive_weight_policy == "drop":
            src, dst, w = src[~nonpos], dst[~nonpos], w[~nonpos]
    del nonpos

    loops = src == dst
    stats["n_self_loops_in_input"] = int(loops.sum())
    stats["n_self_loops_dropped"] = 0
    if self_loop_policy == "drop" and loops.any():
        stats["n_self_loops_dropped"] = int(loops.sum())
        src, dst, w = src[~loops], dst[~loops], w[~loops]
    del loops

    stats["n_duplicate_pairs_removed"] = 0
    stats["duplicate_policy"] = duplicate_policy
    if len(src) == 0:
        matrix = sp.csr_matrix((n, n), dtype=np.int32)
        stats["n_edges"] = 0
        return matrix, stats

    key = src * np.int64(n) + dst
    del src, dst
    order = np.argsort(key, kind="stable")  # stable => 'first' is input-order first
    key = key[order]
    w = w[order]
    del order
    is_start = np.empty(len(key), dtype=bool)
    is_start[0] = True
    np.not_equal(key[1:], key[:-1], out=is_start[1:])
    starts = np.flatnonzero(is_start)
    del is_start
    n_dups = len(key) - len(starts)
    stats["n_duplicate_pairs_removed"] = int(n_dups)
    if n_dups:
        if duplicate_policy == "error":
            dup_pos = np.flatnonzero(key[1:] == key[:-1])[0]
            k = int(key[dup_pos])
            raise ValueError(
                f"{n_dups} duplicate (source, target) edges, e.g. graph indices "
                f"({k // n}, {k % n}). Choose duplicate_policy 'sum', 'max' or 'first' "
                "once the dataset semantics are confirmed."
            )
        logger.warning("Collapsing %d duplicate edges with policy '%s'", n_dups, duplicate_policy)
        if duplicate_policy == "sum":
            w = np.add.reduceat(w.astype(np.int64) if w.dtype.kind in "iu" else w, starts)
        elif duplicate_policy == "max":
            w = np.maximum.reduceat(w, starts)
        else:  # first
            w = w[starts]
    else:
        w = w[starts]
    key = key[starts]

    rows = key // n
    cols = (key % n).astype(np.int32)
    del key
    if w.dtype.kind in "iu":
        w = w.astype(np.int32 if int(w.max()) < 2**31 and int(w.min()) > -(2**31) else np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    matrix = sp.csr_matrix((w, cols, indptr), shape=(n, n))
    stats["n_edges"] = int(matrix.nnz)
    return matrix, stats


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
        "n_zero_out_degree": int((out_deg == 0).sum()),
        "n_zero_in_degree": int((in_deg == 0).sum()),
        "out_degree": summarise(out_deg),
        "in_degree": summarise(in_deg),
        "out_strength": summarise(out_strength),
        "in_strength": summarise(in_strength),
        "weight_dtype": str(matrix.dtype),
        "weight_min": float(matrix.data.min()) if matrix.nnz else None,
        "weight_max": float(matrix.data.max()) if matrix.nnz else None,
    }


# --------------------------------------------------------------------------- #
# Subgraph extraction
# --------------------------------------------------------------------------- #
@dataclass
class SubgraphResult:
    """Induced subgraph. ``node_indices`` are indices into the parent graph."""

    node_indices: np.ndarray  # sorted, parent graph indices; sub index = position
    matrix: sp.csr_matrix  # raw (untransformed) values
    stats: dict[str, Any]


def _parse_filters(filters: Sequence[str]) -> list[tuple[str, set[str]]]:
    parsed = []
    for f in filters:
        if "=" not in f:
            raise ValueError(f"Filter '{f}' must look like 'column=value1,value2'")
        col, vals = f.split("=", 1)
        parsed.append((col.strip(), {v.strip() for v in vals.split(",")}))
    return parsed


def select_nodes(
    matrix: sp.csr_matrix,
    mapping: NeuronIdMapping,
    annotations: Optional[pd.DataFrame] = None,
    schema: Schema = DEFAULT_SCHEMA,
    *,
    node_ids: Optional[Iterable[int]] = None,
    node_indices: Optional[Iterable[int]] = None,
    filters: Sequence[str] = (),
    seed_ids: Optional[Iterable[int]] = None,
    budget: Optional[int] = None,
    strategy: str = "top_degree",
    seed: int = 42,
) -> np.ndarray:
    """Return sorted parent-graph indices of the selected nodes.

    Candidate set = explicit ``node_ids``/``node_indices`` (union) if given,
    else all nodes, then restricted by generic annotation ``filters``
    (``'column=v1,v2'``, compared as strings; no biological meaning is
    assumed). If ``budget`` is smaller than the candidate count, trim with:
      * ``top_degree``: highest total (in+out) strength in the *full* graph,
        ties broken by index.
      * ``random``: seeded uniform sample (induced subgraphs of random nodes
        are extremely sparse; use only for baselines).
      * ``bfs``: expand over undirected connections from ``seed_ids`` layer by
        layer; the last layer is subsampled with the seeded RNG. May return
        fewer than ``budget`` nodes if the reachable set is smaller.
    Deterministic for identical inputs and ``seed``.
    """
    if strategy not in SELECTION_STRATEGIES:
        raise ValueError(f"strategy must be one of {SELECTION_STRATEGIES}")
    n = mapping.n
    rng = np.random.default_rng(seed)
    cand = np.ones(n, dtype=bool)

    if node_ids is not None or node_indices is not None:
        explicit = np.zeros(n, dtype=bool)
        if node_ids is not None:
            explicit[mapping.to_index(np.asarray(list(node_ids), dtype=np.int64))] = True
        if node_indices is not None:
            idx = np.asarray(list(node_indices), dtype=np.int64)
            if len(idx) and (idx.min() < 0 or idx.max() >= n):
                raise IndexError(f"node_indices must lie in [0, {n})")
            explicit[idx] = True
        cand &= explicit

    for col, values in _parse_filters(filters):
        if annotations is None or col not in annotations.columns:
            raise ValueError(f"Cannot filter on unknown annotation column '{col}'")
        matches = annotations.loc[
            annotations[col].astype(str).isin(values), schema.neuron_id
        ].to_numpy()
        match_idx = mapping.to_index(matches, missing="mask")
        mask = np.zeros(n, dtype=bool)
        mask[match_idx[match_idx >= 0]] = True
        cand &= mask

    cand_idx = np.flatnonzero(cand)
    if len(cand_idx) == 0:
        raise ValueError("Node selection is empty")
    if budget is None or budget >= len(cand_idx):
        return cand_idx
    if budget <= 0:
        raise ValueError("budget must be positive")

    if strategy == "random":
        chosen = rng.choice(cand_idx, size=budget, replace=False)
    elif strategy == "top_degree":
        total = np.asarray(matrix.sum(axis=1, dtype=np.float64)).ravel() + np.asarray(
            matrix.sum(axis=0, dtype=np.float64)
        ).ravel()
        # lexsort: last key primary -> descending strength, then ascending index
        order = np.lexsort((cand_idx, -total[cand_idx]))
        chosen = cand_idx[order[:budget]]
    else:  # bfs
        if seed_ids is None:
            raise ValueError("strategy 'bfs' requires seed_ids")
        seeds = mapping.to_index(np.asarray(list(seed_ids), dtype=np.int64))
        seeds = seeds[cand[seeds]]
        if len(seeds) == 0:
            raise ValueError("None of the seed IDs are in the candidate set")
        sym = (matrix.astype(bool) + matrix.T.astype(bool)).tocsr()
        visited = np.zeros(n, dtype=bool)
        frontier = np.unique(seeds)
        chosen_list: list[np.ndarray] = []
        count = 0
        while len(frontier) and count < budget:
            if count + len(frontier) > budget:
                frontier = np.sort(rng.choice(frontier, size=budget - count, replace=False))
            visited[frontier] = True
            chosen_list.append(frontier)
            count += len(frontier)
            if count >= budget:
                break
            nbrs = np.unique(sym[frontier].indices)
            frontier = nbrs[cand[nbrs] & ~visited[nbrs]]
        chosen = np.concatenate(chosen_list)
    return np.sort(chosen)


def extract_subgraph(
    matrix: sp.csr_matrix, node_indices: np.ndarray
) -> SubgraphResult:
    """Induced subgraph on ``node_indices`` (all edges among them are kept)."""
    idx = np.unique(np.asarray(node_indices, dtype=np.int64))
    n = matrix.shape[0]
    if len(idx) == 0:
        raise ValueError("node_indices is empty")
    if idx[0] < 0 or idx[-1] >= n:
        raise IndexError(f"node_indices must lie in [0, {n})")
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


# --------------------------------------------------------------------------- #
# Weight preprocessing
# --------------------------------------------------------------------------- #
def preprocess_edge_weights(
    matrix: sp.csr_matrix, method: str = "raw"
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Return a transformed copy of ``matrix`` (the input is never modified).

    * ``raw``           - unchanged values (default; no normalisation).
    * ``log1p``         - ``log(1 + w)``; requires all weights >= 0.
    * ``row_normalize`` - each *source* row sums to 1 (outgoing-strength
      normalisation). Removes absolute synapse-count differences between
      neurons (hubs no longer dominate) but also removes any information in
      total output strength. Rows with zero strength stay all-zero.
    * ``col_normalize`` - each *target* column sums to 1 (incoming-strength
      normalisation); zero-strength columns stay zero.
    Transforms are computed on the matrix passed in, i.e. on the retained
    subgraph's edges when applied to a subgraph.
    """
    if method not in WEIGHT_TRANSFORMS:
        raise ValueError(f"method must be one of {WEIGHT_TRANSFORMS}")
    info: dict[str, Any] = {"method": method, "input_dtype": str(matrix.dtype)}
    if method == "raw":
        out = matrix.copy()
    elif method == "log1p":
        if matrix.nnz and matrix.data.min() < 0:
            raise ValueError("log1p requires non-negative weights")
        out = matrix.astype(np.float32)
        out.data = np.log1p(matrix.data.astype(np.float64)).astype(np.float32)
    else:
        axis = 1 if method == "row_normalize" else 0
        strength = np.asarray(matrix.sum(axis=axis, dtype=np.float64)).ravel()
        scale = np.zeros_like(strength)
        nonzero = strength > 0
        scale[nonzero] = 1.0 / strength[nonzero]
        info["n_zero_strength_nodes"] = int((~nonzero).sum())
        out = matrix.astype(np.float64).tocsr()
        if axis == 1:
            out.data *= np.repeat(scale, np.diff(out.indptr))
        else:
            out.data *= scale[out.indices]
        out = out.astype(np.float32)
    info["output_dtype"] = str(out.dtype)
    return out, info


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def edges_to_frame(
    matrix: sp.csr_matrix,
    body_ids: np.ndarray,
    processed: Optional[sp.csr_matrix] = None,
) -> pd.DataFrame:
    """Edge list ``(src_index, dst_index, src_body_id, dst_body_id, weights)``.

    ``body_ids[i]`` is the biological ID of (local) node ``i``.
    """
    coo = matrix.tocoo()
    frame = pd.DataFrame(
        {
            "src_index": coo.row.astype(np.int64),
            "dst_index": coo.col.astype(np.int64),
            "src_body_id": body_ids[coo.row],
            "dst_body_id": body_ids[coo.col],
            "raw_weight": coo.data,
        }
    )
    if processed is not None:
        p = processed.tocsr()
        p.sort_indices()
        m = matrix.tocsr()
        m.sort_indices()
        frame["processed_weight"] = p.tocoo().data if (
            np.array_equal(p.indptr, m.indptr) and np.array_equal(p.indices, m.indices)
        ) else np.nan
    return frame


def build_node_metadata(
    annotations: pd.DataFrame, mapping: NeuronIdMapping, schema: Schema = DEFAULT_SCHEMA
) -> pd.DataFrame:
    """Scalar annotation columns aligned to graph indices.

    List-valued columns (e.g. ``somaLocation``) are omitted from the CSV.
    Neurons absent from the annotations (possible with policy 'extend') get
    empty metadata and ``in_annotations == False``.
    """
    scalar = _hashable_columns(annotations)
    meta = annotations[scalar].set_index(schema.neuron_id).reindex(mapping.ids)
    for col in meta.columns:
        if isinstance(meta[col].dtype, pd.CategoricalDtype):
            meta[col] = meta[col].astype(object)
    meta.insert(0, "in_annotations", np.isin(mapping.ids, annotations[schema.neuron_id].to_numpy()))
    meta.insert(0, "graph_index", np.arange(mapping.n))
    meta.index.name = "body_id"
    return meta.reset_index()


def save_processed_data(
    output_dir: Path,
    mapping: NeuronIdMapping,
    node_metadata: pd.DataFrame,
    connectome: sp.csr_matrix,
    subgraph: Optional[SubgraphResult] = None,
    processed_subgraph: Optional[sp.csr_matrix] = None,
    export_full_edge_csv: bool = False,
) -> dict[str, str]:
    """Write artifacts and return ``{name: path}``.

    ``connectome_edges.csv`` (~1.5e8 rows, multiple GB, fully redundant with
    ``connectome_graph.npz``) is only written when ``export_full_edge_csv``.
    Load graphs with ``scipy.sparse.load_npz``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["neuron_id_mapping"] = out / "neuron_id_mapping.csv"
    mapping.to_frame().to_csv(paths["neuron_id_mapping"], index=False)
    paths["node_metadata"] = out / "node_metadata.csv"
    node_metadata.to_csv(paths["node_metadata"], index=False)
    paths["connectome_graph"] = out / "connectome_graph.npz"
    sp.save_npz(paths["connectome_graph"], connectome, compressed=True)

    if export_full_edge_csv:
        paths["connectome_edges"] = out / "connectome_edges.csv"
        coo = connectome.tocoo()
        chunk = 5_000_000
        for i, start in enumerate(range(0, coo.nnz, chunk)):
            sl = slice(start, start + chunk)
            pd.DataFrame(
                {
                    "src_index": coo.row[sl],
                    "dst_index": coo.col[sl],
                    "src_body_id": mapping.ids[coo.row[sl]],
                    "dst_body_id": mapping.ids[coo.col[sl]],
                    "weight": coo.data[sl],
                }
            ).to_csv(paths["connectome_edges"], mode="w" if i == 0 else "a",
                     header=(i == 0), index=False)

    if subgraph is not None:
        sub_ids = mapping.ids[subgraph.node_indices]
        paths["subgraph_node_mapping"] = out / "subgraph_node_mapping.csv"
        pd.DataFrame(
            {
                "subgraph_index": np.arange(len(sub_ids)),
                "parent_graph_index": subgraph.node_indices,
                "body_id": sub_ids,
            }
        ).to_csv(paths["subgraph_node_mapping"], index=False)
        paths["subgraph_edges"] = out / "subgraph_edges.csv"
        edges_to_frame(subgraph.matrix, sub_ids, processed_subgraph).to_csv(
            paths["subgraph_edges"], index=False
        )
        paths["subgraph_graph_raw"] = out / "subgraph_graph_raw.npz"
        sp.save_npz(paths["subgraph_graph_raw"], subgraph.matrix, compressed=True)
        if processed_subgraph is not None:
            paths["subgraph_graph"] = out / "subgraph_graph.npz"
            sp.save_npz(paths["subgraph_graph"], processed_subgraph, compressed=True)
    return {k: str(v) for k, v in paths.items()}


def generate_processing_report(
    config: ProcessingConfig,
    input_info: dict[str, Any],
    inspection: dict[str, Any],
    validation: dict[str, Any],
    mapping: NeuronIdMapping,
    build_stats: dict[str, Any],
    graph_stats: dict[str, Any],
    subgraph: Optional[SubgraphResult],
    weight_info: Optional[dict[str, Any]],
    artifacts: dict[str, str],
    warnings_list: list[str],
) -> dict[str, Any]:
    """Assemble the JSON-serialisable processing report."""
    cfg = asdict(config)
    return {
        "config": cfg,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "inputs": input_info,
        "inspection": inspection,
        "validation": validation,
        "mapping": {"strategy": mapping.strategy, "n_nodes": mapping.n,
                    "ordering": "ascending biological ID"},
        "graph_construction": {
            **build_stats,
            "matrix_orientation": "A[source_index, target_index] = value (CSR)",
            "source_column": DEFAULT_SCHEMA.source,
            "target_column": DEFAULT_SCHEMA.target,
            "weight_column": DEFAULT_SCHEMA.weight,
            "weight_semantics": "UNVERIFIED: assumed non-negative integer connection "
                                "strength (believed synapse count); not calibrated.",
            "direction_semantics": "UNVERIFIED: assumed body_pre -> body_post = signal flow.",
        },
        "graph_stats": graph_stats,
        "subgraph": None if subgraph is None else {
            "selection": {
                "strategy": config.selection_strategy,
                "node_budget": config.node_budget,
                "filters": config.filters,
                "seed": config.seed,
                "biological_claim": "None. Selection is generic; it is not a "
                                    "sensory-to-motor pathway unless built from "
                                    "validated annotations.",
            },
            **subgraph.stats,
        },
        "weight_transform": weight_info,
        "artifacts": artifacts,
        "warnings": warnings_list,
    }


def write_report(report: dict[str, Any], path: Path) -> Path:
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

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=default))
    return path


# --------------------------------------------------------------------------- #
# Pipeline / CLI
# --------------------------------------------------------------------------- #
def _read_id_file(path: Path) -> list[int]:
    """Read integer IDs from a text/CSV file (one per line, or a 'body_id' column)."""
    path = Path(path)
    df = pd.read_csv(path, header=None if path.suffix == ".txt" else "infer")
    col = "body_id" if "body_id" in df.columns else df.columns[0]
    return [int(x) for x in df[col].dropna()]


def run_pipeline(config: ProcessingConfig) -> dict[str, Any]:
    """Run the full pipeline and return the report (also written to disk)."""
    warnings_list: list[str] = []
    annotations, edges, input_info = load_raw_data(
        config.data_dir, config.annotation_file, config.edge_file
    )
    inspection = inspect_data(annotations, edges)
    validation = validate_connectome(annotations, edges)
    if validation["endpoints_not_in_annotations"]["unique_any"]:
        warnings_list.append(
            f"{validation['endpoints_not_in_annotations']['unique_any']} edge endpoint IDs "
            "are not in the annotation table."
        )
    if validation["weights"]["n_zero"] or validation["weights"]["n_negative"]:
        warnings_list.append("Non-positive weights present; see validation.")
    warnings_list.append(input_info["version_note"])

    mapping = build_neuron_id_mapping(
        annotations, edges, DEFAULT_SCHEMA, config.unmapped_endpoint_policy
    )
    matrix, build_stats = build_sparse_connectome(
        edges, mapping, DEFAULT_SCHEMA, config.duplicate_policy,
        config.nonpositive_weight_policy, config.self_loop_policy,
    )
    del edges
    graph_stats = compute_graph_stats(matrix)
    logger.info("Graph: %d nodes, %d edges, density %.3e",
                graph_stats["n_nodes"], graph_stats["n_edges"], graph_stats["density"])

    node_ids = _read_id_file(config.subgraph_ids_file) if config.subgraph_ids_file else None
    seed_ids = _read_id_file(config.seed_ids_file) if config.seed_ids_file else None
    selected = select_nodes(
        matrix, mapping, annotations, DEFAULT_SCHEMA,
        node_ids=node_ids, filters=config.filters, seed_ids=seed_ids,
        budget=config.node_budget, strategy=config.selection_strategy, seed=config.seed,
    )
    subgraph = extract_subgraph(matrix, selected)
    logger.info("Subgraph: %d nodes (%.2f%%), %d edges (%.2f%%)",
                subgraph.stats["n_nodes"], 100 * subgraph.stats["fraction_nodes_retained"],
                subgraph.stats["n_edges"], 100 * subgraph.stats["fraction_edges_retained"])
    processed, weight_info = preprocess_edge_weights(subgraph.matrix, config.weight_transform)

    metadata = build_node_metadata(annotations, mapping)
    artifacts = save_processed_data(
        config.output_dir, mapping, metadata, matrix, subgraph, processed,
        config.export_full_edge_csv,
    )
    report = generate_processing_report(
        config, input_info, inspection, validation, mapping, build_stats,
        graph_stats, subgraph, weight_info, artifacts, warnings_list,
    )
    report_path = write_report(report, Path(config.output_dir) / "processing_report.json")
    report["artifacts"]["processing_report"] = str(report_path)
    write_report(report, report_path)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> ProcessingConfig:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: <data-dir>/processed")
    p.add_argument("--annotation-file", default=DEFAULT_ANNOTATION_FILE)
    p.add_argument("--edge-file", default=DEFAULT_EDGE_FILE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unmapped-endpoint-policy", choices=UNMAPPED_POLICIES, default="error")
    p.add_argument("--duplicate-policy", choices=DUPLICATE_POLICIES, default="error")
    p.add_argument("--nonpositive-weight-policy", choices=NONPOSITIVE_POLICIES, default="error")
    p.add_argument("--self-loop-policy", choices=SELF_LOOP_POLICIES, default="keep")
    p.add_argument("--weight-transform", choices=WEIGHT_TRANSFORMS, default="raw")
    p.add_argument("--node-budget", type=int, default=30000,
                   help="max subgraph size; 0 = no limit")
    p.add_argument("--selection-strategy", choices=SELECTION_STRATEGIES, default="top_degree")
    p.add_argument("--subgraph-ids-file", type=Path, default=None)
    p.add_argument("--seed-ids-file", type=Path, default=None)
    p.add_argument("--filter", dest="filters", action="append", default=[],
                   help="annotation filter 'column=v1,v2' (repeatable)")
    p.add_argument("--export-full-edge-csv", action="store_true")
    a = p.parse_args(argv)
    return ProcessingConfig(
        data_dir=a.data_dir,
        output_dir=a.output_dir or a.data_dir / "processed",
        annotation_file=a.annotation_file,
        edge_file=a.edge_file,
        seed=a.seed,
        unmapped_endpoint_policy=a.unmapped_endpoint_policy,
        duplicate_policy=a.duplicate_policy,
        nonpositive_weight_policy=a.nonpositive_weight_policy,
        self_loop_policy=a.self_loop_policy,
        weight_transform=a.weight_transform,
        node_budget=a.node_budget or None,
        selection_strategy=a.selection_strategy,
        subgraph_ids_file=a.subgraph_ids_file,
        seed_ids_file=a.seed_ids_file,
        filters=a.filters,
        export_full_edge_csv=a.export_full_edge_csv,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_pipeline(parse_args(argv))


if __name__ == "__main__":
    main()
