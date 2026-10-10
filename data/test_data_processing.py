"""Synthetic-data tests for data_processing.py (no real dataset required).

Run:  python -m pytest data/test_data_processing.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import data_processing as dp

S = dp.DEFAULT_SCHEMA


def make_annotations(ids):
    return pd.DataFrame(
        {
            S.neuron_id: np.asarray(ids, dtype=np.int64),
            "superclass": ["x"] * len(ids),
            "somaLocation": [np.array([1, 2, 3])] * len(ids),
        }
    )


def make_edges(rows):
    return pd.DataFrame(rows, columns=[S.source, S.target, S.weight]).astype(np.int64)


def test_direction_preserved_and_mapping_deterministic():
    ann = make_annotations([300, 100, 200])  # unsorted on purpose
    edges = make_edges([(100, 200, 5)])
    m1 = dp.build_neuron_id_mapping(ann, edges)
    m2 = dp.build_neuron_id_mapping(ann.sample(frac=1, random_state=1), edges)
    assert m1.ids.tolist() == m2.ids.tolist() == [100, 200, 300]
    assert m1.to_body_id(m1.to_index(np.array([300, 100]))).tolist() == [300, 100]
    A, _ = dp.build_sparse_connectome(edges, m1)
    assert A[0, 1] == 5 and A[1, 0] == 0  # source row, target column


def test_duplicate_policies():
    ann = make_annotations([1, 2])
    edges = make_edges([(1, 2, 3), (1, 2, 7), (2, 1, 1)])
    m = dp.build_neuron_id_mapping(ann, edges)
    with pytest.raises(ValueError, match="duplicate"):
        dp.build_sparse_connectome(edges, m, duplicate_policy="error")
    for policy, expected in (("sum", 10), ("max", 7), ("first", 3)):
        A, stats = dp.build_sparse_connectome(edges, m, duplicate_policy=policy)
        assert A[0, 1] == expected and A.nnz == 2
        assert stats["n_duplicate_pairs_removed"] == 1


def test_invalid_and_missing_endpoints_detected():
    ann = make_annotations([1, 2])
    edges = make_edges([(1, 99, 1)])
    report = dp.validate_connectome(ann, edges)
    assert report["endpoints_not_in_annotations"]["unique_any"] == 1
    with pytest.raises(ValueError, match="not in the annotation"):
        dp.build_neuron_id_mapping(ann, edges)
    m = dp.build_neuron_id_mapping(ann, edges, unmapped_endpoint_policy="drop_edges")
    A, stats = dp.build_sparse_connectome(edges, m)
    assert A.nnz == 0 and stats["n_dropped_unmapped_endpoint"] == 1
    assert dp.build_neuron_id_mapping(ann, edges, unmapped_endpoint_policy="extend").n == 3

    null_edges = pd.DataFrame({S.source: [1.0, np.nan], S.target: [2.0, 1.0], S.weight: [1, 1]})
    with pytest.raises(ValueError):
        dp.validate_connectome(ann, null_edges)
    with pytest.raises(ValueError, match="duplicated"):
        dp.validate_connectome(make_annotations([1, 1]), make_edges([(1, 1, 1)]))


def test_nonpositive_weights_not_silently_dropped():
    ann = make_annotations([1, 2])
    edges = make_edges([(1, 2, 0)])
    m = dp.build_neuron_id_mapping(ann, edges)
    with pytest.raises(ValueError, match="weight <= 0"):
        dp.build_sparse_connectome(edges, m)
    A, _ = dp.build_sparse_connectome(edges, m, nonpositive_weight_policy="drop")
    assert A.nnz == 0


def test_isolated_nodes_preserved_and_shape():
    ann = make_annotations([1, 2, 3, 4])
    edges = make_edges([(1, 2, 1)])
    m = dp.build_neuron_id_mapping(ann, edges)
    A, _ = dp.build_sparse_connectome(edges, m)
    assert A.shape == (4, 4) and sp.isspmatrix_csr(A)
    stats = dp.compute_graph_stats(A)
    assert stats["n_nodes"] == 4 and stats["n_isolated_nodes"] == 2


def test_self_loop_policy():
    ann = make_annotations([1, 2])
    edges = make_edges([(1, 1, 2), (1, 2, 1)])
    m = dp.build_neuron_id_mapping(ann, edges)
    assert dp.build_sparse_connectome(edges, m)[0].nnz == 2
    assert dp.build_sparse_connectome(edges, m, self_loop_policy="drop")[0].nnz == 1


def _random_graph(n=50, k=300, seed=0):
    rng = np.random.default_rng(seed)
    pairs = {(int(a), int(b)) for a, b in rng.integers(0, n, size=(k, 2))}
    rows = [(a + 1000, b + 1000, int(rng.integers(1, 20))) for a, b in sorted(pairs)]
    ann = make_annotations(list(range(1000, 1000 + n)))
    edges = make_edges(rows)
    m = dp.build_neuron_id_mapping(ann, edges)
    A, _ = dp.build_sparse_connectome(edges, m)
    return ann, edges, m, A


def test_subgraph_keeps_only_internal_edges():
    _, _, m, A = _random_graph()
    sel = np.array([3, 7, 8, 20, 21, 40])
    sub = dp.extract_subgraph(A, sel)
    dense = A.toarray()
    assert np.array_equal(sub.matrix.toarray(), dense[np.ix_(sel, sel)])
    assert sub.stats["n_nodes"] == 6
    assert 0 <= sub.stats["fraction_edges_retained"] <= 1


def test_selection_deterministic_and_budgeted():
    ann, _, m, A = _random_graph()
    for strat, kwargs in (("random", {}), ("top_degree", {}),
                          ("bfs", {"seed_ids": [1000]})):
        a = dp.select_nodes(A, m, ann, budget=10, strategy=strat, seed=1, **kwargs)
        b = dp.select_nodes(A, m, ann, budget=10, strategy=strat, seed=1, **kwargs)
        assert np.array_equal(a, b) and len(a) <= 10 and np.all(np.diff(a) > 0)
    ids = dp.select_nodes(A, m, ann, node_ids=[1001, 1002])
    assert ids.tolist() == [1, 2]
    assert len(dp.select_nodes(A, m, ann, filters=["superclass=x"])) == m.n
    with pytest.raises(KeyError):
        dp.select_nodes(A, m, ann, node_ids=[5])


def test_weight_preprocessing():
    A = sp.csr_matrix(np.array([[0, 1, 3], [0, 0, 0], [2, 0, 0]], dtype=np.int32))
    raw, _ = dp.preprocess_edge_weights(A, "raw")
    assert np.array_equal(raw.toarray(), A.toarray())
    log, _ = dp.preprocess_edge_weights(A, "log1p")
    assert np.allclose(log.data, np.log1p(A.data))
    row, info = dp.preprocess_edge_weights(A, "row_normalize")
    assert np.allclose(row.sum(axis=1).ravel(), [1, 0, 1])  # zero row stays zero
    assert info["n_zero_strength_nodes"] == 1
    col, _ = dp.preprocess_edge_weights(A, "col_normalize")
    assert np.allclose(col.sum(axis=0).ravel(), [1, 1, 1])
    assert A.data.tolist() == [1, 3, 2]  # input untouched


def test_save_and_reload_roundtrip(tmp_path):
    ann, _, m, A = _random_graph()
    sub = dp.extract_subgraph(A, np.arange(0, 30))
    processed, _ = dp.preprocess_edge_weights(sub.matrix, "log1p")
    meta = dp.build_node_metadata(ann, m)
    paths = dp.save_processed_data(tmp_path, m, meta, A, sub, processed, export_full_edge_csv=True)
    B = sp.load_npz(paths["connectome_graph"])
    assert B.shape == A.shape and (B != A).nnz == 0
    P = sp.load_npz(paths["subgraph_graph"])
    assert np.allclose(P.toarray(), processed.toarray())
    mp = pd.read_csv(paths["neuron_id_mapping"])
    assert mp["body_id"].tolist() == m.ids.tolist()
    edges_csv = pd.read_csv(paths["connectome_edges"])
    rebuilt = sp.csr_matrix(
        (edges_csv["weight"], (edges_csv["src_index"], edges_csv["dst_index"])), shape=A.shape
    )
    assert (rebuilt != A).nnz == 0
    sub_edges = pd.read_csv(paths["subgraph_edges"])
    assert len(sub_edges) == sub.matrix.nnz
    assert "somaLocation" not in pd.read_csv(paths["node_metadata"]).columns
    report = dp.generate_processing_report(
        dp.ProcessingConfig(), {}, {}, {}, m, {}, dp.compute_graph_stats(A), sub, None, paths, []
    )
    json.loads(dp.write_report(report, tmp_path / "r.json").read_text())


def test_no_dense_allocation_for_large_graph(monkeypatch):
    import tracemalloc

    def boom(*a, **k):
        raise AssertionError("dense conversion attempted")

    for name in ("toarray", "todense"):
        monkeypatch.setattr(sp.csr_matrix, name, boom)
        monkeypatch.setattr(sp.coo_matrix, name, boom)
    n = 1_000_000  # dense int32 would need ~4 TB
    ids = np.arange(n) + 10
    ann = pd.DataFrame({S.neuron_id: ids})
    edges = make_edges([(10, 11, 1), (11, 12, 2), (999_999, 10, 3)])
    tracemalloc.start()
    m = dp.build_neuron_id_mapping(ann, edges)
    A, _ = dp.build_sparse_connectome(edges, m)
    dp.compute_graph_stats(A)
    dp.extract_subgraph(A, np.array([0, 1, 2]))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert A.shape == (n, n) and A.nnz == 3
    assert peak < 200 * 1024 * 1024

