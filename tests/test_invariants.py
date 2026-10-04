import numpy as np

from rfm_structure.rewiring import (
    assert_structural_invariants,
    attention_fanout_counts,
    attn_true_counts,
    effective_parent_set_sizes,
    temporal_parent_violations,
)


def test_attention_counts_ignore_absent_links():
    node_idxs = np.array([[1, 2, 3]], dtype=np.int64)
    padding = np.array([[False, False, False]])
    nbr = np.array([[[2, -1], [3, -1], [-1, -1]]], dtype=np.int64)
    assert attn_true_counts(node_idxs, nbr, padding) == (5, 2)
    assert effective_parent_set_sizes(nbr, node_idxs, padding).tolist() == [[1, 1, 0]]
    fanout = attention_fanout_counts(node_idxs, nbr, padding)
    assert fanout["feat"].tolist() == [[2, 2, 1]]
    assert fanout["nbr"].tolist() == [[0, 1, 1]]


def test_parent_multiset_check_handles_padding_and_detects_corruption():
    node_idxs = np.array([[10, 11, 12, 13, -1, -1]], dtype=np.int64)
    padding = np.array([[False, False, False, False, True, True]])
    rel = np.full((1, 6, 1), -1, dtype=np.int64)
    rel[0, :3, 0] = 100
    base = np.full((1, 6, 1), -1, dtype=np.int64)
    base[0, :3, 0] = [12, 13, 10]

    assert not assert_structural_invariants(base, base.copy(), rel, node_idxs, padding)

    corrupted = base.copy()
    corrupted[0, 0, 0] = 13
    errors = assert_structural_invariants(base, corrupted, rel, node_idxs, padding)
    assert "parent multiset changed for b=0 relation=100" in errors


def test_temporal_check_reports_known_future_parent_only():
    node_idxs = np.array([[10, 11, 12]], dtype=np.int64)
    padding = np.array([[False, False, False]])
    targets = np.array([[True, False, False]])
    timestamps = np.array([[10, 11, np.iinfo(np.int32).min]], dtype=np.int64)
    nbr = np.array([[[11], [-1], [-1]]], dtype=np.int64)

    errors = temporal_parent_violations(nbr, node_idxs, timestamps, padding, targets)

    assert errors == ["b=0 s=0 k=0: parent 11 timestamp 11 exceeds target timestamp 10"]


def test_repeated_source_cells_cannot_receive_inconsistent_fk_parents():
    nodes = np.array([[10, 10, 11, 11, 12, 13]])
    pad = np.zeros_like(nodes, dtype=bool)
    rel = np.array([[[7], [7], [7], [7], [-1], [-1]]])
    base = np.array([[[12], [12], [13], [13], [-1], [-1]]])
    new = np.array([[[12], [13], [12], [13], [-1], [-1]]])
    # Cell-weighted multisets and per-token key counts still match in this case.
    errors = assert_structural_invariants(base, new, rel, nodes, pad)
    assert any("inconsistent rewired FK copies" in error for error in errors)


def test_out_of_context_parent_slots_are_held_fixed_even_when_counts_match():
    nodes = np.array([[10, 11]])
    pad = np.zeros_like(nodes, dtype=bool)
    rel = np.array([[[7], [-1]]])
    base = np.array([[[99], [-1]]])
    new = np.array([[[98], [-1]]])
    errors = assert_structural_invariants(base, new, rel, nodes, pad)
    assert any("ineligible FK slot changed" in error for error in errors)
