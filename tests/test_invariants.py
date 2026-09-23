import numpy as np

from rfm_structure.rewiring import attn_true_counts, effective_parent_set_sizes


def test_attention_counts_ignore_absent_links():
    node_idxs = np.array([[1, 2, 3]], dtype=np.int64)
    padding = np.array([[False, False, False]])
    nbr = np.array([[[2, -1], [3, -1], [-1, -1]]], dtype=np.int64)
    assert attn_true_counts(node_idxs, nbr, padding) == (5, 2)
    assert effective_parent_set_sizes(nbr, node_idxs, padding).tolist() == [[1, 1, 0]]
