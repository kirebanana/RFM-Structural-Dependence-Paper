"""Visible-label source selection is stricter than non-query task selection."""
import numpy as np
import pytest

from rfm_structure.rewiring import assert_structural_invariants, rewire_f2p_nbr
from rfm_structure.validation import (
    rewire_and_validate_batch,
    visible_support_source_mask,
)


def fixture():
    nodes = np.array([[100, 101, 101, 102, 102, 103, 103, 104, 104,
                       201, 202, 203, 204, 205, -1]])
    padding = np.zeros_like(nodes, dtype=bool)
    padding[0, [6, 8, 14]] = True  # Labels of 103/104 are hidden, but dates remain.
    targets = np.zeros_like(padding)
    targets[0, 0] = True
    task_nodes = np.zeros_like(padding)
    task_nodes[0, :9] = True
    cols = np.array([[7, 8, 7, 8, 7, 8, 7, 8, 7, 9, 9, 9, 9, 9, -1]])
    parents = np.full((*nodes.shape, 2), -1, dtype=np.int64)
    relations = parents.copy()
    parents[0, :9, 0] = [201, 202, 202, 203, 203, 204, 204, 205, 205]
    relations[0, :9, 0] = 7
    parents[0, 1:5, 1] = [203, 203, 204, 204]
    relations[0, 1:5, 1] = 8  # Another relation must not change on selected rows.
    timestamps = np.full_like(nodes, 2)
    timestamps[0, 0] = 10
    return {"node_idxs": nodes, "is_padding": padding, "is_targets": targets,
            "is_task_nodes": task_nodes, "col_name_idxs": cols,
            "number_values": np.zeros((*nodes.shape, 1)), "timestamps": timestamps,
            "f2p_nbr_idxs": parents, "f2p_rel_idxs": relations}


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_support_only_permutation_preserves_query_hidden_rows_other_relations_and_content(seed):
    base = fixture()
    mask = visible_support_source_mask(base)
    assert np.flatnonzero(mask[0]).tolist() == [1, 2, 3, 4]
    evidence = rewire_and_validate_batch(base, seed, 1, {"tables": []}, [7], source_mask=mask)
    assert not evidence["errors"]
    expected = base["f2p_nbr_idxs"].copy()
    expected[0, 1:5, 0] = [203, 203, 202, 202]
    np.testing.assert_array_equal(evidence["batch"]["f2p_nbr_idxs"], expected)
    assert evidence["totals"]["changed_edges"] == 2  # Edge instances, not four cells.
    assert evidence["totals"]["max_changed_edges"] == 2
    for key in base:
        if key != "f2p_nbr_idxs":
            np.testing.assert_array_equal(evidence["batch"][key], base[key])
    assert not assert_structural_invariants(base["f2p_nbr_idxs"], expected, base["f2p_rel_idxs"],
                                            base["node_idxs"], base["is_padding"], enforce_fanout=False)


@pytest.mark.parametrize("visible", [0, 1])
def test_too_few_visible_support_rows_do_not_exchange_with_hidden_rows(visible):
    base = fixture()
    base["is_padding"][0, [2, 4] if visible == 0 else [4]] = True
    mask = visible_support_source_mask(base)
    result = rewire_and_validate_batch(base, 101, 1, {"tables": []}, [7], source_mask=mask)
    np.testing.assert_array_equal(result["batch"]["f2p_nbr_idxs"], base["f2p_nbr_idxs"])
    assert result["totals"]["changed_edges"] == result["totals"]["max_changed_edges"] == 0


def test_duplicate_visible_parent_id_can_be_unchangeable_even_with_two_support_rows():
    base = fixture()
    base["f2p_nbr_idxs"][0, 3:5, 0] = 202
    result = rewire_and_validate_batch(base, 101, 1, {"tables": []}, [7], source_mask=visible_support_source_mask(base))
    assert result["totals"]["max_changed_edges"] == 0
    np.testing.assert_array_equal(result["batch"]["f2p_nbr_idxs"], base["f2p_nbr_idxs"])


def test_nonfinite_support_label_is_not_selected_and_cellwise_masks_are_rejected():
    base = fixture()
    base["number_values"][0, 2, 0] = np.nan
    mask = visible_support_source_mask(base)
    assert not mask[0, 1:3].any()
    mask[0, 3] = False  # Contradicts the label cell of source 102.
    with pytest.raises(ValueError, match="disagrees across cells"):
        rewire_f2p_nbr(base["f2p_nbr_idxs"], base["f2p_rel_idxs"], base["node_idxs"],
                       base["is_padding"], 101, source_mask=mask)


def test_visible_support_mask_does_not_depend_on_label_value():
    base = fixture()
    before = visible_support_source_mask(base)
    base["number_values"][0, [2, 4], 0] = [-2.7, 0.37]
    np.testing.assert_array_equal(visible_support_source_mask(base), before)
