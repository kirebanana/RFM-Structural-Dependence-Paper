import numpy as np

from rfm_structure.rewiring import (
    _self_test,
    assert_structural_invariants,
    rewire_f2p_nbr,
    summarize_query_exposure,
)


def test_synthetic_fixture_passes():
    _self_test()


def test_seeds_produce_distinct_valid_rewirings():
    node_idxs = np.array(
        [[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]],
        dtype=np.int64,
    )
    rel = np.full((2, 6, 3), -1, dtype=np.int64)
    rel[:, :3, 0] = 100
    rel[:, :3, 1] = 200
    nbr = np.full((2, 6, 3), -1, dtype=np.int64)
    nbr[0, :3, 0] = [11, 12, 10]
    nbr[0, :3, 1] = [13, 14, 15]
    nbr[1, :3, 0] = [21, 22, 20]
    nbr[1, :3, 1] = [23, 24, 25]
    padding = np.zeros((2, 6), dtype=bool)

    outputs = []
    for seed in (101, 202, 303):
        stats = {}
        rewired = rewire_f2p_nbr(nbr, rel, node_idxs, padding, seed, stats)
        assert stats["eligible_edges"] == 12
        assert stats["changed_edges"] > 0
        assert stats.get("singleton_edges", 0) == 0
        assert not assert_structural_invariants(
            nbr, rewired, rel, node_idxs, padding
        )
        outputs.append(rewired.tobytes())

    # Some strata have a unique optimal matching; the fixture must therefore
    # require seed sensitivity, not three guaranteed distinct global outputs.
    assert len(set(outputs)) >= 2


def test_exposure_summary_records_query_and_relation_counts():
    node_idxs = np.array([[10, 11, 12, 13]], dtype=np.int64)
    rel = np.array([[[100], [100], [100], [-1]]], dtype=np.int64)
    nbr = np.array([[[12], [13], [10], [-1]]], dtype=np.int64)
    padding = np.array([[False, False, False, False]])
    targets = np.array([[True, False, False, False]])
    col_idxs = np.array([[7, 7, 7, 8]], dtype=np.int64)
    task_nodes = np.array([[True, True, True, False]])
    sequence_stats = []

    rewired = rewire_f2p_nbr(
        nbr,
        rel,
        node_idxs,
        padding,
        seed=101,
        sequence_stats=sequence_stats,
    )
    exposure = summarize_query_exposure(
        node_idxs,
        rel,
        padding,
        targets,
        col_idxs,
        task_nodes,
        sequence_stats,
    )

    assert rewired.shape == nbr.shape
    assert exposure["is_prediction_query"].tolist() == [True]
    assert exposure["target_node_idxs"].tolist() == [10]
    assert exposure["context_token_counts"].tolist() == [4]
    assert exposure["eligible_edges"].tolist() == [3]
    assert exposure["labeled_support_counts"].tolist() == [2]
    assert exposure["relation_query_indices"].tolist() == [0]
    assert exposure["relation_ids"].tolist() == [100]
    assert exposure["relation_eligible_edges"].tolist() == [3]
