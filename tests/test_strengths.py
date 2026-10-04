import numpy as np
import pytest
from experiment_fixtures import cycle_fixture

from rfm_structure.rewiring import assert_structural_invariants, rewire_f2p_nbr


@pytest.mark.parametrize("repeat", [1, 2])
def test_whole_cycle_strengths_identity_intermediate_full_and_invariants(repeat):
    nbr, rel, nodes, pad = cycle_fixture(repeat)
    outputs = []
    counts = []
    for alpha in (0, 0.5, 1):
        stats = {}
        new = rewire_f2p_nbr(nbr, rel, nodes, pad, 101, stats=stats, alpha=alpha)
        assert not assert_structural_invariants(nbr, new, rel, nodes, pad)
        assert stats["eligible_edges"] == 6  # Not repeated cell count.
        assert stats["max_changed_edges"] == 6
        counts.append(stats["changed_edges"])
        outputs.append(new)
        assert np.array_equal(new, rewire_f2p_nbr(nbr, rel, nodes, pad, 101, alpha=alpha))
    assert counts == [0, 2, 6]
    assert np.array_equal(outputs[0], nbr)
    assert np.array_equal(outputs[2], rewire_f2p_nbr(nbr, rel, nodes, pad, 101))


@pytest.mark.parametrize("alpha", [0, 0.5, 1])
def test_duplicate_parent_instances_and_padding_keep_stratum_multiset(alpha):
    nbr, rel, nodes, pad = cycle_fixture()
    nbr[0, :6, 0] = [6, 6, 7, 7, 8, 8]
    pad[0, 9:] = True
    new = rewire_f2p_nbr(nbr, rel, nodes, pad, 202, alpha=alpha)
    # Parent-instance multiset, including duplicates, remains exact.
    assert sorted(new[0, :6, 0]) == [6, 6, 7, 7, 8, 8]
    assert np.array_equal(new[pad], nbr[pad])
    assert not assert_structural_invariants(nbr, new, rel, nodes, pad)
    if alpha == 0:
        assert np.array_equal(new, nbr)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_strengths_are_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        rewire_f2p_nbr(*cycle_fixture(), 101, alpha=alpha)


def test_singleton_and_infeasible_strata_have_zero_maximum_change():
    nodes = np.array([[1, 2, 3]])
    rel = np.array([[[7, 8], [7, -1], [-1, -1]]])
    nbr = np.array([[[1, 3], [1, -1], [-1, -1]]])
    for alpha in (0, 0.5, 1):
        stats = {}
        seq = []
        new = rewire_f2p_nbr(
            nbr, rel, nodes, np.zeros_like(nodes, dtype=bool), 303,
            stats=stats, sequence_stats=seq, alpha=alpha,
        )
        assert np.array_equal(new, nbr)
        assert stats["eligible_edges"] == 3
        assert stats["unrewirable_edges"] == 2
        assert stats["singleton_edges"] == 1
        assert stats["max_changed_edges"] == stats["changed_edges"] == 0
        assert seq[0]["requested_strength"] == alpha


@pytest.mark.parametrize("seed,parents", [
    (101, [7, 6, 9, 8, 11, 10]),
    (202, [9, 11, 10, 6, 8, 7]),
    (303, [8, 10, 6, 11, 7, 9]),
])
def test_full_strength_matches_baseline_37cfe9d_golden_output(seed, parents):
    # Captured from the pre-strength baseline, rather than the new default path.
    output = rewire_f2p_nbr(*cycle_fixture(), seed, alpha=1)
    assert output[0, :6, 0].tolist() == parents
