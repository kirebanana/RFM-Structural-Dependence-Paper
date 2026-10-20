import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiment_fixtures import cycle_fixture, make_batch

from rfm_structure.relation_controls import relation_groups
from rfm_structure.rewiring import assert_structural_invariants, rewire_f2p_nbr
from rfm_structure.validation import real_query_batch, rewire_and_validate_batch


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_targeting_keeps_assignment_stream_and_non_target_relations_fixed(seed):
    nbr, rel, nodes, padding = cycle_fixture()
    nbr = np.repeat(nbr, 2, axis=2)
    rel = np.repeat(rel, 2, axis=2)
    rel[:, :6, 1] = 8
    full = rewire_f2p_nbr(nbr, rel, nodes, padding, seed)
    targeted = rewire_f2p_nbr(nbr, rel, nodes, padding, seed, relation_ids=[8])
    np.testing.assert_array_equal(targeted[:, :, 0], nbr[:, :, 0])
    np.testing.assert_array_equal(targeted[:, :, 1], full[:, :, 1])
    assert not assert_structural_invariants(nbr, targeted, rel, nodes, padding, enforce_fanout=False)
    np.testing.assert_array_equal(targeted, rewire_f2p_nbr(nbr, rel, nodes, padding, seed, relation_ids=[8]))
    np.testing.assert_array_equal(nbr, rewire_f2p_nbr(nbr, rel, nodes, padding, seed, relation_ids=[]))


def test_targeted_exposure_accounts_all_eligible_but_only_targeted_maximum():
    raw = make_batch([0])
    base, _ = real_query_batch(raw)
    evidence = rewire_and_validate_batch(base, 101, relation_ids=[])
    assert evidence["totals"]["eligible_edges"] == 1
    assert evidence["totals"]["changed_edges"] == 0
    assert evidence["totals"]["max_changed_edges"] == 0
    assert not evidence["errors"]


def test_groups_use_relation_semantics_not_numeric_slot_positions():
    registry = {
        42: {"child_table": "driver-dnf", "parent_table": "drivers"},
        10: {"child_table": "results", "parent_table": "drivers"},
        11: {"child_table": "results", "parent_table": "constructors"},
        12: {"child_table": "results", "parent_table": "races"},
        13: {"child_table": "races", "parent_table": "circuits"},
    }
    groups = relation_groups(registry)
    assert groups["task_driver"] == [42]
    assert groups["database_all"] == [10, 11, 12, 13]
    assert groups["database_driver"] == [10]
    assert groups["race_circuit"] == [13]


def test_relation_wrapper_runs_targeted_parents_and_keeps_others_clean():
    from experiment_fixtures import FakeModel, runner_functions

    from rfm_structure.validation import batch_fingerprint

    functions = runner_functions()
    raw = make_batch([0])
    raw["node_idxs"][0, :12] = np.arange(12)
    raw["is_padding"][0, :12] = False
    raw["timestamps"][0, :12] = 5
    raw["timestamps"][0, 0] = 10
    raw["f2p_nbr_idxs"][0, :6, 0] = np.arange(6, 12)
    raw["f2p_rel_idxs"][0, :6, 0] = [7, 7, 7, 8, 8, 8]
    real, _ = real_query_batch(raw)
    holder = SimpleNamespace(batch_mask=raw.pop("batch_mask"), batch=None, batch_index=0,
                             forward_calls=0, fingerprints=[batch_fingerprint(real)])

    class InspectingModel(FakeModel):
        def predict(self, batch, *args):
            if self.calls:
                np.testing.assert_array_equal(batch["f2p_nbr_idxs"][0, :3, 0], np.arange(6, 9))
                assert np.any(batch["f2p_nbr_idxs"][0, 3:6, 0] != np.arange(9, 12))
            return super().predict(batch, *args)

    model = InspectingModel()
    functions["BaseWrap"](model, holder, 101, 8).predict(raw, [48], "cpu", None, True)
    wrapper = functions["RewireWrap"](model, 101, 1, holder, 8, [8])
    wrapper.predict(raw, [48], "cpu", None, True)
    assert wrapper.stats["changed_edges"] == wrapper.stats["max_changed_edges"] == 3


def test_cached_clean_predictions_require_no_model_forward(monkeypatch):
    from rfm_structure.validation import batch_fingerprint

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments"))
    functions = runpy.run_path(str(Path(__file__).parents[1] / "experiments/run_relation_controls.py"))
    raw = make_batch([0, 1], slots=3)
    real, _ = real_query_batch(raw)
    holder = SimpleNamespace(batch_mask=raw.pop("batch_mask"), batch=None, batch_index=0,
                             forward_calls=0, fingerprints=[batch_fingerprint(real)])
    cache = {"node_idxs": np.array([1, 0]), "preds": np.array([0.4, 0.2])}
    wrapper = functions["CachedClean"](None, holder, cache)
    scores = wrapper.predict(raw, [48], "cpu", None, True)[48]
    np.testing.assert_allclose(scores.numpy(), [0.2, 0.4, 0])
    assert holder.forward_calls == 0


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_query_cycle_partition_is_disjoint_and_reconstructs_full_matching(seed):
    nbr, rel, nodes, padding = cycle_fixture(repeat=2)
    full = rewire_f2p_nbr(nbr, rel, nodes, padding, seed)
    query = rewire_f2p_nbr(nbr, rel, nodes, padding, seed,
                           cycle_mode="touching_query", query_node_ids=np.array([0]))
    other = rewire_f2p_nbr(nbr, rel, nodes, padding, seed,
                           cycle_mode="excluding_query", query_node_ids=np.array([0]))
    assert not np.any((query != nbr) & (other != nbr))
    np.testing.assert_array_equal(np.where(query != nbr, query, other), full)
    np.testing.assert_array_equal(other[:, :2], nbr[:, :2])
    np.testing.assert_array_equal(query[:, :2], full[:, :2])
    for output in (query, other):
        assert not assert_structural_invariants(nbr, output, rel, nodes, padding, enforce_fanout=False)
