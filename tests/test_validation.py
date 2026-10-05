import json
from types import SimpleNamespace

import numpy as np
import pytest
from experiment_fixtures import (
    make_batch,
    make_evaluator,
    runner_functions,
    write_metadata,
)

from rfm_structure import validation
from rfm_structure.rewiring import (
    rewire_f2p_nbr,
    summarize_query_exposure,
    summarize_temporal_status,
)
from rfm_structure.validation import (
    capture_evaluator_masks,
    configure_complete_evaluation,
    reconcile_exposure,
    validate_batches,
    write_exposure_npz,
)


def test_temporal_status_counts_unique_source_slots_and_unknown_reasons():
    missing = np.iinfo(np.int32).min
    nodes = np.array([[10, 11, 11, 12], [20, 21, 21, 22]])
    nbr = np.array([[[-1], [12], [12], [-1]], [[-1], [22], [22], [-1]]])
    timestamps = np.array([[10, 5, 5, 12], [missing, 5, 5, missing]])
    targets = np.array([[True, False, False, False]] * 2)
    report = summarize_temporal_status(
        nbr, nodes, timestamps, np.zeros_like(nodes, dtype=bool), targets
    )

    assert report["comparisons"] == 2
    assert report["checked"] == 1
    assert report["known_violations"] == 1
    assert report["unknown"] == 1
    assert report["missing_target"] == 1
    assert report["missing_parent"] == 1
    assert report["missing_both"] == 1
    assert len(report["errors"]) == 1


def test_complete_validation_traverses_702_queries_and_ignores_phantom_tail():
    batches = [make_batch(range(546)), make_batch(range(546, 702), slots=546)]
    report, exposures = validate_batches(batches, np.arange(702), [101, 202, 303], 48)

    assert report["complete"]
    assert report["status"] == "passed"
    assert report["batches_processed"] == 2
    assert report["phantom_rows"] == 390
    assert report["target_node_idxs"] == list(range(702))
    assert report["base_temporal"]["checked"] == 702
    for seed in (101, 202, 303):
        assert report["per_arm"][str(seed)]["totals"]["eligible_edges"] == 702
        assert sum(len(b["target_node_idxs"]) for b in exposures[seed]) == 702


@pytest.mark.parametrize("failure", ["lost", "duplicate", "unexpected", "malformed"])
def test_validation_fails_closed_on_missing_or_malformed_real_targets(failure):
    batch = make_batch([0, 1, 2], slots=4)
    if failure == "lost":
        batch["batch_mask"][1] = False
    elif failure == "duplicate":
        batch["node_idxs"][1, 0] = 0
    elif failure == "unexpected":
        batch["node_idxs"][1, 0] = 999
    else:
        batch["is_targets"][1] = False
    report, _ = validate_batches([batch], np.arange(3), [101], 48)
    assert not report["complete"]
    assert report["status"] == "failed"
    assert report["errors"]
    assert not report["inference_ready"]


@pytest.mark.parametrize("target_missing,parent_missing", [(True, False), (False, True), (True, True)])
def test_unknown_timestamp_comparisons_are_counted_and_require_review(target_missing, parent_missing):
    batch = make_batch([0])
    missing = np.iinfo(np.int32).min
    if target_missing:
        batch["timestamps"][0, 0] = missing
    if parent_missing:
        batch["timestamps"][0, 2] = missing
    report, exposures = validate_batches([batch], [0], [101], 48)

    assert report["complete"]
    assert report["status"] == "temporal_uncertainty"
    assert not report["inference_ready"]
    for counts in (report["base_temporal"], report["per_arm"]["101"]["temporal"]):
        assert counts["comparisons"] == 1
        assert counts["checked"] == 0
        assert counts["unknown"] == 1
        assert counts["missing_target"] == int(target_missing)
        assert counts["missing_parent"] == int(parent_missing)
        assert counts["missing_both"] == int(target_missing and parent_missing)
    assert exposures[101][0]["temporal_unknown"].tolist() == [1]


def test_known_base_future_parent_is_an_incomplete_validation_failure():
    batch = make_batch([0])
    batch["timestamps"][0, 2] = 11
    report, _ = validate_batches([batch], [0], [101], 48)
    assert report["status"] == "failed"
    assert not report["complete"]
    assert report["base_temporal"]["known_violations"] == 1
    assert "base temporal" in report["errors"][0]


def test_exposure_round_trip_keeps_evaluator_order_and_relation_offsets(tmp_path):
    report, exposures = validate_batches(
        [make_batch([2, 0]), make_batch([1], slots=4)], np.arange(3), [101], 48
    )
    assert report["complete"]
    path = tmp_path / "exposure.npz"
    write_exposure_npz(path, exposures[101])
    with np.load(path) as saved:
        assert saved["target_node_idxs"].tolist() == [2, 0, 1]
        assert saved["relation_query_indices"].tolist() == [0, 1, 2]
        assert saved["labeled_support_counts"].tolist() == [2, 2, 2]
        assert saved["context_token_counts"].tolist() == [3, 3, 3]
        assert saved["temporal_checked"].tolist() == [1, 1, 1]


def test_exposure_reconciliation_detects_query_and_aggregate_errors():
    report, exposures = validate_batches([make_batch([0, 1])], [0, 1], [101], 48)
    exposure = exposures[101][0]
    exposure["relation_eligible_edges"][:] = [0, 2]  # Same global sum, wrong query rows.
    with pytest.raises(ValueError, match="relation/query"):
        reconcile_exposure(exposure)
    exposure["relation_eligible_edges"][:] = [1, 1]
    totals = report["per_arm"]["101"]["totals"].copy()
    totals["eligible_edges"] = 3
    with pytest.raises(ValueError, match="query/aggregate"):
        reconcile_exposure(exposure, totals)


def test_infeasible_edges_reconcile_as_eligible_unchanged_unrewirable():
    nodes = np.array([[1, 2]])
    nbr = np.array([[[1], [1]]])
    rel = np.array([[[7], [7]]])
    pad = np.zeros_like(nodes, dtype=bool)
    sequence_stats = []
    stats = {}
    output = rewire_f2p_nbr(nbr, rel, nodes, pad, 101, stats, sequence_stats=sequence_stats)
    assert np.array_equal(output, nbr)
    assert stats["eligible_edges"] == 2
    assert stats["unrewirable_edges"] == 2
    assert stats["unrewirable_strata"] == 1
    exposure = summarize_query_exposure(
        nodes, rel, pad, np.array([[True, False]]), np.zeros_like(nodes),
        np.ones_like(pad), sequence_stats,
    )
    reconcile_exposure(exposure, stats)


@pytest.mark.parametrize("ctx,bs,phantoms", [(48, 5461, 4759), (128, 2048, 1346)])
def test_default_contexts_fit_all_targets_in_one_batch(ctx, bs, phantoms):
    ev = make_evaluator([make_batch(range(702), slots=bs, ctx=ctx)], bs, ctx=ctx)
    assert 2**18 // ctx == bs
    loader = configure_complete_evaluation(ev, 702)
    assert len(loader.dataset) == 1
    report, _ = validate_batches(loader, np.arange(702), [101], ctx)
    assert report["complete"]
    assert report["phantom_rows"] == phantoms


def test_floor_cap_is_avoided_without_changing_sampler_and_mask_survives_pop():
    ev = make_evaluator([make_batch(range(546)), make_batch(range(546, 702), slots=546)], 546)
    old_cap = min(len(ev.eval_loaders[ev.tasks[0]].dataset), max(1, ev.items_per_task // ev.eval_bs))
    assert old_cap == 1
    loader = configure_complete_evaluation(ev, 702)
    assert ev.items_per_task is None
    assert loader.dataset.rustler_dataset.num_items == 702
    assert ev.eval_bs == 546
    holder = SimpleNamespace()
    capture_evaluator_masks(ev, holder)
    real_counts = []
    for _ in range(len(loader.dataset)):
        batch = next(ev.eval_loader_iters[ev.tasks[0]])
        batch.pop("batch_mask")  # evaluate_raw's actual metadata removal.
        real_counts.append(int(holder.batch_mask.sum()))
    assert real_counts == [546, 156]


def test_dry_run_writes_compact_failure_report_without_rt_or_inference(tmp_path):
    write_metadata(tmp_path)
    batch = make_batch(range(702), slots=703)
    batch["batch_mask"][10] = False  # A lost intended item, not harmless overshoot.
    ev = make_evaluator([batch], 703)
    out = tmp_path / "validation"
    out.mkdir()
    assert runner_functions()["dry_run"](ev, tmp_path, [101], out, 24) == 1
    report = json.loads((out / "validation.json").read_text())
    assert report["status"] == "failed"
    assert report["missing_target_ids"] == [10]
    assert not report["complete"]
    assert not (out / "base_context.pt").exists()
    assert not list(out.glob("preds*.npz"))
    assert not list(out.glob("exposure*.npz"))
    with pytest.raises(FileExistsError):
        runner_functions()["dry_run"](ev, tmp_path, [101], out, 24)


def test_dry_run_saves_aligned_exposure_and_unknowns_block_model_loading(tmp_path):
    write_metadata(tmp_path)
    batch = make_batch(range(702), slots=703)
    batch["timestamps"][0, 2] = np.iinfo(np.int32).min
    ev = make_evaluator([batch], 703)
    out = tmp_path / "validation"
    out.mkdir()
    functions = runner_functions()
    with pytest.raises(ValueError, match="uncertain"):
        functions["run"](ev, tmp_path, [101], out, "cpu", 24)
    report = json.loads((out / "validation.json").read_text())
    assert report["status"] == "temporal_uncertainty"
    assert report["complete"]
    assert report["base_temporal"]["unknown"] == 1
    with np.load(out / "validation_exposure_base.npz") as exposure:
        assert exposure["target_node_idxs"].tolist() == list(range(702))
        assert exposure["base_temporal_unknown"].sum() == 1


def test_maximum_matching_retains_existing_80_percent_coverage_gate():
    batch = make_batch([0])
    batch["f2p_nbr_idxs"][0, 0, 0] = 20000
    batch["f2p_rel_idxs"][0, 0, 0] = 7
    report, _ = validate_batches([batch], [0], [101], 48)
    assert report["status"] == "failed"
    assert "80% gate" in report["errors"][0]
    assert report["per_arm"]["101"]["totals"]["eligible_edges"] == 2
    assert report["per_arm"]["101"]["totals"]["changed_edges"] == 0


def test_structural_failure_preserves_offending_batch_counts_and_boundaries():
    batch = make_batch([0])
    batch["node_idxs"][0, :6] = [0, 10, 10, 11, 12, 13]
    batch["is_padding"][0, :6] = False
    batch["is_task_nodes"][0, :6] = True
    batch["timestamps"][0, :6] = [10, 5, 5, 5, 5, 5]
    batch["f2p_nbr_idxs"][0, :6, 0] = [-1, 12, 12, 13, -1, -1]
    batch["f2p_rel_idxs"][0, :6, 0] = [-1, 7, 7, 7, -1, -1]
    report, _ = validate_batches([batch], [0], [101], 48)
    assert not report["complete"]
    assert report["context_token_counts"] == [6]
    assert report["base_temporal"]["comparisons"] == 2
    seed = report["per_arm"]["101"]
    assert seed["accounted_batches"] == 1
    assert seed["validated_batches"] == 0
    assert seed["totals"]["eligible_edges"] == 2
    assert seed["totals"]["changed_edges"] == 2
    assert "nbr attention fanout changed per token" in seed["structural_errors"]


def test_mutating_intervention_on_later_batch_is_rejected(monkeypatch):
    original = validation.rewire_f2p_nbr
    calls = 0

    def mutating_rewire(nbr, rel, node_idxs, padding, seed, **kwargs):
        nonlocal calls
        calls += 1
        output = original(nbr, rel, node_idxs, padding, seed, **kwargs)
        if calls == 2:
            node_idxs[0, 0] = 999
        return output

    monkeypatch.setattr(validation, "rewire_f2p_nbr", mutating_rewire)
    report, _ = validate_batches([make_batch([0]), make_batch([1])], [0, 1], [101], 48)
    assert not report["complete"]
    assert "held-fixed input changed: node_idxs" in report["errors"][0]
    assert report["per_arm"]["101"]["accounted_batches"] == 1


def test_inconsistent_repeated_timestamps_fail_even_with_unknown_target_time():
    missing = np.iinfo(np.int32).min
    nodes = np.array([[0, 1, 1, 2]])
    timestamps = np.array([[missing, 5, 6, 5]])
    nbr = np.array([[[-1, -1], [2, 2], [2, 2], [-1, -1]]])
    report = summarize_temporal_status(
        nbr, nodes, timestamps, np.zeros_like(nodes, dtype=bool),
        np.array([[True, False, False, False]]),
    )
    assert report["comparisons"] == 2  # Two slots, not four repeated cells.
    assert report["unknown"] == 2
    assert report["errors"] == ["b=0: node 1 has inconsistent timestamps"]


def test_phantom_selection_does_not_depend_on_target_flags():
    batch = make_batch([0], slots=2)
    batch["is_targets"][1, :2] = True  # A masked phantom must not become a real query.
    report, _ = validate_batches([batch], [0], [101], 48)
    assert report["complete"]
    assert report["observed_queries"] == 1
    assert report["phantom_rows"] == 1


def test_loader_exception_produces_partial_coverage_report():
    def failing_loader():
        yield make_batch([0])
        raise RuntimeError("sampler failure")

    report, _ = validate_batches(failing_loader(), [0, 1], [101], 48)
    assert report["status"] == "failed"
    assert report["observed_queries"] == 1
    assert report["missing_target_ids"] == [1]
    assert "sampler failure" in report["errors"][0]


def test_sampler_item_count_and_metadata_range_are_independent_gates(tmp_path):
    from rfm_structure.data import load_expected_target_ids

    write_metadata(tmp_path)
    ev = make_evaluator([make_batch(range(701))], 702, count=701)
    assert len(load_expected_target_ids(tmp_path, ev.tasks[0], 702)) == 702
    with pytest.raises(ValueError, match="sampler must contain exactly 702"):
        configure_complete_evaluation(ev, 702)
    (tmp_path / "rel-f1/table_info.json").write_text(json.dumps({
        "driver-dnf:Test": {"num_nodes": 703, "node_idx_offset": 0},
    }))
    with pytest.raises(ValueError, match="task split must contain exactly 702"):
        load_expected_target_ids(tmp_path, ev.tasks[0], 702)


def test_complete_dry_run_does_not_load_rt_and_discloses_local_context(tmp_path):
    write_metadata(tmp_path)
    ev = make_evaluator([make_batch(range(702), slots=703)], 703)
    out = tmp_path / "validation"
    out.mkdir()
    assert runner_functions()["dry_run"](ev, tmp_path, [101, 202, 303], out, 24) == 0
    report = json.loads((out / "validation.json").read_text())
    assert report["complete"]
    assert report["local_ctx"] == 24
    assert report["status"] == "passed"
    assert report["inference_ready"]
    assert not list(out.glob("preds*.npz"))
    assert len(list(out.glob("validation_exposure_rw*.npz"))) == 3


def test_passed_validation_requires_informative_intervention_not_human_approval(tmp_path):
    write_metadata(tmp_path)
    ev = make_evaluator([make_batch(range(702), slots=703)], 703)
    out = tmp_path / "validation"
    out.mkdir()
    with pytest.raises(ValueError, match="uninformative"):
        runner_functions()["run"](ev, tmp_path, [101], out, "cpu", 24)
    report = json.loads((out / "validation.json").read_text())
    assert report["status"] == "passed"
    assert report["inference_ready"]
    assert not list(out.glob("preds*.npz"))


def test_exposure_writer_refuses_to_replace_existing_evidence(tmp_path):
    _, exposures = validate_batches([make_batch([0])], [0], [101], 48)
    path = tmp_path / "exposure.npz"
    write_exposure_npz(path, exposures[101])
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_exposure_npz(path, exposures[101])
    assert path.read_bytes() == original


def test_mismatched_relation_id_array_cannot_pass_exposure_reconciliation():
    _, exposures = validate_batches([make_batch([0])], [0], [101], 48)
    exposure = exposures[101][0]
    exposure["relation_ids"] = np.array([], dtype=np.int64)
    with pytest.raises(ValueError, match="relation IDs"):
        reconcile_exposure(exposure)


def test_corrupt_temporal_accounting_cannot_be_reported_as_passed(monkeypatch):
    original = validation.summarize_temporal_status

    def missing_comparison(*args, **kwargs):
        status = original(*args, **kwargs)
        status["checked"] = 0  # Lose a comparison without declaring it unknown.
        return status

    monkeypatch.setattr(validation, "summarize_temporal_status", missing_comparison)
    report, _ = validate_batches([make_batch([0])], [0], [101], 48)
    assert report["status"] == "failed"
    assert "temporal accounting" in report["errors"][0]


def test_label_and_labeled_support_rules_match_evaluator_for_repeated_cells():
    batch = make_batch([0])
    batch["node_idxs"][0, :6] = [0, 0, 10000, 10000, 20000, 30000]
    batch["is_padding"][0, :6] = False
    batch["is_task_nodes"][0, :6] = [True, True, True, True, False, True]
    batch["col_name_idxs"][0, :6] = [7, 7, 7, 8, 7, 7]
    batch["number_values"][0, :6, 0] = [-0.5, 50, 99, 99, 99, 99]
    batch["f2p_nbr_idxs"][:] = -1
    batch["f2p_rel_idxs"][:] = -1
    batch["timestamps"][0, :6] = [10, 10, 5, 5, 5, 5]
    report, exposures = validate_batches([batch], [0], [101], 48)
    assert report["status"] == "passed"
    assert report["labels"] == [-0.5]
    assert exposures[101][0]["labeled_support_counts"].tolist() == [2]
