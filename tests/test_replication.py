import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiment_fixtures import make_batch

from rfm_structure.exploratory import clustered_auc_delta
from rfm_structure.replication import (
    MINILM_REVISION,
    MINILM_WEIGHTS_SHA256,
    SelectedContextLoader,
    replication_groups,
    selected_query_ids,
    validate_embedding_scope,
)


def test_frozen_selection_is_unique_deterministic_and_independent_of_labels(tmp_path):
    directory = tmp_path / "example"
    directory.mkdir()
    (directory / "table_info.json").write_text(json.dumps({"forecast:Test": {"num_nodes": 1000, "node_idx_offset": 7000}}))
    task = SimpleNamespace(db_name="example", table_name="forecast")
    ids = selected_query_ids(tmp_path, task)
    np.testing.assert_array_equal(ids, selected_query_ids(tmp_path, task))
    assert len(set(ids)) == 512
    assert ids.min() >= 7000 and ids.max() < 8000


def test_metadata_groups_generalize_to_study_and_paper_entities():
    registry = {
        1: {"child_table": "study-outcome", "parent_table": "studies"},
        2: {"child_table": "outcomes", "parent_table": "studies"},
        3: {"child_table": "outcome_analyses", "parent_table": "outcomes"},
    }
    assert replication_groups(registry, "study-outcome", "studies", {"studies", "outcomes", "outcome_analyses"}) == {
        "task_entity": [1], "database": [2, 3],
    }


def test_explicit_query_loader_rejects_sampler_retries_that_change_target():
    dataset = SimpleNamespace(
        sampler=SimpleNamespace(batch_for_nodes_py=lambda *_: make_batch([0, 1])),
        _process_batch=lambda batch: batch,
    )
    loader = SelectedContextLoader(dataset, [0, 1], 48)
    assert len(loader) == 1
    assert next(iter(loader))["node_idxs"].shape[0] == 2
    dataset.sampler.batch_for_nodes_py = lambda *_: make_batch([0, 0])
    with pytest.raises(ValueError, match="exactly the frozen selected queries"):
        next(iter(loader))


def test_single_date_does_not_produce_a_spurious_zero_width_temporal_interval():
    labels = np.tile([0, 1], 8)
    scores = np.arange(16)
    result = clustered_auc_delta(labels, scores, scores[::-1][None, :], np.zeros(16), np.arange(16), replicates=100)
    assert result["date_percentile_95"] is None
    assert result["date_status"] == "single_cluster_not_estimable"
    assert result["crossed_status"] == "only_one_varying_cluster_axis"


def test_seed_paired_reference_contrast_uses_mean_auroc_not_auroc_of_mean_scores():
    from sklearn.metrics import roc_auc_score

    labels = np.array([0, 1, 0, 1, 0, 1])
    reference = np.array([[0, 1, 2, 3, 4, 5], [10, 2, 3, 4, 5, 6]])
    arms = reference[:, ::-1]
    result = clustered_auc_delta(labels, reference, arms, [0, 0, 1, 1, 2, 2], np.arange(6), replicates=100)
    expected = np.mean([roc_auc_score(labels, s) for s in arms]) - np.mean([roc_auc_score(labels, s) for s in reference])
    assert result["point"] == pytest.approx(expected)


def test_scoped_embeddings_reject_a_different_query_panel_before_sampling(tmp_path):
    pytest.importorskip("rt")
    from rfm_structure.replication import replication_evaluator

    directory = tmp_path / "example"
    directory.mkdir()
    (directory / "embedding_scope.json").write_text(json.dumps({
        "status": "complete", "query_ids": [0, 1], "context_fingerprints": {"48": "test"},
    }))
    with pytest.raises(ValueError, match="coverage does not match"):
        replication_evaluator(tmp_path, SimpleNamespace(db_name="example"), [1, 0], 48)


def scoped_embedding_fixture(tmp_path):
    from rfm_structure.provenance import file_identity

    directory = tmp_path / "example"
    directory.mkdir()
    path = directory / "text_emb_all-MiniLM-L12-v2.bin"
    np.zeros((2, 384), dtype=np.uint16).tofile(path)
    scope = {"status": "complete", "encoder_revision": MINILM_REVISION,
             "encoder_dtype": "float32", "stored_dtype": "bfloat16",
             "encoder_weights": {"sha256": MINILM_WEIGHTS_SHA256},
             "embedding_file": {**file_identity(path), "path": "/old/machine/embedding.bin"},
             "query_ids": [0, 1], "context_fingerprints": {"48": "test"}}
    return path, scope


def test_scoped_embedding_identity_checks_current_file_even_when_wrong_vectors_are_finite(tmp_path):
    path, scope = scoped_embedding_fixture(tmp_path)
    validate_embedding_scope(tmp_path, "example", scope)
    # BF16 1.0: still finite, same size, different vectors.
    np.full((2, 384), 0x3F80, dtype=np.uint16).tofile(path)
    vectors = (np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    assert np.isfinite(vectors).all()
    with pytest.raises(ValueError, match="file identity differs"):
        validate_embedding_scope(tmp_path, "example", scope)
    (path.parent / "embedding_scope.json").write_text(json.dumps(scope))
    pytest.importorskip("rt")
    from rfm_structure.replication import replication_evaluator

    with pytest.raises(ValueError, match="file identity differs"):
        replication_evaluator(tmp_path, SimpleNamespace(db_name="example"), [0, 1], 48)


@pytest.mark.parametrize("field,value", [("encoder_revision", "wrong"),
                                          ("encoder_dtype", "bfloat16"),
                                          ("stored_dtype", "float32"),
                                          ("encoder_weights", {"sha256": "wrong"})])
def test_scoped_embeddings_reject_wrong_encoder_protocol(tmp_path, field, value):
    _, scope = scoped_embedding_fixture(tmp_path)
    scope[field] = value
    with pytest.raises(ValueError, match="encoder provenance differs"):
        validate_embedding_scope(tmp_path, "example", scope)


def test_unscoped_fully_encoded_artifacts_do_not_require_scope_provenance(tmp_path):
    validate_embedding_scope(tmp_path, "example", None)


def test_frozen_runner_validation_rejects_nonfinite_embeddings_and_keeps_pairing(tmp_path, monkeypatch):
    from experiment_fixtures import make_evaluator

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments"))
    functions = runpy.run_path(str(Path(__file__).parents[1] / "experiments/run_database_replications.py"))
    raw = make_batch([2, 0, 1])
    raw["number_values"][:, 0, 0] = [0, 1, 0]
    raw["text_values"] = np.zeros((3, 48, 384), dtype=np.float32)
    raw["col_name_values"] = np.zeros((3, 48, 384), dtype=np.float32)
    ev = make_evaluator([raw], 3, count=3)
    policy = {"tables": [{"table": "known", "start": 0, "stop": 30000,
                          "time_col": "date", "label_column": None, "horizon_seconds": 0}]}
    valid = tmp_path / "valid"
    valid.mkdir()
    report = functions["validate_panel"](ev, np.array([2, 0, 1]), {"task_entity": [7], "database": [8]}, policy, valid, None)
    assert report["status"] == "passed"
    assert report["target_node_idxs"] == [2, 0, 1]
    assert len(report["per_arm"]) == 7
    raw["text_values"][0, 0, 0] = np.nan
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    with pytest.raises(ValueError, match="unencoded/nonfinite"):
        functions["validate_panel"](ev, np.array([2, 0, 1]), {"task_entity": [7], "database": [8]}, policy, invalid, None)
    assert json.loads((invalid / "validation.json").read_text())["status"] == "failed"
