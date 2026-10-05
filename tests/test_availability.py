import json

import numpy as np
import pytest
from experiment_fixtures import make_batch

from rfm_structure.data import load_availability_policy
from rfm_structure.validation import (
    available_context,
    real_query_batch,
    validate_batches,
)


def policy():
    return {"version": "test", "tables": [
        {"table": "query:Test", "start": 0, "stop": 1, "time_col": "date", "label_column": 0, "horizon_seconds": 7},
        {"table": "support:Train", "start": 10000, "stop": 10001, "time_col": "date", "label_column": 0, "horizon_seconds": 7},
        {"table": "static:Db", "start": 20000, "stop": 20001, "time_col": None, "label_column": None, "horizon_seconds": 0},
        {"table": "event:Db", "start": 30000, "stop": 30001, "time_col": "date", "label_column": None, "horizon_seconds": 0},
    ]}


def batch():
    raw = make_batch([0])
    raw["node_idxs"][0, :5] = [0, 10000, 10000, 20000, 30000]
    raw["timestamps"][0, :5] = [10, 5, 5, np.iinfo(np.int32).min, 11]
    raw["is_padding"][0, :5] = False
    raw["is_task_nodes"][0, :5] = [True, True, True, False, False]
    raw["col_name_idxs"][0, :5] = [0, 0, 1, 2, 3]
    raw["f2p_nbr_idxs"][0, :5, 0] = [-1, 20000, 20000, -1, -1]
    raw["f2p_rel_idxs"][0, :5, 0] = [-1, 7, 7, -1, -1]
    return raw


def test_common_filter_masks_future_and_unclosed_labels_without_refill_or_mutation():
    raw = batch()
    original, _ = real_query_batch(raw)
    result, counts = available_context(original, policy())

    assert result["is_padding"][0, :5].tolist() == [False, True, False, False, True]
    assert counts["removed_future_tokens"].tolist() == [1]
    assert counts["removed_unmatured_label_tokens"].tolist() == [1]
    assert counts["original_token_counts"].tolist() == [5]
    assert counts["declared_timeless_tokens"].tolist() == [1]
    assert original["is_padding"][0, :5].tolist() == [False] * 5
    for key in original:
        if key != "is_padding":
            np.testing.assert_array_equal(result[key], original[key])


def test_declared_timeless_parents_are_not_unknown_and_mature_labels_are_retained():
    raw = batch()
    raw["timestamps"][0, 1:3] = 3  # Horizon end equals cutoff: label is available.
    report, exposure = validate_batches([raw], [0], [101], 48, [0], policy())

    assert report["status"] == "passed"
    assert report["base_temporal"]["unknown"] == 0
    assert report["base_temporal"]["declared_timeless"] == 1
    assert exposure["base"][0]["labeled_support_counts"].tolist() == [1]
    assert report["availability_totals"]["removed_unmatured_label_tokens"] == 0


@pytest.mark.parametrize("position", [0, 1, 4])
def test_missing_required_timestamps_fail_closed(position):
    raw = batch()
    raw["timestamps"][0, position] = np.iinfo(np.int32).min
    report, _ = validate_batches([raw], [0], [101], 48, [0], policy())
    assert report["status"] == "failed"
    assert "timestamp" in report["errors"][0]


def test_incidence_policy_reports_fanout_changes_without_waiving_hard_invariants():
    raw = make_batch([0])
    raw["node_idxs"][0, :6] = [0, 10, 10, 11, 12, 13]
    raw["is_padding"][0, :6] = False
    raw["timestamps"][0, :6] = [10, 5, 5, 5, 5, 5]
    raw["f2p_nbr_idxs"][0, :6, 0] = [-1, 12, 12, 13, -1, -1]
    raw["f2p_rel_idxs"][0, :6, 0] = [-1, 7, 7, 7, -1, -1]
    rules = {"tables": [{"table": "temporal", "start": 0, "stop": 30000,
                         "time_col": "date", "label_column": None, "horizon_seconds": 0}]}
    report, exposure = validate_batches([raw], [0], [101], 48, [1], rules)

    assert report["status"] == "passed"
    assert report["per_arm"]["a100_s101"]["neighbor_fanout_preserved"] is False
    assert report["per_arm"]["a100_s101"]["structural_error_count"] == 0
    assert exposure["a100_s101"][0]["fanout_nbr_changed_tokens"].sum() > 0
    raw["f2p_nbr_idxs"][0, 1:3, 0] = 10  # Self-link remains a blocking error.
    report, _ = validate_batches([raw], [0], [101], 48, [0], rules)
    assert report["status"] == "failed"


def test_policy_requires_manifest_metadata_and_uses_actual_forecast_horizon(tmp_path):
    pre = tmp_path / "pre" / "rel-f1"
    pre.mkdir(parents=True)
    raw = tmp_path / "raw"
    task_dir = raw / "tasks" / "driver-dnf"
    task_dir.mkdir(parents=True)
    (raw / "manifest.yaml").write_text(json.dumps({"name": "rel-f1", "tables": {"drivers": {"time_col": None}}}))
    (task_dir / "manifest.yaml").write_text(json.dumps({"kind": "forecast", "time_col": "date",
                                                       "target_col": "did_not_finish", "timedelta": "30 days"}))
    (pre / "meta.json").write_text(json.dumps({"source": str(raw), "tasks": [{"name": "driver-dnf"}]}))
    (pre / "table_info.json").write_text(json.dumps({
        "drivers:Db": {"node_idx_offset": 0, "num_nodes": 2},
        "driver-dnf:Test": {"node_idx_offset": 2, "num_nodes": 1},
    }))
    (pre / "column_index.json").write_text(json.dumps({"did_not_finish of driver-dnf": 7}))

    loaded = load_availability_policy(pre.parent)

    assert loaded["tables"][0]["time_col"] is None
    assert loaded["tables"][1]["horizon_seconds"] == 30 * 86400
    assert loaded["source_manifests"]["driver-dnf"]["sha256"]
    (raw / "manifest.yaml").write_text(json.dumps({"name": "rel-f1", "tables": {"drivers": {}}}))
    with pytest.raises(ValueError, match="explicit time-column"):
        load_availability_policy(pre.parent)
