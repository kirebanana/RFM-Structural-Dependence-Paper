"""Portable follow-up evidence must reproduce without RT, raw data or weights."""
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
RELEASE = ROOT / "results/final/2026-10-19"
ANALYSIS = runpy.run_path(str(ROOT / "experiments/analyze_final_results.py"))
CURATION = runpy.run_path(str(ROOT / "experiments/curate_final_results.py"))


def test_final_evidence_reproduces_all_metrics_and_support_contrasts():
    result = ANALYSIS["analyze"](RELEASE)
    recorded = json.loads((RELEASE / "recalculated/analysis.json").read_text())
    assert result == recorded
    assert result["prediction_files"] == 110
    assert len(result["checkpoint_hashes"]) == 2


def test_integrity_check_rejects_missing_or_changed_artifacts(tmp_path):
    (tmp_path / "checksums.json").write_text(json.dumps({"sample": {"bytes": 3, "sha256": "wrong"}}))
    with pytest.raises(ValueError, match="missing artifact"):
        ANALYSIS["verify_checksums"](tmp_path)
    (tmp_path / "sample").write_text("bad")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ANALYSIS["verify_checksums"](tmp_path)


def test_lossless_curation_preserves_nan_dtype_and_original_identity(tmp_path):
    source, dest = tmp_path / "original.npz", tmp_path / "curated.npz"
    np.savez(source, values=np.array([1, np.nan], dtype=np.float32), ids=np.array([2, 3], dtype=np.int64))
    identity = CURATION["file_identity"](source)
    original, curated = CURATION["compress"](source, dest, identity["sha256"])
    assert original == identity
    assert curated["sha256"] == CURATION["file_identity"](dest)["sha256"]
    with pytest.raises(ValueError, match="original checksum mismatch"):
        CURATION["compress"](source, dest, "incorrect")


def test_portable_metadata_removes_machine_paths_but_preserves_hashes():
    assert CURATION["portable"]({"file": {"path": "/machine/data/nodes.rkyv", "sha256": "abc"}}) == {
        "file": {"path": "nodes.rkyv", "sha256": "abc"}}


def test_independent_support_classifier_counts_source_slots_not_repeated_cells(monkeypatch):
    from test_visible_support import fixture

    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    classifier = runpy.run_path(str(ROOT / "experiments/audit_support_links.py"))["classify_links"]
    base = fixture()
    rewired = base["f2p_nbr_idxs"].copy()
    rewired[0, 1:5, 0] = [203, 203, 202, 202]
    result, = classifier(base, rewired, 7)
    assert result["eligible"] == {"query": 1, "visible_label": 2, "hidden_label": 2}
    assert result["changed"] == {"query": 0, "visible_label": 2, "hidden_label": 0}
    assert result["visible_pool_distinct_parents"] == 2


def test_release_inventory_is_mandatory_and_exhaustive(tmp_path):
    with pytest.raises(ValueError, match="requires a complete"):
        ANALYSIS["verify_checksums"](tmp_path)
    (tmp_path / "checksums.json").write_text("{}")
    with pytest.raises(ValueError, match="empty release inventory"):
        ANALYSIS["verify_checksums"](tmp_path)


def refresh_inventory(root):
    import hashlib

    inventory = {}
    for path in root.rglob("*"):
        if path.is_file() and path != root / "checksums.json":
            content = path.read_bytes()
            inventory[str(path.relative_to(root))] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    (root / "checksums.json").write_text(json.dumps(inventory))


@pytest.fixture
def release_copy(tmp_path):
    root = tmp_path / "final/2026-10-19"
    shutil.copytree(RELEASE, root)
    primary = tmp_path / "preliminary/2026-10-05_validated"
    primary.parent.mkdir()
    primary.symlink_to(ROOT / "results/preliminary/2026-10-05_validated", target_is_directory=True)
    return root


@pytest.mark.parametrize("mutation, message", [
    ("missing_run", "six expected"), ("missing_arm", "arm/seed set"),
    ("extra_seed", "matching seeds"), ("missing_context", "two contexts"),
    ("registry_bytes", "authenticated preprocessing"), ("registry_group", "grouping differs"),
    ("f1_broadened_group", "F1 group membership"), ("f1_swapped_group", "F1 group membership"),
    ("f1_cycle_group", "F1 group membership"), ("f1_support_group", "F1 group membership"),
])
def test_coherently_reinventoried_incomplete_or_misdescribed_release_fails(release_copy, mutation, message):
    root = release_copy
    if mutation == "missing_run":
        path = root / "catalog.json"
        value = json.loads(path.read_text())
        del value["runs"]["labeled_support"]
    elif mutation == "missing_arm":
        path = root / "relations/ctx48_lctx24/manifest.json"
        value = json.loads(path.read_text())
        del value["arms"]["task_driver_s303"]
    elif mutation in ("extra_seed", "missing_context"):
        path = root / "relations/controls.json"
        value = json.loads(path.read_text())
        if mutation == "extra_seed":
            value["seeds"].append(404)
        else:
            del value["contexts"]["128"]
    elif mutation == "registry_bytes":
        path = root / "arxiv/relation_index.json"
        value = json.loads(path.read_text())
        value[next(iter(value))]["parent_table"] = "not_the_original_parent"
    elif mutation == "registry_group":
        path = root / "arxiv/replication.json"
        value = json.loads(path.read_text())
        value["groups"]["task_entity"], value["groups"]["database"] = value["groups"]["database"], value["groups"]["task_entity"]
    else:
        run = "task_cycles" if mutation == "f1_cycle_group" else "labeled_support" if mutation == "f1_support_group" else "relations"
        path = root / run / "controls.json"
        value = json.loads(path.read_text())
        if mutation == "f1_broadened_group":
            registry = json.loads((root / run / "ctx48_lctx24/manifest.json").read_text())["relation_index"]
            value["groups"]["database_driver"] = sorted(int(key) for key in registry)
        elif mutation == "f1_swapped_group":
            value["groups"]["database_driver"], value["groups"]["task_driver"] = value["groups"]["task_driver"], value["groups"]["database_driver"]
        else:
            value["groups"][next(iter(value["groups"]))] = [4]  # Ordinary result-row/driver relation, not a task link.
    path.write_text(json.dumps(value))
    refresh_inventory(root)  # A valid checksum inventory alone must not bypass protocol/semantic checks.
    with pytest.raises(ValueError, match=message):
        ANALYSIS["analyze"](root, bootstrap_replicates=2)


@pytest.mark.parametrize("run,field,message", [
    ("relations", "temporal_checked", "temporal accounting"),
    ("trial", "temporal_comparisons", "per-query temporal eligibility"),
    ("labeled_support", "base_temporal_valid", "temporal accounting"),
    ("relations", "availability_declared_timeless_tokens", "availability differs"),
    ("arxiv", "availability_removed_future_tokens", "availability differs"),
])
def test_every_validation_format_rejects_accounting_corruption(run, field, message):
    directory = RELEASE / run / "ctx48_lctx24"
    manifest = json.loads((directory / "manifest.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    arm_name = next(iter(manifest["arms"]))
    arm = manifest["arms"][arm_name]
    read_npz = ANALYSIS["read_npz"]
    exposure = read_npz(directory / arm["exposure"])
    clean_path = directory if "base" in manifest["arms"] else ROOT / "results/preliminary/2026-10-05_validated/ctx48_lctx24"
    clean_manifest = json.loads((clean_path / "manifest.json").read_text())
    clean = read_npz(clean_path / clean_manifest["arms"]["base"]["exposure"])
    exposure[field][0] += 1
    with pytest.raises(ValueError, match=message):
        ANALYSIS["verify_accounting"](exposure, validation, arm_name, arm, clean)


def test_registry_export_rejects_unverified_bytes_and_preserves_exact_source(tmp_path):
    source = RELEASE / "arxiv/relation_index.json"
    state = json.loads((RELEASE / "arxiv/replication.json").read_text())
    identity = state["provenance"]["preprocessing"]["relation_index.json"]
    destination = tmp_path / "relation_index.json"
    CURATION["export_relation_registry"](source, destination, identity)
    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        CURATION["export_relation_registry"](source, destination, identity)
    with pytest.raises(ValueError, match="completed-run preprocessing"):
        CURATION["export_relation_registry"](source, tmp_path / "other.json", {**identity, "sha256": "wrong"})
