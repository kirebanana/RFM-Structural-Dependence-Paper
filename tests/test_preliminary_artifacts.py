"""Verify curated real evidence without RT, external data, or a checkpoint."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rfm_structure.analysis import analyze_matrix, verify_policy_exposure


def test_curated_preliminary_results_have_valid_checksums_pairing_and_accounting():
    root = Path(__file__).parents[1] / "results/preliminary/2026-10-05_validated"
    inventory = json.loads((root / "checksums.json").read_text())
    for name, identity in inventory.items():
        content = (root / name).read_bytes()
        assert len(content) == identity["bytes"]
        assert hashlib.sha256(content).hexdigest() == identity["sha256"]
    analysis = analyze_matrix(root)
    recorded = json.loads((root / "analysis.json").read_text())
    assert analysis["rows"] == recorded["rows"]
    assert analysis["summaries"] == recorded["summaries"]
    assert len(analysis["rows"]) == 14
    assert len(analysis["summaries"]) == 6


@pytest.mark.parametrize("field", ["temporal_checked", "availability_removed_future_tokens", "fanout_feat_absolute_delta"])
def test_saved_policy_accounting_rejects_corrupt_evidence(field):
    root = Path(__file__).parents[1] / "results/preliminary/2026-10-05_validated/ctx48_lctx24"
    manifest = json.loads((root / "manifest.json").read_text())
    with np.load(root / "exposure_a100_s101.npz", allow_pickle=False) as saved:
        exposure = {key: saved[key].copy() for key in saved.files}
    exposure[field][0] += 1
    with pytest.raises(ValueError):
        verify_policy_exposure(exposure, manifest["validation"], "a100_s101")
