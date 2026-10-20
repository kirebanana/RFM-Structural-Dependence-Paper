"""Copy the completed follow-up evidence into a portable, losslessly compressed release.

Original runs are never modified. Prediction/exposure arrays are compared before
and after compression; original file hashes are retained alongside new hashes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from rfm_structure.provenance import file_identity

RUNS = {
    "relations": ("relation_controls_1791303494752864585", "controls.json", "driver-dnf"),
    "task_cycles": ("relation_controls_1791304770103606851", "controls.json", "driver-dnf"),
    "labeled_support": ("relation_controls_1792159329463334922", "controls.json", "driver-dnf"),
    "driver_top3": ("preliminary_1791306037626899928", "matrix.json", "driver-top3"),
    "arxiv": ("replication_rel-arxiv_1791980546835938447", "replication.json", "paper-citation"),
    "trial": ("replication_rel-trial_1791980898619531972", "replication.json", "study-outcome"),
}


def read_json(path):
    return json.loads(path.read_text())


def read_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def portable(value):
    """Remove machine-specific absolute locations, retaining names and identities."""
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return Path(value).name
    return value


def write_json(path, value):
    path.write_text(json.dumps(portable(value), indent=2, allow_nan=False) + "\n")


def compress(source, destination, expected):
    original = file_identity(source)
    if original["sha256"] != expected:
        raise ValueError(f"original checksum mismatch: {source}")
    arrays = read_npz(source)
    np.savez_compressed(destination, **arrays)
    restored = read_npz(destination)
    if set(arrays) != set(restored) or any(
        arrays[key].dtype != restored[key].dtype
        or not np.array_equal(arrays[key], restored[key], equal_nan=True) for key in arrays
    ):
        raise ValueError("lossless compression check failed")
    return original, file_identity(destination)


def export_relation_registry(source, destination, expected):
    """Keep exact original bytes so the run's preprocessing digest authenticates it."""
    identity = file_identity(source)
    if any(identity[key] != expected[key] for key in ("sha256", "bytes")):
        raise ValueError("relation registry differs from completed-run preprocessing identity")
    with destination.open("xb") as handle:
        handle.write(source.read_bytes())


def curate(source_root, output, raw_dirs):
    import pandas as pd

    output.mkdir(parents=True, exist_ok=False)
    catalog = {"schema_version": 1, "baseline": "81e4fa4fdb98ea6ce5973ec4772f91dc4ca9a468",
               "primary": "../../preliminary/2026-10-05_validated", "runs": {}}
    for name, (run_id, state_name, task) in RUNS.items():
        source, dest = source_root / run_id, output / name
        state = read_json(source / state_name)
        if not state["complete"] or state["status"] != "complete":
            raise ValueError(f"incomplete source: {source}")
        dest.mkdir()
        if name in ("arxiv", "trial"):
            registry_identity = state["provenance"]["preprocessing"]["relation_index.json"]
            export_relation_registry(Path(registry_identity["path"]), dest / "relation_index.json", registry_identity)
        database = state.get("database", "rel-f1")
        raw = raw_dirs[database]
        spec = yaml.safe_load((raw / "tasks" / task / "manifest.yaml").read_text())
        table_path = raw / "tasks" / task / "test.parquet"
        table = pd.read_parquet(table_path)
        policy = state.get("availability_policy", state.get("policy"))
        offset = next(item["start"] for item in policy["tables"] if item["table"] == f"{task}:Test")
        clean_root = source if name in ("driver_top3", "arxiv", "trial") else Path("results/preliminary/2026-10-05_validated")
        clean = read_npz(clean_root / "ctx48_lctx24/preds_base.npz")
        panel = table.iloc[clean["node_idxs"] - offset]
        labels = panel[spec["target_col"]].to_numpy().astype(int)
        if not np.array_equal(labels, clean["labels"] > 0):
            raise ValueError("raw labels differ from saved predictions")
        np.savez_compressed(dest / "panel.npz", node_idxs=clean["node_idxs"], labels=labels,
                            dates=panel[spec["time_col"]].astype(str).to_numpy(dtype=str),
                            entities=panel[spec["entity_col"]].to_numpy())
        write_json(dest / "panel_source.json", {"database": database, "task": task, "test_offset": offset,
                   "manifest": spec, "official_test_size": len(table), "raw_test": file_identity(table_path),
                   "raw_manifest": file_identity(raw / "tasks" / task / "manifest.yaml")})
        identities = {}
        for ctx in (48, 128):
            context = f"ctx{ctx}_lctx{ctx // 2}"
            source_dir, dest_dir = source / context, dest / context
            dest_dir.mkdir()
            manifest = read_json(source_dir / "manifest.json")
            for arm in manifest["arms"].values():
                for field, hash_field in (("predictions", "prediction_sha256"), ("exposure", "exposure_sha256")):
                    filename = arm[field]
                    original, compressed = compress(source_dir / filename, dest_dir / filename, arm[hash_field])
                    identities[f"{context}/{filename}"] = {"original": original, "curated": compressed}
                    arm[hash_field] = compressed["sha256"]
            write_json(dest_dir / "manifest.json", manifest)
            write_json(dest_dir / "validation.json", read_json(source_dir / "validation.json"))
        write_json(dest / state_name, state)
        write_json(dest / "artifact_lineage.json", identities)
        catalog["runs"][name] = {"source_run": run_id, "state_file": state_name, "database": database,
                                 "task": task, "clean_reference": "." if clean_root == source else catalog["primary"]}
    write_json(output / "catalog.json", catalog)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/runs"))
    parser.add_argument("--output", type=Path, required=True)
    for name in ("f1", "arxiv", "trial"):
        parser.add_argument(f"--{name}-raw", type=Path, required=True)
    args = parser.parse_args()
    curate(args.source, args.output, {"rel-f1": args.f1_raw, "rel-arxiv": args.arxiv_raw, "rel-trial": args.trial_raw})


if __name__ == "__main__":
    main()
