"""Read the FK relation metadata emitted by RT preprocessing."""

import json
import re
from pathlib import Path

import numpy as np

from .provenance import file_identity


def load_relation_index(pre_dir: str | Path, dataset: str) -> dict[int, dict]:
    """Map relation IDs to child table, FK column, and parent table metadata.

    Reads ``<pre_dir>/<dataset>/relation_index.json`` and converts its JSON
    keys to integers. The runner saves this mapping with its result manifest.
    """
    path = Path(pre_dir).expanduser() / dataset / "relation_index.json"
    with path.open() as handle:
        raw = json.load(handle)
    return {int(relation_id): metadata for relation_id, metadata in raw.items()}


def load_expected_target_ids(pre_dir, task, expected_count):
    """Read the full task-split ID range used by RT's seed-node join.

    Require the whole split, not an arbitrary 702-row subset of a larger task.
    The Db-key preference matches RustlerDataset and RT's submission join.
    """
    path = Path(pre_dir).expanduser() / task.db_name / "table_info.json"
    with path.open() as handle:
        tables = json.load(handle)
    db_key = f"{task.table_name}:Db"
    split_key = f"{task.table_name}:{task.split.capitalize()}"
    info = tables[db_key if db_key in tables else split_key]
    if info["num_nodes"] != expected_count:
        raise ValueError(f"task split must contain exactly {expected_count} targets")
    offset = int(info["node_idx_offset"])
    return np.arange(offset, offset + expected_count, dtype=np.int64)


def load_availability_policy(pre_dir, dataset="rel-f1"):
    """Bind query-time filtering to the local source manifests and node ranges.

    Timelessness comes from an explicit null time_col, never the timestamp
    sentinel. Forecast-label horizons come from task manifests. Source files
    must remain locally available; no inference of availability or downloads.
    """
    import yaml

    directory = Path(pre_dir) / dataset
    meta = json.loads((directory / "meta.json").read_text())
    source = Path(meta["source"])
    manifest_path = source / "manifest.yaml"
    schema = yaml.safe_load(manifest_path.read_text())
    if schema["name"] != dataset:
        raise ValueError("source manifest dataset differs from preprocessing")
    tables = json.loads((directory / "table_info.json").read_text())
    columns = json.loads((directory / "column_index.json").read_text())
    task_specs = {}
    identities = {"dataset_manifest": file_identity(manifest_path)}
    for task in meta["tasks"]:
        path = source / "tasks" / task["name"] / "manifest.yaml"
        spec = yaml.safe_load(path.read_text())
        match = re.fullmatch(r"(\d+) days", str(spec.get("timedelta", "")))
        if spec["kind"] != "forecast" or not match or not spec.get("time_col"):
            raise ValueError("availability requires a forecast task with a day-based horizon")
        task_specs[task["name"]] = spec | {"horizon_seconds": int(match[1]) * 86400}
        identities[task["name"]] = file_identity(path)
    ranges = []
    for name, info in tables.items():
        table, split = name.rsplit(":", 1)
        if split == "Db":
            spec = schema["tables"][table]
            if "time_col" not in spec:
                raise ValueError(f"missing explicit time-column metadata for {table}")
            time_col = spec["time_col"]
            label_col = None
            horizon = 0
        else:
            spec = task_specs[table]
            time_col = spec["time_col"]
            label_col = columns[f"{spec['target_col']} of {table}"]
            horizon = spec["horizon_seconds"]
        ranges.append({
            "table": name, "start": info["node_idx_offset"],
            "stop": info["node_idx_offset"] + info["num_nodes"],
            "time_col": time_col, "label_column": label_col, "horizon_seconds": horizon,
        })
    return {
        "version": "query_available_incidence_v1", "dataset": dataset,
        "cutoff_rule": "temporal rows <= query; support label timestamp + horizon <= query",
        "timeless_rule": "explicit source manifest time_col=null; benchmark static covariates",
        "fanout_rule": "measured consequence of incidence, not a preservation gate",
        "refill_removed_cells": False, "tables": ranges, "source_manifests": identities,
    }
