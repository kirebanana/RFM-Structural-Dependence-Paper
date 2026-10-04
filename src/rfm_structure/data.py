"""Read the FK relation metadata emitted by RT preprocessing."""

import json
from pathlib import Path

import numpy as np


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
