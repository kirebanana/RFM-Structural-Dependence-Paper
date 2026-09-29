"""Read the FK relation metadata emitted by RT preprocessing."""

import json
from pathlib import Path


def load_relation_index(pre_dir: str | Path, dataset: str) -> dict[int, dict]:
    """Map relation IDs to child table, FK column, and parent table metadata.

    Reads ``<pre_dir>/<dataset>/relation_index.json`` and converts its JSON
    keys to integers. The runner saves this mapping with its result manifest.
    """
    path = Path(pre_dir).expanduser() / dataset / "relation_index.json"
    with path.open() as handle:
        raw = json.load(handle)
    return {int(relation_id): metadata for relation_id, metadata in raw.items()}
