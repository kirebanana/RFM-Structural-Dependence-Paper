"""Small helpers for self-describing RT preprocessing artifacts."""

import json
from pathlib import Path


def load_relation_index(pre_dir: str | Path, dataset: str) -> dict[int, dict]:
    """Load the relation registry emitted by the modified Rust preprocessor."""
    path = Path(pre_dir).expanduser() / dataset / "relation_index.json"
    with path.open() as handle:
        raw = json.load(handle)
    return {int(relation_id): metadata for relation_id, metadata in raw.items()}
