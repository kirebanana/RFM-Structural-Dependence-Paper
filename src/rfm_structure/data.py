"""Read project-specific metadata emitted beside RT preprocessing artifacts."""

import json
from pathlib import Path


def load_relation_index(pre_dir: str | Path, dataset: str) -> dict[int, dict]:
    """Load ``<pre_dir>/<dataset>/relation_index.json`` with integer keys.

    The modified Rust preprocessor writes this registry so an ``f2p_rel_idxs``
    value can be interpreted as a child table, foreign-key column, and parent
    table. The helper performs no schema validation and raises normal file/JSON
    exceptions when the artifact is missing or malformed.
    """
    path = Path(pre_dir).expanduser() / dataset / "relation_index.json"
    with path.open() as handle:
        raw = json.load(handle)
    return {int(relation_id): metadata for relation_id, metadata in raw.items()}
