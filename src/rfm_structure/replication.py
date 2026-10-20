"""Frozen small panels for metadata-driven cross-database replication."""
import json
from pathlib import Path

import numpy as np

from .provenance import file_identity
from .validation import numpy_batch

MINILM_REVISION = "a50ef00143b4d5391434df20ae11632588ac25be"
MINILM_WEIGHTS_SHA256 = "d2d541e5f101695ae495eacd867a8d025ecfe8f9674fb23aa6cf93cdb60a5542"


def validate_embedding_scope(pre_dir, database, scope):
    """Authenticate scoped vectors at the current path, not a saved local path.

    Encoder metadata binds the claimed preparation protocol; the file digest
    binds its output. Neither finiteness nor context coverage proves provenance.
    Ordinary fully encoded artifacts without a scope remain supported.
    """
    if scope is None:
        return
    if (scope.get("status") != "complete"
            or scope.get("encoder_revision") != MINILM_REVISION
            or scope.get("encoder_dtype") != "float32"
            or scope.get("stored_dtype") != "bfloat16"
            or scope.get("encoder_weights", {}).get("sha256") != MINILM_WEIGHTS_SHA256):
        raise ValueError("scoped embedding encoder provenance differs from pinned MiniLM FP32/BF16 protocol")
    actual = file_identity(Path(pre_dir) / database / "text_emb_all-MiniLM-L12-v2.bin")
    expected = scope.get("embedding_file", {})
    if any(actual[key] != expected.get(key) for key in ("bytes", "sha256")):
        raise ValueError("scoped embedding file identity differs from coverage manifest")


def selected_query_ids(pre_dir, task, count=512, seed=20261007):
    """Uniformly select official test rows before inspecting labels or scores."""
    tables = json.loads((Path(pre_dir) / task.db_name / "table_info.json").read_text())
    info = tables[f"{task.table_name}:Test"]
    if count < 2 or count > info["num_nodes"]:
        raise ValueError("replication count must fit the official test split")
    chosen = np.random.default_rng(np.random.PCG64(seed)).choice(info["num_nodes"], count, replace=False)
    return chosen.astype(np.int64) + info["node_idx_offset"]


def replication_groups(registry, task_name, entity_table, database_tables):
    """Select task/entity and ordinary-DB relations from schema metadata."""
    groups = {
        "task_entity": sorted(int(r) for r, m in registry.items()
                              if m["child_table"] == task_name and m["parent_table"] == entity_table),
        "database": sorted(int(r) for r, m in registry.items() if m["child_table"] in database_tables),
    }
    if any(not ids for ids in groups.values()) or set(groups["task_entity"]) & set(groups["database"]):
        raise ValueError("replication requires nonempty disjoint task/entity and DB relations")
    return groups


class SelectedContextLoader:
    """Replay explicitly chosen target nodes, with no label-dependent selection.

    Rustler's explicit-node sampler uses the same per-node context seed and
    graph policy as ordinary evaluation. A retry yielding a different target
    fails here. The evaluator sees one batch of selected real queries.
    """

    def __init__(self, rustler_dataset, query_ids, ctx):
        self.rustler_dataset = rustler_dataset
        self.query_ids = np.asarray(query_ids, dtype=np.int64)
        self.ctx = ctx
        self.dataset = self

    def __len__(self):
        return 1

    def __iter__(self):
        native = self.rustler_dataset.sampler.batch_for_nodes_py(self.query_ids.tolist(), 0, self.ctx)
        batch = self.rustler_dataset._process_batch(native)
        arrays = numpy_batch(batch)
        ids = (arrays["node_idxs"] * arrays["is_targets"]).sum(axis=1)
        if not np.array_equal(ids, self.query_ids) or not arrays["batch_mask"].all():
            raise ValueError("explicit-node sampler did not return exactly the frozen selected queries")
        yield batch


def replication_evaluator(pre_dir, task, query_ids, ctx, device="cpu", *, allow_marker_embeddings=False):
    """Build the existing RT evaluator, then replace only target selection."""
    from rt.eval_utils import build_evaluator

    scope_file = Path(pre_dir) / task.db_name / "embedding_scope.json"
    if scope_file.exists() and not allow_marker_embeddings:
        scope = json.loads(scope_file.read_text())
        if scope["status"] != "complete" or scope["query_ids"] != np.asarray(query_ids).tolist() or str(ctx) not in scope["context_fingerprints"]:
            raise ValueError("embedding coverage does not match the frozen replication panel")
        validate_embedding_scope(pre_dir, task.db_name, scope)
    ev = build_evaluator([task], str(pre_dir), embedding_model="all-MiniLM-L12-v2", d_text=384,
                         device=device, ctx_size=ctx, local_ctx_size=ctx // 2,
                         items_per_task=len(query_ids), num_workers=0, context_seed=0, shuffle_seed=0,
                         mmap_populate=False)
    dataset = ev.eval_loaders[task].dataset.rustler_dataset
    loader = SelectedContextLoader(dataset, query_ids, ctx)
    ev.eval_bs = len(query_ids)
    ev.items_per_task = None
    ev.eval_loaders = {task: loader}
    ev.eval_loader_iters = {task: iter(loader)}
    return ev
