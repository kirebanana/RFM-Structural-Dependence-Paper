"""Prepare scoped MiniLM embeddings, or audit coverage without encoding/inference.

Marker vectors encode lookup IDs exactly in four base-128 BF16 digits. Sampling
uses an isolated temporary view of the intact native data, never replacements
inside the source artifact. Fingerprints exclude only the two embedding tensors,
exactly as run_database_replications.validate_panel does. No availability filter
is applied: even placeholders and subsequently masked cells need valid lookups.
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from rfm_structure.provenance import file_identity
from rfm_structure.replication import (
    MINILM_REVISION,
    MINILM_WEIGHTS_SHA256,
    replication_evaluator,
    selected_query_ids,
    validate_embedding_scope,
)
from rfm_structure.validation import batch_fingerprint, numpy_batch

TASKS = {"rel-arxiv": "paper-citation", "rel-trial": "study-outcome"}
CONTEXTS = (48, 128)
DIMENSION = 384
EMBEDDING_FILE = "text_emb_all-MiniLM-L12-v2.bin"
MODEL = "sentence-transformers/all-MiniLM-L12-v2"


def write_markers(path, rows):
    """Create BF16 lookup markers; all four digits are exactly representable."""
    from ml_dtypes import bfloat16

    if not 0 < rows <= 128 ** 4:
        raise ValueError("text row count cannot be represented by marker digits")
    with path.open("xb") as handle:
        handle.truncate(rows * DIMENSION * 2)
    vectors = np.memmap(path, dtype=np.uint16, mode="r+", shape=(rows, DIMENSION))
    for start in range(0, rows, 8192):
        ids = np.arange(start, min(start + 8192, rows), dtype=np.int64)
        digits = ((ids[:, None] // (128 ** np.arange(4))) % 128).astype(np.float32)
        vectors[start:start + len(ids), :4] = digits.astype(bfloat16).view(np.uint16)
    vectors.flush()
    del vectors


def decode_markers(values, rows):
    """Include padded/placeholder lookups too; padding zeros refer to text ID 0."""
    digits = np.asarray(values)[..., :4]
    if (not np.isfinite(digits).all() or np.any(digits < 0)
            or np.any(digits >= 128) or np.any(digits != np.floor(digits))):
        raise ValueError("invalid lookup marker")
    ids = (digits.astype(np.int64) * (128 ** np.arange(4))).sum(axis=-1)
    if np.any(ids >= rows):
        raise ValueError("lookup marker outside text table")
    return np.unique(ids)


def collect_coverage(pre_dir, database, task, query_ids, rows, *, evaluator_factory=None):
    """Replay the frozen explicit-node sampler with temporary marker embeddings."""
    factory = evaluator_factory or replication_evaluator
    temp_root = Path("/tmp/opencode")
    temp_root.mkdir(exist_ok=True)
    covered = set()
    fingerprints = {}
    with tempfile.TemporaryDirectory(prefix="replication-markers-", dir=temp_root) as tmp:
        root = Path(tmp)
        directory = root / database
        directory.mkdir()
        for source in (Path(pre_dir) / database).iterdir():
            if source.name not in (EMBEDDING_FILE, "embedding_scope.json") and source.is_file():
                (directory / source.name).symlink_to(source.resolve())
        write_markers(directory / EMBEDDING_FILE, rows)
        for ctx in CONTEXTS:
            ev = factory(root, task, query_ids, ctx, allow_marker_embeddings=True)
            batches = list(ev.eval_loaders[task])
            if len(batches) != 1:
                raise ValueError("expected one frozen explicit-node batch")
            arrays = numpy_batch(batches[0])
            targets = arrays["is_targets"] & ~arrays["is_padding"]
            actual_ids = (arrays["node_idxs"] * targets).sum(axis=1)
            if (not np.all(targets.sum(axis=1) == 1) or not arrays["batch_mask"].all()
                    or not np.array_equal(actual_ids, query_ids)):
                raise ValueError("marker sampling differs from frozen query panel")
            for key in ("text_values", "col_name_values"):
                covered.update(decode_markers(arrays[key], rows).tolist())
            fingerprints[str(ctx)] = batch_fingerprint({
                key: value for key, value in arrays.items()
                if key not in ("text_values", "col_name_values")
            })
            del ev, batches, arrays
    return sorted(covered), fingerprints


def verify_vector_layout(path, rows, covered):
    """Every covered vector must be finite; every unused component must be NaN."""
    if path.stat().st_size != rows * DIMENSION * 2:
        raise ValueError("embedding file size does not match text rows and BF16 dimension")
    selected = np.zeros(rows, dtype=bool)
    selected[covered] = True
    vectors = np.memmap(path, mode="r", dtype=np.uint16, shape=(rows, DIMENSION))
    for start in range(0, rows, 8192):
        bits = vectors[start:start + 8192]
        special = (bits & 0x7F80) == 0x7F80
        nan = special & ((bits & 0x007F) != 0)
        mask = selected[start:start + len(bits)]
        if special[mask].any() or not nan[~mask].all():
            raise ValueError("covered vectors must be finite and unused vectors must be BF16 NaN")
    del vectors


def check_existing(pre_dir, database, query_ids, rows, covered, fingerprints):
    directory = Path(pre_dir) / database
    scope = json.loads((directory / "embedding_scope.json").read_text())
    expected = {"database": database, "task": TASKS[database], "sampling_seed": 0,
                "selection_seed": 20261007, "contexts": list(CONTEXTS), "text_rows": rows,
                "query_ids": np.asarray(query_ids).tolist(), "encoded_text_ids": covered,
                "encoded_text_count": len(covered), "context_fingerprints": fingerprints}
    for key, value in expected.items():
        if scope.get(key) != value:
            raise ValueError(f"existing embedding scope differs from reconstructed {key}")
    validate_embedding_scope(pre_dir, database, scope)
    verify_vector_layout(directory / EMBEDDING_FILE, rows, covered)
    return scope


def encode_selected(texts, covered, *, device, batch_size):
    """Load only the pinned cached encoder, in FP32 even on CUDA."""
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    snapshot = Path(snapshot_download(MODEL, revision=MINILM_REVISION, local_files_only=True))
    weights = file_identity(snapshot / "model.safetensors")
    if weights["sha256"] != MINILM_WEIGHTS_SHA256:
        raise ValueError("cached MiniLM weights differ from pinned encoder")
    model = SentenceTransformer(str(snapshot), device=device, local_files_only=True,
                                model_kwargs={"dtype": torch.float32})
    if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise ValueError("MiniLM encoder must use FP32 parameters")
    with torch.inference_mode():
        vectors = model.encode([texts[index] for index in covered], batch_size=batch_size,
                               convert_to_tensor=True, convert_to_numpy=False,
                               normalize_embeddings=False, show_progress_bar=True)
    if (vectors.dtype != torch.float32 or tuple(vectors.shape) != (len(covered), DIMENSION)
            or not torch.isfinite(vectors).all()):
        raise ValueError("encoder output must be finite FP32 384-dimensional vectors")
    return vectors.to(torch.bfloat16).cpu().view(torch.int16).numpy().view(np.uint16), weights


def publish_embeddings(directory, rows, covered, encoded, scope):
    """Publish with exclusive hard links, scope last; never replace any artifact."""
    if encoded.dtype != np.uint16 or encoded.shape != (len(covered), DIMENSION):
        raise ValueError("encoded BF16 vector shape/dtype mismatch")
    embedding = directory / EMBEDDING_FILE
    scope_path = directory / "embedding_scope.json"
    if any(path.exists() or path.is_symlink() for path in (embedding, scope_path)):
        raise FileExistsError("refusing to overwrite existing embeddings or scope")
    with tempfile.TemporaryDirectory(prefix=".replication-embedding-", dir=directory) as tmp:
        staged = Path(tmp) / EMBEDDING_FILE
        with staged.open("xb") as handle:
            handle.truncate(rows * DIMENSION * 2)
        vectors = np.memmap(staged, dtype=np.uint16, mode="r+", shape=(rows, DIMENSION))
        vectors[:] = 0x7FC0  # canonical BF16 quiet NaN
        vectors[covered] = encoded
        vectors.flush()
        del vectors
        verify_vector_layout(staged, rows, covered)
        scope = {**scope, "status": "complete", "embedding_file": {
            **file_identity(staged), "path": str(embedding.resolve())}}
        staged_scope = Path(tmp) / "embedding_scope.json"
        staged_scope.write_text(json.dumps(scope, indent=2, allow_nan=False) + "\n")
        os.link(staged, embedding)  # atomic no-clobber publication on same filesystem
        try:
            os.link(staged_scope, scope_path)
        except BaseException:
            # Remove only our own newly published inode, never somebody else's file.
            if embedding.exists() and os.path.samefile(embedding, staged):
                embedding.unlink()
            raise
    return scope


def prepare(pre_dir, database, *, check_only=False, device="cpu", batch_size=64):
    from rt.tasks import tasks_from_preprocessed

    directory = Path(pre_dir) / database
    if not check_only and any((directory / name).exists() or (directory / name).is_symlink()
                              for name in (EMBEDDING_FILE, "embedding_scope.json")):
        raise FileExistsError("refusing to overwrite existing embeddings or scope; use --check-existing")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    started = time.monotonic()
    meta = json.loads((directory / "meta.json").read_text())
    rows = meta["num_text_strings"]
    tasks = [t for t in tasks_from_preprocessed(str(pre_dir), splits=("test",), dbs=[database])
             if t.table_name == TASKS[database]]
    if len(tasks) != 1 or tasks[0].task_type != "clf":
        raise ValueError("expected exactly the frozen replication classification task")
    task = tasks[0]
    query_ids = selected_query_ids(pre_dir, task)
    covered, fingerprints = collect_coverage(pre_dir, database, task, query_ids, rows)
    if check_only:
        return check_existing(pre_dir, database, query_ids, rows, covered, fingerprints)
    texts = json.loads((directory / "text.json").read_text())
    if len(texts) != rows:
        raise ValueError("text table length differs from metadata")
    encoded, weights = encode_selected(texts, covered, device=device, batch_size=batch_size)
    scope = {"database": database, "task": TASKS[database], "sampling_seed": 0,
             "selection_seed": 20261007, "contexts": list(CONTEXTS), "text_rows": rows,
             "query_ids": query_ids.tolist(), "context_fingerprints": fingerprints,
             "encoded_text_ids": covered, "encoded_text_count": len(covered),
             "encoder_revision": MINILM_REVISION, "encoder_dtype": "float32",
             "stored_dtype": "bfloat16", "encoder_weights": weights,
             "seconds": time.monotonic() - started}
    return publish_embeddings(directory, rows, covered, encoded, scope)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-dir", type=Path, required=True)
    parser.add_argument("--database", choices=TASKS, required=True)
    parser.add_argument("--check-existing", action="store_true",
                        help="reconstruct marker coverage only; no encoding, downloads or model inference")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    scope = prepare(args.pre_dir.resolve(), args.database, check_only=args.check_existing,
                    device=args.device, batch_size=args.batch_size)
    print(json.dumps({"status": "verified" if args.check_existing else "prepared",
                      "database": args.database, "queries": len(scope["query_ids"]),
                      "encoded_text_count": scope["encoded_text_count"],
                      "context_fingerprints": scope["context_fingerprints"],
                      "embedding_sha256": scope["embedding_file"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
