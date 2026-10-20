"""Reproducible coverage discovery and no-clobber scoped embedding publication."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rfm_structure.replication import MINILM_REVISION, MINILM_WEIGHTS_SHA256
from rfm_structure.validation import batch_fingerprint


@pytest.fixture
def preparation():
    path = Path(__file__).parents[1] / "experiments/prepare_replication_embeddings.py"
    spec = importlib.util.spec_from_file_location("scoped_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_marker_roundtrip_including_large_ids_and_placeholders(tmp_path, preparation):
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    rows = 128 ** 2 + 3
    path = tmp_path / "markers.bin"
    preparation.write_markers(path, rows)
    vectors = np.memmap(path, mode="r", dtype=bf16, shape=(rows, 384))
    selected = np.array([0, 127, 128, 128 ** 2, rows - 1])
    actual = preparation.decode_markers(vectors[selected].astype(np.float32), rows)
    np.testing.assert_array_equal(actual, selected)
    assert not vectors[:, 4:].any()
    with pytest.raises(FileExistsError):
        preparation.write_markers(path, rows)


def test_marker_rejects_fractional_nonfinite_and_out_of_range(preparation):
    for value in (-1, 128, 0.5, np.nan):
        values = np.zeros((1, 384))
        values[0, 0] = value
        with pytest.raises(ValueError, match="invalid lookup marker"):
            preparation.decode_markers(values, 200)
    values = np.zeros((1, 384))
    values[0, 0] = 5
    with pytest.raises(ValueError, match="outside text table"):
        preparation.decode_markers(values, 5)


def test_coverage_unions_text_schema_and_padding_without_modifying_source(tmp_path, preparation):
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    source = tmp_path / "rel-arxiv"
    source.mkdir()
    (source / "nodes.rkyv").write_bytes(b"intact")
    (source / preparation.EMBEDDING_FILE).write_bytes(b"do not overwrite")
    (source / "embedding_scope.json").write_text("original")
    expected = {}

    def factory(root, task, query_ids, ctx, **kwargs):
        assert kwargs == {"allow_marker_embeddings": True}
        assert (root / "rel-arxiv/nodes.rkyv").read_bytes() == b"intact"
        markers = np.memmap(root / "rel-arxiv" / preparation.EMBEDDING_FILE,
                            mode="r", dtype=bf16, shape=(8, 384))
        arrays = {"node_idxs": np.array([[10, 50], [11, 51]]),
                  "is_targets": np.array([[True, False], [True, False]]),
                  "is_padding": np.array([[False, True], [False, True]]),
                  "batch_mask": np.ones(2, dtype=bool)}
        expected[str(ctx)] = batch_fingerprint(arrays)
        # IDs 0 and 7 occur only at padded positions; still required lookups.
        arrays["text_values"] = markers[[[1, 0], [2 if ctx == 48 else 3, 0]]].astype(np.float32)
        arrays["col_name_values"] = markers[[[4, 7], [5, 0]]].astype(np.float32)
        return SimpleNamespace(eval_loaders={task: [arrays]})

    covered, fingerprints = preparation.collect_coverage(
        tmp_path, "rel-arxiv", "task", np.array([10, 11]), 8, evaluator_factory=factory)
    assert covered == [0, 1, 2, 3, 4, 5, 7]
    assert fingerprints == expected
    assert (source / preparation.EMBEDDING_FILE).read_bytes() == b"do not overwrite"
    assert (source / "embedding_scope.json").read_text() == "original"
    assert sorted(p.name for p in source.iterdir()) == [
        "embedding_scope.json", "nodes.rkyv", preparation.EMBEDDING_FILE]


def published_fixture(tmp_path, preparation):
    directory = tmp_path / "rel-arxiv"
    directory.mkdir()
    scope = {"database": "rel-arxiv", "task": "paper-citation", "sampling_seed": 0,
             "selection_seed": 20261007, "contexts": [48, 128], "text_rows": 8,
             "query_ids": [10, 11], "encoded_text_ids": [0, 2], "encoded_text_count": 2,
             "context_fingerprints": {"48": "first", "128": "second"},
             "encoder_revision": MINILM_REVISION, "encoder_dtype": "float32",
             "stored_dtype": "bfloat16", "encoder_weights": {"sha256": MINILM_WEIGHTS_SHA256}}
    encoded = np.full((2, 384), 0x3F80, dtype=np.uint16)  # finite BF16 1.0
    scope = preparation.publish_embeddings(directory, 8, [0, 2], encoded, scope)
    return directory, scope, encoded


def test_publish_and_check_layout_scope_and_no_overwrite(tmp_path, preparation):
    directory, scope, encoded = published_fixture(tmp_path, preparation)
    returned = preparation.check_existing(tmp_path, "rel-arxiv", [10, 11], 8, [0, 2],
                                           scope["context_fingerprints"])
    assert returned == scope
    before = (directory / preparation.EMBEDDING_FILE).read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        preparation.publish_embeddings(directory, 8, [0, 2], encoded, scope)
    assert (directory / preparation.EMBEDDING_FILE).read_bytes() == before
    assert json.loads((directory / "embedding_scope.json").read_text()) == scope


@pytest.mark.parametrize("key,value", [("query_ids", [11, 10]), ("encoded_text_ids", [0, 3]),
                                      ("encoded_text_count", 3), ("sampling_seed", 1),
                                      ("context_fingerprints", {"48": "wrong", "128": "second"})])
def test_check_existing_rejects_reconstructed_coverage_mismatch(tmp_path, preparation, key, value):
    directory, scope, _ = published_fixture(tmp_path, preparation)
    (directory / "embedding_scope.json").write_text(json.dumps({**scope, key: value}))
    with pytest.raises(ValueError, match=f"reconstructed {key}"):
        preparation.check_existing(tmp_path, "rel-arxiv", [10, 11], 8, [0, 2],
                                   scope["context_fingerprints"])


def test_check_existing_rejects_finite_wrong_file(tmp_path, preparation):
    directory, scope, _ = published_fixture(tmp_path, preparation)
    np.zeros((8, 384), dtype=np.uint16).tofile(directory / preparation.EMBEDDING_FILE)
    with pytest.raises(ValueError, match="file identity differs"):
        preparation.check_existing(tmp_path, "rel-arxiv", [10, 11], 8, [0, 2],
                                   scope["context_fingerprints"])


@pytest.mark.parametrize("row,bits", [(0, 0x7F80), (0, 0x7FC0), (1, 0x3F80)])
def test_layout_rejects_nonfinite_covered_or_finite_unused(tmp_path, preparation, row, bits):
    directory, _, _ = published_fixture(tmp_path, preparation)
    path = directory / preparation.EMBEDDING_FILE
    vectors = np.memmap(path, mode="r+", dtype=np.uint16, shape=(8, 384))
    vectors[row, 0] = bits
    vectors.flush()
    with pytest.raises(ValueError, match="covered vectors must be finite"):
        preparation.verify_vector_layout(path, 8, [0, 2])


def test_failed_scope_publication_removes_only_own_embedding(tmp_path, preparation, monkeypatch):
    directory = tmp_path / "rel-arxiv"
    directory.mkdir()
    original_link = preparation.os.link

    def raced_link(source, target):
        if Path(target).name == "embedding_scope.json":
            Path(target).write_text("another writer")
        return original_link(source, target)

    monkeypatch.setattr(preparation.os, "link", raced_link)
    with pytest.raises(FileExistsError):
        preparation.publish_embeddings(directory, 2, [0], np.zeros((1, 384), dtype=np.uint16), {})
    assert not (directory / preparation.EMBEDDING_FILE).exists()
    assert (directory / "embedding_scope.json").read_text() == "another writer"


def test_check_only_never_encodes_or_loads_encoder(tmp_path, preparation, monkeypatch):
    rt_tasks = pytest.importorskip("rt.tasks")
    directory, scope, _ = published_fixture(tmp_path, preparation)
    (directory / "meta.json").write_text(json.dumps({"num_text_strings": 8}))
    task = SimpleNamespace(table_name="paper-citation", task_type="clf")
    monkeypatch.setattr(rt_tasks, "tasks_from_preprocessed", lambda *a, **kw: [task])
    monkeypatch.setattr(preparation, "selected_query_ids", lambda *a: np.array([10, 11]))
    monkeypatch.setattr(preparation, "collect_coverage", lambda *a: ([0, 2], scope["context_fingerprints"]))

    def forbidden(*args, **kwargs):
        raise AssertionError("coverage-only mode must never load or invoke an encoder")

    monkeypatch.setattr(preparation, "encode_selected", forbidden)
    assert preparation.prepare(tmp_path, "rel-arxiv", check_only=True) == scope
    with pytest.raises(FileExistsError):
        preparation.prepare(tmp_path, "rel-arxiv")


def test_encoding_pins_offline_snapshot_fp32_and_converts_to_bf16(tmp_path, preparation, monkeypatch):
    torch = pytest.importorskip("torch")
    calls = {}

    def snapshot_download(repo, **kwargs):
        calls["download"] = (repo, kwargs)
        return str(tmp_path)

    class Encoder:
        def __init__(self, path, **kwargs):
            calls["model"] = (path, kwargs)

        def parameters(self):
            return [torch.zeros(1, dtype=torch.float32)]

        def encode(self, texts, **kwargs):
            assert not torch.is_grad_enabled()
            calls["encode"] = (texts, kwargs)
            return torch.ones((len(texts), 384), dtype=torch.float32)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Encoder))
    monkeypatch.setattr(preparation, "file_identity", lambda path: {"sha256": MINILM_WEIGHTS_SHA256})
    vectors, weights = preparation.encode_selected(["placeholder", "unused", "schema"], [0, 2],
                                                   device="cpu", batch_size=16)
    assert calls["download"] == (preparation.MODEL, {"revision": MINILM_REVISION, "local_files_only": True})
    assert calls["model"] == (str(tmp_path), {"device": "cpu", "local_files_only": True,
                                             "model_kwargs": {"dtype": torch.float32}})
    assert calls["encode"][0] == ["placeholder", "schema"]
    assert calls["encode"][1] == {"batch_size": 16, "convert_to_tensor": True,
                                  "convert_to_numpy": False, "normalize_embeddings": False,
                                  "show_progress_bar": True}
    assert vectors.dtype == np.uint16 and vectors.shape == (2, 384)
    assert np.all(vectors == 0x3F80)
    assert weights["sha256"] == MINILM_WEIGHTS_SHA256
    monkeypatch.setattr(preparation, "file_identity", lambda path: {"sha256": "wrong"})
    with pytest.raises(ValueError, match="weights differ"):
        preparation.encode_selected(["placeholder"], [0], device="cpu", batch_size=16)


@pytest.mark.parametrize("name", ["embedding_scope.json", "text_emb_all-MiniLM-L12-v2.bin"])
def test_publication_refuses_preexisting_partial_artifact(tmp_path, preparation, name):
    (tmp_path / name).write_bytes(b"existing partial artifact")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        preparation.publish_embeddings(tmp_path, 2, [0], np.zeros((1, 384), dtype=np.uint16), {})
    assert (tmp_path / name).read_bytes() == b"existing partial artifact"
    assert len(list(tmp_path.iterdir())) == 1
