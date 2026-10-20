"""Small native regression: temporal entity tables need not have outgoing FKs."""
import json

import numpy as np
import pytest


def test_native_preprocessor_preserves_time_on_entity_without_foreign_keys(tmp_path):
    pd = pytest.importorskip("pandas")
    native = pytest.importorskip("rt._rustler")
    from rt.eval_utils import build_evaluator
    from rt.tasks import tasks_from_preprocessed

    raw = tmp_path / "raw"
    (raw / "db").mkdir(parents=True)
    (raw / "tasks/forecast").mkdir(parents=True)
    (raw / "manifest.yaml").write_text(json.dumps({
        "name": "temporal-entity-fixture", "manifest_version": 1,
        "val_timestamp": "2005-01-01", "test_timestamp": "2010-01-01",
        "tables": {"entities": {"pkey": "id", "time_col": "created", "fkeys": {}}},
    }))
    pd.DataFrame({"id": [0, 1], "created": pd.to_datetime(["2000-01-01", "2001-01-01"]),
                  "feature": [1.0, 2.0]}).to_parquet(raw / "db/entities.parquet", index=False)
    (raw / "tasks/forecast/manifest.yaml").write_text(json.dumps({
        "name": "forecast", "kind": "forecast", "task_type": "binary_classification",
        "entity_table": "entities", "entity_col": "entity", "target_col": "label",
        "time_col": "timestamp", "timedelta": "30 days", "manifest_version": 1,
    }))
    pd.DataFrame({"timestamp": pd.to_datetime(["2010-01-01", "2010-01-01"]),
                  "entity": [0, 1], "label": [0, 1]}).to_parquet(raw / "tasks/forecast/test.parquet", index=False)
    pd.DataFrame({"timestamp": pd.to_datetime(["2004-01-01", "2004-01-01"]),
                  "entity": [0, 1], "label": [0, 1]}).to_parquet(raw / "tasks/forecast/train.parquet", index=False)
    out = tmp_path / "pre"
    native.preprocess(str(raw), str(out), source=str(raw), skip_tasks=False)
    directory = out / "temporal-entity-fixture"
    meta = json.loads((directory / "meta.json").read_text())
    np.zeros((meta["num_text_strings"], 384), dtype=np.uint16).tofile(directory / "text_emb_all-MiniLM-L12-v2.bin")
    tasks = tasks_from_preprocessed(str(out), splits=("test",), dbs=[meta["name"]])
    ev = build_evaluator(tasks, str(out), embedding_model="all-MiniLM-L12-v2", d_text=384,
                         device="cpu", ctx_size=16, local_ctx_size=8, items_per_task=2,
                         num_workers=0, num_walks=0, mmap_populate=False, tokens_per_gpu=32)
    batch = next(iter(ev.eval_loaders[tasks[0]]))
    info = json.loads((directory / "table_info.json").read_text())["entities:Db"]
    present = ~batch["is_padding"].numpy()
    nodes = batch["node_idxs"].numpy()
    times = batch["timestamps"].numpy()
    for row, date in enumerate(("2000-01-01", "2001-01-01")):
        positions = present & (nodes == info["node_idx_offset"] + row)
        assert positions.any(), "fixture must actually sample the entity row"
        assert np.all(times[positions] == int(pd.Timestamp(date).timestamp()))
