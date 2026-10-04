"""Small synthetic RT-shaped batches and prediction interfaces; no RT weights."""

import json
import math
import runpy
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def make_batch(ids, slots=None, ctx=48):
    """One singleton FK parent per real query, with optional phantom tail."""
    slots = len(ids) if slots is None else slots
    nodes = np.full((slots, ctx), -1, dtype=np.int64)
    nbr = np.full((slots, ctx, 1), -1, dtype=np.int64)
    rel = nbr.copy()
    pad = np.ones((slots, ctx), dtype=bool)
    targets = np.zeros_like(pad)
    timestamps = np.full_like(nodes, np.iinfo(np.int32).min)
    mask = np.zeros(slots, dtype=bool)
    for row, target_id in enumerate(ids):
        nodes[row, :3] = [target_id, target_id + 10000, target_id + 20000]
        nbr[row, 1, 0] = nodes[row, 2]
        rel[row, 1, 0] = 7
        pad[row, :3] = False
        targets[row, 0] = True
        timestamps[row, :3] = [10, 5, 5]
        mask[row] = True
    return {
        "node_idxs": nodes, "f2p_nbr_idxs": nbr, "f2p_rel_idxs": rel,
        "is_padding": pad, "is_targets": targets, "timestamps": timestamps,
        "col_name_idxs": np.zeros_like(nodes), "is_task_nodes": ~pad,
        "number_values": np.ones((*nodes.shape, 1)), "batch_mask": mask,
    }


def cycle_fixture(repeat=1):
    nodes = np.arange(12)[None, :]
    nbr = np.full((1, 12, 1), -1, dtype=np.int64)
    rel = nbr.copy()
    nbr[0, :6, 0] = np.arange(6, 12)
    rel[0, :6, 0] = 7
    return (
        np.repeat(nbr, repeat, axis=1), np.repeat(rel, repeat, axis=1),
        np.repeat(nodes, repeat, axis=1), np.zeros((1, 12 * repeat), dtype=bool),
    )


class FakeDataset:
    def __init__(self, count, bs):
        self.rustler_dataset = SimpleNamespace(num_items=count)
        self.bs = bs

    def __len__(self):
        return math.ceil(self.rustler_dataset.num_items / self.bs)


class FakeLoader:
    def __init__(self, batches, count, bs):
        self.batches = batches
        self.dataset = FakeDataset(count, bs)

    def __iter__(self):
        return iter([{key: value.copy() for key, value in batch.items()} for batch in self.batches])


def make_evaluator(batches, bs, count=702, ctx=48):
    task = namedtuple("Task", "db_name table_name split")("rel-f1", "driver-dnf", "test")
    loader = FakeLoader(batches, count, bs)
    return SimpleNamespace(
        tasks=[task], world_size=1, items_per_task=702, eval_bs=bs, ctx_sizes=[ctx],
        eval_loaders={task: loader}, eval_loader_iters={task: iter(loader)},
    )


def runner_functions():
    return runpy.run_path(str(Path(__file__).parents[1] / "experiments/run_clean_rewire.py"))


def write_metadata(tmp_path):
    db = tmp_path / "rel-f1"
    db.mkdir()
    (db / "table_info.json").write_text(json.dumps({
        "driver-dnf:Test": {"num_nodes": 702, "node_idx_offset": 0},
    }))
    (db / "relation_index.json").write_text(json.dumps({"7": {"fk_column": "parent"}}))


class FakeModel:
    def __init__(self):
        self.calls = 0

    def eval(self):
        pass

    def predict(self, batch, contexts, device, task, bool_as_num):
        self.calls += 1
        return {ctx: batch["node_idxs"][:, 0].astype(float) / 1000 for ctx in contexts}


def with_raw_evaluation(ev):
    """Follow RT's batch/pop/predict/mask behavior using NumPy arrays."""
    def evaluate_raw(nets, contexts, with_node_idxs):
        for net, _ in nets:
            net.eval()
        task = ev.tasks[0]
        ids, labels = [], []
        scores = {name: [] for _, name in nets}
        for batch in ev.eval_loader_iters[task]:
            mask = batch.pop("batch_mask")
            for net, name in nets:
                scores[name].append(net.predict(batch, contexts, "cpu", task, True)[contexts[0]][mask])
            target_ids = (batch["node_idxs"] * batch["is_targets"]).sum(axis=1)[mask]
            ids.extend(target_ids)
            labels.extend((batch["number_values"].squeeze(-1) * batch["is_targets"]).sum(axis=1)[mask])
        yield task, contexts[0], np.asarray(labels), {name: np.concatenate(values) for name, values in scores.items()}, None, np.asarray(ids)

    ev.evaluate_raw = evaluate_raw
    return ev


def evaluator(ctx, uncertain=False, replay_change=False):
    batch = make_batch(range(702), slots=703, ctx=ctx)
    for row in range(702):
        batch["node_idxs"][row, :12] = row + np.arange(12) * 10000
        batch["is_padding"][row, :12] = False
        batch["is_task_nodes"][row, :12] = True
        batch["timestamps"][row, :12] = 5
        batch["timestamps"][row, 0] = 10
        batch["f2p_nbr_idxs"][row, :6, 0] = batch["node_idxs"][row, 6:12]
        batch["f2p_rel_idxs"][row, :6, 0] = 7
    batch["number_values"][:, 0, 0] = np.arange(703) % 2
    if uncertain:
        batch["timestamps"][0, 6] = np.iinfo(np.int32).min
    ev = with_raw_evaluation(make_evaluator([batch], 703, ctx=ctx))
    if replay_change:
        original = ev.evaluate_raw

        def changed(*args, **kwargs):
            ev.eval_loaders[ev.tasks[0]].loader.batches[0]["timestamps"][0, 0] = 9
            yield from original(*args, **kwargs)

        ev.evaluate_raw = changed
    return ev
