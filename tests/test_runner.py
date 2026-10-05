import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiment_fixtures import (
    FakeModel,
    cycle_fixture,
    evaluator,
    make_batch,
    runner_functions,
    write_metadata,
)


@pytest.fixture
def runner():
    return runner_functions()


def test_seven_arms_per_context_clean_once_and_npz_alignment(runner, tmp_path):
    write_metadata(tmp_path)
    models = []

    def factory():
        model = FakeModel()
        models.append(model)
        return model

    root = tmp_path / "matrix"
    result = runner["execute_matrix"](
        {48: evaluator(48), 128: evaluator(128)}, tmp_path, [101, 202, 303], [0, 0.5, 1], root,
        dry_only=False, model_factory=factory,
    )
    assert result["complete"]
    assert len(models) == 2
    assert [model.calls for model in models] == [7, 7]
    for ctx in (48, 128):
        directory = root / f"ctx{ctx}_lctx{ctx // 2}"
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["evidence_kind"] == "synthetic_test"
        assert len(manifest["arms"]) == 7
        assert manifest["n_queries"] == 702
        for arm in manifest["arms"].values():
            with np.load(directory / arm["predictions"]) as preds, np.load(directory / arm["exposure"]) as exposure:
                assert preds["node_idxs"].tolist() == list(range(702))
                assert np.array_equal(preds["node_idxs"], exposure["target_node_idxs"])
                assert exposure["requested_strength"].tolist() == [arm["requested_strength"]] * 702
                assert not np.any(exposure["no_changeable_edges"])
                assert np.all(np.isfinite(exposure["realized_max_fractions"]))
        assert len(list(directory.glob("validation_exposure_*.npz"))) == 7
    with pytest.raises(FileExistsError):
        runner["execute_matrix"]({}, tmp_path, [101], [0, 1], root)


def test_second_context_failure_blocks_every_model_load(runner, tmp_path):
    write_metadata(tmp_path)
    calls = []
    result = runner["execute_matrix"](
        {48: evaluator(48), 128: evaluator(128, uncertain=True)}, tmp_path,
        [101, 202, 303], [0, 0.5, 1], tmp_path / "matrix", dry_only=False,
        model_factory=lambda: calls.append(True),
    )
    assert not result["complete"]
    assert not calls
    assert result["contexts"]["128"] == "temporal_uncertainty"


def test_changed_replay_fails_before_prediction_and_marks_manifest_incomplete(runner, tmp_path):
    write_metadata(tmp_path)
    model = FakeModel()
    root = tmp_path / "matrix"
    with pytest.raises(ValueError, match="validated batch replay"):
        runner["execute_matrix"](
            {48: evaluator(48, replay_change=True), 128: evaluator(128)}, tmp_path,
            [101, 202, 303], [0, 0.5, 1], root, dry_only=False, model_factory=lambda: model,
        )
    assert model.calls == 0
    manifest = json.loads((root / "ctx48_lctx24/manifest.json").read_text())
    assert not manifest["complete"]
    assert manifest["status"] == "failed"
    assert json.loads((root / "matrix.json").read_text())["status"] == "failed"


def test_strength_query_relation_and_run_maxima_reconcile(tmp_path):
    from rfm_structure.validation import validate_batches, write_exposure_npz

    nbr, rel, nodes, pad = cycle_fixture()
    # Add padding to configured context; six changeable edges, all parent rows sampled.
    batch = make_batch([0])
    batch["node_idxs"][0, :12] = nodes[0]
    batch["is_padding"][0, :12] = pad[0]
    batch["is_task_nodes"][0, :12] = True
    batch["timestamps"][0, :12] = 5
    batch["timestamps"][0, 0] = 10
    batch["f2p_nbr_idxs"][0, :12] = nbr[0]
    batch["f2p_rel_idxs"][0, :12] = rel[0]
    report, exposure = validate_batches([batch], [0], [101], 48, [0, 0.5, 1])
    assert report["status"] == "passed"
    assert report["per_arm"]["a050_s101"]["realized_max_fraction"] == pytest.approx(1 / 3)
    arrays = write_exposure_npz(tmp_path / "partial.npz", exposure["a050_s101"])
    assert arrays["changed_edges"].tolist() == [2]
    assert arrays["max_changed_edges"].tolist() == [6]
    assert arrays["relation_max_changed_edges"].tolist() == [6]


def test_last_arm_input_mutation_is_detected_and_failure_is_recorded(runner, tmp_path):
    write_metadata(tmp_path)

    class MutatingModel(FakeModel):
        def predict(self, batch, *args):
            scores = super().predict(batch, *args)
            if self.calls == 7:
                batch["timestamps"][0, 0] = 9
            return scores

    root = tmp_path / "matrix"
    with pytest.raises(ValueError, match="mutated its sampled input"):
        runner["execute_matrix"]({48: evaluator(48), 128: evaluator(128)}, tmp_path,
                                 [101, 202, 303], [0, 0.5, 1], root, dry_only=False, model_factory=MutatingModel)
    status = json.loads((root / "matrix.json").read_text())
    assert not status["complete"]
    assert status["inference_ran"]  # Actual forwards happened, even though the run failed.
    manifest = json.loads((root / "ctx48_lctx24/manifest.json").read_text())
    assert manifest["model_forward_attempts"] == 7
    assert not manifest["complete"]


def test_real_loader_rejects_unavailable_cuda_before_weight_loading(runner, tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"test fixture, never loaded")

    def forbidden(*args, **kwargs):
        pytest.fail("weights must not be loaded when requested CUDA is unavailable")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: False,
    )))
    monkeypatch.setitem(sys.modules, "rt", SimpleNamespace(RelationalTransformer=SimpleNamespace(from_pretrained=forbidden)))
    with pytest.raises(ValueError, match="CUDA runtime is unavailable"):
        runner["load_frozen_model"](checkpoint, "cuda")


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_real_loader_supports_emulated_bf16_and_cpu_first_loading(runner, tmp_path, monkeypatch, device):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"test fixture")
    transitions = []
    dtype = object()
    parameter = SimpleNamespace(numel=lambda: 10, element_size=lambda: 2, requires_grad_=lambda _: None)

    class Model:
        def to(self, value):
            transitions.append(value)
            return self

        def eval(self):
            return self

        def parameters(self):
            return [parameter]

    def load(path, *, device, compile):
        assert device == "cpu"
        assert compile is False
        return Model()

    original_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **k:
                        "MemAvailable: 16000000 kB\n" if str(path) == "/proc/meminfo" else original_read(path, *a, **k))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16=dtype, cuda=SimpleNamespace(
        is_available=lambda: True, empty_cache=lambda: None,
        get_device_capability=lambda: (7, 5), get_device_name=lambda: "GTX 1650",
        mem_get_info=lambda: (512 * 2**20, 4 * 2**30),
        is_bf16_supported=lambda including_emulation: including_emulation,
    )))
    monkeypatch.setitem(sys.modules, "rt", SimpleNamespace(RelationalTransformer=SimpleNamespace(from_pretrained=load)))

    _, provenance = runner["load_frozen_model"](checkpoint, device)

    assert transitions == [dtype, device]
    assert provenance["execution"] == "eager"
    if device == "cuda":
        assert provenance["native_bf16"] is False
        assert provenance["bf16_with_emulation"] is True
        assert provenance["total_vram_bytes"] == 4 * 2**30
    else:
        assert provenance["device"] == "cpu"


@pytest.mark.parametrize("arguments", [
    ["--ctx", "48", "--local-ctx", "64"],
    ["--fractions", "0,0.2,1"],
    ["--seeds", "101,101,303"],
    ["--contexts", "48,48"],
    ["--inference-batch-size", "0"],
])
def test_cli_rejects_invalid_configuration_without_runtime(runner, tmp_path, monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["runner", "--pre-dir", str(tmp_path), "--dry-run", *arguments])
    with pytest.raises(SystemExit) as error:
        runner["main"]()
    assert error.value.code == 2


def test_cross_context_label_mismatch_blocks_all_weight_loads(runner, tmp_path):
    write_metadata(tmp_path)
    first, second = evaluator(48), evaluator(128)
    second.eval_loaders[second.tasks[0]].batches[0]["number_values"][0, 0, 0] = 1
    calls = []
    with pytest.raises(ValueError, match="across contexts"):
        runner["execute_matrix"]({48: first, 128: second}, tmp_path, [101, 202, 303], [0, 0.5, 1],
                                 tmp_path / "matrix", dry_only=False, model_factory=lambda: calls.append(True))
    assert not calls


def test_unused_cuda_cache_is_released_before_resource_measurement(runner, tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"never loaded")
    cache = {"occupied": True}

    def release():
        cache["occupied"] = False

    original_read = Path.read_text

    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return "MemAvailable: 1024 kB\n"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, get_device_capability=lambda: (8, 0), empty_cache=release,
        mem_get_info=lambda: (0 if cache["occupied"] else 16 * 2**30, 16 * 2**30),
    )))
    monkeypatch.setitem(sys.modules, "rt", SimpleNamespace(RelationalTransformer=None))
    # VRAM becomes adequate after cache release; host RAM is still insufficient.
    with pytest.raises(ValueError, match="host RAM"):
        runner["load_frozen_model"](checkpoint, "cuda")
    assert not cache["occupied"]


def test_predictions_receive_rewired_parents_while_queries_and_labels_stay_fixed(runner, tmp_path):
    write_metadata(tmp_path)

    class ParentSensitiveModel(FakeModel):
        def predict(self, batch, contexts, *args):
            predictions = super().predict(batch, contexts, *args)
            return {ctx: scores + batch["f2p_nbr_idxs"][:, 0, 0] / 1e6 for ctx, scores in predictions.items()}

    directory = tmp_path / "context"
    directory.mkdir()
    manifest = runner["run"](evaluator(48), tmp_path, [101], directory, "cpu", 24, model_factory=ParentSensitiveModel)
    assert manifest["complete"]
    with np.load(directory / "preds_base.npz") as clean, np.load(directory / "preds_a050_s101.npz") as partial, np.load(directory / "preds_a100_s101.npz") as full:
        assert np.array_equal(clean["node_idxs"], full["node_idxs"])
        assert np.array_equal(clean["labels"], full["labels"])
        # The first source initially has the smallest parent ID; full matching changes it.
        assert np.all(full["preds"] > clean["preds"])
        assert np.any(partial["preds"] == clean["preds"])
        assert np.any(partial["preds"] > clean["preds"])


def test_microbatches_preserve_contexts_matching_accounting_and_phantom_shapes(runner):
    from rfm_structure.validation import batch_fingerprint, real_query_batch

    raw = make_batch([0, 1, 2], slots=5)
    real, _ = real_query_batch(raw)
    holder = SimpleNamespace(batch_mask=raw.pop("batch_mask"), batch=None, batch_index=0,
                             forward_calls=0, fingerprints=[batch_fingerprint(real)])
    model = FakeModel()
    base = runner["BaseWrap"](model, holder, 101, inference_batch_size=2)
    rewired = runner["RewireWrap"](model, 101, 1.0, holder, inference_batch_size=2)

    clean_scores = base.predict(raw, [48], "cpu", None, True)[48]
    rewired_scores = rewired.predict(raw, [48], "cpu", None, True)[48]

    np.testing.assert_array_equal(clean_scores, [0, 0.001, 0.002, 0, 0])
    np.testing.assert_array_equal(clean_scores, rewired_scores)
    assert model.calls == holder.forward_calls == 4
    assert rewired.stats["eligible_edges"] == 3
    assert rewired.exposure_batches[0]["target_node_idxs"].tolist() == [0, 1, 2]
    assert batch_fingerprint(real_query_batch({**raw, "batch_mask": holder.batch_mask})[0]) == holder.fingerprints[0]


def test_microbatch_input_mutation_is_rejected(runner):
    from rfm_structure.validation import batch_fingerprint, real_query_batch

    raw = make_batch([0, 1, 2])
    real, _ = real_query_batch(raw)
    holder = SimpleNamespace(batch_mask=raw.pop("batch_mask"), batch=None, batch_index=0,
                             forward_calls=0, fingerprints=[batch_fingerprint(real)])

    class MutatingModel(FakeModel):
        def predict(self, batch, *args):
            scores = super().predict(batch, *args)
            batch["timestamps"][0, 0] = 123
            return scores

    wrapper = runner["BaseWrap"](MutatingModel(), holder, 101, inference_batch_size=2)
    with pytest.raises(ValueError, match="mutated its sampled input"):
        wrapper.predict(raw, [48], "cpu", None, True)


def test_availability_filtered_matrix_replays_identical_exposure_in_microbatches(runner, tmp_path):
    write_metadata(tmp_path)
    rules = {"version": "test", "tables": [
        {"table": "queries", "start": 0, "stop": 702, "time_col": "date",
         "label_column": 0, "horizon_seconds": 7},
        {"table": "support", "start": 10000, "stop": 10702, "time_col": "date",
         "label_column": 0, "horizon_seconds": 7},
        {"table": "other", "start": 20000, "stop": 120000, "time_col": "date",
         "label_column": None, "horizon_seconds": 0},
    ]}
    evaluators = {48: evaluator(48), 128: evaluator(128)}
    for ev in evaluators.values():
        ev.eval_loaders[ev.tasks[0]].batches[0]["timestamps"][0, 11] = 11
    root = tmp_path / "matrix"

    result = runner["execute_matrix"](
        evaluators, tmp_path, [101, 202, 303], [0, 0.5, 1], root,
        dry_only=False, model_factory=FakeModel, inference_batch_size=8,
        availability_policy=rules,
    )

    assert result["complete"]
    for ctx in (48, 128):
        directory = root / f"ctx{ctx}_lctx{ctx // 2}"
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["validation"]["availability_totals"]["removed_future_tokens"] == 1
        assert manifest["inference_batch_size"] == 8
        with np.load(directory / "exposure_base.npz") as exposure:
            assert exposure["availability_removed_future_tokens"].sum() == 1
            assert exposure["availability_removed_unmatured_label_tokens"].sum() == 702
            assert exposure["context_token_counts"][0] == 10
