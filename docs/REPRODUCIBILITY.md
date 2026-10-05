# Setup and reproduction

This guide takes you from a fresh checkout to prepared data, validation, and
inference. If you only want to inspect the committed results, start with the
saved-array analysis command in the last section; no model or dataset is needed.

Run commands from the repository root on Linux/WSL. Use `uv` and a Rust toolchain
with native build prerequisites (compiler/linker and Python/native headers as
required by PyO3/maturin). The setup script installs managed Python 3.12.13,
syncs the lockfile and builds the editable RT/Rustler extension:

```bash
bash scripts/setup.sh
uv run pytest -q
uv run ruff check .
```

Pure tests/curated-result analysis need only `uv sync --extra dev`; RT execution
needs the `rt` extra. Data, pretrained embeddings and checkpoints require network
access once or existing local caches. The runner itself uses local paths only.

## 1. Obtain pinned data and checkpoint

The completed run used dataset revision
`d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600` and model revision
`2ed96efc8007c57d589a31ceb435eb768491c439`:

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "stanford-star/relbench-v1",
    repo_type="dataset",
    revision="d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600",
    allow_patterns="rel-f1/*",
    local_dir="artifacts/raw",
)
snapshot_download(
    "stanford-star/rt-plurel",
    revision="2ed96efc8007c57d589a31ceb435eb768491c439",
    allow_patterns="classification/*",
    local_dir="artifacts/rt-plurel",
)
PY
```

The raw directory `artifacts/raw/rel-f1/` must contain `manifest.yaml`, database
parquets in `db/`, and task manifests/split parquets in `tasks/`. The checkpoint
directory `artifacts/rt-plurel/classification/` contains `config.json` and
`model.safetensors`. Expected weight SHA-256:

```text
4a369f065d8f69e1da11ee4326ceb9d32cf78ee69255a438449e971f681b7f83
```

The loader records actual file hashes; a matching architecture alone does not
prove model-family identity. For comparison with this result set, verify the
weight hash against `results/preliminary/2026-10-05_validated/provenance.json`
and the context manifests.

## 2. Preprocess

```bash
bash scripts/preprocess.sh artifacts/raw/rel-f1 artifacts/clean_rewire_preprocessed
```

This invokes the installed native Rustler preprocessing and creates MiniLM-L12-v2
text embeddings (384 dimensions). The embedding step can download its encoder.
Do not use legacy hosted preprocessed files lacking the local relation patch:
the serialized node layout must agree with the built extension.

`--pre-dir` names the **collection root**, not its `rel-f1/` subdirectory.
The latter must contain:

```text
meta.json                 table_info.json          column_index.json
relation_index.json       nodes.rkyv               offsets.rkyv
p2f_adj.rkyv              text.json                text_emb_all-MiniLM-L12-v2.bin
```

Keep the raw source directory in place: `meta.json` records its `source`, and
availability validation reads the dataset/task manifests there. Regenerate
preprocessing if relocating the data rather than treating old absolute paths as
portable instructions.

The completed run reused an existing compatible preprocessing artifact. Fresh
native preprocessing may assign different node/relation IDs or embeddings;
byte-identical reproduction has not been demonstrated. Compare recorded input
hashes, query coverage and semantics rather than promising identical scores.

## 3. Validate, then run

```bash
uv run python experiments/run_clean_rewire.py --dry-run \
  --pre-dir artifacts/clean_rewire_preprocessed

uv run python experiments/run_clean_rewire.py \
  --pre-dir artifacts/clean_rewire_preprocessed \
  --checkpoint-dir artifacts/rt-plurel/classification \
  --device auto --inference-batch-size 8
```

Defaults are contexts 48/24 and 128/64, strengths 0/0.5/1, seeds 101/202/303,
and all 702 driver-DNF test queries. YAML configs are reference records, not
loaded inputs. `--ctx 48 --local-ctx 24 --fractions 1 --dry-run` checks only that
validation subset; full inference requires the complete matrix.

The full command repeats all-context validation before loading weights. Exit 0
means completion, 1 failure, and 2 unresolved temporal uncertainty. Reports are
explicitly incomplete after errors; do not analyze partial inference as a matrix.

The completed run used PyTorch 2.13.0+cu130, a GTX 1650 (4 GB), eager BF16 and
8-query inference microbatches. Model parameter storage was approximately
163 MiB. Inference took about 160 seconds at context 48 and 206 seconds at context
128; these are observed times, not hardware guarantees. CPU is supported but
its full-matrix performance is unmeasured. Microbatching leaves sampling/matching
unchanged; reducing its size can address activation-memory pressure.

## 4. Analyze and inspect artifacts

To reanalyze the versioned result set without loading RT:

```bash
uv sync --extra dev
uv run python experiments/analyze_preliminary.py \
  results/preliminary/2026-10-05_validated --output results/runs/curated_analysis
```

The output directory must be new; choose another name if it already exists.
For a newly generated experiment, use its printed run directory instead:

Every invocation prints a fresh `results/runs/preliminary_RUN_ID/` directory.
Large/local runs are ignored by Git. Each context contains a manifest, compact
validation report, validation exposure, predictions and post-inference exposure:

```text
matrix.json
ctx48_lctx24/ and ctx128_lctx64/
  validation.json
  validation_exposure_<arm>.npz
  preds_<arm>.npz          # labels, preds, node_idxs — each shape (702,)
  exposure_<arm>.npz       # query arrays and flattened relation records
  manifest.json
```

```bash
uv run python experiments/analyze_preliminary.py results/runs/preliminary_RUN_ID \
  --output results/runs/preliminary_RUN_ID/analysis
```

This creates `analysis.json` and `SUMMARY.md` after verifying saved pairing,
hashes, checkpoint consistency and accounting. The curated tracked result set
also supports this command; choose a fresh output path because existing evidence
is not overwritten. See its [README](../results/preliminary/2026-10-05_validated/README.md)
for the source-at-run patch and raw-run trace.
