# Setup and reproduction

This guide takes you from a fresh checkout to prepared data, validation, and
inference. For the curated research release, start with
[final evidence reanalysis](#final-evidence-reanalysis); no model or dataset is needed.

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

## Exploratory follow-up controls

The follow-up runner changes only selected FK relations while replaying the
same driver-DNF contexts and checkpoint as the curated panel. It reuses verified
clean predictions; no-change controls need no model forward. Relation restrictions
keep the matching RNG identical to all-relations corruption.

```bash
uv run python experiments/run_relation_controls.py \
  --pre-dir artifacts/clean_rewire_preprocessed \
  --checkpoint-dir artifacts/rt-plurel/classification
```

Defaults compare task-row/driver links and grouped database driver, constructor,
race and circuit links. Add `--dry-run` for validation only. To compare complete
task-matching cycles touching the query versus cycles confined to other rows,
use `--groups task_query_cycles,task_other_cycles`. These are exploratory controls,
not single-edge replacements or a uniform corruption dose-response experiment.

For a strictly visible-labeled-support control, use
`--groups labeled_support_only`. A support source row must have a retained finite
prediction-column label cell after the common availability mask and must not be
the prediction query. Matching is recomputed within that source pool; query links,
hidden-label source links, other relations and cell values stay fixed. Zero- and
one-row pools are identity controls. This is a fresh restricted matching, not a
subset of `task_other_cycles`: the latter can change both visible- and hidden-label
rows. Their effects cannot be subtracted to attribute a fraction to labels.
The changed FK also grounds dates and other features on the support row, so this
control isolates support-row grounding rather than label-value use alone.

Analyze the two printed run directories together:

```bash
uv run python experiments/analyze_relation_controls.py \
  results/runs/relation_controls_FIRST results/runs/relation_controls_SECOND \
  --raw-dir artifacts/raw/rel-f1 --output results/runs/relation_analysis
```

The analysis checks pairing/hashes and reports cluster-resampling sensitivity
intervals and date/driver-held-out score-sensitivity models. Groups differ in
actual exposure; reported associations are not causal attribution. Raw task
metadata is needed to identify repeated dates/drivers, but no inference runs.

The matrix runner also supports `--task driver-top3` as a same-database
replication. Its full test count comes from preprocessing metadata (726 in the
recorded snapshot), not the driver-DNF count. It applies the same availability
policy with task-manifest label horizons, saves a separate run, and uses the
existing saved-array analyzer. A second task in rel-f1 is not cross-database
replication.

## Frozen cross-database panels

`experiments/run_database_replications.py` supports `rel-arxiv/paper-citation`
and `rel-trial/study-outcome`. It selects 512 official test rows uniformly with
seed 20261007, then uses contexts 48/24 and 128/64 and clean/task-entity/database
conditions with matching seeds 101/202/303. It does not fit additional models or
search corruption strengths. Database tables are intact; preprocessing may omit
unrelated task-label tables.

```bash
uv run python experiments/run_database_replications.py --database rel-arxiv \
  --pre-dir artifacts/replication_preprocessed \
  --checkpoint-dir artifacts/rt-plurel/classification
```

Use `--dry-run` to validate only; use `--database rel-trial` for the second panel.
Source manifests and compatible patched native data are required as above.
After changing Rust preprocessing code, rebuild the editable extension with
`uv sync --extra dev --extra rt --reinstall-package relational-transformer`.
The native regression test checks that temporal entity rows retain their declared
timestamps even when the table has no outgoing FK columns.

The recorded laptop runs used scoped text embeddings: every text/schema lookup
in both frozen contexts was encoded with pinned MiniLM-L12-v2, while unused rows
were marked NaN. `embedding_scope.json` identifies selected query IDs, covered
text IDs, encoder identity, and non-embedding context fingerprints. The runner
rejects scope mismatches or any nonfinite embedding; these artifacts must not be
used as complete general-purpose embeddings for a different query panel. Fully
encoded ordinary RT artifacts are also supported.

### Preparing the recorded cross-database configuration

Download `rel-trial` from `stanford-star/relbench-v1` at
`d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600`, and `rel-arxiv` from
`stanford-star/relbench-v2-extra` at `190016ae95ac78362c331e23f072ca74bcbed3a2`.
Keep the complete database tables but only the selected task directory and its
train/val/test splits in each raw root. Do this by downloading scoped paths to
fresh directories, not by deleting task files from an existing dataset:

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download

for repo, revision, database, task, destination in [
    ("stanford-star/relbench-v1", "d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600",
     "rel-trial", "study-outcome", "artifacts/raw_trial_panel"),
    ("stanford-star/relbench-v2-extra", "190016ae95ac78362c331e23f072ca74bcbed3a2",
     "rel-arxiv", "paper-citation", "artifacts/raw_arxiv_panel"),
]:
    snapshot_download(repo, repo_type="dataset", revision=revision,
        allow_patterns=[f"{database}/manifest.yaml", f"{database}/db/*",
                        f"{database}/tasks/{task}/*"], local_dir=destination)
snapshot_download("sentence-transformers/all-MiniLM-L12-v2",
    revision="a50ef00143b4d5391434df20ae11632588ac25be")
PY

uv sync --extra dev --extra rt --reinstall-package relational-transformer
uv run python - <<'PY'
from pathlib import Path
from rt._rustler import preprocess

for source in ("artifacts/raw_trial_panel/rel-trial", "artifacts/raw_arxiv_panel/rel-arxiv"):
    raw = Path(source).resolve()
    output = Path("artifacts/replication_preprocessed_fresh").resolve()
    if (output / raw.name).exists():
        raise FileExistsError("use a fresh native output directory")
    preprocess(str(raw), str(output), source=str(raw), skip_tasks=False)
PY

uv run python experiments/prepare_replication_embeddings.py \
  --pre-dir artifacts/replication_preprocessed_fresh --database rel-arxiv --device cpu
uv run python experiments/prepare_replication_embeddings.py \
  --pre-dir artifacts/replication_preprocessed_fresh --database rel-trial --device cpu
```

The preparation command refuses to replace embeddings or scope metadata. It
reconstructs coverage with exact BF16 lookup-ID markers in an isolated temporary
native-data view, then encodes the union using the cached pinned FP32 encoder and
stores BF16 vectors. Unused rows are NaN. Use `--device cuda` if desired. Encoding
may differ slightly across hardware/software; this workflow does not establish
byte-identical historical encoder regeneration. New native node/relation IDs may
also differ; evaluate semantic panel selection and recorded identities separately.

For existing recorded artifacts, validate coverage and identity without encoding
or model inference:

```bash
uv run python experiments/prepare_replication_embeddings.py \
  --pre-dir artifacts/replication_preprocessed --database rel-arxiv --check-existing
uv run python experiments/prepare_replication_embeddings.py \
  --pre-dir artifacts/replication_preprocessed --database rel-trial --check-existing
```

Recorded coverage is 34,515 ArXiv and 12,084 Trial strings. Both exact ID sets and
non-embedding context fingerprints reproduced during finalization. Hash checks
authenticate the actual embedding files against scope manifests and the pinned
encoder identity; finite vectors alone are insufficient provenance.

```bash
uv run python experiments/analyze_database_replications.py \
  results/runs/replication_rel-arxiv_RUN results/runs/replication_rel-trial_RUN \
  --output results/runs/cross_database_analysis
```

This checks saved pairing/hashes and reports paired score/AUROC contrasts. These
official test panels have one date: date-resampling uncertainty is not estimable,
and entity-resampling intervals are conditional on that snapshot and chosen seeds.

## Final evidence reanalysis

The [final release](../results/final/2026-10-19/README.md) combines the unchanged
primary panel with compact follow-up evidence. All following analysis commands
use only versioned files; select new output paths if already present.

```bash
uv sync --extra dev
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 uv run python experiments/analyze_final_results.py \
  results/final/2026-10-19 --output results/runs/recheck_final

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 uv run python experiments/analyze_relation_controls.py \
  results/final/2026-10-19/relations results/final/2026-10-19/task_cycles \
  --panel results/final/2026-10-19/relations/panel.npz \
  --output results/runs/recheck_relations

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 uv run python experiments/analyze_database_replications.py \
  results/final/2026-10-19/arxiv results/final/2026-10-19/trial \
  --output results/runs/recheck_cross_database

uv run python experiments/analyze_preliminary.py results/final/2026-10-19/driver_top3 \
  --output results/runs/recheck_top3
```

`analyze_final_results.py` uses sklearn AUROC independently of the project
rank-sum implementation, verifies 110 prediction files and reconciles hashes,
pairing, common checkpoints, availability/temporal evidence, unchanged-query
predictions and classified support-link exposure. It reports actual per-seed
changed-link counts and affected-query coverage, alongside support-only paired
cluster-resampling contrasts. Other commands reproduce the RQ2 models and
cross-database uncertainty calculations, not just summary AUROCs.

The standalone command now requires a mandatory, exhaustive inventory and the
fixed release's exact runs, contexts, seeds, task counts and arm sets. It rejects
a coherently re-inventoried incomplete panel rather than calling it verified.
All saved validation formats undergo the same per-query temporal identities and
common availability/held-fixed checks; older compact JSON is checked against its
complete NPZ fields and verified clean reference where aggregates were not saved.
Its guarantee is saved-evidence consistency/completeness, not independent proof
of inference execution, historical source reconstruction or redistribution rights.

ArXiv/Trial `relation_index.json` files preserve exact original bytes and match
the recorded preprocessing hashes/byte counts. The verifier independently
reconstructs task/entity versus database groups from the registry, task manifest
entity table and availability-policy database tables. No local raw/native files
are required to inspect group semantics.
F1 relation-group memberships are likewise reconstructed from the saved registry
and must match the primary registry; cycle/support groups must be exactly the
task-driver relation set. This rejects semantically incorrect supersets even if
the observed changed edges happen to fall within them.

The corrected RQ2 output separates context-only exposure, assignment-informed
budgets and post-intervention diagnostics. Historical fits retain exact numbers
under explicit name mappings; unchanged prior outputs remain in
`relation_analysis/pre_review/`. Clean-score magnitude shares the clean term with
the response `|corrupted - clean|`, and all RQ2 cohorts remain selected on realized
exposure. Reanalysis does not turn these associations into causal mechanisms.

The support audit can be regenerated from the original compatible F1 native
artifacts on CPU, without loading the model:

```bash
uv run python experiments/audit_support_links.py \
  --pre-dir artifacts/clean_rewire_preprocessed --release results/final/2026-10-19 \
  --output results/runs/recheck_support_links.json
```

It reconstructs both visible-support-only and excluding-query assignments,
checks exact sampled fingerprints and saved per-query changed-edge counts, and
classifies source slots using retained finite label cells. The JSON contains
aggregate and selected-pool diagnostics; its compressed NPZ sidecar contains all
per-query counts. Existing audit outputs are never overwritten.

### Recreating the curated copy from local completed runs

Only needed for evidence maintenance, not metric reproduction:

```bash
uv run python experiments/curate_final_results.py --source results/runs \
  --output results/final/recurated \
  --f1-raw artifacts/raw/rel-f1 --arxiv-raw artifacts/raw_arxiv_panel/rel-arxiv \
  --trial-raw artifacts/raw_trial_panel/rel-trial
```

The script explicitly selects the six completed run IDs listed in `catalog.json`.
It verifies source artifact hashes, compresses arrays losslessly, checks restored
dtype/values, exports aligned minimal panel metadata with raw-source hashes, and
retains original-to-curated identities. It does not copy interrupted/dry runs,
duplicate exposures, weights, raw data or private reports. It uses fresh output
directories and never modifies the source runs. Generate the replay audit and
analysis outputs separately with the commands above, adjusting the release path.
The unchanged primary result set remains a relative reference, not a duplicate.

After final edits, `uv run python experiments/finalize_result_inventory.py
results/final/RELEASE_NAME` records the current source delta/identity and checksums.
This describes finalization source, not an invented exact historical snapshot.
The command refuses to overwrite an existing inventory or source snapshot; create
a new release or explicitly archive only your own generated inventory first.
For an audited amendment, preserve the original inventory and source snapshot,
then pass `--amendment review_2026_10_20` to create a separately named implementation
snapshot. Record the previous inventory hash and changes, and generate the new
inventory only after all analysis/documentation edits are complete. The amended
2026-10-19 release preserves its prior inventory at
`provenance/pre_review_checksums.json` and records the correction in
`review_amendment.json`; prediction/exposure arrays and original run identities
are unchanged.

Final validation:

```bash
uv run pytest -q
uv run ruff check .
git diff --check
git diff --cached --stat
```
