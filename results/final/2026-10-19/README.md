# Final evidence release — 2026-10-19

Start with [SUMMARY.md](SUMMARY.md) for findings and limitations. This release
curates completed follow-up runs after baseline `81e4fa4`; it does not replace or
modify the [primary validated F1 evidence](../../preliminary/2026-10-05_validated/README.md).
No new inference was performed during finalization.
The [audit record](AUDIT.md) documents problems, fixes and independent checks.

## Contents

| Directory/file | Evidence |
|---|---|
| `catalog.json` | Selected original run IDs, task identities and clean references |
| `relations/` | Six relation-specific controls, two contexts, three seeds |
| `task_cycles/` | Query-touching and query-excluding complete-cycle controls |
| `labeled_support/` | Strict visible-labeled-support source-pool control |
| `driver_top3/` | Complete same-database clean/partial/full replication |
| `arxiv/`, `trial/` | Frozen 512-query cross-database panels |
| `recalculated/` | Independent AUROC recalculation, score shifts, budgets and support contrasts |
| `relation_analysis/` | Cluster-resampling sensitivity checks and grouped RQ2 models |
| `relation_analysis/pre_review/` | Unmodified historical RQ2 output before feature classification was corrected |
| `cross_database_analysis/` | Paired cross-database contrasts and conditional entity intervals |
| `support_link_audit.json`, `.npz` | Fresh CPU replay classification of visible, hidden and query links |
| `data_sources.json`, `preparation_verification.json` | Pinned download identities and scoped-embedding coverage/hash checks |
| `arxiv/relation_index.json`, `trial/relation_index.json` | Exact registry bytes authenticated by original preprocessing hashes; portable group reconstruction |
| `checksums.json` | SHA-256 and byte counts for every release file except this inventory itself |
| `source_at_finalization.patch`, `source_identity.json` | Final source delta/identity against the baseline, not reconstructed historical source |
| `source_review_2026_10_20.patch`, `source_review_2026_10_20_identity.json` | Amended implementation snapshot; the earlier source snapshot is preserved |
| `source_review_2026_10_20_followup.patch`, `source_review_2026_10_20_followup_identity.json` | Latest amendment, including independently reviewed F1 semantic-group enforcement |
| `review_amendment.json`, `provenance/pre_review_checksums.json` | Review correction record and the exact previous release inventory |

Each run contains its scientific state file (`controls.json`, `matrix.json` or
`replication.json`), context manifests/validation and `panel.npz`. Panel metadata
contains only query IDs, binary labels, dates and entity IDs, extracted in saved
prediction order from official test files. `panel_source.json` records their
source-file identities and task manifest. Saved-array analysis needs no raw
database, GPU, native extension, text embeddings or model weights.

## Curation and provenance

- There are **110 prediction files** across the primary and follow-up panels:
  14 in the unchanged primary release and 96 here. All were independently
  recalculated with sklearn AUROC and compared to the project's separate rank-sum
  implementation and saved manifest values to absolute tolerance `1e-12`.
- Predictions/exposures were losslessly ZIP-compressed. Curation checked original
  manifest hashes, then dtype and array equality (including NaNs) after compression.
  `artifact_lineage.json` records original and curated hashes; curated manifests
  reference the new compressed file hashes. Compression changes bytes, not arrays.
- Only the exposure file referenced by each completed arm is retained. Duplicate
  validation/inference copies, interrupted/dry runs, raw tables, weights, native
  artifacts, logs and private research reports are excluded. Original local runs
  remain untouched under ignored `results/runs/`.
- Absolute machine paths in copied JSON are reduced to basenames. Content hashes,
  source-run IDs, context fingerprints, checkpoint identity and scientific metadata
  remain. The original absolute locations are not needed to analyze this release.
- Original follow-up runs record a baseline revision plus dirty-source digest,
  rather than a complete per-run source patch. The finalization patch makes the
  current implementation inspectable but cannot prove byte-identical historical
  source reconstruction. This provenance limitation is explicit; no historical
  source snapshot has been invented. The primary release retains its own exact
  source-at-run patch.
- Scoped embedding files are excluded, but replication state files retain their
  coverage IDs, pinned encoder metadata, context fingerprints and file hashes.
  Replaying preparation requires external data/native preprocessing; metric
  reproduction does not.

## Recalculate

From the repository root, choose fresh output paths:

```bash
uv sync --extra dev
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 uv run python experiments/analyze_final_results.py \
  results/final/2026-10-19 --output results/runs/recheck_final
```

This checks release hashes, source-link audit accounting, checkpoint equality,
pairing, validation coverage, availability/temporal accounting, unchanged-query
scores and all AUROCs, then recomputes the support-only paired bootstrap contrasts.
The inventory is mandatory and exhaustive. The verifier enforces the fixed six
follow-up panels plus primary panel, exact arm/seed/context sets, task counts and
pinned checkpoint; a coherently rehashed subset cannot pass as the full release.
It checks per-query temporal identities and common availability fields in every
validation format. Older compact control JSON lacks some aggregate fields;
those are reconciled from complete NPZ fields and checked against the verified
clean reference, not fabricated as historical aggregate evidence. This remains
saved-evidence verification, not proof of sampler/model execution or permissions.
Cross-database group membership is reconstructed from the authenticated registries,
task entity-table metadata and ordinary-database ranges in the availability policy.
F1 relation controls are reconstructed with the documented semantic predicates;
cycle/support groups must equal the task-driver relation set. Control registries
must match the verified primary registry. A rehashed broadened/swapped group
cannot pass merely because all observed changes fit within its supplied list.
See [reproducibility](../../../docs/REPRODUCIBILITY.md#final-evidence-reanalysis)
for grouped RQ2 models, cross-database intervals, curation and replay commands.

Upstream source/checkpoint and dataset redistribution rights still require human
review; see [third-party notices](../../../THIRD_PARTY_NOTICES.md) and the
[redistribution assessment](../../../docs/REDISTRIBUTION.md). No model weights,
embeddings or raw source tables are included in this evidence release.
