# Availability-filtered preliminary results — 5 October 2026

This is the completed **14-arm, 702-query RT-PluRel / rel-f1 driver-DNF** matrix
under `query_available_incidence_v1`. It is separate from the earlier pilot in
the sibling `ctx48_lctx24/` and `ctx128_lctx64/` directories, which used unfiltered
contexts and did not pin its checkpoint revision.

Read [SUMMARY.md](SUMMARY.md) for all seed results and paired score statistics.
At context 48 the mean AUROC changes are small and vary in sign across seeds;
at context 128 all rewired conditions reduce AUROC, with **50% requested strength
hurting more than 100%**. No monotonicity, significance or causal context-size
claim is made.

## What is retained

- `analysis.json`, `SUMMARY.md`: descriptive per-arm and context/strength summaries.
- `matrix.json`: completed matrix and common availability policy.
- `ctx*/manifest.json`: compact configuration, completion, validation, edge counts,
  FK metadata, checkpoint identity and artifact hashes.
- `ctx*/preds_*.npz`: all 14 aligned prediction vectors; labels, raw scores and
  target node IDs, each shape `(702,)`.
- `ctx*/exposure_*.npz`: per-query and relation-level accounting, requested/realized
  strengths, availability removals, temporal status and fanout diagnostics.
- `provenance.json`: original run ID, input hashes, sampling/software settings,
  source identity, validation and raw-data/checkpoint revisions.
- `source_identity.json`, `source_at_run.patch`: exact runtime source identity
  and patch against recorded base commit `a2373fb5cff7581d9171b9127a1f1b4f751d6c6e`.
- `checksums.json`: sizes and SHA-256 hashes of curated evidence files.

NPZ bytes are unchanged from the raw run. Duplicate pre-inference exposure files,
logs, smoke runs, failed runs, model weights and database artifacts are not
included. Validation replay verified the retained exposure against pre-inference
exposure in the original run. Local absolute paths are omitted from the curated
policy; content hashes and run IDs are preserved.

## Configuration and validation

Frozen checkpoint: `stanford-star/rt-plurel/classification`, Hub revision
`2ed96efc8007c57d589a31ceb435eb768491c439`, 85,565,091 stored scalars; weight SHA-256
`4a369f065d8f69e1da11ee4326ceb9d32cf78ee69255a438449e971f681b7f83`.

Contexts/local contexts: 48/24 and 128/64. One clean arm plus 50%/100% requested
strength at seeds 101, 202, 303. Matching is computed before 8-query CUDA BF16
microbatches. Requested strength refers to maximum-matching changed IDs, not
all eligible FK links. Intermediate cycles achieve about 42.0% of maximum at
context 48 and 37.5% at context 128.

Every context contains exactly 702 distinct aligned queries. All arms passed
hard structural, input-equality, replay and exposure checks. Retained parent
comparisons have zero known violations and zero unresolved times. Each context
masked 55 future cells and 1,379 unclosed support-label cells before making arms,
without refill. Fanout changes are measured incidence effects, not held fixed.

## Reanalyze without RT

From the repository root:

```bash
uv sync --extra dev
uv run pytest -q tests/test_preliminary_artifacts.py
uv run python experiments/analyze_preliminary.py \
  results/preliminary/2026-10-05_validated --output results/runs/curated_analysis
```

The test verifies curated checksums and independently recomputes paired metrics
and accounting. The analysis command requires a fresh output directory.

## Trace to the execution source

Original local evidence: `results/runs/preliminary_1791191647758643138/` (ignored,
not promised to exist in a fresh clone). The runtime worktree was modified atop
the recorded base commit; **the base commit alone is not the execution source**.
To inspect that source, use a separate checkout at the base and run
`git apply --unidiff-zero source_at_run.patch` using the artifact's full path.
It is a zero-context patch; `source_identity.json` lists its runtime file hashes.
The milestone commit also contains later review/docs cleanup; this patch
preserves the original execution source independently of those changes.

Temporal validity is conditional on benchmark event dates and explicitly static
covariates. Mature rolling labels are allowed; this is not train-only context.
Sampling precedes filtering, so retrieval selection is not proven historically
leakage-free. Fanout/normalization is part of the structural effect. One task,
one checkpoint and three seeds do not establish general RFM behavior.
