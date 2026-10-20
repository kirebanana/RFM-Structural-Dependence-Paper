# Final scientific and implementation audit

**Amended after independent review, 2026-10-20.** The original validation count
and findings below describe the initial finalization. Review corrections and
their separate source/inventory lineage are recorded in
[review_amendment.json](review_amendment.json); no original prediction, exposure,
run identity or historical source snapshot was replaced.

Baseline: `81e4fa4fdb98ea6ce5973ec4772f91dc4ca9a468`. Scope includes tracked
working-tree changes, untracked research code/tests, completed saved runs and
earlier research reports. Original data, runs and research work were preserved.
No new model inference, staging, commits, pushes or Git identity changes were made.

## Findings and disposition

| Finding | Disposition and result impact |
|---|---|
| Scoped embedding gates checked finite values/coverage but did not reconcile the actual file hash with its scope manifest | Added SHA-256/byte-count and pinned encoder/dtype reconciliation before sampling/inference. Regression rejects a same-sized finite but different embedding file. Both recorded files matched; no completed results invalidated. |
| Scoped preparation procedure was not versioned | Added an exclusive/no-overwrite preparation command and coverage-only audit. Reconstructed exact covered IDs, both context fingerprints and finite/NaN layout for ArXiv/Trial. Did not re-encode; byte-identical encoder regeneration remains unproven. |
| Old excluding-query control mixed visible/hidden-label source rows | Retained its original name/results. Fresh CPU replay reproduces saved exposure and classifies the mix; a distinct restricted-source control has zero changed query/hidden-label links. No subtraction-based attribution. |
| Eligible/singleton counts in restricted controls describe unrestricted strata | Kept counter semantics unchanged. Documented denominators; separate replay arrays report selected visible-pool sizes, singleton pools and distinct parents. |
| Relation/cycle groups and context settings have unequal budgets/coverage | Published per-seed changed-edge and affected-query counts. Clarified union-over-seeds coverage in the relation analyzer's table header. No per-edge causal ranking or matched-dose claim. |
| Trial temporal entities without outgoing FKs lost native row timestamps | Reviewed existing Rust fix and ran compiled-native regression. Completed replication uses rebuilt, hashed artifacts. No reported run used the rejected missing-time data. |
| Earlier “final” extension report predates cross-database/support-only evidence | Preserved it with a historical/superseded notice outside the repository; the public summary uses the complete evidence hierarchy. Private reports are not included in the release. |
| Follow-up runs retain dirty-source digests rather than complete historical source patches | Retained original digests and explicit limitation. Captured finalization source separately, not relabeled as exact source-at-run. |

## Independent-review corrections

- **RQ2 classification:** `targeted_maximum` is assignment-informed, not context-only.
  Corrected names explicitly map to unchanged historical fits (all 96 reproduce).
  Added context-only comparisons; exposure/composition gives date/driver R²
  0.1011/0.1058. Disclosed clean-score/response mathematical coupling and the
  post-assignment affected-record selection. Original output bytes remain in
  `relation_analysis/pre_review/`.
- **Standalone verification:** inventory is mandatory/exhaustive; fixed catalog,
  panel sizes/labels/dates, contexts, seeds, exact arm sets and pinned checkpoint
  are enforced independently of supplied manifests. All formats receive per-query
  temporal identities, availability and held-fixed comparisons, including every
  saved aggregate where available. Compact historical JSON is not treated as
  though it contained missing aggregates. Rehashed incomplete/misdescribed releases,
  forged relation semantics and arithmetic corruption are covered by regression tests.
- **Portable relation semantics:** exact ArXiv/Trial registry bytes match original
  preprocessing SHA-256/byte identities. Group membership is reconstructed from
  child/parent semantics, selected task entity metadata and declared database tables.
  Independent re-review additionally identified a false-acceptance path for
  broadened F1 group lists. It is closed by reconstructing all F1 memberships
  from the primary-matched registry, including task-only cycle/support scopes;
  regression tests exercise broadened, swapped and non-task control groups.
- **Publication permissions:** authoritative-source findings and safe options for
  the existing public repository are documented in
  [REDISTRIBUTION.md](../../../docs/REDISTRIBUTION.md). Technical correctness is
  not permission; no repository visibility, history or published content was changed.
  The archived Ergast database page's own optional-attribution qualification is
  recorded alongside its CC declaration rather than silently omitted.

## Independent verification

- All **110 prediction files**: sklearn AUROC agrees with project rank-sum AUROC
  and original manifests within absolute `1e-12`. Verified hashes, target/label
  pairing, shared checkpoint identity, validated coverage, exposure accounting,
  availability removals and zero unresolved temporal status. Queries without
  changed edges retain exactly their cached clean scores.
- Recomputed complete driver-DNF and driver-top3 matrices, all six relation
  groups, both task-cycle controls, ArXiv/Trial task-versus-database contrasts and
  strict labeled-support controls.
- Recomputed original-run and curated-file analyses: **2,366 numeric fields**
  agree within `1e-12` for relation/RQ2, cross-database and driver-top3 analyses.
  Separately reproduced every saved support-only paired-bootstrap contrast.
- Reviewed paired rank-credit resampling against weighted sklearn AUROC;
  held-out grouping/scaling keep repeated date/driver records together. RQ2 R²
  remains predictive association, not causal evidence.
- Fresh support replay checks exact sampled-input fingerprints and every
  per-query changed-edge count. Embedding replay verifies 34,515 ArXiv and 12,084
  Trial covered strings, with the recorded encoder/file identities.
- Curation checks original hashes and restored array values/dtypes after
  compression. Original runs remain unchanged; duplicate exposures and large
  intermediate/model/data files are excluded.

## Validation and publication boundary

The full local suite passed **153 tests**, including the compiled native temporal
regression, finite-wrong-embedding rejection, restricted-support cases, preparation
coverage/publication tests and portable saved-artifact checks. Ruff and
`git diff --check` passed. The release inventory provides per-file hashes;
`tests/test_final_artifacts.py` recalculates the recorded final metrics.
All four documented saved-array analysis commands also passed with imports of
RT, torch, pandas and sentence-transformers deliberately blocked, confirming
that the curated reanalysis path does not need the inference/data runtime.

No statistical or intervention error requiring new inference was identified.
The [summary](SUMMARY.md) retains nonadditivity, selective partial matching,
post-sampling temporal filtering, one-checkpoint/three-seed scope and single-date
replication limitations. Before publication, a human should review the scientific
wording, third-party licensing/redistribution rights and the explicitly incomplete
historical source/bitwise-reproduction provenance.
