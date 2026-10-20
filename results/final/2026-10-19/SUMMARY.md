# Structural reliance: final research summary

**Scope.** One frozen RT-PluRel classification checkpoint, four RelBench tasks,
three databases, contexts/local contexts 48/24 and 128/64, and matching seeds
101/202/303. The model is not trained. Within each panel the same queries, sampled
cells and common availability preparation are paired across conditions.
This is a controlled input-sensitivity study, not a globally valid database attack
or a confirmatory benchmark of relational foundation models generally.

The [artifact guide](README.md), [methodology](../../../docs/METHODOLOGY.md) and
[independently recalculated tables](recalculated/SUMMARY.md) provide the evidence
and definitions. Every table averages per-seed AUROCs, not predictions before AUROC.

## RQ1 — Does correct FK grounding matter?

### Complete F1 matrices

| Task | Queries | Context | Clean AUROC | Partial mean | Full mean |
|---|---:|---:|---:|---:|---:|
| driver-dnf | 702 | 48 | 0.5613 | 0.5610 | 0.5604 |
| driver-dnf | 702 | 128 | 0.6427 | 0.5893 | 0.6143 |
| driver-top3 | 726 | 48 | 0.5591 | 0.5326 | 0.5393 |
| driver-top3 | 726 | 128 | 0.7168 | 0.6761 | 0.6629 |

Changed grounding produces substantial raw-score movement even when AUROC changes
little. The larger driver-DNF context has clearer ranking damage, but partial
corruption is more damaging than full there. Whole-cycle selection changes relation
composition as well as count: partial/full is **not a clean dose-response**.
Driver-top3 extends the finding within F1, not to an independent database.

### Relation and task-cycle controls on driver-DNF

| Control | Context 48 AUROC | Context 128 AUROC |
|---|---:|---:|
| Clean | 0.5613 | 0.6427 |
| Task-row → driver | 0.5592 | 0.6143 |
| All database links | 0.5713 | 0.6426 |
| Database → driver | 0.5706 | 0.6416 |
| Database → constructor | 0.5614 | 0.6420 |
| Database → race | 0.5600 | 0.6431 |
| Race → circuit | 0.5613 | 0.6438 |
| Task cycles touching query | 0.5593 | 0.5598 |
| Task cycles excluding query | 0.5425 | 0.5845 |
| Visible labeled-support only | 0.5549 | 0.5964 |

Task-link corruption gives larger score shifts than database-link corruption in
these panels. This is a contrast between actual intervention regimes, **not
intrinsic importance per link**. At context 48 task corruption changes 2,960 links
and affects 702 queries, versus 1,172 links and 322 queries for all database links.
At context 128 those budgets are 4,177 versus 10,961 links, both covering 702 queries.
The zero-exposure circuit control at context 48 is an identity control, not evidence
that circuit grounding is unimportant when exposed.

Query-touching cycles also move other rows. Excluding-query cycles contain both
visible- and hidden-label rows. The two cycle effects are nonadditive: separately
they can cause more AUROC damage than their union. It is invalid to sum their
effects or call either an isolated query-edge/visible-support intervention.
See [relation statistics](relation_analysis/SUMMARY.md) and the [per-seed budget
table](recalculated/SUMMARY.md); the relation table's affected-query count is the
union across seeds, whereas the recalculated table lists each seed separately.

### Strict visible-labeled-support control

Fresh replay confirms the old excluding-query control changed 1,954 visible-label
and 1,631 hidden-label links at context 48, and 5,278/2,312 at context 128, summed
over three seeds. These are edge instances, not independent prediction examples.

The new control rematches only non-query rows whose finite label cell is retained
after availability preparation. Independent replay finds **zero changed query
links and zero hidden-label links in every arm**.

| Context | Changed links/seed | Affected queries/seed | Mean abs raw-score shift | Mean AUROC delta | Crossed date/driver 95% interval |
|---|---:|---:|---:|---:|---|
| 48 | 451 | 158/702 | 0.0525 | −0.0064 | [−0.0463, 0.0290] |
| 128 | 1,552 | 390/702 | 0.1110 | −0.0463 | [−0.0973, −0.0033] |

All three per-seed AUROCs are below clean in both contexts; the context-48
cluster-resampling interval includes zero. Correct entity grounding of visible
labeled-support rows therefore affects predictions with the query link fixed,
with clearer performance evidence at context 128. This is **support-row grounding**,
not label-value-only causation: dates and other row features share the link, and
fanout is allowed to change.

Coverage explains part of the full-panel shift contrast. On the 158/390 affected
queries, mean absolute shifts are 0.2332/0.1997: the larger context does not have
the larger affected-subset shift. The two contexts have different affected subsets.
Fresh matching and different parent pools prevent attributing any percentage of
the old mixed effect to visible versus hidden labels. Strict-versus-old-control
AUROC intervals include zero in both contexts. The numerical contrasts and subset
metrics are in [analysis.json](recalculated/analysis.json), with classified link
counts in [support_link_audit.json](support_link_audit.json).

### Cross-database panels

Each panel uniformly selects 512 official test queries, seed 20261007. ArXiv has
225 positive outcomes and Trial 292; each has one query date and 512 entities.

| Database/task | Context | Clean AUROC | Task-link mean | Database-link mean | Task / database mean abs raw-score shift |
|---|---:|---:|---:|---:|---|
| ArXiv/paper-citation | 48 | 0.6401 | 0.6126 | 0.6401 | 0.1104 / 0.0175 |
| ArXiv/paper-citation | 128 | 0.6920 | 0.6595 | 0.6896 | 0.1530 / 0.0269 |
| Trial/study-outcome | 48 | 0.5489 | 0.5393 | 0.5414 | 0.0131 / 0.0022 |
| Trial/study-outcome | 128 | 0.5360 | 0.5048 | 0.4904 | 0.0220 / 0.0040 |

The task-greater-than-database **raw-score sensitivity ordering** recurs in both
databases. The **AUROC damage ordering does not**: Trial context 128 has worse
AUROC under database corruption despite smaller score movement. Its task-minus-
database AUROC contrast is +0.0144, with conditional entity interval
[−0.0384, 0.0660]. Budgets differ: ArXiv context 128 changes 4,588 task versus
21,799 database links; Trial context 128 changes 1,497 versus 14,099. These are
not matched-dose comparisons, and raw-score magnitudes are not calibrated across
tasks. [Cross-database analysis](cross_database_analysis/SUMMARY.md) links to the
saved contrasts. Date uncertainty is non-estimable, not zero, on these single-date
panels. An H&M attempt stopped during preparation for resource limits and produced
no predictions; it is not a completed replication or a negative result.

## RQ2 — What predicts sensitivity?

Context size, sampling composition and available matching pools vary jointly;
larger contexts do not identify a causal context-size effect. Corruption budgets
and coverage are reported explicitly rather than normalized into a causal
“effect per edge.”

Exploratory fixed-ridge models predict log1p(seed-mean absolute score shift) on
8,087 affected query/control records from the original eight F1 relation/cycle
controls. All repetitions of held-out dates/drivers stay in the held-out fold;
scaling fits within training folds, with no tuning.

| Features added to group/context | Date-held-out R² | Driver-held-out R² |
|---|---:|---:|
| None | 0.0995 | 0.0964 |
| Context-only exposure/composition | 0.1011 | 0.1058 |
| Assignment-informed exposure/composition (historical fit) | 0.1005 | 0.1057 |
| Support count alone | 0.1087 | 0.1056 |
| Clean-score magnitude alone | 0.3026 | 0.3007 |
| Assignment-informed exposure, support and clean score (historical fit) | 0.3038 | 0.3026 |

Context-only exposure excludes `targeted_maximum`: that variable is an
assignment-informed budget from the solved matching, equal to the changed-edge
count at full strength, not an untouched-context characteristic. Historical fits
including it retain their numbers under corrected names. All specifications use
an affected-record cohort selected after assignment; “context-only” describes
predictors, not ex ante selection. A name-to-name mapping and the unchanged
[historical analysis](relation_analysis/pre_review/analysis.json) preserve traceability.

Measured context exposure/support counts add little predictive power in this
specification; clean-score magnitude adds more R² but not a comparable improvement
in absolute error. Clean-score magnitude is mathematically coupled to
`|corrupted - clean|` through the shared clean score; its gain is not independent
mechanism evidence. This is model-response prediction on one panel, not causal attribution,
proof of an exposure law, outcome prediction or external-task validation. Fanout
diagnostics are post-intervention and are reported separately. Full pooled and
within-control models, baseline error metrics and intervals are saved in
[relation_analysis/analysis.json](relation_analysis/analysis.json).

## Validity, fixes and remaining limits

- A compiled Rustler regression verifies the timestamp fix for temporal tables
  without outgoing FKs. Completed replications use rebuilt artifacts, not the
  pre-fix missing-time data. No affected completed inference needed rerunning.
- Scoped embedding coverage, encoder and actual file hashes were checked. The
  missing file-to-scope hash gate was added during finalization; recorded files
  matched, so saved results remain valid. A preparation/replay workflow is now
  versioned. Unused embedding rows remain NaN; scope is not general-purpose.
- Availability masking is common across arms but follows sampling. Retained
  temporal cells and mature rolling labels obey the declared rules; retrieval
  itself is not established historically leakage-free, and static schema metadata
  is not proof of historical attribute availability.
- Three fixed seeds, one checkpoint, post-exploration controls, overlapping outcome
  windows, single-date replications and no multiple-testing correction limit
  uncertainty interpretation. Bootstrap intervals are conditional descriptive
  checks, not confirmatory significance or population-generalization claims.
- FK incidence changes fanout/normalization as well as entity association.
  No label-only, pure content, globally valid database or universal RFM claim is made.
- Follow-up provenance retains source digests rather than exact per-run patches;
  see the [artifact guide](README.md). The [redistribution review](../../../docs/REDISTRIBUTION.md)
  distinguishes today's Apache-2.0 RT source from the older unlicensed pinned
  tree, and conflicting pinned collection labels from source-database terms.
  The classification checkpoint is CC BY-NC-SA 4.0, not the MIT `paper/` variant.
  Applicable source/data permissions and compliance remain uncleared; a public
  commit containing the current vendored/data-derived material is not approved.

**Conclusion:** RT-PluRel uses relational grounding in these fixed-context panels,
including the grounding of visible labeled support. Sensitivity depends on the
intervention regime; ranking damage is not monotone in requested corruption or
raw-score movement. The experiments support a bounded structural-sensitivity
claim, not a general ranking of relation importance or a causal exposure mechanism.
