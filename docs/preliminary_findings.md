# Preliminary Findings

The preserved historical inference results evaluated 702 `rel-f1/driver-dnf`
test queries with RT-PluRel. They used context/local-context pairs `(48, 24)`
and `(128, 64)` and rewiring seeds 101, 202, and 303. Following migration, the
real-data rewiring and invariant dry-run was freshly verified for ctx48; a
ctx128 dry-run is being run separately. No post-migration model inference has
been performed, so the AUROC values below are preserved historical evidence,
not fresh migrated-code inference results.

| Context | Base | Rewire 101 | Rewire 202 | Rewire 303 | Mean delta |
|---:|---:|---:|---:|---:|---:|
| 48 | 0.6150 | 0.6052 | 0.6072 | 0.6003 | -0.0108 |
| 128 | 0.6869 | 0.6217 | 0.6250 | 0.6272 | -0.0623 |

Changing valid FK parent identities while retaining the sampled context,
support labels, relation identity, and attention-mask counts reduced AUROC in
all six comparisons. This is preliminary evidence that RT-PluRel uses parent
content, not only the existence or degree pattern of relational links.

## Changed-edge counts

“Changed” counts eligible FK edge instances whose assigned parent identity
differs after rewiring; it does not count test examples or unique query rows.
“Eligible” counts present, in-context FK edge instances considered for
rewiring, including singleton strata that cannot change. Per seed, ctx48 had
4,136 changed / 6,766 eligible edge instances (2,112 singleton edge instances);
ctx128 had 15,163 / 20,294 (1,422 singleton edge instances). The displayed
historical total counts in `results/preliminary/summary.md` sum those counts
across all three seeds. The rewirable fraction excludes singleton edge
instances from its denominator.

The results do **not** establish universal RFM behavior, causal growth of
structural dependence with context size, or leakage-free forecasting. Full
provenance is in `results/preliminary/*/manifest.json` and the source artifacts
are preserved in `~/iis_research_archive`.
