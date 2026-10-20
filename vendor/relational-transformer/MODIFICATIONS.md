# Local research modifications

Upstream: Stanford STAR, Relational Transformer / Rustler,
https://github.com/stanford-star/relational-transformer.
The vendored source originated at revision
`455df27c1458e093eac00133d5bbf41a8263a2e3`.

This is a modified research copy, not an unmodified upstream release:

- Rust preprocessing records FK relation identities and preserves declared row
  timestamps for temporal tables without outgoing foreign keys.
- Native sampling and Python data conversion carry FK relation metadata into
  sampled contexts for controlled, validated parent-identity interventions.
- Local checkpoint/loading and evaluation integration supports the documented
  frozen-model experiment workflow. Model architecture is not changed.

The accompanying Apache-2.0 license is copied from upstream revision
`def35f92bf890741c4be41c4c315886365ee5ffb`, checked 2026-10-20.
Historical snapshot licensing qualifications are retained in the repository's
`THIRD_PARTY_NOTICES.md` and `docs/REDISTRIBUTION.md`; this notice does not assert
that historical distributions were retroactively cleared. Preserve upstream
copyright and attribution notices. No separate upstream NOTICE file was identified
in the reviewed source. Scientific source snapshots remain historical evidence,
not a claim that the amended source was used for every earlier run.
