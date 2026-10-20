# Third-party notices

## Publication decision — 2026-10-20

The project owner approved publication of this academic, noncommercial research
repository using the current upstream Apache-2.0 declaration as the source-license
basis. Its complete text is included at `vendor/relational-transformer/LICENSE`;
local changes and source revisions are identified in `MODIFICATIONS.md` beside it.
The historical qualifications below are retained, not represented as resolved
legal findings. Earlier "uncleared" release-gate statements record the prior
assessment; they are not a claim that the owner has obtained separate permission.

RelBench and the underlying Ergast, AACT/ClinicalTrials.gov and arXiv-physics
sources are attributed below. Data-containing research artifacts retain their
source-specific terms; no blanket Apache license or unrestricted/commercial-use
permission is assigned to them. Raw databases, embeddings, model weights and
private preparation material are excluded. See `docs/REDISTRIBUTION.md` for the
source terms, attribution qualifications and historical processing-date gaps.
Publication approval does not change those source terms or confer rights the
upstream licensors do not control.

Checked **2026-10-20**. These notices identify sources and unresolved licensing;
they do not grant rights or clear redistribution. See
[redistribution review](docs/REDISTRIBUTION.md) for pinned evidence, source terms,
artifact decisions and owner-approved options. Legal/institutional review is
still required; no blanket license is assigned to third-party material here.

## Vendored Relational Transformer / Rustler

`vendor/relational-transformer/` derives from Stanford STAR's
`455df27c1458e093eac00133d5bbf41a8263a2e3`, with local FK-relation and temporal
processing changes. The exact recursive tree contains no license file; the
checked README, Python and Rust metadata establish no license grant for that
checkout. **Redistribution of the pinned copy remains uncleared.** Public
availability alone is not permission to redistribute.

- [Exact upstream tree](https://api.github.com/repos/stanford-star/relational-transformer/git/trees/455df27c1458e093eac00133d5bbf41a8263a2e3?recursive=1)
- [Pinned Python metadata](https://raw.githubusercontent.com/stanford-star/relational-transformer/455df27c1458e093eac00133d5bbf41a8263a2e3/pyproject.toml)

**Correction to the 2026-09-23 notice:** current upstream does now include
Apache-2.0, verified at `def35f92bf890741c4be41c4c315886365ee5ffb`.
[Current license](https://raw.githubusercontent.com/stanford-star/relational-transformer/def35f92bf890741c4be41c4c315886365ee5ffb/LICENSE)
and [license-file history](https://api.github.com/repos/stanford-star/relational-transformer/commits?path=LICENSE&per_page=10)
do not by themselves establish retroactive coverage of the older vendored tree.
PluRel's separate MIT metadata does not settle this checkout's rights. Establish
an applicable grant or use an owner-approved licensed-source/exclusion plan;
do not simply attach today's license to yesterday's copy.

Publication-gate recheck on 2026-10-20 confirmed that the supplied `main/LICENSE`
is Apache-2.0, but did not establish coverage for the legacy copied files. Of 86
tracked vendored paths, comparison of the pinned and current upstream trees finds
3 unchanged at the same path, 12 changed, and 71 absent from the newer tree. Paths
may have moved; this comparison is not a legal ruling, but it does not authenticate
the complete legacy copy as the newly licensed work. No license was attached to
the older copy on that assumption. Publication remains gated on applicable
permission or an approved licensed-source/minimized-release plan.

Source patches can contain upstream code too, including historical
`results/preliminary/2026-10-05_validated/source_at_run.patch` and the curated
`results/final/2026-10-19/source_at_finalization.patch`.

## RelBench collections and source databases

The experiment uses:

- `stanford-star/relbench-v1` at
  `d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600`: `rel-f1` and `rel-trial`.
  Its [pinned card](https://huggingface.co/datasets/stanford-star/relbench-v1/raw/d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600/README.md)
  labels the collection `cc-by-sa-4.0`; its STATS rows also say `CC BY-SA 4.0`.
- `stanford-star/relbench-v2-extra` at
  `190016ae95ac78362c331e23f072ca74bcbed3a2`: `rel-arxiv`.
  Its [pinned card](https://huggingface.co/datasets/stanford-star/relbench-v2-extra/raw/190016ae95ac78362c331e23f072ca74bcbed3a2/README.md)
  says CC BY-SA 4.0 **for the collection**, retaining source-database licenses;
  the pinned `rel-arxiv` STATS row nevertheless says `CC BY-SA 4.0`.

Neither pinned tree contains a standalone license file. Current cards and
LICENSE.md files instead put RelBench's conversion/schema/task/split/statistics
contributions under CDLA-Permissive-2.0 and retain upstream per-database terms.
The detailed review records both revisions and the conflicting STATS values;
later declarations are not assumed retroactive. RelBench **code** being
Apache-2.0 does not license benchmark data.

Source acknowledgments and constraints requiring artifact-specific review:

- **F1:** Ergast Developer API/database; its archived database-image page
  specifies **CC BY-NC-SA 3.0**, unlike the pinned Hub label. Jolpica's successor
  API specifies **CC BY-NC-SA 4.0**; do not substitute it for this snapshot.
- **Trial:** ClinicalTrials.gov via **AACT**, maintained by the Clinical Trials
  Transformation Initiative (CTTI). ClinicalTrials.gov terms address source
  attribution, processing dates, current data, modification descriptions,
  proprietary-rights claims, promotional email use and third-party copyrights.
  No unconditional public-domain/CC BY-SA clearance is inferred.
- **arXiv:** **Tang Haiming, He Sirui, Li Mengjie and Guo Zhimao (2024),
  arXiv-physics: A Large-Scale Physics Citation and Authorship Dataset**,
  [source repository](https://github.com/PKUTHM/arxiv-physics).
  Its checked source [LICENSE.txt](https://raw.githubusercontent.com/PKUTHM/arxiv-physics/d041d56db6a2a7c86cbc7af6774f9db30d1385ba/LICENSE.txt)
  specifies **CC BY 4.0** for metadata/dataset files, with attribution, license
  link and change indication. This is not assumed to be OGB data or permission
  to redistribute full-text arXiv papers.

RelBench benchmark attribution: Robinson et al., *RelBench: A Benchmark for Deep
Learning on Relational Databases*, NeurIPS 2024; Gu et al., *RelBench v2: A
Large-Scale Benchmark and Repository for Relational Data*, DATA-FM at ICLR 2026,
[arXiv:2602.12606](https://arxiv.org/abs/2602.12606). Project transformations include
task-row selection, native preprocessing, temporal availability filtering,
fixed-context FK interventions, and extraction of query-aligned panels and
metrics; see methodology and artifact provenance for exact changes.

## Externally downloaded models

- **Stanford STAR RT-PluRel**, revision
  `2ed96efc8007c57d589a31ceb435eb768491c439`: the experiment's root
  `classification/` checkpoint is **CC BY-NC-SA 4.0**, per the exact
  [card](https://huggingface.co/stanford-star/rt-plurel/raw/2ed96efc8007c57d589a31ceb435eb768491c439/README.md)
  and [LICENSE](https://huggingface.co/stanford-star/rt-plurel/raw/2ed96efc8007c57d589a31ceb435eb768491c439/LICENSE).
  The MIT declaration applies to older `paper/` files, not this checkpoint.
- **Sentence Transformers all-MiniLM-L12-v2**, revision
  `a50ef00143b4d5391434df20ae11632588ac25be`: the exact
  [card](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/raw/a50ef00143b4d5391434df20ae11632588ac25be/README.md)
  declares **Apache-2.0**; no standalone root license file was found.

Model weights, raw source tables and generated embeddings are excluded from the
curated final evidence. That does not automatically settle model-use conditions
or rights in copied task labels, IDs, dates, manifests/SQL and derived panels.
Nor does a model's SA declaration automatically attach to every prediction.

## Existing publication and release gate

The repository is already public. Notices, `.gitignore`, future omissions and
new-commit deletions do not remove existing history or copies. Any removal,
history rewrite, visibility change or rights-holder contact needs explicit owner
approval. No such action is taken or authorized by these notices. Keep public
redistribution uncleared wherever an applicable grant is unestablished.
