# Redistribution review

**Status: not cleared for unrestricted public redistribution.** Evidence checked
2026-10-20. This is a source-evidence record and release checklist, not legal
advice, a new license, or permission from any rights holder. Obtain qualified
legal/institutional review before relying on an interpretation below.

## 1. Scope and immediate decision

Code, model weights, benchmark packaging, underlying databases, and derived
artifacts have different terms. A repository-level license cannot replace
upstream permissions. In particular, **RelBench's Apache-2.0 software license is
not a license for its databases**.[relbench-code]

The GitHub API reports this project's existing repository as **public**, with no
detected license.[project-api] This is not just a proposed future release:
`vendor/relational-transformer/`, historical `results/preliminary/` artifacts,
and source patches already belong to the review scope. The new curated evidence
also includes extracted task labels/IDs/dates and copied task-manifest SQL in
`panel_source.json`, not only original numerical predictions. Its exclusion of
raw tables, weights, and embeddings does not settle those remaining rights.

**The pinned RT source has no identified license grant in the checked tree and
metadata. Its redistribution remains a blocker unless coverage is established
or that material is excluded through an owner-approved release plan.** The
dataset discrepancies and artifact-specific questions below also remain open.
Do not interpret this document as clearing past publication.

## 2. RT source: pinned checkout versus today's upstream

`vendor/relational-transformer/` derives from
`455df27c1458e093eac00133d5bbf41a8263a2e3`, with local changes. The complete
recursive GitHub tree (`truncated: false`) has no LICENSE, COPYING, or NOTICE
path. Its README, Python project metadata, and Rust Cargo metadata contain no
license declaration.[rt-pin-tree], [rt-pin-readme], [rt-pin-python], [rt-pin-rust]

Current upstream is different: `main` resolved to
`def35f92bf890741c4be41c4c315886365ee5ffb`, whose tree contains `LICENSE` and
whose raw license is Apache-2.0.[rt-current-tree], [rt-current-license] The license
file history records its introduction in `4af9ff1dba62b5ffe702565f5f24910bd85aa678`
(2026-10-01), followed by “Relicense from MIT to Apache-2.0” in
`035fec9a7663aa4c7b9a969714ee81aaa629126e` (2026-10-15).[rt-license-history]
This supersedes the old notice's statement that *current* upstream has no
license; it does **not** by itself establish coverage of the older copied tree,
all its contributors, or historical redistribution. Do not assume retroactivity
or copy the new license onto the old checkout as if coverage were settled.

GitHub's own guidance says default copyright applies without a license (all
rights retained); public hosting's view/fork permissions are not a general
redistribution grant.[github-guidance] PluRel software's MIT metadata and the
MIT-licensed `paper/` checkpoints are separate from this RT checkout.

Possible resolutions require a recorded decision: establish an applicable grant
covering the exact copied material; or migrate/port the necessary changes to an
identified licensed revision and test it; or prepare a release excluding copied
RT material. If Apache-2.0 is established as applicable, its section 4 requires
the license copy, change notices, retained attribution/copyright notices and any
applicable upstream NOTICE. A link in this document alone is not that compliance.

## 3. Benchmark evidence at the exact revisions used

These are observations of the pinned files, **not confirmation that the
uploader can license every underlying source**. The two recursive Hub trees
were fully returned (248 and 228 entries, no next-page link); neither contains
a license-named file. Root `LICENSE` requests return 404.[v1-tree], [v2-tree]

| Download used here | Pinned card evidence | Pinned `STATS/databases.parquet`: `name` / `license` / `source_url` |
|---|---|---|
| `stanford-star/relbench-v1` @ `d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600` | README metadata says `cc-by-sa-4.0`; no prose license section or per-source clarification.[v1-card] | `rel-f1` / `CC BY-SA 4.0` / `https://ergast.com/mrd/`; `rel-trial` / `CC BY-SA 4.0` / `https://clinicaltrials.gov/`.[v1-stats] |
| `stanford-star/relbench-v2-extra` @ `190016ae95ac78362c331e23f072ca74bcbed3a2` | Metadata says `cc-by-sa-4.0`; prose says “CC BY-SA 4.0 for the collection. Each database keeps the license of its source; see `STATS/databases.parquet`.”[v2-card] | `rel-arxiv` / `CC BY-SA 4.0` / `https://relbench.stanford.edu`.[v2-stats] |

The Parquet rows above were read from the pinned binary files, not inferred from
the cards or the current Hub viewer. The `rel-arxiv` pinned source URL identifies
RelBench, not the original data producer.

### Current metadata is materially different

On the access date, v1 `main` resolved to
`21d7284df2a9621c71f21bc4a1a30b7e8effb8dd` and v2-extra `main` to
`6b02dae348f68ecf5b08d8587ccdc839c2c33e3e`. Both current README/LICENSE.md
files separate upstream database terms from **CDLA-Permissive-2.0 for their
contributions** (conversion, schemas, task definitions, splits, statistics).
The current cards use `license: other`, `license_name: per-database`, not a
blanket CC BY-SA label.[v1-current-card], [v1-current-license],
[v2-current-card], [v2-current-license]

| Database | Current STATS license value | Current source URL |
|---|---|---|
| `rel-f1` | `CC BY-NC-SA 3.0 (Ergast database images)` | `https://ergast.com/mrd/db/` |
| `rel-trial` | `ClinicalTrials.gov terms (US Government data; attribution required)` | `https://clinicaltrials.gov/about-site/terms-conditions` |
| `rel-arxiv` | `CC BY 4.0` | `https://github.com/PKUTHM/arxiv-physics` |

These current rows were also read directly from Parquet.[v1-current-stats],
[v2-current-stats] Do not silently substitute them for the pinned evidence or
assume later grants apply to all earlier files. They are useful corrections and
source leads, but applicability to the exact downloaded bytes/contributions
needs review. CDLA section 3 imposes no obligations on computational “Results”;
section 2 requires its text with shared Data. Neither provision overrides
separate upstream terms or establishes that the pinned files were received
under CDLA.[cdla]

## 4. Underlying sources and unresolved restrictions

### `rel-f1`: Ergast database images, not a blanket permissive dataset

RelBench's current provenance script identifies a static **Ergast database**
snapshot.[f1-provenance] An archived copy of Ergast's own database-images page
(2024-01-06) explicitly licenses the images under **CC BY-NC-SA 3.0 Unported**.
The live database URL returned unrelated betting content on the access date;
it is not evidence of a replacement grant.[ergast-archive]

The same archived database-images page also says **“Attribution is optional but
appreciated.”** Record that contemporaneous qualification alongside the linked
CC BY-NC-SA declaration; its legal interaction with the standard CC attribution
conditions requires review, rather than treating the page as an unqualified
standard-license statement. It does not on its face eliminate the NC/SA questions.

Ergast's historical API terms describe personal/noncommercial use including
research/education, disallow charging for API-based applications/services or
obtained data, and say API attribution is appreciated but not required. That
API statement alone must not be substituted for the database-image declaration
and its own attribution qualification or the standard CC conditions.
[ergast-api], [cc-ncsa3]

The successor Jolpica API's own terms instead specify **CC BY-NC-SA 4.0** and
noncommercial use.[jolpica] They do not establish that this static historical
snapshot came from Jolpica or relicense it. Retain Ergast attribution, applicable
license/version, source and modification details if a data-containing release
is approved; assess NC and SA obligations and reconcile the conflicting pinned
CC BY-SA 4.0 label before distributing such artifacts. Do not promise commercial
reuse or automatically label all F1 material as Jolpica 4.0.

### `rel-trial`: AACT / ClinicalTrials.gov

RelBench's provenance identifies an AACT snapshot. AACT describes itself as a
relational reorganization of daily ClinicalTrials.gov protocol/results data;
its public availability or its application's software license is not a blanket
license for every record.[trial-provenance], [aact]

ClinicalTrials.gov's own terms apply independently of format/acquisition method.
They say data are available without charge and that publication/distribution
*should*: attribute ClinicalTrials.gov; keep data current; clearly display the
date ClinicalTrials.gov processed the data; and completely describe modifications.
They prohibit asserting proprietary rights in the database and promotional use
of extracted email addresses. They also warn of international copyright and
possible third-party copyrights; these are not an unqualified public-domain or
CC BY-SA grant.[ctgov] The text was verified in the first-party JavaScript asset
used to render the terms page, because a plain HTML fetch returns an app shell.
[ctgov-page-asset]

For a frozen historical panel, record the source snapshot/processing date and
all transformations, distinguish historical research data from current trial
information, and obtain review of how the “current at all times” language
applies. The Hub commit date, experiment date and forecast/query timestamps are
**not substitutes** for the ClinicalTrials.gov processing date. That date and
the exact AACT snapshot's applicable conditions remain to be established here.
Attribute AACT/CTTI and ClinicalTrials.gov separately; do not claim proprietary
rights in extracted source records or treat the pinned CC label as resolving
third-party text rights.

### `rel-arxiv`: arXiv-physics, not assumed OGB provenance

The corrected RelBench source points to **PKUTHM/arxiv-physics**. Its README
reports six tables and 2,733,846 records, matching the pinned RelBench STATS
total, and says RelBench adopted it. At checked source revision
`d041d56db6a2a7c86cbc7af6774f9db30d1385ba`, `LICENSE.txt` expressly licenses
metadata/dataset files under **CC BY 4.0**, requiring attribution, a license link
and indication of changes. It names Tang Haiming, He Sirui, Li Mengjie and Guo
Zhimao, and supplies their 2024 dataset citation.[arxiv-source], [arxiv-license]
RelBench's provenance script traces its conversion to a static `db.zip`
snapshot, but does not itself prove the chain of rights for that exact archive
and all fields.[arxiv-provenance] No reviewed evidence identifies this database
as OGB/`ogbn-arxiv`; do not import an OGB license based on the name.

arXiv's own API terms permit descriptive metadata (including titles, abstracts,
authors, identifiers and classifications) under CC0, while e-print full text
retains its individual copyright/license. The arXiv-physics README also says
citation links were verified via Crossref; Crossref distinguishes unrestricted
bibliographic metadata/references from potentially copyrighted abstracts.
[arxiv-terms], [crossref] These facts do not license full-text papers or establish
the origin of every field. Preserve the source dataset's CC BY attribution and
RelBench attribution, and resolve the relationship between the pinned collection
CC BY-SA label and the source CC BY/current CDLA declarations before assigning
an artifact license.

## 5. Downloaded models (excluded from the curated evidence)

The exact **RT-PluRel** card at
`2ed96efc8007c57d589a31ceb435eb768491c439` and its `LICENSE` specify
**CC BY-NC-SA 4.0 for root `classification/` and `regression/` checkpoints**.
Only the older files under `paper/` retain MIT. This experiment uses
`classification/`, not `paper/`.[rt-model-card], [rt-model-license] Excluding
weights does not waive applicable conditions on use of licensed material, but
the card does not establish that every prediction is Adapted Material. Do not
automatically relicense all predictions/code under the checkpoint's SA terms.

The pinned **sentence-transformers/all-MiniLM-L12-v2** card at
`a50ef00143b4d5391434df20ae11632588ac25be` declares **Apache-2.0**; the checked
root tree has no standalone LICENSE file.[minilm-card], [minilm-tree] Weights and
generated text embeddings are excluded from the curated evidence. Any future
weight redistribution needs a license/notice check; Apache metadata for the
encoder does not license the underlying database text or automatically clear
redistribution of embeddings computed from it.

## 6. Artifact decisions and owner-approved options

CC BY-SA 4.0 sections 1–4 distinguish licensed material, adaptations requiring
permission and substantial database extraction. Attribution/source/license and
change notices can apply to sharing covered material; SA applies when covered
adaptations are shared. CC BY-NC-SA also restricts licensed use to noncommercial
purposes. These licenses grant only rights the licensor controls; neither
“derived” nor “research” alone settles applicability.[cc-sa4], [cc-ncsa4]

| Material | Concrete action before approving distribution |
|---|---|
| Original analysis code and aggregate metrics/plots | Separate original authorship from copied code/text. Review any embedded data; record a project-owner licensing decision. Do not claim a blanket upstream grant. |
| Query-aligned predictions and exposure arrays | Inspect fields and reconstruction/linkage risk. Separate model-generated numbers from source labels, IDs, dates and copied task material. Decide rights/attribution per artifact, not by `.npz` extension or size. |
| `panel.npz`, copied task manifests/SQL, support/query IDs | Treat as potential benchmark/database extraction or adaptation, not “predictions only”; check collection/contribution and source terms, extraction scope, modification notices and any applicable SA/NC conditions. No legal conclusion about substantiality is made here. |
| Raw tables, text, sampled contexts, embeddings and model weights | Keep outside the proposed curated release unless separately reviewed and explicitly approved. Local download/reproduction does not itself confer redistribution rights. |
| Vendored RT, compiled copies and `*.patch` artifacts | Establish license coverage for copied lines as well as whole files. Review `results/preliminary/2026-10-05_validated/source_at_run.patch` and `results/final/2026-10-19/source_at_finalization.patch`; dropping `vendor/` alone can leave source in patches/history. |

Owner-approved paths, **not actions performed by this review**:

1. **Permission/compliance route:** establish an applicable RT grant and the
   exact data/source terms; preserve the evidence, supply required notices and
   licenses, and approve each data-containing artifact. Do not fabricate a grant
   or contact rights holders without authorization.
2. **Licensed-source migration:** port/test against an explicitly licensed RT
   revision, preserve scientific provenance and comply with its terms. This
   changes the source basis and does not by itself cure old published copies.
3. **Minimized separate release:** create an owner-approved clean export of
   original code, synthetic examples, aggregates, provenance hashes and download
   instructions, excluding uncleared copied code/patches and data-containing
   artifacts. Inspect even documentation/SQL for copied content. This lowers
   redistribution scope; it does not retrospectively clear the existing repo.
4. **Existing public repository remediation:** inventory reachable commits,
   tags, branches, releases, hosted artifacts and known forks/copies. Discuss
   temporary visibility changes, targeted removal and/or history rewriting with
   the owner and legal reviewer. **Each requires explicit user approval.**

`.gitignore`, future omission, a notice update, deleting a file in a new commit,
or a fresh export **does not remove already published Git history**. Rewriting
history or making a repository private also cannot recall existing clones or
forks; GitHub explicitly warns that those copies remain.[github-guidance]
No deletion, history rewrite, visibility change, maintainer contact, staging,
commit or push is authorized by this document. Keep the release gate unresolved
where no applicable grant has been established.

## Sources

All URLs below were accessed **2026-10-20**. Commit-qualified links distinguish
experimental pins from current evidence; the archived Ergast page preserves a
historical first-party statement. Current source declarations are not assumed
retroactive.

[project-api]: https://api.github.com/repos/kirebanana/RFM-Structural-Dependence-Paper
[github-guidance]: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository
[relbench-code]: https://raw.githubusercontent.com/snap-stanford/relbench/91c1f6420a03feb61c279aaeb194d7268435f278/LICENSE
[rt-pin-tree]: https://api.github.com/repos/stanford-star/relational-transformer/git/trees/455df27c1458e093eac00133d5bbf41a8263a2e3?recursive=1
[rt-pin-readme]: https://raw.githubusercontent.com/stanford-star/relational-transformer/455df27c1458e093eac00133d5bbf41a8263a2e3/README.md
[rt-pin-python]: https://raw.githubusercontent.com/stanford-star/relational-transformer/455df27c1458e093eac00133d5bbf41a8263a2e3/pyproject.toml
[rt-pin-rust]: https://raw.githubusercontent.com/stanford-star/relational-transformer/455df27c1458e093eac00133d5bbf41a8263a2e3/rustler/Cargo.toml
[rt-current-tree]: https://api.github.com/repos/stanford-star/relational-transformer/git/trees/def35f92bf890741c4be41c4c315886365ee5ffb?recursive=1
[rt-current-license]: https://raw.githubusercontent.com/stanford-star/relational-transformer/def35f92bf890741c4be41c4c315886365ee5ffb/LICENSE
[rt-license-history]: https://api.github.com/repos/stanford-star/relational-transformer/commits?path=LICENSE&per_page=10
[v1-tree]: https://huggingface.co/api/datasets/stanford-star/relbench-v1/tree/d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600?recursive=true&limit=1000
[v2-tree]: https://huggingface.co/api/datasets/stanford-star/relbench-v2-extra/tree/190016ae95ac78362c331e23f072ca74bcbed3a2?recursive=true&limit=1000
[v1-card]: https://huggingface.co/datasets/stanford-star/relbench-v1/raw/d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600/README.md
[v2-card]: https://huggingface.co/datasets/stanford-star/relbench-v2-extra/raw/190016ae95ac78362c331e23f072ca74bcbed3a2/README.md
[v1-stats]: https://huggingface.co/datasets/stanford-star/relbench-v1/resolve/d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600/STATS/databases.parquet
[v2-stats]: https://huggingface.co/datasets/stanford-star/relbench-v2-extra/resolve/190016ae95ac78362c331e23f072ca74bcbed3a2/STATS/databases.parquet
[v1-current-card]: https://huggingface.co/datasets/stanford-star/relbench-v1/raw/21d7284df2a9621c71f21bc4a1a30b7e8effb8dd/README.md
[v1-current-license]: https://huggingface.co/datasets/stanford-star/relbench-v1/raw/21d7284df2a9621c71f21bc4a1a30b7e8effb8dd/LICENSE.md
[v2-current-card]: https://huggingface.co/datasets/stanford-star/relbench-v2-extra/raw/6b02dae348f68ecf5b08d8587ccdc839c2c33e3e/README.md
[v2-current-license]: https://huggingface.co/datasets/stanford-star/relbench-v2-extra/raw/6b02dae348f68ecf5b08d8587ccdc839c2c33e3e/LICENSE.md
[v1-current-stats]: https://huggingface.co/datasets/stanford-star/relbench-v1/resolve/21d7284df2a9621c71f21bc4a1a30b7e8effb8dd/STATS/databases.parquet
[v2-current-stats]: https://huggingface.co/datasets/stanford-star/relbench-v2-extra/resolve/6b02dae348f68ecf5b08d8587ccdc839c2c33e3e/STATS/databases.parquet
[cdla]: https://cdla.dev/permissive-2-0/
[f1-provenance]: https://raw.githubusercontent.com/snap-stanford/relbench/91c1f6420a03feb61c279aaeb194d7268435f278/provenance/f1.py
[ergast-archive]: https://web.archive.org/web/20240106111004id_/https://ergast.com/mrd/db/
[ergast-api]: https://web.archive.org/web/20240221181400id_/http://ergast.com/mrd/terms/
[jolpica]: https://raw.githubusercontent.com/jolpica/jolpica-f1/main/TERMS.md
[cc-ncsa3]: https://creativecommons.org/licenses/by-nc-sa/3.0/legalcode
[trial-provenance]: https://raw.githubusercontent.com/snap-stanford/relbench/91c1f6420a03feb61c279aaeb194d7268435f278/provenance/trial.py
[aact]: https://aact.ctti-clinicaltrials.org/landing
[ctgov]: https://clinicaltrials.gov/about-site/terms-conditions
[ctgov-page-asset]: https://clinicaltrials.gov/chunk-4U7SWTR5.js
[arxiv-source]: https://raw.githubusercontent.com/PKUTHM/arxiv-physics/d041d56db6a2a7c86cbc7af6774f9db30d1385ba/README.md
[arxiv-license]: https://raw.githubusercontent.com/PKUTHM/arxiv-physics/d041d56db6a2a7c86cbc7af6774f9db30d1385ba/LICENSE.txt
[arxiv-provenance]: https://raw.githubusercontent.com/snap-stanford/relbench/91c1f6420a03feb61c279aaeb194d7268435f278/provenance/arxiv.py
[arxiv-terms]: https://info.arxiv.org/help/api/tou.html
[crossref]: https://www.crossref.org/documentation/retrieve-metadata/
[rt-model-card]: https://huggingface.co/stanford-star/rt-plurel/raw/2ed96efc8007c57d589a31ceb435eb768491c439/README.md
[rt-model-license]: https://huggingface.co/stanford-star/rt-plurel/raw/2ed96efc8007c57d589a31ceb435eb768491c439/LICENSE
[minilm-card]: https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/raw/a50ef00143b4d5391434df20ae11632588ac25be/README.md
[minilm-tree]: https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L12-v2/tree/a50ef00143b4d5391434df20ae11632588ac25be
[cc-sa4]: https://creativecommons.org/licenses/by-sa/4.0/legalcode.en
[cc-ncsa4]: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en
