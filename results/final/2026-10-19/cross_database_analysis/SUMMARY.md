# Frozen cross-database replication

Detailed paired contrasts: [analysis.json](analysis.json).

512 uniformly selected official test queries per task; three matching seeds.
Single-date panels: intervals are conditional entity-resampling sensitivity checks, not temporal replication.

| Database/task | Context | Clean AUROC | Task-link AUROC mean | DB-link AUROC mean | Task abs shift | DB abs shift |
|---|---:|---:|---:|---:|---:|---:|
| rel-arxiv/paper-citation | 48 | 0.6401 | 0.6126 | 0.6401 | 0.1104 | 0.0175 |
| rel-arxiv/paper-citation | 128 | 0.6920 | 0.6595 | 0.6896 | 0.1530 | 0.0269 |
| rel-trial/study-outcome | 48 | 0.5489 | 0.5393 | 0.5414 | 0.0131 | 0.0022 |
| rel-trial/study-outcome | 128 | 0.5360 | 0.5048 | 0.4904 | 0.0220 | 0.0040 |

Task/entity and DB groups use different changed-edge counts, not matched dose.
Declared static covariates and post-sampling availability filtering retain the existing interpretation limits.
