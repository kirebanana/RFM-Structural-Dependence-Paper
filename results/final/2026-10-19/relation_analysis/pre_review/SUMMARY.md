# Exploratory relation controls

Detailed contrasts and RQ2 models: [analysis.json](analysis.json).

Same 702 sampled queries and context fingerprints; three matching seeds; no confirmatory claims.
Affected queries are the union across seeds; changed edges are the seed mean. Budgets are not matched.

| Context | Control | AUROC mean | Δ clean | Affected queries (union) | Changed edges (mean) | Mean abs score shift | Crossed-cluster interval for Δ |
|---|---|---:|---:|---:|---:|---:|---|
| 48 | task_driver | 0.5592 | -0.0021 | 702 | 2960 | 0.1274 | [-0.0487, +0.0437] |
| 48 | database_all | 0.5713 | +0.0100 | 322 | 1172 | 0.0092 | [-0.0036, +0.0256] |
| 48 | database_driver | 0.5706 | +0.0093 | 321 | 677 | 0.0092 | [-0.0045, +0.0251] |
| 48 | database_constructor | 0.5614 | +0.0001 | 111 | 222 | 0.0004 | [-0.0012, +0.0020] |
| 48 | database_race | 0.5600 | -0.0013 | 135 | 273 | 0.0010 | [-0.0059, +0.0021] |
| 48 | race_circuit | 0.5613 | +0.0000 | 0 | 0 | 0.0000 | [+0.0000, +0.0000] |
| 128 | task_driver | 0.6143 | -0.0284 | 702 | 4177 | 0.1437 | [-0.0692, +0.0158] |
| 128 | database_all | 0.6426 | -0.0002 | 702 | 10961 | 0.0202 | [-0.0139, +0.0156] |
| 128 | database_driver | 0.6416 | -0.0011 | 670 | 2816 | 0.0202 | [-0.0165, +0.0154] |
| 128 | database_constructor | 0.6420 | -0.0008 | 517 | 1593 | 0.0019 | [-0.0035, +0.0017] |
| 128 | database_race | 0.6431 | +0.0003 | 697 | 3469 | 0.0042 | [-0.0035, +0.0041] |
| 128 | race_circuit | 0.6438 | +0.0011 | 701 | 3083 | 0.0022 | [-0.0014, +0.0038] |
| 48 | task_query_cycles | 0.5593 | -0.0020 | 700 | 1765 | 0.1267 | [-0.0570, +0.0514] |
| 48 | task_other_cycles | 0.5425 | -0.0189 | 474 | 1195 | 0.0931 | [-0.0630, +0.0211] |
| 128 | task_query_cycles | 0.5598 | -0.0829 | 693 | 1647 | 0.1579 | [-0.1471, -0.0226] |
| 128 | task_other_cycles | 0.5845 | -0.0582 | 640 | 2530 | 0.1432 | [-0.1094, -0.0091] |

## Held-out sensitivity prediction

Response: log1p(seed-mean absolute score shift). Out-of-fold R² is not causal evidence.

| Features | Date holdout R² | Driver holdout R² |
|---|---:|---:|
| group_context | 0.0995 | 0.0964 |
| plus_exposure_composition | 0.1005 | 0.1057 |
| plus_support_only | 0.1087 | 0.1056 |
| plus_clean_score_only | 0.3026 | 0.3007 |
| exposure_and_support | 0.0990 | 0.1111 |
| plus_support_clean_score | 0.3025 | 0.3004 |
| all_pre_intervention | 0.3038 | 0.3026 |
| plus_realized_fanout_diagnostics | 0.3030 | 0.3027 |

## Limitations
- Post-exploration controls on the same single-task panel, not confirmatory tests.
- Groups differ in changed-edge count and exposure; effect per edge is not a causal estimate.
- Query-cycle controls include other rows in those cycles, not isolated query-edge replacements.
- Cluster intervals are sensitivity checks, conditional on three seeds and one checkpoint.
- Crossed cluster resampling does not eliminate temporal or outcome-window dependence.
- Held-out sensitivity models predict model response, not labels or causal structure effects.
- Original partial/full strengths differ in relation composition; no clean dose-response claim.
- Availability masks retained inputs after sampling; historical retrieval validity is not proven.
