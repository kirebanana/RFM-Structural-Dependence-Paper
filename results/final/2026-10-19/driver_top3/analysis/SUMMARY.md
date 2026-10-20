# Preliminary structural-reliance results

Detailed per-arm metrics: [analysis.json](analysis.json).

Result set: `results/final/2026-10-19/driver_top3`
Evidence: **real**

Task: `rel-f1/driver-top3`

## Context-by-strength summary

| Context/local | Requested | Seeds | AUROC mean [range] | Δ clean | Global changed/eligible | Global changed/max |
|---|---:|---|---|---:|---:|---:|
| 48/24 | 0% | clean | 0.5591 [0.5591, 0.5591] | +0.0000 | 0.0000 | 0.0000 |
| 48/24 | 50% | 101, 202, 303 | 0.5326 [0.5235, 0.5388] | -0.0265 | 0.2506 | 0.4117 |
| 48/24 | 100% | 101, 202, 303 | 0.5393 [0.5041, 0.5770] | -0.0197 | 0.6087 | 1.0000 |
| 128/64 | 0% | clean | 0.7168 [0.7168, 0.7168] | +0.0000 | 0.0000 | 0.0000 |
| 128/64 | 50% | 101, 202, 303 | 0.6761 [0.6671, 0.6909] | -0.0407 | 0.2849 | 0.3740 |
| 128/64 | 100% | 101, 202, 303 | 0.6629 [0.6490, 0.6706] | -0.0538 | 0.7618 | 1.0000 |

## Per-arm paired statistics

| Context | Requested | Seed | AUROC | Δ clean | Changed/eligible | Changed/max | Mean abs raw shift | Median abs raw shift | Pearson | Unknown comparisons |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 0% | clean | 0.5591 | +0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| 48 | 50% | 101 | 0.5354 | -0.0236 | 0.2487 | 0.4086 | 0.0933 | 0.0137 | 0.6410 | 0 |
| 48 | 50% | 202 | 0.5235 | -0.0355 | 0.2522 | 0.4143 | 0.0945 | 0.0136 | 0.6220 | 0 |
| 48 | 50% | 303 | 0.5388 | -0.0203 | 0.2510 | 0.4124 | 0.0772 | 0.0120 | 0.7113 | 0 |
| 48 | 100% | 101 | 0.5770 | +0.0179 | 0.6087 | 1.0000 | 0.1166 | 0.0211 | 0.5454 | 0 |
| 48 | 100% | 202 | 0.5368 | -0.0222 | 0.6087 | 1.0000 | 0.1084 | 0.0195 | 0.5625 | 0 |
| 48 | 100% | 303 | 0.5041 | -0.0549 | 0.6087 | 1.0000 | 0.1163 | 0.0195 | 0.5501 | 0 |
| 128 | 0% | clean | 0.7168 | +0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| 128 | 50% | 101 | 0.6909 | -0.0259 | 0.2854 | 0.3746 | 0.1082 | 0.0159 | 0.6427 | 0 |
| 128 | 50% | 202 | 0.6702 | -0.0465 | 0.2848 | 0.3739 | 0.1198 | 0.0156 | 0.5932 | 0 |
| 128 | 50% | 303 | 0.6671 | -0.0497 | 0.2846 | 0.3736 | 0.1063 | 0.0156 | 0.6482 | 0 |
| 128 | 100% | 101 | 0.6692 | -0.0476 | 0.7618 | 1.0000 | 0.1156 | 0.0273 | 0.6981 | 0 |
| 128 | 100% | 202 | 0.6490 | -0.0678 | 0.7618 | 1.0000 | 0.1150 | 0.0273 | 0.6751 | 0 |
| 128 | 100% | 303 | 0.6706 | -0.0461 | 0.7618 | 1.0000 | 0.1122 | 0.0269 | 0.6943 | 0 |

## Availability and incidence policy

Policy: `query_available_incidence_v1`. Future rows and unclosed forecast labels are masked before every arm; no refill.
Fanout changes are part of the measured incidence effect, not a held-fixed control.

| Context | Future cells masked | Unclosed label cells masked | Original cells |
|---|---:|---:|---:|
| 48 | 45 | 1418 | 34848 |
| 128 | 55 | 1418 | 92928 |

## Limitations

- Preliminary descriptive evidence; no significance or causal context-size claims.
- This is a total FK-incidence effect: per-token fanout/normalization can change and is recorded, not held fixed.
- Requested strength is relative to seeded maximum changed parent IDs, not all eligible edges.
- Zero eligible/max denominators are undefined; query means omit these rows and report their counts.
- Common filtering masks future rows and unclosed forecast labels without refilling the sampled context.
- Timeless tables follow explicit benchmark schema metadata, not proof of historical attribute availability.
- Retained temporal input uses query-time cutoffs and mature rolling support labels; this is not train-only context.
- Sampling precedes availability masking; historically leakage-free retrieval selection has not been established.
- The historical pilot is separate and its checkpoint revision was not pinned.
