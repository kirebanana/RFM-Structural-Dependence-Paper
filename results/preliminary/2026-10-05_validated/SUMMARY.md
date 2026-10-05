# Preliminary structural-reliance results

Result set: `results/preliminary/2026-10-05_validated`
Evidence: **real**

## Context-by-strength summary

| Context/local | Requested | Seeds | AUROC mean [range] | Δ clean | Global changed/eligible | Global changed/max |
|---|---:|---|---|---:|---:|---:|
| 48/24 | 0% | clean | 0.5613 [0.5613, 0.5613] | +0.0000 | 0.0000 | 0.0000 |
| 48/24 | 50% | 101, 202, 303 | 0.5610 [0.5516, 0.5682] | -0.0003 | 0.2574 | 0.4204 |
| 48/24 | 100% | 101, 202, 303 | 0.5604 [0.5533, 0.5661] | -0.0010 | 0.6122 | 1.0000 |
| 128/64 | 0% | clean | 0.6427 [0.6427, 0.6427] | +0.0000 | 0.0000 | 0.0000 |
| 128/64 | 50% | 101, 202, 303 | 0.5893 [0.5862, 0.5912] | -0.0534 | 0.2803 | 0.3750 |
| 128/64 | 100% | 101, 202, 303 | 0.6143 [0.6071, 0.6246] | -0.0285 | 0.7476 | 1.0000 |

## Per-arm paired statistics

| Context | Requested | Seed | AUROC | Δ clean | Changed/eligible | Changed/max | Mean abs raw shift | Median abs raw shift | Pearson | Unknown comparisons |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 0% | clean | 0.5613 | +0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| 48 | 50% | 101 | 0.5516 | -0.0097 | 0.2568 | 0.4194 | 0.1060 | 0.0186 | 0.6255 | 0 |
| 48 | 50% | 202 | 0.5631 | +0.0018 | 0.2587 | 0.4226 | 0.0996 | 0.0180 | 0.6494 | 0 |
| 48 | 50% | 303 | 0.5682 | +0.0068 | 0.2566 | 0.4192 | 0.0986 | 0.0196 | 0.7007 | 0 |
| 48 | 100% | 101 | 0.5533 | -0.0080 | 0.6122 | 1.0000 | 0.1201 | 0.0254 | 0.5461 | 0 |
| 48 | 100% | 202 | 0.5661 | +0.0048 | 0.6122 | 1.0000 | 0.1272 | 0.0261 | 0.5178 | 0 |
| 48 | 100% | 303 | 0.5616 | +0.0003 | 0.6122 | 1.0000 | 0.1281 | 0.0263 | 0.4720 | 0 |
| 128 | 0% | clean | 0.6427 | +0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 |
| 128 | 50% | 101 | 0.5912 | -0.0516 | 0.2817 | 0.3769 | 0.1326 | 0.0191 | 0.5661 | 0 |
| 128 | 50% | 202 | 0.5862 | -0.0565 | 0.2808 | 0.3756 | 0.1341 | 0.0205 | 0.6014 | 0 |
| 128 | 50% | 303 | 0.5906 | -0.0521 | 0.2785 | 0.3725 | 0.1270 | 0.0176 | 0.6047 | 0 |
| 128 | 100% | 101 | 0.6071 | -0.0356 | 0.7476 | 1.0000 | 0.1238 | 0.0322 | 0.6956 | 0 |
| 128 | 100% | 202 | 0.6110 | -0.0317 | 0.7476 | 1.0000 | 0.1338 | 0.0302 | 0.6772 | 0 |
| 128 | 100% | 303 | 0.6246 | -0.0181 | 0.7476 | 1.0000 | 0.1598 | 0.0333 | 0.5260 | 0 |

## Availability and incidence policy

Policy: `query_available_incidence_v1`. Future rows and unclosed forecast labels are masked before every arm; no refill.
Fanout changes are part of the measured incidence effect, not a held-fixed control.

| Context | Future cells masked | Unclosed label cells masked | Original cells |
|---|---:|---:|---:|
| 48 | 55 | 1379 | 33696 |
| 128 | 55 | 1379 | 89856 |

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
