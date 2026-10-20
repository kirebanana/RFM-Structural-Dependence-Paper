# Independently recalculated saved results

Detailed metrics and support contrasts: [analysis.json](analysis.json).

Verified 110 prediction files.

AUROC averages per-seed AUROCs, not averaged predictions. Counts/coverage list seeds in order 101/202/303.
Eligible counts in the arrays cover all relations; maximum counts apply to the selected pool.

| Panel | Context | Group | Clean AUROC | AUROC mean | Mean abs shift | Changed edges | Affected queries |
|---|---:|---|---:|---:|---:|---|---|
| primary | 48 | base | 0.5613 | 0.5613 | 0.0000 | [0] | [0] |
| primary | 48 | a050 | 0.5613 | 0.5610 | 0.1014 | [1733, 1746, 1732] | [640, 647, 641] |
| primary | 48 | a100 | 0.5613 | 0.5604 | 0.1251 | [4132, 4132, 4132] | [702, 702, 702] |
| primary | 128 | base | 0.6427 | 0.6427 | 0.0000 | [0] | [0] |
| primary | 128 | a050 | 0.6427 | 0.5893 | 0.1312 | [5705, 5686, 5639] | [702, 702, 702] |
| primary | 128 | a100 | 0.6427 | 0.6143 | 0.1391 | [15138, 15138, 15138] | [702, 702, 702] |
| relations | 48 | task_driver | 0.5613 | 0.5592 | 0.1274 | [2960, 2960, 2960] | [702, 702, 702] |
| relations | 48 | database_all | 0.5613 | 0.5713 | 0.0092 | [1172, 1172, 1172] | [322, 322, 322] |
| relations | 48 | database_driver | 0.5613 | 0.5706 | 0.0092 | [677, 677, 677] | [321, 321, 321] |
| relations | 48 | database_constructor | 0.5613 | 0.5614 | 0.0004 | [222, 222, 222] | [111, 111, 111] |
| relations | 48 | database_race | 0.5613 | 0.5600 | 0.0010 | [273, 273, 273] | [135, 135, 135] |
| relations | 48 | race_circuit | 0.5613 | 0.5613 | 0.0000 | [0, 0, 0] | [0, 0, 0] |
| relations | 128 | task_driver | 0.6427 | 0.6143 | 0.1437 | [4177, 4177, 4177] | [702, 702, 702] |
| relations | 128 | database_all | 0.6427 | 0.6426 | 0.0202 | [10961, 10961, 10961] | [702, 702, 702] |
| relations | 128 | database_driver | 0.6427 | 0.6416 | 0.0202 | [2816, 2816, 2816] | [670, 670, 670] |
| relations | 128 | database_constructor | 0.6427 | 0.6420 | 0.0019 | [1593, 1593, 1593] | [517, 517, 517] |
| relations | 128 | database_race | 0.6427 | 0.6431 | 0.0042 | [3469, 3469, 3469] | [697, 697, 697] |
| relations | 128 | race_circuit | 0.6427 | 0.6438 | 0.0022 | [3083, 3083, 3083] | [701, 701, 701] |
| task_cycles | 48 | task_query_cycles | 0.5613 | 0.5593 | 0.1267 | [1781, 1747, 1767] | [669, 666, 674] |
| task_cycles | 48 | task_other_cycles | 0.5613 | 0.5425 | 0.0931 | [1179, 1213, 1193] | [420, 425, 417] |
| task_cycles | 128 | task_query_cycles | 0.6427 | 0.5598 | 0.1579 | [1611, 1662, 1668] | [636, 635, 641] |
| task_cycles | 128 | task_other_cycles | 0.6427 | 0.5845 | 0.1432 | [2566, 2515, 2509] | [587, 593, 571] |
| labeled_support | 48 | labeled_support_only | 0.5613 | 0.5549 | 0.0525 | [451, 451, 451] | [158, 158, 158] |
| labeled_support | 128 | labeled_support_only | 0.6427 | 0.5964 | 0.1110 | [1552, 1552, 1552] | [390, 390, 390] |
| driver_top3 | 48 | base | 0.5591 | 0.5591 | 0.0000 | [0] | [0] |
| driver_top3 | 48 | a050 | 0.5591 | 0.5326 | 0.0883 | [1723, 1747, 1739] | [650, 658, 654] |
| driver_top3 | 48 | a100 | 0.5591 | 0.5393 | 0.1138 | [4217, 4217, 4217] | [726, 726, 726] |
| driver_top3 | 128 | base | 0.7168 | 0.7168 | 0.0000 | [0] | [0] |
| driver_top3 | 128 | a050 | 0.7168 | 0.6761 | 0.1114 | [5917, 5906, 5902] | [726, 726, 726] |
| driver_top3 | 128 | a100 | 0.7168 | 0.6629 | 0.1143 | [15796, 15796, 15796] | [726, 726, 726] |
| arxiv | 48 | base | 0.6401 | 0.6401 | 0.0000 | [0] | [0] |
| arxiv | 48 | task_entity | 0.6401 | 0.6126 | 0.1104 | [2552, 2552, 2552] | [512, 512, 512] |
| arxiv | 48 | database | 0.6401 | 0.6401 | 0.0175 | [5540, 5540, 5540] | [507, 507, 507] |
| arxiv | 128 | base | 0.6920 | 0.6920 | 0.0000 | [0] | [0] |
| arxiv | 128 | task_entity | 0.6920 | 0.6595 | 0.1530 | [4588, 4588, 4588] | [512, 512, 512] |
| arxiv | 128 | database | 0.6920 | 0.6896 | 0.0269 | [21799, 21799, 21799] | [512, 512, 512] |
| trial | 48 | base | 0.5489 | 0.5489 | 0.0000 | [0] | [0] |
| trial | 48 | task_entity | 0.5489 | 0.5393 | 0.0131 | [1476, 1476, 1476] | [512, 512, 512] |
| trial | 48 | database | 0.5489 | 0.5414 | 0.0022 | [809, 809, 809] | [296, 296, 296] |
| trial | 128 | base | 0.5360 | 0.5360 | 0.0000 | [0] | [0] |
| trial | 128 | task_entity | 0.5360 | 0.5048 | 0.0220 | [1497, 1497, 1497] | [512, 512, 512] |
| trial | 128 | database | 0.5360 | 0.4904 | 0.0040 | [14099, 14099, 14099] | [512, 512, 512] |
