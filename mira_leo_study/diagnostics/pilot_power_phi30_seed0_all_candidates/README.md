# 30-degree power-only diagnostic

Seed=0; fixed pilot grid: 90 times x 2 cohorts x 6 compact actions = 1,080 cases.
All candidate groups are marked scheduled. This is not an actual AoI/Algorithm-2 failure frequency.
Association is recomputed every physical slot; no AoI recursion, learning policy or reward pilot is executed.

| Threshold | LP-infeasible configurations | Configurations checked |
|---|---:|---:|
| 20 deg (previous audit) | 8 | 1080 |
| 30 deg (this audit) | 170 | 1080 |

170/170 proven infeasible cases were handled as zero-power transmission failures. Other invalid cases: 0.
By action: a=1: 0, a=2: 3, a=4: 19, a=5: 20, a=8: 52, a=10: 76.
By cohort index: 0: 33, 1: 137.

A higher elevation threshold can change a group from nonedge to edge. The global CSI order is still stochastic, so the fixed class split can remain infeasible.
For details see power_cases.csv and infeasible_cases.json.
