# Seed 0 pilot snapshot power-only diagnostic

Scope: 90 snapshot times x 2 cohorts x 6 compact actions = 1,080 power configurations.
All candidate groups are marked scheduled. This is not an actual Algorithm-2/AoI simulation failure frequency.
Association is recomputed at every 10-ms slot with current config; no AoI recursion, reward pilot or learning policy is executed.

Result: 8 LP-infeasible power configurations, 1,072 feasible, 0 other validation failures.
All 8 belong to cohort index 1 (the second cohort); actions: a=10 seven cases, a=8 one case.

| Time_s | Slot | Cohort index | Satellite | Action | Edge groups | Nonedge groups |
|---:|---:|---:|---|---:|---:|---:|
| 462 | 46200 | 1 | sat_06 | 10 | 1 | 29 |
| 463 | 46300 | 1 | sat_06 | 10 | 1 | 29 |
| 465 | 46500 | 1 | sat_06 | 10 | 1 | 29 |
| 470 | 47000 | 1 | sat_06 | 10 | 1 | 29 |
| 473 | 47300 | 1 | sat_06 | 10 | 2 | 28 |
| 475 | 47500 | 1 | sat_06 | 10 | 2 | 28 |
| 478 | 47800 | 1 | sat_06 | 8 | 1 | 23 |
| 478 | 47800 | 1 | sat_06 | 10 | 1 | 29 |

Actual Algorithm 2 can omit nonpositive-priority nonedge groups, so its actual failure count requires an additional schedule/AoI-aware audit.
These results cover only the pilot snapshot grid at seed 0, not every physical slot in the full 480-second run.
