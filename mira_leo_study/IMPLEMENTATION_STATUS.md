# Current Implementation Status

- User changes: fixed elevation threshold restored to 20 degrees; proven power infeasibility is a zero-power failed update.
- Strict Algorithm 2 and TWC/AIM staircase remain; no allocator group drop or budget relaxation.
- Compact [1,2,4,5,8,10], regrouping 1 s, physical slot 10 ms, SW-UCB 10 s / 10 epochs remain.
- LP status=2 plus no floor error may mute the serving satellite for the slot. Scheduled flags stay unchanged, actual power=0, all its planned updates fail, AoI/reward include that failure, and other satellites operate normally.
- Numerical errors, configuration/floor errors and schedule mutations remain errors.
- Pilot / MIRA / Random / Fixed share this handling. Failed pilot counterfactual samples are retained.
- Infeasible plan diagnostics remain invalid; actual-emission and outage counters are explicit. Status may be COMPLETED_WITH_POWER_OUTAGES.
- New reward context version and checkpoint schema=4 reject old bounds/checkpoints.
- Default root: outputs/fixed_phi20_B005_rg1s_compact_outage.
- Non-simulation tests: 87 total, 82 passed, 5 simulation tests skipped. No real reward pilot or AoI/learning study was run for this delivery.
- Historical power-only diagnostic (not rerun for this reversion): seed 0, 90 snapshot times x 2 cohorts x 6 actions=1080 full-candidate configurations; at 30 degrees 170 LP-infeasible, 910 feasible, 0 other invalid. All 170 infeasible cases became zero-power failures. Earlier 20-degree audit found 8/1080.
- Those diagnostic counts assume all candidates scheduled and are not actual policy outage frequencies.
- AIM-static / GA-grouping / PPO-multicast remain NOT_IMPLEMENTED.

- Current reversion: fixed phi=20 with the same zero-power outage policy. 30-degree bounds are rejected; no simulation/pilot/scan is run for this change.
