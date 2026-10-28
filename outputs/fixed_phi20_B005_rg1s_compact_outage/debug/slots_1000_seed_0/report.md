# Fixed-parameter MIRA-LEO paper results

phi_theta_deg=20.0; B_penalty=0.05; Fixed grouping a=4.
Horizon: 1000 slots; seed: 0; full horizon: False.
AoI is the constellation sum in slots; the running mean uses the entire horizon, including exploration.
Per-satellite mean includes all eight satellites (inactive contributions are zero).
All three policies use the same scheduler, power allocator, SIC, AoI, trajectory and indexed channel trace.
Membership and actions are held for 1 second (100 physical slots); global regrouping epochs index SW-UCB, whose nominal 10-second window contains 10 epochs.
Each physical slot uses current-slot error-free CSI to update worst-user gain, edge labels, strict Algorithm 2, TWC/AIM powers, global SIC and AoI.
Algorithm 2 uses U=62, all edge groups when edge count <= U, otherwise top-U edge groups; only positive-benefit nonedge groups fill remaining capacity.
The allocator uses fixed class pools, current-CSI staircases and continuous residuals. Cross-class SIC repair changes powers only. A proven infeasible LP mutes the satellite for that slot; all its scheduled updates fail and remain in AoI/reward statistics. The schedule is unchanged and actual power is zero.
Power-outage counters are separate from channel decoding failures. invalid_power_or_sic_count counts unhandled/transmitted constraint violations, not explicitly recorded zero-power outages. Numerical/solver errors that do not prove infeasibility still stop the run.
Bandit rewards average the per-slot normalized rewards in each epoch and update the agent that selected the action, including when a handover occurs mid-epoch.
Physical slot t and bandit epoch e are distinct indices; the paper's slot-based regret notation requires this mapping for the two-timescale implementation.
A logical G-RNTI includes service, group index and sorted membership. Only consecutive visible assignments on the same satellite count as regrouping.
Initial acquisition, coverage exit and handover are excluded; the denominator is all eligible subscribed device-service instances times the full evaluated duration.
Action-change counts and whole-message membership-change diagnostics are separate from device-level signalling events.
Additional tau values are project sensitivity points. Other parameters and environment remain fixed.

- MIRA-LEO (SW-UCB): mean AoI=153.57887; last 20% mean=153.71801; power-outage cohort-slots=0; status=COMPLETED.
- Random grouping: mean AoI=153.49887; last 20% mean=153.71801; power-outage cohort-slots=0; status=COMPLETED.
- Fixed grouping (a=4): mean AoI=153.53887; last 20% mean=153.71801; power-outage cohort-slots=0; status=COMPLETED.
MIRA-LEO does not strictly outperform every baseline on full-horizon mean AoI; all measured values and exploration slots are retained. See the scheduler/power/action CSV diagnostics.

AIM-static, GA-grouping and PPO-multicast: NOT_IMPLEMENTED.

| Policy | tau_s | Re-signalling events | Device-service instances | Duration_min | Events/device/min | Action changes | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| mira_sw_ucb | 10.0 | 2160 | 240 | 0.16666666666666666 | 54.0 | 16 | COMPLETED |
| aim_static |  |  |  |  |  |  | NOT_IMPLEMENTED |
| ga_grouping |  |  |  |  |  |  | NOT_IMPLEMENTED |
