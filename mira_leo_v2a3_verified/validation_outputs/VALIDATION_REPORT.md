# MIRA-LEO v2a3 Validation Report

## Scope

Validated timestamped SW-UCB, bounded reward direction, per-satellite Algorithm 3 power and SIC safety, and delay conversion. Real TLE/SGP4 trajectories and statistical performance claims are outside this validation.

The source package `mira_leo_v2a3` was preserved. Changes were made only in `mira_leo_v2a3_verified`. The workspace exposes a `.git` reparse point, but Git reported that this directory is not a repository, so no branch or commit was created.

## Environment

- Python: 3.12.13
- Dependencies used: Python standard library and NumPy already present in the bundled runtime.
- Unit tests: 57/57 passed, 0 failed, 0 errors.

Commands:

- `python -m unittest discover -s mira_leo_v2a3_verified/tests -v`
- `python -m mira_leo_v2a3_verified.validation.run_validation --controlled --strict`
- `python -m mira_leo_v2a3_verified.validation.run_validation --controlled --strict --T 50 --seeds 42`

## Formula Mapping

| Paper definition | Production implementation |
|---|---|
| SW-UCB window and Eq. (16) | `algorithms/sw_ucb.py` timestamped observations |
| Raw objective C - weighted AoI - edge penalty | `aoi/metrics.py::compute_regrouping_reward_details` |
| Algorithm 2 handover-aware priority | `algorithms/scheduler.py` |
| Algorithm 3 subset floors and residual | `algorithms/power_allocation.py` |
| Global SIC order constraint (15c) | power allocator plus `validation/validators.py` |
| Decoder-message SINR chain | `noma/transmission.py::evaluate_noma_decoding` |
| AoI delay recursion | `aoi/metrics.py::update_user_aoi` and `time_units.py` |

The paper requires normalized rewards in [0,1] but does not provide an explicit numerical bound formula. This verified implementation uses conservative fixed bounds derived once from configuration and horizon. It is an implementation choice, not a paper-specified normalization.

## Baseline Evidence

- Existing source tests: 16/16 passed, but all five targeted validation-risk checks failed.
- T=1000, seed=42 source run: reward min -0.012095700570307831, max 0.9957124235748394, 9 out-of-bound rewards.
- Source used one SW-UCB object for all satellites and stored no observation slots.
- Source added K_h=0.025 directly to slot-based AoI; verified conversion is 25 ms / 10 ms = 2.5 slots.
- Source recorded interleaved subset power-order violations but allowed NOMA transmission; verified code rejects them before transmission.

## Results

| ID | Validation | Source result | Verified result | Status | Evidence |
|---|---|---|---|---|---|
| SW-01 | basic sliding window | FAIL | PASS | PASS | test_sw_ucb_window |
| SW-02 | stale arm expires | FAIL | PASS | PASS | test_sw_ucb_window |
| RW-06 | out-of-range detection | FAIL | PASS | PASS | test_reward_bounds |
| PW-07 | per-satellite overflow injection | not covered | PASS | PASS | test_power_constraints |
| SIC-03 | interleaved subset CSI | continued transmission | rejected | PASS | test_sic_decodability |
| DEC-04 | success recomputation | not covered | PASS | PASS | test_sic_decodability |
| TM-04 | single handover | 0.025 slot | 2.5 slots | PASS | test_time_units |
| INT-06 | independent satellite SW-UCB | shared policy | independent | PASS | controlled_scenarios.csv |

Runtime violation totals:

| Violation | Count |
|---|---:|
| stale_sw_ucb_observations | 0 |
| reward_out_of_bounds | 0 |
| nonfinite_rewards | 0 |
| negative_power | 0 |
| satellite_power_overflow | 0 |
| unscheduled_positive_power | 0 |
| power_floor_infeasible_success | 0 |
| sic_power_order | 0 |
| sic_decoding | 0 |
| time_unit | 0 |
| repeated_handover_charge | 0 |

Controlled coverage:

| Branch | Covered |
|---|---|
| edge_only_tested | True |
| nonedge_only_tested | True |
| mixed_edge_nonedge_tested | True |
| interleaved_csi_tested | True |
| handover_event_tested | True |
| multi_satellite_policy_tested | True |

## Time Units

| Quantity | Source unit | Internal unit | Default |
|---|---|---|---:|
| slot_time_s | seconds/slot | seconds/slot | 0.01 |
| handover_delay_s | seconds | continuous slots | 0.025 s = 2.5 slots |
| isl_delay_s | seconds/hop | continuous slots/hop | 0.005 s = 0.5 slots |
| processing_delay_slots | slots | slots | 1.0 |
| propagation delay | seconds from distance/c | slots | divided by slot_time_s |
| AoI state | slots | slots | continuous |

Handover is represented as a one-slot event by the serving-satellite assignment. A repeated event ID is charged only once.

## Modified Files

- `config.py`, `time_units.py`: explicit units and fixed reward bounds.
- `algorithms/sw_ucb.py`: global-slot window with per-satellite state.
- `aoi/metrics.py`, `algorithms/scheduler.py`: reward and delay corrections.
- `algorithms/power_allocation.py`, `noma/transmission.py`: strict global SIC order and pairwise decoding records.
- `main.py`: per-satellite policy/reward and per-slot validation records.
- `validation/*`, `tests/*`: validators, controlled scenarios, reports, and regression coverage.

## Blocked Or Deferred

- The paper does not provide an explicit numerical reward-normalization bound; the fixed linear mapping is an implementation choice.
- The T=1000, seeds 0-4 validation run was not executed in this no-simulation delivery.

## Conclusion

Executable checks status: PASS. Overall validation status: BLOCKED.
The verified code is ready for deterministic software testing. Do not claim complete paper-level validation or introduce real trajectories until the deferred full multi-seed run is executed and the reward normalization choice is accepted or replaced by an explicit paper bound.
