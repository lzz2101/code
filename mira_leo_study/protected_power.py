"""Scheduled geographic-edge plus CSI-promoted 70/30 power allocation.

Geographic flags, membership and Algorithm-2 admission decisions are immutable.
Only a fully validated plan can be applied. Old power helpers are not used here.
"""
from __future__ import annotations

import math
import numpy as np

from .errors import ModelConsistencyError, OptionalDependencyError

POWER_MODE = "csi_expanded_protected_pool"
POOL_RULE_VERSION = "scheduled_geo_plus_strict_weaker_than_strongest_geo_v1"
TOLERANCE = 1e-10


def _order(group):
    return group.gain, group.message_id, group.group_id


def classify_power_pools(scheduled_groups) -> dict:
    scheduled = [g for g in scheduled_groups if g.scheduled]
    if len({g.key for g in scheduled}) != len(scheduled):
        raise ModelConsistencyError("DUPLICATE_GROUP_KEY")
    for group in scheduled:
        if not math.isfinite(group.gain) or group.gain <= 0:
            raise ModelConsistencyError("INVALID_GROUP_GAIN: expected finite positive raw CSI power gain")
        if len(group.member_indices) == 0:
            raise ModelConsistencyError("EMPTY_GROUP")
        if not math.isfinite(group.max_elevation_deg) or not -90 <= group.max_elevation_deg <= 90:
            raise ModelConsistencyError("INVALID_GROUP_ELEVATION")
    ordered = sorted(scheduled, key=_order)
    edge = [g for g in ordered if g.is_edge]
    reference = max((g.gain for g in edge), default=None)
    promoted = [g for g in ordered if not g.is_edge and reference is not None and g.gain < reference]
    protected_ids = {id(g) for g in edge + promoted}
    return dict(ordered=ordered, geo_edge=edge, promoted=promoted,
                protected=[g for g in ordered if id(g) in protected_ids],
                ordinary=[g for g in ordered if id(g) not in protected_ids], g_ref=reference)


def _violations(ordered, powers):
    return [dict(weaker=a.key, stronger=b.key, weaker_gain=a.gain, stronger_gain=b.gain,
                 weaker_power=powers[id(a)], stronger_power=powers[id(b)],
                 magnitude=powers[id(b)] - powers[id(a)])
            for a, b in zip(ordered, ordered[1:]) if powers[id(a)] < powers[id(b)] - TOLERANCE]


def _project_l1(ordered, target, floors, labels, budgets, p_sat):
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        raise OptionalDependencyError("csi_expanded_protected_pool requires scipy") from exc
    n = len(ordered)
    inequalities, limits = [], []
    for index in range(n - 1):
        row = np.zeros(2*n); row[index] = -1.; row[index+1] = 1.
        inequalities.append(row); limits.append(0.)
    row = np.r_[np.ones(n), np.zeros(n)]
    inequalities.append(row); limits.append(p_sat)
    for index, group in enumerate(ordered):
        row = np.zeros(2*n); row[index] = 1.; row[n+index] = -1.
        inequalities.append(row); limits.append(target[id(group)])
        row = np.zeros(2*n); row[index] = -1.; row[n+index] = -1.
        inequalities.append(row); limits.append(-target[id(group)])
    equalities, totals = [], []
    for label, budget in budgets.items():
        indices = [i for i, g in enumerate(ordered) if labels[id(g)] == label]
        if indices:
            row = np.zeros(2*n); row[indices] = 1.
            equalities.append(row); totals.append(budget)
    try:
        result = linprog(np.r_[np.zeros(n), np.ones(n)], A_ub=np.asarray(inequalities),
                         b_ub=np.asarray(limits), A_eq=np.asarray(equalities), b_eq=np.asarray(totals),
                         bounds=[(floors[id(g)], None) for g in ordered]+[(0., None)]*n,
                         method="highs", options={"primal_feasibility_tolerance": 1e-10,
                                                   "dual_feasibility_tolerance": 1e-10})
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return None, 4, f"{type(exc).__name__}: {exc}"
    status = int(result.status)
    solution = None
    if result.success:
        try:
            solution = np.asarray(result.x, dtype=float)
        except (ValueError, TypeError):
            return None, 4, "LP returned an invalid solution array"
        if solution.ndim != 1 or len(solution) != 2*n or not np.all(np.isfinite(solution)):
            return None, 4, "LP returned a nonfinite or incorrectly sized solution"
        solution = solution[:n]
    return solution, status, str(result.message)


def allocate_csi_expanded_protected_power(config, scheduled_groups, *, apply=False) -> dict:
    pools = classify_power_pools(scheduled_groups)
    ordered, protected, ordinary = pools["ordered"], pools["protected"], pools["ordinary"]
    p_sat = float(config.get("channel.normalized_satellite_power"))
    v_phi = float(config.get("algorithm.V_phi"))
    if not math.isfinite(p_sat) or p_sat <= 0 or not math.isfinite(v_phi) or v_phi <= 0:
        raise ModelConsistencyError("INVALID_POWER_PARAMETERS")
    mixed = bool(protected and ordinary)
    budgets = dict(protected=(.7*p_sat if mixed else p_sat if protected else 0.),
                   ordinary=(.3*p_sat if mixed else p_sat if ordinary else 0.))
    labels = {id(g): label for label, subset in (("protected", protected), ("ordinary", ordinary)) for g in subset}
    floors, target, floor_totals, floor_violations = {}, {}, {}, []
    for label, subset in (("protected", protected), ("ordinary", ordinary)):
        for index, group in enumerate(subset):
            floors[id(group)] = v_phi*(len(subset)-index)
        floor_totals[label] = sum(floors[id(g)] for g in subset)
        if floor_totals[label] > budgets[label]+TOLERANCE:
            floor_violations.append(dict(pool=label, floor_total=floor_totals[label], budget=budgets[label]))
        residual = budgets[label]-floor_totals[label]
        for group in subset:
            target[id(group)] = floors[id(group)]+residual/len(subset) if residual >= 0 else floors[id(group)]
    candidate = dict(target)
    before = _violations(ordered, target) if not floor_violations else []
    projection_used = bool(before and not floor_violations)
    lp_status, lp_message, reason = None, "", ""
    if floor_violations:
        reason = "FLOOR_EXCEEDS_POOL"
    elif projection_used:
        solution, lp_status, lp_message = _project_l1(ordered, target, floors, labels, budgets, p_sat)
        if solution is None:
            reason = "POOL_SIC_INCOMPATIBLE" if lp_status == 2 else "LP_NUMERIC_FAILURE"
        else:
            candidate = {id(g): float(p) for g, p in zip(ordered, solution)}
    after = _violations(ordered, candidate)
    budget_violations = []
    if not reason:
        for label, subset in (("protected", protected), ("ordinary", ordinary)):
            power = sum(candidate[id(g)] for g in subset)
            if abs(power-budgets[label]) > TOLERANCE:
                budget_violations.append(dict(pool=label, power=power, budget=budgets[label]))
        finite = all(math.isfinite(p) and p >= 0 for p in candidate.values())
        floor_valid = all(candidate[id(g)] >= floors[id(g)]-TOLERANCE for g in ordered)
        if (not finite or not floor_valid or after or budget_violations
                or sum(candidate.values()) > p_sat+TOLERANCE):
            reason = "LP_NUMERIC_FAILURE" if projection_used else "INVALID_TARGET_CONSTRAINTS"
    valid = not bool(reason)
    actual = candidate if valid else {id(g): 0. for g in ordered}
    allocated = {label: sum(actual[id(g)] for g in subset)
                 for label, subset in (("protected", protected), ("ordinary", ordinary))}
    promoted_ids = {id(g) for g in pools["promoted"]}
    details = []
    for rank, group in enumerate(ordered, 1):
        data = dict(power_pool=labels[id(group)], power_pool_promoted=id(group) in promoted_ids,
                    power_pool_csi_reference_gain=pools["g_ref"], global_csi_rank=rank,
                    target_power=target[id(group)], power_floor=floors[id(group)],
                    pool_target_power=budgets[labels[id(group)]],
                    pool_allocated_power=allocated[labels[id(group)]],
                    geo_edge=bool(group.is_edge), gain=group.gain, max_elevation_deg=group.max_elevation_deg)
        details.append(dict(key=group.key, **data))
        if apply:
            group.metadata.update(data, power_failure_no_transmission=not valid)
            group.power = actual[id(group)]
            if not valid:
                group.success = group.rate_pass = group.sic_pass = False
                group.sinr = group.rate_bps_hz = 0.
    return dict(mode=POWER_MODE, power_pool_rule_version=POOL_RULE_VERSION, valid=valid,
                status=("SIC_POWER_PROJECTED" if projection_used else "OK") if valid else "POWER_INFEASIBLE_NO_TRANSMISSION",
                transmission_failure=not valid, transmission_attempted=bool(valid and ordered),
                power_infeasible_reason=reason, power_failure_reason=reason,
                geo_edge_count_scheduled=len(pools["geo_edge"]), promoted_weak_nonedge_count=len(pools["promoted"]),
                protected_count=len(protected), ordinary_count=len(ordinary), g_ref=pools["g_ref"],
                pool_budget_mode="mixed_70_30" if mixed else "protected_full" if protected else "ordinary_full" if ordinary else "empty",
                protected_budget=budgets["protected"], ordinary_budget=budgets["ordinary"],
                protected_power_actual=allocated["protected"], ordinary_power_actual=allocated["ordinary"],
                protected_floor_total=floor_totals["protected"], ordinary_floor_total=floor_totals["ordinary"],
                pool_target_power=budgets, pool_allocated_power=allocated, floor_total=floor_totals,
                group_details=details, target_powers=target, powers=actual, final_powers=actual, floors=floors,
                total_power=sum(candidate.values()), actual_total_power=sum(actual.values()),
                sic_projection_used=projection_used, projection_infeasible=bool(reason),
                projection_solver_status=lp_status, projection_solver_message=lp_message,
                lp_status=lp_status, sic_order_violations_before=before,
                sic_order_violations_after=after if valid else [], sic_order_violations=after if valid else [],
                candidate_sic_order_violations=after,
                floor_violations=floor_violations, budget_violations=budget_violations, removed_groups=[],
                # Read compatibility aliases mean POWER POOLS, not geographic classes.
                edge_count=len(protected), nonedge_count=len(ordinary),
                edge_budget=budgets["protected"], nonedge_budget=budgets["ordinary"],
                edge_floor_total=floor_totals["protected"], nonedge_floor_total=floor_totals["ordinary"],
                edge_residual=budgets["protected"]-floor_totals["protected"],
                nonedge_residual=budgets["ordinary"]-floor_totals["ordinary"])
