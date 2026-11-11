"""Post-update P1=C-Delta in slot units; the bandit penalty is separate."""
from __future__ import annotations

import math


def p1_slot_value(successful_groups, weighted_delta_slots):
    if successful_groups < 0 or not math.isfinite(weighted_delta_slots) or weighted_delta_slots < 0:
        raise ValueError("P1 requires nonnegative success count and finite post-update AoI")
    return float(successful_groups-weighted_delta_slots)


def objective_summary(*, sum_c, sum_delta, slots, warm_c, warm_delta, warm_slots,
                      warmup_s, slot_dt_s):
    value = float(sum_c-sum_delta)
    warm_value = float(warm_c-warm_delta) if warm_slots else None
    return dict(sum_C=int(sum_c), mean_C=sum_c/slots, sum_weighted_delta_slots=float(sum_delta),
                mean_weighted_delta_slots=sum_delta/slots, sum_P1=value, mean_P1_per_slot=value/slots,
                post_warmup_sum_C=int(warm_c) if warm_slots else None,
                post_warmup_mean_C=warm_c/warm_slots if warm_slots else None,
                post_warmup_sum_weighted_delta_slots=float(warm_delta) if warm_slots else None,
                post_warmup_mean_weighted_delta_slots=warm_delta/warm_slots if warm_slots else None,
                post_warmup_sum_P1=warm_value,
                post_warmup_mean_P1=warm_value/warm_slots if warm_slots else None,
                post_warmup_slots=warm_slots, post_warmup_start_time_s=warmup_s,
                post_warmup_end_time_s_exclusive=slots*slot_dt_s,
                P1_convention="successful groups minus post-update weighted-sum AoI in slots; no penalty",
                achieved_objective_not_global_optimum=True)
