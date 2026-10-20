"""Reward function from the paper's RL appendix.

    r_t = alpha * q_T * 1[terminal] + gamma_info * g_t - beta * c_t - delta * p_red

with the information gain g_t = 2*dIoC + 5*dCVE + 3*dTTP. The tool-cost term is
disabled (beta = 0) because tool latency is not part of the released trace, and
the redundancy penalty counts repeated (tool_type, parameters) actions.
"""

from __future__ import annotations

from .actions import PERSPECTIVE_FOCUS_INDICES, synergy_bonus


ALPHA = 100.0
GAMMA_INFO = 1.0
BETA = 0.0
DELTA = 1.0

W_IOC = 2.0
W_CVE = 5.0
W_TTP = 3.0

PERSPECTIVE_COVERAGE_BONUS = 10.0
PERSPECTIVE_COUNT = 4

HIGH_EFFICIENCY_BONUS = 20.0
HIGH_EFFICIENCY_MAX_ACTIONS = 10
HIGH_EFFICIENCY_MIN_IOCS = 15
MEDIUM_EFFICIENCY_BONUS = 10.0
MEDIUM_EFFICIENCY_MAX_ACTIONS = 15
MEDIUM_EFFICIENCY_MIN_IOCS = 10


def information_gain(delta_ioc: int, delta_cve: int, delta_ttp: int) -> float:
    """Weighted count of newly discovered indicators, CVEs, and TTPs."""
    if delta_ioc < 0 or delta_cve < 0 or delta_ttp < 0:
        raise ValueError("information gain deltas must be non-negative")
    return W_IOC * float(delta_ioc) + W_CVE * float(delta_cve) + W_TTP * float(delta_ttp)


def step_reward(
    *,
    delta_ioc: int,
    delta_cve: int,
    delta_ttp: int,
    duplicate_actions: int,
    cost: float = 0.0,
) -> float:
    """Per-step reward: information gain minus the redundancy penalty."""
    if duplicate_actions < 0:
        raise ValueError("duplicate_actions must be non-negative")
    gain = information_gain(delta_ioc, delta_cve, delta_ttp)
    return GAMMA_INFO * gain - BETA * float(cost) - DELTA * float(duplicate_actions)


def efficiency_bonus(actions_taken: int, ioc_total: int) -> float:
    """Tiered bonus for reaching a useful IoC yield in few actions."""
    if actions_taken < HIGH_EFFICIENCY_MAX_ACTIONS and ioc_total >= HIGH_EFFICIENCY_MIN_IOCS:
        return HIGH_EFFICIENCY_BONUS
    if actions_taken < MEDIUM_EFFICIENCY_MAX_ACTIONS and ioc_total >= MEDIUM_EFFICIENCY_MIN_IOCS:
        return MEDIUM_EFFICIENCY_BONUS
    return 0.0


def terminal_bonus(
    *,
    quality_score: float,
    perspectives_used: set[str],
    actions_taken: int,
    ioc_total: int,
) -> float:
    """Terminal reward: report quality, perspective coverage, efficiency, synergy."""
    if actions_taken < 0 or ioc_total < 0:
        raise ValueError("actions_taken and ioc_total must be non-negative")
    # Only contract perspectives count, matching synergy_bonus.
    known = {name for name in perspectives_used if name in PERSPECTIVE_FOCUS_INDICES}
    coverage = PERSPECTIVE_COVERAGE_BONUS * len(known) / PERSPECTIVE_COUNT
    return (
        ALPHA * float(quality_score)
        + coverage
        + efficiency_bonus(actions_taken, ioc_total)
        + synergy_bonus(perspectives_used)
    )
