"""Action space for the MultiAction DQN investigation policy.

The 18 focus types are the flattened contract focus list (Infrastructure,
Technical, Social, Actor) plus a terminal ``STOP`` action, giving the 19-way
action space described in the paper's RL appendix.
"""

from __future__ import annotations

from itertools import combinations

from ..contract import REQUIRED_FOCUS_TYPES, REQUIRED_PERSPECTIVES


PERSPECTIVES: list[str] = list(REQUIRED_PERSPECTIVES)

FOCUS_TYPES: list[str] = [focus for perspective in PERSPECTIVES for focus in REQUIRED_FOCUS_TYPES[perspective]]

STOP_ACTION = "STOP"

ACTION_SPACE: list[str] = [*FOCUS_TYPES, STOP_ACTION]

STOP_INDEX: int = len(FOCUS_TYPES)

FOCUS_PERSPECTIVE: dict[str, str] = {
    focus: perspective for perspective in PERSPECTIVES for focus in REQUIRED_FOCUS_TYPES[perspective]
}

PERSPECTIVE_FOCUS_INDICES: dict[str, list[int]] = {
    perspective: [index for index, focus in enumerate(FOCUS_TYPES) if FOCUS_PERSPECTIVE[focus] == perspective]
    for perspective in PERSPECTIVES
}

# Perspective-pair synergy weights used by the terminal reward. Higher values
# mean the two perspectives corroborate each other more often in practice.
SYNERGY: dict[tuple[str, str], float] = {
    ("Technical", "Actor"): 0.9,
    ("Infrastructure", "Technical"): 0.8,
    ("Infrastructure", "Social"): 0.6,
    ("Social", "Actor"): 0.7,
    ("Infrastructure", "Actor"): 0.5,
    ("Technical", "Social"): 0.4,
}

_SYNERGY_LOOKUP: dict[frozenset[str], float] = {frozenset(pair): value for pair, value in SYNERGY.items()}


def synergy_bonus(perspectives: set[str]) -> float:
    """Sum the synergy weight of every known perspective pair that was used."""
    known = sorted(name for name in perspectives if name in PERSPECTIVE_FOCUS_INDICES)
    total = sum(_SYNERGY_LOOKUP.get(frozenset(pair), 0.0) for pair in combinations(known, 2))
    return round(total, 6)


def focus_index(focus_type: str) -> int:
    """Return the action index of ``focus_type`` or raise for an unknown name."""
    try:
        return ACTION_SPACE.index(focus_type)
    except ValueError as error:  # pragma: no cover - defensive, callers normalise first
        raise ValueError(f"unknown action: {focus_type!r}") from error
