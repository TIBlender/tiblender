"""Action-space tests for the MultiAction DQN policy."""

from tiblender_runtime.contract import REQUIRED_FOCUS_TYPES, REQUIRED_PERSPECTIVES
from tiblender_runtime.rl.actions import (
    ACTION_SPACE,
    FOCUS_PERSPECTIVE,
    FOCUS_TYPES,
    PERSPECTIVE_FOCUS_INDICES,
    PERSPECTIVES,
    STOP_ACTION,
    STOP_INDEX,
    SYNERGY,
    synergy_bonus,
)


def test_action_space_has_eighteen_focus_types_plus_stop():
    assert len(FOCUS_TYPES) == 18
    assert len(ACTION_SPACE) == 19
    assert ACTION_SPACE[-1] == STOP_ACTION == "STOP"
    assert STOP_INDEX == 18
    assert len(set(ACTION_SPACE)) == 19


def test_focus_types_follow_contract_order():
    expected = [focus for perspective in REQUIRED_PERSPECTIVES for focus in REQUIRED_FOCUS_TYPES[perspective]]

    assert FOCUS_TYPES == expected
    assert PERSPECTIVES == list(REQUIRED_PERSPECTIVES)


def test_perspective_focus_indices_partition_the_focus_space():
    collected: list[int] = []
    for perspective in PERSPECTIVES:
        indices = PERSPECTIVE_FOCUS_INDICES[perspective]
        assert indices == sorted(indices)
        for index in indices:
            assert FOCUS_PERSPECTIVE[FOCUS_TYPES[index]] == perspective
        collected.extend(indices)

    assert sorted(collected) == list(range(18))
    assert len(collected) == 18


def test_synergy_bonus_sums_pairwise_values():
    assert synergy_bonus(set()) == 0.0
    assert synergy_bonus({"Technical"}) == 0.0
    assert synergy_bonus({"Technical", "Actor"}) == 0.9
    assert synergy_bonus({"Actor", "Technical"}) == 0.9
    expected = SYNERGY[("Infrastructure", "Technical")] + SYNERGY[("Infrastructure", "Social")] + SYNERGY[("Technical", "Social")]
    assert synergy_bonus({"Infrastructure", "Technical", "Social"}) == round(expected, 6)


def test_synergy_bonus_ignores_unknown_perspectives():
    assert synergy_bonus({"Technical", "Unknown"}) == 0.0
