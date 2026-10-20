"""Reward-function tests mirroring the paper's RL appendix."""

import pytest

from tiblender_runtime.rl.actions import synergy_bonus
from tiblender_runtime.rl.reward import (
    ALPHA,
    BETA,
    DELTA,
    GAMMA_INFO,
    W_CVE,
    W_IOC,
    W_TTP,
    information_gain,
    step_reward,
    terminal_bonus,
)


def test_reward_constants_match_the_paper():
    assert (ALPHA, GAMMA_INFO, BETA, DELTA) == (100.0, 1.0, 0.0, 1.0)
    assert (W_IOC, W_CVE, W_TTP) == (2.0, 5.0, 3.0)


def test_information_gain_is_the_weighted_sum():
    assert information_gain(0, 0, 0) == 0.0
    assert information_gain(2, 1, 0) == 9.0
    assert information_gain(1, 1, 1) == 10.0


def test_step_reward_subtracts_duplicate_penalty():
    assert step_reward(delta_ioc=2, delta_cve=1, delta_ttp=0, duplicate_actions=1) == 8.0
    assert step_reward(delta_ioc=0, delta_cve=0, delta_ttp=0, duplicate_actions=0) == 0.0
    assert step_reward(delta_ioc=0, delta_cve=0, delta_ttp=0, duplicate_actions=2) == -2.0


def test_step_reward_ignores_cost_because_beta_is_zero():
    assert step_reward(delta_ioc=1, delta_cve=0, delta_ttp=0, duplicate_actions=0, cost=25.0) == 2.0


def test_step_reward_rejects_negative_deltas():
    with pytest.raises(ValueError):
        step_reward(delta_ioc=-1, delta_cve=0, delta_ttp=0, duplicate_actions=0)


def test_terminal_bonus_high_efficiency_tier():
    value = terminal_bonus(
        quality_score=0.9,
        perspectives_used={"Infrastructure", "Technical", "Social", "Actor"},
        actions_taken=8,
        ioc_total=20,
    )
    expected = 100.0 * 0.9 + 10.0 + 20.0 + synergy_bonus({"Infrastructure", "Technical", "Social", "Actor"})

    assert value == pytest.approx(expected)


def test_terminal_bonus_medium_efficiency_tier():
    value = terminal_bonus(
        quality_score=0.5,
        perspectives_used={"Technical", "Actor"},
        actions_taken=12,
        ioc_total=11,
    )

    assert value == pytest.approx(100.0 * 0.5 + 10.0 * 2 / 4 + 10.0 + 0.9)


def test_terminal_bonus_no_efficiency_tier():
    value = terminal_bonus(
        quality_score=0.0,
        perspectives_used={"Technical"},
        actions_taken=20,
        ioc_total=3,
    )

    assert value == pytest.approx(10.0 * 1 / 4)


def test_terminal_bonus_ignores_unknown_perspectives():
    value = terminal_bonus(
        quality_score=0.0,
        perspectives_used={"Technical", "Marketing"},
        actions_taken=20,
        ioc_total=0,
    )

    assert value == pytest.approx(10.0 * 1 / 4)


def test_terminal_bonus_requires_ioc_threshold_for_the_high_tier():
    low_iocs = terminal_bonus(quality_score=0.0, perspectives_used=set(), actions_taken=5, ioc_total=12)

    assert low_iocs == pytest.approx(10.0)
