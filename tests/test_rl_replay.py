"""Replay-buffer and trajectory-to-transition tests."""

import numpy as np
import pytest

from tiblender_runtime.rl.actions import ACTION_SPACE, STOP_INDEX
from tiblender_runtime.rl.features import CLUSTER_DIM, STATE_DIM
from tiblender_runtime.rl.replay import ReplayBuffer, Transition, transitions_from_trajectory
from tiblender_runtime.rl.reward import step_reward, terminal_bonus


def _transition(value: float) -> Transition:
    return Transition(state=(value,), action=0, reward=value, next_state=(value,), done=False)


def test_transition_is_immutable():
    transition = _transition(1.0)

    with pytest.raises(Exception):
        transition.reward = 2.0  # type: ignore[misc]


def test_replay_buffer_evicts_oldest_beyond_capacity():
    buffer = ReplayBuffer(capacity=3)
    buffer.extend([_transition(float(index)) for index in range(5)])

    assert len(buffer) == 3
    assert [transition.reward for transition in buffer.transitions] == [2.0, 3.0, 4.0]


def test_replay_buffer_sample_is_deterministic_for_a_seeded_rng():
    buffer = ReplayBuffer(capacity=10)
    buffer.extend([_transition(float(index)) for index in range(10)])

    first = buffer.sample(4, np.random.default_rng(0))
    second = buffer.sample(4, np.random.default_rng(0))

    assert len(first) == 4
    assert [t.reward for t in first] == [t.reward for t in second]


def test_replay_buffer_sample_uses_replacement_when_short():
    buffer = ReplayBuffer(capacity=10)
    buffer.extend([_transition(1.0), _transition(2.0)])

    assert len(buffer.sample(5, np.random.default_rng(1))) == 5


def test_replay_buffer_roundtrip_preserves_transitions():
    buffer = ReplayBuffer(capacity=4)
    buffer.extend([_transition(1.5), Transition(state=(0.5,), action=1, reward=2.5, next_state=(0.5,), done=True, num_perspectives=3)])
    restored = ReplayBuffer.from_dict(buffer.to_dict())

    assert restored.capacity == 4
    assert restored.transitions == buffer.transitions
    assert restored.transitions[1].num_perspectives == 3


def test_replay_buffer_loads_payloads_without_num_perspectives():
    payload = {
        "capacity": 2,
        "size": 1,
        "transitions": [{"state": [0.0], "action": 0, "reward": 1.0, "next_state": [0.0], "done": False}],
    }

    restored = ReplayBuffer.from_dict(payload)

    assert restored.transitions[0].num_perspectives == 0


def _trajectory() -> list[dict]:
    return [
        {
            "step": 1,
            "perspective": "Infrastructure",
            "focus_type": "domain_registration_and_whois",
            "tool_type": "WHOIS_LOOKUP",
            "parameters": {"domain": "bad.example"},
            "status": "success",
            "new_iocs": 2,
            "new_cves": 1,
            "new_ttps": 0,
            "duplicate": False,
        },
        {
            "step": 2,
            "perspective": "Technical",
            "focus_type": "cve_exploitation_analysis",
            "tool_type": "NVD_CVE_LOOKUP",
            "parameters": {"cve": "CVE-2026-3055"},
            "status": "success",
            "new_iocs": 0,
            "new_cves": 0,
            "new_ttps": 1,
            "duplicate": True,
        },
    ]


def test_transitions_from_trajectory_appends_a_terminal_stop_action():
    transitions = transitions_from_trajectory(
        _trajectory(),
        cluster_state=np.zeros(CLUSTER_DIM),
        final_quality=0.8,
        perspectives_used={"Infrastructure", "Technical"},
        ioc_total=16,
    )

    assert len(transitions) == 3
    assert [t.action for t in transitions] == [
        ACTION_SPACE.index("domain_registration_and_whois"),
        ACTION_SPACE.index("cve_exploitation_analysis"),
        STOP_INDEX,
    ]
    assert [t.done for t in transitions] == [False, False, True]
    assert len(transitions[0].state) == STATE_DIM
    assert [t.num_perspectives for t in transitions] == [2, 2, 2]
    assert transitions[0].next_state == transitions[1].state
    assert transitions[0].reward == step_reward(delta_ioc=2, delta_cve=1, delta_ttp=0, duplicate_actions=0)
    assert transitions[1].reward == step_reward(delta_ioc=0, delta_cve=0, delta_ttp=1, duplicate_actions=1)
    assert transitions[2].reward == pytest.approx(
        terminal_bonus(
            quality_score=0.8,
            perspectives_used={"Infrastructure", "Technical"},
            actions_taken=2,
            ioc_total=16,
        )
    )


def test_transitions_from_trajectory_accepts_legacy_workflow_records():
    legacy = [
        {
            "step": 1,
            "perspective": "Infrastructure",
            "focus_types": ["domain_registration_and_whois", "hosting_and_osint"],
            "tool_type": "WHOIS_LOOKUP",
            "query_parameters": {"domain": "bad.example"},
            "result_status": "success",
            "discovered_iocs": {
                "urls": ["http://bad.example/login"],
                "domains": ["bad.example"],
                "ips": [],
                "hashes": [],
                "cves": ["CVE-2026-3055"],
            },
            "confidence": "HIGH",
        },
        {
            "step": 2,
            "perspective": "Infrastructure",
            "focus_types": ["domain_registration_and_whois"],
            "tool_type": "WHOIS_LOOKUP",
            "query_parameters": {"domain": "bad.example"},
            "result_status": "success",
            "discovered_iocs": {
                "urls": ["http://bad.example/login"],
                "domains": ["bad.example", "other.example"],
                "ips": [],
                "hashes": [],
                "cves": ["CVE-2026-3055"],
            },
            "confidence": "LOW",
        },
    ]

    transitions = transitions_from_trajectory(
        legacy,
        cluster_state=np.zeros(CLUSTER_DIM),
        final_quality=0.0,
        perspectives_used={"Infrastructure"},
        ioc_total=3,
    )

    assert len(transitions) == 3
    # Both legacy steps ran from the Infrastructure perspective only.
    assert {t.num_perspectives for t in transitions} == {1}
    # Step 1 discovers two IoC values and one CVE, and is not a duplicate action.
    assert transitions[0].reward == step_reward(delta_ioc=2, delta_cve=1, delta_ttp=0, duplicate_actions=0)
    # Step 2 repeats the same (tool_type, parameters) pair and only adds one new domain.
    assert transitions[1].reward == step_reward(delta_ioc=1, delta_cve=0, delta_ttp=0, duplicate_actions=1)


def test_transitions_from_trajectory_returns_nothing_for_an_empty_trajectory():
    assert transitions_from_trajectory(
        [],
        cluster_state=np.zeros(CLUSTER_DIM),
        final_quality=1.0,
        perspectives_used=set(),
        ioc_total=0,
    ) == []


def test_transitions_from_trajectory_rejects_a_wrong_cluster_state():
    with pytest.raises(ValueError):
        transitions_from_trajectory(
            _trajectory(),
            cluster_state=np.zeros(7),
            final_quality=1.0,
            perspectives_used=set(),
            ioc_total=0,
        )


def test_transitions_from_a_step_three_context_trajectory():
    """The Step 3 context trajectory feeds the replay converter unchanged."""
    trajectory = [
        {
            "step": 1,
            "perspective": "Infrastructure",
            "focus_type": "domain_registration_and_whois",
            "tool_type": "WHOIS_LOOKUP",
            "parameters": {"domain": "bad.example"},
            "status": "success",
            "new_iocs": 2,
            "new_cves": 0,
            "new_ttps": 0,
            "duplicate": False,
            "confidence": "MEDIUM",
            "mode": "multi_perspective",
        },
        {
            "step": 2,
            "perspective": "Technical",
            "focus_type": "cve_exploitation_analysis",
            "tool_type": "NVD_CVE_LOOKUP",
            "parameters": {"cve": "CVE-2026-12345"},
            "status": "success",
            "new_iocs": 0,
            "new_cves": 1,
            "new_ttps": 0,
            "duplicate": False,
            "confidence": "HIGH",
            "mode": "multi_perspective",
        },
    ]

    transitions = transitions_from_trajectory(
        trajectory,
        cluster_state=np.zeros(CLUSTER_DIM),
        final_quality=0.8,
        perspectives_used={"Infrastructure", "Technical"},
        ioc_total=2,
    )

    assert [transition.action for transition in transitions] == [
        ACTION_SPACE.index("domain_registration_and_whois"),
        ACTION_SPACE.index("cve_exploitation_analysis"),
        STOP_INDEX,
    ]
    assert all(transition.num_perspectives == 2 for transition in transitions)
    assert [transition.done for transition in transitions] == [False, False, True]
    assert all(len(transition.state) == STATE_DIM for transition in transitions)


def test_build_trajectory_records_prefers_a_supplied_context_trajectory():
    from tiblender_runtime.models import Cluster, IocSet, Severity, ThreatCategory
    from tiblender_runtime.rl.replay import build_trajectory_records

    cluster = Cluster(
        id="cluster-ctx",
        topic_name="context campaign",
        description="context campaign",
        posts=[],
        iocs=IocSet(),
        severity=Severity.HIGH,
        threat_category=ThreatCategory.MALWARE,
    )
    trajectory = [
        {
            "step": 1,
            "perspective": "Actor",
            "focus_type": "campaign_tracking",
            "tool_type": "SEARCH_ENGINE",
            "parameters": {"query": "bad.example campaign"},
            "status": "success",
            "new_iocs": 0,
            "new_cves": 0,
            "new_ttps": 1,
            "duplicate": False,
            "confidence": "MEDIUM",
            "mode": "rl_guided",
        }
    ]

    records = build_trajectory_records(cluster, [], context_trajectory=trajectory)

    assert records == [{"cluster_id": "cluster-ctx", "cluster_name": "context campaign", **trajectory[0]}]
    assert build_trajectory_records(cluster, []) == []
