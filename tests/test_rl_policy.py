"""Mode-selection tests for the RL-guided investigation policy."""

import json
import logging
from datetime import UTC, datetime, timedelta

from tiblender_runtime.models import Cluster, IocSet, NormalizedPost, Post, Severity, ThreatCategory
from tiblender_runtime.rl.actions import FOCUS_TYPES
from tiblender_runtime.rl.features import cluster_embedding
from tiblender_runtime.rl.policy import (
    SIMILARITY_WEIGHTS,
    InvestigationStrategy,
    brand_tokens,
    select_investigation_strategy,
    similarity_features,
    similarity_to_report,
)
from tiblender_runtime.rl_policy import ACTION_SPACE
from tiblender_runtime.rl_policy import select_investigation_strategy as shim_select
from tiblender_runtime.settings import Settings


NOW = datetime(2026, 9, 9, 12, 0, 0, tzinfo=UTC)


def _cluster() -> Cluster:
    post = NormalizedPost(
        post=Post(id="p1", platform="x", content="Citrix NetScaler CVE-2026-3055 exploitation bad.example"),
        normalized_content="Citrix NetScaler CVE-2026-3055 active exploitation bad.example",
        iocs=IocSet(domains=["bad.example"], cves=["CVE-2026-3055"]),
        threat_category=ThreatCategory.VULNERABILITY,
    )
    return Cluster(
        id="cluster-netscaler",
        topic_name="Citrix NetScaler CVE-2026-3055 exploitation",
        description="Citrix NetScaler active exploitation campaign",
        posts=[post],
        iocs=post.iocs,
        severity=Severity.CRITICAL,
        threat_category=ThreatCategory.VULNERABILITY,
    )


def _matching_history_entry(cluster: Cluster) -> dict:
    return {
        "cluster_id": "cluster-netscaler-prev",
        "report_id": "TIBLENDER-PREV",
        "signature": [],
        "platforms": ["x"],
        "category": "vulnerability",
        "severity": "CRITICAL",
        "cves": ["CVE-2026-3055"],
        "iocs": ["bad.example"],
        "brands": ["Citrix", "NetScaler"],
        "created_at": (NOW - timedelta(days=1)).isoformat().replace("+00:00", "Z"),
        "embedding": cluster_embedding(cluster).tolist(),
    }


def _unrelated_history_entry() -> dict:
    other = Cluster(
        id="cluster-other",
        topic_name="Retail gift card refund fraud",
        description="Refund fraud ring reselling gift cards",
        posts=[
            NormalizedPost(
                post=Post(id="p9", platform="telegram", content="gift card refund fraud service"),
                normalized_content="gift card refund fraud service reseller",
                iocs=IocSet(domains=["refunds.example"]),
                threat_category=ThreatCategory.FRAUD,
            )
        ],
        iocs=IocSet(domains=["refunds.example"]),
        severity=Severity.MEDIUM,
        threat_category=ThreatCategory.FRAUD,
    )
    return {
        "cluster_id": "cluster-other",
        "report_id": "TIBLENDER-OTHER",
        "signature": [],
        "platforms": ["telegram"],
        "category": "fraud",
        "severity": "MEDIUM",
        "cves": [],
        "iocs": ["refunds.example"],
        "brands": ["Refund"],
        "created_at": (NOW - timedelta(days=45)).isoformat().replace("+00:00", "Z"),
        "embedding": cluster_embedding(other).tolist(),
    }


def _write_state(path, history: list[dict]) -> None:
    path.write_text(json.dumps({"policy_type": "multiaction_dqn", "version": 2, "report_history": history}), encoding="utf-8")


def test_similarity_weights_match_the_specification():
    assert SIMILARITY_WEIGHTS == {
        "cosine": 2.0,
        "ioc_jaccard": 3.0,
        "cve_jaccard": 3.0,
        "brand_jaccard": 1.5,
        "category_match": 1.0,
        "platform_match": 0.5,
        "temporal_decay": 1.0,
        "bias": -3.0,
    }


def test_similarity_features_report_every_component():
    features = similarity_features(_cluster(), _matching_history_entry(_cluster()), now=NOW)

    assert set(features) == {
        "cosine",
        "ioc_jaccard",
        "cve_jaccard",
        "brand_jaccard",
        "category_match",
        "platform_match",
        "temporal_decay",
    }
    assert features["cosine"] > 0.99
    assert features["ioc_jaccard"] == 1.0
    assert features["cve_jaccard"] == 1.0
    assert features["category_match"] == 1.0
    assert features["platform_match"] == 1.0
    assert 0.85 < features["temporal_decay"] < 0.88


def test_similarity_to_report_passes_the_gate_for_a_matching_past_report():
    assert similarity_to_report(_cluster(), _matching_history_entry(_cluster()), now=NOW) >= 0.60


def test_similarity_to_report_stays_below_the_gate_for_an_unrelated_report():
    assert similarity_to_report(_cluster(), _unrelated_history_entry(), now=NOW) < 0.60


def test_brand_tokens_keep_long_capitalised_words():
    assert brand_tokens("Citrix NetScaler CVE-2026-3055 exploitation") == ["Citrix", "NetScaler"]
    assert brand_tokens("") == []


def test_cold_start_without_a_policy_state_file(tmp_path):
    strategy = select_investigation_strategy(_cluster(), Settings(), tmp_path / "missing.json")

    assert strategy == InvestigationStrategy(
        mode="multi_perspective",
        similarity_score=0.0,
        action_priorities=[],
        focus_by_perspective={},
        q_values={},
    )


def test_cold_start_when_report_history_is_empty(tmp_path):
    path = tmp_path / "policy_state.json"
    _write_state(path, [])

    strategy = select_investigation_strategy(_cluster(), Settings(), path)

    assert strategy.mode == "multi_perspective"
    assert strategy.similarity_score == 0.0
    assert strategy.matched_report_id == ""


def test_malformed_policy_state_falls_back_to_multi_perspective(tmp_path, caplog):
    path = tmp_path / "policy_state.json"
    path.write_text("{not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="tiblender_runtime.rl.policy"):
        strategy = select_investigation_strategy(_cluster(), Settings(), path)

    assert strategy.mode == "multi_perspective"
    assert strategy.q_values == {}
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert "unreadable" in caplog.records[0].getMessage()


def test_disabled_backend_always_returns_multi_perspective(tmp_path):
    path = tmp_path / "policy_state.json"
    _write_state(path, [_matching_history_entry(_cluster())])

    strategy = select_investigation_strategy(_cluster(), Settings(rl_policy_backend="disabled"), path)

    assert strategy.mode == "multi_perspective"
    assert strategy.action_priorities == []


def test_rl_guided_mode_for_a_matching_past_report(tmp_path, monkeypatch):
    monkeypatch.setattr("tiblender_runtime.rl.policy._now", lambda: NOW)
    path = tmp_path / "policy_state.json"
    _write_state(path, [_unrelated_history_entry(), _matching_history_entry(_cluster())])

    strategy = select_investigation_strategy(_cluster(), Settings(), path)

    assert strategy.mode == "rl_guided"
    assert strategy.similarity_score >= 0.60
    assert strategy.matched_report_id == "TIBLENDER-PREV"
    assert strategy.action_priorities
    assert all(action in ACTION_SPACE for action in strategy.action_priorities)
    assert all(action in FOCUS_TYPES for action in strategy.action_priorities)
    assert len(strategy.action_priorities) <= Settings().rl_action_top_k
    assert strategy.focus_by_perspective
    for focuses in strategy.focus_by_perspective.values():
        for focus in focuses:
            assert focus in FOCUS_TYPES
            assert focus in strategy.q_values
    assert strategy.q_values


def test_multi_perspective_when_similarity_is_below_the_gate(tmp_path, monkeypatch):
    monkeypatch.setattr("tiblender_runtime.rl.policy._now", lambda: NOW)
    path = tmp_path / "policy_state.json"
    _write_state(path, [_unrelated_history_entry()])

    strategy = select_investigation_strategy(_cluster(), Settings(), path)

    assert strategy.mode == "multi_perspective"
    assert strategy.action_priorities == []
    assert 0.0 < strategy.similarity_score < 0.60


def test_focus_by_perspective_survives_a_tight_action_top_k(tmp_path, monkeypatch):
    monkeypatch.setattr("tiblender_runtime.rl.policy._now", lambda: NOW)
    path = tmp_path / "policy_state.json"
    _write_state(path, [_matching_history_entry(_cluster())])

    strategy = select_investigation_strategy(_cluster(), Settings(rl_action_top_k=1), path)

    assert strategy.mode == "rl_guided"
    assert len(strategy.action_priorities) == 1
    selected = [focus for focuses in strategy.focus_by_perspective.values() for focus in focuses]
    assert len(selected) >= 1
    # Every selected focus keeps a Q value even when the flat priority list is cut.
    assert all(focus in strategy.q_values for focus in selected)


def test_shim_reexports_the_public_policy_api(tmp_path):
    assert len(ACTION_SPACE) == 19
    assert shim_select is select_investigation_strategy
