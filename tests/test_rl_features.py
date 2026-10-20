"""State-feature tests for the MultiAction DQN policy."""

import numpy as np

from tiblender_runtime.models import Cluster, IocSet, NormalizedPost, Post, Severity, ThreatCategory
from tiblender_runtime.rl.features import (
    CLUSTER_DIM,
    CONTEXT_DIM,
    HISTORY_DIM,
    STATE_DIM,
    cluster_embedding,
    cluster_tokens,
    context_embedding,
    history_embedding,
    state_vector,
)


def _cluster(topic: str = "PayPay phishing activity") -> Cluster:
    post = NormalizedPost(
        post=Post(id="p1", platform="x", content="PayPay phishing kit bad.example"),
        normalized_content="PayPay phishing kit steals credentials bad.example CVE-2026-3055",
        iocs=IocSet(domains=["bad.example"], cves=["CVE-2026-3055"]),
        threat_category=ThreatCategory.PHISHING,
    )
    return Cluster(
        id="cluster-paypay",
        topic_name=topic,
        description="PayPay phishing kit targeting mobile wallet users",
        posts=[post],
        iocs=post.iocs,
        severity=Severity.HIGH,
        threat_category=ThreatCategory.PHISHING,
    )


def test_dimensions_match_the_320_dimensional_contract():
    assert (CLUSTER_DIM, CONTEXT_DIM, HISTORY_DIM, STATE_DIM) == (128, 128, 64, 320)


def test_cluster_embedding_is_deterministic_and_finite():
    first = cluster_embedding(_cluster())
    second = cluster_embedding(_cluster())

    assert first.shape == (128,)
    assert first.dtype == np.float64
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert float(np.linalg.norm(first)) > 0.0


def test_cluster_embedding_separates_unrelated_clusters():
    same = float(np.dot(cluster_embedding(_cluster()), cluster_embedding(_cluster())))
    other = _cluster(topic="Ransomware negotiation leak site")
    other.posts[0].normalized_content = "LockBit ransomware negotiation leak portal onion address"
    other.iocs = IocSet(domains=["leak.onion"])
    other.threat_category = ThreatCategory.RANSOMWARE
    cross = float(np.dot(cluster_embedding(_cluster()), cluster_embedding(other)))

    assert same > cross


def test_cluster_tokens_include_topic_and_ioc_terms():
    tokens = cluster_tokens(_cluster())

    assert "paypay" in tokens
    assert "bad.example" in tokens


def test_context_embedding_is_zero_before_investigation():
    vector = context_embedding({}, {})

    assert vector.shape == (128,)
    assert not vector.any()


def test_context_embedding_encodes_counts_and_coverage():
    stats = {
        "iocs": 6,
        "cves": 2,
        "ttps": 1,
        "results": 4,
        "successes": 3,
        "confidence": {"LOW": 1, "MEDIUM": 2, "HIGH": 1},
    }
    per_perspective = {"Infrastructure": {"results": 2, "successes": 2}, "Actor": {"results": 2, "successes": 1}}

    vector = context_embedding(stats, per_perspective)

    assert vector.shape == (128,)
    assert vector[0] == np.log1p(6.0)
    assert vector[1] == np.log1p(2.0)
    assert vector[2] == np.log1p(1.0)
    assert vector[6] == 0.75
    assert vector[7] == 1.0
    assert vector[8] == 0.0
    assert vector[10] == 1.0


def test_history_embedding_encodes_actions_and_step():
    vector = history_embedding([0, 0, 5], 3)

    assert vector.shape == (64,)
    assert vector[0] == 1.0
    assert vector[5] == 1.0
    assert vector[1] == 0.0
    assert vector[19 + 3] == 1.0
    assert vector[19 + 0] == 0.0


def test_history_embedding_is_zero_at_cold_start_except_step_slot():
    vector = history_embedding([], 0)

    assert vector[19] == 1.0
    assert not vector[:19].any()


def test_state_vector_concatenates_to_320():
    vector = state_vector(_cluster(), {}, {}, [], 0)

    assert vector.shape == (320,)
    assert np.array_equal(vector[:128], cluster_embedding(_cluster()))
    assert not vector[128:256].any()
    assert np.array_equal(vector[256:], history_embedding([], 0))
