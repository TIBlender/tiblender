"""Shared builders for the two LLM-fallback test modules.

Not a test module: `tests/test_llm_fallbacks.py` covers chunking, truncation
handling and per-report attribution; `tests/test_llm_fallback_reasons.py`
covers the reason categories, the logging contract and concurrency. Both need
the same minimal settings/post/cluster/finding/decision objects, so they live
here rather than being duplicated or imported across test modules.
"""

from __future__ import annotations

from tiblender_runtime.models import (
    Cluster,
    EvaluationDecision,
    EvaluationRound,
    EvaluationVerdict,
    IocSet,
    NormalizedPost,
    PerspectiveFinding,
    Post,
    Severity,
    ThreatCategory,
)
from tiblender_runtime.settings import Settings


def make_settings(**overrides) -> Settings:
    base = {
        "llm_provider": "openai",
        "llm_model": "model",
        "openai_api_key": "key",
        "allow_heuristic_reports": False,
    }
    base.update(overrides)
    return Settings(**base)


def make_posts(count: int) -> list[Post]:
    return [
        Post(
            id=f"p{index}",
            platform="x",
            content=f"Phishing kit reuse reported for bad{index}.example",
            timestamp="2026-04-01T00:00:00Z",
            metadata={"language": "en"},
        )
        for index in range(1, count + 1)
    ]


def make_normalized_post(post_id: str = "p1") -> NormalizedPost:
    post = Post(
        id=post_id,
        platform="x",
        content="CVE-2026-12345 exploitation at bad.example",
        timestamp="2026-04-01T00:00:00Z",
    )
    return NormalizedPost(
        post=post,
        normalized_content="CVE-2026-12345 exploitation campaign bad.example",
        iocs=IocSet(domains=["bad.example"], cves=["CVE-2026-12345"]),
        threat_category=ThreatCategory.VULNERABILITY,
    )


def make_cluster(cluster_id: str = "cluster-test") -> Cluster:
    post = make_normalized_post()
    return Cluster(
        id=cluster_id,
        topic_name="CVE-2026-12345 exploitation campaign",
        description="test cluster",
        posts=[post],
        iocs=post.iocs,
        severity=Severity.HIGH,
        threat_category=ThreatCategory.VULNERABILITY,
    )


def make_finding() -> PerspectiveFinding:
    return PerspectiveFinding(
        perspective="Infrastructure",
        focus_types=["domain_registration_and_whois"],
        queries=[],
        tool_results=[],
        insights=["WHOIS and CT should be checked."],
        confidence=0.7,
    )


def make_decision() -> EvaluationDecision:
    return EvaluationDecision(
        verdict=EvaluationVerdict.PROCEED,
        rounds=[
            EvaluationRound(
                round_index=1,
                judge="llm-judge-a",
                verdict=EvaluationVerdict.PROCEED,
                quality_score=0.9,
                reason="Sufficient evidence.",
            )
        ],
        final_quality_score=0.9,
        reason="Sufficient evidence.",
    )


def make_cluster_with_posts(cluster_id: str, post_ids: list[str]) -> Cluster:
    posts = [
        NormalizedPost(
            post=Post(
                id=post_id,
                platform="x",
                content=f"Phishing kit reuse reported for {post_id}.example",
                timestamp="2026-04-01T00:00:00Z",
            ),
            normalized_content=f"Phishing kit reuse campaign {post_id}.example",
            iocs=IocSet(domains=[f"{post_id}.example"]),
            threat_category=ThreatCategory.PHISHING,
        )
        for post_id in post_ids
    ]
    return Cluster(
        id=cluster_id,
        topic_name=f"Campaign {cluster_id}",
        description="test cluster",
        posts=posts,
        iocs=IocSet.from_many([post.iocs for post in posts]),
        severity=Severity.MEDIUM,
        threat_category=ThreatCategory.PHISHING,
    )


