from tiblender_runtime.contract import REQUIRED_PERSPECTIVES
from tiblender_runtime.investigation import investigate_cluster, investigate_followups
from tiblender_runtime.models import Cluster, IocSet, NormalizedPost, Post, Severity, ThreatCategory, ToolQuery
from tiblender_runtime.rl.policy import InvestigationStrategy
from tiblender_runtime.settings import Settings
from tiblender_runtime.tools import ToolRunner


def _runner() -> ToolRunner:
    return ToolRunner(Settings(enable_real_tools=False, allow_heuristic_reports=True))


def _strategy(mode: str = "multi_perspective", **kwargs) -> InvestigationStrategy:
    return InvestigationStrategy(mode=mode, similarity_score=0.0, **kwargs)


def _settings(**kwargs) -> Settings:
    return Settings(enable_real_tools=False, allow_heuristic_reports=True, **kwargs)


def _cluster() -> Cluster:
    post = NormalizedPost(
        post=Post(id="p1", platform="x", content="Payload uses bad.example"),
        normalized_content="Payload uses bad.example",
        iocs=IocSet(domains=["bad.example"]),
        threat_category=ThreatCategory.MALWARE,
    )
    return Cluster(
        id="cluster-test",
        topic_name="bad.example payload campaign",
        description="bad.example payload campaign",
        posts=[post],
        iocs=post.iocs,
        severity=Severity.MEDIUM,
        threat_category=ThreatCategory.MALWARE,
    )


def test_investigate_cluster_runs_every_perspective_with_planned_queries():
    cluster = _cluster()
    planned = [ToolQuery("Infrastructure", "WHOIS_LOOKUP", {"domain": "bad.example"}, priority=9)]

    findings = investigate_cluster(
        cluster,
        settings=_settings(enable_pivot_iocs=False),
        runner=_runner(),
        strategy=_strategy("llm_planned_multi_perspective"),
        planned_queries=planned,
    )

    assert [finding.perspective for finding in findings] == list(REQUIRED_PERSPECTIVES)
    assert all(finding.mode == "llm_planned_multi_perspective" for finding in findings)
    infrastructure = findings[0]
    assert infrastructure.queries[0].tool_type == "WHOIS_LOOKUP"
    assert infrastructure.queries[0].parameters == {"domain": "bad.example"}
    assert infrastructure.queries[0].focus_type == "domain_registration_and_whois"


def test_investigate_cluster_never_queries_an_unverified_indicator():
    cluster = _cluster()
    planned = [ToolQuery("Infrastructure", "WHOIS_LOOKUP", {"domain": "not-in-cluster.example"}, priority=9)]

    findings = investigate_cluster(
        cluster,
        settings=_settings(enable_pivot_iocs=False),
        runner=_runner(),
        strategy=_strategy(),
        planned_queries=planned,
    )

    parameters = [query.parameters for finding in findings for query in finding.queries]
    assert {"domain": "not-in-cluster.example"} not in parameters
    assert all("bad.example" in str(item) or "query" in item for item in parameters)


def test_investigate_cluster_honours_the_query_budget_per_perspective():
    findings = investigate_cluster(
        _cluster(),
        settings=_settings(max_queries_per_perspective=1, enable_pivot_iocs=False),
        runner=_runner(),
        strategy=_strategy(),
        planned_queries=[],
    )

    assert all(len(finding.queries) <= 1 for finding in findings)


def test_investigate_followups_maps_requests_back_to_grounded_tool_calls():
    findings = investigate_followups(
        _cluster(),
        ["Infrastructure WHOIS_LOOKUP: validate bad.example registration"],
        tool_runner=_runner(),
        max_queries_per_perspective=2,
    )

    assert [finding.perspective for finding in findings] == ["Infrastructure"]
    assert findings[0].mode == "multi_perspective_follow_up"
    assert findings[0].queries[0].tool_type == "WHOIS_LOOKUP"
    assert findings[0].queries[0].parameters == {"domain": "bad.example"}


def test_investigate_followups_falls_back_to_a_full_multi_perspective_sweep():
    findings = investigate_followups(
        _cluster(),
        [],
        tool_runner=_runner(),
        max_queries_per_perspective=2,
    )

    assert [finding.perspective for finding in findings] == list(REQUIRED_PERSPECTIVES)
    assert all(finding.mode == "multi_perspective_follow_up" for finding in findings)
