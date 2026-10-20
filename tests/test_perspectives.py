"""Perspective agents: focus selection, query generation, pivots, parallel runs."""

import threading
import time

from tiblender_runtime.contract import REQUIRED_FOCUS_TYPES, REQUIRED_PERSPECTIVES
from tiblender_runtime.llm import LLMGenerationError
from tiblender_runtime.models import (
    Cluster,
    IocSet,
    NormalizedPost,
    Post,
    Severity,
    ThreatCategory,
    ToolQuery,
    ToolResult,
)
from tiblender_runtime.perspectives import (
    FOCUS_TOOLS,
    PERSPECTIVE_TOOLS,
    build_perspective_queries,
    investigate_cluster,
    pivot_queries,
    select_focus_types,
)
from tiblender_runtime.rl.policy import InvestigationStrategy
from tiblender_runtime.settings import Settings


def _post(post_id: str, text: str, platform: str = "x") -> NormalizedPost:
    return NormalizedPost(
        post=Post(id=post_id, platform=platform, content=text),
        normalized_content=text,
        iocs=IocSet(),
        threat_category=ThreatCategory.MALWARE,
    )


def _cluster(
    *,
    iocs: IocSet | None = None,
    category: ThreatCategory = ThreatCategory.MALWARE,
    posts: list[NormalizedPost] | None = None,
) -> Cluster:
    resolved_posts = posts or [_post("p1", "Winos payload delivered from bad.example CVE-2026-11111")]
    return Cluster(
        id="cluster-perspectives",
        topic_name="Winos payload campaign",
        description="Winos payload delivered from bad.example",
        posts=resolved_posts,
        iocs=iocs if iocs is not None else IocSet(domains=["bad.example"], cves=["CVE-2026-11111"]),
        severity=Severity.HIGH,
        threat_category=category,
    )


class _FakeRunner:
    """Records the thread each perspective ran on and replays canned results."""

    def __init__(self, *, failing: set[str] | None = None, delay: float = 0.0) -> None:
        self.settings = Settings(enable_real_tools=False, allow_heuristic_reports=True)
        self.failing = failing or set()
        self.delay = delay
        self.thread_ids: list[int] = []
        self.executed: list[ToolQuery] = []
        self._lock = threading.Lock()

    def execute_many(self, queries: list[ToolQuery]) -> list[ToolResult]:
        with self._lock:
            self.thread_ids.append(threading.get_ident())
        if self.delay:
            time.sleep(self.delay)
        return [self.execute(query) for query in queries]

    def execute(self, query: ToolQuery) -> ToolResult:
        if query.perspective in self.failing:
            raise RuntimeError(f"{query.perspective} tool exploded")
        with self._lock:
            self.executed.append(query)
        return ToolResult(
            tool_type=query.tool_type,
            query_params=dict(query.parameters),
            summary=f"{query.tool_type} ok",
            discovered_iocs=IocSet(),
        )


def test_focus_tools_cover_every_contract_focus_type():
    contract_focuses = {focus for focuses in REQUIRED_FOCUS_TYPES.values() for focus in focuses}

    assert set(FOCUS_TOOLS) == contract_focuses
    assert all(tools for tools in FOCUS_TOOLS.values())
    assert set(PERSPECTIVE_TOOLS) == set(REQUIRED_PERSPECTIVES)


def test_heuristic_focus_selection_returns_two_to_four_valid_focuses():
    settings = Settings(allow_heuristic_reports=True)
    cluster = _cluster(iocs=IocSet(domains=["bad.example", "worse.example"], ips=["203.0.113.10"]))

    focuses = select_focus_types(cluster, "Infrastructure", hints=[], settings=settings)

    assert 2 <= len(focuses) <= 4
    assert set(focuses) <= set(REQUIRED_FOCUS_TYPES["Infrastructure"])
    assert len(set(focuses)) == len(focuses)
    assert "domain_registration_and_whois" in focuses
    assert "related_domain_and_infrastructure_pivoting" in focuses
    assert focuses == select_focus_types(cluster, "Infrastructure", hints=[], settings=settings)


def test_heuristic_focus_selection_puts_valid_hints_first():
    settings = Settings(allow_heuristic_reports=True)
    cluster = _cluster()

    focuses = select_focus_types(
        cluster,
        "Infrastructure",
        hints=["hosting_and_osint", "ttp_mapping"],
        settings=settings,
    )

    assert focuses[0] == "hosting_and_osint"
    assert "ttp_mapping" not in focuses
    assert set(focuses) <= set(REQUIRED_FOCUS_TYPES["Infrastructure"])


def test_heuristic_focus_selection_covers_every_perspective_without_signals():
    settings = Settings(allow_heuristic_reports=True)
    cluster = _cluster(iocs=IocSet(), category=ThreatCategory.OTHER)

    for perspective in REQUIRED_PERSPECTIVES:
        focuses = select_focus_types(cluster, perspective, hints=[], settings=settings)
        assert 2 <= len(focuses) <= 4
        assert set(focuses) <= set(REQUIRED_FOCUS_TYPES[perspective])


def test_focus_selection_pads_only_to_the_two_focus_minimum():
    settings = Settings(allow_heuristic_reports=True)
    unsupported = _cluster(iocs=IocSet(), category=ThreatCategory.OTHER)
    supported = _cluster(iocs=IocSet(domains=["bad.example", "worse.example"], ips=["203.0.113.10"]))

    assert select_focus_types(unsupported, "Infrastructure", hints=[], settings=settings) == [
        "domain_registration_and_whois",
        "dns_and_passive_dns_analysis",
    ]
    assert len(select_focus_types(supported, "Infrastructure", hints=[], settings=settings)) == 4


def test_llm_focus_selection_keeps_exactly_the_two_focuses_it_returned(monkeypatch):
    monkeypatch.setattr(
        "tiblender_runtime.perspectives.select_focus_types_with_llm",
        lambda settings, cluster, perspective: ["ssl_tls_certificate_analysis", "hosting_and_osint"],
    )
    settings = Settings(
        llm_provider="openai",
        llm_model="model",
        openai_api_key="key",
        allow_heuristic_reports=False,
    )

    focuses = select_focus_types(_cluster(), "Infrastructure", hints=[], settings=settings)

    assert focuses == ["ssl_tls_certificate_analysis", "hosting_and_osint"]


def test_llm_focus_selection_pads_a_single_valid_focus_up_to_two(monkeypatch):
    from tiblender_runtime.perspectives import _focus_types_with_notes

    monkeypatch.setattr(
        "tiblender_runtime.perspectives.select_focus_types_with_llm",
        lambda settings, cluster, perspective: ["phishing_campaign_tracking"],
    )
    settings = Settings(
        llm_provider="openai",
        llm_model="model",
        openai_api_key="key",
        allow_heuristic_reports=False,
    )

    focuses, notes = _focus_types_with_notes(_cluster(), "Social", hints=[], settings=settings)

    assert len(focuses) == 2
    assert focuses[0] == "phishing_campaign_tracking"
    assert "Social llm focus selection padded (1 valid → 2)" in notes


def test_query_builder_uses_only_verified_cluster_iocs():
    settings = Settings(allow_heuristic_reports=True)
    cluster = _cluster()

    queries = build_perspective_queries(
        cluster,
        "Infrastructure",
        ["domain_registration_and_whois", "dns_and_passive_dns_analysis", "ssl_tls_certificate_analysis"],
        planned=[],
        settings=settings,
    )

    assert queries
    assert {query.tool_type for query in queries} == {"WHOIS_LOOKUP", "PASSIVE_DNS_LOOKUP", "TLS_CT_LOOKUP"}
    assert all(query.perspective == "Infrastructure" for query in queries)
    assert all(query.focus_type in REQUIRED_FOCUS_TYPES["Infrastructure"] for query in queries)
    assert {query.parameters["domain"] for query in queries} == {"bad.example"}


def test_query_builder_emits_nothing_without_a_usable_indicator():
    settings = Settings(allow_heuristic_reports=True)
    cluster = Cluster(
        id="cluster-empty",
        topic_name="Unnamed activity",
        description="No indicators were recovered.",
        posts=[_post("p1", "No indicators were recovered from this post.")],
        iocs=IocSet(),
        severity=Severity.MEDIUM,
        threat_category=ThreatCategory.OTHER,
    )

    queries = build_perspective_queries(
        cluster,
        "Infrastructure",
        ["domain_registration_and_whois", "dns_and_passive_dns_analysis"],
        planned=[],
        settings=settings,
    )

    assert queries == []


def test_query_builder_rejects_generic_single_word_planned_queries():
    settings = Settings(allow_heuristic_reports=True)
    cluster = _cluster()
    planned = [
        ToolQuery("Social", "X_SEARCH", {"query": "malware"}, priority=9),
        ToolQuery("Social", "X_SEARCH", {"query": "bad.example phishing kit"}, priority=8),
        ToolQuery("Social", "REDDIT_SEARCH", {"query": "evil.invalid"}, priority=7),
    ]

    queries = build_perspective_queries(
        cluster,
        "Social",
        ["phishing_campaign_tracking"],
        planned=planned,
        settings=settings,
    )

    emitted = [query.parameters.get("query", "") for query in queries]
    assert "malware" not in emitted
    assert "evil.invalid" not in emitted
    assert emitted[0] == "bad.example phishing kit"
    assert queries[0].focus_type == "phishing_campaign_tracking"


def test_query_builder_tags_planner_queries_with_a_focus_of_their_own_perspective():
    settings = Settings(allow_heuristic_reports=True)
    cluster = _cluster()
    planned = [
        ToolQuery("Actor", "SEARCH_ENGINE", {"query": "bad.example winos"}, priority=9),
        ToolQuery("Actor", "REDDIT_SEARCH", {"query": "bad.example winos"}, priority=8),
    ]
    focus_types = ["campaign_tracking", "apt_group_and_threat_actor_investigation"]

    queries = build_perspective_queries(cluster, "Actor", focus_types, planned=planned, settings=settings)

    tagged = {query.tool_type: query.focus_type for query in queries}
    assert tagged["SEARCH_ENGINE"] in REQUIRED_FOCUS_TYPES["Actor"]
    assert tagged["SEARCH_ENGINE"] == "campaign_tracking"
    assert tagged["REDDIT_SEARCH"] == "apt_group_and_threat_actor_investigation"
    assert all(query.focus_type in focus_types for query in queries)


def test_query_builder_uses_up_to_two_indicators_per_focus():
    settings = Settings(allow_heuristic_reports=True)
    cluster = _cluster(iocs=IocSet(domains=["bad.example", "worse.example"]))

    queries = build_perspective_queries(
        cluster,
        "Infrastructure",
        ["domain_registration_and_whois"],
        planned=[],
        settings=settings,
    )

    assert [query.tool_type for query in queries] == ["WHOIS_LOOKUP", "WHOIS_LOOKUP"]
    assert {query.parameters["domain"] for query in queries} == {"bad.example", "worse.example"}


def test_query_builder_falls_back_when_every_llm_query_is_ungrounded(monkeypatch):
    def ungrounded(settings, cluster, perspective, focus_types):
        del settings, cluster, perspective, focus_types
        return [
            ToolQuery("Infrastructure", "WHOIS_LOOKUP", {"domain": "invented.example"}, priority=9),
            ToolQuery("Infrastructure", "TLS_CT_LOOKUP", {"domain": "also-invented.example"}, priority=8),
        ]

    monkeypatch.setattr("tiblender_runtime.perspectives.generate_perspective_queries_with_llm", ungrounded)
    monkeypatch.setattr(
        "tiblender_runtime.perspectives.select_focus_types_with_llm",
        lambda settings, cluster, perspective: ["domain_registration_and_whois", "dns_and_passive_dns_analysis"],
    )
    runner = _FakeRunner()
    settings = Settings(
        llm_provider="openai",
        llm_model="model",
        openai_api_key="key",
        allow_heuristic_reports=False,
        enable_real_tools=False,
        enable_pivot_iocs=False,
    )

    findings = investigate_cluster(
        _cluster(),
        settings=settings,
        runner=runner,
        strategy=InvestigationStrategy(mode="multi_perspective", similarity_score=0.0),
        planned_queries=[],
    )

    infrastructure = next(finding for finding in findings if finding.perspective == "Infrastructure")
    assert infrastructure.queries
    assert {query.parameters.get("domain") for query in infrastructure.queries} == {"bad.example"}
    assert any("llm queries dropped: 2 ungrounded" in insight for insight in infrastructure.insights)


def test_query_builder_caps_queries_per_perspective():
    settings = Settings(allow_heuristic_reports=True, max_queries_per_perspective=2)
    cluster = _cluster()

    queries = build_perspective_queries(
        cluster,
        "Infrastructure",
        list(REQUIRED_FOCUS_TYPES["Infrastructure"])[:4],
        planned=[],
        settings=settings,
    )

    assert len(queries) == 2


def test_pivot_queries_target_only_new_infrastructure_at_depth_one():
    cluster = _cluster()
    results = [
        ToolResult(
            tool_type="PASSIVE_DNS_LOOKUP",
            query_params={"domain": "bad.example"},
            summary="pdns",
            discovered_iocs=IocSet(domains=["pivot.example", "bad.example", "github.com"]),
        )
    ]

    queries = pivot_queries("Infrastructure", results, cluster, depth=1)

    assert len(queries) == 3
    assert {query.tool_type for query in queries} == {"WHOIS_LOOKUP", "PASSIVE_DNS_LOOKUP", "TLS_CT_LOOKUP"}
    assert {query.parameters["domain"] for query in queries} == {"pivot.example"}
    assert all(query.parameters["pivot"] is True for query in queries)
    assert pivot_queries("Infrastructure", results, cluster, depth=0) == []
    assert pivot_queries("Social", results, cluster, depth=1) == []


def test_pivot_queries_ignore_unsuccessful_results():
    cluster = _cluster()
    results = [
        ToolResult(
            tool_type="PASSIVE_DNS_LOOKUP",
            query_params={"domain": "bad.example"},
            summary="unavailable",
            status="unavailable",
            discovered_iocs=IocSet(domains=["pivot.example"]),
        )
    ]

    assert pivot_queries("Infrastructure", results, cluster, depth=1) == []


def test_investigate_cluster_returns_four_findings_in_fixed_order():
    runner = _FakeRunner()
    settings = Settings(allow_heuristic_reports=True, enable_real_tools=False, enable_pivot_iocs=False)
    strategy = InvestigationStrategy(mode="rl_guided", similarity_score=0.9)

    findings = investigate_cluster(
        _cluster(),
        settings=settings,
        runner=runner,
        strategy=strategy,
        planned_queries=[],
    )

    assert [finding.perspective for finding in findings] == list(REQUIRED_PERSPECTIVES)
    assert all(finding.mode == "rl_guided" for finding in findings)
    assert all(2 <= len(finding.focus_types) <= 4 for finding in findings)
    assert all(len(finding.queries) == len(finding.tool_results) for finding in findings)
    assert all(query.focus_type for finding in findings for query in finding.queries)


def test_investigate_cluster_applies_rl_focus_hints():
    runner = _FakeRunner()
    settings = Settings(allow_heuristic_reports=True, enable_real_tools=False, enable_pivot_iocs=False)
    strategy = InvestigationStrategy(
        mode="rl_guided",
        similarity_score=0.9,
        focus_by_perspective={"Actor": ["campaign_tracking"]},
    )

    findings = investigate_cluster(
        _cluster(),
        settings=settings,
        runner=runner,
        strategy=strategy,
        planned_queries=[],
    )

    actor = next(finding for finding in findings if finding.perspective == "Actor")
    assert actor.focus_types[0] == "campaign_tracking"


def test_investigate_cluster_runs_perspectives_concurrently():
    runner = _FakeRunner(delay=0.05)
    settings = Settings(
        allow_heuristic_reports=True,
        enable_real_tools=False,
        enable_pivot_iocs=False,
        perspective_workers=4,
    )

    findings = investigate_cluster(
        _cluster(),
        settings=settings,
        runner=runner,
        strategy=InvestigationStrategy(mode="multi_perspective", similarity_score=0.0),
        planned_queries=[],
    )

    assert len(findings) == 4
    assert len(set(runner.thread_ids)) >= 2


def test_investigate_cluster_records_a_failed_perspective_without_aborting():
    runner = _FakeRunner(failing={"Social"})
    settings = Settings(allow_heuristic_reports=True, enable_real_tools=False, enable_pivot_iocs=False)

    findings = investigate_cluster(
        _cluster(),
        settings=settings,
        runner=runner,
        strategy=InvestigationStrategy(mode="multi_perspective", similarity_score=0.0),
        planned_queries=[],
    )

    social = next(finding for finding in findings if finding.perspective == "Social")
    assert social.tool_results == []
    assert any(insight.startswith("perspective failed: ") for insight in social.insights)
    assert [finding.perspective for finding in findings] == list(REQUIRED_PERSPECTIVES)
    assert any(finding.tool_results for finding in findings if finding.perspective != "Social")


def test_investigate_cluster_pivots_one_level_from_discovered_infrastructure():
    class _PivotRunner(_FakeRunner):
        def execute(self, query: ToolQuery) -> ToolResult:
            result = super().execute(query)
            if query.tool_type == "PASSIVE_DNS_LOOKUP" and not query.parameters.get("pivot"):
                return ToolResult(
                    tool_type=query.tool_type,
                    query_params=dict(query.parameters),
                    summary="pdns",
                    discovered_iocs=IocSet(domains=["pivot.example"]),
                )
            return result

    runner = _PivotRunner()
    settings = Settings(
        allow_heuristic_reports=True,
        enable_real_tools=False,
        enable_pivot_iocs=True,
        pivot_depth=1,
    )

    findings = investigate_cluster(
        _cluster(),
        settings=settings,
        runner=runner,
        strategy=InvestigationStrategy(mode="multi_perspective", similarity_score=0.0),
        planned_queries=[],
    )

    pivoted = [query for finding in findings for query in finding.queries if query.parameters.get("pivot")]
    assert pivoted
    assert {query.parameters["domain"] for query in pivoted} == {"pivot.example"}
    infrastructure = next(finding for finding in findings if finding.perspective == "Infrastructure")
    assert len(infrastructure.queries) == len(infrastructure.tool_results)


def test_llm_focus_selection_failure_falls_back_to_the_heuristic(monkeypatch):
    def explode(*args, **kwargs):
        raise LLMGenerationError("focus selection unavailable")

    monkeypatch.setattr("tiblender_runtime.perspectives.select_focus_types_with_llm", explode)
    monkeypatch.setattr("tiblender_runtime.perspectives.generate_perspective_queries_with_llm", explode)
    runner = _FakeRunner()
    settings = Settings(
        llm_provider="openai",
        llm_model="model",
        openai_api_key="key",
        allow_heuristic_reports=False,
        enable_real_tools=False,
        enable_pivot_iocs=False,
    )

    findings = investigate_cluster(
        _cluster(),
        settings=settings,
        runner=runner,
        strategy=InvestigationStrategy(mode="multi_perspective", similarity_score=0.0),
        planned_queries=[],
    )

    infrastructure = next(finding for finding in findings if finding.perspective == "Infrastructure")
    assert set(infrastructure.focus_types) <= set(REQUIRED_FOCUS_TYPES["Infrastructure"])
    assert any("heuristic" in insight for insight in infrastructure.insights)
