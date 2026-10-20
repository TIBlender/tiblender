"""Context aggregation: confidence rules, benign filtering, stats, trajectory."""

from tiblender_runtime.context import aggregate_context, context_to_judge_payload
from tiblender_runtime.models import (
    Cluster,
    IocSet,
    NormalizedPost,
    PerspectiveFinding,
    Post,
    Severity,
    ThreatCategory,
    ToolQuery,
    ToolResult,
)


def _post(post_id: str, text: str, platform: str = "x") -> NormalizedPost:
    return NormalizedPost(
        post=Post(id=post_id, platform=platform, content=text),
        normalized_content=text,
        iocs=IocSet(),
        threat_category=ThreatCategory.MALWARE,
    )


def _cluster(posts: list[NormalizedPost], iocs: IocSet) -> Cluster:
    return Cluster(
        id="cluster-ctx",
        topic_name="example payload campaign",
        description="example payload campaign",
        posts=posts,
        iocs=iocs,
        severity=Severity.HIGH,
        threat_category=ThreatCategory.MALWARE,
    )


def _tool_result(
    tool_type: str,
    params: dict,
    *,
    status: str = "success",
    summary: str = "",
    discovered: IocSet | None = None,
    error: str = "",
) -> ToolResult:
    return ToolResult(
        tool_type=tool_type,
        query_params=dict(params),
        summary=summary,
        discovered_iocs=discovered or IocSet(),
        status=status,
        error=error,
    )


def _finding(
    perspective: str,
    results: list[ToolResult],
    *,
    focus_types: list[str] | None = None,
) -> PerspectiveFinding:
    queries = [
        ToolQuery(perspective, result.tool_type, dict(result.query_params), focus_type="ttp_mapping")
        for result in results
    ]
    return PerspectiveFinding(
        perspective=perspective,
        focus_types=focus_types or ["ttp_mapping"],
        queries=queries,
        tool_results=results,
        insights=[],
        confidence=0.6,
    )


def _by_value(context) -> dict:
    return {evidence.value: evidence for evidence in context.iocs}


def test_direct_confidence_tiers_follow_independent_source_counts():
    posts = [
        _post("p1", "triple.example double.example single.example"),
        _post("p2", "triple.example double.example", platform="reddit"),
        _post("p3", "triple.example", platform="telegram"),
    ]
    cluster = _cluster(posts, IocSet(domains=["triple.example", "double.example", "single.example"]))

    context = aggregate_context(cluster, [])

    evidence = _by_value(context)
    assert evidence["triple.example"].confidence == "HIGH"
    assert evidence["double.example"].confidence == "MEDIUM"
    assert evidence["single.example"].confidence == "LOW"
    assert evidence["triple.example"].sources == ["post:p1", "post:p2", "post:p3"]
    assert all(item.extraction_type == "Direct" for item in context.iocs)
    assert context.stats["ioc_total"] == 3
    assert context.stats["ioc_high"] == 1
    assert context.stats["ioc_medium"] == 1
    assert context.stats["ioc_low"] == 1
    assert context.stats["direct_total"] == 3
    assert context.stats["pivot_total"] == 0
    assert context.stats["tool_success_rate"] == 0.0
    assert context.stats["perspectives_active"] == 0


def test_a_link_to_a_known_malicious_domain_raises_the_queried_indicator_one_tier():
    """Paper §III rule 2: the +1 needs a link to a *known malicious* domain.

    Here passive DNS shows host.example sharing infrastructure with
    other.example, which the cluster's own posts already named as an
    indicator -- that is the co-residency the paper's rule is about.
    """
    posts = [
        _post("p1", "host.example and other.example"),
        _post("p2", "host.example", platform="reddit"),
    ]
    cluster = _cluster(posts, IocSet(domains=["host.example", "other.example"]))
    result = _tool_result(
        "PASSIVE_DNS_LOOKUP",
        {"domain": "host.example"},
        summary="shared hosting record",
        discovered=IocSet(domains=["other.example"]),
    )

    context = aggregate_context(cluster, [_finding("Infrastructure", [result])])

    evidence = _by_value(context)
    # host.example is the investigated indicator: two posts (MEDIUM) plus the
    # known-malicious-link corroboration. other.example only earns the tool source.
    assert evidence["host.example"].confidence == "HIGH"
    assert "shared_hosting_link:+1" in evidence["host.example"].adjustments
    assert evidence["other.example"].confidence == "MEDIUM"
    assert evidence["other.example"].adjustments == []
    assert "tool:PASSIVE_DNS_LOOKUP:host.example" in evidence["other.example"].sources


def test_repeating_one_tool_answer_is_not_a_second_independent_source():
    """Rule 1 counts *independent* reports of an indicator.

    A source is one distinct (tool, queried indicator) answer, so asking the
    same tool the same question twice never lifts a tier; asking a second tool
    does.
    """
    cluster = _cluster([_post("p1", "bad.example")], IocSet(domains=["bad.example"]))
    repeated = _finding(
        "Infrastructure",
        [
            _tool_result("WHOIS_LOOKUP", {"domain": "bad.example"}, discovered=IocSet(domains=["bad.example"])),
            _tool_result("WHOIS_LOOKUP", {"domain": "bad.example"}, discovered=IocSet(domains=["bad.example"])),
        ],
    )
    distinct = _finding(
        "Technical",
        [
            _tool_result("WHOIS_LOOKUP", {"domain": "bad.example"}, discovered=IocSet(domains=["bad.example"])),
            _tool_result("TLS_CT_LOOKUP", {"domain": "bad.example"}, discovered=IocSet(domains=["bad.example"])),
        ],
    )

    once = _by_value(aggregate_context(cluster, [repeated]))["bad.example"]
    twice = _by_value(aggregate_context(cluster, [distinct]))["bad.example"]

    assert once.sources == ["post:p1", "tool:WHOIS_LOOKUP:bad.example"]
    assert once.confidence == "MEDIUM"
    assert len(twice.sources) == 3
    assert twice.confidence == "HIGH"


def test_a_single_resolution_answer_never_makes_a_pivot_reportable():
    """A pivot resting on one tool answer is an unconfirmed relationship.

    A small resolution set is not a second opinion about the names in it; the
    deployed scorer calls a single-evidence-type pivot LOW, and a LOW pivot is
    never published.
    """
    cluster = _cluster([_post("p1", "185.158.133.1")], IocSet(ips=["185.158.133.1"]))
    findings = [
        _finding(
            "Infrastructure",
            [
                _tool_result(
                    "PASSIVE_DNS_LOOKUP",
                    {"ip": "185.158.133.1"},
                    discovered=IocSet(domains=["only-answer.example"]),
                )
            ],
        )
    ]

    context = aggregate_context(cluster, findings)

    assert "only-answer.example" not in _by_value(context)


def test_a_contradiction_is_never_cancelled_by_another_rule():
    """Rule 3 subtracts a tier and nothing adds it back.

    A two-source pivot would be MEDIUM; a disconfirming provider answer makes
    it LOW, and a LOW pivot is dropped rather than reported.
    """
    cluster = _cluster([_post("p1", "seed.example")], IocSet(domains=["seed.example"]))
    findings = [
        _finding(
            "Infrastructure",
            [
                _tool_result(
                    "PASSIVE_DNS_LOOKUP",
                    {"domain": "seed.example"},
                    discovered=IocSet(domains=["contested.example"]),
                ),
                _tool_result(
                    "TLS_CT_LOOKUP",
                    {"domain": "seed.example"},
                    discovered=IocSet(domains=["contested.example"]),
                ),
                _tool_result(
                    "WHOIS_LOOKUP",
                    {"domain": "contested.example"},
                    status="unavailable",
                    error="RDAP: no data for this object",
                ),
            ],
        )
    ]

    context = aggregate_context(cluster, findings)

    assert "contested.example" not in _by_value(context)


def test_contradiction_lowers_confidence_one_tier():
    posts = [
        _post("p1", "gone.example nxd.example"),
        _post("p2", "gone.example nxd.example", platform="reddit"),
        _post("p3", "gone.example nxd.example", platform="telegram"),
    ]
    cluster = _cluster(posts, IocSet(domains=["gone.example", "nxd.example"]))
    findings = [
        _finding(
            "Infrastructure",
            [
                _tool_result(
                    "WHOIS_LOOKUP",
                    {"domain": "gone.example"},
                    status="error",
                    error="RDAP lookup failed: no such host gone.example",
                ),
                _tool_result(
                    "PASSIVE_DNS_LOOKUP",
                    {"domain": "nxd.example"},
                    status="unavailable",
                    error="PASSIVE_DNS_LOOKUP unavailable: NXDOMAIN for nxd.example",
                ),
            ],
        )
    ]

    context = aggregate_context(cluster, findings)

    evidence = _by_value(context)
    assert evidence["gone.example"].confidence == "MEDIUM"
    assert evidence["nxd.example"].confidence == "MEDIUM"
    assert "contradiction:-1" in evidence["gone.example"].adjustments
    assert "contradiction:-1" in evidence["nxd.example"].adjustments


def test_internal_tool_failure_is_not_treated_as_a_contradiction():
    posts = [
        _post("p1", "steady.example"),
        _post("p2", "steady.example", platform="reddit"),
        _post("p3", "steady.example", platform="telegram"),
    ]
    cluster = _cluster(posts, IocSet(domains=["steady.example"]))
    findings = [
        _finding(
            "Infrastructure",
            [
                _tool_result(
                    "WHOIS_LOOKUP",
                    {"domain": "steady.example"},
                    status="error",
                    error="RuntimeError: invalid JSON response",
                ),
                _tool_result(
                    "TLS_CT_LOOKUP",
                    {"domain": "steady.example"},
                    status="unavailable",
                    error="no recorded tool response for TLS_CT_LOOKUP with parameters {'domain': 'steady.example'}",
                ),
            ],
        )
    ]

    context = aggregate_context(cluster, findings)

    evidence = _by_value(context)
    assert evidence["steady.example"].confidence == "HIGH"
    assert evidence["steady.example"].adjustments == []


def test_post_sources_require_a_token_boundary_match():
    posts = [
        _post("p1", "evil.example is the payload host"),
        _post("p2", "notevil.example is a different site", platform="reddit"),
    ]
    cluster = _cluster(posts, IocSet(domains=["evil.example"]))

    context = aggregate_context(cluster, [])

    assert _by_value(context)["evil.example"].sources == ["post:p1"]


def test_benign_indicators_are_filtered_from_the_context():
    posts = [_post("p1", "bad.example and github.com")]
    cluster = _cluster(posts, IocSet(domains=["bad.example", "github.com"]))
    result = _tool_result(
        "SEARCH_ENGINE",
        {"query": "bad.example"},
        summary="search",
        discovered=IocSet(domains=["bleepingcomputer.com"]),
    )

    context = aggregate_context(cluster, [_finding("Actor", [result])])

    values = [item.value for item in context.iocs]
    assert "bad.example" in values
    assert "github.com" not in values
    assert "bleepingcomputer.com" not in values
    assert context.filtered_benign == ["bleepingcomputer.com", "github.com"]


def test_ttps_are_collected_from_tool_summaries_and_post_text():
    posts = [_post("p1", "bad.example operators rely on T1071 traffic")]
    cluster = _cluster(posts, IocSet(domains=["bad.example"]))
    result = _tool_result(
        "SEARCH_ENGINE",
        {"query": "bad.example campaign"},
        summary="Report maps the campaign to T1566.001 and T1059 techniques.",
    )

    context = aggregate_context(cluster, [_finding("Actor", [result])])

    assert context.ttps == ["T1059", "T1071", "T1566.001"]


def test_per_perspective_stats_count_query_outcomes():
    posts = [_post("p1", "bad.example")]
    cluster = _cluster(posts, IocSet(domains=["bad.example"]))
    finding = _finding(
        "Infrastructure",
        [
            _tool_result(
                "WHOIS_LOOKUP",
                {"domain": "bad.example"},
                summary="rdap",
                discovered=IocSet(domains=["bad.example"]),
            ),
            _tool_result("TLS_CT_LOOKUP", {"domain": "bad.example"}, status="unavailable", error="no ct data"),
            _tool_result("PASSIVE_DNS_LOOKUP", {"domain": "bad.example"}, status="error", error="boom"),
        ],
        focus_types=["domain_registration_and_whois", "ssl_tls_certificate_analysis"],
    )

    context = aggregate_context(cluster, [finding])

    bucket = context.per_perspective["Infrastructure"]
    assert bucket["focus_types"] == ["domain_registration_and_whois", "ssl_tls_certificate_analysis"]
    assert bucket["queries"] == 3
    assert bucket["success"] == 1
    assert bucket["unavailable"] == 1
    assert bucket["error"] == 1
    assert bucket["discovered"] == ["bad.example"]
    assert bucket["diagnostics"] == []
    assert context.stats["tool_success_rate"] == 0.333
    assert context.stats["perspectives_active"] == 1


def test_per_perspective_reports_a_query_result_pairing_mismatch():
    posts = [_post("p1", "bad.example")]
    cluster = _cluster(posts, IocSet(domains=["bad.example"]))
    finding = PerspectiveFinding(
        perspective="Infrastructure",
        focus_types=["domain_registration_and_whois"],
        queries=[
            ToolQuery("Infrastructure", "WHOIS_LOOKUP", {"domain": "bad.example"}, focus_type="domain_registration_and_whois"),
            ToolQuery("Infrastructure", "TLS_CT_LOOKUP", {"domain": "bad.example"}, focus_type="ssl_tls_certificate_analysis"),
        ],
        tool_results=[_tool_result("WHOIS_LOOKUP", {"domain": "bad.example"}, summary="rdap")],
        insights=[],
        confidence=0.6,
    )

    context = aggregate_context(cluster, [finding])

    bucket = context.per_perspective["Infrastructure"]
    assert bucket["queries"] == 2
    assert bucket["success"] == 1
    assert bucket["diagnostics"] == ["2 query/queries paired with 1 result(s)"]
    assert len(context.trajectory) == 1


def test_pivot_iocs_are_typed_and_low_confidence_pivots_are_dropped():
    posts = [_post("p1", "host.example")]
    cluster = _cluster(posts, IocSet(domains=["host.example"]))
    findings = [
        _finding(
            "Infrastructure",
            [
                # WHOIS is not a resolution tool, so what it names alone is an
                # unconfirmed pivot: no second tool, no resolution behind it.
                _tool_result(
                    "WHOIS_LOOKUP",
                    {"domain": "host.example"},
                    summary="whois",
                    discovered=IocSet(domains=["weak.example", "strong.example"]),
                ),
                _tool_result(
                    "TLS_CT_LOOKUP",
                    {"domain": "host.example"},
                    summary="ct",
                    discovered=IocSet(domains=["strong.example"]),
                ),
            ],
        )
    ]

    context = aggregate_context(cluster, findings)

    evidence = _by_value(context)
    assert "weak.example" not in evidence
    assert evidence["strong.example"].extraction_type == "Pivot"
    assert evidence["strong.example"].confidence == "MEDIUM"
    assert evidence["strong.example"].tools == ["TLS_CT_LOOKUP", "WHOIS_LOOKUP"]
    assert evidence["strong.example"].perspectives == ["Infrastructure"]
    assert context.stats["pivot_total"] == 1
    assert context.trajectory[0]["new_iocs"] == 2


def test_trajectory_uses_the_reinforcement_learning_step_schema():
    posts = [_post("p1", "bad.example")]
    cluster = _cluster(posts, IocSet(domains=["bad.example"]))
    query = ToolQuery(
        "Infrastructure",
        "WHOIS_LOOKUP",
        {"domain": "bad.example"},
        focus_type="domain_registration_and_whois",
    )
    results = [
        _tool_result("WHOIS_LOOKUP", {"domain": "bad.example"}, summary="rdap"),
        _tool_result("WHOIS_LOOKUP", {"domain": "bad.example"}, summary="rdap again"),
    ]
    finding = PerspectiveFinding(
        perspective="Infrastructure",
        focus_types=["domain_registration_and_whois"],
        queries=[query, query],
        tool_results=results,
        insights=[],
        confidence=0.6,
    )

    context = aggregate_context(cluster, [finding])

    assert [record["step"] for record in context.trajectory] == [1, 2]
    assert context.trajectory[0]["focus_type"] == "domain_registration_and_whois"
    assert context.trajectory[0]["perspective"] == "Infrastructure"
    assert context.trajectory[0]["status"] == "success"
    assert context.trajectory[0]["duplicate"] is False
    assert context.trajectory[1]["duplicate"] is True
    assert context.trajectory[0]["confidence"] == "MEDIUM"
    assert context.trajectory[0]["mode"] == "multi_perspective"
    assert set(context.trajectory[0]) == {
        "step",
        "perspective",
        "focus_type",
        "tool_type",
        "parameters",
        "status",
        "new_iocs",
        "new_cves",
        "new_ttps",
        "duplicate",
        "confidence",
        "mode",
    }


def test_judge_payload_exposes_context_summary_fields():
    posts = [_post("p1", "bad.example CVE-2026-11111")]
    cluster = _cluster(posts, IocSet(domains=["bad.example"], cves=["CVE-2026-11111"]))
    result = _tool_result("NVD_CVE_LOOKUP", {"cve": "CVE-2026-11111"}, summary="T1190 exploitation")

    context = aggregate_context(cluster, [_finding("Technical", [result])])
    payload = context_to_judge_payload(context)

    assert set(payload) == {"iocs", "cves", "ttps", "per_perspective", "stats", "filtered_benign"}
    assert payload["cves"] == ["CVE-2026-11111"]
    assert payload["ttps"] == ["T1190"]
    assert set(payload["iocs"][0]) == {
        "value",
        "type",
        "extraction_type",
        "confidence",
        "source_count",
        "perspectives",
    }
    payload["stats"]["ioc_total"] = 999
    payload["per_perspective"]["Technical"]["focus_types"].append("tampered")
    assert context.stats["ioc_total"] != 999
    assert "tampered" not in context.per_perspective["Technical"]["focus_types"]


def test_a_shared_hosting_ip_contributes_no_co_tenant_iocs_and_earns_no_bump():
    """Rule: an IP whose resolution set exceeds the cap is shared hosting.

    Its co-resident domains are other people's sites, not this campaign's
    infrastructure, so none of them becomes an indicator. A crowded address is
    a false-positive risk, not corroboration, so the queried IP earns no tier
    bump either -- only a marker recording that its co-tenants were dropped.
    """
    co_tenants = [f"co-tenant-{index:02d}.example" for index in range(35)]
    cluster = _cluster(
        [_post("p1", "payload served from 185.158.133.1")],
        IocSet(ips=["185.158.133.1"]),
    )
    finding = _finding(
        "Infrastructure",
        [
            _tool_result(
                "PASSIVE_DNS_LOOKUP",
                {"ip": "185.158.133.1"},
                discovered=IocSet(domains=co_tenants),
            )
        ],
    )

    context = aggregate_context(cluster, [finding])
    by_value = _by_value(context)

    assert [value for value in by_value if value.startswith("co-tenant-")] == []
    assert by_value["185.158.133.1"].extraction_type == "Direct"
    assert by_value["185.158.133.1"].adjustments == ["shared_hosting_environment:co_tenants_dropped"]
    # One post named it and nothing else did: rule 1's single-source tier.
    assert by_value["185.158.133.1"].confidence == "LOW"


def test_a_small_resolution_set_yields_corroborated_pivot_iocs():
    """Below the cap the IP is not shared hosting, so its resolutions pivot.

    Each one still needs a second independent source before it is reportable:
    a single tool answer is an unconfirmed pivot.
    """
    resolved = ["a.evil-campaign.example", "b.evil-campaign.example", "c.evil-campaign.example"]
    cluster = _cluster(
        [_post("p1", "payload served from 185.158.133.1")],
        IocSet(ips=["185.158.133.1"]),
    )
    findings = [
        _finding(
            "Infrastructure",
            [
                _tool_result("PASSIVE_DNS_LOOKUP", {"ip": "185.158.133.1"}, discovered=IocSet(domains=resolved)),
                _tool_result("TLS_CT_LOOKUP", {"domain": "evil-campaign.example"}, discovered=IocSet(domains=resolved)),
            ],
        )
    ]

    context = aggregate_context(cluster, findings)
    by_value = _by_value(context)

    for value in resolved:
        assert by_value[value].extraction_type == "Pivot"
        assert by_value[value].confidence in {"MEDIUM", "HIGH"}


def test_the_resolution_cap_is_configurable_per_run():
    resolved = ["a.evil-campaign.example", "b.evil-campaign.example", "c.evil-campaign.example"]
    cluster = _cluster([_post("p1", "185.158.133.1")], IocSet(ips=["185.158.133.1"]))
    findings = [
        _finding(
            "Infrastructure",
            [
                _tool_result("PASSIVE_DNS_LOOKUP", {"ip": "185.158.133.1"}, discovered=IocSet(domains=resolved)),
                _tool_result("TLS_CT_LOOKUP", {"domain": "evil-campaign.example"}, discovered=IocSet(domains=resolved)),
            ],
        )
    ]

    context = aggregate_context(cluster, findings, max_resolutions_per_ip=2)

    assert [item.value for item in context.iocs if item.extraction_type == "Pivot"] == []
