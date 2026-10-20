import json

from tiblender_runtime import clock
from tiblender_runtime.context import aggregate_context
from tiblender_runtime.models import (
    Cluster,
    EvaluationDecision,
    EvaluationRound,
    EvaluationVerdict,
    IocConfidence,
    IocSet,
    NormalizedPost,
    PerspectiveFinding,
    Post,
    Severity,
    ThreatCategory,
    ToolResult,
)
from tiblender_runtime.reporting import build_report, write_report
from tiblender_runtime.services import ServiceStatus


def _decision() -> EvaluationDecision:
    return EvaluationDecision(
        verdict=EvaluationVerdict.PROCEED,
        rounds=[
            EvaluationRound(
                round_index=1,
                judge="llm-judge-a",
                verdict=EvaluationVerdict.PROCEED,
                quality_score=0.82,
                reason="Sufficient evidence.",
            )
        ],
        final_quality_score=0.82,
        reason="Sufficient evidence.",
    )


def _cluster() -> Cluster:
    post = NormalizedPost(
        post=Post(
            id="p1",
            platform="x",
            content="Phishing campaign uses https://bad.example/login and bad.example",
            timestamp="2026-04-11T00:00:00Z",
        ),
        normalized_content="Phishing campaign uses https://bad.example/login and bad.example",
        iocs=IocSet(urls=["https://bad.example/login"], domains=["bad.example"]),
        threat_category=ThreatCategory.PHISHING,
    )
    return Cluster(
        id="cluster-test",
        topic_name="Credential phishing campaign",
        description="Credential phishing campaign",
        posts=[post],
        iocs=post.iocs,
        severity=Severity.HIGH,
        threat_category=ThreatCategory.PHISHING,
    )


def test_report_schema_carries_exactly_the_eleven_released_sections():
    """`report_schema` is the released `report_text`, section for section.

    Runtime-only views (`english_report`, the full TTP objects, the source
    post bodies, the tool results) live at the report root instead.
    """
    report = build_report(_cluster(), [], _decision())

    english_report = report["english_report"]

    assert report["title"] == report["cluster_name"]
    assert set(report["report_schema"]) == {
        "cluster_name",
        "severity",
        "summary",
        "key_findings",
        "recommendations",
        "iocs",
        "cves",
        "ttps",
        "threat_actor",
        "timeline",
        "sources",
    }
    # The released shapes: technique ids as strings, one source row per post.
    assert all(isinstance(item, str) for item in report["report_schema"]["ttps"])
    assert all(
        set(row) == {"source_number", "source_ref", "summary"}
        for row in report["report_schema"]["sources"]
    )
    assert [row["source_ref"] for row in report["report_schema"]["sources"]] == [
        item["source_ref"] for item in report["source_posts"]["items"]
    ]
    assert "finding_groups" not in report["report_schema"]
    assert set(english_report) == {"summary_en", "key_findings_en"}
    assert english_report["summary_en"] == report["executive_summary"]
    assert isinstance(english_report["key_findings_en"], list)
    assert all("extraction_type" in item for item in report["report_schema"]["iocs"])
    assert all("discovery_path" not in item for item in report["report_schema"]["iocs"])


def test_markdown_uses_singular_hash_label():
    cluster = _cluster()
    cluster.iocs = IocSet(hashes=["65dca34b5a28c3d9f8a8eb8a0e60a0b1d6d1d4b64db48ab681fd8db1d80abde7"])

    report = build_report(cluster, [], _decision())

    assert "| hash |" in report["markdown_report"]
    assert "| hashe |" not in report["markdown_report"]


def test_unavailable_404_tool_iocs_are_not_promoted_but_source_iocs_are_reported():
    finding = PerspectiveFinding(
        perspective="Infrastructure",
        focus_types=["domain_registration_and_whois"],
        queries=[],
        tool_results=[
            ToolResult(
                tool_type="WHOIS_LOOKUP",
                query_params={"domain": "bad.example"},
                summary="WHOIS_LOOKUP unavailable: HTTP 404: Not Found",
                discovered_iocs=IocSet(domains=["tool-only.example"]),
                confidence=IocConfidence.LOW,
                status="unavailable",
                error="HTTP 404: Not Found",
            )
        ],
        insights=[],
        confidence=0.3,
    )

    report = build_report(_cluster(), [finding], _decision())

    assert "bad.example" in report["iocs"]["domains"]
    assert "https://bad.example/login" in report["iocs"]["urls"]
    assert "tool-only.example" not in report["iocs"]["domains"]
    assert any(item["value"] == "bad.example" for item in report["report_schema"]["iocs"])
    assert report["sources"]["tool_results"] == []
    assert report["perspective_findings"][0]["tool_results"] == []


def test_markdown_contains_required_report_sections():
    report = build_report(_cluster(), [], _decision())

    markdown = report["markdown_report"]

    assert "## Executive Summary" in markdown
    assert "## Key Findings" in markdown
    assert "## IoCs" in markdown
    assert "## CVEs" in markdown
    assert "## TTPs" in markdown
    assert "## Threat Actor" in markdown
    assert "## Attack Timeline" in markdown
    assert "## Recommendations" in markdown
    assert "## Source Evidence" in markdown


def test_report_filters_reference_iocs_and_calibrates_unconfirmed_critical_report():
    posts = []
    for platform in ("x", "reddit", "discord"):
        posts.append(
            NormalizedPost(
                post=Post(
                    id=f"{platform}-1",
                    platform=platform,
                    content=(
                        "Citrix NetScaler CVE-2026-3055 is being discussed after public reporting, "
                        "but the posts do not include attacker infrastructure."
                    ),
                    timestamp="2026-04-11T00:00:00Z",
                ),
                normalized_content=(
                    "Citrix NetScaler CVE-2026-3055 is being discussed after public reporting, "
                    "but the posts do not include attacker infrastructure."
                ),
                iocs=IocSet(cves=["CVE-2026-3055"]),
                threat_category=ThreatCategory.VULNERABILITY,
            )
        )
    cluster = Cluster(
        id="cluster-netscaler",
        topic_name="CVE-2026-3055 NetScaler exploitation discussion",
        description="CVE-2026-3055 NetScaler exploitation discussion",
        posts=posts,
        iocs=IocSet(cves=["CVE-2026-3055"]),
        severity=Severity.CRITICAL,
        threat_category=ThreatCategory.VULNERABILITY,
    )
    finding = PerspectiveFinding(
        perspective="Technical",
        focus_types=["related_webpage_and_technical_document_retrieval"],
        queries=[],
        tool_results=[
            ToolResult(
                tool_type="WEB_CRAWL",
                query_params={
                    "url": (
                        "https://securityaffairs.com/190131/hacking/urgent-alert-netscaler-bug-"
                        "cve-2026-3055-probed-by-attackers-could-leak-sensitive-data.html"
                    )
                },
                summary="Crawled public news article about CVE-2026-3055.",
                discovered_iocs=IocSet(
                    urls=[
                        "https://securityaffairs.com/190131/hacking/urgent-alert-netscaler-bug-"
                        "cve-2026-3055-probed-by-attackers-could-leak-sensitive-data.html"
                    ],
                    domains=[
                        "securityaffairs.com",
                        "securityaffairs.co",
                        "pierluigi.paganini",
                        "urgent-alert-netscaler-bug-cve-2026-3055-probed-by-attackers-could-leak-"
                        "sensitive-data.html",
                    ],
                    cves=["CVE-2023-4966", "CVE-2025-32975", "CVE-2026-3055", "CVE-2026-41940"],
                ),
                confidence=IocConfidence.MEDIUM,
                status="success",
            )
        ],
        insights=[],
        confidence=0.5,
    )

    report = build_report(cluster, [finding], _decision())

    assert report["threat_severity"] == "MEDIUM"
    assert report["iocs"]["urls"] == []
    assert report["iocs"]["domains"] == []
    assert report["iocs"]["cves"] == ["CVE-2026-3055"]
    assert "CVE-2023-4966" not in report["iocs"]["cves"]
    assert "securityaffairs.com" not in json.dumps(report["stix_bundle"])
    vulnerabilities = [item for item in report["stix_bundle"]["objects"] if item["type"] == "vulnerability"]
    assert [item["name"] for item in vulnerabilities] == ["CVE-2026-3055"]


def test_stix_bundle_declares_stix_21_and_refs_indicator_objects():
    report = build_report(_cluster(), [], _decision())

    bundle = report["stix_bundle"]

    assert bundle["type"] == "bundle"
    assert bundle["spec_version"] == "2.1"
    assert bundle["id"].startswith("bundle--")
    assert all(item["spec_version"] == "2.1" for item in bundle["objects"])
    report_object = next(item for item in bundle["objects"] if item["type"] == "report")
    indicator_ids = {item["id"] for item in bundle["objects"] if item["type"] == "indicator"}
    related_ids = {item["id"] for item in bundle["objects"] if item["type"] != "report"}
    assert indicator_ids
    assert indicator_ids <= set(report_object["object_refs"])
    assert set(report_object["object_refs"]) == related_ids


def test_write_report_emits_json_markdown_stix_and_pdf(tmp_path):
    report = build_report(_cluster(), [], _decision())

    write_report(report, tmp_path)

    assert (tmp_path / "reports" / f"{report['report_id']}.json").exists()
    assert (tmp_path / "markdown" / f"{report['report_id']}.md").exists()
    assert (tmp_path / "stix" / f"{report['report_id']}.json").exists()
    pdf_path = tmp_path / "pdf" / f"{report['report_id']}.pdf"
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF-1.4")


def test_report_json_excludes_raw_provider_payloads():
    finding = PerspectiveFinding(
        perspective="Social",
        focus_types=["phishing_campaign_tracking"],
        queries=[],
        tool_results=[
            ToolResult(
                tool_type="SEARCH_ENGINE",
                query_params={"query": "bad.example"},
                summary="Search engine query returned one English result.",
                discovered_iocs=IocSet(),
                status="success",
                raw_json='{"title":"raw provider payload"}',
            )
        ],
        insights=[],
        confidence=0.6,
    )

    report = build_report(_cluster(), [finding], _decision())

    assert "raw_json" not in json.dumps(report["sources"])
    assert "raw_json" not in json.dumps(report["perspective_findings"])
    assert "raw provider payload" not in json.dumps(report)


def test_report_sources_omit_author_metadata():
    cluster = _cluster()
    cluster.posts[0].post.author = "source_handle"

    report = build_report(cluster, [], _decision())

    assert all("author" not in post for post in report["sources"]["social_media_posts"])
    assert "source_handle" not in json.dumps(report)


def test_build_report_uses_validated_section_specific_report_fields():
    report = build_report(
        _cluster(),
        [],
        _decision(),
        report_fields={
            "cluster_name": "Validated phishing report",
            "severity": "MEDIUM",
            "executive_summary": "Credential theft infrastructure was consolidated from source evidence.",
            "key_findings": [{"heading": "Source Evidence", "items": ["The URL was observed in the source post."]}],
            "recommendations": [
                {
                    "action": "Block the observed credential-theft URL.",
                    "rationale": "The URL was directly observed in source evidence.",
                }
            ],
            "threat_actor": "Unknown",
            "timeline": [
                {
                    "timestamp": "2026-04-11T00:00:00Z",
                    "event": "Source post reported the phishing URL.",
                    "significance": "Initial evidence.",
                }
            ],
            "ttps": [
                {
                    "mitre_id": "T1566",
                    "tactic": "Initial Access",
                    "technique": "Phishing",
                    "evidence": "Credential collection language was observed.",
                }
            ],
        },
    )

    assert report["cluster_name"] == "Validated phishing report"
    assert report["threat_severity"] == "MEDIUM"
    assert report["report_schema"]["key_findings"][0]["heading"] == "Source Evidence"
    assert report["report_schema"]["recommendations"][0]["rationale"] == "The URL was directly observed in source evidence."
    assert report["report_schema"]["threat_actor"] == "Unknown"
    assert report["report_schema"]["timeline"][0]["event"] == "Source post reported the phishing URL."
    assert report["report_schema"]["ttps"] == ["T1566"]
    assert report["ttps"][0]["technique"] == "Phishing"


def test_build_report_rejects_internal_cluster_id_as_report_name():
    cluster = _cluster()
    report = build_report(
        cluster,
        [],
        _decision(),
        report_fields={"cluster_name": cluster.id},
    )

    assert report["cluster_name"] == cluster.topic_name
    assert report["report_schema"]["cluster_name"] == cluster.topic_name


def test_build_report_defaults_to_offline_deterministic_generation_mode():
    report = build_report(_cluster(), [], _decision())

    assert report["metadata"]["generation_mode"] == "offline_deterministic"
    assert report["metadata"]["services"] == {}
    assert report["metadata"]["runtime_version"] == "1.0.0"
    assert report["metadata"]["clock_fixed"] is clock.is_fixed()
    assert report["iocs_evidence"] == []


def test_build_report_generation_mode_argument_overrides_heuristic_metadata():
    report = build_report(
        _cluster(),
        [],
        _decision(),
        report_fields={"metadata": {"generation_mode": "offline_deterministic", "llm_used": False}},
        generation_mode="llm",
    )

    assert report["metadata"]["generation_mode"] == "llm"
    assert report["metadata"]["llm_used"] is False


def test_build_report_records_resolved_services_matrix_sorted_by_name():
    services = {
        "llm": ServiceStatus("llm", "live", "configured"),
        "embedding": ServiceStatus("embedding", "heuristic", "hashing backend"),
    }

    report = build_report(_cluster(), [], _decision(), services=services)

    assert list(report["metadata"]["services"]) == ["embedding", "llm"]
    assert report["metadata"]["services"] == {"embedding": "heuristic", "llm": "live"}


def test_build_report_uses_investigation_context_for_evidence_and_report_schema_iocs():
    cluster = _cluster()
    finding = PerspectiveFinding(
        perspective="Infrastructure",
        focus_types=["domain_registration_and_whois"],
        queries=[],
        tool_results=[
            ToolResult(
                tool_type="PASSIVE_DNS_LOOKUP",
                query_params={"domain": "bad.example"},
                summary="PASSIVE_DNS_LOOKUP resolved bad.example.",
                discovered_iocs=IocSet(domains=["bad.example"]),
                confidence=IocConfidence.MEDIUM,
                status="success",
            )
        ],
        insights=[],
        confidence=0.5,
    )
    context = aggregate_context(cluster, [finding])

    report = build_report(cluster, [finding], _decision(), context=context)

    assert report["iocs_evidence"]
    assert report["iocs_evidence"] == sorted(report["iocs_evidence"], key=lambda item: item["value"])
    assert all(set(item) >= {"value", "ioc_type", "extraction_type", "confidence", "sources"} for item in report["iocs_evidence"])

    evidence_by_value = {item.value: item for item in context.iocs}
    for ioc_row in report["report_schema"]["iocs"]:
        evidence = evidence_by_value.get(ioc_row["value"])
        if evidence is not None:
            assert ioc_row["extraction_type"] == evidence.extraction_type
    # Per-indicator confidence is carried by the evidence list, not by the
    # released IoC rows, which mirror the published dataset's field set.
    assert all("confidence" not in row for row in report["report_schema"]["iocs"])
    assert {item["value"]: item["confidence"] for item in report["iocs_evidence"]} == {
        value: item.confidence for value, item in evidence_by_value.items()
    }


def _infrastructure_finding(results: list[ToolResult]) -> PerspectiveFinding:
    return PerspectiveFinding(
        perspective="Infrastructure",
        focus_types=["related_domain_and_infrastructure_pivoting"],
        queries=[],
        tool_results=results,
        insights=[],
        confidence=0.6,
    )


def _result(tool_type: str, params: dict, discovered: IocSet) -> ToolResult:
    return ToolResult(
        tool_type=tool_type,
        query_params=dict(params),
        summary="",
        discovered_iocs=discovered,
        status="success",
    )


def test_shared_hosting_co_tenants_never_reach_the_flat_ioc_list():
    """The regression this guards: a passive-DNS answer for one shared IP put
    31 unrelated co-resident sites into `iocs.domains` while `iocs_evidence`
    scored only the five indicators that had evidence behind them."""
    cluster = _cluster()
    cluster.iocs = IocSet(
        urls=["https://bad.example/login"], domains=["bad.example"], ips=["185.158.133.1"]
    )
    co_tenants = [f"co-tenant-{index:02d}.example" for index in range(35)]
    findings = [
        _infrastructure_finding(
            [_result("PASSIVE_DNS_LOOKUP", {"ip": "185.158.133.1"}, IocSet(domains=co_tenants))]
        )
    ]
    context = aggregate_context(cluster, findings)

    report = build_report(cluster, findings, _decision(), context=context)

    assert not [value for value in report["iocs"]["domains"] if value.startswith("co-tenant-")]
    assert report["iocs"]["ips"] == ["185.158.133.1"]


def test_every_reported_ioc_agrees_with_the_scored_evidence_list():
    """`iocs`, `report_schema.iocs` and `iocs_evidence` are one decision.

    A value in the flat list with no evidence entry is an unverified
    infrastructure relationship presented as an indicator.
    """
    cluster = _cluster()
    cluster.iocs = IocSet(
        urls=["https://bad.example/login"], domains=["bad.example"], ips=["185.158.133.1"]
    )
    resolved = ["a.pivot-target.example", "b.pivot-target.example"]
    findings = [
        _infrastructure_finding(
            [
                _result("PASSIVE_DNS_LOOKUP", {"ip": "185.158.133.1"}, IocSet(domains=resolved)),
                _result("TLS_CT_LOOKUP", {"domain": "bad.example"}, IocSet(domains=resolved)),
                _result(
                    "WHOIS_LOOKUP",
                    {"domain": "bad.example"},
                    IocSet(domains=["uncorroborated-pivot.example"]),
                ),
            ]
        )
    ]
    context = aggregate_context(cluster, findings)

    report = build_report(cluster, findings, _decision(), context=context)

    evidence = {item["value"]: item for item in report["iocs_evidence"]}
    for row in report["report_schema"]["iocs"]:
        assert row["extraction_type"] in {"Direct", "Pivot"}
        if row["extraction_type"] == "Pivot":
            assert row["value"] in evidence, f"{row['value']} is an unscored pivot indicator"
            assert evidence[row["value"]]["confidence"] in {"LOW", "MEDIUM", "HIGH"}
            assert row["platforms"] == []
    assert "uncorroborated-pivot.example" not in report["iocs"]["domains"]
    for value in resolved:
        assert value in report["iocs"]["domains"]
        assert evidence[value]["extraction_type"] == "Pivot"


def test_a_contradicted_pivot_never_reaches_the_flat_list_as_high_confidence():
    """The two lists are one decision, including on contradiction markers.

    `context._apply_contradictions` and `_disqualified_iocs` recognise
    different provider phrasings ("no data" is only in the first). While the
    flat list was recomputed independently, a pivot the scorer had lowered to
    LOW and dropped still shipped in `report_schema.iocs` at the report-level
    fallback confidence, with no `iocs_evidence` entry behind it.
    """
    cluster = _cluster()
    cluster.iocs = IocSet(urls=["https://bad.example/login"], domains=["bad.example"])
    findings = [
        _infrastructure_finding(
            [
                _result("WHOIS_LOOKUP", {"domain": "bad.example"}, IocSet(domains=["pivot.example"])),
                _result("PASSIVE_DNS_LOOKUP", {"domain": "bad.example"}, IocSet(domains=["pivot.example"])),
                ToolResult(
                    tool_type="WHOIS_LOOKUP",
                    query_params={"domain": "pivot.example"},
                    summary="",
                    discovered_iocs=IocSet(),
                    status="unavailable",
                    error="RDAP: no data for this object",
                ),
            ]
        )
    ]
    context = aggregate_context(cluster, findings)

    report = build_report(cluster, findings, _decision(), context=context)

    evidence = {item["value"]: item for item in report["iocs_evidence"]}
    assert "pivot.example" not in evidence
    assert "pivot.example" not in report["iocs"]["domains"]
    assert all(row["value"] in evidence for row in report["report_schema"]["iocs"])


def test_released_ioc_rows_carry_platforms_and_drop_confidence():
    """The released-facing IoC list mirrors the dataset: {type, value,
    extraction_type, platforms}. Per-indicator confidence stays in
    ``iocs_evidence``, never in the released list."""
    report = build_report(_cluster(), [], _decision())

    rows = report["report_schema"]["iocs"]

    assert rows
    for row in rows:
        assert set(row) == {"type", "value", "extraction_type", "platforms"}
        assert "confidence" not in row
        assert isinstance(row["platforms"], list)


def test_direct_iocs_are_attributed_to_their_source_post_platforms():
    """Direct indicators carry the platforms of the posts they appear in;
    pivots carry an empty list, exactly as the released dataset records it."""
    report = build_report(_cluster(), [], _decision())

    by_value = {row["value"]: row for row in report["report_schema"]["iocs"]}

    assert by_value["bad.example"]["extraction_type"] == "Direct"
    assert by_value["bad.example"]["platforms"] == ["x"]


def test_pivot_iocs_carry_no_platform_attribution():
    cluster = _cluster()
    finding = _infrastructure_finding(
        [
            _result("PASSIVE_DNS_LOOKUP", {"domain": "bad.example"}, IocSet(ips=["185.158.133.1"])),
            _result("WHOIS_LOOKUP", {"ip": "185.158.133.1"}, IocSet(ips=["185.158.133.1"])),
        ]
    )
    context = aggregate_context(cluster, [finding])

    report = build_report(cluster, [finding], _decision(), context=context)

    by_value = {row["value"]: row for row in report["report_schema"]["iocs"]}
    assert by_value["185.158.133.1"]["extraction_type"] == "Pivot"
    assert by_value["185.158.133.1"]["platforms"] == []


def test_recommendations_carry_a_deterministic_horizon():
    report = build_report(_cluster(), [], _decision())

    rows = report["report_schema"]["recommendations"]

    assert rows
    for row in rows:
        assert set(row) == {"action", "rationale", "horizon"}
        assert row["horizon"] in {"immediate", "short_term", "long_term"}


def test_horizon_is_derived_from_the_leading_action_verb():
    from tiblender_runtime.reporting import recommendation_horizon

    assert recommendation_horizon("Block the listed malicious domains.") == "immediate"
    assert recommendation_horizon("Hunt for the listed IoCs in historical telemetry.") == "short_term"
    assert recommendation_horizon("Integrate the indicators into the detection pipeline.") == "long_term"


def test_llm_supplied_horizon_is_preserved_and_invalid_values_are_derived():
    report = build_report(
        _cluster(),
        [],
        _decision(),
        {
            "recommendations": [
                {"action": "Block bad.example at the perimeter.", "rationale": "Directly observed.", "horizon": "long_term"},
                {"action": "Block bad.example at the proxy.", "rationale": "Directly observed.", "horizon": "nonsense"},
            ]
        },
    )

    rows = report["report_schema"]["recommendations"]
    assert rows[0]["horizon"] == "long_term"
    assert rows[1]["horizon"] == "immediate"


def test_source_posts_expose_text_not_content_preview():
    report = build_report(_cluster(), [], _decision())

    posts = report["sources"]["social_media_posts"]

    assert posts
    for post in posts:
        assert "text" in post
        assert "content_preview" not in post
    assert "Phishing campaign uses" in posts[0]["text"]


def test_markdown_renders_platforms_and_grouped_recommendations():
    report = build_report(_cluster(), [], _decision())

    markdown = report["markdown_report"]

    assert "| Type | Value | Extraction Type | Source Platforms |" in markdown
    assert "Confidence |" not in markdown
    assert "### Immediate" in markdown
