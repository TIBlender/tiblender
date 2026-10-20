"""Deterministic report assembly, redaction, and multi-format export."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import clock, language
from .context import InvestigationContext, IocEvidence
from .iocs import filter_reportable_iocs, url_hostname
from .models import Cluster, EvaluationDecision, IocSet, PerspectiveFinding, to_jsonable
from .pdf import pdf_bytes
from .pivots import corroborated_tool_iocs, max_resolutions_per_ip
from .recommendations import (
    recommendation_row,
    recommendation_horizon,
)
from .report_markdown import markdown_report
from .services import ServiceStatus
from .stix import stix_bundle

RUNTIME_VERSION = "1.0.0"


def _now() -> str:
    return clock.now_iso()


_SOURCE_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_DOMAIN_TOKEN_RE = re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,24}\b")


def _redact_contextual_references(text: str) -> str:
    # Keep actionable indicators intact while replacing benign reference links
    # that often appear in social posts and tool summaries.
    def replace_url(match: re.Match[str]) -> str:
        raw = match.group(0)
        suffix = ""
        while raw and raw[-1] in ".,;:!?)]}>\"'":
            suffix = raw[-1] + suffix
            raw = raw[:-1]
        if filter_reportable_iocs(IocSet(urls=[raw])).urls:
            return raw + suffix
        return "[contextual reference URL]" + suffix

    def replace_domain(match: re.Match[str]) -> str:
        value = match.group(0)
        if filter_reportable_iocs(IocSet(domains=[value])).domains:
            return value
        if value.lower().endswith(".html"):
            return "contextual reference document"
        return "contextual reference domain"

    redacted = _SOURCE_URL_RE.sub(replace_url, text)
    return _DOMAIN_TOKEN_RE.sub(replace_domain, redacted)


def _platform_contributions(cluster: Cluster) -> dict[str, str]:
    contributions: dict[str, list[str]] = {}
    for post in cluster.posts:
        contributions.setdefault(post.platform, []).append(_redact_contextual_references(post.normalized_content))
    return {
        platform: " / ".join(items[:2])[:500]
        for platform, items in sorted(contributions.items())
    }


def _platforms_by_value(cluster: Cluster) -> dict[str, list[str]]:
    """Map every indicator observed in a source post to the platforms of the
    posts carrying it.

    This is the attribution the released dataset records in
    ``iocs.items[].platforms``: a Direct indicator names the platforms whose
    posts it was read from, and a value seen only through enrichment gets no
    entry here (and therefore an empty list).
    """
    platforms: dict[str, set[str]] = {}
    for post in cluster.posts:
        if not post.platform:
            continue
        for value in _ioc_values(post.iocs):
            platforms.setdefault(value.lower(), set()).add(post.platform)
    return {value: sorted(names) for value, names in platforms.items()}


def _report_iocs(
    iocs: IocSet,
    direct_values: set[str] | None = None,
    evidence_by_value: dict[str, IocEvidence] | None = None,
    platforms_by_value: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    # The released IoC list mirrors the published dataset's record shape:
    # type, value, extraction_type and the platform attribution. Per-indicator
    # confidence stays in `iocs_evidence`, which keeps the scoring rules and
    # their adjustment markers together with the value they were computed for.
    # When a Step 3 InvestigationContext is available its per-indicator
    # extraction_type (derived from source-count and corroboration rules) takes
    # precedence; otherwise fall back to the Direct/Pivot membership check used
    # without a context.
    direct_values = direct_values or set()
    evidence_by_value = evidence_by_value or {}
    platforms_by_value = platforms_by_value or {}
    rows: list[dict[str, Any]] = []
    for field, label in (
        ("domains", "domain"),
        ("ips", "ip"),
        ("urls", "url"),
        ("hashes", "hash"),
        ("cves", "cve"),
    ):
        for value in getattr(iocs, field):
            evidence = evidence_by_value.get(value)
            extraction_type = (
                evidence.extraction_type
                if evidence is not None
                else ("Direct" if value in direct_values else "Pivot")
            )
            rows.append(
                {
                    "type": label,
                    "value": value,
                    "extraction_type": extraction_type,
                    # A pivot was never read off a post, so it carries no
                    # platform attribution.
                    "platforms": (
                        list(platforms_by_value.get(value.lower(), []))
                        if extraction_type == "Direct"
                        else []
                    ),
                }
            )
    return rows


def _ioc_values(iocs: IocSet) -> set[str]:
    return set(iocs.urls + iocs.domains + iocs.ips + iocs.hashes + iocs.cves)


_IOC_FIELDS = ("urls", "domains", "ips", "hashes", "cves")
_IOC_FIELD_BY_TYPE = {"url": "urls", "domain": "domains", "ip": "ips", "hash": "hashes", "cve": "cves"}


def _context_ioc_set(context: InvestigationContext, *, tools_only: bool = False) -> IocSet:
    """The Step 3 scored evidence list, as an :class:`IocSet`.

    This is the single decision the published ``iocs`` block is derived from,
    so no value can reach ``report_schema.iocs`` without a matching
    ``iocs_evidence`` entry carrying its confidence and adjustment markers.
    """
    buckets: dict[str, list[str]] = {field_name: [] for field_name in _IOC_FIELDS}
    for item in context.iocs:
        field_name = _IOC_FIELD_BY_TYPE.get(item.ioc_type)
        if field_name is None or (tools_only and not item.tools):
            continue
        buckets[field_name].append(item.value)
    return IocSet(**{field_name: sorted(set(values)) for field_name, values in buckets.items()})


def _restrict_to(iocs: IocSet, allowed: set[str]) -> IocSet:
    return IocSet(
        **{field_name: [value for value in getattr(iocs, field_name) if value in allowed] for field_name in _IOC_FIELDS}
    )


def _host(value: str) -> str:
    return url_hostname(value)


def _disqualified_iocs(tool_results: list[Any]) -> IocSet:
    # A failed enrichment query is evidence against promoting that pivot value.
    # This protects reports from dead links, NXDOMAINs, and similar false leads.
    domains: set[str] = set()
    urls: set[str] = set()
    ips: set[str] = set()
    failure_markers = ("HTTP 404", "NXDOMAIN", "Name or service not known", "No address associated with hostname")
    for result in tool_results:
        if result.status not in {"unavailable", "error"}:
            continue
        text = f"{result.error} {result.summary}".lower()
        if not any(marker.lower() in text for marker in failure_markers):
            continue
        params = result.query_params
        if result.tool_type in {"WHOIS_LOOKUP", "PASSIVE_DNS_LOOKUP", "TLS_CT_LOOKUP"}:
            if params.get("domain"):
                domains.add(str(params["domain"]).lower())
            if params.get("ip"):
                ips.add(str(params["ip"]))
        if result.tool_type == "WEB_CRAWL" and params.get("url"):
            url = str(params["url"])
            urls.add(url)
            host = _host(url)
            if host:
                domains.add(host)
    return filter_reportable_iocs(IocSet(urls=sorted(urls), domains=sorted(domains), ips=sorted(ips)))


def _filter_report_iocs(iocs: IocSet, blocked: IocSet, source_cves: set[str] | None = None) -> IocSet:
    clean = filter_reportable_iocs(iocs)
    blocked_domains = set(blocked.domains)
    blocked_urls = set(blocked.urls)
    blocked_ips = set(blocked.ips)
    allowed_cves = {value.upper() for value in source_cves} if source_cves is not None else None
    return IocSet(
        urls=[url for url in clean.urls if url not in blocked_urls and _host(url) not in blocked_domains],
        domains=[domain for domain in clean.domains if domain not in blocked_domains],
        ips=[ip for ip in clean.ips if ip not in blocked_ips],
        hashes=list(clean.hashes),
        cves=[cve for cve in clean.cves if allowed_cves is None or cve in allowed_cves],
    )


def _has_actionable_network_ioc(iocs: IocSet) -> bool:
    return bool(iocs.urls or iocs.domains or iocs.ips or iocs.hashes)


def _evidence_text(cluster: Cluster, findings: list[PerspectiveFinding]) -> str:
    parts: list[str] = [cluster.topic_name, cluster.description]
    parts.extend(post.normalized_content for post in cluster.posts)
    for finding in findings:
        parts.extend(finding.insights)
        parts.extend(result.summary for result in finding.tool_results)
    return " ".join(part for part in parts if part).lower()


def _calibrated_severity(cluster: Cluster, findings: list[PerspectiveFinding], discovered: IocSet) -> str:
    # Source severity is adjusted downward unless concrete activity, victim or
    # scope evidence, and actionable IoCs support a high-impact label.
    original = cluster.severity.value
    if original not in {"CRITICAL", "HIGH"}:
        return original
    evidence = _evidence_text(cluster, findings)
    active_terms = re.search(r"\b(active|exploited|exploitation|campaign|in[- ]the[- ]wild|compromised|malware|phishing)\b", evidence)
    victim_terms = re.search(r"\b(victim|stolen|breach|exfiltrat|ransom|leak|compromised account|affected organization)\b", evidence)
    critical_scope = re.search(r"\b(critical infrastructure|government|healthcare|energy|telecom|financial sector|wide[- ]scale)\b", evidence)
    platform_count = len({post.platform for post in cluster.posts if post.platform})
    actionable_iocs = _has_actionable_network_ioc(discovered)
    if original == "CRITICAL":
        if active_terms and actionable_iocs and platform_count >= 2 and (victim_terms or critical_scope):
            return "CRITICAL"
        if active_terms and actionable_iocs:
            return "HIGH"
        return "MEDIUM"
    if original == "HIGH" and not (actionable_iocs or victim_terms or active_terms):
        return "MEDIUM"
    return original


def _report_recommendations(actions: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for action in actions:
        action_text = _strip_followup_suffix(action)
        if not action_text:
            continue
        rows.append(
            recommendation_row(
                action_text,
                "Prioritized from the report evidence, severity calibration, and available IoCs.",
            )
        )
    if not rows:
        rows.append(
            recommendation_row(
                "Continue monitoring the described campaign and preserve source evidence.",
                "Evidence is credible but does not support an immediate blocking action.",
            )
        )
    return rows[:5]


def _validated_recommendations(raw: Any, fallback_actions: list[str]) -> list[dict[str, str]]:
    if isinstance(raw, list):
        rows: list[dict[str, str]] = []
        for item in raw:
            horizon: Any = None
            if isinstance(item, str):
                action = _strip_followup_suffix(_redact_contextual_references(item.strip()))
                rationale = "Prioritized from the report evidence, severity calibration, and available IoCs."
            elif isinstance(item, dict):
                action = _strip_followup_suffix(_redact_contextual_references(str(item.get("action", "")).strip()))
                rationale = _redact_contextual_references(str(item.get("rationale", "")).strip()) or (
                    "Prioritized from the report evidence, severity calibration, and available IoCs."
                )
                horizon = item.get("horizon")
            else:
                continue
            if action:
                rows.append(recommendation_row(action, rationale, horizon))
        if rows:
            return rows[:5]
    return _report_recommendations(fallback_actions)


def _validated_key_findings(raw: Any, findings: list[PerspectiveFinding]) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        groups: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, str):
                text = _redact_contextual_references(item.strip())
                if text:
                    groups.append({"heading": "Finding", "items": [text]})
                continue
            if not isinstance(item, dict):
                continue
            heading = _redact_contextual_references(str(item.get("heading") or "Finding").strip())
            raw_items = item.get("items") or item.get("findings") or []
            if isinstance(raw_items, str):
                raw_items = [raw_items]
            if not isinstance(raw_items, list):
                raw_items = []
            items = [
                _redact_contextual_references(str(value).strip())
                for value in raw_items
                if str(value).strip()
            ]
            if heading and items:
                groups.append({"heading": heading, "items": items[:5]})
        if groups:
            return groups[:8]
    return _key_findings(findings)


def _validated_timeline(raw: Any, cluster: Cluster) -> list[dict[str, str]]:
    if isinstance(raw, list):
        events: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            timestamp = str(item.get("timestamp", "")).strip()
            event = _redact_contextual_references(str(item.get("event", "")).strip())
            significance = _redact_contextual_references(str(item.get("significance", "")).strip())
            if timestamp and event and significance:
                events.append({"timestamp": timestamp, "event": event, "significance": significance})
        if events:
            return events[:10]
    return language.timeline_events(cluster)


def _validated_ttps(raw: Any, cluster: Cluster) -> list[dict[str, str]]:
    if isinstance(raw, list):
        ttps: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            mitre_id = str(item.get("mitre_id", "")).strip().upper()
            if not re.fullmatch(r"T\d{4}(?:\.\d{3})?", mitre_id):
                continue
            tactic = _redact_contextual_references(str(item.get("tactic", "")).strip())
            technique = _redact_contextual_references(str(item.get("technique", "")).strip())
            evidence = _redact_contextual_references(str(item.get("evidence", "")).strip())
            if tactic and technique and evidence:
                ttps.append({"mitre_id": mitre_id, "tactic": tactic, "technique": technique, "evidence": evidence})
        if ttps:
            return ttps[:8]
    return language.ttp_guess(cluster)


def _strip_followup_suffix(action: str) -> str:
    return re.sub(r"\s*Follow-ups?:.*$", "", action, flags=re.IGNORECASE).strip()


def _validated_cluster_name(raw_name: str, cluster: Cluster) -> str:
    name = _redact_contextual_references(raw_name.strip())
    lowered = name.lower()
    if not name:
        return cluster.topic_name
    if lowered == cluster.id.lower():
        return cluster.topic_name
    if " " not in name and re.fullmatch(r"cluster-[a-z0-9-]+", lowered):
        return cluster.topic_name
    return name


def _english_report(report: dict[str, Any], findings: list[PerspectiveFinding]) -> dict[str, Any]:
    # Tool summaries quote provider text, which can be in any language; this
    # block is the report's English view, so a non-English line is dropped
    # rather than published (the same line stays under sources.tool_results).
    key_findings: list[str] = []
    for finding in findings:
        for insight in finding.insights:
            if insight and insight not in key_findings and language.text_is_english(insight):
                key_findings.append(insight)
        for result in finding.tool_results:
            summary = _public_tool_summary(result)
            if not language.text_is_english(summary):
                continue
            if result.status == "success" and summary and summary not in key_findings:
                key_findings.append(summary)
    if not key_findings:
        key_findings.append(
            f"{report['cluster_name']} was assessed as {report['threat_severity']} based on source-post IoCs and evaluation evidence."
        )
    return {"summary_en": report["executive_summary"], "key_findings_en": key_findings[:8]}


def _key_findings(findings: list[PerspectiveFinding]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for finding in findings:
        items = list(finding.insights)
        # Key findings should highlight confirmed enrichment and perspective
        # context. Failed lookups remain available under sources.tool_results.
        items.extend(
            _public_tool_summary(result)
            for result in finding.tool_results[:5]
            if result.status == "success"
        )
        if not items:
            items.append("No additional reportable tool findings were required for this perspective.")
        groups.append({"heading": finding.perspective, "items": items})
    return groups


def _is_reference_query(value: str, *, is_url: bool = False) -> bool:
    if not value:
        return False
    if is_url:
        host = _host(value)
        return bool(host and not filter_reportable_iocs(IocSet(urls=[value])).urls)
    return not bool(filter_reportable_iocs(IocSet(domains=[value])).domains)


def _public_tool_summary(result: Any) -> str:
    params = result.query_params
    if result.status != "success":
        marker = re.search(r"HTTP\s+\d+", f"{result.error} {result.summary}", flags=re.IGNORECASE)
        detail = f" ({marker.group(0).upper()})" if marker else ""
        return f"{result.tool_type} returned {result.status}{detail}; no additional reportable IoCs were promoted."
    if result.tool_type == "WEB_CRAWL" and _is_reference_query(str(params.get("url", "")), is_url=True):
        return (
            "WEB_CRAWL retrieved a contextual reference page for the cluster topic; "
            "no attack infrastructure was promoted from that reference page."
        )
    if result.tool_type in {"WHOIS_LOOKUP", "PASSIVE_DNS_LOOKUP", "TLS_CT_LOOKUP"} and _is_reference_query(str(params.get("domain", ""))):
        return (
            f"{result.tool_type} was retained as contextual enrichment; "
            "the queried value is not treated as attack infrastructure."
        )
    return result.summary


def _public_query_params(result: Any) -> dict[str, Any]:
    params = dict(result.query_params)
    if result.tool_type == "WEB_CRAWL" and _is_reference_query(str(params.get("url", "")), is_url=True):
        return {"scope": "contextual_reference"}
    if result.tool_type in {"WHOIS_LOOKUP", "PASSIVE_DNS_LOOKUP", "TLS_CT_LOOKUP"} and _is_reference_query(str(params.get("domain", ""))):
        return {"scope": "contextual_reference"}
    return params


def _public_tool_result(result: Any) -> dict[str, Any]:
    # Public report JSON keeps status, summary, and filtered IoCs, but drops raw
    # provider payloads to avoid leaking search results or transient URLs.
    payload = to_jsonable(result)
    payload.pop("raw_json", None)
    if not payload.get("timestamp"):
        payload.pop("timestamp", None)
    payload["query_params"] = _public_query_params(result)
    payload["summary"] = _public_tool_summary(result)
    payload["discovered_iocs"] = to_jsonable(filter_reportable_iocs(result.discovered_iocs))
    return payload


def _public_finding(finding: PerspectiveFinding) -> dict[str, Any]:
    payload = to_jsonable(finding)
    payload["tool_results"] = [
        _public_tool_result(result)
        for result in finding.tool_results
        if result.status == "success"
    ]
    return payload


def _source_post_text(post: Any) -> str:
    return _redact_contextual_references(post.normalized_content)[:500]


def build_report(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    decision: EvaluationDecision,
    report_fields: dict[str, Any] | None = None,
    *,
    context: InvestigationContext | None = None,
    services: dict[str, ServiceStatus] | None = None,
    generation_mode: str = "",
    llm_stage_fallbacks: list[str] | None = None,
    pivot_max_resolutions_per_ip: int | None = None,
) -> dict[str, Any]:
    # Build one canonical report object first, then derive all public formats
    # from it to keep redaction, IoC filtering, and metadata consistent.
    report_fields = report_fields or {}
    report_id = f"TIBLENDER-RUN-{cluster.id.removeprefix('cluster-')}"
    generated_at = _now()
    tool_results = [result for finding in findings for result in finding.tool_results]
    public_tool_results = [
        _public_tool_result(result)
        for result in tool_results
        if result.status == "success"
    ]
    # Failed lookups disqualify the queried indicator from promotion, which
    # keeps obvious dead links and NXDOMAIN pivots out of the final IoC set.
    blocked_iocs = _disqualified_iocs(tool_results)
    source_iocs = filter_reportable_iocs(cluster.iocs)
    if context is not None:
        # One decision, computed once: `iocs`, `report_schema.iocs` and
        # `iocs_evidence` are all views of the Step 3 scored evidence list, so
        # they cannot disagree about which values are indicators, about their
        # confidence, or about the adjustment markers behind it.
        scored_iocs = _context_ioc_set(context)
        # A failed enrichment lookup disqualifies a *pivot* -- a lead the run
        # could not verify. It does not delete an indicator the posts
        # themselves named: rule 3 has already lowered that one a tier.
        discovered = _filter_report_iocs(
            scored_iocs, blocked_iocs, source_cves=set(source_iocs.cves)
        ).merge(_restrict_to(scored_iocs, _ioc_values(source_iocs)))
        tool_report_iocs = _restrict_to(_context_ioc_set(context, tools_only=True), _ioc_values(discovered))
    else:
        # No Step 3 context (a report built straight from findings): apply the
        # same pivot bar here. A pivot-derived value is an indicator only once
        # a second independent tool answer returns it, and never when it is
        # merely co-resident on a shared-hosting IP.
        tool_iocs = corroborated_tool_iocs(
            tool_results,
            direct_values=_ioc_values(cluster.iocs),
            limit=max_resolutions_per_ip(pivot_max_resolutions_per_ip),
        )
        tool_report_iocs = _filter_report_iocs(tool_iocs, blocked_iocs, source_cves=set(source_iocs.cves))
        discovered = source_iocs.merge(tool_report_iocs)
    field_severity = str(report_fields.get("severity", "")).strip().upper()
    severity = field_severity if field_severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"} else _calibrated_severity(cluster, findings, discovered)
    confidence = "HIGH" if decision.final_quality_score >= 0.80 else "MEDIUM" if decision.final_quality_score >= 0.55 else "LOW"
    cluster_name = _validated_cluster_name(str(report_fields.get("cluster_name", "")), cluster)
    executive_summary = str(report_fields.get("executive_summary", "")).strip() or (
        f"{cluster.topic_name} was consolidated from {len(cluster.posts)} social-media posts across "
        f"{len({post.platform for post in cluster.posts})} platform(s). The report preserves direct IoCs, "
        "adds perspective-specific evidence, and records the evaluation loop judgement before publication."
    )
    immediate_actions = report_fields.get("immediate_actions") or [
        "Block or monitor the listed malicious domains, URLs, IPs, and hashes according to local policy.",
        "Search SIEM/EDR telemetry for the listed IoCs and associated CVE or campaign terms.",
        "Preserve source evidence and re-run enrichment if new platform posts appear in the next cycle.",
    ]
    immediate_actions = [_strip_followup_suffix(str(action)) for action in immediate_actions]
    report_recommendations = _validated_recommendations(report_fields.get("recommendations"), immediate_actions)
    report_key_findings = _validated_key_findings(report_fields.get("key_findings"), findings)
    report_ttps = _validated_ttps(report_fields.get("ttps"), cluster)
    report_threat_actor = _redact_contextual_references(str(report_fields.get("threat_actor", "")).strip()) or "Unknown"
    report_timeline = _validated_timeline(report_fields.get("timeline"), cluster)
    generation_metadata = dict(report_fields.get("metadata") or {})
    resolved_generation_mode = (
        generation_mode
        or str(generation_metadata.get("generation_mode") or "").strip()
        or "offline_deterministic"
    )
    # "llm_partial": the LLM produced this report, but at least one stage of
    # its pipeline fell back to a deterministic path (named in
    # metadata.llm_stage_fallbacks).
    if resolved_generation_mode not in {"llm", "llm_partial", "offline_deterministic"}:
        resolved_generation_mode = "offline_deterministic"
    stage_fallbacks = [str(stage) for stage in (llm_stage_fallbacks or [])]
    services_matrix = {
        name: status.mode for name, status in sorted((services or {}).items(), key=lambda item: item[0])
    }
    evidence_by_value = {item.value: item for item in context.iocs} if context is not None else {}
    report = {
        "report_id": report_id,
        "title": cluster_name,
        "generated_at": generated_at,
        "cluster_id": cluster.id,
        "cluster_name": cluster_name,
        "contains_threat_info": True,
        "content_type": "ACTIONABLE_THREAT",
        "threat_category": cluster.threat_category.value,
        "threat_severity": severity,
        "executive_summary": executive_summary,
        "iocs": to_jsonable(discovered),
        "ioc_validation": {
            "source_observed": sorted(_ioc_values(source_iocs)),
            "tool_discovered": sorted(_ioc_values(tool_report_iocs)),
            "tool_disqualified": sorted(_ioc_values(blocked_iocs)),
        },
        "ttps": report_ttps,
        "cves": [{"cve_id": cve, "exploited_in_wild": severity in {"HIGH", "CRITICAL"}} for cve in discovered.cves],
        "sources": {
            "social_media_posts": [
                {
                    "id": post.id,
                    "platform": post.platform,
                    "timestamp": post.post.timestamp,
                    # The post body keeps the language it was posted in; the
                    # English description of it is `summary` (language.py).
                    "text": _source_post_text(post),
                    "summary": language.source_post_summary(
                        post, indicator_count=len(post.iocs.all_values())
                    ),
                }
                for post in cluster.posts
            ],
            "tool_results": public_tool_results,
            "total_sources_count": len(cluster.posts) + len(public_tool_results),
        },
        "source_posts": {
            "count": len(cluster.posts),
            "items": [
                {
                    "source_ref": post.id,
                    "platform": post.platform,
                    "timestamp": post.post.timestamp,
                    "text": _source_post_text(post),
                }
                for post in cluster.posts
            ],
        },
        "perspective_findings": [_public_finding(finding) for finding in findings],
        "platform_unique_contributions": _platform_contributions(cluster),
        "evidence_assessment": {
            "confidence_level": confidence,
            "quality_score": decision.final_quality_score,
            "decision": decision.verdict.value,
            "reason": decision.reason,
        },
        "judgement_trace": [
            {
                "decision": "severity",
                "value": severity,
                "reason": (
                    "Severity is calibrated from actionable IoCs, active-threat language, victim or scope evidence, "
                    "and independent platform coverage."
                ),
                "supporting_evidence_ids": [post.id for post in cluster.posts],
            },
            {
                "decision": "publish",
                "value": decision.verdict.value,
                "reason": decision.reason,
                "supporting_evidence_ids": [f"round-{round_item.round_index}" for round_item in decision.rounds],
            },
        ],
        "evaluation_rounds": [to_jsonable(round_item) for round_item in decision.rounds],
        "recommendations": {
            "immediate_actions": immediate_actions,
            "mitigation_priority": severity,
        },
        "analyst_notes": _redact_contextual_references(str(report_fields.get("analyst_notes", "")).strip()),
        "metadata": {
            "generated_by": "TIBlender Pipeline",
            "report_version": "runtime-0.1",
            **generation_metadata,
            "generation_mode": resolved_generation_mode,
            "llm_stage_fallbacks": stage_fallbacks,
            "services": services_matrix,
            "runtime_version": RUNTIME_VERSION,
            "clock_fixed": clock.is_fixed(),
        },
    }
    report["report_schema"] = {
        # report_schema mirrors the structured fields expected by the browser
        # viewer and downstream consumers, while preserving the richer root JSON.
        "cluster_name": report["cluster_name"],
        "severity": report["threat_severity"],
        "summary": report["executive_summary"],
        "key_findings": report_key_findings,
        "recommendations": report_recommendations,
        "iocs": _report_iocs(
            discovered,
            direct_values=_ioc_values(source_iocs),
            evidence_by_value=evidence_by_value,
            platforms_by_value=_platforms_by_value(cluster),
        ),
        "cves": report["cves"],
        # report_text carries exactly the 11 released sections, in the
        # released shapes: `ttps` are the MITRE technique ids as strings and
        # `sources` is one `{source_number, source_ref, summary}` row per
        # cited post. The runtime's richer views -- the full TTP objects, the
        # source-post bodies and the tool results -- stay at the report root,
        # outside report_schema (see docs/FIDELITY.md).
        "ttps": [item["mitre_id"] for item in report_ttps],
        "threat_actor": report_threat_actor,
        "timeline": report_timeline,
        "sources": [
            {
                "source_number": number,
                "source_ref": post.id,
                "summary": language.source_post_summary(
                    post, indicator_count=len(post.iocs.all_values())
                ),
            }
            for number, post in enumerate(cluster.posts, start=1)
        ],
    }
    # The language guard is the one place both generation paths pass through:
    # a section that is not English -- a model that answered in the source
    # language, or a tool summary quoting a non-English page -- is replaced
    # with prose composed from the cluster's own facts. Source-post text is
    # never touched; it keeps the language it was posted in.
    report["report_schema"], rewritten_sections = language.enforce_english(
        report["report_schema"], cluster=cluster, findings=findings
    )
    if rewritten_sections:
        report["metadata"]["language_fallback"] = True
        report["metadata"]["language_fallback_sections"] = rewritten_sections
    report["cluster_name"] = report["title"] = report["report_schema"]["cluster_name"]
    report["executive_summary"] = report["report_schema"]["summary"]
    report["key_findings"] = report["report_schema"]["key_findings"]
    report["threat_actor"] = report["report_schema"]["threat_actor"]
    report["timeline"] = report["report_schema"]["timeline"]
    if "ttps" in rewritten_sections:
        report["ttps"] = language.ttp_guess(cluster)
        report["report_schema"]["ttps"] = [item["mitre_id"] for item in report["ttps"]]
    # The two root-level prose fields that are not report_schema sections.
    if not language.section_is_english(immediate_actions):
        report["recommendations"]["immediate_actions"] = [
            row["action"] for row in report["report_schema"]["recommendations"]
        ]
    if not language.text_is_english(report["analyst_notes"]):
        report["analyst_notes"] = decision.reason
    # Runtime-only, deliberately outside report_schema so report_text carries
    # exactly the 11 released sections.
    report["english_report"] = _english_report(report, findings)
    report["markdown_report"] = markdown_report(report)
    report["stix_bundle"] = stix_bundle(report)
    report["iocs_evidence"] = (
        [asdict(item) for item in sorted(context.iocs, key=lambda item: item.value)]
        if context is not None
        else []
    )
    return report


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    # All exports are emitted together so a sample run can demonstrate JSON,
    # Markdown, STIX, and PDF output from the same source report.
    report_path = output_dir / "reports" / f"{report['report_id']}.json"
    markdown_path = output_dir / "markdown" / f"{report['report_id']}.md"
    stix_path = output_dir / "stix" / f"{report['report_id']}.json"
    pdf_path = output_dir / "pdf" / f"{report['report_id']}.pdf"
    for path in (report_path.parent, markdown_path.parent, stix_path.parent, pdf_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(report["markdown_report"], encoding="utf-8")
    stix_path.write_text(json.dumps(report["stix_bundle"], ensure_ascii=False, indent=2), encoding="utf-8")
    pdf_path.write_bytes(pdf_bytes(report["markdown_report"]))
