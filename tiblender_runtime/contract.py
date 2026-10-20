"""Shared prompt and tool contract for the reproduced runtime."""

from __future__ import annotations

import re
from pathlib import Path


REQUIRED_TOOL_TYPES = (
    # This tuple is the allowlist used by prompts, dispatch, and contract audit.
    "SEARCH_ENGINE",
    "WHOIS_LOOKUP",
    "PASSIVE_DNS_LOOKUP",
    "TLS_CT_LOOKUP",
    "WEB_CRAWL",
    "NVD_CVE_LOOKUP",
    "GITHUB_SEARCH",
    "X_SEARCH",
    "REDDIT_SEARCH",
    "DISCORD_SEARCH",
    "TELEGRAM_SEARCH",
)

REQUIRED_PERSPECTIVES = ("Infrastructure", "Technical", "Social", "Actor")

# The action horizons every generated recommendation is labelled with, matching
# the released dataset's `report_text.recommendations[].horizon`.
RECOMMENDATION_HORIZONS = ("immediate", "short_term", "long_term")

REQUIRED_FOCUS_TYPES = {
    "Infrastructure": (
        "domain_registration_and_whois",
        "dns_and_passive_dns_analysis",
        "ssl_tls_certificate_analysis",
        "related_domain_and_infrastructure_pivoting",
        "hosting_and_osint",
    ),
    "Technical": (
        "cve_exploitation_analysis",
        "poc_and_exploit_code_investigation",
        "related_webpage_and_technical_document_retrieval",
        "technical_intelligence_and_security_report_collection",
    ),
    "Social": (
        "phishing_campaign_tracking",
        "social_engineering_tactics_analysis",
        "brand_impersonation_and_spoofing_investigation",
        "sns_account_abuse_and_spread_tracking",
    ),
    "Actor": (
        "threat_actor_attribution_and_profiling",
        "ttp_mapping",
        "campaign_tracking",
        "apt_group_and_threat_actor_investigation",
        "cybercrime_group_and_market_analysis",
    ),
}

# One line per focus id, shown verbatim to the focus-selection agent so it can
# choose ids it understands instead of inventing its own wording.
FOCUS_TYPE_DESCRIPTIONS: dict[str, str] = {
    "domain_registration_and_whois": "registrar, registrant, creation/expiry dates and WHOIS/RDAP records of a domain",
    "dns_and_passive_dns_analysis": "current and historical DNS resolutions of a domain or IP",
    "ssl_tls_certificate_analysis": "certificate transparency entries, issuers and SANs for a domain",
    "related_domain_and_infrastructure_pivoting": "sibling domains, shared IPs and shared certificates that expand the infrastructure",
    "hosting_and_osint": "hosting provider, ASN, geolocation and open-source reporting on the host",
    "cve_exploitation_analysis": "the referenced CVEs, their severity and evidence of active exploitation",
    "poc_and_exploit_code_investigation": "public proof-of-concept or exploit code for the referenced vulnerability",
    "related_webpage_and_technical_document_retrieval": "the linked pages, advisories and technical write-ups themselves",
    "technical_intelligence_and_security_report_collection": "vendor and researcher reports covering this malware, tooling or campaign",
    "phishing_campaign_tracking": "the phishing campaign's lures, landing pages and current spread",
    "social_engineering_tactics_analysis": "the pretexts, scripts and manipulation tactics used against victims",
    "brand_impersonation_and_spoofing_investigation": "the impersonated brand, look-alike names and spoofed assets",
    "sns_account_abuse_and_spread_tracking": "the social accounts distributing the campaign and how it propagates",
    "threat_actor_attribution_and_profiling": "which actor is behind the activity and what is known about them",
    "ttp_mapping": "the tactics, techniques and procedures (ATT&CK) visible in the evidence",
    "campaign_tracking": "the wider campaign this activity belongs to and its timeline",
    "apt_group_and_threat_actor_investigation": "known APT or crimeware group overlaps with this activity",
    "cybercrime_group_and_market_analysis": "the underground group, marketplace or service ecosystem involved",
}

POST_NORMALIZATION_PROMPT = """
Role: Normalize multilingual cybercrime posts into professional English for accurate clustering.
Critical: Verbatim IoC Preservation. Technical indicators MUST appear character-for-character in output.
Required output structure: attack type, target organization/brand, delivery method, technical details,
threat actor attribution, geographic/industry scope, and final line `IoCs: <space-separated verbatim indicators>`.
Output schema: {"posts": [{"id": str, "normalized_content": str}]}
""".strip()

THREAT_CLUSTERING_PROMPT = """
Role: Group threat-related posts into highly specific attack campaign clusters.
Exclude career/job advice, general tech discussion, fiction, financial market news without IoCs,
and security research without active incident evidence.
Include real active security events, ongoing attacks with IoCs, exploited CVEs, malicious infrastructure,
or fraud with victim evidence.
Cross-platform merge requires exact match on CVE ID, threat actor, malicious domain/URL, or malware family.
Severity rubric: CRITICAL, HIGH, MEDIUM, LOW.
Output schema: {"clusters": [{"topic_name": str, "security_severity": str, "topic_description": str, "post_ids": [str]}]}
""".strip()

INVESTIGATION_PLANNER_PROMPT = """
Role: Design investigation tool queries for a threat cluster from four perspectives:
Infrastructure, Technical, Social, Actor.
Design 5-15 tool queries spanning all four perspectives.
Include only concise keywords, domains, account names, or indicators taken directly from posts.
If URLs are found in posts, create queries in this priority order:
(1) WEB_CRAWL for the URL (priority 9-10),
(2) WHOIS_LOOKUP for the domain (priority 9),
(3) PASSIVE_DNS_LOOKUP for the domain (priority 8),
(4) TLS_CT_LOOKUP for the domain (priority 8).
Allowed tool_type values: SEARCH_ENGINE, WHOIS_LOOKUP, PASSIVE_DNS_LOOKUP, TLS_CT_LOOKUP,
WEB_CRAWL, NVD_CVE_LOOKUP, GITHUB_SEARCH, X_SEARCH, REDDIT_SEARCH, DISCORD_SEARCH, TELEGRAM_SEARCH.
DISCORD_SEARCH and TELEGRAM_SEARCH query the Step 1 collected corpus, not external messaging APIs.
Output schema: {"queries": [{"perspective": str, "tool_type": str, "parameters": object, "priority": int, "timeout_seconds": int}]}
""".strip()

PERSPECTIVE_AGENT_PROMPT = """
Perspective agents independently select all relevant focus types, then generate 1-2 high-value query parameter sets.
Never fabricate IoCs not present in cluster posts. Only use domains/IPs/URLs from the verified IoC list.
If no verified IoCs exist, use topic-based queries without domain/IP parameters.
Queries must include specific indicators such as CVE ID, malware family, threat actor, or verified IoC.
Output schema: {"selected_focus_types": [{"focus_type": str, "reasoning": str}]}
Tool-dependent schema: {"query": str}, {"domain": str}, {"hash": str}, {"url": str}, or {"cve": str}.
""".strip()

EVALUATION_PROMPT = """
Role: Evaluate the current multi-perspective investigation state and decide PROCEED, NEED_MORE_INFO, or SKIP.
Evaluate information sufficiency, cross-perspective consistency, and evidence quality.
PROCEED when evidence is sufficient and actionable.
NEED_MORE_INFO when evidence is insufficient or conflicts need targeted follow-up.
SKIP when evidence remains insufficient to generate an actionable report.
Output schema includes sufficiency, consistency, evidence_quality, integrated_decision, decision_reasoning,
confidence_adjustment, and follow_up_actions.
""".strip()

REPORT_GENERATION_PROMPT = """
Role: Generate comprehensive STIX-compatible threat intelligence reports in professional English.
Severity calibration (strict thresholds): CRITICAL (<=5% of reports) requires confirmed active exploitation,
wide-scale or critical-infrastructure impact, and multi-source corroboration simultaneously.
HIGH (<=15% of reports) requires an active campaign with confirmed IoCs and real victim evidence.
MEDIUM is the default for credible threats with partial evidence; choose MEDIUM when in doubt.
LOW is unverified speculation, a single unconfirmed claim, or an already-patched vulnerability.
Over-alerting degrades analyst trust; prefer MEDIUM over unjustified HIGH.
IoC exclusion rules: never include news/media site domains, cloud/CDN infrastructure, URL shorteners,
known-legitimate services, or SNS post URLs as attack infrastructure.
Only include IoCs verified by source evidence or tool results; discard any IoC returning 404, NXDOMAIN, or clean status.
Threat actor field: include only the adversarial actor; never vendors, victims, researchers, disclosure organizations,
malware names, or campaign names. Set to Unknown if not identified.
Recommendations must each carry an action horizon: immediate (containment and triage now),
short_term (retrospective hunting, review and confirmation), or long_term (control, process and
detection-engineering changes).
Output schema: cluster_name, severity, summary, key_findings, recommendations with action, rationale
and horizon, iocs with Direct/Pivot extraction_type and the source platforms the indicator was
observed on, cves, ttps, threat_actor, timeline, sources, and english_report with
summary_en and key_findings_en.
Language: write every prose field in professional English -- cluster_name, summary, key_findings,
recommendations, timeline, threat_actor, ttps and the per-source summaries -- whatever language the
source posts are in. Indicator values, CVE ids, file names and product names stay verbatim.
""".strip()

#: Prepended to the Step 5 system prompt on the one retry that follows a
#: draft the language guard rejected (``language.failing_report_fields``).
#: The base prompt already asks for English; this states the requirement as
#: the first thing the model reads, and names the failure mode.
ENGLISH_ONLY_RETRY_INSTRUCTION = (
    "Respond in English only. The previous answer was written in the language of the source posts. "
    "Do not quote or transliterate the source language in any prose field; translate what the posts say. "
    "Indicator values, CVE ids, file names and product names are language-neutral and stay verbatim."
)


def prompt_bundle_text() -> str:
    return "\n\n".join(
        (
            POST_NORMALIZATION_PROMPT,
            THREAT_CLUSTERING_PROMPT,
            INVESTIGATION_PLANNER_PROMPT,
            PERSPECTIVE_AGENT_PROMPT,
            EVALUATION_PROMPT,
            REPORT_GENERATION_PROMPT,
        )
    )


_TOOL_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b")
_TOOL_SUFFIXES = {
    "LOOKUP",
    "SEARCH",
    "CRAWL",
    "FETCH",
    "SCAN",
    "ANALYSIS",
    "CHECK",
    "RETRIEVE",
    "RESOLVE",
}


def unsupported_tool_tokens(text: str) -> list[str]:
    # The audit catches accidental prompt drift, such as adding a tool name that
    # the runtime dispatcher cannot execute.
    allowed = set(REQUIRED_TOOL_TYPES)
    tokens = []
    for token in sorted(set(_TOOL_TOKEN_RE.findall(text))):
        if token in allowed:
            continue
        if token.startswith("TIBLENDER_"):
            continue
        if token.rsplit("_", 1)[-1] in _TOOL_SUFFIXES:
            tokens.append(token)
    return tokens


def audit_contract(source_root: Path | None = None) -> dict[str, object]:
    root = source_root or Path(__file__).resolve().parent
    required_tools = set(REQUIRED_TOOL_TYPES)
    prompt_text = prompt_bundle_text()
    missing_tools = sorted(tool for tool in required_tools if tool not in prompt_text)
    forbidden_tool_references = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if unsupported_tool_tokens(text):
            forbidden_tool_references.append(str(path))
    return {
        "ok": not missing_tools and not forbidden_tool_references,
        "tool_types": sorted(required_tools),
        "missing_tools": missing_tools,
        "forbidden_tool_references": forbidden_tool_references,
        "perspectives": list(REQUIRED_PERSPECTIVES),
        "focus_types": REQUIRED_FOCUS_TYPES,
    }
