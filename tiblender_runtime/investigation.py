"""Follow-up investigation for Step 4 judge requests.

The parallel perspective agents of Step 3 live in :mod:`perspectives`;
``investigate_cluster`` is re-exported here so the workflow and existing callers
keep one investigation entry point.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .contract import REQUIRED_FOCUS_TYPES
from .iocs import extract_iocs
from .models import Cluster, PerspectiveFinding, ToolQuery
from .perspectives import investigate_cluster
from .rl.policy import InvestigationStrategy
from .tools import ToolRunner

__all__ = ["PERSPECTIVE_FOCUSES", "investigate_cluster", "investigate_followups"]

FOLLOW_UP_MODE = "multi_perspective_follow_up"

PERSPECTIVE_FOCUSES = {
    # Keep the focus taxonomy in one place so prompts, tests, and RL action
    # names remain aligned with the four investigation perspectives.
    perspective: list(focuses)
    for perspective, focuses in REQUIRED_FOCUS_TYPES.items()
}

_FOLLOWUP_TOOLS = (
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
    "SEARCH_ENGINE",
)
_QUERY_TOOLS = {"SEARCH_ENGINE", "X_SEARCH", "REDDIT_SEARCH", "DISCORD_SEARCH", "TELEGRAM_SEARCH", "GITHUB_SEARCH"}


def _first(values: Iterable[str], limit: int = 3) -> list[str]:
    return list(values)[:limit]


def _query_for_ioc(perspective: str, tool_type: str, value: str, priority: int) -> ToolQuery:
    key = "domain"
    if value.startswith("http"):
        key = "url"
    elif value.startswith("CVE-"):
        key = "cve"
    elif value.count(".") == 3 and all(part.isdigit() for part in value.split(".")):
        key = "ip"
    elif len(value) in {32, 40, 64} and all(char in "0123456789abcdef" for char in value.lower()):
        key = "hash"
    return ToolQuery(perspective=perspective, tool_type=tool_type, parameters={key: value}, priority=priority)


def _tool_from_followup(text: str) -> str:
    upper = text.upper()
    for tool in _FOLLOWUP_TOOLS:
        if tool in upper:
            return tool
    lowered = text.lower()
    if any(token in lowered for token in ("whois", "rdap", "registration")):
        return "WHOIS_LOOKUP"
    if any(token in lowered for token in ("passive dns", "pdns", "resolution")):
        return "PASSIVE_DNS_LOOKUP"
    if any(token in lowered for token in ("certificate", "tls", "ct log")):
        return "TLS_CT_LOOKUP"
    if "cve" in lowered:
        return "NVD_CVE_LOOKUP"
    if any(token in lowered for token in ("github", "poc", "exploit code")):
        return "GITHUB_SEARCH"
    if any(token in lowered for token in ("crawl", "landing page", "web page")):
        return "WEB_CRAWL"
    if "reddit" in lowered:
        return "REDDIT_SEARCH"
    if "telegram" in lowered:
        return "TELEGRAM_SEARCH"
    if "discord" in lowered:
        return "DISCORD_SEARCH"
    if " x " in f" {lowered} " or "twitter" in lowered:
        return "X_SEARCH"
    return "SEARCH_ENGINE"


def _perspective_for_tool(tool_type: str) -> str:
    if tool_type in {"WHOIS_LOOKUP", "PASSIVE_DNS_LOOKUP", "TLS_CT_LOOKUP"}:
        return "Infrastructure"
    if tool_type in {"WEB_CRAWL", "NVD_CVE_LOOKUP", "GITHUB_SEARCH"}:
        return "Technical"
    if tool_type in {"X_SEARCH", "REDDIT_SEARCH", "DISCORD_SEARCH", "TELEGRAM_SEARCH"}:
        return "Social"
    return "Actor"


def _first_ioc_for_tool(tool_type: str, cluster: Cluster, text: str) -> str:
    extracted = extract_iocs(text)
    merged = cluster.iocs.merge(extracted)
    if tool_type == "WEB_CRAWL":
        return _first(merged.urls, 1)[0] if merged.urls else ""
    if tool_type == "NVD_CVE_LOOKUP":
        return _first(merged.cves, 1)[0] if merged.cves else ""
    if tool_type == "GITHUB_SEARCH":
        if merged.cves:
            return merged.cves[0]
        if merged.hashes:
            return merged.hashes[0]
        return cluster.topic_name
    if tool_type in {"WHOIS_LOOKUP", "PASSIVE_DNS_LOOKUP", "TLS_CT_LOOKUP"}:
        if merged.domains:
            return merged.domains[0]
        if merged.ips and tool_type == "WHOIS_LOOKUP":
            return merged.ips[0]
        return ""
    if tool_type in _QUERY_TOOLS:
        values = merged.cves + merged.domains + merged.urls + merged.ips + merged.hashes
        return values[0] if values else cluster.topic_name
    return cluster.topic_name


def _followup_query(cluster: Cluster, followup: str, priority: int) -> ToolQuery | None:
    # Step 4 judges return short natural-language follow-up requests. This
    # mapper turns them back into concrete tool calls grounded in cluster IoCs.
    tool_type = _tool_from_followup(followup)
    perspective = _perspective_for_tool(tool_type)
    value = _first_ioc_for_tool(tool_type, cluster, followup)
    if not value:
        return None
    if tool_type in _QUERY_TOOLS:
        key = "query"
        if tool_type == "GITHUB_SEARCH" and value.upper().startswith("CVE-"):
            key = "cve"
        return ToolQuery(perspective, tool_type, {key: value}, priority=priority)
    return _query_for_ioc(perspective, tool_type, value, priority)


def investigate_followups(
    cluster: Cluster,
    requested_followups: list[str],
    *,
    tool_runner: ToolRunner,
    max_queries_per_perspective: int = 2,
) -> list[PerspectiveFinding]:
    # Follow-up results are returned as normal perspective findings so the next
    # evaluation round can score them using the same evidence schema.
    queries_by_perspective: dict[str, list[ToolQuery]] = {}
    for index, followup in enumerate(requested_followups):
        query = _followup_query(cluster, followup, priority=10 - min(index, 5))
        if query is None:
            continue
        queries_by_perspective.setdefault(query.perspective, []).append(query)

    if not queries_by_perspective:
        # No follow-up mapped to a concrete tool call: repeat the standard
        # multi-perspective sweep (never RL-guided) over the same cluster.
        settings = replace(
            tool_runner.settings,
            max_queries_per_perspective=max_queries_per_perspective,
        )
        return investigate_cluster(
            cluster,
            settings=settings,
            runner=tool_runner,
            strategy=InvestigationStrategy(mode=FOLLOW_UP_MODE, similarity_score=0.0),
            planned_queries=[],
        )

    findings: list[PerspectiveFinding] = []
    for perspective, queries in sorted(queries_by_perspective.items()):
        limited = queries[:max_queries_per_perspective]
        findings.append(
            PerspectiveFinding(
                perspective=perspective,
                focus_types=PERSPECTIVE_FOCUSES[perspective],
                queries=limited,
                tool_results=tool_runner.execute_many(limited),
                insights=[f"Follow-up investigation addressed Step 4 request(s): {'; '.join(requested_followups[:3])}"],
                confidence=0.70,
                mode=FOLLOW_UP_MODE,
            )
        )
    return findings
