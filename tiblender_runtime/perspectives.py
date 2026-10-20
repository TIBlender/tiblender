"""Step 3 perspective agents.

Four independent agents (Infrastructure, Technical, Social, Actor) each select
2-4 investigation focuses, derive tool queries from the cluster's verified IoCs,
run in parallel, and pivot one level from newly discovered infrastructure.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from .contract import REQUIRED_FOCUS_TYPES, REQUIRED_PERSPECTIVES
from .iocs import filter_reportable_iocs
from .llm import (
    LLMGenerationError,
    _query_is_specific,
    generate_perspective_queries_with_llm,
    select_focus_types_with_llm,
)
from .llm_fallbacks import category_of, record_fallback, record_padding
from .models import Cluster, IocSet, PerspectiveFinding, ToolQuery, ToolResult
from .rl.policy import InvestigationStrategy
from .settings import Settings
from .tools import ToolRunner

logger = logging.getLogger(__name__)

PERSPECTIVE_TOOLS: dict[str, list[str]] = {
    "Infrastructure": ["WHOIS_LOOKUP", "PASSIVE_DNS_LOOKUP", "TLS_CT_LOOKUP", "SEARCH_ENGINE"],
    "Technical": ["WEB_CRAWL", "NVD_CVE_LOOKUP", "GITHUB_SEARCH", "SEARCH_ENGINE"],
    "Social": ["X_SEARCH", "REDDIT_SEARCH", "DISCORD_SEARCH", "TELEGRAM_SEARCH"],
    "Actor": ["SEARCH_ENGINE", "X_SEARCH", "REDDIT_SEARCH"],
}

# Each contract focus type maps to the ordered tools that can answer it.
FOCUS_TOOLS: dict[str, list[str]] = {
    "domain_registration_and_whois": ["WHOIS_LOOKUP"],
    "dns_and_passive_dns_analysis": ["PASSIVE_DNS_LOOKUP"],
    "ssl_tls_certificate_analysis": ["TLS_CT_LOOKUP"],
    "related_domain_and_infrastructure_pivoting": ["PASSIVE_DNS_LOOKUP", "TLS_CT_LOOKUP"],
    "hosting_and_osint": ["WHOIS_LOOKUP", "SEARCH_ENGINE"],
    "cve_exploitation_analysis": ["NVD_CVE_LOOKUP", "SEARCH_ENGINE"],
    "poc_and_exploit_code_investigation": ["GITHUB_SEARCH"],
    "related_webpage_and_technical_document_retrieval": ["WEB_CRAWL", "SEARCH_ENGINE"],
    "technical_intelligence_and_security_report_collection": ["SEARCH_ENGINE"],
    "phishing_campaign_tracking": ["X_SEARCH", "REDDIT_SEARCH", "SEARCH_ENGINE"],
    "social_engineering_tactics_analysis": ["REDDIT_SEARCH", "TELEGRAM_SEARCH"],
    "brand_impersonation_and_spoofing_investigation": ["X_SEARCH", "SEARCH_ENGINE"],
    "sns_account_abuse_and_spread_tracking": ["X_SEARCH", "DISCORD_SEARCH", "TELEGRAM_SEARCH"],
    "threat_actor_attribution_and_profiling": ["SEARCH_ENGINE"],
    "ttp_mapping": ["SEARCH_ENGINE"],
    "campaign_tracking": ["SEARCH_ENGINE", "X_SEARCH"],
    "apt_group_and_threat_actor_investigation": ["SEARCH_ENGINE", "REDDIT_SEARCH"],
    "cybercrime_group_and_market_analysis": ["TELEGRAM_SEARCH", "DISCORD_SEARCH", "SEARCH_ENGINE"],
}

PIVOT_FOCUS = "related_domain_and_infrastructure_pivoting"
PIVOT_PERSPECTIVES = ("Infrastructure", "Technical")
_PIVOT_TOOLS_BY_TYPE = {
    "domain": ("WHOIS_LOOKUP", "PASSIVE_DNS_LOOKUP", "TLS_CT_LOOKUP"),
    "ip": ("WHOIS_LOOKUP", "PASSIVE_DNS_LOOKUP"),
}
_MAX_PIVOT_QUERIES = 3
_MAX_QUERIES_PER_FOCUS = 2
_MAX_INDICATORS_PER_FOCUS = 2
_MIN_FOCUS_TYPES = 2
_MAX_FOCUS_TYPES = 4

_SEARCH_TOOLS = frozenset({"SEARCH_ENGINE", "X_SEARCH", "REDDIT_SEARCH", "DISCORD_SEARCH", "TELEGRAM_SEARCH"})
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,}")
_TTP_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_TOPIC_STOPWORDS = {"campaign", "threat", "actor", "attack", "malicious", "cyber", "group", "activity"}


def _focus_for_tool(tool_type: str, perspective: str, focus_types: list[str] | None = None) -> str:
    """Focus of ``perspective`` served by ``tool_type``.

    The agent's own selected focuses are preferred, so a query is attributed to
    the investigation step that asked for it; the perspective's contract order
    is the fallback. A focus of another perspective is never returned, which
    keeps the RL trajectory's action attribution correct.
    """
    contract = list(REQUIRED_FOCUS_TYPES.get(perspective, ()))
    for focus in [*(focus_types or []), *contract]:
        if focus in contract and tool_type in FOCUS_TOOLS.get(focus, []):
            return focus
    return ""


class _Signals:
    """Read-only view of the cluster evidence used for focus and query choice."""

    __slots__ = ("iocs", "category", "platforms", "text", "topic_terms")

    def __init__(self, cluster: Cluster) -> None:
        self.iocs = filter_reportable_iocs(cluster.iocs)
        self.category = cluster.threat_category.value
        self.platforms = {post.platform for post in cluster.posts if post.platform}
        self.text = " ".join(
            [cluster.topic_name, cluster.description, *[post.normalized_content for post in cluster.posts]]
        ).lower()
        indicators = {value.lower() for value in self.iocs.all_values()}
        self.topic_terms = [
            token
            for token in dict.fromkeys(match.group(0).lower() for match in _TOKEN_RE.finditer(cluster.topic_name))
            if token not in _TOPIC_STOPWORDS and token not in indicators
        ]


def _infrastructure_scores(signals: _Signals) -> dict[str, int]:
    scores: dict[str, int] = {}
    if signals.iocs.domains or signals.iocs.urls:
        scores["domain_registration_and_whois"] = 2
        scores["dns_and_passive_dns_analysis"] = 2
        scores["ssl_tls_certificate_analysis"] = 2
    if signals.iocs.ips:
        scores["dns_and_passive_dns_analysis"] = scores.get("dns_and_passive_dns_analysis", 0) + 2
        scores["hosting_and_osint"] = scores.get("hosting_and_osint", 0) + 2
    if len(signals.iocs.domains) >= 2:
        scores["related_domain_and_infrastructure_pivoting"] = 3
    return scores


def _technical_scores(signals: _Signals) -> dict[str, int]:
    scores: dict[str, int] = {}
    if signals.iocs.cves:
        scores["cve_exploitation_analysis"] = 3
        scores["poc_and_exploit_code_investigation"] = 2
    if signals.iocs.urls:
        scores["related_webpage_and_technical_document_retrieval"] = 2
    if signals.iocs.hashes or signals.category == "malware":
        scores["technical_intelligence_and_security_report_collection"] = 2
    return scores


def _social_scores(signals: _Signals) -> dict[str, int]:
    scores = {"social_engineering_tactics_analysis": 1}
    if signals.category in {"phishing", "fraud"}:
        scores["phishing_campaign_tracking"] = 3
        scores["brand_impersonation_and_spoofing_investigation"] = 2
    if len(signals.platforms) >= 2:
        scores["sns_account_abuse_and_spread_tracking"] = 2
    return scores


def _actor_scores(signals: _Signals) -> dict[str, int]:
    scores = {"campaign_tracking": 1}
    if signals.category in {"apt", "ransomware"}:
        scores["apt_group_and_threat_actor_investigation"] = 3
        scores["threat_actor_attribution_and_profiling"] = 2
    if signals.category == "malware" or "c2" in signals.text:
        scores["cybercrime_group_and_market_analysis"] = 2
    if _TTP_RE.search(signals.text):
        scores["ttp_mapping"] = 2
    return scores


_SCORERS = {
    "Infrastructure": _infrastructure_scores,
    "Technical": _technical_scores,
    "Social": _social_scores,
    "Actor": _actor_scores,
}


def _heuristic_focus_types(cluster: Cluster, perspective: str) -> list[str]:
    contract = list(REQUIRED_FOCUS_TYPES[perspective])
    scores = _SCORERS[perspective](_Signals(cluster))
    scored = [focus for focus in contract if scores.get(focus, 0) > 0]
    return sorted(scored, key=lambda focus: (-scores[focus], contract.index(focus)))


def _finalise_focus_types(perspective: str, hints: list[str], ranked: list[str]) -> list[str]:
    # RL hints lead and the perspective's own ranking follows, capped at four.
    # The contract order only pads up to the two-focus minimum, so an agent
    # investigates unsupported focuses only when it has nothing better.
    contract = list(REQUIRED_FOCUS_TYPES[perspective])
    ordered: list[str] = []
    for focus in [*hints, *ranked]:
        if focus in contract and focus not in ordered:
            ordered.append(focus)
        if len(ordered) == _MAX_FOCUS_TYPES:
            break
    for focus in contract:
        if len(ordered) >= _MIN_FOCUS_TYPES:
            break
        if focus not in ordered:
            ordered.append(focus)
    return ordered


def _focus_types_with_notes(
    cluster: Cluster,
    perspective: str,
    *,
    hints: list[str],
    settings: Settings,
) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    if settings.llm_configured() and not settings.allow_heuristic_reports:
        # One perspective unit: retried once, then this perspective (only)
        # falls back to the deterministic focus ranking. A zero-valid answer
        # (or an LLMGenerationError after retry) is the only case that counts
        # as a fallback; a single valid focus is padded up to the two-focus
        # minimum with the heuristic ranking instead (not a fallback).
        try:
            selected = select_focus_types_with_llm(settings, cluster, perspective)
        except LLMGenerationError as first_error:
            try:
                selected = select_focus_types_with_llm(settings, cluster, perspective)
            except LLMGenerationError as error:
                # `notes` becomes PerspectiveFinding.insights, which reporting
                # renders into the published report, so it carries a
                # code-authored sentence naming the stage and the raise-site
                # category -- never the failure text, which may embed the
                # model's draft of scraped posts. The raw text stays in
                # `detail`, which only reaches DEBUG.
                reason = f"{first_error} / retry: {error}"
                notes.append(
                    f"{perspective} focus selection fell back to the heuristic planner "
                    f"(reason: {category_of(error)})."
                )
                record_fallback(
                    "perspective_units",
                    cluster_id=cluster.id,
                    category=category_of(error),
                    summary=f"{perspective} focus selection rejected after one retry",
                    detail=reason,
                )
                ranked = _heuristic_focus_types(cluster, perspective)
                return _finalise_focus_types(perspective, hints, ranked), notes
        if len(selected) < _MIN_FOCUS_TYPES:
            notes.append(f"{perspective} llm focus selection padded (1 valid → 2)")
            record_padding("perspective_focus_padded", cluster_id=cluster.id)
            ranked = _heuristic_focus_types(cluster, perspective)
            return _finalise_focus_types(perspective, selected, ranked), notes
        return _finalise_focus_types(perspective, hints, selected), notes
    else:
        notes.append(f"{perspective} focus selection used the heuristic planner (no LLM agent configured).")
    ranked = _heuristic_focus_types(cluster, perspective)
    return _finalise_focus_types(perspective, hints, ranked), notes


def select_focus_types(cluster: Cluster, perspective: str, *, hints: list[str], settings: Settings) -> list[str]:
    """Select the 2-4 focus types this perspective agent will investigate."""
    focus_types, _notes = _focus_types_with_notes(cluster, perspective, hints=hints, settings=settings)
    return focus_types


def _is_grounded(query: ToolQuery, cluster: Cluster) -> bool:
    # A query may only carry indicators the cluster actually observed; free-text
    # searches must still name a specific indicator or two cluster terms.
    iocs = filter_reportable_iocs(cluster.iocs)
    allowed = {
        "domain": {value.lower() for value in iocs.domains},
        "ip": set(iocs.ips),
        "url": set(iocs.urls),
        "cve": {value.upper() for value in iocs.cves},
        "hash": {value.lower() for value in iocs.hashes},
    }
    for name, values in allowed.items():
        raw = query.parameters.get(name)
        if raw is None:
            continue
        value = str(raw).strip()
        normalised = value.lower() if name in {"domain", "hash"} else value.upper() if name == "cve" else value
        if normalised not in values:
            return False
    search = str(query.parameters.get("query", "")).strip()
    if search and not _query_is_specific(search, cluster):
        return False
    return bool(query.parameters)


def _search_indicators(signals: _Signals) -> list[str]:
    indicators = [*signals.iocs.cves, *signals.iocs.domains, *signals.iocs.hashes]
    if indicators:
        return indicators[:_MAX_INDICATORS_PER_FOCUS]
    return signals.topic_terms[:1]


def _search_queries(cluster: Cluster, signals: _Signals) -> list[str]:
    # One search string per indicator: "<indicator> <topic term>", kept only
    # when it still names something specific about this cluster.
    searches: list[str] = []
    for indicator in _search_indicators(signals):
        terms = [term for term in signals.topic_terms if term != indicator.lower()]
        candidate = f"{indicator} {terms[0]}".strip() if terms else indicator
        if candidate and candidate not in searches and _query_is_specific(candidate, cluster):
            searches.append(candidate)
    return searches


def _tool_parameter_sets(tool_type: str, cluster: Cluster, signals: _Signals) -> list[dict[str, str]]:
    # Up to two verified indicators per tool, so clusters carrying several
    # domains, URLs, or CVEs are investigated beyond their first indicator.
    iocs = signals.iocs
    limit = _MAX_INDICATORS_PER_FOCUS
    if tool_type in {"WHOIS_LOOKUP", "PASSIVE_DNS_LOOKUP"}:
        candidates = [{"domain": value} for value in iocs.domains] + [{"ip": value} for value in iocs.ips]
        return candidates[:limit]
    if tool_type == "TLS_CT_LOOKUP":
        return [{"domain": value} for value in iocs.domains[:limit]]
    if tool_type == "WEB_CRAWL":
        return [{"url": value} for value in iocs.urls[:limit]]
    if tool_type == "NVD_CVE_LOOKUP":
        return [{"cve": value} for value in iocs.cves[:limit]]
    if tool_type == "GITHUB_SEARCH":
        values = iocs.cves[:limit] or iocs.hashes[:limit]
        return [{"query": value} for value in values]
    if tool_type in _SEARCH_TOOLS:
        return [{"query": search} for search in _search_queries(cluster, signals)]
    return []


def _heuristic_queries(cluster: Cluster, perspective: str, focus_types: list[str]) -> list[ToolQuery]:
    signals = _Signals(cluster)
    queries: list[ToolQuery] = []
    for index, focus in enumerate(focus_types):
        tools = FOCUS_TOOLS.get(focus, [])
        parameter_sets = {tool_type: _tool_parameter_sets(tool_type, cluster, signals) for tool_type in tools}
        produced = 0
        # Cover every tool of the focus with its first indicator before using a
        # second indicator, so a two-tool focus never spends its budget on one.
        for indicator_index in range(_MAX_INDICATORS_PER_FOCUS):
            for tool_type in tools:
                if produced >= _MAX_QUERIES_PER_FOCUS:
                    break
                candidates = parameter_sets[tool_type]
                if indicator_index >= len(candidates):
                    continue
                queries.append(
                    ToolQuery(
                        perspective=perspective,
                        tool_type=tool_type,
                        parameters=candidates[indicator_index],
                        priority=max(1, 9 - index),
                        focus_type=focus,
                    )
                )
                produced += 1
    return queries


def _dedupe(queries: list[ToolQuery]) -> list[ToolQuery]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[ToolQuery] = []
    for query in queries:
        key = (query.perspective, query.tool_type, str(sorted(query.parameters.items())))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(query)
    return deduped


def _planned_for_perspective(
    planned: list[ToolQuery],
    perspective: str,
    cluster: Cluster,
    focus_types: list[str],
) -> list[ToolQuery]:
    selected: list[ToolQuery] = []
    for query in planned:
        if query.perspective != perspective or not _is_grounded(query, cluster):
            continue
        focus = query.focus_type if query.focus_type in focus_types else ""
        selected.append(
            replace(query, focus_type=focus or _focus_for_tool(query.tool_type, perspective, focus_types))
        )
    return selected


def _perspective_queries_with_notes(
    cluster: Cluster,
    perspective: str,
    focus_types: list[str],
    *,
    planned: list[ToolQuery],
    settings: Settings,
) -> tuple[list[ToolQuery], list[str]]:
    notes: list[str] = []
    generated: list[ToolQuery] = []
    if settings.llm_configured() and not settings.allow_heuristic_reports:
        # One perspective unit: retried once, then this perspective (only)
        # falls back to the verified-IoC heuristic query builder.
        try:
            generated = generate_perspective_queries_with_llm(settings, cluster, perspective, focus_types)
        except LLMGenerationError as first_error:
            try:
                generated = generate_perspective_queries_with_llm(settings, cluster, perspective, focus_types)
            except LLMGenerationError as error:
                # Same contract as the focus-selection note above: the
                # published insight is code-authored, the failure text stays
                # in `detail` (DEBUG only).
                reason = f"{first_error} / retry: {error}"
                notes.append(
                    f"{perspective} query generation fell back to the heuristic builder "
                    f"(reason: {category_of(error)})."
                )
                record_fallback(
                    "perspective_units",
                    cluster_id=cluster.id,
                    category=category_of(error),
                    summary=f"{perspective} query generation rejected after one retry",
                    detail=reason,
                )

    def tagged(queries: list[ToolQuery]) -> list[ToolQuery]:
        return [
            replace(
                query,
                focus_type=(query.focus_type if query.focus_type in focus_types else "")
                or _focus_for_tool(query.tool_type, perspective, focus_types),
            )
            for query in queries
            if _is_grounded(query, cluster)
        ]

    grounded = tagged(generated)
    dropped = len(generated) - len(grounded)
    if dropped:
        notes.append(f"{perspective} llm queries dropped: {dropped} ungrounded")
    if not grounded:
        # Grounding can reject every generated query; the perspective still
        # investigates, using the verified-IoC heuristic builder.
        grounded = tagged(_heuristic_queries(cluster, perspective, focus_types))
    queries = _dedupe([*_planned_for_perspective(planned, perspective, cluster, focus_types), *grounded])
    return queries[: max(1, settings.max_queries_per_perspective)], notes


def build_perspective_queries(
    cluster: Cluster,
    perspective: str,
    focus_types: list[str],
    *,
    planned: list[ToolQuery],
    settings: Settings,
) -> list[ToolQuery]:
    """Build the executable tool queries for one perspective agent."""
    queries, _notes = _perspective_queries_with_notes(
        cluster,
        perspective,
        focus_types,
        planned=planned,
        settings=settings,
    )
    return queries


def _new_infrastructure(results: list[ToolResult], cluster: Cluster) -> list[tuple[str, str]]:
    known = {value.lower() for value in cluster.iocs.domains} | set(cluster.iocs.ips)
    discovered: list[tuple[str, str]] = []
    for result in results:
        if result.status != "success":
            continue
        reportable = filter_reportable_iocs(
            IocSet(domains=list(result.discovered_iocs.domains), ips=list(result.discovered_iocs.ips))
        )
        for value in sorted(reportable.domains):
            if value.lower() not in known and (value, "domain") not in discovered:
                discovered.append((value, "domain"))
        for value in sorted(reportable.ips):
            if value not in known and (value, "ip") not in discovered:
                discovered.append((value, "ip"))
    return discovered


def pivot_queries(perspective: str, results: list[ToolResult], cluster: Cluster, *, depth: int) -> list[ToolQuery]:
    """One-level infrastructure pivot from indicators the tools just discovered."""
    if depth < 1 or perspective not in PIVOT_PERSPECTIVES:
        return []
    queries: list[ToolQuery] = []
    for value, value_type in _new_infrastructure(results, cluster):
        for tool_type in _PIVOT_TOOLS_BY_TYPE[value_type]:
            if len(queries) >= _MAX_PIVOT_QUERIES:
                return queries
            queries.append(
                ToolQuery(
                    perspective=perspective,
                    tool_type=tool_type,
                    parameters={value_type: value, "pivot": True},
                    priority=4,
                    focus_type=PIVOT_FOCUS,
                )
            )
    return queries


def _perspective_confidence(results: list[ToolResult]) -> float:
    if not results:
        return 0.30
    successes = sum(1 for result in results if result.status == "success")
    return round(min(0.90, 0.45 + 0.10 * successes), 2)


def run_perspective(
    cluster: Cluster,
    perspective: str,
    focus_types: list[str],
    queries: list[ToolQuery],
    *,
    runner: ToolRunner,
    pivot_depth: int,
    mode: str,
    notes: list[str] | None = None,
) -> PerspectiveFinding:
    """Execute one perspective's queries, pivot once, and report the evidence."""
    executed = list(queries)
    results = runner.execute_many(executed) if executed else []
    pivots = pivot_queries(perspective, results, cluster, depth=pivot_depth)
    if pivots:
        results = [*results, *runner.execute_many(pivots)]
        executed = [*executed, *pivots]
    successes = sum(1 for result in results if result.status == "success")
    insights = [
        *(notes or []),
        f"{perspective} agent investigated {', '.join(focus_types) or 'no focus type'} with "
        f"{len(executed)} tool query/queries ({successes} successful).",
    ]
    if pivots:
        insights.append(f"{perspective} agent pivoted one level into {len(pivots)} newly discovered indicator query/queries.")
    return PerspectiveFinding(
        perspective=perspective,
        focus_types=focus_types,
        queries=executed,
        tool_results=results,
        insights=insights,
        confidence=_perspective_confidence(results),
        mode=mode,
    )


def _failed_finding(perspective: str, mode: str, error: Exception) -> PerspectiveFinding:
    # `insights` is published in the report body, so it names the exception
    # type this code caught and nothing the model or a collected post wrote:
    # an escaping LLMGenerationError carries the model's draft in its message.
    logger.debug("%s perspective agent failed for this cluster: %s", perspective, error)
    return PerspectiveFinding(
        perspective=perspective,
        focus_types=[],
        queries=[],
        tool_results=[],
        insights=[f"perspective failed: {perspective} agent raised {type(error).__name__}"],
        confidence=0.0,
        mode=mode,
    )


def investigate_cluster(
    cluster: Cluster,
    *,
    settings: Settings,
    runner: ToolRunner,
    strategy: InvestigationStrategy,
    planned_queries: list[ToolQuery],
) -> list[PerspectiveFinding]:
    """Run the four perspective agents in parallel and return their findings.

    Results are always collected in contract perspective order, and a failing
    agent yields a finding that records the failure instead of aborting the
    cluster investigation.
    """
    pivot_depth = settings.pivot_depth if settings.enable_pivot_iocs else 0

    def investigate(perspective: str) -> PerspectiveFinding:
        hints = list(strategy.focus_by_perspective.get(perspective, []))
        focus_types, focus_notes = _focus_types_with_notes(
            cluster,
            perspective,
            hints=hints,
            settings=settings,
        )
        queries, query_notes = _perspective_queries_with_notes(
            cluster,
            perspective,
            focus_types,
            planned=planned_queries or [],
            settings=settings,
        )
        return run_perspective(
            cluster,
            perspective,
            focus_types,
            queries,
            runner=runner,
            pivot_depth=pivot_depth,
            mode=strategy.mode,
            notes=[*focus_notes, *query_notes],
        )

    with ThreadPoolExecutor(max_workers=max(1, settings.perspective_workers)) as pool:
        futures = {perspective: pool.submit(investigate, perspective) for perspective in REQUIRED_PERSPECTIVES}

    findings: list[PerspectiveFinding] = []
    for perspective in REQUIRED_PERSPECTIVES:
        try:
            findings.append(futures[perspective].result())
        except Exception as error:  # noqa: BLE001 - one agent must never abort the cluster
            findings.append(_failed_finding(perspective, strategy.mode, error))
    return findings
