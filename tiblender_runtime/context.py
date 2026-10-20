"""Step 3 context aggregation.

Perspective findings are merged into one :class:`InvestigationContext`: every
indicator is labelled ``Direct`` or ``Pivot``, scored with the three paper
confidence rules (independent sources, shared hosting, contradiction), benign
indicators are dropped, and the per-step trajectory is recorded for the RL
feedback layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .iocs import filter_reportable_iocs, refang
from .legitimate_hosts import is_suppressed_host
from .models import Cluster, IocSet, PerspectiveFinding, ToolResult
from .pivots import (
    keeps_discovered_values,
    max_resolutions_per_ip as resolve_max_resolutions,
    queried_value as _queried_value,
    shared_hosting_queries,
)

DIRECT = "Direct"
PIVOT = "Pivot"
# Paper §III, rule 2: passive DNS or CT showing the queried indicator sharing a
# hosting IP or certificate with a *known malicious* domain raises it one tier.
SHARED_HOSTING_LINK = "shared_hosting_link:+1"
# Not that rule. A resolution set larger than the cap only says the address is
# crowded, so its co-tenants were dropped. Recorded as evidence, never scored:
# the deployed system treats a crowded address as a false-positive risk
# (`ioc_confidence._evaluate_pivot_ioc` -> LOW), never as corroboration.
SHARED_HOSTING_ENVIRONMENT = "shared_hosting_environment:co_tenants_dropped"
CONTRADICTION = "contradiction:-1"

_TIERS = ("LOW", "MEDIUM", "HIGH")
_TTP_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_SHARED_HOSTING_TOOLS = frozenset({"PASSIVE_DNS_LOOKUP", "TLS_CT_LOOKUP"})
_CONTRADICTION_TOOLS = frozenset({"WHOIS_LOOKUP", "PASSIVE_DNS_LOOKUP", "TLS_CT_LOOKUP", "WEB_CRAWL"})
_CONTRADICTION_MARKERS = ("404", "nxdomain", "name resolution", "not found", "no such host", "no data")
_FIELD_BY_TYPE = {
    "url": "urls",
    "domain": "domains",
    "ip": "ips",
    "hash": "hashes",
    "cve": "cves",
}
_TYPE_BY_FIELD = {field_name: ioc_type for ioc_type, field_name in _FIELD_BY_TYPE.items()}
# Ordered like ``rl.replay``: CVEs are counted separately from other indicators.
_VALUE_FIELDS = ("urls", "domains", "ips", "hashes")


@dataclass(slots=True)
class IocEvidence:
    value: str
    ioc_type: str
    extraction_type: str
    confidence: str
    sources: list[str] = field(default_factory=list)
    perspectives: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InvestigationContext:
    cluster_id: str
    iocs: list[IocEvidence] = field(default_factory=list)
    cves: list[str] = field(default_factory=list)
    ttps: list[str] = field(default_factory=list)
    per_perspective: dict[str, dict] = field(default_factory=dict)
    trajectory: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    filtered_benign: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _Candidate:
    """Mutable accumulator turned into an immutable :class:`IocEvidence`."""

    value: str
    ioc_type: str
    extraction_type: str
    sources: set[str] = field(default_factory=set)
    perspectives: set[str] = field(default_factory=set)
    tools: set[str] = field(default_factory=set)
    adjustments: set[str] = field(default_factory=set)


def _normalise(value: str, ioc_type: str) -> str:
    text = str(value).strip()
    if ioc_type in {"domain", "hash"}:
        return text.lower().strip(".")
    if ioc_type == "cve":
        return text.upper()
    return text


def _typed_values(iocs: IocSet) -> list[tuple[str, str]]:
    """Return ``(value, ioc_type)`` pairs in a stable field-then-value order."""
    pairs: list[tuple[str, str]] = []
    for field_name in ("urls", "domains", "ips", "hashes", "cves"):
        ioc_type = _TYPE_BY_FIELD[field_name]
        for value in sorted(getattr(iocs, field_name)):
            normalised = _normalise(value, ioc_type)
            if normalised and (normalised, ioc_type) not in pairs:
                pairs.append((normalised, ioc_type))
    return pairs


def _is_reportable(value: str, ioc_type: str) -> bool:
    # A value is benign when the report-grade filter rejects it, or when it is a
    # known-legitimate platform, vendor, or reference domain.
    if ioc_type == "domain" and is_suppressed_host(value):
        return False
    field_name = _FIELD_BY_TYPE[ioc_type]
    probe = filter_reportable_iocs(IocSet(**{field_name: [value]}))
    return value in getattr(probe, field_name)


def _value_pattern(value: str) -> re.Pattern[str]:
    # Token-boundary match so "notevil.example" or "1.2.3.40" never counts as a
    # mention of "evil.example" or "1.2.3.4".
    return re.compile(rf"(?<![A-Za-z0-9.\-]){re.escape(value)}(?![A-Za-z0-9\-])", re.IGNORECASE)


def _post_sources(cluster: Cluster, value: str) -> list[str]:
    if not value:
        return []
    pattern = _value_pattern(value)
    sources: list[str] = []
    for post in cluster.posts:
        haystack = refang(f"{post.post.content}\n{post.normalized_content}")
        if pattern.search(haystack):
            source = f"post:{post.id}"
            if source not in sources:
                sources.append(source)
    return sources


def _successful(findings: list[PerspectiveFinding]) -> Iterable[tuple[str, ToolResult]]:
    for finding in findings:
        for result in finding.tool_results:
            if result.status == "success":
                yield finding.perspective, result


def _collect_candidates(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    *,
    resolution_limit: int,
) -> tuple[dict[str, _Candidate], list[str]]:
    # The report-grade filter is applied *before* the Direct set is built, so
    # the scored evidence list is a superset of what a report may publish as
    # Direct: `reporting.build_report` derives its flat `iocs` from this
    # structure, and a post-text indicator must never be missing from it.
    benign = {
        value
        for value, ioc_type in _typed_values(cluster.iocs)
        if not _is_reportable(value, ioc_type)
    }
    direct = _typed_values(filter_reportable_iocs(cluster.iocs))
    direct_values = {value for value, _ in direct}
    candidates: dict[str, _Candidate] = {}

    for value, ioc_type in direct:
        candidate = _Candidate(value=value, ioc_type=ioc_type, extraction_type=DIRECT)
        candidate.sources.update(_post_sources(cluster, value))
        candidates[value] = candidate

    for perspective, result in _successful(findings):
        queried = _queried_value(result)
        # A resolution set larger than the cap describes shared hosting: the
        # names in it are co-tenants, not this campaign's infrastructure. Only
        # values the posts already named survive, as corroboration of an
        # indicator we had independent reason to believe.
        keep_discovered = keeps_discovered_values(result, resolution_limit)
        for value, ioc_type in _typed_values(result.discovered_iocs):
            if not keep_discovered and value not in direct_values:
                continue
            if not _is_reportable(value, ioc_type):
                benign.add(value)
                continue
            candidate = candidates.get(value)
            if candidate is None:
                candidate = _Candidate(
                    value=value,
                    ioc_type=ioc_type,
                    extraction_type=DIRECT if value in direct_values else PIVOT,
                )
                candidates[value] = candidate
            # One source per distinct (tool, queried indicator) answer: asking
            # the same tool the same question twice is not a second, and
            # independent report. This is what "independent sources" means for
            # a tool answer here, and it matches the deployed scorer, which
            # also counts an answer that merely names the queried indicator
            # (`ioc_confidence.evaluate_ioc_confidence`).
            candidate.sources.add(f"tool:{result.tool_type}:{queried}")
            candidate.perspectives.add(perspective)
            candidate.tools.add(result.tool_type)

    return candidates, sorted(benign)


def _apply_shared_hosting(
    candidates: dict[str, _Candidate],
    findings: list[PerspectiveFinding],
    *,
    resolution_limit: int,
) -> None:
    # Rule 2, as the paper words it: the +1 is earned only when passive DNS or
    # CT reveals a link to a *known malicious* domain sharing the same hosting
    # IP or certificate. The cluster's Direct indicators -- the values the
    # posts themselves named -- are that known-malicious set here.
    for _perspective, result in _successful(findings):
        if result.tool_type not in _SHARED_HOSTING_TOOLS:
            continue
        queried = _candidate_key(candidates, _queried_value(result))
        if queried is None or candidates[queried].extraction_type != DIRECT:
            continue
        for value, _ioc_type in _typed_values(result.discovered_iocs):
            other = _candidate_key(candidates, value)
            if other is None or other == queried or candidates[other].extraction_type != DIRECT:
                continue
            candidates[queried].adjustments.add(SHARED_HOSTING_LINK)
    # A crowded resolution set is *not* that rule. It marks a shared-hosting
    # environment, whose co-tenants _collect_candidates already dropped; the
    # marker records that they were dropped and carries no tier change.
    for shared in shared_hosting_queries(
        (result for _perspective, result in _successful(findings)), resolution_limit
    ):
        key = _candidate_key(candidates, shared)
        if key is not None:
            candidates[key].adjustments.add(SHARED_HOSTING_ENVIRONMENT)


def _is_contradiction(result: ToolResult) -> bool:
    # Only a disconfirming provider answer is evidence against an indicator: an
    # internal failure (timeout, parser crash, missing recording) says nothing
    # about the indicator itself and must never lower its confidence.
    if result.tool_type not in _CONTRADICTION_TOOLS:
        return False
    if result.status not in {"error", "unavailable"}:
        return False
    lowered = result.error.lower()
    return any(marker in lowered for marker in _CONTRADICTION_MARKERS)


def _apply_contradictions(candidates: dict[str, _Candidate], findings: list[PerspectiveFinding]) -> None:
    # Rule 3: a resolution, registration, or crawl failure for an indicator is
    # evidence against it, so its confidence drops one tier.
    for finding in findings:
        for result in finding.tool_results:
            if not _is_contradiction(result):
                continue
            key = _candidate_key(candidates, _queried_value(result))
            if key is not None:
                candidates[key].adjustments.add(CONTRADICTION)


def _candidate_key(candidates: dict[str, _Candidate], value: str) -> str | None:
    if not value:
        return None
    for normalised in (value, value.lower().strip("."), value.upper()):
        if normalised in candidates:
            return normalised
    return None


def _tier(candidate: _Candidate) -> str:
    # Rule 1 (paper §III): three or more independent sources = HIGH, two =
    # MEDIUM, one = LOW. Rule 2 adds a tier, rule 3 removes one; nothing else
    # moves an indicator, and no floor cancels a contradiction.
    count = len(candidate.sources)
    index = 2 if count >= 3 else 1 if count == 2 else 0
    if SHARED_HOSTING_LINK in candidate.adjustments:
        index += 1
    if CONTRADICTION in candidate.adjustments:
        index -= 1
    return _TIERS[max(0, min(len(_TIERS) - 1, index))]


def _evidence(candidates: dict[str, _Candidate]) -> list[IocEvidence]:
    evidence: list[IocEvidence] = []
    for candidate in sorted(candidates.values(), key=lambda item: (item.ioc_type, item.value)):
        confidence = _tier(candidate)
        if candidate.extraction_type == PIVOT and confidence == "LOW":
            # A pivot resting on one tool answer is an unconfirmed
            # infrastructure relationship and never reaches the report.
            continue
        evidence.append(
            IocEvidence(
                value=candidate.value,
                ioc_type=candidate.ioc_type,
                extraction_type=candidate.extraction_type,
                confidence=confidence,
                sources=sorted(candidate.sources),
                perspectives=sorted(candidate.perspectives),
                tools=sorted(candidate.tools),
                adjustments=sorted(candidate.adjustments),
            )
        )
    return evidence


def _ttps(cluster: Cluster, findings: list[PerspectiveFinding]) -> list[str]:
    texts = [post.normalized_content for post in cluster.posts]
    texts.extend(result.summary for finding in findings for result in finding.tool_results)
    found: set[str] = set()
    for text in texts:
        found.update(_TTP_RE.findall(text or ""))
    return sorted(found)


def _per_perspective(findings: list[PerspectiveFinding], benign: set[str]) -> dict[str, dict]:
    buckets: dict[str, dict] = {}
    for finding in findings:
        bucket = buckets.setdefault(
            finding.perspective,
            {
                "focus_types": [],
                "queries": 0,
                "success": 0,
                "unavailable": 0,
                "error": 0,
                "discovered": [],
                "diagnostics": [],
            },
        )
        for focus in finding.focus_types:
            if focus not in bucket["focus_types"]:
                bucket["focus_types"].append(focus)
        bucket["queries"] += len(finding.queries)
        if len(finding.queries) != len(finding.tool_results):
            # Queries and results are paired positionally; a mismatch means one
            # side was truncated and the trajectory below is incomplete.
            bucket["diagnostics"].append(
                f"{len(finding.queries)} query/queries paired with {len(finding.tool_results)} result(s)"
            )
        discovered: set[str] = set(bucket["discovered"])
        for result in finding.tool_results:
            if result.status in {"success", "unavailable", "error"}:
                bucket[result.status] += 1
            if result.status != "success":
                continue
            discovered.update(
                value for value, _ioc_type in _typed_values(result.discovered_iocs) if value not in benign
            )
        bucket["discovered"] = sorted(discovered)
    return buckets


def _stats(evidence: list[IocEvidence], findings: list[PerspectiveFinding], per_perspective: dict[str, dict]) -> dict:
    results = [result for finding in findings for result in finding.tool_results]
    successes = sum(1 for result in results if result.status == "success")
    return {
        "ioc_total": len(evidence),
        "ioc_high": sum(1 for item in evidence if item.confidence == "HIGH"),
        "ioc_medium": sum(1 for item in evidence if item.confidence == "MEDIUM"),
        "ioc_low": sum(1 for item in evidence if item.confidence == "LOW"),
        "pivot_total": sum(1 for item in evidence if item.extraction_type == PIVOT),
        "direct_total": sum(1 for item in evidence if item.extraction_type == DIRECT),
        "tool_success_rate": round(successes / len(results), 3) if results else 0.0,
        "perspectives_active": sum(1 for bucket in per_perspective.values() if bucket["success"]),
    }


def _signature(query_tool: str, parameters: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    stable = tuple(sorted((str(key), str(value)) for key, value in parameters.items() if key != "pivot"))
    return query_tool, stable


def _trajectory(cluster: Cluster, findings: list[PerspectiveFinding], benign: set[str]) -> list[dict]:
    known_values = {value for value, _ioc_type in _typed_values(cluster.iocs)}
    known_cves = {value.upper() for value in cluster.iocs.cves}
    seen_ttps = {ttp for post in cluster.posts for ttp in _TTP_RE.findall(post.normalized_content or "")}
    seen_signatures: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    records: list[dict] = []

    for finding in findings:
        for query, result in zip(finding.queries, finding.tool_results, strict=False):
            signature = _signature(query.tool_type, query.parameters)
            duplicate = signature in seen_signatures
            seen_signatures.add(signature)

            discovered = result.discovered_iocs
            new_values = {
                _normalise(value, _TYPE_BY_FIELD[field_name])
                for field_name in _VALUE_FIELDS
                for value in getattr(discovered, field_name)
            } - known_values - benign
            new_cves = {value.upper() for value in discovered.cves} - known_cves
            new_ttps = set(_TTP_RE.findall(result.summary or "")) - seen_ttps
            known_values |= new_values
            known_cves |= new_cves
            seen_ttps |= new_ttps

            records.append(
                {
                    "step": len(records) + 1,
                    "perspective": finding.perspective,
                    "focus_type": query.focus_type or (finding.focus_types[0] if finding.focus_types else ""),
                    "tool_type": query.tool_type,
                    "parameters": dict(query.parameters),
                    "status": result.status,
                    "new_iocs": len(new_values),
                    "new_cves": len(new_cves),
                    "new_ttps": len(new_ttps),
                    "duplicate": duplicate,
                    "confidence": result.confidence.value,
                    "mode": finding.mode,
                }
            )
    return records


def _resolution_limit(override: int | None) -> int:
    return resolve_max_resolutions(override)


def aggregate_context(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    *,
    max_resolutions_per_ip: int | None = None,
) -> InvestigationContext:
    """Merge perspective findings into the evidence context the judges consume."""
    resolution_limit = _resolution_limit(max_resolutions_per_ip)
    candidates, filtered_benign = _collect_candidates(cluster, findings, resolution_limit=resolution_limit)
    _apply_shared_hosting(candidates, findings, resolution_limit=resolution_limit)
    _apply_contradictions(candidates, findings)
    evidence = _evidence(candidates)
    benign = set(filtered_benign)
    per_perspective = _per_perspective(findings, benign)
    return InvestigationContext(
        cluster_id=cluster.id,
        iocs=evidence,
        cves=[item.value for item in evidence if item.ioc_type == "cve"],
        ttps=_ttps(cluster, findings),
        per_perspective=per_perspective,
        trajectory=_trajectory(cluster, findings, benign),
        stats=_stats(evidence, findings, per_perspective),
        filtered_benign=filtered_benign,
    )


def context_to_judge_payload(context: InvestigationContext) -> dict:
    """Bounded, evidence-only view of the context for the Step 4 judge prompt."""
    return {
        "iocs": [
            {
                "value": item.value,
                "type": item.ioc_type,
                "extraction_type": item.extraction_type,
                "confidence": item.confidence,
                "source_count": len(item.sources),
                "perspectives": list(item.perspectives),
            }
            for item in context.iocs
        ],
        "cves": list(context.cves),
        "ttps": list(context.ttps),
        "per_perspective": {
            perspective: {
                key: list(value) if isinstance(value, list) else value
                for key, value in bucket.items()
            }
            for perspective, bucket in context.per_perspective.items()
        },
        "stats": dict(context.stats),
        "filtered_benign": list(context.filtered_benign),
    }
