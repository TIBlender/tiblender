"""Dynamic, trend-adapted collection query generation for the next cycle."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .collectors.keywords import KeywordCatalog
from .llm import LLMGenerationError, _query_is_specific
from .llm_fallbacks import category_of, record_fallback
from .models import Cluster, ThreatCategory
from .rl.policy import brand_tokens
from .settings import Settings
from .trends import ClusterSnapshot, TrendSignal

logger = logging.getLogger(__name__)

# Dynamic queries drive keyword search, and only X and Reddit are collected
# by keyword search. Telegram and Discord are collected exhaustively from
# the communities the account has joined, so they are never query targets
# here. (In the deployed system the same dynamic keywords also feed the
# Telegram/Discord channel-discovery collectors, which live outside this
# repository -- see docs/FIDELITY.md.)
_PLATFORMS = ("x", "reddit")
_LOCALIZED_PLATFORMS = ("x",)
_MAX_DYNAMIC_QUERIES = 200

# Maps the runtime's ThreatCategory values onto the collector keyword
# catalog's four category ids (config/keywords.json).
_CATEGORY_TO_CATALOG_ID = {
    ThreatCategory.PHISHING: "phishing_fraud",
    ThreatCategory.FRAUD: "phishing_fraud",
    ThreatCategory.MALWARE: "malware_c2",
    ThreatCategory.ABUSE_INFRASTRUCTURE: "malware_c2",
    ThreatCategory.SUPPLY_CHAIN: "malware_c2",
    ThreatCategory.VULNERABILITY: "vulnerability",
    ThreatCategory.APT: "apt_ransomware",
    ThreatCategory.RANSOMWARE: "apt_ransomware",
    ThreatCategory.OTHER: "malware_c2",
}


@dataclass(slots=True)
class DynamicQuery:
    query: str
    platforms: list[str]
    trend_types: list[str]
    priority: int
    language: str
    source_report_id: str
    source_cluster_id: str
    generated_at: str
    expires_at: str
    rationale: str


def _priority_for(trend_types: list[str]) -> int:
    bonus = 1 if "EARLY_WARNING" in trend_types else 0
    return min(10, 6 + len(trend_types) + bonus)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _seeds(cluster: Cluster) -> list[tuple[str, str]]:
    """Evidence-grounded seed terms: CVEs, queryable domains, then brand tokens."""
    from .feedback import _queryable_domains

    seeds: list[tuple[str, str]] = []
    for cve in cluster.iocs.cves[:2]:
        seeds.append((cve, "CVE"))
    for domain in _queryable_domains(cluster.iocs.domains)[:2]:
        seeds.append((domain, "domain"))
    for brand in brand_tokens(cluster.topic_name)[:2]:
        seeds.append((brand, "brand/malware token"))
    return seeds


def heuristic_dynamic_queries(
    signal: TrendSignal,
    snapshot: ClusterSnapshot,
    cluster: Cluster,
    *,
    catalog: KeywordCatalog,
    now: datetime,
    ttl_hours: int,
    report_id: str,
) -> list[DynamicQuery]:
    """Deterministic fallback: one query per seed for X and Reddit in English,
    plus one localized X variant per non-English language observed on the
    cluster's posts."""
    seeds = _seeds(cluster)
    if not seeds:
        return []

    priority = _priority_for(signal.trend_types)
    generated_at = _iso(now)
    expires_at = _iso(now + timedelta(hours=ttl_hours))
    category_id = _CATEGORY_TO_CATALOG_ID.get(cluster.threat_category, "malware_c2")
    trend_label = ", ".join(signal.trend_types)

    queries: list[DynamicQuery] = []
    for seed, source_label in seeds:
        queries.append(
            DynamicQuery(
                query=seed,
                platforms=list(_PLATFORMS),
                trend_types=list(signal.trend_types),
                priority=priority,
                language="en",
                source_report_id=report_id,
                source_cluster_id=cluster.id,
                generated_at=generated_at,
                expires_at=expires_at,
                rationale=f"{trend_label} trend detected on cluster {cluster.id}; seeded from {source_label} '{seed}'.",
            )
        )
        for language in snapshot.languages:
            if language == "en":
                continue
            terms = catalog.terms_for(category_id, language)
            if not terms:
                continue
            localized_term = terms[0]
            queries.append(
                DynamicQuery(
                    query=f"{seed} {localized_term}",
                    platforms=list(_LOCALIZED_PLATFORMS),
                    trend_types=list(signal.trend_types),
                    priority=priority,
                    language=language,
                    source_report_id=report_id,
                    source_cluster_id=cluster.id,
                    generated_at=generated_at,
                    expires_at=expires_at,
                    rationale=(
                        f"{trend_label} trend detected on cluster {cluster.id}; localized ({language}) variant "
                        f"seeded from {source_label} '{seed}' plus catalog term '{localized_term}'."
                    ),
                )
            )
    return queries


def _trend_payload(signal: TrendSignal, snapshot: ClusterSnapshot, cluster: Cluster) -> dict[str, Any]:
    from .feedback import _queryable_domains

    return {
        "trend_types": list(signal.trend_types),
        "evidence": signal.evidence,
        "cluster": {
            "id": cluster.id,
            "topic_name": cluster.topic_name,
            "description": cluster.description,
            "category": cluster.threat_category.value,
            "severity": cluster.severity.value,
            "cves": sorted(set(cluster.iocs.cves)),
            "domains": _queryable_domains(cluster.iocs.domains),
        },
        "platforms": snapshot.platforms,
        "languages": snapshot.languages,
    }


def _validated_llm_queries(
    raw_queries: list[Any],
    signal: TrendSignal,
    cluster: Cluster,
    *,
    catalog: KeywordCatalog,
    now: datetime,
    ttl_hours: int,
    report_id: str,
) -> list[DynamicQuery]:
    allowed_platforms = set(_PLATFORMS)
    allowed_languages = set(catalog.languages)
    priority = _priority_for(signal.trend_types)
    generated_at = _iso(now)
    expires_at = _iso(now + timedelta(hours=ttl_hours))
    default_rationale = f"{', '.join(signal.trend_types)} trend detected on cluster {cluster.id}."

    valid: list[DynamicQuery] = []
    for raw in raw_queries:
        if not isinstance(raw, dict):
            continue
        query_text = str(raw.get("query", "")).strip()
        platforms_raw = raw.get("platforms")
        if not query_text or not isinstance(platforms_raw, list) or not platforms_raw:
            continue
        # Narrow the draft's platform list to the ones this runtime actually
        # collects by keyword search rather than discarding the whole query:
        # a proposal of ["x", "telegram"] still carries a perfectly good X
        # query, and rejecting it outright would push the batch into the
        # heuristic fallback over a platform name the prompt already forbids.
        # A query left with no platform at all is still dropped.
        normalized_platforms = {str(item).strip().lower() for item in platforms_raw} & allowed_platforms
        if not normalized_platforms:
            continue
        # Fixed collector order (matches heuristic_dynamic_queries and
        # collectors.base), not alphabetical, so platform lists are stable
        # and consistent regardless of the LLM's own ordering.
        platforms = [platform for platform in _PLATFORMS if platform in normalized_platforms]
        language = str(raw.get("language", "en")).strip().lower() or "en"
        if language not in allowed_languages:
            continue
        if not _query_is_specific(query_text, cluster):
            continue
        rationale = str(raw.get("rationale", "")).strip() or default_rationale
        valid.append(
            DynamicQuery(
                query=query_text,
                platforms=platforms,
                trend_types=list(signal.trend_types),
                priority=priority,
                language=language,
                source_report_id=report_id,
                source_cluster_id=cluster.id,
                generated_at=generated_at,
                expires_at=expires_at,
                rationale=rationale,
            )
        )
    return valid


def _with_llm_fallback(signal: TrendSignal, marker: str) -> TrendSignal:
    # TrendSignal is frozen: record the fallback marker on a new instance
    # (a copy of evidence plus the marker) rather than mutating the
    # caller-owned signal in place.
    return replace(signal, evidence={**signal.evidence, "llm_fallback": marker})


def generate_dynamic_queries(
    signals: list[TrendSignal],
    snapshots_by_id: dict[str, ClusterSnapshot],
    clusters_by_id: dict[str, Cluster],
    reports_by_cluster: dict[str, str],
    *,
    settings: Settings,
    catalog: KeywordCatalog,
    now: datetime,
) -> tuple[list[DynamicQuery], list[TrendSignal]]:
    """Generate, per trend signal, an LLM-adapted or heuristic set of queries.

    Deduplicates by ``(query.lower(), tuple(platforms))`` keeping the highest
    priority entry, sorts by ``(-priority, query)``, and caps the result at
    :data:`_MAX_DYNAMIC_QUERIES`.

    Returns ``(queries, signals)`` where ``signals`` is the input list with
    any LLM-fallback signal replaced (never mutated) by a new
    :class:`TrendSignal` carrying a code-authored ``evidence["llm_fallback"]``
    marker: ``"llm_error:<category>"`` on an actual
    :class:`LLMGenerationError` (the category from the raise site, never the
    error message, which may embed the model's draft of scraped posts), or
    ``"no_valid_queries"`` when the LLM call succeeded but none of its
    proposed queries passed validation.
    """
    use_llm = settings.llm_configured() and not settings.allow_heuristic_reports
    generated: list[DynamicQuery] = []
    updated_signals: list[TrendSignal] = []

    for signal in signals:
        cluster = clusters_by_id.get(signal.cluster_id)
        snapshot = snapshots_by_id.get(signal.cluster_id)
        if cluster is None or snapshot is None:
            updated_signals.append(signal)
            continue
        report_id = reports_by_cluster.get(signal.cluster_id, "")
        ttl_hours = settings.trend_window_hours

        per_signal: list[DynamicQuery] = []
        current_signal = signal
        if use_llm:
            try:
                from .llm_feedback import generate_dynamic_queries_with_llm

                raw_queries = generate_dynamic_queries_with_llm(
                    settings, _trend_payload(signal, snapshot, cluster)
                )
                per_signal = _validated_llm_queries(
                    raw_queries,
                    signal,
                    cluster,
                    catalog=catalog,
                    now=now,
                    ttl_hours=ttl_hours,
                    report_id=report_id,
                )
                if not per_signal:
                    # The LLM responded, but nothing it proposed was
                    # grounded/allowed; distinguish this from "not
                    # configured" so a misbehaving LLM is visible.
                    current_signal = _with_llm_fallback(signal, "no_valid_queries")
                    record_fallback(
                        "dynamic_query_signals",
                        cluster_id=cluster.id,
                        category="invalid_response",
                        summary="no LLM-proposed query passed grounding validation",
                    )
            except LLMGenerationError as error:
                # The failure text can embed up to 500 characters of the
                # model's own draft (a summary of scraped posts), so it only
                # ever reaches DEBUG. The single content-free WARNING for this
                # fallback is the one record_fallback builds with
                # llm_fallbacks.format_fallback_line.
                logger.debug(
                    "LLM dynamic query generation failed for cluster %s; falling back to heuristic: %s",
                    cluster.id,
                    error,
                )
                current_signal = _with_llm_fallback(signal, f"llm_error:{category_of(error)}")
                record_fallback(
                    "dynamic_query_signals",
                    cluster_id=cluster.id,
                    category=category_of(error),
                    summary="dynamic query draft rejected after one retry",
                    detail=str(error),
                )

        if not per_signal:
            per_signal = heuristic_dynamic_queries(
                signal,
                snapshot,
                cluster,
                catalog=catalog,
                now=now,
                ttl_hours=ttl_hours,
                report_id=report_id,
            )
        generated.extend(per_signal)
        updated_signals.append(current_signal)

    deduped: dict[tuple[str, tuple[str, ...]], DynamicQuery] = {}
    for query in generated:
        key = (query.query.lower(), tuple(query.platforms))
        existing = deduped.get(key)
        if existing is None or query.priority > existing.priority:
            deduped[key] = query

    ordered = sorted(deduped.values(), key=lambda item: (-item.priority, item.query))
    return ordered[:_MAX_DYNAMIC_QUERIES], updated_signals


def write_dynamic_queries(queries: list[DynamicQuery], path: Path) -> None:
    """Write queries as a JSON array consumed by ``collectors.keywords.load_dynamic_queries``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(query) for query in queries], indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
