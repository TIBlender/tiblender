"""Keyword catalog, static/dynamic query loading, and query-plan assembly."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .base import CollectionQuery, QueryPlan

logger = logging.getLogger(__name__)

_MIN_DYNAMIC_PRIORITY = 1
_MAX_DYNAMIC_PRIORITY = 10

# Dynamic (trend-adapted) queries drive keyword *search* only, and only X
# and Reddit are collected by keyword search. Telegram and Discord are
# collected exhaustively from the communities the account has joined, so a
# dynamic query naming them has nothing to drive in this runtime.
DYNAMIC_QUERY_PLATFORMS: tuple[str, ...] = ("x", "reddit")


@dataclass(frozen=True, slots=True)
class KeywordTerm:
    term: str
    language: str
    category: str


@dataclass(slots=True)
class KeywordCatalog:
    categories: list[str]
    languages: list[str]
    terms: list[KeywordTerm]
    platform_queries: dict[str, list[CollectionQuery]]

    @classmethod
    def load(cls, path: Path) -> "KeywordCatalog":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        categories = [entry["id"] for entry in payload["categories"]]
        terms_section = payload.get("terms", {})
        languages_seen: set[str] = set()
        terms: list[KeywordTerm] = []
        for category in categories:
            for language, term_list in terms_section.get(category, {}).items():
                languages_seen.add(language)
                for term in term_list:
                    terms.append(KeywordTerm(term=term, language=language, category=category))
        platform_queries: dict[str, list[CollectionQuery]] = {}
        for platform, entries in payload.get("platform_queries", {}).items():
            platform_queries[platform] = [
                CollectionQuery(
                    platform=platform,
                    query=entry["query"],
                    source="static",
                    category=entry.get("category", ""),
                    language=entry.get("language", "en"),
                )
                for entry in entries
            ]
        return cls(
            categories=categories,
            languages=sorted(languages_seen),
            terms=terms,
            platform_queries=platform_queries,
        )

    def security_terms(self) -> list[str]:
        return sorted({term.term.lower() for term in self.terms})

    def terms_for(self, category: str, language: str) -> list[str]:
        return [term.term for term in self.terms if term.category == category and term.language == language]


def load_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        lines.append(stripped)
    return lines


class _Unparsable:
    """Sentinel distinguishing 'no expiry given' from 'expiry given but invalid'."""


_UNPARSABLE = _Unparsable()


def _parse_expiry(value: object) -> datetime | None | _Unparsable:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return _UNPARSABLE


def _coerce_priority(value: object) -> int | None:
    # A missing/invalid priority must not crash the whole batch (it would
    # otherwise blow up later as `-q.priority` in the sort key), so a
    # non-coercible value is treated as a skip, not a hard failure.
    try:
        priority = int(value)
    except (TypeError, ValueError):
        return None
    return max(_MIN_DYNAMIC_PRIORITY, min(_MAX_DYNAMIC_PRIORITY, priority))


def load_dynamic_queries(path: Path, *, now: datetime, max_per_platform: int) -> list[CollectionQuery]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"malformed dynamic queries file (expected a JSON array): {path}")

    expanded: list[CollectionQuery] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            logger.warning("dynamic query record %d skipped: not a JSON object (%s)", index, path)
            continue
        query = record.get("query")
        platforms = record.get("platforms")
        if not query or not isinstance(platforms, list) or not platforms:
            logger.warning(
                "dynamic query record %d skipped: missing/invalid 'query' or 'platforms' (%s)", index, path
            )
            continue
        expires_at = _parse_expiry(record.get("expires_at"))
        if expires_at is _UNPARSABLE:
            logger.warning("dynamic query record %d skipped: unparsable 'expires_at' (%s)", index, path)
            continue
        if expires_at is not None and expires_at < now:
            continue
        priority = _coerce_priority(record.get("priority", 5))
        if priority is None:
            logger.warning("dynamic query record %d skipped: non-numeric 'priority' (%s)", index, path)
            continue
        language = record.get("language", "en")
        trend_types = tuple(record.get("trend_types", []))
        source_report_id = record.get("source_report_id", "")
        for platform in platforms:
            if platform not in DYNAMIC_QUERY_PLATFORMS:
                logger.debug(
                    "dynamic query record %d: platform %r ignored (dynamic queries target %s only)",
                    index,
                    platform,
                    ", ".join(DYNAMIC_QUERY_PLATFORMS),
                )
                continue
            expanded.append(
                CollectionQuery(
                    platform=platform,
                    query=query,
                    source="dynamic",
                    language=language,
                    priority=priority,
                    trend_types=trend_types,
                    source_report_id=source_report_id,
                )
            )

    expanded.sort(key=lambda q: (-q.priority, q.query, q.platform))

    capped: list[CollectionQuery] = []
    counts: dict[str, int] = {}
    for query_obj in expanded:
        used = counts.get(query_obj.platform, 0)
        if used >= max_per_platform:
            continue
        counts[query_obj.platform] = used + 1
        capped.append(query_obj)
    return capped


def build_query_plan(
    catalog: KeywordCatalog,
    *,
    subreddits: list[str],
    dynamic: list[CollectionQuery],
) -> QueryPlan:
    """Assemble the keyword-search plan for the query-driven platforms.

    Only X and Reddit are query driven; Telegram and Discord collect
    exhaustively from joined communities and ignore the plan entirely.
    """
    queries: list[CollectionQuery] = []
    for platform_queries in catalog.platform_queries.values():
        queries.extend(platform_queries)
    for subreddit in subreddits:
        queries.append(CollectionQuery(platform="reddit", query=subreddit, source="subreddit"))
    queries.extend(dynamic)
    return QueryPlan(queries=queries)
