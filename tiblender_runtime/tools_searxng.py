"""SearXNG request building and result normalization for ``SEARCH_ENGINE``.

Split out of :mod:`tiblender_runtime.tools` so the adapter module stays within
the package's file-size budget. These helpers are pure functions over the
query string and the provider payload -- they issue no request themselves, so
they are also the part of the search path that is testable without a server.

The IoC rule they implement is the conservative one: a search snippet only
yields a ``Pivot`` indicator when the snippet both mentions the indicator the
search was about and reads as security evidence. Without that pair of gates a
SearXNG answer for a campaign name would promote every domain on the results
page to campaign infrastructure.
"""

from __future__ import annotations

import re
from typing import Any

from .iocs import extract_iocs
from .models import IocSet
from .tools_social import _search_result_text

_HASH_RE = re.compile(r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b")

# Snippet vocabulary that marks a result as describing security evidence
# rather than merely mentioning the queried string.
_SECURITY_MARKERS = (
    "ioc",
    "indicator",
    "malware",
    "c2",
    "rat",
    "trojan",
    "backdoor",
    "phishing",
    "ransomware",
    "exploit",
    "vulnerability",
    "compromise",
    "hunting",
    "supply chain",
    "postinstall",
    "threat",
    "attack",
)


def _normalize_domain_filter_values(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    normalized: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        candidate = item.strip().lower().rstrip(".")
        if candidate:
            normalized.append(candidate)
    return normalized


def _build_searxng_query(query: str, include_domains: Any = None, exclude_domains: Any = None) -> str:
    parts = [query.strip()]
    parts.extend(f"site:{domain}" for domain in _normalize_domain_filter_values(include_domains))
    parts.extend(f"-site:{domain}" for domain in _normalize_domain_filter_values(exclude_domains))
    return " ".join(part for part in parts if part).strip()


def _normalize_searxng_results(payload: Any, max_results: int) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        record: dict[str, Any] = {
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "content": item.get("content") or item.get("snippet") or "",
        }
        if item.get("engine"):
            record["engine"] = item["engine"]
        if item.get("category"):
            record["category"] = item["category"]
        if item.get("score") is not None:
            record["score"] = item["score"]
        normalized.append(record)
        if len(normalized) >= max_results:
            break
    return normalized


def _query_ioc_seeds(search: str) -> set[str]:
    """Return IoC-like values from the search query for contextual matching."""
    iocs = extract_iocs(search)
    seeds = {value.lower() for value in iocs.all_values()}
    for value in _HASH_RE.findall(search):
        seeds.add(value.lower())
    return seeds


def _has_security_context(text: str) -> bool:
    """Check whether a search snippet is describing security evidence."""
    lowered = text.lower()
    return any(marker in lowered for marker in _SECURITY_MARKERS)


def _search_snippet_iocs(items: list[dict[str, Any]], search: str) -> IocSet:
    """Extract Pivot IoCs only from snippets tied to the queried indicator."""
    seeds = _query_ioc_seeds(search)
    if not seeds:
        return IocSet()
    eligible_text: list[str] = []
    for item in items:
        text = _search_result_text([item], "title", "content")
        lowered = text.lower()
        if not any(seed in lowered for seed in seeds):
            continue
        if not _has_security_context(text):
            continue
        eligible_text.append(text)
    iocs = extract_iocs("\n".join(eligible_text))
    hash_seeds = {seed for seed in seeds if len(seed) in {32, 40, 64} and all(char in "0123456789abcdef" for char in seed)}
    if hash_seeds:
        iocs.hashes = [value for value in iocs.hashes if value in hash_seeds]
    return iocs


__all__ = [
    "_build_searxng_query",
    "_has_security_context",
    "_normalize_domain_filter_values",
    "_normalize_searxng_results",
    "_query_ioc_seeds",
    "_search_snippet_iocs",
]
