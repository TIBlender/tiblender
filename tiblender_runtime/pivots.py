"""Shared-hosting and corroboration rules for one-level pivot indicators.

A one-level pivot (§III of the paper) asks an infrastructure tool what else the
queried indicator touches. Two things can come back:

* a small, specific resolution set -- the campaign's other hosts;
* every site that happens to share a CDN edge or a hosting IP.

The second kind is not evidence of anything. Emitting it as an indicator states
an infrastructure relationship the run never verified, so this module encodes
the two rules that keep it out:

1. **Shared hosting.** A resolution answer for an *IP* that returns more than
   ``TIBLENDER_PIVOT_MAX_RESOLUTIONS_PER_IP`` (default 5) values marks that IP
   as a shared-hosting environment. Its co-tenants are dropped. No confidence
   bump follows: the paper's +1 is for a link to a *known malicious* domain on
   the same IP or certificate, and the deployed scorer treats a crowded
   address as a false-positive risk rather than as corroboration.
2. **Corroboration.** A pivot-derived value becomes an indicator only when at
   least two independent tool answers returned it, or when it is already a
   Direct indicator from post text. One tool answer is an unconfirmed pivot.
   Answers are counted one per distinct ``(tool type, queried indicator)``
   pair, so asking the same tool the same question twice is not corroboration.

``context.aggregate_context`` scores every indicator with these rules, and
``reporting.build_report`` derives the flat ``iocs`` list from that one scored
structure whenever a context is available, so the report's ``iocs``,
``report_schema.iocs`` and ``iocs_evidence`` cannot disagree about which
values are indicators or about their confidence. ``corroborated_tool_iocs``
below reproduces the same bar for the context-free path (a report built
straight from findings, as in the collector sample run).
"""

from __future__ import annotations

import os
from typing import Any, Iterable

from .models import IocSet

DEFAULT_MAX_RESOLUTIONS_PER_IP = 5
ENV_MAX_RESOLUTIONS_PER_IP = "TIBLENDER_PIVOT_MAX_RESOLUTIONS_PER_IP"

# Tools whose answer for one indicator is a *resolution set*: other names or
# addresses sharing infrastructure with it.
RESOLUTION_TOOLS = frozenset({"PASSIVE_DNS_LOOKUP", "TLS_CT_LOOKUP"})

# Independent tool answers a pivot value needs before it is reportable. Matches
# the paper's confidence rule (>=2 tools -> MEDIUM) and ``context._tier``.
MIN_PIVOT_SOURCES = 2

_PARAMETER_ORDER = ("domain", "ip", "url", "cve", "hash", "query")
_VALUE_FIELDS = ("urls", "domains", "ips", "hashes", "cves")


def max_resolutions_per_ip(override: int | None = None) -> int:
    """Resolutions considered per IP before it counts as shared hosting."""
    if override is not None:
        return max(1, int(override))
    raw = os.getenv(ENV_MAX_RESOLUTIONS_PER_IP, "")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_RESOLUTIONS_PER_IP


def queried_value(result: Any) -> str:
    """The indicator a tool result was asked about, or ""."""
    for name in _PARAMETER_ORDER:
        raw = result.query_params.get(name)
        if raw:
            return str(raw).strip()
    return ""


def resolution_count(iocs: IocSet) -> int:
    """Number of network indicators a resolution answer returned."""
    return len(iocs.domains) + len(iocs.ips) + len(iocs.urls)


def is_shared_hosting_result(result: Any, limit: int) -> bool:
    """True when this answer describes an IP that hosts many unrelated names.

    Scoped to answers keyed on an ``ip``: a certificate-transparency answer for
    a *domain* returns that domain's own subdomains, which are related by
    registration rather than merely co-resident.
    """
    if result.tool_type not in RESOLUTION_TOOLS:
        return False
    if not str(result.query_params.get("ip") or "").strip():
        return False
    return resolution_count(result.discovered_iocs) > limit


def shared_hosting_queries(results: Iterable[Any], limit: int) -> set[str]:
    """Queried values whose own resolution set marks them as shared hosting."""
    found: set[str] = set()
    for result in results:
        if getattr(result, "status", "") != "success":
            continue
        if not is_shared_hosting_result(result, limit):
            continue
        value = queried_value(result)
        if value:
            found.add(value)
    return found


def source_key(result: Any) -> str:
    """Identity of one tool answer, used to count independent corroboration."""
    return f"tool:{result.tool_type}:{queried_value(result)}"


def keeps_discovered_values(result: Any, limit: int) -> bool:
    """False when a result's discovered values are shared-hosting co-tenants."""
    return not is_shared_hosting_result(result, limit)


def corroborated_tool_iocs(
    results: Iterable[Any],
    *,
    direct_values: set[str],
    limit: int,
) -> IocSet:
    """Tool-discovered indicators that survive both pivot rules.

    A value is kept when it is one of two things:

    * already a Direct indicator from post text (``direct_values``) -- a tool
      answer merely confirms it and it is never held to the pivot bar;
    * returned by at least ``MIN_PIVOT_SOURCES`` independent tool answers,
      counted one per distinct ``(tool type, queried indicator)`` pair.

    Everything else is an uncorroborated pivot: a relationship the run never
    verified, which must not ship as an indicator.
    """
    lowered_direct = {value.lower() for value in direct_values}
    sources: dict[tuple[str, str], set[str]] = {}
    for result in results:
        if getattr(result, "status", "") != "success":
            continue
        if not keeps_discovered_values(result, limit):
            continue
        key = source_key(result)
        for field_name in _VALUE_FIELDS:
            for value in getattr(result.discovered_iocs, field_name):
                sources.setdefault((field_name, value), set()).add(key)
    kept: dict[str, list[str]] = {field_name: [] for field_name in _VALUE_FIELDS}
    for (field_name, value), keys in sources.items():
        if value.lower() in lowered_direct or len(keys) >= MIN_PIVOT_SOURCES:
            kept[field_name].append(value)
    return IocSet(**{field_name: sorted(set(values)) for field_name, values in kept.items()})


__all__ = [
    "DEFAULT_MAX_RESOLUTIONS_PER_IP",
    "ENV_MAX_RESOLUTIONS_PER_IP",
    "MIN_PIVOT_SOURCES",
    "RESOLUTION_TOOLS",
    "corroborated_tool_iocs",
    "is_shared_hosting_result",
    "keeps_discovered_values",
    "max_resolutions_per_ip",
    "queried_value",
    "resolution_count",
    "shared_hosting_queries",
    "source_key",
]
