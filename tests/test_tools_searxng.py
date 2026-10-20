"""Unit tests for the SearXNG query/result helpers.

These are the pure half of the `SEARCH_ENGINE` adapter (`tools_searxng.py`),
so they are exercised here without a server; `tests/test_tools.py` covers the
request path that uses them.
"""

from __future__ import annotations

from tiblender_runtime import tools
from tiblender_runtime.tools_searxng import (
    _build_searxng_query,
    _has_security_context,
    _normalize_domain_filter_values,
    _normalize_searxng_results,
    _query_ioc_seeds,
    _search_snippet_iocs,
)


def test_tools_uses_the_extracted_searxng_helpers():
    """`tools` keeps one implementation, imported from the split-out module."""
    from tiblender_runtime import tools_searxng

    assert tools._build_searxng_query is tools_searxng._build_searxng_query
    assert tools._normalize_searxng_results is tools_searxng._normalize_searxng_results
    assert tools._search_snippet_iocs is tools_searxng._search_snippet_iocs


def test_domain_filter_values_are_normalized_and_non_strings_dropped():
    assert _normalize_domain_filter_values(None) == []
    assert _normalize_domain_filter_values("  Example.COM.  ") == ["example.com"]
    assert _normalize_domain_filter_values(["A.test", 7, "", "b.test"]) == ["a.test", "b.test"]


def test_search_query_carries_site_include_and_exclude_filters():
    assert (
        _build_searxng_query("apt28 router", include_domains="Example.com", exclude_domains=["bad.example"])
        == "apt28 router site:example.com -site:bad.example"
    )
    assert _build_searxng_query("  spaced  ") == "spaced"


def test_result_normalization_keeps_known_fields_and_honours_the_cap():
    payload = {
        "results": [
            {"title": "t1", "url": "https://a.example", "content": "c1", "engine": "duck", "score": 1.5},
            "not-a-dict",
            {"title": "t2", "url": "https://b.example", "snippet": "c2", "category": "general"},
            {"title": "t3", "url": "https://c.example", "content": "c3"},
        ]
    }
    items = _normalize_searxng_results(payload, max_results=2)
    assert [item["url"] for item in items] == ["https://a.example", "https://b.example"]
    assert items[0]["engine"] == "duck" and items[0]["score"] == 1.5
    assert items[1]["content"] == "c2" and items[1]["category"] == "general"
    assert "engine" not in items[1]
    assert _normalize_searxng_results([], max_results=5) == []
    assert _normalize_searxng_results({"results": "nope"}, max_results=5) == []


def test_query_seeds_include_extracted_iocs_and_bare_hashes():
    digest = "a" * 64
    seeds = _query_ioc_seeds(f"sfrclak.com {digest}")
    assert "sfrclak.com" in seeds
    assert digest in seeds


def test_security_context_marks_only_security_snippets():
    assert _has_security_context("Indicator of compromise observed")
    assert not _has_security_context("A recipe for sourdough bread")


def test_snippets_yield_iocs_only_when_tied_to_the_query_and_security_related():
    items = [
        # Mentions the queried indicator and reads as security evidence.
        {"title": "sfrclak.com malware c2", "content": "Payload staged at http://sfrclak.com:8000/6202033"},
        # Mentions it, but is not security evidence: its host must not leak.
        {"title": "sfrclak.com domain for sale", "content": "Visit https://unrelated-marketplace.example"},
        # Security evidence about something else entirely.
        {"title": "other ransomware campaign", "content": "See https://elsewhere.example"},
    ]
    iocs = _search_snippet_iocs(items, "sfrclak.com supply chain attack")
    assert "sfrclak.com" in iocs.domains
    assert "unrelated-marketplace.example" not in iocs.domains
    assert "elsewhere.example" not in iocs.domains


def test_a_query_with_no_indicator_yields_nothing():
    items = [{"title": "ransomware roundup", "content": "https://elsewhere.example"}]
    assert _search_snippet_iocs(items, "weekly threat roundup").all_values() == []


def test_hashes_are_restricted_to_the_queried_digest():
    queried = "b" * 64
    other = "c" * 64
    items = [{"title": f"malware {queried}", "content": f"related sample {other}"}]
    iocs = _search_snippet_iocs(items, queried)
    assert iocs.hashes == [queried]
