"""Unit tests for the pure helper functions in tools/record_tool_responses.py.

These cover the WEB_CRAWL host allowlist, --resume recording-exists check,
--only/--skip filtering, and per-tool sleep floors without making any
network calls or invoking ToolRunner.
"""

from __future__ import annotations

from pathlib import Path

from tools.record_tool_responses import (
    DEFAULT_ALLOWED_CRAWL_HOSTS,
    recording_file_path,
    extract_host,
    filter_entries,
    is_allowed_crawl_host,
    is_already_recorded,
    parse_host_list,
    parse_tool_set,
    repair_truncated_raw_json,
    resolve_sleep_seconds,
    summarize_recordings,
)


def test_parse_host_list_normalizes_case_whitespace_and_dedupes():
    assert parse_host_list(" Github.com , github.com ,CVE.org") == ("cve.org", "github.com")


def test_parse_host_list_empty_string_is_empty_tuple():
    assert parse_host_list("") == ()
    assert parse_host_list("  ,  ") == ()


def test_extract_host_lowercases_and_strips_path():
    assert extract_host("https://WWW.Example.COM/a/b?q=1") == "www.example.com"


def test_extract_host_returns_empty_for_hostless_or_invalid_url():
    assert extract_host("not a url") == ""
    assert extract_host("") == ""


def test_is_allowed_crawl_host_exact_and_subdomain_match():
    allowed = ("github.com", "nist.gov")
    assert is_allowed_crawl_host("github.com", allowed)
    assert is_allowed_crawl_host("gist.github.com", allowed)
    assert is_allowed_crawl_host("nvd.nist.gov", allowed)
    assert is_allowed_crawl_host("GITHUB.COM", allowed)


def test_is_allowed_crawl_host_rejects_untrusted_and_lookalike_hosts():
    allowed = ("github.com",)
    assert not is_allowed_crawl_host("evilgithub.com", allowed)
    assert not is_allowed_crawl_host("github.com.evil.example", allowed)
    assert not is_allowed_crawl_host("", allowed)
    assert not is_allowed_crawl_host("medium.com", allowed)


def test_default_allowed_crawl_hosts_covers_base_and_fixture_publishers():
    for host in ("github.com", "thehackernews.com", "bleepingcomputer.com", "helpnetsecurity.com"):
        assert host in DEFAULT_ALLOWED_CRAWL_HOSTS
    for host in ("horizon3.ai", "pcmag.com", "safedep.io", "security-next.com", "undernews.fr"):
        assert host in DEFAULT_ALLOWED_CRAWL_HOSTS
    # News/CERT hosts added for the rebuilt fixture bundle.
    for host in (
        "reuters.com",
        "cnn.com",
        "apnews.com",
        "bbc.com",
        "darkreading.com",
        "infosecurity-magazine.com",
        "ncsc.gov.uk",
        "ic3.gov",
    ):
        assert host in DEFAULT_ALLOWED_CRAWL_HOSTS


def test_resolve_sleep_seconds_uses_floor_when_larger_than_base():
    assert resolve_sleep_seconds("NVD_CVE_LOOKUP", 1.0) == 6.0
    assert resolve_sleep_seconds("GITHUB_SEARCH", 1.0) == 7.0
    # VirusTotal free tier: 4 requests/minute, so 16 s between calls.
    assert resolve_sleep_seconds("PASSIVE_DNS_LOOKUP", 1.0) == 16.0
    assert resolve_sleep_seconds("TLS_CT_LOOKUP", 1.0) == 2.0


def test_resolve_sleep_seconds_uses_base_when_larger_than_floor():
    assert resolve_sleep_seconds("NVD_CVE_LOOKUP", 10.0) == 10.0


def test_resolve_sleep_seconds_defaults_to_base_for_unlisted_tool():
    assert resolve_sleep_seconds("WHOIS_LOOKUP", 1.5) == 1.5


def test_parse_tool_set_uppercases_and_splits():
    assert parse_tool_set("whois_lookup, tls_ct_lookup") == {"WHOIS_LOOKUP", "TLS_CT_LOOKUP"}


def test_parse_tool_set_none_for_empty_input():
    assert parse_tool_set("") is None
    assert parse_tool_set(None) is None


def test_filter_entries_only_keeps_matching_tool_types():
    entries = [{"tool_type": "WHOIS_LOOKUP"}, {"tool_type": "TLS_CT_LOOKUP"}, {"tool_type": "NVD_CVE_LOOKUP"}]
    filtered = filter_entries(entries, only={"WHOIS_LOOKUP", "NVD_CVE_LOOKUP"}, skip=None)
    assert [e["tool_type"] for e in filtered] == ["WHOIS_LOOKUP", "NVD_CVE_LOOKUP"]


def test_filter_entries_skip_drops_matching_tool_types():
    entries = [{"tool_type": "WHOIS_LOOKUP"}, {"tool_type": "TLS_CT_LOOKUP"}]
    filtered = filter_entries(entries, only=None, skip={"TLS_CT_LOOKUP"})
    assert [e["tool_type"] for e in filtered] == ["WHOIS_LOOKUP"]


def test_filter_entries_only_and_skip_combine():
    entries = [{"tool_type": "A"}, {"tool_type": "B"}, {"tool_type": "C"}]
    filtered = filter_entries(entries, only={"A", "B"}, skip={"B"})
    assert [e["tool_type"] for e in filtered] == ["A"]


def test_recording_file_path_is_deterministic_and_matches_recording_key(tmp_path: Path):
    from tiblender_runtime.tool_fixtures import recording_key

    parameters = {"domain": "example.net"}
    path = recording_file_path(tmp_path, "WHOIS_LOOKUP", parameters)
    assert path == tmp_path / "WHOIS_LOOKUP" / f"{recording_key('WHOIS_LOOKUP', parameters)}.json"
    # Case-insensitive parameter normalization inside recording_key keeps the
    # path stable regardless of how the caller cased the domain.
    assert path == recording_file_path(tmp_path, "WHOIS_LOOKUP", {"domain": "EXAMPLE.net"})


def test_is_already_recorded_false_when_missing_true_after_write(tmp_path: Path):
    parameters = {"domain": "example.net"}
    assert is_already_recorded(tmp_path, "WHOIS_LOOKUP", parameters) is False
    path = recording_file_path(tmp_path, "WHOIS_LOOKUP", parameters)
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    assert is_already_recorded(tmp_path, "WHOIS_LOOKUP", parameters) is True


def test_summarize_recordings_counts_by_tool_type_and_status(tmp_path: Path):
    (tmp_path / "WHOIS_LOOKUP").mkdir()
    (tmp_path / "WHOIS_LOOKUP" / "a.json").write_text('{"status": "success"}', encoding="utf-8")
    (tmp_path / "WHOIS_LOOKUP" / "b.json").write_text('{"status": "unavailable"}', encoding="utf-8")
    (tmp_path / "TLS_CT_LOOKUP").mkdir()
    (tmp_path / "TLS_CT_LOOKUP" / "c.json").write_text('{"status": "success"}', encoding="utf-8")
    (tmp_path / "recording_summary.json").write_text("{}", encoding="utf-8")

    counts = summarize_recordings(tmp_path)

    assert counts["WHOIS_LOOKUP"] == {"success": 1, "unavailable": 1, "error": 0}
    assert counts["TLS_CT_LOOKUP"] == {"success": 1, "unavailable": 0, "error": 0}
    assert "recording_summary" not in counts


def test_summarize_recordings_missing_root_returns_empty_dict(tmp_path: Path):
    assert summarize_recordings(tmp_path / "does-not-exist") == {}


def test_summarize_recordings_skips_unreadable_file(tmp_path: Path):
    (tmp_path / "WHOIS_LOOKUP").mkdir()
    (tmp_path / "WHOIS_LOOKUP" / "broken.json").write_text("{not valid json", encoding="utf-8")
    assert summarize_recordings(tmp_path) == {}


def test_repair_truncated_raw_json_leaves_valid_json_unchanged():
    valid = '{"a": 1, "b": [1, 2, 3]}'
    assert repair_truncated_raw_json(valid) == valid


def test_repair_truncated_raw_json_leaves_empty_string_unchanged():
    assert repair_truncated_raw_json("") == ""


def test_repair_truncated_raw_json_wraps_truncated_payload_as_valid_json():
    truncated = '{"items": [{"full_name": "foo/bar", "description": "cut off mid-str'
    import json as _json

    repaired = repair_truncated_raw_json(truncated)
    parsed = _json.loads(repaired)  # must not raise
    assert parsed["raw_prefix"] == truncated
    assert "note" in parsed


def test_repair_truncated_raw_json_does_not_lose_any_original_text():
    truncated = "not json at all \" unterminated"
    repaired = repair_truncated_raw_json(truncated)
    import json as _json

    assert _json.loads(repaired)["raw_prefix"] == truncated


def test_the_crawl_allowlist_is_a_subset_of_the_runtime_ioc_allowlist():
    """A host trusted enough to fetch is never campaign infrastructure.

    The two lists used to be maintained separately and drifted apart by 14
    hosts, so CT/passive-DNS/crawl evidence recorded from publishers and
    national CERTs was extracted as indicators. Both now derive from
    ``tiblender_runtime.legitimate_hosts``.
    """
    from tiblender_runtime.legitimate_hosts import PUBLISHER_HOSTS, is_suppressed_host

    # The publisher half, so a recorded host's *subdomains* are suppressed too.
    assert set(DEFAULT_ALLOWED_CRAWL_HOSTS) <= set(PUBLISHER_HOSTS)
    assert all(is_suppressed_host(f"beta.{host}") for host in DEFAULT_ALLOWED_CRAWL_HOSTS)
    # Not vacuous: the allowlist is non-empty and every entry really matches.
    assert DEFAULT_ALLOWED_CRAWL_HOSTS
    for host in DEFAULT_ALLOWED_CRAWL_HOSTS:
        assert is_allowed_crawl_host(f"www.{host}", DEFAULT_ALLOWED_CRAWL_HOSTS)
