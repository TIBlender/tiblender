"""Validates the recorded tool-response bundle under fixtures/tools/.

These tests load the actual recording files produced by
`tools/record_tool_responses.py` (not synthetic in-test payloads) against every
query in fixtures/tools/queries.json, and check the guarantees the recording
step is supposed to uphold:
schema completeness, real (non-fabricated) recorded_at/status/error values,
that the WEB_CRAWL host allowlist was actually enforced on disk, that
recording_summary.json matches the files on disk, and that at least one
successful recording was made for the tools that could be exercised
live from this environment.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from tiblender_runtime.tool_fixtures import recording_key
from tools.record_tool_responses import (
    CRAWL_SKIPPED_ERROR,
    DEFAULT_ALLOWED_CRAWL_HOSTS,
    extract_host,
    is_allowed_crawl_host,
)

_FS_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _FS_ROOT / "fixtures" / "tools"
_QUERIES_PATH = _TOOLS_DIR / "queries.json"
_SUMMARY_PATH = _TOOLS_DIR / "recording_summary.json"

_REQUIRED_KEYS = {
    "tool_type",
    "parameters",
    "recorded_at",
    "source_url",
    "status",
    "summary",
    "discovered_iocs",
    "raw",
    "error",
}
_VALID_STATUSES = {"success", "unavailable", "error"}

# At least one recorded query must succeed for each of these tool types; all
# five are reachable anonymously (no API key).
_REQUIRED_SUCCESS_TOOL_TYPES = (
    "WHOIS_LOOKUP",
    "TLS_CT_LOOKUP",
    "NVD_CVE_LOOKUP",
    "GITHUB_SEARCH",
    # Passive DNS is recorded from VirusTotal, the only provider, which
    # answers this environment.
    "PASSIVE_DNS_LOOKUP",
)

_ISO_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


def _load_queries() -> list[dict]:
    return json.loads(_QUERIES_PATH.read_text(encoding="utf-8"))


def _tool_dirs() -> list[Path]:
    if not _TOOLS_DIR.exists():
        return []
    return sorted(path for path in _TOOLS_DIR.iterdir() if path.is_dir())


def _recording_files() -> list[Path]:
    files: list[Path] = []
    for tool_dir in _tool_dirs():
        files.extend(sorted(tool_dir.glob("*.json")))
    return files


def _load_recording(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def queries() -> list[dict]:
    assert _QUERIES_PATH.exists(), f"missing {_QUERIES_PATH}"
    return _load_queries()


@pytest.fixture(scope="module")
def recording_files() -> list[Path]:
    files = _recording_files()
    assert files, f"no recording files found under {_TOOLS_DIR}"
    return files


@pytest.fixture(scope="module")
def recordings_by_tool(recording_files: list[Path]) -> dict[str, list[dict]]:
    by_tool: dict[str, list[dict]] = {}
    for path in recording_files:
        payload = _load_recording(path)
        by_tool.setdefault(payload.get("tool_type", ""), []).append(payload)
    return by_tool


def test_fixtures_tools_dir_has_a_subdirectory_per_tool_type(queries: list[dict]):
    expected_tool_types = {entry["tool_type"] for entry in queries}
    actual_tool_types = {path.name for path in _tool_dirs()}
    missing = expected_tool_types - actual_tool_types
    assert not missing, f"no recording directory created for tool_type(s): {sorted(missing)}"


def test_every_recording_has_required_keys(recording_files: list[Path]):
    for path in recording_files:
        payload = _load_recording(path)
        missing = _REQUIRED_KEYS - payload.keys()
        assert not missing, f"{path} is missing keys: {sorted(missing)}"


def test_every_recording_status_is_valid_and_error_status_has_error_text(recording_files: list[Path]):
    for path in recording_files:
        payload = _load_recording(path)
        status = payload["status"]
        assert status in _VALID_STATUSES, f"{path} has invalid status {status!r}"
        if status == "error":
            assert payload["error"].strip(), f"{path} has status=error but an empty error field"


def test_every_recording_recorded_at_is_iso8601_z(recording_files: list[Path]):
    for path in recording_files:
        recorded_at = _load_recording(path)["recorded_at"]
        assert _ISO_Z_RE.match(recorded_at), f"{path} has non-ISO8601Z recorded_at {recorded_at!r}"
        # Must also be a real, parseable wall-clock timestamp (never a fixed
        # placeholder like 1970-01-01), guarding against a fixed test clock
        # having been active during recording.
        parsed = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
        assert parsed.year >= 2026, f"{path} recorded_at {recorded_at!r} looks like a fixed/placeholder clock"


def test_every_recording_tool_type_matches_its_folder(recording_files: list[Path]):
    for path in recording_files:
        payload = _load_recording(path)
        assert payload["tool_type"] == path.parent.name, f"{path} tool_type {payload['tool_type']!r} != folder {path.parent.name!r}"


def test_every_recording_parameters_is_non_empty(recording_files: list[Path]):
    for path in recording_files:
        parameters = _load_recording(path)["parameters"]
        assert parameters, f"{path} has empty parameters"


def test_every_recording_filename_is_the_deterministic_recording_key(recording_files: list[Path]):
    for path in recording_files:
        payload = _load_recording(path)
        expected_key = recording_key(payload["tool_type"], payload["parameters"])
        assert path.stem == expected_key, f"{path} filename does not match recording_key({payload['tool_type']!r}, ...)"


def test_every_query_in_queries_json_has_a_recording_file(queries: list[dict]):
    missing = []
    for entry in queries:
        tool_type = entry["tool_type"]
        parameters = entry.get("parameters", {})
        expected_path = _TOOLS_DIR / tool_type / f"{recording_key(tool_type, parameters)}.json"
        if not expected_path.exists():
            missing.append((tool_type, parameters))
    assert not missing, f"queries.json entries with no recorded tool response: {missing}"


@pytest.mark.parametrize("tool_type", _REQUIRED_SUCCESS_TOOL_TYPES)
def test_required_tool_types_have_at_least_one_successful_recording(tool_type: str, recordings_by_tool: dict[str, list[dict]]):
    statuses = [payload["status"] for payload in recordings_by_tool.get(tool_type, [])]
    assert "success" in statuses, f"no successful recording made for {tool_type} (statuses seen: {statuses})"


def test_passive_dns_recordings_are_recorded_from_virustotal(recordings_by_tool: dict[str, list[dict]]):
    """Passive DNS is recorded live from VirusTotal's resolutions endpoint,
    the only passive DNS provider, and the bundle carries real evidence.
    """
    recordings = recordings_by_tool.get("PASSIVE_DNS_LOOKUP", [])
    assert recordings, "expected recorded PASSIVE_DNS_LOOKUP recordings"
    successes = [payload for payload in recordings if payload["status"] == "success"]
    assert successes, "no successful PASSIVE_DNS_LOOKUP recording made"
    for payload in successes:
        assert payload["raw"].get("provider") == "virustotal", f"unexpected provider in {payload['parameters']}"
        assert "virustotal.com/api/v3/" in payload["source_url"]
        assert payload["summary"].startswith("VirusTotal passive DNS for ")


def test_passive_dns_recording_keys_are_addressable_by_parameters_alone(recordings_by_tool: dict[str, list[dict]]):
    """Every bundled recording is addressable by tool type and parameters alone."""
    from tiblender_runtime.settings import Settings
    from tiblender_runtime.tool_fixtures import ToolRecordingStore

    recordings = recordings_by_tool.get("PASSIVE_DNS_LOOKUP", [])
    assert recordings, "expected recorded PASSIVE_DNS_LOOKUP recordings"
    settings = Settings(fixtures_dir=_TOOLS_DIR.parent)
    store = ToolRecordingStore(settings.fixtures_dir / "tools")

    for payload in recordings:
        parameters = payload["parameters"]
        key = recording_key("PASSIVE_DNS_LOOKUP", parameters)
        assert (_TOOLS_DIR / "PASSIVE_DNS_LOOKUP" / f"{key}.json").exists()
        replayed = store.lookup("PASSIVE_DNS_LOOKUP", parameters)
        assert replayed is not None, f"{parameters} unreachable"
        assert replayed["summary"] == payload["summary"]


def test_web_crawl_recordings_outside_allowlist_are_unavailable_and_skipped(recordings_by_tool: dict[str, list[dict]]):
    crawl_recordings = recordings_by_tool.get("WEB_CRAWL", [])
    assert crawl_recordings, "expected recorded WEB_CRAWL recordings"
    checked_untrusted = 0
    for payload in crawl_recordings:
        url = payload["parameters"].get("url", "")
        host = extract_host(url)
        if is_allowed_crawl_host(host, DEFAULT_ALLOWED_CRAWL_HOSTS):
            continue
        checked_untrusted += 1
        assert payload["status"] == "unavailable", f"untrusted-host WEB_CRAWL {url!r} was not recorded as unavailable"
        assert CRAWL_SKIPPED_ERROR in payload["error"], f"untrusted-host WEB_CRAWL {url!r} error does not cite {CRAWL_SKIPPED_ERROR!r}: {payload['error']!r}"
        assert payload["discovered_iocs"] == {"urls": [], "domains": [], "ips": [], "hashes": [], "cves": []}
    assert checked_untrusted > 0, "expected at least one WEB_CRAWL query outside the allowlist in the fixture bundle"


def test_no_web_crawl_success_outside_the_allowlist(recordings_by_tool: dict[str, list[dict]]):
    for payload in recordings_by_tool.get("WEB_CRAWL", []):
        if payload["status"] != "success":
            continue
        host = extract_host(payload["parameters"].get("url", ""))
        assert is_allowed_crawl_host(host, DEFAULT_ALLOWED_CRAWL_HOSTS), f"WEB_CRAWL succeeded for untrusted host {host!r}"


def test_recording_summary_json_exists_and_matches_recording_files_on_disk(recordings_by_tool: dict[str, list[dict]]):
    assert _SUMMARY_PATH.exists(), f"missing {_SUMMARY_PATH}"
    summary = json.loads(_SUMMARY_PATH.read_text(encoding="utf-8"))

    for key in ("recorded_at", "allow_crawl_hosts", "searxng_used"):
        assert key in summary, f"recording_summary.json missing key {key!r}"
    assert _ISO_Z_RE.match(summary["recorded_at"]), f"recording_summary.json recorded_at is not ISO8601Z: {summary['recorded_at']!r}"

    non_meta_keys = set(summary) - {"recorded_at", "allow_crawl_hosts", "searxng_used"}
    assert non_meta_keys == set(recordings_by_tool), f"recording_summary.json tool_type keys {non_meta_keys} != recording folders {set(recordings_by_tool)}"

    for tool_type, recordings in recordings_by_tool.items():
        expected = {"success": 0, "unavailable": 0, "error": 0}
        for payload in recordings:
            expected[payload["status"]] = expected.get(payload["status"], 0) + 1
        assert summary[tool_type] == expected, f"recording_summary.json counts for {tool_type} do not match recording files on disk"


def test_no_secret_like_values_leak_into_recording_raw_or_source_url(recording_files: list[Path]):
    # Providers never echo request headers back in a response body, so these
    # patterns (the header/query-param names tools.py sends credentials
    # under, serialized as JSON keys) should never legitimately appear in a
    # recorded tool response. Matched as quoted-JSON-key/header-literal patterns
    # rather than bare words, so ordinary provider text (e.g. a GitHub repo
    # description mentioning "authorization bypass" for a CVE) cannot
    # false-positive this check.
    secret_key_markers = ('"authorization":', '"apikey":', "x-apikey")
    for path in recording_files:
        text = path.read_text(encoding="utf-8").lower()
        for marker in secret_key_markers:
            assert marker not in text, f"{path} appears to contain a leaked credential marker {marker!r}"


def test_no_recording_on_disk_is_orphaned_from_queries_json(queries: list[dict], recording_files: list[Path]):
    """The reverse of the coverage test: nothing ships that nothing asks for.

    120 recordings for a previous fixture bundle survived a rebuild once and
    were only found by hand; a recording with no `queries.json` entry is dead
    weight that also makes `recording_summary.json` describe files the
    bundle no longer uses.
    """
    expected = {
        (entry["tool_type"], recording_key(entry["tool_type"], entry.get("parameters", {})))
        for entry in queries
    }
    orphans = [
        str(path.relative_to(_TOOLS_DIR))
        for path in recording_files
        if (path.parent.name, path.stem) not in expected
    ]
    assert not orphans, f"recorded tool responses with no queries.json entry: {orphans}"


def test_github_search_queries_use_the_parameter_name_the_runtime_issues(queries: list[dict]):
    """`recording_key` hashes the *named* parameters, so `{"cve": X}` and
    `{"query": X}` are different keys for the same GitHub request.
    `perspectives._tool_parameter_sets` issues GITHUB_SEARCH as `{"query": ...}`;
    when the builder emitted `{"cve": ...}` every recorded GitHub response was
    unreachable by construction (0 of 11 runtime calls replayed).
    """
    from tiblender_runtime.models import Cluster, IocSet, NormalizedPost, Post, Severity, ThreatCategory
    from tiblender_runtime.perspectives import _Signals, _tool_parameter_sets

    github_entries = [entry for entry in queries if entry["tool_type"] == "GITHUB_SEARCH"]
    assert github_entries, "queries.json has no GITHUB_SEARCH entries"
    for entry in github_entries:
        assert set(entry["parameters"]) == {"query"}, (
            f"GITHUB_SEARCH query {entry['parameters']!r} uses a parameter name the runtime never issues"
        )

    text = "CVE-2026-33186 exploited in the wild"
    cluster = Cluster(
        id="c1",
        topic_name="CVE-2026-33186 exploitation",
        description=text,
        posts=[
            NormalizedPost(
                post=Post(id="p1", platform="x", content=text),
                normalized_content=text,
                iocs=IocSet(cves=["CVE-2026-33186"]),
                threat_category=ThreatCategory.VULNERABILITY,
            )
        ],
        iocs=IocSet(cves=["CVE-2026-33186"]),
        severity=Severity.HIGH,
        threat_category=ThreatCategory.VULNERABILITY,
    )
    issued = _tool_parameter_sets("GITHUB_SEARCH", cluster, _Signals(cluster))
    assert issued and all(set(parameters) == {"query"} for parameters in issued), (
        f"the runtime now issues GITHUB_SEARCH as {issued!r}; queries.json must use the same parameter name"
    )
