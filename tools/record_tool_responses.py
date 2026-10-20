#!/usr/bin/env python3
"""Record tool responses from live providers for deterministic offline replay.

Usage:
    python tools/record_tool_responses.py --fixtures fixtures --queries fixtures/tools/queries.json

queries.json is a JSON list of {"tool_type": "WHOIS_LOOKUP", "parameters": {"domain": "..."}}
objects. Settings are loaded the same way as the runtime CLI (.env plus the
environment); this script then forces live mode so ToolRunner contacts real
providers, and records each response (success or unavailable) into the tool
recording store so `offline`/`auto` cycles can replay it without network access.

Useful flags for a large queries file:
    --only WHOIS_LOOKUP,TLS_CT_LOOKUP   only record these tool_types
    --skip SEARCH_ENGINE                 skip these tool_types
    --sleep 2.0                          pace calls (per-tool floors may raise this)
    --resume                             skip queries that already have a recording file
    --allow-crawl-hosts host1,host2      WEB_CRAWL host allowlist (suffix match)

A summary table is printed at the end and written to
fixtures/tools/recording_summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiblender_runtime.legitimate_hosts import WEB_CRAWL_ALLOWED_HOSTS  # noqa: E402
from tiblender_runtime.models import IocSet, ToolQuery, ToolResult  # noqa: E402
from tiblender_runtime.services import resolve_services  # noqa: E402
from tiblender_runtime.settings import Settings  # noqa: E402
from tiblender_runtime.tool_fixtures import ToolRecordingStore, recording_key  # noqa: E402
from tiblender_runtime.tools import ToolRunner  # noqa: E402

# Hosts a WEB_CRAWL recording may fetch live. Every other WEB_CRAWL query is
# recorded as an "unavailable" response without ever contacting the host:
# never fetch, never open in a browser.
#
# The list itself lives in tiblender_runtime/legitimate_hosts.py, where it is a
# subset of LEGITIMATE_HOSTS by construction: a host trusted enough to fetch is
# a host whose own subdomains must never be extracted as campaign
# infrastructure. See fixtures/README.md ("WEB_CRAWL host allowlist") for the
# per-host rationale, and for hosts considered and deliberately left out
# (e.g. medium.com, threatcluster.io, ow.ly, bit.ly, and www.kkinstagram.com).
DEFAULT_ALLOWED_CRAWL_HOSTS: tuple[str, ...] = WEB_CRAWL_ALLOWED_HOSTS

# Rate-limit floors (seconds) for providers that reject/throttle unauthenticated
# callers quickly. --sleep sets the pacing default for every other tool; these
# floors win whenever they are larger than --sleep.
DEFAULT_TOOL_SLEEP_FLOORS: dict[str, float] = {
    "NVD_CVE_LOOKUP": 6.0,
    "GITHUB_SEARCH": 7.0,
    # VirusTotal's free tier allows 4 requests/minute (500/day), and the
    # recorder issues one request per query plus one on retry, so it paces
    # itself at 16 s per call rather than relying on --sleep being passed.
    "PASSIVE_DNS_LOOKUP": 16.0,
    "TLS_CT_LOOKUP": 2.0,
}

CRAWL_SKIPPED_ERROR = "crawl_skipped_untrusted_host"


def _load_queries(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of {{tool_type, parameters}} objects")
    return payload


def parse_host_list(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated host list into a normalized, sorted, deduplicated tuple."""
    hosts = {item.strip().lower().rstrip(".") for item in raw.split(",") if item.strip()}
    return tuple(sorted(hosts))


def extract_host(url: str) -> str:
    """Return the lowercase hostname of url, or "" when it has none/is unparsable."""
    try:
        return (urllib.parse.urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def is_allowed_crawl_host(host: str, allowed_hosts: tuple[str, ...]) -> bool:
    """True when host equals or is a subdomain of one of allowed_hosts (suffix match)."""
    host = host.lower().rstrip(".")
    if not host:
        return False
    for allowed in allowed_hosts:
        allowed = allowed.lower().rstrip(".")
        if not allowed:
            continue
        if host == allowed or host.endswith("." + allowed):
            return True
    return False


def resolve_sleep_seconds(tool_type: str, base_sleep: float, floors: dict[str, float] | None = None) -> float:
    """Effective pacing sleep for a tool: --sleep, raised to the tool's rate-limit floor."""
    floors = DEFAULT_TOOL_SLEEP_FLOORS if floors is None else floors
    return max(base_sleep, floors.get(tool_type, 0.0))


def parse_tool_set(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {item.strip().upper() for item in raw.split(",") if item.strip()}


def filter_entries(entries: list[dict], *, only: set[str] | None, skip: set[str] | None) -> list[dict]:
    """Pure filter backing --only/--skip: keep entries whose tool_type passes both."""
    skip = skip or set()
    filtered = []
    for entry in entries:
        tool_type = entry["tool_type"]
        if only is not None and tool_type not in only:
            continue
        if tool_type in skip:
            continue
        filtered.append(entry)
    return filtered


def recording_file_path(fixtures_tools_root: Path, tool_type: str, parameters: dict[str, Any]) -> Path:
    """Where ToolRecordingStore would read/write this query's recording (mirrors ToolRecordingStore._path)."""
    return fixtures_tools_root / tool_type / f"{recording_key(tool_type, parameters)}.json"


def is_already_recorded(fixtures_tools_root: Path, tool_type: str, parameters: dict[str, Any]) -> bool:
    """Pure predicate backing --resume: True when a recording file already exists on disk."""
    return recording_file_path(fixtures_tools_root, tool_type, parameters).exists()


def summarize_recordings(fixtures_tools_root: Path) -> dict[str, dict[str, int]]:
    """Scan every recorded tool response on disk and count statuses per tool_type.

    Scanning disk (rather than accumulating counts only for queries processed
    in this invocation) keeps the summary correct across --resume/--only/--skip
    runs that only touch part of the bundle.
    """
    counts: dict[str, dict[str, int]] = {}
    if not fixtures_tools_root.exists():
        return counts
    for path in sorted(fixtures_tools_root.rglob("*.json")):
        if path.parent == fixtures_tools_root:
            continue  # e.g. recording_summary.json itself
        tool_type = path.parent.name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        status = payload.get("status", "unknown")
        bucket = counts.setdefault(tool_type, {"success": 0, "unavailable": 0, "error": 0})
        bucket[status] = bucket.get(status, 0) + 1
    return counts


def _skip_untrusted_crawl_result(query: ToolQuery, url: str, host: str) -> ToolResult:
    reason = f"{CRAWL_SKIPPED_ERROR}: host {host or '<unknown>'} is not in the WEB_CRAWL allowlist"
    return ToolResult(
        tool_type=query.tool_type,
        query_params=dict(query.parameters),
        summary=f"WEB_CRAWL skipped untrusted host {host or '<unknown>'} for {url}",
        result_url=url,
        timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        discovered_iocs=IocSet(),
        status="unavailable",
        error=reason,
        raw_json="",
    )


def repair_truncated_raw_json(raw_json: str) -> str:
    """Guard against ToolResult.raw_json being invalid JSON.

    tools.py's _json() helper truncates the *serialized string* of a live
    provider payload at a fixed character cap, which can cut a JSON value
    off mid-string and leave raw_json unparsable. ToolRecordingStore.record()
    does `json.loads(result.raw_json)` and would raise on that -- losing a
    recorded provider answer to a serialization artifact. This wraps the
    (possibly truncated) text as an opaque string field of a new,
    always-valid JSON object, so the recorded bytes are kept as they came.
    A raw_json that already parses is returned unchanged.
    """
    if not raw_json:
        return raw_json
    try:
        json.loads(raw_json)
        return raw_json
    except json.JSONDecodeError:
        return json.dumps(
            {
                "note": "raw payload was truncated by the live tool's JSON size cap; repaired for recording storage",
                "raw_prefix": raw_json,
            },
            ensure_ascii=False,
        )


def _safe_result_for_recording(result: ToolResult) -> ToolResult:
    repaired = repair_truncated_raw_json(result.raw_json)
    if repaired == result.raw_json:
        return result
    return replace(result, raw_json=repaired)


def _execute_with_one_retry(runner: ToolRunner, query: ToolQuery, sleep_seconds: float) -> ToolResult:
    result = runner.execute(query)
    if result.status == "success":
        return result
    # Give the provider exactly one more chance after the same pacing
    # interval; never retry indefinitely.
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return runner.execute(query)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record live tool responses as offline tool recordings.")
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures"), help="Fixtures root directory")
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="Path to a JSON file listing {tool_type, parameters} queries to record",
    )
    parser.add_argument("--only", default="", help="Comma-separated tool_type allowlist, e.g. WHOIS_LOOKUP,TLS_CT_LOOKUP")
    parser.add_argument("--skip", default="", help="Comma-separated tool_type denylist")
    parser.add_argument(
        "--allow-crawl-hosts",
        default=",".join(DEFAULT_ALLOWED_CRAWL_HOSTS),
        help="Comma-separated WEB_CRAWL host allowlist (suffix match)",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to pace between calls (per-tool floors may raise this)")
    parser.add_argument("--resume", action="store_true", help="Skip queries that already have a recorded tool response file")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    settings = replace(settings, mode="live", fixtures_dir=args.fixtures, enable_real_tools=True)
    store = ToolRecordingStore(args.fixtures / "tools")
    runner = ToolRunner(settings, services=resolve_services(settings), recordings=store)

    allowed_hosts = parse_host_list(args.allow_crawl_hosts)
    only = parse_tool_set(args.only)
    skip = parse_tool_set(args.skip)

    entries = filter_entries(_load_queries(args.queries), only=only, skip=skip)

    rows: list[tuple[str, str, str]] = []
    resumed = 0
    for entry in entries:
        tool_type = entry["tool_type"]
        parameters = entry.get("parameters", {})
        if args.resume and is_already_recorded(store.root, tool_type, parameters):
            resumed += 1
            continue

        query = ToolQuery(entry.get("perspective", "Infrastructure"), tool_type, parameters)
        sleep_seconds = resolve_sleep_seconds(tool_type, args.sleep)

        if tool_type == "WEB_CRAWL":
            url = parameters.get("url", "")
            host = extract_host(url)
            if not is_allowed_crawl_host(host, allowed_hosts):
                result = _skip_untrusted_crawl_result(query, url, host)
            else:
                result = _execute_with_one_retry(runner, query, sleep_seconds)
        else:
            result = _execute_with_one_retry(runner, query, sleep_seconds)

        result = _safe_result_for_recording(result)
        store.record(tool_type, parameters, result, source_url=result.result_url)
        rows.append((tool_type, json.dumps(parameters, sort_keys=True), result.status))

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    tool_width = max((len(row[0]) for row in rows), default=len("tool_type"))
    param_width = max((len(row[1]) for row in rows), default=len("parameters"))
    print(f"{'tool_type':<{tool_width}}  {'parameters':<{param_width}}  status")
    for tool_type, parameters, status in rows:
        print(f"{tool_type:<{tool_width}}  {parameters:<{param_width}}  {status}")
    if resumed:
        print(f"skipped {resumed} already-recorded quer{'y' if resumed == 1 else 'ies'} (--resume)")

    counts = summarize_recordings(store.root)
    print()
    print(f"{'tool_type':<20}  success  unavailable  error")
    for tool_type in sorted(counts):
        row = counts[tool_type]
        print(f"{tool_type:<20}  {row['success']:<7}  {row['unavailable']:<11}  {row['error']}")

    summary_payload: dict[str, Any] = {
        "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "allow_crawl_hosts": list(allowed_hosts),
        "searxng_used": bool(os.environ.get("SEARXNG_BASE_URL")),
        **counts,
    }
    summary_path = args.fixtures / "tools" / "recording_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary_payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nrecorded {store.count()} tool response(s) under {store.root}")
    print(f"summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
