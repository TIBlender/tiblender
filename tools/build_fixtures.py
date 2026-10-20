#!/usr/bin/env python3
"""Build anonymized, 3-cycle collector fixtures from the released dataset.

Reads the released dataset's `raw_dataset/reports.jsonl` -- the same
8,288-report file the paper's evaluation tables are recomputed from -- and
produces API-shaped fixture files for the X, Reddit, Telegram, and Discord
fixture transports (`tiblender_runtime/collectors/fixtures.py`), plus a
manifest and a tool-query list for tool-response recording.

Every fixture post is an anonymized, defanged excerpt of one released
`source_posts.items[]` entry and keeps that entry's `source_ref`, so each
post in the bundle can be traced back to a report in the dataset.

Selection, windowing, and noise assignment live in
`tools/fixture_selection.py`; per-platform payload shapes live in
`tools/fixture_shapes.py`. This module is the CLI: it wires those together
and writes the fixture files, manifest, tool-query list, and README.

Usage:
    python3 tools/build_fixtures.py \\
        --dataset dataset/raw_dataset/reports.jsonl \\
        --out fixtures/collectors --cycles 3 --seed 20260909

Deterministic given the same input file and seed: no wall clock, no unseeded
randomness.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import random
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiblender_runtime.iocs import VALID_TLDS, refang  # noqa: E402
from tiblender_runtime.legitimate_hosts import REFERENCE_HOSTS  # noqa: E402
from tools.fixture_selection import (  # noqa: E402
    ALL_PLATFORMS,
    CYCLE_DAYS,
    Window,
    assign_noise,
    build_noise_pool,
    choose_band_start,
    compute_windows,
    counts_by_cycle_platform,
    grow_selection,
    load_dataset_reports,
    prepare_primary_posts,
    select_candidates,
)
from tools.fixture_shapes import (  # noqa: E402
    PreparedPost,
    base36_id,
    discord_guild_and_channel_responses,
    discord_messages_response,
    iso_z,
    reddit_response,
    telegram_channel_id,
    telegram_dialogs_response,
    telegram_messages_response,
    x_response,
)
from tools.anonymize import pseudonym  # noqa: E402
from tools.fixtures_readme import write_readme  # noqa: E402
from tools.harvest_tool_queries import (  # noqa: E402
    HARVESTED_FILENAME,
    ORIGIN_POST_DERIVED,
    load_harvested,
    merge_queries,
)

PSEUDONYM_SALT = "tiblender-fixture"

CATEGORY_SUBREDDIT = {
    "Phishing/Fraud": "phishing",
    "Malware/C2": "Malware",
    "Vulnerability": "netsec",
    "APT/Ransomware": "cybersecurity",
}
DEFAULT_SUBREDDIT = "netsec"

# Same ASCII-only boundary rationale as `tools/anonymize.py`: Python's `\b`
# treats CJK characters as word characters, so a plain `\b` misses a CVE
# id/IP/domain directly abutting Japanese text (common in this corpus).
_LEFT_BOUND = r"(?<![A-Za-z0-9_])"
_RIGHT_BOUND = r"(?![A-Za-z0-9_])"

_CVE_RE = re.compile(_LEFT_BOUND + r"CVE-\d{4}-\d{4,8}" + _RIGHT_BOUND, re.IGNORECASE)
_IP_RE = re.compile(_LEFT_BOUND + r"(?:\d{1,3}\.){3}\d{1,3}" + _RIGHT_BOUND)
_DOMAIN_RE = re.compile(_LEFT_BOUND + r"(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,24}" + _RIGHT_BOUND)
_URL_RE = re.compile(r"https?://[^\s)\]}>,\"']+", re.IGNORECASE)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_public_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _is_low_quality_crawl_url(url: str) -> bool:
    """True when `url` is not a good WEB_CRAWL candidate.

    Covers two cases: the URL still carries a source redaction placeholder
    (our own `_URL_RE`'s excluded-character class stops matching at a "]",
    so a URL like "https://medium.com/[HANDLE_REDACTED]/post" comes out
    truncated as "https://medium.com/[HANDLE_REDACTED" -- dropping any
    candidate containing "[", "]", or "REDACTED" catches that without
    needing a smarter URL regex), and the URL has no path beyond the host
    (crawling just a bare domain is rarely a useful WEB_CRAWL target here).
    Checked before `urlparse` so a bracket-bearing candidate never reaches
    it (a stray "[" can otherwise make `urlparse` misread the netloc as an
    IPv6 literal and raise).
    """
    if "[" in url or "]" in url or "REDACTED" in url.upper():
        return True
    parsed = urllib.parse.urlparse(url)
    path = parsed.path or ""
    return path in ("", "/")


# ---------------------------------------------------------------------------
# Fixture file assembly
# ---------------------------------------------------------------------------


def _build_platform_fixture(
    platform: str,
    cycle_posts: list[PreparedPost],
    window: Window,
    source_reports: list[str],
    notes: dict,
) -> dict:
    responses: list[dict] = []

    if platform == "x":
        responses.append(x_response(cycle_posts, salt=PSEUDONYM_SALT))

    elif platform == "reddit":
        by_subreddit: dict[str, list[PreparedPost]] = {}
        for post in cycle_posts:
            subreddit = CATEGORY_SUBREDDIT.get(post.category, DEFAULT_SUBREDDIT)
            by_subreddit.setdefault(subreddit, []).append(post)
        for subreddit in sorted(by_subreddit):
            responses.append(reddit_response(by_subreddit[subreddit], subreddit=subreddit, salt=PSEUDONYM_SALT))

    elif platform == "telegram":
        by_category: dict[str, list[PreparedPost]] = {}
        for post in cycle_posts:
            by_category.setdefault(post.category or "general", []).append(post)
        chats: list[dict] = []
        message_responses: list[dict] = []
        for category in sorted(by_category):
            channel_username = "tichan" + base36_id(f"{category}:{window.cycle_index}", PSEUDONYM_SALT, 8)
            channel_title = "channel-" + pseudonym(f"{category}:{window.cycle_index}", salt=PSEUDONYM_SALT)
            chats.append(
                {
                    "id": telegram_channel_id(channel_username, salt=PSEUDONYM_SALT),
                    "username": channel_username,
                    "title": channel_title,
                }
            )
            message_responses.append(
                telegram_messages_response(
                    by_category[category],
                    channel_username=channel_username,
                    salt=PSEUDONYM_SALT,
                )
            )
        # One joined-channel listing, then that channel's history.
        responses.append(telegram_dialogs_response(chats))
        responses.extend(message_responses)

    elif platform == "discord":
        guilds_response, channels_response = discord_guild_and_channel_responses()
        responses.append(guilds_response)
        responses.append(channels_response)
        responses.append(discord_messages_response(cycle_posts, salt=PSEUDONYM_SALT))

    return {
        "platform": platform,
        "window": {
            "start": iso_z(window.start),
            "end": iso_z(window.end),
            "cycle_index": window.cycle_index,
            "label": window.label,
        },
        "provenance": "anonymized_dataset_excerpt",
        "source_reports": sorted(source_reports),
        # Traceability: one entry per post replayed by this file, in the same
        # order, carrying the released `source_ref` it excerpts. Kept outside
        # `responses` so the replayed payloads stay exactly API-shaped.
        "posts": [source_ref_entry(post) for post in sorted(cycle_posts, key=_post_order)],
        "notes": notes,
        "responses": responses,
    }


def _post_order(post: PreparedPost) -> tuple:
    return (post.posted_at, post.post_ref)


def source_ref_entry(post: PreparedPost) -> dict:
    """The provenance record shipped for one fixture post."""
    return {
        "source_ref": post.post_ref,
        "platform": post.platform,
        "report_id": post.report_id,
        "kind": post.kind,
        "timestamp": iso_z(post.posted_at),
    }


def _dedup_sorted(entries: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for entry in entries:
        key = json.dumps(entry, sort_keys=True)
        seen[key] = entry
    return [json.loads(key) for key in sorted(seen)]


def _build_tool_queries(posts: list[PreparedPost]) -> list[dict]:
    domains: set[str] = set()
    ips: set[str] = set()
    cves: set[str] = set()
    urls: set[str] = set()

    for post in posts:
        refanged = refang(post.text)
        for match in _CVE_RE.finditer(refanged):
            cves.add(match.group(0).upper())
        for match in _IP_RE.finditer(refanged):
            if _is_public_ip(match.group(0)):
                ips.add(match.group(0))
        for match in _DOMAIN_RE.finditer(refanged):
            domain = match.group(0).lower().strip(".")
            if domain.rsplit(".", 1)[-1] not in VALID_TLDS:
                continue
            # REFERENCE_HOSTS, not the runtime's wider LEGITIMATE_DOMAINS: this
            # scan defines the committed query set, so it must not shrink when
            # the IoC allowlist grows and orphan the recordings behind it.
            if domain in REFERENCE_HOSTS or any(domain.endswith(f".{legit}") for legit in REFERENCE_HOSTS):
                continue
            domains.add(domain)
        # Refanged, exactly like the domain/IP/CVE scans above and like the
        # runtime's own `iocs.extract_iocs`: fixture text ships defanged, so
        # matching the raw text would silently drop every crawlable URL (and
        # with it its recorded tool response) while the runtime still queries
        # it.
        for match in _URL_RE.finditer(refanged):
            url = match.group(0).rstrip(".,;:!?)]}>\"'")
            if _is_low_quality_crawl_url(url):
                continue
            urls.add(url)

    entries: list[dict] = []
    for domain in domains:
        entries.append({"tool_type": "WHOIS_LOOKUP", "parameters": {"domain": domain}})
        entries.append({"tool_type": "PASSIVE_DNS_LOOKUP", "parameters": {"domain": domain}})
        entries.append({"tool_type": "TLS_CT_LOOKUP", "parameters": {"domain": domain}})
    for ip in ips:
        entries.append({"tool_type": "WHOIS_LOOKUP", "parameters": {"ip": ip}})
        entries.append({"tool_type": "PASSIVE_DNS_LOOKUP", "parameters": {"ip": ip}})
    for cve in cves:
        entries.append({"tool_type": "NVD_CVE_LOOKUP", "parameters": {"cve": cve}})
        # `query`, not `cve`: the recording key is the sha1 of the *named*
        # parameters (tool_fixtures.recording_key), and
        # perspectives._tool_parameter_sets issues GITHUB_SEARCH as
        # {"query": <cve>}. Keying these as {"cve": ...} made every recorded
        # GitHub response unreachable by construction -- a different key for
        # the same request. `tools.py::_github_search` reads
        # `query|cve|hash`, so the recorded payloads are identical either way.
        entries.append({"tool_type": "GITHUB_SEARCH", "parameters": {"query": cve}})
        entries.append({"tool_type": "SEARCH_ENGINE", "parameters": {"query": cve}})
    for url in urls:
        entries.append({"tool_type": "WEB_CRAWL", "parameters": {"url": url}})
    # Provenance for queries.json: everything above is visible in the
    # fixture post text, so a rebuild derives it without running anything.
    return [{**entry, "origin": ORIGIN_POST_DERIVED} for entry in _dedup_sorted(entries)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_harvested_path(out: Path) -> Path:
    """Locate harvested_queries.json for an --out anywhere on disk.

    An out-of-tree rebuild (`--out <scratch>/collectors`, the command that
    proves the bundle is byte-reproducible) has no `tools/`
    directory beside it. Resolving only there silently dropped the whole
    second pass and produced a smaller queries.json and a different
    fixtures/README.md -- a false mismatch. Fall back to the committed file in
    the package tree, and say out loud when there is none.
    """
    beside_out = out.parent / "tools" / HARVESTED_FILENAME
    if beside_out.exists():
        return beside_out
    in_package = Path(__file__).resolve().parent.parent / "fixtures" / "tools" / HARVESTED_FILENAME
    if in_package.exists():
        print(f"note: no {beside_out}; using the committed {in_package}")
        return in_package
    print(f"warning: no harvested queries file at {beside_out} or {in_package}; queries.json will be post-derived only")
    return beside_out


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="released dataset raw_dataset/reports.jsonl (the only fixture source)",
    )
    parser.add_argument("--out", type=Path, required=True, help="output collectors fixtures directory")
    parser.add_argument("--cycles", type=int, default=3, help="number of cycles to build")
    parser.add_argument("--seed", type=int, default=20260909, help="deterministic seed for noise selection")
    return parser.parse_args(argv)


def build(args: argparse.Namespace) -> dict:
    reports = load_dataset_reports(args.dataset)
    if not reports:
        raise ValueError(f"{args.dataset} contains no reports with dated source posts")

    band_start = choose_band_start(reports, args.cycles)
    windows = compute_windows(band_start, args.cycles)

    candidates = select_candidates(reports, windows)
    selected = grow_selection(candidates, windows)
    selected_report_ids = {candidate.report_id for candidate in selected}

    primary_posts = prepare_primary_posts(selected)
    primary_counts = counts_by_cycle_platform(primary_posts)

    rng = random.Random(args.seed)
    noise_pool = build_noise_pool(reports, selected_report_ids, windows)
    noise_posts = assign_noise(noise_pool, primary_counts, windows, rng)

    all_posts = primary_posts + noise_posts

    noise_notes_by_cycle: dict[int, list[str]] = {}
    for post in noise_posts:
        noise_notes_by_cycle.setdefault(post.cycle_index, []).append(post.post_ref)

    source_reports_by_cycle: dict[int, set[str]] = {window.cycle_index: set() for window in windows}
    for post in primary_posts:
        source_reports_by_cycle[post.cycle_index].add(post.report_id)

    manifest_cycles = []
    for window in windows:
        cycle_notes = {
            "noise_post_refs": sorted(noise_notes_by_cycle.get(window.cycle_index, [])),
        }
        cycle_counts: dict[str, int] = {}
        cycle_post_records: list[dict] = []
        for platform in ALL_PLATFORMS:
            platform_posts = [
                post for post in all_posts if post.cycle_index == window.cycle_index and post.platform == platform
            ]
            cycle_counts[platform] = len(platform_posts)
            fixture = _build_platform_fixture(
                platform,
                platform_posts,
                window,
                sorted(source_reports_by_cycle[window.cycle_index]),
                cycle_notes,
            )
            cycle_post_records.extend(fixture["posts"])
            _write_json(args.out / platform / f"{window.label}.json", fixture)
        manifest_cycles.append(
            {
                "cycle_index": window.cycle_index,
                "label": window.label,
                "window": {"start": iso_z(window.start), "end": iso_z(window.end)},
                "counts": cycle_counts,
                "total_posts": sum(cycle_counts.values()),
                "report_ids": sorted(source_reports_by_cycle[window.cycle_index]),
                # One record per fixture post in this cycle, so every post in
                # the bundle is resolvable to a released report and post.
                "posts": sorted(cycle_post_records, key=lambda record: (record["platform"], record["source_ref"])),
            }
        )

    manifest = {
        "generated_by": "tools/build_fixtures.py",
        "seed": args.seed,
        "cycles_requested": args.cycles,
        "cycle_days": CYCLE_DAYS,
        "collection_band": {
            "start": iso_z(windows[0].start),
            "end": iso_z(windows[-1].end),
        },
        # Provenance records the *file name* and content hash of the source,
        # never the author's absolute path: a shipped artifact must not carry
        # a hardcoded path or username from the machine that built it.
        "dataset_source": {
            "file": Path(args.dataset).name,
            "sha256": _sha256_file(args.dataset),
            # Reports that carry at least one dated source post on a collected
            # platform -- the population selection actually draws from, not the
            # file's full line count.
            "reports_with_dated_posts": len(reports),
        },
        "selected_report_ids": sorted(selected_report_ids),
        "cycles": manifest_cycles,
        "total_posts": len(all_posts),
        "primary_post_count": len(primary_posts),
        "noise_post_count": len(noise_posts),
    }
    _write_json(args.out / "manifest.json", manifest)

    post_derived_queries = _build_tool_queries(all_posts)
    # The builder can only see fixture *post text*. The pipeline also issues
    # free-text cluster search phrases and pivots on indicators found in
    # recorded evidence -- requests no static scan can predict. Those are
    # harvested from a deterministic offline run by
    # tools/harvest_tool_queries.py and committed alongside, so a rebuild on a
    # machine with no network still produces the full query set.
    harvested_path = _resolve_harvested_path(args.out)
    harvested = load_harvested(harvested_path)
    print(f"harvested tool queries: {len(harvested)} from {harvested_path}")
    tool_queries = merge_queries(post_derived_queries, harvested)
    _write_json(args.out.parent / "tools" / "queries.json", tool_queries)

    write_readme(args.out.parent / "README.md", manifest, tool_queries)

    print(
        f"band {windows[0].start.date()}..{windows[-1].end.date()}: selected {len(selected)} reports, "
        f"{len(all_posts)} total posts across {len(windows)} cycles"
    )
    for cycle in manifest_cycles:
        print(f"  {cycle['label']}: {cycle['counts']} (total {cycle['total_posts']})")
    print(f"wrote manifest to {args.out / 'manifest.json'}")
    print(
        f"wrote {len(tool_queries)} tool queries to {args.out.parent / 'tools' / 'queries.json'} "
        f"({len(post_derived_queries)} derived from post text, {len(harvested)} harvested from the offline run)"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
