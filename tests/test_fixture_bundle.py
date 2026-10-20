"""Validates the committed, generated fixture bundle under fixtures/collectors/.

These tests load the actual fixture files produced by
`tools/build_fixtures.py` (not synthetic in-test payloads) and check the
guarantees `tools/build_fixtures.py` is supposed to uphold: provenance,
contiguous increasing cycle windows, no leaked handles/social links in any
text field, per-cycle platform coverage, a minimum total post count,
traceability of every post back to the released dataset, and that every
platform's fixture actually parses into RawPosts whose `posted_at` falls
inside that cycle's window.

Nothing here reads the dataset itself: the code must be checkable
on its own, so provenance is verified against the bundled manifest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tiblender_runtime.collectors.discord import parse_messages as parse_discord_messages
from tiblender_runtime.collectors.fixtures import FixtureFile, load_fixture
from tiblender_runtime.collectors.reddit import parse_listing
from tiblender_runtime.collectors.telegram import ChannelRef, parse_messages as parse_telegram_messages
from tiblender_runtime.collectors.x import parse_search_payload
from tools.anonymize import _ALL_SOCIAL_HOSTS, _BARE_SOCIAL_HOST_RE

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "collectors"
_PLATFORMS = ("x", "reddit", "telegram", "discord")
_CYCLES = (1, 2, 3)

_HANDLE_RE = re.compile(r"@[A-Za-z0-9_]{4,}")
_MIN_TOTAL_POSTS = 90

# Every platform-hosted URL shape the package promises is stripped
# (README.md "social links", fixtures/README.md §Anonymization,
# docs/FIDELITY.md "no ... social-post URLs survive anonymization"). Driven
# off the anonymizer's own host list so the test tracks the documented
# policy instead of two hand-picked needles -- the two-needle version missed
# a live, signed `https://preview.redd.it/...?s=<sig>` media URL, which is a
# pointer straight back to the source post.
#
# `discord.com` is matched only in its invite form: the Discord *API* path
# (`https://discord.com/api/v10/...`) is part of the replayed response shape,
# not post text, and is not a pointer to any post.
_PLATFORM_URL_NEEDLES = tuple(
    sorted(
        {f"{host}/" for host in _ALL_SOCIAL_HOSTS if host != "discord.com"}
        | {f"//{host}" for host in _ALL_SOCIAL_HOSTS if host != "discord.com"}
        | {"discord.com/invite", "://discord.com/channels"}
    )
)

# The list the docs enumerate by name; assert-guarded below so shortening
# _ALL_SOCIAL_HOSTS can never silently shrink what this test covers.
_DOCUMENTED_PLATFORM_HOSTS = (
    "reddit.com",
    "redd.it",
    "x.com",
    "twitter.com",
    "t.co",
    "t.me",
    "telegram.me",
    "discord.gg",
    "discordapp.com",
    "linkedin.com",
    "lnkd.in",
)


def _fixture_path(platform: str, cycle: int) -> Path:
    return _FIXTURES_DIR / platform / f"cycle_{cycle:03d}.json"


def _load(platform: str, cycle: int) -> FixtureFile:
    return load_fixture(_fixture_path(platform, cycle))


def _raw_payload(platform: str, cycle: int) -> dict:
    return json.loads(_fixture_path(platform, cycle).read_text(encoding="utf-8"))


def _iter_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _posts_for(fixture: FixtureFile, platform: str) -> list:
    if platform == "x":
        posts = []
        for response in fixture.responses_for("tweets/search/recent"):
            query = response["request"].get("query", "")
            posts.extend(parse_search_payload(response["payload"], source="test", query=query))
        return posts

    if platform == "reddit":
        posts = []
        for response in fixture.responses_for("r/{sub}/new"):
            posts.extend(parse_listing(response["payload"], source="subreddit", query=""))
        return posts

    if platform == "telegram":
        posts = []
        for response in fixture.responses_for("messages"):
            channel_id = str(response["request"].get("channel_id", ""))
            channel = ChannelRef(id=channel_id, username="testchan", title="Test Channel", discovery="test")
            posts.extend(parse_telegram_messages(channel, response["payload"]))
        return posts

    if platform == "discord":
        posts = []
        guild_responses = fixture.responses_for("users/@me/guilds")
        channel_responses = fixture.responses_for("guilds/{id}/channels")
        guild = guild_responses[0]["payload"][0] if guild_responses and guild_responses[0]["payload"] else {}
        channel = (
            channel_responses[0]["payload"][0] if channel_responses and channel_responses[0]["payload"] else {}
        )
        for response in fixture.responses_for("channels/{id}/messages"):
            posts.extend(parse_discord_messages(guild, channel, response["payload"]))
        return posts

    raise ValueError(f"unknown platform: {platform}")


@pytest.mark.parametrize("platform", _PLATFORMS)
@pytest.mark.parametrize("cycle", _CYCLES)
def test_fixture_loads_with_expected_provenance(platform, cycle):
    fixture = _load(platform, cycle)
    assert fixture.platform == platform
    assert fixture.provenance == "anonymized_dataset_excerpt"
    assert fixture.window.cycle_index == cycle
    assert fixture.window.label == f"cycle_{cycle:03d}"


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_windows_are_contiguous_and_increasing_across_cycles(platform):
    windows = [_load(platform, cycle).window for cycle in _CYCLES]
    for earlier, later in zip(windows, windows[1:]):
        assert earlier.start < earlier.end
        assert earlier.end < later.start
        assert later.cycle_index == earlier.cycle_index + 1


@pytest.mark.parametrize("platform", _PLATFORMS)
@pytest.mark.parametrize("cycle", _CYCLES)
def test_no_handles_or_social_links_leak_into_any_text_field(platform, cycle):
    payload = _raw_payload(platform, cycle)
    for text in _iter_strings(payload):
        lowered = text.lower()
        assert not _HANDLE_RE.search(text), f"leaked handle in {platform}/cycle_{cycle:03d}: {text!r}"
        for needle in _PLATFORM_URL_NEEDLES:
            assert needle not in lowered, (
                f"leaked platform URL {needle!r} in {platform}/cycle_{cycle:03d}: {text!r}"
            )
        # The needles above all carry a "/" or a scheme, so a bare host in
        # prose ("see redd.it for the thread") slipped past them. The
        # anonymizer does strip that shape; this closes the gap in the guard.
        bare = _BARE_SOCIAL_HOST_RE.search(text)
        assert bare is None, (
            f"leaked bare platform host {bare.group(0)!r} in {platform}/cycle_{cycle:03d}: {text!r}"
        )


def test_the_platform_url_scan_covers_every_documented_platform_host():
    """Guards the guard: the needles must cover the hosts the docs name."""
    missing = [
        host
        for host in _DOCUMENTED_PLATFORM_HOSTS
        if f"{host}/" not in _PLATFORM_URL_NEEDLES and f"//{host}" not in _PLATFORM_URL_NEEDLES
    ]
    assert not missing, f"documented platform hosts absent from the fixture-text scan: {missing}"
    assert "discord.com/invite" in _PLATFORM_URL_NEEDLES
    # Not vacuous: the bare-host pattern really catches a scheme-less,
    # path-less mention, which the substring needles cannot see.
    assert _BARE_SOCIAL_HOST_RE.search("see redd.it for the thread") is not None
    assert _BARE_SOCIAL_HOST_RE.search("no platform host here") is None


def test_every_cycle_has_at_least_one_post_for_every_platform():
    for cycle in _CYCLES:
        for platform in _PLATFORMS:
            fixture = _load(platform, cycle)
            posts = _posts_for(fixture, platform)
            assert len(posts) >= 1, f"{platform}/cycle_{cycle:03d} has no posts"


def test_total_posts_across_bundle_is_at_least_90():
    total = 0
    for platform in _PLATFORMS:
        for cycle in _CYCLES:
            total += len(_posts_for(_load(platform, cycle), platform))
    assert total >= _MIN_TOTAL_POSTS


@pytest.mark.parametrize("platform", _PLATFORMS)
@pytest.mark.parametrize("cycle", _CYCLES)
def test_platform_fixture_parses_into_posts_inside_window(platform, cycle):
    fixture = _load(platform, cycle)
    posts = _posts_for(fixture, platform)
    assert len(posts) >= 1

    dated_posts = [post for post in posts if post.posted_at]
    assert dated_posts, f"{platform}/cycle_{cycle:03d} produced no dated RawPosts"
    for post in dated_posts:
        assert fixture.window.contains(post.posted_at), (
            f"{platform}/cycle_{cycle:03d} post {post.post_id} posted_at={post.posted_at!r} "
            f"outside window {fixture.window.start}..{fixture.window.end}"
        )


def _endpoints(platform: str, cycle: int) -> set[str]:
    payload = _raw_payload(platform, cycle)
    return {response["request"]["endpoint"] for response in payload["responses"]}


@pytest.mark.parametrize("cycle", _CYCLES)
def test_telegram_fixture_replays_joined_dialogs_not_keyword_search(cycle):
    assert _endpoints("telegram", cycle) == {"dialogs", "messages"}

    fixture = _load("telegram", cycle)
    dialogs = fixture.responses_for("dialogs")
    assert len(dialogs) == 1, "exactly one joined-channel listing per cycle"
    chats = dialogs[0]["payload"]["chats"]
    assert chats
    for chat in chats:
        assert set(chat) == {"id", "username", "title"}
    # Every listed channel has its own message history in the same fixture.
    listed = {str(chat["id"]) for chat in chats}
    with_messages = {
        str(response["request"]["channel_id"]) for response in fixture.responses_for("messages")
    }
    assert listed == with_messages


@pytest.mark.parametrize("cycle", _CYCLES)
def test_discord_fixture_has_no_discovery_endpoints(cycle):
    assert _endpoints("discord", cycle) == {
        "users/@me/guilds",
        "guilds/{id}/channels",
        "channels/{id}/messages",
    }


@pytest.mark.parametrize("cycle", _CYCLES)
def test_the_joined_community_fixtures_replay_through_their_own_transports(cycle):
    """The fixture transports the runtime actually uses must read these files."""
    from tiblender_runtime.collectors.discord import DiscordFixtureTransport
    from tiblender_runtime.collectors.telegram import TelegramFixtureTransport

    telegram = TelegramFixtureTransport(_load("telegram", cycle))
    channels = telegram.dialog_channels()
    assert channels
    assert all(channel.discovery == "joined" for channel in channels)
    assert any(telegram.channel_messages(channel, min_id=0, limit=100) for channel in channels)

    discord = DiscordFixtureTransport(_load("discord", cycle))
    guilds = discord.my_guilds()
    assert guilds
    guild_channels = discord.guild_channels(str(guilds[0]["id"]))
    assert guild_channels
    assert discord.channel_messages(str(guild_channels[0]["id"]), after=None, limit=100)


def test_discord_message_ids_increase_with_time_across_cycles():
    """A Discord snowflake encodes its own creation timestamp, so ids only ever
    grow -- that is what makes the `after` watermark a safe incremental cursor.
    All three cycles replay the same channel, so a fixture whose ids did not
    grow with time would have a later cycle's messages silently suppressed by
    an earlier cycle's watermark."""
    seen: list[tuple[int, str]] = []
    for cycle in _CYCLES:
        payloads = [
            response["payload"]
            for response in _load("discord", cycle).responses
            if response["request"]["endpoint"] == "channels/{id}/messages"
        ]
        for payload in payloads:
            for message in payload:
                seen.append((int(message["id"]), message["timestamp"]))

    by_time = sorted(seen, key=lambda item: item[1])
    ids_in_time_order = [identifier for identifier, _ in by_time]
    assert ids_in_time_order == sorted(ids_in_time_order), (
        "discord fixture message ids are not monotone in time: "
        f"{by_time}"
    )
    assert len(set(ids_in_time_order)) == len(ids_in_time_order)


@pytest.mark.parametrize("cycle", _CYCLES)
def test_discord_fixture_transport_honours_an_after_watermark(cycle):
    """Every cycle after the first is replayed with the previous cycle's
    watermark. The transport must return exactly the recorded messages whose id
    is greater than it -- not nothing, which is what matching the watermark
    against the recorded request keys used to produce."""
    from tiblender_runtime.collectors.discord import DiscordFixtureTransport

    discord = DiscordFixtureTransport(_load("discord", cycle))
    guild_id = str(discord.my_guilds()[0]["id"])
    channel_id = str(discord.guild_channels(guild_id)[0]["id"])

    recorded = discord.channel_messages(channel_id, after=None, limit=100)
    ids = sorted(int(message["id"]) for message in recorded)
    assert len(ids) >= 2

    watermark = ids[0]
    kept = discord.channel_messages(channel_id, after=str(watermark), limit=100)
    assert [int(message["id"]) for message in kept] == [i for i in ids if i > watermark]
    assert all(int(message["id"]) > watermark for message in kept)

    # A watermark at or above the newest recorded id legitimately means "no new
    # messages"; anything below it must still replay.
    assert discord.channel_messages(channel_id, after=str(ids[-1]), limit=100) == []


# ---------------------------------------------------------------------------
# Dataset traceability
# ---------------------------------------------------------------------------

_MANIFEST_PATH = _FIXTURES_DIR / "manifest.json"
# `TIBLENDER-YYYY-MM-DD-<hex>` with an optional `-<hex>` suffix, the released
# report-id shape; `source_ref` is that id plus a `:p<N>` post suffix.
_REPORT_ID_RE = re.compile(r"^TIBLENDER-\d{4}-\d{2}-\d{2}-[0-9a-f]{6,32}(?:-[0-9a-f]{6,32})?$")
# `source_ref` is a released report id plus one or more colon-separated post
# selectors (`:p3`, `:bulk:p0`, ...), so it is matched more loosely than an id.
_SOURCE_REF_RE = re.compile(r"^TIBLENDER-[0-9A-Za-z-]+(?::[0-9A-Za-z-]+)+$")


def _manifest() -> dict:
    assert _MANIFEST_PATH.is_file(), f"missing fixture manifest: {_MANIFEST_PATH}"
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_report_ids_are_well_formed_released_ids():
    manifest = _manifest()
    selected = manifest["selected_report_ids"]
    assert selected, "manifest lists no selected report ids"
    for report_id in selected:
        assert _REPORT_ID_RE.match(report_id), f"malformed released report id in manifest: {report_id!r}"
    assert len(set(selected)) == len(selected), "manifest repeats a selected report id"


def test_manifest_records_the_dataset_source_by_name_only():
    source = _manifest()["dataset_source"]
    assert source["file"] == "reports.jsonl"
    assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"]), "dataset sha256 is not a sha256 digest"
    # A shipped artifact must never carry the build machine's path.
    assert "/" not in source["file"]


@pytest.mark.parametrize("platform", _PLATFORMS)
@pytest.mark.parametrize("cycle", _CYCLES)
def test_every_fixture_post_is_traceable_to_the_manifest(platform, cycle):
    manifest = _manifest()
    by_cycle = {entry["cycle_index"]: entry for entry in manifest["cycles"]}
    manifest_refs = {record["source_ref"] for record in by_cycle[cycle]["posts"]}
    selected = set(manifest["selected_report_ids"])

    payload = _raw_payload(platform, cycle)
    records = payload["posts"]
    assert records, f"{platform}/cycle_{cycle:03d} carries no post provenance records"
    for record in records:
        source_ref = record["source_ref"]
        assert _SOURCE_REF_RE.match(source_ref), f"malformed source_ref {source_ref!r}"
        assert source_ref in manifest_refs, f"{source_ref!r} is not listed in the manifest"
        assert record["platform"] == platform
        assert _REPORT_ID_RE.match(record["report_id"]), f"malformed report id {record['report_id']!r}"
        # A released `source_ref` is not always prefixed with the id of the
        # report that cites it (clusters are re-keyed as they grow), so the
        # binding checked here is the manifest's, not a string prefix.
        if record["kind"] == "primary":
            assert record["report_id"] in selected, (
                f"primary post {source_ref!r} cites report {record['report_id']!r}, "
                "which the manifest does not list as selected"
            )
        else:
            assert record["kind"] == "noise", f"unknown post kind {record['kind']!r}"
            assert record["report_id"] not in selected


def test_the_fixture_files_and_the_manifest_agree_on_every_post():
    manifest = _manifest()
    from_files: set[tuple[int, str]] = set()
    for cycle in _CYCLES:
        for platform in _PLATFORMS:
            for record in _raw_payload(platform, cycle)["posts"]:
                from_files.add((cycle, record["source_ref"]))
    from_manifest = {
        (entry["cycle_index"], record["source_ref"])
        for entry in manifest["cycles"]
        for record in entry["posts"]
    }
    assert from_files == from_manifest
    assert len(from_files) == manifest["total_posts"]


def test_no_fixture_post_source_ref_is_reused_across_cycles():
    seen: dict[str, int] = {}
    for cycle in _CYCLES:
        for platform in _PLATFORMS:
            for record in _raw_payload(platform, cycle)["posts"]:
                source_ref = record["source_ref"]
                assert source_ref not in seen, (
                    f"{source_ref!r} appears in cycle {seen[source_ref]} and cycle {cycle}"
                )
                seen[source_ref] = cycle
