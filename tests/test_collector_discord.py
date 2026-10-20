"""Tests for the Discord collector: message parsing, guild/channel selection and
caps, incremental watermarks, the fixture transport, and live-transport header
selection / rate-limit handling."""

import io
import logging
import random
import urllib.error
from datetime import datetime

import pytest

from tiblender_runtime.collectors.base import CollectionWindow, CollectorError, QueryPlan
from tiblender_runtime.collectors.discord import (
    MAX_MESSAGES_PER_CHANNEL_PER_CYCLE,
    TEXT_CHANNEL_TYPES,
    DiscordCollector,
    DiscordFixtureTransport,
    DiscordLiveTransport,
    discord_auth_headers,
    parse_messages,
)
from tiblender_runtime.collectors.fixtures import FixtureFile
from tiblender_runtime.collectors.http import HttpClient, HttpError


class _FakeTransport:
    def __init__(self, *, guilds=None, channels=None, messages=None, channel_errors=None):
        self._guilds = guilds or []
        self._channels = channels or {}
        self._messages = messages or {}
        self._channel_errors = channel_errors or {}
        self.calls = []

    def my_guilds(self):
        return self._guilds

    def guild_channels(self, guild_id):
        return self._channels.get(guild_id, [])

    def channel_messages(self, channel_id, *, after, limit):
        self.calls.append((channel_id, after, limit))
        if channel_id in self._channel_errors:
            raise HttpError(self._channel_errors[channel_id], "denied")
        pages = self._messages.get(channel_id, [])
        index = (
            0
            if after is None
            else next((i + 1 for i, page in enumerate(pages) if page and page[-1]["id"] == after), len(pages))
        )
        if index >= len(pages):
            return []
        return pages[index]


def _window(start="2026-03-20T09:00:00Z", end="2026-03-20T10:00:00Z", cycle_index=1):
    return CollectionWindow(
        start=datetime.fromisoformat(start.replace("Z", "+00:00")),
        end=datetime.fromisoformat(end.replace("Z", "+00:00")),
        cycle_index=cycle_index,
        label=f"cycle_{cycle_index:03d}",
    )


def _message(
    message_id,
    *,
    content="hello",
    bot=False,
    timestamp="2026-03-20T09:10:00.123000+00:00",
    attachments=None,
    username="alice",
):
    return {
        "id": str(message_id),
        "content": content,
        "timestamp": timestamp,
        "author": {"username": username, "bot": bot},
        "attachments": attachments or [],
    }


def _guild(guild_id="10", name="Sec Guild"):
    return {"id": guild_id, "name": name}


def _channel(channel_id="100", name="general", channel_type=0):
    return {"id": channel_id, "name": name, "type": channel_type}


# ---------------------------------------------------------------------------
# parse_messages
# ---------------------------------------------------------------------------


class TestParseMessages:
    def test_skips_bot_messages_and_empty_content(self):
        messages = [
            _message(1, content="real message"),
            _message(2, bot=True, content="bot message"),
            _message(3, content="   "),
        ]
        posts = parse_messages(_guild(), _channel(), messages)
        assert [p.external_id for p in posts] == ["100:1"]

    def test_url_and_external_id(self):
        posts = parse_messages(_guild(guild_id="10"), _channel(channel_id="100"), [_message(42)])
        assert posts[0].external_id == "100:42"
        assert posts[0].url == "https://discord.com/channels/10/100/42"
        assert posts[0].author_ref == "alice"

    def test_timestamp_normalized_to_z_seconds(self):
        posts = parse_messages(_guild(), _channel(), [_message(1, timestamp="2026-03-20T09:10:00.123000+00:00")])
        assert posts[0].posted_at == "2026-03-20T09:10:00Z"

    def test_attachments_classified_image_vs_file(self):
        attachments = [
            {"url": "https://cdn/img.png", "content_type": "image/png"},
            {"url": "https://cdn/doc.pdf", "content_type": "application/pdf", "ocr_text": "scanned text"},
        ]
        posts = parse_messages(_guild(), _channel(), [_message(1, attachments=attachments)])
        assert posts[0].attachments == [
            {"type": "image", "url": "https://cdn/img.png", "ocr_text": ""},
            {"type": "file", "url": "https://cdn/doc.pdf", "ocr_text": "scanned text"},
        ]

    def test_metadata_shape(self):
        posts = parse_messages(_guild(), _channel(), [_message(1)])
        # Every guild now reaches the parser from /users/@me/guilds, so the
        # provenance is a constant and there is no query to record.
        assert posts[0].metadata == {
            "source": "channel",
            "guild_id": "10",
            "guild_name": "Sec Guild",
            "channel_id": "100",
            "channel_name": "general",
        }

    def test_guild_payload_keys_are_never_read_as_provenance(self):
        posts = parse_messages(
            {"id": "10", "name": "Sec Guild", "source": "discovery", "query": "stealer logs"},
            _channel(),
            [_message(1)],
        )
        assert posts[0].metadata["source"] == "channel"
        assert "query" not in posts[0].metadata

    def test_sorted_by_message_id_as_int(self):
        messages = [_message(20), _message(3), _message(100)]
        posts = parse_messages(_guild(), _channel(), messages)
        assert [p.external_id for p in posts] == ["100:3", "100:20", "100:100"]


# ---------------------------------------------------------------------------
# discord_auth_headers
# ---------------------------------------------------------------------------


class TestDiscordAuthHeaders:
    def test_user_token_is_sent_bare_with_a_browser_user_agent(self):
        headers = discord_auth_headers(user_token="usr", bot_token="")
        assert headers["Authorization"] == "usr"
        assert headers["User-Agent"].startswith("Mozilla/5.0")

    def test_user_token_is_preferred_over_a_bot_token(self):
        headers = discord_auth_headers(user_token="usr", bot_token="bot")
        assert headers["Authorization"] == "usr"

    def test_bot_token_is_the_fallback(self):
        headers = discord_auth_headers(user_token="", bot_token="bot")
        assert headers["Authorization"] == "Bot bot"
        assert headers["User-Agent"] == "DiscordBot (tiblender, 1.0)"

    def test_blank_user_token_falls_back_to_the_bot_token(self):
        headers = discord_auth_headers(user_token="   ", bot_token="bot")
        assert headers["Authorization"] == "Bot bot"

    def test_no_credential_raises_collector_error(self):
        with pytest.raises(CollectorError, match="TIBLENDER_DISCORD_USER_TOKEN"):
            discord_auth_headers(user_token="", bot_token="")


# ---------------------------------------------------------------------------
# DiscordCollector.collect
# ---------------------------------------------------------------------------


class TestDiscordCollectorCollect:
    def test_filters_guilds_by_guild_filter(self):
        transport = _FakeTransport(
            guilds=[_guild("1", name="A"), _guild("2", name="B")],
            channels={"1": [_channel("11")], "2": [_channel("22")]},
            messages={"11": [[_message(1)]], "22": [[_message(2)]]},
        )
        collector = DiscordCollector(transport, guild_filter=["1"])
        result = collector.collect(_window(), QueryPlan(queries=[]))
        assert result.stats["guilds"] == 1
        assert {p.metadata["guild_id"] for p in result.posts} == {"1"}

    def test_guild_channel_filter_line_restricts_channels(self):
        transport = _FakeTransport(
            guilds=[_guild("1", name="A")],
            channels={"1": [_channel("11"), _channel("12")]},
            messages={"11": [[_message(1)]], "12": [[_message(2)]]},
        )
        collector = DiscordCollector(transport, guild_filter=["1:11"])
        result = collector.collect(_window(), QueryPlan(queries=[]))
        assert result.stats["channels"] == 1
        assert {p.metadata["channel_id"] for p in result.posts} == {"11"}

    def test_no_guild_filter_collects_every_joined_guild(self):
        transport = _FakeTransport(
            guilds=[_guild("2", name="B"), _guild("1", name="A")],
            channels={"1": [_channel("11")], "2": [_channel("22")]},
            messages={"11": [[_message(1)]], "22": [[_message(2)]]},
        )
        collector = DiscordCollector(transport, guild_filter=[])
        result = collector.collect(_window(), QueryPlan(queries=[]))
        assert result.stats["guilds"] == 2
        assert {p.metadata["guild_id"] for p in result.posts} == {"1", "2"}

    def test_max_guilds_caps_the_run_in_guild_id_order(self):
        transport = _FakeTransport(
            guilds=[_guild("3"), _guild("1"), _guild("2")],
            channels={"1": [_channel("11")], "2": [_channel("22")], "3": [_channel("33")]},
            messages={"11": [[_message(1)]], "22": [[_message(2)]], "33": [[_message(3)]]},
        )
        collector = DiscordCollector(transport, guild_filter=[], max_guilds=2)
        result = collector.collect(_window(), QueryPlan(queries=[]))
        assert result.stats["guilds"] == 2
        assert {p.metadata["guild_id"] for p in result.posts} == {"1", "2"}

    def test_max_guilds_zero_means_no_cap(self):
        transport = _FakeTransport(
            guilds=[_guild("1"), _guild("2"), _guild("3")],
            channels={"1": [_channel("11")], "2": [_channel("22")], "3": [_channel("33")]},
        )
        collector = DiscordCollector(transport, guild_filter=[], max_guilds=0)
        result = collector.collect(_window(), QueryPlan(queries=[]))
        assert result.stats["guilds"] == 3

    def test_max_channels_per_guild_caps_each_guild_in_channel_id_order(self):
        transport = _FakeTransport(
            guilds=[_guild("1"), _guild("2")],
            channels={
                "1": [_channel("13"), _channel("11"), _channel("12")],
                "2": [_channel("21"), _channel("22")],
            },
            messages={cid: [[_message(int(cid))]] for cid in ("11", "12", "13", "21", "22")},
        )
        collector = DiscordCollector(transport, guild_filter=[], max_channels_per_guild=2)
        result = collector.collect(_window(), QueryPlan(queries=[]))
        assert result.stats["channels"] == 4
        assert {p.metadata["channel_id"] for p in result.posts} == {"11", "12", "21", "22"}

    def test_max_channels_per_guild_zero_means_no_cap(self):
        transport = _FakeTransport(
            guilds=[_guild("1")],
            channels={"1": [_channel("11"), _channel("12"), _channel("13")]},
        )
        collector = DiscordCollector(transport, guild_filter=[], max_channels_per_guild=0)
        result = collector.collect(_window(), QueryPlan(queries=[]))
        assert result.stats["channels"] == 3

    def test_only_text_channel_types_are_kept(self):
        transport = _FakeTransport(
            guilds=[_guild("1")],
            channels={
                "1": [
                    _channel("11", channel_type=0),
                    _channel("12", channel_type=2),
                    _channel("13", channel_type=5),
                ]
            },
            messages={"11": [[_message(1)]], "13": [[_message(2)]]},
        )
        collector = DiscordCollector(transport, guild_filter=[])
        result = collector.collect(_window(), QueryPlan(queries=[]))
        assert result.stats["channels"] == 2
        assert TEXT_CHANNEL_TYPES == {0, 5}

    def test_collect_ignores_the_query_plan_entirely(self):
        from tiblender_runtime.collectors.base import CollectionQuery

        transport = _FakeTransport(
            guilds=[_guild("1")], channels={"1": [_channel("11")]}, messages={"11": [[_message(1)]]}
        )
        collector = DiscordCollector(transport, guild_filter=[])
        plan = QueryPlan(queries=[CollectionQuery("discord", "phishing kit", "static")])
        result = collector.collect(_window(), plan)
        assert [p.external_id for p in result.posts] == ["11:1"]

    def test_after_comes_from_state_last_message_id(self):
        transport = _FakeTransport(
            guilds=[_guild("1")],
            channels={"1": [_channel("11")]},
            messages={"11": [[_message(5), _message(6)]]},
        )
        collector = DiscordCollector(
            transport, guild_filter=[], state={"last_message_id": {"11": "3"}}
        )
        collector.collect(_window(), QueryPlan(queries=[]))
        assert transport.calls[0] == ("11", "3", 100)

    def test_window_filters_out_of_range_posts(self):
        transport = _FakeTransport(
            guilds=[_guild("1")],
            channels={"1": [_channel("11")]},
            messages={
                "11": [
                    [
                        _message(1, timestamp="2026-03-20T08:00:00+00:00"),
                        _message(2, timestamp="2026-03-20T09:30:00+00:00"),
                    ]
                ]
            },
        )
        collector = DiscordCollector(transport, guild_filter=[])
        result = collector.collect(_window(), QueryPlan(queries=[]))
        assert [p.external_id for p in result.posts] == ["11:2"]

    def test_pagination_stops_below_page_size_and_updates_last_message_id(self):
        transport = _FakeTransport(
            guilds=[_guild("1")],
            channels={"1": [_channel("11")]},
            messages={"11": [[_message(1), _message(2)], [_message(3)]]},
        )
        collector = DiscordCollector(transport, guild_filter=[], max_posts_per_channel=2)
        result = collector.collect(_window(), QueryPlan(queries=[]))
        assert result.stats["pages"] == 2
        assert [p.external_id for p in result.posts] == ["11:1", "11:2", "11:3"]
        assert collector.export_state() == {"last_message_id": {"11": "3"}}

    def test_pagination_stops_after_three_pages_even_if_full(self):
        def full_page(start):
            return [_message(i) for i in range(start, start + 2)]

        transport = _FakeTransport(
            guilds=[_guild("1")],
            channels={"1": [_channel("11")]},
            messages={"11": [full_page(1), full_page(3), full_page(5), full_page(7)]},
        )
        collector = DiscordCollector(transport, guild_filter=[], max_posts_per_channel=2)
        result = collector.collect(_window(), QueryPlan(queries=[]))
        assert result.stats["pages"] == 3
        assert len(transport.calls) == 3

    def test_channel_403_is_recorded_and_does_not_crash(self):
        transport = _FakeTransport(
            guilds=[_guild("1")],
            channels={"1": [_channel("11"), _channel("12")]},
            messages={"12": [[_message(1)]]},
            channel_errors={"11": 403},
        )
        collector = DiscordCollector(transport, guild_filter=[])
        result = collector.collect(_window(), QueryPlan(queries=[]))
        assert result.errors == ["channel 11: HTTP 403"]
        assert result.stats["errors"] == 1
        assert [p.external_id for p in result.posts] == ["12:1"]

    def test_channel_404_is_recorded_and_does_not_crash(self):
        transport = _FakeTransport(
            guilds=[_guild("1")], channels={"1": [_channel("11")]}, channel_errors={"11": 404}
        )
        collector = DiscordCollector(transport, guild_filter=[])
        result = collector.collect(_window(), QueryPlan(queries=[]))
        assert result.errors == ["channel 11: HTTP 404"]

    def test_non_404_403_channel_error_propagates(self):
        transport = _FakeTransport(
            guilds=[_guild("1")], channels={"1": [_channel("11")]}, channel_errors={"11": 500}
        )
        collector = DiscordCollector(transport, guild_filter=[])
        with pytest.raises(HttpError):
            collector.collect(_window(), QueryPlan(queries=[]))

    def test_configured_guild_that_is_not_joined_is_recorded_as_an_error(self):
        """A stale id in config/discord_guilds.txt must not produce a silent
        zero-post run, exactly as an unresolvable Telegram channel does not."""
        transport = _FakeTransport(
            guilds=[_guild("1")], channels={"1": [_channel("11")]}, messages={"11": [[_message(1)]]}
        )
        collector = DiscordCollector(transport, guild_filter=["1", "999", "42:7"])
        result = collector.collect(_window(), QueryPlan(queries=[]))

        assert result.errors == [
            "guild 42: not joined or not visible to this account",
            "guild 999: not joined or not visible to this account",
        ]
        assert result.stats["errors"] == 2
        assert result.stats["guilds"] == 1
        assert [p.external_id for p in result.posts] == ["11:1"]

    def test_no_guild_filter_reports_no_missing_guild_errors(self):
        transport = _FakeTransport(guilds=[_guild("1")], channels={"1": [_channel("11")]})
        result = DiscordCollector(transport, guild_filter=[]).collect(_window(), QueryPlan(queries=[]))
        assert result.errors == []

    def test_per_channel_page_bound_is_three_pages_of_one_hundred(self):
        assert MAX_MESSAGES_PER_CHANNEL_PER_CYCLE == 300

    def test_stats_keys_exact(self):
        transport = _FakeTransport(
            guilds=[_guild("1")], channels={"1": [_channel("11")]}, messages={"11": [[_message(1)]]}
        )
        collector = DiscordCollector(transport, guild_filter=[])
        result = collector.collect(_window(), QueryPlan(queries=[]))
        assert set(result.stats.keys()) == {"guilds", "channels", "posts", "pages", "errors"}

    def test_mode_is_carried_into_result(self):
        transport = _FakeTransport(guilds=[])
        collector = DiscordCollector(transport, guild_filter=[], mode="fixture")
        result = collector.collect(_window(), QueryPlan(queries=[]))
        assert result.mode == "fixture"


# ---------------------------------------------------------------------------
# DiscordFixtureTransport
# ---------------------------------------------------------------------------


class TestDiscordFixtureTransport:
    def _fixture(self):
        # One recorded payload per channel, exactly as tools/build_fixtures.py
        # emits it: the `after` watermark is applied in code, never matched on
        # the recorded request (no recorded request carries an `after` key).
        return FixtureFile(
            platform="discord",
            window=_window(),
            provenance="recorded:test",
            responses=[
                {"request": {"endpoint": "users/@me/guilds"}, "payload": [{"id": "1", "name": "A"}]},
                {
                    "request": {"endpoint": "guilds/{id}/channels", "guild_id": "1"},
                    "payload": [{"id": "11", "type": 0}],
                },
                {
                    "request": {"endpoint": "channels/{id}/messages", "channel_id": "11"},
                    "payload": [_message(30), _message(10), _message(20)],
                },
            ],
        )

    def test_each_endpoint_matched(self):
        transport = DiscordFixtureTransport(self._fixture())
        assert transport.my_guilds() == [{"id": "1", "name": "A"}]
        assert transport.guild_channels("1") == [{"id": "11", "type": 0}]
        assert transport.channel_messages("11", after=None, limit=100) == [
            _message(10),
            _message(20),
            _message(30),
        ]

    def test_after_watermark_filters_the_recorded_payload_by_message_id(self):
        transport = DiscordFixtureTransport(self._fixture())

        assert transport.channel_messages("11", after="10", limit=100) == [_message(20), _message(30)]
        assert transport.channel_messages("11", after="20", limit=100) == [_message(30)]
        assert transport.channel_messages("11", after="30", limit=100) == []

    def test_a_watermark_that_is_not_in_the_payload_still_filters_by_value(self):
        transport = DiscordFixtureTransport(self._fixture())

        assert transport.channel_messages("11", after="15", limit=100) == [_message(20), _message(30)]

    def test_limit_caps_the_replayed_page_oldest_first(self):
        transport = DiscordFixtureTransport(self._fixture())

        assert transport.channel_messages("11", after=None, limit=2) == [_message(10), _message(20)]

    def test_unknown_ids_replay_as_empty(self):
        transport = DiscordFixtureTransport(self._fixture())
        assert transport.guild_channels("999") == []
        assert transport.channel_messages("999", after=None, limit=100) == []

    def test_a_non_numeric_watermark_names_the_channel_and_the_value(self):
        """A corrupted history/collector_state.json must not surface as a bare
        ValueError from an int() deep inside the replay path."""
        transport = DiscordFixtureTransport(self._fixture())

        with pytest.raises(CollectorError, match="11") as error:
            transport.channel_messages("11", after="not-a-snowflake", limit=100)
        assert "not-a-snowflake" in str(error.value)

    def test_a_non_numeric_recorded_message_id_names_the_channel_and_the_id(self):
        fixture = self._fixture()
        fixture.responses[-1]["payload"] = [_message("abc")]
        transport = DiscordFixtureTransport(fixture)

        with pytest.raises(CollectorError, match="11") as error:
            transport.channel_messages("11", after=None, limit=100)
        assert "abc" in str(error.value)

    def test_a_non_positive_limit_means_no_cap_explicitly(self):
        transport = DiscordFixtureTransport(self._fixture())

        assert transport.channel_messages("11", after=None, limit=0) == [
            _message(10),
            _message(20),
            _message(30),
        ]

    def test_the_fixture_transport_has_no_discovery_surface(self):
        transport = DiscordFixtureTransport(self._fixture())
        for retired in ("invite", "directory_search", "directory_outcomes"):
            assert not hasattr(transport, retired)


# ---------------------------------------------------------------------------
# DiscordLiveTransport
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self, n=-1):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _live(*, user_token="", bot_token="", opener=None, sleeper=None, **kwargs):
    client = HttpClient(opener=opener or (lambda request, timeout=0: _Resp(b"[]")), sleeper=lambda s: None)
    return DiscordLiveTransport(
        client,
        user_token=user_token,
        bot_token=bot_token,
        sleeper=sleeper if sleeper is not None else (lambda s: None),
        **kwargs,
    )


class TestDiscordLiveTransport:
    def test_bot_authorization_header_sent(self):
        seen = {}

        def opener(request, timeout=0):
            seen["auth"] = request.get_header("Authorization")
            seen["ua"] = request.get_header("User-agent")
            return _Resp(b"[]")

        transport = _live(bot_token="tok", opener=opener)
        transport.my_guilds()
        assert seen["auth"] == "Bot tok"
        assert seen["ua"] == "DiscordBot (tiblender, 1.0)"
        assert transport.uses_user_token is False

    def test_user_token_header_is_sent_bare_with_a_browser_user_agent(self):
        seen = {}

        def opener(request, timeout=0):
            seen["auth"] = request.get_header("Authorization")
            seen["ua"] = request.get_header("User-agent")
            return _Resp(b"[]")

        transport = _live(user_token="usr", bot_token="tok", opener=opener)
        transport.my_guilds()
        assert seen["auth"] == "usr"
        assert seen["ua"].startswith("Mozilla/5.0")
        assert transport.uses_user_token is True

    def test_missing_credentials_raise_before_any_request(self):
        with pytest.raises(CollectorError):
            _live()

    def test_polite_delay_between_two_and_five_seconds_before_each_request(self):
        slept: list[float] = []
        transport = _live(bot_token="tok", sleeper=slept.append, rng=random.Random(7))
        transport.my_guilds()
        transport.guild_channels("1")
        transport.channel_messages("11", after=None, limit=100)
        assert len(slept) == 3
        assert all(2.0 <= delay <= 5.0 for delay in slept)

    def test_channel_messages_passes_after_and_limit_as_query_params(self):
        seen = {}

        def opener(request, timeout=0):
            seen["url"] = request.full_url
            return _Resp(b"[]")

        transport = _live(bot_token="tok", opener=opener)
        transport.channel_messages("11", after="42", limit=25)
        assert "channels/11/messages" in seen["url"]
        assert "after=42" in seen["url"] and "limit=25" in seen["url"]

    def test_rate_limited_retries_up_to_three_times_then_raises_collector_error(self):
        calls = []

        def opener(request, timeout=0):
            calls.append(1)
            raise urllib.error.HTTPError(request.full_url, 429, "slow", {"Retry-After": "0.01"}, io.BytesIO(b"{}"))

        sleeps: list[float] = []
        transport = _live(bot_token="tok", opener=opener, sleeper=sleeps.append, delay_range=(0.0, 0.0))
        with pytest.raises(CollectorError):
            transport.my_guilds()
        assert len(calls) == 4  # 1 initial + 3 retries
        assert [delay for delay in sleeps if delay] == [0.01, 0.01, 0.01]

    def test_rate_limited_then_success_returns_result(self):
        state = {"n": 0}

        def opener(request, timeout=0):
            state["n"] += 1
            if state["n"] <= 2:
                raise urllib.error.HTTPError(
                    request.full_url, 429, "slow", {"Retry-After": "0.01"}, io.BytesIO(b"{}")
                )
            return _Resp(b'[{"id": "1"}]')

        transport = _live(bot_token="tok", opener=opener)
        assert transport.my_guilds() == [{"id": "1"}]
        assert state["n"] == 3

    def test_channel_messages_403_propagates_http_error_for_collector_to_handle(self):
        def opener(request, timeout=0):
            raise urllib.error.HTTPError(request.full_url, 403, "denied", {}, io.BytesIO(b""))

        client = HttpClient(opener=opener, sleeper=lambda s: None, max_retries=0)
        transport = DiscordLiveTransport(client, bot_token="tok", sleeper=lambda s: None)
        with pytest.raises(HttpError) as exc:
            transport.channel_messages("11", after=None, limit=100)
        assert exc.value.status == 403

    def test_the_live_transport_never_contacts_a_directory_site(self, caplog):
        called: list[str] = []

        def opener(request, timeout=0):
            called.append(request.full_url)
            return _Resp(b"[]")

        transport = _live(bot_token="tok", opener=opener)
        with caplog.at_level(logging.WARNING):
            transport.my_guilds()
        assert all(url.startswith("https://discord.com/api/v10/") for url in called)
        assert not any(hasattr(transport, name) for name in ("directory_search", "invite"))
