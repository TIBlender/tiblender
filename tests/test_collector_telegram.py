"""Tests for the Telegram collector: message conversion/parsing, the joined-dialog
and explicit-channel-list collection paths, incremental watermarks, and the
fixture/live transports."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from tiblender_runtime.collectors.base import CollectionWindow, CollectorError, QueryPlan
from tiblender_runtime.collectors.fixtures import load_fixture
from tiblender_runtime.collectors.telegram import (
    ChannelRef,
    TelegramCollector,
    TelegramFixtureTransport,
    TelethonTransport,
    message_to_dict,
    parse_messages,
)


# ---------------------------------------------------------------------------
# message_to_dict
# ---------------------------------------------------------------------------


def _make_message(**overrides):
    base = dict(
        id=42,
        date=datetime(2026, 3, 20, 10, 30, 0, tzinfo=UTC),
        message="hello world",
        views=10,
        forwards=2,
        forward=None,
        photo=None,
        document=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_message_to_dict_basic_fields():
    result = message_to_dict(_make_message())
    assert result == {
        "id": 42,
        "date": "2026-03-20T10:30:00Z",
        "message": "hello world",
        "views": 10,
        "forwards": 2,
        "fwd_from_username": "",
        "media": {"type": "", "ocr_text": ""},
    }


def test_message_to_dict_photo_media():
    result = message_to_dict(_make_message(photo=object()))
    assert result["media"] == {"type": "photo", "ocr_text": ""}


def test_message_to_dict_document_media():
    result = message_to_dict(_make_message(document=object()))
    assert result["media"] == {"type": "document", "ocr_text": ""}


def test_message_to_dict_forward_username():
    forward = SimpleNamespace(chat=SimpleNamespace(username="origchan"))
    result = message_to_dict(_make_message(forward=forward))
    assert result["fwd_from_username"] == "origchan"


def test_message_to_dict_forward_without_chat_username():
    forward = SimpleNamespace(chat=None)
    result = message_to_dict(_make_message(forward=forward))
    assert result["fwd_from_username"] == ""


# ---------------------------------------------------------------------------
# parse_messages
# ---------------------------------------------------------------------------


def _channel(**overrides):
    base = dict(id="100", username="mychan", title="My Chan", discovery="joined")
    base.update(overrides)
    return ChannelRef(**base)


def _payload(**overrides):
    base = {
        "id": 1,
        "date": "2026-03-20T10:00:00Z",
        "message": "hi",
        "views": 1,
        "forwards": 0,
        "fwd_from_username": "",
        "media": {"type": "", "ocr_text": ""},
    }
    base.update(overrides)
    return base


def test_parse_messages_url_present_when_username_set():
    posts = parse_messages(_channel(), [_payload()])
    assert posts[0].url == "https://t.me/mychan/1"
    assert posts[0].external_id == "100:1"
    assert posts[0].author_ref == "mychan"


def test_parse_messages_url_empty_when_no_username():
    posts = parse_messages(_channel(username=""), [_payload()])
    assert posts[0].url == ""


def test_parse_messages_no_attachments_when_no_media():
    posts = parse_messages(_channel(), [_payload()])
    assert posts[0].attachments == []


def test_parse_messages_attachment_with_ocr_text():
    posts = parse_messages(_channel(), [_payload(media={"type": "photo", "ocr_text": "some text"})])
    assert posts[0].attachments == [{"type": "photo", "url": "", "ocr_text": "some text"}]


def test_parse_messages_metadata_fields():
    posts = parse_messages(
        _channel(discovery="channel_list"),
        [_payload(id=5, views=7, forwards=3)],
    )
    assert posts[0].metadata == {
        "source": "channel_list",
        "channel_id": "100",
        "channel_title": "My Chan",
        "views": 7,
        "forwards": 3,
    }


def test_channel_ref_carries_no_vestigial_seed_or_snowball_fields():
    fields = {field.name for field in dataclasses.fields(ChannelRef)}
    assert fields == {"id", "username", "title", "discovery"}


# ---------------------------------------------------------------------------
# TelegramFixtureTransport
# ---------------------------------------------------------------------------


def _write_fixture(tmp_path, responses):
    payload = {
        "platform": "telegram",
        "window": {
            "start": "2026-03-20T09:00:00Z",
            "end": "2026-03-20T10:00:00Z",
            "cycle_index": 1,
            "label": "cycle_001",
        },
        "provenance": "test",
        "responses": responses,
    }
    path = tmp_path / "cycle_001.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_fixture(path)


def _dialogs_response(chats):
    return {"request": {"endpoint": "dialogs"}, "payload": {"chats": chats}}


def test_fixture_transport_dialog_channels(tmp_path):
    fixture = _write_fixture(
        tmp_path,
        [_dialogs_response([{"id": 1, "username": "joinedchan", "title": "Joined Chan"}])],
    )
    transport = TelegramFixtureTransport(fixture)
    assert transport.dialog_channels() == [
        ChannelRef(id="1", username="joinedchan", title="Joined Chan", discovery="joined")
    ]


def test_fixture_transport_dialog_channels_empty_without_a_dialogs_response(tmp_path):
    fixture = _write_fixture(tmp_path, [{"request": {"endpoint": "messages", "channel_id": "1"}, "payload": []}])
    assert TelegramFixtureTransport(fixture).dialog_channels() == []


def test_fixture_transport_resolve_channel_by_username_and_id(tmp_path):
    fixture = _write_fixture(
        tmp_path,
        [_dialogs_response([{"id": 1, "username": "joinedchan", "title": "Joined Chan"}])],
    )
    transport = TelegramFixtureTransport(fixture)
    expected = ChannelRef(id="1", username="joinedchan", title="Joined Chan", discovery="channel_list")
    assert transport.resolve_channel("@joinedchan") == expected
    assert transport.resolve_channel("joinedchan") == expected
    assert transport.resolve_channel("1") == expected
    assert transport.resolve_channel("nope") is None


def test_fixture_transport_channel_messages_filters_and_sorts(tmp_path):
    fixture = _write_fixture(
        tmp_path,
        [
            {
                "request": {"endpoint": "messages", "channel_id": "1"},
                "payload": [_payload(id=3), _payload(id=1), _payload(id=2)],
            }
        ],
    )
    transport = TelegramFixtureTransport(fixture)
    channel = ChannelRef(id="1", username="joinedchan", title="Joined Chan")
    messages = transport.channel_messages(channel, min_id=1, limit=10)
    assert [m["id"] for m in messages] == [2, 3]


def test_fixture_transport_has_no_search_or_snowball_surface(tmp_path):
    fixture = _write_fixture(tmp_path, [_dialogs_response([])])
    transport = TelegramFixtureTransport(fixture)
    for retired in ("search_channels", "recommendations"):
        assert not hasattr(transport, retired)


# ---------------------------------------------------------------------------
# TelethonTransport (not-installed path, and `_call` retry/error-mapping with
# an injected fake client + fake telethon.errors -- no real Telethon needed)
# ---------------------------------------------------------------------------


def test_telethon_transport_raises_when_not_installed(monkeypatch):
    from tiblender_runtime.collectors import telegram as telegram_module

    monkeypatch.setattr(telegram_module.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(CollectorError, match="telethon is not installed"):
        TelethonTransport(api_id="1", api_hash="hash", session="session")


class _FakeFloodWaitError(Exception):
    def __init__(self, seconds):
        self.seconds = seconds
        super().__init__(f"flood wait {seconds}s")


class _FakeChannelPrivateError(Exception):
    pass


class _FakeUsernameNotOccupiedError(Exception):
    pass


class _FakeChatAdminRequiredError(Exception):
    pass


class _FakeUsernameInvalidError(Exception):
    pass


class _FakeChannel:
    def __init__(self, id, username="", title=""):
        self.id = id
        self.username = username
        self.title = title


def _fake_telethon_module():
    return SimpleNamespace(
        errors=SimpleNamespace(
            FloodWaitError=_FakeFloodWaitError,
            ChannelPrivateError=_FakeChannelPrivateError,
            UsernameNotOccupiedError=_FakeUsernameNotOccupiedError,
            ChatAdminRequiredError=_FakeChatAdminRequiredError,
            UsernameInvalidError=_FakeUsernameInvalidError,
        ),
        tl=SimpleNamespace(types=SimpleNamespace(Channel=_FakeChannel)),
    )


def _make_injected_transport(run_until_complete, *, delay_seconds=0.0):
    # Bypasses __init__ (and the real telethon import) entirely: `_call` only
    # ever touches `self._client.loop.run_until_complete`, `self._telethon`,
    # `self._delay_seconds` and `self._connected`, so those are the only
    # attributes a test double needs.
    transport = TelethonTransport.__new__(TelethonTransport)
    transport._telethon = _fake_telethon_module()
    transport._client = SimpleNamespace(loop=SimpleNamespace(run_until_complete=run_until_complete))
    transport._delay_seconds = delay_seconds
    transport._connected = True
    return transport


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    from tiblender_runtime.collectors import telegram as telegram_module

    monkeypatch.setattr(telegram_module, "_sleep", lambda seconds: None)


def test_call_flood_wait_under_300s_retries_once_and_returns_result():
    calls = {"n": 0}

    def run_until_complete(_coro):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _FakeFloodWaitError(seconds=5)
        return "retried-result"

    transport = _make_injected_transport(run_until_complete)
    assert transport._call(lambda: "marker") == "retried-result"
    assert calls["n"] == 2


def test_call_flood_wait_over_300s_raises_collector_error():
    def run_until_complete(_coro):
        raise _FakeFloodWaitError(seconds=301)

    transport = _make_injected_transport(run_until_complete)
    with pytest.raises(CollectorError, match="flood wait"):
        transport._call(lambda: "marker")


def test_call_flood_wait_twice_raises_after_retry_exhausted():
    def run_until_complete(_coro):
        raise _FakeFloodWaitError(seconds=5)

    transport = _make_injected_transport(run_until_complete)
    with pytest.raises(CollectorError, match="flood wait retry failed"):
        transport._call(lambda: "marker")


@pytest.mark.parametrize(
    "error_cls",
    [_FakeChannelPrivateError, _FakeUsernameNotOccupiedError, _FakeChatAdminRequiredError, _FakeUsernameInvalidError],
)
def test_call_channel_access_errors_return_none(error_cls):
    def run_until_complete(_coro):
        raise error_cls()

    transport = _make_injected_transport(run_until_complete)
    assert transport._call(lambda: "marker") is None


def test_call_flood_wait_then_access_error_returns_none():
    calls = {"n": 0}

    def run_until_complete(_coro):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _FakeFloodWaitError(seconds=5)
        raise _FakeChannelPrivateError()

    transport = _make_injected_transport(run_until_complete)
    assert transport._call(lambda: "marker") is None
    assert calls["n"] == 2


def test_call_generic_exception_raises_collector_error():
    def run_until_complete(_coro):
        raise RuntimeError("network blew up")

    transport = _make_injected_transport(run_until_complete)
    with pytest.raises(CollectorError, match="telegram call failed: RuntimeError: network blew up"):
        transport._call(lambda: "marker")


def test_dialog_channels_maps_channel_entities(monkeypatch):
    entities = [_FakeChannel(1, "chan_a", "Chan A"), _FakeChannel(2, "", "Chan B")]

    def run_until_complete(coro):
        # The transport's coroutine is not awaited here; the fake client
        # answers with the entities the real `get_dialogs()` filter would.
        coro.close()
        return entities

    transport = _make_injected_transport(run_until_complete)
    assert transport.dialog_channels() == [
        ChannelRef(id="1", username="chan_a", title="Chan A", discovery="joined"),
        ChannelRef(id="2", username="", title="Chan B", discovery="joined"),
    ]


def test_resolve_channel_maps_one_entity_and_none_on_access_error():
    def run_until_complete(coro):
        coro.close()
        return _FakeChannel(7, "chan_c", "Chan C")

    transport = _make_injected_transport(run_until_complete)
    assert transport.resolve_channel("chan_c") == ChannelRef(
        id="7", username="chan_c", title="Chan C", discovery="channel_list"
    )

    def denied(coro):
        coro.close()
        raise _FakeChannelPrivateError()

    assert _make_injected_transport(denied).resolve_channel("chan_c") is None


class _AsyncMessages:
    """Minimal async iterator standing in for Telethon's `iter_messages`."""

    def __init__(self, messages):
        self._messages = list(messages)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._messages):
            raise StopAsyncIteration
        message = self._messages[self._index]
        self._index += 1
        return message


class _FakeTelethonClient:
    """A client whose `iter_messages` honours `min_id`, `limit` and `reverse`
    the way Telethon's does, so the collector's paging can be tested for real."""

    def __init__(self, messages, *, channels=None):
        self._messages = sorted(messages, key=lambda message: message.id)
        self._channels = channels if channels is not None else [_FakeChannel(1, "chana", "Chan A")]
        self.iter_calls: list[dict] = []
        self.loop = SimpleNamespace(run_until_complete=lambda coro: asyncio.run(coro))

    async def get_dialogs(self):
        return [SimpleNamespace(entity=entity) for entity in self._channels]

    async def get_entity(self, reference):
        del reference
        return self._channels[0]

    def iter_messages(self, entity, *, min_id, limit, reverse=False):
        del entity
        self.iter_calls.append({"min_id": min_id, "limit": limit, "reverse": reverse})
        newer = [message for message in self._messages if message.id > min_id]
        # Telethon walks newest -> oldest unless `reverse` is set.
        ordered = newer if reverse else sorted(newer, key=lambda message: -message.id)
        return _AsyncMessages(ordered[:limit])


def _client_transport(client, *, delay_seconds=0.0):
    transport = TelethonTransport.__new__(TelethonTransport)
    transport._telethon = _fake_telethon_module()
    transport._client = client
    transport._delay_seconds = delay_seconds
    transport._connected = True
    return transport


def _live_message(message_id, minute):
    return _make_message(
        id=message_id,
        date=datetime(2026, 3, 20, 9, minute, 0, tzinfo=UTC),
        message=f"m{message_id}",
    )


def test_live_channel_messages_pages_oldest_first():
    client = _FakeTelethonClient([_live_message(i, i) for i in range(1, 8)])
    transport = _client_transport(client)
    channel = ChannelRef(id="1", username="chana", title="Chan A")

    messages = transport.channel_messages(channel, min_id=2, limit=3)

    assert client.iter_calls == [{"min_id": 2, "limit": 3, "reverse": True}]
    # The three OLDEST messages above the watermark, not the three newest.
    assert [message["id"] for message in messages] == [3, 4, 5]


def test_live_backlog_larger_than_one_page_is_collected_across_cycles_without_gaps():
    """A busy channel must not lose the messages between the watermark and the
    newest page: every id is collected exactly once and the watermark ends at
    the newest id."""
    client = _FakeTelethonClient([_live_message(i, i) for i in range(1, 8)])
    transport = _client_transport(client)

    state: dict = {}
    collected: list[str] = []
    for _cycle in range(3):
        collector = TelegramCollector(transport, messages_per_channel=3, state=state)
        result = collector.collect(_window(), QueryPlan([]))
        collected.extend(post.external_id for post in result.posts)
        state = collector.export_state()

    assert collected == [f"1:{message_id}" for message_id in range(1, 8)]
    assert len(collected) == len(set(collected))
    assert state == {"last_message_id": {"1": 7}}
    assert all(call["reverse"] is True for call in client.iter_calls)
    # Each cycle resumes exactly where the previous one stopped.
    assert [call["min_id"] for call in client.iter_calls] == [0, 3, 6]


def test_live_dialog_channels_reads_joined_channel_entities():
    client = _FakeTelethonClient([], channels=[_FakeChannel(5, "chanb", "Chan B")])
    transport = _client_transport(client)
    assert transport.dialog_channels() == [
        ChannelRef(id="5", username="chanb", title="Chan B", discovery="joined")
    ]


def test_transport_has_no_search_or_snowball_surface():
    transport = _make_injected_transport(lambda coro: None)
    for retired in ("search_channels", "recommendations"):
        assert not hasattr(transport, retired)


# ---------------------------------------------------------------------------
# TelegramCollector
# ---------------------------------------------------------------------------


def _msg(id_, date, text="hello", views=0, forwards=0, media_type="", ocr_text=""):
    return {
        "id": id_,
        "date": date,
        "message": text,
        "views": views,
        "forwards": forwards,
        "fwd_from_username": "",
        "media": {"type": media_type, "ocr_text": ocr_text},
    }


class _FakeTransport:
    def __init__(self, dialogs=None, messages_map=None, resolve_map=None, raise_on=None):
        self.dialogs = dialogs or []
        self.messages_map = messages_map or {}
        self.resolve_map = resolve_map or {}
        self.raise_on = raise_on or {}
        self.dialog_calls = 0
        self.resolve_calls: list[str] = []
        self.message_calls: list[tuple[str, int]] = []

    def dialog_channels(self):
        self.dialog_calls += 1
        if "dialogs" in self.raise_on:
            raise CollectorError(self.raise_on["dialogs"])
        return list(self.dialogs)

    def resolve_channel(self, reference):
        self.resolve_calls.append(reference)
        if reference in self.raise_on.get("resolve", {}):
            raise CollectorError(self.raise_on["resolve"][reference])
        return self.resolve_map.get(reference)

    def channel_messages(self, channel, *, min_id, limit):
        self.message_calls.append((channel.id, min_id))
        if channel.id in self.raise_on.get("messages", {}):
            raise CollectorError(self.raise_on["messages"][channel.id])
        msgs = self.messages_map.get(channel.id, [])
        filtered = sorted((m for m in msgs if m["id"] > min_id), key=lambda m: m["id"])
        return filtered[:limit]


def _window():
    return CollectionWindow(
        start=datetime(2026, 3, 20, 9, 0, tzinfo=UTC),
        end=datetime(2026, 3, 20, 11, 0, tzinfo=UTC),
        cycle_index=1,
        label="cycle_001",
    )


def _joined(id_, username, title="Chan"):
    return ChannelRef(id=id_, username=username, title=title, discovery="joined")


def test_collect_reads_every_joined_channel_when_no_channel_list_is_configured():
    transport = _FakeTransport(
        dialogs=[_joined("1", "chana"), _joined("2", "chanb")],
        messages_map={
            "1": [_msg(1, "2026-03-20T09:30:00Z")],
            "2": [_msg(1, "2026-03-20T09:35:00Z")],
        },
    )
    collector = TelegramCollector(transport)
    result = collector.collect(_window(), QueryPlan([]))

    assert transport.dialog_calls == 1
    assert transport.resolve_calls == []
    assert result.stats == {"channels": 2, "posts": 2, "errors": 0}
    assert {post.metadata["source"] for post in result.posts} == {"joined"}


def test_collect_uses_the_configured_channel_list_instead_of_dialogs():
    transport = _FakeTransport(
        dialogs=[_joined("9", "unused")],
        resolve_map={"@chana": ChannelRef(id="1", username="chana", title="A", discovery="channel_list")},
        messages_map={"1": [_msg(1, "2026-03-20T09:30:00Z")], "9": [_msg(1, "2026-03-20T09:30:00Z")]},
    )
    collector = TelegramCollector(transport, channels=["@chana"])
    result = collector.collect(_window(), QueryPlan([]))

    assert transport.dialog_calls == 0
    assert transport.resolve_calls == ["@chana"]
    assert [post.external_id for post in result.posts] == ["1:1"]
    assert result.posts[0].metadata["source"] == "channel_list"


def test_collect_records_an_unresolvable_configured_channel_and_continues():
    transport = _FakeTransport(
        resolve_map={"chanb": ChannelRef(id="2", username="chanb", title="B", discovery="channel_list")},
        messages_map={"2": [_msg(1, "2026-03-20T09:30:00Z")]},
    )
    collector = TelegramCollector(transport, channels=["chana", "chanb"])
    result = collector.collect(_window(), QueryPlan([]))

    assert result.errors == ["chana: channel not found or not accessible"]
    assert [post.external_id for post in result.posts] == ["2:1"]


def test_collect_records_a_resolve_error_and_continues():
    transport = _FakeTransport(raise_on={"resolve": {"chana": "boom"}})
    collector = TelegramCollector(transport, channels=["chana"])
    result = collector.collect(_window(), QueryPlan([]))
    assert result.errors == ["chana: boom"]
    assert result.stats == {"channels": 0, "posts": 0, "errors": 1}


def test_collect_records_a_dialog_listing_error_without_crashing():
    transport = _FakeTransport(raise_on={"dialogs": "flood"})
    collector = TelegramCollector(transport)
    result = collector.collect(_window(), QueryPlan([]))
    assert result.errors == ["dialogs: flood"]
    assert result.stats["channels"] == 0


def test_collect_respects_max_channels():
    transport = _FakeTransport(
        dialogs=[_joined("1", "a"), _joined("2", "b"), _joined("3", "c")],
        messages_map={"1": [_msg(1, "2026-03-20T09:30:00Z")]},
    )
    collector = TelegramCollector(transport, max_channels=2)
    result = collector.collect(_window(), QueryPlan([]))
    assert result.stats["channels"] == 2
    assert [call[0] for call in transport.message_calls] == ["1", "2"]


def test_collect_caps_channels_by_lowest_id_regardless_of_transport_order():
    """get_dialogs() returns Telegram's recency order, which changes whenever a
    channel receives a message; the cap must not depend on it."""
    transport = _FakeTransport(dialogs=[_joined("30", "c"), _joined("10", "a"), _joined("20", "b")])
    collector = TelegramCollector(transport, max_channels=2)
    result = collector.collect(_window(), QueryPlan([]))

    assert result.stats["channels"] == 2
    assert [call[0] for call in transport.message_calls] == ["10", "20"]

    reordered = _FakeTransport(dialogs=[_joined("20", "b"), _joined("30", "c"), _joined("10", "a")])
    TelegramCollector(reordered, max_channels=2).collect(_window(), QueryPlan([]))
    assert [call[0] for call in reordered.message_calls] == ["10", "20"]


def test_collect_orders_channels_by_numeric_id_and_sorts_unparsable_ids_last():
    transport = _FakeTransport(dialogs=[_joined("abc", "x"), _joined("30", "c"), _joined("10", "a")])
    TelegramCollector(transport).collect(_window(), QueryPlan([]))
    assert [call[0] for call in transport.message_calls] == ["10", "30", "abc"]


def test_collect_max_channels_zero_means_no_cap():
    transport = _FakeTransport(dialogs=[_joined(str(i), f"c{i}") for i in range(1, 6)])
    collector = TelegramCollector(transport, max_channels=0)
    result = collector.collect(_window(), QueryPlan([]))
    assert result.stats["channels"] == 5


def test_collect_dedupes_repeated_channel_ids():
    transport = _FakeTransport(
        resolve_map={
            "chana": ChannelRef(id="1", username="chana", title="A", discovery="channel_list"),
            "1": ChannelRef(id="1", username="chana", title="A", discovery="channel_list"),
        },
        messages_map={"1": [_msg(1, "2026-03-20T09:30:00Z")]},
    )
    collector = TelegramCollector(transport, channels=["chana", "1"])
    result = collector.collect(_window(), QueryPlan([]))
    assert result.stats["channels"] == 1


def test_collect_incremental_min_id_and_export_state():
    transport = _FakeTransport(
        dialogs=[_joined("1", "chana")],
        messages_map={
            "1": [
                _msg(1, "2026-03-20T09:10:00Z"),
                _msg(2, "2026-03-20T09:20:00Z"),
                _msg(5, "2026-03-20T09:40:00Z"),
            ]
        },
    )
    collector = TelegramCollector(transport, state={"last_message_id": {"1": 1}})
    result = collector.collect(_window(), QueryPlan([]))

    assert transport.message_calls == [("1", 1)]
    assert [p.external_id for p in result.posts] == ["1:2", "1:5"]
    assert collector.export_state() == {"last_message_id": {"1": 5}}


def test_collect_window_filtering_still_advances_last_message_id():
    transport = _FakeTransport(
        dialogs=[_joined("1", "chana")],
        messages_map={
            "1": [
                _msg(1, "2026-03-20T08:00:00Z"),  # before window
                _msg(2, "2026-03-20T09:30:00Z"),  # inside window
                _msg(3, "2026-03-20T12:00:00Z"),  # after window
            ]
        },
    )
    collector = TelegramCollector(transport)
    result = collector.collect(_window(), QueryPlan([]))

    assert [p.external_id for p in result.posts] == ["1:2"]
    assert collector.export_state() == {"last_message_id": {"1": 3}}


def test_collect_records_per_channel_errors_and_continues():
    transport = _FakeTransport(
        dialogs=[_joined("1", "chana"), _joined("2", "chanb")],
        messages_map={"2": [_msg(1, "2026-03-20T09:30:00Z")]},
        raise_on={"messages": {"1": "boom"}},
    )
    collector = TelegramCollector(transport)
    result = collector.collect(_window(), QueryPlan([]))
    assert result.errors == ["chana: boom"]
    assert [p.external_id for p in result.posts] == ["2:1"]
    assert result.stats["errors"] == 1


def test_collect_ignores_the_query_plan():
    from tiblender_runtime.collectors.base import CollectionQuery

    transport = _FakeTransport(
        dialogs=[_joined("1", "chana")], messages_map={"1": [_msg(1, "2026-03-20T09:30:00Z")]}
    )
    collector = TelegramCollector(transport)
    plan = QueryPlan([CollectionQuery("telegram", "stealer logs", "dynamic", priority=9)])
    result = collector.collect(_window(), plan)
    assert result.posts[0].metadata["source"] == "joined"
    assert "query" not in result.posts[0].metadata


def test_collector_mode_is_stored_and_stats_keys_are_exact():
    transport = _FakeTransport()
    collector = TelegramCollector(transport, mode="fixture")
    result = collector.collect(_window(), QueryPlan([]))
    assert result.mode == "fixture"
    assert result.stats == {"channels": 0, "posts": 0, "errors": 0}
