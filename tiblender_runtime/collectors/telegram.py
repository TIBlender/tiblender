"""Telegram collector: exhaustive incremental history from joined channels.

`TelethonTransport` talks to the real Telegram API via Telethon (imported lazily
and only inside its constructor, so this module imports fine without the
package installed). `TelegramFixtureTransport` replays recorded fixture
responses for offline/deterministic runs. `TelegramCollector` drives both
through the shared `TelegramTransport` protocol.

Collection is exhaustive over communities the account is already a member of,
never query-driven: either the channels listed in `config/telegram_channels.txt`
or, when that file lists none, the channels/supergroups returned by
`client.get_dialogs()`. Finding and joining channels is an operational task
performed outside this repository.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from typing import Protocol

from .base import CollectionResult, CollectionWindow, CollectorError, QueryPlan, RawPost
from .fixtures import FixtureFile

_CHANNEL_ACCESS_ERROR_NAMES = (
    "ChannelPrivateError",
    "UsernameNotOccupiedError",
    "ChatAdminRequiredError",
    "UsernameInvalidError",
)

_MAX_CALL_ATTEMPTS = 2

# Module-level so tests can `monkeypatch.setattr(telegram, "_sleep", lambda s: None)`
# to skip real delays/backoff without waiting on the clock.
_sleep = time.sleep


@dataclass(frozen=True, slots=True)
class ChannelRef:
    id: str
    username: str
    title: str
    # How this channel entered the run: "joined" (from the account's dialogs)
    # or "channel_list" (named in config/telegram_channels.txt).
    discovery: str = "joined"


class TelegramTransport(Protocol):
    def dialog_channels(self) -> list[ChannelRef]: ...

    def resolve_channel(self, reference: str) -> ChannelRef | None: ...

    def channel_messages(self, channel: ChannelRef, *, min_id: int, limit: int) -> list[dict]: ...


def message_to_dict(message: object) -> dict:
    """Convert a Telethon `Message`-shaped object into the plain dict transports return."""
    forward = getattr(message, "forward", None)
    fwd_from_username = ""
    if forward is not None:
        chat = getattr(forward, "chat", None)
        if chat is not None:
            fwd_from_username = getattr(chat, "username", None) or ""

    if getattr(message, "photo", None):
        media_type = "photo"
    elif getattr(message, "document", None):
        media_type = "document"
    else:
        media_type = ""

    return {
        "id": message.id,
        "date": message.date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message": getattr(message, "message", "") or "",
        "views": getattr(message, "views", None) or 0,
        "forwards": getattr(message, "forwards", None) or 0,
        "fwd_from_username": fwd_from_username,
        "media": {"type": media_type, "ocr_text": ""},
    }


def parse_messages(channel: ChannelRef, messages: list[dict]) -> list[RawPost]:
    posts: list[RawPost] = []
    for message in messages:
        media = message.get("media") or {}
        media_type = media.get("type", "")
        attachments = (
            [{"type": media_type, "url": "", "ocr_text": media.get("ocr_text", "")}] if media_type else []
        )
        url = f"https://t.me/{channel.username}/{message['id']}" if channel.username else ""
        posts.append(
            RawPost(
                platform="telegram",
                external_id=f"{channel.id}:{message['id']}",
                content=message.get("message", ""),
                posted_at=message["date"],
                author_ref=channel.username,
                url=url,
                attachments=attachments,
                metadata={
                    "source": channel.discovery,
                    "channel_id": channel.id,
                    "channel_title": channel.title,
                    "views": message.get("views", 0),
                    "forwards": message.get("forwards", 0),
                },
            )
        )
    return posts


def _channel_ref(chat: object, *, discovery: str) -> ChannelRef:
    return ChannelRef(
        id=str(getattr(chat, "id", "")),
        username=getattr(chat, "username", "") or "",
        title=getattr(chat, "title", "") or "",
        discovery=discovery,
    )


class TelethonTransport:
    """Live MTProto transport. Telethon is imported lazily so this module (and
    every other transport/collector in it) works without the package installed.
    """

    def __init__(self, api_id: str, api_hash: str, session: str, *, delay_seconds: float = 1.0):
        if importlib.util.find_spec("telethon") is None:
            raise CollectorError("telethon is not installed; install with pip install telethon")

        import telethon
        from telethon import TelegramClient

        self._telethon = telethon
        self._client = TelegramClient(session, int(api_id), api_hash)
        self._delay_seconds = delay_seconds
        self._connected = False

    def _access_errors(self) -> tuple[type[Exception], ...]:
        errors = self._telethon.errors
        return tuple(getattr(errors, name) for name in _CHANNEL_ACCESS_ERROR_NAMES)

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        self._client.loop.run_until_complete(self._client.connect())
        if not self._client.loop.run_until_complete(self._client.is_user_authorized()):
            raise CollectorError(
                "telegram session is not authorized; run `python -m tiblender_runtime telegram-login`"
            )
        self._connected = True

    def _call(self, make_coro):
        """Run one MTProto call, retrying at most once after a short FloodWait.

        Every attempt shares the same exception handling: a FloodWait beyond
        the retry budget or a second consecutive FloodWait becomes a
        `CollectorError`; the four channel-access errors map to an empty
        result (`None`); any other exception (network errors, other RPC
        errors, ...) is wrapped in a `CollectorError` rather than propagating
        and crashing the whole collection cycle.
        """
        self._ensure_connected()
        pending_wait = self._delay_seconds
        last_flood_error = None
        for _attempt in range(_MAX_CALL_ATTEMPTS):
            _sleep(pending_wait)
            try:
                return self._client.loop.run_until_complete(make_coro())
            except self._telethon.errors.FloodWaitError as error:
                if error.seconds > 300:
                    raise CollectorError(f"telegram flood wait exceeds retry budget: {error.seconds}s") from error
                last_flood_error = error
                pending_wait = error.seconds
                continue
            except self._access_errors():
                return None
            except Exception as error:
                raise CollectorError(f"telegram call failed: {type(error).__name__}: {error}") from error
        raise CollectorError(f"telegram flood wait retry failed: {last_flood_error.seconds}s") from last_flood_error

    def dialog_channels(self) -> list[ChannelRef]:
        """Every channel/supergroup the signed-in account has joined."""
        channel_type = self._telethon.tl.types.Channel

        async def make():
            dialogs = await self._client.get_dialogs()
            return [dialog.entity for dialog in dialogs if isinstance(dialog.entity, channel_type)]

        result = self._call(make)
        return [_channel_ref(entity, discovery="joined") for entity in (result or [])]

    def resolve_channel(self, reference: str) -> ChannelRef | None:
        target: str | int = reference
        if reference.lstrip("-").isdigit():
            target = int(reference)

        async def make():
            return await self._client.get_entity(target)

        entity = self._call(make)
        if entity is None:
            return None
        return _channel_ref(entity, discovery="channel_list")

    def channel_messages(self, channel: ChannelRef, *, min_id: int, limit: int) -> list[dict]:
        """The oldest `limit` messages newer than `min_id`, oldest first.

        `reverse=True` is what makes collection incremental rather than merely
        recent. Telethon iterates newest -> oldest by default, so a channel
        that produced more than `limit` messages since the last cycle would
        return only its newest `limit`, and the caller's watermark -- which
        advances to the highest id it saw -- would jump straight past the
        messages in between, permanently. Walking oldest -> oldest+limit from
        the watermark keeps the advance contiguous: a backlog is drained over
        successive cycles instead of skipped, and this matches what
        `TelegramFixtureTransport.channel_messages` replays offline.
        """

        async def make():
            entity = await self._client.get_entity(channel.username or int(channel.id))
            messages = []
            async for message in self._client.iter_messages(
                entity, min_id=min_id, limit=limit, reverse=True
            ):
                messages.append(message_to_dict(message))
            return messages

        result = self._call(make)
        return result or []


class TelegramFixtureTransport:
    """Replays recorded fixture responses; used for offline/replay collection cycles."""

    def __init__(self, fixture: FixtureFile):
        self._fixture = fixture

    def _chats(self) -> list[dict]:
        responses = self._fixture.responses_for("dialogs")
        if not responses:
            return []
        return responses[0]["payload"].get("chats", [])

    def dialog_channels(self) -> list[ChannelRef]:
        return [
            ChannelRef(
                id=str(chat["id"]),
                username=chat.get("username", "") or "",
                title=chat.get("title", "") or "",
                discovery="joined",
            )
            for chat in self._chats()
        ]

    def resolve_channel(self, reference: str) -> ChannelRef | None:
        wanted = reference.lstrip("@").lower()
        for chat in self._chats():
            username = str(chat.get("username", "") or "").lower()
            if wanted in {username, str(chat["id"]).lower()}:
                return ChannelRef(
                    id=str(chat["id"]),
                    username=chat.get("username", "") or "",
                    title=chat.get("title", "") or "",
                    discovery="channel_list",
                )
        return None

    def channel_messages(self, channel: ChannelRef, *, min_id: int, limit: int) -> list[dict]:
        responses = self._fixture.responses_for("messages", channel_id=channel.id)
        if not responses:
            return []
        messages = [message for message in responses[0]["payload"] if message["id"] > min_id]
        messages.sort(key=lambda message: message["id"])
        return messages[:limit]


def _channel_sort_key(channel: ChannelRef) -> tuple[int, int, str]:
    """Order channels by numeric id, with any non-numeric id sorted last by text."""
    identifier = channel.id.lstrip("-")
    if identifier.isdigit():
        return (0, int(channel.id), "")
    return (1, 0, channel.id)


class TelegramCollector:
    platform = "telegram"

    def __init__(
        self,
        transport: TelegramTransport,
        *,
        channels: list[str] | None = None,
        max_channels: int = 0,
        messages_per_channel: int = 100,
        mode: str = "live",
        state: dict | None = None,
    ):
        self.transport = transport
        # Explicit channel references (usernames or numeric ids) from
        # config/telegram_channels.txt. Empty -> every joined channel.
        self.channel_references = [reference for reference in (channels or []) if reference.strip()]
        # 0 means "no cap": collect every joined channel.
        self.max_channels = max(0, int(max_channels))
        self.messages_per_channel = messages_per_channel
        self.mode = mode
        self._last_message_id: dict[str, int] = dict((state or {}).get("last_message_id", {}))

    def export_state(self) -> dict:
        return {"last_message_id": dict(sorted(self._last_message_id.items()))}

    def _channels(self, errors: list[str]) -> list[ChannelRef]:
        if self.channel_references:
            resolved: list[ChannelRef] = []
            for reference in self.channel_references:
                try:
                    channel = self.transport.resolve_channel(reference)
                except CollectorError as error:
                    errors.append(f"{reference}: {error}")
                    continue
                if channel is None:
                    errors.append(f"{reference}: channel not found or not accessible")
                    continue
                resolved.append(channel)
        else:
            try:
                resolved = list(self.transport.dialog_channels())
            except CollectorError as error:
                errors.append(f"dialogs: {error}")
                resolved = []

        deduped: list[ChannelRef] = []
        seen: set[str] = set()
        for channel in resolved:
            if channel.id in seen:
                continue
            seen.add(channel.id)
            deduped.append(channel)
        # Sort before capping, exactly like DiscordCollector's guild/channel
        # selection: `get_dialogs()` hands back Telegram's own recency order, so
        # an uncapped run is unaffected but a capped one would otherwise collect
        # a different subset every time a channel received a message.
        deduped.sort(key=_channel_sort_key)
        if self.max_channels:
            deduped = deduped[: self.max_channels]
        return deduped

    def _fetch_messages(self, channel: ChannelRef, errors: list[str]) -> list[dict]:
        min_id = self._last_message_id.get(channel.id, 0)
        try:
            messages = self.transport.channel_messages(channel, min_id=min_id, limit=self.messages_per_channel)
        except CollectorError as error:
            errors.append(f"{channel.username or channel.id}: {error}")
            return []
        if messages:
            max_id = max(message["id"] for message in messages)
            self._last_message_id[channel.id] = max(self._last_message_id.get(channel.id, 0), max_id)
        return messages

    def collect(self, window: CollectionWindow, plan: QueryPlan) -> CollectionResult:
        del plan  # telegram collection is channel driven, not query driven
        errors: list[str] = []
        channels = self._channels(errors)

        posts: list[RawPost] = []
        for channel in channels:
            messages = self._fetch_messages(channel, errors)
            windowed = [message for message in messages if window.contains(message["date"])]
            posts.extend(parse_messages(channel, windowed))

        stats = {
            "channels": len(channels),
            "posts": len(posts),
            "errors": len(errors),
        }
        return CollectionResult(platform=self.platform, mode=self.mode, posts=posts, errors=errors, stats=stats)
