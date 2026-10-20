"""Discord collector: message collection from the guilds the account has joined.

Collection is exhaustive over joined communities, never query-driven: the
account's guilds (`/users/@me/guilds`) are enumerated, their text/announcement
channels listed, and each channel's new messages read with an `after`
watermark. Which servers the account is a member of -- finding them and
joining them -- is an operational task performed outside this repository.

Two credential shapes are supported. A user-account token
(`TIBLENDER_DISCORD_USER_TOKEN`) is preferred when set, because that is what
the deployment runs on: it is sent as a bare `Authorization` value with a
browser-like `User-Agent`. Otherwise a bot token
(`TIBLENDER_DISCORD_BOT_TOKEN`) is sent as `Authorization: Bot <token>`.
Automating a user account is subject to Discord's terms; see
`docs/CREDENTIALS.md` and the paper's ethics section.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import UTC, datetime
from typing import Callable, Protocol

from .base import (
    CollectionResult,
    CollectionWindow,
    CollectorError,
    QueryPlan,
    RawPost,
)
from .fixtures import FixtureFile
from .http import HttpClient, HttpError, RateLimited

logger = logging.getLogger(__name__)

TEXT_CHANNEL_TYPES = {0, 5}  # GUILD_TEXT, GUILD_ANNOUNCEMENT

_LIVE_BASE = "https://discord.com/api/v10"
_BOT_USER_AGENT = "DiscordBot (tiblender, 1.0)"
# A user-account token is what a browser session carries, so requests made
# with one identify as a browser rather than as a bot.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
_MAX_RATE_LIMIT_RETRIES = 3
_MAX_MESSAGE_PAGES = 3
_DEFAULT_MESSAGES_PER_PAGE = 100
# Politeness bound on one cycle: at most three pages of 100 messages per
# channel. Nothing is lost to it -- the `after` watermark is anchored at the
# oldest message this cycle has not seen, so paging is contiguous and a
# backlog is drained over successive cycles rather than skipped.
MAX_MESSAGES_PER_CHANNEL_PER_CYCLE = _MAX_MESSAGE_PAGES * _DEFAULT_MESSAGES_PER_PAGE
# Polite spacing between live API calls: the collector reads many channels in
# one pass and must not behave like a scraper even when Discord would allow it.
_MIN_REQUEST_DELAY_SECONDS = 2.0
_MAX_REQUEST_DELAY_SECONDS = 5.0


def discord_auth_headers(*, user_token: str = "", bot_token: str = "") -> dict[str, str]:
    """Pick the Authorization/User-Agent pair for the configured credential.

    A user-account token wins over a bot token when both are set: the
    production collector runs on a user account that is a member of the
    monitored servers, and the bot token is only the fallback for a
    bot-based deployment.
    """
    user = (user_token or "").strip()
    if user:
        return {"Authorization": user, "User-Agent": _BROWSER_USER_AGENT}
    bot = (bot_token or "").strip()
    if bot:
        return {"Authorization": f"Bot {bot}", "User-Agent": _BOT_USER_AGENT}
    raise CollectorError(
        "discord requires TIBLENDER_DISCORD_USER_TOKEN or TIBLENDER_DISCORD_BOT_TOKEN"
    )


class DiscordTransport(Protocol):
    def my_guilds(self) -> list[dict]: ...

    def guild_channels(self, guild_id: str) -> list[dict]: ...

    def channel_messages(self, channel_id: str, *, after: str | None, limit: int) -> list[dict]: ...


class DiscordLiveTransport:
    """Talks to the real Discord HTTP API with a user or bot token."""

    def __init__(
        self,
        client: HttpClient,
        *,
        user_token: str = "",
        bot_token: str = "",
        sleeper: Callable[[float], None] = time.sleep,
        delay_range: tuple[float, float] = (_MIN_REQUEST_DELAY_SECONDS, _MAX_REQUEST_DELAY_SECONDS),
        rng: random.Random | None = None,
    ) -> None:
        self._headers = discord_auth_headers(user_token=user_token, bot_token=bot_token)
        self._client = client
        self._sleeper = sleeper
        self._delay_range = delay_range
        self._rng = rng if rng is not None else random.Random()
        self.uses_user_token = not self._headers["Authorization"].startswith("Bot ")

    def _polite_delay(self) -> None:
        low, high = self._delay_range
        self._sleeper(self._rng.uniform(low, high) if high > low else low)

    def _authed_get(self, path: str, params: dict | None = None) -> dict | list:
        url = f"{_LIVE_BASE}/{path}"
        attempts = 0
        while True:
            self._polite_delay()
            try:
                return self._client.get_json(url, params=params, headers=dict(self._headers))
            except RateLimited as error:
                attempts += 1
                if attempts > _MAX_RATE_LIMIT_RETRIES:
                    raise CollectorError(
                        f"discord rate limited after {_MAX_RATE_LIMIT_RETRIES} retries: {path}"
                    ) from error
                self._sleeper(error.retry_after)

    def my_guilds(self) -> list[dict]:
        return self._authed_get("users/@me/guilds")

    def guild_channels(self, guild_id: str) -> list[dict]:
        return self._authed_get(f"guilds/{guild_id}/channels")

    def channel_messages(self, channel_id: str, *, after: str | None, limit: int) -> list[dict]:
        params: dict[str, object] = {"limit": limit}
        if after is not None:
            params["after"] = after
        return self._authed_get(f"channels/{channel_id}/messages", params)


def _snowflake(value: object, *, channel_id: str, source: str) -> int:
    """Parse a Discord snowflake, naming what failed instead of raising a bare ValueError.

    ``load_fixture`` validates response *structure*, not id types, and the
    ``after`` watermark comes from ``history/collector_state.json`` -- a file an
    operator can hand-edit or truncate. Either way the failure must name the
    channel and the offending value rather than surfacing as an unlabelled
    ``int()`` error from inside the replay path.
    """
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise CollectorError(
            f"discord channel {channel_id}: {source} is not a numeric snowflake: {value!r}"
        ) from error


class DiscordFixtureTransport:
    """Replays a recorded FixtureFile instead of calling the live Discord API."""

    def __init__(self, fixture: FixtureFile) -> None:
        self._fixture = fixture

    def my_guilds(self) -> list[dict]:
        responses = self._fixture.responses_for("users/@me/guilds")
        return responses[0]["payload"] if responses else []

    def guild_channels(self, guild_id: str) -> list[dict]:
        responses = self._fixture.responses_for("guilds/{id}/channels", guild_id=guild_id)
        return responses[0]["payload"] if responses else []

    def channel_messages(self, channel_id: str, *, after: str | None, limit: int) -> list[dict]:
        """Replay the channel's recorded page, honoring the ``after`` watermark.

        The watermark is applied *in code* over the recorded payload, exactly
        as :class:`~tiblender_runtime.collectors.telegram.TelegramFixtureTransport`
        applies ``min_id``. Matching it against the recorded request keys
        instead would never match -- no recorded request carries an ``after``
        key -- so every cycle after the first would silently replay as zero
        messages once cycle 1 stored a watermark.
        """
        responses = self._fixture.responses_for("channels/{id}/messages", channel_id=channel_id)
        if not responses:
            return []
        messages = list(responses[0]["payload"])
        by_id = [
            (_snowflake(message.get("id"), channel_id=channel_id, source="recorded message id"), message)
            for message in messages
        ]
        if after is not None:
            watermark = _snowflake(after, channel_id=channel_id, source="stored after watermark")
            by_id = [item for item in by_id if item[0] > watermark]
        by_id.sort(key=lambda item: item[0])
        messages = [message for _identifier, message in by_id]
        # limit <= 0 means "no cap", matching the collector's own `0 = no cap`
        # convention for its per-guild/per-channel bounds.
        return messages[:limit] if limit > 0 else messages


def _normalize_timestamp(raw: str) -> str:
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_attachment(attachment: dict) -> dict:
    content_type = str(attachment.get("content_type", ""))
    return {
        "type": "image" if content_type.startswith("image/") else "file",
        "url": attachment.get("url", ""),
        "ocr_text": attachment.get("ocr_text", ""),
    }


def parse_messages(guild: dict, channel: dict, messages: list[dict]) -> list[RawPost]:
    guild_id = str(guild.get("id", ""))
    guild_name = guild.get("name", "")
    channel_id = str(channel.get("id", ""))
    channel_name = channel.get("name", "")

    parsed: list[tuple[int, RawPost]] = []
    for message in messages:
        author = message.get("author", {})
        if author.get("bot"):
            continue
        content = message.get("content", "")
        if not content.strip():
            continue
        message_id = str(message["id"])
        post = RawPost(
            platform="discord",
            external_id=f"{channel_id}:{message_id}",
            content=content,
            posted_at=_normalize_timestamp(message["timestamp"]),
            author_ref=author.get("username", ""),
            url=f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}",
            attachments=[_parse_attachment(a) for a in message.get("attachments", [])],
            metadata={
                # Constant: every guild reaches this parser from
                # /users/@me/guilds, i.e. a channel of a joined server. There
                # is no other provenance now that discovery is gone.
                "source": "channel",
                "guild_id": guild_id,
                "guild_name": guild_name,
                "channel_id": channel_id,
                "channel_name": channel_name,
            },
        )
        parsed.append((int(message_id), post))

    parsed.sort(key=lambda item: item[0])
    return [post for _, post in parsed]


def _parse_guild_filter(lines: list[str]) -> dict[str, set[str] | None]:
    parsed: dict[str, set[str] | None] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped:
            guild_id, channel_part = stripped.split(":", 1)
            channel_ids = {c.strip() for c in channel_part.split(",") if c.strip()}
            parsed[guild_id.strip()] = channel_ids or None
        else:
            parsed[stripped] = None
    return parsed


class DiscordCollector:
    platform = "discord"

    def __init__(
        self,
        transport: DiscordTransport,
        *,
        guild_filter: list[str],
        max_guilds: int = 0,
        max_channels_per_guild: int = 0,
        max_posts_per_channel: int = _DEFAULT_MESSAGES_PER_PAGE,
        mode: str = "live",
        state: dict | None = None,
    ) -> None:
        self._transport = transport
        self._guild_filter = _parse_guild_filter(list(guild_filter))
        # 0 means "no cap": collect every joined guild / every text channel.
        self._max_guilds = max(0, int(max_guilds))
        self._max_channels_per_guild = max(0, int(max_channels_per_guild))
        self._max_posts_per_channel = max_posts_per_channel
        self.mode = mode
        self._last_message_id: dict[str, str] = dict((state or {}).get("last_message_id", {}))

    def _selected_guilds(self, errors: list[str]) -> list[dict]:
        joined = self._transport.my_guilds()
        joined_ids = {str(guild.get("id")) for guild in joined}
        # A stale or mistyped id in config/discord_guilds.txt would otherwise
        # produce a silent zero-post run; name it, exactly as the Telegram
        # collector names a channel it cannot resolve.
        for guild_id in sorted(self._guild_filter):
            if guild_id not in joined_ids:
                errors.append(f"guild {guild_id}: not joined or not visible to this account")

        guilds = [
            guild
            for guild in joined
            if not self._guild_filter or str(guild.get("id")) in self._guild_filter
        ]
        guilds.sort(key=lambda guild: int(guild["id"]))
        if self._max_guilds:
            guilds = guilds[: self._max_guilds]
        return guilds

    def _selected_channels(self, guild_id: str) -> list[dict]:
        allowed_channel_ids = self._guild_filter.get(guild_id)
        channels = [
            channel
            for channel in self._transport.guild_channels(guild_id)
            if channel.get("type") in TEXT_CHANNEL_TYPES
            and (allowed_channel_ids is None or str(channel["id"]) in allowed_channel_ids)
        ]
        channels.sort(key=lambda channel: int(channel["id"]))
        if self._max_channels_per_guild:
            channels = channels[: self._max_channels_per_guild]
        return channels

    def collect(self, window: CollectionWindow, plan: QueryPlan) -> CollectionResult:
        del plan  # discord collection is guild/channel driven, not query driven
        errors: list[str] = []
        stats = {"guilds": 0, "channels": 0, "posts": 0, "pages": 0, "errors": 0}
        posts: list[RawPost] = []

        for guild in self._selected_guilds(errors):
            guild_id = str(guild["id"])
            stats["guilds"] += 1
            for channel in self._selected_channels(guild_id):
                channel_id = str(channel["id"])
                stats["channels"] += 1
                self._collect_channel(guild, channel, channel_id, window, posts, errors, stats)

        stats["posts"] = len(posts)
        stats["errors"] = len(errors)
        posts.sort(key=lambda post: int(post.external_id.split(":", 1)[1]))
        return CollectionResult(platform=self.platform, mode=self.mode, posts=posts, errors=errors, stats=stats)

    def _collect_channel(
        self,
        guild: dict,
        channel: dict,
        channel_id: str,
        window: CollectionWindow,
        posts: list[RawPost],
        errors: list[str],
        stats: dict[str, int],
    ) -> None:
        after = self._last_message_id.get(channel_id)
        max_seen: int | None = None
        page_size = self._max_posts_per_channel
        for _page in range(_MAX_MESSAGE_PAGES):
            try:
                messages = self._transport.channel_messages(channel_id, after=after, limit=page_size)
            except HttpError as error:
                if error.status in (403, 404):
                    errors.append(f"channel {channel_id}: HTTP {error.status}")
                    break
                raise
            stats["pages"] += 1
            if not messages:
                break
            page_max = max(int(message["id"]) for message in messages)
            max_seen = page_max if max_seen is None else max(max_seen, page_max)
            for post in parse_messages(guild, channel, messages):
                if window.contains(post.posted_at):
                    posts.append(post)
            if len(messages) < page_size:
                break
            after = str(page_max)
        if max_seen is not None:
            self._last_message_id[channel_id] = str(max_seen)

    def export_state(self) -> dict:
        return {"last_message_id": dict(sorted(self._last_message_id.items()))}
