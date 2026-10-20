"""Deterministic id generation and per-platform payload shape builders.

Each `*_response(...)` helper returns one `{"request": ..., "payload": ...}`
entry, matching exactly what `tiblender_runtime.collectors.fixtures.FixtureFile`
and the platform parsers (`x.parse_search_payload`, `reddit.parse_listing`,
`telegram.parse_messages`, `discord.parse_messages`) expect.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

_BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"

# Fixed Discord demo guild/channel reused across every cycle (fixed, not
# derived) so replaying any cycle talks about the same guild/channel.
DISCORD_GUILD_ID = "900100200300400500"
DISCORD_GUILD_NAME = "artifact-demo-guild"
DISCORD_CHANNEL_ID = "910100200300400500"
DISCORD_CHANNEL_NAME = "threat-intel"

# Discord's own snowflake epoch (2015-01-01T00:00:00Z) in milliseconds.
_DISCORD_EPOCH_MS = 1_420_070_400_000


@dataclass(frozen=True, slots=True)
class PreparedPost:
    """A single anonymized post, ready to be rendered into a platform payload."""

    platform: str
    post_ref: str
    report_id: str
    category: str
    kind: str  # "primary" | "noise" | "dataset_preview"
    cycle_index: int
    text: str  # already anonymized
    posted_at: datetime  # tz-aware, UTC
    author: str  # pseudonym


def _digest_int(seed: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{seed}".encode("utf-8")).hexdigest(), 16)


def numeric_id(seed: str, salt: str, length: int) -> str:
    """A deterministic all-digit id of exactly `length` digits, never leading with 0."""
    digits = str(_digest_int(seed, salt) % (10**length)).zfill(length)
    if digits[0] == "0":
        digits = "1" + digits[1:]
    return digits


def base36_id(seed: str, salt: str, length: int = 7) -> str:
    """A deterministic lowercase base36 id of exactly `length` characters."""
    remaining = _digest_int(seed, salt)
    chars: list[str] = []
    for _ in range(length):
        remaining, remainder = divmod(remaining, 36)
        chars.append(_BASE36_ALPHABET[remainder])
    return "".join(reversed(chars))


def iso_z(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def x_response(posts: list[PreparedPost], *, salt: str, query: str = "artifact") -> dict:
    tweets: list[dict] = []
    users: list[dict] = []
    seen_authors: set[str] = set()
    for post in posts:
        author_id = numeric_id(f"{post.post_ref}:author", salt, 12)
        tweets.append(
            {
                "id": numeric_id(post.post_ref, salt, 18),
                "author_id": author_id,
                "created_at": post.posted_at.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "lang": "en",
                "text": post.text,
                "entities": {"urls": []},
                "public_metrics": {"retweet_count": 0, "like_count": 0},
            }
        )
        if author_id not in seen_authors:
            seen_authors.add(author_id)
            users.append({"id": author_id, "username": post.author})
    return {
        "request": {"endpoint": "tweets/search/recent", "query": query},
        "payload": {
            "data": tweets,
            "includes": {"users": users},
            "meta": {"result_count": len(tweets)},
        },
    }


def reddit_response(posts: list[PreparedPost], *, subreddit: str, salt: str) -> dict:
    children: list[dict] = []
    for post in posts:
        post_id = base36_id(post.post_ref, salt, 7)
        title = post.text[:120]
        selftext = post.text[120:].strip()
        children.append(
            {
                "kind": "t3",
                "data": {
                    "id": post_id,
                    "title": title,
                    "selftext": selftext,
                    "author": post.author,
                    "created_utc": int(post.posted_at.timestamp()),
                    "permalink": f"/r/{subreddit}/comments/{post_id}/",
                    "subreddit": subreddit,
                    "score": 0,
                    "num_comments": 0,
                    "is_self": True,
                    "stickied": False,
                },
            }
        )
    return {
        "request": {"endpoint": "r/{sub}/new", "subreddit": subreddit},
        "payload": {"kind": "Listing", "data": {"children": children, "after": None}},
    }


def telegram_channel_id(channel_username: str, *, salt: str) -> str:
    return numeric_id(f"telegram-channel:{channel_username}", salt, 10)


def telegram_dialogs_response(chats: list[dict]) -> dict:
    """The joined-channel listing replayed for `client.get_dialogs()`.

    One response per fixture file lists every channel the demo account is a
    member of; the collector reads them all (collection is exhaustive over
    joined communities, not query driven).
    """
    return {"request": {"endpoint": "dialogs"}, "payload": {"chats": chats}}


def telegram_messages_response(posts: list[PreparedPost], *, channel_username: str, salt: str) -> dict:
    channel_id = telegram_channel_id(channel_username, salt=salt)
    ordered = sorted(posts, key=lambda post: (post.posted_at, post.post_ref))
    base_message_id = int(numeric_id(f"telegram-messages:{channel_username}", salt, 6))
    messages = []
    for index, post in enumerate(ordered):
        messages.append(
            {
                "id": base_message_id + index,
                "date": iso_z(post.posted_at),
                "message": post.text,
                "views": 0,
                "forwards": 0,
                "fwd_from_username": "",
                "media": {},
            }
        )
    return {
        "request": {"endpoint": "messages", "channel_id": channel_id},
        "payload": messages,
    }


def discord_guild_and_channel_responses() -> tuple[dict, dict]:
    guilds_response = {
        "request": {"endpoint": "users/@me/guilds"},
        "payload": [{"id": DISCORD_GUILD_ID, "name": DISCORD_GUILD_NAME}],
    }
    channels_response = {
        "request": {"endpoint": "guilds/{id}/channels", "guild_id": DISCORD_GUILD_ID},
        "payload": [{"id": DISCORD_CHANNEL_ID, "name": DISCORD_CHANNEL_NAME, "type": 0}],
    }
    return guilds_response, channels_response


def discord_snowflake(posted_at: datetime, seed: str, salt: str) -> str:
    """A Discord-shaped snowflake id: timestamp in the high bits, hash in the low.

    Real snowflakes encode their own creation time
    (``(ms - DISCORD_EPOCH) << 22 | worker | process | increment``), which is
    what makes the collector's ``after`` watermark a safe incremental cursor:
    a message posted later always has a larger id. A purely hash-derived id
    does not have that property, and because every cycle replays the same
    demo channel, cycle 2's watermark would then silently suppress most of
    cycle 3's messages on replay. The low 22 bits stay deterministic-random so
    two posts in the same millisecond still get distinct ids.
    """
    milliseconds = int(posted_at.timestamp() * 1000) - _DISCORD_EPOCH_MS
    low_bits = _digest_int(seed, salt) % (1 << 22)
    return str((max(0, milliseconds) << 22) | low_bits)


def discord_messages_response(posts: list[PreparedPost], *, salt: str) -> dict:
    ordered = sorted(posts, key=lambda post: (post.posted_at, post.post_ref))
    messages = []
    for post in ordered:
        messages.append(
            {
                "id": discord_snowflake(post.posted_at, post.post_ref, salt),
                "timestamp": post.posted_at.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"),
                "content": post.text,
                "author": {"id": numeric_id(f"{post.post_ref}:author", salt, 12), "username": post.author, "bot": False},
                "attachments": [],
            }
        )
    return {
        "request": {"endpoint": "channels/{id}/messages", "channel_id": DISCORD_CHANNEL_ID},
        "payload": messages,
    }
