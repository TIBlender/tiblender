"""Reddit collector: OAuth2 app-only client against the official Reddit API.

Two transports share one parsing path (``parse_listing``): ``RedditLiveTransport``
talks to ``oauth.reddit.com`` with rotating app credentials, and
``RedditFixtureTransport`` replays recorded fixture payloads for offline cycles.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Callable, Protocol

from .base import CollectionResult, CollectionWindow, CollectorError, QueryPlan, RawPost
from .fixtures import FixtureFile
from .http import HttpClient, HttpError, RateLimited

_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_OAUTH_BASE = "https://oauth.reddit.com"
_TOKEN_EXPIRY_SAFETY_SECONDS = 60
_MAX_PAGES_PER_QUERY = 3
_EMPTY_LISTING: dict = {"kind": "Listing", "data": {"children": [], "after": None}}


class _MalformedTokenResponse(ValueError):
    """Raised when a token endpoint response is unusable.

    Subclasses ValueError so it (and json.JSONDecodeError, also a ValueError
    subclass) is caught by the same rotate-to-next-account handling as a
    rejected/rate-limited account, instead of escaping the rotation loop.
    """


def _validated_access_token(payload: object) -> str:
    if not isinstance(payload, dict):
        raise _MalformedTokenResponse(f"malformed token response: expected an object, got {type(payload).__name__}")
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise _MalformedTokenResponse("malformed token response: missing or empty access_token")
    return token


def _validated_expires_in(payload: dict) -> float:
    raw = payload.get("expires_in", 3600)
    try:
        return float(raw)
    except (TypeError, ValueError) as error:
        raise _MalformedTokenResponse(f"malformed token response: non-numeric expires_in ({raw!r})") from error


class RedditTransport(Protocol):
    def subreddit_new(self, subreddit: str, *, limit: int, after: str | None) -> dict: ...

    def search(self, query: str, *, limit: int, after: str | None) -> dict: ...


class RedditLiveTransport:
    """Talks to oauth.reddit.com using app-only (client_credentials) tokens.

    Rotates across ``accounts`` round-robin, starting from the last account
    that successfully served a request, whenever the current one is rate
    limited or rejected (401/403). Any other HTTP failure is not retried on
    another account; it propagates immediately as a ``CollectorError``.
    """

    def __init__(
        self,
        accounts: list[tuple[str, str]],
        client: HttpClient,
        *,
        user_agent: str,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._accounts = list(accounts)
        self._client = client
        self._user_agent = user_agent
        self._now = now
        self._tokens: dict[int, tuple[str, float]] = {}
        self._current_index = 0

    def subreddit_new(self, subreddit: str, *, limit: int, after: str | None) -> dict:
        params = {"limit": limit, "raw_json": 1}
        if after is not None:
            params["after"] = after
        return self._get(f"{_OAUTH_BASE}/r/{subreddit}/new", params)

    def search(self, query: str, *, limit: int, after: str | None) -> dict:
        params = {"q": query, "sort": "new", "t": "day", "limit": limit, "raw_json": 1}
        if after is not None:
            params["after"] = after
        return self._get(f"{_OAUTH_BASE}/search", params)

    def _get(self, url: str, params: dict) -> dict:
        def attempt(token: str) -> dict:
            headers = {"Authorization": f"bearer {token}", "User-Agent": self._user_agent}
            return self._client.get_json(url, params=params, headers=headers)

        return self._with_account_rotation(attempt)

    def _with_account_rotation(self, attempt: Callable[[str], dict]) -> dict:
        account_count = len(self._accounts)
        if account_count == 0:
            raise CollectorError("all Reddit accounts rate-limited or rejected")

        start = self._current_index
        last_error: Exception | None = None
        for offset in range(account_count):
            index = (start + offset) % account_count
            try:
                token = self._token_for(index)
                result = attempt(token)
            except RateLimited as error:
                last_error = error
                continue
            except HttpError as error:
                if error.status in (401, 403):
                    last_error = error
                    continue
                raise CollectorError(str(error)) from error
            except (ValueError, KeyError, TypeError) as error:
                # Malformed token/listing responses (including a non-JSON
                # body surfacing as json.JSONDecodeError, a ValueError
                # subclass) are treated like a rejected account: try the
                # next one instead of letting a bad response from one
                # account starve every other healthy account.
                last_error = error
                continue
            self._current_index = index
            return result
        raise CollectorError(f"all Reddit accounts rate-limited or rejected: {last_error}") from last_error

    def _token_for(self, index: int) -> str:
        cached = self._tokens.get(index)
        now = self._now()
        if cached is not None and now < cached[1]:
            return cached[0]

        client_id, client_secret = self._accounts[index]
        payload = self._client.post_form(
            _TOKEN_URL,
            {"grant_type": "client_credentials"},
            headers={"User-Agent": self._user_agent},
            basic_auth=(client_id, client_secret),
        )
        token = _validated_access_token(payload)
        expires_in = _validated_expires_in(payload)
        expires_at = now + expires_in - _TOKEN_EXPIRY_SAFETY_SECONDS
        self._tokens[index] = (token, expires_at)
        return token


class RedditFixtureTransport:
    """Replays recorded Reddit listing payloads from a ``FixtureFile``."""

    def __init__(self, fixture: FixtureFile) -> None:
        self._fixture = fixture

    def subreddit_new(self, subreddit: str, *, limit: int, after: str | None) -> dict:
        match: dict[str, str] = {"subreddit": subreddit}
        if after is not None:
            match["after"] = after
        responses = self._fixture.responses_for("r/{sub}/new", **match)
        if responses:
            return responses[0]["payload"]
        return _EMPTY_LISTING

    def search(self, query: str, *, limit: int, after: str | None) -> dict:
        match: dict[str, str] = {"query": query}
        if after is not None:
            match["after"] = after
        responses = self._fixture.responses_for("search", **match)
        if not responses:
            fallback_match: dict[str, str] = {}
            if after is not None:
                fallback_match["after"] = after
            responses = self._fixture.responses_for("search", **fallback_match)
        if responses:
            return responses[0]["payload"]
        return _EMPTY_LISTING


def parse_listing(payload: dict, *, source: str, query: str) -> list[RawPost]:
    """Turn a Reddit ``Listing`` payload into ``RawPost`` records (t3 children only)."""
    children = payload.get("data", {}).get("children", [])
    posts: list[RawPost] = []
    for child in children:
        if child.get("kind") != "t3":
            continue
        posts.append(_parse_post(child.get("data", {}), source=source, query=query))
    return posts


def _parse_post(data: dict, *, source: str, query: str) -> RawPost:
    title = str(data.get("title") or "")
    selftext = str(data.get("selftext") or "").strip()
    content = title if not selftext else f"{title}\n\n{selftext}"

    return RawPost(
        platform="reddit",
        external_id=str(data.get("id", "")),
        content=content,
        posted_at=_posted_at(data.get("created_utc")),
        language="",
        author_ref=str(data.get("author") or ""),
        url=f"https://www.reddit.com{data.get('permalink', '')}",
        attachments=_attachments(data),
        metadata={
            "source": source,
            "query": query,
            "subreddit": data.get("subreddit") or "",
            "score": data.get("score") or 0,
            "num_comments": data.get("num_comments") or 0,
            "link_flair_text": data.get("link_flair_text") or "",
            "is_self": bool(data.get("is_self", False)),
            "stickied": bool(data.get("stickied", False)),
        },
    )


def _posted_at(created_utc: object) -> str:
    if created_utc is None:
        return ""
    try:
        timestamp = float(created_utc)
    except (TypeError, ValueError):
        return ""
    return datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _attachments(data: dict) -> list[dict]:
    if data.get("post_hint") != "image":
        return []
    images = data.get("preview", {}).get("images", [])
    if not images:
        return []
    url = images[0].get("source", {}).get("url")
    if not url:
        return []
    return [{"type": "image", "url": url, "alt_text": ""}]


def _parse_window_edge(iso_timestamp: str, tzinfo) -> datetime:
    parsed = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tzinfo)
    return parsed


class RedditCollector:
    """Collects Reddit posts for a query plan via subreddit ``new`` and site search."""

    platform = "reddit"

    def __init__(self, transport: RedditTransport, *, max_posts_per_query: int = 100, mode: str = "live") -> None:
        self._transport = transport
        self._limit = min(max_posts_per_query, 100)
        self._mode = mode

    def collect(self, window: CollectionWindow, plan: QueryPlan) -> CollectionResult:
        posts: list[RawPost] = []
        errors: list[str] = []
        stats = {"subreddits": 0, "searches": 0, "posts": 0, "pages": 0, "errors": 0}

        for query in plan.for_platform(self.platform):
            if query.source == "subreddit":
                stats["subreddits"] += 1
            else:
                stats["searches"] += 1
            try:
                posts.extend(self._collect_query(query, window, stats, errors))
            except Exception as error:  # noqa: BLE001 - captured per-query, collection continues
                errors.append(f"{query.query}: {error}")

        stats["posts"] = len(posts)
        stats["errors"] = len(errors)
        return CollectionResult(platform=self.platform, mode=self._mode, posts=posts, errors=errors, stats=stats)

    def _collect_query(
        self, query, window: CollectionWindow, stats: dict[str, int], errors: list[str]
    ) -> list[RawPost]:
        is_subreddit = query.source == "subreddit"
        kept: list[RawPost] = []
        skipped_missing_created_utc = 0
        after: str | None = None

        for _page in range(_MAX_PAGES_PER_QUERY):
            if is_subreddit:
                payload = self._transport.subreddit_new(query.query, limit=self._limit, after=after)
            else:
                payload = self._transport.search(query.query, limit=self._limit, after=after)
            stats["pages"] += 1

            page_posts = parse_listing(payload, source=query.source, query=query.query)
            # A post with a missing/invalid created_utc gets posted_at="" from
            # parse_listing; it can't be dated, so it must not feed the
            # window filter (window.contains("") would raise ValueError) nor
            # the early-stop "oldest post on this page" check below, or one
            # malformed post would wrongly truncate pagination for every
            # later, legitimately dated page.
            dated_posts = [post for post in page_posts if post.posted_at]
            skipped_missing_created_utc += len(page_posts) - len(dated_posts)
            kept.extend(post for post in dated_posts if window.contains(post.posted_at))

            after = payload.get("data", {}).get("after")
            if not after:
                break
            if is_subreddit and dated_posts:
                oldest_iso = min(post.posted_at for post in dated_posts)
                oldest = _parse_window_edge(oldest_iso, window.start.tzinfo)
                if oldest < window.start:
                    break

        if skipped_missing_created_utc > 0:
            errors.append(f"{query.query}: {skipped_missing_created_utc} post(s) without created_utc skipped")

        return kept
