"""X (Twitter) API v2 collector: recent search + user timelines.

Runs the same payload parsing against a live ``XLiveTransport`` (real API
calls, bearer-token rotation) or an offline ``XFixtureTransport`` (bundled
fixture files), so ``XCollector`` never knows which one it is talking to.
"""

from __future__ import annotations

from typing import Protocol

from .base import CollectionQuery, CollectionResult, CollectionWindow, CollectorError, QueryPlan, RawPost
from .fixtures import FixtureFile
from .http import HttpClient, HttpError, RateLimited, TokenPool

_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
_USERS_BY_URL = "https://api.twitter.com/2/users/by"
_USER_TWEETS_URL = "https://api.twitter.com/2/users/{user_id}/tweets"

_TWEET_FIELDS = "attachments,author_id,created_at,entities,id,lang,public_metrics,referenced_tweets,text"
_EXPANSIONS = "author_id,attachments.media_keys"
_USER_FIELDS = "username"
_MEDIA_FIELDS = "type,url,preview_image_url,alt_text"

_SEARCH_MIN_RESULTS = 10
_SEARCH_MAX_RESULTS = 100
_TIMELINE_MAX_RESULTS = 100
_TOKEN_REJECTED_STATUSES = (401, 403)

_MAX_SEARCH_PAGES = 5
_MAX_TIMELINE_PAGES = 2
_USER_BATCH_SIZE = 100


class XTransport(Protocol):
    def recent_search(
        self, query: str, *, start_time: str, end_time: str, max_results: int, next_token: str | None
    ) -> dict: ...

    def users_by_usernames(self, usernames: list[str]) -> dict: ...

    def user_tweets(
        self, user_id: str, *, start_time: str, end_time: str, pagination_token: str | None
    ) -> dict: ...


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


class XLiveTransport:
    """Calls the live X API v2, rotating bearer tokens on rate limit/rejection."""

    def __init__(self, tokens: list[str], client: HttpClient, *, pool: TokenPool | None = None) -> None:
        self._client = client
        self._pool = pool if pool is not None else TokenPool(tokens)

    def recent_search(
        self, query: str, *, start_time: str, end_time: str, max_results: int, next_token: str | None
    ) -> dict:
        params = {
            "query": query,
            "max_results": _clamp(max_results, _SEARCH_MIN_RESULTS, _SEARCH_MAX_RESULTS),
            "start_time": start_time,
            "end_time": end_time,
            "tweet.fields": _TWEET_FIELDS,
            "expansions": _EXPANSIONS,
            "user.fields": _USER_FIELDS,
            "media.fields": _MEDIA_FIELDS,
        }
        if next_token:
            params["next_token"] = next_token
        return self._get(_SEARCH_URL, params)

    def users_by_usernames(self, usernames: list[str]) -> dict:
        params = {
            "usernames": ",".join(usernames[:_USER_BATCH_SIZE]),
            "user.fields": _USER_FIELDS,
        }
        return self._get(_USERS_BY_URL, params)

    def user_tweets(
        self, user_id: str, *, start_time: str, end_time: str, pagination_token: str | None
    ) -> dict:
        params = {
            "start_time": start_time,
            "end_time": end_time,
            "max_results": _TIMELINE_MAX_RESULTS,
            "tweet.fields": _TWEET_FIELDS,
            "expansions": _EXPANSIONS,
            "user.fields": _USER_FIELDS,
            "media.fields": _MEDIA_FIELDS,
        }
        if pagination_token:
            params["pagination_token"] = pagination_token
        return self._get(_USER_TWEETS_URL.format(user_id=user_id), params)

    def _get(self, url: str, params: dict) -> dict:
        for token in self._pool.available():
            headers = {"Authorization": f"Bearer {token}"}
            try:
                return self._client.get_json(url, params=params, headers=headers)
            except RateLimited:
                self._pool.mark_rate_limited(token)
                continue
            except HttpError as error:
                if error.status in _TOKEN_REJECTED_STATUSES:
                    continue
                raise CollectorError(str(error)) from error
        raise CollectorError("all X bearer tokens rate-limited or rejected")


class XFixtureTransport:
    """Replays bundled fixture responses instead of calling the live API."""

    def __init__(self, fixture: FixtureFile) -> None:
        self._fixture = fixture

    def recent_search(
        self, query: str, *, start_time: str, end_time: str, max_results: int, next_token: str | None
    ) -> dict:
        if next_token is None:
            responses = self._fixture.responses_for("tweets/search/recent", query=query)
            if not responses:
                responses = self._fixture.responses_for("tweets/search/recent")
        else:
            responses = self._fixture.responses_for(
                "tweets/search/recent", query=query, next_token=next_token
            )
        return responses[0]["payload"] if responses else {"data": []}

    def users_by_usernames(self, usernames: list[str]) -> dict:
        responses = self._fixture.responses_for("users/by")
        return responses[0]["payload"] if responses else {"data": []}

    def user_tweets(
        self, user_id: str, *, start_time: str, end_time: str, pagination_token: str | None
    ) -> dict:
        responses = self._fixture.responses_for("users/{id}/tweets", user_id=user_id)
        return responses[0]["payload"] if responses else {"data": []}


def _expand_text(text: str, entities: dict) -> str:
    for url_entry in entities.get("urls", []) if entities else []:
        short_url = url_entry.get("url")
        expanded_url = url_entry.get("expanded_url")
        if short_url and expanded_url:
            text = text.replace(short_url, expanded_url)
    return text


def _build_attachments(tweet: dict, media_by_key: dict) -> list[dict]:
    media_keys = (tweet.get("attachments") or {}).get("media_keys") or []
    attachments: list[dict] = []
    for key in media_keys:
        media = media_by_key.get(key)
        if not media:
            continue
        attachments.append(
            {
                "type": media.get("type", ""),
                "url": media.get("url") or media.get("preview_image_url", ""),
                "alt_text": media.get("alt_text", ""),
            }
        )
    return attachments


def parse_search_payload(payload: dict, *, source: str, query: str) -> list[RawPost]:
    """Parse a v2 search/timeline payload into RawPosts. Never raises on missing keys."""
    data = payload.get("data") or []
    includes = payload.get("includes") or {}
    users_by_id = {user.get("id"): user for user in includes.get("users", [])}
    media_by_key = {media.get("media_key"): media for media in includes.get("media", [])}

    posts: list[RawPost] = []
    for tweet in data:
        external_id = str(tweet.get("id", ""))
        author_id = str(tweet.get("author_id", ""))
        author = users_by_id.get(author_id)
        author_ref = author.get("username") if author and author.get("username") else author_id
        entities = tweet.get("entities") or {}
        content = _expand_text(str(tweet.get("text", "")), entities)
        metrics = tweet.get("public_metrics") or {}
        metadata = {
            "source": source,
            "query": query,
            "retweet_count": metrics.get("retweet_count", 0),
            "like_count": metrics.get("like_count", 0),
            "hashtags": [tag.get("tag", "") for tag in entities.get("hashtags", [])],
            "referenced_tweets": [ref.get("type", "") for ref in tweet.get("referenced_tweets", [])],
        }
        posts.append(
            RawPost(
                platform="x",
                external_id=external_id,
                content=content,
                posted_at=str(tweet.get("created_at") or ""),
                language=str(tweet.get("lang", "")),
                author_ref=author_ref,
                url=f"https://x.com/{author_ref}/status/{external_id}",
                attachments=_build_attachments(tweet, media_by_key),
                metadata=metadata,
            )
        )
    return posts


def _format_time(moment) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _chunked(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _next_token(payload: dict) -> str | None:
    return (payload.get("meta") or {}).get("next_token")


def _within_window(posts: list[RawPost], window: CollectionWindow) -> list[RawPost]:
    return [post for post in posts if post.posted_at and window.contains(post.posted_at)]


class XCollector:
    platform = "x"

    def __init__(
        self,
        transport: XTransport,
        *,
        users: list[str],
        max_posts_per_query: int = 100,
        mode: str = "live",
    ) -> None:
        self._transport = transport
        self._users = list(users)
        self._max_posts_per_query = max_posts_per_query
        self._mode = mode

    def collect(self, window: CollectionWindow, plan: QueryPlan) -> CollectionResult:
        queries = plan.for_platform("x")
        start_time = _format_time(window.start)
        end_time = _format_time(window.end)

        posts: list[RawPost] = []
        errors: list[str] = []
        pages = 0

        for query in queries:
            try:
                query_posts, query_pages = self._collect_query(query, window, start_time, end_time)
            except (CollectorError, HttpError) as error:
                errors.append(f"{query.query}: {error}")
                continue
            posts.extend(query_posts)
            pages += query_pages

        for batch in _chunked(self._users, _USER_BATCH_SIZE):
            try:
                lookup = self._transport.users_by_usernames(batch)
            except (CollectorError, HttpError) as error:
                errors.append(f"{','.join(batch)}: {error}")
                continue
            for user in lookup.get("data") or []:
                user_id = str(user.get("id", ""))
                username = str(user.get("username", "")) or user_id
                try:
                    timeline_posts, timeline_pages = self._collect_timeline(
                        user_id, username, window, start_time, end_time
                    )
                except (CollectorError, HttpError) as error:
                    errors.append(f"{username}: {error}")
                    continue
                posts.extend(timeline_posts)
                pages += timeline_pages

        stats = {
            "queries": len(queries),
            "timeline_users": len(self._users),
            "posts": len(posts),
            "pages": pages,
            "errors": len(errors),
        }
        return CollectionResult(platform=self.platform, mode=self._mode, posts=posts, errors=errors, stats=stats)

    def _collect_query(
        self, query: CollectionQuery, window: CollectionWindow, start_time: str, end_time: str
    ) -> tuple[list[RawPost], int]:
        posts: list[RawPost] = []
        next_token: str | None = None
        pages = 0
        max_results = _clamp(self._max_posts_per_query, _SEARCH_MIN_RESULTS, _SEARCH_MAX_RESULTS)
        for _ in range(_MAX_SEARCH_PAGES):
            payload = self._transport.recent_search(
                query.query,
                start_time=start_time,
                end_time=end_time,
                max_results=max_results,
                next_token=next_token,
            )
            pages += 1
            posts.extend(_within_window(parse_search_payload(payload, source=query.source, query=query.query), window))
            next_token = _next_token(payload)
            if not next_token or len(posts) >= self._max_posts_per_query:
                break
        return posts, pages

    def _collect_timeline(
        self, user_id: str, username: str, window: CollectionWindow, start_time: str, end_time: str
    ) -> tuple[list[RawPost], int]:
        posts: list[RawPost] = []
        pagination_token: str | None = None
        pages = 0
        for _ in range(_MAX_TIMELINE_PAGES):
            payload = self._transport.user_tweets(
                user_id, start_time=start_time, end_time=end_time, pagination_token=pagination_token
            )
            pages += 1
            posts.extend(_within_window(parse_search_payload(payload, source="timeline", query=username), window))
            pagination_token = _next_token(payload)
            if not pagination_token:
                break
        return posts, pages
