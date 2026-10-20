"""Tool result primitives and social/corpus search adapters.

The shared query/result helpers live here so both this module and
:mod:`tiblender_runtime.tools` can use them without an import cycle;
``tools`` re-exports every public name so existing call sites keep working.
X and Reddit fall back to the Step 1 collected corpus, while Discord and
Telegram search that corpus exclusively - no private platform API is contacted.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from pathlib import Path
from typing import Any

from . import clock
from .iocs import extract_iocs
from .models import IocConfidence, IocSet, ToolQuery, ToolResult
from .settings import Settings


class ToolUnavailableError(RuntimeError):
    """Raised when an external provider is unavailable, unauthorized, or rate-limited."""


def _now() -> str:
    return clock.now_iso()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)[:12000]


def _param(query: ToolQuery, *names: str) -> str:
    for name in names:
        value = query.parameters.get(name)
        if value:
            return str(value).strip()
    return ""


def _result(
    query: ToolQuery,
    *,
    summary: str,
    status: str = "success",
    result_url: str = "",
    discovered_iocs: IocSet | None = None,
    confidence: IocConfidence = IocConfidence.MEDIUM,
    error: str = "",
    raw: Any = None,
) -> ToolResult:
    return ToolResult(
        tool_type=query.tool_type,
        query_params=dict(query.parameters),
        summary=summary,
        result_url=result_url,
        timestamp=_now(),
        discovered_iocs=discovered_iocs or IocSet(),
        confidence=confidence,
        status=status,
        error=error,
        raw_json=_json(raw) if raw is not None else "",
    )


def _unavailable(query: ToolQuery, reason: str) -> ToolResult:
    return _result(query, summary=reason, status="unavailable", confidence=IocConfidence.LOW, error=reason)


def _search_result_text(items: list[dict[str, Any]], *keys: str) -> str:
    parts: list[str] = []
    for item in items:
        for key in keys:
            value = item.get(key)
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(parts)


def _read_secret_lines(path_value: str) -> list[str]:
    # Missing production-mounted secret files are tolerated so the runtime can
    # still run from environment variables or bundled collected posts.
    if not path_value:
        return []
    path = Path(path_value).expanduser()
    if not path.exists() or not path.is_file():
        return []
    values: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values.append(urllib.parse.unquote(line))
    return values


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _x_bearer_tokens(settings: Settings) -> list[str]:
    # The production X collector rotates file-backed bearer tokens and falls
    # back to environment credentials. This compact runtime preserves that
    # credential order without retaining token values in outputs.
    return _dedupe(
        [
            settings.x_bearer_token,
            *_read_secret_lines(settings.x_bearer_token_file),
            *_read_secret_lines(settings.x_bearer_fallback_token_file),
        ]
    )


def _reddit_accounts(settings: Settings) -> list[tuple[str, str]]:
    accounts: list[tuple[str, str]] = []
    for line in _read_secret_lines(settings.reddit_accounts_file):
        client_id, sep, client_secret = line.partition(":")
        if sep and client_id.strip() and client_secret.strip():
            accounts.append((client_id.strip(), client_secret.strip()))
    if settings.reddit_client_id and settings.reddit_client_secret:
        accounts.append((settings.reddit_client_id.strip(), settings.reddit_client_secret.strip()))
    for raw_account in settings.reddit_extra_accounts.split(","):
        client_id, sep, client_secret = raw_account.strip().partition(":")
        if sep and client_id.strip() and client_secret.strip():
            accounts.append((client_id.strip(), client_secret.strip()))
    return list(dict.fromkeys(accounts))


class SocialSearchMixin:
    """Social-platform and collected-corpus searches composed into ToolRunner."""

    def _corpus_matches(self, query: ToolQuery, platform: str | None = None) -> list[dict[str, Any]]:
        # Platform collectors populate the corpus before report synthesis. When
        # a live provider is temporarily unavailable, this preserves the same
        # investigation path over already collected social posts.
        search = _param(query, "query").lower()
        terms = [term for term in re.split(r"\s+", search) if term]
        matched: list[dict[str, Any]] = []
        for post in self.corpus:
            if platform and post.platform.lower() != platform:
                continue
            text = post.normalized_content or post.post.content
            lowered = text.lower()
            if not terms or any(term in lowered for term in terms):
                matched.append(
                    {
                        "id": post.id,
                        "platform": post.platform,
                        "timestamp": post.post.timestamp,
                        "content": text[:700],
                    }
                )
        return matched

    def _corpus_search_result(
        self,
        query: ToolQuery,
        *,
        platform: str | None,
        label: str,
        include_iocs: bool,
    ) -> ToolResult | None:
        matched = self._corpus_matches(query, platform=platform)
        if not matched:
            return None
        combined = _search_result_text(matched, "content")
        summary = (
            f"{label} collected corpus search for {_param(query, 'query') or '<empty>'}: "
            f"{len(matched)} collected post(s). {combined[:500]}"
        )
        return _result(
            query,
            summary=summary,
            discovered_iocs=extract_iocs(combined) if include_iocs else IocSet(),
            raw={
                "fallback_source": "collected_corpus",
                "result_count": len(matched),
                "posts": matched[: self.settings.max_tool_results],
            },
        )

    def _x_search(self, query: ToolQuery) -> ToolResult:
        # Imported at call time so the HTTP helpers keep living in `tools` (the
        # module tests patch) without creating an import cycle with this module.
        from .tools import _headers, _http_json

        search = _param(query, "query")
        if not search:
            return _unavailable(query, "X_SEARCH requires a query parameter.")
        params = urllib.parse.urlencode(
            {
                "query": search,
                "max_results": 10,
                "tweet.fields": "attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,lang,public_metrics,referenced_tweets,source,text",
                "user.fields": "created_at,description,id,location,name,public_metrics,username,verified,verified_type",
                "media.fields": "alt_text,duration_ms,height,media_key,preview_image_url,public_metrics,type,url,width",
                "expansions": "attachments.media_keys,author_id,entities.mentions.username,referenced_tweets.id,referenced_tweets.id.author_id",
            },
            safe=",:",
        )
        url = f"https://api.twitter.com/2/tweets/search/recent?{params}"
        tokens = _x_bearer_tokens(self.settings)
        if tokens:
            for token in tokens:
                try:
                    data, final_url = _http_json(
                        url,
                        headers=_headers({"Authorization": f"Bearer {token}"}),
                        timeout=query.timeout_seconds or self.settings.tool_timeout_seconds,
                    )
                except ToolUnavailableError:
                    continue
                tweets = data.get("data", []) if isinstance(data, dict) else []
                text = _search_result_text(tweets, "text")
                summary = f"X recent search for {search}: {len(tweets)} tweet(s). {text[:500]}"
                return _result(
                    query,
                    summary=summary,
                    result_url=final_url,
                    discovered_iocs=extract_iocs(text),
                    raw={"result_count": len(tweets), "tweets": tweets[: self.settings.max_tool_results]},
                )
        corpus_result = self._corpus_search_result(query, platform="x", label="X", include_iocs=True)
        if corpus_result:
            return corpus_result
        return _unavailable(query, "X_SEARCH unavailable: requires usable X bearer credentials or collected X posts.")

    def _reddit_search(self, query: ToolQuery) -> ToolResult:
        search = _param(query, "query")
        if not search:
            return _unavailable(query, "REDDIT_SEARCH requires a query parameter.")
        accounts = _reddit_accounts(self.settings)
        query_markers = [value.lower() for value in extract_iocs(search).all_values()]
        if accounts:
            try:
                import praw
            except ImportError:
                praw = None
            if praw is not None:
                for client_id, client_secret in accounts:
                    try:
                        reddit = praw.Reddit(
                            client_id=client_id,
                            client_secret=client_secret,
                            user_agent=self.settings.reddit_user_agent,
                        )
                        submissions = list(
                            reddit.subreddit("all").search(
                                search,
                                sort="new",
                                limit=self.settings.max_tool_results,
                            )
                        )
                    except Exception:  # noqa: BLE001
                        continue
                    posts = [
                        {
                            "title": getattr(submission, "title", "") or "",
                            "selftext": getattr(submission, "selftext", "") or "",
                            "permalink": f"https://reddit.com{getattr(submission, 'permalink', '')}",
                            "url": getattr(submission, "url", "") or "",
                            "created_utc": getattr(submission, "created_utc", None),
                        }
                        for submission in submissions
                    ]
                    if query_markers:
                        posts = [
                            post
                            for post in posts
                            if any(
                                marker
                                in " ".join(str(post.get(key) or "") for key in ("title", "selftext", "url")).lower()
                                for marker in query_markers
                            )
                        ]
                        if not posts:
                            continue
                    text = _search_result_text(posts, "title", "selftext", "url")
                    summary = f"Reddit OAuth search for {search}: {len(posts)} post(s). {text[:500]}"
                    return _result(
                        query,
                        summary=summary,
                        result_url="https://www.reddit.com/search/",
                        discovered_iocs=extract_iocs(text),
                        raw={"result_count": len(posts), "posts": posts[: self.settings.max_tool_results]},
                    )
        corpus_result = self._corpus_search_result(query, platform="reddit", label="Reddit", include_iocs=True)
        if corpus_result:
            return corpus_result
        return _unavailable(query, "REDDIT_SEARCH requires usable Reddit API credentials or collected Reddit posts.")

    def _discord_search(self, query: ToolQuery) -> ToolResult:
        return self._internal_platform_search(query, "discord")

    def _telegram_search(self, query: ToolQuery) -> ToolResult:
        return self._internal_platform_search(query, "telegram")

    def _internal_platform_search(self, query: ToolQuery, platform: str) -> ToolResult:
        # Discord and Telegram search operate over the already-collected corpus
        # in this compact implementation; no private platform API is contacted here.
        search = _param(query, "query").lower()
        matched = self._corpus_matches(query, platform=platform)
        combined = _search_result_text(matched, "content")
        summary = (
            f"{platform.title()} internal corpus search for {search or '<empty>'}: "
            f"{len(matched)} matching collected post(s). {combined[:500]}"
        )
        return _result(
            query,
            summary=summary,
            discovered_iocs=extract_iocs(combined),
            raw={"result_count": len(matched), "posts": matched[: self.settings.max_tool_results]},
        )
