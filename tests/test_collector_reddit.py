from datetime import UTC, datetime

import pytest

from tiblender_runtime.collectors.base import (
    CollectionQuery,
    CollectionWindow,
    CollectorError,
    QueryPlan,
    RawPost,
)
from tiblender_runtime.collectors.fixtures import FixtureFile
from tiblender_runtime.collectors.http import HttpError, RateLimited
from tiblender_runtime.collectors.reddit import (
    RedditCollector,
    RedditFixtureTransport,
    RedditLiveTransport,
    parse_listing,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeHttpClient:
    """Records calls; replays canned responses/exceptions from post_form/get_json."""

    def __init__(self, *, token_responses=None, get_responses=None):
        self.post_calls: list[dict] = []
        self.get_calls: list[dict] = []
        self._token_responses = list(token_responses or [])
        self._get_responses = list(get_responses or [])

    def post_form(self, url, data, *, headers=None, basic_auth=None):
        self.post_calls.append({"url": url, "data": data, "headers": headers, "basic_auth": basic_auth})
        result = self._token_responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def get_json(self, url, *, params=None, headers=None):
        self.get_calls.append({"url": url, "params": params, "headers": headers})
        result = self._get_responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeTransport:
    """A RedditTransport double that returns queued listing pages and records calls."""

    def __init__(self, *, subreddit_pages=None, search_pages=None):
        self.subreddit_pages = list(subreddit_pages or [])
        self.search_pages = list(search_pages or [])
        self.subreddit_calls: list[dict] = []
        self.search_calls: list[dict] = []

    def subreddit_new(self, subreddit, *, limit, after):
        self.subreddit_calls.append({"subreddit": subreddit, "limit": limit, "after": after})
        if not self.subreddit_pages:
            return {"kind": "Listing", "data": {"children": [], "after": None}}
        return self.subreddit_pages.pop(0)

    def search(self, query, *, limit, after):
        self.search_calls.append({"query": query, "limit": limit, "after": after})
        if not self.search_pages:
            return {"kind": "Listing", "data": {"children": [], "after": None}}
        return self.search_pages.pop(0)


class ExplodingTransport:
    def subreddit_new(self, subreddit, *, limit, after):
        raise CollectorError("boom")

    def search(self, query, *, limit, after):
        raise CollectorError("boom")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _child(
    id_,
    title,
    *,
    selftext="",
    created_utc=1700000000.0,
    permalink="/r/netsec/comments/abc/title/",
    author="alice",
    subreddit="netsec",
    score=10,
    num_comments=2,
    link_flair_text=None,
    is_self=True,
    stickied=False,
    post_hint=None,
    preview_url=None,
    kind="t3",
    extra=None,
):
    data = {
        "id": id_,
        "title": title,
        "selftext": selftext,
        "created_utc": created_utc,
        "permalink": permalink,
        "author": author,
        "subreddit": subreddit,
        "score": score,
        "num_comments": num_comments,
        "link_flair_text": link_flair_text,
        "is_self": is_self,
        "stickied": stickied,
    }
    if post_hint is not None:
        data["post_hint"] = post_hint
    if preview_url is not None:
        data["preview"] = {"images": [{"source": {"url": preview_url}}]}
    if extra:
        data.update(extra)
    return {"kind": kind, "data": data}


def _listing(children, after=None):
    return {"kind": "Listing", "data": {"children": children, "after": after}}


# ---------------------------------------------------------------------------
# parse_listing
# ---------------------------------------------------------------------------


def test_parse_listing_content_is_title_plus_selftext_when_present():
    payload = _listing([_child("a1", "Title A", selftext="Body text here")])
    posts = parse_listing(payload, source="subreddit", query="netsec")
    assert posts[0].content == "Title A\n\nBody text here"


def test_parse_listing_content_is_title_only_when_selftext_empty():
    payload = _listing([_child("a1", "Title A", selftext="")])
    posts = parse_listing(payload, source="subreddit", query="netsec")
    assert posts[0].content == "Title A"


def test_parse_listing_content_is_title_only_when_selftext_whitespace():
    payload = _listing([_child("a1", "Title A", selftext="   \n  ")])
    posts = parse_listing(payload, source="subreddit", query="netsec")
    assert posts[0].content == "Title A"


def test_parse_listing_permalink_becomes_absolute_url():
    payload = _listing([_child("a1", "T", permalink="/r/netsec/comments/xyz/t/")])
    posts = parse_listing(payload, source="subreddit", query="netsec")
    assert posts[0].url == "https://www.reddit.com/r/netsec/comments/xyz/t/"


def test_parse_listing_created_utc_converts_to_iso_z():
    payload = _listing([_child("a1", "T", created_utc=1700000000.0)])
    posts = parse_listing(payload, source="subreddit", query="netsec")
    assert posts[0].posted_at == "2023-11-14T22:13:20Z"


def test_parse_listing_external_id_author_and_language():
    payload = _listing([_child("a1", "T", author="bob")])
    posts = parse_listing(payload, source="subreddit", query="netsec")
    post = posts[0]
    assert post.external_id == "a1"
    assert post.author_ref == "bob"
    assert post.language == ""
    assert post.platform == "reddit"


def test_parse_listing_stickied_and_full_metadata_shape():
    payload = _listing(
        [_child("a1", "T", subreddit="netsec", score=42, num_comments=7, link_flair_text="News", is_self=False, stickied=True)]
    )
    posts = parse_listing(payload, source="subreddit", query="netsec")
    assert posts[0].metadata == {
        "source": "subreddit",
        "query": "netsec",
        "subreddit": "netsec",
        "score": 42,
        "num_comments": 7,
        "link_flair_text": "News",
        "is_self": False,
        "stickied": True,
    }


def test_parse_listing_metadata_defaults_when_fields_missing():
    payload = _listing([{"kind": "t3", "data": {"id": "a1", "title": "T", "created_utc": 1700000000.0, "permalink": "/x/"}}])
    posts = parse_listing(payload, source="search", query="ransomware")
    assert posts[0].metadata == {
        "source": "search",
        "query": "ransomware",
        "subreddit": "",
        "score": 0,
        "num_comments": 0,
        "link_flair_text": "",
        "is_self": False,
        "stickied": False,
    }
    assert posts[0].attachments == []


def test_parse_listing_image_attachment_from_preview():
    payload = _listing(
        [_child("a1", "T", post_hint="image", preview_url="https://preview.redd.it/a1.jpg?auto=webp&s=abc")]
    )
    posts = parse_listing(payload, source="subreddit", query="netsec")
    assert posts[0].attachments == [
        {"type": "image", "url": "https://preview.redd.it/a1.jpg?auto=webp&s=abc", "alt_text": ""}
    ]


def test_parse_listing_no_attachment_when_post_hint_not_image():
    payload = _listing([_child("a1", "T", post_hint="link")])
    posts = parse_listing(payload, source="subreddit", query="netsec")
    assert posts[0].attachments == []


def test_parse_listing_no_attachment_when_preview_missing_despite_image_hint():
    payload = _listing([_child("a1", "T", post_hint="image")])
    posts = parse_listing(payload, source="subreddit", query="netsec")
    assert posts[0].attachments == []


def test_parse_listing_skips_non_t3_children():
    payload = _listing([_child("a1", "T", kind="t1"), _child("a2", "U", kind="t3")])
    posts = parse_listing(payload, source="subreddit", query="netsec")
    assert [p.external_id for p in posts] == ["a2"]


def test_parse_listing_empty_children_returns_empty_list():
    assert parse_listing(_listing([]), source="subreddit", query="netsec") == []


def test_parse_listing_missing_created_utc_yields_empty_posted_at():
    payload = _listing([_child("a1", "T", created_utc=None)])
    posts = parse_listing(payload, source="subreddit", query="netsec")
    assert posts[0].posted_at == ""


def test_parse_listing_non_numeric_created_utc_yields_empty_posted_at():
    payload = _listing([{"kind": "t3", "data": {"id": "a1", "title": "T", "created_utc": "not-a-number", "permalink": "/x/"}}])
    posts = parse_listing(payload, source="subreddit", query="netsec")
    assert posts[0].posted_at == ""


# ---------------------------------------------------------------------------
# RedditFixtureTransport
# ---------------------------------------------------------------------------


def _fixture(responses):
    return FixtureFile(
        platform="reddit",
        window=CollectionWindow(
            start=datetime(2026, 3, 20, 9, tzinfo=UTC), end=datetime(2026, 3, 20, 10, tzinfo=UTC), cycle_index=1, label="cycle_001"
        ),
        provenance="recorded:2026-03-20T10:00:00Z",
        responses=responses,
    )


def test_fixture_transport_subreddit_new_matches_subreddit():
    payload = _listing([_child("a1", "T")])
    fixture = _fixture([{"request": {"endpoint": "r/{sub}/new", "subreddit": "netsec"}, "payload": payload}])
    transport = RedditFixtureTransport(fixture)
    assert transport.subreddit_new("netsec", limit=25, after=None) == payload


def test_fixture_transport_subreddit_new_empty_listing_when_no_match():
    fixture = _fixture([{"request": {"endpoint": "r/{sub}/new", "subreddit": "other"}, "payload": _listing([_child("a1", "T")])}])
    transport = RedditFixtureTransport(fixture)
    assert transport.subreddit_new("netsec", limit=25, after=None) == {"kind": "Listing", "data": {"children": [], "after": None}}


def test_fixture_transport_subreddit_new_pagination_matches_after():
    page1 = _listing([_child("a1", "T")], after="t3_a1")
    page2 = _listing([_child("a2", "U")], after=None)
    fixture = _fixture(
        [
            {"request": {"endpoint": "r/{sub}/new", "subreddit": "netsec"}, "payload": page1},
            {"request": {"endpoint": "r/{sub}/new", "subreddit": "netsec", "after": "t3_a1"}, "payload": page2},
        ]
    )
    transport = RedditFixtureTransport(fixture)
    assert transport.subreddit_new("netsec", limit=25, after=None) == page1
    assert transport.subreddit_new("netsec", limit=25, after="t3_a1") == page2


def test_fixture_transport_search_matches_query():
    payload = _listing([_child("a1", "T")])
    fixture = _fixture([{"request": {"endpoint": "search", "query": "ransomware"}, "payload": payload}])
    transport = RedditFixtureTransport(fixture)
    assert transport.search("ransomware", limit=25, after=None) == payload


def test_fixture_transport_search_falls_back_to_first_generic_response():
    payload = _listing([_child("a1", "T")])
    fixture = _fixture([{"request": {"endpoint": "search"}, "payload": payload}])
    transport = RedditFixtureTransport(fixture)
    assert transport.search("unmatched-query", limit=25, after=None) == payload


def test_fixture_transport_search_empty_listing_when_nothing_matches():
    fixture = _fixture([])
    transport = RedditFixtureTransport(fixture)
    assert transport.search("anything", limit=25, after=None) == {"kind": "Listing", "data": {"children": [], "after": None}}


# ---------------------------------------------------------------------------
# RedditCollector.collect
# ---------------------------------------------------------------------------


def _window():
    return CollectionWindow(start=datetime(2026, 3, 20, 9, tzinfo=UTC), end=datetime(2026, 3, 20, 10, tzinfo=UTC), cycle_index=1, label="cycle_001")


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_collect_routes_subreddit_source_to_subreddit_new():
    window = _window()
    child = _child("a1", "T", created_utc=window.start.timestamp() + 60)
    transport = FakeTransport(subreddit_pages=[_listing([child], after=None)])
    plan = QueryPlan([CollectionQuery(platform="reddit", query="netsec", source="subreddit")])
    collector = RedditCollector(transport, mode="live")

    result = collector.collect(window, plan)

    assert len(transport.subreddit_calls) == 1
    assert transport.subreddit_calls[0]["subreddit"] == "netsec"
    assert transport.search_calls == []
    assert [p.external_id for p in result.posts] == ["a1"]
    assert result.mode == "live"
    assert result.platform == "reddit"


def test_collect_routes_dynamic_source_to_search():
    window = _window()
    child = _child("a1", "T", created_utc=window.start.timestamp() + 60)
    transport = FakeTransport(search_pages=[_listing([child], after=None)])
    plan = QueryPlan([CollectionQuery(platform="reddit", query="ransomware", source="dynamic")])
    collector = RedditCollector(transport)

    result = collector.collect(window, plan)

    assert len(transport.search_calls) == 1
    assert transport.search_calls[0]["query"] == "ransomware"
    assert transport.subreddit_calls == []
    assert [p.external_id for p in result.posts] == ["a1"]


def test_collect_filters_posts_outside_window():
    window = _window()
    inside = _child("in", "In window", created_utc=window.start.timestamp() + 60)
    outside = _child("out", "Outside window", created_utc=window.end.timestamp() + 60)
    transport = FakeTransport(subreddit_pages=[_listing([inside, outside], after=None)])
    plan = QueryPlan([CollectionQuery(platform="reddit", query="netsec", source="subreddit")])
    collector = RedditCollector(transport)

    result = collector.collect(window, plan)

    assert [p.external_id for p in result.posts] == ["in"]


def test_collect_pagination_stops_when_after_is_null():
    window = _window()
    child = _child("a1", "T", created_utc=window.start.timestamp() + 60)
    transport = FakeTransport(subreddit_pages=[_listing([child], after=None)])
    plan = QueryPlan([CollectionQuery(platform="reddit", query="netsec", source="subreddit")])
    collector = RedditCollector(transport)

    collector.collect(window, plan)

    assert len(transport.subreddit_calls) == 1


def test_collect_pagination_continues_while_after_present_up_to_three_pages():
    window = _window()
    recent = window.start.timestamp() + 3000
    page1 = _listing([_child("p1", "T", created_utc=recent)], after="cursor1")
    page2 = _listing([_child("p2", "T", created_utc=recent)], after="cursor2")
    page3 = _listing([_child("p3", "T", created_utc=recent)], after="cursor3")
    transport = FakeTransport(subreddit_pages=[page1, page2, page3])
    plan = QueryPlan([CollectionQuery(platform="reddit", query="netsec", source="subreddit")])
    collector = RedditCollector(transport)

    result = collector.collect(window, plan)

    assert len(transport.subreddit_calls) == 3
    assert transport.subreddit_calls[1]["after"] == "cursor1"
    assert transport.subreddit_calls[2]["after"] == "cursor2"
    assert {p.external_id for p in result.posts} == {"p1", "p2", "p3"}


def test_collect_pagination_stops_early_on_window_cutoff_for_subreddit():
    window = _window()
    in_window = window.start.timestamp() + 60
    before_window = window.start.timestamp() - 3600
    page1 = _listing([_child("p1", "T", created_utc=in_window), _child("p2", "T", created_utc=before_window)], after="cursor1")
    transport = FakeTransport(subreddit_pages=[page1, _listing([_child("p3", "T")], after=None)])
    plan = QueryPlan([CollectionQuery(platform="reddit", query="netsec", source="subreddit")])
    collector = RedditCollector(transport)

    result = collector.collect(window, plan)

    assert len(transport.subreddit_calls) == 1
    assert [p.external_id for p in result.posts] == ["p1"]


def test_collect_stats_keys_exactly():
    window = _window()
    child = _child("a1", "T", created_utc=window.start.timestamp() + 60)
    transport = FakeTransport(subreddit_pages=[_listing([child], after=None)], search_pages=[_listing([], after=None)])
    plan = QueryPlan(
        [
            CollectionQuery(platform="reddit", query="netsec", source="subreddit"),
            CollectionQuery(platform="reddit", query="ransomware", source="dynamic"),
        ]
    )
    collector = RedditCollector(transport)

    result = collector.collect(window, plan)

    assert set(result.stats.keys()) == {"subreddits", "searches", "posts", "pages", "errors"}
    assert result.stats["subreddits"] == 1
    assert result.stats["searches"] == 1
    assert result.stats["posts"] == 1
    assert result.stats["pages"] == 2
    assert result.stats["errors"] == 0


def test_collect_appends_per_query_error_and_continues():
    window = _window()
    good_child = _child("a1", "T", created_utc=window.start.timestamp() + 60)
    good_transport = FakeTransport(search_pages=[_listing([good_child], after=None)])

    class MixedTransport:
        def subreddit_new(self, subreddit, *, limit, after):
            raise CollectorError("all Reddit accounts rate-limited or rejected")

        def search(self, query, *, limit, after):
            return good_transport.search(query, limit=limit, after=after)

    plan = QueryPlan(
        [
            CollectionQuery(platform="reddit", query="netsec", source="subreddit"),
            CollectionQuery(platform="reddit", query="ransomware", source="dynamic"),
        ]
    )
    collector = RedditCollector(MixedTransport())

    result = collector.collect(window, plan)

    assert result.errors == ["netsec: all Reddit accounts rate-limited or rejected"]
    assert result.stats["errors"] == 1
    assert [p.external_id for p in result.posts] == ["a1"]


def test_collect_skips_missing_created_utc_without_truncating_pagination():
    window = _window()
    in_window_1 = window.start.timestamp() + 60
    in_window_2 = window.start.timestamp() + 120
    page1 = _listing(
        [_child("good1", "Good1", created_utc=in_window_1), _child("bad1", "Bad1", created_utc=None)],
        after="cursor1",
    )
    page2 = _listing([_child("good2", "Good2", created_utc=in_window_2)], after=None)
    transport = FakeTransport(subreddit_pages=[page1, page2])
    plan = QueryPlan([CollectionQuery(platform="reddit", query="netsec", source="subreddit")])
    collector = RedditCollector(transport)

    result = collector.collect(window, plan)

    assert len(transport.subreddit_calls) == 2
    assert {p.external_id for p in result.posts} == {"good1", "good2"}
    assert any("without created_utc skipped" in error for error in result.errors)
    assert result.stats["errors"] == len(result.errors)


def test_collect_ignores_non_reddit_platform_queries():
    window = _window()
    transport = FakeTransport()
    plan = QueryPlan([CollectionQuery(platform="x", query="netsec", source="static")])
    collector = RedditCollector(transport)

    result = collector.collect(window, plan)

    assert result.posts == []
    assert transport.subreddit_calls == [] and transport.search_calls == []


# ---------------------------------------------------------------------------
# RedditLiveTransport
# ---------------------------------------------------------------------------


def test_live_transport_obtains_token_via_basic_auth_and_caches():
    client = FakeHttpClient(
        token_responses=[{"access_token": "tok-1", "expires_in": 3600}],
        get_responses=[_listing([]), _listing([])],
    )
    transport = RedditLiveTransport([("id1", "secret1")], client, user_agent="tiblender/1.0")

    transport.subreddit_new("netsec", limit=25, after=None)
    transport.subreddit_new("netsec", limit=25, after=None)

    assert len(client.post_calls) == 1
    assert client.post_calls[0]["url"] == "https://www.reddit.com/api/v1/access_token"
    assert client.post_calls[0]["data"] == {"grant_type": "client_credentials"}
    assert client.post_calls[0]["basic_auth"] == ("id1", "secret1")
    assert client.post_calls[0]["headers"]["User-Agent"] == "tiblender/1.0"

    for call in client.get_calls:
        assert call["headers"]["Authorization"] == "bearer tok-1"
        assert call["headers"]["User-Agent"] == "tiblender/1.0"
    assert client.get_calls[0]["url"] == "https://oauth.reddit.com/r/netsec/new"
    assert client.get_calls[0]["params"]["limit"] == 25
    assert client.get_calls[0]["params"]["raw_json"] == 1


def test_live_transport_search_endpoint_and_params():
    client = FakeHttpClient(
        token_responses=[{"access_token": "tok-1", "expires_in": 3600}],
        get_responses=[_listing([])],
    )
    transport = RedditLiveTransport([("id1", "secret1")], client, user_agent="ua")

    transport.search("ransomware", limit=10, after="c1")

    call = client.get_calls[0]
    assert call["url"] == "https://oauth.reddit.com/search"
    assert call["params"] == {"q": "ransomware", "sort": "new", "t": "day", "limit": 10, "after": "c1", "raw_json": 1}


def test_live_transport_refetches_token_after_expiry():
    now = {"t": 1000.0}
    client = FakeHttpClient(
        token_responses=[
            {"access_token": "tok-1", "expires_in": 100},
            {"access_token": "tok-2", "expires_in": 100},
        ],
        get_responses=[_listing([]), _listing([])],
    )
    transport = RedditLiveTransport([("id1", "secret1")], client, user_agent="ua", now=lambda: now["t"])

    transport.subreddit_new("netsec", limit=25, after=None)
    now["t"] += 200  # past expires_in - 60
    transport.subreddit_new("netsec", limit=25, after=None)

    assert len(client.post_calls) == 2
    assert client.get_calls[1]["headers"]["Authorization"] == "bearer tok-2"


def test_live_transport_rotates_account_on_rate_limited_and_remembers_last_used():
    client = FakeHttpClient(
        token_responses=[
            {"access_token": "tok-a", "expires_in": 3600},
            {"access_token": "tok-b", "expires_in": 3600},
        ],
        get_responses=[
            RateLimited(429, "slow down", retry_after=1.0),
            _listing([_child("a1", "T")]),
            _listing([_child("a2", "T")]),
        ],
    )
    transport = RedditLiveTransport([("id-a", "secret-a"), ("id-b", "secret-b")], client, user_agent="ua")

    first = transport.subreddit_new("netsec", limit=25, after=None)
    second = transport.subreddit_new("netsec", limit=25, after=None)

    assert first["data"]["children"][0]["data"]["id"] == "a1"
    assert second["data"]["children"][0]["data"]["id"] == "a2"
    # second call reuses account B directly (no rotation attempt on account A, no new token fetch)
    assert len(client.post_calls) == 2
    assert len(client.get_calls) == 3
    assert client.get_calls[1]["headers"]["Authorization"] == "bearer tok-b"
    assert client.get_calls[2]["headers"]["Authorization"] == "bearer tok-b"


def test_live_transport_raises_collector_error_when_all_accounts_exhausted():
    client = FakeHttpClient(
        token_responses=[
            {"access_token": "tok-a", "expires_in": 3600},
            {"access_token": "tok-b", "expires_in": 3600},
        ],
        get_responses=[
            RateLimited(429, "slow", retry_after=1.0),
            HttpError(403, "forbidden"),
        ],
    )
    transport = RedditLiveTransport([("id-a", "secret-a"), ("id-b", "secret-b")], client, user_agent="ua")

    with pytest.raises(CollectorError, match="all Reddit accounts rate-limited or rejected"):
        transport.subreddit_new("netsec", limit=25, after=None)


def test_live_transport_rotates_on_malformed_token_response_missing_access_token():
    client = FakeHttpClient(
        token_responses=[
            {"expires_in": 3600},  # missing access_token entirely
            {"access_token": "tok-b", "expires_in": 3600},
        ],
        get_responses=[_listing([_child("a1", "T")])],
    )
    transport = RedditLiveTransport([("id-a", "secret-a"), ("id-b", "secret-b")], client, user_agent="ua")

    result = transport.subreddit_new("netsec", limit=25, after=None)

    assert result["data"]["children"][0]["data"]["id"] == "a1"
    assert len(client.post_calls) == 2  # account A's malformed token, then account B's
    assert len(client.get_calls) == 1  # only account B's listing call was ever made
    assert client.get_calls[0]["headers"]["Authorization"] == "bearer tok-b"


def test_live_transport_rotates_on_malformed_token_response_empty_access_token():
    client = FakeHttpClient(
        token_responses=[
            {"access_token": ""},
            {"access_token": "tok-b", "expires_in": 3600},
        ],
        get_responses=[_listing([_child("a1", "T")])],
    )
    transport = RedditLiveTransport([("id-a", "secret-a"), ("id-b", "secret-b")], client, user_agent="ua")

    result = transport.subreddit_new("netsec", limit=25, after=None)

    assert result["data"]["children"][0]["data"]["id"] == "a1"


def test_live_transport_rotates_on_non_numeric_expires_in():
    client = FakeHttpClient(
        token_responses=[
            {"access_token": "tok-a", "expires_in": "soon"},
            {"access_token": "tok-b", "expires_in": 3600},
        ],
        get_responses=[_listing([_child("a1", "T")])],
    )
    transport = RedditLiveTransport([("id-a", "secret-a"), ("id-b", "secret-b")], client, user_agent="ua")

    result = transport.subreddit_new("netsec", limit=25, after=None)

    assert result["data"]["children"][0]["data"]["id"] == "a1"


def test_live_transport_raises_collector_error_when_all_token_responses_malformed():
    client = FakeHttpClient(
        token_responses=[{"expires_in": 3600}, {"access_token": ""}],
        get_responses=[],
    )
    transport = RedditLiveTransport([("id-a", "secret-a"), ("id-b", "secret-b")], client, user_agent="ua")

    with pytest.raises(CollectorError, match="all Reddit accounts rate-limited or rejected"):
        transport.subreddit_new("netsec", limit=25, after=None)

    assert len(client.post_calls) == 2
    assert client.get_calls == []


def test_live_transport_propagates_other_http_errors_as_collector_error_without_rotating():
    client = FakeHttpClient(
        token_responses=[{"access_token": "tok-a", "expires_in": 3600}],
        get_responses=[HttpError(500, "server exploded")],
    )
    transport = RedditLiveTransport([("id-a", "secret-a"), ("id-b", "secret-b")], client, user_agent="ua")

    with pytest.raises(CollectorError):
        transport.subreddit_new("netsec", limit=25, after=None)

    # only the first account was tried; no rotation on a plain 500
    assert len(client.post_calls) == 1
    assert len(client.get_calls) == 1
