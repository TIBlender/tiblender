from datetime import UTC, datetime

from tiblender_runtime.collectors.base import CollectionQuery, CollectionWindow, CollectorError, QueryPlan
from tiblender_runtime.collectors.fixtures import FixtureFile
from tiblender_runtime.collectors.http import RateLimited
from tiblender_runtime.collectors.x import XCollector, XFixtureTransport, XLiveTransport, parse_search_payload

PAYLOAD = {
    "data": [
        {"id": "1", "author_id": "u1", "created_at": "2026-03-20T09:10:00.000Z", "lang": "ja", "text": "偽サイト https://t.co/abc #phishing",
         "entities": {"urls": [{"url": "https://t.co/abc", "expanded_url": "https://login-example.net/verify"}], "hashtags": [{"tag": "phishing"}]},
         "attachments": {"media_keys": ["m1"]}, "public_metrics": {"retweet_count": 2, "like_count": 5}},
        {"id": "2", "author_id": "u1", "created_at": "2026-03-20T12:00:00.000Z", "lang": "en", "text": "outside window"},
    ],
    "includes": {"users": [{"id": "u1", "username": "alice"}], "media": [{"media_key": "m1", "type": "photo", "url": "https://pbs.example/m1.jpg"}]},
    "meta": {"result_count": 2},
}
WINDOW = CollectionWindow(datetime(2026, 3, 20, 9, tzinfo=UTC), datetime(2026, 3, 20, 10, tzinfo=UTC), 1, "cycle_001")


class FakeTransport:
    def __init__(self): self.calls = []
    def recent_search(self, query, *, start_time, end_time, max_results, next_token):
        self.calls.append(("search", query, next_token)); return PAYLOAD
    def users_by_usernames(self, usernames): self.calls.append(("users", tuple(usernames))); return {"data": [{"id": "u9", "username": usernames[0]}]}
    def user_tweets(self, user_id, *, start_time, end_time, pagination_token):
        self.calls.append(("timeline", user_id)); return {"data": [{"id": "9", "author_id": "u9", "created_at": "2026-03-20T09:30:00.000Z", "lang": "en", "text": "timeline post CVE-2026-0001"}], "includes": {"users": [{"id": "u9", "username": "bob"}]}}


def test_parse_expands_tco_and_attachments():
    posts = parse_search_payload(PAYLOAD, source="static", query="q")
    assert posts[0].content == "偽サイト https://login-example.net/verify #phishing"
    assert posts[0].attachments == [{"type": "photo", "url": "https://pbs.example/m1.jpg", "alt_text": ""}]
    assert posts[0].author_ref == "alice" and posts[0].url == "https://x.com/alice/status/1" and posts[0].language == "ja"
    assert posts[0].metadata["hashtags"] == ["phishing"] and posts[0].metadata["source"] == "static"


def test_parse_marks_referenced_tweet_types():
    payload = {"data": [{"id": "3", "author_id": "u1", "created_at": "2026-03-20T09:15:00Z", "text": "rt",
                          "referenced_tweets": [{"type": "retweeted", "id": "1"}]}]}
    posts = parse_search_payload(payload, source="static", query="q")
    assert posts[0].metadata["referenced_tweets"] == ["retweeted"]


def test_parse_missing_created_at_defaults_to_empty_string():
    payload = {"data": [{"id": "5", "author_id": "u1", "text": "no timestamp"}]}
    posts = parse_search_payload(payload, source="static", query="q")
    assert posts[0].posted_at == "" and posts[0].external_id == "5"


def test_collect_filters_window_and_runs_timelines():
    transport = FakeTransport()
    collector = XCollector(transport, users=["bob"], mode="fixture")
    result = collector.collect(WINDOW, QueryPlan([CollectionQuery("x", "q", "static")]))
    ids = sorted(p.external_id for p in result.posts)
    assert ids == ["1", "9"] and result.stats["queries"] == 1 and result.stats["timeline_users"] == 1
    assert next(p for p in result.posts if p.external_id == "9").metadata["source"] == "timeline"
    assert result.mode == "fixture" and result.platform == "x"
    assert result.stats == {"queries": 1, "timeline_users": 1, "posts": 2, "pages": 2, "errors": 0}


class _NoTimestampTransport:
    def recent_search(self, query, *, start_time, end_time, max_results, next_token):
        return {"data": [{"id": "5", "author_id": "u1", "text": "no ts"}]}
    def users_by_usernames(self, usernames):
        return {"data": []}
    def user_tweets(self, user_id, *, start_time, end_time, pagination_token):
        return {"data": []}


def test_collect_drops_posts_without_created_at():
    collector = XCollector(_NoTimestampTransport(), users=[], mode="fixture")
    result = collector.collect(WINDOW, QueryPlan([CollectionQuery("x", "q", "static")]))
    assert result.posts == [] and result.stats["posts"] == 0


class _PaginatingTransport:
    def __init__(self): self.calls = 0
    def recent_search(self, query, *, start_time, end_time, max_results, next_token):
        self.calls += 1
        if next_token is None:
            return {"data": [{"id": "1", "author_id": "u1", "created_at": "2026-03-20T09:10:00Z", "text": "a"}], "meta": {"next_token": "tok"}}
        return {"data": [{"id": "2", "author_id": "u1", "created_at": "2026-03-20T09:20:00Z", "text": "b"}]}
    def users_by_usernames(self, usernames):
        return {"data": []}
    def user_tweets(self, user_id, *, start_time, end_time, pagination_token):
        return {"data": []}


def test_collect_paginates_until_next_token_missing():
    transport = _PaginatingTransport()
    collector = XCollector(transport, users=[], mode="fixture")
    result = collector.collect(WINDOW, QueryPlan([CollectionQuery("x", "q", "static")]))
    ids = sorted(p.external_id for p in result.posts)
    assert ids == ["1", "2"] and transport.calls == 2 and result.stats["pages"] == 2


class _FailingTransport:
    def recent_search(self, query, *, start_time, end_time, max_results, next_token):
        raise CollectorError("boom")
    def users_by_usernames(self, usernames):
        return {"data": []}
    def user_tweets(self, user_id, *, start_time, end_time, pagination_token):
        return {"data": []}


def test_collect_captures_per_query_errors_and_continues():
    collector = XCollector(_FailingTransport(), users=[], mode="fixture")
    plan = QueryPlan([CollectionQuery("x", "q1", "static"), CollectionQuery("x", "q2", "static")])
    result = collector.collect(WINDOW, plan)
    assert result.posts == []
    assert result.errors == ["q1: boom", "q2: boom"]
    assert result.stats["errors"] == 2 and result.stats["queries"] == 2


class _FailingUserLookupTransport:
    def recent_search(self, query, *, start_time, end_time, max_results, next_token):
        return {"data": []}
    def users_by_usernames(self, usernames):
        raise CollectorError("lookup failed")
    def user_tweets(self, user_id, *, start_time, end_time, pagination_token):
        return {"data": []}


def test_collect_counts_configured_users_even_when_lookup_fails():
    collector = XCollector(_FailingUserLookupTransport(), users=["bob", "carol"], mode="fixture")
    result = collector.collect(WINDOW, QueryPlan([]))
    assert result.stats["timeline_users"] == 2
    assert result.errors == ["bob,carol: lookup failed"]
    assert result.stats["errors"] == 1
    assert result.posts == []


class _FailingTimelineTransport:
    def recent_search(self, query, *, start_time, end_time, max_results, next_token):
        return {"data": []}
    def users_by_usernames(self, usernames):
        return {"data": [{"id": "u1", "username": "bob"}]}
    def user_tweets(self, user_id, *, start_time, end_time, pagination_token):
        raise CollectorError("timeline failed")


def test_collect_records_timeline_fetch_errors_and_continues():
    collector = XCollector(_FailingTimelineTransport(), users=["bob"], mode="fixture")
    result = collector.collect(WINDOW, QueryPlan([]))
    assert result.stats["timeline_users"] == 1
    assert result.errors == ["bob: timeline failed"]
    assert result.posts == []


def test_live_transport_rotates_tokens_on_rate_limit():
    class Client:
        def __init__(self): self.headers = []
        def get_json(self, url, *, params=None, headers=None):
            self.headers.append(headers["Authorization"])
            if headers["Authorization"] == "Bearer t1": raise RateLimited(429, "", retry_after=1)
            return {"data": []}
    client = Client()
    transport = XLiveTransport(["t1", "t2"], client)
    assert transport.recent_search("q", start_time="s", end_time="e", max_results=10, next_token=None) == {"data": []}
    assert client.headers == ["Bearer t1", "Bearer t2"]


def test_live_transport_raises_collector_error_when_all_tokens_fail():
    class Client:
        def get_json(self, url, *, params=None, headers=None):
            raise RateLimited(429, "", retry_after=1)
    transport = XLiveTransport(["t1", "t2"], Client())
    try:
        transport.recent_search("q", start_time="s", end_time="e", max_results=10, next_token=None)
        assert False, "expected CollectorError"
    except CollectorError as error:
        assert "all X bearer tokens" in str(error)


def test_live_transport_skips_token_on_401_then_propagates_other_http_errors():
    from tiblender_runtime.collectors.http import HttpError

    class Client:
        def __init__(self): self.headers = []
        def get_json(self, url, *, params=None, headers=None):
            self.headers.append(headers["Authorization"])
            if headers["Authorization"] == "Bearer t1": raise HttpError(401, "unauthorized")
            raise HttpError(500, "boom")
    client = Client()
    transport = XLiveTransport(["t1", "t2"], client)
    try:
        transport.recent_search("q", start_time="s", end_time="e", max_results=10, next_token=None)
        assert False, "expected CollectorError"
    except CollectorError:
        pass
    assert client.headers == ["Bearer t1", "Bearer t2"]


def _fixture(responses):
    return FixtureFile(platform="x", window=WINDOW, provenance="test", responses=responses)


def test_fixture_transport_matches_query_then_falls_back_to_generic():
    fixture = _fixture([
        {"request": {"endpoint": "tweets/search/recent", "query": "artifact"}, "payload": {"data": [{"id": "g1"}]}},
        {"request": {"endpoint": "tweets/search/recent", "query": "phishing"}, "payload": {"data": [{"id": "p1"}]}},
    ])
    transport = XFixtureTransport(fixture)
    assert transport.recent_search("phishing", start_time="s", end_time="e", max_results=10, next_token=None) == {"data": [{"id": "p1"}]}
    assert transport.recent_search("no-match", start_time="s", end_time="e", max_results=10, next_token=None) == {"data": [{"id": "g1"}]}


def test_fixture_transport_returns_empty_when_no_match():
    fixture = _fixture([])
    transport = XFixtureTransport(fixture)
    assert transport.recent_search("q", start_time="s", end_time="e", max_results=10, next_token=None) == {"data": []}
    assert transport.users_by_usernames(["alice"]) == {"data": []}
    assert transport.user_tweets("u1", start_time="s", end_time="e", pagination_token=None) == {"data": []}


def test_fixture_transport_paginates_by_next_token():
    fixture = _fixture([
        {"request": {"endpoint": "tweets/search/recent", "query": "q"}, "payload": {"data": [{"id": "1"}], "meta": {"next_token": "tok"}}},
        {"request": {"endpoint": "tweets/search/recent", "query": "q", "next_token": "tok"}, "payload": {"data": [{"id": "2"}]}},
    ])
    transport = XFixtureTransport(fixture)
    first = transport.recent_search("q", start_time="s", end_time="e", max_results=10, next_token=None)
    assert first["data"][0]["id"] == "1"
    second = transport.recent_search("q", start_time="s", end_time="e", max_results=10, next_token="tok")
    assert second["data"][0]["id"] == "2"


def test_fixture_transport_users_and_timeline_endpoints():
    fixture = _fixture([
        {"request": {"endpoint": "users/by"}, "payload": {"data": [{"id": "u1", "username": "alice"}]}},
        {"request": {"endpoint": "users/{id}/tweets", "user_id": "u1"}, "payload": {"data": [{"id": "9"}]}},
    ])
    transport = XFixtureTransport(fixture)
    assert transport.users_by_usernames(["alice"]) == {"data": [{"id": "u1", "username": "alice"}]}
    assert transport.user_tweets("u1", start_time="s", end_time="e", pagination_token=None) == {"data": [{"id": "9"}]}
