import io, json, urllib.error
import pytest
import tiblender_runtime.collectors.http as http_module
from tiblender_runtime.collectors.http import HttpClient, HttpError, RateLimited, TokenPool


class _Resp:
    def __init__(self, body: bytes, headers=None): self._b = body; self.headers = headers or {}
    def read(self, n=-1): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def geturl(self): return "http://x"


def test_get_json_builds_query_string_and_parses():
    seen = {}
    def opener(request, timeout=0):
        seen["url"] = request.full_url; seen["ua"] = request.get_header("User-agent")
        return _Resp(b'{"ok": true}')
    client = HttpClient(opener=opener, sleeper=lambda s: None)
    assert client.get_json("http://api/x", params={"q": "a b", "n": 2}) == {"ok": True}
    assert seen["url"] == "http://api/x?q=a+b&n=2" and seen["ua"].startswith("tiblender/")


def test_429_raises_rate_limited_with_retry_after():
    def opener(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 429, "slow", {"Retry-After": "7"}, io.BytesIO(b"{}"))
    with pytest.raises(RateLimited) as exc:
        HttpClient(opener=opener, sleeper=lambda s: None).get_json("http://api/x")
    assert exc.value.retry_after == 7.0


def test_429_falls_back_to_epoch_ratelimit_reset_header(monkeypatch):
    monkeypatch.setattr(http_module, "_time", lambda: 1000.0)
    def opener(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 429, "slow", {"x-ratelimit-reset": "1010"}, io.BytesIO(b"{}"))
    with pytest.raises(RateLimited) as exc:
        HttpClient(opener=opener, sleeper=lambda s: None).get_json("http://api/x")
    assert exc.value.retry_after == 10.0


def test_429_epoch_ratelimit_reset_floors_at_one_second(monkeypatch):
    monkeypatch.setattr(http_module, "_time", lambda: 1000.0)
    def opener(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 429, "slow", {"x-ratelimit-reset": "1000.2"}, io.BytesIO(b"{}"))
    with pytest.raises(RateLimited) as exc:
        HttpClient(opener=opener, sleeper=lambda s: None).get_json("http://api/x")
    assert exc.value.retry_after == 1.0


def test_429_retry_after_header_takes_priority_over_epoch_reset(monkeypatch):
    monkeypatch.setattr(http_module, "_time", lambda: 1000.0)
    def opener(request, timeout=0):
        raise urllib.error.HTTPError(
            request.full_url, 429, "slow", {"Retry-After": "3", "x-ratelimit-reset": "1999"}, io.BytesIO(b"{}")
        )
    with pytest.raises(RateLimited) as exc:
        HttpClient(opener=opener, sleeper=lambda s: None).get_json("http://api/x")
    assert exc.value.retry_after == 3.0


def test_5xx_retries_then_raises():
    calls = []
    def opener(request, timeout=0):
        calls.append(1); raise urllib.error.HTTPError(request.full_url, 503, "down", {}, io.BytesIO(b""))
    with pytest.raises(HttpError):
        HttpClient(opener=opener, sleeper=lambda s: None, max_retries=2).get_json("http://api/x")
    assert len(calls) == 3


def test_post_form_sends_basic_auth():
    seen = {}
    def opener(request, timeout=0):
        seen["auth"] = request.get_header("Authorization"); seen["data"] = request.data
        return _Resp(b'{"access_token": "t"}')
    client = HttpClient(opener=opener, sleeper=lambda s: None)
    assert client.post_form("http://api/token", {"grant_type": "client_credentials"}, basic_auth=("id", "secret"))["access_token"] == "t"
    assert seen["auth"].startswith("Basic ") and seen["data"] == b"grant_type=client_credentials"


def test_token_pool_cooldown():
    t = {"now": 0.0}
    pool = TokenPool(["a", "b"], cooldown_seconds=10, clock=lambda: t["now"])
    pool.mark_rate_limited("a")
    assert pool.available() == ["b"]
    t["now"] = 11.0
    assert pool.available() == ["a", "b"]
