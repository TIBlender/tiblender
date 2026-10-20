"""Stdlib HTTP client with retry/backoff and a rotating token pool for platform APIs."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

_RETRY_IMMEDIATE_STATUSES = (401, 403, 404)


class HttpError(RuntimeError):
    """Raised for any non-retryable (or retry-exhausted) HTTP failure."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body


class RateLimited(HttpError):
    """Raised immediately on HTTP 429, carrying the server-advised retry delay."""

    def __init__(self, status: int, body: str, *, retry_after: float):
        super().__init__(status, body)
        self.retry_after = retry_after


def _header_value(headers: object, name: str) -> str | None:
    # Real HTTPError.headers (email.message.Message) is case-insensitive, but
    # tests hand in a plain dict, so fall back to a manual scan.
    if not headers:
        return None
    getter = getattr(headers, "get", None)
    if getter is not None:
        value = getter(name)
        if value is not None:
            return value
    items = getattr(headers, "items", None)
    if items is not None:
        for key, value in items():
            if key.lower() == name.lower():
                return value
    return None


# Module-level so tests can `monkeypatch.setattr(http, "_time", ...)` to pin
# "now" for the epoch-based x-ratelimit-reset fallback without real sleeps.
_time: Callable[[], float] = time.time


def _first_header_seconds(headers: object, *names: str) -> float | None:
    for name in names:
        value = _header_value(headers, name)
        if value is None:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def _retry_after_seconds(error: urllib.error.HTTPError, text_body: str) -> float:
    # Order: Retry-After (seconds) > x-ratelimit-reset-after variants
    # (seconds) > x-ratelimit-reset variants (epoch seconds) > JSON body
    # retry_after > a conservative 1.0 s default.
    seconds = _first_header_seconds(error.headers, "Retry-After")
    if seconds is not None:
        return seconds
    seconds = _first_header_seconds(error.headers, "x-ratelimit-reset-after", "x-rate-limit-reset-after")
    if seconds is not None:
        return seconds
    epoch_seconds = _first_header_seconds(error.headers, "x-ratelimit-reset", "x-rate-limit-reset")
    if epoch_seconds is not None:
        return max(1.0, epoch_seconds - _time())
    if text_body:
        try:
            payload = json.loads(text_body)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "retry_after" in payload:
            try:
                return float(payload["retry_after"])
            except (TypeError, ValueError):
                pass
    return 1.0


class HttpClient:
    """Thin urllib wrapper: JSON GET/POST with 429/5xx handling and injectable I/O."""

    def __init__(
        self,
        *,
        user_agent: str = "tiblender/0.1",
        timeout: int = 30,
        max_retries: int = 3,
        sleeper: Callable[[float], None] = time.sleep,
        opener: Callable = urllib.request.urlopen,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleeper = sleeper
        self.opener = opener

    def _build_headers(self, extra: dict[str, str] | None) -> dict[str, str]:
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        if extra:
            headers.update(extra)
        return headers

    def get_json(self, url: str, *, params: dict | None = None, headers: dict | None = None) -> dict | list:
        full_url = url if not params else f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(full_url, headers=self._build_headers(headers), method="GET")
        return self._send(request)

    def post_form(
        self,
        url: str,
        data: dict,
        *,
        headers: dict | None = None,
        basic_auth: tuple[str, str] | None = None,
    ) -> dict:
        body = urllib.parse.urlencode(data).encode("ascii")
        request_headers = self._build_headers(headers)
        request_headers["Content-Type"] = "application/x-www-form-urlencoded"
        if basic_auth is not None:
            username, secret = basic_auth
            token = base64.b64encode(f"{username}:{secret}".encode("utf-8")).decode("ascii")
            request_headers["Authorization"] = f"Basic {token}"
        request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
        return self._send(request)

    def _send(self, request: urllib.request.Request) -> dict | list:
        attempt = 0
        while True:
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    raw = response.read()
                return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as error:
                raw_body = error.read() if hasattr(error, "read") else b""
                text_body = raw_body.decode("utf-8", errors="replace") if raw_body else ""
                if error.code == 429:
                    raise RateLimited(error.code, text_body, retry_after=_retry_after_seconds(error, text_body)) from error
                if error.code in _RETRY_IMMEDIATE_STATUSES:
                    raise HttpError(error.code, text_body) from error
                if 500 <= error.code < 600 and attempt < self.max_retries:
                    self.sleeper(2**attempt)
                    attempt += 1
                    continue
                raise HttpError(error.code, text_body) from error
            except urllib.error.URLError as error:
                raise HttpError(0, str(error.reason)) from error


class TokenPool:
    """Rotates API tokens, skipping any still cooling down after a rate limit."""

    def __init__(
        self,
        tokens: list[str],
        *,
        cooldown_seconds: float = 900.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tokens = list(tokens)
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._cooldown_until: dict[str, float] = {}

    def available(self) -> list[str]:
        now = self._clock()
        return [token for token in self._tokens if self._cooldown_until.get(token, 0.0) <= now]

    def mark_rate_limited(self, token: str) -> None:
        self._cooldown_until[token] = self._clock() + self._cooldown_seconds

    def __len__(self) -> int:
        return len(self._tokens)
