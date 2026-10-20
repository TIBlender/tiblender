"""VirusTotal HTTP plumbing for the PASSIVE_DNS_LOOKUP tool.

Split out of :mod:`tiblender_runtime.tools` (kept under its 800-line budget).
Holds only the transport concerns -- status-to-reason mapping and the single
Retry-After retry the free tier's 4-requests-per-minute limit makes necessary.
The response parsing and the ToolResult live with the other tool adapters.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .tools_social import ToolUnavailableError


class _VirusTotalHttpError(Exception):
    """A VirusTotal HTTP status that maps to a specific unavailable reason."""

    def __init__(self, status: int, detail: str = "") -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.detail = detail


# VirusTotal free tier allows 4 requests/minute; a 429 is a pacing signal, not
# a dead provider, so one retry after Retry-After is worth the wait.
_VT_MAX_RETRY_AFTER_SECONDS = 60.0
_VT_DEFAULT_RETRY_AFTER_SECONDS = 16.0


def _vt_retry_after_seconds(raw: str | None) -> float:
    try:
        value = float((raw or "").strip())
    except ValueError:
        return _VT_DEFAULT_RETRY_AFTER_SECONDS
    if value <= 0:
        return _VT_DEFAULT_RETRY_AFTER_SECONDS
    return min(value, _VT_MAX_RETRY_AFTER_SECONDS)


def _vt_unavailable_reason(error: _VirusTotalHttpError) -> str:
    if error.status in {401, 403}:
        return "VirusTotal rejected the API key (HTTP %d): check VIRUSTOTAL_API_KEY." % error.status
    if error.status == 404:
        return "VirusTotal has no record for this indicator (HTTP 404: not found)."
    if error.status == 429:
        return "VirusTotal rate limit exceeded (HTTP 429) after one retry."
    return f"VirusTotal request failed: {error.detail[:400] or error}"


def _vt_get_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout: int,
    max_bytes: int = 400_000,
    sleeper=None,
) -> tuple[Any, str]:
    """GET a VirusTotal endpoint, retrying a 429 once after ``Retry-After``."""
    sleeper = sleeper if sleeper is not None else time.sleep
    attempts = 0
    while True:
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read(max_bytes).decode(
                    response.headers.get_content_charset() or "utf-8", errors="replace"
                )
                final_url = response.geturl()
            break
        except urllib.error.HTTPError as error:
            detail = error.read(800).decode("utf-8", errors="replace")
            if error.code == 429 and attempts == 0:
                attempts += 1
                sleeper(_vt_retry_after_seconds(error.headers.get("Retry-After")))
                continue
            raise _VirusTotalHttpError(error.code, detail) from error
        except (urllib.error.URLError, OSError) as error:
            raise ToolUnavailableError(str(getattr(error, "reason", error))) from error
    try:
        return json.loads(text), final_url
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON response from VirusTotal: {error}") from error
