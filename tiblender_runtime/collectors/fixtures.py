"""Collector fixture-file loading for offline/replay collection cycles."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .base import CollectionWindow

logger = logging.getLogger(__name__)

_PLATFORMS = ("x", "reddit", "telegram", "discord")
_REQUIRED_TOP_KEYS = ("platform", "window", "provenance", "responses")
_REQUIRED_WINDOW_KEYS = ("start", "end", "cycle_index", "label")


@dataclass(slots=True)
class FixtureFile:
    platform: str
    window: CollectionWindow
    provenance: str
    responses: list[dict]

    def responses_for(self, endpoint: str, **match) -> list[dict]:
        """Recorded responses for ``endpoint`` whose request carries ``match``.

        An empty result has two very different causes, and a caller that cannot
        tell them apart silently reports "no data" for a recording that does
        hold data:

        * the recorder wrote this key down but never with this value -- an
          ordinary miss (a runtime-generated dynamic query has no recording),
          logged at debug level;
        * the recorder never wrote this key down *at all*, so no value of it
          could ever match -- a fixture/collector contract mismatch, which is
          always a bug and is logged as a warning. Matching the Discord
          ``after`` watermark against the recorded request keys was exactly
          this: it silently zeroed every cycle after the first.
        """
        matched: list[dict] = []
        endpoint_requests: list[dict] = []
        for response in self.responses:
            request = response.get("request", {})
            if request.get("endpoint") != endpoint:
                continue
            endpoint_requests.append(request)
            if all(str(request.get(key)) == str(value) for key, value in match.items()):
                matched.append(response)
        if endpoint_requests and match and not matched:
            unknown = sorted(
                key for key in match if not any(key in request for request in endpoint_requests)
            )
            described = ", ".join(f"{key}={value}" for key, value in sorted(match.items()))
            if unknown:
                # Lookup *values* can be a runtime-generated dynamic query,
                # i.e. text the model derived from collected posts, so WARNING
                # names only the keys; the values go to DEBUG.
                logger.warning(
                    "%s fixture records endpoint %r but no recorded request carries the key(s) %s, "
                    "so a lookup on the key(s) %s can never match; replaying as no data",
                    self.platform,
                    endpoint,
                    ", ".join(unknown),
                    ", ".join(sorted(match)),
                )
                logger.debug(
                    "%s fixture lookup on endpoint %r that can never match: %s",
                    self.platform,
                    endpoint,
                    described,
                )
            else:
                logger.debug(
                    "%s fixture records endpoint %r but no recorded request matches %s",
                    self.platform,
                    endpoint,
                    described,
                )
        return matched


def _malformed(path: Path) -> ValueError:
    return ValueError(f"malformed fixture file: {path}")


def _is_valid_response_entry(entry: object) -> bool:
    # Each response must carry a request dict with a string endpoint so
    # responses_for() can match on it without an AttributeError deep inside
    # a collector's replay path.
    if not isinstance(entry, dict):
        return False
    request = entry.get("request")
    if not isinstance(request, dict):
        return False
    return isinstance(request.get("endpoint"), str) and bool(request.get("endpoint"))


def load_fixture(path: Path) -> FixtureFile:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _malformed(path) from error

    if not isinstance(payload, dict) or any(key not in payload for key in _REQUIRED_TOP_KEYS):
        raise _malformed(path)

    window_payload = payload["window"]
    if not isinstance(window_payload, dict) or any(key not in window_payload for key in _REQUIRED_WINDOW_KEYS):
        raise _malformed(path)

    responses = payload["responses"]
    if not isinstance(responses, list) or not all(_is_valid_response_entry(entry) for entry in responses):
        raise _malformed(path)

    try:
        start = datetime.fromisoformat(str(window_payload["start"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(window_payload["end"]).replace("Z", "+00:00"))
        window = CollectionWindow(
            start=start,
            end=end,
            cycle_index=int(window_payload["cycle_index"]),
            label=str(window_payload["label"]),
        )
    except (TypeError, ValueError) as error:
        raise _malformed(path) from error

    return FixtureFile(
        platform=str(payload["platform"]),
        window=window,
        provenance=str(payload["provenance"]),
        responses=responses,
    )


def fixture_path(fixtures_dir: Path, platform: str, cycle_index: int) -> Path:
    return Path(fixtures_dir) / "collectors" / platform / f"cycle_{cycle_index:03d}.json"


def list_fixture_cycles(fixtures_dir: Path) -> list[int]:
    fixtures_dir = Path(fixtures_dir)
    indices_by_platform: list[set[int]] = []
    for platform in _PLATFORMS:
        platform_dir = fixtures_dir / "collectors" / platform
        indices: set[int] = set()
        if platform_dir.is_dir():
            for candidate in platform_dir.glob("cycle_*.json"):
                suffix = candidate.stem.removeprefix("cycle_")
                if suffix.isdigit():
                    indices.add(int(suffix))
        indices_by_platform.append(indices)
    if not indices_by_platform or any(not indices for indices in indices_by_platform):
        return []
    return sorted(set.intersection(*indices_by_platform))
