"""Recorded tool responses: deterministic offline substitutes for live tool calls.

Tool recordings are captured once from real provider responses (see
tools/record_tool_responses.py) and replayed during offline/fixture-mode cycles so
outputs stay reproducible without contacting external services.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from . import clock
from .models import ToolResult

# Only these parameters distinguish a query for recording lookup purposes;
# priority/timeout/perspective do not affect the recorded provider response.
_KEY_PARAM_NAMES = ("url", "domain", "ip", "cve", "hash", "query")


class RecordingCorruptedError(RuntimeError):
    """Raised when a recorded tool response file exists but cannot be read or parsed."""


def recording_key(tool_type: str, parameters: dict[str, Any]) -> str:
    normalized = {
        name: str(parameters[name]).strip().lower()
        for name in _KEY_PARAM_NAMES
        if parameters.get(name) not in (None, "")
    }
    payload = json.dumps({"tool_type": tool_type, "parameters": normalized}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def key_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """The subset of ``parameters`` the recording key is computed from.

    Values keep their original form (a URL's case matters when the recorder
    fetches it again); only the *names* are filtered, exactly as
    :func:`recording_key` does.
    """
    return {name: parameters[name] for name in _KEY_PARAM_NAMES if parameters.get(name) not in (None, "")}


class ToolRecordingReplayLog:
    """Per-cycle tally of which recording lookups hit and which missed.

    The offline demo is byte-deterministic, so the set of external-tool
    requests it issues is fixed -- but the fixture builder can only anticipate
    the requests derivable from post text, never the free-text cluster search
    phrases or the domains/IPs pivoted out of recorded evidence. Without this
    log those unanticipated requests were invisible: they degraded silently to
    ``status=unavailable`` and the bundle could ship recordings that nothing
    ever reads while claiming a coverage it did not have.

    The misses this collects are the exact input to
    ``tools/harvest_tool_queries.py``, and the rates land in
    ``cycle_summary.json`` so ``verify-sample`` pins them.

    Thread-safe: cluster workers investigate in parallel, but the payload is
    ordered by key, never by arrival, so it stays byte-identical across runs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # (tool_type, canonical parameters json) -> [hits, misses, parameters]
        self._entries: dict[tuple[str, str], list[Any]] = {}

    def observe(self, tool_type: str, parameters: dict[str, Any], *, hit: bool) -> None:
        relevant = key_parameters(parameters)
        index = (tool_type, json.dumps(relevant, sort_keys=True, ensure_ascii=False))
        with self._lock:
            entry = self._entries.setdefault(index, [0, 0, relevant])
            entry[0 if hit else 1] += 1

    @staticmethod
    def _rate(hits: int, calls: int) -> float:
        # Rounded so the value is byte-stable in committed expected output.
        return round(hits / calls, 4) if calls else 0.0

    def rates(self) -> dict[str, Any]:
        """Per-tool and overall replay rates, ordered by tool name."""
        with self._lock:
            entries = {index: list(entry) for index, entry in self._entries.items()}
        by_tool: dict[str, dict[str, Any]] = {}
        for (tool_type, _), (hits, misses, _) in entries.items():
            bucket = by_tool.setdefault(tool_type, {"calls": 0, "hits": 0, "misses": 0})
            bucket["calls"] += hits + misses
            bucket["hits"] += hits
            bucket["misses"] += misses
        for bucket in by_tool.values():
            bucket["hit_rate"] = self._rate(bucket["hits"], bucket["calls"])
        calls = sum(bucket["calls"] for bucket in by_tool.values())
        hits = sum(bucket["hits"] for bucket in by_tool.values())
        return {
            "by_tool": {tool_type: by_tool[tool_type] for tool_type in sorted(by_tool)},
            "calls": calls,
            "hits": hits,
            "misses": calls - hits,
            "hit_rate": self._rate(hits, calls),
        }

    def misses(self) -> list[dict[str, Any]]:
        """Every request key that had no recording, ordered deterministically."""
        with self._lock:
            entries = {index: list(entry) for index, entry in self._entries.items()}
        return [
            {
                "tool_type": tool_type,
                "parameters": parameters,
                "recording_key": recording_key(tool_type, parameters),
                "calls": misses,
            }
            for (tool_type, _), (_, misses, parameters) in sorted(entries.items())
            if misses
        ]

    def payload(self) -> dict[str, Any]:
        """The full ``intermediate/tool_recording_misses.json`` document."""
        return {
            "note": (
                "External-tool request keys this cycle issued with no recorded response. "
                "Regenerate the recordings with `make harvest-tool-queries`."
            ),
            "replay": self.rates(),
            "misses": self.misses(),
        }


class ToolRecordingStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, tool_type: str, parameters: dict[str, Any]) -> Path:
        key = recording_key(tool_type, parameters)
        return self.root / tool_type / f"{key}.json"

    def lookup(self, tool_type: str, parameters: dict[str, Any]) -> dict[str, Any] | None:
        # A recording can be missing (nothing recorded yet) or, rarely, unreadable
        # (disk corruption, a crash mid-write before atomic record() shipped,
        # a hand-edited fixture). The former is a normal "no data" case; the
        # latter must never propagate as a raw exception out of ToolRunner.
        path = self._path(tool_type, parameters)
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
            return json.loads(text)
        except (OSError, json.JSONDecodeError) as error:
            raise RecordingCorruptedError(f"recording {path} is unreadable or corrupted: {error}") from error

    def record(
        self,
        tool_type: str,
        parameters: dict[str, Any],
        result: ToolResult,
        source_url: str,
    ) -> Path:
        # Sorted keys and sorted IoC lists keep recorded tool responses byte-stable
        # across recording runs so diffs only show real provider changes.
        path = self._path(tool_type, parameters)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tool_type": tool_type,
            "parameters": dict(parameters),
            "recorded_at": clock.now_iso(),
            "source_url": source_url,
            "status": result.status,
            "summary": result.summary,
            "discovered_iocs": {
                "urls": sorted(result.discovered_iocs.urls),
                "domains": sorted(result.discovered_iocs.domains),
                "ips": sorted(result.discovered_iocs.ips),
                "hashes": sorted(result.discovered_iocs.hashes),
                "cves": sorted(result.discovered_iocs.cves),
            },
            "raw": json.loads(result.raw_json) if result.raw_json else {},
            "error": result.error,
        }
        text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
        # Write-then-rename: a crash or concurrent recorder run mid-write can
        # never leave a truncated recording at the real path, since os.replace
        # is atomic on the same filesystem.
        descriptor, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.remove(tmp_name)
            except OSError:
                pass
            raise
        return path

    def count(self) -> int:
        if not self.root.exists():
            return 0
        return sum(1 for _ in self.root.rglob("*.json"))
