"""Shared collector types: collection windows, raw posts, query plans, results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CollectionWindow:
    start: datetime
    end: datetime
    cycle_index: int
    label: str

    @classmethod
    def ending_at(cls, end: datetime, hours: float, cycle_index: int) -> "CollectionWindow":
        start = end - timedelta(hours=hours)
        return cls(start=start, end=end, cycle_index=cycle_index, label=f"cycle_{cycle_index:03d}")

    def contains(self, iso_timestamp: str) -> bool:
        parsed = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=self.start.tzinfo)
        return self.start <= parsed < self.end


@dataclass(slots=True)
class RawPost:
    platform: str
    external_id: str
    content: str
    posted_at: str
    language: str = ""
    author_ref: str = ""
    url: str = ""
    attachments: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def post_id(self) -> str:
        return f"{self.platform}-{self.external_id}"


@dataclass(frozen=True, slots=True)
class CollectionQuery:
    platform: str
    query: str
    source: str
    category: str = ""
    language: str = "en"
    priority: int = 5
    trend_types: tuple[str, ...] = ()
    source_report_id: str = ""


@dataclass(slots=True)
class QueryPlan:
    queries: list[CollectionQuery]

    def for_platform(self, platform: str) -> list[CollectionQuery]:
        matching = [q for q in self.queries if q.platform == platform]
        return sorted(matching, key=lambda q: (-q.priority, q.query))

    def dynamic(self) -> list[CollectionQuery]:
        return [q for q in self.queries if q.source == "dynamic"]


@dataclass(slots=True)
class CollectionResult:
    platform: str
    mode: str
    posts: list[RawPost]
    errors: list[str]
    stats: dict[str, int]


class PlatformCollector(Protocol):
    platform: str

    def collect(self, window: CollectionWindow, plan: QueryPlan) -> CollectionResult: ...


class CollectorError(RuntimeError):
    """Raised when a platform collector cannot complete its collection cycle."""
