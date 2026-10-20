"""Typed data structures shared across the compact TIBlender runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ThreatCategory(StrEnum):
    PHISHING = "phishing"
    MALWARE = "malware"
    RANSOMWARE = "ransomware"
    VULNERABILITY = "vulnerability"
    APT = "apt"
    FRAUD = "fraud"
    SUPPLY_CHAIN = "supply_chain"
    ABUSE_INFRASTRUCTURE = "abuse_infrastructure"
    OTHER = "other"


class EvaluationVerdict(StrEnum):
    PROCEED = "PROCEED"
    NEED_MORE_INFO = "NEED_MORE_INFO"
    SKIP = "SKIP"


class IocConfidence(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(slots=True)
class Post:
    # URL and author can be read from private JSONL inputs, but report builders
    # omit them from public outputs.
    id: str
    platform: str
    content: str
    timestamp: str = ""
    author: str = ""
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IocSet:
    urls: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    ips: list[str] = field(default_factory=list)
    hashes: list[str] = field(default_factory=list)
    cves: list[str] = field(default_factory=list)

    def all_values(self) -> list[str]:
        return sorted(set(self.urls + self.domains + self.ips + self.hashes + self.cves))

    def merge(self, other: "IocSet") -> "IocSet":
        return IocSet(
            urls=sorted(set(self.urls + other.urls)),
            domains=sorted(set(self.domains + other.domains)),
            ips=sorted(set(self.ips + other.ips)),
            hashes=sorted(set(self.hashes + other.hashes)),
            cves=sorted(set(self.cves + other.cves)),
        )

    @classmethod
    def from_many(cls, items: list["IocSet"]) -> "IocSet":
        merged = cls()
        for item in items:
            merged = merged.merge(item)
        return merged


@dataclass(slots=True)
class NormalizedPost:
    post: Post
    normalized_content: str
    iocs: IocSet = field(default_factory=IocSet)
    threat_category: ThreatCategory = ThreatCategory.OTHER
    language: str = "unknown"
    is_threat_related: bool = True

    @property
    def id(self) -> str:
        return self.post.id

    @property
    def platform(self) -> str:
        return self.post.platform


@dataclass(slots=True)
class Cluster:
    id: str
    topic_name: str
    description: str
    posts: list[NormalizedPost]
    iocs: IocSet
    severity: Severity
    threat_category: ThreatCategory
    filter_reason: str = ""
    clustering_signals: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolQuery:
    # focus_type records which perspective focus generated the query so the
    # context aggregator and the RL trajectory can attribute each tool call.
    perspective: str
    tool_type: str
    parameters: dict[str, Any]
    priority: int = 5
    timeout_seconds: int = 30
    focus_type: str = ""


@dataclass(slots=True)
class ToolResult:
    # raw_json supports intermediate diagnostics but is stripped from public
    # report fields before export.
    tool_type: str
    query_params: dict[str, Any]
    summary: str
    result_url: str = ""
    timestamp: str = ""
    discovered_iocs: IocSet = field(default_factory=IocSet)
    confidence: IocConfidence = IocConfidence.MEDIUM
    status: str = "success"
    error: str = ""
    raw_json: str = ""


@dataclass(slots=True)
class PerspectiveFinding:
    perspective: str
    focus_types: list[str]
    queries: list[ToolQuery]
    tool_results: list[ToolResult]
    insights: list[str]
    confidence: float
    mode: str = "multi_perspective"


@dataclass(slots=True)
class EvaluationRound:
    round_index: int
    judge: str
    verdict: EvaluationVerdict
    quality_score: float
    reason: str
    requested_followups: list[str] = field(default_factory=list)
    sufficiency: dict[str, Any] = field(default_factory=dict)
    consistency: dict[str, Any] = field(default_factory=dict)
    evidence_quality: dict[str, Any] = field(default_factory=dict)
    follow_up_actions: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationDecision:
    verdict: EvaluationVerdict
    rounds: list[EvaluationRound]
    final_quality_score: float
    reason: str


@dataclass(slots=True)
class CycleResult:
    posts_loaded: int
    normalized_posts: int
    clusters_generated: int
    clusters_investigated: int
    reports_generated: int
    output_dir: Path


def to_jsonable(value: Any) -> Any:
    # Centralized serialization keeps dataclass, enum, path, and set handling
    # consistent across reports, intermediate files, and tests.
    if isinstance(value, Post):
        payload = asdict(value)
        # Released runtime traces keep source records deidentified; private
        # source handles and URLs are not written even if present in inputs.
        payload.pop("author", None)
        payload.pop("url", None)
        return {key: to_jsonable(item) for key, item in payload.items()}
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value
