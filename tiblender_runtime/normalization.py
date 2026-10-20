"""Deterministic post loading and normalization before clustering."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

from .iocs import extract_iocs
from .models import NormalizedPost, Post, ThreatCategory, to_jsonable

_CATEGORY_KEYWORDS: list[tuple[ThreatCategory, tuple[str, ...]]] = [
    # The compact runtime uses keyword categories here; the full system can
    # replace this stage with richer multilingual normalization.
    (ThreatCategory.VULNERABILITY, ("cve-", "zero-day", "zero day", "rce", "exploit", "vulnerability")),
    (ThreatCategory.MALWARE, ("malware", "c2", "payload", "dropper", "backdoor", "botnet", "trojan")),
    (ThreatCategory.RANSOMWARE, ("ransomware", "extortion", "double extortion")),
    (ThreatCategory.APT, ("apt", "lazarus", "kimsuky", "apt28", "apt29", "nation-state", "state-sponsored")),
    (ThreatCategory.PHISHING, ("phishing", "credential", "fake login", "smishing", "spoof", "drainer")),
    (ThreatCategory.FRAUD, ("fraud", "scam", "wallet", "seed phrase", "impersonation")),
    (ThreatCategory.SUPPLY_CHAIN, ("supply chain", "package", "dependency", "npm", "pypi", "github action")),
]


def load_posts(path: Path) -> list[Post]:
    # Input may contain platform URLs for private runs, but downstream reports
    # deliberately avoid publishing those URL fields.
    posts: list[Post] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            content = str(payload.get("content", "")).strip()
            if not content:
                continue
            posts.append(
                Post(
                    id=str(payload.get("id") or f"post-{line_number}"),
                    platform=str(payload.get("platform") or "unknown").lower(),
                    content=content,
                    timestamp=str(payload.get("timestamp") or ""),
                    author=str(payload.get("author") or ""),
                    url=str(payload.get("url") or payload.get("post_platform_url") or ""),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
    return posts


def detect_category(text: str) -> ThreatCategory:
    lowered = text.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return category
    return ThreatCategory.OTHER


def _brief_summary(text: str, ioc_values: Iterable[str], category: ThreatCategory) -> str:
    # Keep normalized content short and stable while preserving every extracted
    # IoC verbatim for clustering and later report construction.
    compact = re.sub(r"\s+", " ", text).strip()
    compact = compact.replace("[\u6dfb\u4ed8\u30e1\u30c7\u30a3\u30a2]", "[attached media]")
    first_sentence = compact[:280]
    iocs = " ".join(sorted(ioc_values)[:8])
    category_label = category.value.replace("_", " ")
    first_sentence = first_sentence.rstrip(" .")
    if iocs:
        return f"{category_label.title()} signal: {first_sentence}. IoCs: {iocs}"
    return f"{category_label.title()} context: {first_sentence}"


def normalize_posts(
    posts: list[Post],
    *,
    min_text_chars: int = 24,
    normalized_content_by_id: Mapping[str, str] | None = None,
) -> list[NormalizedPost]:
    # The compact runtime uses deterministic normalization so tests and sample
    # runs do not depend on an LLM before clustering begins. Full runs can pass
    # LLM-generated English summaries through normalized_content_by_id.
    normalized: list[NormalizedPost] = []
    llm_summaries = normalized_content_by_id or {}
    for post in posts:
        llm_summary = str(llm_summaries.get(post.id, "")).strip()
        evidence_text = f"{post.content}\n{llm_summary}".strip()
        iocs = extract_iocs(evidence_text)
        category = detect_category(evidence_text)
        enough_context = len((llm_summary or post.content).strip()) >= min_text_chars
        is_threat_related = (category is not ThreatCategory.OTHER or bool(iocs.all_values())) and enough_context
        normalized.append(
            NormalizedPost(
                post=post,
                normalized_content=llm_summary or _brief_summary(post.content, iocs.all_values(), category),
                iocs=iocs,
                threat_category=category,
                language=str(post.metadata.get("language", "unknown")),
                is_threat_related=is_threat_related,
            )
        )
    return normalized


def write_normalized_jsonl(posts: list[NormalizedPost], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for post in posts:
            handle.write(json.dumps(to_jsonable(post), ensure_ascii=False) + "\n")
