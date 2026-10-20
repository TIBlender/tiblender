"""Corpus writer: RawPost -> posts.jsonl record shape, dedupe, and collection_report.json."""

from __future__ import annotations

import json
from pathlib import Path

from .. import clock
from .base import CollectionResult, RawPost


def to_post_record(post: RawPost) -> dict:
    return {
        "id": post.post_id,
        "platform": post.platform,
        "content": post.content,
        "timestamp": post.posted_at,
        "author": post.author_ref,
        "url": post.url,
        "metadata": {**post.metadata, "language": post.language, "attachments": post.attachments},
    }


def dedupe_posts(posts: list[RawPost]) -> list[RawPost]:
    seen: set[str] = set()
    deduped: list[RawPost] = []
    for post in posts:
        if post.post_id in seen:
            continue
        seen.add(post.post_id)
        deduped.append(post)
    deduped.sort(key=lambda post: (post.posted_at, post.post_id))
    return deduped


def write_posts_jsonl(posts: list[RawPost], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for post in posts:
            handle.write(json.dumps(to_post_record(post), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return len(posts)


def write_collection_report(
    results: list[CollectionResult],
    stats: dict,
    path: Path,
    *,
    total_posts: int | None = None,
    warnings: list[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    per_platform = {
        result.platform: {
            "mode": result.mode,
            "posts": len(result.posts),
            "errors": list(result.errors),
            "stats": dict(result.stats),
        }
        for result in results
    }
    kept = stats.get("kept", 0)
    # total_posts is the number of posts actually written to posts.jsonl
    # (i.e. after the caller's cross-platform dedupe). When the caller
    # doesn't pass it (e.g. a direct/legacy call), fall back to the
    # pre-dedupe "kept" count so "deduplicated" degrades to zero rather than
    # a misleading negative/undefined value.
    resolved_total_posts = kept if total_posts is None else total_posts
    preprocess_stats = dict(stats)
    preprocess_stats["deduplicated"] = kept - resolved_total_posts
    payload = {
        "per_platform": per_platform,
        "preprocess": preprocess_stats,
        "total_posts": resolved_total_posts,
        "warnings": list(warnings) if warnings else [],
        "generated_at": clock.now_iso(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
