import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tiblender_runtime.collectors.keywords import (
    DYNAMIC_QUERY_PLATFORMS,
    KeywordCatalog,
    build_query_plan,
    load_dynamic_queries,
    load_lines,
)

CONFIG = Path("config")


def test_catalog_loads_bundle_with_nine_languages_and_four_categories():
    catalog = KeywordCatalog.load(CONFIG / "keywords.json")
    assert set(catalog.languages) == {"en", "ja", "ko", "ru", "zh", "es", "pt", "ar", "fr"}
    assert catalog.categories == ["phishing_fraud", "malware_c2", "vulnerability", "apt_ransomware"]
    assert len(catalog.terms) >= 100
    assert "フィッシング" in catalog.terms_for("phishing_fraud", "ja")
    # Only the query-driven platforms carry static queries: Telegram and
    # Discord are collected exhaustively from joined communities.
    assert sorted(catalog.platform_queries) == ["reddit", "x"]


def test_load_lines_skips_comments():
    subs = load_lines(CONFIG / "subreddits.txt")
    assert "netsec" in subs and all(not s.startswith("#") for s in subs) and len(subs) == 42


def test_dynamic_queries_drop_expired_and_cap(tmp_path):
    now = datetime(2026, 3, 20, tzinfo=UTC)
    payload = [
        {"query": "old", "platforms": ["x"], "trend_types": ["VOLUME"], "priority": 9, "language": "en", "expires_at": "2026-03-19T00:00:00Z", "source_report_id": "r1"},
        {"query": "new1", "platforms": ["x", "reddit"], "trend_types": ["ESCALATION"], "priority": 8, "language": "en", "expires_at": "2026-03-27T00:00:00Z", "source_report_id": "r2"},
        {"query": "new2", "platforms": ["x"], "trend_types": ["VOLUME"], "priority": 7, "language": "ja", "expires_at": "2026-03-27T00:00:00Z", "source_report_id": "r3"},
    ]
    path = tmp_path / "dynamic_queries.json"; path.write_text(json.dumps(payload), encoding="utf-8")
    queries = load_dynamic_queries(path, now=now, max_per_platform=1)
    assert [(q.platform, q.query) for q in queries] == [("reddit", "new1"), ("x", "new1")]
    assert queries[0].source == "dynamic" and queries[0].trend_types == ("ESCALATION",)


def test_build_query_plan_merges_static_subreddits_and_dynamic():
    catalog = KeywordCatalog.load(CONFIG / "keywords.json")
    plan = build_query_plan(catalog, subreddits=["netsec"], dynamic=[])
    assert any(q.source == "subreddit" and q.query == "netsec" for q in plan.for_platform("reddit"))
    assert plan.for_platform("x")
    # Telegram and Discord are not query driven and get no plan entries.
    assert plan.for_platform("telegram") == []
    assert plan.for_platform("discord") == []


def test_telegram_channel_list_ships_with_comments_only():
    assert load_lines(CONFIG / "telegram_channels.txt") == []
    assert not (CONFIG / "telegram_seed_queries.txt").exists()


def test_dynamic_queries_target_x_and_reddit_only():
    assert DYNAMIC_QUERY_PLATFORMS == ("x", "reddit")


def test_dynamic_queries_ignore_non_search_platforms(tmp_path, caplog):
    now = datetime(2026, 3, 20, tzinfo=UTC)
    payload = [
        {"query": "CVE-2026-1", "platforms": ["x", "telegram", "discord", "reddit"], "priority": 7},
        {"query": "CVE-2026-2", "platforms": ["telegram"], "priority": 7},
    ]
    path = tmp_path / "dynamic_queries.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with caplog.at_level(logging.DEBUG, logger="tiblender_runtime.collectors.keywords"):
        queries = load_dynamic_queries(path, now=now, max_per_platform=10)

    assert sorted((q.platform, q.query) for q in queries) == [
        ("reddit", "CVE-2026-1"),
        ("x", "CVE-2026-1"),
    ]
    messages = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]
    assert any("telegram" in message and "ignored" in message for message in messages)
    assert any("discord" in message and "ignored" in message for message in messages)


def test_dynamic_queries_skip_non_numeric_priority_with_warning(tmp_path, caplog):
    now = datetime(2026, 3, 20, tzinfo=UTC)
    payload = [
        {"query": "bad", "platforms": ["x"], "priority": "not-a-number", "expires_at": "2026-03-27T00:00:00Z"},
        {"query": "good", "platforms": ["x"], "priority": 6, "expires_at": "2026-03-27T00:00:00Z"},
    ]
    path = tmp_path / "dynamic_queries.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        queries = load_dynamic_queries(path, now=now, max_per_platform=10)
    assert [q.query for q in queries] == ["good"]
    assert any("priority" in record.getMessage() and "record 0" in record.getMessage() for record in caplog.records)


def test_dynamic_queries_clamps_priority_to_1_10(tmp_path):
    now = datetime(2026, 3, 20, tzinfo=UTC)
    payload = [
        {"query": "too-high", "platforms": ["x"], "priority": 99},
        {"query": "too-low", "platforms": ["x"], "priority": -5},
    ]
    path = tmp_path / "dynamic_queries.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    queries = {q.query: q.priority for q in load_dynamic_queries(path, now=now, max_per_platform=10)}
    assert queries["too-high"] == 10
    assert queries["too-low"] == 1


def test_dynamic_queries_skip_invalid_query_and_platforms_with_warning(tmp_path, caplog):
    now = datetime(2026, 3, 20, tzinfo=UTC)
    payload = [
        {"query": "", "platforms": ["x"]},
        {"query": "no-platforms", "platforms": []},
        {"platforms": ["x"]},
        {"query": "kept", "platforms": ["x"]},
    ]
    path = tmp_path / "dynamic_queries.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        queries = load_dynamic_queries(path, now=now, max_per_platform=10)
    assert [q.query for q in queries] == ["kept"]
    warnings = [record.getMessage() for record in caplog.records]
    assert sum("query" in message and "platforms" in message for message in warnings) == 3


def test_dynamic_queries_skip_unparsable_expires_at_with_warning(tmp_path, caplog):
    now = datetime(2026, 3, 20, tzinfo=UTC)
    payload = [
        {"query": "bad-expiry", "platforms": ["x"], "expires_at": "not-a-date"},
        {"query": "kept", "platforms": ["x"]},
    ]
    path = tmp_path / "dynamic_queries.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        queries = load_dynamic_queries(path, now=now, max_per_platform=10)
    assert [q.query for q in queries] == ["kept"]
    assert any("expires_at" in record.getMessage() and "record 0" in record.getMessage() for record in caplog.records)


def test_dynamic_queries_raises_value_error_on_non_list_payload(tmp_path):
    path = tmp_path / "dynamic_queries.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_dynamic_queries(path, now=datetime(2026, 3, 20, tzinfo=UTC), max_per_platform=10)
