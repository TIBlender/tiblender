import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tiblender_runtime.collectors.keywords import KeywordCatalog, load_dynamic_queries
from tiblender_runtime.dynamic_queries import (
    DynamicQuery,
    generate_dynamic_queries,
    heuristic_dynamic_queries,
    write_dynamic_queries,
)
from tiblender_runtime.llm import LLMGenerationError
from tiblender_runtime.models import Cluster, IocSet, NormalizedPost, Post, Severity, ThreatCategory
from tiblender_runtime.settings import Settings
from tiblender_runtime.trends import ClusterSnapshot, TrendSignal

CATALOG = KeywordCatalog.load(Path("config") / "keywords.json")
NOW = datetime(2026, 3, 20, tzinfo=UTC)


def _cluster(
    *,
    cluster_id: str = "cluster-1",
    cves: list[str] | None = None,
    domains: list[str] | None = None,
    topic_name: str = "Axios Stealer campaign",
    category: ThreatCategory = ThreatCategory.MALWARE,
) -> Cluster:
    iocs = IocSet(cves=cves or [], domains=domains or [])
    post = NormalizedPost(
        post=Post(id="p1", platform="x", content="Payload dropper active."),
        normalized_content="Payload dropper active.",
        iocs=iocs,
    )
    return Cluster(
        id=cluster_id,
        topic_name=topic_name,
        description="A malware campaign.",
        posts=[post],
        iocs=iocs,
        severity=Severity.HIGH,
        threat_category=category,
    )


def _snapshot(cluster: Cluster, *, languages: list[str] | None = None) -> ClusterSnapshot:
    return ClusterSnapshot(
        cluster_id=cluster.id,
        cycle_index=1,
        observed_at="2026-03-20T00:00:00Z",
        embedding=[1.0, 0.0],
        iocs=cluster.iocs.all_values(),
        cves=list(cluster.iocs.cves),
        severity=cluster.severity.value,
        platforms=["x"],
        post_count=len(cluster.posts),
        category=cluster.threat_category.value,
        topic_name=cluster.topic_name,
        strategic_value=0.5,
        languages=languages if languages is not None else ["en"],
    )


def _signal(cluster_id: str, trend_types: list[str]) -> TrendSignal:
    return TrendSignal(cluster_id=cluster_id, trend_types=trend_types, adjacent_history=1, evidence={})


# --- heuristic_dynamic_queries --------------------------------------------------


def test_heuristic_seeds_the_cve_across_the_search_platforms_in_english():
    cluster = _cluster(cves=["CVE-2026-3055"])
    snapshot = _snapshot(cluster)
    signal = _signal(cluster.id, ["VOLUME"])

    queries = heuristic_dynamic_queries(
        signal, snapshot, cluster, catalog=CATALOG, now=NOW, ttl_hours=168, report_id="TIBLENDER-1"
    )

    cve_queries = [q for q in queries if q.query == "CVE-2026-3055"]
    assert len(cve_queries) == 1
    # Only X and Reddit are collected by keyword search.
    assert cve_queries[0].platforms == ["x", "reddit"]
    assert cve_queries[0].language == "en"
    assert cve_queries[0].source_report_id == "TIBLENDER-1"
    assert cve_queries[0].source_cluster_id == cluster.id


def test_heuristic_adds_a_japanese_variant_for_x_only():
    cluster = _cluster(cves=["CVE-2026-3055"], category=ThreatCategory.VULNERABILITY)
    snapshot = _snapshot(cluster, languages=["en", "ja"])
    signal = _signal(cluster.id, ["EARLY_WARNING"])

    queries = heuristic_dynamic_queries(
        signal, snapshot, cluster, catalog=CATALOG, now=NOW, ttl_hours=168, report_id="TIBLENDER-2"
    )

    ja_queries = [q for q in queries if q.language == "ja" and q.query.startswith("CVE-2026-3055")]
    assert len(ja_queries) == 1
    assert ja_queries[0].platforms == ["x"]
    assert ja_queries[0].query != "CVE-2026-3055"  # localized term appended


def test_heuristic_expiry_equals_now_plus_ttl_hours():
    cluster = _cluster(cves=["CVE-2026-3055"])
    snapshot = _snapshot(cluster)
    signal = _signal(cluster.id, ["VOLUME"])

    queries = heuristic_dynamic_queries(
        signal, snapshot, cluster, catalog=CATALOG, now=NOW, ttl_hours=168, report_id="r1"
    )

    assert all(q.expires_at == "2026-03-27T00:00:00Z" for q in queries)
    assert all(q.generated_at == "2026-03-20T00:00:00Z" for q in queries)


def test_heuristic_falls_back_to_domain_and_brand_seeds_without_a_cve():
    cluster = _cluster(cves=[], domains=["bad-example.com"], topic_name="AxiosStealer campaign")
    snapshot = _snapshot(cluster)
    signal = _signal(cluster.id, ["PROPAGATION"])

    queries = heuristic_dynamic_queries(
        signal, snapshot, cluster, catalog=CATALOG, now=NOW, ttl_hours=168, report_id="r1"
    )

    seeds = {q.query for q in queries}
    assert "bad-example.com" in seeds
    assert any(seed.startswith("Axios") or seed.startswith("Stealer") for seed in seeds if seed != "bad-example.com")


def test_heuristic_returns_no_queries_without_any_grounded_seed():
    cluster = _cluster(cves=[], domains=[], topic_name="lowercase only")
    snapshot = _snapshot(cluster)
    signal = _signal(cluster.id, ["VOLUME"])

    queries = heuristic_dynamic_queries(
        signal, snapshot, cluster, catalog=CATALOG, now=NOW, ttl_hours=168, report_id="r1"
    )

    assert queries == []


def test_heuristic_priority_formula_caps_at_ten():
    cluster = _cluster(cves=["CVE-2026-3055"])
    snapshot = _snapshot(cluster)
    signal = _signal(cluster.id, ["VOLUME", "PROPAGATION", "ESCALATION", "EARLY_WARNING"])

    queries = heuristic_dynamic_queries(
        signal, snapshot, cluster, catalog=CATALOG, now=NOW, ttl_hours=168, report_id="r1"
    )

    # 6 + 4 trend types + 1 EARLY_WARNING bonus = 11, capped at 10.
    assert all(q.priority == 10 for q in queries)


def test_heuristic_priority_formula_without_early_warning_bonus():
    cluster = _cluster(cves=["CVE-2026-3055"])
    snapshot = _snapshot(cluster)
    signal = _signal(cluster.id, ["ESCALATION"])

    queries = heuristic_dynamic_queries(
        signal, snapshot, cluster, catalog=CATALOG, now=NOW, ttl_hours=168, report_id="r1"
    )

    assert all(q.priority == 7 for q in queries)  # 6 + 1 trend type, no bonus


# --- generate_dynamic_queries: dedupe / sort / cap ------------------------------


def _heuristic_settings() -> Settings:
    return Settings(llm_provider="heuristic", allow_heuristic_reports=True)


def test_generate_dynamic_queries_dedupes_keeping_the_higher_priority():
    cluster_a = _cluster(cluster_id="cluster-a", cves=["CVE-2026-3055"])
    cluster_b = _cluster(cluster_id="cluster-b", cves=["CVE-2026-3055"])
    snapshot_a = _snapshot(cluster_a)
    snapshot_b = _snapshot(cluster_b)
    signal_a = _signal("cluster-a", ["VOLUME"])  # priority 7
    signal_b = _signal("cluster-b", ["VOLUME", "PROPAGATION", "ESCALATION"])  # priority 9

    queries, _signals = generate_dynamic_queries(
        [signal_a, signal_b],
        {"cluster-a": snapshot_a, "cluster-b": snapshot_b},
        {"cluster-a": cluster_a, "cluster-b": cluster_b},
        {},
        settings=_heuristic_settings(),
        catalog=CATALOG,
        now=NOW,
    )

    matches = [q for q in queries if q.query == "CVE-2026-3055" and q.platforms == ["x", "reddit"]]
    assert len(matches) == 1
    assert matches[0].priority == 9


def test_generate_dynamic_queries_sorts_by_priority_desc_then_query():
    cluster_a = _cluster(cluster_id="cluster-a", cves=["CVE-2026-0001"])
    cluster_b = _cluster(cluster_id="cluster-b", cves=["CVE-2026-9999"])
    snapshots = {"cluster-a": _snapshot(cluster_a), "cluster-b": _snapshot(cluster_b)}
    clusters = {"cluster-a": cluster_a, "cluster-b": cluster_b}
    signals = [_signal("cluster-a", ["VOLUME"]), _signal("cluster-b", ["VOLUME"])]

    queries, _signals = generate_dynamic_queries(
        signals, snapshots, clusters, {}, settings=_heuristic_settings(), catalog=CATALOG, now=NOW
    )

    same_priority = [q for q in queries if q.query in {"CVE-2026-0001", "CVE-2026-9999"}]
    assert [q.query for q in same_priority] == ["CVE-2026-0001", "CVE-2026-9999"]


def test_generate_dynamic_queries_caps_the_total_at_two_hundred():
    clusters = {}
    snapshots = {}
    signals = []
    for index in range(205):
        cluster_id = f"cluster-{index}"
        cluster = _cluster(cluster_id=cluster_id, cves=[f"CVE-2026-{index:04d}"])
        clusters[cluster_id] = cluster
        snapshots[cluster_id] = _snapshot(cluster)
        signals.append(_signal(cluster_id, ["VOLUME"]))

    queries, _signals = generate_dynamic_queries(
        signals, snapshots, clusters, {}, settings=_heuristic_settings(), catalog=CATALOG, now=NOW
    )

    assert len(queries) == 200


# --- generate_dynamic_queries: LLM path -----------------------------------------


def _llm_settings() -> Settings:
    return Settings(llm_provider="openai", llm_model="model", openai_api_key="key")


def test_generate_dynamic_queries_uses_the_llm_when_configured(monkeypatch):
    cluster = _cluster(cves=["CVE-2026-3055"])
    snapshot = _snapshot(cluster)
    signal = _signal(cluster.id, ["VOLUME"])
    captured = {}

    def fake_llm(settings, trend_payload):
        captured["payload"] = trend_payload
        return [
            {
                "query": "CVE-2026-3055 exploit chatter",
                "platforms": ["x", "reddit"],
                "language": "en",
                "rationale": "LLM rationale",
            }
        ]

    monkeypatch.setattr("tiblender_runtime.llm_feedback.generate_dynamic_queries_with_llm", fake_llm)

    queries, signals = generate_dynamic_queries(
        [signal],
        {cluster.id: snapshot},
        {cluster.id: cluster},
        {cluster.id: "TIBLENDER-9"},
        settings=_llm_settings(),
        catalog=CATALOG,
        now=NOW,
    )

    assert [q.query for q in queries] == ["CVE-2026-3055 exploit chatter"]
    assert queries[0].rationale == "LLM rationale"
    assert queries[0].source_report_id == "TIBLENDER-9"
    assert captured["payload"]["trend_types"] == ["VOLUME"]
    assert "llm_fallback" not in signal.evidence
    assert "llm_fallback" not in signals[0].evidence


def test_generate_dynamic_queries_falls_back_to_heuristic_on_llm_error_and_records_evidence(monkeypatch):
    cluster = _cluster(cves=["CVE-2026-3055"])
    snapshot = _snapshot(cluster)
    signal = _signal(cluster.id, ["VOLUME"])

    def fake_llm(settings, trend_payload):
        raise LLMGenerationError("boom: victim ACME Corp paid", category="json_parse")

    monkeypatch.setattr("tiblender_runtime.llm_feedback.generate_dynamic_queries_with_llm", fake_llm)

    queries, signals = generate_dynamic_queries(
        [signal],
        {cluster.id: snapshot},
        {cluster.id: cluster},
        {},
        settings=_llm_settings(),
        catalog=CATALOG,
        now=NOW,
    )

    assert any(q.query == "CVE-2026-3055" for q in queries)
    # The caller's own signal instance is never mutated (TrendSignal is
    # frozen); the fallback marker only appears on the *returned* signal,
    # which is a new object built via dataclasses.replace.
    assert "llm_fallback" not in signal.evidence
    assert signals[0] is not signal
    # The marker is code-authored: the raise-site category, never the error
    # message, which may embed the model's draft of scraped post text.
    assert signals[0].evidence["llm_fallback"] == "llm_error:json_parse"
    assert "ACME" not in signals[0].evidence["llm_fallback"]
    assert signals[0].cluster_id == signal.cluster_id
    assert signals[0].trend_types == signal.trend_types


def test_generate_dynamic_queries_marks_no_valid_queries_when_llm_yields_nothing_usable(monkeypatch):
    cluster = _cluster(cves=["CVE-2026-3055"])
    snapshot = _snapshot(cluster)
    signal = _signal(cluster.id, ["VOLUME"])

    def fake_llm(settings, trend_payload):
        # Not grounded in the cluster's evidence, and an unsupported platform.
        return [{"query": "generic unrelated text", "platforms": ["facebook"], "language": "en"}]

    monkeypatch.setattr("tiblender_runtime.llm_feedback.generate_dynamic_queries_with_llm", fake_llm)

    queries, signals = generate_dynamic_queries(
        [signal],
        {cluster.id: snapshot},
        {cluster.id: cluster},
        {},
        settings=_llm_settings(),
        catalog=CATALOG,
        now=NOW,
    )

    assert any(q.query == "CVE-2026-3055" for q in queries)
    # Distinguishable from an outright LLM failure: the LLM responded, it
    # just proposed nothing valid.
    assert "llm_fallback" not in signal.evidence
    assert signals[0].evidence["llm_fallback"] == "no_valid_queries"


def test_generate_dynamic_queries_skips_the_llm_when_heuristics_are_forced(monkeypatch):
    cluster = _cluster(cves=["CVE-2026-3055"])
    snapshot = _snapshot(cluster)
    signal = _signal(cluster.id, ["VOLUME"])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called when allow_heuristic_reports is true")

    monkeypatch.setattr("tiblender_runtime.llm_feedback.generate_dynamic_queries_with_llm", fail_if_called)

    settings = Settings(llm_provider="openai", llm_model="model", openai_api_key="key", allow_heuristic_reports=True)
    queries, _signals = generate_dynamic_queries(
        [signal], {cluster.id: snapshot}, {cluster.id: cluster}, {}, settings=settings, catalog=CATALOG, now=NOW
    )

    assert any(q.query == "CVE-2026-3055" for q in queries)


# --- write_dynamic_queries / load_dynamic_queries round trip --------------------


def test_write_dynamic_queries_round_trips_through_load_dynamic_queries(tmp_path):
    queries = [
        DynamicQuery(
            query="CVE-2026-3055",
            platforms=["x", "reddit"],
            trend_types=["VOLUME"],
            priority=8,
            language="en",
            source_report_id="TIBLENDER-1",
            source_cluster_id="cluster-1",
            generated_at="2026-03-20T00:00:00Z",
            expires_at="2026-03-27T00:00:00Z",
            rationale="VOLUME trend detected.",
        )
    ]
    path = tmp_path / "dynamic_queries.json"

    write_dynamic_queries(queries, path)
    loaded = load_dynamic_queries(path, now=NOW, max_per_platform=30)

    assert json.loads(path.read_text(encoding="utf-8"))[0]["query"] == "CVE-2026-3055"
    assert {q.platform for q in loaded} == {"x", "reddit"}
    assert all(q.source == "dynamic" and q.trend_types == ("VOLUME",) for q in loaded)


def _generate_with_llm_draft(monkeypatch, draft):
    cluster = _cluster(cves=["CVE-2026-3055"])
    snapshot = _snapshot(cluster)
    signal = _signal(cluster.id, ["VOLUME"])
    monkeypatch.setattr(
        "tiblender_runtime.llm_feedback.generate_dynamic_queries_with_llm",
        lambda settings, trend_payload: draft,
    )
    return generate_dynamic_queries(
        [signal],
        {cluster.id: snapshot},
        {cluster.id: cluster},
        {cluster.id: "TIBLENDER-9"},
        settings=_llm_settings(),
        catalog=CATALOG,
        now=NOW,
    )


def test_llm_drafted_platform_list_is_narrowed_to_the_search_platforms(monkeypatch):
    """A draft naming a non-search platform keeps its usable half rather than
    losing the whole query to the heuristic fallback."""
    queries, signals = _generate_with_llm_draft(
        monkeypatch,
        [{"query": "CVE-2026-3055 guild chatter", "platforms": ["x", "discord"], "language": "en"}],
    )

    assert [q.query for q in queries] == ["CVE-2026-3055 guild chatter"]
    assert queries[0].platforms == ["x"]
    assert "llm_fallback" not in signals[0].evidence


def test_llm_draft_with_no_search_platform_left_falls_back_to_the_heuristic(monkeypatch):
    """Narrowing is not a licence to emit a query with no platform: when
    nothing survives, the query is dropped and the heuristic path runs."""
    queries, signals = _generate_with_llm_draft(
        monkeypatch,
        [
            {"query": "CVE-2026-3055 channel", "platforms": ["telegram"], "language": "en"},
            {"query": "CVE-2026-3055 server", "platforms": ["discord"], "language": "en"},
        ],
    )

    assert queries
    assert all(set(query.platforms) <= {"x", "reddit"} for query in queries)
    assert all(query.platforms for query in queries)
    assert "CVE-2026-3055 channel" not in {query.query for query in queries}
    assert signals[0].evidence["llm_fallback"] == "no_valid_queries"


def test_dynamic_query_prompt_names_only_the_search_platforms():
    from tiblender_runtime.llm_feedback import _dynamic_query_messages

    system = _dynamic_query_messages({"trend_types": []})[0]["content"]
    assert "a subset of x, reddit)" in system
    assert "never propose them as query platforms" in system
