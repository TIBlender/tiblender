import json
import logging
import sys
import types
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tiblender_runtime.models import (
    Cluster,
    EvaluationDecision,
    EvaluationVerdict,
    IocSet,
    NormalizedPost,
    Post,
    Severity,
    ThreatCategory,
)
from tiblender_runtime.trends import (
    ClusterSnapshot,
    TrendSignal,
    append_history,
    cosine,
    detect_trends,
    load_history,
    snapshot_cluster,
    strategic_value,
    trend_embedding,
)


def _snapshot(
    cluster_id: str,
    *,
    cycle_index: int,
    embedding: list[float],
    severity: str = "MEDIUM",
    platforms: list[str] | None = None,
    strategic_value_: float = 0.0,
    languages: list[str] | None = None,
    observed_at: str = "2026-03-20T00:00:00Z",
    post_count: int = 3,
) -> ClusterSnapshot:
    return ClusterSnapshot(
        cluster_id=cluster_id,
        cycle_index=cycle_index,
        observed_at=observed_at,
        embedding=embedding,
        iocs=[],
        cves=[],
        severity=severity,
        platforms=platforms if platforms is not None else ["x"],
        post_count=post_count,
        category="malware",
        topic_name=f"{cluster_id} topic",
        strategic_value=strategic_value_,
        languages=languages if languages is not None else ["en"],
    )


def _cluster(cluster_id: str = "cluster-1", *, cves: list[str] | None = None) -> Cluster:
    post = NormalizedPost(
        post=Post(id="p1", platform="x", content="Payload dropper active."),
        normalized_content="Payload dropper active.",
        iocs=IocSet(cves=cves or []),
    )
    return Cluster(
        id=cluster_id,
        topic_name="Dropper campaign",
        description="A dropper campaign.",
        posts=[post],
        iocs=post.iocs,
        severity=Severity.MEDIUM,
        threat_category=ThreatCategory.MALWARE,
    )


# --- trend_embedding / cosine -------------------------------------------------


def test_trend_embedding_hashing_is_deterministic_and_matches_feature_count():
    first = trend_embedding("Citrix NetScaler exploitation campaign", backend="hashing", features=64)
    second = trend_embedding("Citrix NetScaler exploitation campaign", backend="hashing", features=64)

    assert first == second
    assert len(first) == 64


def test_trend_embedding_hashing_differs_for_different_text():
    first = trend_embedding("alpha campaign indicators", backend="hashing", features=64)
    second = trend_embedding("totally unrelated other content", backend="hashing", features=64)

    assert first != second


def test_trend_and_clustering_hashing_embeddings_share_one_tokenizer():
    """Both call themselves "the hashing embedding" and both are 384-d, but
    they used different token patterns, so they were not the same space --
    which makes the shared tau_sim=0.80 gate much harder to reason about."""
    import tiblender_runtime.clustering as clustering_module
    import tiblender_runtime.trends as trends_module

    assert trends_module.HASHING_TOKEN_PATTERN is clustering_module.HASHING_TOKEN_PATTERN

    text = "CVE-2026-31337 exploited by cb-terminal.dev via a C2 payload; 3 victims"
    matrix, _backend = clustering_module._hashing_embedding_matrix([text], n_features=64)

    assert trends_module._hashing_embedding(text, features=64) == pytest.approx(list(matrix[0]))


def test_trend_embedding_minilm_falls_back_to_hashing_when_unavailable(monkeypatch, caplog):
    import tiblender_runtime.trends as trends_module

    def fake_hashing(text, *, features):
        return [0.5] * features

    monkeypatch.setattr(trends_module, "_hashing_embedding", fake_hashing)

    class _BoomModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no local model cache")

    # Inject a stand-in module rather than setattr-ing on the real one:
    # monkeypatch.setattr("sentence_transformers.SentenceTransformer", ...)
    # imports sentence_transformers, so the test only ran on machines that
    # happened to have it installed. This exercises the fallback path with
    # and without the real package present.
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=_BoomModel)
    )

    with caplog.at_level(logging.WARNING):
        vector = trend_embedding("some text", backend="minilm", features=16)

    assert vector == [0.5] * 16
    assert any("minilm" in record.getMessage() for record in caplog.records)


def test_trend_embedding_loads_the_minilm_model_once_per_process(monkeypatch):
    """`feedback._embed_for` calls this once per cluster snapshot, so an
    uncached loader rebuilt SentenceTransformer 35 times for a bundled
    three-cycle run (134 for the recurring set) -- most of the MiniLM
    overhead. Clustering already loads its model once per cycle."""
    import tiblender_runtime.trends as trends_module

    loads: list[tuple] = []

    class _CountingModel:
        def __init__(self, *args, **kwargs):
            loads.append((args, tuple(sorted(kwargs.items()))))

        def encode(self, texts, **kwargs):
            import numpy as np

            return np.array([[0.25, 0.75] for _ in texts])

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=_CountingModel)
    )
    trends_module.load_trend_model.cache_clear()
    try:
        vectors = [trend_embedding(f"snapshot {index}", backend="minilm", features=16) for index in range(12)]
    finally:
        trends_module.load_trend_model.cache_clear()

    assert len(loads) == 1, f"model was constructed {len(loads)} times"
    assert vectors == [[0.25, 0.75]] * 12


def test_cosine_of_identical_vectors_is_one():
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_of_zero_vector_is_zero_not_a_crash():
    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0


# --- snapshot_cluster ----------------------------------------------------------


def test_snapshot_cluster_collects_languages_excluding_unknown():
    post_en = NormalizedPost(post=Post(id="p1", platform="x", content="a"), normalized_content="a", language="en")
    post_ja = NormalizedPost(post=Post(id="p2", platform="telegram", content="b"), normalized_content="b", language="ja")
    post_unknown = NormalizedPost(post=Post(id="p3", platform="x", content="c"), normalized_content="c", language="unknown")
    cluster = Cluster(
        id="cluster-lang",
        topic_name="Topic",
        description="Desc",
        posts=[post_en, post_ja, post_unknown],
        iocs=IocSet(),
        severity=Severity.LOW,
        threat_category=ThreatCategory.OTHER,
    )

    snapshot = snapshot_cluster(
        cluster,
        cycle_index=1,
        observed_at="2026-03-20T00:00:00Z",
        strategic_value=0.1,
        embed=lambda text: [1.0, 0.0],
    )

    assert snapshot.languages == ["en", "ja"]
    assert snapshot.platforms == ["telegram", "x"]
    assert snapshot.embedding == [1.0, 0.0]


# --- load_history / append_history ---------------------------------------------


def test_load_history_prunes_snapshots_outside_the_window(tmp_path):
    path = tmp_path / "cluster_snapshots.jsonl"
    now = datetime(2026, 3, 20, tzinfo=UTC)
    recent = _snapshot("c1", cycle_index=1, embedding=[1.0], observed_at=(now - timedelta(hours=10)).isoformat().replace("+00:00", "Z"))
    stale = _snapshot("c2", cycle_index=1, embedding=[1.0], observed_at=(now - timedelta(hours=200)).isoformat().replace("+00:00", "Z"))
    append_history(path, [recent, stale])

    loaded = load_history(path, now=now, window_hours=168)

    assert [snapshot.cluster_id for snapshot in loaded] == ["c1"]


def test_load_history_drops_snapshots_newer_than_now_with_a_warning(tmp_path, caplog):
    # With a pinned clock (every offline cycle pins it to its own window end)
    # a re-run into a populated output root would otherwise let cycle 1 read
    # the previous run's cycle-2/3 snapshots as "history".
    path = tmp_path / "cluster_snapshots.jsonl"
    now = datetime(2026, 3, 20, tzinfo=UTC)
    current = _snapshot("c1", cycle_index=1, embedding=[1.0], observed_at=now.isoformat().replace("+00:00", "Z"))
    future = _snapshot(
        "c2",
        cycle_index=3,
        embedding=[1.0],
        observed_at=(now + timedelta(hours=48)).isoformat().replace("+00:00", "Z"),
    )
    append_history(path, [current, future])

    with caplog.at_level(logging.WARNING):
        loaded = load_history(path, now=now, window_hours=168)

    assert [snapshot.cluster_id for snapshot in loaded] == ["c1"]
    assert any("after the current clock" in record.getMessage() for record in caplog.records)


def test_load_history_returns_empty_list_when_file_is_missing(tmp_path):
    assert load_history(tmp_path / "missing.jsonl", now=datetime(2026, 3, 20, tzinfo=UTC), window_hours=168) == []


def test_load_history_skips_malformed_lines_with_a_warning(tmp_path, caplog):
    path = tmp_path / "cluster_snapshots.jsonl"
    now = datetime(2026, 3, 20, tzinfo=UTC)
    good = _snapshot("c1", cycle_index=1, embedding=[1.0], observed_at=now.isoformat().replace("+00:00", "Z"))
    path.write_text(json.dumps(asdict(good)) + "\n" + "{not json\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        loaded = load_history(path, now=now, window_hours=168)

    assert [snapshot.cluster_id for snapshot in loaded] == ["c1"]
    assert any("malformed" in record.getMessage() for record in caplog.records)


def test_append_history_then_load_history_round_trips(tmp_path):
    path = tmp_path / "cluster_snapshots.jsonl"
    now = datetime(2026, 3, 20, tzinfo=UTC)
    snapshot = _snapshot("c1", cycle_index=1, embedding=[1.0, 0.0], observed_at=now.isoformat().replace("+00:00", "Z"))

    append_history(path, [snapshot])
    append_history(path, [])  # no-op, must not raise or write an empty line

    loaded = load_history(path, now=now, window_hours=168)

    assert len(loaded) == 1
    assert loaded[0].embedding == [1.0, 0.0]
    assert len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]) == 1


# --- detect_trends: VOLUME -------------------------------------------------------


def _adjacent_history(post_counts: list[int], embedding: list[float]) -> list[ClusterSnapshot]:
    return [
        _snapshot(f"h{index}", cycle_index=index + 1, embedding=embedding, post_count=count)
        for index, count in enumerate(post_counts)
    ]


def test_volume_never_fires_without_adjacent_history():
    """A cluster nobody has seen before cannot be a volume anomaly. The old
    cluster-count statistic fired on exactly these -- mu=sigma=0 and a current
    count of 1 -- which made VOLUME the noisiest axis on the bundled data."""
    embedding = [1.0, 0.0]
    current = [_snapshot("c0", cycle_index=1, embedding=embedding, post_count=50)]

    signals = detect_trends(current, [], tau_sim=0.80, k=1.5, theta_ew=0.75)

    assert not any("VOLUME" in signal.trend_types for signal in signals)


def test_volume_does_not_flag_a_current_post_count_in_line_with_similar_history():
    # post counts [3, 3, 3, 3] -> mu=3, sigma=0, threshold=3. The current
    # cluster matches the baseline exactly and is under volume_min_posts.
    embedding = [1.0, 0.0]
    history = _adjacent_history([3, 3, 3, 3], embedding)
    current = [_snapshot("c0", cycle_index=5, embedding=embedding, post_count=3)]

    signals = detect_trends(current, history, tau_sim=0.80, k=1.5, theta_ew=0.75)

    assert not any("VOLUME" in signal.trend_types for signal in signals)


def test_volume_flags_a_post_count_spike_over_similar_history():
    embedding = [1.0, 0.0]
    history = _adjacent_history([3, 3, 3, 3], embedding)
    current = [_snapshot("c0", cycle_index=5, embedding=embedding, post_count=8)]

    signals = detect_trends(current, history, tau_sim=0.80, k=1.5, theta_ew=0.75)

    signal = next(signal for signal in signals if signal.cluster_id == "c0")
    assert "VOLUME" in signal.trend_types
    assert signal.evidence["mu"] == pytest.approx(3.0)
    assert signal.evidence["sigma"] == pytest.approx(0.0)
    assert signal.evidence["threshold"] == pytest.approx(3.0)
    assert signal.evidence["current_count"] == 8
    # The baseline size is `adjacent` (the deployed detector's
    # metadata.matched_snapshots); it is not repeated under a second key.
    assert signal.evidence["adjacent"] == 4
    assert signal.adjacent_history == 4
    assert "matched" not in signal.evidence


def test_volume_fires_exactly_at_the_threshold_on_a_zero_variance_baseline():
    """The comparison is `>=`, matching the deployed detector, so a cluster
    sitting exactly on a flat baseline qualifies. Isolating the boundary keeps
    a `>` regression distinguishable from a `>=` one."""
    embedding = [1.0, 0.0]
    history = _adjacent_history([4, 4, 4, 4], embedding)
    current = [_snapshot("c0", cycle_index=5, embedding=embedding, post_count=4)]

    signals = detect_trends(current, history, tau_sim=0.80, k=1.5, theta_ew=0.75)

    signal = next(signal for signal in signals if signal.cluster_id == "c0")
    assert "VOLUME" in signal.trend_types
    assert signal.evidence["threshold"] == pytest.approx(4.0)
    assert signal.evidence["current_count"] == 4


def test_volume_does_not_fire_just_below_a_zero_variance_baseline():
    embedding = [1.0, 0.0]
    history = _adjacent_history([4, 4, 4, 4], embedding)
    current = [_snapshot("c0", cycle_index=5, embedding=embedding, post_count=3)]

    signals = detect_trends(current, history, tau_sim=0.80, k=1.5, theta_ew=0.75)

    assert not any("VOLUME" in signal.trend_types for signal in signals)


def test_volume_needs_at_least_min_similar_snapshots_of_adjacent_history():
    """Three similar snapshots are too thin a baseline for a mu/sigma test."""
    embedding = [1.0, 0.0]
    history = _adjacent_history([3, 3, 3], embedding)
    current = [_snapshot("c0", cycle_index=5, embedding=embedding, post_count=8)]

    signals = detect_trends(current, history, tau_sim=0.80, k=1.5, theta_ew=0.75)

    assert not any("VOLUME" in signal.trend_types for signal in signals)


def test_volume_honours_the_minimum_current_post_count_floor():
    embedding = [1.0, 0.0]
    history = _adjacent_history([3, 3, 3, 3], embedding)
    current = [_snapshot("c0", cycle_index=5, embedding=embedding, post_count=8)]

    signals = detect_trends(
        current, history, tau_sim=0.80, k=1.5, theta_ew=0.75, volume_min_posts=10
    )

    assert not any("VOLUME" in signal.trend_types for signal in signals)


def test_volume_statistic_is_over_post_counts_not_cluster_counts():
    """A campaign that recurs as one cluster per cycle and grows 3 -> 5 -> 8
    posts must be detectable. Under the cluster-count statistic its current
    count stayed at 1 and it could never exceed mu + k*sigma."""
    embedding = [1.0, 0.0]
    history = _adjacent_history([3, 3, 5, 5], embedding)
    current = [_snapshot("c0", cycle_index=5, embedding=embedding, post_count=8)]

    signals = detect_trends(current, history, tau_sim=0.80, k=1.5, theta_ew=0.75)

    signal = next(signal for signal in signals if signal.cluster_id == "c0")
    assert "VOLUME" in signal.trend_types
    assert signal.evidence["mu"] == pytest.approx(4.0)
    assert signal.evidence["sigma"] == pytest.approx(1.0)
    assert signal.evidence["threshold"] == pytest.approx(5.5)


def test_volume_ignores_same_cycle_sibling_clusters():
    """Adjacency inside the current cycle is not evidence of a volume trend."""
    embedding = [1.0, 0.0]
    current = [_snapshot(f"c{index}", cycle_index=1, embedding=embedding, post_count=9) for index in range(5)]

    signals = detect_trends(current, [], tau_sim=0.80, k=1.5, theta_ew=0.75)

    assert not any("VOLUME" in signal.trend_types for signal in signals)


# --- detect_trends: PROPAGATION --------------------------------------------------


def test_propagation_flags_a_platform_not_seen_in_adjacent_history():
    embedding = [1.0, 0.0]
    history = [_snapshot("h1", cycle_index=1, embedding=embedding, platforms=["x"])]
    current = [_snapshot("c0", cycle_index=2, embedding=embedding, platforms=["x", "telegram"])]

    signals = detect_trends(current, history, tau_sim=0.80, k=1.5, theta_ew=0.75)

    signal = next(signal for signal in signals if signal.cluster_id == "c0")
    assert "PROPAGATION" in signal.trend_types
    assert signal.evidence["new_platforms"] == ["telegram"]


def test_propagation_does_not_flag_without_adjacent_history():
    embedding = [1.0, 0.0]
    unrelated = [0.0, 1.0]
    history = [_snapshot("h1", cycle_index=1, embedding=unrelated, platforms=["x"])]
    current = [_snapshot("c0", cycle_index=2, embedding=embedding, platforms=["x", "telegram"])]

    signals = detect_trends(current, history, tau_sim=0.80, k=1.5, theta_ew=0.75)

    assert not any(signal.cluster_id == "c0" for signal in signals)


def test_propagation_does_not_flag_when_platforms_are_a_subset():
    embedding = [1.0, 0.0]
    history = [_snapshot("h1", cycle_index=1, embedding=embedding, platforms=["x", "telegram", "reddit"])]
    current = [_snapshot("c0", cycle_index=2, embedding=embedding, platforms=["x"])]

    signals = detect_trends(current, history, tau_sim=0.80, k=1.5, theta_ew=0.75)

    assert not any(signal.cluster_id == "c0" and "PROPAGATION" in signal.trend_types for signal in signals)


# --- detect_trends: ESCALATION ---------------------------------------------------


def test_escalation_flags_a_severity_increase_over_adjacent_history():
    embedding = [1.0, 0.0]
    history = [_snapshot("h1", cycle_index=1, embedding=embedding, severity="MEDIUM")]
    current = [_snapshot("c0", cycle_index=2, embedding=embedding, severity="CRITICAL")]

    signals = detect_trends(current, history, tau_sim=0.80, k=1.5, theta_ew=0.75)

    signal = next(signal for signal in signals if signal.cluster_id == "c0")
    assert "ESCALATION" in signal.trend_types
    assert signal.evidence["max_history_severity"] == 1


def test_escalation_does_not_flag_equal_or_lower_severity():
    embedding = [1.0, 0.0]
    history = [_snapshot("h1", cycle_index=1, embedding=embedding, severity="HIGH")]
    current = [_snapshot("c0", cycle_index=2, embedding=embedding, severity="HIGH")]

    signals = detect_trends(current, history, tau_sim=0.80, k=1.5, theta_ew=0.75)

    assert not any(signal.cluster_id == "c0" and "ESCALATION" in signal.trend_types for signal in signals)


# --- detect_trends: EARLY_WARNING ------------------------------------------------


def test_early_warning_flags_at_or_above_theta():
    embedding = [1.0, 0.0]
    current = [_snapshot("c0", cycle_index=1, embedding=embedding, strategic_value_=0.75)]

    signals = detect_trends(current, [], tau_sim=0.80, k=1.5, theta_ew=0.75)

    signal = next(signal for signal in signals if signal.cluster_id == "c0")
    assert "EARLY_WARNING" in signal.trend_types
    assert signal.evidence["strategic_value"] == pytest.approx(0.75)


def test_early_warning_does_not_flag_below_theta():
    embedding = [1.0, 0.0]
    current = [_snapshot("c0", cycle_index=1, embedding=embedding, strategic_value_=0.74)]

    signals = detect_trends(current, [], tau_sim=0.80, k=1.5, theta_ew=0.75)

    assert signals == []


def test_trend_signal_is_frozen_and_immutable():
    import dataclasses

    import pytest as _pytest

    signal = TrendSignal(cluster_id="c0", trend_types=["EARLY_WARNING"], adjacent_history=0, evidence={"strategic_value": 0.9})

    with _pytest.raises(dataclasses.FrozenInstanceError):
        signal.evidence = {"other": 1}  # type: ignore[misc]

    replaced = dataclasses.replace(signal, evidence={**signal.evidence, "llm_fallback": "boom"})
    assert replaced is not signal
    assert "llm_fallback" not in signal.evidence
    assert replaced.evidence["llm_fallback"] == "boom"
    assert replaced.cluster_id == signal.cluster_id


def test_detect_trends_can_flag_multiple_axes_at_once():
    embedding = [1.0, 0.0]
    history = [
        _snapshot("h1", cycle_index=1, embedding=embedding, severity="LOW", platforms=["x"]),
        _snapshot("h2", cycle_index=2, embedding=embedding, severity="LOW", platforms=["x"]),
    ]
    current = [
        _snapshot(
            "c0",
            cycle_index=3,
            embedding=embedding,
            severity="CRITICAL",
            platforms=["x", "discord"],
            strategic_value_=0.9,
        )
    ]

    signals = detect_trends(current, history, tau_sim=0.80, k=1.5, theta_ew=0.75)

    signal = next(signal for signal in signals if signal.cluster_id == "c0")
    assert set(signal.trend_types) >= {"PROPAGATION", "ESCALATION", "EARLY_WARNING"}


# --- strategic_value -----------------------------------------------------------


def _decision(quality: float) -> EvaluationDecision:
    return EvaluationDecision(verdict=EvaluationVerdict.PROCEED, rounds=[], final_quality_score=quality, reason="ok")


def test_strategic_value_combines_quality_ioc_ratio_and_cve_presence():
    cluster = _cluster(cves=["CVE-2026-1111"])
    value = strategic_value(cluster, _decision(0.8), {"ioc_total": 10, "ioc_high": 5})

    expected = 0.5 * 0.8 + 0.3 * (5 / 10) + 0.2 * 1.0
    assert value == pytest.approx(expected)


def test_strategic_value_defaults_quality_to_zero_when_no_decision():
    cluster = _cluster(cves=[])
    value = strategic_value(cluster, None, {"ioc_total": 0, "ioc_high": 0})

    assert value == pytest.approx(0.0)


def test_strategic_value_is_clipped_to_the_unit_interval():
    cluster = _cluster(cves=["CVE-2026-1111"])
    value = strategic_value(cluster, _decision(1.0), {"ioc_total": 1, "ioc_high": 1})

    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(1.0)


def test_strategic_value_handles_zero_ioc_total_without_dividing_by_zero():
    cluster = _cluster(cves=[])
    value = strategic_value(cluster, _decision(0.5), {"ioc_total": 0, "ioc_high": 3})

    # ioc_total=0 must not raise ZeroDivisionError; the divisor floors at 1,
    # and the unclipped combination here exceeds 1.0 so the result is clipped.
    unclipped = 0.5 * 0.5 + 0.3 * (3 / 1) + 0.0
    assert unclipped > 1.0
    assert value == pytest.approx(1.0)
