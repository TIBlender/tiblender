import json

from tiblender_runtime.context import aggregate_context
from tiblender_runtime.feedback import _cluster_transitions, run_feedback
from tiblender_runtime.models import (
    Cluster,
    EvaluationDecision,
    EvaluationVerdict,
    IocSet,
    NormalizedPost,
    PerspectiveFinding,
    Post,
    Severity,
    ThreatCategory,
    ToolQuery,
    ToolResult,
)
from tiblender_runtime.rl.actions import ACTION_SPACE
from tiblender_runtime.settings import Settings
from tiblender_runtime.trends import ClusterSnapshot, append_history


def _cluster() -> Cluster:
    post = NormalizedPost(
        post=Post(id="p1", platform="x", content="Citrix NetScaler CVE-2026-3055 active probing bad.example"),
        normalized_content="Citrix NetScaler CVE-2026-3055 active probing bad.example",
        iocs=IocSet(cves=["CVE-2026-3055"], domains=["citrix-netscaler-under-active-recon-for.html", "bad.example"]),
        threat_category=ThreatCategory.VULNERABILITY,
    )
    return Cluster(
        id="cluster-netscaler",
        topic_name="CVE-2026-3055 exploitation campaign",
        description="Citrix NetScaler active probing",
        posts=[post],
        iocs=post.iocs,
        severity=Severity.CRITICAL,
        threat_category=ThreatCategory.VULNERABILITY,
    )


def _feedback_settings(tmp_path) -> Settings:
    return Settings(
        history_dir=tmp_path / "history",
        output_dir=tmp_path / "output",
        config_dir="config",
        llm_provider="heuristic",
        allow_heuristic_reports=True,
    )


def _seed_history(history_dir, *, embedding: list[float]) -> None:
    path = history_dir / "cluster_snapshots.jsonl"
    for cycle_index in (1, 2, 3):
        append_history(
            path,
            [
                ClusterSnapshot(
                    cluster_id="cluster-history",
                    cycle_index=cycle_index,
                    observed_at="2026-03-19T00:00:00Z",
                    embedding=embedding,
                    iocs=[],
                    cves=[],
                    severity="MEDIUM",
                    platforms=["x"],
                    post_count=2,
                    category="vulnerability",
                    topic_name="Prior cycle campaign",
                    strategic_value=0.2,
                    languages=["en"],
                )
            ],
        )


def test_run_feedback_writes_history_and_dynamic_queries(tmp_path):
    from tiblender_runtime import clock

    settings = _feedback_settings(tmp_path)
    clock.set_fixed_clock("2026-03-20T00:00:00Z")
    try:
        _seed_history(settings.history_dir, embedding=[0.1] * settings.embedding_features)

        cluster = _cluster()
        decision = EvaluationDecision(
            verdict=EvaluationVerdict.PROCEED, rounds=[], final_quality_score=0.9, reason="ok"
        )
        state = {
            "clusters": [cluster],
            "decisions_by_cluster": {cluster.id: decision},
            "contexts_by_cluster": {},
            "reports": [{"cluster_id": cluster.id, "report_id": "TIBLENDER-1"}],
            "cycle_index": 4,
        }

        queries, signals = run_feedback(state, settings)

        history_path = settings.history_dir / "cluster_snapshots.jsonl"
        history_lines = [line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(history_lines) == 4  # 3 seeded + 1 appended this cycle

        dynamic_path = settings.output_dir / "dynamic_queries.json"
        assert dynamic_path.exists()
        payload = json.loads(dynamic_path.read_text(encoding="utf-8"))
        assert isinstance(payload, list)

        assert isinstance(queries, list)
        assert isinstance(signals, list)
    finally:
        clock.set_fixed_clock(None)


def test_run_feedback_detects_early_warning_from_high_quality_evaluation(tmp_path):
    from tiblender_runtime import clock

    settings = _feedback_settings(tmp_path)
    clock.set_fixed_clock("2026-03-20T00:00:00Z")
    try:
        cluster = _cluster()
        # final_quality_score=1.0, ioc_high/ioc_total=1.0, and a CVE present:
        # 0.5*1.0 + 0.3*1.0 + 0.2*1.0 = 1.0 (clipped), comfortably >= theta_ew=0.75.
        decision = EvaluationDecision(
            verdict=EvaluationVerdict.PROCEED, rounds=[], final_quality_score=1.0, reason="ok"
        )
        context_stats = {"ioc_total": 2, "ioc_high": 2}
        state = {
            "clusters": [cluster],
            "decisions_by_cluster": {cluster.id: decision},
            "contexts_by_cluster": {},
            "reports": [],
            "cycle_index": 1,
        }
        # Inject context stats via a lightweight stand-in that only needs .stats.
        class _Context:
            stats = context_stats

        state["contexts_by_cluster"] = {cluster.id: _Context()}

        queries, signals = run_feedback(state, settings)

        assert any("EARLY_WARNING" in signal.trend_types for signal in signals)
    finally:
        clock.set_fixed_clock(None)


def test_run_feedback_returns_empty_when_there_are_no_clusters(tmp_path):
    settings = _feedback_settings(tmp_path)
    state = {"clusters": [], "decisions_by_cluster": {}, "contexts_by_cluster": {}, "reports": []}

    queries, signals = run_feedback(state, settings)

    assert queries == []
    assert signals == []
    assert (settings.output_dir / "dynamic_queries.json").exists()


def test_update_policy_state_offsets_the_seed_by_the_training_round(tmp_path, monkeypatch):
    import json

    import numpy as np

    from tiblender_runtime import feedback

    path = tmp_path / "policy_state.json"
    path.write_text(json.dumps({"training_history": [{"status": "bc_only_insufficient_transitions"}] * 3}), encoding="utf-8")
    seeds: list[int] = []
    real_default_rng = np.random.default_rng

    def spy(seed):
        seeds.append(int(seed))
        return real_default_rng(seed)

    monkeypatch.setattr(feedback.np.random, "default_rng", spy)
    settings = Settings(reference_rl_hidden_dim=8)

    feedback.update_policy_state([_cluster()], [], {}, path, settings)

    # The first draw is the training rng; later ones are the network's own init.
    assert seeds[0] == settings.seed + 3


def test_update_policy_state_seed_offset_uses_the_cycle_index_when_given(tmp_path, monkeypatch):
    # The real cycle index is a property of the run, not of whatever state
    # file happens to be on disk: a re-run into a populated output root must
    # draw cycle 1's RNG stream exactly as a clean run does.
    import json

    import numpy as np

    from tiblender_runtime import feedback

    path = tmp_path / "policy_state.json"
    path.write_text(json.dumps({"training_history": [{"status": "x"}] * 3}), encoding="utf-8")
    seeds: list[int] = []
    real_default_rng = np.random.default_rng

    def spy(seed):
        seeds.append(int(seed))
        return real_default_rng(seed)

    monkeypatch.setattr(feedback.np.random, "default_rng", spy)
    settings = Settings(reference_rl_hidden_dim=8)

    feedback.update_policy_state([_cluster()], [], {}, path, settings, cycle_index=1)
    assert seeds[0] == settings.seed  # cycle_index - 1 == 0, not len(training_history) == 3

    seeds.clear()
    feedback.update_policy_state([_cluster()], [], {}, path, settings, cycle_index=3)
    assert seeds[0] == settings.seed + 2


def test_update_policy_state_recovers_from_a_malformed_file(tmp_path):
    from tiblender_runtime.feedback import update_policy_state

    path = tmp_path / "policy_state.json"
    path.write_text("{not json", encoding="utf-8")

    state = update_policy_state([_cluster()], [], {}, path, Settings(reference_rl_hidden_dim=8))

    assert state["policy_type"] == "multiaction_dqn"
    assert state["training_status"] == "bc_only_insufficient_transitions"
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 2


def test_policy_transitions_use_the_context_focus_attribution():
    """Training actions follow the focus that ran each query, not the first focus."""
    cluster = _cluster()
    query = ToolQuery(
        "Actor",
        "SEARCH_ENGINE",
        {"query": "bad.example campaign"},
        focus_type="campaign_tracking",
    )
    finding = PerspectiveFinding(
        perspective="Actor",
        focus_types=["threat_actor_attribution_and_profiling", "campaign_tracking"],
        queries=[query],
        tool_results=[
            ToolResult(
                tool_type="SEARCH_ENGINE",
                query_params=dict(query.parameters),
                summary="search",
            )
        ],
        insights=[],
        confidence=0.6,
    )
    findings_by_cluster = {cluster.id: [finding]}
    context = aggregate_context(cluster, [finding])

    with_context = _cluster_transitions([cluster], [], findings_by_cluster, {cluster.id: context})
    without_context = _cluster_transitions([cluster], [], findings_by_cluster)

    assert with_context[0].action == ACTION_SPACE.index("campaign_tracking")
    assert without_context[0].action == ACTION_SPACE.index("threat_actor_attribution_and_profiling")
