"""LangGraph orchestration for one TIBlender collection-to-report cycle."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph

from .clustering import cluster_posts
from .context import InvestigationContext, aggregate_context
from .dynamic_queries import DynamicQuery
from .evaluation import evaluate_cluster
from .feedback import TrendEmbedder, run_feedback, update_policy_state
from .investigation import investigate_cluster, investigate_followups
from .llm import compose_report_fields, normalize_post_batch, plan_investigation_queries, refine_clusters_with_llm
from .llm_fallbacks import (
    fallback_counters,
    fallback_reason_counters,
    reset_fallbacks,
    stages_for_cluster,
)
from .models import (
    Cluster,
    CycleResult,
    EvaluationDecision,
    EvaluationVerdict,
    NormalizedPost,
    PerspectiveFinding,
    Post,
    Severity,
    to_jsonable,
)
from .normalization import load_posts, normalize_posts, write_normalized_jsonl
from .paths import relative_path
from .reporting import build_report, write_report
from .rl.replay import build_trajectory_records
from .rl_policy import load_policy_state, select_investigation_strategy
from .services import resolve_services, resolved_embedding_backend
from .settings import Settings
from .tool_fixtures import ToolRecordingReplayLog
from .tools import ToolRunner
from .trends import TREND_TYPES, TrendSignal

# Both embedding provenance labels spell a runtime fallback the same way
# (see clustering._embedding_matrix and trends.trend_embedding_with_backend).
_FALLBACK_MARKER = "_fallback:"

WORKFLOW_NODES = [
    "normalize_posts",
    "cluster_posts",
    "investigate_clusters",
    "evaluate_clusters",
    "generate_reports",
    "cross_cycle_feedback",
]

_SEVERITY_ORDER = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


class CycleState(TypedDict, total=False):
    # LangGraph passes this state dictionary between explicit pipeline nodes.
    # Keeping the keys typed makes the simplified runtime easier to audit.
    settings: Settings
    posts: list[Post]
    normalized: list[NormalizedPost]
    threat_related: list[NormalizedPost]
    clusters: list[Cluster]
    findings_by_cluster: dict[str, list[PerspectiveFinding]]
    contexts_by_cluster: dict[str, InvestigationContext]
    decisions_by_cluster: dict[str, EvaluationDecision]
    reportable_clusters: list[Cluster]
    reports: list[dict]
    cycle_index: int
    dynamic_queries: list[DynamicQuery]
    trend_signals: list[TrendSignal]
    policy_state: dict
    trajectories: list[dict]
    clusters_investigated: int
    embedding_backend_used: str
    # One replay log per cycle, shared by both tool-running nodes, so
    # cycle_summary.json reports the cycle's real recording coverage.
    replay_log: ToolRecordingReplayLog


def _meets_threshold(severity: Severity, minimum: Severity) -> bool:
    return _SEVERITY_ORDER[severity] >= _SEVERITY_ORDER[minimum]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _trajectory_records(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    context: InvestigationContext | None = None,
) -> list[dict]:
    # Persist tool-action trajectories separately from reports so the feedback
    # layer can learn from successful investigation paths in the next cycle. The
    # record shape is owned by the RL package, which also consumes it.
    return build_trajectory_records(
        cluster,
        findings,
        context_trajectory=context.trajectory if context is not None else None,
    )


def _normalize_posts_node(state: CycleState) -> CycleState:
    # Step 1: load collected social posts, normalize text, and retain only
    # threat-related records for clustering.
    settings = state["settings"]
    posts = load_posts(settings.input_path)
    normalized_content_by_id = {}
    if posts and not settings.allow_heuristic_reports:
        normalized_content_by_id = normalize_post_batch(settings, posts)
    normalized = normalize_posts(
        posts,
        min_text_chars=settings.min_text_chars,
        normalized_content_by_id=normalized_content_by_id,
    )
    write_normalized_jsonl(normalized, settings.output_dir / "intermediate" / "normalized_posts.jsonl")
    return {
        "posts": posts,
        "normalized": normalized,
        "threat_related": [post for post in normalized if post.is_threat_related],
    }


def _cluster_posts_node(state: CycleState) -> CycleState:
    # Step 2: combine IoC-anchored grouping with semantic clustering so posts
    # without explicit indicators can still join the right campaign context.
    settings = state["settings"]
    clusters = cluster_posts(
        state["threat_related"],
        jaccard_threshold=settings.jaccard_threshold,
        absorption_threshold=settings.absorption_threshold,
        min_cluster_size=settings.min_cluster_size,
        min_samples=settings.min_samples,
        cluster_epsilon=settings.cluster_epsilon,
        merge_similarity_threshold=settings.merge_similarity_threshold,
        merge_max_time_gap_seconds=settings.merge_max_time_gap_seconds,
        merge_min_context_overlap=settings.merge_min_context_overlap,
        embedding_backend=resolved_embedding_backend(settings, settings.embedding_backend),
        embedding_model=settings.embedding_model,
        embedding_features=settings.embedding_features,
        embedding_fallback=settings.embedding_fallback,
        embedding_local_files_only=settings.embedding_local_files_only,
        embedding_device=settings.embedding_device,
        hdbscan_backend=settings.hdbscan_backend,
    )
    if clusters and not settings.allow_heuristic_reports:
        clusters = refine_clusters_with_llm(settings, clusters)
    _write_json(settings.output_dir / "intermediate" / "clusters.json", clusters)
    # Every cluster carries the same provenance label, produced by the one
    # embedding matrix this node built. With no clusters no embedding was
    # attempted, so the pin-resolved request is the honest answer.
    backend_used = next(
        (
            str(cluster.clustering_signals["embedding_backend"])
            for cluster in clusters
            if cluster.clustering_signals.get("embedding_backend")
        ),
        resolved_embedding_backend(settings, settings.embedding_backend),
    )
    return {"clusters": clusters, "embedding_backend_used": backend_used}



def _map_clusters_in_order(clusters: list[Cluster], worker: Callable, settings: Settings) -> list:
    """Run ``worker`` over each cluster in a worker pool, results in input order.

    Step 3 and Step 4 treat clusters as independent units -- the paper's worker
    pool -- so they can run concurrently. Determinism is preserved because the
    results are indexed by cluster position rather than completion time, the
    shared ``ToolRunner`` only reads tool recordings, and no worker writes into a
    structure another worker reads.
    """
    if not clusters:
        return []
    workers = max(1, settings.cluster_workers)
    if len(clusters) == 1 or workers == 1:
        return [worker(cluster) for cluster in clusters]
    with ThreadPoolExecutor(max_workers=min(workers, len(clusters))) as pool:
        futures = [pool.submit(worker, cluster) for cluster in clusters]
    return [future.result() for future in futures]


def _investigate_clusters_node(state: CycleState) -> CycleState:
    # Step 3: run multi-perspective investigation. A prior policy state can
    # reorder tool actions, but the same four perspectives remain available.
    settings = state["settings"]
    # One ToolRunner is shared by every cluster worker: it is stateless apart
    # from read-only recording lookups, so sharing it keeps offline cycles
    # byte-identical while avoiding a per-cluster service re-resolution.
    tool_runner = ToolRunner(settings, corpus=state["normalized"], replay_log=state.get("replay_log"))
    policy_state_path = settings.resolved_history_dir() / "policy_state.json"
    # Read once per node instead of re-parsing policy_state.json per cluster.
    policy_state = load_policy_state(policy_state_path)
    findings_by_cluster: dict[str, list[PerspectiveFinding]] = {}
    contexts_by_cluster: dict[str, InvestigationContext] = {}

    eligible = [cluster for cluster in state["clusters"] if _meets_threshold(cluster.severity, settings.min_severity)]

    def investigate(cluster: Cluster) -> tuple[list[PerspectiveFinding], InvestigationContext]:
        strategy = select_investigation_strategy(
            cluster, settings, policy_state_path, policy_state=policy_state
        )
        planned_queries = []
        if not settings.allow_heuristic_reports:
            planned_queries = plan_investigation_queries(settings, cluster)
        cluster.clustering_signals["rl_strategy"] = {
            "mode": strategy.mode,
            "similarity_score": strategy.similarity_score,
            "matched_report_id": strategy.matched_report_id,
            "action_priorities": strategy.action_priorities,
            "focus_by_perspective": strategy.focus_by_perspective,
            "q_values": strategy.q_values,
        }
        findings = investigate_cluster(
            cluster,
            settings=settings,
            runner=tool_runner,
            strategy=strategy,
            planned_queries=planned_queries,
        )
        return findings, aggregate_context(
            cluster, findings, max_resolutions_per_ip=settings.pivot_max_resolutions_per_ip
        )

    for cluster, (findings, context) in zip(eligible, _map_clusters_in_order(eligible, investigate, settings)):
        findings_by_cluster[cluster.id] = findings
        contexts_by_cluster[cluster.id] = context
    investigated = len(eligible)

    _write_json(
        settings.output_dir / "intermediate" / "investigation_contexts.json",
        contexts_by_cluster,
    )
    return {
        "findings_by_cluster": findings_by_cluster,
        "contexts_by_cluster": contexts_by_cluster,
        "clusters_investigated": investigated,
    }


def _evaluate_clusters_node(state: CycleState) -> CycleState:
    # Step 4: ask independent judges whether the evidence is reportable, needs
    # targeted follow-up, or should be suppressed.
    settings = state["settings"]
    tool_runner = ToolRunner(settings, corpus=state["normalized"], replay_log=state.get("replay_log"))
    decisions_by_cluster: dict[str, EvaluationDecision] = {}
    contexts_by_cluster: dict[str, InvestigationContext] = dict(state.get("contexts_by_cluster", {}))
    reportable_clusters: list[Cluster] = []

    evaluated = [cluster for cluster in state["clusters"] if state["findings_by_cluster"].get(cluster.id) is not None]
    initial_contexts = dict(contexts_by_cluster)

    def evaluate(cluster: Cluster) -> tuple[EvaluationDecision, InvestigationContext | None]:
        findings = state["findings_by_cluster"][cluster.id]
        # Each cluster worker keeps its own refreshed context and the caller
        # merges them back in cluster order, so a concurrent evaluation can
        # never interleave writes into the shared context map.
        refreshed_context: InvestigationContext | None = None

        def followup_executor(requested_followups: list[str], round_index: int) -> list[PerspectiveFinding]:
            del round_index
            # Step 4 follow-ups are routed back through the same tool runner so
            # re-evaluation receives concrete evidence instead of free text.
            return investigate_followups(
                cluster,
                requested_followups,
                tool_runner=tool_runner,
                max_queries_per_perspective=max(1, min(settings.max_queries_per_perspective, 2)),
            )

        def context_refresh(combined: list[PerspectiveFinding]) -> InvestigationContext:
            # Follow-up evidence changes the confidence tiers, so the judges of
            # the next round see a context re-aggregated from every finding.
            nonlocal refreshed_context
            refreshed_context = aggregate_context(cluster, combined)
            return refreshed_context

        decision = evaluate_cluster(
            cluster,
            findings,
            max_rounds=settings.max_evaluation_rounds,
            settings=settings,
            followup_executor=followup_executor,
            context=initial_contexts.get(cluster.id),
            context_refresh=context_refresh,
        )
        return decision, refreshed_context

    for cluster, (decision, refreshed) in zip(evaluated, _map_clusters_in_order(evaluated, evaluate, settings)):
        decisions_by_cluster[cluster.id] = decision
        if refreshed is not None:
            contexts_by_cluster[cluster.id] = refreshed
        if decision.verdict is EvaluationVerdict.PROCEED:
            reportable_clusters.append(cluster)

    _write_json(settings.output_dir / "intermediate" / "evaluation_decisions.json", decisions_by_cluster)
    _write_json(settings.output_dir / "intermediate" / "investigation_contexts.json", contexts_by_cluster)
    # Written here rather than in the feedback node because these two nodes
    # are the only ones that run external tools: by this point the cycle's
    # request set is complete.
    replay_log = state.get("replay_log")
    if replay_log is not None:
        _write_json(settings.output_dir / "intermediate" / "tool_recording_misses.json", replay_log.payload())
    return {
        "decisions_by_cluster": decisions_by_cluster,
        "reportable_clusters": reportable_clusters,
        "findings_by_cluster": state["findings_by_cluster"],
        "contexts_by_cluster": contexts_by_cluster,
    }


def _generate_reports_node(state: CycleState) -> CycleState:
    # Step 5: compose LLM-backed report fields and export all supported formats.
    settings = state["settings"]
    services = resolve_services(settings)
    contexts_by_cluster = state.get("contexts_by_cluster", {})
    generation_mode = "offline_deterministic" if settings.allow_heuristic_reports else "llm"
    reports: list[dict] = []
    for cluster in state["reportable_clusters"]:
        findings = state["findings_by_cluster"][cluster.id]
        decision = state["decisions_by_cluster"][cluster.id]
        report_fields = compose_report_fields(settings, cluster, findings, decision)
        # A cluster whose pipeline used the LLM everywhere is "llm"; one where
        # a stage fell back *for this report* is "llm_partial", with the stage
        # names recorded so the degradation is auditable per report. The scope
        # is this cluster and its own posts: a normalization chunk carrying one
        # of them, a refinement batch carrying this cluster, or one of this
        # cluster's own planner/perspective/judge/report-field calls. Another
        # cluster's failed batch never reaches this report.
        stage_fallbacks = stages_for_cluster(cluster.id, [post.id for post in cluster.posts])
        report = build_report(
            cluster,
            findings,
            decision,
            report_fields=report_fields,
            context=contexts_by_cluster.get(cluster.id),
            services=services,
            generation_mode=(
                "llm_partial" if stage_fallbacks and generation_mode == "llm" else generation_mode
            ),
            llm_stage_fallbacks=stage_fallbacks,
            pivot_max_resolutions_per_ip=settings.pivot_max_resolutions_per_ip,
        )
        write_report(report, settings.output_dir)
        reports.append(report)
    return {"reports": reports}


def _trend_counts(signals: list[TrendSignal]) -> dict[str, int]:
    counts = {trend_type: 0 for trend_type in TREND_TYPES}
    for signal in signals:
        for trend_type in signal.trend_types:
            counts[trend_type] = counts.get(trend_type, 0) + 1
    return counts


def _history_size(settings: Settings) -> int:
    history_path = settings.resolved_history_dir() / "cluster_snapshots.jsonl"
    if not history_path.exists():
        return 0
    return sum(1 for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip())


def _output_root(settings: Settings) -> Path:
    """Root the cycle's own output paths are recorded relative to.

    Under the multi-cycle scheduler this is ``<output_root>`` (the parent of
    the shared ``history/`` directory, with ``output_dir`` being one
    ``cycle_NNN/`` beneath it); for a single ``run`` it is ``output_dir``
    itself.
    """
    history_dir = settings.resolved_history_dir()
    if history_dir.name == "history":
        return history_dir.parent
    return Path(settings.output_dir)


def _cross_cycle_feedback_node(state: CycleState) -> CycleState:
    # Cross-cycle feedback records useful trends and successful tool paths for
    # the next cycle's collection queries and RL-guided investigation order.
    settings = state["settings"]
    contexts_by_cluster = state.get("contexts_by_cluster", {})
    trend_embedder = TrendEmbedder(settings)
    dynamic_queries, trend_signals = run_feedback(state, settings, embed=trend_embedder)
    policy_state = update_policy_state(
        state["clusters"],
        state["reports"],
        state["findings_by_cluster"],
        settings.resolved_history_dir() / "policy_state.json",
        settings,
        contexts_by_cluster=contexts_by_cluster,
        cycle_index=state.get("cycle_index"),
    )

    trajectories: list[dict] = []
    for cluster in state["clusters"]:
        findings = state["findings_by_cluster"].get(cluster.id, [])
        trajectories.extend(_trajectory_records(cluster, findings, contexts_by_cluster.get(cluster.id)))
    _write_json(settings.output_dir / "intermediate" / "investigation_trajectories.json", trajectories)

    summary = _cycle_summary(
        state, dynamic_queries, trend_signals, policy_state, trajectories, trend_embedder
    )
    _write_json(settings.output_dir / "cycle_summary.json", summary)
    return {
        "dynamic_queries": dynamic_queries,
        "trend_signals": trend_signals,
        "policy_state": policy_state,
        "trajectories": trajectories,
    }


def _cycle_summary(
    state: CycleState,
    dynamic_queries: list[DynamicQuery],
    trend_signals: list[TrendSignal],
    policy_state: dict,
    trajectories: list[dict],
    trend_embedder: TrendEmbedder,
) -> dict:
    settings = state["settings"]
    reports = state.get("reports", [])
    output_root = _output_root(settings)
    # `resolved_*` answers "did the offline pin change the request?"; only the
    # `actual_*` labels answer "what did this run embed with?", which also
    # covers the runtime fallback taken when the model cannot be loaded.
    actual_backend = state.get(
        "embedding_backend_used", resolved_embedding_backend(settings, settings.embedding_backend)
    )
    actual_trend_backend = trend_embedder.backend_used
    return {
        "workflow_engine": "langgraph_stategraph",
        "workflow_nodes": WORKFLOW_NODES,
        "posts_loaded": len(state.get("posts", [])),
        "normalized_posts": len(state.get("normalized", [])),
        "threat_related_posts": len(state.get("threat_related", [])),
        "clusters_generated": len(state.get("clusters", [])),
        "clusters_investigated": state.get("clusters_investigated", 0),
        "reports_generated": len(reports),
        # How much of this cycle's external-tool traffic actually replayed
        # from a recorded response. Reported (not just logged) because
        # README/FIDELITY cite the coverage, and verify-sample pins it.
        "tool_recording_replay": (
            state["replay_log"].rates() if state.get("replay_log") is not None else {}
        ),
        # Paths are recorded relative to the run's output root: an absolute
        # host path would differ between the host and the container and make
        # committed expected output machine-specific.
        "output_dir": relative_path(settings.output_dir, output_root),
        "pipeline_steps": [
            "Step 1 adaptive collection/post normalization",
            "Step 2 dual-track campaign clustering",
            "Step 3 multi-perspective investigation",
            "Step 4 adaptive evaluation loop with follow-up investigation",
            "Step 5 structured JSON/Markdown/PDF/STIX report generation",
            "Cross-cycle trend query and policy feedback",
        ],
        "report_generation": {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.azure_openai_deployment or settings.llm_model,
            "allow_heuristic_reports": settings.allow_heuristic_reports,
            "mode": "offline_deterministic" if settings.allow_heuristic_reports else "llm",
        },
        "post_normalization": {
            "mode": "offline_deterministic" if settings.allow_heuristic_reports else "llm_english_summarization",
            "summary_source": "local_rules" if settings.allow_heuristic_reports else "post_normalization_agent",
        },
        "cluster_refinement": {
            "mode": "offline_deterministic" if settings.allow_heuristic_reports else "llm_campaign_splitting",
        },
        # What the run *asked for* versus what it actually embedded with.
        # `hyperparameters.embedding_backend` echoes the request; offline mode
        # pins both backends to the deterministic hashing vectorizer
        # (services.resolved_embedding_backend), so without this block a
        # summary could claim bge_large/minilm while
        # intermediate/clusters.json recorded sklearn_hashing_vectorizer.
        "embedding_resolution": {
            "requested_backend": settings.embedding_backend,
            "resolved_backend": resolved_embedding_backend(settings, settings.embedding_backend),
            "actual_backend": actual_backend,
            "requested_trend_backend": settings.trend_embedding_backend,
            "resolved_trend_backend": resolved_embedding_backend(settings, settings.trend_embedding_backend),
            "actual_trend_backend": actual_trend_backend,
            "offline_pin_applied": settings.is_offline(),
            "embedding_fallback_applied": _FALLBACK_MARKER in actual_backend
            or _FALLBACK_MARKER in actual_trend_backend,
        },
        # Per-stage count of units that could not be generated by the LLM and
        # used their deterministic path instead, plus two bookkeeping counters
        # that are not stages: "perspective_focus_padded" (padding, not a
        # fallback) and "unattributed_fallbacks" (how many of the fallbacks
        # counted here carried no cluster/post scope, so they could not reach
        # any report). All keys are always present.
        "llm_fallbacks": fallback_counters(),
        # Why those fallbacks happened, bucketed by a category this code
        # attaches where the failure is raised: truncated / timeout /
        # http_status / network / empty_content / json_parse /
        # invalid_response / unknown. All keys are always present.
        #
        # Invariant: sum(llm_fallback_reasons) == sum of the *stage* counters
        # above, i.e. llm_fallbacks.FALLBACK_KEYS. The two bookkeeping
        # counters are excluded -- padding has no failure reason to bucket,
        # and unattributed_fallbacks is a flag over events already counted.
        "llm_fallback_reasons": fallback_reason_counters(),
        # How many of this cycle's reports had a report_text section rewritten
        # by the deterministic English composer because it was not English
        # (tiblender_runtime/language.py). Both generation paths pass through
        # the same guard, so this counts leaks from either.
        "language_guard": {
            "reports_checked": len(reports),
            "reports_rewritten": sum(
                1 for report in reports if (report.get("metadata") or {}).get("language_fallback")
            ),
            "sections_rewritten": sorted(
                {
                    str(section)
                    for report in reports
                    for section in ((report.get("metadata") or {}).get("language_fallback_sections") or [])
                }
            ),
        },
        "investigation_planning": {
            "mode": "offline_deterministic" if settings.allow_heuristic_reports else "llm_strategy_planner",
        },
        "feedback": {
            "dynamic_query_count": len(dynamic_queries),
            "trajectory_count": len(trajectories),
            "policy_type": policy_state.get("policy_type", ""),
            "training_status": policy_state.get("training_status", ""),
            "policy_state_path": relative_path(
                settings.resolved_history_dir() / "policy_state.json", output_root
            ),
            "trajectory_path": relative_path(
                settings.output_dir / "intermediate" / "investigation_trajectories.json", output_root
            ),
            "trend_counts": _trend_counts(trend_signals),
            "history_size": _history_size(settings),
        },
        "hyperparameters": {
            "llm_temperature": settings.llm_temperature,
            "llm_max_output_tokens": settings.llm_max_output_tokens,
            "llm_batch_size": settings.llm_batch_size,
            "llm_cluster_batch_size": settings.llm_cluster_batch_size,
            "llm_max_concurrent": settings.llm_max_concurrent,
            "cluster_workers": settings.cluster_workers,
            "llm_timeout_seconds": settings.llm_timeout_seconds,
            "min_text_chars": settings.min_text_chars,
            "jaccard_threshold": settings.jaccard_threshold,
            "absorption_threshold": settings.absorption_threshold,
            "min_cluster_size": settings.min_cluster_size,
            "min_samples": settings.min_samples,
            "cluster_epsilon": settings.cluster_epsilon,
            "embedding_backend": settings.embedding_backend,
            "embedding_model": settings.embedding_model,
            "embedding_features": settings.embedding_features,
            "embedding_fallback": settings.embedding_fallback,
            "embedding_local_files_only": settings.embedding_local_files_only,
            "hdbscan_backend": settings.hdbscan_backend,
            "merge_similarity_threshold": settings.merge_similarity_threshold,
            "merge_max_time_gap_seconds": settings.merge_max_time_gap_seconds,
            "merge_min_context_overlap": settings.merge_min_context_overlap,
            "min_severity": settings.min_severity.value,
            "max_evaluation_rounds": settings.max_evaluation_rounds,
            "rl_similarity_gate": settings.rl_similarity_gate,
            "rl_max_steps": settings.rl_max_steps,
            "rl_action_top_k": settings.rl_action_top_k,
            "rl_policy_backend": settings.rl_policy_backend,
            "reference_rl_hidden_dim": settings.reference_rl_hidden_dim,
            "reference_rl_dropout": settings.reference_rl_dropout,
            "reference_rl_bc_epochs": settings.reference_rl_bc_epochs,
            "reference_rl_bc_batch_size": settings.reference_rl_bc_batch_size,
            "reference_rl_bc_learning_rate": settings.reference_rl_bc_learning_rate,
            "reference_rl_bc_patience": settings.reference_rl_bc_patience,
            "reference_rl_bc_validation_split": settings.reference_rl_bc_validation_split,
            "reference_rl_dqn_episodes": settings.reference_rl_dqn_episodes,
            "reference_rl_dqn_learning_rate": settings.reference_rl_dqn_learning_rate,
            "reference_rl_replay_capacity": settings.reference_rl_replay_capacity,
            "reference_rl_training_min_transitions": settings.reference_rl_training_min_transitions,
            "reference_rl_epsilon_init": settings.reference_rl_epsilon_init,
            "reference_rl_epsilon_min": settings.reference_rl_epsilon_min,
            "reference_rl_epsilon_decay": settings.reference_rl_epsilon_decay,
            "reference_rl_discount": settings.reference_rl_discount,
            "trend_window_hours": settings.trend_window_hours,
            "trend_similarity_threshold": settings.trend_similarity_threshold,
            "volume_sigma_k": settings.volume_sigma_k,
            "trend_min_similar_snapshots": settings.trend_min_similar_snapshots,
            "trend_volume_min_posts": settings.trend_volume_min_posts,
            "early_warning_threshold": settings.early_warning_threshold,
            "trend_embedding_backend": settings.trend_embedding_backend,
            "max_queries_per_perspective": settings.max_queries_per_perspective,
            "perspective_workers": settings.perspective_workers,
            "pivot_depth": settings.pivot_depth,
            "enable_pivot_iocs": settings.enable_pivot_iocs,
            "enable_real_tools": settings.enable_real_tools,
            "tool_timeout_seconds": settings.tool_timeout_seconds,
            "max_tool_results": settings.max_tool_results,
            "tool_max_bytes": settings.tool_max_bytes,
            "web_crawl_max_bytes": settings.web_crawl_max_bytes,
            "web_crawl_render_js": settings.web_crawl_render_js,
            "web_crawl_wait_ms": settings.web_crawl_wait_ms,
            "search_engine_provider": settings.search_engine_provider,
            "search_engine_max_results": settings.search_engine_max_results,
            "searxng_base_url": settings.searxng_base_url,
        },
    }


def build_cycle_graph():
    # The graph keeps each pipeline stage explicit; this makes intermediate
    # files, state handoff, and later feedback updates easier to inspect.
    builder = StateGraph(CycleState)
    builder.add_node("normalize_posts", _normalize_posts_node)
    builder.add_node("cluster_posts", _cluster_posts_node)
    builder.add_node("investigate_clusters", _investigate_clusters_node)
    builder.add_node("evaluate_clusters", _evaluate_clusters_node)
    builder.add_node("generate_reports", _generate_reports_node)
    builder.add_node("cross_cycle_feedback", _cross_cycle_feedback_node)
    builder.set_entry_point("normalize_posts")
    builder.add_edge("normalize_posts", "cluster_posts")
    builder.add_edge("cluster_posts", "investigate_clusters")
    builder.add_edge("investigate_clusters", "evaluate_clusters")
    builder.add_edge("evaluate_clusters", "generate_reports")
    builder.add_edge("generate_reports", "cross_cycle_feedback")
    builder.add_edge("cross_cycle_feedback", END)
    return builder.compile()


def run_graph_cycle(settings: Settings, cycle_index: int = 1) -> CycleResult:
    settings.validate_report_generation()
    settings.prepare_output_dirs()
    reset_fallbacks()
    graph = build_cycle_graph()
    final_state = graph.invoke(
        {"settings": settings, "cycle_index": cycle_index, "replay_log": ToolRecordingReplayLog()}
    )
    return CycleResult(
        posts_loaded=len(final_state.get("posts", [])),
        normalized_posts=len(final_state.get("normalized", [])),
        clusters_generated=len(final_state.get("clusters", [])),
        clusters_investigated=final_state.get("clusters_investigated", 0),
        reports_generated=len(final_state.get("reports", [])),
        output_dir=settings.output_dir,
    )
