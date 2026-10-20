"""Cross-cycle trend-query generation and policy-state updates."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from . import clock
from .context import InvestigationContext
from .dynamic_queries import DynamicQuery, generate_dynamic_queries, write_dynamic_queries
from .llm import LLMGenerationError
from .llm_fallbacks import category_of, record_fallback
from .models import Cluster, EvaluationDecision, PerspectiveFinding, Severity, to_jsonable
from .rl.features import cluster_embedding, cluster_tokens
from .rl.policy import brand_tokens
from .rl.replay import Transition, build_trajectory_records, transitions_from_trajectory
from .rl.trainer import train_policy
from .services import resolved_embedding_backend
from .settings import Settings
from . import trends as trends_module
from .trends import ClusterSnapshot, TrendSignal


logger = logging.getLogger(__name__)

_NON_QUERY_DOMAIN_SUFFIXES = (".html", ".htm", ".php", ".aspx", ".jsp", ".json", ".xml")

_SEVERITY_ORDER = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
_MIN_TRAINING_SEVERITY = Severity.MEDIUM
REPORT_HISTORY_LIMIT = 2_000

__all__ = [
    "DynamicQuery",
    "TrendSignal",
    "report_history_entry",
    "run_feedback",
    "update_policy_state",
]


def _queryable_domains(domains: list[str]) -> list[str]:
    queryable: list[str] = []
    for domain in domains:
        normalized = domain.strip().lower()
        if not normalized or normalized.endswith(_NON_QUERY_DOMAIN_SUFFIXES):
            continue
        if "/" in normalized or normalized.count(".") < 1:
            continue
        queryable.append(normalized)
    return queryable


def _strategic_value_payload(cluster: Cluster, decision: EvaluationDecision | None, context_stats: dict) -> dict[str, Any]:
    return {
        "cluster": {
            "id": cluster.id,
            "topic_name": cluster.topic_name,
            "description": cluster.description,
            "category": cluster.threat_category.value,
            "severity": cluster.severity.value,
            "cves": sorted(set(cluster.iocs.cves)),
            "post_count": len(cluster.posts),
        },
        "evaluation": {"final_quality_score": decision.final_quality_score if decision is not None else 0.0},
        "context_stats": context_stats,
    }


def _strategic_value_for(
    cluster: Cluster,
    decision: EvaluationDecision | None,
    context_stats: dict,
    settings: Settings,
) -> float:
    # The LLM variant is only attempted when an LLM is actually configured and
    # heuristics have not been forced; any generation failure falls back to
    # the deterministic heuristic rather than aborting the cycle.
    if settings.llm_configured() and not settings.allow_heuristic_reports:
        try:
            from .llm_feedback import strategic_value_with_llm

            payload = _strategic_value_payload(cluster, decision, context_stats)
            return strategic_value_with_llm(settings, payload)
        except LLMGenerationError as error:
            # DEBUG only: the failure text can embed the model's draft, which
            # summarises scraped posts. record_fallback below emits the single
            # content-free WARNING for this fallback.
            logger.debug(
                "LLM strategic value scoring failed for cluster %s; falling back to heuristic: %s",
                cluster.id,
                error,
            )
            record_fallback(
                "strategic_value_clusters",
                cluster_id=cluster.id,
                category=category_of(error),
                summary="strategic value draft rejected after one retry",
                detail=str(error),
            )
    return trends_module.strategic_value(cluster, decision, context_stats)


class TrendEmbedder:
    """Callable snapshot embedder that remembers which backend it really used.

    The offline pin is applied once, up front; the *runtime* fallback (the
    MiniLM weights not being loadable) can only be observed by embedding, so
    ``backend_used`` is filled in by the first call and is what
    ``cycle_summary.json`` reports as ``actual_trend_backend``. Before any call
    it reports the pin-resolved request, which is the honest answer for a cycle
    that produced no clusters to snapshot.
    """

    def __init__(self, settings: Settings) -> None:
        self._backend = resolved_embedding_backend(settings, settings.trend_embedding_backend)
        self._features = settings.embedding_features
        self.backend_used = self._backend

    def __call__(self, text: str) -> list[float]:
        vector, backend_used = trends_module.trend_embedding_with_backend(
            text, backend=self._backend, features=self._features
        )
        self.backend_used = backend_used
        return vector


def _embed_for(settings: Settings) -> TrendEmbedder:
    return TrendEmbedder(settings)


def run_feedback(
    state: dict, settings: Settings, *, embed: TrendEmbedder | None = None
) -> tuple[list[DynamicQuery], list[TrendSignal]]:
    """Cross-cycle trend detection and dynamic query generation.

    Snapshots every current cluster, loads the rolling history window, runs
    the four-axis trend detector, generates next-cycle dynamic queries
    (LLM-adapted when configured, heuristic otherwise), appends this cycle's
    snapshots to history, writes ``dynamic_queries.json``, and returns
    ``(queries, signals)``.
    """
    from .collectors.keywords import KeywordCatalog

    clusters: list[Cluster] = state.get("clusters", []) or []
    decisions_by_cluster: dict[str, EvaluationDecision] = state.get("decisions_by_cluster", {}) or {}
    contexts_by_cluster: dict[str, InvestigationContext] = state.get("contexts_by_cluster", {}) or {}
    reports: list[dict] = state.get("reports", []) or []
    cycle_index = int(state.get("cycle_index", 1))

    now = clock.now()
    observed_at = clock.now_iso()
    # The caller may pass its own embedder so it can read back which trend
    # embedding backend the cycle actually used (see TrendEmbedder).
    embed = embed if embed is not None else _embed_for(settings)

    snapshots: list[ClusterSnapshot] = []
    for cluster in clusters:
        decision = decisions_by_cluster.get(cluster.id)
        context = contexts_by_cluster.get(cluster.id)
        context_stats = context.stats if context is not None else {}
        value = _strategic_value_for(cluster, decision, context_stats, settings)
        snapshots.append(
            trends_module.snapshot_cluster(
                cluster,
                cycle_index=cycle_index,
                observed_at=observed_at,
                strategic_value=value,
                embed=embed,
            )
        )

    history_path = settings.resolved_history_dir() / "cluster_snapshots.jsonl"
    history = trends_module.load_history(history_path, now=now, window_hours=settings.trend_window_hours)

    signals = trends_module.detect_trends(
        snapshots,
        history,
        tau_sim=settings.trend_similarity_threshold,
        k=settings.volume_sigma_k,
        theta_ew=settings.early_warning_threshold,
        min_similar_snapshots=settings.trend_min_similar_snapshots,
        volume_min_posts=settings.trend_volume_min_posts,
    )

    catalog = KeywordCatalog.load(Path(settings.config_dir) / "keywords.json")
    snapshots_by_id = {snapshot.cluster_id: snapshot for snapshot in snapshots}
    clusters_by_id = {cluster.id: cluster for cluster in clusters}
    reports_by_cluster = {
        str(report["cluster_id"]): str(report.get("report_id", ""))
        for report in reports
        if report.get("cluster_id")
    }

    # generate_dynamic_queries returns signals with any LLM-fallback marker
    # applied via dataclasses.replace, never by mutating the detect_trends
    # output; use its returned signals (not the pre-fallback ones above) as
    # the cycle's trend signals from here on.
    queries, signals = generate_dynamic_queries(
        signals,
        snapshots_by_id,
        clusters_by_id,
        reports_by_cluster,
        settings=settings,
        catalog=catalog,
        now=now,
    )

    trends_module.append_history(history_path, snapshots)
    write_dynamic_queries(queries, Path(settings.output_dir) / "dynamic_queries.json")

    return queries, signals


def _read_previous_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as error:
        logger.warning("policy state at %s is unreadable (%s); starting from a cold policy", path, error)
        return {}
    if not isinstance(previous, dict):
        logger.warning("policy state at %s is not an object; starting from a cold policy", path)
        return {}
    return previous


def _successful_perspectives(findings: list[PerspectiveFinding]) -> set[str]:
    return {
        finding.perspective
        for finding in findings
        if any(result.status == "success" for result in finding.tool_results)
    }


def _discovered_ioc_total(findings: list[PerspectiveFinding]) -> int:
    values: set[str] = set()
    for finding in findings:
        for result in finding.tool_results:
            values.update(result.discovered_iocs.all_values())
    return len(values)


def _cluster_transitions(
    clusters: list[Cluster],
    reports: list[dict],
    findings_by_cluster: dict[str, list[PerspectiveFinding]],
    contexts_by_cluster: dict[str, InvestigationContext] | None = None,
) -> list[Transition]:
    # Only investigated clusters at or above MEDIUM severity contribute training
    # trajectories; below that the runtime never runs Step 3.
    quality_by_cluster = {
        str(report.get("cluster_id", "")): float(report.get("evidence_assessment", {}).get("quality_score", 0.0))
        for report in reports
    }
    contexts = contexts_by_cluster or {}
    transitions: list[Transition] = []
    for cluster in clusters:
        findings = findings_by_cluster.get(cluster.id) or []
        if not findings or _SEVERITY_ORDER[cluster.severity] < _SEVERITY_ORDER[_MIN_TRAINING_SEVERITY]:
            continue
        # Step 3 already recorded the per-query focus type; using its trajectory
        # keeps every training action attributed to the focus that ran it.
        context = contexts.get(cluster.id)
        transitions.extend(
            transitions_from_trajectory(
                build_trajectory_records(
                    cluster,
                    findings,
                    context_trajectory=context.trajectory if context is not None else None,
                ),
                cluster_state=cluster_embedding(cluster),
                final_quality=quality_by_cluster.get(cluster.id, 0.0),
                perspectives_used=_successful_perspectives(findings),
                ioc_total=_discovered_ioc_total(findings),
            )
        )
    return transitions


def report_history_entry(cluster: Cluster, report: dict) -> dict:
    """Similarity-matching record kept for one generated report."""
    return {
        "cluster_id": cluster.id,
        "report_id": str(report.get("report_id", "")),
        "signature": sorted(set(cluster_tokens(cluster))),
        "platforms": sorted({post.platform for post in cluster.posts if post.platform}),
        "category": cluster.threat_category.value,
        "severity": str(report.get("threat_severity") or cluster.severity.value),
        "cves": sorted(set(cluster.iocs.cves)),
        "iocs": sorted(set(cluster.iocs.urls + cluster.iocs.domains + cluster.iocs.ips + cluster.iocs.hashes)),
        "brands": brand_tokens(cluster.topic_name),
        "created_at": str(report.get("generated_at") or clock.now_iso()),
        "embedding": cluster_embedding(cluster).tolist(),
    }


def _append_report_history(previous: list[dict], clusters: list[Cluster], reports: list[dict]) -> list[dict]:
    cluster_by_id = {cluster.id: cluster for cluster in clusters}
    history = list(previous)
    for report in reports:
        cluster = cluster_by_id.get(str(report.get("cluster_id", "")))
        if cluster is None:
            continue
        history.append(report_history_entry(cluster, report))
    return history[-REPORT_HISTORY_LIMIT:]


def update_policy_state(
    clusters: list[Cluster],
    reports: list[dict],
    findings_by_cluster: dict[str, list[PerspectiveFinding]],
    path: Path,
    settings: Settings,
    contexts_by_cluster: dict[str, InvestigationContext] | None = None,
    cycle_index: int | None = None,
) -> dict:
    # Policy state is incremental: a new cycle reads the previous file, replays
    # this cycle's trajectories through behaviour cloning plus DQN fine-tuning,
    # records the reports used for similarity matching, then writes it back.
    previous = _read_previous_state(path)
    transitions = _cluster_transitions(clusters, reports, findings_by_cluster, contexts_by_cluster)
    # Every cycle draws a different, still reproducible, RNG stream. The real
    # cycle index is used when the caller knows it (the scheduler always does),
    # because it is a property of the run rather than of whatever state file
    # happens to be on disk: deriving the offset from the previous training
    # history instead made a re-run into a populated output root start cycle 1
    # at a different offset than a clean run did, silently changing the policy.
    if cycle_index is not None:
        cycle_offset = max(int(cycle_index) - 1, 0)
    else:
        cycle_offset = len(previous.get("training_history") or [])
    state = train_policy(previous, transitions, settings, rng=np.random.default_rng(settings.seed + cycle_offset))
    state["report_history"] = _append_report_history(state.get("report_history") or [], clusters, reports)
    state["clusters_seen"] = len(clusters)
    # Written compactly: the replay buffer holds up to 10,000 x 320-float states,
    # and indenting them would inflate the file by more than an order of magnitude.
    path.write_text(
        json.dumps(to_jsonable(state), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return state
