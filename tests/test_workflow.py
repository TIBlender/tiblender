import importlib.machinery
import json
import sys
import types
from pathlib import Path

import pytest

from tiblender_runtime.pipeline import run_cycle
from tiblender_runtime.settings import Settings
from tiblender_runtime.workflow import build_cycle_graph


def test_build_cycle_graph_exposes_workflow_nodes():
    graph = build_cycle_graph()

    assert graph is not None


def test_run_cycle_records_langgraph_workflow_summary(tmp_path: Path):
    settings = Settings(
        input_path=Path("sample_runs/collector_experiment_sample/sample_run_posts.jsonl"),
        output_dir=tmp_path,
        history_dir=tmp_path / "history",
        llm_provider="heuristic",
        allow_heuristic_reports=True,
        enable_real_tools=False,
    )

    result = run_cycle(settings)

    summary = json.loads((tmp_path / "cycle_summary.json").read_text(encoding="utf-8"))
    assert result.reports_generated >= 1
    assert summary["workflow_engine"] == "langgraph_stategraph"
    assert summary["workflow_nodes"] == [
        "normalize_posts",
        "cluster_posts",
        "investigate_clusters",
        "evaluate_clusters",
        "generate_reports",
        "cross_cycle_feedback",
    ]
    assert summary["post_normalization"]["mode"] == "offline_deterministic"
    assert summary["cluster_refinement"]["mode"] == "offline_deterministic"
    assert summary["investigation_planning"]["mode"] == "offline_deterministic"
    assert summary["feedback"]["trajectory_count"] >= 1
    assert summary["hyperparameters"]["perspective_workers"] >= 1


def test_run_cycle_writes_investigation_contexts_and_step_trajectories(tmp_path: Path):
    settings = Settings(
        input_path=Path("sample_runs/collector_experiment_sample/sample_run_posts.jsonl"),
        output_dir=tmp_path,
        history_dir=tmp_path / "history",
        llm_provider="heuristic",
        allow_heuristic_reports=True,
        enable_real_tools=False,
    )

    run_cycle(settings)

    contexts = json.loads((tmp_path / "intermediate" / "investigation_contexts.json").read_text(encoding="utf-8"))
    assert contexts
    context = next(iter(contexts.values()))
    assert set(context) >= {"cluster_id", "iocs", "cves", "ttps", "per_perspective", "trajectory", "stats", "filtered_benign"}
    assert set(context["per_perspective"]) == {"Infrastructure", "Technical", "Social", "Actor"}
    assert all(item["extraction_type"] in {"Direct", "Pivot"} for item in context["iocs"])
    assert all(item["confidence"] in {"HIGH", "MEDIUM", "LOW"} for item in context["iocs"])

    trajectories = json.loads((tmp_path / "intermediate" / "investigation_trajectories.json").read_text(encoding="utf-8"))
    assert trajectories
    assert set(trajectories[0]) >= {
        "cluster_id",
        "step",
        "perspective",
        "focus_type",
        "tool_type",
        "parameters",
        "status",
        "new_iocs",
        "new_cves",
        "new_ttps",
        "duplicate",
    }


def _cycle_settings(tmp_path: Path, cluster_workers: int) -> Settings:
    return Settings(
        input_path=Path("sample_runs/collector_experiment_sample/sample_run_posts.jsonl"),
        output_dir=tmp_path,
        history_dir=tmp_path / "history",
        llm_provider="heuristic",
        allow_heuristic_reports=True,
        enable_real_tools=False,
        cluster_workers=cluster_workers,
    )


def test_cluster_workers_and_llm_max_concurrent_are_recorded_as_hyperparameters(tmp_path: Path):
    run_cycle(_cycle_settings(tmp_path, 4))

    hyperparameters = json.loads((tmp_path / "cycle_summary.json").read_text(encoding="utf-8"))["hyperparameters"]

    assert hyperparameters["cluster_workers"] == 4
    assert hyperparameters["llm_max_concurrent"] >= 1


def test_cluster_results_keep_their_order_with_four_workers(tmp_path: Path):
    """Four cluster workers must produce exactly the serial result."""
    concurrent = run_cycle(_cycle_settings(tmp_path / "concurrent", 4))
    serial = run_cycle(_cycle_settings(tmp_path / "serial", 1))

    def cluster_order(root: Path) -> list[str]:
        contexts = json.loads((root / "intermediate" / "investigation_contexts.json").read_text(encoding="utf-8"))
        return list(contexts)

    assert cluster_order(tmp_path / "concurrent") == cluster_order(tmp_path / "serial")
    assert concurrent.reports_generated == serial.reports_generated

    def decisions(root: Path) -> dict:
        return json.loads((root / "intermediate" / "evaluation_decisions.json").read_text(encoding="utf-8"))

    assert decisions(tmp_path / "concurrent") == decisions(tmp_path / "serial")


def test_investigate_node_reads_policy_state_once_per_node(tmp_path: Path, monkeypatch):
    from tiblender_runtime import workflow as workflow_module

    reads: list[Path] = []
    original = workflow_module.load_policy_state

    def counting_load(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(workflow_module, "load_policy_state", counting_load)

    run_cycle(_cycle_settings(tmp_path, 4))

    assert len(reads) == 1


@pytest.mark.slow
def test_offline_run_cycles_summary_is_identical_at_one_and_four_cluster_workers(tmp_path: Path):
    """The real determinism claim: a full offline cycle, recordings and all.

    The per-node assertions above compare structures inside one process; this
    runs `run_cycles` end to end in offline mode at both worker counts and
    compares the same `summarize_output` reduction that `verify-sample` uses,
    so a concurrency-dependent difference anywhere in the tree would surface.
    """
    from tiblender_runtime.scheduler import run_cycles
    from tiblender_runtime.verify import summarize_output

    def run(root: Path, cluster_workers: int) -> dict:
        # Default offline settings, exactly as `verify-sample` runs them, so
        # recordings and the deterministic report path are both exercised. Two
        # cycles, because cycle 1 legitimately produces no report and cycle 2
        # is the first with report IoCs to compare.
        settings = Settings(
            output_dir=root,
            history_dir=root / "history",
            mode="offline",
            cluster_workers=cluster_workers,
            llm_max_concurrent=cluster_workers,
        )
        run_cycles(
            settings,
            cycles=2,
            output_root=root,
            interval_seconds=0,
            sleeper=lambda _seconds: None,
        )
        return summarize_output(root)

    serial = run(tmp_path / "w1", 1)
    concurrent = run(tmp_path / "w4", 4)

    assert concurrent == serial
    # Guard against the comparison going vacuous if the fixture bundle changes.
    cycle_two = concurrent["cycles"]["cycle_002"]
    assert cycle_two["reports"] >= 1 and cycle_two["iocs_by_report"]


def test_run_cycle_records_the_resolved_embedding_backends_not_only_the_requested_ones(tmp_path: Path):
    """`TIBLENDER_MODE=offline` pins both embedding backends to hashing for
    determinism. Recording only the requested backend made `cycle_summary.json`
    claim reference embeddings were used while `intermediate/clusters.json`
    correctly said `sklearn_hashing_vectorizer`."""
    settings = Settings(
        mode="offline",
        input_path=Path("sample_runs/collector_experiment_sample/sample_run_posts.jsonl"),
        output_dir=tmp_path,
        history_dir=tmp_path / "history",
        llm_provider="heuristic",
        allow_heuristic_reports=True,
        enable_real_tools=False,
        embedding_backend="bge_large",
        trend_embedding_backend="minilm",
    )

    run_cycle(settings)

    summary = json.loads((tmp_path / "cycle_summary.json").read_text(encoding="utf-8"))
    resolution = summary["embedding_resolution"]
    assert resolution["requested_backend"] == "bge_large"
    assert resolution["requested_trend_backend"] == "minilm"
    assert resolution["resolved_backend"] == "hashing"
    assert resolution["resolved_trend_backend"] == "hashing"
    assert resolution["offline_pin_applied"] is True
    assert resolution["actual_backend"] == "sklearn_hashing_vectorizer"
    assert resolution["actual_trend_backend"] == "sklearn_hashing_vectorizer"
    assert resolution["embedding_fallback_applied"] is False
    # The requested values stay visible in hyperparameters, unchanged.
    assert summary["hyperparameters"]["embedding_backend"] == "bge_large"
    assert summary["hyperparameters"]["trend_embedding_backend"] == "minilm"


def test_run_cycle_reports_no_offline_pin_outside_offline_mode(tmp_path: Path):
    settings = Settings(
        mode="live",
        input_path=Path("sample_runs/collector_experiment_sample/sample_run_posts.jsonl"),
        output_dir=tmp_path,
        history_dir=tmp_path / "history",
        llm_provider="heuristic",
        allow_heuristic_reports=True,
        enable_real_tools=False,
        embedding_backend="hashing",
        trend_embedding_backend="hashing",
    )

    run_cycle(settings)

    resolution = json.loads((tmp_path / "cycle_summary.json").read_text(encoding="utf-8"))["embedding_resolution"]
    assert resolution["offline_pin_applied"] is False
    assert resolution["resolved_backend"] == "hashing"
    assert resolution["resolved_trend_backend"] == "hashing"


def test_run_cycle_records_a_runtime_embedding_fallback_not_just_the_offline_pin(tmp_path: Path, monkeypatch):
    """Outside offline mode there is no pin, so `resolved_backend` echoes the
    request. When the model itself cannot be loaded the run silently falls back
    to hashing -- `intermediate/clusters.json` records that, and the summary
    must not keep claiming the reference backend."""
    import tiblender_runtime.trends as trends_module

    class _UnavailableModel:
        def __init__(self, *args, **kwargs):
            raise OSError("no local model cache")

    # A real module object (not SimpleNamespace): services._resolve_embedding
    # calls importlib.util.find_spec, which needs __spec__ to be set.
    stand_in = types.ModuleType("sentence_transformers")
    stand_in.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", None)
    stand_in.SentenceTransformer = _UnavailableModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", stand_in)
    trends_module.load_trend_model.cache_clear()

    settings = Settings(
        mode="live",
        input_path=Path("sample_runs/collector_experiment_sample/sample_run_posts.jsonl"),
        output_dir=tmp_path,
        history_dir=tmp_path / "history",
        llm_provider="heuristic",
        allow_heuristic_reports=True,
        enable_real_tools=False,
        embedding_backend="bge_large",
        trend_embedding_backend="minilm",
    )

    try:
        run_cycle(settings)
    finally:
        trends_module.load_trend_model.cache_clear()

    summary = json.loads((tmp_path / "cycle_summary.json").read_text(encoding="utf-8"))
    resolution = summary["embedding_resolution"]
    assert resolution["offline_pin_applied"] is False
    # No pin, so the "resolved" (pin-only) view still echoes the request ...
    assert resolution["resolved_backend"] == "bge_large"
    assert resolution["resolved_trend_backend"] == "minilm"
    # ... but what the run actually embedded with was hashing, both times.
    assert "hashing" in resolution["actual_backend"]
    assert "fallback" in resolution["actual_backend"]
    assert "hashing" in resolution["actual_trend_backend"]
    assert "fallback" in resolution["actual_trend_backend"]
    assert resolution["embedding_fallback_applied"] is True

    clusters = json.loads((tmp_path / "intermediate" / "clusters.json").read_text(encoding="utf-8"))
    assert clusters, "expected at least one cluster to compare provenance against"
    assert clusters[0]["clustering_signals"]["embedding_backend"] == resolution["actual_backend"]
