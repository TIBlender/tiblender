"""Chunked LLM batch calls, truncation detection, and per-unit fallbacks.

Every LLM stage must degrade one unit at a time -- one chunk of posts, one
cluster's plan, one perspective, one judge round, one report's prose, one
trend signal -- instead of aborting the cycle, and every degradation must be
counted (``cycle_summary.json``'s ``llm_fallbacks``) and named on the
affected report (``metadata.llm_stage_fallbacks`` / ``generation_mode``).
"""

from __future__ import annotations

import json

import pytest

from llm_fallback_builders import (
    make_cluster,
    make_cluster_with_posts,
    make_decision,
    make_finding,
    make_posts,
    make_settings,
)
from tiblender_runtime.llm import ChatCompletion, LLMGenerationError, _extract_json_object
from tiblender_runtime.models import Post
from tiblender_runtime.llm_fallbacks import fallback_counters, reset_fallbacks, stages_for_cluster


@pytest.fixture(autouse=True)
def _clean_fallback_counters():
    reset_fallbacks()
    yield
    reset_fallbacks()


def _truncated_posts_document(posts: list[Post]) -> str:
    """A posts document cut off mid-object, as a length-capped provider sends."""
    body = ",".join(
        f'{{"id":"{post.id}","normalized_content":"Phishing kit reuse summary."}}' for post in posts[:1]
    )
    return '{"posts":[' + body + ',{"id":"' + posts[-1].id


# --- chunking -----------------------------------------------------------


def test_normalize_post_batch_chunks_and_recovers_from_a_truncated_reply(monkeypatch):
    from tiblender_runtime.llm import normalize_post_batch

    max_posts_per_reply = 2
    chunk_sizes: list[int] = []

    def fake_completion(settings, messages):
        del settings
        payload = json.loads(messages[1]["content"])
        posts = payload["posts"]
        chunk_sizes.append(len(posts))
        if len(posts) > max_posts_per_reply:
            # The provider stopped at its output cap: partial document.
            return ChatCompletion(
                content='{"posts":[{"id":"' + posts[0]["id"] + '","normalized_content":"Trunc',
                finish_reason="length",
            )
        return ChatCompletion(
            content=json.dumps(
                {
                    "posts": [
                        {"id": item["id"], "normalized_content": f"Phishing kit reuse for {item['id']}."}
                        for item in posts
                    ]
                }
            ),
            finish_reason="stop",
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion_with_meta", fake_completion)
    posts = make_posts(8)

    normalized = normalize_post_batch(make_settings(llm_batch_size=4), posts)

    assert sorted(normalized) == sorted(post.id for post in posts)
    # Two chunks of four, each halved into two chunks of two after truncation.
    assert chunk_sizes == [4, 2, 2, 4, 2, 2]
    assert fallback_counters()["normalization_posts"] == 0


def test_normalize_post_batch_falls_back_for_posts_no_chunk_size_can_normalize(monkeypatch):
    from tiblender_runtime.llm import normalize_post_batch

    def always_truncated(settings, messages):
        del settings
        posts = json.loads(messages[1]["content"])["posts"]
        return ChatCompletion(content=_truncated_posts_document(make_posts(len(posts))), finish_reason="length")

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion_with_meta", always_truncated)
    posts = make_posts(4)

    normalized = normalize_post_batch(make_settings(llm_batch_size=4), posts)

    assert normalized == {}
    assert fallback_counters()["normalization_posts"] == 4
    # The affected posts mark their cluster's report as partially generated.
    assert stages_for_cluster("cluster-test", [post.id for post in posts]) == ["post_normalization"]


def test_refine_clusters_keeps_unrefined_candidates_when_the_chunk_cannot_be_refined(monkeypatch):
    from tiblender_runtime.llm import refine_clusters_with_llm

    def always_truncated(settings, messages):
        del settings, messages
        return ChatCompletion(content='{"clusters":[{"topic_name":"Part', finish_reason="length")

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion_with_meta", always_truncated)
    clusters = [make_cluster("cluster-a")]

    refined = refine_clusters_with_llm(make_settings(llm_cluster_batch_size=10), clusters)

    assert [cluster.id for cluster in refined] == ["cluster-a"]
    assert fallback_counters()["cluster_refinement_batches"] == 1
    # Only the batch's own clusters are affected -- not the whole cycle.
    assert stages_for_cluster("cluster-a") == ["cluster_refinement"]
    assert stages_for_cluster("cluster-b") == []


# --- truncation detection and JSON repair -------------------------------


def test_chat_completion_with_meta_reports_length_finish_reason_as_truncation(monkeypatch):
    from tiblender_runtime.llm import _chat_completion_with_meta

    def fake_request_json(url, headers, body, timeout):
        del url, headers, body, timeout
        return {
            "choices": [{"finish_reason": "length", "message": {"content": '{"posts":[{"id":"p1"'}}],
            "usage": {"completion_tokens": 4096},
        }

    monkeypatch.setattr("tiblender_runtime.llm._request_json", fake_request_json)

    completion = _chat_completion_with_meta(make_settings(), [{"role": "user", "content": "x"}])

    assert completion.truncated is True
    assert completion.finish_reason == "length"
    assert completion.usage["completion_tokens"] == 4096


def test_chat_completion_with_meta_reports_a_complete_reply_as_untruncated(monkeypatch):
    from tiblender_runtime.llm import _chat_completion_with_meta

    monkeypatch.setattr(
        "tiblender_runtime.llm._request_json",
        lambda url, headers, body, timeout: {
            "choices": [{"finish_reason": "stop", "message": {"content": '{"posts":[]}'}}]
        },
    )

    completion = _chat_completion_with_meta(make_settings(), [{"role": "user", "content": "x"}])

    assert completion.truncated is False


def test_extract_json_object_repairs_a_document_truncated_mid_element():
    truncated = (
        '{"posts": [{"id": "x-1", "normalized_content": "First summary."}, '
        '{"id": "x-2", "normalized_content": "Second summary."}, {"id": "x-3", "normalized_'
    )

    parsed = _extract_json_object(truncated)

    # The complete elements survive; the partial one is dropped, not invented.
    assert [item["id"] for item in parsed["posts"]] == ["x-1", "x-2"]


def test_extract_json_object_repairs_missing_trailing_braces_only():
    parsed = _extract_json_object('{"posts": [{"id": "x-1", "normalized_content": "Only summary."}]')

    assert [item["id"] for item in parsed["posts"]] == ["x-1"]


def test_extract_json_object_rejects_prose_and_unrepairable_documents():
    with pytest.raises(LLMGenerationError):
        _extract_json_object("I cannot answer that request.")
    with pytest.raises(LLMGenerationError):
        # Nothing completed before the cut, so there is nothing to keep.
        _extract_json_object('{"posts": [{"id": "x-1", "normalized_')


# --- per-unit fallbacks -------------------------------------------------


def test_planner_falls_back_to_no_planned_queries_after_one_retry(monkeypatch):
    from tiblender_runtime.llm import plan_investigation_queries

    calls: list[int] = []

    def failing(settings, messages):
        del settings, messages
        calls.append(1)
        raise LLMGenerationError("planner draft is not JSON")

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion", failing)

    queries = plan_investigation_queries(make_settings(), make_cluster())

    assert queries == []
    assert len(calls) == 2  # one attempt plus exactly one retry
    assert fallback_counters()["planner_clusters"] == 1
    assert stages_for_cluster("cluster-test") == ["investigation_planning"]


def test_one_perspective_falls_back_without_affecting_the_others(monkeypatch):
    from tiblender_runtime.perspectives import select_focus_types

    def failing_for_infrastructure(settings, cluster, perspective):
        del settings, cluster
        if perspective == "Infrastructure":
            raise LLMGenerationError("focus draft is not JSON")
        return ["phishing_campaign_tracking", "social_engineering_tactics_analysis"]

    monkeypatch.setattr(
        "tiblender_runtime.perspectives.select_focus_types_with_llm",
        failing_for_infrastructure,
    )
    cluster = make_cluster()
    settings = make_settings()

    infrastructure = select_focus_types(cluster, "Infrastructure", hints=[], settings=settings)
    social = select_focus_types(cluster, "Social", hints=[], settings=settings)

    assert infrastructure  # deterministic focus ranking still produced focuses
    assert social == ["phishing_campaign_tracking", "social_engineering_tactics_analysis"]
    assert fallback_counters()["perspective_units"] == 1
    assert stages_for_cluster("cluster-test") == ["perspective_queries"]


def test_a_single_valid_focus_is_padded_and_not_counted_as_a_fallback(monkeypatch):
    from tiblender_runtime.perspectives import select_focus_types

    def one_valid_focus(settings, cluster, perspective):
        del settings, cluster
        assert perspective == "Social"
        # The LLM already dropped "not_a_real_focus_type"; only one valid
        # focus type survives that filtering in select_focus_types_with_llm.
        # This mirrors the real thin-Social/Actor-cluster case where the
        # heuristic ranking itself only offers one weak candidate.
        return ["phishing_campaign_tracking"]

    monkeypatch.setattr(
        "tiblender_runtime.perspectives.select_focus_types_with_llm",
        one_valid_focus,
    )
    cluster = make_cluster()
    settings = make_settings()

    focuses = select_focus_types(cluster, "Social", hints=[], settings=settings)

    assert len(focuses) == 2
    assert focuses[0] == "phishing_campaign_tracking"
    assert fallback_counters()["perspective_focus_padded"] == 1
    assert fallback_counters()["perspective_units"] == 0
    # Padding must never surface as a per-report fallback stage.
    assert stages_for_cluster("cluster-test") == []


def test_one_failing_judge_round_is_scored_deterministically_and_the_other_still_counts(monkeypatch):
    from tiblender_runtime.evaluation import evaluate_cluster

    def fake_chat(settings, messages):
        del messages
        if settings.llm_model == "judge-a-model":
            raise LLMGenerationError("judge A draft is not JSON")
        return json.dumps(
            {
                "verdict": "PROCEED",
                "quality_score": 0.9,
                "reason": "judge B accepts the evidence",
                "sufficiency": {"score": 0.9, "rationale": "Sufficient."},
                "consistency": {"score": 0.9, "rationale": "Consistent."},
                "evidence_quality": {"score": 0.9, "rationale": "High quality."},
                "follow_up_actions": [],
            }
        )

    monkeypatch.setattr("tiblender_runtime.evaluation._chat_completion", fake_chat)
    settings = make_settings(
        judge_a_provider="openai",
        judge_a_model="judge-a-model",
        judge_a_api_key="judge-a-key",
        judge_b_provider="openai",
        judge_b_model="judge-b-model",
        judge_b_api_key="judge-b-key",
    )

    decision = evaluate_cluster(make_cluster(), [make_finding()], max_rounds=2, settings=settings)

    judges = [round_item.judge for round_item in decision.rounds]
    assert "rule-judge-a" in judges  # the fallback round is labelled as deterministic
    assert "llm-judge-b" in judges  # the other judge still produced an LLM judgement
    assert fallback_counters()["judge_rounds"] >= 1
    assert stages_for_cluster("cluster-test") == ["evaluation_judges"]


def test_report_fields_fall_back_to_the_deterministic_composer(monkeypatch):
    from tiblender_runtime.llm import compose_report_fields

    calls: list[int] = []

    def failing(settings, messages):
        del settings, messages
        calls.append(1)
        raise LLMGenerationError("report draft is not valid JSON")

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion", failing)

    fields = compose_report_fields(make_settings(), make_cluster(), [make_finding()], make_decision())

    assert fields["executive_summary"]
    assert fields["immediate_actions"]
    assert len(calls) == 2
    assert fallback_counters()["report_fields"] == 1
    assert stages_for_cluster("cluster-test") == ["report_fields"]


def test_dynamic_query_generation_falls_back_per_trend_signal(monkeypatch):
    from tiblender_runtime import clock
    from tiblender_runtime.dynamic_queries import generate_dynamic_queries
    from tiblender_runtime.collectors.keywords import KeywordCatalog
    from tiblender_runtime.trends import TrendSignal, snapshot_cluster

    cluster = make_cluster()
    snapshot = snapshot_cluster(
        cluster,
        cycle_index=1,
        observed_at="2026-04-01T00:00:00Z",
        strategic_value=0.9,
        embed=lambda text: [0.0, 1.0],
    )
    signal = TrendSignal(
        cluster_id=cluster.id,
        trend_types=["EARLY_WARNING"],
        adjacent_history=1,
        evidence={},
    )

    def failing(settings, payload):
        del settings, payload
        raise LLMGenerationError("dynamic query draft is not JSON")

    monkeypatch.setattr("tiblender_runtime.llm_feedback.generate_dynamic_queries_with_llm", failing)

    queries, signals = generate_dynamic_queries(
        [signal],
        {cluster.id: snapshot},
        {cluster.id: cluster},
        {},
        settings=make_settings(),
        catalog=KeywordCatalog.load("config/keywords.json"),
        now=clock.now(),
    )

    assert queries  # the deterministic template path still produced queries
    assert signals[0].evidence.get("llm_fallback")
    assert fallback_counters()["dynamic_query_signals"] == 1


def test_report_metadata_marks_a_partially_generated_report(monkeypatch):
    from tiblender_runtime.llm_fallbacks import record_fallback
    from tiblender_runtime.reporting import build_report

    record_fallback("report_fields", cluster_id="cluster-test", category="json_parse", summary="test")
    stages = stages_for_cluster("cluster-test")

    report = build_report(
        make_cluster(),
        [make_finding()],
        make_decision(),
        report_fields={"metadata": {"generation_mode": "llm", "llm_used": True}},
        generation_mode="llm_partial",
        llm_stage_fallbacks=stages,
    )

    assert report["metadata"]["generation_mode"] == "llm_partial"
    assert report["metadata"]["llm_stage_fallbacks"] == ["report_fields"]


def test_report_metadata_stays_llm_when_no_stage_fell_back():
    from tiblender_runtime.reporting import build_report

    report = build_report(
        make_cluster(),
        [make_finding()],
        make_decision(),
        report_fields={"metadata": {"generation_mode": "llm", "llm_used": True}},
        generation_mode="llm",
        llm_stage_fallbacks=[],
    )

    assert report["metadata"]["generation_mode"] == "llm"
    assert report["metadata"]["llm_stage_fallbacks"] == []


# --- per-report attribution ---------------------------------------------


def test_one_failing_refinement_batch_marks_only_that_batchs_clusters(monkeypatch):
    """A single bad batch must not make every report of the cycle partial.

    Cluster refinement used to be recorded as a cycle-global stage, so one
    failing batch out of many downgraded every report's ``generation_mode``
    to ``llm_partial`` -- which is what the live three-cycle run observed
    (one batch fallback, ten reports all ``llm_partial``).
    """
    from tiblender_runtime.llm import refine_clusters_with_llm

    clusters = [make_cluster_with_posts(f"cluster-{index}", [f"p{index}"]) for index in range(1, 7)]
    failing_batch_ids = {"cluster-3", "cluster-4"}

    def refine_all_but_one_batch(settings, messages):
        del settings
        payload = json.loads(messages[1]["content"])
        batch_ids = {item["id"] for item in payload["candidate_clusters"]}
        if batch_ids & failing_batch_ids:
            return ChatCompletion(content='{"clusters":[{"topic_name":"Part', finish_reason="length")
        return ChatCompletion(
            content=json.dumps(
                {
                    "clusters": [
                        {
                            "topic_name": "Refined campaign",
                            "security_severity": "MEDIUM",
                            "topic_description": "Refined campaign cluster.",
                            "post_ids": [post["id"] for post in item["posts"]],
                        }
                        for item in payload["candidate_clusters"]
                    ]
                }
            ),
            finish_reason="stop",
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion_with_meta", refine_all_but_one_batch)

    refined = refine_clusters_with_llm(make_settings(llm_cluster_batch_size=2), clusters)

    # One of the three batches failed; the retry halved it and both halves
    # failed too, so two batch fallbacks are counted -- covering exactly the
    # two candidate clusters of that batch and nothing else.
    assert fallback_counters()["cluster_refinement_batches"] == 2
    assert {"cluster-3", "cluster-4"} <= {cluster.id for cluster in refined}
    partial = {
        cluster.id
        for cluster in clusters
        if stages_for_cluster(cluster.id, [post.id for post in cluster.posts])
    }
    assert partial == failing_batch_ids


def test_a_refinement_fallback_is_attributed_by_post_id_as_well_as_cluster_id(monkeypatch):
    """Attribution survives a cluster id that the pipeline later rewrites."""
    from tiblender_runtime.llm import refine_clusters_with_llm

    monkeypatch.setattr(
        "tiblender_runtime.llm._chat_completion_with_meta",
        lambda settings, messages: ChatCompletion(content='{"clusters":[{"topic', finish_reason="length"),
    )

    refine_clusters_with_llm(make_settings(llm_cluster_batch_size=4), [make_cluster_with_posts("cluster-a", ["p1", "p2"])])

    assert stages_for_cluster("some-other-id", ["p2"]) == ["cluster_refinement"]
    assert stages_for_cluster("some-other-id", ["p9"]) == []


def test_a_normalization_chunk_fallback_marks_only_the_clusters_holding_its_posts(monkeypatch):
    from tiblender_runtime.llm import normalize_post_batch

    posts = make_posts(4)

    def fail_only_the_second_chunk(settings, messages):
        del settings
        chunk = json.loads(messages[1]["content"])["posts"]
        if any(item["id"] in {"p3", "p4"} for item in chunk):
            return ChatCompletion(content='{"posts":[{"id":"p3","normalized_', finish_reason="length")
        return ChatCompletion(
            content=json.dumps(
                {"posts": [{"id": item["id"], "normalized_content": "Summary."} for item in chunk]}
            ),
            finish_reason="stop",
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion_with_meta", fail_only_the_second_chunk)

    normalize_post_batch(make_settings(llm_batch_size=2), posts)

    assert stages_for_cluster("cluster-early", ["p1", "p2"]) == []
    assert stages_for_cluster("cluster-late", ["p3", "p4"]) == ["post_normalization"]


def test_a_planner_fallback_never_reaches_another_cluster(monkeypatch):
    from tiblender_runtime.llm import plan_investigation_queries

    monkeypatch.setattr(
        "tiblender_runtime.llm._chat_completion",
        lambda settings, messages: (_ for _ in ()).throw(LLMGenerationError("planner draft is not JSON")),
    )

    plan_investigation_queries(make_settings(), make_cluster("cluster-one"))

    assert stages_for_cluster("cluster-one") == ["investigation_planning"]
    assert stages_for_cluster("cluster-two") == []


def test_stage_names_come_back_in_pipeline_order_whatever_order_they_failed_in():
    from tiblender_runtime.llm_fallbacks import record_fallback

    record_fallback("report_fields", cluster_id="c1", category="json_parse", summary="report draft")
    record_fallback("normalization_posts", post_ids=["p1"], category="truncated", summary="chunk")
    record_fallback("planner_clusters", cluster_id="c1", category="json_parse", summary="planner draft")

    assert stages_for_cluster("c1", ["p1"]) == [
        "post_normalization",
        "investigation_planning",
        "report_fields",
    ]
