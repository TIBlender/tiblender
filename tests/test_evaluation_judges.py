import json

from tiblender_runtime.context import InvestigationContext, IocEvidence, aggregate_context
from tiblender_runtime.evaluation import evaluate_cluster
from tiblender_runtime.models import (
    Cluster,
    EvaluationVerdict,
    IocSet,
    NormalizedPost,
    PerspectiveFinding,
    Post,
    Severity,
    ThreatCategory,
)
from tiblender_runtime.contract import EVALUATION_PROMPT
from tiblender_runtime.settings import Settings


def _cluster() -> Cluster:
    post = NormalizedPost(
        post=Post(id="p1", platform="x", content="CVE-2026-12345 exploitation at bad.example"),
        normalized_content="CVE-2026-12345 exploitation campaign bad.example",
        iocs=IocSet(domains=["bad.example"], cves=["CVE-2026-12345"]),
        threat_category=ThreatCategory.VULNERABILITY,
    )
    return Cluster(
        id="cluster-test",
        topic_name="CVE-2026-12345 exploitation campaign",
        description="test cluster",
        posts=[post],
        iocs=post.iocs,
        severity=Severity.HIGH,
        threat_category=ThreatCategory.VULNERABILITY,
    )


def _finding() -> PerspectiveFinding:
    return PerspectiveFinding(
        perspective="Infrastructure",
        focus_types=["domain_registration_and_whois"],
        queries=[],
        tool_results=[],
        insights=["WHOIS and CT should be checked."],
        confidence=0.7,
    )


def test_evaluation_uses_two_independent_llm_judges(monkeypatch):
    prompts = []

    def fake_chat(settings, messages):
        prompts.append((settings.llm_model, messages[0]["content"]))
        return json.dumps(
            {
                "verdict": "PROCEED",
                "quality_score": 0.91,
                "reason": f"{settings.llm_model} accepts the evidence",
                "requested_followups": [],
            }
        )

    monkeypatch.setattr("tiblender_runtime.evaluation._chat_completion", fake_chat)
    settings = Settings(
        llm_provider="openai",
        llm_model="report-model",
        openai_api_key="report-key",
        judge_a_provider="openai",
        judge_a_model="judge-a-model",
        judge_a_api_key="judge-a-key",
        judge_b_provider="openai",
        judge_b_model="judge-b-model",
        judge_b_api_key="judge-b-key",
    )

    decision = evaluate_cluster(_cluster(), [_finding()], max_rounds=3, settings=settings)

    assert decision.verdict is EvaluationVerdict.PROCEED
    assert {round_item.judge for round_item in decision.rounds} == {"llm-judge-a", "llm-judge-b"}
    assert {model for model, _prompt in prompts} == {"judge-a-model", "judge-b-model"}
    assert all(EVALUATION_PROMPT in prompt for _model, prompt in prompts)
    assert all("Evaluate independently" in prompt for _model, prompt in prompts)
    assert all("sufficiency" in prompt for _model, prompt in prompts)
    assert all("follow_up_actions" in prompt for _model, prompt in prompts)
    assert all("materially change whether the report is actionable" in prompt for _model, prompt in prompts)
    assert all("could improve evidence" not in prompt for _model, prompt in prompts)


def test_evaluation_records_structured_axes_and_followup_actions(monkeypatch):
    responses = iter(
        [
            {
                "verdict": "NEED_MORE_INFO",
                "quality_score": 0.52,
                "reason": "Need domain registration evidence.",
                "sufficiency": {"score": 0.45, "rationale": "Infrastructure evidence is incomplete."},
                "consistency": {"score": 0.70, "rationale": "Source posts agree on the same domain."},
                "evidence_quality": {"score": 0.50, "rationale": "The IoC needs validation."},
                "follow_up_actions": [
                    {
                        "perspective": "Infrastructure",
                        "focus_types": ["domain_registration_and_whois"],
                        "tools": ["WHOIS_LOOKUP"],
                        "rationale": "Validate bad.example registration.",
                    }
                ],
            },
            {
                "verdict": "PROCEED",
                "quality_score": 0.86,
                "reason": "Enough",
                "sufficiency": {"score": 0.86, "rationale": "Sufficient."},
                "consistency": {"score": 0.88, "rationale": "Consistent."},
                "evidence_quality": {"score": 0.84, "rationale": "High quality."},
                "follow_up_actions": [],
            },
            {
                "verdict": "PROCEED",
                "quality_score": 0.88,
                "reason": "Enough",
                "sufficiency": {"score": 0.86, "rationale": "Sufficient."},
                "consistency": {"score": 0.88, "rationale": "Consistent."},
                "evidence_quality": {"score": 0.84, "rationale": "High quality."},
                "follow_up_actions": [],
            },
            {
                "verdict": "PROCEED",
                "quality_score": 0.89,
                "reason": "Enough",
                "sufficiency": {"score": 0.86, "rationale": "Sufficient."},
                "consistency": {"score": 0.88, "rationale": "Consistent."},
                "evidence_quality": {"score": 0.84, "rationale": "High quality."},
                "follow_up_actions": [],
            },
        ]
    )

    def fake_chat(settings, messages):
        del settings, messages
        return json.dumps(next(responses))

    monkeypatch.setattr("tiblender_runtime.evaluation._chat_completion", fake_chat)
    settings = Settings(llm_provider="openai", llm_model="main", openai_api_key="key")
    followup_calls = []

    def followup_executor(requested_followups, round_index):
        followup_calls.append((requested_followups, round_index))
        return [_finding()]

    decision = evaluate_cluster(
        _cluster(),
        [_finding()],
        max_rounds=2,
        settings=settings,
        followup_executor=followup_executor,
    )

    first_round = decision.rounds[0]
    assert first_round.sufficiency["rationale"] == "Infrastructure evidence is incomplete."
    assert first_round.consistency["score"] == 0.7
    assert first_round.evidence_quality["rationale"] == "The IoC needs validation."
    assert first_round.follow_up_actions == [
        {
            "perspective": "Infrastructure",
            "focus_types": ["domain_registration_and_whois"],
            "tools": ["WHOIS_LOOKUP"],
            "rationale": "Validate bad.example registration.",
        }
    ]
    assert first_round.requested_followups == ["Infrastructure WHOIS_LOOKUP domain_registration_and_whois: Validate bad.example registration."]
    assert followup_calls == [(["Infrastructure WHOIS_LOOKUP domain_registration_and_whois: Validate bad.example registration."], 1)]


def test_evaluation_conservatively_requests_followup_before_proceeding(monkeypatch):
    responses = iter(
        [
            {"verdict": "NEED_MORE_INFO", "quality_score": 0.52, "reason": "Need WHOIS", "requested_followups": ["WHOIS_LOOKUP bad.example"]},
            {"verdict": "PROCEED", "quality_score": 0.86, "reason": "Enough", "requested_followups": []},
            {"verdict": "PROCEED", "quality_score": 0.88, "reason": "Enough", "requested_followups": []},
            {"verdict": "PROCEED", "quality_score": 0.89, "reason": "Enough", "requested_followups": []},
        ]
    )

    def fake_chat(settings, messages):
        del settings, messages
        return json.dumps(next(responses))

    monkeypatch.setattr("tiblender_runtime.evaluation._chat_completion", fake_chat)
    settings = Settings(llm_provider="openai", llm_model="main", openai_api_key="key")

    followup_calls = []

    def followup_executor(requested_followups, round_index):
        followup_calls.append((requested_followups, round_index))
        return [_finding()]

    decision = evaluate_cluster(
        _cluster(),
        [_finding()],
        max_rounds=2,
        settings=settings,
        followup_executor=followup_executor,
    )

    assert decision.verdict is EvaluationVerdict.PROCEED
    assert any(round_item.verdict is EvaluationVerdict.NEED_MORE_INFO for round_item in decision.rounds)
    assert decision.rounds[0].requested_followups == ["WHOIS_LOOKUP bad.example"]
    assert followup_calls == [(["WHOIS_LOOKUP bad.example"], 1)]


def test_evaluation_skips_when_max_rounds_end_without_consensus(monkeypatch):
    def fake_chat(settings, messages):
        del settings, messages
        return json.dumps(
            {
                "verdict": "NEED_MORE_INFO",
                "quality_score": 0.99,
                "reason": "Still incomplete",
                "requested_followups": ["WEB_CRAWL bad.example"],
            }
        )

    monkeypatch.setattr("tiblender_runtime.evaluation._chat_completion", fake_chat)
    settings = Settings(llm_provider="openai", llm_model="main", openai_api_key="key")

    decision = evaluate_cluster(_cluster(), [_finding()], max_rounds=1, settings=settings)

    assert decision.verdict is EvaluationVerdict.SKIP
    assert decision.final_quality_score < 0.80
    assert "maximum evaluation rounds" in decision.reason.lower()


def test_judge_prompt_includes_the_aggregated_investigation_context(monkeypatch):
    payloads = []

    def fake_chat(settings, messages):
        del settings
        payloads.append(json.loads(messages[1]["content"]))
        return json.dumps({"verdict": "PROCEED", "quality_score": 0.9, "reason": "ok", "requested_followups": []})

    monkeypatch.setattr("tiblender_runtime.evaluation._chat_completion", fake_chat)
    settings = Settings(llm_provider="openai", llm_model="main", openai_api_key="key")
    cluster = _cluster()
    context = aggregate_context(cluster, [_finding()])

    evaluate_cluster(cluster, [_finding()], max_rounds=1, settings=settings, context=context)

    assert payloads
    investigation_context = payloads[0]["investigation_context"]
    assert set(investigation_context) == {"iocs", "cves", "ttps", "per_perspective", "stats", "filtered_benign"}
    assert investigation_context["cves"] == ["CVE-2026-12345"]
    assert investigation_context["stats"]["ioc_total"] == 2


def _context_with_stats(ioc_high: int) -> InvestigationContext:
    return InvestigationContext(
        cluster_id="cluster-test",
        iocs=[
            IocEvidence(
                value=f"scored-{index}.example",
                ioc_type="domain",
                extraction_type="Direct",
                confidence="HIGH" if index < ioc_high else "LOW",
            )
            for index in range(3)
        ],
        stats={
            "ioc_total": 3,
            "ioc_high": ioc_high,
            "ioc_medium": 0,
            "ioc_low": 3 - ioc_high,
            "pivot_total": 0,
            "direct_total": 3,
            "tool_success_rate": 0.0,
            "perspectives_active": 0,
        },
    )


def test_heuristic_judge_score_rises_with_confirmed_indicators():
    cluster = _cluster()

    weak = evaluate_cluster(cluster, [_finding()], max_rounds=1, context=_context_with_stats(0))
    strong = evaluate_cluster(cluster, [_finding()], max_rounds=1, context=_context_with_stats(3))
    without_context = evaluate_cluster(cluster, [_finding()], max_rounds=1)

    assert strong.final_quality_score > weak.final_quality_score
    assert without_context.final_quality_score != strong.final_quality_score
    assert strong.rounds[0].quality_score == strong.final_quality_score
    assert 0.0 <= weak.final_quality_score <= 1.0


def test_a_cluster_with_no_indicator_cve_or_ttp_is_never_reported():
    """Paper §III-F / §IV-F: a report has to describe an actionable threat.

    The offline evidence score reaches 0.65 on platform diversity, tool
    reliability and severity alone, so a cluster whose Step 3 scoring left no
    indicator, CVE or TTP used to clear the 0.62 bar and publish an empty
    report. Actionability is now a precondition, not a score term.
    """
    cluster = _cluster()
    empty = InvestigationContext(cluster_id="cluster-test", stats={"ioc_total": 0, "ioc_high": 0})

    decision = evaluate_cluster(cluster, [_finding()], max_rounds=1, context=empty)

    assert decision.verdict is EvaluationVerdict.SKIP
    assert "actionable" in decision.reason.lower()


def test_the_actionability_gate_accepts_an_indicator_a_cve_or_a_ttp():
    """The gate asks for an indicator *or* a CVE *or* a TTP, not all three."""
    from tiblender_runtime.evaluation import _is_actionable

    empty = InvestigationContext(cluster_id="c")

    assert not _is_actionable(empty)
    assert _is_actionable(InvestigationContext(cluster_id="c", ttps=["T1566"]))
    assert _is_actionable(InvestigationContext(cluster_id="c", cves=["CVE-2026-12345"]))
    assert _is_actionable(
        InvestigationContext(
            cluster_id="c",
            iocs=[IocEvidence(value="bad.example", ioc_type="domain", extraction_type="Direct", confidence="LOW")],
        )
    )
    # No context at all (a report built straight from findings) is not the
    # gate's business: it has no scored evidence list to judge.
    assert _is_actionable(None)
