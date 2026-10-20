"""Adaptive evidence evaluation with independent judges and follow-up loops."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from typing import Any, Callable

from .context import InvestigationContext, context_to_judge_payload
from .llm import LLMGenerationError, _chat_completion, _clean_actions, _extract_json_object, _sanitize_report_text
from .llm_fallbacks import category_of, record_fallback
from .models import Cluster, EvaluationDecision, EvaluationRound, EvaluationVerdict, PerspectiveFinding, Severity, to_jsonable
from .contract import EVALUATION_PROMPT
from .settings import Settings

FollowupExecutor = Callable[[list[str], int], list[PerspectiveFinding]]
ContextRefresh = Callable[[list[PerspectiveFinding]], InvestigationContext]
_ALLOWED_FOLLOWUP_TOOLS = {
    "WHOIS_LOOKUP",
    "PASSIVE_DNS_LOOKUP",
    "TLS_CT_LOOKUP",
    "WEB_CRAWL",
    "NVD_CVE_LOOKUP",
    "GITHUB_SEARCH",
    "X_SEARCH",
    "REDDIT_SEARCH",
    "DISCORD_SEARCH",
    "TELEGRAM_SEARCH",
    "SEARCH_ENGINE",
}
_ALLOWED_PERSPECTIVES = {"Infrastructure", "Technical", "Social", "Actor"}


def _context_evidence_score(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    context: InvestigationContext,
) -> float:
    # With an aggregated context the offline judge scores the evidence the LLM
    # judge also sees: confirmed indicators, indicator volume, tool reliability,
    # and perspective coverage. Independent platforms and severity stay in the
    # rubric so a fixture/offline cycle - where no live tool can succeed - is
    # still scored on the source evidence it does have.
    stats = context.stats
    platforms = {post.platform for post in cluster.posts}
    attempted = sum(1 for finding in findings if finding.tool_results)
    score = 0.20
    score += min(0.25, 0.08 * len(platforms))
    score += 0.20 * min(1.0, stats.get("ioc_high", 0) / 3)
    score += 0.15 * min(1.0, stats.get("ioc_total", 0) / 5)
    score += 0.10 * float(stats.get("tool_success_rate", 0.0))
    score += 0.05 * min(1.0, stats.get("perspectives_active", 0) / 4)
    score += 0.05 * min(1.0, attempted / 4)
    if cluster.severity in {Severity.HIGH, Severity.CRITICAL}:
        score += 0.10
    return round(min(score, 1.0), 3)


def _is_actionable(context: InvestigationContext | None) -> bool:
    """True when Step 3 left the cluster something a defender can act on.

    The paper's report-level false-positive definition is "the report does not
    describe an actionable threat", and §III-F's report carries indicators, CVEs
    and TTPs. A cluster whose scored evidence holds none of the three has
    nothing actionable to publish, however well it scores on platform
    diversity or tool reliability, so it is suppressed rather than reported.
    """
    if context is None:
        return True
    return bool(context.iocs or context.cves or context.ttps)


def _evidence_score(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    context: InvestigationContext | None = None,
) -> float:
    # The offline judge uses the same signals as the LLM judge input so smoke
    # runs exercise the control flow without pretending to be a full LLM review.
    if context is not None:
        return _context_evidence_score(cluster, findings, context)
    platforms = {post.platform for post in cluster.posts}
    ioc_count = len(cluster.iocs.all_values())
    perspective_count = sum(1 for finding in findings if finding.tool_results)
    score = 0.20
    score += min(0.25, 0.08 * len(platforms))
    score += min(0.25, 0.04 * ioc_count)
    score += min(0.20, 0.05 * perspective_count)
    if cluster.severity in {Severity.HIGH, Severity.CRITICAL}:
        score += 0.10
    return min(score, 1.0)


def _judge_messages(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    round_index: int,
    judge_label: str,
    context: InvestigationContext | None = None,
) -> list[dict[str, str]]:
    # The judge sees only normalized cluster evidence and tool summaries. Raw
    # provider payloads stay out of the prompt to keep decisions evidence-bound.
    payload = {
        "round_index": round_index,
        "cluster": {
            "id": cluster.id,
            "topic_name": cluster.topic_name,
            "severity": cluster.severity.value,
            "threat_category": cluster.threat_category.value,
            "iocs": to_jsonable(cluster.iocs),
            "platforms": sorted({post.platform for post in cluster.posts}),
            "post_count": len(cluster.posts),
        },
        "findings": [
            {
                "perspective": finding.perspective,
                "focus_types": finding.focus_types,
                "insights": finding.insights,
                "tool_results": [
                    {
                        "tool_type": result.tool_type,
                        "status": result.status,
                        "summary": result.summary[:800],
                        "discovered_iocs": to_jsonable(result.discovered_iocs),
                    }
                    for result in finding.tool_results
                ],
            }
            for finding in findings
        ],
    }
    if context is not None:
        payload["investigation_context"] = context_to_judge_payload(context)
    return [
        {
            "role": "system",
            "content": (
                EVALUATION_PROMPT
                + "\n\n"
                f"Judge label: {judge_label}. Evaluate independently without consulting another judge. "
                "Return strict JSON only with keys: verdict, quality_score, reason, sufficiency, consistency, "
                "evidence_quality, follow_up_actions. "
                "verdict must be PROCEED, NEED_MORE_INFO, or SKIP. "
                "sufficiency, consistency, and evidence_quality must each be objects with score and rationale. "
                "Use NEED_MORE_INFO only when a configured follow-up would materially change whether the report is actionable. "
                "Do not request follow-up solely because attribution, passive DNS, or payload details might be further enriched "
                "when direct IoCs, impact, and mitigation evidence are already consistent. "
                "follow_up_actions must be an array of objects with perspective, focus_types, tools, and rationale. "
                "tools may name only configured tool types."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _clean_axis(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"score": 0.0, "rationale": "Judge did not provide this axis assessment."}
    try:
        score = float(raw.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    rationale = _sanitize_report_text(str(raw.get("rationale", "")).strip()) or "Judge did not provide a rationale."
    return {"score": max(0.0, min(1.0, round(score, 3))), "rationale": rationale}


def _clean_follow_up_actions(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    actions: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        perspective = str(item.get("perspective", "")).strip()
        if perspective not in _ALLOWED_PERSPECTIVES:
            continue
        raw_focus = item.get("focus_types", [])
        if isinstance(raw_focus, str):
            raw_focus = [raw_focus]
        focus_types = [str(value).strip() for value in raw_focus if str(value).strip()] if isinstance(raw_focus, list) else []
        raw_tools = item.get("tools", [])
        if isinstance(raw_tools, str):
            raw_tools = [raw_tools]
        tools = []
        if isinstance(raw_tools, list):
            tools = [
                str(value).strip().upper()
                for value in raw_tools
                if str(value).strip().upper() in _ALLOWED_FOLLOWUP_TOOLS
            ]
        rationale = _sanitize_report_text(str(item.get("rationale", "")).strip())
        if perspective and tools and rationale:
            actions.append(
                {
                    "perspective": perspective,
                    "focus_types": focus_types,
                    "tools": tools,
                    "rationale": rationale,
                }
            )
    return actions[:8]


def _followup_strings_from_actions(actions: list[dict[str, Any]]) -> list[str]:
    followups: list[str] = []
    for action in actions:
        parts = [
            str(action.get("perspective", "")).strip(),
            " ".join(str(tool).strip().upper() for tool in action.get("tools", []) if str(tool).strip()),
            " ".join(str(focus).strip() for focus in action.get("focus_types", []) if str(focus).strip()),
        ]
        prefix = " ".join(part for part in parts if part)
        rationale = str(action.get("rationale", "")).strip()
        text = f"{prefix}: {rationale}" if prefix else rationale
        if text and text not in followups:
            followups.append(text)
    return followups


def _legacy_followup_actions(followups: list[str]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for followup in followups:
        upper = followup.upper()
        tools = [tool for tool in _ALLOWED_FOLLOWUP_TOOLS if tool in upper]
        perspective = "Actor"
        if any(tool in {"WHOIS_LOOKUP", "PASSIVE_DNS_LOOKUP", "TLS_CT_LOOKUP"} for tool in tools):
            perspective = "Infrastructure"
        elif any(tool in {"WEB_CRAWL", "NVD_CVE_LOOKUP", "GITHUB_SEARCH"} for tool in tools):
            perspective = "Technical"
        elif any(tool in {"X_SEARCH", "REDDIT_SEARCH", "DISCORD_SEARCH", "TELEGRAM_SEARCH"} for tool in tools):
            perspective = "Social"
        actions.append(
            {
                "perspective": perspective,
                "focus_types": [],
                "tools": tools or ["SEARCH_ENGINE"],
                "rationale": followup,
            }
        )
    return actions


def _normalise_judge_response(raw: dict[str, Any]) -> tuple[EvaluationVerdict, float, str, list[str], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    # Clamp malformed LLM output back into the three-state contract so the
    # workflow can proceed deterministically or fail only on transport errors.
    verdict_text = str(raw.get("verdict", "NEED_MORE_INFO")).strip().upper()
    if verdict_text not in {item.value for item in EvaluationVerdict}:
        verdict_text = "NEED_MORE_INFO"
    try:
        score = float(raw.get("quality_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    followups = raw.get("requested_followups", [])
    if isinstance(followups, str):
        followups = [followups]
    if not isinstance(followups, list):
        followups = []
    structured_followups = _clean_follow_up_actions(raw.get("follow_up_actions"))
    cleaned_followups = _followup_strings_from_actions(structured_followups) or _clean_actions(followups)
    if not structured_followups and cleaned_followups:
        structured_followups = _legacy_followup_actions(cleaned_followups)
    return (
        EvaluationVerdict(verdict_text),
        max(0.0, min(1.0, score)),
        _sanitize_report_text(str(raw.get("reason", "")).strip()) or "Judge did not provide a reason.",
        cleaned_followups,
        _clean_axis(raw.get("sufficiency")),
        _clean_axis(raw.get("consistency")),
        _clean_axis(raw.get("evidence_quality")),
        structured_followups,
    )


def _heuristic_evaluate_cluster(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    max_rounds: int,
    followup_executor: FollowupExecutor | None = None,
    context: InvestigationContext | None = None,
    context_refresh: ContextRefresh | None = None,
) -> EvaluationDecision:
    rounds: list[EvaluationRound] = []
    score = _evidence_score(cluster, findings, context)

    for index in range(1, max_rounds + 1):
        if cluster.severity is Severity.LOW:
            verdict = EvaluationVerdict.SKIP
            reason = "Cluster is below the configured severity threshold and lacks actionable evidence."
        elif not _is_actionable(context):
            # No indicator, CVE, or TTP survived Step 3: there is nothing to
            # publish. Follow-ups still get their rounds to find one.
            if index < max_rounds:
                verdict = EvaluationVerdict.NEED_MORE_INFO
                reason = "No indicator, CVE, or TTP survived evidence scoring; targeted follow-up is required."
            else:
                verdict = EvaluationVerdict.SKIP
                reason = "No indicator, CVE, or TTP survived evidence scoring, so the cluster has nothing actionable to report."
        elif score >= 0.62:
            verdict = EvaluationVerdict.PROCEED
            reason = "Independent source, IoC, and perspective evidence are sufficient for a structured TI report."
        elif index < max_rounds:
            verdict = EvaluationVerdict.NEED_MORE_INFO
            reason = "Evidence is plausible but needs additional corroboration before reporting."
            score += 0.08
        else:
            verdict = EvaluationVerdict.SKIP
            reason = "Evidence remained insufficient after the configured evaluation loop."

        rounds.append(
            EvaluationRound(
                round_index=index,
                judge="rule-judge-a" if index % 2 else "rule-judge-b",
                verdict=verdict,
                quality_score=round(score, 3),
                reason=reason,
                requested_followups=[] if verdict is not EvaluationVerdict.NEED_MORE_INFO else ["collect_more_independent_sources"],
                sufficiency={
                    "score": round(min(score + 0.05, 1.0), 3),
                    "rationale": "Source and indicator volume were scored with the local smoke-test rubric.",
                },
                consistency={
                    "score": round(score, 3),
                    "rationale": "Smoke-test consistency is estimated from platform count and available perspective evidence.",
                },
                evidence_quality={
                    "score": round(max(score - 0.05, 0.0), 3),
                    "rationale": "Smoke-test evidence quality is estimated from IoC count and successful perspective coverage.",
                },
                follow_up_actions=[] if verdict is not EvaluationVerdict.NEED_MORE_INFO else [
                    {
                        "perspective": "Social",
                        "focus_types": ["campaign_tracking"],
                        "tools": ["SEARCH_ENGINE"],
                        "rationale": "Collect more independent corroborating sources.",
                    }
                ],
            )
        )
        if verdict in {EvaluationVerdict.PROCEED, EvaluationVerdict.SKIP}:
            return EvaluationDecision(verdict=verdict, rounds=rounds, final_quality_score=round(score, 3), reason=reason)
        if followup_executor is not None and index < max_rounds:
            findings.extend(followup_executor(["collect_more_independent_sources"], index))
            if context_refresh is not None:
                context = context_refresh(findings)
            score = _evidence_score(cluster, findings, context)

    final = rounds[-1]
    return EvaluationDecision(
        verdict=final.verdict,
        rounds=rounds,
        final_quality_score=final.quality_score,
        reason=final.reason,
    )


JudgeOutput = tuple[
    EvaluationVerdict,
    float,
    str,
    list[str],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]


def _heuristic_judge_output(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    context: InvestigationContext | None,
    reason_suffix: str,
) -> JudgeOutput:
    # Deterministic stand-in for one judge whose LLM call could not be
    # completed. It scores the same evidence the LLM judge is shown, so a
    # single failed judge round degrades that round instead of the cycle.
    score = _evidence_score(cluster, findings, context)
    if cluster.severity is Severity.LOW:
        verdict = EvaluationVerdict.SKIP
        reason = "Cluster is below the configured severity threshold and lacks actionable evidence."
    elif score >= 0.62:
        verdict = EvaluationVerdict.PROCEED
        reason = "Independent source, IoC, and perspective evidence are sufficient for a structured TI report."
    else:
        verdict = EvaluationVerdict.NEED_MORE_INFO
        reason = "Evidence is plausible but needs additional corroboration before reporting."
    rubric = "Scored with the deterministic rubric because the judge LLM call did not return a usable draft."
    return (
        verdict,
        round(score, 3),
        f"{reason} {reason_suffix}",
        [],
        {"score": round(min(score + 0.05, 1.0), 3), "rationale": rubric},
        {"score": round(score, 3), "rationale": rubric},
        {"score": round(max(score - 0.05, 0.0), 3), "rationale": rubric},
        [],
    )


def _judge_round(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    *,
    round_index: int,
    label: str,
    settings: Settings,
    context: InvestigationContext | None,
) -> tuple[str, JudgeOutput]:
    """Run one judge for one round: retry once, then score deterministically.

    Returns the judge name to record alongside its output so a fallback round
    is visible in ``evaluation_rounds`` as ``rule-judge-<label>`` rather than
    being mistaken for an LLM judgement. The other judge of the round is
    unaffected and still counts toward the verdict.
    """
    judge_settings = settings.for_judge(label)
    messages = _judge_messages(cluster, findings, round_index, label.upper(), context)
    errors: list[LLMGenerationError] = []
    for _attempt in range(2):
        try:
            content = _chat_completion(judge_settings, messages)
            return f"llm-judge-{label}", _normalise_judge_response(_extract_json_object(content))
        except LLMGenerationError as error:
            errors.append(error)
    record_fallback(
        "judge_rounds",
        cluster_id=cluster.id,
        category=category_of(errors[-1] if errors else None),
        summary=f"judge {label} round {round_index} rejected after one retry",
        detail=" / retry: ".join(str(error) for error in errors),
    )
    return (
        f"rule-judge-{label}",
        _heuristic_judge_output(cluster, findings, context, "Judge LLM unavailable for this round."),
    )


def _llm_evaluate_cluster(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    *,
    max_rounds: int,
    settings: Settings,
    followup_executor: FollowupExecutor | None = None,
    context: InvestigationContext | None = None,
    context_refresh: ContextRefresh | None = None,
) -> EvaluationDecision:
    # Each round asks two judge configurations independently. A single SKIP is
    # conservative enough to suppress publication for the cluster.
    rounds: list[EvaluationRound] = []
    final_reason = ""
    for round_index in range(1, max_rounds + 1):
        judge_outputs = []
        # The two judges of a round are independent, so they are asked
        # concurrently; results are still consumed in ("a", "b") order.
        with ThreadPoolExecutor(max_workers=2) as pool:
            judge_futures = [
                pool.submit(
                    _judge_round,
                    cluster,
                    findings,
                    round_index=round_index,
                    label=label,
                    settings=settings,
                    context=context,
                )
                for label in ("a", "b")
            ]
        for judge_future in judge_futures:
            judge_name, judge_output = judge_future.result()
            verdict, score, reason, followups, sufficiency, consistency, evidence_quality, follow_up_actions = judge_output
            judge_outputs.append(judge_output)
            rounds.append(
                EvaluationRound(
                    round_index=round_index,
                    judge=judge_name,
                    verdict=verdict,
                    quality_score=round(score, 3),
                    reason=reason,
                    requested_followups=followups,
                    sufficiency=sufficiency,
                    consistency=consistency,
                    evidence_quality=evidence_quality,
                    follow_up_actions=follow_up_actions,
                )
            )

        verdicts = [item[0] for item in judge_outputs]
        scores = [item[1] for item in judge_outputs]
        reasons = [item[2] for item in judge_outputs]
        final_score = round(mean(scores), 3)
        final_reason = " / ".join(reasons)
        if EvaluationVerdict.SKIP in verdicts:
            return EvaluationDecision(EvaluationVerdict.SKIP, rounds, final_score, final_reason)
        if all(verdict is EvaluationVerdict.PROCEED for verdict in verdicts):
            if _is_actionable(context):
                return EvaluationDecision(EvaluationVerdict.PROCEED, rounds, final_score, final_reason)
            # Both judges agreed, but Step 3 scored no indicator, CVE, or TTP:
            # there is no actionable content for a report to carry.
            return EvaluationDecision(
                EvaluationVerdict.SKIP,
                rounds,
                min(final_score, 0.74),
                f"No indicator, CVE, or TTP survived evidence scoring; suppressing report. {final_reason}",
            )

        requested_followups = sorted({item for _verdict, _score, _reason, followups, _sufficiency, _consistency, _evidence_quality, _follow_up_actions in judge_outputs for item in followups})
        if requested_followups and followup_executor is not None and round_index < max_rounds:
            findings.extend(followup_executor(requested_followups, round_index))
            if context_refresh is not None:
                context = context_refresh(findings)

        if round_index == max_rounds:
            capped_score = min(final_score, 0.74)
            return EvaluationDecision(
                EvaluationVerdict.SKIP,
                rounds,
                capped_score,
                f"Maximum evaluation rounds reached without conservative consensus; suppressing report. {final_reason}",
            )
    return EvaluationDecision(EvaluationVerdict.SKIP, rounds, rounds[-1].quality_score if rounds else 0.0, final_reason)


def evaluate_cluster(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    *,
    max_rounds: int = 3,
    settings: Settings | None = None,
    followup_executor: FollowupExecutor | None = None,
    context: InvestigationContext | None = None,
    context_refresh: ContextRefresh | None = None,
) -> EvaluationDecision:
    # Normal runs require two configured LLM judges. The heuristic path exists
    # only for local smoke tests and exercises the same follow-up control flow.
    def heuristic() -> EvaluationDecision:
        return _heuristic_evaluate_cluster(
            cluster,
            findings,
            max_rounds,
            followup_executor=followup_executor,
            context=context,
            context_refresh=context_refresh,
        )

    if settings is None or settings.allow_heuristic_reports:
        return heuristic()
    try:
        if settings.for_judge("a").llm_configured() and settings.for_judge("b").llm_configured():
            return _llm_evaluate_cluster(
                cluster,
                findings,
                max_rounds=max_rounds,
                settings=settings,
                followup_executor=followup_executor,
                context=context,
                context_refresh=context_refresh,
            )
    except LLMGenerationError:
        raise
    return heuristic()
