"""LLM agent functions: Step 1 post normalization, Step 2 cluster refinement,
Step 3 investigation planning and perspective queries, and Step 5 report
field composition.

Split out of ``llm.py`` (kept at/under its 800-line budget), which retains
the provider client (``_chat_completion`` and friends), JSON-object
extraction, and the shared text validators these agents call. Every call
into ``llm.py`` goes through the ``llm`` module object (``_llm.<name>``)
rather than a direct name import, so tests that do
``monkeypatch.setattr("tiblender_runtime.llm._chat_completion", ...)`` still
intercept calls made from here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from . import llm as _llm
from .contract import (
    ENGLISH_ONLY_RETRY_INSTRUCTION,
    INVESTIGATION_PLANNER_PROMPT,
    PERSPECTIVE_AGENT_PROMPT,
    POST_NORMALIZATION_PROMPT,
    REQUIRED_TOOL_TYPES,
    REPORT_GENERATION_PROMPT,
    THREAT_CLUSTERING_PROMPT,
)
from .iocs import extract_iocs, filter_reportable_iocs
from . import language
from .llm import LLMGenerationError
from .llm_fallbacks import category_of, record_fallback
from .llm_focus import _perspective_payload, select_focus_types_with_llm  # noqa: F401  (re-exported)
from .models import (
    Cluster,
    EvaluationDecision,
    IocSet,
    NormalizedPost,
    PerspectiveFinding,
    Post,
    Severity,
    ThreatCategory,
    ToolQuery,
    to_jsonable,
)
from .settings import Settings


def _compact_reportable_iocs(iocs: IocSet, source_cves: set[str] | None = None) -> IocSet:
    clean = filter_reportable_iocs(iocs)
    if source_cves is not None:
        clean.cves = [cve for cve in clean.cves if cve in source_cves]
    return clean


def _compact_cluster_payload(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    decision: EvaluationDecision,
) -> dict[str, Any]:
    # Keep the LLM input bounded and de-identified; source post URLs are not
    # required for report writing and are intentionally omitted.
    tool_results = [result for finding in findings for result in finding.tool_results]
    source_iocs = filter_reportable_iocs(cluster.iocs)
    source_cves = set(source_iocs.cves)
    tool_iocs = _compact_reportable_iocs(
        IocSet.from_many([result.discovered_iocs for result in tool_results]),
        source_cves,
    )
    discovered = source_iocs.merge(tool_iocs)
    return {
        "cluster": {
            "id": cluster.id,
            "topic_name": cluster.topic_name,
            "description": cluster.description,
            "threat_category": cluster.threat_category.value,
            "severity": cluster.severity.value,
            "iocs": to_jsonable(discovered),
            "post_count": len(cluster.posts),
            "platforms": sorted({post.platform for post in cluster.posts}),
            "clustering_signals": cluster.clustering_signals,
        },
        "source_posts": [
            {
                "id": post.id,
                "platform": post.platform,
                "timestamp": post.post.timestamp,
                "normalized_content": post.normalized_content[:600],
            }
            for post in cluster.posts[:12]
        ],
        "perspective_findings": [
            {
                "perspective": finding.perspective,
                "focus_types": finding.focus_types,
                "insights": finding.insights[:6],
                "tool_results": [
                    {
                        "tool_type": result.tool_type,
                        "query_params": result.query_params,
                        "summary": result.summary,
                        "confidence": result.confidence.value,
                        "discovered_iocs": to_jsonable(_compact_reportable_iocs(result.discovered_iocs, source_cves)),
                    }
                    for result in finding.tool_results[:8]
                ],
            }
            for finding in findings
        ],
        "evaluation": to_jsonable(decision),
    }


def _heuristic_report_fields(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    decision: EvaluationDecision,
    settings: Settings,
) -> dict[str, Any]:
    """Step 5 prose without an LLM, composed in English by ``language.py``."""
    return language.deterministic_report_fields(cluster, findings, decision, settings)


def _messages(payload: dict[str, Any], *, english_only: bool = False) -> list[dict[str, str]]:
    allowed_tools = ", ".join(REQUIRED_TOOL_TYPES)
    return [
        {
            "role": "system",
            "content": (
                (ENGLISH_ONLY_RETRY_INSTRUCTION + "\n\n" if english_only else "")
                + REPORT_GENERATION_PROMPT
                + "\n\n"
                "Use only the supplied evidence. Do not invent IoCs, sources, CVEs, actors, or tool results. "
                "Generate the report sections as structured fields. Return strict JSON only with keys: "
                "cluster_name, severity, executive_summary, key_findings, recommendations, immediate_actions, "
                "analyst_notes, threat_actor, timeline, ttps. "
                "Write executive_summary, immediate_actions, and analyst_notes in professional English. "
                "executive_summary must be one concise analyst-facing paragraph. "
                "immediate_actions must be an array of 3 to 5 concrete defensive actions. "
                "key_findings must be an array of objects with heading and items. "
                "recommendations must be an array of objects with action, rationale and horizon, "
                "where horizon is one of immediate, short_term or long_term. "
                "timeline must be an array of timestamp, event, significance objects. "
                "ttps must be an array of mitre_id, tactic, technique, evidence objects. "
                f"Follow-up collection may name only these allowed tools: {allowed_tools}."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def _post_normalization_messages(posts: list[Post]) -> list[dict[str, str]]:
    payload = {
        "posts": [
            {
                "id": post.id,
                "platform": post.platform,
                "timestamp": post.timestamp,
                "language_hint": post.metadata.get("language", "unknown"),
                "source_iocs": extract_iocs(post.content).all_values(),
                "content": post.content[:4000],
            }
            for post in posts
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                POST_NORMALIZATION_PROMPT
                + "\n\n"
                "Use only the supplied post text. Do not invent IoCs, CVEs, brands, actors, victims, or tools. "
                "Write every normalized_content value in professional English, even when the source post is not English. "
                "Preserve every supplied source_ioc verbatim in normalized_content. "
                "Return strict JSON only with key posts. Each item must include id and normalized_content."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def _clean_normalized_content(text: str) -> str:
    compact = " ".join(text.strip().split())
    # Some LLMs fill absent structured fields as comma-separated "none"
    # fragments. Drop only standalone placeholders, preserving real prose and
    # IoCs for clustering and report evidence.
    parts = [part.strip() for part in compact.split(",")]
    parts = [part for part in parts if part.lower() not in {"none", "unknown", "n/a", "na"}]
    cleaned = ", ".join(parts)
    cleaned = re.sub(r"\s*,\s*(IoCs?:)", r". \1", cleaned)
    # Dropping the placeholders above can leave a dangling "IoCs:" label with
    # nothing behind it (the model wrote "IoCs: none"); an empty label is
    # noise in every downstream summary, so remove the label too.
    cleaned = re.sub(
        r"\s*\bIoCs?:(?:\s*(?:none|unknown|n/a|na)\b[.\s]*)?\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _ensure_source_iocs_preserved(summary: str, post: Post) -> str:
    source_iocs = extract_iocs(post.content).all_values()
    if not source_iocs:
        return summary
    summary_iocs = set(extract_iocs(summary).all_values())
    missing = [ioc for ioc in source_iocs if ioc not in summary_iocs]
    if not missing:
        return summary
    separator = "" if summary.endswith((".", "!", "?")) else "."
    return f"{summary}{separator} IoCs: {' '.join(missing)}"


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    step = max(1, size)
    return [items[index : index + step] for index in range(0, len(items), step)]


def _map_chunks_in_order(chunks: list[list[Any]], worker, max_workers: int) -> list[Any]:
    """Run ``worker`` over every chunk concurrently, results in chunk order.

    Chunks are independent provider calls, so waiting for each one before
    starting the next only adds latency. Results are indexed by chunk position
    and reassembled in that order, so concurrency can never reorder a cycle's
    output; the per-chunk retry/fallback semantics live inside ``worker`` and
    are unchanged.
    """
    if not chunks:
        return []
    if len(chunks) == 1 or max_workers <= 1:
        return [worker(chunk) for chunk in chunks]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as pool:
        futures = [pool.submit(worker, chunk) for chunk in chunks]
    return [future.result() for future in futures]


def _normalize_chunk(settings: Settings, posts: list[Post]) -> dict[str, str]:
    """One provider call for one chunk of posts. Raises on an unusable reply."""
    completion = _llm._chat_completion_with_meta(settings, _post_normalization_messages(posts))
    if completion.truncated:
        # finish_reason == "length": the reply is a partial document even if
        # its prefix parses, so the caller must retry with a smaller chunk
        # rather than silently normalize only the posts that fit.
        raise LLMGenerationError(
            f"LLM post normalization reply was truncated (finish_reason=length) for {len(posts)} posts",
            category="truncated",
        )
    parsed = _llm._extract_json_object(completion.content)
    raw_posts = parsed.get("posts")
    if not isinstance(raw_posts, list):
        raise LLMGenerationError(
            "LLM post normalization draft must include a posts array", category="invalid_response"
        )
    expected = {post.id: post for post in posts}
    normalized: dict[str, str] = {}
    for item in raw_posts:
        if not isinstance(item, dict):
            continue
        post_id = str(item.get("id", "")).strip()
        if post_id not in expected:
            continue
        content = _clean_normalized_content(str(item.get("normalized_content", "")))
        if not content:
            continue
        normalized[post_id] = _ensure_source_iocs_preserved(content, expected[post_id])
    return normalized


def _try_normalize_chunk(
    settings: Settings, posts: list[Post]
) -> tuple[dict[str, str], LLMGenerationError | None]:
    """Normalize one chunk, returning the error object rather than its text.

    The caller needs the failure's *category*, which this code attached where
    the error was raised; returning ``str(error)`` would throw that away and
    force a guess from a message that embeds the provider's own words.
    """
    try:
        return _normalize_chunk(settings, posts), None
    except LLMGenerationError as error:
        return {}, error


def normalize_post_batch(settings: Settings, posts: list[Post]) -> dict[str, str]:
    # Step 1 normalizes multilingual social posts into English summaries before
    # clustering. The output remains bounded and de-identified: source URLs and
    # authors are not sent unless they are part of the post content itself.
    # Posts are sent in chunks: one request carrying a whole cycle's posts is
    # what makes a provider stop at its output-token cap mid-document.
    if not posts:
        return {}
    per_chunk = _map_chunks_in_order(
        _chunks(posts, settings.llm_batch_size),
        lambda chunk: _normalize_chunk_with_retry(settings, chunk),
        settings.llm_max_concurrent,
    )
    normalized: dict[str, str] = {}
    for chunk_result in per_chunk:
        normalized.update(chunk_result)
    return normalized


def _normalize_chunk_with_retry(settings: Settings, chunk: list[Post]) -> dict[str, str]:
    """Normalize one chunk, retrying once with halved chunks before falling back.

    Posts still missing after the retry are simply left out of the returned
    map: ``normalization.normalize_posts`` then gives them the deterministic
    ``_brief_summary`` instead, and the omission is counted so it shows up in
    ``cycle_summary.json`` and in the affected reports' metadata.
    """
    normalized, error = _try_normalize_chunk(settings, chunk)
    missing = [post for post in chunk if post.id not in normalized]
    if missing and len(chunk) > 1:
        half = max(1, len(chunk) // 2)
        for sub_chunk in _chunks(missing, half):
            partial, sub_error = _try_normalize_chunk(settings, sub_chunk)
            normalized.update(partial)
            error = sub_error or error
        missing = [post for post in chunk if post.id not in normalized]
    if missing:
        if error is None:
            # The provider answered with valid JSON that simply left these
            # post ids out -- a malformed answer, not an unclassifiable one.
            category, summary = "invalid_response", "provider omitted normalized_content for some posts"
        else:
            category, summary = category_of(error), "post normalization chunk rejected after one retry"
        record_fallback(
            "normalization_posts",
            count=len(missing),
            post_ids=[post.id for post in missing],
            category=category,
            summary=summary,
            detail=str(error) if error is not None else "",
        )
    return normalized


def _cluster_refinement_messages(clusters: list[Cluster]) -> list[dict[str, str]]:
    payload = {
        "candidate_clusters": [
            {
                "id": cluster.id,
                "topic_name": cluster.topic_name,
                "description": cluster.description,
                "severity": cluster.severity.value,
                "threat_category": cluster.threat_category.value,
                "iocs": to_jsonable(filter_reportable_iocs(cluster.iocs)),
                "posts": [
                    {
                        "id": post.id,
                        "platform": post.platform,
                        "timestamp": post.post.timestamp,
                        "normalized_content": post.normalized_content[:800],
                        "iocs": to_jsonable(filter_reportable_iocs(post.iocs)),
                    }
                    for post in cluster.posts
                ],
            }
            for cluster in clusters
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                THREAT_CLUSTERING_PROMPT
                + "\n\n"
                "Refine only the supplied candidate clusters. Split mixed clusters when posts describe different campaigns. "
                "Use only supplied post ids; never create, rename, or infer source posts. "
                "Return strict JSON only with key clusters. Each cluster item must include topic_name, "
                "security_severity, topic_description, and post_ids."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def _cluster_digest(post_ids: list[str]) -> str:
    return hashlib.sha1(",".join(sorted(post_ids)).encode("utf-8")).hexdigest()[:12]


def _dominant_category(posts: list[NormalizedPost]) -> ThreatCategory:
    counts = Counter(post.threat_category for post in posts if post.threat_category is not ThreatCategory.OTHER)
    return counts.most_common(1)[0][0] if counts else ThreatCategory.OTHER


def _severity(value: object, default: Severity = Severity.MEDIUM) -> Severity:
    try:
        return Severity(str(value).strip().upper())
    except ValueError:
        return default


def refine_clusters_with_llm(settings: Settings, clusters: list[Cluster]) -> list[Cluster]:
    # Step 2 starts with deterministic dual-track candidates and then asks the
    # clustering agent to split mixed candidates into campaign-level units.
    # Candidates go out in chunks for the same reason posts do; a chunk that
    # cannot be refined keeps its unrefined deterministic clusters.
    if not clusters:
        return []
    per_chunk = _map_chunks_in_order(
        _chunks(clusters, settings.llm_cluster_batch_size),
        lambda chunk: _refine_chunk_with_retry(settings, chunk),
        settings.llm_max_concurrent,
    )
    refined: list[Cluster] = []
    for chunk_result in per_chunk:
        refined.extend(chunk_result)
    return refined


def _record_refinement_fallback(chunk: list[Cluster], error: LLMGenerationError) -> None:
    """Attribute one failed refinement batch to exactly its own candidates.

    A failing batch keeps its unrefined candidate clusters, so those cluster
    ids (and their posts, which survive whatever the ids do) are the precise
    scope of the damage. Recording the batch instead of the whole cycle is
    what keeps ``generation_mode`` per report: the candidates that were
    refined successfully in another batch stay ``llm``.
    """
    record_fallback(
        "cluster_refinement_batches",
        cluster_ids=[cluster.id for cluster in chunk],
        post_ids=[post.id for cluster in chunk for post in cluster.posts],
        category=category_of(error),
        summary="cluster refinement batch rejected after one retry",
        detail=str(error),
    )


def _refine_chunk_with_retry(settings: Settings, chunk: list[Cluster]) -> list[Cluster]:
    """Refine one chunk of candidates, retrying once with halved chunks.

    Falls back to the unrefined candidates of the failing chunk rather than
    aborting the cycle, and records the fallback for ``cycle_summary.json``
    against that chunk's own clusters and posts only.
    """
    try:
        return _refine_cluster_chunk(settings, chunk)
    except LLMGenerationError as first_error:
        error = first_error
    if len(chunk) == 1:
        _record_refinement_fallback(chunk, error)
        return list(chunk)

    half = max(1, len(chunk) // 2)
    refined: list[Cluster] = []
    for sub_chunk in _chunks(chunk, half):
        try:
            refined.extend(_refine_cluster_chunk(settings, sub_chunk))
        except LLMGenerationError as sub_error:
            _record_refinement_fallback(sub_chunk, sub_error)
            refined.extend(sub_chunk)
    return refined


def _refine_cluster_chunk(settings: Settings, clusters: list[Cluster]) -> list[Cluster]:
    completion = _llm._chat_completion_with_meta(settings, _cluster_refinement_messages(clusters))
    if completion.truncated:
        raise LLMGenerationError(
            f"LLM cluster refinement reply was truncated (finish_reason=length) for {len(clusters)} clusters",
            category="truncated",
        )
    parsed = _llm._extract_json_object(completion.content)
    raw_clusters = parsed.get("clusters")
    if not isinstance(raw_clusters, list):
        raise LLMGenerationError(
            "LLM cluster refinement draft must include a clusters array", category="invalid_response"
        )

    post_by_id = {post.id: post for cluster in clusters for post in cluster.posts}
    source_cluster_by_post = {
        post.id: cluster.id
        for cluster in clusters
        for post in cluster.posts
    }
    refined: list[Cluster] = []
    used_posts: set[str] = set()
    for item in raw_clusters:
        if not isinstance(item, dict):
            continue
        raw_post_ids = item.get("post_ids", [])
        if not isinstance(raw_post_ids, list):
            continue
        post_ids = []
        for raw_post_id in raw_post_ids:
            post_id = str(raw_post_id).strip()
            if post_id in post_by_id and post_id not in used_posts and post_id not in post_ids:
                post_ids.append(post_id)
        if not post_ids:
            continue
        posts = [post_by_id[post_id] for post_id in post_ids]
        used_posts.update(post_ids)
        iocs = IocSet.from_many([post.iocs for post in posts])
        topic_name = _llm._sanitize_report_text(str(item.get("topic_name", "")).strip()) or "Refined threat campaign"
        description = _llm._sanitize_report_text(str(item.get("topic_description", "")).strip()) or "LLM-refined campaign cluster."
        category = _dominant_category(posts)
        source_cluster_ids = sorted({source_cluster_by_post[post_id] for post_id in post_ids})
        refined.append(
            Cluster(
                id=f"cluster-{_cluster_digest(post_ids)}",
                topic_name=topic_name,
                description=description,
                posts=posts,
                iocs=iocs,
                severity=_severity(item.get("security_severity")),
                threat_category=category,
                clustering_signals={
                    "llm_campaign_refinement": True,
                    "source_cluster_ids": source_cluster_ids,
                    "post_count": len(posts),
                },
            )
        )
    if not refined:
        raise LLMGenerationError(
            "LLM cluster refinement did not return any valid campaign clusters", category="invalid_response"
        )
    return refined


def _planner_messages(cluster: Cluster) -> list[dict[str, str]]:
    payload = {
        "cluster": {
            "id": cluster.id,
            "topic_name": cluster.topic_name,
            "description": cluster.description,
            "severity": cluster.severity.value,
            "threat_category": cluster.threat_category.value,
            "iocs": to_jsonable(filter_reportable_iocs(cluster.iocs)),
            "platforms": sorted({post.platform for post in cluster.posts}),
            "posts": [
                {
                    "id": post.id,
                    "platform": post.platform,
                    "normalized_content": post.normalized_content[:700],
                }
                for post in cluster.posts[:8]
            ],
        }
    }
    return [
        {
            "role": "system",
            "content": (
                INVESTIGATION_PLANNER_PROMPT
                + "\n\n"
                "Use only supplied IoCs, CVEs, campaign names, malware names, package names, or source-post terms. "
                "Do not output generic instructional phrases as queries. Return strict JSON only with key queries."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def plan_investigation_queries(settings: Settings, cluster: Cluster) -> list[ToolQuery]:
    """Step 3 planner for one cluster, retried once and never fatal.

    A cluster whose plan cannot be generated investigates with the
    perspective agents' own (heuristic or LLM) queries instead of aborting
    the cycle; the fallback is counted for ``cycle_summary.json`` and marks
    that cluster's report ``llm_partial``.
    """
    try:
        return _plan_investigation_queries_once(settings, cluster)
    except LLMGenerationError as first_error:
        reason = str(first_error)
    try:
        return _plan_investigation_queries_once(settings, cluster)
    except LLMGenerationError as error:
        record_fallback(
            "planner_clusters",
            cluster_id=cluster.id,
            category=category_of(error),
            summary="investigation planner draft rejected after one retry",
            detail=f"{reason} / retry: {error}",
        )
        return []


def _plan_investigation_queries_once(settings: Settings, cluster: Cluster) -> list[ToolQuery]:
    # The runtime validates every query against the tool contract and source
    # evidence before it can be executed.
    parsed = _llm._extract_json_object(_llm._chat_completion(settings, _planner_messages(cluster)))
    raw_queries = parsed.get("queries")
    if not isinstance(raw_queries, list):
        raise LLMGenerationError(
            "LLM investigation planner draft must include a queries array", category="invalid_response"
        )
    queries: list[ToolQuery] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_queries:
        if not isinstance(raw, dict):
            continue
        query = _llm._validated_tool_query(raw, cluster)
        if query is None:
            continue
        key = (query.perspective, query.tool_type, json.dumps(query.parameters, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return sorted(queries, key=lambda item: item.priority, reverse=True)[:15]


def _perspective_query_messages(cluster: Cluster, perspective: str, focus_types: list[str]) -> list[dict[str, str]]:
    payload = {**_perspective_payload(cluster, perspective), "selected_focus_types": list(focus_types)}
    allowed_tools = ", ".join(REQUIRED_TOOL_TYPES)
    return [
        {
            "role": "system",
            "content": (
                PERSPECTIVE_AGENT_PROMPT
                + "\n\n"
                f"You are the {perspective} perspective agent. Subtask (ii): generate 1 to 2 tool queries per "
                "selected focus type. Every domain, IP, URL, hash, and CVE parameter must come from "
                "verified_iocs; never invent an indicator. Search queries must name a verified indicator, a CVE, "
                "a malware family, or a brand from the cluster. "
                f"Allowed tool_type values: {allowed_tools}. "
                "Return strict JSON only with key queries, an array of objects with focus_type, tool_type, "
                "parameters, priority, and timeout_seconds."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def generate_perspective_queries_with_llm(
    settings: Settings,
    cluster: Cluster,
    perspective: str,
    focus_types: list[str],
) -> list[ToolQuery]:
    # Subtask (ii): concrete tool calls for the selected focuses. Every query is
    # re-validated against the cluster's verified IoCs before it is executed.
    parsed = _llm._extract_json_object(
        _llm._chat_completion(settings, _perspective_query_messages(cluster, perspective, focus_types))
    )
    raw_queries = parsed.get("queries")
    if not isinstance(raw_queries, list):
        raise LLMGenerationError(
            "LLM perspective agent draft must include a queries array", category="invalid_response"
        )
    allowed_focuses = set(focus_types)
    queries: list[ToolQuery] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_queries:
        if not isinstance(raw, dict):
            continue
        query = _llm._validated_tool_query({**raw, "perspective": perspective}, cluster)
        if query is None:
            continue
        focus = str(raw.get("focus_type", "")).strip()
        query = replace(query, focus_type=focus if focus in allowed_focuses else "")
        key = (query.tool_type, json.dumps(query.parameters, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
    if not queries:
        raise LLMGenerationError(
            f"LLM perspective agent returned no usable {perspective} queries", category="invalid_response"
        )
    return queries


def compose_report_fields(
    settings: Settings,
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    decision: EvaluationDecision,
) -> dict[str, Any]:
    """Step 5 report prose for one cluster, retried once and never fatal.

    ``validate_report_generation`` still fails fast when a live run has no
    LLM configured at all; a per-cluster generation failure instead falls
    back to the deterministic report fields for that one report.
    """
    # The LLM only supplies prose fields. IoCs, STIX objects, timelines, and
    # source evidence are assembled deterministically from cluster/tool data.
    if settings.allow_heuristic_reports:
        return _heuristic_report_fields(cluster, findings, decision, settings)

    settings.validate_report_generation()
    try:
        return _english_or_raise(_compose_report_fields_once(settings, cluster, findings, decision))
    except LLMGenerationError as first_error:
        reason = str(first_error)
        category = category_of(first_error)
    try:
        # The retry states the language requirement explicitly when (and only
        # when) that is what the first answer got wrong; every other failure
        # retries with the unchanged prompt.
        return _english_or_raise(
            _compose_report_fields_once(
                settings, cluster, findings, decision,
                english_only=category == "non_english_output",
            )
        )
    except LLMGenerationError as error:
        record_fallback(
            "report_fields",
            cluster_id=cluster.id,
            category=category_of(error),
            summary="report field composition rejected after one retry",
            detail=f"{reason} / retry: {error}",
        )
        return _heuristic_report_fields(cluster, findings, decision, settings)


def _english_or_raise(fields: dict[str, Any]) -> dict[str, Any]:
    """Return ``fields`` unchanged, or raise when a prose field is not English.

    The category is authored here, at the raise site, from the guard's own
    verdict (:func:`language.failing_report_fields`) -- the failing text is
    the model's draft of scraped posts and is never pattern-matched.
    """
    failing = language.failing_report_fields(fields)
    if failing:
        raise LLMGenerationError(
            "LLM report draft is not English in: " + ", ".join(failing),
            category="non_english_output",
        )
    return fields


def _compose_report_fields_once(
    settings: Settings,
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    decision: EvaluationDecision,
    *,
    english_only: bool = False,
) -> dict[str, Any]:
    payload = _compact_cluster_payload(cluster, findings, decision)
    parsed = _llm._extract_json_object(
        _llm._chat_completion(settings, _messages(payload, english_only=english_only))
    )
    executive_summary = _llm._sanitize_report_text(str(parsed.get("executive_summary", "")).strip())
    recommendations = _llm._clean_recommendations(parsed.get("recommendations"))
    immediate_actions = _llm._clean_actions(parsed.get("immediate_actions"))
    if not immediate_actions and recommendations:
        immediate_actions = [item["action"] for item in recommendations]
    if not executive_summary or not immediate_actions:
        raise LLMGenerationError(
            "LLM report draft must include non-empty executive_summary and immediate_actions fields",
            category="invalid_response",
        )
    severity_value = str(parsed.get("severity", "")).strip().upper()
    if severity_value not in {item.value for item in Severity}:
        severity_value = ""
    return {
        "cluster_name": _llm._clean_string(parsed.get("cluster_name", "")),
        "severity": severity_value,
        "executive_summary": executive_summary,
        "key_findings": _llm._clean_key_findings(parsed.get("key_findings")),
        "recommendations": recommendations,
        "immediate_actions": immediate_actions,
        "analyst_notes": _llm._sanitize_report_text(str(parsed.get("analyst_notes", "")).strip()),
        "threat_actor": _llm._clean_string(parsed.get("threat_actor", "")),
        "timeline": _llm._clean_timeline(parsed.get("timeline")),
        "ttps": _llm._clean_ttps(parsed.get("ttps")),
        "metadata": {
            "generation_mode": "llm",
            "llm_used": True,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.azure_openai_deployment
            if settings.llm_provider.strip().lower() in {"azure", "azure_openai", "foundry", "microsoft_foundry"}
            else settings.llm_model,
        },
    }
