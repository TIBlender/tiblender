"""LLM provider client and shared text validators.

Agent-level functions (Step 1 post normalization, Step 2 cluster refinement,
Step 3 investigation planning/perspective queries, Step 5 report field
composition) live in :mod:`tiblender_runtime.llm_agents` to keep this module
under its line budget; they are re-exported lazily below (see
``__getattr__``) so every existing ``from .llm import <agent function>``
caller, and every ``monkeypatch.setattr("tiblender_runtime.llm.<name>", ...)``
in the test suite, keeps working unchanged.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .contract import RECOMMENDATION_HORIZONS, REQUIRED_PERSPECTIVES, REQUIRED_TOOL_TYPES, unsupported_tool_tokens
from .llm_fallbacks import REASON_KEYS
from .iocs import filter_reportable_iocs
from .models import Cluster, ToolQuery
from .settings import Settings

_AGENT_REEXPORTS = frozenset(
    {
        "normalize_post_batch",
        "refine_clusters_with_llm",
        "plan_investigation_queries",
        "select_focus_types_with_llm",
        "generate_perspective_queries_with_llm",
        "compose_report_fields",
    }
)


def __getattr__(name: str) -> Any:
    # PEP 562 lazy module attribute: llm_agents.py imports `from . import llm
    # as _llm` at module scope, so importing it eagerly from here (at load
    # time) would be circular. Resolving it lazily on first attribute access
    # means llm.py never imports llm_agents until something actually asks for
    # one of these names, by which point both modules load cleanly regardless
    # of which one a caller imports first.
    if name in _AGENT_REEXPORTS:
        from . import llm_agents

        return getattr(llm_agents, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class LLMGenerationError(RuntimeError):
    """Raised when the configured LLM cannot produce a usable report draft.

    ``category`` is the machine-readable cause, authored **here, at the raise
    site**, and drawn from :data:`llm_fallbacks.REASON_KEYS`. The fallback
    accounting buckets by it instead of pattern-matching ``str(error)``,
    because that message routinely embeds the provider's reply or the model's
    own draft -- i.e. text derived from scraped, adversary-authored posts.
    """

    def __init__(self, *args: object, category: str = "unknown") -> None:
        super().__init__(*args)
        self.category = category if category in REASON_KEYS else "unknown"


_TOOL_TO_PERSPECTIVE = {
    "WHOIS_LOOKUP": "Infrastructure",
    "PASSIVE_DNS_LOOKUP": "Infrastructure",
    "TLS_CT_LOOKUP": "Infrastructure",
    "WEB_CRAWL": "Technical",
    "NVD_CVE_LOOKUP": "Technical",
    "GITHUB_SEARCH": "Technical",
    "X_SEARCH": "Social",
    "REDDIT_SEARCH": "Social",
    "DISCORD_SEARCH": "Social",
    "TELEGRAM_SEARCH": "Social",
    "SEARCH_ENGINE": "Actor",
}

_QUERY_STOPWORDS = {
    "about",
    "actor",
    "campaign",
    "cluster",
    "context",
    "cyber",
    "investigate",
    "malicious",
    "more",
    "this",
    "threat",
}


def _topic_tokens(cluster: Cluster) -> set[str]:
    text = f"{cluster.topic_name} {cluster.description} " + " ".join(post.normalized_content for post in cluster.posts)
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,}", text)
        if token.lower() not in _QUERY_STOPWORDS
    }


def _query_is_specific(value: str, cluster: Cluster) -> bool:
    lowered = value.lower()
    markers = [marker.lower() for marker in filter_reportable_iocs(cluster.iocs).all_values()]
    if any(marker and marker in lowered for marker in markers):
        return True
    query_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,}", value)
        if token.lower() not in _QUERY_STOPWORDS
    }
    return len(query_tokens & _topic_tokens(cluster)) >= 2


def _clean_priority(value: object) -> int:
    try:
        return max(1, min(10, int(value)))
    except (TypeError, ValueError):
        return 5


def _clean_timeout(value: object) -> int:
    try:
        return max(1, min(120, int(value)))
    except (TypeError, ValueError):
        return 30


def _validated_tool_query(raw: dict[str, Any], cluster: Cluster) -> ToolQuery | None:
    tool_type = str(raw.get("tool_type", "")).strip().upper()
    if tool_type not in REQUIRED_TOOL_TYPES:
        return None
    parameters = raw.get("parameters")
    if not isinstance(parameters, dict):
        return None
    perspective = str(raw.get("perspective", "")).strip() or _TOOL_TO_PERSPECTIVE.get(tool_type, "Actor")
    if perspective not in REQUIRED_PERSPECTIVES:
        perspective = _TOOL_TO_PERSPECTIVE.get(tool_type, "Actor")

    iocs = filter_reportable_iocs(cluster.iocs)
    allowed_domains = set(iocs.domains)
    allowed_urls = set(iocs.urls)
    allowed_ips = set(iocs.ips)
    allowed_hashes = set(iocs.hashes)
    allowed_cves = set(iocs.cves)

    clean_parameters: dict[str, Any] = {}
    if tool_type == "WEB_CRAWL":
        url = str(parameters.get("url", "")).strip()
        if url not in allowed_urls:
            return None
        clean_parameters["url"] = url
    elif tool_type in {"PASSIVE_DNS_LOOKUP", "TLS_CT_LOOKUP"}:
        domain = str(parameters.get("domain", "")).strip().lower()
        if domain not in allowed_domains:
            return None
        clean_parameters["domain"] = domain
    elif tool_type == "WHOIS_LOOKUP":
        domain = str(parameters.get("domain", "")).strip().lower()
        ip = str(parameters.get("ip", "")).strip()
        if domain and domain in allowed_domains:
            clean_parameters["domain"] = domain
        elif ip and ip in allowed_ips:
            clean_parameters["ip"] = ip
        else:
            return None
    elif tool_type == "NVD_CVE_LOOKUP":
        cve = str(parameters.get("cve", "")).strip().upper()
        if cve not in allowed_cves:
            return None
        clean_parameters["cve"] = cve
    elif tool_type == "GITHUB_SEARCH":
        cve = str(parameters.get("cve", "")).strip().upper()
        hash_value = str(parameters.get("hash", "")).strip().lower()
        query = str(parameters.get("query", "")).strip()
        if cve and cve in allowed_cves:
            clean_parameters["cve"] = cve
        elif hash_value and hash_value in allowed_hashes:
            clean_parameters["hash"] = hash_value
        elif query and _query_is_specific(query, cluster):
            clean_parameters["query"] = query
        else:
            return None
    else:
        query = str(parameters.get("query", "")).strip()
        if not query or not _query_is_specific(query, cluster):
            return None
        clean_parameters["query"] = query

    return ToolQuery(
        perspective=perspective,
        tool_type=tool_type,
        parameters=clean_parameters,
        priority=_clean_priority(raw.get("priority")),
        timeout_seconds=_clean_timeout(raw.get("timeout_seconds")),
    )


def _request_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # A 2xx body that is not valid UTF-8 is a malformed response, not
            # an abort: decoding with errors="replace" (as the HTTPError arm
            # below already does) turns it into the invalid_response arm via
            # json.JSONDecodeError instead of letting UnicodeDecodeError --
            # which none of these arms catch -- escape uncategorised.
            payload = response.read().decode("utf-8", errors="replace")
            return json.loads(payload)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2000]
        raise LLMGenerationError(
            f"LLM provider returned HTTP {error.code}: {detail}", category="http_status"
        ) from error
    except TimeoutError as error:
        # socket.timeout is an alias of TimeoutError since 3.10, and urllib
        # raises it directly for the read timeout.
        raise LLMGenerationError(f"LLM provider request timed out: {error}", category="timeout") from error
    except json.JSONDecodeError as error:
        # A 2xx reply whose body is not JSON is a malformed response, not a
        # transport failure.
        raise LLMGenerationError(
            f"LLM provider reply was not valid JSON: {error}", category="invalid_response"
        ) from error
    except OSError as error:
        if isinstance(getattr(error, "reason", None), TimeoutError):
            raise LLMGenerationError(
                f"LLM provider request timed out: {error}", category="timeout"
            ) from error
        raise LLMGenerationError(f"LLM provider request failed: {error}", category="network") from error


def _request_chat_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout: int) -> dict[str, Any]:
    # Some OpenAI-compatible deployments reject optional Chat Completions
    # parameters. Retrying without them preserves compatibility without hiding
    # real transport or authentication failures.
    retry_body = dict(body)
    for _ in range(3):
        try:
            return _request_json(url, headers, retry_body, timeout=timeout)
        except LLMGenerationError as error:
            detail = str(error)
            if "temperature" in detail and "temperature" in retry_body:
                retry_body = dict(retry_body)
                retry_body.pop("temperature", None)
                continue
            if "response_format" in detail and "response_format" in retry_body:
                retry_body = dict(retry_body)
                retry_body.pop("response_format", None)
                continue
            raise
    return _request_json(url, headers, retry_body, timeout=timeout)


def _normalise_chat_content(raw_content: Any) -> str:
    if isinstance(raw_content, str):
        return raw_content
    if not isinstance(raw_content, list):
        return ""
    parts: list[str] = []
    for item in raw_content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("content")
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(text, dict) and isinstance(text.get("value"), str):
            parts.append(text["value"])
    return "".join(parts)


def _chat_response_content(response: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise LLMGenerationError(
            f"LLM provider response did not include chat content: {response}", category="invalid_response"
        ) from error
    if not isinstance(choice, dict) or not isinstance(message, dict):
        raise LLMGenerationError(
            f"LLM provider response did not include chat content: {response}", category="invalid_response"
        )
    return _normalise_chat_content(message.get("content")), choice, message


def _empty_content_error(response: dict[str, Any], choice: dict[str, Any], message: dict[str, Any]) -> LLMGenerationError:
    usage = response.get("usage") if isinstance(response, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    completion_details = usage.get("completion_tokens_details")
    completion_details = completion_details if isinstance(completion_details, dict) else {}
    # A capped reply with no content at all is truncation; anything else is a
    # provider returning nothing usable.
    category = "truncated" if str(choice.get("finish_reason", "")).strip().lower() == "length" else "empty_content"
    return LLMGenerationError(
        "LLM provider returned empty chat content "
        f"(finish_reason={choice.get('finish_reason')!r}, "
        f"message_keys={sorted(message.keys())}, "
        f"completion_tokens={usage.get('completion_tokens')!r}, "
        f"reasoning_tokens={completion_details.get('reasoning_tokens')!r}). "
        "Increase TIBLENDER_LLM_MAX_OUTPUT_TOKENS or use a deployment that returns chat message content.",
        category=category,
    )


def _expanded_token_budget(body: dict[str, Any]) -> int | None:
    try:
        current = int(body.get("max_completion_tokens", 0))
    except (TypeError, ValueError):
        current = 0
    expanded = min(max(current * 2, 8192), 16384)
    return expanded if expanded > current else None


@dataclass(frozen=True, slots=True)
class ChatCompletion:
    """One provider reply plus the metadata callers need to trust it.

    ``truncated`` is the provider's ``finish_reason == "length"``: the model
    stopped because it ran out of output budget, so ``content`` is a partial
    document even when it happens to parse. Batch agents use this to shrink
    the chunk and retry instead of accepting half a result.
    """

    content: str
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def _chat_completion_with_meta(settings: Settings, messages: list[dict[str, str]]) -> ChatCompletion:
    # Keep one request shape for OpenAI-compatible providers and Azure-style
    # deployments so the rest of the pipeline can remain provider-agnostic.
    provider = settings.llm_provider.strip().lower()
    body: dict[str, Any] = {
        "messages": messages,
        "temperature": settings.llm_temperature,
        "max_completion_tokens": settings.llm_max_output_tokens,
        "response_format": {"type": "json_object"},
    }
    if provider in {"azure", "azure_openai", "foundry", "microsoft_foundry"}:
        deployment = urllib.parse.quote(settings.azure_openai_deployment, safe="")
        api_version = urllib.parse.quote(settings.azure_openai_api_version, safe="")
        url = (
            f"{settings.azure_openai_endpoint.rstrip('/')}/openai/deployments/"
            f"{deployment}/chat/completions?api-version={api_version}"
        )
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "tiblender/0.1",
            "api-key": settings.openai_api_key,
        }
    else:
        body["model"] = settings.llm_model
        url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "tiblender/0.1",
            "Authorization": f"Bearer {settings.openai_api_key}",
        }

    response = _request_chat_json(url, headers, body, timeout=settings.llm_timeout_seconds)
    content, choice, message = _chat_response_content(response)
    if not content.strip() and choice.get("finish_reason") == "length":
        # Empty content with a length finish reason usually means the token
        # budget was too low for strict JSON; retry once with a bounded budget.
        expanded_budget = _expanded_token_budget(body)
        if expanded_budget:
            retry_body = dict(body)
            retry_body["max_completion_tokens"] = expanded_budget
            response = _request_chat_json(url, headers, retry_body, timeout=settings.llm_timeout_seconds)
            content, choice, message = _chat_response_content(response)
    if not content.strip():
        raise _empty_content_error(response, choice, message)
    usage = response.get("usage") if isinstance(response, dict) else None
    return ChatCompletion(
        content=content,
        finish_reason=str(choice.get("finish_reason") or ""),
        usage=usage if isinstance(usage, dict) else {},
    )


def _chat_completion(settings: Settings, messages: list[dict[str, str]]) -> str:
    """Reply text only, for callers that do not inspect the finish reason."""
    return _chat_completion_with_meta(settings, messages).content


def _strip_code_fences(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    return stripped


def _closable_json_prefix(text: str) -> str | None:
    """Longest prefix of a truncated JSON object that can be closed as-is.

    Scans for the last point at which a nested value finished cleanly while
    the outer containers were still open, then closes those containers. That
    repairs a document cut off mid-element (the provider stopped at its token
    limit) by *dropping* the incomplete tail; it never invents field values,
    and returns ``None`` for anything that is not a bracket-balanced prefix
    of an object (mismatched brackets, prose, or nothing complete yet).
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    best_end = -1
    best_stack: tuple[str, ...] = ()
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if not stack or stack[-1] != char:
                return None
            stack.pop()
            if stack:
                best_end = index + 1
                best_stack = tuple(stack)
    if best_end < 0 or not best_stack:
        return None
    return text[:best_end] + "".join(reversed(best_stack))


def _repaired_json_object(stripped: str) -> dict[str, Any] | None:
    """Conservative repair of a JSON object truncated by an output-token cap."""
    if not stripped.startswith("{"):
        return None
    candidate = _closable_json_prefix(stripped)
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_json_object(content: str) -> dict[str, Any]:
    # Accept fenced JSON from less strict models, but still require a single
    # object so downstream report construction never consumes free-form prose.
    stripped = _strip_code_fences(content)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(stripped[start : end + 1])
            except json.JSONDecodeError as error:
                repaired = _repaired_json_object(stripped)
                if repaired is None:
                    raise LLMGenerationError(
                        f"LLM report draft is not valid JSON: {error}: {content[:500]}",
                        category="json_parse",
                    ) from error
                parsed = repaired
        else:
            repaired = _repaired_json_object(stripped)
            if repaired is None:
                raise LLMGenerationError(
                    f"LLM report draft is not JSON: {content[:500]}", category="json_parse"
                )
            parsed = repaired
    if not isinstance(parsed, dict):
        raise LLMGenerationError("LLM report draft must be a JSON object", category="invalid_response")
    return parsed


def _clean_actions(raw_actions: Any) -> list[str]:
    if isinstance(raw_actions, str):
        candidates = [raw_actions]
    elif isinstance(raw_actions, list):
        candidates = raw_actions
    else:
        candidates = []
    cleaned: list[str] = []
    for action in candidates:
        text = _sanitize_report_text(str(action).strip())
        if not text:
            continue
        if text not in cleaned:
            cleaned.append(text)
    return cleaned


def _sanitize_report_text(text: str) -> str:
    sanitized = text.strip()
    if not sanitized:
        return ""
    if _contains_non_contract_collection_reference(sanitized):
        return _allowed_tool_followup_sentence()
    return sanitized


def _allowed_tool_followup_sentence() -> str:
    return (
        "Run configured follow-up collection only with "
        + ", ".join(REQUIRED_TOOL_TYPES[:-1])
        + f", and {REQUIRED_TOOL_TYPES[-1]} as applicable."
    )


def _contains_non_contract_collection_reference(text: str) -> bool:
    return bool(unsupported_tool_tokens(text))


def _clean_string(value: object) -> str:
    return _sanitize_report_text(str(value).strip())


def _clean_key_findings(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    findings: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            text = _clean_string(item)
            if text:
                findings.append({"heading": "Finding", "items": [text]})
            continue
        if not isinstance(item, dict):
            continue
        heading = _clean_string(item.get("heading") or item.get("perspective") or "Finding")
        raw_items = item.get("items") or item.get("findings") or []
        if isinstance(raw_items, str):
            raw_items = [raw_items]
        if not isinstance(raw_items, list):
            raw_items = []
        items = [_clean_string(value) for value in raw_items]
        items = [value for value in items if value]
        if heading and items:
            findings.append({"heading": heading, "items": items[:5]})
    return findings[:8]


def _clean_recommendations(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    recommendations: list[dict[str, str]] = []
    for item in raw:
        horizon = ""
        if isinstance(item, str):
            action = _clean_string(item)
            rationale = "Prioritized from the supplied report evidence."
        elif isinstance(item, dict):
            action = _clean_string(item.get("action", ""))
            rationale = _clean_string(item.get("rationale", "")) or "Prioritized from the supplied report evidence."
            horizon = str(item.get("horizon", "")).strip().lower()
        else:
            continue
        if action:
            row = {"action": action, "rationale": rationale}
            # An unusable or absent horizon is left out here; report assembly
            # derives it deterministically from the action wording.
            if horizon in RECOMMENDATION_HORIZONS:
                row["horizon"] = horizon
            recommendations.append(row)
    return recommendations[:5]


def _clean_timeline(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    timeline: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        timestamp = str(item.get("timestamp", "")).strip()
        event = _clean_string(item.get("event", ""))
        significance = _clean_string(item.get("significance", ""))
        if timestamp and event and significance:
            timeline.append({"timestamp": timestamp, "event": event, "significance": significance})
    return timeline[:10]


def _clean_ttps(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    ttps: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        mitre_id = str(item.get("mitre_id", "")).strip().upper()
        if not re.fullmatch(r"T\d{4}(?:\.\d{3})?", mitre_id):
            continue
        tactic = _clean_string(item.get("tactic", ""))
        technique = _clean_string(item.get("technique", ""))
        evidence = _clean_string(item.get("evidence", ""))
        if tactic and technique and evidence:
            ttps.append({"mitre_id": mitre_id, "tactic": tactic, "technique": technique, "evidence": evidence})
    return ttps[:8]
