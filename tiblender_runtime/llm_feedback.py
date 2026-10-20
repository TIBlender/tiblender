"""LLM helpers for cross-cycle trend feedback: strategic value and dynamic queries.

Kept as a separate module from ``llm.py`` (already at its 800-line budget):
these two functions reuse llm.py's chat-completion plumbing and JSON-object
extraction but own their own prompts and result shapes.
"""

from __future__ import annotations

import json
from typing import Any, Callable, TypeVar

from .llm import LLMGenerationError, _chat_completion, _extract_json_object
from .llm_fallbacks import category_of
from .settings import Settings

_T = TypeVar("_T")

STRATEGIC_VALUE_PROMPT = (
    "Assess the strategic value of this threat cluster for early warning (0.0-1.0): "
    "novelty, potential impact, actionability. Use only the supplied cluster and context "
    "evidence; never invent facts not present in the payload."
)

DYNAMIC_QUERY_PROMPT = (
    "Generate collection queries adapted to the threat's language, geographic scope and "
    "context. Use only the supplied cluster and trend evidence; never invent IoCs, brands, "
    "campaign names, or platforms. Every query must reference a verified indicator, CVE, or "
    "topic token from the supplied cluster."
)


def _strategic_value_messages(cluster_payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                STRATEGIC_VALUE_PROMPT
                + "\n\n"
                "Return strict JSON only with keys strategic_value (a float between 0.0 and 1.0) "
                "and rationale (a short string)."
            ),
        },
        {"role": "user", "content": json.dumps(cluster_payload, ensure_ascii=False, indent=2)},
    ]


def _retry_once(call: Callable[[], _T]) -> _T:
    """Run ``call``; on a generation failure run it once more.

    A truncated or malformed draft is usually transient, so every per-unit
    caller gets one retry before it falls back to its deterministic path.
    The second failure is raised with both reasons so the fallback record
    names what actually happened.
    """
    try:
        return call()
    except LLMGenerationError as first_error:
        reason = str(first_error)
    try:
        return call()
    except LLMGenerationError as error:
        # The retry's own category is the one that decided the fallback.
        raise LLMGenerationError(
            f"{reason} / retry: {error}", category=category_of(error)
        ) from error


def strategic_value_with_llm(settings: Settings, cluster_payload: dict[str, Any]) -> float:
    """LLM variant of :func:`trends.strategic_value`. Raises on any malformed draft."""
    return _retry_once(lambda: _strategic_value_once(settings, cluster_payload))


def _strategic_value_once(settings: Settings, cluster_payload: dict[str, Any]) -> float:
    parsed = _extract_json_object(_chat_completion(settings, _strategic_value_messages(cluster_payload)))
    try:
        value = float(parsed["strategic_value"])
    except (KeyError, TypeError, ValueError) as error:
        raise LLMGenerationError(
            "LLM strategic value draft must include a numeric strategic_value field",
            category="invalid_response",
        ) from error
    return max(0.0, min(1.0, value))


def _dynamic_query_messages(trend_payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                DYNAMIC_QUERY_PROMPT
                + "\n\n"
                'Return strict JSON only with key "queries": an array of objects with query, '
                "platforms (a subset of x, reddit), language, and rationale. Only X and Reddit are "
                "collected by keyword search; Telegram and Discord are collected exhaustively from "
                "the communities the account has joined, so never propose them as query platforms."
            ),
        },
        {"role": "user", "content": json.dumps(trend_payload, ensure_ascii=False, indent=2)},
    ]


def generate_dynamic_queries_with_llm(settings: Settings, trend_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """LLM variant that proposes dynamic queries for one trend signal.

    Returns the raw ``queries`` array; the caller (``dynamic_queries.py``) is
    responsible for validating each entry against the cluster's evidence.
    """
    return _retry_once(lambda: _dynamic_queries_once(settings, trend_payload))


def _dynamic_queries_once(settings: Settings, trend_payload: dict[str, Any]) -> list[dict[str, Any]]:
    parsed = _extract_json_object(_chat_completion(settings, _dynamic_query_messages(trend_payload)))
    raw_queries = parsed.get("queries")
    if not isinstance(raw_queries, list):
        raise LLMGenerationError(
            "LLM dynamic query draft must include a queries array", category="invalid_response"
        )
    return raw_queries
