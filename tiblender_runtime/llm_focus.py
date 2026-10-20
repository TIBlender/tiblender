"""Subtask (i) of the Step 3 perspective agents: focus-type selection.

Kept out of :mod:`tiblender_runtime.llm_agents` (which stays under its
800-line budget) because focus selection carries its own prompt catalogue,
name normalisation and follow-up request. Padding a single-focus answer is
the caller's job (``perspectives._focus_types_with_notes``); everything here
tries to make padding unnecessary in the first place.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from . import llm as _llm
from .contract import FOCUS_TYPE_DESCRIPTIONS, PERSPECTIVE_AGENT_PROMPT, REQUIRED_FOCUS_TYPES
from .llm import LLMGenerationError
from .llm_fallbacks import category_of
from .iocs import filter_reportable_iocs
from .models import Cluster, to_jsonable
from .settings import Settings

logger = logging.getLogger(__name__)


def _perspective_payload(cluster: Cluster, perspective: str) -> dict[str, Any]:
    return {
        "perspective": perspective,
        "available_focus_types": list(REQUIRED_FOCUS_TYPES[perspective]),
        "cluster": {
            "id": cluster.id,
            "topic_name": cluster.topic_name,
            "description": cluster.description,
            "severity": cluster.severity.value,
            "threat_category": cluster.threat_category.value,
            "verified_iocs": to_jsonable(filter_reportable_iocs(cluster.iocs)),
            "platforms": sorted({post.platform for post in cluster.posts}),
            "posts": [
                {"id": post.id, "platform": post.platform, "normalized_content": post.normalized_content[:600]}
                for post in cluster.posts[:8]
            ],
        },
    }


MAX_SELECTED_FOCUS_TYPES = 4
MIN_SELECTED_FOCUS_TYPES = 2

_FOCUS_LEADING_ENUMERATION_RE = re.compile(r"^\s*\d+\s*[.)\-:]\s*")
_FOCUS_TRAILING_NUMBER_RE = re.compile(r"_\d+$")


def focus_alias_map(perspective: str) -> dict[str, str]:
    """Normalized spellings that resolve to a focus id of ``perspective``.

    Every alias is derived from the id itself, never hand-maintained: the id
    verbatim and the human label the id reads as ("DNS And Passive Dns
    Analysis") both normalize onto the same key, so an otherwise valid answer
    written with the label is accepted instead of discarded.
    """
    aliases: dict[str, str] = {}
    for focus in REQUIRED_FOCUS_TYPES[perspective]:
        aliases[focus] = focus
        aliases[_normalize_focus_token(focus.replace("_", " "))] = focus
    return aliases


def _normalize_focus_token(raw: str) -> str:
    token = str(raw).strip().lower()
    token = _FOCUS_LEADING_ENUMERATION_RE.sub("", token)
    token = token.replace("-", "_")
    token = re.sub(r"\s+", "_", token)
    token = re.sub(r"[^a-z0-9_]+", "", token)
    return re.sub(r"_+", "_", token).strip("_")


def normalize_focus_name(raw: str, perspective: str) -> str:
    """Resolve ``raw`` to a focus id of ``perspective``, or "" when it is none.

    Accepts an exact id, an id with a trailing period or index number, and the
    human label of an id, in any casing and with spaces or hyphens for
    underscores.
    """
    aliases = focus_alias_map(perspective)
    token = _normalize_focus_token(raw)
    if token in aliases:
        return aliases[token]
    trimmed = _FOCUS_TRAILING_NUMBER_RE.sub("", token)
    return aliases.get(trimmed, "")


def _focus_catalog(perspective: str) -> str:
    return "\n".join(
        f"- {focus}: {FOCUS_TYPE_DESCRIPTIONS[focus]}" for focus in REQUIRED_FOCUS_TYPES[perspective]
    )


def _focus_selection_instructions(perspective: str) -> str:
    focuses = REQUIRED_FOCUS_TYPES[perspective]
    maximum = min(MAX_SELECTED_FOCUS_TYPES, len(focuses))
    example = json.dumps(
        {
            "selected_focus_types": [
                {"focus_type": focuses[0], "reasoning": "one short sentence"},
                {"focus_type": focuses[1], "reasoning": "one short sentence"},
            ]
        },
        ensure_ascii=False,
    )
    return (
        f"You are the {perspective} perspective agent. Subtask (i): choose the investigation focuses "
        "that the supplied cluster evidence actually supports.\n"
        f"The only allowed focus ids for the {perspective} perspective are:\n"
        f"{_focus_catalog(perspective)}\n"
        f"Select at least {MIN_SELECTED_FOCUS_TYPES} and at most {maximum} focuses, using the ids verbatim "
        "exactly as written above. Never invent an id, never reword one, and never select an id from "
        "another perspective.\n"
        'JSON schema: {"selected_focus_types": [{"focus_type": "<one of the ids above>", '
        '"reasoning": "<why this cluster supports it>"}]}\n'
        f"Example of a valid answer: {example}\n"
        "Return strict JSON only, with no prose around it."
    )


def _focus_selection_messages(cluster: Cluster, perspective: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PERSPECTIVE_AGENT_PROMPT + "\n\n" + _focus_selection_instructions(perspective)},
        {"role": "user", "content": json.dumps(_perspective_payload(cluster, perspective), ensure_ascii=False, indent=2)},
    ]


def _additional_focus_messages(cluster: Cluster, perspective: str, selected: list[str]) -> list[dict[str, str]]:
    remaining = [focus for focus in REQUIRED_FOCUS_TYPES[perspective] if focus not in selected]
    catalog = "\n".join(f"- {focus}: {FOCUS_TYPE_DESCRIPTIONS[focus]}" for focus in remaining)
    return [
        {
            "role": "system",
            "content": (
                PERSPECTIVE_AGENT_PROMPT
                + "\n\n"
                f"You are the {perspective} perspective agent. "
                + (
                    f"You selected only {', '.join(selected)}, but this "
                    if selected
                    else "You selected no focus at all, but this "
                )
                + f"investigation needs at least {MIN_SELECTED_FOCUS_TYPES} focuses. Choose at least "
                f"{max(MIN_SELECTED_FOCUS_TYPES - len(selected), 1)} more focus for the same cluster from "
                "the remaining ids, using the ids verbatim. Pick the ones this cluster's evidence comes "
                "closest to supporting; an imperfect but relevant focus is better than none:\n"
                f"{catalog}\n"
                'JSON schema: {"selected_focus_types": [{"focus_type": "<one of the remaining ids>", '
                '"reasoning": "<why this cluster supports it>"}]}\n'
                "Return strict JSON only, with no prose around it."
            ),
        },
        {"role": "user", "content": json.dumps(_perspective_payload(cluster, perspective), ensure_ascii=False, indent=2)},
    ]


def _parse_focus_selection(content: str, perspective: str) -> list[str]:
    parsed = _llm._extract_json_object(content)
    raw_focuses = parsed.get("selected_focus_types")
    if not isinstance(raw_focuses, list):
        raise LLMGenerationError(
            "LLM focus selection draft must include a selected_focus_types array", category="invalid_response"
        )
    selected: list[str] = []
    for item in raw_focuses:
        raw = item.get("focus_type", "") if isinstance(item, dict) else item
        focus = normalize_focus_name(raw, perspective)
        if focus and focus not in selected:
            selected.append(focus)
    return selected


def select_focus_types_with_llm(settings: Settings, cluster: Cluster, perspective: str) -> list[str]:
    # Subtask (i) of the perspective agent: choose the investigation focuses.
    # Only contract focus types of the requesting perspective are accepted.
    if perspective not in REQUIRED_FOCUS_TYPES:
        # A programming error, not a provider failure: no cause bucket fits.
        raise LLMGenerationError(f"unknown investigation perspective: {perspective}", category="unknown")
    selected = _parse_focus_selection(
        _llm._chat_completion(settings, _focus_selection_messages(cluster, perspective)), perspective
    )
    if len(selected) < MIN_SELECTED_FOCUS_TYPES:
        # An under-answer is not a failure. On a thin cluster the agent often
        # replies with one focus, or with none at all ("no infrastructure
        # evidence here"); asked again with the remaining ids spelled out and
        # permission to pick the closest relevant focus, it answers. That
        # extra request is an LLM call, so it counts as neither padding nor a
        # fallback -- both of which only apply if it too comes back short.
        selected = _extend_focus_selection(settings, cluster, perspective, selected)
    if not selected:
        raise LLMGenerationError(
            f"LLM focus selection returned no valid {perspective} focus types", category="invalid_response"
        )
    return selected[:MAX_SELECTED_FOCUS_TYPES]


def _extend_focus_selection(
    settings: Settings,
    cluster: Cluster,
    perspective: str,
    selected: list[str],
) -> list[str]:
    try:
        additional = _parse_focus_selection(
            _llm._chat_completion(settings, _additional_focus_messages(cluster, perspective, selected)),
            perspective,
        )
    except LLMGenerationError as error:
        # Content-free at INFO: the raise-site category only. The failure text
        # may embed the model's draft of scraped posts, so it stays at DEBUG.
        logger.info(
            "%s focus follow-up request returned no additional focuses for cluster %s (reason=%s)",
            perspective,
            cluster.id,
            category_of(error),
        )
        logger.debug(
            "%s focus follow-up detail for cluster %s: %s",
            perspective,
            cluster.id,
            error,
        )
        return selected
    extended = list(selected)
    for focus in additional:
        if focus not in extended:
            extended.append(focus)
    return extended
