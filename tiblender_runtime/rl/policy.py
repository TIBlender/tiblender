"""RL-guided vs multi-perspective mode selection for Step 3."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import clock
from ..models import Cluster
from ..settings import Settings
from .actions import FOCUS_TYPES
from .features import cluster_embedding, state_vector
from .network import select_composite_action
from .trainer import load_network


logger = logging.getLogger(__name__)

DISABLED_BACKEND = "disabled"
MULTI_PERSPECTIVE = "multi_perspective"
RL_GUIDED = "rl_guided"

TEMPORAL_HALF_LIFE_DAYS = 7.0
PLATFORM_MATCH_THRESHOLD = 0.5
MIN_BRAND_CHARS = 4

SIMILARITY_WEIGHTS: dict[str, float] = {
    "cosine": 2.0,
    "ioc_jaccard": 3.0,
    "cve_jaccard": 3.0,
    "brand_jaccard": 1.5,
    "category_match": 1.0,
    "platform_match": 0.5,
    "temporal_decay": 1.0,
    "bias": -3.0,
}

_BRAND_RE = re.compile(r"^[A-Z][A-Za-z]*$")


@dataclass(slots=True)
class InvestigationStrategy:
    mode: str
    similarity_score: float
    action_priorities: list[str] = field(default_factory=list)
    focus_by_perspective: dict[str, list[str]] = field(default_factory=dict)
    q_values: dict[str, float] = field(default_factory=dict)
    matched_report_id: str = ""


def _now() -> datetime:
    return clock.now()


def _cold_start() -> InvestigationStrategy:
    return InvestigationStrategy(
        mode=MULTI_PERSPECTIVE,
        similarity_score=0.0,
        action_priorities=[],
        focus_by_perspective={},
        q_values={},
    )


def brand_tokens(text: str) -> list[str]:
    """Capitalised words of at least four characters, used as brand proxies."""
    tokens: list[str] = []
    for raw in str(text or "").split():
        token = raw.strip(".,:;()[]{}\"'`")
        if len(token) >= MIN_BRAND_CHARS and _BRAND_RE.match(token) and token not in tokens:
            tokens.append(token)
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return 0.0
    denominator = float(np.linalg.norm(left)) * float(np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _ClusterProfile:
    """Cluster-side similarity inputs, computed once per cluster."""

    embedding: np.ndarray
    iocs: frozenset[str]
    cves: frozenset[str]
    brands: frozenset[str]
    platforms: frozenset[str]
    category: str


def _profile(cluster: Cluster) -> _ClusterProfile:
    return _ClusterProfile(
        embedding=cluster_embedding(cluster),
        iocs=frozenset(cluster.iocs.urls + cluster.iocs.domains + cluster.iocs.ips + cluster.iocs.hashes),
        cves=frozenset(value.upper() for value in cluster.iocs.cves),
        brands=frozenset(brand_tokens(cluster.topic_name)),
        platforms=frozenset(post.platform.strip().lower() for post in cluster.posts if post.platform),
        category=cluster.threat_category.value,
    )


def _features(profile: _ClusterProfile, past: dict, now: datetime) -> dict[str, float]:
    created_at = _parse_timestamp(past.get("created_at"))
    if created_at is None:
        temporal_decay = 0.0
    else:
        age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
        temporal_decay = math.exp(-age_days / TEMPORAL_HALF_LIFE_DAYS)

    platform_overlap = _jaccard(set(profile.platforms), {str(name).lower() for name in past.get("platforms") or []})
    return {
        "cosine": _cosine(profile.embedding, np.asarray(past.get("embedding") or [], dtype=np.float64)),
        "ioc_jaccard": _jaccard(set(profile.iocs), {str(value) for value in past.get("iocs") or []}),
        "cve_jaccard": _jaccard(set(profile.cves), {str(value).upper() for value in past.get("cves") or []}),
        "brand_jaccard": _jaccard(set(profile.brands), {str(value) for value in past.get("brands") or []}),
        "category_match": 1.0 if profile.category == str(past.get("category", "")) else 0.0,
        "platform_match": 1.0 if platform_overlap >= PLATFORM_MATCH_THRESHOLD else 0.0,
        "temporal_decay": temporal_decay,
    }


def _logistic(features: dict[str, float]) -> float:
    logit = SIMILARITY_WEIGHTS["bias"] + sum(SIMILARITY_WEIGHTS[name] * value for name, value in features.items())
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))


def similarity_features(cluster: Cluster, past: dict, *, now: datetime) -> dict[str, float]:
    """Feature vector comparing a new cluster with one past report."""
    return _features(_profile(cluster), past, now)


def similarity_to_report(cluster: Cluster, past: dict, *, now: datetime) -> float:
    """Logistic similarity in [0, 1] between a cluster and a past report."""
    return _logistic(similarity_features(cluster, past, now=now))


def _best_match(cluster: Cluster, report_history: list[dict], now: datetime) -> tuple[float, str]:
    # The cluster-side profile is computed once; report_history can hold up to
    # 2,000 entries and re-embedding the cluster for each one would dominate.
    profile = _profile(cluster)
    best_score = 0.0
    best_id = ""
    for past in report_history:
        if not isinstance(past, dict):
            continue
        score = _logistic(_features(profile, past, now))
        if score > best_score:
            best_score = score
            best_id = str(past.get("report_id", ""))
    return best_score, best_id


def load_policy_state(policy_state_path: Path) -> dict | None:
    """Read and parse ``policy_state.json`` once, for reuse across clusters."""
    return _load_state(policy_state_path)


def _load_state(policy_state_path: Path) -> dict | None:
    if not policy_state_path.exists():
        return None
    try:
        state = json.loads(policy_state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as error:
        logger.warning("policy state at %s is unreadable (%s); using multi-perspective mode", policy_state_path, error)
        return None
    if not isinstance(state, dict):
        logger.warning("policy state at %s is not an object; using multi-perspective mode", policy_state_path)
        return None
    return state


def _rank_focus_types(q_focus: np.ndarray) -> list[str]:
    order = sorted(range(len(FOCUS_TYPES)), key=lambda index: (-q_focus[index], index))
    return [FOCUS_TYPES[index] for index in order]


def select_investigation_strategy(
    cluster: Cluster,
    settings: Settings,
    policy_state_path: Path,
    *,
    policy_state: dict | None = None,
) -> InvestigationStrategy:
    """Choose between the RL-guided ordering and the default multi-perspective sweep.

    RL guidance is used only when the cluster resembles a previously reported
    campaign (logistic similarity at or above ``rl_similarity_gate``); otherwise
    the full four-perspective investigation runs unchanged.

    ``policy_state`` lets a caller that investigates many clusters read and
    parse ``policy_state.json`` once and pass the same parsed object to every
    cluster, instead of re-reading the file per cluster. Passing it is purely
    an optimisation: the selection is identical either way.
    """
    if settings.rl_policy_backend.strip().lower() == DISABLED_BACKEND:
        return _cold_start()

    state = policy_state if policy_state is not None else _load_state(policy_state_path)
    if state is None:
        return _cold_start()
    report_history = state.get("report_history")
    if not isinstance(report_history, list) or not report_history:
        return _cold_start()

    score, matched_report_id = _best_match(cluster, report_history, _now())

    def fallback() -> InvestigationStrategy:
        return InvestigationStrategy(
            mode=MULTI_PERSPECTIVE,
            similarity_score=round(score, 4),
            matched_report_id=matched_report_id,
        )

    if score < settings.rl_similarity_gate:
        return fallback()

    network = load_network(state.get("network"), settings)
    features = state_vector(cluster, {}, {}, [], 0).reshape(1, -1)
    try:
        out = network.forward(features)
    except ValueError as error:
        logger.warning("stored policy network rejected the state vector (%s); using multi-perspective mode", error)
        return fallback()

    # Step 0: nothing has been investigated yet, so the stop head is not consulted
    # (select_composite_action only honours it from step 1 onwards).
    pairs = select_composite_action(out, step=0)

    q_focus = np.asarray(out["q_focus"], dtype=np.float64).reshape(-1)
    ranked = _rank_focus_types(q_focus)
    chosen = {focus for _, focus in pairs}
    selected = [focus for focus in ranked if focus in chosen]
    # Selected focuses lead the ordering; the remaining focuses follow in Q order
    # so Step 3 still has a full fallback ranking.
    priorities = (selected + [focus for focus in ranked if focus not in chosen])[: max(1, settings.rl_action_top_k)]

    # focus_by_perspective always reports the full composite action, even when
    # rl_action_top_k truncates the flat priority list.
    focus_by_perspective: dict[str, list[str]] = {}
    for perspective, focus in pairs:
        focus_by_perspective.setdefault(perspective, []).append(focus)
    scored = priorities + [focus for focus in selected if focus not in priorities]

    return InvestigationStrategy(
        mode=RL_GUIDED,
        similarity_score=round(score, 4),
        action_priorities=priorities,
        focus_by_perspective=focus_by_perspective,
        q_values={focus: round(float(q_focus[FOCUS_TYPES.index(focus)]), 4) for focus in scored},
        matched_report_id=matched_report_id,
    )
