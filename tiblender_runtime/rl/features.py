"""State features for the MultiAction DQN policy.

The 320-dimensional state is the concatenation of a 128-d cluster embedding, a
128-d investigation-context embedding, and a 64-d action-history embedding.
Every component is deterministic: text is folded into fixed buckets with a
stable hash so no vector database or model download is required.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from ..models import Cluster, Severity, ThreatCategory
from .actions import ACTION_SPACE, FOCUS_PERSPECTIVE, PERSPECTIVES


CLUSTER_DIM = 128
CONTEXT_DIM = 128
HISTORY_DIM = 64
STATE_DIM = CLUSTER_DIM + CONTEXT_DIM + HISTORY_DIM

_SEVERITIES: list[str] = [severity.value for severity in Severity]
_CATEGORIES: list[str] = [category.value for category in ThreatCategory]
_PLATFORMS: list[str] = ["x", "reddit", "discord", "telegram"]
_IOC_FIELDS: list[str] = ["urls", "domains", "ips", "hashes", "cves"]
_CONFIDENCES: list[str] = ["LOW", "MEDIUM", "HIGH"]

# Trailing slots of the cluster embedding that hold structured (non-hashed) data.
_CLUSTER_SLOTS = len(_SEVERITIES) + len(_CATEGORIES) + len(_PLATFORMS) + len(_IOC_FIELDS)
_CLUSTER_HASH_DIM = CLUSTER_DIM - _CLUSTER_SLOTS

_MIN_TOKEN_CHARS = 3


def _hash_bucket(token: str, width: int) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % width


def cluster_tokens(cluster: Cluster) -> list[str]:
    """Lower-cased vocabulary of a cluster: topic, description, text, and IoCs."""
    parts = [cluster.topic_name, cluster.description, cluster.threat_category.value, cluster.severity.value]
    for post in cluster.posts:
        parts.append(post.normalized_content)
        parts.append(post.platform)
    parts.extend(cluster.iocs.all_values())
    text = " ".join(str(part) for part in parts).lower()
    for character in ",;:()[]{}<>\"'`|\\\n\t":
        text = text.replace(character, " ")
    return [token.strip(".-_/") for token in text.split() if len(token.strip(".-_/")) >= _MIN_TOKEN_CHARS]


def _platform_set(cluster: Cluster) -> set[str]:
    return {post.platform.strip().lower() for post in cluster.posts if post.platform}


def cluster_embedding(cluster: Cluster, dim: int = CLUSTER_DIM) -> np.ndarray:
    """Hashed bag of cluster tokens plus severity, category, platform, and IoC slots."""
    if dim <= _CLUSTER_SLOTS:
        raise ValueError(f"cluster embedding dim must exceed {_CLUSTER_SLOTS}")
    hash_dim = dim - _CLUSTER_SLOTS
    vector = np.zeros(dim, dtype=np.float64)
    for token in cluster_tokens(cluster):
        vector[_hash_bucket(token, hash_dim)] += 1.0
    norm = float(np.linalg.norm(vector[:hash_dim]))
    if norm:
        vector[:hash_dim] /= norm

    offset = hash_dim
    severity = cluster.severity.value
    if severity in _SEVERITIES:
        vector[offset + _SEVERITIES.index(severity)] = 1.0
    offset += len(_SEVERITIES)

    category = cluster.threat_category.value
    if category in _CATEGORIES:
        vector[offset + _CATEGORIES.index(category)] = 1.0
    offset += len(_CATEGORIES)

    platforms = _platform_set(cluster)
    for index, platform in enumerate(_PLATFORMS):
        if platform in platforms:
            vector[offset + index] = 1.0
    offset += len(_PLATFORMS)

    for index, field in enumerate(_IOC_FIELDS):
        vector[offset + index] = math.log1p(len(getattr(cluster.iocs, field)))
    return vector


def context_embedding(stats: dict, per_perspective: dict, dim: int = CONTEXT_DIM) -> np.ndarray:
    """Investigation progress so far; all-zero before the first tool result.

    Layout: 0-2 log1p IoC/CVE/TTP counts, 3-5 confidence shares (LOW/MEDIUM/
    HIGH), 6 overall success rate, 7-10 per-perspective coverage flags, 11-14
    per-perspective success rates. Remaining slots are reserved and stay zero.
    """
    vector = np.zeros(dim, dtype=np.float64)
    if not stats and not per_perspective:
        return vector

    vector[0] = math.log1p(max(0.0, float(stats.get("iocs", 0))))
    vector[1] = math.log1p(max(0.0, float(stats.get("cves", 0))))
    vector[2] = math.log1p(max(0.0, float(stats.get("ttps", 0))))

    confidence = stats.get("confidence") or {}
    confidence_total = float(sum(max(0, int(confidence.get(name, 0))) for name in _CONFIDENCES))
    if confidence_total:
        for index, name in enumerate(_CONFIDENCES):
            vector[3 + index] = max(0, int(confidence.get(name, 0))) / confidence_total

    results = float(stats.get("results", 0))
    if results > 0:
        vector[6] = float(stats.get("successes", 0)) / results

    for index, perspective in enumerate(PERSPECTIVES):
        bucket = per_perspective.get(perspective) or {}
        bucket_results = float(bucket.get("results", 0))
        vector[7 + index] = 1.0 if bucket_results > 0 else 0.0
        if bucket_results > 0:
            vector[11 + index] = float(bucket.get("successes", 0)) / bucket_results
    return vector


def history_embedding(actions: list[int], step: int, max_steps: int = 10, dim: int = HISTORY_DIM) -> np.ndarray:
    """Multi-hot of past actions, a one-hot step index, and summary counts."""
    if dim < len(ACTION_SPACE) + max_steps + 2 + len(PERSPECTIVES):
        raise ValueError("history embedding dim is too small for the action space")
    vector = np.zeros(dim, dtype=np.float64)
    valid = [index for index in actions if 0 <= index < len(ACTION_SPACE)]
    for index in valid:
        vector[index] = 1.0

    step_slot = len(ACTION_SPACE) + min(max(step, 0), max_steps - 1)
    vector[step_slot] = 1.0

    offset = len(ACTION_SPACE) + max_steps
    vector[offset] = min(1.0, len(valid) / max_steps)
    vector[offset + 1] = len(set(valid)) / len(ACTION_SPACE)
    for position, perspective in enumerate(PERSPECTIVES):
        count = sum(
            1
            for index in valid
            if index < len(ACTION_SPACE) - 1 and FOCUS_PERSPECTIVE[ACTION_SPACE[index]] == perspective
        )
        vector[offset + 2 + position] = min(1.0, count / max_steps)
    return vector


def state_vector(
    cluster: Cluster,
    stats: dict,
    per_perspective: dict,
    actions: list[int],
    step: int,
) -> np.ndarray:
    """Concatenate the cluster, context, and history embeddings into 320 dims."""
    return np.concatenate(
        (
            cluster_embedding(cluster),
            context_embedding(stats, per_perspective),
            history_embedding(actions, step),
        )
    )
