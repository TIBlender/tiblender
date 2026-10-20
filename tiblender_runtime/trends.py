"""Cross-cycle trend detection: rolling cluster snapshots and the four trend axes.

Implements the paper's cross-cycle trend-based query adaptation: a rolling
history of cluster snapshots (window ``W``), an adjacency criterion
``sim(c, h) >= tau_sim``, and four trend axes evaluated in parallel per
current snapshot: Volume, Propagation, Escalation, and Early_Warning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .clustering import HASHING_BACKEND_LABEL, HASHING_TOKEN_PATTERN
from .models import Cluster, EvaluationDecision, Severity

logger = logging.getLogger(__name__)

TREND_TYPES = ("VOLUME", "PROPAGATION", "ESCALATION", "EARLY_WARNING")

TREND_MINILM_MODEL = "all-MiniLM-L6-v2"

_SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}


@dataclass(slots=True)
class ClusterSnapshot:
    cluster_id: str
    cycle_index: int
    observed_at: str
    embedding: list[float]
    iocs: list[str]
    cves: list[str]
    severity: str
    platforms: list[str]
    post_count: int
    category: str
    topic_name: str
    strategic_value: float
    languages: list[str]


@dataclass(frozen=True, slots=True)
class TrendSignal:
    # Frozen: callers (dynamic_queries.generate_dynamic_queries, and tests)
    # hold references to signals returned by detect_trends. Recording an LLM
    # fallback marker must never mutate a caller-owned instance in place;
    # use dataclasses.replace(signal, evidence={**signal.evidence, ...}) to
    # produce a new TrendSignal instead.
    cluster_id: str
    trend_types: list[str]
    adjacent_history: int
    evidence: dict[str, Any] = field(default_factory=dict)


def _snapshot_text(cluster: Cluster) -> str:
    parts = [f"{cluster.topic_name}. {cluster.description}"]
    for post in cluster.posts[:3]:
        text = post.normalized_content or post.post.content
        if text:
            parts.append(text)
    return " ".join(parts)


def snapshot_cluster(
    cluster: Cluster,
    *,
    cycle_index: int,
    observed_at: str,
    strategic_value: float,
    embed: Callable[[str], list[float]],
) -> ClusterSnapshot:
    languages = sorted(
        {
            post.language
            for post in cluster.posts
            if post.language and post.language not in {"unknown", ""}
        }
    )
    return ClusterSnapshot(
        cluster_id=cluster.id,
        cycle_index=cycle_index,
        observed_at=observed_at,
        embedding=embed(_snapshot_text(cluster)),
        iocs=cluster.iocs.all_values(),
        cves=sorted(set(cluster.iocs.cves)),
        severity=cluster.severity.value,
        platforms=sorted({post.platform for post in cluster.posts if post.platform}),
        post_count=len(cluster.posts),
        category=cluster.threat_category.value,
        topic_name=cluster.topic_name,
        strategic_value=strategic_value,
        languages=languages,
    )


@lru_cache(maxsize=4)
def load_trend_model(model_name: str, local_files_only: bool):
    """Load (once per process) the sentence-transformers model for trend snapshots.

    ``feedback._embed_for`` calls :func:`trend_embedding` once per cluster
    snapshot, so constructing the model inside it rebuilt it 35 times for a
    bundled three-cycle run. The cache is keyed by the arguments that pick the
    weights, so switching model or offline mode still reloads. Failures are not
    cached (``lru_cache`` does not memoize exceptions), so the hashing fallback
    stays available on every call.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, local_files_only=local_files_only)


def trend_embedding_with_backend(text: str, *, backend: str, features: int = 384) -> tuple[list[float], str]:
    """Embed one snapshot text, and report which backend actually produced it.

    The label follows `clustering._embedding_matrix`'s convention so the two
    provenance strings can be read side by side: ``sklearn_hashing_vectorizer``,
    ``minilm:all-MiniLM-L6-v2``, or
    ``minilm:...->sklearn_hashing_vectorizer_fallback:<ExcName>`` when the model
    could not be loaded. Callers that only need the vector use
    :func:`trend_embedding`.
    """
    normalized_backend = (backend or "hashing").strip().lower()
    if normalized_backend in {"minilm", "all-minilm-l6-v2", "sentence_transformers"}:
        try:
            model = load_trend_model(TREND_MINILM_MODEL, True)
            vector = model.encode(
                [text],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            return [float(value) for value in vector], f"{normalized_backend}:{TREND_MINILM_MODEL}"
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "trend embedding backend 'minilm' is unavailable (%s); falling back to hashing", error
            )
            return (
                _hashing_embedding(text, features=features),
                f"{normalized_backend}:{TREND_MINILM_MODEL}"
                f"->{HASHING_BACKEND_LABEL}_fallback:{type(error).__name__}",
            )
    return _hashing_embedding(text, features=features), HASHING_BACKEND_LABEL


def trend_embedding(text: str, *, backend: str, features: int = 384) -> list[float]:
    """Embed one snapshot text with the configured trend embedding backend.

    ``"hashing"`` (the default) uses a deterministic scikit-learn
    HashingVectorizer. ``"minilm"`` uses sentence-transformers'
    all-MiniLM-L6-v2 model loaded from local files only, falling back to
    hashing (with a warning) when the model cache is unavailable.
    """
    vector, _backend_used = trend_embedding_with_backend(text, backend=backend, features=features)
    return vector


def _hashing_embedding(text: str, *, features: int) -> list[float]:
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(
        n_features=features,
        alternate_sign=False,
        norm="l2",
        token_pattern=HASHING_TOKEN_PATTERN,
    )
    matrix = vectorizer.transform([text]).toarray()
    return [float(value) for value in matrix[0]]


def _parse_observed_at(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def load_history(path: Path, *, now: datetime, window_hours: int) -> list[ClusterSnapshot]:
    """Load cluster snapshots observed within ``window_hours`` of ``now``.

    The window is closed on both sides: snapshots older than the cutoff are
    dropped, and so are snapshots observed AFTER ``now``. The future side
    matters whenever the clock is pinned (offline runs pin it to the cycle's
    own window end): a re-run into an output root that still holds a previous
    run's later-cycle snapshots would otherwise let cycle 1 "remember" cycles
    2 and 3, changing its trend counts and dynamic queries.

    Each line is one JSON-encoded :class:`ClusterSnapshot`. Malformed lines are
    skipped with a warning (and counted) rather than aborting the whole load.
    """
    path = Path(path)
    if not path.exists():
        return []
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    cutoff = now - timedelta(hours=window_hours)
    snapshots: list[ClusterSnapshot] = []
    skipped = 0
    from_the_future = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            snapshot = ClusterSnapshot(**payload)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            skipped += 1
            logger.warning(
                "cluster snapshot history line %d in %s is malformed (%s); skipping", line_number, path, error
            )
            continue
        observed_at = _parse_observed_at(snapshot.observed_at)
        if observed_at is None or observed_at < cutoff:
            continue
        if observed_at > now:
            from_the_future += 1
            continue
        snapshots.append(snapshot)
    if skipped:
        logger.warning("skipped %d malformed cluster snapshot history line(s) in %s", skipped, path)
    if from_the_future:
        logger.warning(
            "dropped %d cluster snapshot(s) in %s observed after the current clock (%s)",
            from_the_future,
            path,
            now.isoformat(),
        )
    return snapshots


def append_history(path: Path, snapshots: list[ClusterSnapshot]) -> None:
    if not snapshots:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(snapshot), ensure_ascii=False, sort_keys=True) for snapshot in snapshots]
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def cosine(a: list[float], b: list[float]) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not denom:
        return 0.0
    return float(np.dot(left, right) / denom)


def detect_trends(
    current: list[ClusterSnapshot],
    history: list[ClusterSnapshot],
    *,
    tau_sim: float,
    k: float,
    theta_ew: float,
    min_similar_snapshots: int = 4,
    volume_min_posts: int = 4,
) -> list[TrendSignal]:
    """Evaluate the four trend axes for every current snapshot.

    Adjacency is ``cosine(c, h) >= tau_sim``. Volume compares this cluster's
    own **post count** against the population mean/std of the post counts of
    the similar historical snapshots; Propagation looks for platforms not seen
    in adjacent history; Escalation looks for a severity increase over
    adjacent history; and Early_Warning simply thresholds the snapshot's
    strategic value.
    """
    signals: list[TrendSignal] = []

    for snapshot in current:
        adjacent_history = [h for h in history if cosine(snapshot.embedding, h.embedding) >= tau_sim]

        trend_types: list[str] = []
        evidence: dict[str, Any] = {"adjacent": len(adjacent_history)}

        # VOLUME: the deployed Layer 4 statistic. The baseline is the post
        # counts of the similar historical snapshots, and the test subject is
        # this cluster's own post count -- not a count of mutually similar
        # clusters, which fired on every cluster that had no history at all
        # (mu = sigma = 0, current_count = 1) while a campaign recurring as one
        # cluster per cycle could never exceed its own flat baseline.
        # Two gates guard the statistic: a baseline thinner than
        # ``min_similar_snapshots`` is not worth a mu/sigma test, and a cluster
        # below ``volume_min_posts`` is too small to call a spike whatever the
        # baseline says. ``evidence["adjacent"]`` is the size of that baseline
        # -- what the deployed detector records as ``metadata.matched_snapshots``.
        # The comparison is ``>=``, exactly as deployed, so a cluster sitting
        # exactly on a zero-variance baseline qualifies.
        if len(adjacent_history) >= min_similar_snapshots:
            counts = np.array([h.post_count for h in adjacent_history], dtype=float)
            mu = float(counts.mean())
            sigma = float(counts.std())
            threshold = mu + k * sigma
            current_count = int(snapshot.post_count)
            evidence["mu"] = mu
            evidence["sigma"] = sigma
            evidence["threshold"] = threshold
            evidence["current_count"] = current_count
            if current_count >= threshold and current_count >= volume_min_posts:
                trend_types.append("VOLUME")

        # PROPAGATION: a platform not seen among any adjacent history snapshot.
        if adjacent_history:
            adjacent_platforms: set[str] = set()
            for h in adjacent_history:
                adjacent_platforms.update(h.platforms)
            new_platforms = sorted(set(snapshot.platforms) - adjacent_platforms)
            evidence["new_platforms"] = new_platforms
            if new_platforms:
                trend_types.append("PROPAGATION")

        # ESCALATION: severity strictly above the highest adjacent-history severity.
        if adjacent_history:
            max_history_severity = max(_SEVERITY_RANK[Severity(h.severity)] for h in adjacent_history)
            evidence["max_history_severity"] = max_history_severity
            if _SEVERITY_RANK[Severity(snapshot.severity)] > max_history_severity:
                trend_types.append("ESCALATION")

        # EARLY_WARNING: strategic value threshold, independent of history.
        evidence["strategic_value"] = snapshot.strategic_value
        if snapshot.strategic_value >= theta_ew:
            trend_types.append("EARLY_WARNING")

        if trend_types:
            signals.append(
                TrendSignal(
                    cluster_id=snapshot.cluster_id,
                    trend_types=trend_types,
                    adjacent_history=len(adjacent_history),
                    evidence=evidence,
                )
            )

    return signals


def strategic_value(cluster: Cluster, decision: EvaluationDecision | None, context_stats: dict) -> float:
    """Heuristic strategic-value score used for Early_Warning and as the LLM fallback."""
    quality = decision.final_quality_score if decision is not None else 0.0
    ioc_total = int(context_stats.get("ioc_total", 0) or 0)
    ioc_high = int(context_stats.get("ioc_high", 0) or 0)
    cve_component = 1.0 if cluster.iocs.cves else 0.0
    value = 0.5 * quality + 0.3 * (ioc_high / max(ioc_total, 1)) + 0.2 * cve_component
    return float(min(1.0, max(0.0, value)))
