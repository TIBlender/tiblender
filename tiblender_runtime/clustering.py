"""Dual-track clustering for IoC-anchored and semantic threat signals."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from typing import Iterable

from .iocs import extract_iocs
from .models import Cluster, IocSet, NormalizedPost, Severity, ThreatCategory

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{2,}", re.IGNORECASE)
# The one tokenizer every deterministic hashing embedding uses. Clustering and
# cross-cycle trend snapshots both call themselves "the hashing embedding" and
# are both 384-d, so they must tokenize identically or they are not the same
# vector space -- and the tau_sim=0.80 adjacency gate would be comparing
# distances measured with two different rulers. `trends._hashing_embedding`
# imports this constant rather than restating it.
HASHING_TOKEN_PATTERN = r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9_\-]{2,}\b"
# Provenance label recorded for a vector produced by that vectorizer.
HASHING_BACKEND_LABEL = "sklearn_hashing_vectorizer"
_STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "this",
    "that",
    "uses",
    "seen",
    "post",
    "signal",
    "context",
    "domain",
    "campaign",
    "threat",
    "malicious",
}
_KNOWN_BRANDS = ("Adobe Reader", "Adobe", "PayPay", "Microsoft", "Langflow", "Checkmarx", "CPUID")


class _UnionFind:
    """Merge post IDs that share enough IoC evidence before semantic grouping."""

    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _hashing_embedding_matrix(texts: list[str], n_features: int = 384):
    try:
        from sklearn.feature_extraction.text import HashingVectorizer

        # HashingVectorizer keeps clustering deterministic and avoids shipping a
        # fitted model with the compact runtime.
        vectorizer = HashingVectorizer(
            n_features=max(32, n_features),
            alternate_sign=False,
            norm="l2",
            token_pattern=HASHING_TOKEN_PATTERN,
        )
        matrix = vectorizer.transform(texts).toarray()
        return matrix, HASHING_BACKEND_LABEL
    except Exception:  # noqa: BLE001
        import numpy as np

        # Keep the clustering pipeline runnable in minimal environments where
        # scikit-learn is unavailable.
        matrix = np.zeros((len(texts), max(32, n_features)), dtype=float)
        for row, text in enumerate(texts):
            for token in _tokens(text):
                column = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % matrix.shape[1]
                matrix[row, column] += 1.0
            norm = math.sqrt(float((matrix[row] * matrix[row]).sum()))
            if norm:
                matrix[row] /= norm
        return matrix, "local_hashing_embedding"


def _sentence_transformer_embedding_matrix(
    texts: list[str],
    model_name: str,
    device: str = "",
    local_files_only: bool = True,
):
    # The reference profile can use BAAI/bge-large-en-v1.5 embeddings. The model
    # dependency is optional so the compact runtime can still run offline.
    from sentence_transformers import SentenceTransformer

    kwargs = {"local_files_only": local_files_only}
    if device:
        kwargs["device"] = device
    try:
        model = SentenceTransformer(model_name, **kwargs)
    except TypeError as exc:
        if local_files_only:
            raise RuntimeError("sentence-transformers local-only loading is unavailable") from exc
        kwargs.pop("local_files_only", None)
        model = SentenceTransformer(model_name, **kwargs)
    return model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def _embedding_matrix(
    posts: list[NormalizedPost],
    n_features: int = 384,
    backend: str = "hashing",
    model_name: str = "BAAI/bge-large-en-v1.5",
    fallback: bool = True,
    device: str = "",
    local_files_only: bool = True,
):
    texts = [post.normalized_content or post.post.content for post in posts]
    normalized_backend = (backend or "hashing").strip().lower()
    if normalized_backend in {"bge", "bge_large", "sentence_transformers", "reference"}:
        try:
            matrix = _sentence_transformer_embedding_matrix(
                texts,
                model_name,
                device=device,
                local_files_only=local_files_only,
            )
            return matrix, f"{normalized_backend}:{model_name}"
        except Exception as exc:  # noqa: BLE001
            if not fallback:
                raise
            matrix, fallback_backend = _hashing_embedding_matrix(texts, n_features=n_features)
            return matrix, f"{normalized_backend}:{model_name}->{fallback_backend}_fallback:{type(exc).__name__}"
    if normalized_backend not in {"hashing", "sklearn", "sklearn_hashing_vectorizer", "local_hashing"} and not fallback:
        raise ValueError(f"unsupported embedding backend: {backend}")
    matrix, hashing_backend = _hashing_embedding_matrix(texts, n_features=n_features)
    if normalized_backend in {"hashing", "sklearn", "sklearn_hashing_vectorizer", "local_hashing"}:
        return matrix, hashing_backend
    return matrix, f"{normalized_backend}->{hashing_backend}_fallback"


def _cosine(left, right) -> float:
    try:
        import numpy as np

        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        return float(np.dot(left, right) / denom) if denom else 0.0
    except Exception:  # noqa: BLE001
        dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
        right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _centroid(rows):
    try:
        import numpy as np

        return np.asarray(rows).mean(axis=0)
    except Exception:  # noqa: BLE001
        count = len(rows)
        width = len(rows[0]) if rows else 0
        return [sum(float(row[index]) for row in rows) / count for index in range(width)]


def _fallback_density_labels(matrix, min_cluster_size: int, epsilon: float) -> list[int]:
    # Small offline runs need deterministic grouping even when HDBSCAN cannot
    # be imported or does not find a dense component.
    labels = [-1 for _ in range(len(matrix))]
    next_label = 0
    similarity_threshold = max(0.20, 1.0 - epsilon - 0.55)
    for index, row in enumerate(matrix):
        if labels[index] != -1:
            continue
        neighbors = [
            other_index
            for other_index, other in enumerate(matrix)
            if _cosine(row, other) >= similarity_threshold
        ]
        if len(neighbors) < min_cluster_size:
            continue
        for neighbor in neighbors:
            labels[neighbor] = next_label
        next_label += 1
    return labels


def _density_labels(matrix, min_cluster_size: int, min_samples: int, epsilon: float, backend: str) -> tuple[list[int], str]:
    if len(matrix) < max(2, min_cluster_size):
        return [-1 for _ in range(len(matrix))], "insufficient_samples"
    if backend == "fallback":
        return _fallback_density_labels(matrix, min_cluster_size, epsilon), "local_density_hdbscan_fallback"
    try:
        from sklearn.cluster import HDBSCAN

        clusterer = HDBSCAN(
            min_cluster_size=max(2, min_cluster_size),
            min_samples=max(1, min_samples),
            cluster_selection_epsilon=max(0.0, epsilon),
            metric="euclidean",
        )
        labels = [int(label) for label in clusterer.fit_predict(matrix)]
        if any(label >= 0 for label in labels):
            return labels, "sklearn.cluster.HDBSCAN"
        fallback_labels = _fallback_density_labels(matrix, min_cluster_size, epsilon)
        if any(label >= 0 for label in fallback_labels):
            return fallback_labels, "sklearn.cluster.HDBSCAN+local_density_fallback"
        return labels, "sklearn.cluster.HDBSCAN"
    except Exception:  # noqa: BLE001
        return _fallback_density_labels(matrix, min_cluster_size, epsilon), "local_density_hdbscan_fallback"


def _ioc_similarity(left: IocSet, right: IocSet) -> float:
    return _jaccard(set(left.all_values()), set(right.all_values()))


def _cluster_id(post_ids: Iterable[str]) -> str:
    digest = hashlib.sha1(",".join(sorted(post_ids)).encode("utf-8")).hexdigest()[:12]
    return f"cluster-{digest}"


def _dominant_category(posts: list[NormalizedPost]) -> ThreatCategory:
    counts = Counter(post.threat_category for post in posts if post.threat_category is not ThreatCategory.OTHER)
    return counts.most_common(1)[0][0] if counts else ThreatCategory.OTHER


def _topic_name(posts: list[NormalizedPost], iocs: IocSet, category: ThreatCategory) -> str:
    text = " ".join(post.normalized_content + " " + post.post.content for post in posts)
    for cve in iocs.cves:
        return f"{cve} exploitation campaign"
    lowered = text.lower()
    for brand in _KNOWN_BRANDS:
        if brand.lower() in lowered:
            if "zero" in lowered and "day" in lowered:
                return f"{brand} zero-day exploitation"
            return f"{brand} {category.value.replace('_', ' ')} activity"
    domain = iocs.domains[0] if iocs.domains else ""
    if domain:
        return f"{domain} {category.value.replace('_', ' ')} campaign"
    return f"{category.value.replace('_', ' ').title()} cluster"


def _severity(posts: list[NormalizedPost], iocs: IocSet, category: ThreatCategory, min_cluster_size: int) -> Severity:
    platforms = {post.platform for post in posts}
    text = " ".join(post.normalized_content.lower() for post in posts)
    has_active = any(word in text for word in ("active", "exploitation", "victim", "stolen", "c2", "payload"))
    if len(posts) < min_cluster_size and not iocs.all_values():
        return Severity.LOW
    if len(platforms) >= 3 and has_active and (iocs.cves or iocs.hashes or iocs.ips or iocs.domains):
        return Severity.CRITICAL if category in {ThreatCategory.VULNERABILITY, ThreatCategory.MALWARE} else Severity.HIGH
    if len(platforms) >= 2 and iocs.all_values():
        return Severity.HIGH
    if iocs.all_values() or category is not ThreatCategory.OTHER:
        return Severity.MEDIUM
    return Severity.LOW


def _make_cluster(posts: list[NormalizedPost], min_cluster_size: int, signals: dict[str, object]) -> Cluster:
    iocs = IocSet.from_many([post.iocs for post in posts])
    category = _dominant_category(posts)
    severity = _severity(posts, iocs, category, min_cluster_size)
    filter_reason = ""
    if severity is Severity.LOW and len(posts) < min_cluster_size:
        filter_reason = "below_min_cluster_size"
    topic = _topic_name(posts, iocs, category)
    description = " / ".join(post.normalized_content for post in posts[:3])
    return Cluster(
        id=_cluster_id(post.id for post in posts),
        topic_name=topic,
        description=description,
        posts=sorted(posts, key=lambda item: item.id),
        iocs=iocs,
        severity=severity,
        threat_category=category,
        filter_reason=filter_reason,
        clustering_signals=signals,
    )


def cluster_posts(
    posts: list[NormalizedPost],
    *,
    jaccard_threshold: float = 0.25,
    absorption_threshold: float = 0.75,
    min_cluster_size: int = 3,
    min_samples: int = 2,
    cluster_epsilon: float = 0.08,
    merge_similarity_threshold: float = 0.92,
    merge_max_time_gap_seconds: int = 6 * 60 * 60,
    merge_min_context_overlap: int = 2,
    embedding_backend: str = "hashing",
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    embedding_features: int = 384,
    embedding_fallback: bool = True,
    embedding_local_files_only: bool = True,
    embedding_device: str = "",
    hdbscan_backend: str = "sklearn",
) -> list[Cluster]:
    if not posts:
        return []
    for post in posts:
        if not post.iocs.all_values():
            post.iocs = extract_iocs(f"{post.post.content}\n{post.normalized_content}")

    # Track 1 forms high-confidence anchors from exact/near-exact IoC overlap.
    by_id = {post.id: post for post in posts}
    post_order = list(posts)
    embeddings, embedding_backend_used = _embedding_matrix(
        post_order,
        n_features=embedding_features,
        backend=embedding_backend,
        model_name=embedding_model,
        fallback=embedding_fallback,
        device=embedding_device,
        local_files_only=embedding_local_files_only,
    )
    embedding_by_id = {post.id: embeddings[index] for index, post in enumerate(post_order)}
    ioc_posts = [post for post in posts if post.iocs.all_values()]
    non_ioc_posts = [post for post in posts if not post.iocs.all_values()]
    uf = _UnionFind(post.id for post in ioc_posts)

    for index, left in enumerate(ioc_posts):
        for right in ioc_posts[index + 1 :]:
            if _ioc_similarity(left.iocs, right.iocs) >= jaccard_threshold:
                uf.union(left.id, right.id)

    anchor_groups: dict[str, list[NormalizedPost]] = defaultdict(list)
    for post in ioc_posts:
        anchor_groups[uf.find(post.id)].append(post)

    anchor_vectors = {
        group_id: _centroid([embedding_by_id[post.id] for post in group_posts])
        for group_id, group_posts in anchor_groups.items()
    }
    dual_track_hdbscan_backend = "not_run"

    # Track 2 absorbs context-only posts into IoC anchors when the semantic
    # evidence is strong enough; the rest falls through to density clustering.
    unabsorbed: list[NormalizedPost] = []
    for post in non_ioc_posts:
        if not anchor_vectors:
            unabsorbed.append(post)
            continue
        best_group, best_score = max(
            ((group_id, _cosine(embedding_by_id[post.id], vector)) for group_id, vector in anchor_vectors.items()),
            key=lambda item: item[1],
        )
        if best_score >= absorption_threshold:
            anchor_groups[best_group].append(post)
            anchor_vectors[best_group] = _centroid([embedding_by_id[item.id] for item in anchor_groups[best_group]])
        else:
            unabsorbed.append(post)

    ioc_post_ids = {item.id for item in ioc_posts}
    clusters: list[Cluster] = []
    for group_id, group_posts in anchor_groups.items():
        clusters.append(
            _make_cluster(
                group_posts,
                min_cluster_size,
                {
                    "mode": "dual_track",
                    "ioc_track": True,
                    "embedding_backend": embedding_backend_used,
                    "embedding_model": embedding_model,
                    "embedding_features": embedding_features,
                    "embedding_local_files_only": embedding_local_files_only,
                    "hdbscan_backend": dual_track_hdbscan_backend,
                    "non_ioc_absorbed": sum(1 for post in group_posts if post.id not in ioc_post_ids),
                    "anchor": group_id,
                    "tau_j": jaccard_threshold,
                    "tau_abs": absorption_threshold,
                    "min_samples": min_samples,
                    "cluster_epsilon": cluster_epsilon,
                    "tau_merge": merge_similarity_threshold,
                    "merge_max_time_gap_seconds": merge_max_time_gap_seconds,
                    "merge_min_context_overlap": merge_min_context_overlap,
                },
            )
        )

    if unabsorbed:
        # Remaining posts are grouped semantically so campaigns without explicit
        # IoCs still produce reviewable clusters rather than disappearing.
        matrix = [embedding_by_id[post.id] for post in unabsorbed]
        labels, hdbscan_backend_used = _density_labels(matrix, min_cluster_size, min_samples, cluster_epsilon, hdbscan_backend)
        semantic_groups: dict[int, list[NormalizedPost]] = defaultdict(list)
        singleton_index = -1
        for post, label in zip(unabsorbed, labels, strict=False):
            if label < 0:
                semantic_groups[singleton_index].append(post)
                singleton_index -= 1
            else:
                semantic_groups[label].append(post)
    else:
        semantic_groups = {}

    for label, group in semantic_groups.items():
        clusters.append(
            _make_cluster(
                group,
                min_cluster_size,
                {
                    "mode": "semantic_track",
                    "ioc_track": False,
                    "embedding_backend": embedding_backend_used,
                    "embedding_model": embedding_model,
                    "embedding_features": embedding_features,
                    "embedding_local_files_only": embedding_local_files_only,
                    "hdbscan_backend": hdbscan_backend_used,
                    "hdbscan_label": label,
                    "tau_abs": absorption_threshold,
                    "min_samples": min_samples,
                    "cluster_epsilon": cluster_epsilon,
                    "tau_merge": merge_similarity_threshold,
                    "merge_max_time_gap_seconds": merge_max_time_gap_seconds,
                    "merge_min_context_overlap": merge_min_context_overlap,
                },
            )
        )

    return sorted(clusters, key=lambda cluster: (cluster.severity.value, -len(cluster.posts), cluster.topic_name))
