from tiblender_runtime.clustering import cluster_posts
from tiblender_runtime.models import NormalizedPost, Post, Severity


def _post(post_id: str, content: str, summary: str, platform: str = "x") -> NormalizedPost:
    return NormalizedPost(
        post=Post(id=post_id, platform=platform, content=content, timestamp="2026-04-11T00:00:00Z"),
        normalized_content=summary,
    )


def test_dual_track_clustering_merges_shared_iocs_and_absorbs_context_posts():
    posts = [
        _post("x1", "C2 hxxps://ado-read-parser[.]com seen in Adobe zero-day", "Adobe Reader zero-day exploitation uses C2 domain ado-read-parser.com"),
        _post("r1", "Malicious PDF sample calls ado-read-parser[.]com", "Adobe Reader malicious PDF exploitation uses domain ado-read-parser.com", "reddit"),
        _post("d1", "JS engine abuse observed in Adobe Reader zero-day", "Adobe Reader zero-day uses JavaScript engine abuse and PDF delivery", "discord"),
        _post("x2", "Unrelated PayPay credential phishing", "PayPay phishing campaign steals credentials"),
    ]

    clusters = cluster_posts(posts, jaccard_threshold=0.25, absorption_threshold=0.18, min_cluster_size=2)

    adobe = next(cluster for cluster in clusters if "adobe" in cluster.topic_name.lower())
    assert {post.id for post in adobe.posts} == {"x1", "r1", "d1"}
    assert adobe.severity in {Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL}
    assert adobe.clustering_signals["embedding_backend"]
    assert adobe.clustering_signals["hdbscan_backend"]
    assert adobe.clustering_signals["non_ioc_absorbed"] == 1
    assert "tau_abs" in adobe.clustering_signals


def test_low_singletons_are_retained_for_filtering_accountability():
    posts = [_post("solo", "general security news", "General awareness post without actionable indicators")]

    clusters = cluster_posts(posts, min_cluster_size=3)

    assert len(clusters) == 1
    assert clusters[0].severity == Severity.LOW
    assert clusters[0].filter_reason == "below_min_cluster_size"


def test_non_ioc_track_uses_hdbscan_semantic_density():
    posts = [
        _post("p1", "Credential lure discussed in Japanese SMS", "Japanese mobile payment phishing lure steals credentials"),
        _post("p2", "Fake account verification kit reported", "Japanese payment phishing kit asks victims to verify credentials", "telegram"),
        _post("p3", "Patch notes for unrelated software", "Software maintenance update fixes minor bugs", "reddit"),
    ]

    clusters = cluster_posts(posts, min_cluster_size=2, min_samples=1, absorption_threshold=0.20)

    phishing = next(cluster for cluster in clusters if {post.id for post in cluster.posts} == {"p1", "p2"})
    assert phishing.clustering_signals["mode"] == "semantic_track"
    assert phishing.clustering_signals["hdbscan_backend"]
    assert phishing.clustering_signals["hdbscan_label"] >= 0


def test_reference_embedding_backend_falls_back_without_model(monkeypatch):
    def _offline_model(*args, **kwargs):
        raise RuntimeError("model cache is unavailable")

    monkeypatch.setattr("tiblender_runtime.clustering._sentence_transformer_embedding_matrix", _offline_model, raising=False)
    posts = [
        _post("x1", "C2 hxxps://ado-read-parser[.]com seen in Adobe zero-day", "Adobe Reader zero-day exploitation uses C2 domain ado-read-parser.com"),
        _post("r1", "Malicious PDF sample calls ado-read-parser[.]com", "Adobe Reader malicious PDF exploitation uses domain ado-read-parser.com", "reddit"),
        _post("d1", "JS engine abuse observed in Adobe Reader zero-day", "Adobe Reader zero-day uses JavaScript engine abuse and PDF delivery", "discord"),
    ]

    clusters = cluster_posts(
        posts,
        min_cluster_size=2,
        embedding_backend="bge_large",
        embedding_model="BAAI/bge-large-en-v1.5",
        embedding_features=1024,
        embedding_fallback=True,
    )

    assert clusters
    signals = clusters[0].clustering_signals
    assert signals["embedding_backend"].startswith("bge_large")
    assert signals["embedding_model"] == "BAAI/bge-large-en-v1.5"
    assert signals["embedding_features"] == 1024
    assert "fallback" in signals["embedding_backend"]
