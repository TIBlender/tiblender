from tiblender_runtime.models import Post
from tiblender_runtime.normalization import normalize_posts


def test_normalize_posts_uses_llm_english_summary_and_preserves_iocs():
    posts = [
        Post(
            id="p1",
            platform="x",
            content="Apple IDの認証を装うフィッシングを確認。hxxps://xyahyy[.]com/login を使用。",
            timestamp="2026-04-13T06:51:56Z",
            metadata={"language": "ja"},
        )
    ]

    normalized = normalize_posts(
        posts,
        normalized_content_by_id={
            "p1": "Apple ID phishing lure targeting Japanese users. IoCs: https://xyahyy.com/login"
        },
    )

    assert normalized[0].normalized_content.startswith("Apple ID phishing lure")
    assert "Apple IDの認証" not in normalized[0].normalized_content
    assert normalized[0].iocs.urls == ["https://xyahyy.com/login"]
    assert normalized[0].is_threat_related is True
