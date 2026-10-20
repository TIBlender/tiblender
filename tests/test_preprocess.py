import logging

from tiblender_runtime.collectors.base import RawPost
from tiblender_runtime.collectors.preprocess import (
    PreprocessDecision,
    default_ocr,
    has_security_keyword,
    merge_ocr_text,
    platform_noise_filter,
    preprocess,
)

SECURITY_TERMS = ["phishing", "ransomware", "stealer logs"]


def _post(platform="x", content="hello world", metadata=None, attachments=None, external_id="1"):
    return RawPost(
        platform=platform,
        external_id=external_id,
        content=content,
        posted_at="2026-09-01T00:10:00Z",
        metadata=metadata or {},
        attachments=attachments or [],
    )


# ---------------------------------------------------------------------------
# platform_noise_filter
# ---------------------------------------------------------------------------


class TestXNoiseFilter:
    def test_retweet_is_filtered(self):
        post = _post(platform="x", content="a real post about malware", metadata={"referenced_tweets": ["retweeted"]})
        assert platform_noise_filter(post) == "retweet"

    def test_low_text_after_stripping_urls_mentions_hashtags_is_filtered(self):
        post = _post(platform="x", content="@user #tag https://example.com/x")
        assert platform_noise_filter(post) == "no_text"

    def test_real_text_is_kept(self):
        post = _post(platform="x", content="New stealer campaign observed targeting banks")
        assert platform_noise_filter(post) is None


class TestRedditNoiseFilter:
    def test_stickied_is_filtered(self):
        post = _post(platform="reddit", content="weekly megathread", metadata={"stickied": True})
        assert platform_noise_filter(post) == "stickied"

    def test_removed_is_filtered_case_insensitive(self):
        post = _post(platform="reddit", content="[Removed]")
        assert platform_noise_filter(post) == "removed"

    def test_deleted_is_filtered(self):
        post = _post(platform="reddit", content="[deleted]")
        assert platform_noise_filter(post) == "removed"

    def test_normal_post_is_kept(self):
        post = _post(platform="reddit", content="New phishing kit targeting banks discovered today")
        assert platform_noise_filter(post) is None


class TestTelegramNoiseFilter:
    def test_joined_channel_service_message_is_filtered(self):
        post = _post(platform="telegram", content="Alice joined the channel")
        assert platform_noise_filter(post) == "service_message"

    def test_pinned_message_service_message_is_filtered(self):
        post = _post(platform="telegram", content="Bob pinned a message")
        assert platform_noise_filter(post) == "service_message"

    def test_joined_group_service_message_is_filtered(self):
        post = _post(platform="telegram", content="Carol joined the group")
        assert platform_noise_filter(post) == "service_message"

    def test_empty_content_is_filtered_as_generic_empty_not_service_message(self):
        # "service_message" is reserved for the actual service phrases; a
        # plain empty message falls through to the generic "empty" rule.
        post = _post(platform="telegram", content="")
        assert platform_noise_filter(post) == "empty"

    def test_normal_message_is_kept(self):
        post = _post(platform="telegram", content="Fresh stealer logs dump posted in this channel")
        assert platform_noise_filter(post) is None


class TestDiscordNoiseFilter:
    def test_bot_message_is_filtered(self):
        post = _post(platform="discord", content="a real message from a bot", metadata={"bot": True})
        assert platform_noise_filter(post) == "bot"

    def test_low_text_is_filtered(self):
        post = _post(platform="discord", content="@someone #tag https://example.com/x")
        assert platform_noise_filter(post) == "no_text"

    def test_normal_message_is_kept(self):
        post = _post(platform="discord", content="New ransomware affiliate program advertised today")
        assert platform_noise_filter(post) is None


class TestGenericEmptyFilter:
    def test_blank_reddit_post_is_filtered_as_empty(self):
        post = _post(platform="reddit", content="   ")
        assert platform_noise_filter(post) == "empty"


# ---------------------------------------------------------------------------
# has_security_keyword
# ---------------------------------------------------------------------------


def test_has_security_keyword_matches_term_case_insensitively():
    assert has_security_keyword("New PHISHING kit for sale", SECURITY_TERMS) is True


def test_has_security_keyword_matches_cve_regex():
    assert has_security_keyword("Actively exploited: CVE-2026-12345", []) is True


def test_has_security_keyword_matches_via_ioc_extraction():
    assert has_security_keyword("C2 endpoint is 45.33.32.156", []) is True


def test_has_security_keyword_returns_false_when_nothing_matches():
    assert has_security_keyword("Just a normal chat about lunch plans", SECURITY_TERMS) is False


# ---------------------------------------------------------------------------
# merge_ocr_text
# ---------------------------------------------------------------------------


def test_merge_ocr_text_uses_existing_ocr_text_from_fixture():
    post = _post(
        content="screenshot attached",
        attachments=[{"type": "image", "url": "https://example.com/a.png", "ocr_text": "login-example.net password"}],
    )
    merged = merge_ocr_text(post, ocr=None)
    assert merged.content == "screenshot attached\n[OCR] login-example.net password"
    assert merged.metadata["ocr_merged"] == 1
    assert merged is not post
    assert post.metadata == {}  # original never mutated


def test_merge_ocr_text_calls_ocr_callable_for_image_without_ocr_text():
    calls = []

    def fake_ocr(url):
        calls.append(url)
        return "extracted text"

    post = _post(content="screenshot", attachments=[{"type": "image", "url": "https://example.com/b.png"}])
    merged = merge_ocr_text(post, ocr=fake_ocr)
    assert calls == ["https://example.com/b.png"]
    assert merged.content == "screenshot\n[OCR] extracted text"
    assert merged.metadata["ocr_merged"] == 1


def test_merge_ocr_text_never_raises_when_ocr_callable_fails_for_one_image():
    def flaky_ocr(url):
        if url == "https://example.com/bad.png":
            raise RuntimeError("tesseract crashed")
        return "good text"

    post = _post(
        content="two screenshots",
        attachments=[
            {"type": "image", "url": "https://example.com/bad.png"},
            {"type": "image", "url": "https://example.com/good.png"},
        ],
    )
    merged = merge_ocr_text(post, ocr=flaky_ocr)
    assert merged.content == "two screenshots\n[OCR] good text"
    assert merged.metadata["ocr_merged"] == 1
    assert merged.metadata["ocr_failed"] == 1


def test_merge_ocr_text_records_failure_via_logging_and_metadata(caplog):
    def always_fails(url):
        raise RuntimeError("tesseract crashed")

    post = _post(
        content="one bad screenshot",
        attachments=[{"type": "image", "url": "https://example.com/bad.png"}],
    )
    with caplog.at_level("DEBUG", logger="tiblender_runtime.collectors.preprocess"):
        merged = merge_ocr_text(post, ocr=always_fails)

    # No text was extracted, so content is untouched, but the failure is
    # still surfaced on the returned post rather than silently swallowed.
    assert merged.content == post.content
    assert merged.metadata["ocr_failed"] == 1
    assert "ocr_merged" not in merged.metadata
    # The attachment URL is collected data and the OCR message can quote the
    # image's own text, so both stay at DEBUG; WARNING names the type only.
    warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert warnings
    assert all(
        "https://example.com/bad.png" not in record.getMessage() and "tesseract crashed" not in record.getMessage()
        for record in warnings
    )
    assert any("RuntimeError" in record.getMessage() for record in warnings)
    assert any(
        record.levelno == logging.DEBUG
        and "https://example.com/bad.png" in record.getMessage()
        and "tesseract crashed" in record.getMessage()
        for record in caplog.records
    )


def test_merge_ocr_text_unchanged_when_nothing_merges():
    post = _post(content="no attachments here")
    merged = merge_ocr_text(post, ocr=None)
    assert merged is post


def test_merge_ocr_text_skips_non_image_attachments_without_ocr_text():
    post = _post(content="a file", attachments=[{"type": "file", "url": "https://example.com/report.pdf"}])
    merged = merge_ocr_text(post, ocr=lambda url: "should not be called")
    assert merged is post


# ---------------------------------------------------------------------------
# default_ocr
# ---------------------------------------------------------------------------


def test_default_ocr_returns_none_when_pytesseract_missing(monkeypatch):
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "pytesseract":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert default_ocr() is None


# ---------------------------------------------------------------------------
# preprocess()
# ---------------------------------------------------------------------------


def test_preprocess_keeps_short_post_with_cve():
    post = _post(platform="x", content="CVE-2026-9999", external_id="short-cve")
    kept, stats = preprocess(posts=[post], min_text_chars=24, security_terms=SECURITY_TERMS, ocr=None)
    assert kept == [post]
    assert stats["short_no_keyword"] == 0
    assert stats["kept"] == 1


def test_preprocess_drops_short_post_without_keyword():
    post = _post(platform="x", content="lol nice", external_id="short-plain")
    kept, stats = preprocess(posts=[post], min_text_chars=24, security_terms=SECURITY_TERMS, ocr=None)
    assert kept == []
    assert stats["short_no_keyword"] == 1
    assert stats["kept"] == 0


def test_preprocess_stats_arithmetic():
    posts = [
        _post(platform="x", content="retweet post", metadata={"referenced_tweets": ["retweeted"]}, external_id="1"),
        _post(platform="reddit", content="[removed]", external_id="2"),
        _post(platform="x", content="lol", external_id="3"),
        _post(
            platform="x",
            content="incident write-up with screenshot",
            attachments=[{"type": "image", "url": "https://example.com/x.png", "ocr_text": "malicious-domain.example"}],
            external_id="4",
        ),
        _post(platform="x", content="A long enough normal post about a security incident today", external_id="5"),
    ]
    kept, stats = preprocess(posts=posts, min_text_chars=24, security_terms=SECURITY_TERMS, ocr=None)

    assert stats["input"] == 5
    assert stats["noise_filtered"] == 2  # retweet + removed
    assert stats["short_no_keyword"] == 1  # "lol"
    assert stats["ocr_merged"] == 1
    assert stats["kept"] == 2
    assert [post.post_id for post in kept] == ["x-4", "x-5"]


def test_preprocess_returns_preprocess_decision_compatible_reasons():
    # PreprocessDecision is part of the public interface even though preprocess()
    # itself returns (kept_posts, stats); exercise it directly for coverage.
    post = _post(platform="reddit", content="[removed]")
    decision = PreprocessDecision(keep=False, reason="removed", post=post)
    assert decision.keep is False
    assert decision.reason == "removed"
    assert decision.post is post
