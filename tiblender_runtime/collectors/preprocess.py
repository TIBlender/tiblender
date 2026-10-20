"""Step-1 preprocessing: platform noise filters, the short-post-without-security-
keyword rule, and OCR text merge.

``preprocess()`` runs each ``RawPost`` through, in order: a platform-specific
noise filter, an OCR text merge (so newly-merged OCR text can rescue an
otherwise-short post), and finally the short-and-no-keyword drop rule.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import logging
import re
from dataclasses import dataclass
from typing import Callable

from ..iocs import extract_iocs
from .base import RawPost
from .http import HttpClient

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)
_MIN_ALNUM_CHARS = 3

_TELEGRAM_SERVICE_PHRASES = ("joined the channel", "pinned a message", "joined the group")


@dataclass(frozen=True, slots=True)
class PreprocessDecision:
    keep: bool
    reason: str
    post: RawPost


def _strip_noise(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(" ", text)
    return text


def _alnum_count(text: str) -> int:
    return sum(1 for char in text if char.isalnum())


def _has_little_text(content: str) -> bool:
    return _alnum_count(_strip_noise(content)) < _MIN_ALNUM_CHARS


def platform_noise_filter(post: RawPost) -> str | None:
    content = post.content or ""
    platform = post.platform

    if platform == "x":
        if "retweeted" in (post.metadata.get("referenced_tweets") or []):
            return "retweet"
        if _has_little_text(content):
            return "no_text"
    elif platform == "reddit":
        if post.metadata.get("stickied"):
            return "stickied"
        if content.strip().lower().startswith(("[removed]", "[deleted]")):
            return "removed"
    elif platform == "telegram":
        lowered = content.lower()
        if any(phrase in lowered for phrase in _TELEGRAM_SERVICE_PHRASES):
            return "service_message"
    elif platform == "discord":
        if post.metadata.get("bot"):
            return "bot"
        if _has_little_text(content):
            return "no_text"

    if not content.strip():
        return "empty"
    return None


def has_security_keyword(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    if any(term.lower() in lowered for term in terms):
        return True
    if _CVE_RE.search(text):
        return True
    if extract_iocs(text).all_values():
        return True
    return False


def merge_ocr_text(post: RawPost, *, ocr: Callable[[str], str] | None) -> RawPost:
    merged_texts: list[str] = []
    failed_count = 0
    for attachment in post.attachments:
        existing_text = attachment.get("ocr_text") or ""
        if existing_text:
            merged_texts.append(existing_text)
            continue
        if ocr is not None and attachment.get("type") == "image" and attachment.get("url"):
            url = attachment["url"]
            try:
                extracted = ocr(url)
            except Exception as error:  # noqa: BLE001 - a failed OCR call must never abort collection
                failed_count += 1
                # The attachment URL comes from a collected post and the OCR
                # error can quote the image's own text, so WARNING names only
                # the exception type; the URL and the message go to DEBUG.
                logger.warning("OCR failed for one collected attachment (%s)", type(error).__name__)
                logger.debug("OCR failed for attachment url=%s: %s", url, error)
                continue
            if extracted:
                merged_texts.append(extracted)

    if not merged_texts and not failed_count:
        return post

    new_content = post.content
    new_metadata = dict(post.metadata)
    if merged_texts:
        new_content = new_content + "".join(f"\n[OCR] {text}" for text in merged_texts)
        new_metadata["ocr_merged"] = len(merged_texts)
    if failed_count:
        new_metadata["ocr_failed"] = failed_count
    return dataclasses.replace(post, content=new_content, metadata=new_metadata)


def default_ocr() -> Callable[[str], str] | None:
    """Live OCR callable backed by pytesseract + PIL, when both are importable."""
    if importlib.util.find_spec("pytesseract") is None or importlib.util.find_spec("PIL") is None:
        return None

    import io
    import urllib.request

    import pytesseract
    from PIL import Image

    client = HttpClient()

    def _ocr(url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": client.user_agent})
        with client.opener(request, timeout=client.timeout) as response:
            raw = response.read()
        image = Image.open(io.BytesIO(raw))
        return pytesseract.image_to_string(image).strip()

    return _ocr


def preprocess(
    posts: list[RawPost],
    *,
    min_text_chars: int,
    security_terms: list[str],
    ocr: Callable[[str], str] | None,
) -> tuple[list[RawPost], dict[str, int]]:
    stats = {"input": len(posts), "noise_filtered": 0, "short_no_keyword": 0, "kept": 0, "ocr_merged": 0}
    kept: list[RawPost] = []

    for post in posts:
        noise_reason = platform_noise_filter(post)
        if noise_reason is not None:
            stats["noise_filtered"] += 1
            continue

        merged_post = merge_ocr_text(post, ocr=ocr)
        if merged_post.metadata.get("ocr_merged"):
            stats["ocr_merged"] += 1

        is_short = len(merged_post.content.strip()) < min_text_chars
        if is_short and not has_security_keyword(merged_post.content, security_terms):
            stats["short_no_keyword"] += 1
            continue

        kept.append(merged_post)

    stats["kept"] = len(kept)
    return kept, stats
