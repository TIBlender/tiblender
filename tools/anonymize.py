"""Text anonymization for building fixtures from real released post data.

`anonymize_text` strips anything that could identify a real person or lead
back to the original post (handles, emails, phone numbers, links to social
platforms) while preserving IoC-looking tokens (CVE ids, hashes, IPs, and
defanged/real domains) so downstream IoC extraction still works on the
anonymized fixture text.

`defang_text` rewrites any still-clickable indicator -- every
`http(s)://host/...` URL and every bare domain whose host is not a known
legitimate publisher/vendor host -- into the standard defanged form
(`hxxp://host[.]tld`, `host[.]tld`). The
bundled fixtures are excerpts of real released posts, so they carry real
campaign infrastructure; defanging keeps the shipped text non-clickable while
`tiblender_runtime.iocs.refang` still recovers the original indicator for
extraction, so nothing about the pipeline's behavior changes.

`pseudonym` derives a stable, non-reversible replacement handle/author name
for a real value (e.g. an author ref) so the same input always maps to the
same fixture-facing pseudonym.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiblender_runtime.iocs import VALID_TLDS
from tiblender_runtime.legitimate_hosts import REFERENCE_HOSTS

# Hosts that identify a specific social platform account/post; a URL to one
# of these (with or without scheme/www) is removed entirely rather than kept
# as a "real" IoC domain.
_SOCIAL_HOSTS = (
    "x.com",
    "twitter.com",
    "t.co",
    "t.me",
    "telegram.me",
    "telegram.org",
    "discord.gg",
    "discord.com",
    "discordapp.com",
    "discordapp.net",
    "reddit.com",
    "redd.it",
    "redditmedia.com",
    "redditstatic.com",
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "youtu.be",
    "linkedin.com",
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_HANDLE_RE = re.compile(r"@[A-Za-z0-9_]{2,}")

# Phone-shaped digit runs only: an optional leading "+", digit groups of 1-4
# digits (optionally parenthesized), separated by at most one space/./-
# between groups, repeated 1-4 times. A trailing digit-count check (done in
# `_strip_phone_numbers`) additionally requires >=9 digits so short numeric
# fragments (e.g. a bare "2026-03-27" date, which would otherwise satisfy
# this shape at only 8 digits) are left alone even if this regex matches.
_PHONE_RE = re.compile(r"\+?\(?\d{1,4}\)?(?:[\s.-]?\(?\d{2,4}\)?){1,4}")
_PHONE_MIN_DIGITS = 9

# Link-shortener hosts operated by a social platform: a URL to one is a
# pointer back to a specific post just as much as the platform host itself,
# so they are stripped on the same terms.
_SOCIAL_SHORTENER_HOSTS = (
    "lnkd.in",
    "fb.me",
)

_ALL_SOCIAL_HOSTS = _SOCIAL_HOSTS + _SOCIAL_SHORTENER_HOSTS

# Any subdomain depth is tolerated, not just `www.`: a platform serves
# post-derived media and permalinks from its own subdomains
# (`preview.redd.it`, `i.redd.it`, `v.redd.it`, `old.reddit.com`,
# `cdn.discordapp.com`), and a signed media URL is a *stronger* pointer back
# to the original post than the permalink is. Restricting the prefix to
# `www.` let `https://preview.redd.it/<id>.jpg?...&s=<signature>` ship
# clickable out of a bundle whose documented guarantee is that no
# social-post URL survives.
_SOCIAL_URL_RE = re.compile(
    r"(?<![A-Za-z0-9.-])(?:https?://)?(?:[A-Za-z0-9-]+\.)*(?:"
    + "|".join(re.escape(host) for host in _ALL_SOCIAL_HOSTS)
    + r")(?![A-Za-z0-9-])(?:/[^\s]*)?",
    re.IGNORECASE,
)

# A platform host mentioned in prose with neither a scheme nor a path
# ("see redd.it for the thread"). `_SOCIAL_URL_RE` already strips that shape,
# but the fixture-bundle guard scanned for `host/` and `//host` substrings
# only, so this one form was invisible to it. Exported for that guard. The
# leading and trailing look-arounds keep it off any host that *is* part of a
# URL, so a legitimately shipped API path is not re-flagged here.
_BARE_SOCIAL_HOST_RE = re.compile(
    r"(?<![A-Za-z0-9.:/@-])(?:[A-Za-z0-9-]+\.)*(?:"
    + "|".join(re.escape(host) for host in _ALL_SOCIAL_HOSTS)
    + r")(?![A-Za-z0-9:/@-])",
    re.IGNORECASE,
)

# Python's `\b` is Unicode-aware, so CJK characters (hiragana/katakana/kanji)
# count as "word" characters -- `\bCVE-...` then fails to match a CVE id
# immediately preceded by Japanese/Chinese/Korean text with no space -- a
# particle glued straight onto the id, which is common in the non-English
# source posts of this corpus -- silently letting the unprotected CVE id fall through to
# the phone-number stripper. These tokens are always ASCII by construction,
# so an explicit ASCII-only boundary (not `\b`) is used instead.
_LEFT_BOUND = r"(?<![A-Za-z0-9_])"
_RIGHT_BOUND = r"(?![A-Za-z0-9_])"

# Tokens that must survive anonymization even though they can look numeric
# (IPs, CVE ids, ISO dates/timestamps) or would otherwise collide with the
# phone-number stripper.
_ISO_DATETIME_RE = re.compile(
    _LEFT_BOUND + r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?" + _RIGHT_BOUND
)
_CVE_RE = re.compile(_LEFT_BOUND + r"CVE-\d{4}-\d{4,8}" + _RIGHT_BOUND, re.IGNORECASE)
_HASH_RE = re.compile(
    _LEFT_BOUND + r"[a-fA-F0-9]{64}" + _RIGHT_BOUND
    + "|" + _LEFT_BOUND + r"[a-fA-F0-9]{40}" + _RIGHT_BOUND
    + "|" + _LEFT_BOUND + r"[a-fA-F0-9]{32}" + _RIGHT_BOUND
)
_HXXP_RE = re.compile(_LEFT_BOUND + r"hxxps?://\S+", re.IGNORECASE)
_BRACKET_DOMAIN_RE = re.compile(
    _LEFT_BOUND + r"[a-zA-Z0-9-]+(?:\[\.\]|\(\.\)|\{\.\})[a-zA-Z0-9.\[\](){}-]*[a-zA-Z0-9\])}]"
)
_IP_RE = re.compile(_LEFT_BOUND + r"(?:\d{1,3}\.){3}\d{1,3}" + _RIGHT_BOUND)
_DOMAIN_RE = re.compile(_LEFT_BOUND + r"(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,24}" + _RIGHT_BOUND)

_PROTECT_RE = re.compile(
    "|".join(
        pattern.pattern
        for pattern in (
            _ISO_DATETIME_RE,
            _CVE_RE,
            _HASH_RE,
            _HXXP_RE,
            _BRACKET_DOMAIN_RE,
            _IP_RE,
            _DOMAIN_RE,
        )
    ),
    re.IGNORECASE,
)

_WHITESPACE_RE = re.compile(r"\s+")

_PSEUDONYM_PREFIX = "u"
_PSEUDONYM_DIGEST_LENGTH = 8


def _protect_ioc_tokens(text: str) -> tuple[str, dict[str, str]]:
    """Replace IoC-looking tokens with placeholders so later removal regexes
    (in particular the phone-number stripper) cannot mangle them.
    """
    placeholders: dict[str, str] = {}

    def _replace(match: re.Match) -> str:
        token = f"\x00IOC{len(placeholders)}\x00"
        placeholders[token] = match.group(0)
        return token

    protected = _PROTECT_RE.sub(_replace, text)
    return protected, placeholders


def _restore_ioc_tokens(text: str, placeholders: dict[str, str]) -> str:
    for token, original in placeholders.items():
        text = text.replace(token, original)
    return text


def _strip_phone_numbers(text: str) -> str:
    """Remove phone-shaped digit runs with >=9 total digits.

    The digit-count check (rather than relying on `_PHONE_RE` alone) is what
    keeps a short numeric fragment that happens to be phone-shaped -- e.g. a
    bare `2026-03-27` date, which is only 8 digits -- from being stripped
    even if it were not already protected as an ISO date/timestamp token.
    """

    def _replace(match: re.Match) -> str:
        digit_count = sum(character.isdigit() for character in match.group(0))
        return " " if digit_count >= _PHONE_MIN_DIGITS else match.group(0)

    return _PHONE_RE.sub(_replace, text)


# Defanging. `_FANGED_URL_RE` matches only a live scheme (never `hxxp://`,
# which is already defanged), and both patterns require a plain-dot hostname,
# so an already-defanged `evil[.]com` cannot match and be defanged twice.
_FANGED_URL_RE = re.compile(r"(?<![A-Za-z0-9.\-])(https?)://([A-Za-z0-9.\-]+\.[A-Za-z]{2,24})", re.IGNORECASE)
_BARE_DOMAIN_RE = re.compile(_LEFT_BOUND + r"(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,24}" + _RIGHT_BOUND)


def _is_social_host(host: str) -> bool:
    """True for a social/platform host (or any subdomain of one)."""
    lowered = host.lower().strip(".")
    return any(lowered == social or lowered.endswith("." + social) for social in _ALL_SOCIAL_HOSTS)


def _is_legitimate_host(host: str) -> bool:
    """True for a publisher/vendor host whose URL may ship clickable.

    Driven by ``legitimate_hosts.REFERENCE_HOSTS``, not by the runtime's wider
    ``LEGITIMATE_DOMAINS``. The two answer different questions: the runtime
    list says what is not an *indicator* (and grows as more publisher and CERT
    hosts turn up in recorded evidence), while this one says what may ship
    **clickable** in published fixture text. Widening an allowlist must never
    un-defang URLs in an already-published bundle, so this list stays put.

    ``REFERENCE_HOSTS`` also lists the social platforms, because the extractor
    must not treat `reddit.com` or `t.me` as campaign infrastructure. A
    post-derived platform URL is the one thing this module must never let
    through, so social hosts are excluded here regardless of that membership:
    whatever survives `_SOCIAL_URL_RE` is defanged rather than exempted.
    """
    lowered = host.lower().strip(".")
    if _is_social_host(lowered):
        return False
    return any(lowered == legit or lowered.endswith("." + legit) for legit in REFERENCE_HOSTS)


# Darknet service TLDs are not in the runtime's VALID_TLDS (they are never
# extracted as IoCs), but a leak-site address is exactly the kind of live
# infrastructure that must not ship clickable in fixture text.
_EXTRA_DEFANG_TLDS = frozenset({"onion", "i2p"})


def _should_defang(host: str) -> bool:
    """True for a *bare* host (no scheme) that must ship defanged.

    Restricted to hosts whose TLD the runtime itself recognizes
    (`VALID_TLDS`, plus darknet service TLDs), so ordinary prose that merely
    looks domain-shaped ("payload.Then", "config.yaml") is never rewritten,
    and excluding known publisher/vendor/platform domains
    (`legitimate_hosts.REFERENCE_HOSTS`), which stay readable as references.
    """
    lowered = host.lower().strip(".")
    if "." not in lowered or _is_legitimate_host(lowered):
        return False
    return lowered.rsplit(".", 1)[-1] in VALID_TLDS or lowered.rsplit(".", 1)[-1] in _EXTRA_DEFANG_TLDS


def _should_defang_url_host(host: str) -> bool:
    """True for the host of an `http(s)://...` URL that must ship defanged.

    A scheme removes the ambiguity `_should_defang` guards against: after
    `https://` the next token *is* a hostname, so there is no prose to
    protect and no reason to restrict this to the runtime's curated
    `VALID_TLDS`. Without this, a live link on a TLD outside that list
    (`https://lnkd.in/...`, `https://app.any.run/...`) would ship clickable
    even though the whole point of defanging is that nothing in the bundle is.
    Known publisher/vendor/platform domains stay clickable, as before.
    """
    lowered = host.lower().strip(".")
    return "." in lowered and not _is_legitimate_host(lowered)


def _defang_host(host: str) -> str:
    return host.replace(".", "[.]")


def defang_text(text: str) -> str:
    """Defang live URLs and bare domains that are not known-legitimate hosts.

    `http://evil.example/x` becomes `hxxp://evil[.]example/x` and a bare
    `evil.example` becomes `evil[.]example`. Already-defanged tokens, IP
    addresses, CVE ids, hashes, and legitimate publisher/vendor domains are
    left exactly as they are.
    """
    if not text:
        return ""

    def _replace_url(match: re.Match) -> str:
        scheme, host = match.group(1), match.group(2)
        if not _should_defang_url_host(host):
            return match.group(0)
        defanged_scheme = "hxxps" if scheme.lower() == "https" else "hxxp"
        return f"{defanged_scheme}://{_defang_host(host)}"

    result = _FANGED_URL_RE.sub(_replace_url, text)

    def _replace_domain(match: re.Match) -> str:
        host = match.group(0)
        if not _should_defang(host):
            return host
        return _defang_host(host)

    # URLs were rewritten first, so their hosts now contain "[.]" and can no
    # longer match this bare-domain pattern.
    return _BARE_DOMAIN_RE.sub(_replace_domain, result)


def anonymize_text(text: str) -> str:
    """Remove author/contact identifiers and social-platform links from `text`
    while preserving IoC-looking tokens (CVE ids, hashes, IPs, defanged and
    real domains). Whitespace is collapsed to single spaces.
    """
    if not text:
        return ""

    result = text
    # Emails first: an email's local part would otherwise be partially eaten
    # by the @handle stripper and leave a mangled domain fragment behind.
    result = _EMAIL_RE.sub(" ", result)
    result = _HANDLE_RE.sub(" ", result)
    result = _SOCIAL_URL_RE.sub(" ", result)

    # Defang before protecting: the shipped fixture text must never contain a
    # clickable link to real campaign infrastructure (or to a named victim's
    # site). Social links are already gone by this point, so nothing that
    # survives here is a platform URL.
    result = defang_text(result)

    # Protect IoC-looking tokens (CVE ids, hashes, IPs, domains, ISO
    # dates/timestamps) before running the phone-number stripper, whose
    # digit-run pattern would otherwise also match a plain IP address, a
    # long CVE id, or a date/timestamp like "2026-03-27 21:05:07.129051".
    protected, placeholders = _protect_ioc_tokens(result)
    protected = _strip_phone_numbers(protected)
    result = _restore_ioc_tokens(protected, placeholders)

    return _WHITESPACE_RE.sub(" ", result).strip()


def pseudonym(value: str, salt: str = "tiblender-fixture") -> str:
    """Deterministic, non-reversible pseudonym for `value` (e.g. an author ref)."""
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return _PSEUDONYM_PREFIX + digest[:_PSEUDONYM_DIGEST_LENGTH]
