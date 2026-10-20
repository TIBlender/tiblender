"""IoC extraction and reportability filters for source and tool text."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

from .legitimate_hosts import LEGITIMATE_HOSTS, is_suppressed_host
from .models import IocSet


# One source of truth for publisher/vendor/platform hosts: the WEB_CRAWL
# recorder allowlist is a subset of it, so evidence recorded *from* a
# publisher or national CERT can never be extracted as campaign
# infrastructure. Immutable, and only the union of the two suppression sets --
# `legitimate_hosts.is_suppressed_host` is what decides membership, because the
# two halves match differently (subdomain suffix vs parent domain only).
LEGITIMATE_DOMAINS: frozenset[str] = LEGITIMATE_HOSTS

VALID_TLDS = {
    "com",
    "net",
    "org",
    "io",
    "co",
    "jp",
    "uk",
    "ru",
    "cn",
    "kr",
    "de",
    "fr",
    "br",
    "au",
    "info",
    "biz",
    "site",
    "online",
    "top",
    "xyz",
    "zone",
    "dev",
    "app",
    "shop",
    "link",
    "click",
    "live",
    "cloud",
    "cfd",
    "icu",
    "example",
    "test",
}

_DOMAIN_RE = re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,24}\b")
_URL_RE = re.compile(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+", re.IGNORECASE)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,8}\b", re.IGNORECASE)
_HASH_RE = re.compile(r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b")


def refang(text: str) -> str:
    # Social posts commonly defang indicators; normalize only common patterns
    # before extraction so benign prose is not aggressively rewritten.
    replacements = [
        (r"[\[\(\{]\s*://\s*[\]\)\}]", "://"),
        (r"[\[\(\{]\s*:\s*[\]\)\}]", ":"),
        (r"[\[\(\{]\s*\.\s*[\]\)\}]", "."),
        (r"[\[\(\{]\s*dot\s*[\]\)\}]", "."),
        (r"[\[\(\{]\s*@\s*[\]\)\}]", "@"),
    ]
    refanged = text
    for pattern, repl in replacements:
        refanged = re.sub(pattern, repl, refanged, flags=re.IGNORECASE)
    refanged = re.sub(
        r"(?<![a-zA-Z])hxxps?\s*://",
        lambda match: "https://" if match.group(0).lower().startswith("hxxps") else "http://",
        refanged,
        flags=re.IGNORECASE,
    )
    refanged = re.sub(
        r"(?<![a-zA-Z])ttps?\s*://",
        lambda match: "https://" if match.group(0).lower().startswith("ttps") else "http://",
        refanged,
        flags=re.IGNORECASE,
    )
    return refanged


_URL_TRAILING_PUNCTUATION =".,;:!?)]}>\"'」』(["


def _strip_trailing_punctuation(url: str) -> str:
    """Drop sentence and markdown punctuation the URL regex swallowed.

    ``_URL_RE`` accepts "[", "]", "(" and ")" because real URLs contain them,
    so a markdown link written as ``[docs](http://host/docs](`` yields a match
    ending in "](". Stripping only the closing half left that suffix on the
    URL, and the malformed value reached a recorded WEB_CRAWL request key.
    """
    return url.rstrip(_URL_TRAILING_PUNCTUATION)


def _is_legitimate_domain(domain: str) -> bool:
    return is_suppressed_host(domain)


def _is_valid_domain(domain: str) -> bool:
    # Domain validation is intentionally conservative: invalid TLDs and known
    # reference domains are excluded before they reach the report layer.
    lowered = domain.lower().strip(".")
    if not lowered or _is_legitimate_domain(lowered):
        return False
    if _IP_RE.fullmatch(lowered):
        return False
    parts = lowered.split(".")
    if len(parts) < 2:
        return False
    if parts[-1] not in VALID_TLDS:
        return False
    return all(part and not part.startswith("-") and not part.endswith("-") for part in parts)


def _is_public_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def url_hostname(url: str) -> str:
    """Lowercase hostname of ``url``, or "" when it has none or is unparsable.

    Real post text contains malformed URLs -- an unbalanced "[" makes
    ``urlparse`` raise ``ValueError("Invalid IPv6 URL")``. An indicator we
    cannot parse is simply not an indicator; it must never abort a cycle.
    """
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def extract_iocs(text: str) -> IocSet:
    # Extract candidates first, then apply allow/deny rules so source posts do
    # not promote news sites, platform URLs, or infrastructure providers as IoCs.
    refanged = refang(text)
    domain_scan_text = list(refanged)

    urls: set[str] = set()
    domains: set[str] = set()
    ips: set[str] = set()
    hashes: set[str] = set()
    cves: set[str] = set()

    for match in _URL_RE.finditer(refanged):
        url = _strip_trailing_punctuation(match.group(0))
        for index in range(match.start(), match.end()):
            domain_scan_text[index] = " "
        host = url_hostname(url)
        if host and (_is_valid_domain(host) or _is_public_ip(host)):
            urls.add(url)
            if _is_valid_domain(host):
                domains.add(host)
            elif _is_public_ip(host):
                ips.add(host)

    for match in _DOMAIN_RE.finditer("".join(domain_scan_text)):
        domain = match.group(0).lower().strip(".")
        if _is_valid_domain(domain):
            domains.add(domain)

    for match in _IP_RE.finditer(refanged):
        ip = match.group(0)
        if _is_public_ip(ip):
            ips.add(ip)

    for match in _HASH_RE.finditer(refanged):
        hashes.add(match.group(0).lower())

    for match in _CVE_RE.finditer(refanged):
        cves.add(match.group(0).upper())

    return IocSet(
        urls=sorted(urls),
        domains=sorted(domains),
        ips=sorted(ips),
        hashes=sorted(hashes),
        cves=sorted(cves),
    )


def filter_reportable_iocs(iocs: IocSet) -> IocSet:
    # Report output uses the stricter filter a second time because tool output
    # can include benign reference domains unrelated to the threat itself.
    clean_urls: set[str] = set()
    clean_domains: set[str] = set()
    clean_ips: set[str] = set()
    for url in iocs.urls:
        host = url_hostname(url)
        if host and _is_valid_domain(host):
            clean_urls.add(url)
            clean_domains.add(host)
        elif host and _is_public_ip(host):
            clean_urls.add(url)
            clean_ips.add(host)
    for domain in iocs.domains:
        lowered = domain.lower().strip(".")
        if _is_valid_domain(lowered):
            clean_domains.add(lowered)
    for ip in iocs.ips:
        if _is_public_ip(ip):
            clean_ips.add(ip)
    return IocSet(
        urls=sorted(clean_urls),
        domains=sorted(clean_domains),
        ips=sorted(clean_ips),
        hashes=sorted({value.lower() for value in iocs.hashes}),
        cves=sorted({value.upper() for value in iocs.cves}),
    )
