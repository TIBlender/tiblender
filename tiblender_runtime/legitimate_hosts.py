"""One source of truth for hostnames that are never campaign infrastructure.

Two lists used to exist side by side and drift apart:

* the IoC allowlist, so the extractor does not promote a platform, vendor or
  news host to "campaign infrastructure";
* ``tools/record_tool_responses.py::DEFAULT_ALLOWED_CRAWL_HOSTS`` -- the hosts
  the maintainer-only recorder is allowed to fetch.

A host is on the crawl allowlist precisely because it is a publisher, vendor,
standards body or national CERT. Once the recordings replayed, evidence
recorded *from* those hosts (CT logs, passive DNS, the crawl itself) started
arriving as indicators, and a report shipped five ``*.ncsc.gov.uk`` hostnames
as campaign infrastructure. Deriving both lists here makes that impossible:
``WEB_CRAWL_ALLOWED_HOSTS`` is a subset of ``PUBLISHER_HOSTS`` by
construction.

Suppression has **two shapes**, because a hostname can be "not an indicator"
for two different reasons:

* :data:`PUBLISHER_HOSTS` -- publishers, vendors, standards bodies, national
  CERTs, social platforms and link shorteners. Nothing under them is ever
  campaign infrastructure, so the match is exact **or subdomain suffix**:
  ``ncsc.gov.uk`` covers ``www.ncsc.gov.uk`` and ``authoring.ncsc.gov.uk``.
  It is assembled from three inputs: the reference list, the crawl allowlist,
  and :data:`NON_CRAWLED_PUBLISHER_HOSTS` for publishers that only ever
  appear as a link inside a collected post.
* :data:`SHARED_SERVICE_PARENTS` -- object storage, edge and static-site
  services that host one tenant per hostname (``pages.dev``, ``r2.dev``,
  ``amazonaws.com``, ...). Paper §V-A ("Indicator granularity on shared
  services") says the actionable indicator *is* the specific host or object
  path and that naming the parent domain would be overly broad, so here the
  match is **exact only**: bare ``pages.dev`` is never an indicator, while
  ``malicious-tenant.pages.dev`` stays reportable.

Use :func:`is_suppressed_host` rather than testing either set directly; it
applies the right matching rule to each.
"""

from __future__ import annotations

# Hosts the maintainer-only recorder may fetch live (suffix match). Every one is a
# vendor advisory, standards body, national CERT, or established publisher; see
# fixtures/README.md ("WEB_CRAWL host allowlist") for the per-host rationale and
# for hosts considered and deliberately left out.
WEB_CRAWL_ALLOWED_HOSTS: tuple[str, ...] = (
    "nvd.nist.gov",
    "nist.gov",
    "cve.org",
    "mitre.org",
    "github.com",
    "cisa.gov",
    "msrc.microsoft.com",
    "security.googleblog.com",
    "bleepingcomputer.com",
    "thehackernews.com",
    "securityweek.com",
    "helpnetsecurity.com",
    "horizon3.ai",
    "pcmag.com",
    "safedep.io",
    "security-next.com",
    "undernews.fr",
    "reuters.com",
    "cnn.com",
    "apnews.com",
    "bbc.com",
    "darkreading.com",
    "infosecurity-magazine.com",
    "ncsc.gov.uk",
    "ic3.gov",
    # Twenty further additions, made when reviewing a rebuilt queries.json's
    # skipped WEB_CRAWL hosts for established news outlets and first-party
    # security vendors not yet covered. See fixtures/README.md for the
    # per-host rationale.
    "nytimes.com",
    "washingtonpost.com",
    "wsj.com",
    "theguardian.com",
    "npr.org",
    "pbs.org",
    "newsweek.com",
    "theregister.com",
    "technologyreview.com",
    "securityonline.info",
    "microsoft.com",
    "crowdstrike.com",
    "malwarebytes.com",
    "elastic.co",
    "esentire.com",
    "infoblox.com",
    "reversinglabs.com",
    "specterops.io",
    "stepsecurity.io",
    "0patch.com",
)

# Platforms, shorteners, cloud/CDN providers and reference sites that appear in
# collected posts and in tool answers as references rather than as indicators.
#
# This is also the list the *authoring* tools use -- `tools/anonymize.py`
# decides from it which URLs may ship clickable in the fixture text, and
# `tools/build_fixtures.py` which hosts it does not scan as candidate
# infrastructure. Those two answer a different question from IoC extraction
# ("what may we publish?" rather than "what is an indicator?"), and widening
# an *allowlist* must never make an already-published bundle ship more live
# URLs clickable, so they deliberately do not see the crawl-allowlist
# additions below.
REFERENCE_HOSTS: frozenset[str] = frozenset(
    {
        "github.com",
        "nvd.nist.gov",
        "gitlab.com",
        "bitbucket.org",
        "npm.org",
        "npmjs.org",
        "npmjs.com",
        "reddit.com",
        "redd.it",
        "twitter.com",
        "x.com",
        "telegram.org",
        "t.me",
        "discord.com",
        "google.com",
        "microsoft.com",
        "apple.com",
        "amazon.com",
        "cloudflare.com",
        "virus" + "total.com",
        "nist.gov",
        "mitre.org",
        "cve.org",
        "cisa.gov",
        "bleepingcomputer.com",
        "thehackernews.com",
        "securityweek.com",
        "helpnetsecurity.com",
        "securityaffairs.com",
        "securityaffairs.co",
        "securityonline.info",
        "theregister.com",
        "watchtowr.com",
        "cloudfront.net",
        "amazonaws.com",
        "azureedge.net",
        "vercel.app",
        "pages.dev",
        "workers.dev",
        "t.co",
        "bit.ly",
        "tinyurl.com",
        "goo.gl",
        "ow.ly",
        "cutt.ly",
        "shorturl.at",
    }
)

# Shared services that host one tenant per hostname. Suppressed EXACTLY, never
# by suffix: paper §V-A makes the specific host the actionable indicator, so
# `evil.pages.dev` must stay reportable while bare `pages.dev` -- which the
# paper calls "overly broad", since the service hosts many legitimate buckets
# -- must not. `cloudflaressl.com` is here for the same reason a certificate's
# `*.sni.cloudflaressl.com` name is not the campaign: the SNI hostname belongs
# to the CDN, and the bare parent is meaningless as an indicator.
SHARED_SERVICE_PARENTS: frozenset[str] = frozenset(
    {
        "amazonaws.com",
        "azureedge.net",
        "cloudflaressl.com",
        "cloudfront.net",
        "github.io",
        "pages.dev",
        "r2.dev",
        "vercel.app",
        "workers.dev",
    }
)

# Publishers that reach the pipeline only through *post text* -- a collected
# post links one of their articles -- and that the recorder therefore never
# fetches, so they cannot arrive via `WEB_CRAWL_ALLOWED_HOSTS`. Without an
# entry here such a host stays a scored `Direct` indicator and its article URL
# ships as campaign infrastructure, which paper §III-F's Step 5 exclusion rules
# (news/CDN/shortener/SNS URLs) forbid.
#
# Deliberately NOT added to `REFERENCE_HOSTS`: that list also drives
# `tools/anonymize.py`, and moving a host into it would un-defang an
# already-published fixture bundle (see the module docstring and
# `tests/test_iocs.py::test_a_wider_ioc_allowlist_never_un_defangs_published_fixture_text`).
NON_CRAWLED_PUBLISHER_HOSTS: frozenset[str] = frozenset(
    {
        # Korean blockchain/crypto news outlet, linked from a collected
        # Telegram post about the npm "Axios" supply-chain compromise.
        "blockmedia.co.kr",
    }
)

# Never an indicator, at any depth: publishers, vendors, standards bodies,
# national CERTs, social platforms, link shorteners. Every reference host that
# is not a shared-service parent, plus every host trusted enough to fetch,
# plus the publishers only ever seen in post text.
PUBLISHER_HOSTS: frozenset[str] = frozenset(
    (REFERENCE_HOSTS - SHARED_SERVICE_PARENTS)
    | set(WEB_CRAWL_ALLOWED_HOSTS)
    | NON_CRAWLED_PUBLISHER_HOSTS
)

# Every host suppressed in *some* form. Membership alone says nothing about
# which rule applies -- call `is_suppressed_host` for that.
LEGITIMATE_HOSTS: frozenset[str] = PUBLISHER_HOSTS | SHARED_SERVICE_PARENTS


def is_suppressed_host(host: str) -> bool:
    """True when ``host`` may never be reported as campaign infrastructure.

    ``PUBLISHER_HOSTS`` match exactly or as a subdomain suffix;
    ``SHARED_SERVICE_PARENTS`` match exactly only.
    """
    lowered = str(host).strip().lower().strip(".")
    if not lowered:
        return False
    if lowered in SHARED_SERVICE_PARENTS:
        return True
    return any(lowered == host_name or lowered.endswith("." + host_name) for host_name in PUBLISHER_HOSTS)


__all__ = [
    "LEGITIMATE_HOSTS",
    "NON_CRAWLED_PUBLISHER_HOSTS",
    "PUBLISHER_HOSTS",
    "REFERENCE_HOSTS",
    "SHARED_SERVICE_PARENTS",
    "WEB_CRAWL_ALLOWED_HOSTS",
    "is_suppressed_host",
]
