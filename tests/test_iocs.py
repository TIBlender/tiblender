from tiblender_runtime.iocs import extract_iocs, refang


def test_refang_preserves_verbatim_indicators():
    text = "hxxps://ado-read-parser[.]com/a.pdf and 169.40.2[.]68"

    assert refang(text) == "https://ado-read-parser.com/a.pdf and 169.40.2.68"


def test_extract_iocs_filters_reference_domains_and_keeps_cves_hashes():
    text = (
        "Researcher post links https://github.com/advisory and malicious "
        "hxxps://ado-read-parser[.]com/dropper.pdf, C2 169.40.2[.]68, "
        "CVE-2026-34621, sha256 "
        "65dca34b5a28c3d9f8a8eb8a0e60a0b1d6d1d4b64db48ab681fd8db1d80abde7."
    )

    iocs = extract_iocs(text)

    assert "ado-read-parser.com" in iocs.domains
    assert "https://ado-read-parser.com/dropper.pdf" in iocs.urls
    assert "169.40.2.68" in iocs.ips
    assert "CVE-2026-34621" in iocs.cves
    assert "65dca34b5a28c3d9f8a8eb8a0e60a0b1d6d1d4b64db48ab681fd8db1d80abde7" in iocs.hashes
    assert "github.com" not in iocs.domains


def test_extract_iocs_does_not_promote_news_urls_or_path_fragments():
    text = (
        "Advisory discussion references CVE-2026-3055 via "
        "https://securityaffairs.com/190131/hacking/urgent-alert-netscaler-bug-cve-2026-3055-"
        "probed-by-attackers-could-leak-sensitive-data.html and "
        "https://thehackernews.com/2026/03/citrix-netscaler-under-active-recon-for.html."
    )

    iocs = extract_iocs(text)

    assert "CVE-2026-3055" in iocs.cves
    assert "securityaffairs.com" not in iocs.domains
    assert "thehackernews.com" not in iocs.domains
    assert "citrix-netscaler-under-active-recon-for.html" not in iocs.domains
    assert "data.html" not in iocs.domains
    assert not iocs.urls


def test_extract_iocs_filters_package_registry_reference_domains():
    text = (
        "The npm incident mentions packages[.]npm[.]org and registry[.]npmjs[.]org as package "
        "distribution context, while sfrclak[.]com is the suspicious payload host."
    )

    iocs = extract_iocs(text)

    assert "sfrclak.com" in iocs.domains
    assert "packages.npm.org" not in iocs.domains
    assert "registry.npmjs.org" not in iocs.domains


def test_extract_iocs_rejects_crawl_byline_and_reference_documents():
    text = (
        "By Pierluigi Paganini. Read more at securityaffairs.co. "
        "The public report discusses CVE-2026-3055 but does not publish attacker infrastructure."
    )

    iocs = extract_iocs(text)

    assert "CVE-2026-3055" in iocs.cves
    assert "pierluigi.paganini" not in iocs.domains
    assert "securityaffairs.co" not in iocs.domains


def test_a_malformed_url_is_skipped_instead_of_aborting_extraction():
    """An unbalanced "[" makes urlparse raise ValueError("Invalid IPv6 URL").

    Real collected post text contains such URLs; one of them used to abort a
    whole live cycle inside the Step 1 normalization node.
    """
    from tiblender_runtime.iocs import extract_iocs

    iocs = extract_iocs("bad https://exa[mple.com/path good https://real.example/p CVE-2026-1234")

    assert iocs.urls == ["https://real.example/p"]
    assert iocs.domains == ["real.example"]
    assert iocs.cves == ["CVE-2026-1234"]


def test_url_hostname_returns_empty_for_an_unparsable_url():
    from tiblender_runtime.iocs import url_hostname

    assert url_hostname("https://exa[mple.com/path") == ""
    assert url_hostname("not a url") == ""
    assert url_hostname("https://Real.Example/p") == "real.example"


def test_filter_reportable_iocs_drops_a_malformed_url_without_raising():
    from tiblender_runtime.iocs import filter_reportable_iocs
    from tiblender_runtime.models import IocSet

    clean = filter_reportable_iocs(IocSet(urls=["https://exa[mple.com/p", "https://real.example/p"]))

    assert clean.urls == ["https://real.example/p"]


def test_publisher_and_cert_hosts_are_never_extracted_as_iocs():
    """A recorded CT/passive-DNS/crawl answer names its own publisher's hosts.

    Those hostnames are evidence *about* a campaign, never the campaign's own
    infrastructure, so they must not reach a report's indicator list. The one
    source of truth is ``legitimate_hosts.LEGITIMATE_HOSTS``, which the
    WEB_CRAWL recorder allowlist is a subset of.
    """
    from tiblender_runtime.iocs import extract_iocs

    text = (
        "CT log for www.ncsc.gov.uk lists authoring.ncsc.gov.uk, beta.ncsc.gov.uk, "
        "cyberessentials.ncsc.gov.uk and cyberfirst.ncsc.gov.uk. See also "
        "https://www.bleepingcomputer.com/news/x/ and https://www.reuters.com/tech/y "
        "plus evil-campaign.example for contrast."
    )

    iocs = extract_iocs(text)

    assert iocs.domains == ["evil-campaign.example"]
    assert iocs.urls == []


def test_every_web_crawl_allowlisted_host_is_a_legitimate_host():
    from tiblender_runtime.iocs import LEGITIMATE_DOMAINS
    from tiblender_runtime.legitimate_hosts import (
        LEGITIMATE_HOSTS,
        PUBLISHER_HOSTS,
        REFERENCE_HOSTS,
        SHARED_SERVICE_PARENTS,
        WEB_CRAWL_ALLOWED_HOSTS,
    )

    # A host trusted enough to fetch is a publisher, vendor, standards body or
    # national CERT, so it belongs to the half that is suppressed *with* its
    # subdomains -- never to the shared-service half.
    assert set(WEB_CRAWL_ALLOWED_HOSTS) <= set(PUBLISHER_HOSTS)
    assert not set(WEB_CRAWL_ALLOWED_HOSTS) & set(SHARED_SERVICE_PARENTS)
    assert set(REFERENCE_HOSTS) <= set(LEGITIMATE_HOSTS)
    assert set(LEGITIMATE_DOMAINS) == set(LEGITIMATE_HOSTS)
    assert set(LEGITIMATE_HOSTS) == set(PUBLISHER_HOSTS) | set(SHARED_SERVICE_PARENTS)
    assert not set(PUBLISHER_HOSTS) & set(SHARED_SERVICE_PARENTS)
    # The crawl allowlist really does add hosts the reference list never had;
    # without that the subset assertion above would be vacuous.
    assert set(WEB_CRAWL_ALLOWED_HOSTS) - set(REFERENCE_HOSTS)


def test_a_news_publisher_named_only_in_post_text_is_still_suppressed():
    """Paper §III-F: a news URL is a reference, never an actionable indicator.

    ``blockmedia.co.kr`` reaches the pipeline the way most publishers do --
    a collected post links one of its articles -- but the recorder never
    fetches it, so it cannot come in through the WEB_CRAWL allowlist. Without
    a home of its own such a host stays a scored `Direct` indicator and the
    article URL ships as campaign infrastructure.
    """
    from tiblender_runtime.iocs import extract_iocs
    from tiblender_runtime.legitimate_hosts import (
        NON_CRAWLED_PUBLISHER_HOSTS,
        PUBLISHER_HOSTS,
        is_suppressed_host,
    )

    assert set(NON_CRAWLED_PUBLISHER_HOSTS) <= set(PUBLISHER_HOSTS)
    assert is_suppressed_host("blockmedia.co.kr")
    assert is_suppressed_host("www.blockmedia.co.kr")
    assert not is_suppressed_host("blockmedia.co.kr.evil.example")

    iocs = extract_iocs(
        "news: https://www.blockmedia.co.kr/archives/1071614?utm_source=telegram "
        "about the payload on evil-campaign.example"
    )

    assert iocs.domains == ["evil-campaign.example"]
    assert iocs.urls == []


def test_publisher_hosts_are_suppressed_together_with_their_subdomains():
    """Exact-or-suffix, and the suffix must be a real label boundary."""
    from tiblender_runtime.legitimate_hosts import is_suppressed_host

    assert is_suppressed_host("github.com")
    assert is_suppressed_host("www.github.com")
    assert is_suppressed_host("GITHUB.COM")
    assert is_suppressed_host("x.ncsc.gov.uk")
    # Neighbours that merely end in the same characters are not the host.
    assert not is_suppressed_host("notgithub.com")
    assert not is_suppressed_host("github.com.evil.example")
    assert not is_suppressed_host("ncsc.gov.uk.attacker.example")


def test_shared_service_parents_are_suppressed_as_the_parent_domain_only():
    """Paper §V-A: the actionable indicator is the specific host, not the service.

    Naming ``pages.dev`` would be overly broad -- the service hosts many
    legitimate tenants -- but a malicious tenant's own hostname is exactly
    what a defender needs, so it must stay reportable.
    """
    from tiblender_runtime.iocs import extract_iocs
    from tiblender_runtime.legitimate_hosts import is_suppressed_host

    for parent in ("pages.dev", "workers.dev", "r2.dev", "vercel.app", "amazonaws.com", "cloudfront.net", "github.io"):
        assert is_suppressed_host(parent), parent
        assert not is_suppressed_host(f"malicious-tenant.{parent}"), parent

    iocs = extract_iocs("payload staged at https://x.pages.dev/a and mirrored on pages.dev")

    assert iocs.domains == ["x.pages.dev"]
    assert iocs.urls == ["https://x.pages.dev/a"]


def test_a_wider_ioc_allowlist_never_un_defangs_published_fixture_text():
    """The authoring tools ship on REFERENCE_HOSTS, not on the IoC allowlist.

    Adding a CERT or news host so its own hostnames stop being extracted as
    campaign infrastructure must not, as a side effect, make that host's URLs
    ship clickable in an already-published fixture bundle.
    """
    from tools.anonymize import anonymize_text

    defanged = anonymize_text("advisory at https://www.ncsc.gov.uk/news/apt28-routers")

    assert "https://www.ncsc.gov.uk" not in defanged
    assert "ncsc[.]gov[.]uk" in defanged


def test_filter_reportable_iocs_drops_publisher_subdomains():
    from tiblender_runtime.iocs import filter_reportable_iocs
    from tiblender_runtime.models import IocSet

    clean = filter_reportable_iocs(
        IocSet(
            domains=["www.ncsc.gov.uk", "security.googleblog.com", "evil-campaign.example"],
            urls=["https://www.ncsc.gov.uk/news/a", "https://evil-campaign.example/a"],
        )
    )

    assert clean.domains == ["evil-campaign.example"]
    assert clean.urls == ["https://evil-campaign.example/a"]


def test_a_markdown_link_suffix_is_stripped_from_an_extracted_url():
    """`[docs](http://host/docs](` shapes appear in real collected post text.

    The trailing "](" used to survive into the URL and, from there, into a
    WEB_CRAWL request key shipped in the fixture bundle.
    """
    from tiblender_runtime.iocs import extract_iocs

    iocs = extract_iocs("see [the docs](http://docs.evil-campaign.example/docs]( for more")

    assert iocs.urls == ["http://docs.evil-campaign.example/docs"]
    assert iocs.domains == ["docs.evil-campaign.example"]


def test_pivot_max_resolutions_per_ip_reads_the_documented_environment_variable(monkeypatch):
    from tiblender_runtime import pivots

    assert pivots.max_resolutions_per_ip() == pivots.DEFAULT_MAX_RESOLUTIONS_PER_IP == 5
    monkeypatch.setenv("TIBLENDER_PIVOT_MAX_RESOLUTIONS_PER_IP", "9")
    assert pivots.max_resolutions_per_ip() == 9
    # An explicit override always wins over the environment.
    assert pivots.max_resolutions_per_ip(3) == 3
    monkeypatch.setenv("TIBLENDER_PIVOT_MAX_RESOLUTIONS_PER_IP", "not-a-number")
    assert pivots.max_resolutions_per_ip() == pivots.DEFAULT_MAX_RESOLUTIONS_PER_IP
