from tools.anonymize import anonymize_text, defang_text, pseudonym


def test_removes_at_handles():
    text = "great writeup by @security_researcher123 on this campaign"
    result = anonymize_text(text)
    assert "@security_researcher123" not in result
    assert "great writeup by" in result
    assert "on this campaign" in result


def test_removes_email_addresses():
    text = "contact the analyst at threat.intel@example.com for details"
    result = anonymize_text(text)
    assert "threat.intel@example.com" not in result
    assert "@" not in result


def test_removes_phone_numbers():
    text = "reach the hotline at +1 (555) 123-4567 for reports"
    result = anonymize_text(text)
    assert "555" not in result
    assert "123-4567" not in result


def test_removes_social_platform_urls():
    text = "see the thread https://x.com/someuser/status/12345 for the original post"
    result = anonymize_text(text)
    assert "x.com" not in result
    assert "see the thread" in result


def test_removes_telegram_and_discord_links():
    text = "join t.me/somechannel or discord.gg/abcd1234 to follow along"
    result = anonymize_text(text)
    assert "t.me/" not in result
    assert "discord.gg/" not in result


def test_preserves_defanged_domain():
    text = "the payload phones home to evil[.]example over HTTPS"
    result = anonymize_text(text)
    assert "evil[.]example" in result


def test_preserves_cve_id():
    text = "actively exploited per CVE-2026-1234 in the wild"
    result = anonymize_text(text)
    assert "CVE-2026-1234" in result


def test_preserves_cve_id_directly_adjacent_to_japanese_text():
    # Python's `\b` treats CJK characters as word characters, so a CVE id
    # with no space on either side of Japanese text is a real regression
    # risk: the leading/trailing boundary must not silently fail here.
    text = "問題はCVE-2025-66376として追跡される"
    result = anonymize_text(text)
    assert "CVE-2025-66376" in result


def test_preserves_domain_directly_adjacent_to_japanese_text():
    # The domain must survive intact (defanged, not mangled by the
    # phone-number stripper) even with no ASCII boundary around it.
    text = "関連ドメインmcp-guard.abcas.jpは宣伝サイトである"
    result = anonymize_text(text)
    assert "mcp-guard[.]abcas[.]jp" in result


def test_preserves_ip_address():
    text = "beacons observed connecting to 203.0.113.5 on port 443"
    result = anonymize_text(text)
    assert "203.0.113.5" in result


def test_preserves_hash():
    text = "the dropped sample hashes to 44d88612fea8a8f36de82e1278abb02f"
    result = anonymize_text(text)
    assert "44d88612fea8a8f36de82e1278abb02f" in result


def test_preserves_hash_and_ip_together():
    text = "sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 beaconed to 203.0.113.5"
    result = anonymize_text(text)
    assert "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" in result
    assert "203.0.113.5" in result


def test_preserves_bare_iso_date():
    text = "**Attack Date:** 2026-03-27 was when the intrusion began"
    result = anonymize_text(text)
    assert "2026-03-27" in result


def test_preserves_iso_timestamp_with_microseconds():
    text = "**Attack Date:** 2026-03-27 21:05:07.129051"
    result = anonymize_text(text)
    assert result == "**Attack Date:** 2026-03-27 21:05:07.129051"


def test_preserves_iso_timestamp_with_t_separator():
    text = "recorded at 2026-03-27T21:05:07 UTC"
    result = anonymize_text(text)
    assert "2026-03-27T21:05:07" in result


def test_removes_international_phone_number():
    text = "call the tip line at +81 90 1234 5678 if you have info"
    result = anonymize_text(text)
    assert "1234" not in result
    assert "5678" not in result


def test_removes_parenthesized_phone_number():
    text = "our hotline is (555) 123-4567 during business hours"
    result = anonymize_text(text)
    assert "555" not in result
    assert "123-4567" not in result


def test_social_url_requires_host_boundary():
    preserved = anonymize_text("check out notx.com/path for details")
    assert "notx[.]com/path" in preserved

    removed = anonymize_text("check out https://x.com/user/status/1 for details")
    assert "x.com" not in removed


def test_collapses_whitespace():
    text = "hello   @user\n\nworld"
    result = anonymize_text(text)
    assert "  " not in result
    assert result == "hello world"


def test_combined_removal_and_preservation():
    text = (
        "@analyst1 flagged CVE-2026-9999 tied to evil[.]example and 203.0.113.5, "
        "see https://t.me/leakchannel or email me at leak@example.org"
    )
    result = anonymize_text(text)
    assert "@analyst1" not in result
    assert "t.me/" not in result
    assert "leak@example.org" not in result
    assert "CVE-2026-9999" in result
    assert "evil[.]example" in result
    assert "203.0.113.5" in result


def test_pseudonym_is_stable_for_same_input():
    assert pseudonym("alice") == pseudonym("alice")


def test_pseudonym_differs_for_different_input():
    assert pseudonym("alice") != pseudonym("bob")


def test_pseudonym_format():
    value = pseudonym("alice")
    assert value.startswith("u")
    assert len(value) == 9  # "u" + 8 hex chars


def test_pseudonym_respects_salt():
    assert pseudonym("alice", salt="salt-a") != pseudonym("alice", salt="salt-b")


# ---------------------------------------------------------------------------
# defang_text: bundled fixtures are excerpts of real posts, so any live
# indicator (and any named-victim site) must ship non-clickable.
# ---------------------------------------------------------------------------


def test_defangs_bare_http_url():
    assert defang_text("payload from http://kdmpop.com/x.exe") == "payload from hxxp://kdmpop[.]com/x.exe"


def test_defangs_https_url_scheme_and_host():
    assert defang_text("see https://evil.example.com/login") == "see hxxps://evil[.]example[.]com/login"


def test_defangs_bare_domain():
    assert defang_text("c2 at malicious-host.top today") == "c2 at malicious-host[.]top today"


def test_keeps_already_defanged_tokens_unchanged():
    text = "hxxp://bad[.]com and evil[.]net and hxxps://other[.]org"
    assert defang_text(text) == text


def test_keeps_legitimate_publisher_and_vendor_domains():
    text = "reported by bleepingcomputer.com, patch on github.com/foo, advisory at nvd.nist.gov"
    assert defang_text(text) == text


def test_keeps_legitimate_domain_urls_clickable():
    text = "https://github.com/org/repo/issues/1"
    assert defang_text(text) == text


def test_does_not_defang_ip_addresses_or_cve_ids():
    text = "203.0.113.5 exploited CVE-2026-1234 on 2026-03-27"
    assert defang_text(text) == text


def test_does_not_defang_prose_or_file_names_with_unknown_tlds():
    text = "ran the payload.Then it read config.yaml and main.py"
    assert defang_text(text) == text


def test_defanging_preserves_extracted_iocs():
    # The whole point: defanged fixture text must extract to exactly the same
    # indicators as the original, because extraction refangs first.
    from tiblender_runtime.iocs import extract_iocs

    text = "drop from http://kdmpop.com and beacon to evil.example.com (203.0.113.5)"
    assert extract_iocs(defang_text(text)) == extract_iocs(text)


def test_anonymize_text_defangs_victim_site_url():
    text = "The chaos ransomware group has added http://kdmpop.com to its victims."
    result = anonymize_text(text)
    assert "http://kdmpop.com" not in result
    assert "hxxp://kdmpop[.]com" in result


def test_defangs_darknet_leak_site_urls():
    # A .onion leak-site address is never extracted as an IoC (its TLD is not
    # in VALID_TLDS), but it is live infrastructure and must not ship
    # clickable in fixture text either.
    text = "Claim URL: http://hptqq2o2qjva7lcaaq.onion/post/QbsoKI"
    assert defang_text(text) == "Claim URL: hxxp://hptqq2o2qjva7lcaaq[.]onion/post/QbsoKI"


def test_removes_platform_media_urls_on_any_subdomain():
    """A signed platform media URL is a pointer back to the original post.

    `preview.redd.it/<id>.jpg?...&s=<signature>` identifies one Reddit
    submission as precisely as its permalink does, and the same shape exists
    for every platform (`i.redd.it`, `v.redd.it`, `old.reddit.com`,
    `cdn.discordapp.com`, `media.discordapp.net`). Tolerating only `www.` as
    a subdomain let exactly this URL ship clickable in the bundle.
    """
    signed = (
        "context https://preview.redd.it/3qzbvpopuotg1.jpg"
        "?width=1080&format=pjpg&auto=webp&s=fd2b2f26c53700b9fcbc35679af73fe9db263640 end"
    )
    result = anonymize_text(signed)
    assert "redd.it" not in result
    assert "preview" not in result
    assert "s=fd2b2f26" not in result
    assert result.startswith("context") and result.endswith("end")

    for url in (
        "https://i.redd.it/abc123.png",
        "https://v.redd.it/xyz/DASH_720.mp4",
        "https://old.reddit.com/r/netsec/comments/abc/title/",
        "https://cdn.discordapp.com/attachments/1/2/x.png",
        "https://media.discordapp.net/attachments/1/2/x.png",
        "https://mobile.twitter.com/someone/status/9",
        "https://web.telegram.org.example.invalid",
    ):
        stripped = anonymize_text(f"see {url} now")
        assert "discordapp" not in stripped, url
        assert "redd" not in stripped, url
        assert "twitter" not in stripped, url


def test_removes_social_link_shortener_urls():
    result = anonymize_text("write-up at https://lnkd.in/eXaMpLe1 today")
    assert "lnkd" not in result
    assert "write-up at" in result


def test_platform_hosts_are_never_treated_as_clickable_publishers():
    """`redd.it`/`reddit.com`/`x.com`/`t.me` are on the runtime's IoC
    allowlist so the extractor does not mistake them for campaign
    infrastructure. That must not make a platform URL a "legitimate
    publisher" URL the defanger leaves clickable.
    """
    for host in ("redd.it", "preview.redd.it", "reddit.com", "x.com", "t.me", "discord.com"):
        assert defang_text(f"https://{host}/thing") == f"hxxps://{host.replace('.', '[.]')}/thing"
    # Genuine publisher/vendor hosts are unaffected.
    assert defang_text("https://github.com/org/repo") == "https://github.com/org/repo"
