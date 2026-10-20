import io
import json
import sys
import types
import urllib.error

import pytest

from tiblender_runtime.models import ToolQuery, ToolResult
from tiblender_runtime.normalization import normalize_posts
from tiblender_runtime.models import Post
from tiblender_runtime.services import ServiceStatus
from tiblender_runtime.settings import Settings
from tiblender_runtime.tool_fixtures import ToolRecordingStore
from tiblender_runtime.tools import ToolRunner, ToolUnavailableError
from tiblender_runtime.tools import _http_text


def test_tool_runner_rejects_non_contract_tool():
    runner = ToolRunner(Settings(enable_real_tools=False))
    forbidden_tool = "UNLISTED" + "_" + "LOOKUP"

    with pytest.raises(ValueError, match="not in tool contract"):
        runner.execute(ToolQuery("Technical", forbidden_tool, {"url": "https://example.com"}))


def test_web_crawl_executes_http_and_extracts_iocs(monkeypatch):
    def fake_http_text(url, *, headers=None, timeout=30, max_bytes=200000):
        assert url == "https://example.test/path"
        return (
            "<html><title>Investigation</title><body>CVE-2026-12345 uses "
            "https://bad.example.test/payload and 8.8.8.8</body></html>",
            "https://example.test/path",
        )

    monkeypatch.setattr("tiblender_runtime.tools._http_text", fake_http_text)
    runner = ToolRunner(Settings(enable_real_tools=True, web_crawl_render_js=False))

    result = runner.execute(ToolQuery("Technical", "WEB_CRAWL", {"url": "https://example.test/path"}))

    assert result.status == "success"
    assert result.tool_type == "WEB_CRAWL"
    assert "CVE-2026-12345" in result.discovered_iocs.cves
    assert "bad.example.test" in result.discovered_iocs.domains


def test_web_crawl_uses_javascript_renderer_and_extracts_rendered_iocs(monkeypatch):
    def fail_static_fetch(*args, **kwargs):
        del args, kwargs
        raise AssertionError("static HTTP fetch should not run when JavaScript rendering succeeds")

    def fake_render(url, *, timeout, max_bytes, wait_ms):
        assert url == "https://example.test/spa"
        assert timeout == 20
        assert max_bytes == 200000
        assert wait_ms == 1000
        return (
            "<html><title>Rendered App</title><body>Client-rendered IoC "
            "https://rendered-bad.example.test/payload CVE-2026-54321</body></html>",
            "https://example.test/spa",
            {"javascript_rendered": True, "blocked_resource_types": ["image", "media", "font"]},
        )

    monkeypatch.setattr("tiblender_runtime.tools._http_text", fail_static_fetch)
    monkeypatch.setattr("tiblender_runtime.tools._render_javascript_page", fake_render)
    runner = ToolRunner(Settings(enable_real_tools=True, web_crawl_render_js=True, web_crawl_wait_ms=1000))

    result = runner.execute(ToolQuery("Technical", "WEB_CRAWL", {"url": "https://example.test/spa"}, timeout_seconds=20))

    assert result.status == "success"
    assert "CVE-2026-54321" in result.discovered_iocs.cves
    assert "rendered-bad.example.test" in result.discovered_iocs.domains
    raw = json.loads(result.raw_json)
    assert raw["javascript_rendered"] is True
    assert raw["blocked_resource_types"] == ["image", "media", "font"]


def test_nvd_lookup_executes_json_api(monkeypatch):
    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        assert "services.nvd.nist.gov/rest/json/cves/2.0" in url
        assert "CVE-2026-12345" in url
        return {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2026-12345",
                        "descriptions": [{"lang": "en", "value": "Example vulnerability"}],
                    }
                }
            ]
        }, url

    monkeypatch.setattr("tiblender_runtime.tools._http_json", fake_http_json)
    runner = ToolRunner(Settings(enable_real_tools=True))

    result = runner.execute(ToolQuery("Technical", "NVD_CVE_LOOKUP", {"cve": "CVE-2026-12345"}))

    assert result.status == "success"
    assert result.discovered_iocs.cves == ["CVE-2026-12345"]
    assert "Example vulnerability" in result.summary


def test_nvd_lookup_retries_without_rejected_api_key(monkeypatch):
    calls = []

    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        del timeout
        calls.append(bool(headers and headers.get("apiKey")))
        if headers and headers.get("apiKey"):
            raise ToolUnavailableError("HTTP 404:")
        return {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2023-4966",
                        "descriptions": [{"lang": "en", "value": "Citrix NetScaler vulnerability"}],
                    }
                }
            ]
        }, url

    monkeypatch.setattr("tiblender_runtime.tools._http_json", fake_http_json)
    runner = ToolRunner(Settings(enable_real_tools=True, nvd_api_key="rejected-key"))

    result = runner.execute(ToolQuery("Technical", "NVD_CVE_LOOKUP", {"cve": "CVE-2023-4966"}))

    assert result.status == "success"
    assert calls == [True, False]
    assert result.discovered_iocs.cves == ["CVE-2023-4966"]


def test_search_engine_extracts_snippet_iocs_without_promoting_result_urls(monkeypatch):
    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        del headers, timeout
        assert "searxng.test/search" in url
        return {
            "results": [
                {
                    "title": "Axios npm supply-chain incident response",
                    "url": "https://example-news.test/axios-writeup",
                    "content": (
                        "Hunting guidance lists sfrclak.com, callnrwise.com, calltan.com, "
                        "142.11.206.73, and 23.254.167.216 as related indicators."
                    ),
                }
            ]
        }, url

    monkeypatch.setattr("tiblender_runtime.tools._http_json", fake_http_json)
    runner = ToolRunner(
        Settings(
            enable_real_tools=True,
            search_engine_provider="searxng",
            searxng_base_url="http://searxng.test",
        )
    )

    result = runner.execute(ToolQuery("Technical", "SEARCH_ENGINE", {"query": "sfrclak.com supply chain attack"}))

    assert result.status == "success"
    assert "callnrwise.com" in result.discovered_iocs.domains
    assert "calltan.com" in result.discovered_iocs.domains
    assert "142.11.206.73" in result.discovered_iocs.ips
    assert "23.254.167.216" in result.discovered_iocs.ips
    assert "example-news.test" not in result.discovered_iocs.domains


def test_search_engine_ignores_unrelated_snippet_iocs_for_hash_query(monkeypatch):
    searched_hash = "92ff08773995ebc8d55ec4b8e1a225d0d1e51efa4ef88b8849d0071230c9645a"

    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        del headers, timeout
        assert searched_hash in url
        return {
            "results": [
                {
                    "title": "Unrelated sports result",
                    "url": "https://search-result.example/item",
                    "content": "A random snippet mentions nba.com but not the searched hash.",
                },
                {
                    "title": "Hash hunting note",
                    "url": "https://security-writeup.example/hash",
                    "content": (
                        f"IoC {searched_hash} was observed with sfrclak.com and "
                        "142.11.206.73 in the same campaign. Related tooling also mentions "
                        "58401c195fe0a6204b42f5f90995ece5fab74ce7c69c67a24c61a057325af668."
                    ),
                },
            ]
        }, url

    monkeypatch.setattr("tiblender_runtime.tools._http_json", fake_http_json)
    runner = ToolRunner(
        Settings(
            enable_real_tools=True,
            search_engine_provider="searxng",
            searxng_base_url="http://searxng.test",
        )
    )

    result = runner.execute(ToolQuery("Technical", "SEARCH_ENGINE", {"query": searched_hash}))

    assert result.status == "success"
    assert "sfrclak.com" in result.discovered_iocs.domains
    assert "142.11.206.73" in result.discovered_iocs.ips
    assert result.discovered_iocs.hashes == [searched_hash]
    assert "nba.com" not in result.discovered_iocs.domains


def test_whois_lookup_does_not_promote_rdap_metadata_to_iocs(monkeypatch):
    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        assert "rdap.org/domain/ado-read-parser.com" in url
        return {
            "handle": "2956898792_DOMAIN_COM-VRSN",
            "status": ["active"],
            "nameservers": [{"ldhName": "NS1.DNSOWL.COM"}],
            "links": [{"href": "https://rdap.verisign.com/com/v1/domain/ADO-READ-PARSER.COM"}],
            "events": [{"eventAction": "registration", "eventDate": "2025-02-06T11:42:40Z"}],
        }, url

    monkeypatch.setattr("tiblender_runtime.tools._http_json", fake_http_json)
    runner = ToolRunner(Settings(enable_real_tools=True))

    result = runner.execute(ToolQuery("Infrastructure", "WHOIS_LOOKUP", {"domain": "ado-read-parser.com"}))

    assert result.status == "success"
    assert result.discovered_iocs.domains == ["ado-read-parser.com"]
    assert "rdap.verisign.com" not in result.discovered_iocs.domains
    assert "ns1.dnsowl.com" not in result.discovered_iocs.domains


def test_passive_dns_zero_results_are_successful_lookup(monkeypatch):
    def fake_vt_get_json(url, *, headers, timeout, max_bytes=None, sleeper=None):
        del headers, timeout, sleeper
        assert "virustotal.com/api/v3/domains/sfrclak.com/resolutions" in url
        return {"data": []}, url

    monkeypatch.setattr("tiblender_runtime.tools._vt_get_json", fake_vt_get_json)
    runner = ToolRunner(Settings(enable_real_tools=True, mode="live", virustotal_api_key="vtkey"))

    result = runner.execute(ToolQuery("Infrastructure", "PASSIVE_DNS_LOOKUP", {"domain": "sfrclak.com"}))

    assert result.status == "success"
    assert "0 resolution(s)" in result.summary
    assert result.discovered_iocs.all_values() == []
    assert result.raw_json


def test_tls_ct_lookup_falls_back_to_certspotter_when_crtsh_is_unavailable(monkeypatch):
    calls = []

    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        del headers, timeout
        calls.append(url)
        if "crt.sh" in url:
            raise ToolUnavailableError("The read operation timed out")
        assert "api.certspotter.com/v1/issuances" in url
        return [
            {
                "dns_names": ["example.com", "www.example.com"],
                "cert_sha256": "abc123",
                "not_before": "2026-04-01T00:00:00Z",
                "not_after": "2026-07-01T00:00:00Z",
                "revoked": False,
            }
        ], url

    monkeypatch.setattr("tiblender_runtime.tools._http_json", fake_http_json)
    runner = ToolRunner(Settings(enable_real_tools=True))

    result = runner.execute(ToolQuery("Infrastructure", "TLS_CT_LOOKUP", {"domain": "example.com"}))

    assert result.status == "success"
    assert [url for url in calls if "crt.sh" in url]
    assert [url for url in calls if "api.certspotter.com/v1/issuances" in url]
    assert result.discovered_iocs.domains == ["example.com", "www.example.com"]
    assert "Certificate transparency lookup for example.com" in result.summary


def test_tls_ct_lookup_returns_sanitized_unavailable_when_all_ct_providers_fail(monkeypatch):
    calls = []

    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        del headers, timeout
        calls.append(url)
        if "crt.sh" in url:
            raise RuntimeError("HTTP 502: <html>Bad Gateway</html>")
        raise ToolUnavailableError("HTTP 404: provider-specific detail")

    monkeypatch.setattr("tiblender_runtime.tools._http_json", fake_http_json)
    runner = ToolRunner(Settings(enable_real_tools=True))

    result = runner.execute(ToolQuery("Infrastructure", "TLS_CT_LOOKUP", {"domain": "example.com"}))

    assert result.status == "unavailable"
    assert [url for url in calls if "crt.sh" in url]
    assert [url for url in calls if "api.certspotter.com/v1/issuances" in url]
    assert "TLS CT providers unavailable for example.com" in result.summary
    assert "HTTP 502" not in result.summary
    assert "Bad Gateway" not in result.summary


def test_x_search_does_not_fabricate_without_key():
    runner = ToolRunner(Settings(enable_real_tools=True, x_bearer_token=""))

    result = runner.execute(ToolQuery("Social", "X_SEARCH", {"query": "CVE-2026-12345"}))

    assert result.status == "unavailable"
    assert result.discovered_iocs.all_values() == []


def test_tool_runner_marks_unauthorized_provider_as_unavailable(monkeypatch):
    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        raise ToolUnavailableError("HTTP 401: Unauthorized")

    monkeypatch.setattr("tiblender_runtime.tools._http_json", fake_http_json)
    runner = ToolRunner(Settings(enable_real_tools=True, x_bearer_token="invalid"))

    result = runner.execute(ToolQuery("Social", "X_SEARCH", {"query": "CVE-2026-12345"}))

    assert result.status == "unavailable"
    assert "unavailable" in result.summary


def test_x_search_rotates_tokens_from_file_before_falling_back(monkeypatch, tmp_path):
    token_file = tmp_path / "x_tokens.txt"
    token_file.write_text("good-token\n", encoding="utf-8")
    seen_authorizations = []

    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        del timeout
        seen_authorizations.append(headers["Authorization"])
        if headers["Authorization"] == "Bearer bad-token":
            raise ToolUnavailableError("HTTP 401: Unauthorized")
        assert "api.twitter.com/2/tweets/search/recent" in url
        return {"data": [{"text": "CVE-2026-12345 active exploitation bad.example.test"}]}, url

    monkeypatch.setattr("tiblender_runtime.tools._http_json", fake_http_json)
    runner = ToolRunner(
        Settings(
            enable_real_tools=True,
            x_bearer_token="bad-token",
            x_bearer_token_file=str(token_file),
        )
    )

    result = runner.execute(ToolQuery("Social", "X_SEARCH", {"query": "CVE-2026-12345"}))

    assert result.status == "success"
    assert seen_authorizations == ["Bearer bad-token", "Bearer good-token"]
    assert result.discovered_iocs.cves == ["CVE-2026-12345"]


def test_x_search_uses_collected_x_corpus_when_tokens_are_unusable(monkeypatch):
    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        del url, headers, timeout
        raise ToolUnavailableError("HTTP 403: Forbidden")

    monkeypatch.setattr("tiblender_runtime.tools._http_json", fake_http_json)
    posts = normalize_posts(
        [
            Post(id="x1", platform="x", content="CVE-2026-12345 exploitation signal bad.example.test"),
            Post(id="r1", platform="reddit", content="unrelated text"),
        ]
    )
    runner = ToolRunner(Settings(enable_real_tools=True, x_bearer_token="expired-token"), corpus=posts)

    result = runner.execute(ToolQuery("Social", "X_SEARCH", {"query": "CVE-2026-12345"}))

    assert result.status == "success"
    assert "collected post" in result.summary
    assert json.loads(result.raw_json)["fallback_source"] == "collected_corpus"
    assert result.discovered_iocs.cves == ["CVE-2026-12345"]


def test_http_text_marks_404_as_unavailable(monkeypatch):
    def raise_404(request, timeout=30):
        del request, timeout
        raise urllib.error.HTTPError(
            url="https://rdap.org/domain/missing.example",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", raise_404)

    with pytest.raises(ToolUnavailableError, match="HTTP 404"):
        _http_text("https://rdap.org/domain/missing.example")


def test_http_text_marks_dns_failure_as_unavailable(monkeypatch):
    def raise_dns_failure(request, timeout=30):
        del request, timeout
        raise urllib.error.URLError(OSError("Name or service not known"))

    monkeypatch.setattr("urllib.request.urlopen", raise_dns_failure)

    with pytest.raises(ToolUnavailableError, match="Name or service not known"):
        _http_text("https://nx.example.test")


def test_discord_and_telegram_search_use_internal_corpus_without_tokens():
    posts = normalize_posts(
        [
            Post(id="d1", platform="discord", content="Discord thread mentions CVE-2026-12345 and bad.example"),
            Post(id="t1", platform="telegram", content="Telegram post mentions CVE-2026-12345 and 8.8.8.8"),
        ]
    )
    runner = ToolRunner(Settings(enable_real_tools=True), corpus=posts)

    discord = runner.execute(ToolQuery("Social", "DISCORD_SEARCH", {"query": "CVE-2026-12345"}))
    telegram = runner.execute(ToolQuery("Social", "TELEGRAM_SEARCH", {"query": "CVE-2026-12345"}))

    assert discord.status == "success"
    assert telegram.status == "success"
    assert discord.discovered_iocs.cves == ["CVE-2026-12345"]
    assert telegram.discovered_iocs.cves == ["CVE-2026-12345"]


def test_reddit_search_executes_oauth_api(monkeypatch):
    class FakeSubmission:
        title = "CVE discussion"
        selftext = "CVE-2026-12345 bad.example.test"
        permalink = "/r/test/comments/1/x"
        url = "https://reddit.com/r/test/comments/1/x"
        created_utc = 1760000000

    class FakeSubreddit:
        def search(self, query, sort="new", limit=5):
            assert query == "CVE-2026-12345"
            assert sort == "new"
            assert limit == 5
            return [FakeSubmission()]

    class FakeReddit:
        def __init__(self, *, client_id, client_secret, user_agent):
            assert client_id == "client"
            assert client_secret == "secret"
            assert user_agent == "test-agent"

        def subreddit(self, name):
            assert name == "all"
            return FakeSubreddit()

    monkeypatch.setitem(sys.modules, "praw", types.SimpleNamespace(Reddit=FakeReddit))
    runner = ToolRunner(
        Settings(
            enable_real_tools=True,
            reddit_client_id="client",
            reddit_client_secret="secret",
            reddit_user_agent="test-agent",
        )
    )

    result = runner.execute(ToolQuery("Social", "REDDIT_SEARCH", {"query": "CVE-2026-12345"}))

    assert result.status == "success"
    assert result.discovered_iocs.cves == ["CVE-2026-12345"]
    assert json.loads(result.raw_json)["result_count"] == 1


def test_reddit_search_uses_collected_reddit_corpus_without_credentials():
    posts = normalize_posts(
        [
            Post(id="r1", platform="reddit", content="CVE-2026-12345 exploitation signal bad.example.test"),
            Post(id="x1", platform="x", content="unrelated text"),
        ]
    )
    runner = ToolRunner(Settings(enable_real_tools=True), corpus=posts)

    result = runner.execute(ToolQuery("Social", "REDDIT_SEARCH", {"query": "CVE-2026-12345"}))

    assert result.status == "success"
    assert "collected post" in result.summary
    assert json.loads(result.raw_json)["fallback_source"] == "collected_corpus"
    assert result.discovered_iocs.cves == ["CVE-2026-12345"]


def test_x_search_uses_collected_x_corpus_in_fixture_mode():
    # With recorded collector fixtures present the x service resolves to
    # fixture mode; a recording miss must still search the collected corpus
    # rather than returning an empty result.
    posts = normalize_posts(
        [
            Post(id="x1", platform="x", content="CVE-2026-12345 exploitation signal bad.example.test"),
            Post(id="r1", platform="reddit", content="unrelated text"),
        ]
    )
    runner = ToolRunner(
        Settings(enable_real_tools=True),
        corpus=posts,
        services={"x": ServiceStatus("x", "fixture", "test")},
    )

    result = runner.execute(ToolQuery("Social", "X_SEARCH", {"query": "CVE-2026-12345"}))

    assert result.status == "success"
    assert json.loads(result.raw_json)["fallback_source"] == "collected_corpus"
    assert result.discovered_iocs.cves == ["CVE-2026-12345"]


def test_fixture_mode_recording_miss_stays_unavailable_for_non_social_tools():
    runner = ToolRunner(
        Settings(enable_real_tools=True),
        services={"whois": ServiceStatus("whois", "fixture", "test")},
    )

    result = runner.execute(ToolQuery("Infrastructure", "WHOIS_LOOKUP", {"domain": "bad.example.test"}))

    assert result.status == "unavailable"
    assert "no recorded tool response" in result.summary


def test_reddit_search_ignores_oauth_results_without_query_ioc(monkeypatch):
    class IrrelevantSubmission:
        title = "Unrelated discussion"
        selftext = "No vulnerability reference in this post"
        permalink = "/r/test/comments/1/x"
        url = "https://reddit.com/r/test/comments/1/x"
        created_utc = 1760000000

    class FakeSubreddit:
        def search(self, query, sort="new", limit=5):
            del query, sort, limit
            return [IrrelevantSubmission()]

    class FakeReddit:
        def __init__(self, *, client_id, client_secret, user_agent):
            del client_id, client_secret, user_agent

        def subreddit(self, name):
            del name
            return FakeSubreddit()

    monkeypatch.setitem(sys.modules, "praw", types.SimpleNamespace(Reddit=FakeReddit))
    posts = normalize_posts(
        [Post(id="r1", platform="reddit", content="CVE-2026-12345 exploitation signal bad.example.test")]
    )
    runner = ToolRunner(
        Settings(enable_real_tools=True, reddit_client_id="client", reddit_client_secret="secret"),
        corpus=posts,
    )

    result = runner.execute(ToolQuery("Social", "REDDIT_SEARCH", {"query": "CVE-2026-12345 exploitation campaign"}))

    assert result.status == "success"
    assert json.loads(result.raw_json)["fallback_source"] == "collected_corpus"
    assert result.discovered_iocs.cves == ["CVE-2026-12345"]


def test_search_engine_does_not_promote_result_urls_to_iocs_or_titles(monkeypatch):
    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        del headers, timeout
        assert url.startswith("http://searxng:8080/search?")
        assert "format=json" in url
        return {
            "results": [
                {
                    "title": "Provider result title",
                    "url": "https://www.sciencedirect.com/science/article/pii/51d00b2f24215f371a518a062b7fa46938a508e24ef9344cf58b2bf8a9dd18bd",
                    "content": "benign academic result",
                }
            ]
        }, url

    monkeypatch.setattr("tiblender_runtime.tools._http_json", fake_http_json)
    runner = ToolRunner(Settings(enable_real_tools=True))

    result = runner.execute(ToolQuery("Actor", "SEARCH_ENGINE", {"query": "PayPay phishing attribution"}))

    assert result.status == "success"
    assert result.discovered_iocs.all_values() == []
    assert "Provider result title" not in result.summary


def test_search_engine_uses_collected_corpus_when_searxng_is_unavailable(monkeypatch):
    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        del url, headers, timeout
        raise ToolUnavailableError("HTTP 403: Forbidden")

    monkeypatch.setattr("tiblender_runtime.tools._http_json", fake_http_json)
    posts = normalize_posts(
        [
            Post(id="x1", platform="x", content="Citrix NetScaler CVE-2026-12345 exploitation signal"),
            Post(id="d1", platform="discord", content="unrelated text"),
        ]
    )
    runner = ToolRunner(Settings(enable_real_tools=True), corpus=posts)

    result = runner.execute(ToolQuery("Actor", "SEARCH_ENGINE", {"query": "NetScaler CVE-2026-12345"}))

    assert result.status == "success"
    assert "collected post" in result.summary
    assert json.loads(result.raw_json)["fallback_source"] == "collected_corpus"
    assert result.discovered_iocs.all_values() == []


def test_search_engine_uses_searxng_json_api_and_normalizes_results(monkeypatch):
    calls = []

    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        del headers, timeout
        calls.append(url)
        return {
            "results": [
                {
                    "title": "OpenAI result",
                    "url": "https://example.com/openai",
                    "content": "first snippet",
                    "engine": "brave",
                    "category": "general",
                    "score": 13.5,
                },
                {
                    "title": "ignored",
                    "url": "https://example.com/ignored",
                    "content": "ignored snippet",
                },
            ]
        }, url

    monkeypatch.setattr("tiblender_runtime.tools._http_json", fake_http_json)
    settings = Settings(enable_real_tools=True, searxng_base_url="http://searxng:8080")
    runner = ToolRunner(settings)

    result = runner.execute(
        ToolQuery(
            "Actor",
            "SEARCH_ENGINE",
            {
                "query": "OpenAI research",
                "max_results": 1,
                "include_domains": ["example.com"],
                "exclude_domains": ["bad.example"],
            },
        )
    )

    assert calls == ["http://searxng:8080/search?q=OpenAI+research+site%3Aexample.com+-site%3Abad.example&format=json"]
    assert result.status == "success"
    assert json.loads(result.raw_json) == {
        "result_count": 1,
        "items": [
            {
                "title": "OpenAI result",
                "url": "https://example.com/openai",
                "content": "first snippet",
                "engine": "brave",
                "category": "general",
                "score": 13.5,
            }
        ],
    }


def _record_whois_recording(store: ToolRecordingStore, query: ToolQuery, summary: str) -> None:
    store.record(
        "WHOIS_LOOKUP",
        query.parameters,
        ToolResult(tool_type="WHOIS_LOOKUP", query_params=query.parameters, summary=summary),
        source_url="https://rdap.org/domain/cached.example",
    )


def test_auto_mode_falls_back_to_recording_when_live_call_is_unavailable(tmp_path, monkeypatch):
    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        del url, headers, timeout
        raise ToolUnavailableError("HTTP 503: temporarily unavailable")

    monkeypatch.setattr("tiblender_runtime.tools._http_json", fake_http_json)
    store = ToolRecordingStore(tmp_path)
    query = ToolQuery("Infrastructure", "WHOIS_LOOKUP", {"domain": "cached.example"})
    _record_whois_recording(store, query, "cached registrar")

    services = {"whois": ServiceStatus("whois", "live", "auto mode with credentials")}
    runner = ToolRunner(Settings(mode="auto"), services=services, recordings=store)

    result = runner.execute(query)

    assert result.status == "success"
    assert result.summary == "cached registrar"
    assert json.loads(result.raw_json)["source"] == "recording"


def test_live_mode_never_consults_recording_on_live_call_failure(tmp_path, monkeypatch):
    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        del url, headers, timeout
        raise ToolUnavailableError("HTTP 503: temporarily unavailable")

    monkeypatch.setattr("tiblender_runtime.tools._http_json", fake_http_json)
    store = ToolRecordingStore(tmp_path)
    query = ToolQuery("Infrastructure", "WHOIS_LOOKUP", {"domain": "cached.example"})
    _record_whois_recording(store, query, "cached registrar")

    services = {"whois": ServiceStatus("whois", "live", "live mode")}
    runner = ToolRunner(Settings(mode="live"), services=services, recordings=store)

    result = runner.execute(query)

    assert result.status == "unavailable"
    assert result.summary != "cached registrar"
    assert "HTTP 503" in result.summary


def test_a_javascript_render_failure_never_records_the_recording_host_path():
    """Playwright names the *author's* browser cache when Chromium is missing.

    Recorded verbatim into a WEB_CRAWL fixture, that ships the recording
    machine's username inside the bundle and means nothing on any other host.
    Only the cause is kept, on one line, with absolute paths removed.
    """
    from tiblender_runtime.tools import _machine_independent_reason

    error = RuntimeError(
        "BrowserType.launch: Executable doesn't exist at "
        "/home/alice/.cache/ms-playwright/chromium_headless_shell-1155/chrome-linux/headless_shell\n"
        "=================================\n"
        "Looks like Playwright was just installed or updated.\n"
    )

    reason = _machine_independent_reason(error)

    assert "/home/alice" not in reason
    assert "\n" not in reason
    assert "BrowserType.launch" in reason
    assert len(reason) <= 300


def test_the_bounded_reason_keeps_urls_and_short_messages_intact():
    from tiblender_runtime.tools import _machine_independent_reason

    assert _machine_independent_reason(RuntimeError("timeout after 30000ms")) == "timeout after 30000ms"
    assert "https://example.test/a/b" in _machine_independent_reason(
        RuntimeError("navigation to https://example.test/a/b failed")
    )
    assert _machine_independent_reason(RuntimeError("")) == "RuntimeError"
