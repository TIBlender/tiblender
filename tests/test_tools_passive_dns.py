"""PASSIVE_DNS_LOOKUP against VirusTotal: dispatch, parsing and failure modes.

Split out of tests/test_tools.py to keep both files under the 800-line budget.
"""

import io
import json
import urllib.error

import pytest

from tiblender_runtime.models import ToolQuery
from tiblender_runtime.settings import Settings
from tiblender_runtime.tools import ToolRunner


class _EmptyHeaders:
    """Minimal stand-in for an http.client.HTTPMessage in VirusTotal tests."""

    @staticmethod
    def get_content_charset():
        return "utf-8"

    @staticmethod
    def get(_name, default=None):
        return default


def _vt_resolution(host: str, ip: str, *, malicious: int = 3, suspicious: int = 1) -> dict:
    return {
        "attributes": {
            "host_name": host,
            "ip_address": ip,
            "date": 1782489793,
            "resolver": "VirusTotal",
            "host_name_last_analysis_stats": {"malicious": malicious, "suspicious": suspicious},
            "ip_address_last_analysis_stats": {"malicious": 2, "suspicious": 0},
        }
    }


def test_passive_dns_defaults_to_virustotal_and_extracts_resolutions(monkeypatch):
    calls = {}

    def fake_vt_get_json(url, *, headers, timeout, max_bytes=None, sleeper=None):
        del timeout, sleeper
        calls["url"] = url
        calls["headers"] = headers
        return ({"data": [_vt_resolution("bad.example.net", "203.0.113.10")]}, url)

    monkeypatch.setattr("tiblender_runtime.tools._vt_get_json", fake_vt_get_json)
    runner = ToolRunner(Settings(enable_real_tools=True, mode="live", virustotal_api_key="vtkey"))

    result = runner.execute(ToolQuery("Infrastructure", "PASSIVE_DNS_LOOKUP", {"domain": "example.net"}))

    assert calls["url"] == "https://www.virustotal.com/api/v3/domains/example.net/resolutions?limit=20"
    assert calls["headers"]["x-apikey"] == "vtkey"
    assert result.status == "success"
    assert "VirusTotal passive DNS for example.net: 1 resolution(s)" in result.summary
    assert "bad.example.net" in result.discovered_iocs.domains
    raw = json.loads(result.raw_json)
    assert raw["provider"] == "virustotal"
    assert raw["malicious_count"] == 5 and raw["suspicious_count"] == 1
    assert raw["records"][0]["resolver"] == "VirusTotal"
    assert raw["records"][0]["date"] == 1782489793


def test_passive_dns_virustotal_uses_the_ip_addresses_collection_for_an_ip(monkeypatch):
    calls = {}

    def fake_vt_get_json(url, *, headers, timeout, max_bytes=None, sleeper=None):
        del headers, timeout, sleeper
        calls["url"] = url
        return ({"data": [_vt_resolution("host.example.org", "198.51.100.7")]}, url)

    monkeypatch.setattr("tiblender_runtime.tools._vt_get_json", fake_vt_get_json)
    runner = ToolRunner(Settings(enable_real_tools=True, mode="live", virustotal_api_key="vtkey"))

    result = runner.execute(ToolQuery("Infrastructure", "PASSIVE_DNS_LOOKUP", {"ip": "198.51.100.7"}))

    assert calls["url"] == "https://www.virustotal.com/api/v3/ip_addresses/198.51.100.7/resolutions?limit=20"
    assert result.status == "success" and "host.example.org" in result.summary


def test_passive_dns_virustotal_without_a_key_is_unavailable():
    runner = ToolRunner(Settings(enable_real_tools=True, mode="live"))

    result = runner.execute(ToolQuery("Infrastructure", "PASSIVE_DNS_LOOKUP", {"domain": "example.net"}))

    assert result.status == "unavailable" and "VIRUSTOTAL_API_KEY" in result.error


@pytest.mark.parametrize(
    ("status", "marker"),
    [(401, "rejected the API key"), (403, "rejected the API key"), (404, "not found")],
)
def test_passive_dns_virustotal_auth_and_missing_statuses_are_unavailable(monkeypatch, status, marker):
    from tiblender_runtime.tools import _VirusTotalHttpError

    def fake_vt_get_json(url, *, headers, timeout, max_bytes=None, sleeper=None):
        del url, headers, timeout, sleeper
        raise _VirusTotalHttpError(status, "denied")

    monkeypatch.setattr("tiblender_runtime.tools._vt_get_json", fake_vt_get_json)
    runner = ToolRunner(Settings(enable_real_tools=True, mode="live", virustotal_api_key="vtkey"))

    result = runner.execute(ToolQuery("Infrastructure", "PASSIVE_DNS_LOOKUP", {"domain": "example.net"}))

    assert result.status == "unavailable" and marker in result.error


def test_vt_get_json_retries_a_429_once_after_retry_after(monkeypatch):
    from tiblender_runtime import tools as tools_module

    attempts = {"count": 0}
    slept: list[float] = []

    class _Response:
        headers = _EmptyHeaders()

        def read(self, _size):
            return b'{"data": []}'

        def geturl(self):
            return "https://www.virustotal.com/api/v3/domains/x/resolutions?limit=40"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout=30):
        del request, timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise urllib.error.HTTPError(
                "https://www.virustotal.com/", 429, "Too Many Requests", {"Retry-After": "5"}, io.BytesIO(b"slow down")
            )
        return _Response()

    monkeypatch.setattr(tools_module.urllib.request, "urlopen", fake_urlopen)

    data, _url = tools_module._vt_get_json(
        "https://www.virustotal.com/api/v3/domains/x/resolutions?limit=40",
        headers={"x-apikey": "k"},
        timeout=10,
        sleeper=slept.append,
    )

    assert attempts["count"] == 2 and slept == [5.0] and data == {"data": []}


def test_vt_get_json_gives_up_after_one_429_retry(monkeypatch):
    from tiblender_runtime import tools as tools_module
    from tiblender_runtime.tools import _VirusTotalHttpError

    def fake_urlopen(request, timeout=30):
        del request, timeout
        raise urllib.error.HTTPError(
            "https://www.virustotal.com/", 429, "Too Many Requests", {"Retry-After": "1"}, io.BytesIO(b"slow down")
        )

    monkeypatch.setattr(tools_module.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(_VirusTotalHttpError) as excinfo:
        tools_module._vt_get_json(
            "https://www.virustotal.com/api/v3/domains/x/resolutions?limit=40",
            headers={"x-apikey": "k"},
            timeout=10,
            sleeper=lambda _seconds: None,
        )

    assert excinfo.value.status == 429


def test_a_recorded_recording_replays_through_the_tool_runner(tmp_path):
    """A recorded tool response is replayed by recording key, independent of settings."""
    from tiblender_runtime.services import ServiceStatus
    from tiblender_runtime.tool_fixtures import ToolRecordingStore, recording_key

    parameters = {"domain": "example.net"}
    query = ToolQuery("Infrastructure", "PASSIVE_DNS_LOOKUP", parameters)
    store = ToolRecordingStore(tmp_path)

    key_when_recorded = recording_key("PASSIVE_DNS_LOOKUP", parameters)
    recorded = ToolRunner(
        Settings(enable_real_tools=True, mode="offline"),
        services={"passive_dns": ServiceStatus("passive_dns", "fixture", "test")},
        recordings=store,
    )
    store.record(
        "PASSIVE_DNS_LOOKUP",
        parameters,
        recorded.execute(query),  # unavailable: no recording yet, which is fine to record
        source_url="https://www.virustotal.com/api/v3/domains/example.net/resolutions?limit=40",
    )

    assert (tmp_path / "PASSIVE_DNS_LOOKUP" / f"{key_when_recorded}.json").exists()

    replayed = ToolRunner(
        Settings(enable_real_tools=True, mode="offline"),
        services={"passive_dns": ServiceStatus("passive_dns", "fixture", "test")},
        recordings=store,
    ).execute(query)

    assert json.loads(replayed.raw_json)["source"] == "recording"
    assert json.loads(replayed.raw_json)["recording_source_url"].startswith("https://www.virustotal.com/")


def test_a_recorded_passive_dns_answer_is_truncated_to_the_configured_limit(tmp_path):
    """Recordings captured under the old limit=40 request replay as 20.

    The deployed tool truncates the provider payload to its first 20
    resolutions, and the live request here now asks for exactly that many. The
    bundled recordings predate the change, so the same cap is applied on
    replay -- deterministically, over the recording's stored order -- instead
    of re-recording every provider answer. The queried indicator is the query
    side of the pair, not a resolution, and is always kept.
    """
    import json as _json

    from tiblender_runtime.services import ServiceStatus
    from tiblender_runtime.tool_fixtures import ToolRecordingStore

    parameters = {"ip": "198.51.100.7"}
    query = ToolQuery("Infrastructure", "PASSIVE_DNS_LOOKUP", parameters)
    store = ToolRecordingStore(tmp_path)
    path = tmp_path / "PASSIVE_DNS_LOOKUP"
    path.mkdir(parents=True, exist_ok=True)
    from tiblender_runtime.tool_fixtures import recording_key

    (path / f"{recording_key('PASSIVE_DNS_LOOKUP', parameters)}.json").write_text(
        _json.dumps(
            {
                "tool_type": "PASSIVE_DNS_LOOKUP",
                "parameters": parameters,
                "status": "success",
                "summary": "40 resolutions",
                "discovered_iocs": {
                    "urls": [],
                    "domains": [f"host-{index:02d}.example" for index in range(40)],
                    "ips": ["198.51.100.7"],
                    "hashes": [],
                    "cves": [],
                },
                "raw": {},
                "error": "",
            }
        ),
        encoding="utf-8",
    )

    def _replay(limit: int):
        return ToolRunner(
            Settings(enable_real_tools=True, mode="offline", passive_dns_limit=limit),
            services={"passive_dns": ServiceStatus("passive_dns", "fixture", "test")},
            recordings=store,
        ).execute(query)

    replayed = _replay(20)

    assert len(replayed.discovered_iocs.domains) == 20
    assert replayed.discovered_iocs.domains == [f"host-{index:02d}.example" for index in range(20)]
    # The queried address is the query side of every pair, never a resolution.
    assert replayed.discovered_iocs.ips == ["198.51.100.7"]
    # Non-vacuous: raising the cap really does return the whole recording.
    assert len(_replay(40).discovered_iocs.domains) == 40
