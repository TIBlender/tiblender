import json
from pathlib import Path

from tiblender_runtime.models import IocSet, ToolQuery, ToolResult
from tiblender_runtime.services import ServiceStatus
from tiblender_runtime.settings import Settings
from tiblender_runtime.tool_fixtures import RecordingCorruptedError, ToolRecordingStore, recording_key
from tiblender_runtime.tools import ToolRunner


def test_recording_key_is_stable_and_case_insensitive():
    assert recording_key("WHOIS_LOOKUP", {"domain": "Example.NET"}) == recording_key("WHOIS_LOOKUP", {"domain": "example.net"})
    assert recording_key("WHOIS_LOOKUP", {"domain": "a.net"}) != recording_key("PASSIVE_DNS_LOOKUP", {"domain": "a.net"})


def test_record_then_lookup_roundtrip(tmp_path: Path):
    store = ToolRecordingStore(tmp_path)
    query = ToolQuery("Infrastructure", "WHOIS_LOOKUP", {"domain": "example.net"})
    result = ToolResult(tool_type="WHOIS_LOOKUP", query_params=query.parameters, summary="registrar X", discovered_iocs=IocSet(domains=["example.net"]))
    path = store.record("WHOIS_LOOKUP", query.parameters, result, source_url="https://rdap.org/domain/example.net")
    assert path.exists()
    payload = store.lookup("WHOIS_LOOKUP", {"domain": "EXAMPLE.net"})
    assert payload["summary"] == "registrar X" and payload["discovered_iocs"]["domains"] == ["example.net"]
    assert store.count() == 1


def test_fixture_mode_uses_recording_and_marks_source(tmp_path: Path):
    store = ToolRecordingStore(tmp_path)
    query = ToolQuery("Technical", "NVD_CVE_LOOKUP", {"cve": "CVE-2021-44228"})
    store.record("NVD_CVE_LOOKUP", query.parameters, ToolResult(tool_type="NVD_CVE_LOOKUP", query_params=query.parameters, summary="log4shell", discovered_iocs=IocSet(cves=["CVE-2021-44228"])), source_url="https://services.nvd.nist.gov/")
    services = {"nvd": ServiceStatus("nvd", "fixture", "offline")}
    runner = ToolRunner(Settings(mode="offline"), services=services, recordings=store)
    result = runner.execute(query)
    assert result.status == "success" and "log4shell" in result.summary
    assert json.loads(result.raw_json)["source"] == "recording"


def test_fixture_mode_without_recording_is_unavailable(tmp_path: Path):
    runner = ToolRunner(Settings(mode="offline"), services={"whois": ServiceStatus("whois", "fixture", "offline")}, recordings=ToolRecordingStore(tmp_path))
    result = runner.execute(ToolQuery("Infrastructure", "WHOIS_LOOKUP", {"domain": "nowhere.example"}))
    assert result.status == "unavailable" and "recorded tool response" in result.error


def test_lookup_raises_recording_corrupted_error_for_unparseable_file(tmp_path: Path):
    store = ToolRecordingStore(tmp_path)
    query = ToolQuery("Infrastructure", "WHOIS_LOOKUP", {"domain": "broken.example"})
    path = store.record(
        "WHOIS_LOOKUP",
        query.parameters,
        ToolResult(tool_type="WHOIS_LOOKUP", query_params=query.parameters, summary="registrar X"),
        source_url="https://rdap.org/domain/broken.example",
    )
    path.write_text("{not valid json", encoding="utf-8")

    try:
        store.lookup("WHOIS_LOOKUP", query.parameters)
        raise AssertionError("expected RecordingCorruptedError")
    except RecordingCorruptedError as error:
        assert str(path) in str(error)


def test_execute_returns_unavailable_for_corrupted_recording_instead_of_raising(tmp_path: Path):
    store = ToolRecordingStore(tmp_path)
    query = ToolQuery("Infrastructure", "WHOIS_LOOKUP", {"domain": "broken.example"})
    path = store.record(
        "WHOIS_LOOKUP",
        query.parameters,
        ToolResult(tool_type="WHOIS_LOOKUP", query_params=query.parameters, summary="registrar X"),
        source_url="https://rdap.org/domain/broken.example",
    )
    # Simulate a crash mid-write or a hand-corrupted fixture on disk.
    path.write_text("{not valid json", encoding="utf-8")

    services = {"whois": ServiceStatus("whois", "fixture", "offline")}
    runner = ToolRunner(Settings(mode="offline"), services=services, recordings=store)

    result = runner.execute(query)

    assert result.status == "unavailable"
    assert "corrupted" in result.error or "unreadable" in result.error


# ---------------------------------------------------------------------------
# ToolRecordingReplayLog
# ---------------------------------------------------------------------------


def _log():
    from tiblender_runtime.tool_fixtures import ToolRecordingReplayLog

    return ToolRecordingReplayLog()


def test_replay_log_counts_hits_and_misses_per_tool():
    log = _log()
    log.observe("WHOIS_LOOKUP", {"domain": "a.example"}, hit=True)
    log.observe("WHOIS_LOOKUP", {"domain": "b.example"}, hit=False)
    log.observe("SEARCH_ENGINE", {"query": "supply chain"}, hit=False)

    rates = log.rates()
    assert rates["calls"] == 3
    assert rates["hits"] == 1
    assert rates["misses"] == 2
    assert rates["hit_rate"] == round(1 / 3, 4)
    assert rates["by_tool"]["WHOIS_LOOKUP"] == {"calls": 2, "hits": 1, "misses": 1, "hit_rate": 0.5}
    assert rates["by_tool"]["SEARCH_ENGINE"]["hit_rate"] == 0.0


def test_replay_log_reports_every_missed_key_with_its_recording_key():
    from tiblender_runtime.tool_fixtures import recording_key

    log = _log()
    log.observe("SEARCH_ENGINE", {"query": "supply chain"}, hit=False)
    log.observe("SEARCH_ENGINE", {"query": "supply chain"}, hit=False)
    log.observe("NVD_CVE_LOOKUP", {"cve": "CVE-2026-1"}, hit=True)

    misses = log.misses()
    assert [miss["tool_type"] for miss in misses] == ["SEARCH_ENGINE"]
    assert misses[0]["parameters"] == {"query": "supply chain"}
    assert misses[0]["calls"] == 2
    assert misses[0]["recording_key"] == recording_key("SEARCH_ENGINE", {"query": "supply chain"})


def test_replay_log_ignores_parameters_the_recording_key_ignores():
    """`priority`/`timeout` never change the provider response, so two calls
    differing only in those are one request key, not two."""
    log = _log()
    log.observe("TLS_CT_LOOKUP", {"domain": "a.example", "priority": 1}, hit=False)
    log.observe("TLS_CT_LOOKUP", {"domain": "a.example", "priority": 9}, hit=False)
    misses = log.misses()
    assert len(misses) == 1
    assert misses[0]["calls"] == 2
    assert misses[0]["parameters"] == {"domain": "a.example"}


def test_replay_log_payload_is_order_independent():
    """Cluster workers run in parallel, so the payload must depend on the
    request keys, not on which thread observed first."""
    forward, backward = _log(), _log()
    observations = [
        ("WHOIS_LOOKUP", {"domain": "b.example"}, False),
        ("SEARCH_ENGINE", {"query": "z"}, False),
        ("WHOIS_LOOKUP", {"domain": "a.example"}, True),
    ]
    for tool_type, parameters, hit in observations:
        forward.observe(tool_type, parameters, hit=hit)
    for tool_type, parameters, hit in reversed(observations):
        backward.observe(tool_type, parameters, hit=hit)
    assert forward.payload() == backward.payload()


def test_replay_log_is_thread_safe():
    import threading

    log = _log()

    def worker(index: int) -> None:
        for step in range(50):
            log.observe("SEARCH_ENGINE", {"query": f"q{step}"}, hit=index % 2 == 0)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rates = log.rates()
    assert rates["calls"] == 8 * 50
    assert rates["hits"] == 4 * 50
