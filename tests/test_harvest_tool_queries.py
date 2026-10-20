"""Tests for the deterministic tool-query harvest step.

`tools/build_fixtures.py` derives tool queries from fixture *post text*
only. The pipeline additionally issues free-text cluster search phrases and
pivots on indicators found in recorded evidence, and those requests were
invisible: they degraded to `status=unavailable` while the package claimed
its recorded responses stood in for the live tools.
`tools/harvest_tool_queries.py` runs the deterministic offline demo, reads
the misses the runtime now logs, and merges them into the query list.

These tests cover the pure parts (merge/collect/format) plus the wiring that
makes the runtime write the misses at all; the credentialed recording step
itself is exercised by `tests/test_record_tool_responses.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.harvest_tool_queries import (
    HARVESTED_FILENAME,
    ORIGIN_HARVESTED,
    ORIGIN_POST_DERIVED,
    collect_misses,
    collect_replay_rates,
    format_rate_table,
    load_harvested,
    merge_queries,
    write_harvested,
)

_FS_ROOT = Path(__file__).resolve().parents[1]
_HARVESTED_PATH = _FS_ROOT / "fixtures" / "tools" / HARVESTED_FILENAME
_QUERIES_PATH = _FS_ROOT / "fixtures" / "tools" / "queries.json"


def test_merge_queries_deduplicates_by_request_key_and_sorts():
    merged = merge_queries(
        [{"tool_type": "WHOIS_LOOKUP", "parameters": {"domain": "b.example"}}],
        [
            {"tool_type": "WHOIS_LOOKUP", "parameters": {"domain": "a.example"}},
            # Same request key as the existing entry, plus an ignored parameter.
            {"tool_type": "WHOIS_LOOKUP", "parameters": {"domain": "b.example", "priority": 3}},
        ],
    )
    assert merged == [
        {"tool_type": "WHOIS_LOOKUP", "parameters": {"domain": "a.example"}, "origin": ORIGIN_HARVESTED},
        {"tool_type": "WHOIS_LOOKUP", "parameters": {"domain": "b.example"}, "origin": ORIGIN_HARVESTED},
    ]


def test_merge_queries_never_harvests_a_corpus_backed_social_tool():
    """A social tool's fixture-mode miss falls back to the Step 1 collected
    corpus; it is not a recording gap and cannot be recorded from a provider.
    """
    merged = merge_queries(
        [],
        [
            {"tool_type": "X_SEARCH", "parameters": {"query": "ransomware"}},
            {"tool_type": "TELEGRAM_SEARCH", "parameters": {"query": "ransomware"}},
            {"tool_type": "SEARCH_ENGINE", "parameters": {"query": "ransomware"}},
        ],
    )
    assert merged == [
        {"tool_type": "SEARCH_ENGINE", "parameters": {"query": "ransomware"}, "origin": ORIGIN_HARVESTED}
    ]


def test_merge_queries_drops_entries_with_no_key_bearing_parameter():
    assert merge_queries([], [{"tool_type": "WHOIS_LOOKUP", "parameters": {"priority": 1}}]) == []


def test_merge_queries_is_idempotent():
    once = merge_queries([], [{"tool_type": "SEARCH_ENGINE", "parameters": {"query": "supply chain"}}])
    assert merge_queries(once, once) == once


def test_load_harvested_accepts_both_shapes_and_a_missing_file(tmp_path):
    assert load_harvested(tmp_path / "absent.json") == []

    listed = tmp_path / "list.json"
    listed.write_text(json.dumps([{"tool_type": "SEARCH_ENGINE", "parameters": {"query": "x"}}]), encoding="utf-8")
    assert load_harvested(listed)[0]["tool_type"] == "SEARCH_ENGINE"

    documented = tmp_path / "doc.json"
    write_harvested(documented, [{"tool_type": "SEARCH_ENGINE", "parameters": {"query": "x"}}])
    assert load_harvested(documented) == [{"tool_type": "SEARCH_ENGINE", "parameters": {"query": "x"}}]
    assert "note" in json.loads(documented.read_text(encoding="utf-8"))


def test_load_harvested_rejects_a_malformed_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"queries": "not-a-list"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_harvested(path)


def _write_run(root: Path, cycles: dict[str, tuple[dict, list[dict]]]) -> None:
    for label, (replay, misses) in cycles.items():
        cycle = root / label
        (cycle / "intermediate").mkdir(parents=True)
        (cycle / "cycle_summary.json").write_text(
            json.dumps({"tool_recording_replay": replay}), encoding="utf-8"
        )
        (cycle / "intermediate" / "tool_recording_misses.json").write_text(
            json.dumps({"misses": misses}), encoding="utf-8"
        )


def test_collect_misses_and_rates_aggregate_across_cycles(tmp_path):
    _write_run(
        tmp_path,
        {
            "cycle_001": (
                {"by_tool": {"WHOIS_LOOKUP": {"calls": 4, "hits": 3, "misses": 1}}},
                [{"tool_type": "WHOIS_LOOKUP", "parameters": {"domain": "a.example"}}],
            ),
            "cycle_002": (
                {"by_tool": {"WHOIS_LOOKUP": {"calls": 6, "hits": 6, "misses": 0}}},
                # The same key missing again in a later cycle is one entry.
                [{"tool_type": "WHOIS_LOOKUP", "parameters": {"domain": "a.example"}}],
            ),
        },
    )
    assert collect_misses(tmp_path) == [
        {"tool_type": "WHOIS_LOOKUP", "parameters": {"domain": "a.example"}, "origin": ORIGIN_HARVESTED}
    ]
    assert collect_replay_rates(tmp_path) == {
        "WHOIS_LOOKUP": {"calls": 10, "hits": 9, "misses": 1, "hit_rate": 0.9}
    }


def test_format_rate_table_has_a_row_per_tool_and_an_overall_row():
    table = format_rate_table(
        {
            "SEARCH_ENGINE": {"calls": 10, "hits": 10, "misses": 0, "hit_rate": 1.0},
            "WHOIS_LOOKUP": {"calls": 10, "hits": 5, "misses": 5, "hit_rate": 0.5},
        }
    )
    assert "SEARCH_ENGINE" in table and "WHOIS_LOOKUP" in table
    assert table.splitlines()[-1].startswith("ALL")
    assert "75.0 %" in table


def _request_key(entry: dict) -> str:
    return json.dumps({"tool_type": entry["tool_type"], "parameters": entry["parameters"]}, sort_keys=True)


def test_the_shipped_harvested_file_is_merged_into_queries_json():
    """The committed harvest is what makes `make build-fixtures` reproducible
    with no network: the builder reads it, it never re-derives it."""
    harvested = load_harvested(_HARVESTED_PATH)
    assert harvested, "fixtures/tools/harvested_queries.json is empty"
    queries = json.loads(_QUERIES_PATH.read_text(encoding="utf-8"))
    keys = {_request_key(entry) for entry in queries}
    missing = [entry for entry in harvested if _request_key(entry) not in keys]
    assert not missing, f"harvested queries absent from queries.json: {missing}"


def test_every_shipped_query_declares_where_it_came_from():
    """A reader of queries.json alone must be able to tell the two
    passes apart: what the builder derives from fixture post text, and what
    the offline demo was observed to issue."""
    queries = json.loads(_QUERIES_PATH.read_text(encoding="utf-8"))
    assert queries
    origins = {entry.get("origin") for entry in queries}
    assert origins <= {ORIGIN_POST_DERIVED, ORIGIN_HARVESTED}
    assert origins == {ORIGIN_POST_DERIVED, ORIGIN_HARVESTED}, "both passes must be represented"

    harvested_keys = {_request_key(entry) for entry in load_harvested(_HARVESTED_PATH)}
    for entry in queries:
        if entry["origin"] == ORIGIN_HARVESTED:
            assert _request_key(entry) in harvested_keys


def test_origin_is_not_part_of_the_recording_key():
    """Adding provenance must not move a single recording file."""
    from tiblender_runtime.tool_fixtures import recording_key

    plain = {"tool_type": "WHOIS_LOOKUP", "parameters": {"domain": "a.example"}}
    tagged = {**plain, "origin": ORIGIN_HARVESTED}
    assert recording_key(plain["tool_type"], plain["parameters"]) == recording_key(
        tagged["tool_type"], tagged["parameters"]
    )


def test_a_post_derived_request_stays_post_derived_when_it_is_also_harvested():
    merged = merge_queries(
        [{"tool_type": "WHOIS_LOOKUP", "parameters": {"domain": "a.example"}, "origin": ORIGIN_POST_DERIVED}],
        [{"tool_type": "WHOIS_LOOKUP", "parameters": {"domain": "a.example"}, "origin": ORIGIN_HARVESTED}],
    )
    assert merged == [
        {"tool_type": "WHOIS_LOOKUP", "parameters": {"domain": "a.example"}, "origin": ORIGIN_POST_DERIVED}
    ]


def test_an_out_of_tree_rebuild_still_finds_the_committed_harvested_queries(tmp_path, capsys):
    """`--out <scratch>/collectors` used to silently drop the second pass and
    yield a smaller queries.json -- a false mismatch for anyone checking the rebuild."""
    from tools.build_fixtures import _resolve_harvested_path

    resolved = _resolve_harvested_path(tmp_path / "collectors")

    assert resolved == _HARVESTED_PATH
    assert "using the committed" in capsys.readouterr().out


def test_a_cycle_writes_its_recording_misses_file(tmp_path):
    """End-to-end wiring: the runtime must actually emit the file the harvest
    reads, with the deterministic shape it expects.
    """
    from dataclasses import replace as dataclass_replace

    from tiblender_runtime.settings import Settings
    from tiblender_runtime.workflow import run_graph_cycle

    posts = tmp_path / "posts.jsonl"
    posts.write_text(
        json.dumps(
            {
                "id": "p1",
                "platform": "x",
                "content": (
                    "Active exploitation of CVE-2026-99999 with payloads staged on "
                    "harvest-probe.example and 203.0.113.7"
                ),
                "created_at": "2026-04-01T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    settings = dataclass_replace(
        Settings(
            allow_heuristic_reports=True,
            mode="offline",
            enable_real_tools=True,
            input_path=posts,
            output_dir=tmp_path / "cycle_001",
        ),
        fixtures_dir=tmp_path / "empty_fixtures",
    )
    run_graph_cycle(settings, cycle_index=1)

    payload = json.loads(
        (tmp_path / "cycle_001" / "intermediate" / "tool_recording_misses.json").read_text(encoding="utf-8")
    )
    assert set(payload) == {"note", "replay", "misses"}
    assert payload["replay"]["calls"] >= 1
    # No recording store at all, so nothing can replay.
    assert payload["replay"]["hit_rate"] == 0.0
    assert payload["misses"], "a cycle with no recordings must report its missed keys"
    for miss in payload["misses"]:
        assert set(miss) == {"tool_type", "parameters", "recording_key", "calls"}
        assert miss["parameters"], "a harvestable miss must carry the request parameters"
    # And the same information reaches cycle_summary.json, which verify-sample pins.
    summary = json.loads((tmp_path / "cycle_001" / "cycle_summary.json").read_text(encoding="utf-8"))
    assert summary["tool_recording_replay"]["by_tool"]
