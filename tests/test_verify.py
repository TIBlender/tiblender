"""Tests for the deterministic output summarizer and expected-output verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiblender_runtime.verify import (
    VerificationError,
    compare_summaries,
    summarize_output,
    verify_sample,
    write_expected,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_cycle(cycle_dir: Path, *, report_id: str, domains: list[str], dynamic_queries: list[str]) -> None:
    _write_json(
        cycle_dir / "cycle_summary.json",
        {
            "posts_loaded": 4,
            "threat_related_posts": 3,
            "clusters_generated": 1,
            "reports_generated": 1,
            "feedback": {
                "trend_counts": {"VOLUME": 1, "PROPAGATION": 0, "ESCALATION": 0, "EARLY_WARNING": 1},
                "training_status": "bc_and_dqn",
                "history_size": 5,
            },
            "generated_at": "2026-09-01T00:00:00Z",
        },
    )
    reports_dir = cycle_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        reports_dir / f"{report_id}.json",
        {
            "report_id": report_id,
            "generated_at": "2026-09-01T00:05:00Z",
            "iocs": {"domains": domains, "ips": [], "cves": [], "hashes": [], "urls": []},
        },
    )
    _write_json(
        cycle_dir / "dynamic_queries.json",
        [{"query": query, "trend_types": ["VOLUME"]} for query in dynamic_queries],
    )
    _write_json(
        cycle_dir / "collection_report.json",
        {"per_platform": {"x": {"mode": "fixture", "posts": 4, "errors": [], "stats": {}}}, "generated_at": "2026-09-01T00:00:01Z"},
    )


def _build_output_root(root: Path) -> None:
    _build_cycle(root / "cycle_001", report_id="RPT-1", domains=["evil.example"], dynamic_queries=["evil.example"])
    _write_json(root / "run_summary.json", {"mode": "offline", "seed": 42, "cycles": [], "services": {}, "fixed_clock": True, "warnings": []})


# ---------------------------------------------------------------------------
# summarize_output
# ---------------------------------------------------------------------------


def test_summarize_output_reduces_one_cycle_to_the_comparable_numbers(tmp_path: Path):
    _build_output_root(tmp_path)

    summary = summarize_output(tmp_path)

    assert summary["mode"] == "offline"
    assert summary["seed"] == 42
    cycle = summary["cycles"]["cycle_001"]
    assert cycle["posts_loaded"] == 4
    assert cycle["threat_related"] == 3
    assert cycle["clusters"] == 1
    assert cycle["reports"] == 1
    assert cycle["report_ids"] == ["RPT-1"]
    assert cycle["iocs_by_report"] == {"RPT-1": ["evil.example"]}
    assert cycle["dynamic_queries"] == ["evil.example"]
    assert cycle["trend_counts"] == {"EARLY_WARNING": 1, "ESCALATION": 0, "PROPAGATION": 0, "VOLUME": 1}
    assert cycle["training_status"] == "bc_and_dqn"
    assert cycle["history_size"] == 5
    assert cycle["services"] == {"x": "fixture"}
    # Wall-clock timestamps are not part of the deterministic summary.
    assert "generated_at" not in cycle


def test_summarize_output_ignores_wall_clock_fields_when_run_summary_missing(tmp_path: Path):
    _build_cycle(tmp_path / "cycle_001", report_id="RPT-1", domains=[], dynamic_queries=[])

    summary = summarize_output(tmp_path)

    assert summary["mode"] == ""
    assert summary["seed"] is None
    assert "cycle_001" in summary["cycles"]


# ---------------------------------------------------------------------------
# compare_summaries
# ---------------------------------------------------------------------------


def test_compare_summaries_is_empty_for_identical_summaries(tmp_path: Path):
    _build_output_root(tmp_path)
    summary = summarize_output(tmp_path)

    assert compare_summaries(summary, summary) == []


def test_compare_summaries_reports_a_diff_when_a_report_id_changes(tmp_path: Path):
    _build_output_root(tmp_path)
    expected = summarize_output(tmp_path)

    actual = json.loads(json.dumps(expected))
    actual["cycles"]["cycle_001"]["report_ids"] = ["RPT-2"]
    actual["cycles"]["cycle_001"]["iocs_by_report"] = {"RPT-2": ["evil.example"]}

    diffs = compare_summaries(actual, expected)

    assert diffs
    assert any("cycles.cycle_001.report_ids" in diff for diff in diffs)


def test_compare_summaries_ignores_generated_at_and_timestamp_keys():
    expected = {"mode": "offline", "generated_at": "2026-01-01T00:00:00Z", "cycles": {}}
    actual = {"mode": "offline", "generated_at": "2026-02-02T00:00:00Z", "timestamp": "irrelevant", "cycles": {}}

    assert compare_summaries(actual, expected) == []


def test_compare_summaries_reports_missing_and_unexpected_keys():
    expected = {"a": 1, "b": 2}
    actual = {"a": 1, "c": 3}

    diffs = compare_summaries(actual, expected)

    assert any("b: missing in actual" in diff for diff in diffs)
    assert any("c: unexpected in actual" in diff for diff in diffs)


# ---------------------------------------------------------------------------
# write_expected / verify_sample round trip
# ---------------------------------------------------------------------------


def test_write_expected_then_verify_sample_round_trips_to_zero(tmp_path: Path):
    output_root = tmp_path / "output"
    _build_output_root(output_root)
    expected_dir = tmp_path / "expected"

    write_expected(output_root, expected_dir)

    assert (expected_dir / "summary.json").exists()
    assert (expected_dir / "cycle_001" / "cycle_summary.json").exists()
    assert (expected_dir / "cycle_001" / "dynamic_queries.json").exists()

    assert verify_sample(output_root, expected_dir / "summary.json") == 0


def test_verify_sample_returns_one_and_prints_diffs_on_mismatch(tmp_path: Path, capsys):
    output_root = tmp_path / "output"
    _build_output_root(output_root)
    expected_dir = tmp_path / "expected"
    write_expected(output_root, expected_dir)

    # Mutate the actual output after the expected snapshot was taken.
    _build_cycle(output_root / "cycle_001", report_id="RPT-CHANGED", domains=["evil.example"], dynamic_queries=["evil.example"])

    exit_code = verify_sample(output_root, expected_dir / "summary.json")

    assert exit_code == 1
    printed = capsys.readouterr().out
    assert "report_ids" in printed


# ---------------------------------------------------------------------------
# VerificationError: missing/malformed inputs (never a raw FileNotFoundError
# or json.JSONDecodeError).
# ---------------------------------------------------------------------------


def test_verify_sample_raises_verification_error_for_a_missing_expected_file(tmp_path: Path):
    output_root = tmp_path / "output"
    _build_output_root(output_root)

    with pytest.raises(VerificationError, match="missing"):
        verify_sample(output_root, tmp_path / "does-not-exist" / "summary.json")


def test_verify_sample_raises_verification_error_for_malformed_expected_json(tmp_path: Path):
    output_root = tmp_path / "output"
    _build_output_root(output_root)
    expected_path = tmp_path / "summary.json"
    expected_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(VerificationError, match="malformed JSON"):
        verify_sample(output_root, expected_path)


def test_verify_sample_raises_verification_error_for_a_missing_output_root(tmp_path: Path):
    output_root = tmp_path / "output"
    _build_output_root(output_root)
    expected_dir = tmp_path / "expected"
    write_expected(output_root, expected_dir)

    with pytest.raises(VerificationError, match="output root"):
        verify_sample(tmp_path / "does-not-exist-output", expected_dir / "summary.json")


def test_summarize_output_raises_verification_error_for_malformed_cycle_summary_json(tmp_path: Path):
    output_root = tmp_path / "output"
    _build_output_root(output_root)
    (output_root / "cycle_001" / "cycle_summary.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(VerificationError, match="malformed JSON"):
        summarize_output(output_root)


def test_write_expected_raises_verification_error_for_a_missing_output_root(tmp_path: Path):
    with pytest.raises(VerificationError, match="output root"):
        write_expected(tmp_path / "does-not-exist-output", tmp_path / "expected")
