"""End-to-end offline three-cycle sample run against the committed expected output.

This is the determinism/regression gate for the multi-cycle scheduler: running
the bundled fixtures/tool recordings through ``run_cycles`` with the real (default)
hyperparameters must reproduce the exact numbers committed to
``sample_runs/offline_three_cycles/expected/``. It is marked ``slow`` (it runs
real RL training three times) but carries no marker filter in this repo, so
it still runs by default.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tiblender_runtime.demo_report import write_demo_report
from tiblender_runtime.scheduler import run_cycles
from tiblender_runtime.settings import Settings
from tiblender_runtime.verify import verify_sample

sys.path.insert(0, str(Path(__file__).resolve().parent))

from expected_numbers import (  # noqa: E402
    MIN_TOOL_REPLAY_RATE,
    expected_overall_replay_rate,
    expected_replay_rates,
)

_EXPECTED_DIR = Path("sample_runs/offline_three_cycles/expected")
_EXPECTED_PATH = _EXPECTED_DIR / "summary.json"


@pytest.mark.slow
def test_offline_three_cycle_sample_run_matches_committed_expected_output(tmp_path: Path):
    assert _EXPECTED_PATH.exists(), "expected summary must be generated and committed (write-expected)"

    settings = Settings(mode="offline", seed=20260909)
    output_root = tmp_path / "output"

    records = run_cycles(settings, cycles=3, output_root=output_root, interval_seconds=0)

    assert [record.cycle_index for record in records] == [1, 2, 3]
    assert verify_sample(output_root, _EXPECTED_PATH) == 0

    # `verify-sample` only compares the *derived* expected/summary.json (post,
    # cluster and report counts, report ids, IoCs, dynamic queries, trend
    # counts, service modes) -- it never reads the committed per-cycle
    # cycle_summary.json copies, which could therefore go stale unnoticed
    # after a schema change. Diff them here so they cannot.
    for cycle_index in (1, 2, 3):
        name = f"cycle_{cycle_index:03d}/cycle_summary.json"
        committed = json.loads((_EXPECTED_DIR / name).read_text(encoding="utf-8"))
        produced = json.loads((output_root / name).read_text(encoding="utf-8"))
        assert committed == produced, (
            f"{name} is stale; regenerate with "
            "`tiblender write-expected --output-root <run> --expected-dir "
            "sample_runs/offline_three_cycles/expected`"
        )

    # `demo/run_demo.sh` ends by writing this document, and the README names
    # its path. It must hold every report the run published -- a reader who
    # opens only output/report.json must not be reading a subset -- and be
    # byte-stable, since the demo is run more than once.
    report_json = tmp_path / "report.json"
    write_demo_report(output_root, report_json)
    first = report_json.read_bytes()
    write_demo_report(output_root, report_json)
    assert report_json.read_bytes() == first

    document = json.loads(first)
    expected_summary = json.loads(_EXPECTED_PATH.read_text(encoding="utf-8"))
    expected_ids = {
        report_id
        for cycle in expected_summary["cycles"].values()
        for report_id in cycle["report_ids"]
    }
    assert {report["report_id"] for report in document["reports"]} == expected_ids
    # Not the wall clock: the last cycle's collection window.
    assert document["generated_at"] == document["cycles"][-1]["window"]["end"]


def test_every_tool_replays_at_least_the_documented_coverage_floor():
    """The bundle must actually stand in for the tools it claims to.

    `README.md` says recorded tool responses stand in for the live external
    services. That was false for GITHUB_SEARCH (0 % of calls replayed --
    every recording was keyed under a parameter name the runtime never
    issues) and misleading for SEARCH_ENGINE (8 %), and nothing failed. The
    floor is derived from the committed expected run, so it cannot drift
    away from what the pipeline does.
    """
    rates = expected_replay_rates()
    assert rates, (
        "the committed expected summary carries no tool_recording_replay block; "
        "regenerate it with `tiblender write-expected`"
    )
    below = {
        tool_type: row["hit_rate"]
        for tool_type, row in rates.items()
        if row["hit_rate"] < MIN_TOOL_REPLAY_RATE
    }
    assert not below, (
        f"these tools replay below the {MIN_TOOL_REPLAY_RATE:.0%} floor: {below}. "
        "Run `make harvest-tool-queries`, then record the harvested keys."
    )
    assert expected_overall_replay_rate() >= MIN_TOOL_REPLAY_RATE


def test_the_replay_coverage_block_covers_every_recording_backed_tool():
    """A tool that stopped being called at all would pass the floor test by
    vanishing from the table; pin the tool set instead."""
    expected_tools = {
        "GITHUB_SEARCH",
        "NVD_CVE_LOOKUP",
        "PASSIVE_DNS_LOOKUP",
        "SEARCH_ENGINE",
        "TLS_CT_LOOKUP",
        "WEB_CRAWL",
        "WHOIS_LOOKUP",
    }
    assert set(expected_replay_rates()) == expected_tools
