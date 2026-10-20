"""Multi-cycle scheduler tests: window resolution, cross-cycle state carry,
and run_cycles' offline degrade-gracefully behavior."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import tiblender_runtime.scheduler as scheduler_module
from tiblender_runtime.collectors.base import CollectionWindow
from tiblender_runtime.collectors.fixtures import fixture_path, load_fixture
from tiblender_runtime.scheduler import (
    CycleRecord,
    clear_previous_run,
    cycle_dir,
    resolve_window,
    run_cycles,
    run_one_cycle,
)
from tiblender_runtime.settings import Settings


def _fast_settings(**overrides) -> Settings:
    # Real bundled fixtures/recordings, but a much smaller RL training budget so
    # these wiring tests stay fast; test_offline_three_cycles.py uses the real
    # (default) hyperparameters for the committed expected-output comparison.
    base = dict(
        mode="offline",
        seed=20260909,
        reference_rl_hidden_dim=16,
        reference_rl_bc_epochs=2,
        reference_rl_bc_batch_size=8,
        reference_rl_dqn_episodes=2,
        reference_rl_replay_capacity=2000,
    )
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------------
# cycle_dir
# ---------------------------------------------------------------------------


def test_cycle_dir_formats_three_digit_index(tmp_path: Path):
    assert cycle_dir(tmp_path, 1) == tmp_path / "cycle_001"
    assert cycle_dir(tmp_path, 12) == tmp_path / "cycle_012"


# ---------------------------------------------------------------------------
# resolve_window
# ---------------------------------------------------------------------------


def test_resolve_window_offline_uses_the_x_fixture_window():
    settings = _fast_settings()

    window = resolve_window(settings, cycle_index=1, window_end=None)

    expected = load_fixture(fixture_path(settings.fixtures_dir, "x", 1)).window
    assert window == expected


def test_resolve_window_falls_back_to_ending_at_when_fixture_missing(tmp_path: Path):
    settings = _fast_settings(fixtures_dir=tmp_path / "fixtures")  # no fixtures at all -> x is "unavailable"
    window_end = datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)

    window = resolve_window(settings, cycle_index=7, window_end=window_end)

    assert window == CollectionWindow.ending_at(window_end, settings.collect_window_hours, 7)


def test_resolve_window_falls_back_when_that_specific_cycle_has_no_fixture():
    settings = _fast_settings()
    window_end = datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)

    # Fixtures exist for cycles 1-3 only.
    window = resolve_window(settings, cycle_index=999, window_end=window_end)

    assert window == CollectionWindow.ending_at(window_end, settings.collect_window_hours, 999)


# ---------------------------------------------------------------------------
# run_cycles: happy path over the bundled fixtures
# ---------------------------------------------------------------------------


def test_run_cycles_offline_produces_three_cycle_dirs_and_run_summary(tmp_path: Path):
    settings = _fast_settings()
    sleeps: list[float] = []

    records = run_cycles(
        settings,
        cycles=3,
        output_root=tmp_path / "output",
        interval_seconds=0,
        sleeper=sleeps.append,
    )

    assert [record.cycle_index for record in records] == [1, 2, 3]
    assert all(isinstance(record, CycleRecord) for record in records)
    for index in (1, 2, 3):
        this_dir = tmp_path / "output" / f"cycle_{index:03d}"
        assert (this_dir / "posts.jsonl").exists()
        assert (this_dir / "cycle_summary.json").exists()
        assert (this_dir / "cycle_record.json").exists()
        assert (this_dir / "collection_report.json").exists()

    assert sleeps == [0, 0]  # cycles - 1

    run_summary = json.loads((tmp_path / "output" / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["mode"] == "offline"
    assert run_summary["seed"] == settings.seed
    assert run_summary["fixed_clock"] is True
    assert len(run_summary["cycles"]) == 3
    assert run_summary["services"]["x"] in {"fixture", "live", "unavailable"}
    assert run_summary["warnings"] == []


def test_run_cycles_shares_history_dir_and_policy_state_lives_there(tmp_path: Path):
    settings = _fast_settings()
    output_root = tmp_path / "output"

    run_cycles(settings, cycles=2, output_root=output_root, interval_seconds=0)

    history_dir = output_root / "history"
    assert (history_dir / "collector_state.json").exists()
    policy_state_path = history_dir / "policy_state.json"
    assert policy_state_path.exists()
    policy_state = json.loads(policy_state_path.read_text(encoding="utf-8"))
    assert len(policy_state.get("training_history") or []) >= 1
    assert policy_state.get("report_history") is not None


def test_run_cycles_shares_history_across_cycles_with_a_relative_output_root_named_output(tmp_path: Path, monkeypatch):
    # Regression: resolved_history_dir() used to
    # compare history_dir to its own default *by value*, so the scheduler's
    # explicit history_dir=output_root/"history" was silently discarded
    # whenever output_root was named "output" (the documented/common case),
    # each cycle falling back to its own output_dir instead and isolating
    # history per cycle. A tmp_path-absolute output_root never reproduced
    # this (it never equals the literal default string), so this test uses a
    # RELATIVE output_root=Path("output") from a chdir'd cwd -- the exact
    # shape the documented `run-cycles --output-root output` invocation uses.
    fixtures_dir = Path("fixtures").resolve()
    config_dir = Path("config").resolve()
    settings = _fast_settings(fixtures_dir=fixtures_dir, config_dir=config_dir)
    monkeypatch.chdir(tmp_path)
    output_root = Path("output")

    run_cycles(settings, cycles=2, output_root=output_root, interval_seconds=0)

    shared_policy_state = tmp_path / "output" / "history" / "policy_state.json"
    shared_snapshots = tmp_path / "output" / "history" / "cluster_snapshots.jsonl"
    assert shared_policy_state.exists()
    assert shared_snapshots.exists()
    # Neither cycle_00N/ subtree may have its own isolated history/ directory.
    assert not (tmp_path / "output" / "cycle_001" / "history").exists()
    assert not (tmp_path / "output" / "cycle_002" / "history").exists()

    def _cycle_summary(index: int) -> dict:
        path = tmp_path / "output" / f"cycle_{index:03d}" / "cycle_summary.json"
        return json.loads(path.read_text(encoding="utf-8"))

    summary_1 = _cycle_summary(1)
    summary_2 = _cycle_summary(2)
    # Both cycles must record the exact same (shared) policy_state_path --
    # not e.g. "output/cycle_001/history/..." vs "output/cycle_002/history/...".
    assert summary_1["feedback"]["policy_state_path"] == summary_2["feedback"]["policy_state_path"]
    # Recorded relative to the run's output root (never an absolute host path).
    assert summary_1["feedback"]["policy_state_path"] == "history/policy_state.json"
    # Cycle 2 sees cycle 1's snapshots plus its own -> a strictly larger
    # cumulative history size, proof the history file is genuinely shared
    # rather than each cycle starting from an empty file of its own.
    assert summary_2["feedback"]["history_size"] > summary_1["feedback"]["history_size"]

    line_count = sum(1 for line in shared_snapshots.read_text(encoding="utf-8").splitlines() if line.strip())
    assert line_count == summary_1["clusters_generated"] + summary_2["clusters_generated"]


def test_cluster_snapshot_history_grows_across_cycles(tmp_path: Path):
    settings = _fast_settings()
    output_root = tmp_path / "output"

    def _history_line_count() -> int:
        path = output_root / "history" / "cluster_snapshots.jsonl"
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    run_one_cycle(settings, cycle_index=1, output_root=output_root)
    after_one = _history_line_count()
    run_one_cycle(settings, cycle_index=2, output_root=output_root)
    after_two = _history_line_count()

    assert after_two >= after_one


def test_run_cycles_third_cycle_consumes_second_cycles_dynamic_queries(tmp_path: Path, monkeypatch):
    # Over the bundled fixtures, cycle 1 has no trend history yet (so it emits
    # no dynamic queries), cycle 2 detects EARLY_WARNING trends and writes a
    # non-empty dynamic_queries.json, and cycle 3 is the one that should
    # actually consume it.
    settings = _fast_settings()
    output_root = tmp_path / "output"
    original_collect_cycle = scheduler_module.collect_cycle
    captured_paths: list[Path | None] = []

    def _spy(settings_arg, *, window, dynamic_queries_path, output_dir, services=None):
        captured_paths.append(dynamic_queries_path)
        return original_collect_cycle(
            settings_arg, window=window, dynamic_queries_path=dynamic_queries_path, output_dir=output_dir, services=services
        )

    monkeypatch.setattr(scheduler_module, "collect_cycle", _spy)

    run_cycles(settings, cycles=3, output_root=output_root, interval_seconds=0)

    assert captured_paths[0] is None
    assert captured_paths[1] == cycle_dir(output_root, 1) / "dynamic_queries.json"
    assert captured_paths[2] == cycle_dir(output_root, 2) / "dynamic_queries.json"
    # Cycle 2's dynamic_queries.json must exist and be non-empty for it to be
    # meaningfully "consumed" by cycle 3.
    dynamic_queries = json.loads(captured_paths[2].read_text(encoding="utf-8"))
    assert isinstance(dynamic_queries, list)
    assert len(dynamic_queries) > 0


# ---------------------------------------------------------------------------
# run_cycles: offline degrade-gracefully and failure reporting
# ---------------------------------------------------------------------------


def test_run_cycles_offline_more_cycles_than_fixtures_warns_instead_of_failing(tmp_path: Path):
    settings = _fast_settings()
    output_root = tmp_path / "output"

    records = run_cycles(settings, cycles=5, output_root=output_root, interval_seconds=0)

    # Only 3 fixture cycles are bundled.
    assert [record.cycle_index for record in records] == [1, 2, 3]
    run_summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["warnings"]
    assert "failed_cycle" not in run_summary


def test_run_cycles_writes_partial_summary_and_reraises_on_failure(tmp_path: Path, monkeypatch):
    settings = _fast_settings()
    output_root = tmp_path / "output"

    calls = {"count": 0}
    original_run_one_cycle = scheduler_module.run_one_cycle

    def _flaky(settings_arg, *, cycle_index, output_root, window_end=None, resume=False):
        calls["count"] += 1
        if cycle_index == 2:
            raise RuntimeError("boom")
        return original_run_one_cycle(
            settings_arg, cycle_index=cycle_index, output_root=output_root, window_end=window_end, resume=resume
        )

    monkeypatch.setattr(scheduler_module, "run_one_cycle", _flaky)

    with pytest.raises(RuntimeError, match="boom"):
        run_cycles(settings, cycles=3, output_root=output_root, interval_seconds=0)

    run_summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["failed_cycle"]["cycle_index"] == 2
    assert "boom" in run_summary["failed_cycle"]["error"]
    assert len(run_summary["cycles"]) == 1  # cycle 1 succeeded before the failure


# ---------------------------------------------------------------------------
# Determinism: two independent runs over the same fixtures must agree.
# ---------------------------------------------------------------------------


def test_run_cycles_is_deterministic_across_independent_runs(tmp_path: Path):
    from tiblender_runtime.verify import summarize_output

    settings = _fast_settings()

    run_cycles(settings, cycles=2, output_root=tmp_path / "run_a", interval_seconds=0)
    run_cycles(settings, cycles=2, output_root=tmp_path / "run_b", interval_seconds=0)

    summary_a = summarize_output(tmp_path / "run_a")
    summary_b = summarize_output(tmp_path / "run_b")

    assert summary_a == summary_b


# ---------------------------------------------------------------------------
# Fresh start: a second run into the SAME output root must reproduce the
# first run rather than continue it.
# ---------------------------------------------------------------------------


def test_run_cycles_twice_into_the_same_root_is_identical(tmp_path: Path):
    from tiblender_runtime.verify import summarize_output

    settings = _fast_settings()
    output_root = tmp_path / "output"

    run_cycles(settings, cycles=3, output_root=output_root, interval_seconds=0)
    first = summarize_output(output_root)

    run_cycles(settings, cycles=3, output_root=output_root, interval_seconds=0)
    second = summarize_output(output_root)

    assert second == first


def test_run_cycles_records_fresh_start_in_run_summary(tmp_path: Path):
    settings = _fast_settings()
    output_root = tmp_path / "output"

    run_cycles(settings, cycles=1, output_root=output_root, interval_seconds=0)
    assert json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))["fresh_start"] is True

    run_cycles(settings, cycles=1, output_root=output_root, interval_seconds=0, resume=True)
    assert json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))["fresh_start"] is False


def test_run_cycles_fresh_start_removes_previous_history_and_cycle_dirs(tmp_path: Path):
    settings = _fast_settings()
    output_root = tmp_path / "output"

    run_cycles(settings, cycles=2, output_root=output_root, interval_seconds=0)
    stale_marker = output_root / "cycle_009"
    stale_marker.mkdir()
    (stale_marker / "leftover.json").write_text("{}", encoding="utf-8")
    history_file = output_root / "history" / "cluster_snapshots.jsonl"
    lines_after_first = len(history_file.read_text(encoding="utf-8").splitlines())

    run_cycles(settings, cycles=2, output_root=output_root, interval_seconds=0)

    assert not stale_marker.exists()
    # History was rebuilt from scratch, not appended to.
    assert len(history_file.read_text(encoding="utf-8").splitlines()) == lines_after_first


def test_run_cycles_resume_keeps_history_and_cycle_dirs(tmp_path: Path):
    settings = _fast_settings()
    output_root = tmp_path / "output"

    run_cycles(settings, cycles=2, output_root=output_root, interval_seconds=0)
    history_file = output_root / "history" / "cluster_snapshots.jsonl"
    lines_after_first = len(history_file.read_text(encoding="utf-8").splitlines())
    marker = output_root / "cycle_001" / "marker.txt"
    marker.write_text("kept", encoding="utf-8")

    run_cycles(settings, cycles=2, output_root=output_root, interval_seconds=0, resume=True)

    assert marker.exists()
    assert len(history_file.read_text(encoding="utf-8").splitlines()) > lines_after_first


def test_run_one_cycle_cycle_one_starts_fresh_unless_resuming(tmp_path: Path):
    settings = _fast_settings()
    output_root = tmp_path / "output"

    run_one_cycle(settings, cycle_index=1, output_root=output_root)
    marker = output_root / "cycle_001" / "marker.txt"
    marker.write_text("stale", encoding="utf-8")

    run_one_cycle(settings, cycle_index=1, output_root=output_root, resume=True)
    assert marker.exists()

    run_one_cycle(settings, cycle_index=1, output_root=output_root)
    assert not marker.exists()


def test_clear_previous_run_reports_what_it_removed(tmp_path: Path):
    output_root = tmp_path / "output"
    (output_root / "history").mkdir(parents=True)
    (output_root / "cycle_001").mkdir()
    (output_root / "cycle_002").mkdir()
    keep = output_root / "run_summary.json"
    keep.write_text("{}", encoding="utf-8")

    removed = clear_previous_run(output_root)

    assert removed == ["history", "cycle_001", "cycle_002"]
    assert keep.exists()
    assert clear_previous_run(output_root) == []


def test_repeat_run_still_matches_the_committed_expected_output(tmp_path: Path):
    """A second run into a populated root must still pass verify-sample.

    This is the exact regression this guards against: the repeat
    run used to produce 11 posts in cycle 1 instead of 17 (collector
    watermarks), 4 clusters instead of 8 (future trend snapshots), and a
    shifted RL seed stream.
    """
    from tiblender_runtime.verify import verify_sample

    expected_path = Path("sample_runs/offline_three_cycles/expected/summary.json")
    settings = Settings(mode="offline", seed=20260909)
    output_root = tmp_path / "output"

    run_cycles(settings, cycles=3, output_root=output_root, interval_seconds=0)
    run_cycles(settings, cycles=3, output_root=output_root, interval_seconds=0)

    assert verify_sample(output_root, expected_path) == 0


def test_cycle_record_stores_a_path_relative_to_the_output_root(tmp_path: Path):
    settings = _fast_settings()
    output_root = tmp_path / "output"

    run_cycles(settings, cycles=1, output_root=output_root, interval_seconds=0)

    record = json.loads((output_root / "cycle_001" / "cycle_record.json").read_text(encoding="utf-8"))
    assert record["cycle_dir"] == "cycle_001"
    run_summary = json.loads((output_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["cycles"][0]["cycle_dir"] == "cycle_001"
    assert str(tmp_path) not in (output_root / "cycle_001" / "cycle_record.json").read_text(encoding="utf-8")
    summary = json.loads((output_root / "cycle_001" / "cycle_summary.json").read_text(encoding="utf-8"))
    assert summary["output_dir"] == "cycle_001"
    assert summary["feedback"]["trajectory_path"] == "cycle_001/intermediate/investigation_trajectories.json"


# ---------------------------------------------------------------------------
# Live-mode preflight: README/PIPELINE_OVERVIEW promise a credential-less live
# run "fails fast before writing outputs". It used to complete Step 1
# collection first, leaving posts.jsonl, collection_report.json,
# history/collector_state.json and run_summary.json behind before exiting 2.
# ---------------------------------------------------------------------------


def test_live_run_cycles_fails_before_creating_the_output_root(tmp_path):
    settings = _fast_settings(mode="live")
    output_root = tmp_path / "liveout"

    with pytest.raises(scheduler_module.LiveServicesUnavailableError) as excinfo:
        run_cycles(settings, cycles=1, output_root=output_root, interval_seconds=0)

    assert not output_root.exists(), (
        "a live run with no configured services must write nothing at all: "
        f"{sorted(p.name for p in output_root.iterdir())}"
    )
    message = str(excinfo.value)
    assert "llm" in message and "TIBLENDER_LLM_PROVIDER" in message


def test_live_single_cycle_fails_before_creating_the_cycle_directory(tmp_path):
    settings = _fast_settings(mode="live")
    output_root = tmp_path / "liveout"

    with pytest.raises(scheduler_module.LiveServicesUnavailableError):
        run_one_cycle(settings, cycle_index=1, output_root=output_root)

    assert not cycle_dir(output_root, 1).exists()


def test_live_preflight_passes_once_the_llm_and_one_collector_are_configured(tmp_path):
    settings = _fast_settings(
        mode="live",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        openai_api_key="k",
        x_bearer_token="t",
    )

    assert scheduler_module.live_preflight_error(settings) is None


def test_offline_and_auto_runs_are_never_blocked_by_the_live_preflight():
    assert scheduler_module.live_preflight_error(_fast_settings(mode="offline")) is None
    assert scheduler_module.live_preflight_error(_fast_settings(mode="auto")) is None
