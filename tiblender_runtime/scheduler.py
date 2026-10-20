"""Multi-cycle scheduler: collect -> run, carrying state across cycles.

Each cycle collects posts (consuming the previous cycle's dynamic queries),
runs the single-cycle pipeline against those posts, and writes a
``cycle_record.json`` summarizing the cycle. State that must survive across
cycles -- dynamic queries, RL policy state, trend history, and collector
state -- all lives under one shared ``<output_root>/history`` directory so
cycle N+1 can build on cycle N.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import clock
from .collectors.base import CollectionWindow
from .collectors.fixtures import fixture_path, list_fixture_cycles, load_fixture
from .collectors.runner import collect_cycle
from .paths import relative_path
from .services import (
    # Re-exported: the preflight is part of the scheduler's contract (both
    # entry points below run it), so callers catch/inspect it from here.
    LiveServicesUnavailableError,  # noqa: F401
    live_preflight_error,  # noqa: F401
    require_live_services,
    resolve_services,
)
from .settings import Settings
from .workflow import run_graph_cycle

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CycleRecord:
    cycle_index: int
    window: CollectionWindow
    cycle_dir: Path
    posts: int
    reports: int
    dynamic_queries: int
    trend_counts: dict[str, int]
    training_status: str
    # Share of this cycle's external-tool calls answered from a recorded
    # response (0.0-1.0); 0.0 when the cycle ran no recording-backed tool.
    tool_recording_hit_rate: float = 0.0


def cycle_dir(output_root: Path, index: int) -> Path:
    return Path(output_root) / f"cycle_{index:03d}"


def resolve_window(settings: Settings, *, cycle_index: int, window_end: datetime | None) -> CollectionWindow:
    """Resolve the collection window for one cycle.

    When the x collector is running from a recorded fixture and a fixture
    file exists for ``cycle_index``, that fixture's own window is used (so
    the offline sample stays anchored to the timestamps baked into the
    fixtures). Otherwise the window ends at ``window_end`` (or ``clock.now()``
    when not given) and spans ``settings.collect_window_hours``.
    """
    services = resolve_services(settings)
    x_status = services["x"]
    if x_status.mode == "fixture":
        candidate = fixture_path(settings.fixtures_dir, "x", cycle_index)
        if candidate.exists():
            return load_fixture(candidate).window
    end = window_end or clock.now()
    return CollectionWindow.ending_at(end, settings.collect_window_hours, cycle_index)


def _uses_fixed_clock(settings: Settings) -> bool:
    return settings.uses_fixed_clock()


def _window_to_dict(window: CollectionWindow) -> dict:
    return {
        "start": window.start.isoformat().replace("+00:00", "Z"),
        "end": window.end.isoformat().replace("+00:00", "Z"),
        "cycle_index": window.cycle_index,
        "label": window.label,
    }


def clear_previous_run(output_root: Path) -> list[str]:
    """Delete everything a previous run left under ``output_root``.

    Removes the shared ``history/`` directory (collector watermarks, trend
    snapshots, RL policy state) and every ``cycle_*`` directory, returning the
    names removed. Without this a second ``run-cycles`` into the same root is
    not a repeat of the first run at all: the collector state file suppresses
    fixture posts already seen, the trend history already holds the previous
    run's later-cycle snapshots, and the RL seed stream is offset by the
    previous run's training history -- so cycle 1 produces fewer posts, fewer
    clusters and a different policy than it did on a clean root.
    """
    output_root = Path(output_root)
    removed: list[str] = []
    history_dir = output_root / "history"
    if history_dir.exists():
        shutil.rmtree(history_dir, ignore_errors=True)
        removed.append(history_dir.name)
    for path in sorted(output_root.glob("cycle_*")):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
    if removed:
        logger.info("fresh start: removed %s under %s", ", ".join(removed), output_root)
    return removed


def _cycle_record_to_dict(record: CycleRecord, output_root: Path) -> dict:
    return {
        "cycle_index": record.cycle_index,
        "window": _window_to_dict(record.window),
        # Relative to output_root so committed/compared output never carries
        # an absolute host path.
        "cycle_dir": relative_path(record.cycle_dir, output_root),
        "posts": record.posts,
        "reports": record.reports,
        "dynamic_queries": record.dynamic_queries,
        "trend_counts": dict(sorted(record.trend_counts.items())),
        "training_status": record.training_status,
        "tool_recording_hit_rate": record.tool_recording_hit_rate,
    }


def _write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def run_one_cycle(
    settings: Settings,
    *,
    cycle_index: int,
    output_root: Path,
    window_end: datetime | None = None,
    resume: bool = False,
) -> CycleRecord:
    """Collect and run one cycle, sharing history/state under ``output_root``.

    Cycle 1 starts from a clean ``output_root`` (previous ``history/`` and
    ``cycle_*`` directories are removed) unless ``resume=True``, so re-running
    into an existing root reproduces the first run instead of continuing it.
    """
    # Before anything is created on disk: a live run without the services it
    # needs can only fail, and it must not leave a half-written cycle behind.
    require_live_services(settings)

    output_root = Path(output_root)
    if cycle_index == 1 and not resume:
        clear_previous_run(output_root)
    this_cycle_dir = cycle_dir(output_root, cycle_index)
    window = resolve_window(settings, cycle_index=cycle_index, window_end=window_end)

    # Everything cross-cycle (dynamic queries excepted -- those live per-cycle
    # so the next cycle can find the previous one's file) lives under one
    # shared history directory for the whole run, never settings' own default.
    base_settings = replace(settings, history_dir=output_root / "history")

    previous_dynamic_queries_path = (
        cycle_dir(output_root, cycle_index - 1) / "dynamic_queries.json" if cycle_index > 1 else None
    )

    was_fixed = clock.is_fixed()
    previous_fixed_value = clock.now() if was_fixed else None
    try:
        if _uses_fixed_clock(settings):
            clock.set_fixed_clock(window.end)

        outcome = collect_cycle(
            base_settings,
            window=window,
            dynamic_queries_path=previous_dynamic_queries_path,
            output_dir=this_cycle_dir,
        )

        cycle_settings = replace(base_settings, input_path=outcome.posts_path, output_dir=this_cycle_dir)
        run_graph_cycle(cycle_settings, cycle_index=cycle_index)
    finally:
        clock.set_fixed_clock(previous_fixed_value)

    cycle_summary_path = this_cycle_dir / "cycle_summary.json"
    cycle_summary: dict = {}
    if cycle_summary_path.exists():
        cycle_summary = json.loads(cycle_summary_path.read_text(encoding="utf-8"))
    feedback = cycle_summary.get("feedback", {})

    record = CycleRecord(
        cycle_index=cycle_index,
        window=window,
        cycle_dir=this_cycle_dir,
        posts=int(cycle_summary.get("posts_loaded", 0)),
        reports=int(cycle_summary.get("reports_generated", 0)),
        dynamic_queries=int(feedback.get("dynamic_query_count", 0)),
        trend_counts=dict(feedback.get("trend_counts", {})),
        training_status=str(feedback.get("training_status", "")),
        tool_recording_hit_rate=float((cycle_summary.get("tool_recording_replay") or {}).get("hit_rate", 0.0)),
    )
    _write_json(this_cycle_dir / "cycle_record.json", _cycle_record_to_dict(record, output_root))
    return record


def _write_run_summary(
    output_root: Path,
    settings: Settings,
    records: list[CycleRecord],
    *,
    warnings: list[str],
    fresh_start: bool,
    failed_cycle: dict | None = None,
) -> None:
    services = resolve_services(settings)
    payload = {
        "cycles": [_cycle_record_to_dict(record, output_root) for record in records],
        "mode": settings.mode,
        "services": {name: status.mode for name, status in services.items()},
        "seed": settings.seed,
        "fixed_clock": _uses_fixed_clock(settings),
        "fresh_start": fresh_start,
        "warnings": warnings,
    }
    if failed_cycle is not None:
        payload["failed_cycle"] = failed_cycle
    _write_json(Path(output_root) / "run_summary.json", payload)


def run_cycles(
    settings: Settings,
    *,
    cycles: int,
    output_root: Path,
    interval_seconds: int,
    start_index: int = 1,
    resume: bool = False,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[CycleRecord]:
    """Run ``cycles`` sequential cycles, sleeping ``interval_seconds`` between them.

    The run starts FRESH by default: any ``history/`` and ``cycle_*``
    directories a previous run left under ``output_root`` are removed first,
    so running twice into the same root produces the same output both times.
    Pass ``resume=True`` to continue from the existing state instead. Which
    of the two happened is recorded as ``run_summary.json["fresh_start"]``.

    In offline mode, if more cycles are requested than the bundled fixtures
    cover, the run stops at the last available fixture cycle and records a
    warning in ``run_summary.json`` instead of failing.
    """
    # Checked here as well as in run_one_cycle so the output root itself is
    # never created for a run that cannot start (a failure inside the loop
    # below would already have written run_summary.json).
    require_live_services(settings)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    fresh_start = not resume
    if fresh_start:
        clear_previous_run(output_root)

    indices = list(range(start_index, start_index + cycles))
    warnings: list[str] = []
    if settings.is_offline():
        fixture_cycles = list_fixture_cycles(settings.fixtures_dir)
        if fixture_cycles:
            last_fixture = max(fixture_cycles)
            allowed = [index for index in indices if index <= last_fixture]
            if len(allowed) < len(indices):
                warnings.append(
                    f"offline mode: requested cycles through {indices[-1]} but bundled fixtures only "
                    f"cover up to cycle {last_fixture}; stopping after cycle {last_fixture}"
                )
            indices = allowed
        else:
            warnings.append("offline mode: no collector fixture cycles are available; running zero cycles")
            indices = []

    records: list[CycleRecord] = []
    index = start_index
    try:
        for position, index in enumerate(indices):
            # The whole root was already cleared (or deliberately kept) above;
            # per-cycle clearing would wipe cycle 1's own output mid-run.
            record = run_one_cycle(settings, cycle_index=index, output_root=output_root, resume=True)
            records.append(record)
            if position < len(indices) - 1:
                sleeper(interval_seconds)
    except Exception as error:  # noqa: BLE001 - a failed cycle must be reported, not silently lost
        # Content-free at WARNING: an escaping LLMGenerationError carries the
        # model's draft of scraped posts in its message. The full text goes to
        # DEBUG and to run_summary.json's ``failed_cycle.error``, which is a
        # local diagnostic rather than a log line meant to be shared.
        logger.warning("cycle %d failed (%s); see run_summary.json", index, type(error).__name__)
        logger.debug("cycle %d failure detail: %s", index, error)
        _write_run_summary(
            output_root,
            settings,
            records,
            warnings=warnings,
            fresh_start=fresh_start,
            failed_cycle={"cycle_index": index, "error": f"{type(error).__name__}: {error}"},
        )
        raise

    _write_run_summary(output_root, settings, records, warnings=warnings, fresh_start=fresh_start)
    return records
