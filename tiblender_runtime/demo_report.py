"""One JSON document for the whole demo run: ``output/report.json``.

The README documents that the demo writes one structured threat
intelligence document at ``output/report.json`` without external
credentials. A cycle run writes one file per report under
``cycle_00N/reports/``; this module gathers them, the per-cycle counters, and
the ``verify-sample`` summary into that single document, so a reader can
open one path instead of walking the tree.

Nothing is recomputed here and nothing is summarized away: each report is the
report file verbatim. ``generated_at`` is taken from the run's own last
collection window rather than from the wall clock, so re-running the demo
rewrites the same bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import clock, verify

REPORT_FILENAME = "report.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cycle_records(run_summary: dict) -> list[dict]:
    cycles = run_summary.get("cycles")
    if not isinstance(cycles, list):
        return []
    return [cycle for cycle in cycles if isinstance(cycle, dict)]


def _generated_at(cycles: list[dict]) -> str:
    """The end of the last collection window, or the clock if there is none.

    Using the run's own window keeps the document byte-identical across runs
    of the offline demo, where the runtime pins its clock to that same window
    end. A live run (no fixed clock) falls back to the clock.
    """
    for cycle in reversed(cycles):
        window = cycle.get("window")
        if isinstance(window, dict) and window.get("end"):
            return str(window["end"])
    return clock.now_iso()


def _cycle_dir(output_root: Path, cycle: dict) -> Path:
    name = cycle.get("cycle_dir")
    if not name:
        index = int(cycle.get("cycle_index") or 0)
        name = f"cycle_{index:03d}"
    return output_root / str(name)


def _reports_of(cycle_dir: Path) -> list[dict]:
    reports_dir = cycle_dir / "reports"
    if not reports_dir.is_dir():
        return []
    return [_read_json(path) for path in sorted(reports_dir.glob("*.json"))]


def build_demo_report(output_root: Path) -> dict:
    """Gather one demo run into a single JSON-serializable document.

    Raises :class:`FileNotFoundError` when ``output_root`` holds no
    ``run_summary.json`` -- i.e. when the demo has not run.
    """
    output_root = Path(output_root)
    run_summary_path = output_root / "run_summary.json"
    if not run_summary_path.is_file():
        raise FileNotFoundError(
            f"{run_summary_path} does not exist: run `tiblender demo` before building the demo report"
        )
    run_summary = _read_json(run_summary_path)
    cycles = _cycle_records(run_summary)

    reports: list[dict] = []
    for cycle in cycles:
        reports.extend(_reports_of(_cycle_dir(output_root, cycle)))

    return {
        "generated_at": _generated_at(cycles),
        "cycles": cycles,
        "reports": reports,
        "summary": verify.summarize_output(output_root),
    }


def write_demo_report(output_root: Path, path: Path) -> Path:
    """Write :func:`build_demo_report`'s document to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = build_demo_report(output_root)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = ["REPORT_FILENAME", "build_demo_report", "write_demo_report"]
