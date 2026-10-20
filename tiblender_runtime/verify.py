"""Deterministic summarization and comparison of multi-cycle scheduler output.

``summarize_output`` reduces a scheduler output tree to the numbers that must
be reproducible run-to-run (post/cluster/report counts, report ids, IoC
values, dynamic queries, trend counts, training status, and per-platform
service modes) while dropping anything that legitimately varies between runs
(wall-clock timestamps). ``compare_summaries`` diffs two such summaries, and
``verify_sample``/``write_expected`` wire that into the committed
``sample_runs/offline_three_cycles/expected/`` fixture.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

_IGNORED_KEYS = {"generated_at", "recorded_at", "timestamp"}


class VerificationError(Exception):
    """A required input file for verification is missing or malformed.

    Raised (never a raw ``FileNotFoundError``/``JSONDecodeError``) so the CLI
    can tell a genuine verification mismatch (exit 1) apart from a
    configuration/input problem (exit 2) -- a missing expected summary or a
    corrupted output file is not evidence that the run itself disagrees with
    the expected numbers.
    """


def _read_json(path: Path) -> Any:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise VerificationError(f"missing or unreadable file: {path} ({error})") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise VerificationError(f"malformed JSON in {path}: {error}") from error


def _require_output_root(output_root: Path) -> Path:
    output_root = Path(output_root)
    if not output_root.is_dir():
        raise VerificationError(f"output root does not exist or is not a directory: {output_root}")
    return output_root


def _report_iocs(report: dict) -> list[str]:
    values: set[str] = set()
    for entries in (report.get("iocs") or {}).values():
        if isinstance(entries, list):
            values.update(str(value) for value in entries)
    return sorted(values)


def _summarize_reports(cycle_path: Path) -> tuple[list[str], dict[str, list[str]]]:
    reports_dir = cycle_path / "reports"
    if not reports_dir.is_dir():
        return [], {}
    report_ids: list[str] = []
    iocs_by_report: dict[str, list[str]] = {}
    for report_path in sorted(reports_dir.glob("*.json")):
        report = _read_json(report_path)
        report_id = str(report.get("report_id") or report_path.stem)
        report_ids.append(report_id)
        iocs_by_report[report_id] = _report_iocs(report)
    return sorted(report_ids), iocs_by_report


def _summarize_dynamic_queries(cycle_path: Path) -> list[str]:
    path = cycle_path / "dynamic_queries.json"
    if not path.exists():
        return []
    payload = _read_json(path)
    if not isinstance(payload, list):
        return []
    return sorted(str(item.get("query", "")) for item in payload if isinstance(item, dict))


def _summarize_services(cycle_path: Path) -> dict[str, str]:
    path = cycle_path / "collection_report.json"
    if not path.exists():
        return {}
    payload = _read_json(path)
    per_platform = payload.get("per_platform", {}) if isinstance(payload, dict) else {}
    return {str(platform): str(info.get("mode", "")) for platform, info in per_platform.items()}


def _summarize_cycle(cycle_path: Path) -> dict:
    cycle_summary_path = cycle_path / "cycle_summary.json"
    cycle_summary: dict = _read_json(cycle_summary_path) if cycle_summary_path.exists() else {}
    feedback = cycle_summary.get("feedback", {}) if isinstance(cycle_summary, dict) else {}

    report_ids, iocs_by_report = _summarize_reports(cycle_path)

    return {
        "posts_loaded": cycle_summary.get("posts_loaded", 0),
        "threat_related": cycle_summary.get("threat_related_posts", 0),
        "clusters": cycle_summary.get("clusters_generated", 0),
        "reports": cycle_summary.get("reports_generated", 0),
        "report_ids": report_ids,
        "iocs_by_report": iocs_by_report,
        "dynamic_queries": _summarize_dynamic_queries(cycle_path),
        "trend_counts": dict(sorted((feedback.get("trend_counts") or {}).items())),
        "training_status": feedback.get("training_status", ""),
        # Direct evidence of cross-cycle history sharing: this is the cumulative line count of the shared
        # cluster_snapshots.jsonl at the time this cycle ran, so it must grow
        # cycle over cycle rather than resetting to each cycle's own cluster
        # count -- tracking it here means a future regression shows up as a
        # verify-sample mismatch, not just a scheduler unit-test failure.
        "history_size": feedback.get("history_size", 0),
        # Recorded-response replay coverage for this cycle. Comparing it here
        # means a regression that silently stops replaying a tool (a re-keyed
        # parameter name, a deleted recording) shows up as a verify-sample
        # diff instead of as quietly thinner evidence.
        "tool_recording_replay": cycle_summary.get("tool_recording_replay", {}),
        "services": _summarize_services(cycle_path),
    }


def summarize_output(output_root: Path) -> dict:
    """Reduce one scheduler output tree to its deterministic, comparable numbers.

    Raises :class:`VerificationError` if ``output_root`` does not exist, or
    if any JSON file inside it is malformed.
    """
    output_root = _require_output_root(output_root)
    run_summary_path = output_root / "run_summary.json"
    mode = ""
    seed: int | None = None
    if run_summary_path.exists():
        run_summary = _read_json(run_summary_path)
        mode = run_summary.get("mode", "")
        seed = run_summary.get("seed")

    cycles: dict[str, dict] = {}
    for cycle_path in sorted(path for path in output_root.glob("cycle_*") if path.is_dir()):
        cycles[cycle_path.name] = _summarize_cycle(cycle_path)

    return {"mode": mode, "seed": seed, "cycles": cycles}


def _diff_value(actual: Any, expected: Any, path: str, diffs: list[str]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key in _IGNORED_KEYS:
                continue
            child_path = f"{path}.{key}" if path else str(key)
            if key not in actual:
                diffs.append(f"{child_path}: missing in actual (expected {expected[key]!r})")
            elif key not in expected:
                diffs.append(f"{child_path}: unexpected in actual (got {actual[key]!r})")
            else:
                _diff_value(actual[key], expected[key], child_path, diffs)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: length mismatch (expected {len(expected)}, got {len(actual)})")
            return
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _diff_value(actual_item, expected_item, f"{path}[{index}]", diffs)
        return
    if actual != expected:
        diffs.append(f"{path}: expected {expected!r}, got {actual!r}")


def compare_summaries(actual: dict, expected: dict) -> list[str]:
    """Path-qualified diffs between two summaries; ``[]`` when equal.

    Keys named ``generated_at``/``recorded_at``/``timestamp`` are ignored at
    any nesting level, since those legitimately vary between runs.
    """
    diffs: list[str] = []
    _diff_value(actual, expected, "", diffs)
    return diffs


def verify_sample(output_root: Path, expected_path: Path) -> int:
    """Compare ``output_root`` against a committed expected summary.

    Prints any diffs to stdout and returns ``0`` when they match, ``1``
    otherwise. Raises :class:`VerificationError` (never a raw
    ``FileNotFoundError``/``JSONDecodeError``) if ``expected_path`` is
    missing/malformed or ``output_root`` is missing/contains malformed JSON
    -- those are configuration/input problems, not a verification mismatch.
    """
    expected = _read_json(Path(expected_path))
    actual = summarize_output(Path(output_root))
    diffs = compare_summaries(actual, expected)
    if diffs:
        for diff in diffs:
            print(diff, file=sys.stdout)
        return 1
    print("verify_sample: ok (0 diffs)")
    return 0


def write_expected(output_root: Path, expected_dir: Path) -> None:
    """Regenerate the committed expected-output fixture (maintainer-only).

    Writes ``expected_dir/summary.json`` plus a copy of each cycle's
    ``cycle_summary.json`` and ``dynamic_queries.json``. Raises
    :class:`VerificationError` if ``output_root`` is missing or contains
    malformed JSON.
    """
    output_root = _require_output_root(output_root)
    expected_dir = Path(expected_dir)
    expected_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_output(output_root)
    (expected_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    for cycle_path in sorted(path for path in output_root.glob("cycle_*") if path.is_dir()):
        dest_dir = expected_dir / cycle_path.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("cycle_summary.json", "dynamic_queries.json"):
            source = cycle_path / filename
            if source.exists():
                shutil.copy2(source, dest_dir / filename)
