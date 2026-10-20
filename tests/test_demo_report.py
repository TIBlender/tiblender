"""The one-document demo report behind `demo/run_demo.sh`.

The README promises one structured threat intelligence document at
output/report.json; this is that document, and these tests pin the two
properties the promise rests on: it carries
every report the run produced, and it is byte-identical across runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiblender_runtime.demo_report import build_demo_report, write_demo_report


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def run_tree(tmp_path: Path) -> Path:
    root = tmp_path / "output"
    _write(
        root / "run_summary.json",
        {
            "cycles": [
                {
                    "cycle_dir": "cycle_001",
                    "cycle_index": 1,
                    "posts": 20,
                    "reports": 2,
                    "window": {"start": "2026-03-31T00:00:00Z", "end": "2026-04-03T23:59:59Z"},
                },
                {
                    "cycle_dir": "cycle_002",
                    "cycle_index": 2,
                    "posts": 33,
                    "reports": 1,
                    "window": {"start": "2026-04-04T00:00:00Z", "end": "2026-04-07T23:59:59Z"},
                },
            ]
        },
    )
    for cycle, ids in (("cycle_001", ["TIBLENDER-RUN-bbb", "TIBLENDER-RUN-aaa"]), ("cycle_002", ["TIBLENDER-RUN-ccc"])):
        for report_id in ids:
            _write(root / cycle / "reports" / f"{report_id}.json", {"report_id": report_id})
        _write(root / cycle / "cycle_summary.json", {"posts_loaded": 20, "reports_generated": len(ids)})
    return root


def test_the_demo_report_carries_every_report_of_every_cycle_in_order(run_tree: Path):
    document = build_demo_report(run_tree)

    assert [report["report_id"] for report in document["reports"]] == [
        "TIBLENDER-RUN-aaa",
        "TIBLENDER-RUN-bbb",
        "TIBLENDER-RUN-ccc",
    ]
    assert [cycle["cycle_dir"] for cycle in document["cycles"]] == ["cycle_001", "cycle_002"]
    assert document["summary"]["cycles"]["cycle_001"]["reports"] == 2


def test_generated_at_comes_from_the_run_not_the_wall_clock(run_tree: Path):
    """The last cycle's window end, so two runs write the same bytes."""
    first = build_demo_report(run_tree)
    second = build_demo_report(run_tree)

    assert first["generated_at"] == "2026-04-07T23:59:59Z"
    assert json.dumps(first) == json.dumps(second)


def test_writing_it_twice_produces_byte_identical_files(run_tree: Path, tmp_path: Path):
    target = tmp_path / "report.json"

    write_demo_report(run_tree, target)
    first = target.read_bytes()
    write_demo_report(run_tree, target)

    assert target.read_bytes() == first
    assert json.loads(first)["reports"]


def test_a_run_tree_without_a_summary_is_a_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_demo_report(tmp_path / "missing")
