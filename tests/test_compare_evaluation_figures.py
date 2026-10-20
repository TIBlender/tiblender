"""The comparison behind `evaluation/run_evaluation.sh`'s `evaluation: ok`."""

from __future__ import annotations

import json
from pathlib import Path

from tools.compare_evaluation_figures import compare, differences, main


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def test_identical_bytes_report_ok(tmp_path: Path, capsys):
    payload = {"reports": 8288, "feeds": {"PhishTank": 12.5}}
    expected = _write(tmp_path / "expected.json", payload)
    actual = _write(tmp_path / "actual.json", payload)

    assert compare(expected, actual) == 0
    assert capsys.readouterr().out.strip() == "evaluation: ok"


def test_a_moved_number_is_named_with_its_path(tmp_path: Path, capsys):
    expected = _write(tmp_path / "expected.json", {"feeds": {"PhishTank": {"overlap": 12.5}}})
    actual = _write(tmp_path / "actual.json", {"feeds": {"PhishTank": {"overlap": 13.0}}})

    assert compare(expected, actual) == 1
    err = capsys.readouterr().err
    assert "feeds.PhishTank.overlap" in err
    assert "12.5" in err and "13.0" in err


def test_added_and_removed_keys_are_both_reported():
    found = differences({"a": 1, "b": 2}, {"a": 1, "c": 3})

    assert any("b: missing" in line for line in found)
    assert any("c: unexpected" in line for line in found)


def test_list_length_changes_are_reported_once():
    found = differences({"rows": [1, 2, 3]}, {"rows": [1, 2]})

    assert found == ["rows: length 3 expected, got 2"]


def test_same_values_but_different_bytes_still_fail(tmp_path: Path, capsys):
    """The dataset promises byte equality, not just equal values."""
    expected = tmp_path / "expected.json"
    actual = tmp_path / "actual.json"
    expected.write_text('{"a": 1, "b": 2}', encoding="utf-8")
    actual.write_text('{"b": 2, "a": 1}', encoding="utf-8")

    assert compare(expected, actual) == 1
    assert "not the same bytes" in capsys.readouterr().err


def test_a_missing_file_is_a_configuration_error_not_a_failed_claim(tmp_path: Path):
    expected = _write(tmp_path / "expected.json", {"a": 1})

    assert main(["--expected", str(expected), "--actual", str(tmp_path / "nope.json")]) == 2
