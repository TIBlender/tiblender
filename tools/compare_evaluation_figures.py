#!/usr/bin/env python3
"""Compare a recomputed `evaluation_figures.json` with the released one.

`evaluation/run_evaluation.sh` runs this after executing the dataset's
notebook. The dataset's own README states the contract this enforces:
executing `notebook/evaluation_figures.ipynb` must regenerate
`notebook/evaluation_figures.json` **byte-identically**.

A byte comparison alone would say only "different", so on a mismatch this
also walks the two documents and names the leaf paths that disagree -- that
is the difference between "the artifact failed" and "these three numbers
moved".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MAX_REPORTED_DIFFERENCES = 40


def _walk(expected: Any, actual: Any, path: str, out: list[str]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else str(key)
            if key not in expected:
                out.append(f"{child}: unexpected (got {actual[key]!r})")
            elif key not in actual:
                out.append(f"{child}: missing (expected {expected[key]!r})")
            else:
                _walk(expected[key], actual[key], child, out)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f"{path}: length {len(expected)} expected, got {len(actual)}")
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            _walk(left, right, f"{path}[{index}]", out)
        return
    if expected != actual:
        out.append(f"{path}: expected {expected!r}, got {actual!r}")


def differences(expected: Any, actual: Any) -> list[str]:
    """Leaf paths on which the two documents disagree, in a stable order."""
    out: list[str] = []
    _walk(expected, actual, "", out)
    return out


def compare(expected_path: Path, actual_path: Path) -> int:
    expected_bytes = expected_path.read_bytes()
    actual_bytes = actual_path.read_bytes()
    if expected_bytes == actual_bytes:
        print("evaluation: ok")
        return 0

    try:
        found = differences(json.loads(expected_bytes), json.loads(actual_bytes))
    except json.JSONDecodeError as error:
        print(f"evaluation: FAILED ({actual_path} is not valid JSON: {error})", file=sys.stderr)
        return 1

    if not found:
        # Same values, different bytes: key order, indentation or trailing
        # newline. Still a failure -- the dataset promises byte equality
        # -- but say which kind, so it is not mistaken for moved numbers.
        print(
            f"evaluation: FAILED ({actual_path} holds the same values as {expected_path} "
            "but not the same bytes: formatting or key order differs)",
            file=sys.stderr,
        )
        return 1

    print(f"evaluation: FAILED ({len(found)} value(s) differ from {expected_path})", file=sys.stderr)
    for line in found[:MAX_REPORTED_DIFFERENCES]:
        print(f"  {line}", file=sys.stderr)
    if len(found) > MAX_REPORTED_DIFFERENCES:
        print(f"  ... and {len(found) - MAX_REPORTED_DIFFERENCES} more", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True, help="the released evaluation_figures.json")
    parser.add_argument("--actual", type=Path, required=True, help="the file the notebook just wrote")
    args = parser.parse_args(argv)
    for path in (args.expected, args.actual):
        if not path.is_file():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2
    return compare(args.expected, args.actual)


if __name__ == "__main__":
    raise SystemExit(main())
