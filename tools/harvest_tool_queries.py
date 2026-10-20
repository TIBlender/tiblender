#!/usr/bin/env python3
"""Harvest the external-tool request keys the offline demo actually issues.

Usage:
    python tools/harvest_tool_queries.py            # run the demo, merge the misses
    python tools/harvest_tool_queries.py --dry-run  # report only, write nothing

`tools/build_fixtures.py` can only derive tool queries from the *text* of the
fixture posts: every domain, IP, CVE and URL it can see. That covers the first
round of an investigation and nothing after it. The pipeline then issues
requests the builder has no way to compute -- free-text cluster search phrases
(`"supply chain"`, `"edition.cnn.com microsoft"`), and the domains, IPs and
URLs pivoted out of *recorded* evidence rather than out of post text. Those
requests found no recording, degraded to `status=unavailable`, and left the
bundle shipping recordings that nothing reads while the docs claimed the
recordings stood in for the live tools.

The offline three-cycle run is byte-deterministic, so the set of requests it
issues is a fixed, reproducible input. This script:

1. runs `run-cycles --cycles 3 --interval 0` offline into a temporary output
   root (never `output/`),
2. reads every cycle's `intermediate/tool_recording_misses.json` (written by
   `workflow._evaluate_clusters_node` from `ToolRecordingReplayLog`),
3. merges the missed keys into `fixtures/tools/harvested_queries.json`, and
4. regenerates `fixtures/tools/queries.json` (and `fixtures/README.md`) by
   re-running the builder, which unions the post-derived queries with the
   harvested ones.

Recording the merged file is a separate, credentialed step:

    python tools/record_tool_responses.py --fixtures fixtures \
        --queries fixtures/tools/queries.json --resume

Repeat harvest -> record until a fresh run reports zero misses. Two rounds are
normal: recording the first round's misses gives the pipeline more evidence,
which produces further pivots.

The harvested file is committed, so `make build-fixtures` stays reproducible
on a machine with no network: the builder reads it, it never re-derives it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiblender_runtime.tool_fixtures import key_parameters  # noqa: E402

HARVESTED_FILENAME = "harvested_queries.json"

# Provenance carried on every entry of fixtures/tools/queries.json. It is not
# part of the recording key (tool_fixtures.recording_key hashes tool_type and
# the named parameters only), so adding it moves no recording and forces no
# re-recording -- it just lets a reader of queries.json alone tell which
# requests the builder derived from fixture post text and which the offline
# demo was observed to issue.
ORIGIN_POST_DERIVED = "post_derived"
ORIGIN_HARVESTED = "harvested"
ORIGINS = (ORIGIN_POST_DERIVED, ORIGIN_HARVESTED)

# Committed run trees that are also harvested, in addition to the demo the
# script runs itself. The legacy single-cycle sample under `sample_runs/` is
# regenerated and shipped like the demo is, so its external-tool requests
# deserve recordings for the same reason. Paths are relative to the package
# root and skipped when absent.
EXTRA_RUN_ROOTS: tuple[Path, ...] = (
    Path("sample_runs") / "collector_experiment_sample" / "sample_run_output",
)

_HARVESTED_NOTE = (
    "External-tool request keys harvested from the deterministic offline three-cycle "
    "demo by tools/harvest_tool_queries.py. These are the requests the pipeline issues "
    "that tools/build_fixtures.py cannot derive from fixture post text: free-text "
    "cluster search phrases, and indicators pivoted out of recorded evidence. "
    "tools/build_fixtures.py merges them into fixtures/tools/queries.json. Committed so "
    "a fixture rebuild needs no network."
)

# Social tools are answered from the Step 1 collected corpus, never from a
# provider, so a missing recording for one is not a coverage gap and must not
# enter queries.json.
_NEVER_HARVEST = frozenset({"X_SEARCH", "REDDIT_SEARCH", "TELEGRAM_SEARCH", "DISCORD_SEARCH"})


def load_harvested(path: Path) -> list[dict[str, Any]]:
    """Read a harvested-queries file, tolerating its absence."""
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("queries", [])
    if not isinstance(payload, list):
        raise ValueError(f"{path} must hold a JSON list of queries or an object with a 'queries' list")
    return payload


def _entry_key(entry: dict[str, Any]) -> str:
    return json.dumps(
        {"tool_type": entry["tool_type"], "parameters": key_parameters(entry.get("parameters", {}))},
        sort_keys=True,
        ensure_ascii=False,
    )


def merge_queries(
    existing: list[dict[str, Any]],
    harvested: list[dict[str, Any]],
    *,
    default_origin: str = ORIGIN_HARVESTED,
) -> list[dict[str, Any]]:
    """Union of two query lists, deduplicated by request key and sorted.

    Each entry keeps its own ``origin`` when it carries one, else
    ``default_origin``. A request both the post text produces and the demo
    issues is ``post_derived``: the builder can derive it with no run at all.

    Pure so the merge is unit-testable without running a cycle.
    """
    merged: dict[str, dict[str, Any]] = {}
    for entry in list(existing) + list(harvested):
        tool_type = entry["tool_type"]
        if tool_type in _NEVER_HARVEST:
            continue
        parameters = key_parameters(entry.get("parameters", {}))
        if not parameters:
            continue
        origin = entry.get("origin") if entry.get("origin") in ORIGINS else default_origin
        key = _entry_key(entry)
        previous = merged.get(key)
        if previous is not None and previous["origin"] == ORIGIN_POST_DERIVED:
            origin = ORIGIN_POST_DERIVED
        merged[key] = {"tool_type": tool_type, "parameters": parameters, "origin": origin}
    return [merged[key] for key in sorted(merged)]


def _misses_files(output_root: Path) -> list[Path]:
    """The misses files under a run tree.

    `run-cycles` writes one per `cycle_*` directory; a single `run` writes one
    directly under its own output directory.
    """
    candidates = [
        cycle_dir / "intermediate" / "tool_recording_misses.json"
        for cycle_dir in sorted(path for path in output_root.glob("cycle_*") if path.is_dir())
    ]
    candidates.append(output_root / "intermediate" / "tool_recording_misses.json")
    return [path for path in candidates if path.exists()]


def collect_misses(output_root: Path) -> list[dict[str, Any]]:
    """Every missed request key across a run tree, deduplicated and sorted."""
    misses: list[dict[str, Any]] = []
    for path in _misses_files(output_root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for miss in payload.get("misses", []):
            misses.append(
                {
                    "tool_type": miss["tool_type"],
                    "parameters": miss["parameters"],
                    "origin": ORIGIN_HARVESTED,
                }
            )
    return merge_queries([], misses)


def collect_replay_rates(output_root: Path) -> dict[str, dict[str, Any]]:
    """Per-tool replay totals across every cycle of a run."""
    totals: dict[str, dict[str, Any]] = {}
    for cycle_dir in sorted(path for path in output_root.glob("cycle_*") if path.is_dir()):
        path = cycle_dir / "cycle_summary.json"
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8")).get("tool_recording_replay", {})
        for tool_type, row in (replay.get("by_tool") or {}).items():
            bucket = totals.setdefault(tool_type, {"calls": 0, "hits": 0, "misses": 0})
            bucket["calls"] += int(row["calls"])
            bucket["hits"] += int(row["hits"])
            bucket["misses"] += int(row["misses"])
    for bucket in totals.values():
        bucket["hit_rate"] = round(bucket["hits"] / bucket["calls"], 4) if bucket["calls"] else 0.0
    return {tool_type: totals[tool_type] for tool_type in sorted(totals)}


def format_rate_table(totals: dict[str, dict[str, Any]]) -> str:
    """The per-tool replay table the README cites, as plain text."""
    lines = [f"{'tool_type':<22}  {'calls':>5}  {'hits':>5}  {'misses':>6}  hit rate"]
    calls = hits = 0
    for tool_type, row in totals.items():
        calls += row["calls"]
        hits += row["hits"]
        lines.append(
            f"{tool_type:<22}  {row['calls']:>5}  {row['hits']:>5}  {row['misses']:>6}  {row['hit_rate'] * 100:6.1f} %"
        )
    overall = (hits / calls * 100) if calls else 0.0
    lines.append(f"{'ALL':<22}  {calls:>5}  {hits:>5}  {calls - hits:>6}  {overall:6.1f} %")
    return "\n".join(lines)


def run_offline_cycles(output_root: Path, *, cycles: int) -> None:
    """Run the deterministic offline demo into ``output_root``."""
    environment = dict(os.environ)
    environment.update(
        {
            "TIBLENDER_MODE": "offline",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TIBLENDER_CLOCK": "auto",
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
        }
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tiblender_runtime",
            "run-cycles",
            "--output-root",
            str(output_root),
            "--cycles",
            str(cycles),
            "--interval",
            "0",
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        env=environment,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def rebuild_fixtures(dataset: Path, *, cycles: int, seed: int) -> None:
    """Re-run the builder so queries.json/README.md pick up the merged file."""
    root = Path(__file__).resolve().parent.parent
    environment = dict(os.environ)
    environment.update({"TIBLENDER_MODE": "offline", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"})
    subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "build_fixtures.py"),
            "--dataset",
            str(dataset),
            "--out",
            str(root / "fixtures" / "collectors"),
            "--cycles",
            str(cycles),
            "--seed",
            str(seed),
        ],
        cwd=str(root),
        env=environment,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def write_harvested(path: Path, queries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"note": _HARVESTED_NOTE, "queries": queries}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures"), help="Fixtures root directory")
    parser.add_argument("--cycles", type=int, default=3, help="Number of cycles to run (default 3)")
    parser.add_argument("--seed", type=int, default=20260909, help="Fixture builder seed")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/raw_dataset/reports.jsonl"),
        help="Released dataset the builder rebuilds fixtures from",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report the misses; write nothing")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where to run the demo (default: a temporary directory, removed afterwards)",
    )
    args = parser.parse_args(argv)

    harvested_path = args.fixtures / "tools" / HARVESTED_FILENAME

    with tempfile.TemporaryDirectory(prefix="tiblender-harvest-") as scratch:
        output_root = args.output_root if args.output_root is not None else Path(scratch) / "run"
        run_offline_cycles(output_root, cycles=args.cycles)
        misses = collect_misses(output_root)
        totals = collect_replay_rates(output_root)

    package_root = Path(__file__).resolve().parent.parent
    extra: list[dict[str, Any]] = []
    for relative in EXTRA_RUN_ROOTS:
        extra.extend(collect_misses(package_root / relative))
    if extra:
        print(f"(+{len(extra)} missed key(s) from committed sample runs)")
        misses = merge_queries(misses, extra)

    print(format_rate_table(totals))
    print()
    if not misses:
        print("no missing recordings: every external-tool request replayed")
        return 0

    by_tool: dict[str, int] = {}
    for miss in misses:
        by_tool[miss["tool_type"]] = by_tool.get(miss["tool_type"], 0) + 1
    print(f"{len(misses)} distinct request key(s) with no recording:")
    for tool_type in sorted(by_tool):
        print(f"  {tool_type:<22}  {by_tool[tool_type]}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    existing = load_harvested(harvested_path)
    merged = merge_queries(existing, misses)
    write_harvested(harvested_path, merged)
    print(f"\n{harvested_path}: {len(existing)} -> {len(merged)} harvested quer{'y' if len(merged) == 1 else 'ies'}")

    rebuild_fixtures(args.dataset, cycles=args.cycles, seed=args.seed)
    print(f"rebuilt {args.fixtures / 'tools' / 'queries.json'}")
    print(
        "\nnext: python tools/record_tool_responses.py --fixtures fixtures "
        "--queries fixtures/tools/queries.json --resume"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
