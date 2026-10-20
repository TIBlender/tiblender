# Committed Expected Output — Offline Three-Cycle Demo

This directory is the **ground truth** every offline demo run is checked
against. When `docker compose up --build` or `make offline` prints

```
verify_sample: ok (0 diffs)
```

it means the run just performed agrees, field by field, with the numbers
committed here. Nothing in this directory is produced at run time; it is a
fixture that ships with the repository.

## What `expected/` contains

- **`expected/summary.json`** — the whole comparison surface in one file:
  `mode`, `seed`, and one entry per cycle (`cycle_001`, `cycle_002`,
  `cycle_003`) holding
  - `posts_loaded`, `threat_related`, `clusters`, `reports`
  - `report_ids` — every generated report's id, sorted
  - `iocs_by_report` — the sorted IoC value set per report id
  - `dynamic_queries` — the trend-derived query strings that cycle wrote
  - `trend_counts` — per-axis firing counts (`EARLY_WARNING`, `VOLUME`,
    `PROPAGATION`, `ESCALATION`)
  - `training_status` — the RL trainer's own verdict for that cycle
  - `history_size` — the cumulative size of the shared
    `history/cluster_snapshots.jsonl` at that point, which is direct evidence
    that state really is carried across cycles
  - `services` — the resolved per-platform collector mode for that cycle
- **`expected/cycle_00N/cycle_summary.json`** and
  **`expected/cycle_00N/dynamic_queries.json`** — verbatim copies of the two
  per-cycle files the run writes, kept for inspection and diffing by hand.
  `verify-sample` itself reads only `summary.json`.

The `expected/` tree deliberately does **not** contain the reports, Markdown,
PDF, or STIX exports: those are large, and everything about them that must be
reproducible (their ids and their IoC sets) is already in `summary.json`.

## How `verify-sample` compares

```bash
tiblender verify-sample --output-root output
```

reduces the run's own `output/` tree to a summary of exactly the same shape
(`tiblender_runtime/verify.py`, `summarize_output`) and diffs it against
`expected/summary.json`:

- **Exit 0** and `verify_sample: ok (0 diffs)` on a match.
- **Exit 1** with every difference printed as a `path: expected X, got Y`
  line — a real disagreement between the run and the expectation.
- **Exit 2** on a missing or malformed input file — a configuration problem,
  not evidence that the run disagrees.

**Timestamps are excluded.** Keys named `generated_at`, `recorded_at` and
`timestamp` are skipped at every nesting level, because they legitimately
differ between runs. Everything else is compared exactly.

`--expected PATH` points the comparison at a different summary; the default is
this directory's `expected/summary.json`.

## How it was produced, and how to regenerate it

By running the demo and then snapshotting the result:

```bash
TIBLENDER_MODE=offline PYTHONHASHSEED=0 tiblender run-cycles \
  --output-root output --cycles 3 --interval 0
tiblender write-expected --output-root output \
  --expected-dir sample_runs/offline_three_cycles/expected
```

`write-expected` is **maintainer-only** and is not part of running the demo:
it overwrites the expectation with whatever the current code produces, which
would turn a genuine regression into a silent pass. Regenerate it only when a
deliberate change to the pipeline or to the bundled fixtures is meant to move
these numbers, and review the resulting diff as carefully as the code change
itself. Run `make clean-output` first so the tree being snapshotted holds
nothing from an earlier run.
