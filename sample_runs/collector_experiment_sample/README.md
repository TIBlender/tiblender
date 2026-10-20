# Collector Experiment Sample Run

This directory contains the bounded input used by the executable sample run and
the retained output from running the TIBlender runtime over a representative
npm supply-chain cluster.

- Period: `2026-03-16T00:00:00Z` to `2026-04-17T00:00:00Z` (end exclusive)
- Platforms: `x`, `reddit`, `discord`
- Scope: deidentified social posts with source-post URLs and account names omitted
- Sample topic: Axios npm supply-chain compromise

## Files

- `sample_run_posts.jsonl`: six deidentified posts used for the executable sample run.
- `manifest.json`: sample-run input, period, platform counts, and generated-output metadata.
- `sample_run_output/`: stored output from `python3 -m tiblender_runtime run` in offline mode, with real tools replayed from recorded tool responses and deterministic heuristic report generation (no live LLM). It holds `cycle_summary.json`, `dynamic_queries.json`, and the `history/`, `intermediate/`, `markdown/`, `pdf/`, `reports/` and `stix/` subdirectories. No run log is retained: the run is reproducible from the input above, and `cycle_summary.json` records what each stage did.

## Sample Run

The sample run uses English-rendered posts from one Axios npm supply-chain
cluster. The input spans X, Reddit, and Discord so the runtime exercises
cross-platform clustering, multi-perspective investigation, offline
deterministic evidence evaluation, report generation, and cross-cycle feedback.

This directory keeps only the minimal executable input and its retained output.
The sample input keeps post text, timestamps, platform labels, and visible
indicators needed to reproduce the runtime path.

The run produced JSON, Markdown, PDF, and STIX 2.1 reports, intermediate
normalized posts, investigation trajectories, dynamic queries, and (under
`sample_run_output/history/`) policy state and cluster-snapshot history.
`intermediate/tool_recording_misses.json` records which external-tool requests
found a recorded response; all 8 of this run's do
(`cycle_summary.json`'s `tool_recording_replay.hit_rate` is `1.0`).

## Regenerating it, and why re-running it is not byte-identical

```bash
TIBLENDER_MODE=offline PYTHONHASHSEED=0 python3 -m tiblender_runtime run \
  --input sample_runs/collector_experiment_sample/sample_run_posts.jsonl \
  --output sample_runs/collector_experiment_sample/sample_run_output
```

Unlike the three-cycle demo, this sample is **not** byte-pinned by
`verify-sample`, and it cannot be: `run` (unlike `run-cycles`) has no
collection window, so `TIBLENDER_CLOCK=auto` has nothing to anchor a fixed
clock on and falls back to wall time. A re-run therefore differs from the
committed copy **only** in wall-clock stamps — `generated_at`, evidence
`observed_at`/`timestamp`, `expires_at`, the STIX `modified`/`published`/
`valid_from` fields, and the PDF's "Generated At" line. Everything that
identifies the run is stable and unchanged across regenerations: the report id
`TIBLENDER-RUN-252a5d11fa1a`, the cluster, the IoC set, the severity and
strategic-value verdicts, and every count in `manifest.json`. Read a timestamp
diff here as the absence of a fixed clock, not as non-determinism.
