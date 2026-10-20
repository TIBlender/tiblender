# TIBlender — Reference

Reference material for TIBlender; start with
[the README's Quick Start](../README.md#quick-start) for how to run the demo.

- [CLI reference](#cli-reference)
- [Configuration paths](#configuration-paths)
- [Output layout](#output-layout)
- [Recorded-response replay coverage](#recorded-response-replay-coverage)
- [Modes and determinism](#modes-and-determinism)

## CLI reference

Every subcommand below is `tiblender <command>` once installed (`pip install
-e .` or via `install.sh`), or `python3 -m tiblender_runtime <command>` from
the package root without installing. In the container, the entry point is the
CLI itself, so `docker compose run --rm tiblender <command> [flags]` runs any
of them.

| Command | Flags | What it does |
|---|---|---|
| `demo` | `--output-root DIR` (default `output`) `--cycles N` (default 3) `--interval SECONDS` (default 0) `--viewer-dir DIR` (default `viewer`) `--expected PATH` `--report-json PATH` `--resume` | The bundled offline demo in one command: `run-cycles`, then `verify-sample`, then `build-viewer`, stopping at the first non-zero exit code. `--report-json` additionally collects every report of the run into one document. This is the container image's default command (`CMD`), which passes `--report-json output/report.json`, so `docker compose up --build` runs it and writes `output/report.json`. |
| `run` | `--input PATH` `--output DIR` `--llm-provider NAME` `--llm-model NAME` `--allow-heuristic-reports` | Runs one pipeline cycle (Steps 1–5 + feedback) over a JSONL input file. `--allow-heuristic-reports` permits deterministic smoke reports without an LLM configured; normal live runs require an LLM profile and fail with exit code 2 without one. |
| `collect` | `--output DIR` (required) `--cycle N` `--window-end ISO8601` `--dynamic-queries PATH` | Runs one collection cycle across all four platforms (live or fixture, per `tiblender doctor`), applies preprocessing/dedup, and writes `posts.jsonl` + `collection_report.json`. |
| `cycle` | `--output-root DIR` (required) `--cycle N` (required) `--window-end ISO8601` `--resume` | Runs one multi-cycle scheduler cycle: `collect` then `run`, sharing `history/` (trend snapshots, RL policy state, collector state) under `--output-root`. Cycle 1 starts fresh (see `run-cycles`); `--resume` continues from existing state. |
| `run-cycles` | `--output-root DIR` (required) `--cycles N` (required) `--interval SECONDS` `--start-index N` `--resume` | Runs a sequence of scheduler cycles, sleeping `--interval` seconds between them (default: `TIBLENDER_CYCLE_INTERVAL_SECONDS`, 3600; the offline demo passes `--interval 0` explicitly, from the Makefile `offline` recipe and the Dockerfile `CMD`). **Starts fresh by default**: any `history/` and `cycle_*/` directories a previous run left under `--output-root` are deleted first, so running twice into the same directory reproduces the first run instead of continuing it (collector watermarks, trend history, and the RL seed stream would all otherwise carry over). Pass `--resume` to continue from that state; either way `run_summary.json` records `"fresh_start"`. |
| `verify-sample` | `--output-root DIR` (required) `--expected PATH` (default: `sample_runs/offline_three_cycles/expected/summary.json`) | Compares a `run-cycles` output tree against a committed expected summary and prints `verify_sample: ok (0 diffs)` on a match. Exits 0 on a match, 1 with every diff printed, and 2 on a missing or malformed input file. The compared fields are listed in the [README](../README.md#expected-output). |
| `write-expected` | `--output-root DIR` (required) `--expected-dir DIR` (required) | Maintainer-only: regenerates the committed `sample_runs/offline_three_cycles/expected/` fixture from a fresh `run-cycles` output tree. |
| `build-viewer` | `--reports-root DIR` (required) `--viewer-dir DIR` (required) | Builds `<viewer-dir>/reports_viewer_data.js` from every `reports/*.json` file under `--reports-root` (a single `output/reports/` tree or a multi-cycle `output/cycle_00N/reports/` tree). Prints the data-file path, report count, and skipped-file count. |
| `doctor` | — | Prints the resolved mode (`live`/`fixture`/`heuristic`/`unavailable`) and reason for every collector/tool/LLM/embedding/OCR service, with credentials redacted. |
| `show-config` | — | Prints the fully resolved `.env`/environment configuration as JSON, with API keys and tokens replaced by `<set>`/`""`. |
| `audit-contract` | — | Checks the Step 3 tool contract (11 allowed tool types, 4 perspectives, per-perspective focus types) against the prompt/dispatch code; exits 0 if consistent, 1 if not. |
| `telegram-login` | `--session NAME` (required) `--api-id ID` `--api-hash HASH` | One-time interactive Telegram user-session login (see [`CREDENTIALS.md`](CREDENTIALS.md#telegram)). Exits 2 with a clear message if `telethon` is not installed or if no `api_id`/`api_hash` is available (flag or `TIBLENDER_TELEGRAM_API_ID`/`TIBLENDER_TELEGRAM_API_HASH`). |

## Configuration paths

`TIBLENDER_ENV_FILE` (default `.env`) selects which dotenv file
`Settings.from_env()` loads before reading environment variables directly.
The other path settings locate this package's own directories and files —
override them only for non-standard layouts (e.g. running from outside the
package root, or pointing multiple environments at shared fixtures):

| Variable | Default | Meaning |
|---|---|---|
| `TIBLENDER_INPUT_PATH` | `sample_runs/collector_experiment_sample/sample_run_posts.jsonl` | Input JSONL for a single `run` (overridden by `run --input`; `cycle`/`run-cycles` don't use it). |
| `TIBLENDER_OUTPUT_DIR` | `output` | Output directory for a single `run` (overridden by `run --output`; `cycle`/`run-cycles` use `--output-root` instead). |
| `TIBLENDER_FIXTURES_DIR` | `fixtures` | Root of the collector fixtures and tool recordings. |
| `TIBLENDER_CONFIG_DIR` | `config` | Root of the keyword catalog and seed lists (see [`config/README.md`](../config/README.md)). |
| `TIBLENDER_HISTORY_DIR` | unset (falls back to `<output_dir>/history`) | Shared cross-cycle state directory (RL policy state, trend snapshots, collector state); `cycle`/`run-cycles` set this explicitly to `<output-root>/history` so state survives across cycles. |

## Output layout

- `posts.jsonl` — Step 1 collected/preprocessed posts (one JSON object per line).
- `collection_report.json` — per-platform collection stats and errors from `collect`.
- `reports/*.json` — full structured Step 5 report JSON (metadata, IoCs, sources, evaluation rounds, report schema fields, Markdown rendering, STIX bundle).
- `markdown/*.md`, `pdf/*.pdf`, `stix/*.json` — report sidecars generated alongside each `reports/*.json`.
- `dynamic_queries.json` — trend-derived query candidates for the next cycle's `QueryPlan` (see [`config/README.md`](../config/README.md)).
- `policy_state.json` (under `history/` in multi-cycle runs) — the RL policy's learned weights, replay-buffer summary, and training history.
- `cluster_snapshots.jsonl` (under `history/`) — rolling cross-cycle cluster history used by trend detection.
- `intermediate/normalized_posts.jsonl`, `intermediate/clusters.json`, `intermediate/evaluation_decisions.json`, `intermediate/investigation_trajectories.json` — per-stage intermediate state for a single `run`.
- `cycle_summary.json` — per-cycle counts, feedback counts, resolved hyperparameters, and the LangGraph node sequence.
- `run_summary.json` (multi-cycle `run-cycles` only) — mode, seed, and every cycle's `cycle_summary.json` in one file.
- `report.json` (written only when `--report-json` is passed; `demo/run_demo.sh` and the image's default command always pass it) — every report of the run, its per-cycle counters and the `verify-sample` summary in one document.
- `evaluation/` (written by `evaluation/run_evaluation.sh`) — the notebook copy it executes, the executed notebook, and the `evaluation_figures.json` it recomputes.
- `viewer/reports_viewer_data.js` — generated by `build-viewer`; not part of this source tree until built.

## Recorded-response replay coverage

`cycle_00N_tool_recording_hit_rate`, printed by every cycle
([README](../README.md#expected-output)), is the
share of that cycle's external-tool calls answered from a recorded response
rather than degrading to `status=unavailable`. Measured over the committed
three-cycle run:

| tool | calls | replayed | hit rate |
| --- | ---: | ---: | ---: |
| GITHUB_SEARCH | 12 | 12 | 100.0 % |
| NVD_CVE_LOOKUP | 10 | 10 | 100.0 % |
| PASSIVE_DNS_LOOKUP | 46 | 45 | 97.8 % |
| SEARCH_ENGINE | 83 | 82 | 98.8 % |
| TLS_CT_LOOKUP | 20 | 20 | 100.0 % |
| WEB_CRAWL | 11 | 11 | 100.0 % |
| WHOIS_LOOKUP | 60 | 58 | 96.7 % |
| **all recording-backed tools** | **242** | **238** | **98.4 %** |

"Replayed" counts a recorded `unavailable` too: a recorded provider 404, or a
`WEB_CRAWL` host outside the recording allowlist, is real recorded evidence of
what the provider answered, not a gap. `tests/expected_numbers.py` derives
this table's numbers from the committed expected run and
`tests/test_offline_three_cycles.py` fails the suite if any tool drops below
95 %, so the claim cannot go stale. The four social tools
(`X_SEARCH`/`REDDIT_SEARCH`/`TELEGRAM_SEARCH`/`DISCORD_SEARCH`) are excluded:
they are not recording-backed but answered from the Step 1 collected corpus.

The query list behind the table is built in two reproducible offline passes —
`make build-fixtures` derives every request the fixture *post text* implies,
`make harvest-tool-queries` runs the demo and collects the requests it cannot
predict (free-text cluster search phrases, indicators pivoted out of recorded
evidence) into `fixtures/tools/harvested_queries.json`, which the builder
merges back in. See [`fixtures/README.md`](../fixtures/README.md).

## Modes and determinism

| Mode | LLM/judges | Collectors (X/Reddit/Telegram/Discord) | Tools (WHOIS/PDNS/CT/crawl/search/NVD/GitHub) |
|---|---|---|---|
| `offline` | heuristic (deterministic) | recorded fixtures only | recorded tool responses only; no fixture/recording → `unavailable` |
| `auto` (CLI default) | live if configured, else heuristic | live if credentials configured, else fixture, else `unavailable` | live if network-reachable, else recording |
| `live` | live (required; fails without a configured provider unless `--allow-heuristic-reports`) | live if credentials configured, else `unavailable` (never silently falls back to fixture data) | live if network-reachable, else `unavailable` |

`TIBLENDER_MODE` selects the mode (`.env`/environment; Docker Compose defaults
to `offline`). `tiblender doctor` reports the resolved per-service mode and
reason, and every cycle's `cycle_summary.json` and report `metadata.services`
record it too.

A `live` run whose required services are not configured **exits 2 before
creating anything on disk**: `run-cycles` and `cycle` resolve the same service
table `tiblender doctor` prints, and refuse to start when the LLM is
unavailable (and `TIBLENDER_ALLOW_HEURISTIC_REPORTS` is not set) or when none
of the four collectors has credentials. No cycle directory, no `posts.jsonl`,
no `run_summary.json` — nothing is written, and the message names the missing
piece.

### Per-report generation mode

Each report records how its own content was produced, in
`metadata.generation_mode`:

| `generation_mode` | Meaning |
|---|---|
| `llm` | Every LLM-backed stage of that report's pipeline produced a usable draft. |
| `llm_partial` | The run was LLM-backed, but at least one stage fell back to its deterministic path after a retry (truncated or malformed draft, provider error) **for this report**. The stages are named in `metadata.llm_stage_fallbacks`. |
| `offline_deterministic` | No LLM was used (offline/heuristic run). |

`generation_mode` is decided **per report, not per cycle**. A report is
`llm_partial` only when the fallback actually touched it: a post-normalization
chunk that carried one of its own posts, a cluster-refinement batch that
carried its own cluster, or one of its own investigation-planning,
perspective, judge or report-field calls. A fallback in another cluster's
batch leaves this report `llm`, so a cycle routinely mixes `llm` and
`llm_partial` reports.

Cycle-wide accounting lives in `cycle_summary.json` instead: `llm_fallbacks`
counts the fallbacks per stage and `llm_fallback_reasons` counts the same
events by cause (`truncated`, `timeout`, `http_status`, `network`,
`empty_content`, `json_parse`, `invalid_response`, `unknown`). The cause is a
category this code attaches where the failure is raised — never inferred from
the failure text, which carries the provider's reply and the model's own draft
of scraped post content. The reason totals equal the sum of the **per-stage**
counters; `llm_fallbacks` also carries two bookkeeping counters that are
outside that sum, `perspective_focus_padded` (padding is not a fallback, so it
has no cause) and `unattributed_fallbacks` (how many fallbacks carried no
cluster/post scope and so could not reach any report). Every fallback also
emits one WARNING naming the stage, the cause, the affected cluster ids and a
short phrase written by this code; the provider's and the model's own words go
to DEBUG, so a WARNING-level log carries no collected post text. Nothing else
in the package logs an `LLMGenerationError` above DEBUG, and the fallback notes
that reach a published report name the stage and the cause category only.

Batch agents chunk their input (`TIBLENDER_LLM_BATCH_SIZE` posts per
normalization call, `TIBLENDER_LLM_CLUSTER_BATCH_SIZE` clusters per
refinement call) so one provider reply never has to carry a whole cycle; a
reply cut off at the provider's output cap (`finish_reason=length`) is
detected as truncation and retried with a halved chunk before any fallback.

### What makes an offline run reproducible

Determinism in `offline`/`auto`-without-credentials mode comes from:

- **Fixed seed**: `TIBLENDER_SEED` (default `20260909`) seeds every random
  draw (hashing embeddings, HDBSCAN-compatible clustering, the RL policy).
- **Fixed clock**: `TIBLENDER_CLOCK=auto` resolves to a fixed clock whenever
  the run is fully offline; the "current time" for report/STIX IDs and
  timestamps is derived from each cycle's collection window end, not wall
  time. Report and STIX bundle IDs are derived from cluster content hashes.
- **Recorded tool responses/fixtures**: every answer is a recorded live
  provider response. A tool or collector with no recording/fixture for a
  given query reports `status=unavailable`, exactly as it would live with no
  credentials.
- **Fixed viewer build stamp**: `build-viewer` under a fixed clock stamps the
  bundle's top-level `generated_at` from the newest report it bundles instead
  of from wall time, so rebuilding the same reports rewrites the same bytes
  (see [`viewer/README.md`](../viewer/README.md)).

Two consecutive offline runs into the same `./output` therefore produce
byte-identical trees, `viewer/reports_viewer_data.js` included.

`tiblender verify-sample` compares a `run-cycles --output-root` tree against
the committed `sample_runs/offline_three_cycles/expected/` fixture; see the
`verify-sample` row of the [CLI reference](#cli-reference) for its exit codes.

**The reference embeddings do not apply in offline mode.**
`TIBLENDER_MODE=offline` pins *both* embedding backends — Step 2 clustering
(`TIBLENDER_EMBEDDING_BACKEND`) and the cross-cycle trend snapshots
(`TIBLENDER_TREND_EMBEDDING_BACKEND`) — to the deterministic hashing
vectorizer, so that model weights, device, and library version cannot leak
into a run that is meant to be bit-reproducible. Under that pin
`TIBLENDER_EMBEDDING_BACKEND`, `TIBLENDER_EMBEDDING_MODEL`,
`TIBLENDER_TREND_EMBEDDING_BACKEND` and `TIBLENDER_EMBEDDING_FALLBACK` have no
effect, and `TIBLENDER_EMBEDDING_FALLBACK=false` does not raise, because the
pin is applied before the backend is dispatched. Using `bge_large` or `minilm`
for real needs all three of: `TIBLENDER_MODE=auto` or `live` (which also
switches the network-only tools to live), the `sentence-transformers`
dependency from `--with-reference`, and the model weights already present in
the local cache — the loaders pass
`local_files_only=TIBLENDER_EMBEDDING_LOCAL_FILES_ONLY` (default `true`) and
never download at run time. `tiblender doctor` reports the pin in its
`embedding` row, and every `cycle_summary.json` records the requested *and*
the resolved backend under `embedding_resolution`.
