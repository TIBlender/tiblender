# Pipeline Overview

One scheduler cycle (`tiblender cycle` / `run-cycles`, or the `Dockerfile`'s
default command) is `collect` (four-platform adaptive collection) followed by
one LangGraph-backed `run` (the five in-cycle steps + cross-cycle feedback).
`tiblender_runtime/scheduler.py` drives both and shares state that must
survive across cycles — dynamic queries, RL policy state, trend history, and
per-collector state — under one `<output_root>/history/` directory.

## Stage → Module Mapping

| Stage | Module(s) | Notes |
|---|---|---|
| Collection (Step 1, adaptive collection) | `collectors/{x,reddit,telegram,discord}.py`, `collectors/base.py`, `collectors/http.py`, `collectors/keywords.py`, `collectors/fixtures.py`, `collectors/preprocess.py`, `collectors/corpus.py`, `collectors/runner.py` | Four official-API collectors sharing one `parse_*` path between a live transport and a fixture transport. X and Reddit are keyword-search driven (static + dynamic queries); Telegram and Discord collect exhaustively from the communities the configured account has already joined and ignore the query plan. `preprocess.py` applies platform-specific noise rules, the short-text+no-keyword discard, and OCR merge; `corpus.py` deduplicates by `(platform, external_id)` into `posts.jsonl`. |
| Step 1 (normalization + IoC extraction) | `normalization.py`, `iocs.py`, `llm.py`, `language.py` | The post-normalization LLM prompt converts multilingual posts into English threat summaries with IoCs preserved verbatim; the heuristic path reproduces the same `Non-threat post:` judgment deterministically offline. The collected corpus is multilingual and the bundled fixtures carry non-English posts, so this path runs in the demo. A published report quotes the post body in the language it was posted in (`sources.social_media_posts[].text`, `source_posts.items[].text`, the Markdown "Source Evidence" section) and writes everything *about* it in English -- see Step 5. |
| Step 2 (campaign-level clustering) | `clustering.py` | IoC-Jaccard track (`τ_J=0.25`, `τ_abs=0.75`) + configurable embeddings + semantic absorption + HDBSCAN-compatible density clustering (`min_cluster_size=3`, `min_samples=2`, `ε=0.08`), then LLM/heuristic campaign splitting and severity refinement (`θ_sev=MEDIUM`). |
| Step 3 (multi-perspective / RL-guided investigation) | `investigation.py`, `perspectives.py`, `context.py`, `tools.py`, `tool_fixtures.py`, `tools_social.py`, `tools_searxng.py`, `rl_policy.py`, `rl/` | A planner (LLM/heuristic) generates queries; four perspective agents (Infrastructure, Technical, Social, Actor) run in parallel threads, each selecting focus types, generating queries, and pivoting one hop; `context.py`'s `aggregate_context` merges cross-perspective IoCs with the confidence rules below. `rl/policy.py` decides RL-Guided vs. Multi-Perspective mode per cluster. |
| Step 4 (adaptive evaluation loop) | `evaluation.py` | Two independently-configurable LLM judges score sufficiency/consistency/evidence-quality and return `PROCEED`/`NEED_MORE_INFO`/`SKIP`; conservative consensus (any `SKIP` suppresses the report, all judges must `PROCEED`); `NEED_MORE_INFO` triggers a targeted Multi-Perspective follow-up, up to `N_max=3` rounds. |
| Step 5 (report generation) | `reporting.py`, `stix.py`, `pdf.py` | 11-section structured report (cluster_name, severity, executive_summary, key_findings, recommendations with an `immediate`/`short_term`/`long_term` horizon, iocs with `extraction_type` and the source `platforms` the indicator was observed on, cves, ttps, threat_actor, timeline, sources whose posts carry `text`); per-indicator confidence is kept in `iocs_evidence` rather than on the released IoC rows, matching the published dataset's record shape. `stix.py` builds the STIX 2.1 bundle, `pdf.py` renders the PDF sidecar. IoC disqualification rules (news/CDN/shortener/SNS URLs, 404/NXDOMAIN/clean results) are applied before export. **English-only, both generation paths.** `language.py` holds one guard (`text_is_english`) and one deterministic English composer. The offline stand-in composes its Step 5 prose from the cluster's facts -- category vocabulary, platform and post counts, indicators, CVEs, brands, English keyword matches -- and never quotes a collected post; the LLM path is asked for English, retried once with an explicit English-only instruction when the guard rejects a draft, and falls back to the same composer on a second rejection (`non_english_output`). `build_report` applies the guard once more to the assembled sections, so a non-English tool summary quoted into `key_findings` is rewritten too; such a report carries `metadata.language_fallback=true` and `cycle_summary.json` counts it under `language_guard`. `ttps` ships as the technique ids and `sources` as `{source_number, source_ref, summary}`, the released record's shapes; the runtime's richer views (full TTP objects, post bodies, tool results, `english_report`) live at the report root, outside the 11 sections. |
| Cross-cycle feedback: trends | `trends.py` | Rolling `history/cluster_snapshots.jsonl` (window `W=168h`); adjacency `sim(c,h) ≥ τ_sim=0.80`; four axes — Volume (this cluster's post count ≥ μ+kσ of the post counts of its similar historical snapshots, k=1.5, gated on ≥4 such snapshots and ≥4 posts in the current cluster), Propagation (platform-set growth), Escalation (severity increase), Early_Warning (strategic-value score ≥ `θ_ew=0.75`). |
| Cross-cycle feedback: dynamic queries | `dynamic_queries.py` | Trend-triggered clusters produce `DynamicQuery` records (LLM or deterministic templates from `config/keywords.json`) written to `dynamic_queries.json`; consumed by the next cycle's `QueryPlan` (see [`config/README.md`](config/README.md)). |
| Cross-cycle feedback: RL policy | `rl/features.py`, `rl/network.py`, `rl/reward.py`, `rl/replay.py`, `rl/trainer.py`, `rl/policy.py`, `rl/actions.py` | Behaviour cloning each cycle on that cycle's severity≥MEDIUM trajectories, then DQN fine-tuning once the replay buffer holds ≥1,000 transitions; state persists in `history/policy_state.json`. |
| Orchestration | `workflow.py`, `pipeline.py`, `scheduler.py`, `feedback.py`, `llm_feedback.py`, `verify.py`, `viewer_data.py` | `workflow.py` builds the LangGraph `StateGraph`; `pipeline.py`/`scheduler.py` are the single-cycle and multi-cycle CLI entry points; `verify.py` powers `verify-sample`/`write-expected`; `viewer_data.py` powers `build-viewer`. |
| Configuration / contract | `settings.py`, `contract.py`, `services.py`, `models.py`, `clock.py` | `settings.py` resolves every `TIBLENDER_*` env var; `contract.py` defines the tool/perspective/focus allowlists used by prompts, dispatch, and `audit-contract`; `services.py` resolves each service's live/fixture/heuristic/unavailable mode per `TIBLENDER_MODE`. |

## LangGraph Node Sequence

`workflow.py` builds a `StateGraph` with a linear node sequence (no
conditional edges beyond each node's own internal control flow):

```
normalize_posts -> cluster_posts -> investigate_clusters -> evaluate_clusters -> generate_reports -> cross_cycle_feedback -> END
```

This mirrors the production control flow while running every node in a
single process (the production 32-worker pool is not used; parallelism is
limited to the four Step 3 perspective threads and the two Step 4 judges).

## Tool Contract

The Step 3 tool allowlist is defined in `tiblender_runtime/contract.py`
(`REQUIRED_TOOL_TYPES`) and checked by `tiblender audit-contract`:

| Tool | Purpose | Offline source |
|---|---|---|
| `SEARCH_ENGINE` | Web search | SearXNG JSON API (compose profile `live`); recorded tool response otherwise |
| `WHOIS_LOOKUP` | Domain registration | rdap.org; recorded tool response otherwise |
| `PASSIVE_DNS_LOOKUP` | Historical DNS resolutions | VirusTotal `/api/v3/{domains,ip_addresses}/{indicator}/resolutions?limit=40` (the paper's provider, keyed by `VIRUSTOTAL_API_KEY`); recorded tool response otherwise |
| `TLS_CT_LOOKUP` | Certificate transparency | crt.sh / certspotter; recorded tool response otherwise |
| `WEB_CRAWL` | Self-operated page retrieval | Playwright/Chromium by default, falling back to static HTTP; responses recorded only for allowlisted publisher hosts |
| `NVD_CVE_LOOKUP` | CVE detail lookup | NVD API (key optional); recorded tool response otherwise |
| `GITHUB_SEARCH` | Code/repo search | GitHub API (token optional); recorded tool response otherwise |
| `X_SEARCH` | X post search | X API v2 bearer token; falls back to bounded in-cycle corpus match |
| `REDDIT_SEARCH` | Reddit post search | Reddit OAuth2; falls back to bounded in-cycle corpus match |
| `DISCORD_SEARCH` | Discord message search | Step 1 input corpus only (not a live API call) |
| `TELEGRAM_SEARCH` | Telegram message search | Step 1 input corpus only (not a live API call) |

Every external call returns `status ∈ {success, unavailable, error}`;
missing credentials, provider failures, HTTP 404s, and DNS resolution
failures are recorded as `unavailable`, and unexpected
runtime failures are recorded as `error`.

`REQUIRED_PERSPECTIVES` are `Infrastructure`, `Technical`, `Social`, and
`Actor`; each has its own `REQUIRED_FOCUS_TYPES` list in `contract.py` (e.g.
Infrastructure covers `domain_registration_and_whois`,
`dns_and_passive_dns_analysis`, `ssl_tls_certificate_analysis`,
`related_domain_and_infrastructure_pivoting`, `hosting_and_osint`).

## Context Aggregation Confidence Rules

`context.py`'s `aggregate_context()` assigns IoC confidence from
cross-perspective evidence:

- ≥3 independent sources → `HIGH`; 2 sources → `MEDIUM`; 1 source → `LOW`. A
  source is a post (a different poster or platform) or one distinct
  `(tool, queried indicator)` answer; asking the same tool the same question
  twice is not a second source.
- Passive DNS / CT evidence linking the queried indicator to a **known
  malicious domain** — one the cluster's own posts named as an indicator —
  sharing its hosting IP or certificate raises confidence one level
  (`shared_hosting_link:+1`).
- A disconfirming provider marker (tool contradiction) lowers confidence one
  level (`contradiction:-1`). Nothing adds that tier back.
- Suppressed hosts are excluded outright; IoCs that remain `LOW` `Pivot`s
  after aggregation are excluded from the report. The host list is
  `tiblender_runtime/legitimate_hosts.py`, in two halves:
  `PUBLISHER_HOSTS` (publishers, vendors, standards bodies, national CERTs,
  social platforms, shorteners) matched by exact host **or subdomain suffix**,
  and `SHARED_SERVICE_PARENTS` (`pages.dev`, `workers.dev`, `r2.dev`,
  `vercel.app`, `amazonaws.com`, `cloudfront.net`, `github.io`,
  `cloudflaressl.com`, `azureedge.net`) matched **exactly**, because paper §V-A
  makes the specific tenant host the actionable indicator and the bare parent
  overly broad. The WEB_CRAWL recording allowlist is a subset of
  `PUBLISHER_HOSTS` by construction — a publisher or national CERT trusted
  enough to fetch can never have its own hostnames extracted as campaign
  infrastructure. `PUBLISHER_HOSTS` also absorbs `NON_CRAWLED_PUBLISHER_HOSTS`
  — publishers that only ever appear as a link inside a collected post, so the
  recorder never fetches them and the crawl allowlist cannot cover them
  (`blockmedia.co.kr`). Because extraction feeds clustering, changing this list
  changes which posts cluster together, not only which values are reported.
- `extraction_type` is `Direct` (found directly) or `Pivot` (found via a
  one-hop pivot from a directly-found IoC; `TIBLENDER_PIVOT_DEPTH=1`).
- A resolution answer (passive DNS / CT) for an IP that returns more than
  `TIBLENDER_PIVOT_MAX_RESOLUTIONS_PER_IP=5` values describes a **shared
  hosting environment**: the co-resident names are dropped instead of reported
  and the queried indicator is marked
  `shared_hosting_environment:co_tenants_dropped`. That marker carries **no**
  tier change — crowded hosting is a false-positive risk, not corroboration.
- A `Pivot` value not corroborated by a second independent tool answer never
  enters the report, and a report's `iocs`, `report_schema.iocs` and
  `iocs_evidence` are computed once from the same scored structure, so the
  flat list never states an infrastructure relationship the run did not verify
  and never disagrees with the evidence list about confidence.
- Step 4 will not publish a cluster whose scored evidence holds no indicator,
  no CVE and no TTP: there would be nothing actionable in the report.

## LLM chunking and per-unit fallbacks

Batch agents never send a whole cycle in one request: post normalization
sends `TIBLENDER_LLM_BATCH_SIZE` posts per call and cluster refinement
`TIBLENDER_LLM_CLUSTER_BATCH_SIZE` candidates per call, because a single
oversized request is what makes a provider stop at its output-token cap and
return a truncated JSON document (`finish_reason=length`, detected
explicitly).

Chunks of one batch agent are independent provider calls, so they are sent
concurrently — up to `TIBLENDER_LLM_MAX_CONCURRENT=8` at a time — and their
results are reassembled by chunk index, never by completion order. Step 3
and Step 4 apply the same idea one level up: clusters are investigated and
evaluated in a worker pool of `TIBLENDER_CLUSTER_WORKERS=4` (the paper's
cluster worker pool), with results collected in cluster order, the two judges
of a round asked concurrently, and `policy_state.json` read once per node
instead of once per cluster. Both settings are recorded in
`cycle_summary.json`'s `hyperparameters`; setting either to `1` runs the
stage serially and produces byte-identical output.

Two known limits of that concurrency:
`TIBLENDER_LLM_MAX_CONCURRENT` and `TIBLENDER_CLUSTER_WORKERS` bound each pool
separately rather than capping total in-flight LLM or VirusTotal requests, so
the worst case is their product and a provider rate limit is absorbed by the
per-unit retry rather than prevented. And a chunk pool always waits for every
chunk: there is no fail-fast, so one slow chunk holds up the stage it belongs
to even when another has already failed.

Every LLM-backed unit — one chunk, one cluster's plan, one perspective's
focus/query selection, one judge round, one report's prose, one trend
signal's dynamic queries — is retried once and then falls back to its
deterministic path instead of aborting the cycle. Nothing is hidden:

- `cycle_summary.json` carries an `llm_fallbacks` block counting
  `normalization_posts`, `cluster_refinement_batches`, `planner_clusters`,
  `perspective_units`, `judge_rounds`, `report_fields`,
  `dynamic_query_signals`, and `strategic_value_clusters`;
- a sibling `llm_fallback_reasons` block counts the same events by cause —
  `truncated` (`finish_reason=length`), `timeout`, `http_status` (a non-2xx
  provider reply), `network`, `empty_content`, `json_parse`,
  `invalid_response` (well-formed JSON missing the required fields) and
  `unknown`. The cause is authored by the code at the point the failure is
  raised (`LLMGenerationError.category`), never matched against the failure
  text: that text embeds the provider's reply and the model's own draft, and
  the draft summarises scraped, adversary-authored posts, so pattern-matching
  it would let a collected post steer a published histogram. Both blocks
  always carry every key, so zero is distinguishable from "not recorded";
- the two blocks' totals agree **over the per-stage counters**:
  `sum(llm_fallback_reasons) == sum(llm_fallbacks[k] for k in the eight stage
  keys)`. `llm_fallbacks` also carries two counters that are not stages and
  are therefore outside that sum — `perspective_focus_padded`, because padding
  is not a fallback and has no failure cause, and `unattributed_fallbacks`,
  a flag counting how many of the fallbacks above carried no cluster and no
  post scope (so they could not reach any report's `generation_mode`; no
  current call site does this). Such an event is additionally logged at ERROR
  only for the six stages that run *before* Step 5 builds the reports
  (`llm_fallbacks._PRE_REPORT_KEYS`); the two feedback stages,
  `dynamic_query_signals` and `strategic_value_clusters`, run after report
  generation and legitimately reach no report, so an unscoped fallback there
  is counted but not ERROR-logged;
- every fallback logs exactly one WARNING, built by
  `llm_fallbacks.format_fallback_line`, naming the stage, the cause
  category, the cluster ids it touched, the number of posts, and a bounded
  phrase written by this code. The provider's and the model's own words go to
  DEBUG only — no WARNING or ERROR anywhere in the package formats an
  `LLMGenerationError` — so a WARNING-level log can be read and shared without
  carrying collected post text. The same rule covers the report body: a
  perspective note about a fallback names the stage and the cause category,
  never the failure text;
- each affected report lists the stage names in
  `metadata.llm_stage_fallbacks` and its `metadata.generation_mode` becomes
  `llm_partial` instead of `llm`;
- a judge round that fell back is recorded as `rule-judge-<label>` in
  `evaluation_rounds`, never as an LLM judgement.

**`generation_mode` is per report, not per cycle.** The scope recorded with
each fallback is exactly the unit that failed — the post ids of a
normalization chunk, the cluster ids and post ids of a refinement batch, the
single cluster of a planner/perspective/judge/report-field call — and a
report is `llm_partial` only when that scope covers its own cluster or one of
its own posts. One failing batch among several therefore downgrades only the
reports built from that batch; the rest of the cycle stays `llm`.

The perspective focus-selection agent is a partial exception: when the LLM
returns at least one valid focus type for a perspective (just not the
two-focus minimum), the valid focus is kept and padded up to the minimum
with the deterministic focus ranking instead of discarding the whole answer.
That is not a fallback — it does not add a stage to
`metadata.llm_stage_fallbacks` and never downgrades `metadata.generation_mode`
— but it is still counted, in the same `llm_fallbacks` block, as
`perspective_focus_padded`. Only a zero-valid answer (or an error after
retry) is the real `perspective_units` fallback above.

A `live` run whose required services are unavailable fails fast before any
output is written: `run-cycles`/`cycle` resolve the same service table
`tiblender doctor` reports and exit 2 without creating the cycle directory
when the LLM is unconfigured (and `TIBLENDER_ALLOW_HEURISTIC_REPORTS` is not
set), or when none of the four collectors has credentials. Nothing —
`posts.jsonl`, `collection_report.json`, `history/collector_state.json`,
`run_summary.json` — is written by such a run.

## Hyperparameters

Every value below has a `TIBLENDER_*` (or documented alias) environment
variable in `.env.example`; defaults shown are `settings.py`'s.

**LLM runtime** — `TIBLENDER_LLM_PROVIDER=azure_openai`,
`TIBLENDER_LLM_MODEL`, `TIBLENDER_LLM_TEMPERATURE=0.0`,
`TIBLENDER_LLM_MAX_OUTPUT_TOKENS=4096`, `TIBLENDER_LLM_TIMEOUT_SECONDS=60`,
`TIBLENDER_LLM_BATCH_SIZE=20`, `TIBLENDER_LLM_CLUSTER_BATCH_SIZE=10`,
`TIBLENDER_LLM_MAX_CONCURRENT=8`,
`TIBLENDER_ALLOW_HEURISTIC_REPORTS=false`. Judge overrides (each mirrors the
main profile's fields; unset ones inherit the main LLM profile):
`TIBLENDER_JUDGE_A_PROVIDER`, `TIBLENDER_JUDGE_A_MODEL`,
`TIBLENDER_JUDGE_A_OPENAI_BASE_URL`, `TIBLENDER_JUDGE_A_OPENAI_API_KEY`,
`TIBLENDER_JUDGE_A_AZURE_OPENAI_ENDPOINT`,
`TIBLENDER_JUDGE_A_AZURE_OPENAI_DEPLOYMENT`,
`TIBLENDER_JUDGE_A_AZURE_OPENAI_API_KEY`,
`TIBLENDER_JUDGE_A_AZURE_OPENAI_API_VERSION`, and the matching
`TIBLENDER_JUDGE_B_PROVIDER`, `TIBLENDER_JUDGE_B_MODEL`,
`TIBLENDER_JUDGE_B_OPENAI_BASE_URL`, `TIBLENDER_JUDGE_B_OPENAI_API_KEY`,
`TIBLENDER_JUDGE_B_AZURE_OPENAI_ENDPOINT`,
`TIBLENDER_JUDGE_B_AZURE_OPENAI_DEPLOYMENT`,
`TIBLENDER_JUDGE_B_AZURE_OPENAI_API_KEY`,
`TIBLENDER_JUDGE_B_AZURE_OPENAI_API_VERSION`.

**Step 1** — `TIBLENDER_MIN_TEXT_CHARS=24`.

**Step 2 (clustering)** — `TIBLENDER_JACCARD_THRESHOLD=0.25`,
`TIBLENDER_ABSORPTION_THRESHOLD=0.75`, `TIBLENDER_MIN_CLUSTER_SIZE=3`,
`TIBLENDER_MIN_SAMPLES=2`, `TIBLENDER_CLUSTER_EPSILON=0.08`,
`TIBLENDER_EMBEDDING_BACKEND=hashing` (`bge_large` optional, 1024-d),
`TIBLENDER_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5`,
`TIBLENDER_EMBEDDING_FEATURES=384`, `TIBLENDER_EMBEDDING_FALLBACK=true`,
`TIBLENDER_EMBEDDING_LOCAL_FILES_ONLY=true`, `TIBLENDER_EMBEDDING_DEVICE`
(unset = library default; e.g. `cpu`/`cuda`, reference backends only),
`TIBLENDER_HDBSCAN_BACKEND=sklearn`,
`TIBLENDER_MERGE_SIMILARITY_THRESHOLD=0.92`,
`TIBLENDER_MERGE_MAX_TIME_GAP_SECONDS=21600`,
`TIBLENDER_MERGE_MIN_CONTEXT_OVERLAP=2`, `TIBLENDER_MIN_SEVERITY=MEDIUM`.

`TIBLENDER_MODE=offline` **pins both embedding backends** — this one and
`TIBLENDER_TREND_EMBEDDING_BACKEND` — to the deterministic hashing
vectorizer (`services.resolved_embedding_backend`), so that a run meant to be
bit-reproducible cannot have model weights, device, or library version leak
into its output. Under that pin `TIBLENDER_EMBEDDING_BACKEND`,
`TIBLENDER_EMBEDDING_MODEL`, `TIBLENDER_TREND_EMBEDDING_BACKEND` and
`TIBLENDER_EMBEDDING_FALLBACK` have no effect, and
`TIBLENDER_EMBEDDING_FALLBACK=false` does not raise, because the pin is
applied before the backend is dispatched. `bge_large`/`minilm` therefore
require `TIBLENDER_MODE=auto` or `live`, the optional `sentence-transformers`
dependency (`bash install.sh --with-reference`), and the model weights already in
the local cache — the loaders pass
`local_files_only=TIBLENDER_EMBEDDING_LOCAL_FILES_ONLY` (default `true`) and
never download at run time. `tiblender doctor` surfaces the pin in its
`embedding` row, and every `cycle_summary.json` records both the requested and
the resolved backend under `embedding_resolution`
(`requested_backend`, `resolved_backend`, `requested_trend_backend`,
`resolved_trend_backend`, `offline_pin_applied`).

**Step 3 (investigation / RL)** — `TIBLENDER_RL_SIMILARITY_GATE=0.60`
(`θ_rl`), `TIBLENDER_RL_MAX_STEPS=10` (`T_max`), `TIBLENDER_RL_ACTION_TOP_K=5`,
`TIBLENDER_RL_POLICY_BACKEND=multiaction_dqn` (or `disabled`),
`TIBLENDER_PIVOT_DEPTH=1`, `TIBLENDER_PIVOT_MAX_RESOLUTIONS_PER_IP=5`,
`TIBLENDER_PASSIVE_DNS_LIMIT=20`,
`TIBLENDER_PERSPECTIVE_WORKERS=4`,
`TIBLENDER_CLUSTER_WORKERS=4`,
`TIBLENDER_MAX_QUERIES_PER_PERSPECTIVE=5`, `TIBLENDER_ENABLE_PIVOT_IOCS=true`.
Reference RL network/training:
`TIBLENDER_REFERENCE_RL_HIDDEN_DIM=256`, `TIBLENDER_REFERENCE_RL_DROPOUT=0.10`,
`TIBLENDER_REFERENCE_RL_BC_EPOCHS=20`, `TIBLENDER_REFERENCE_RL_BC_BATCH_SIZE=32`,
`TIBLENDER_REFERENCE_RL_BC_LEARNING_RATE=0.0003`,
`TIBLENDER_REFERENCE_RL_BC_PATIENCE=3`,
`TIBLENDER_REFERENCE_RL_BC_VALIDATION_SPLIT=0.10`,
`TIBLENDER_REFERENCE_RL_DQN_EPISODES=50`,
`TIBLENDER_REFERENCE_RL_DQN_LEARNING_RATE=0.0001`,
`TIBLENDER_REFERENCE_RL_REPLAY_CAPACITY=10000`,
`TIBLENDER_REFERENCE_RL_TRAINING_MIN_TRANSITIONS=1000`,
`TIBLENDER_REFERENCE_RL_EPSILON_INIT=1.0`,
`TIBLENDER_REFERENCE_RL_EPSILON_MIN=0.01`,
`TIBLENDER_REFERENCE_RL_EPSILON_DECAY=0.9`,
`TIBLENDER_REFERENCE_RL_DISCOUNT=0.99`.

**Step 4 (evaluation)** — `TIBLENDER_MAX_EVAL_ROUNDS=3` (`N_max`).

**Tools** — `TIBLENDER_ENABLE_REAL_TOOLS=true`,
`TIBLENDER_TOOL_TIMEOUT_SECONDS=20`, `TIBLENDER_MAX_TOOL_RESULTS=5`,
`TIBLENDER_TOOL_MAX_BYTES=2097152` (2 MiB read cap for a tool's JSON response
body; NVD exceeds 200 KB for a widely-referenced CVE, and a cap below the real
payload truncates the body mid-string so the lookup fails to parse),
`TIBLENDER_WEB_CRAWL_MAX_BYTES=200000`, `TIBLENDER_WEB_CRAWL_RENDER_JS=true`,
`TIBLENDER_WEB_CRAWL_WAIT_MS=1000`, `SEARCH_ENGINE_PROVIDER=searxng`,
`SEARCH_ENGINE_MAX_RESULTS=10`, `SEARXNG_BASE_URL=http://searxng:8080`,
`NVD_API_KEY`/`TIBLENDER_NVD_API_KEY`, `GITHUB_TOKEN`/`TIBLENDER_GITHUB_TOKEN`,
`VIRUSTOTAL_API_KEY`/`TIBLENDER_VIRUSTOTAL_API_KEY`.

**Collection** — `TIBLENDER_COLLECT_WINDOW_HOURS=1.0`,
`TIBLENDER_MAX_DYNAMIC_QUERIES_PER_PLATFORM=30`,
`TIBLENDER_MAX_POSTS_PER_QUERY=100`, `TIBLENDER_OCR_ENABLED=true`,
`TIBLENDER_TELEGRAM_MAX_CHANNELS=0`, `TIBLENDER_DISCORD_MAX_GUILDS=0`,
`TIBLENDER_DISCORD_MAX_CHANNELS_PER_GUILD=0` (all three: `0` = no cap),
`TIBLENDER_X_BEARER_TOKEN`/`TIBLENDER_TWITTER_BEARER_TOKEN`/`TIBLENDER_X_BEARER_TOKEN_FILE`,
`TIBLENDER_REDDIT_CLIENT_ID`/`TIBLENDER_REDDIT_CLIENT_SECRET`,
`TIBLENDER_TELEGRAM_API_ID`/`TIBLENDER_TELEGRAM_API_HASH`/`TIBLENDER_TELEGRAM_SESSION`,
`TIBLENDER_DISCORD_USER_TOKEN` (preferred) / `TIBLENDER_DISCORD_BOT_TOKEN`.

**Cross-cycle feedback** — `TIBLENDER_TREND_WINDOW_HOURS=168` (`W`),
`TIBLENDER_TREND_SIM_THRESHOLD=0.80` (`τ_sim`),
`TIBLENDER_VOLUME_SIGMA_K=1.5` (`k`),
`TIBLENDER_TREND_MIN_SIMILAR_SNAPSHOTS=4`,
`TIBLENDER_TREND_VOLUME_MIN_POSTS=4`,
`TIBLENDER_EARLY_WARNING_THRESHOLD=0.75` (`θ_ew`),
`TIBLENDER_TREND_EMBEDDING_BACKEND=hashing` (`minilm` optional).

**Scheduler** — `TIBLENDER_CYCLE_INTERVAL_SECONDS=3600`.

**Determinism** — `TIBLENDER_SEED=20260909`, `TIBLENDER_CLOCK=auto`,
`TIBLENDER_MODE=auto` (`offline`/`live` also accepted).

Run `tiblender show-config` to see every resolved value for the current
environment (credentials redacted), and `tiblender audit-contract` to check
the tool/perspective/focus contract for consistency.
