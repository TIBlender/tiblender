# Config

Static configuration for the Step 1 collection layer: the keyword catalog
plus per-platform seed lists. All files here are read by
`tiblender_runtime/collectors/keywords.py` (`KeywordCatalog.load` /
`load_lines`) at the start of each collection cycle (`TIBLENDER_CONFIG_DIR`,
default `config`) and combined with the previous cycle's dynamic queries
(see below) into one `QueryPlan`.

## `keywords.json`

The multilingual security-term catalog and each platform's static search
queries. Schema:

```jsonc
{
  "version": 1,
  "categories": [
    {"id": "phishing_fraud", "label": "Phishing/Fraud"},
    {"id": "malware_c2", "label": "Malware/C2"},
    {"id": "vulnerability", "label": "Vulnerability"},
    {"id": "apt_ransomware", "label": "APT/Ransomware"}
  ],
  "terms": {
    "<category id>": {
      "<language code>": ["term 1", "term 2", "..."]
    }
  },
  "platform_queries": {
    "<platform>": [
      {"category": "<category id>", "language": "<language code>", "query": "<platform-native search string>"}
    ]
  }
}
```

- `terms` currently covers 9 languages (en, ja, ko, ru, zh, es, pt, ar, fr)
  across the 4 categories; `KeywordCatalog.security_terms()` flattens this
  into the lowercase vocabulary used by `preprocess.py`'s short-text-discard
  rule (a short post is kept only if it contains a security term or an
  IoC/CVE pattern).
- `platform_queries` holds each keyword-search platform's ready-to-use
  static search strings — already platform-native syntax, e.g. X's queries
  append `-is:retweet -is:quote`. These run every cycle unconditionally,
  alongside any dynamic queries for that platform. Only `x` and `reddit`
  appear here: Telegram and Discord are collected exhaustively from the
  communities the configured account has joined, so they are not query
  driven.

## `x_users.txt`, `subreddits.txt`, `telegram_channels.txt`, `discord_guilds.txt`

Plain-text seed lists, one entry per line; blank lines and lines starting
with `#` are ignored (loaded by `collectors/keywords.py::load_lines`, except
`discord_guilds.txt` which has its own `guild_id[:channel_id,...]` parser in
`collectors/discord.py`):

| File | Format | Used by |
|---|---|---|
| `x_users.txt` | one X handle per line | X collector: timeline collection for monitored accounts, in addition to keyword search. Ships with commented-out examples only. |
| `subreddits.txt` | one subreddit name per line | Reddit collector: `/r/{sub}/new` static collection for each listed subreddit. Ships with the production watch list. |
| `telegram_channels.txt` | one `@username`, bare username, or numeric channel id per line | Telegram collector: which already-joined channels/supergroups to read history from. Ships with comments and one commented-out example only — when it lists nothing, the collector reads every channel the account has joined (`client.get_dialogs()`), ordered by numeric id before `TIBLENDER_TELEGRAM_MAX_CHANNELS` is applied. |
| `discord_guilds.txt` | one `guild_id` or `guild_id:channel_id,channel_id` per line | Discord collector: which already-joined guilds/channels to read messages from. Ships with a commented-out example only — when it lists nothing, the collector reads every guild the account has joined; a listed guild the account has not joined is recorded as an error. A read-gated channel that answers 403 still counts against `TIBLENDER_DISCORD_MAX_CHANNELS_PER_GUILD` (the cap is applied before readability is known), and each channel yields at most 300 messages per cycle. See `docs/CREDENTIALS.md`. |

## Dynamic Query Merging

Each cycle's `QueryPlan` (`collectors/keywords.py::build_query_plan`) is the
union of:

1. `keywords.json`'s `platform_queries` (static, every cycle).
2. `subreddits.txt` entries, wrapped as `subreddit`-sourced queries.
3. The previous cycle's **dynamic queries** — `dynamic_queries.json`,
   written by `tiblender_runtime/dynamic_queries.py` at the end of a cycle
   that detected a trend (see `PIPELINE_OVERVIEW.md`'s Cross-Cycle Feedback
   row) and consumed by `collectors/keywords.py::load_dynamic_queries` at
   the start of the next one. It lives under the cycle's output directory
   (`output/cycle_00N/dynamic_queries.json` in a multi-cycle run), not under
   `config/` — `config/` only holds the *static* seed lists above.

`dynamic_queries.json` is a JSON array of objects shaped like:

```json
[
  {
    "query": "sfrclak.com",
    "platforms": ["x", "reddit"],
    "trend_types": ["EARLY_WARNING"],
    "priority": 8,
    "language": "en",
    "source_report_id": "TIBLENDER-RUN-252a5d11fa1a",
    "source_cluster_id": "cluster-252a5d11fa1a",
    "generated_at": "2026-09-09T14:37:39Z",
    "expires_at": "2026-09-16T14:37:39Z",
    "rationale": "EARLY_WARNING trend detected on cluster cluster-252a5d11fa1a; seeded from domain 'sfrclak.com'."
  }
]
```

Loading rules (`load_dynamic_queries`):

- Each record expands into one `CollectionQuery` per listed platform.
- **TTL / expiry**: a record whose `expires_at` is in the past (relative to
  the cycle's clock) is dropped entirely. `expires_at` defaults to
  `generated_at + TIBLENDER_TREND_WINDOW_HOURS` hours (168h/7 days by
  default) when `dynamic_queries.py` writes the file.
- **Per-platform cap**: after sorting all expanded queries by
  `(-priority, query, platform)`, at most
  `TIBLENDER_MAX_DYNAMIC_QUERIES_PER_PLATFORM` (default 30) queries are kept
  *per platform* — lower-priority queries beyond the cap for a platform are
  dropped, not carried over.
  `priority` (1–10; `EARLY_WARNING`-only trends get a +1 bonus, see
  `dynamic_queries.py::_priority_for`) breaks ties before the cap is
  applied.
- A malformed record (not an object, missing `query`/`platforms`,
  unparsable `expires_at`, or a non-numeric `priority`) is skipped and
  logged, never a hard failure — one bad record must not stop collection for
  the rest of the plan.

`dynamic_queries.py` itself generates these records only for clusters where
`trends.py` detected at least one trend axis, using an LLM prompt (language/
region/context-aware) when configured, or a deterministic template
(CVE id / domain / malware name / brand token, localized with
`keywords.json`'s per-language terms) offline.
