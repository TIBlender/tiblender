# Credentials

Every credential in this document is optional. Without it, the corresponding
service degrades to a recorded fixture/tool response (offline/auto mode) or
`unavailable` (live mode) — and `tiblender doctor`
reports exactly which mode each service resolved to. Copy `.env.example` to
`.env` and fill in only the sections you need.

## X (Twitter)

**Unlocks:** the X collector (recent search + monitored-user timelines) and
the `X_SEARCH` investigation tool.

1. Apply for the X Developer Portal (<https://developer.x.com/>) and create a
   project/app. Recent search (`GET /2/tweets/search/recent`) requires the
   paid **Basic** access tier or higher — it is not available on the free
   tier.
2. Generate an App-only **Bearer Token** for the app.
3. Set one of:
   - `TIBLENDER_X_BEARER_TOKEN=<token>` (or the generic `X_BEARER_TOKEN` /
     `TWITTER_BEARER_TOKEN` aliases), or
   - `TIBLENDER_X_BEARER_TOKEN_FILE=<path>` pointing at a file containing one
     or more newline-separated bearer tokens, for rotation across a token
     pool. `TIBLENDER_X_BEARER_FALLBACK_TOKEN_FILE` is consulted if every
     token in the primary file is rejected/rate-limited.
4. Optionally list accounts to monitor directly (in addition to keyword
   search) in `config/x_users.txt`, one handle per line.

**Without a token:** the X collector replays `fixtures/collectors/x/` and
`X_SEARCH` falls back to matching the collected in-cycle corpus, exactly as
it would with expired/invalid credentials in live mode (`unavailable`).

## Reddit

**Unlocks:** the Reddit collector (`/r/{sub}/new` + full-site search) and the
`REDDIT_SEARCH` investigation tool.

1. Create a Reddit account, then go to
   <https://www.reddit.com/prefs/apps> and click **create another app...**.
2. Choose type **script** and set the redirect URI to
   `http://localhost:8080`. It is never used: this is a
   client-credentials-only integration, not a user OAuth flow.
3. Note the client id (under the app name) and client secret.
4. Set `TIBLENDER_REDDIT_CLIENT_ID` and `TIBLENDER_REDDIT_CLIENT_SECRET` (or
   the generic `REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET` aliases).
   `TIBLENDER_REDDIT_USER_AGENT` defaults to `tiblender/0.1`; Reddit requires
   a descriptive, unique user agent string per its API rules.
5. Optionally set `TIBLENDER_REDDIT_ACCOUNTS_FILE` (multiple app
   credentials for rotation) or `TIBLENDER_REDDIT_EXTRA_ACCOUNTS` (inline
   JSON). The production subreddit watch list lives in
   `config/subreddits.txt`, one subreddit per line.

**Without credentials:** the collector replays `fixtures/collectors/reddit/`
and `REDDIT_SEARCH` falls back to the in-cycle corpus.

## Telegram

**Unlocks:** the Telegram collector — exhaustive incremental message history
from the channels/supergroups the account has already joined. Telegram's Bot
API cannot read arbitrary public channel history, so this integration uses
Telethon (a user session) instead of a bot — see `docs/FIDELITY.md` for why.

1. Log in at <https://my.telegram.org/apps> with a Telegram account and
   create an application to get an `api_id` (integer) and `api_hash`
   (string).
2. Set `TIBLENDER_TELEGRAM_API_ID` and `TIBLENDER_TELEGRAM_API_HASH` in
   `.env`.
3. Create the user session once, interactively:

   ```bash
   python3 -m tiblender_runtime telegram-login --session config/telegram_session
   ```

   (or `tiblender telegram-login --session ...` once installed). This
   requires the optional `telethon` dependency (already pinned in
   `requirements.txt`) and prompts on stdin for your phone number, the login
   code Telegram sends you, and your two-factor password if one is set. It
   prints the resulting session file path
   (`<session>.session`) on success. Exits with code 2 and a clear message
   if `telethon` is not installed, or if no `api_id`/`api_hash` is available
   (via `--api-id`/`--api-hash` or the environment variables above).
4. Set `TIBLENDER_TELEGRAM_SESSION` to the same session name/path used in
   step 3, so the collector reuses the authorized session instead of
   prompting again.
5. Choose what to collect. `config/telegram_channels.txt` takes one channel
   per line (a public `@username`, a bare username, or a numeric channel id);
   it ships with comments and one commented-out example only. When that file
   lists no channel, the collector reads **every** channel and supergroup the
   signed-in account has joined (`client.get_dialogs()`).
   `TIBLENDER_TELEGRAM_MAX_CHANNELS` bounds one run (`0`, the default, means
   no cap); channels are ordered by numeric id before the cap is applied, so a
   capped run always picks the same subset rather than following Telegram's
   dialog recency order. Message history is incremental: a per-channel
   `min_id` watermark is carried in `history/collector_state.json`, and each
   cycle reads the **oldest** unseen messages above it (up to
   `messages_per_channel`, 100) rather than the newest, so a channel that
   produced more messages than one page holds is drained contiguously over
   successive cycles instead of having its backlog skipped. FloodWait replies
   are backed off and retried once.

**Finding and joining channels is an operational task, not part of this
repository.** Channel discovery (keyword search, snowball expansion through
message links, mentions, forwards and channel recommendations) and the
decision to join a channel belong to the deployment, where they are
semi-automated with analyst verification. This code collects only from
communities the configured account is already a member of.

**Without a session:** the collector replays
`fixtures/collectors/telegram/`.

## Discord

**Unlocks:** the Discord collector — message collection from the
text/announcement channels of the guilds the credential's account has already
joined (`/users/@me/guilds` → `/guilds/{id}/channels` →
`/channels/{id}/messages`, with an `after` watermark per channel).

Two credential shapes are supported, and the **user-account token wins when
both are set**, because that is what the deployment runs on:

1. **User-account token (preferred).** `TIBLENDER_DISCORD_USER_TOKEN` (or the
   generic `DISCORD_USER_TOKEN` alias) is the token of a user account that is
   a member of the target servers; obtain it through your own compliant
   process. It is sent as a bare `Authorization: <token>` header with a
   browser-like `User-Agent`. Automating a user account is subject to
   Discord's terms of service; the paper's ethics section covers this
   deployment, and operating it that way is your decision to make and
   document.
2. **Bot token (alternative).** Create an application at the Discord
   Developer Portal (<https://discord.com/developers/applications>), add a
   **Bot**, reset/copy its token into `TIBLENDER_DISCORD_BOT_TOKEN` (or the
   generic `DISCORD_BOT_TOKEN` alias), and enable the **Message Content
   Intent** so message text is readable. Invite the bot with scope `bot` and
   permissions **View Channels** + **Read Message History** (no message-send
   permission is needed). The token is sent as `Authorization: Bot <token>`.
   A bot only ever sees guilds it has been invited into, which is usually a
   much narrower view than a member account's.

**Scoping a run.** `config/discord_guilds.txt` optionally narrows collection
to specific guilds (and, with `guild_id:channel_id,...`, specific channels);
it ships with a commented-out example only, and an empty list means every
joined guild. `TIBLENDER_DISCORD_MAX_GUILDS` and
`TIBLENDER_DISCORD_MAX_CHANNELS_PER_GUILD` bound one run (`0`, the default,
means no cap) — useful for a first live smoke run against an account that is
a member of dozens of servers. A guild id listed in `config/discord_guilds.txt`
that the account has not joined is recorded as an error rather than silently
ignored. Live requests are spaced 2–5 s apart and HTTP 429 replies are retried
up to three times honouring `Retry-After`; a 403/404 on one channel is recorded
as a per-channel error and never aborts the cycle.

A channel that answers 403 still counts against
`TIBLENDER_DISCORD_MAX_CHANNELS_PER_GUILD`, because the cap is applied to the
channel listing before readability is known — on a member account expect a
capped run to read noticeably fewer channels than the cap suggests. Each
readable channel yields at most three pages of 100 messages (300) per cycle;
that is a politeness bound, not a loss, because the `after` watermark is
anchored at the oldest message the cycle has not seen, so a backlog is drained
over successive cycles.

**Finding and joining servers is an operational task, not part of this
repository.** Server discovery (web search, server-directory sites, invite
scoring) and the decision to join are performed in the deployment, where they
are semi-automated with analyst verification. Nothing in this repository
discovers or joins a server, and no invite-candidate queue is produced.

**Without a token:** the collector replays `fixtures/collectors/discord/`.

## VirusTotal (passive DNS, only provider)

**Unlocks:** passive DNS lookups via the `PASSIVE_DNS_LOOKUP` tool, which
calls VirusTotal's
`GET /api/v3/domains/{domain}/resolutions?limit=40` and
`GET /api/v3/ip_addresses/{ip}/resolutions?limit=40` endpoints with an
`x-apikey` header. This is the provider the paper used.

1. Create a free account at <https://www.virustotal.com/> and copy the API
   key from your profile.
2. Set `VIRUSTOTAL_API_KEY` (or `TIBLENDER_VIRUSTOTAL_API_KEY`).

**Rate limits:** the VirusTotal free tier allows **4 requests per minute and
500 per day**. `tools/record_tool_responses.py` therefore sleeps **at least 16
seconds** between `PASSIVE_DNS_LOOKUP` calls (a per-tool floor that `--sleep`
cannot lower), and the runtime retries a single HTTP 429 after the
`Retry-After` delay before reporting `unavailable`.

**Failure handling:** HTTP 401/403 reports `unavailable` naming the
VirusTotal key, HTTP 404 reports `unavailable` "not found", and a persistent
429 reports `unavailable` after one retry.

**Without a key:** `doctor` and `show-config` report `passive_dns` as
`unavailable` with the reason naming the missing `VIRUSTOTAL_API_KEY`, and
offline/auto cycles replay the recorded tool responses.

## NVD / GitHub (optional)

**Unlocks:** higher rate limits for `NVD_CVE_LOOKUP` (NVD API) and
`GITHUB_SEARCH` (GitHub code/repo search). Both tools work unauthenticated
at a lower rate limit without a key.

- NVD: request a key at <https://nvd.nist.gov/developers/request-an-api-key>
  and set `NVD_API_KEY`/`TIBLENDER_NVD_API_KEY`.
- GitHub: create a personal access token (no special scopes needed for
  public code search) at
  <https://github.com/settings/personal-access-tokens> and set
  `GITHUB_TOKEN`/`GITHUB_API_KEY`/`TIBLENDER_GITHUB_TOKEN`.

## LLM (Steps 1, 2, 3, 4, 5)

**Unlocks:** live LLM-backed post normalization, cluster refinement,
investigation planning, judge evaluation, and report generation. Without a
configured provider, offline/auto runs fall back automatically to
deterministic heuristic reports
(`metadata.generation_mode=offline_deterministic`,
`metadata.llm_used=false`); live-mode runs fail with exit code 2 unless
`--allow-heuristic-reports`/`TIBLENDER_ALLOW_HEURISTIC_REPORTS=true` is set
for a smoke test.

Fill in **exactly one** of the three profiles in `.env.example`:

- **Profile A — Azure OpenAI / Azure AI Foundry** (default provider):
  `TIBLENDER_LLM_PROVIDER=azure_openai`, `TIBLENDER_AZURE_OPENAI_ENDPOINT`,
  `TIBLENDER_AZURE_OPENAI_DEPLOYMENT`, `TIBLENDER_AZURE_OPENAI_API_VERSION`,
  `TIBLENDER_AZURE_OPENAI_API_KEY` (generic `AZURE_OPENAI_*` aliases also
  accepted).
- **Profile B — OpenAI API or another OpenAI-compatible endpoint**:
  `TIBLENDER_LLM_PROVIDER=openai`, `TIBLENDER_LLM_MODEL`,
  `TIBLENDER_OPENAI_BASE_URL`, `TIBLENDER_OPENAI_API_KEY`.
- **Profile C — custom OpenAI-compatible report endpoint**:
  `TIBLENDER_LLM_PROVIDER=openai_compatible`, `TIBLENDER_REPORT_ENDPOINT`,
  `TIBLENDER_REPORT_DEPLOYMENT`, `TIBLENDER_REPORT_API_KEY`.

The paper used Grok 4 Fast (via Microsoft Foundry) for the generation agents;
this code accepts any OpenAI-compatible or Azure endpoint instead, so
Grok 4 Fast is not required or assumed.

**Judges (Step 4):** the paper used GPT-OSS-120B and Llama 4 Maverick as two
independent judges. Here, `TIBLENDER_JUDGE_A_*` and `TIBLENDER_JUDGE_B_*`
(provider/model/base URL/API key/Azure endpoint-deployment-version, mirroring
the main profile fields) let you point each judge at an independent
deployment. Leave them unset and both judges inherit the main LLM profile
above — set both only when reproducing genuinely independent judge
deployments.

## SearXNG (search tool, live mode only)

**Unlocks:** live `SEARCH_ENGINE` results instead of a recorded tool
response. No
API key is needed — SearXNG is a self-hosted, keyless search aggregator.

```bash
docker compose --profile live up --build
```

starts the bundled `searxng` service alongside `tiblender` and points
`SEARXNG_BASE_URL` at `http://searxng:8080` (the compose network alias). The
default compose profile does **not** start SearXNG, so `SEARCH_ENGINE`
resolves to a recorded tool response (or `unavailable`) unless `--profile
live`
is used or `SEARXNG_BASE_URL` is set to point at an externally-reachable
SearXNG instance.

`tiblender doctor` says which of the two you are in: outside offline mode it
opens a short TCP connection to `SEARXNG_BASE_URL` before reporting, so a
configured-but-not-running instance shows as

```json
{ "name": "search_engine", "mode": "unavailable",
  "reason": "search engine base URL http://searxng:8080 is configured but not reachable: ..." }
```

rather than as `live`. `SEARXNG_BASE_URL` has a non-empty default
(`http://searxng:8080`), so "the variable is set" alone never means the search
tool has a backend. That probe is the only outbound connection `doctor` makes,
and it is skipped entirely in offline mode.
