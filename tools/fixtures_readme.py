"""The generated `fixtures/README.md`: template plus its computed sections.

`fixtures/README.md` is **generator-owned** -- `tools/build_fixtures.py`
rewrites it on every build so its per-cycle table and its WEB_CRAWL host
inventory can never drift from the bundle they describe. The prose lives
here rather than in the builder so the builder stays a readable CLI: it
wires selection, shaping and writing together, and this module owns the
document.

Edit the fixture documentation here, then re-run `make build-fixtures`.
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

def _readme_header(manifest: dict) -> str:
    """The generated, always-current numbers at the top of fixtures/README.md."""
    band = manifest["collection_band"]
    rows = []
    for cycle in manifest["cycles"]:
        counts = cycle["counts"]
        rows.append(
            "| {label} | {start} .. {end} | {x} | {reddit} | {telegram} | {discord} | {total} |".format(
                label=cycle["label"],
                start=cycle["window"]["start"][:10],
                end=cycle["window"]["end"][:10],
                x=counts["x"],
                reddit=counts["reddit"],
                telegram=counts["telegram"],
                discord=counts["discord"],
                total=cycle["total_posts"],
            )
        )
    return (
        "The bundle currently holds **{reports} released reports / {total} posts** "
        "({primary} primary + {noise} noise) across the collection band "
        "{band_start} .. {band_end}:\n\n"
        "| cycle | window (UTC) | x | reddit | telegram | discord | posts |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "{rows}\n"
    ).format(
        reports=len(manifest["selected_report_ids"]),
        total=manifest["total_posts"],
        primary=manifest["primary_post_count"],
        noise=manifest["noise_post_count"],
        band_start=band["start"][:10],
        band_end=band["end"][:10],
        rows="\n".join(rows),
    )


def _target_ramp_text() -> str:
    """The per-cycle primary-post ramp, read from the selector's own constant.

    Written out rather than quoted by hand so the prose cannot drift from
    `tools/fixture_selection.py::TARGET_PRIMARY_POSTS`.
    """
    from tools.fixture_selection import TARGET_PRIMARY_POSTS

    return " / ".join(str(value) for value in TARGET_PRIMARY_POSTS)


def _crawl_host_section(tool_queries: list[dict], allowed_hosts: tuple[str, ...]) -> str:
    """Generated inventory of the WEB_CRAWL hosts the current queries.json holds."""
    from tools.record_tool_responses import extract_host, is_allowed_crawl_host

    allowed: set[str] = set()
    skipped: set[str] = set()
    for entry in tool_queries:
        if entry["tool_type"] != "WEB_CRAWL":
            continue
        host = extract_host(entry["parameters"].get("url", ""))
        if not host:
            continue
        (allowed if is_allowed_crawl_host(host, allowed_hosts) else skipped).add(host)

    def _listing(hosts: set[str]) -> str:
        return "\n".join(f"- `{host}`" for host in sorted(hosts)) or "- (none)"

    return (
        "### Hosts in the current `queries.json`\n\n"
        "Regenerated with the fixtures. Hosts that the allowlist accepts, and "
        "will therefore actually be fetched by the next recording run:\n\n"
        f"{_listing(allowed)}\n\n"
        "Hosts the allowlist rejects. These are never contacted and never "
        "opened in a browser; each is recorded as "
        "`status: \"unavailable\"` with "
        "`error: \"crawl_skipped_untrusted_host: ...\"`:\n\n"
        f"{_listing(skipped)}\n"
    )


def write_readme(path: Path, manifest: dict, tool_queries: list[dict]) -> None:
    from tools.record_tool_responses import DEFAULT_ALLOWED_CRAWL_HOSTS

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Collector fixtures

`fixtures/collectors/<platform>/cycle_00{1,2,3}.json` are API-shaped fixture
files replayed by the offline `*FixtureTransport` classes in
`tiblender_runtime/collectors/`. They are **anonymized, defanged excerpts of
the released dataset** -- `dataset/raw_dataset/reports.jsonl`, the same
8,288-report file the paper's evaluation tables are recomputed from.

"""
        + _readme_header(manifest)
        + """
## Provenance

Every fixture file has `"provenance": "anonymized_dataset_excerpt"`, lists the
released report ids it draws from in `source_reports`, and carries a top-level
`posts` array with one entry per replayed post:
`{source_ref, platform, report_id, kind, timestamp}`. `source_ref` is the
released `source_posts.items[].source_ref` verbatim, so every post in the
bundle can be looked up in the released dataset's `raw_dataset/reports.jsonl`.
`fixtures/collectors/manifest.json` repeats the same records per cycle, plus
the selected report ids and the source file's name and sha256.

**Collection band.** The demo replays three *consecutive* collection windows
out of the 31-day deployment. `tools/build_fixtures.py` first picks one
contiguous band of `cycles * CYCLE_DAYS` (3 x 4 = 12) days: the band start
that maximizes how many released reports have at least
`MIN_POSTS_PER_REPORT` (3) dated posts inside it *and* carry an indicator
verbatim in their post text, ties broken by the earliest day. That is a
deterministic function of the dataset file alone. The band is then split into
three equal, back-to-back four-day windows, so the three windows are
contiguous and increasing by construction -- which is what the collectors'
incremental watermarks assume.

**Candidates.** A released report qualifies for exactly one cycle -- the
window holding most of its dated posts, earliest window on a tie -- and
contributes only its posts inside that window, capped at
`MAX_POSTS_PER_REPORT` (8, earliest first). It qualifies when at least
`MIN_POSTS_PER_REPORT` (3) posts remain *and* at least one of the report's
own indicators (an `iocs.items[].value` or a `report_text.cves` entry)
appears verbatim, after refanging, in the retained post text. So every
selected report really does carry an IoC or CVE into the cycle it lands in.
Undated posts are skipped (they cannot be placed in a window), as are posts
on platforms with no fixture transport. Language is not a selection
criterion: the released corpus is multilingual (Japanese, Korean, Russian,
Chinese, Arabic) and the bundle carries those posts, so the demo exercises
the multilingual path the paper describes. The runtime composes every
`report_text` section in English whatever language the posts were written in
(`tiblender_runtime/language.py`), and publishes the post body as source
evidence in the language it was posted in -- which is what the released dataset
does with `source_posts.items[].text`.

**Selection rule (explicit).** Per cycle, candidates are ordered
cross-platform-first, then by `created_at`, then by report id, and picked in
two passes:

1. *Platform coverage* -- for each of `x`, `reddit`, `telegram`, `discord` in
   turn, if no already-picked report covers it, take the first candidate that
   does. This is what puts all four platforms in every cycle from the
   selected reports, with no out-of-window filler.
2. *Volume* -- keep adding candidates in order until the cycle holds at least
   `TARGET_PRIMARY_POSTS[cycle - 1]` posts ("""
        + _target_ramp_text()
        + """, a ramp that keeps
   the per-cycle totals small and the whole bundle above 90 posts).

Then, bundle-wide, any of the four `paper_category` values still missing gets
its first candidate added. Only after selection is complete is *noise* added:
up to `MAX_NOISE_PER_CYCLE` (6) low-signal posts per cycle, drawn from
**non-selected** released reports in the same window and only from posts that
carry no IoC at all -- so noise can never introduce an indicator, a tool
query, or a recorded tool response of its own. There is no backfill path: if
a platform were ever missing from a cycle, that is a selection failure to
fix.

## Regenerated identifiers

Post, author, guild, and channel ids are regenerated deterministically by
`tools/fixture_shapes.py` (`numeric_id`/`base36_id`, seeded by the released
`source_ref` and the build salt) -- no real platform id is ever shipped.

Discord message ids are the one exception to plain hashing: they are built by
`fixture_shapes.discord_snowflake` as a real Discord snowflake, with the
message's own timestamp in the high bits (`(ms - 1420070400000) << 22`) and
the seeded hash in the low 22 bits. Discord ids must grow with time because
all three cycles replay the same demo channel and the collector reads it
incrementally with an `after` watermark; a purely hash-derived id would let
cycle 2's watermark silently suppress most of cycle 3's messages on replay.
Telegram message ids are monotone within a channel for the same reason
(`base_message_id + index`, consumed through `min_id`).

Subreddit, Telegram channel, and Discord guild/channel names are fixed demo
names derived from the report category and cycle index, not real community
names.

## Anonymization

All post text passes through `tools/anonymize.py::anonymize_text`, which
removes `@handles`, email addresses, phone numbers, and links to social
platforms (x.com, twitter.com, t.co, t.me, telegram.me, telegram.org,
discord.gg, discord.com, discordapp.com/.net, reddit.com, redd.it,
redditmedia.com, redditstatic.com, facebook.com, instagram.com, youtube.com,
youtu.be, linkedin.com, and the platform link shorteners lnkd.in / fb.me)
while preserving IoC-looking tokens (CVE ids, hashes, IPs, ISO
dates/timestamps, and defanged/real domains) so the anonymized text still
carries the same threat-intel signal and dates like
`2026-03-27 21:05:07.129051` are never mistaken for a phone number. The
phone-number stripper only removes digit runs that are both phone-shaped
(grouped digits, `+`/parens/single separators) and carry >=9 digits, and
social-link matching requires a real host boundary so e.g. `notx.com/path`
is left alone while `x.com/...` is removed.

**Any subdomain counts.** A platform serves post-derived media and
permalinks from its own subdomains, and a signed media URL such as
`https://preview.redd.it/<id>.jpg?...&s=<signature>` points back to one
specific submission just as precisely as its permalink does. Matching only
`www.` would leave such a URL in the bundle, so `preview.`, `i.`, `v.`,
`old.`, `cdn.` and every other subdomain are stripped on the same terms. These hosts are also excluded from the
publisher/vendor exemption below: `reddit.com`, `redd.it`, `x.com`, `t.me`
and `discord.com` appear in `tiblender_runtime.legitimate_hosts.REFERENCE_HOSTS`
because the IoC extractor must not mistake them for campaign
infrastructure, which is not a reason to let a post URL ship clickable.

Author/channel identities are replaced with
`tools/anonymize.py::pseudonym`, a stable, non-reversible
`"u" + sha256(...)[:8]` derived from the released `source_ref` -- never from
a real handle.

The released `source_posts.items[].text` still contains live URLs and some
`@handles`; the fixtures do not. This step is what makes the difference, and
`tests/test_anonymize.py` plus the bundle checks in
`tests/test_fixture_bundle.py` hold it in place.

## Defanging

`anonymize_text` also defangs every live indicator it leaves behind
(`tools/anonymize.py::defang_text`): `http(s)://host/...` becomes
`hxxp(s)://host[.]tld/...` and a bare domain becomes `host[.]tld`, for any
host that is not a known publisher/vendor/platform domain
(`tiblender_runtime.legitimate_hosts.REFERENCE_HOSTS`), including darknet (`.onion`)
leak-site addresses. These fixtures are excerpts of real posts, so they carry
real campaign infrastructure; no campaign indicator in the fixture post text
is clickable (publisher, vendor and platform links stay as posted). This is
behavior-neutral -- `iocs.extract_iocs` refangs before extracting, so the
pipeline sees exactly the same indicators, and `fixtures/tools/queries.json`
(hence every recording key) is byte-identical with and without defanging. See
`docs/FIDELITY.md` for why real indicators are bundled at all rather than
reserved documentation domains.

## Rebuilding

```
make build-fixtures
```

which is

```
python3 tools/build_fixtures.py \\
    --dataset dataset/raw_dataset/reports.jsonl \\
    --out fixtures/collectors \\
    --cycles 3 \\
    --seed 20260909
```

The path is relative to the repository root, where the released dataset
sits in `dataset/`. The build is deterministic given the
same dataset file and seed: no wall clock, no unseeded randomness.
`fixtures/collectors/manifest.json` records per-cycle counts, the selected
report ids, every post's `source_ref`, and the sha256 of the dataset file, so
a rebuild can be verified byte-for-byte. `fixtures/tools/queries.json` lists
every IoC found in the fixture posts as `{tool_type, parameters}` recording
targets for `tools/record_tool_responses.py`.

A rebuild changes which indicators the bundle carries, so the recorded tool
responses under `fixtures/tools/` have to be re-recorded for the new query
list; until then those queries replay as `status: "unavailable"`, which the
pipeline handles as an ordinary "no data" tool result.

## The second pass: harvested tool queries

The builder can only see fixture **post text**. The pipeline also issues
requests no static scan of that text can predict:

- free-text cluster search phrases (`"supply chain"`,
  `"edition.cnn.com microsoft"`, `"<md5> malware"`), which only exist once
  clustering has run, and
- domains, IPs and URLs pivoted out of *recorded evidence* rather than out of
  post text.

Without a second pass, those requests would find no recording and degrade to
`status: "unavailable"`. The offline three-cycle run is byte-deterministic,
so the set of requests it issues is a fixed, reproducible input, and it is
harvested rather than guessed:

```bash
make harvest-tool-queries       # no credentials, no network
```

runs the demo, reads each cycle's
`intermediate/tool_recording_misses.json` (written from the runtime's
`ToolRecordingReplayLog`), merges the missed keys into
`fixtures/tools/harvested_queries.json`, and re-runs this builder, which
unions the post-derived queries with the harvested ones into `queries.json`.
Then record the new keys:

```bash
python3 tools/record_tool_responses.py --fixtures fixtures \
    --queries fixtures/tools/queries.json --resume
```

Repeat until `make harvest-tool-queries` reports zero misses. `WEB_CRAWL`
still honours the allowlist below: a harvested URL on an unlisted host is
recorded as `unavailable` without ever being fetched.

`harvested_queries.json` is **committed**, so `make build-fixtures` stays
reproducible with no network — the builder reads that file, it never
re-derives it. Each cycle also reports its replay coverage in
`cycle_summary.json` (`tool_recording_replay`), which `verify-sample`
compares and the test suite holds to a floor.

Every entry in `queries.json` carries an `origin` field naming the pass it
came from, so the two are distinguishable without hand-diffing:

| `origin` | meaning |
| --- | --- |
| `post_derived` | the builder scanned it out of the anonymized fixture post text |
| `harvested` | the offline demo was observed to issue it (second pass, above) |

A request both passes produce is `post_derived`: a rebuild derives it with no
run at all. `origin` is **not** part of the recording key
(`tiblender_runtime/tool_fixtures.py::recording_key` hashes the tool type and
the named parameters only), so it moves no recording file and forces no
re-recording.

## WEB_CRAWL host allowlist

`tools/record_tool_responses.py::DEFAULT_ALLOWED_CRAWL_HOSTS` is the fixed
list of hosts a `WEB_CRAWL` recording may actually fetch during recording
(matched by exact host or subdomain suffix). Every `WEB_CRAWL` query in
`fixtures/tools/queries.json` whose URL host is *not* on this list is
recorded as `status: "unavailable"` with
`error: "crawl_skipped_untrusted_host: ..."`, and the host is never
contacted and never opened in a browser. This section is the per-host
rationale for that list.

**Base list** (`nvd.nist.gov`, `nist.gov`, `cve.org`, `mitre.org`,
`github.com`, `cisa.gov`, `msrc.microsoft.com`, `security.googleblog.com`,
`bleepingcomputer.com`, `thehackernews.com`, `securityweek.com`,
`helpnetsecurity.com`): primary standards bodies (NIST/NVD, CVE.org, MITRE),
a government CERT (CISA), vendor security-response centers (Microsoft's
MSRC, Google's security blog), and established security-news publications
with editorial vetting.

**Five additions**, made when scanning an earlier `queries.json` for
legitimate publisher/vendor hosts not already covered by the base list. They
are kept because the criteria, not the individual URLs, are what the list
encodes:

- `horizon3.ai` -- Horizon3.ai, an offensive-security research vendor;
  CVE writeups live on the vendor's own `attack-research/vulnerabilities/`
  blog section -- first-party vendor vulnerability disclosure, the same kind
  of source as MSRC or Google's security blog.
- `pcmag.com` -- PCMag, a long-established consumer/enterprise tech
  publication with a dedicated security section; regional editions such as
  `gr.pcmag.com` are covered by the suffix match.
- `safedep.io` -- SafeDep, a software-supply-chain-security vendor
  publishing its own advisories about malicious packages -- first-party
  vendor research, the same kind of source as `helpnetsecurity.com` or
  `securityweek.com`.
- `security-next.com` -- Security NEXT, a Japanese security-news
  publication, numbered news articles on the outlet's own domain.
- `undernews.fr` -- UnderNews, a French cybersecurity news publication.

**Eight further additions**, made when scanning the rebuilt `queries.json`
for hosts meeting the same criteria -- established news/security
publications and government CERTs -- that the list did not yet cover:

- `reuters.com` -- Reuters, an international wire service with editorial
  standards and bylined reporting.
- `cnn.com` -- CNN, an established international news organization;
  regional editions such as `edition.cnn.com` are covered by the suffix
  match.
- `apnews.com` -- the Associated Press, a non-profit wire service and one
  of the primary originators of the reporting other outlets syndicate.
- `bbc.com` -- the BBC, an established international news organization
  with published editorial guidelines.
- `darkreading.com` -- Dark Reading, a long-running security-news
  publication with editorial vetting, the same kind of source as
  `securityweek.com` or `bleepingcomputer.com`.
- `infosecurity-magazine.com` -- Infosecurity Magazine, an established
  information-security trade publication.
- `ncsc.gov.uk` -- the UK National Cyber Security Centre, a government
  CERT publishing first-party advisories, the same kind of source as
  `cisa.gov`.
- `ic3.gov` -- the FBI's Internet Crime Complaint Center, a US government
  body publishing first-party public-service announcements and alerts.

**Twenty further additions**, made when reviewing a rebuilt `queries.json`
for hosts skipped as untrusted that meet the same criteria -- established,
editorially-vetted news organizations, and first-party security-vendor
research and advisories:

- `nytimes.com`, `washingtonpost.com`, `wsj.com`, `theguardian.com`,
  `npr.org`, `pbs.org`, `newsweek.com` -- established international news
  organizations with editorial standards, the same category as `reuters.com`,
  `cnn.com`, `apnews.com` and `bbc.com`.
- `theregister.com` -- The Register, a long-running technology and security
  news publication with editorial vetting, the same kind of source as
  `bleepingcomputer.com` or `darkreading.com`.
- `technologyreview.com` -- MIT Technology Review, an established technology
  publication.
- `securityonline.info` -- an established security-news publication already
  vetted elsewhere in this codebase (`REFERENCE_HOSTS`) for the same
  first-party-reporting reason as `security-next.com`.
- `microsoft.com` -- Microsoft's main vendor domain, alongside the
  already-listed `msrc.microsoft.com` security-response subdomain.
- `crowdstrike.com`, `malwarebytes.com`, `elastic.co`, `esentire.com`,
  `infoblox.com`, `reversinglabs.com` -- established security vendors
  publishing first-party threat research and advisories on their own
  domains, the same category as `horizon3.ai` or `safedep.io`.
- `specterops.io`, `stepsecurity.io`, `0patch.com` -- smaller but established
  security-research vendors (offensive-security research, supply-chain
  security, and Windows micropatching respectively) publishing first-party
  advisories on their own domains, the same category as `horizon3.ai` or
  `safedep.io`.

**Categories deliberately excluded:**

- Link shorteners (`ow.ly`, `bit.ly`, `amzn.to`): the shortener host itself
  carries no editorial identity and hides the real destination until
  resolved, which this recorder never does -- fetching it would mean
  recording whatever page the shortener happens to redirect to at
  recording time, an unverifiable and unstable target.
- `.onion` and `.i2p` addresses: darknet hidden services are exactly the
  kind of untrusted, unverifiable host this allowlist exists to keep out --
  never fetched, never opened in a browser, regardless of what the linking
  post claims about them.
- Open publishing platforms (`medium.com`, `substack.com`, ...): anyone can
  publish under the shared host, so the host itself carries no vetting
  signal the way a named vendor or outlet does.
- Threat-intel aggregators and clustering services: not primary publishers;
  they link on to source reporting rather than being it.
- Anything with no established security-publisher or vendor reputation, and
  in particular brand-lookalike hostnames -- exactly the kind of domain this
  allowlist's exact/subdomain-suffix match is designed to reject rather than
  special-case.

Rebuild note: this list is a static Python tuple, not derived from
`queries.json`. If a fixture rebuild introduces a new `WEB_CRAWL` host,
verify it against the criteria above before adding it to
`DEFAULT_ALLOWED_CRAWL_HOSTS`, and extend this section with the reasoning.
Nothing breaks if that review does not happen: an unreviewed host is simply
never fetched.

"""
        + _crawl_host_section(tool_queries, DEFAULT_ALLOWED_CRAWL_HOSTS),
        encoding="utf-8",
    )


