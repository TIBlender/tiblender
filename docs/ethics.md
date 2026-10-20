# Ethics

The paper's ethics statement, restated in terms of what this repository
contains and what running it does. It covers both halves of the repository:
the code and the released dataset (`dataset/`, whose own README states the
same policy for the released records).

The README's Ethics section is the short summary; this document is the
statement behind it.

## What was collected, and from where

Collection was limited to **publicly accessible or semi-public communities**
on X, Reddit, Telegram and Discord — the four platforms and the single 31-day
window described in the paper, and nothing else. **No private messages and no
private servers were collected.** Nothing was collected from a community the
collecting account had not already joined, and no account, group or server
was joined in order to collect from it beyond the public scope described in
the paper.

The dataset is therefore not representative of all platforms, time periods,
languages or threat communities, and should not be presented as such.

## Anonymization

The released records and the bundled fixtures are excerpts of **publicly
posted content collected during the deployment**.

**Released dataset (`dataset/`).** Each source post keeps only its text,
platform and timestamp, plus a regenerated `source_ref`. Author identifiers
and other identifying metadata of the post (the author's name, handle and
account id, the post's own id and permalink, guild and channel ids) are
removed. The post text itself is kept as posted, so it can mention other
accounts (for example security organizations, researchers or threat
actors), email addresses (often the sender addresses of phishing
campaigns) and links to other posts. The generated reports cite such
accounts and addresses where they are part of the threat evidence.

**Bundled fixtures (`fixtures/`).** The fixture post text is anonymized
further: `@handles`, email addresses, phone numbers and links to social
platforms are removed, campaign indicators are defanged, and post, author,
guild and channel identifiers are regenerated deterministically. The method
and its guarantees are documented per field in
[`../fixtures/README.md`](../fixtures/README.md) (`tools/anonymize.py`,
`tools/fixture_shapes.py`, tested in `tests/test_anonymize.py`).

**Neither the dataset nor the code may be used to re-identify the authors
of the collected posts or the people mentioned in them.** Removing author
identifiers is designed to prevent it, and this statement forbids it.

## Live indicators

The repository carries the campaigns' own indicators, so the recorded
provider answers describe the indicators the posts actually name — and **the
repository is not uniformly defanged**.

Some indicators inside collected post text are defanged (`hxxp://host[.]tld`),
as the posters themselves wrote them, and the pipeline refangs those before
extraction (`tiblender_runtime/iocs.py::refang`) so extraction is unaffected.
Three places therefore necessarily carry the plain, clickable form: the
recorded tool responses (a WHOIS / passive-DNS / CT query *is* the plain
domain); the generated reports and their Markdown/PDF/STIX sidecars together
with the committed `sample_runs/offline_three_cycles/expected/summary.json`,
which are the pipeline's own **refanged, machine-readable extracted IoC
values**; and part of the collected post text itself, where vendor, news and
write-up links were collected as posted. Every one of these is an indicator
of real, possibly malicious infrastructure — **do not open them.**

These indicator lists are **live research data, not a curated blocklist**.
`dataset/README.md` states the same for the released records.

## Platform terms and account credentials

Live collection uses the platforms' own interfaces with the operator's own
credentials. Live Telegram collection uses an MTProto **user** session and
live Discord collection prefers a **user-account** token, because neither
platform's bot API can read the channel history the paper analyses; both read
only communities the account has already joined. Using a user-account
credential for automated reading may conflict with a platform's terms of
service, and anyone enabling live mode is responsible for their own
compliance with those terms and with local law.

**The repository ships no token, session or credential of any kind.** The
offline demo, the evaluation script and the test suite all run without live
mode, from bundled evidence.

## Privacy

The offline demo performs no analytics, telemetry or tracking. It makes no
outbound network call once built, and the static viewers load no external
script, font or resource. Nothing in this repository can reveal a user's
identity.

## No offensive capability

The repository contains no exploit, malware or attack code. Its
security-sensitive surface is limited to reading public posts and querying
threat-intelligence lookup services.

## Intended use

This repository is intended for research on cross-platform threat
intelligence and for reproducing the paper's results. The deterministic
demonstration runs without external credentials; live operation depends on third-party platforms, external tools, LLM
providers, network access and user-provided credentials. No destructive
action is expected of either path.

## Licence

Licence: the code is released under the **TIBlender Research and Evaluation
License** (`LICENSE` at the repository root), which permits use,
modification, and sharing for non-commercial research only, and grants no
patent rights; the dataset is released under **CC BY-NC 4.0**
(`dataset/LICENSE`, <https://creativecommons.org/licenses/by-nc/4.0/>), with
attribution by citing the ACSAC 2026 paper. `dataset/README.md` states the
same pair for the data side.
