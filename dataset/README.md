# TIBlender Dataset

This directory contains the report dataset from the 31-day TIBlender deployment
(March 16 to April 16, 2026; X, Reddit, Telegram, Discord), the evaluation evidence behind the
paper's tables, a notebook that recomputes those tables from the evidence, and a static browser
for reading the reports.

## Layout

- `raw_dataset/reports.jsonl` — one JSON object per report (8,288 lines).
- `raw_dataset/evaluation_evidence/` — per-record evaluation evidence (table below).
- `raw_dataset/manifest.json` — file hashes and the field list.
- `notebook/evaluation_figures.ipynb` — recomputes every evaluation figure from `raw_dataset/`;
  `evaluation_figures_executed.{ipynb,html}` are the executed copies and `evaluation_figures.json` holds the values.
- `viewer/index.html` — static report browser; open it directly or serve the directory
  (`python3 -m http.server 8765 --directory viewer`). Data is bundled in `viewer/reports_viewer_data.js`.

## Report records

| Field | Meaning |
| --- | --- |
| `report_id`, `report_index` | Stable identifier and export order. |
| `created_at` | Report generation timestamp; `null` when no timestamp is available for the report. |
| `paper_category` | `Vulnerability`, `Phishing/Fraud`, `APT/Ransomware`, `Malware/C2`. |
| `severity` | `CRITICAL`, `HIGH` or `MEDIUM` (same as `report_text.severity`). Only clusters at MEDIUM or above reach report generation. CRITICAL was assigned during the deployment to confirmed broad exploitation: a CISA-KEV-listed CVE with active exploitation reported in a HIGH report, or two or more indicators with 11+ VirusTotal malicious detections. The VirusTotal counts that fed that rule are reputation observations made at run time and are not part of the released record fields (see "IoC reputation observations" below). |
| `platforms` | Platforms whose posts support the report: `x`, `reddit`, `telegram`, `discord`. |
| `report_text` | The Step 5 output schema with 11 sections: `cluster_name`, `severity`, `summary`, `key_findings[{heading,items}]`, `recommendations[{action,rationale,horizon}]`, `iocs[{type,value,extraction_type}]`, `cves`, `ttps`, `threat_actor`, `timeline[{timestamp,event,significance}]`, `sources[{source_number,source_ref,summary}]`. |
| `iocs.items[]` | `type`, `value`, `extraction_type` (`Direct`/`Pivot`), `platforms` (the platforms whose posts the indicator was attributed to). |
| `source_posts.items[]` | `source_ref`, `platform`, `timestamp`, `text` (post text in its original language). |

`recommendations[].horizon` is `immediate`, `short_term` or `long_term`. It is present on
66,723 of the 67,264 recommendations; 376 reports carry at least one recommendation with no
`horizon`, and consumers should treat the field as optional.

`iocs.items` is the extracted indicator record and the list every figure in the notebook is
computed from: 17,404 rows across the 8,288 reports, each carrying `platforms`.
`report_text.iocs` is the same indicator list as it appears in the report body, holding `type`,
`value` and `extraction_type` only. The two views list the same indicators in every one of the
8,288 reports (verified: 17,404 rows on both sides, with no report where one view holds an
indicator the other does not); `iocs.items` adds the `platforms` attribution. Use `iocs.items`
when the platform attribution is needed.

### IoC reputation observations

The released report records do **not** carry per-IoC VirusTotal detection counts or per-domain
Tranco membership. Those observations live only in the evaluation evidence, one row per
indicator, in `raw_dataset/evaluation_evidence/same_input_direct_records.jsonl` and
`baseline_ioc_records.jsonl` (`vt_malicious`, `tranco_top1m`), which is the population the
paper's IoC-quality tables are computed from. The notebook's population-check section rebuilds
the TIBlender (D) indicator set from `reports.jsonl` alone and confirms it matches those
evidence files indicator-for-indicator, so the two views describe the same indicators.

## Evaluation evidence

| File | Unit | Used for |
| --- | --- | --- |
| `same_input_direct_records.jsonl` | one row per unique Direct IoC and platform (Phishing/Fraud, Malware/C2) | TIBlender (D) rows of the same-input comparison: counts, VirusTotal distribution, Tranco match |
| `baseline_ioc_records.jsonl` | one row per IoC extracted by a baseline system on the same input (Tweezers, DarkGram, Reddit2CTI, iocsearcher) | baseline rows, novelty and coverage |
| `public_feed_overlap_records.jsonl` | one row per feed-scoped IoC/CVE (PhishTank: Phishing/Fraud domains and URLs; URLhaus: Malware/C2 domains and URLs; MalwareBazaar: Malware/C2 hashes; CISA KEV: Vulnerability CVEs; TweetFeed: all domains, URLs, IPs and hashes; OTX Pulses: all types including CVEs) | feed denominators, overlap, feed-scoped absence |
| `public_feed_timing_records.jsonl` | one row per overlapping IoC/CVE | TIBlender detection time vs feed listing time: early-detection rate and lead time |
| `pipeline_overview.json` | deployment totals | raw posts per platform, clusters, Step 2 / Step 4 filtering, runtime |
| `ablation_membership.jsonl` + `ablation_parameters.json` | one row per report and ablation condition | reports lost, affected or unchanged under A1–A5 and the yield parameters |
| `skip_cluster_units.jsonl` | one row per Step 4 SKIP cluster (3,269) | low-quality clusters stopped by the adaptive gate; the A2 counterfactual units and the population of the SKIP-cluster review sample |
| `manual_review_records.jsonl` + `manual_review_patterns.json` | one row per sampled report or cluster | quality audit (500 feed-overlap + 500 feed-absent reports), false-negative reviews (1,000 LOW clusters, 1,000 Step 4 SKIP clusters) |

## What the notebook computes

Operational overview; report, category and severity counts; cross-platform reports and rate; IoCs per
report; platform contribution and single-platform removal loss; same-input comparison with baselines
(totals, per-category counts, novelty, coverage, and — from the evidence files — VirusTotal
detection distribution and Tranco match); the TIBlender (D) population check that rebuilds the
same-input indicator set from `reports.jsonl` alone;
public-feed denominators, overlap, feed-scoped absence, early-detection rate and mean lead time;
ablation membership counts and parameters; manual quality-audit and false-negative review tables.

## Expected values

`notebook/evaluation_figures.json` holds the values this dataset produces:
executing `notebook/evaluation_figures.ipynb` must regenerate that file
byte-identically. The headline numbers it reproduces are below; every other
value in the notebook's output is in the same JSON.

Deployment and reports (`overview`, `reports`):

| Value | Expected |
| --- | --- |
| Raw posts collected | 873,973 (Telegram 579,300; Discord 217,249; X 48,299; Reddit 29,125) |
| Unique clusters | 184,240 (Step 2 LOW-severity filtered: 172,683; Step 4 SKIP: 3,269) |
| Actionable reports | 8,288 |
| Categories | Vulnerability 4,194; Phishing/Fraud 1,817; APT/Ransomware 1,420; Malware/C2 857 |
| Severity | MEDIUM 5,626; HIGH 1,768; CRITICAL 894 |
| Cross-platform reports | 2,003 (24.2 %) |
| IoCs | 17,404 total; 2.1 per report |

Same-input comparison (`same_input`, Coverage of each baseline by TIBlender
Direct IoCs): X — Tweezers 28.2 %, iocsearcher 73.9 %; Telegram — DarkGram
14.3 %, iocsearcher 45.3 %; Reddit — Reddit2CTI 5.1 %, iocsearcher 82.6 %;
Discord — iocsearcher 64.5 %.

Population check (`same_input_population_check`): the TIBlender (D) indicator
set rebuilt from `reports.jsonl` matches the same-input evidence exactly on
every platform — X 866, Reddit 140, Telegram 72, Discord 51 indicators, with 0
reports-only and 0 evidence-only indicators in each case. Indicators appearing
in both a Phishing/Fraud and a Malware/C2 report (X 15, Reddit 3, Telegram 2,
Discord 5) are counted once, under the category recorded in the evidence file.

Public feeds (`public_feeds`); "Ahead" is the early-detection rate among
overlapping indicators and "Mean lead" is signed as described in the next
section:

| Feed | Denominator | Overlap | FSA | Ahead | Mean lead (h) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PhishTank | 4,914 | 227 | 95.4 % | 71 (31.3 %) | −72.0 |
| URLhaus | 787 | 3 | 99.6 % | 1 (33.3 %) | −377.0 |
| MalwareBazaar | 53 | 9 | 83.0 % | 8 (88.9 %) | −24.0 |
| CISA KEV | 301 | 45 | 85.0 % | 8 (17.8 %) | −94.0 |
| TweetFeed | 14,935 | 324 | 97.8 % | 52 (16.0 %) | −157.2 |
| OTX Pulses | 15,627 | 518 | 96.7 % | 71 (13.7 %) | −108.6 |
| **All six** | — | **1,126** | — | **211 (18.7 %)** | — |

## Sign convention for lead time

Lead time is signed as in the paper: `lead_hours = detection_time −
feed_listing_time`, so a **negative value means TIBlender detected the
indicator first**. `-72.0 h` for PhishTank means TIBlender was 72 hours ahead
of PhishTank's listing, not 72 hours behind it. The paper reports the same
quantity as an absolute lead ("a mean lead time of 72 hours").

Each row of `raw_dataset/evaluation_evidence/public_feed_timing_records.jsonl`
carries both timestamps (`detection_time`, `feed_listing_time`), the boolean
`detected_before_feed`, and `lead_hours`, so the convention can be checked
directly on any record.

## KEV report reconciliation

The notebook's `record_consistency` section checks the evaluation date range, the 301
Vulnerability-category CVEs, 1,357 reports referencing KEV-listed CVEs, 95 reports
published before the corresponding listing, and the per-CVE early-detection counts.
The maximum lead is 13.4 days. CVE-2026-35616 has first detection on April 5 and
KEV listing on April 6, consistent with the midnight-UTC listing convention and
0.4-day rounded lead; the appendix's April 6 first-detection date is inconsistent
with that convention.

OTX campaign-level pairing is not computed. Ablation parameters read from
`ablation_parameters.json` are stored aggregate values, not independently
reconstructed from per-investigation evidence.

## Anonymization, ethics and licence

These records are excerpts of publicly posted social media content collected
during the 31-day deployment. Each source post keeps only its text, platform
and timestamp, plus a regenerated `source_ref`; author identifiers and other
identifying metadata of the post (the author's name, handle and account id,
the post's own id and permalink) are removed. The post text is kept as
posted, so it can mention other accounts (for example security
organizations, researchers or threat actors), email addresses (often the
sender addresses of phishing campaigns) and links to other posts, and the
reports cite such accounts and addresses where they are part of the threat
evidence. Content hashes are retained so that records stay joinable across
the evidence files. The code's bundled fixtures are anonymized further, as
documented in `../fixtures/README.md`.

**This dataset must not be used to re-identify the authors of the collected
posts or the people mentioned in them.** Removing author identifiers is
designed to prevent it, and the ethics statement forbids it.

The indicators in these records are **live research data, not a curated
blocklist**, and are not uniformly defanged: treat every indicator as live
and do not open it.

Licence: this dataset (`raw_dataset/`, `notebook/`, `viewer/`) is released
under the **Creative Commons Attribution-NonCommercial 4.0 International
License (CC BY-NC 4.0)**, <https://creativecommons.org/licenses/by-nc/4.0/>;
the notice ships as `LICENSE` in this directory. Attribution: cite the ACSAC
2026 paper. The accompanying code at the repository root is licensed
separately, under the **TIBlender Research and Evaluation License**
(`../LICENSE`).

## File checks

```bash
jq empty raw_dataset/manifest.json raw_dataset/evaluation_evidence/*.json
jq -e empty raw_dataset/reports.jsonl raw_dataset/evaluation_evidence/*.jsonl && echo "dataset json ok"
```

Without `jq` (the `.jsonl` files must be parsed line by line; checking `*.json`
alone would leave two thirds of the evidence unread):

```bash
python3 -c "import json,pathlib; [json.loads(l) for p in pathlib.Path('raw_dataset').rglob('*.jsonl') for l in p.open() if l.strip()]; [json.loads(p.read_text()) for p in pathlib.Path('raw_dataset').rglob('*.json')]; print('dataset json ok')"
```
