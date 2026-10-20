# sfrclak.com malware and command-and-control campaign

**Report ID**: TIBLENDER-RUN-252a5d11fa1a
**Generated At**: 2026-09-16T08:49:00Z
**Severity**: HIGH
**Category**: malware

## Executive Summary

6 post(s) from discord, reddit and x posted on 2026-03-31 were consolidated into one malware and command-and-control cluster across 3 platform(s), rated CRITICAL. The cluster carries 1 domain, 2 file hashes. Matched threat vocabulary: malware, payload, npm. This summary is composed from the cluster's indicators, counts and vocabulary matches; the source posts are quoted under source evidence in the language they were posted in.

## Key Findings

- **Infrastructure**
  - Infrastructure focus selection used the heuristic planner (no LLM agent configured).
  - Infrastructure agent investigated domain_registration_and_whois, dns_and_passive_dns_analysis, ssl_tls_certificate_analysis with 5 tool query/queries (5 successful).
  - Infrastructure agent pivoted one level into 2 newly discovered indicator query/queries.
- **Technical**
  - Technical focus selection used the heuristic planner (no LLM agent configured).
  - Technical agent investigated technical_intelligence_and_security_report_collection, cve_exploitation_analysis with 2 tool query/queries (2 successful).
- **Social**
  - Social focus selection used the heuristic planner (no LLM agent configured).
  - Social agent investigated sns_account_abuse_and_spread_tracking, social_engineering_tactics_analysis with 4 tool query/queries (4 successful).
- **Actor**
  - Actor focus selection used the heuristic planner (no LLM agent configured).
  - Actor agent investigated apt_group_and_threat_actor_investigation, cybercrime_group_and_market_analysis, campaign_tracking with 5 tool query/queries (5 successful).
- **Indicators**
  - Extracted from the cluster's posts: 1 domain, 2 file hashes.
  - Severity CRITICAL across 3 platform(s).

## IoCs

| Type | Value | Extraction Type | Source Platforms |
|------|-------|-----------------|------------------|
| domain | `sfrclak.com` | Direct | discord, reddit, x |
| ip | `142.11.206.73` | Pivot | - |
| hash | `92ff08773995ebc8d55ec4b8e1a225d0d1e51efa4ef88b8849d0071230c9645a` | Direct | reddit, x |
| hash | `f7d335205b8d7b20208fb3ef93ee6dc817905dc3ae0c10a0b164f4e7d07121cd` | Direct | reddit, x |

## CVEs

| CVE ID | Exploited In Wild |
|--------|-------------------|
| none | - |

## TTPs

| MITRE ID | Tactic | Technique | Evidence |
|----------|--------|-----------|----------|
| `T1105` | Command and Control | Ingress Tool Transfer | Payload/C2 context appears in the cluster. |

## Threat Actor

Unknown

## Attack Timeline

| Timestamp | Event | Significance |
|-----------|-------|--------------|
| 2026-03-31T03:20:27Z | x source post sample-x-axios-0001 | Primary source evidence used for clustering and report generation. |
| 2026-03-31T04:22:49Z | x source post sample-x-axios-0002 | Primary source evidence used for clustering and report generation. |
| 2026-03-31T04:52:56Z | x source post sample-x-axios-0003 | Primary source evidence used for clustering and report generation. |
| 2026-03-31T07:54:11Z | discord source post sample-discord-axios-0001 | Primary source evidence used for clustering and report generation. |
| 2026-03-31T13:53:00Z | reddit source post sample-reddit-axios-0001 | Primary source evidence used for clustering and report generation. |
| 2026-03-31T14:56:44Z | reddit source post sample-reddit-axios-0002 | Primary source evidence used for clustering and report generation. |

## Recommendations

### Immediate

1. **Block or monitor the listed domains, URLs and IP addresses according to local policy.** The cluster carries network indicators that can be actioned directly.
2. **Add the listed file hashes to endpoint blocklists and hunt for prior executions.** File hashes in the cluster identify the delivered sample.
3. **Search SIEM and EDR telemetry for the listed indicators and the associated campaign terms.** Retrospective search establishes whether the campaign already reached the estate.

### Short-Term

1. **Preserve the source evidence and re-run enrichment if new platform posts appear in the next cycle.** Evidence is credible but the cluster is still developing.


## Source Evidence

- `sample-discord-axios-0001` (discord, 2026-03-31T07:54:11Z): Malware signal: Discord incident channel is treating the Axios npm compromise as one campaign across x, reddit, and discord evidence. The affected package versions, sfrclak[.]com payload host, and payload hashes are linked to the same malicious postinstall activity. IoCs: sfrclak.com
- `sample-reddit-axios-0001` (reddit, 2026-03-31T13:53:00Z): Malware signal: Reddit triage thread for the Axios npm compromise confirms this is not a simple typosquat. The same sfrclak[.]com payload host, malicious postinstall behavior, and affected versions axios@1.14.1 / axios@0.30.4 are listed as one supply-chain campaign. IoCs: sfrclak.com
- `sample-reddit-axios-0002` (reddit, 2026-03-31T14:56:44Z): Other signal: Axios supply-chain response notes: audit CI runners and developer workstations that installed axios@1.14.1, axios@0.30.4, or plain-crypto-js@4.2.1. Known indicators include sfrclak[.]com, f7d335205b8d7b20208fb3ef93ee6dc817905dc3ae0c10a0b164f4e7d07121cd, and 92ff08773995ebc8d55ec4. IoCs: 92ff08773995ebc8d55ec4b8e1a225d0d1e51efa4ef88b8849d0071230c9645a f7d335205b8d7b20208fb3ef93ee6dc817905dc3ae0c10a0b164f4e7d07121cd sfrclak.com
- `sample-x-axios-0001` (x, 2026-03-31T03:20:27Z): Malware signal: Axios npm supply-chain compromise: affected packages include axios@1.14.1, axios@0.30.4, and plain-crypto-js@4.2.1. The active payload host sfrclak[.]com is being shared for CI and developer workstation hunting. IoCs: sfrclak.com
- `sample-x-axios-0002` (x, 2026-03-31T04:22:49Z): Malware signal: Additional Axios campaign triage links the same malicious npm postinstall flow to sfrclak[.]com. Reported payload hashes are f7d335205b8d7b20208fb3ef93ee6dc817905dc3ae0c10a0b164f4e7d07121cd and 92ff08773995ebc8d55ec4b8e1a225d0d1e51efa4ef88b8849d0071230c9645a. IoCs: 92ff08773995ebc8d55ec4b8e1a225d0d1e51efa4ef88b8849d0071230c9645a f7d335205b8d7b20208fb3ef93ee6dc817905dc3ae0c10a0b164f4e7d07121cd sfrclak.com
- `sample-x-axios-0003` (x, 2026-03-31T04:52:56Z): Malware signal: Axios incident update: the malicious npm postinstall path contacts sfrclak[.]com after users install affected package versions. Hunt for axios@1.14.1, axios@0.30.4, plain-crypto-js@4.2.1, and the two payload hashes already circulating in the thread. IoCs: sfrclak.com