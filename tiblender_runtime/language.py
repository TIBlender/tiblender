"""English-only report prose: the guard, and the deterministic composer.

The pipeline collects posts in every language the four platforms carry
(Japanese, Korean, Russian, Chinese, Arabic, ...). The *report* it publishes
is English whichever path produced it, so that one report can be read without
a translator no matter which platform the evidence came from.

Two mechanisms enforce that, and both live here:

* :func:`text_is_english` -- the guard. It strips the tokens that are
  language-neutral by nature (URLs, defanged hosts, IP addresses, file names,
  CVE ids, hashes) and then requires the remainder to carry no character from
  a non-Latin script and to be overwhelmingly Latin-lettered. A malware
  sample's own Russian file name is an indicator value, not prose, so it
  passes; a Russian *sentence* does not.

* :func:`enforce_english` -- the composer. Any ``report_schema`` section that
  fails the guard is replaced with prose this module composes from the
  cluster's language-neutral facts: the category vocabulary, the platform and
  post counts, the extracted indicators, CVEs and brands, and the English
  keyword matches. It never quotes a collected post.

Both paths run through the composer. The offline stand-in composes its report
fields here from the start; the LLM path is asked for English, retried once
when it answers in another language, and then falls back to the same
composer (see ``llm_agents.compose_report_fields``). A report the composer
had to rewrite carries ``metadata.language_fallback = true`` and the section
names under ``metadata.language_fallback_sections``; ``cycle_summary.json``
counts them under ``language_guard``.

**What stays in its original language.** Source evidence is quoted, not
composed. The report root's ``source_posts.items[].text``,
``sources.social_media_posts[].text`` and ``platform_unique_contributions``
carry the anonymized post body exactly as collected, which is also what the
released dataset does with ``source_posts.items[].text``. Every
*summary* of that evidence -- including ``report_text.sources[].summary`` --
is English.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .models import Cluster, PerspectiveFinding

#: Scripts a published report never writes prose in. Indicator values that
#: happen to use one of them are excluded before the scan (see
#: ``_ALLOWED_TOKEN_RE``).
_NON_LATIN_SCRIPT_RE = re.compile(
    "["
    # Written as escapes so this file is itself pure ASCII (the package's
    # English-only scan reads source files, not just documentation).
    "\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9d"   # hiragana, katakana, halfwidth katakana
    "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"   # han
    "\u1100-\u11ff\u3130-\u318f\uac00-\ud7af"   # hangul
    "\u0400-\u052f\u2de0-\u2dff\ua640-\ua69f"   # cyrillic
    "\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff"  # arabic
    "\u0e00-\u0e7f"                                # thai
    "\u0900-\u097f"                                # devanagari
    "]"
)

#: Tokens that carry no language: they are values, not words. A token is
#: compared after its surrounding quotes and sentence punctuation are
#: stripped.
_ALLOWED_TOKEN_RE = re.compile(
    r"""^(?:
          h[tx]{2}ps?://\S+                      # URL, fanged or defanged
        | CVE-\d{4}-\d{4,8}                      # CVE id
        | [A-Fa-f0-9]{32,128}                    # md5 / sha1 / sha256 ...
        | \d{1,3}(?:\[?\.\]?\d{1,3}){3}          # IPv4, fanged or defanged
        | \S*\[\.\]\S*                           # defanged host
        | \S*[^\s.]\.[A-Za-z0-9]{1,12}           # host or file name with an extension
        | [@#]\S+                                # handle or hashtag
    )$""",
    re.VERBOSE | re.IGNORECASE,
)

_TOKEN_TRIM = "'\"`()[]{}<>,;:!?*_~" + "\u201c\u201d\u2018\u2019\u00ab\u00bb\u2026"

#: Of the letters left after the value tokens are removed, at least this
#: share must be Latin. The script scan above already rejects the seven
#: scripts the corpus actually carries; the ratio is the backstop for every
#: other alphabet (Greek, Hebrew, Armenian, ...).
MIN_LATIN_RATIO = 0.9

#: ``report_schema`` sections the guard checks. ``severity``, ``iocs``,
#: ``cves`` and the source posts' own ``text`` are excluded: the first three
#: are enumerations and indicator values, and the fourth is quoted evidence
#: that keeps the language it was posted in.
GUARDED_SECTIONS: tuple[str, ...] = (
    "cluster_name",
    "summary",
    "key_findings",
    "recommendations",
    "ttps",
    "threat_actor",
    "timeline",
    "sources",
)

#: Prose keys of a Step 5 field draft, used to guard the LLM's answer before
#: the report is assembled.
GUARDED_REPORT_FIELDS: tuple[str, ...] = (
    "cluster_name",
    "executive_summary",
    "analyst_notes",
    "threat_actor",
    "immediate_actions",
    "recommendations",
    "key_findings",
    "timeline",
    "ttps",
)

#: Per-section keys whose values are identifiers or enumerations rather than
#: prose, so the guard skips them.
_NON_PROSE_KEYS = frozenset({"timestamp", "horizon", "mitre_id", "source_number", "source_ref", "id", "platform", "text"})

_BRANDS = ("Adobe Reader", "Adobe", "PayPay", "Microsoft", "Langflow", "Checkmarx", "CPUID")

_CATEGORY_LABELS = {
    "phishing": "phishing",
    "fraud": "fraud",
    "malware": "malware and command-and-control",
    "ransomware": "ransomware",
    "apt": "state-aligned intrusion",
    "vulnerability": "vulnerability exploitation",
    "supply_chain": "supply-chain compromise",
    "other": "threat",
}

#: English vocabulary the composer reports as "matched" when it appears in a
#: cluster's evidence. Every term is English by construction, so quoting the
#: match cannot carry another language into the report.
_ENGLISH_KEYWORDS = (
    "phishing", "smishing", "credential", "fake login", "drainer", "scam", "fraud",
    "wallet", "seed phrase", "impersonation", "malware", "c2", "payload", "dropper",
    "backdoor", "botnet", "trojan", "loader", "stealer", "ransomware", "extortion",
    "leak site", "zero-day", "exploit", "remote code execution", "rce", "vulnerability",
    "patch", "supply chain", "dependency", "npm", "pypi", "nation-state",
    "state-sponsored", "apt", "takeover", "breach", "exfiltration",
)

_IOC_TYPE_LABELS = (
    ("domains", "domain", "domains"),
    ("urls", "URL", "URLs"),
    ("ips", "IPv4 address", "IPv4 addresses"),
    ("hashes", "file hash", "file hashes"),
    ("cves", "CVE", "CVEs"),
)


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def _is_latin_letter(char: str) -> bool:
    try:
        return unicodedata.name(char).startswith("LATIN")
    except ValueError:
        return False


def strip_value_tokens(text: str) -> str:
    """``text`` with every language-neutral value token removed."""
    kept: list[str] = []
    for token in text.split():
        trimmed = token.strip(_TOKEN_TRIM)
        if trimmed and _ALLOWED_TOKEN_RE.match(trimmed):
            continue
        kept.append(token)
    return " ".join(kept)


def text_is_english(value: Any) -> bool:
    """Whether one string reads as English prose.

    Empty and value-only strings pass: there is nothing to translate. A
    string fails when a non-Latin script survives the value-token strip, or
    when fewer than :data:`MIN_LATIN_RATIO` of its remaining letters are
    Latin.
    """
    text = str(value or "")
    if not text.strip():
        return True
    stripped = strip_value_tokens(text)
    if _NON_LATIN_SCRIPT_RE.search(stripped):
        return False
    letters = [char for char in stripped if char.isalpha()]
    if not letters:
        return True
    latin = sum(1 for char in letters if _is_latin_letter(char))
    return latin / len(letters) >= MIN_LATIN_RATIO


def _prose_strings(value: Any, *, key: str = "") -> list[str]:
    if key in _NON_PROSE_KEYS:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for name, item in value.items() for text in _prose_strings(item, key=str(name))]
    if isinstance(value, list):
        return [text for item in value for text in _prose_strings(item, key=key)]
    return []


def section_is_english(value: Any) -> bool:
    """Whether every prose string inside one report section is English."""
    return all(text_is_english(text) for text in _prose_strings(value))


def failing_sections(report_schema: dict[str, Any]) -> list[str]:
    """The :data:`GUARDED_SECTIONS` of ``report_schema`` that are not English."""
    return [
        section
        for section in GUARDED_SECTIONS
        if section in report_schema and not section_is_english(report_schema[section])
    ]


def failing_report_fields(fields: dict[str, Any]) -> list[str]:
    """The :data:`GUARDED_REPORT_FIELDS` of a Step 5 draft that are not English."""
    return [
        name
        for name in GUARDED_REPORT_FIELDS
        if name in fields and not section_is_english(fields[name])
    ]


# ---------------------------------------------------------------------------
# Language-neutral facts about one cluster
# ---------------------------------------------------------------------------


def _category_label(cluster: Cluster) -> str:
    return _CATEGORY_LABELS.get(cluster.threat_category.value, "threat")


def _evidence_text(cluster: Cluster) -> str:
    parts = [post.normalized_content for post in cluster.posts]
    parts.extend(post.post.content for post in cluster.posts)
    return " ".join(part for part in parts if part).lower()


def english_keyword_matches(cluster: Cluster, *, limit: int = 6) -> list[str]:
    """English catalogue terms present in the cluster's evidence.

    Only the English vocabulary is reported, so the match itself is English
    even when the post that carried it was not.
    """
    evidence = _evidence_text(cluster)
    return [term for term in _ENGLISH_KEYWORDS if term in evidence][:limit]


def brand_matches(cluster: Cluster, *, limit: int = 3) -> list[str]:
    evidence = _evidence_text(cluster)
    matched: list[str] = []
    for brand in _BRANDS:
        if brand.lower() in evidence and not any(brand in seen for seen in matched):
            matched.append(brand)
    return matched[:limit]


def _indicator_counts(cluster: Cluster) -> list[str]:
    phrases: list[str] = []
    for attribute, singular, plural in _IOC_TYPE_LABELS:
        count = len(set(getattr(cluster.iocs, attribute, []) or []))
        if count:
            phrases.append(f"{count} {singular if count == 1 else plural}")
    return phrases


def _platform_phrase(cluster: Cluster) -> str:
    platforms = sorted({post.platform for post in cluster.posts if post.platform})
    if not platforms:
        return "an unrecorded platform"
    if len(platforms) == 1:
        return platforms[0]
    return ", ".join(platforms[:-1]) + f" and {platforms[-1]}"


def _post_dates(cluster: Cluster) -> tuple[str, str]:
    dates = sorted({str(post.post.timestamp)[:10] for post in cluster.posts if post.post.timestamp})
    if not dates:
        return "", ""
    return dates[0], dates[-1]


# ---------------------------------------------------------------------------
# The deterministic English composer
# ---------------------------------------------------------------------------


def english_cluster_name(cluster: Cluster) -> str:
    """A cluster name built from the category and its leading indicator."""
    category = _category_label(cluster).title()
    if cluster.iocs.cves:
        return f"{sorted(cluster.iocs.cves)[0]} exploitation activity"
    brands = brand_matches(cluster, limit=1)
    if brands:
        return f"{brands[0]} {_category_label(cluster)} activity"
    if cluster.iocs.domains:
        return f"{sorted(cluster.iocs.domains)[0]} {_category_label(cluster)} campaign"
    if cluster.iocs.ips:
        return f"{sorted(cluster.iocs.ips)[0]} {_category_label(cluster)} infrastructure"
    return f"{category} cluster"


def english_summary(cluster: Cluster) -> str:
    """One English paragraph composed from the cluster's facts alone."""
    post_count = len(cluster.posts)
    platform_count = len({post.platform for post in cluster.posts if post.platform})
    first, last = _post_dates(cluster)
    window = f" posted between {first} and {last}" if first and last and first != last else (
        f" posted on {first}" if first else ""
    )
    sentences = [
        f"{post_count} post(s) from {_platform_phrase(cluster)}{window} were consolidated into one "
        f"{_category_label(cluster)} cluster across {platform_count} platform(s), rated "
        f"{cluster.severity.value}."
    ]
    indicators = _indicator_counts(cluster)
    if indicators:
        sentences.append("The cluster carries " + ", ".join(indicators) + ".")
    else:
        sentences.append("No reportable indicator was extracted from the cluster's posts.")
    cves = sorted(set(cluster.iocs.cves))
    if cves:
        sentences.append("Referenced CVE(s): " + ", ".join(cves[:6]) + ".")
    brands = brand_matches(cluster)
    if brands:
        sentences.append("Named product(s): " + ", ".join(brands) + ".")
    keywords = english_keyword_matches(cluster)
    if keywords:
        sentences.append("Matched threat vocabulary: " + ", ".join(keywords) + ".")
    sentences.append(
        "This summary is composed from the cluster's indicators, counts and vocabulary matches; "
        "the source posts are quoted under source evidence in the language they were posted in."
    )
    return " ".join(sentences)


def english_key_findings(cluster: Cluster, findings: list[PerspectiveFinding]) -> list[dict[str, Any]]:
    """Key findings composed from perspective metadata and indicator counts."""
    groups: list[dict[str, Any]] = []
    for finding in findings:
        successes = sum(1 for result in finding.tool_results if result.status == "success")
        items = [insight for insight in finding.insights if text_is_english(insight)]
        # The perspective's own insight lines already say what it ran; add the
        # composed line only when none of them survived the guard.
        if not any(item.startswith(f"{finding.perspective} agent investigated") for item in items):
            items.append(
                f"{finding.perspective} agent ran {len(finding.queries)} tool query/queries "
                f"({successes} with a usable answer) over {', '.join(finding.focus_types) or 'no focus type'}."
            )
        groups.append({"heading": finding.perspective, "items": items[:5]})
    indicators = _indicator_counts(cluster)
    groups.append(
        {
            "heading": "Indicators",
            "items": [
                "Extracted from the cluster's posts: " + (", ".join(indicators) or "no reportable indicator") + ".",
                f"Severity {cluster.severity.value} across {len({post.platform for post in cluster.posts})} platform(s).",
            ],
        }
    )
    return groups[:8]


def english_recommendations(cluster: Cluster) -> list[dict[str, str]]:
    """Default English actions, chosen by which indicator types are present."""
    from .recommendations import recommendation_row

    rows: list[dict[str, str]] = []
    if cluster.iocs.domains or cluster.iocs.urls or cluster.iocs.ips:
        rows.append(
            recommendation_row(
                "Block or monitor the listed domains, URLs and IP addresses according to local policy.",
                "The cluster carries network indicators that can be actioned directly.",
            )
        )
    if cluster.iocs.hashes:
        rows.append(
            recommendation_row(
                "Add the listed file hashes to endpoint blocklists and hunt for prior executions.",
                "File hashes in the cluster identify the delivered sample.",
            )
        )
    if cluster.iocs.cves:
        rows.append(
            recommendation_row(
                "Confirm patch status for the referenced CVEs on internet-facing systems.",
                "The cluster names CVEs reported as exploited in the source posts.",
            )
        )
    rows.append(
        recommendation_row(
            "Search SIEM and EDR telemetry for the listed indicators and the associated campaign terms.",
            "Retrospective search establishes whether the campaign already reached the estate.",
        )
    )
    rows.append(
        recommendation_row(
            "Preserve the source evidence and re-run enrichment if new platform posts appear in the next cycle.",
            "Evidence is credible but the cluster is still developing.",
        )
    )
    return rows[:5]


def ttp_guess(cluster: Cluster) -> list[dict[str, str]]:
    """Conservative MITRE mapping for one cluster, in English."""
    category = cluster.threat_category.value
    if category == "phishing":
        return [{"mitre_id": "T1566", "tactic": "Initial Access", "technique": "Phishing", "evidence": "Phishing language or credential collection indicators appear in source posts."}]
    if category == "malware":
        return [{"mitre_id": "T1105", "tactic": "Command and Control", "technique": "Ingress Tool Transfer", "evidence": "Payload/C2 context appears in the cluster."}]
    if category == "vulnerability":
        return [{"mitre_id": "T1190", "tactic": "Initial Access", "technique": "Exploit Public-Facing Application", "evidence": "CVE or exploitation context appears in the cluster."}]
    return [{"mitre_id": "T1595", "tactic": "Reconnaissance", "technique": "Active Scanning", "evidence": "Threat context is present but technique mapping is conservative."}]


def timeline_events(cluster: Cluster) -> list[dict[str, str]]:
    """One English timeline row per source post, in posting order."""
    return [
        {
            "timestamp": post.post.timestamp,
            "event": f"{post.platform} source post {post.id}",
            "significance": "Primary source evidence used for clustering and report generation.",
        }
        for post in sorted(cluster.posts, key=lambda item: item.post.timestamp or item.id)
    ]


def source_post_summary(post: Any, *, indicator_count: int = 0) -> str:
    """An English one-line summary of one source post.

    The post body itself is published separately, in the language it was
    posted in; this is the English description of it.
    """
    platform = str(getattr(post, "platform", "") or "unknown").title()
    date = str(getattr(getattr(post, "post", None), "timestamp", "") or "")[:10] or "an unrecorded date"
    if indicator_count:
        return f"{platform} post from {date} naming {indicator_count} indicator(s)."
    return f"{platform} post from {date} with no extracted indicator."


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


def _english_sources(sources: Any, cluster: Cluster) -> Any:
    """``report_text.sources`` with every non-English summary recomposed.

    The released shape is one ``{source_number, source_ref, summary}`` row per
    cited post. The post *body* lives outside this section (at the report
    root, under ``sources`` and ``source_posts``) and keeps the language it
    was posted in; only the summary is rewritten here.
    """
    if not isinstance(sources, list):
        return sources
    by_ref = {post.id: post for post in cluster.posts}
    rows = []
    for entry in sources:
        if not isinstance(entry, dict):
            rows.append(entry)
            continue
        post = by_ref.get(str(entry.get("source_ref", "")))
        summary = entry.get("summary", "")
        if post is not None and not text_is_english(summary):
            summary = source_post_summary(post, indicator_count=len(post.iocs.all_values()))
        rows.append({**entry, "summary": summary})
    return rows


def _english_section(section: str, cluster: Cluster, findings: list[PerspectiveFinding], current: Any) -> Any:
    if section == "cluster_name":
        return english_cluster_name(cluster)
    if section == "summary":
        return english_summary(cluster)
    if section == "key_findings":
        return english_key_findings(cluster, findings)
    if section == "recommendations":
        return english_recommendations(cluster)
    if section == "ttps":
        return [item["mitre_id"] for item in ttp_guess(cluster)]
    if section == "threat_actor":
        return "Unknown"
    if section == "timeline":
        return timeline_events(cluster)
    if section == "sources":
        return _english_sources(current, cluster)
    return current


def enforce_english(
    report_schema: dict[str, Any],
    *,
    cluster: Cluster,
    findings: list[PerspectiveFinding],
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(schema, rewritten)`` with every guarded section in English.

    The input is never mutated. ``rewritten`` names the sections the composer
    had to replace; it is empty when the report was already English.
    """
    rewritten = failing_sections(report_schema)
    if not rewritten:
        return report_schema, []
    replaced = {
        section: _english_section(section, cluster, findings, report_schema.get(section))
        for section in rewritten
    }
    return {**report_schema, **replaced}, rewritten


def deterministic_report_fields(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    decision: Any,
    settings: Any,
) -> dict[str, Any]:
    """Step 5 report fields composed without an LLM, in English.

    Every field is built from the cluster's language-neutral facts -- the
    category vocabulary, the platform and post counts, the extracted
    indicators, CVEs and brands, and the English keyword matches. A collected
    post is never quoted, so the stand-in produces English prose whatever
    language the posts were written in.
    """
    recommendations = english_recommendations(cluster)
    return {
        "cluster_name": english_cluster_name(cluster),
        "executive_summary": english_summary(cluster),
        "key_findings": english_key_findings(cluster, findings),
        "recommendations": recommendations,
        "immediate_actions": [row["action"] for row in recommendations],
        "ttps": ttp_guess(cluster),
        "timeline": timeline_events(cluster),
        "analyst_notes": decision.reason,
        "metadata": {
            "generation_mode": "offline_deterministic",
            "llm_used": False,
            "llm_provider": settings.llm_provider,
            "llm_model": "",
        },
    }



__all__ = [
    "GUARDED_REPORT_FIELDS",
    "GUARDED_SECTIONS",
    "MIN_LATIN_RATIO",
    "brand_matches",
    "deterministic_report_fields",
    "english_cluster_name",
    "english_key_findings",
    "english_keyword_matches",
    "english_recommendations",
    "english_summary",
    "enforce_english",
    "failing_report_fields",
    "failing_sections",
    "section_is_english",
    "source_post_summary",
    "strip_value_tokens",
    "text_is_english",
    "timeline_events",
    "ttp_guess",
]
