"""Markdown rendering of one assembled report.

Kept out of :mod:`tiblender_runtime.reporting` so report *assembly* and
report *rendering* stay separately readable and each module stays inside the
package's 800-line budget. The renderer reads the canonical report object
that ``reporting.build_report`` returns and writes nothing back to it.
"""

from __future__ import annotations

from typing import Any

from .recommendations import markdown_recommendation_lines


def markdown_report(report: dict[str, Any]) -> str:
    ioc_lines = []
    for item in report["report_schema"]["iocs"]:
        platforms = ", ".join(item["platforms"]) or "-"
        ioc_lines.append(
            f"| {item['type']} | `{item['value']}` | {item['extraction_type']} | {platforms} |"
        )
    if not ioc_lines:
        ioc_lines.append("| none | - | - | - |")
    cve_lines = [
        f"| `{item['cve_id']}` | {str(item['exploited_in_wild']).lower()} |"
        for item in report["cves"]
    ] or ["| none | - |"]
    ttp_lines = [
        f"| `{item['mitre_id']}` | {item['tactic']} | {item['technique']} | {item['evidence']} |"
        for item in report["ttps"]
    ] or ["| none | - | - | - |"]
    finding_lines: list[str] = []
    for group in report["report_schema"]["key_findings"]:
        finding_lines.append(f"- **{group['heading']}**")
        finding_lines.extend(f"  - {item}" for item in group["items"][:4])
    if not finding_lines:
        finding_lines.append("- No additional tool findings were required.")
    recommendation_lines = markdown_recommendation_lines(report["report_schema"]["recommendations"])
    timeline_lines = [
        f"| {item['timestamp']} | {item['event']} | {item['significance']} |"
        for item in report["report_schema"]["timeline"]
    ] or ["| none | - | - |"]
    source_lines = [
        f"- `{post['id']}` ({post['platform']}, {post['timestamp']}): {post['text']}"
        for post in report["sources"]["social_media_posts"]
    ]
    if not source_lines:
        source_lines.append("- No source posts were attached.")
    return "\n".join(
        [
            f"# {report['cluster_name']}",
            "",
            f"**Report ID**: {report['report_id']}",
            f"**Generated At**: {report['generated_at']}",
            f"**Severity**: {report['threat_severity']}",
            f"**Category**: {report['threat_category']}",
            "",
            "## Executive Summary",
            "",
            report["executive_summary"],
            "",
            "## Key Findings",
            "",
            *finding_lines,
            "",
            "## IoCs",
            "",
            "| Type | Value | Extraction Type | Source Platforms |",
            "|------|-------|-----------------|------------------|",
            *ioc_lines,
            "",
            "## CVEs",
            "",
            "| CVE ID | Exploited In Wild |",
            "|--------|-------------------|",
            *cve_lines,
            "",
            "## TTPs",
            "",
            "| MITRE ID | Tactic | Technique | Evidence |",
            "|----------|--------|-----------|----------|",
            *ttp_lines,
            "",
            "## Threat Actor",
            "",
            report["report_schema"]["threat_actor"],
            "",
            "## Attack Timeline",
            "",
            "| Timestamp | Event | Significance |",
            "|-----------|-------|--------------|",
            *timeline_lines,
            "",
            "## Recommendations",
            "",
            *recommendation_lines,
            "",
            "## Source Evidence",
            "",
            *source_lines,
        ]
    )


__all__ = ["markdown_report"]
