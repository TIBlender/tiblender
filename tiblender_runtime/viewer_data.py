"""Builds the browser report viewer's data file from generated runtime
reports.

The viewer at ``viewer/`` (adapted from the dataset viewer)
reads ``window.TIBLENDER_ARTIFACT_DATA`` and dereferences a fixed set of
per-report fields (see ``viewer/app.js``); the mapping here must keep those
field names exact so the viewer renders without console errors.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import clock

SCHEMA_VERSION = "1.0"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ViewerBuildResult:
    """Outcome of :func:`write_viewer_data`: where it wrote, how many report
    files it read successfully, and how many it had to skip."""

    path: Path
    reports_written: int
    skipped: int


def _viewer_platforms(report: dict[str, Any]) -> list[str]:
    posts = report.get("sources", {}).get("social_media_posts", [])
    return sorted({str(post.get("platform", "")) for post in posts if post.get("platform")})


def _viewer_ioc_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    # Mirrors the released dataset's IoC field set, so this viewer and the
    # dataset viewer render the same columns.
    return [
        {
            "type": item.get("type", ""),
            "value": item.get("value", ""),
            "extraction_type": item.get("extraction_type", ""),
            "platforms": list(item.get("platforms", [])),
        }
        for item in report.get("report_schema", {}).get("iocs", [])
    ]


def _viewer_source_posts(report: dict[str, Any]) -> dict[str, Any]:
    sources = report.get("sources", {})
    posts = sources.get("social_media_posts", [])
    return {
        "count": sources.get("total_sources_count", len(posts)),
        "items": [
            {
                "source_ref": post.get("id", ""),
                "platform": post.get("platform", ""),
                "timestamp": post.get("timestamp", ""),
                "text": post.get("text", ""),
            }
            for post in posts
        ],
    }


def _viewer_key_findings(report: dict[str, Any]) -> list[str]:
    # app.js's findingsHtml expects a flat array of strings; the runtime
    # report groups findings by perspective/heading, so flatten while
    # de-duplicating in encounter order.
    findings: list[str] = []
    for group in report.get("report_schema", {}).get("key_findings", []):
        for item in group.get("items", []):
            if item and item not in findings:
                findings.append(item)
    return findings


def _viewer_recommendations(report: dict[str, Any]) -> dict[str, Any]:
    # Recommendations are grouped by the horizon the runtime assigned, so the
    # viewer's three action lists mean the same thing as the released
    # dataset's `recommendations[].horizon`.
    grouped: dict[str, list[str]] = {"immediate": [], "short_term": [], "long_term": []}
    for item in report.get("report_schema", {}).get("recommendations", []):
        action = item.get("action", "")
        if not action:
            continue
        horizon = str(item.get("horizon", "")).strip().lower()
        grouped.get(horizon, grouped["immediate"]).append(action)
    return {
        "mitigation_priority": report.get("threat_severity", ""),
        "immediate_actions": grouped["immediate"],
        "short_term_actions": grouped["short_term"],
        "long_term_actions": grouped["long_term"],
        "detection_rules": [],
    }


def _viewer_record(report: dict[str, Any], index: int) -> dict[str, Any]:
    ttps = report.get("ttps", []) or []
    cves = report.get("cves", []) or []
    return {
        "id": report.get("report_id", ""),
        "report_index": index,
        "created_at": report.get("generated_at", ""),
        "title": report.get("cluster_name", ""),
        "category": report.get("threat_category", ""),
        "severity": report.get("threat_severity", ""),
        "platforms": _viewer_platforms(report),
        "ttp_ids": [item.get("mitre_id", "") for item in ttps if item.get("mitre_id")],
        "cves": [item.get("cve_id", "") for item in cves if item.get("cve_id")],
        "threat_actor": report.get("threat_actor", ""),
        "summary": report.get("executive_summary", ""),
        "iocs": {
            "count": len(report.get("report_schema", {}).get("iocs", [])),
            "items": _viewer_ioc_items(report),
        },
        "source_posts": _viewer_source_posts(report),
        "report_text": {
            "title": report.get("cluster_name", ""),
            "summary": report.get("executive_summary", ""),
            "key_findings": _viewer_key_findings(report),
            "infection_chain": "",
            "recommendations": _viewer_recommendations(report),
        },
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    platform_counts: dict[str, int] = {}
    total_ioc_count = 0
    for record in records:
        category = record["category"] or "Uncategorized"
        severity = record["severity"] or "UNKNOWN"
        category_counts[category] = category_counts.get(category, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        for platform in record["platforms"]:
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
        total_ioc_count += record["iocs"]["count"]
    return {
        "report_count": len(records),
        "category_counts": category_counts,
        "severity_counts": severity_counts,
        "platform_counts": platform_counts,
        "total_ioc_count": total_ioc_count,
    }


def build_viewer_payload(
    reports: list[dict[str, Any]],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Map generated runtime reports into the viewer's payload shape.

    ``generated_at`` defaults to the runtime clock (wall time unless a fixed
    clock is in effect); pass an explicit stamp to make a rebuild of the same
    reports byte-identical.
    """
    records = [_viewer_record(report, index) for index, report in enumerate(reports)]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or clock.now_iso(),
        "summary": _summary(records),
        "reports": records,
    }


def fixed_clock_build_time(reports: list[dict[str, Any]]) -> str | None:
    """A build stamp derived from the run itself, for fixed-clock runs.

    Every report carries the fixed clock of the cycle that produced it, so the
    newest of them is "the time this body of reports is current as of" and does
    not move when the same reports are bundled again. Returns ``None`` when
    there is no report to derive a stamp from, leaving the caller on the
    runtime clock.
    """
    stamps = sorted(
        str(report.get("generated_at", "")) for report in reports if report.get("generated_at")
    )
    return stamps[-1] if stamps else None


def collect_report_paths(reports_root: Path) -> list[Path]:
    """Find every generated report JSON under ``reports_root``.

    Matches both a single-cycle ``output/reports/*.json`` tree and a
    multi-cycle ``output/cycle_00N/reports/*.json`` tree.
    """
    return sorted(Path(reports_root).glob("**/reports/*.json"))


def _load_report(path: Path) -> dict[str, Any] | None:
    # A cycle in progress (or interrupted) can leave a truncated or partially
    # written report JSON under reports_root; build-viewer must skip it
    # rather than crash the whole viewer build.
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        logger.warning("Skipping unreadable or malformed report file %s: %s", path, error)
        return None


def write_viewer_data(
    reports_root: Path,
    viewer_dir: Path,
    *,
    fixed_clock: bool = False,
) -> ViewerBuildResult:
    """Write ``viewer_dir/reports_viewer_data.js`` from every report found
    under ``reports_root``.

    With ``fixed_clock`` (offline/`TIBLENDER_CLOCK=fixed` runs) the payload's
    build stamp comes from the reports themselves instead of wall time, so two
    consecutive offline runs produce a byte-identical file.

    Returns the written path together with how many report files were read
    successfully and how many were skipped (logged as warnings) for being
    unreadable or not valid JSON.
    """
    report_paths = collect_report_paths(reports_root)
    reports: list[dict[str, Any]] = []
    skipped = 0
    for path in report_paths:
        report = _load_report(path)
        if report is None:
            skipped += 1
            continue
        reports.append(report)
    payload = build_viewer_payload(
        reports,
        generated_at=fixed_clock_build_time(reports) if fixed_clock else None,
    )

    viewer_dir = Path(viewer_dir)
    viewer_dir.mkdir(parents=True, exist_ok=True)
    data_path = viewer_dir / "reports_viewer_data.js"
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    data_path.write_text(f"window.TIBLENDER_ARTIFACT_DATA={body};", encoding="utf-8")
    return ViewerBuildResult(path=data_path, reports_written=len(reports), skipped=skipped)
