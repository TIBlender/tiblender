"""STIX 2.1 bundle export, derived from the same filtered report object used
for JSON, Markdown, and PDF export so all formats agree on promoted
indicators."""

from __future__ import annotations

import re
import uuid
from typing import Any

_STIX_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://tiblender.local/stix")


def _stix_id(object_type: str, *parts: object) -> str:
    key = "|".join(str(part) for part in parts)
    return f"{object_type}--{uuid.uuid5(_STIX_NAMESPACE, object_type + '|' + key)}"


def _stix_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _hash_algorithm(value: str) -> str:
    if re.fullmatch(r"[A-Fa-f0-9]{32}", value):
        return "MD5"
    if re.fullmatch(r"[A-Fa-f0-9]{40}", value):
        return "SHA-1"
    return "SHA-256"


def _stix_indicator(name: str, value: str, pattern: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "indicator",
        "spec_version": "2.1",
        "id": _stix_id("indicator", name, value),
        "created": report["generated_at"],
        "modified": report["generated_at"],
        "name": name,
        "description": f"Indicator observed in {report['cluster_name']}.",
        "pattern": pattern,
        "pattern_type": "stix",
        "valid_from": report["generated_at"],
        "labels": [report["threat_category"]],
        "confidence": {"LOW": 35, "MEDIUM": 60, "HIGH": 85}.get(report["evidence_assessment"]["confidence_level"], 50),
    }


def stix_bundle(report: dict[str, Any]) -> dict[str, Any]:
    # STIX export is derived from the same filtered report object so JSON,
    # Markdown, PDF, and STIX cannot disagree on promoted indicators.
    related_objects: list[dict[str, Any]] = []
    for domain in report["iocs"]["domains"]:
        related_objects.append(
            _stix_indicator(
                f"Domain indicator: {domain}",
                domain,
                f"[domain-name:value = '{_stix_quote(domain)}']",
                report,
            )
        )
    for ip in report["iocs"]["ips"]:
        stix_type = "ipv6-addr" if ":" in ip else "ipv4-addr"
        related_objects.append(
            _stix_indicator(
                f"IP indicator: {ip}",
                ip,
                f"[{stix_type}:value = '{_stix_quote(ip)}']",
                report,
            )
        )
    for url in report["iocs"]["urls"]:
        related_objects.append(
            _stix_indicator(
                f"URL indicator: {url}",
                url,
                f"[url:value = '{_stix_quote(url)}']",
                report,
            )
        )
    for hash_value in report["iocs"]["hashes"]:
        algorithm = _hash_algorithm(hash_value)
        related_objects.append(
            _stix_indicator(
                f"File hash indicator: {hash_value}",
                hash_value,
                f"[file:hashes.'{algorithm}' = '{_stix_quote(hash_value.lower())}']",
                report,
            )
        )
    for cve in report["iocs"]["cves"]:
        related_objects.append(
            {
                "type": "vulnerability",
                "spec_version": "2.1",
                "id": _stix_id("vulnerability", cve),
                "created": report["generated_at"],
                "modified": report["generated_at"],
                "name": cve,
                "external_references": [{"source_name": "cve", "external_id": cve}],
            }
        )
    for ttp in report["ttps"]:
        related_objects.append(
            {
                "type": "attack-pattern",
                "spec_version": "2.1",
                "id": _stix_id("attack-pattern", ttp["mitre_id"], ttp["technique"]),
                "created": report["generated_at"],
                "modified": report["generated_at"],
                "name": ttp["technique"],
                "description": ttp["evidence"],
                "external_references": [
                    {
                        "source_name": "mitre-attack",
                        "external_id": ttp["mitre_id"],
                    }
                ],
            }
        )
    report_object = {
        "type": "report",
        "spec_version": "2.1",
        "id": _stix_id("report", report["report_id"]),
        "created": report["generated_at"],
        "modified": report["generated_at"],
        "name": report["cluster_name"],
        "description": report["executive_summary"],
        "published": report["generated_at"],
        "report_types": ["threat-report"],
        "labels": [report["threat_category"], report["threat_severity"].lower()],
        "object_refs": [item["id"] for item in related_objects],
    }
    return {
        "type": "bundle",
        "spec_version": "2.1",
        "id": _stix_id("bundle", report["report_id"]),
        "objects": [report_object, *related_objects],
    }
