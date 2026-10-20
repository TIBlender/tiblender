import json
import logging
from pathlib import Path

from tiblender_runtime.cli import main
from tiblender_runtime.viewer_data import build_viewer_payload, write_viewer_data


def _sample_report(report_id: str = "TIBLENDER-RUN-1", *, platform: str = "x") -> dict:
    return {
        "report_id": report_id,
        "generated_at": "2026-04-11T00:00:00Z",
        "cluster_name": "Credential phishing campaign",
        "threat_category": "phishing",
        "threat_severity": "HIGH",
        "executive_summary": "Credential phishing infrastructure was consolidated from source evidence.",
        "threat_actor": "Unknown",
        "ttps": [
            {
                "mitre_id": "T1566",
                "tactic": "Initial Access",
                "technique": "Phishing",
                "evidence": "Credential collection language was observed.",
            }
        ],
        "cves": [{"cve_id": "CVE-2026-1234", "exploited_in_wild": True}],
        "sources": {
            "social_media_posts": [
                {
                    "id": "p1",
                    "platform": platform,
                    "timestamp": "2026-04-11T00:00:00Z",
                    "text": "Phishing campaign uses bad.example",
                }
            ],
            "tool_results": [],
            "total_sources_count": 1,
        },
        "report_schema": {
            "iocs": [
                {"type": "domain", "value": "bad.example", "extraction_type": "Direct", "platforms": ["x"]},
            ],
            "key_findings": [
                {"heading": "Source Evidence", "items": ["The domain was observed in the source post."]}
            ],
            "recommendations": [
                {"action": "Block bad.example.", "rationale": "Directly observed in source evidence.", "horizon": "immediate"},
                {"action": "Hunt for bad.example in telemetry.", "rationale": "Retrospective coverage.", "horizon": "short_term"},
                {"action": "Integrate the indicator into detection content.", "rationale": "Durable coverage.", "horizon": "long_term"},
            ],
        },
    }


def test_build_viewer_payload_maps_report_fields_app_js_dereferences():
    payload = build_viewer_payload([_sample_report()])

    assert payload["schema_version"] == "1.0"
    assert payload["generated_at"]
    assert payload["summary"]["report_count"] == 1
    assert payload["summary"]["category_counts"] == {"phishing": 1}
    assert payload["summary"]["severity_counts"] == {"HIGH": 1}
    assert payload["summary"]["platform_counts"] == {"x": 1}

    record = payload["reports"][0]
    assert record["id"] == "TIBLENDER-RUN-1"
    assert record["report_index"] == 0
    assert record["created_at"] == "2026-04-11T00:00:00Z"
    assert record["title"] == "Credential phishing campaign"
    assert record["category"] == "phishing"
    assert record["severity"] == "HIGH"
    assert record["platforms"] == ["x"]
    assert record["ttp_ids"] == ["T1566"]
    assert record["cves"] == ["CVE-2026-1234"]
    assert record["threat_actor"] == "Unknown"
    assert record["summary"] == "Credential phishing infrastructure was consolidated from source evidence."
    assert record["iocs"]["count"] == 1
    assert record["iocs"]["items"] == [
        {"type": "domain", "value": "bad.example", "extraction_type": "Direct", "platforms": ["x"]}
    ]
    assert record["source_posts"]["count"] == 1
    assert record["source_posts"]["items"] == [
        {
            "source_ref": "p1",
            "platform": "x",
            "timestamp": "2026-04-11T00:00:00Z",
            "text": "Phishing campaign uses bad.example",
        }
    ]
    assert record["report_text"]["key_findings"] == ["The domain was observed in the source post."]
    assert record["report_text"]["recommendations"]["immediate_actions"] == ["Block bad.example."]
    assert record["report_text"]["recommendations"]["short_term_actions"] == [
        "Hunt for bad.example in telemetry."
    ]
    assert record["report_text"]["recommendations"]["long_term_actions"] == [
        "Integrate the indicator into detection content."
    ]
    assert record["report_text"]["recommendations"]["detection_rules"] == []


def test_write_viewer_data_writes_window_assignment_with_valid_json(tmp_path: Path):
    reports_dir = tmp_path / "output" / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "TIBLENDER-RUN-1.json").write_text(json.dumps(_sample_report()), encoding="utf-8")

    viewer_dir = tmp_path / "viewer"
    result = write_viewer_data(tmp_path / "output", viewer_dir)

    assert result.path == viewer_dir / "reports_viewer_data.js"
    assert result.reports_written == 1
    assert result.skipped == 0
    text = result.path.read_text(encoding="utf-8")
    assert text.startswith("window.TIBLENDER_ARTIFACT_DATA=")
    assert text.endswith(";")

    body = text.removeprefix("window.TIBLENDER_ARTIFACT_DATA=").rstrip(";")
    payload = json.loads(body)
    assert payload["summary"]["report_count"] == 1
    assert len(payload["reports"]) == 1
    assert payload["reports"][0]["id"] == "TIBLENDER-RUN-1"


def test_write_viewer_data_skips_malformed_report_files_and_logs_warning(tmp_path: Path, caplog):
    reports_dir = tmp_path / "output" / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "TIBLENDER-RUN-1.json").write_text(json.dumps(_sample_report()), encoding="utf-8")
    malformed_path = reports_dir / "TIBLENDER-RUN-BROKEN.json"
    malformed_path.write_text("{not valid json", encoding="utf-8")

    viewer_dir = tmp_path / "viewer"
    with caplog.at_level(logging.WARNING, logger="tiblender_runtime.viewer_data"):
        result = write_viewer_data(tmp_path / "output", viewer_dir)

    assert result.reports_written == 1
    assert result.skipped == 1
    body = result.path.read_text(encoding="utf-8").removeprefix("window.TIBLENDER_ARTIFACT_DATA=").rstrip(";")
    payload = json.loads(body)
    assert payload["summary"]["report_count"] == 1
    assert [record["id"] for record in payload["reports"]] == ["TIBLENDER-RUN-1"]
    assert any(str(malformed_path) in record.message for record in caplog.records)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_write_viewer_data_collects_multi_cycle_report_tree(tmp_path: Path):
    output_root = tmp_path / "output"
    cycle_one = output_root / "cycle_001" / "reports"
    cycle_two = output_root / "cycle_002" / "reports"
    cycle_one.mkdir(parents=True)
    cycle_two.mkdir(parents=True)
    (cycle_one / "TIBLENDER-RUN-1.json").write_text(json.dumps(_sample_report("TIBLENDER-RUN-1", platform="x")), encoding="utf-8")
    (cycle_two / "TIBLENDER-RUN-2.json").write_text(
        json.dumps(_sample_report("TIBLENDER-RUN-2", platform="reddit")), encoding="utf-8"
    )

    viewer_dir = tmp_path / "viewer"
    result = write_viewer_data(output_root, viewer_dir)

    assert result.reports_written == 2
    assert result.skipped == 0
    body = result.path.read_text(encoding="utf-8").removeprefix("window.TIBLENDER_ARTIFACT_DATA=").rstrip(";")
    payload = json.loads(body)
    assert payload["summary"]["report_count"] == 2
    assert {record["id"] for record in payload["reports"]} == {"TIBLENDER-RUN-1", "TIBLENDER-RUN-2"}
    assert payload["summary"]["platform_counts"] == {"x": 1, "reddit": 1}


def test_cli_build_viewer_command_prints_data_path_and_report_count(tmp_path: Path, capsys):
    reports_dir = tmp_path / "output" / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "TIBLENDER-RUN-1.json").write_text(json.dumps(_sample_report()), encoding="utf-8")
    viewer_dir = tmp_path / "viewer"

    exit_code = main(
        [
            "build-viewer",
            "--reports-root",
            str(tmp_path / "output"),
            "--viewer-dir",
            str(viewer_dir),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert f"viewer_data={viewer_dir / 'reports_viewer_data.js'}" in captured.out
    assert "reports=1" in captured.out
    assert "skipped=0" in captured.out
    assert (viewer_dir / "reports_viewer_data.js").exists()


# ---------------------------------------------------------------------------
# Determinism: two consecutive offline runs are byte-identical. They were not, because the viewer bundle's
# top-level `generated_at` was stamped from the wall clock even when the run
# itself was on the pipeline's fixed clock.
# ---------------------------------------------------------------------------


def _three_cycle_tree(root: Path) -> Path:
    for index, (report_id, stamp) in enumerate(
        (("TIBLENDER-RUN-1", "2026-04-11T00:00:00Z"), ("TIBLENDER-RUN-2", "2026-04-13T06:30:00Z")),
        start=2,
    ):
        reports_dir = root / f"cycle_{index:03d}" / "reports"
        reports_dir.mkdir(parents=True)
        report = _sample_report(report_id)
        report["generated_at"] = stamp
        (reports_dir / f"{report_id}.json").write_text(json.dumps(report), encoding="utf-8")
    return root


def test_fixed_clock_build_stamps_the_newest_report_time(tmp_path: Path):
    output_root = _three_cycle_tree(tmp_path / "output")

    write_viewer_data(output_root, tmp_path / "viewer", fixed_clock=True)

    payload = _payload(tmp_path / "viewer" / "reports_viewer_data.js")
    assert payload["generated_at"] == "2026-04-13T06:30:00Z"


def test_two_fixed_clock_builds_of_the_same_reports_are_byte_identical(tmp_path: Path):
    output_root = _three_cycle_tree(tmp_path / "output")
    data_path = tmp_path / "viewer" / "reports_viewer_data.js"

    write_viewer_data(output_root, tmp_path / "viewer", fixed_clock=True)
    first = data_path.read_bytes()
    write_viewer_data(output_root, tmp_path / "viewer", fixed_clock=True)

    assert data_path.read_bytes() == first


def test_a_wall_clock_build_still_stamps_the_current_time(tmp_path: Path):
    from tiblender_runtime import clock

    output_root = _three_cycle_tree(tmp_path / "output")

    write_viewer_data(output_root, tmp_path / "viewer", fixed_clock=False)

    payload = _payload(tmp_path / "viewer" / "reports_viewer_data.js")
    assert payload["generated_at"] != "2026-04-13T06:30:00Z"
    assert payload["generated_at"].endswith("Z")
    assert payload["generated_at"] <= clock.now_iso()


def test_cli_build_viewer_uses_the_fixed_clock_in_offline_mode(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TIBLENDER_MODE", "offline")
    monkeypatch.setenv("TIBLENDER_CLOCK", "auto")
    output_root = _three_cycle_tree(tmp_path / "output")

    assert main(["build-viewer", "--reports-root", str(output_root), "--viewer-dir", str(tmp_path / "viewer")]) == 0

    payload = _payload(tmp_path / "viewer" / "reports_viewer_data.js")
    assert payload["generated_at"] == "2026-04-13T06:30:00Z"


def _payload(data_path: Path) -> dict:
    body = data_path.read_text(encoding="utf-8")
    assert body.startswith("window.TIBLENDER_ARTIFACT_DATA=") and body.endswith(";")
    return json.loads(body[len("window.TIBLENDER_ARTIFACT_DATA=") : -1])
