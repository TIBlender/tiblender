import json
from pathlib import Path

import pytest

from tiblender_runtime.pipeline import run_cycle
from tiblender_runtime.settings import MissingLLMConfigurationError, Settings


def test_run_cycle_rejects_report_generation_without_llm(tmp_path: Path):
    input_path = Path("sample_runs/collector_experiment_sample/sample_run_posts.jsonl")
    settings = Settings(
        input_path=input_path,
        output_dir=tmp_path,
        history_dir=tmp_path / "history",
        llm_provider="heuristic",
        mode="live",
    )

    with pytest.raises(MissingLLMConfigurationError):
        run_cycle(settings)

    assert not (tmp_path / "reports").exists()


def test_run_cycle_writes_reports_and_feedback(tmp_path: Path):
    input_path = Path("sample_runs/collector_experiment_sample/sample_run_posts.jsonl")
    settings = Settings(
        input_path=input_path,
        output_dir=tmp_path,
        history_dir=tmp_path / "history",
        llm_provider="heuristic",
        allow_heuristic_reports=True,
        enable_real_tools=False,
    )

    result = run_cycle(settings)

    assert result.reports_generated >= 1
    assert (tmp_path / "cycle_summary.json").exists()
    assert (tmp_path / "dynamic_queries.json").exists()
    assert (tmp_path / "history" / "policy_state.json").exists()

    report_files = sorted((tmp_path / "reports").glob("*.json"))
    assert report_files
    report = json.loads(report_files[0].read_text(encoding="utf-8"))

    assert report["threat_severity"] in {"MEDIUM", "HIGH", "CRITICAL"}
    assert report["executive_summary"]
    assert report["iocs"]["domains"] or report["iocs"]["ips"] or report["iocs"]["cves"] or report["iocs"]["hashes"]
    assert report["platform_unique_contributions"]
    assert all("author" not in post for post in report["sources"]["social_media_posts"])
    assert report["judgement_trace"]
    assert report["metadata"]["generation_mode"] == "offline_deterministic"
    assert report["metadata"]["llm_used"] is False
    assert all("gpt" not in round_item["judge"].lower() for round_item in report["evaluation_rounds"])
    assert all("llama" not in round_item["judge"].lower() for round_item in report["evaluation_rounds"])
    assert set(report["report_schema"]) == {
        "cluster_name",
        "severity",
        "summary",
        "key_findings",
        "recommendations",
        "iocs",
        "cves",
        "ttps",
        "threat_actor",
        "timeline",
        "sources",
    }
    assert set(report["english_report"]) == {"summary_en", "key_findings_en"}
    assert all("extraction_type" in item for item in report["report_schema"]["iocs"])


def test_cycle_summary_records_the_fallback_reason_histogram(tmp_path: Path):
    """``llm_fallback_reasons`` sits beside ``llm_fallbacks`` and always has
    every key, so a consumer can tell "no fallback" from "not recorded"."""
    from tiblender_runtime.llm_fallbacks import (
        FALLBACK_KEYS,
        PAD_KEYS,
        REASON_KEYS,
        UNATTRIBUTED_KEY,
    )

    settings = Settings(
        input_path=Path("sample_runs/collector_experiment_sample/sample_run_posts.jsonl"),
        output_dir=tmp_path,
        history_dir=tmp_path / "history",
        llm_provider="heuristic",
        allow_heuristic_reports=True,
        enable_real_tools=False,
    )

    run_cycle(settings)
    summary = json.loads((tmp_path / "cycle_summary.json").read_text(encoding="utf-8"))

    assert set(summary["llm_fallback_reasons"]) == set(REASON_KEYS)
    assert set(summary["llm_fallbacks"]) == set(FALLBACK_KEYS) | set(PAD_KEYS) | {UNATTRIBUTED_KEY}
    # The documented invariant: the reason histogram totals the *stage*
    # counters. The two bookkeeping counters sit in the same block but are
    # outside that sum.
    assert sum(summary["llm_fallback_reasons"].values()) == sum(
        summary["llm_fallbacks"][key] for key in FALLBACK_KEYS
    )
    # The deterministic offline stand-in never calls a provider, so it can
    # never fall back: both blocks stay at zero and every report stays
    # offline_deterministic rather than llm_partial.
    assert sum(summary["llm_fallback_reasons"].values()) == 0
    assert summary["llm_fallbacks"][UNATTRIBUTED_KEY] == 0
    for path in sorted((tmp_path / "reports").glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["metadata"]["generation_mode"] == "offline_deterministic"
        assert report["metadata"]["llm_stage_fallbacks"] == []
