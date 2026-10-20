from pathlib import Path

from tiblender_runtime.investigation import investigate_cluster
from tiblender_runtime.models import Cluster, IocSet, NormalizedPost, Post, Severity, ThreatCategory
from tiblender_runtime.contract import REQUIRED_TOOL_TYPES, audit_contract, prompt_bundle_text
from tiblender_runtime.rl.policy import InvestigationStrategy
from tiblender_runtime.settings import Settings
from tiblender_runtime.tools import ToolRunner


def _cluster_with_url() -> Cluster:
    post = NormalizedPost(
        post=Post(
            id="p1",
            platform="x",
            content="Adobe Reader zero-day payload uses hxxps://ado-read-parser[.]com/a.pdf",
            timestamp="2026-04-11T00:00:00Z",
        ),
        normalized_content="Adobe Reader zero-day payload uses ado-read-parser.com",
        iocs=IocSet(
            urls=["https://ado-read-parser.com/a.pdf"],
            domains=["ado-read-parser.com"],
            hashes=["65dca34b5a28c3d9f8a8eb8a0e60a0b1d6d1d4b64db48ab681fd8db1d80abde7"],
            cves=["CVE-2026-34621"],
        ),
        threat_category=ThreatCategory.VULNERABILITY,
    )
    return Cluster(
        id="cluster-test",
        topic_name="Adobe Reader zero-day exploitation",
        description="Adobe Reader zero-day exploitation",
        posts=[post],
        iocs=post.iocs,
        severity=Severity.HIGH,
        threat_category=ThreatCategory.VULNERABILITY,
    )


def test_tool_contract_uses_web_crawl_for_url_evidence():
    assert "WEB_CRAWL" in REQUIRED_TOOL_TYPES

    prompt_text = prompt_bundle_text()
    assert "WEB_CRAWL" in prompt_text


def test_investigation_uses_self_crawler_for_url_evidence():
    settings = Settings(
        enable_real_tools=False,
        allow_heuristic_reports=True,
        max_queries_per_perspective=10,
    )
    findings = investigate_cluster(
        _cluster_with_url(),
        settings=settings,
        runner=ToolRunner(settings),
        strategy=InvestigationStrategy(mode="multi_perspective", similarity_score=0.0),
        planned_queries=[],
    )
    tool_types = [query.tool_type for finding in findings for query in finding.queries]
    result_tool_types = [result.tool_type for finding in findings for result in finding.tool_results]

    assert "WEB_CRAWL" in tool_types
    assert set(tool_types).issubset(set(REQUIRED_TOOL_TYPES))
    assert set(result_tool_types).issubset(set(REQUIRED_TOOL_TYPES))


def test_runtime_source_has_no_non_contract_tool_reference():
    source_root = Path("tiblender_runtime")
    offenders = []
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        result = audit_contract(path.parent)
        if str(path) in result["forbidden_tool_references"]:
            offenders.append(str(path))

    assert offenders == []


def test_contract_audit_rejects_non_contract_tool_tokens(tmp_path):
    source_file = tmp_path / "bad_tool.py"
    source_file.write_text("TOOL = 'UNLISTED" + "_LOOKUP'\n", encoding="utf-8")

    result = audit_contract(tmp_path)

    assert result["ok"] is False
    assert result["forbidden_tool_references"] == [str(source_file)]


def test_contract_audit_reports_alignment():
    result = audit_contract()

    assert result["ok"] is True
    assert result["forbidden_tool_references"] == []
    assert result["missing_tools"] == []
