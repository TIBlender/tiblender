import json

import pytest

from tiblender_runtime.llm import LLMGenerationError, _chat_completion
from tiblender_runtime.models import Cluster, EvaluationDecision, EvaluationRound, EvaluationVerdict, IocSet, NormalizedPost, PerspectiveFinding, Post, Severity, ThreatCategory
from tiblender_runtime.settings import Settings


def _completion(content: str, finish_reason: str = "stop"):
    """Build the ChatCompletion a patched provider call returns."""
    from tiblender_runtime.llm import ChatCompletion

    return ChatCompletion(content=content, finish_reason=finish_reason)


def test_chat_completion_uses_max_completion_tokens_for_azure(monkeypatch):
    captured = {}

    def fake_request_json(url, headers, body, timeout):
        captured["body"] = body
        return {"choices": [{"message": {"content": '{"executive_summary":"x","immediate_actions":["y"]}'}}]}

    monkeypatch.setattr("tiblender_runtime.llm._request_json", fake_request_json)
    settings = Settings(
        llm_provider="azure_openai",
        openai_api_key="test-key",
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_deployment="deployment",
        azure_openai_api_version="2026-01-01",
        llm_max_output_tokens=321,
    )

    _chat_completion(settings, [{"role": "user", "content": "test"}])

    assert captured["body"]["max_completion_tokens"] == 321
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert "max_tokens" not in captured["body"]


def test_chat_completion_retries_without_temperature_when_provider_rejects_it(monkeypatch):
    bodies = []

    def fake_request_json(url, headers, body, timeout):
        bodies.append(body.copy())
        if len(bodies) == 1:
            raise LLMGenerationError('"param": "temperature"')
        return {"choices": [{"message": {"content": '{"executive_summary":"x","immediate_actions":["y"]}'}}]}

    monkeypatch.setattr("tiblender_runtime.llm._request_json", fake_request_json)
    settings = Settings(
        llm_provider="azure_openai",
        openai_api_key="test-key",
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_deployment="deployment",
        azure_openai_api_version="2026-01-01",
        llm_temperature=0.1,
    )

    _chat_completion(settings, [{"role": "user", "content": "test"}])

    assert bodies[0]["temperature"] == pytest.approx(0.1)
    assert "temperature" not in bodies[1]


def test_chat_completion_retries_without_response_format_when_provider_rejects_it(monkeypatch):
    bodies = []

    def fake_request_json(url, headers, body, timeout):
        bodies.append(body.copy())
        if len(bodies) == 1:
            raise LLMGenerationError('"param": "response_format"')
        return {"choices": [{"message": {"content": '{"executive_summary":"x","immediate_actions":["y"]}'}}]}

    monkeypatch.setattr("tiblender_runtime.llm._request_json", fake_request_json)
    settings = Settings(
        llm_provider="azure_openai",
        openai_api_key="test-key",
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_deployment="deployment",
        azure_openai_api_version="2026-01-01",
    )

    _chat_completion(settings, [{"role": "user", "content": "test"}])

    assert bodies[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in bodies[1]


def test_chat_completion_retries_with_larger_budget_when_content_is_empty_length(monkeypatch):
    bodies = []

    def fake_request_json(url, headers, body, timeout):
        bodies.append(body.copy())
        if len(bodies) == 1:
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": ""},
                    }
                ],
                "usage": {"completion_tokens": 128, "completion_tokens_details": {"reasoning_tokens": 128}},
            }
        return {"choices": [{"finish_reason": "stop", "message": {"content": '{"executive_summary":"x","immediate_actions":["y"]}'}}]}

    monkeypatch.setattr("tiblender_runtime.llm._request_json", fake_request_json)
    settings = Settings(
        llm_provider="azure_openai",
        openai_api_key="test-key",
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_deployment="deployment",
        azure_openai_api_version="2026-01-01",
        llm_max_output_tokens=128,
    )

    content = _chat_completion(settings, [{"role": "user", "content": "test"}])

    assert content
    assert bodies[0]["max_completion_tokens"] == 128
    assert bodies[1]["max_completion_tokens"] == 8192


def test_report_prompt_requires_english_report_text():
    from tiblender_runtime.llm_agents import _messages
    from tiblender_runtime.contract import REPORT_GENERATION_PROMPT

    messages = _messages({"cluster": {"topic_name": "test"}})

    assert REPORT_GENERATION_PROMPT in messages[0]["content"]
    assert "professional English" in messages[0]["content"]
    assert "professional " + "Japanese" not in messages[0]["content"]
    assert messages[0]["content"].startswith("Role: Generate comprehensive")


def test_post_normalization_prompt_requires_english_and_preserves_iocs(monkeypatch):
    from tiblender_runtime.contract import POST_NORMALIZATION_PROMPT
    from tiblender_runtime.llm import normalize_post_batch

    captured = {}

    def fake_chat_completion(settings, messages):
        captured["messages"] = messages
        return _completion(
            '{"posts":[{"id":"p1","normalized_content":"Apple ID phishing lure targeting Japanese users."}]}'
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion_with_meta", fake_chat_completion)
    settings = Settings(llm_provider="openai", llm_model="model", openai_api_key="key")
    posts = [
        Post(
            id="p1",
            platform="x",
            content="Apple IDの認証を装うフィッシングを確認。hxxps://xyahyy[.]com/login を使用。",
            timestamp="2026-04-13T06:51:56Z",
            metadata={"language": "ja"},
        )
    ]

    normalized = normalize_post_batch(settings, posts)

    assert POST_NORMALIZATION_PROMPT in captured["messages"][0]["content"]
    assert "professional English" in captured["messages"][0]["content"]
    assert normalized["p1"].startswith("Apple ID phishing lure")
    assert "https://xyahyy.com/login" in normalized["p1"]


def test_post_normalization_removes_standalone_absence_placeholders(monkeypatch):
    from tiblender_runtime.llm import normalize_post_batch

    def fake_chat_completion(settings, messages):
        del settings, messages
        return _completion(
            '{"posts":[{"id":"p1","normalized_content":'
            '"Supply chain attack, Axios npm packages, none, none, IoCs: sfrclak.com"}]}'
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion_with_meta", fake_chat_completion)
    settings = Settings(llm_provider="openai", llm_model="model", openai_api_key="key")
    posts = [
        Post(
            id="p1",
            platform="x",
            content="Axios compromise using sfrclak[.]com",
            timestamp="2026-03-31T03:20:27Z",
            metadata={"language": "en"},
        )
    ]

    normalized = normalize_post_batch(settings, posts)

    assert normalized["p1"] == "Supply chain attack, Axios npm packages. IoCs: sfrclak.com"


def test_post_normalization_drops_an_ioc_label_left_empty_by_placeholder_removal(monkeypatch):
    from tiblender_runtime.llm import normalize_post_batch

    def fake_chat_completion(settings, messages):
        del settings, messages
        return _completion(
            '{"posts":[{"id":"p1","normalized_content":'
            '"Credential phishing lure targeting bank customers, IoCs: none"}]}'
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion_with_meta", fake_chat_completion)
    settings = Settings(llm_provider="openai", llm_model="model", openai_api_key="key")
    posts = [
        Post(
            id="p1",
            platform="x",
            content="Credential phishing lure targeting bank customers",
            timestamp="2026-03-31T03:20:27Z",
            metadata={"language": "en"},
        )
    ]

    normalized = normalize_post_batch(settings, posts)

    assert normalized["p1"] == "Credential phishing lure targeting bank customers."


def _normalized_post(post_id: str, content: str, platform: str = "x") -> NormalizedPost:
    iocs = IocSet(domains=["bad.example"]) if "bad.example" in content else IocSet()
    return NormalizedPost(
        post=Post(id=post_id, platform=platform, content=content, timestamp="2026-04-01T00:00:00Z"),
        normalized_content=content,
        iocs=iocs,
        threat_category=ThreatCategory.MALWARE if "payload" in content.lower() else ThreatCategory.PHISHING,
    )


def _llm_cluster() -> Cluster:
    posts = [
        _normalized_post("p1", "Axios supply-chain payload uses bad.example", "x"),
        _normalized_post("p2", "Credential phishing lure mentions bad.example", "reddit"),
    ]
    return Cluster(
        id="cluster-input",
        topic_name="Mixed threat cluster",
        description="Mixed threat cluster",
        posts=posts,
        iocs=IocSet.from_many([post.iocs for post in posts]),
        severity=Severity.MEDIUM,
        threat_category=ThreatCategory.MALWARE,
    )


def test_refine_clusters_with_llm_splits_campaign_units_and_validates_post_ids(monkeypatch):
    from tiblender_runtime.contract import THREAT_CLUSTERING_PROMPT
    from tiblender_runtime.llm import refine_clusters_with_llm

    captured = {}

    def fake_chat_completion(settings, messages):
        captured["messages"] = messages
        return _completion(
            '{"clusters":['
            '{"topic_name":"Axios supply-chain payload","security_severity":"HIGH",'
            '"topic_description":"Malicious npm package payload activity.","post_ids":["p1"]},'
            '{"topic_name":"Credential phishing lure","security_severity":"MEDIUM",'
            '"topic_description":"Credential phishing activity.","post_ids":["p2","missing-post"]}'
            ']}'
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion_with_meta", fake_chat_completion)
    settings = Settings(llm_provider="openai", llm_model="model", openai_api_key="key")

    refined = refine_clusters_with_llm(settings, [_llm_cluster()])

    assert THREAT_CLUSTERING_PROMPT in captured["messages"][0]["content"]
    assert [cluster.topic_name for cluster in refined] == ["Axios supply-chain payload", "Credential phishing lure"]
    assert [cluster.severity for cluster in refined] == [Severity.HIGH, Severity.MEDIUM]
    assert [[post.id for post in cluster.posts] for cluster in refined] == [["p1"], ["p2"]]
    assert all(cluster.clustering_signals["llm_campaign_refinement"] is True for cluster in refined)


def test_plan_investigation_queries_uses_llm_allowlist_and_discards_generic_queries(monkeypatch):
    from tiblender_runtime.contract import INVESTIGATION_PLANNER_PROMPT
    from tiblender_runtime.llm import plan_investigation_queries

    captured = {}

    def fake_chat_completion(settings, messages):
        captured["messages"] = messages
        return (
            '{"queries":['
            '{"perspective":"Infrastructure","tool_type":"WHOIS_LOOKUP","parameters":{"domain":"bad.example"},"priority":9},'
            '{"perspective":"Infrastructure","tool_type":"UNLISTED_LOOKUP","parameters":{"domain":"bad.example"},"priority":9},'
            '{"perspective":"Actor","tool_type":"SEARCH_ENGINE","parameters":{"query":"investigate this"},"priority":5},'
            '{"perspective":"Social","tool_type":"X_SEARCH","parameters":{"query":"bad.example Axios payload"},"priority":7}'
            ']}'
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion", fake_chat_completion)
    settings = Settings(llm_provider="openai", llm_model="model", openai_api_key="key")

    queries = plan_investigation_queries(settings, _llm_cluster())

    assert INVESTIGATION_PLANNER_PROMPT in captured["messages"][0]["content"]
    assert [(query.perspective, query.tool_type, query.parameters) for query in queries] == [
        ("Infrastructure", "WHOIS_LOOKUP", {"domain": "bad.example"}),
        ("Social", "X_SEARCH", {"query": "bad.example Axios payload"}),
    ]


def test_select_focus_types_with_llm_keeps_the_single_valid_focus(monkeypatch):
    """A follow-up that also answers with one focus leaves padding to the caller."""
    from tiblender_runtime.llm import select_focus_types_with_llm

    calls = []

    def fake_chat_completion(settings, messages):
        del settings
        calls.append(messages)
        return (
            '{"selected_focus_types":['
            '{"focus_type":"hosting_and_osint","reasoning":"ok"},'
            '{"focus_type":"not_a_real_focus_type","reasoning":"invalid"}'
            ']}'
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion", fake_chat_completion)
    settings = Settings(llm_provider="openai", llm_model="model", openai_api_key="key")

    selected = select_focus_types_with_llm(settings, _llm_cluster(), "Infrastructure")

    assert selected == ["hosting_and_osint"]
    assert len(calls) == 2, "a single valid focus must trigger exactly one follow-up request"


def test_focus_selection_prompt_lists_every_allowed_id_with_a_description():
    from tiblender_runtime.contract import FOCUS_TYPE_DESCRIPTIONS, REQUIRED_FOCUS_TYPES
    from tiblender_runtime.llm_focus import _focus_selection_messages

    for perspective, focuses in REQUIRED_FOCUS_TYPES.items():
        system = _focus_selection_messages(_llm_cluster(), perspective)[0]["content"]
        for focus in focuses:
            assert f"- {focus}: {FOCUS_TYPE_DESCRIPTIONS[focus]}" in system
        assert f"Select at least 2 and at most {min(4, len(focuses))} focuses, using the ids verbatim" in system
        assert '"selected_focus_types"' in system


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("dns_and_passive_dns_analysis", "dns_and_passive_dns_analysis"),
        ("DNS_AND_PASSIVE_DNS_ANALYSIS", "dns_and_passive_dns_analysis"),
        ("  hosting_and_osint  ", "hosting_and_osint"),
        ("Hosting And Osint", "hosting_and_osint"),
        ("hosting-and-osint", "hosting_and_osint"),
        ("ssl_tls_certificate_analysis.", "ssl_tls_certificate_analysis"),
        ("2. ssl_tls_certificate_analysis", "ssl_tls_certificate_analysis"),
        ("ssl_tls_certificate_analysis_2", "ssl_tls_certificate_analysis"),
        ("ttp_mapping", ""),
        ("not_a_real_focus_type", ""),
        ("", ""),
    ],
)
def test_normalize_focus_name_accepts_ids_labels_and_decorated_ids(raw, expected):
    from tiblender_runtime.llm_focus import normalize_focus_name

    assert normalize_focus_name(raw, "Infrastructure") == expected


def test_select_focus_types_accepts_focuses_written_as_human_labels(monkeypatch):
    from tiblender_runtime.llm import select_focus_types_with_llm

    def fake_chat_completion(settings, messages):
        del settings, messages
        return (
            '{"selected_focus_types":['
            '{"focus_type":"Domain Registration And Whois","reasoning":"ok"},'
            '{"focus_type":"DNS and passive DNS analysis","reasoning":"ok"}'
            ']}'
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion", fake_chat_completion)
    settings = Settings(llm_provider="openai", llm_model="model", openai_api_key="key")

    selected = select_focus_types_with_llm(settings, _llm_cluster(), "Infrastructure")

    assert selected == ["domain_registration_and_whois", "dns_and_passive_dns_analysis"]


def test_single_focus_triggers_one_follow_up_request_over_the_remaining_ids(monkeypatch):
    from tiblender_runtime.llm import select_focus_types_with_llm

    calls = []

    def fake_chat_completion(settings, messages):
        del settings
        calls.append(messages)
        if len(calls) == 1:
            return '{"selected_focus_types":[{"focus_type":"hosting_and_osint","reasoning":"ok"}]}'
        return '{"selected_focus_types":[{"focus_type":"dns_and_passive_dns_analysis","reasoning":"more"}]}'

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion", fake_chat_completion)
    settings = Settings(llm_provider="openai", llm_model="model", openai_api_key="key")

    selected = select_focus_types_with_llm(settings, _llm_cluster(), "Infrastructure")

    assert selected == ["hosting_and_osint", "dns_and_passive_dns_analysis"]
    follow_up = calls[1][0]["content"]
    catalog = follow_up.split("better than none:")[-1]
    assert "hosting_and_osint" not in catalog, "the already-selected focus must not be offered again"
    assert "dns_and_passive_dns_analysis" in catalog


def test_two_valid_focuses_never_trigger_a_follow_up_request(monkeypatch):
    from tiblender_runtime.llm import select_focus_types_with_llm

    calls = []

    def fake_chat_completion(settings, messages):
        del settings
        calls.append(messages)
        return (
            '{"selected_focus_types":['
            '{"focus_type":"hosting_and_osint","reasoning":"ok"},'
            '{"focus_type":"dns_and_passive_dns_analysis","reasoning":"ok"}'
            ']}'
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion", fake_chat_completion)
    settings = Settings(llm_provider="openai", llm_model="model", openai_api_key="key")

    assert len(select_focus_types_with_llm(settings, _llm_cluster(), "Infrastructure")) == 2
    assert len(calls) == 1


def test_a_failing_follow_up_request_leaves_the_single_focus_intact(monkeypatch):
    from tiblender_runtime.llm import select_focus_types_with_llm

    calls = []

    def fake_chat_completion(settings, messages):
        del settings
        calls.append(messages)
        if len(calls) == 1:
            return '{"selected_focus_types":[{"focus_type":"hosting_and_osint","reasoning":"ok"}]}'
        return "not json at all"

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion", fake_chat_completion)
    settings = Settings(llm_provider="openai", llm_model="model", openai_api_key="key")

    assert select_focus_types_with_llm(settings, _llm_cluster(), "Infrastructure") == ["hosting_and_osint"]
    assert len(calls) == 2


def test_select_focus_types_with_llm_raises_only_when_zero_focuses_are_valid(monkeypatch):
    from tiblender_runtime.llm import select_focus_types_with_llm

    def fake_chat_completion(settings, messages):
        del settings, messages
        return '{"selected_focus_types":[{"focus_type":"not_a_real_focus_type","reasoning":"invalid"}]}'

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion", fake_chat_completion)
    settings = Settings(llm_provider="openai", llm_model="model", openai_api_key="key")

    with pytest.raises(LLMGenerationError):
        select_focus_types_with_llm(settings, _llm_cluster(), "Infrastructure")


def test_compose_report_fields_accepts_section_specific_report_json(monkeypatch):
    from tiblender_runtime.llm import compose_report_fields

    def fake_chat_completion(settings, messages):
        del settings, messages
        return (
            '{"cluster_name":"Axios supply-chain payload",'
            '"severity":"HIGH",'
            '"executive_summary":"Axios package consumers should investigate a malicious postinstall payload.",'
            '"key_findings":[{"heading":"Technical","items":["A malicious payload indicator was observed."]}],'
            '"recommendations":[{"action":"Audit recent npm installs.","rationale":"Affected versions were discussed in source posts."}],'
            '"immediate_actions":["Audit recent npm installs."],'
            '"analyst_notes":"Evidence is based on source posts and tool results.",'
            '"threat_actor":"Unknown",'
            '"timeline":[{"timestamp":"2026-04-01T00:00:00Z","event":"Source posts reported the payload.","significance":"Initial social evidence."}],'
            '"ttps":[{"mitre_id":"T1195","tactic":"Initial Access","technique":"Supply Chain Compromise","evidence":"Malicious package behavior was reported."}]}'
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion", fake_chat_completion)
    settings = Settings(llm_provider="openai", llm_model="model", openai_api_key="key")

    fields = compose_report_fields(
        settings,
        _llm_cluster(),
        findings=[],
        decision=EvaluationDecision(
            verdict=EvaluationVerdict.PROCEED,
            rounds=[EvaluationRound(1, "llm-judge-a", EvaluationVerdict.PROCEED, 0.82, "ok")],
            final_quality_score=0.82,
            reason="ok",
        ),
    )

    assert fields["cluster_name"] == "Axios supply-chain payload"
    assert fields["severity"] == "HIGH"
    assert fields["key_findings"][0]["heading"] == "Technical"
    assert fields["recommendations"][0]["rationale"].startswith("Affected versions")
    assert fields["threat_actor"] == "Unknown"
    assert fields["timeline"][0]["event"].startswith("Source posts")
    assert fields["ttps"][0]["mitre_id"] == "T1195"


def test_extract_json_object_wraps_malformed_json_as_llm_error():
    from tiblender_runtime.llm import _extract_json_object

    with pytest.raises(LLMGenerationError, match="not valid JSON"):
        _extract_json_object('{"executive_summary": "x", "immediate_actions": ["missing quote]}')


def test_clean_actions_replaces_non_contract_tool_names():
    from tiblender_runtime.llm import _clean_actions

    disallowed = "UNLISTED" + "_" + "LOOKUP"
    actions = _clean_actions(
        [
            f"Run {disallowed} before publication."
        ]
    )

    assert len(actions) == 1
    assert disallowed not in actions[0]
    assert "WEB_CRAWL" in actions[0]


def test_report_text_rewrites_non_contract_tool_tokens():
    from tiblender_runtime.llm import _sanitize_report_text
    from tiblender_runtime.contract import REQUIRED_TOOL_TYPES

    disallowed = "UNLISTED" + "_" + "FETCH"
    text = _sanitize_report_text(
        f"Run {disallowed} before publication."
    )

    assert disallowed not in text
    assert all(tool in text for tool in REQUIRED_TOOL_TYPES)


def test_report_text_replaces_unknown_tool_tokens_with_tool_allowlist():
    from tiblender_runtime.llm import _sanitize_report_text
    from tiblender_runtime.contract import REQUIRED_TOOL_TYPES

    disallowed = "UNLISTED" + "_" + "RETRIEVE"
    text = _sanitize_report_text(
        f"Run {disallowed} before publication."
    )

    assert disallowed not in text
    assert all(tool in text for tool in REQUIRED_TOOL_TYPES)


# ---------------------------------------------------------------------------
# Concurrent chunk execution (TIBLENDER_LLM_MAX_CONCURRENT)
# ---------------------------------------------------------------------------


def _chunk_post(index: int) -> Post:
    return Post(
        id=f"p{index}",
        platform="x",
        author="a",
        content=f"Threat post {index} mentions bad{index}.example",
        timestamp="2026-09-09T00:00:00Z",
        url="",
        metadata={},
    )


def test_normalize_post_batch_runs_chunks_concurrently_and_keeps_order(monkeypatch):
    import threading

    from tiblender_runtime.llm import normalize_post_batch

    started = threading.Barrier(4, timeout=10)
    seen_threads: set[int] = set()
    lock = threading.Lock()

    def fake_chat_completion_with_meta(settings, messages):
        del settings
        payload = json.loads(messages[1]["content"])
        with lock:
            seen_threads.add(threading.get_ident())
        # Every chunk must be in flight at once, or this times out.
        started.wait()
        return _completion(
            json.dumps(
                {
                    "posts": [
                        {"id": item["id"], "normalized_content": f"Normalized {item['id']}."}
                        for item in payload["posts"]
                    ]
                }
            )
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion_with_meta", fake_chat_completion_with_meta)
    settings = Settings(
        llm_provider="openai",
        llm_model="model",
        openai_api_key="key",
        llm_batch_size=2,
        llm_max_concurrent=8,
    )
    posts = [_chunk_post(index) for index in range(8)]

    normalized = normalize_post_batch(settings, posts)

    assert list(normalized) == [post.id for post in posts]
    assert len(seen_threads) == 4


def test_normalize_post_batch_result_is_identical_serially_and_concurrently(monkeypatch):
    from tiblender_runtime.llm import normalize_post_batch

    def fake_chat_completion_with_meta(settings, messages):
        del settings
        payload = json.loads(messages[1]["content"])
        return _completion(
            json.dumps(
                {
                    "posts": [
                        {"id": item["id"], "normalized_content": f"Normalized {item['id']}."}
                        for item in payload["posts"]
                    ]
                }
            )
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion_with_meta", fake_chat_completion_with_meta)
    posts = [_chunk_post(index) for index in range(9)]

    def run(max_concurrent: int) -> dict:
        return normalize_post_batch(
            Settings(
                llm_provider="openai",
                llm_model="model",
                openai_api_key="key",
                llm_batch_size=2,
                llm_max_concurrent=max_concurrent,
            ),
            posts,
        )

    assert run(8) == run(1)


def test_refine_clusters_with_llm_preserves_chunk_order_under_concurrency(monkeypatch):
    from tiblender_runtime.llm import refine_clusters_with_llm

    def fake_chat_completion_with_meta(settings, messages):
        del settings
        payload = json.loads(messages[1]["content"])
        return _completion(
            json.dumps(
                {
                    "clusters": [
                        {
                            "topic_name": f"Refined {candidate['id']}",
                            "security_severity": "MEDIUM",
                            "topic_description": "Refined campaign unit.",
                            "post_ids": [post["id"] for post in candidate["posts"]],
                        }
                        for candidate in payload["candidate_clusters"]
                    ]
                }
            )
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion_with_meta", fake_chat_completion_with_meta)
    clusters = []
    for index in range(6):
        cluster = _llm_cluster()
        cluster.id = f"cluster-{index}"
        clusters.append(cluster)

    def run(max_concurrent: int) -> list[str]:
        settings = Settings(
            llm_provider="openai",
            llm_model="model",
            openai_api_key="key",
            llm_cluster_batch_size=1,
            llm_max_concurrent=max_concurrent,
        )
        return [refined.topic_name for refined in refine_clusters_with_llm(settings, clusters)]

    concurrent = run(8)
    assert concurrent == [f"Refined cluster-{index}" for index in range(6)]
    assert concurrent == run(1)


def test_fallback_counters_are_correct_when_chunks_fail_concurrently(monkeypatch):
    from tiblender_runtime.llm import normalize_post_batch
    from tiblender_runtime.llm_fallbacks import fallback_counters, reset_fallbacks

    def fake_chat_completion_with_meta(settings, messages):
        del settings, messages
        raise LLMGenerationError("provider refused")

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion_with_meta", fake_chat_completion_with_meta)
    reset_fallbacks()
    settings = Settings(
        llm_provider="openai",
        llm_model="model",
        openai_api_key="key",
        llm_batch_size=2,
        llm_max_concurrent=8,
    )
    posts = [_chunk_post(index) for index in range(16)]

    assert normalize_post_batch(settings, posts) == {}
    assert fallback_counters()["normalization_posts"] == 16
    reset_fallbacks()


def test_zero_valid_focuses_also_trigger_one_follow_up_request(monkeypatch):
    """A thin cluster often gets "no focus applies"; ask once more before failing.

    Verified live: for real clusters with no verified indicators the agent
    answers with an empty selected_focus_types array, and answers with two
    valid ids when the remaining ids are spelled out for it. Treating the
    empty answer as an outright failure is what used to drive the
    perspective_units fallback counter up.
    """
    from tiblender_runtime.llm import select_focus_types_with_llm

    calls = []

    def fake_chat_completion(settings, messages):
        del settings
        calls.append(messages)
        if len(calls) == 1:
            return '{"selected_focus_types":[]}'
        return (
            '{"selected_focus_types":['
            '{"focus_type":"domain_registration_and_whois","reasoning":"closest"},'
            '{"focus_type":"dns_and_passive_dns_analysis","reasoning":"closest"}'
            ']}'
        )

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion", fake_chat_completion)
    settings = Settings(llm_provider="openai", llm_model="model", openai_api_key="key")

    selected = select_focus_types_with_llm(settings, _llm_cluster(), "Infrastructure")

    assert selected == ["domain_registration_and_whois", "dns_and_passive_dns_analysis"]
    assert len(calls) == 2
    assert "You selected no focus at all" in calls[1][0]["content"]


def test_zero_focuses_twice_still_raises_for_the_caller_to_fall_back(monkeypatch):
    from tiblender_runtime.llm import select_focus_types_with_llm

    calls = []

    def fake_chat_completion(settings, messages):
        del settings
        calls.append(messages)
        return '{"selected_focus_types":[]}'

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion", fake_chat_completion)
    settings = Settings(llm_provider="openai", llm_model="model", openai_api_key="key")

    with pytest.raises(LLMGenerationError):
        select_focus_types_with_llm(settings, _llm_cluster(), "Infrastructure")
    assert len(calls) == 2


def test_clean_recommendations_keeps_a_valid_horizon_and_drops_an_invalid_one():
    from tiblender_runtime.llm import _clean_recommendations

    rows = _clean_recommendations(
        [
            {"action": "Block the host.", "rationale": "Observed.", "horizon": "long_term"},
            {"action": "Block the host.", "rationale": "Observed.", "horizon": "eventually"},
            {"action": "Block the host.", "rationale": "Observed."},
        ]
    )

    assert rows[0]["horizon"] == "long_term"
    # An unusable horizon is omitted here so report assembly derives it.
    assert "horizon" not in rows[1]
    assert "horizon" not in rows[2]


def test_report_generation_prompt_requests_the_recommendation_horizon():
    from tiblender_runtime.contract import REPORT_GENERATION_PROMPT

    assert "horizon" in REPORT_GENERATION_PROMPT
    for horizon in ("immediate", "short_term", "long_term"):
        assert horizon in REPORT_GENERATION_PROMPT
