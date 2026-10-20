"""The English-only report guard, on both generation paths.

The corpus this pipeline collects is multilingual; the report it publishes is
not. These tests pin that contract from three directions:

* the guard itself (`tiblender_runtime.language.text_is_english`): what counts
  as prose and what counts as a language-neutral value;
* the offline deterministic path, end to end, on Japanese, Korean, Russian,
  Chinese and Arabic posts with and without indicators;
* the LLM path with a fake provider that answers in the source language --
  one retry carrying the explicit English instruction, then the deterministic
  composer.

The post bodies below are written for this module. They deliberately carry
non-English text, which is the input the guard exists to handle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_fallback_builders import make_cluster, make_decision, make_finding, make_settings
from tiblender_runtime import language
from tiblender_runtime.llm_fallbacks import fallback_counters, fallback_reason_counters, reset_fallbacks
from tiblender_runtime.pipeline import run_cycle
from tiblender_runtime.settings import Settings


@pytest.fixture(autouse=True)
def _clean_fallback_counters():
    reset_fallbacks()
    yield
    reset_fallbacks()


# --- the guard ----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Credential phishing campaign targeting exchange customers.",
        "Block bybiiusdt[.]com and hxxps://kdmpop[.]com/login at the perimeter.",
        "CVE-2026-21509 is exploited in the wild; patch Office endpoints.",
        "Sample 27cc2459" + "a" * 56 + " was uploaded from 203.0.113.9.",
        # A Russian *file name* is an indicator value, not prose: the released
        # dataset carries exactly this case in 39 of its 8,288 reports.
        "RTF file 'Peregovornaya_tablitsa.rtf' and "
        "'Переговорная_таблица.rtf' "
        "are the same lure.",
        "",
        "   ",
    ],
)
def test_english_prose_and_value_tokens_pass_the_guard(text):
    assert language.text_is_english(text)


@pytest.mark.parametrize(
    "text",
    [
        "フィッシングキャンペーンが確認されました。",
        "피싱 캠페인이 확인되었습니다.",
        "Хакеры взломали аккаунт разработчика.",
        "黑客攻击了加密货币交易所。",
        "حملة تصيد جديدة تستهدف المستخدمين.",
        # Mixed: an English sentence that still quotes a whole source sentence.
        "The post reports a phishing campaign: "
        "フィッシングサイトに誘導されます。",
    ],
)
def test_non_english_prose_fails_the_guard(text):
    assert not language.text_is_english(text)


def test_the_guard_skips_identifier_and_evidence_keys():
    section = {
        "social_media_posts": [
            {
                "id": "p1",
                "platform": "telegram",
                "timestamp": "2026-04-01T00:00:00Z",
                # Quoted source evidence: keeps the language it was posted in.
                "text": "フィッシングキャンペーン",
                "summary": "Telegram post from 2026-04-01 naming 1 indicator(s).",
            }
        ]
    }
    assert language.section_is_english(section)

    section["social_media_posts"][0]["summary"] = "フィッシングの投稿"
    assert not language.section_is_english(section)


# --- the offline deterministic path, end to end -------------------------


#: One indicator per language group, so each group clusters on its own.
_MULTILINGUAL_POSTS: list[tuple[str, str, str]] = [
    # (platform, language tag, body)
    ("x", "ja", "フィッシングサイト jp-lure[.]example を確認。利用者は認証情報を入力しないでください。"),
    ("x", "ja", "注意：jp-lure[.]example からのメールが尊大な被害を出しています。CVE-2026-21509 を悪用。"),
    ("telegram", "ja", "jp-lure[.]example のフィッシングキットが拡散中です。CVE-2026-21509 の情報もあります。"),
    ("reddit", "ja", "jp-lure[.]example をブロックしました。同じインフラが別のドメインでも使われています。"),
    ("discord", "ja", "jp-lure[.]example についての報告。CVE-2026-21509 のパッチを当ててください。"),
    ("x", "ko", "피싱 사이트 kr-lure[.]example 가 확인되었습니다. 자격 증명을 입력하지 마십시오."),
    ("telegram", "ko", "kr-lure[.]example 에서 배포되는 악성코드가 확인되었습니다. CVE-2026-21509 악용."),
    ("discord", "ko", "kr-lure[.]example 도메인을 차단했습니다. 동일한 인프라가 계속 사용됩니다."),
    ("telegram", "ru", "Хакеры используют ru-lure[.]example для рассылки вредоносных файлов."),
    ("telegram", "ru", "Файл 'Переговорная_таблица.rtf' с ru-lure[.]example эксплуатирует CVE-2026-21509."),
    ("x", "ru", "Новая волна атак с ru-lure[.]example продолжается вторую неделю."),
    ("x", "zh", "黑客使用 cn-lure[.]example 发起钓鱼攻击，请勿输入凭据。"),
    ("telegram", "zh", "cn-lure[.]example 上发现恶意载荷，利用 CVE-2026-21509 漏洞。"),
    ("discord", "zh", "已封锁 cn-lure[.]example 域名，相关基础设施仍在活动。"),
    ("telegram", "ar", "حملة تصيد تستخدم ar-lure[.]example لسرقة بيانات الاعتماد."),
    ("telegram", "ar", "تم رصد ar-lure[.]example وهو يوزع برمجيات خبيثة عبر CVE-2026-21509."),
    ("x", "ar", "حظرنا النطاق ar-lure[.]example لكن البنية التحتية ما زالت نشطة."),
    # Two posts with no indicator at all, one CJK and one Cyrillic: they must
    # not put source-language text into any report either.
    ("x", "ja", "今日のセキュリティニュースのまとめです。詳細は後ほど共有します。"),
    ("telegram", "ru", "Обзор новостей безопасности за неделю без конкретных индикаторов."),
]


def _write_multilingual_posts(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, (platform, tag, body) in enumerate(_MULTILINGUAL_POSTS, start=1):
            handle.write(
                json.dumps(
                    {
                        "id": f"mlp{index:03d}",
                        "platform": platform,
                        "content": body,
                        "timestamp": f"2026-04-{index % 27 + 1:02d}T09:00:00Z",
                        "metadata": {"language": tag},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


@pytest.fixture(scope="module")
def multilingual_run(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("multilingual")
    input_path = root / "posts.jsonl"
    _write_multilingual_posts(input_path)
    settings = Settings(
        input_path=input_path,
        output_dir=root / "out",
        history_dir=root / "history",
        llm_provider="heuristic",
        allow_heuristic_reports=True,
        enable_real_tools=False,
        min_cluster_size=2,
    )
    run_cycle(settings)
    return root / "out"


def _reports(output_dir: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "reports").glob("*.json"))
    ]


def test_offline_multilingual_posts_produce_reports(multilingual_run: Path):
    assert _reports(multilingual_run), "the multilingual fixture must reach report generation"


def test_every_report_text_section_is_english(multilingual_run: Path):
    for report in _reports(multilingual_run):
        failing = language.failing_sections(report["report_schema"])
        assert not failing, f"{report['report_id']} has non-English section(s): {failing}"
        for section in ("cluster_name", "summary", "threat_actor"):
            assert language.text_is_english(report["report_schema"][section])


def test_source_post_text_is_preserved_in_its_own_language(multilingual_run: Path):
    bodies = {body for _, _, body in _MULTILINGUAL_POSTS}
    published = {
        post["text"]
        for report in _reports(multilingual_run)
        for post in report["sources"]["social_media_posts"]
    }
    assert published, "reports must publish their source posts"
    # Offline, `sources[].text` is the normalized post: a code-authored
    # English prefix around the post body. The body itself must survive
    # verbatim, in the language it was posted in.
    carried = {body for body in bodies if any(body[:40] in text for text in published)}
    assert carried, "no source post body survived into the published report"
    # And at least one of them is genuinely not English -- otherwise this
    # fixture would not be exercising the multilingual path at all.
    assert any(not language.text_is_english(body) for body in carried)
    assert any(not language.text_is_english(text) for text in published)


def test_every_source_post_carries_an_english_summary(multilingual_run: Path):
    for report in _reports(multilingual_run):
        for post in report["sources"]["social_media_posts"]:
            assert post["summary"]
            assert language.text_is_english(post["summary"])


def test_markdown_and_stix_sidecars_are_english_apart_from_source_evidence(multilingual_run: Path):
    for path in sorted((multilingual_run / "markdown").glob("*.md")):
        body = path.read_text(encoding="utf-8")
        # The "Source Evidence" section quotes the posts; everything the
        # pipeline *wrote* comes before it.
        prose = body.split("## Source Evidence")[0]
        assert language.text_is_english(prose), f"{path.name} is not English above Source Evidence"
    for path in sorted((multilingual_run / "stix").glob("*.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        for obj in bundle["objects"]:
            for key in ("name", "description", "labels"):
                assert language.section_is_english(obj.get(key, ""))


def test_pdf_sidecars_are_written(multilingual_run: Path):
    assert sorted((multilingual_run / "pdf").glob("*.pdf"))


def test_cycle_summary_reports_the_language_guard(multilingual_run: Path):
    summary = json.loads((multilingual_run / "cycle_summary.json").read_text(encoding="utf-8"))
    guard = summary["language_guard"]
    assert guard["reports_checked"] == len(_reports(multilingual_run))
    assert guard["reports_rewritten"] >= 0
    assert isinstance(guard["sections_rewritten"], list)


# --- the LLM path -------------------------------------------------------


_JAPANESE_DRAFT = {
    "cluster_name": "フィッシングキャンペーン",
    "severity": "HIGH",
    "executive_summary": "このレポートはフィッシングキャンペーンについて説明します。",
    "immediate_actions": ["ドメインをブロックしてください。"],
    "analyst_notes": "証拠は十分です。",
    "threat_actor": "Unknown",
    "key_findings": [],
    "recommendations": [],
    "timeline": [],
    "ttps": [],
}

_RUSSIAN_DRAFT = {
    **_JAPANESE_DRAFT,
    "cluster_name": "Фишинговая кампания",
    "executive_summary": "В отчёте описана фишинговая кампания против клиентов бирж.",
    "immediate_actions": ["Заблокируйте домен."],
}

_KOREAN_DRAFT = {
    **_JAPANESE_DRAFT,
    "cluster_name": "피싱 캠페인",
    "executive_summary": "이 보고서는 피싱 캠페인을 설명합니다.",
    "immediate_actions": ["도메인을 차단하십시오."],
}

_ENGLISH_DRAFT = {
    "cluster_name": "Credential phishing campaign",
    "severity": "HIGH",
    "executive_summary": "Exchange customers are targeted by a credential phishing campaign.",
    "immediate_actions": ["Block the reported domain at the perimeter."],
    "analyst_notes": "Evidence is sufficient.",
    "threat_actor": "Unknown",
    "key_findings": [],
    "recommendations": [],
    "timeline": [],
    "ttps": [],
}


def _fake_provider(monkeypatch, drafts: list[dict]) -> list[list[dict[str, str]]]:
    """Answer each call with the next draft; record the messages sent."""
    sent: list[list[dict[str, str]]] = []
    queue = list(drafts)

    def fake_completion(settings, messages):
        sent.append(messages)
        return json.dumps(queue.pop(0) if queue else drafts[-1], ensure_ascii=False)

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion", fake_completion)
    return sent


@pytest.mark.parametrize("draft", [_JAPANESE_DRAFT, _RUSSIAN_DRAFT, _KOREAN_DRAFT])
def test_a_non_english_draft_is_retried_with_the_explicit_instruction(monkeypatch, draft):
    from tiblender_runtime.contract import ENGLISH_ONLY_RETRY_INSTRUCTION
    from tiblender_runtime.llm_agents import compose_report_fields

    sent = _fake_provider(monkeypatch, [draft, _ENGLISH_DRAFT])
    fields = compose_report_fields(make_settings(), make_cluster(), [make_finding()], make_decision())

    assert len(sent) == 2, "a non-English draft must be retried exactly once"
    assert ENGLISH_ONLY_RETRY_INSTRUCTION not in sent[0][0]["content"]
    assert ENGLISH_ONLY_RETRY_INSTRUCTION in sent[1][0]["content"]
    assert fields["executive_summary"] == _ENGLISH_DRAFT["executive_summary"]
    assert fields["metadata"]["generation_mode"] == "llm"
    assert fallback_counters()["report_fields"] == 0


@pytest.mark.parametrize("draft", [_JAPANESE_DRAFT, _RUSSIAN_DRAFT, _KOREAN_DRAFT])
def test_a_second_non_english_draft_falls_back_to_the_english_composer(monkeypatch, draft):
    from tiblender_runtime.llm_agents import compose_report_fields

    sent = _fake_provider(monkeypatch, [draft, draft])
    fields = compose_report_fields(make_settings(), make_cluster(), [make_finding()], make_decision())

    assert len(sent) == 2
    assert fields["metadata"]["generation_mode"] == "offline_deterministic"
    assert language.text_is_english(fields["executive_summary"])
    assert not language.failing_report_fields(fields)
    assert fallback_counters()["report_fields"] == 1
    assert fallback_reason_counters()["non_english_output"] == 1


def test_an_english_draft_is_never_retried(monkeypatch):
    from tiblender_runtime.llm_agents import compose_report_fields

    sent = _fake_provider(monkeypatch, [_ENGLISH_DRAFT])
    fields = compose_report_fields(make_settings(), make_cluster(), [make_finding()], make_decision())

    assert len(sent) == 1
    assert fields["cluster_name"] == _ENGLISH_DRAFT["cluster_name"]
    assert fallback_counters()["report_fields"] == 0


def test_a_non_english_section_that_survives_is_rewritten_at_assembly(monkeypatch):
    """The last net: a section that reaches ``build_report`` in another
    language is replaced by the composer and the report says so."""
    from tiblender_runtime.reporting import build_report

    cluster = make_cluster()
    report = build_report(
        cluster,
        [make_finding()],
        make_decision(),
        report_fields={
            "cluster_name": "フィッシングキャンペーン",
            "executive_summary": "このレポートはキャンペーンを説明します。",
            "threat_actor": "不明な攻撃者",
        },
    )

    assert report["metadata"]["language_fallback"] is True
    assert set(report["metadata"]["language_fallback_sections"]) >= {"cluster_name", "summary", "threat_actor"}
    assert not language.failing_sections(report["report_schema"])
    # The root-level views follow the rewritten schema.
    assert report["cluster_name"] == report["report_schema"]["cluster_name"]
    assert report["executive_summary"] == report["report_schema"]["summary"]
    assert language.text_is_english(report["markdown_report"].split("## Source Evidence")[0])


def test_an_english_report_is_not_marked_as_a_language_fallback():
    from tiblender_runtime.reporting import build_report

    report = build_report(make_cluster(), [make_finding()], make_decision())

    assert "language_fallback" not in report["metadata"]
    assert not language.failing_sections(report["report_schema"])


# --- the committed expected output --------------------------------------


def test_the_committed_sample_report_is_english():
    """The bundled demo may draw on non-English posts; its reports may not."""
    root = Path("sample_runs/collector_experiment_sample/sample_run_output/reports")
    reports = sorted(root.glob("*.json"))
    assert reports, "the committed sample run must carry at least one report"
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        failing = language.failing_sections(report["report_schema"])
        assert not failing, f"{path.name} has non-English section(s): {failing}"
