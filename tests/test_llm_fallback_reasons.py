"""Fallback reason categories, the logging contract, and concurrency.

The companion module `test_llm_fallbacks.py` covers chunking, truncation
handling and per-report attribution. This one covers *why* a fallback was
recorded, what may and may not appear in the log, and that several workers
failing at once still produce a deterministic result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_fallback_builders import make_cluster, make_decision, make_finding, make_posts, make_settings
from tiblender_runtime.llm import ChatCompletion, LLMGenerationError
from tiblender_runtime.llm_fallbacks import fallback_counters, reset_fallbacks, stages_for_cluster


@pytest.fixture(autouse=True)
def _clean_fallback_counters():
    reset_fallbacks()
    yield
    reset_fallbacks()


# --- fallback reasons ---------------------------------------------------
#
# The category is authored by this code where the failure is raised, never
# inferred from the failure text: that text carries the provider's reply and
# the model's own draft, and the draft summarises scraped, adversary-authored
# posts. A post is untrusted input to the classifier if the classifier reads
# the text at all, so it does not.


def test_the_category_comes_from_the_raise_site_not_the_message():
    from tiblender_runtime.llm_fallbacks import category_of

    assert category_of(LLMGenerationError("anything at all", category="http_status")) == "http_status"
    # A raise site that forgot, an unknown category, and a third-party
    # exception all land in "unknown" rather than being guessed.
    assert category_of(LLMGenerationError("LLM provider returned HTTP 500: boom")) == "unknown"
    assert category_of(LLMGenerationError("x", category="not-a-category")) == "unknown"
    assert category_of(ValueError("the request timed out")) == "unknown"
    assert category_of(None) == "unknown"


def test_every_llm_generation_error_raised_by_the_runtime_declares_a_category():
    """A new raise site cannot silently fall into ``unknown``."""
    import ast

    def names_the_error(func: ast.expr) -> bool:
        # Both call shapes count: the bare name this package uses today, and
        # the attribute form (`llm.LLMGenerationError(...)`) a future module
        # could import instead, which would otherwise escape the guard.
        if isinstance(func, ast.Name):
            return func.id == "LLMGenerationError"
        return isinstance(func, ast.Attribute) and func.attr == "LLMGenerationError"

    # No attribute-form site exists today, so only a direct probe proves the
    # matcher would catch one.
    probe = ast.parse("_llm.LLMGenerationError('x')").body[0].value
    assert names_the_error(probe.func)
    assert names_the_error(ast.parse("LLMGenerationError('x')").body[0].value.func)
    assert not names_the_error(ast.parse("ValueError('x')").body[0].value.func)

    missing = []
    for path in sorted(Path("tiblender_runtime").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and names_the_error(node.func)
                and not any(keyword.arg == "category" for keyword in node.keywords)
            ):
                missing.append(f"{path}:{node.lineno}")

    assert missing == [], f"LLMGenerationError raised without a category: {missing}"


def test_a_post_body_that_reads_like_another_category_cannot_move_the_histogram(monkeypatch):
    """Scraped post text inside a JSON-parse failure stays ``json_parse``.

    The model's draft is a summary of collected posts. A post that says
    "timed out" (or "was truncated", or "must include") used to be enough to
    move a counter in the histogram this artifact publishes, because the
    classifier matched markers anywhere in the failure message.
    """
    from tiblender_runtime.llm import compose_report_fields
    from tiblender_runtime.llm_fallbacks import fallback_reason_counters

    adversary_text = (
        "The actor advertised a timeout bypass kit; the panel timed out and the "
        "listing was truncated, buyers must include a referral code."
    )
    monkeypatch.setattr(
        "tiblender_runtime.llm._chat_completion",
        lambda settings, messages: adversary_text,
    )

    compose_report_fields(make_settings(), make_cluster(), [make_finding()], make_decision())

    reasons = fallback_reason_counters()
    assert reasons["json_parse"] == 1
    assert reasons["timeout"] == 0
    assert reasons["truncated"] == 0
    assert reasons["invalid_response"] == 0


def test_an_http_error_body_that_mentions_a_timeout_stays_http_status(monkeypatch):
    import urllib.error
    from tiblender_runtime.llm import _chat_completion

    class _Body:
        @staticmethod
        def read():
            return b'{"error":{"message":"Request timed out upstream"}}'

    def raise_http(request, timeout):
        del request, timeout
        raise urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", raise_http)
    monkeypatch.setattr(urllib.error.HTTPError, "read", lambda self: _Body.read(), raising=False)

    with pytest.raises(LLMGenerationError) as caught:
        _chat_completion(make_settings(), [{"role": "user", "content": "x"}])

    assert caught.value.category == "http_status"


def test_a_read_timeout_is_a_timeout_and_a_malformed_body_is_an_invalid_response(monkeypatch):
    from tiblender_runtime.llm import _request_json

    def raise_timeout(request, timeout):
        del request, timeout
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", raise_timeout)
    with pytest.raises(LLMGenerationError) as caught:
        _request_json("http://provider.invalid", {}, {}, 5)
    assert caught.value.category == "timeout"

    class _Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def read():
            return b"not json at all"

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: _Reply())
    with pytest.raises(LLMGenerationError) as caught:
        _request_json("http://provider.invalid", {}, {}, 5)
    # A 2xx reply with a broken body is a malformed response, not a transport
    # failure -- it must not be counted as `network`.
    assert caught.value.category == "invalid_response"


def test_a_non_utf8_2xx_body_is_an_invalid_response_not_an_abort(monkeypatch):
    """A 2xx body that is not UTF-8 must fall back, not crash the stage.

    ``bytes.decode("utf-8")`` raises ``UnicodeDecodeError``, which none of the
    categorised arms catch, so it used to escape ``_request_json`` uncaught and
    abort the cycle instead of falling back with a reason.
    """
    from tiblender_runtime.llm import _request_json

    class _Latin1Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def read():
            # A UTF-16 (or otherwise mis-encoded) body: invalid UTF-8 bytes.
            return '{"choices": []}'.encode("utf-16")

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: _Latin1Reply())
    with pytest.raises(LLMGenerationError) as caught:
        _request_json("http://provider.invalid", {}, {}, 5)
    assert caught.value.category == "invalid_response"


def test_a_connection_failure_is_network(monkeypatch):
    from tiblender_runtime.llm import _request_json

    def raise_oserror(request, timeout):
        del request, timeout
        raise OSError("[Errno 111] Connection refused")

    monkeypatch.setattr("urllib.request.urlopen", raise_oserror)
    with pytest.raises(LLMGenerationError) as caught:
        _request_json("http://provider.invalid", {}, {}, 5)
    assert caught.value.category == "network"


def test_a_provider_that_omits_post_ids_is_an_invalid_response_not_unknown(monkeypatch):
    """The most common normalization fallback must not land in ``unknown``.

    A well-formed reply that simply leaves some post ids out raises no error
    at all, so there is nothing to take a category from; the site names one.
    """
    from tiblender_runtime.llm import normalize_post_batch
    from tiblender_runtime.llm_fallbacks import fallback_reason_counters

    monkeypatch.setattr(
        "tiblender_runtime.llm._chat_completion_with_meta",
        lambda settings, messages: ChatCompletion(content='{"posts": []}', finish_reason="stop"),
    )

    normalize_post_batch(make_settings(llm_batch_size=4), make_posts(2))

    assert fallback_counters()["normalization_posts"] == 2
    assert fallback_reason_counters()["invalid_response"] == 2
    assert fallback_reason_counters()["unknown"] == 0


def test_a_truncated_normalization_chunk_is_categorised_as_truncated(monkeypatch):
    from tiblender_runtime.llm import normalize_post_batch
    from tiblender_runtime.llm_fallbacks import fallback_reason_counters

    monkeypatch.setattr(
        "tiblender_runtime.llm._chat_completion_with_meta",
        lambda settings, messages: ChatCompletion(
            content='{"posts":[{"id":"p1","normalized_', finish_reason="length"
        ),
    )

    normalize_post_batch(make_settings(llm_batch_size=4), make_posts(4))

    assert fallback_reason_counters()["truncated"] == 4


def test_the_reason_histogram_totals_the_per_stage_fallback_counters():
    """The documented invariant, stated exactly.

    ``llm_fallbacks`` also carries two bookkeeping counters that are not stage
    counters -- ``perspective_focus_padded`` (padding is not a fallback, so it
    has no cause) and ``unattributed_fallbacks`` (a flag over the fallbacks
    already counted) -- so the invariant is over ``FALLBACK_KEYS`` only.
    """
    from tiblender_runtime.llm_fallbacks import (
        FALLBACK_KEYS,
        REASON_KEYS,
        UNATTRIBUTED_KEY,
        fallback_reason_counters,
        record_fallback,
        record_padding,
    )

    record_fallback("normalization_posts", count=4, post_ids=["p1"], category="truncated", summary="chunk")
    record_fallback("planner_clusters", cluster_id="c1", category="http_status", summary="planner")
    record_fallback("report_fields", cluster_id="c1", category="json_parse", summary="report")
    record_padding("perspective_focus_padded", cluster_id="c1")

    reasons = fallback_reason_counters()
    counters = fallback_counters()

    assert sorted(reasons) == sorted(REASON_KEYS)  # every key always present
    assert reasons["truncated"] == 4
    assert reasons["http_status"] == 1
    assert reasons["json_parse"] == 1
    assert sum(reasons.values()) == sum(counters[key] for key in FALLBACK_KEYS) == 6
    # Padding is counted but is not a fallback, so it is outside the invariant.
    assert counters["perspective_focus_padded"] == 1
    assert counters[UNATTRIBUTED_KEY] == 0


def test_padding_is_absent_from_the_reason_histogram():
    from tiblender_runtime.llm_fallbacks import fallback_reason_counters, record_padding

    record_padding("perspective_focus_padded", cluster_id="c1")

    assert sum(fallback_reason_counters().values()) == 0


def test_a_fallback_with_no_scope_is_counted_as_unattributed_not_dropped(caplog):
    """An unscoped fallback can reach no report, so it must be visible.

    That invisibility is exactly what the old cycle-global attribution had;
    a future call site that forgets its scope has to be loud, not silent.
    """
    from tiblender_runtime.llm_fallbacks import UNATTRIBUTED_KEY, record_fallback

    with caplog.at_level("DEBUG", logger="tiblender_runtime.llm_fallbacks"):
        record_fallback("planner_clusters", category="json_parse", summary="planner draft")

    assert fallback_counters()[UNATTRIBUTED_KEY] == 1
    assert stages_for_cluster("any-cluster", ["any-post"]) == []
    warning = next(record for record in caplog.records if record.levelname == "WARNING")
    assert "<unattributed>" in warning.getMessage()
    assert any(record.levelname == "ERROR" for record in caplog.records)


def test_a_scoped_fallback_is_not_counted_as_unattributed():
    from tiblender_runtime.llm_fallbacks import UNATTRIBUTED_KEY, record_fallback

    record_fallback("planner_clusters", cluster_id="c1", category="json_parse", summary="planner draft")

    assert fallback_counters()[UNATTRIBUTED_KEY] == 0


# --- logging ------------------------------------------------------------


def test_the_warning_line_is_content_free_and_bounded(caplog):
    """WARNING must never carry the model draft or post text; DEBUG may."""
    from tiblender_runtime.llm_fallbacks import record_fallback

    post_text = "victim ACME Corp paid the ransom; contact darkmarket@example.onion for the decryptor"
    provider_detail = f"LLM report draft is not JSON: {post_text}"

    with caplog.at_level("DEBUG", logger="tiblender_runtime.llm_fallbacks"):
        record_fallback(
            "report_fields",
            cluster_id="cluster-a",
            category="json_parse",
            summary="report field composition rejected after one retry",
            detail=provider_detail,
        )

    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "report_fields" in message
    assert "reason=json_parse" in message
    assert "cluster-a" in message
    # Nothing the provider or the model said reaches WARNING.
    assert post_text not in message
    assert "ACME" not in message
    assert "darkmarket" not in message
    assert provider_detail not in message

    debug = [record for record in caplog.records if record.levelname == "DEBUG"]
    assert any(provider_detail in record.getMessage() for record in debug)


def test_the_formatter_sanitises_a_stage_or_category_it_did_not_author():
    """Every field on the line is code-authored, including the two names.

    ``format_fallback_line`` is the single place a fallback WARNING is built,
    so a caller passing through a stage or category it read from somewhere
    else must not be able to put free text on that line.
    """
    from tiblender_runtime.llm_fallbacks import format_fallback_line

    line = format_fallback_line(
        "victim ACME Corp <injected stage>",
        category="contact darkmarket@example.onion",
        cluster_ids=["c1"],
        post_count=2,
        summary="report field composition rejected after one retry",
    )

    assert "ACME" not in line
    assert "darkmarket" not in line
    assert "unknown_stage" in line
    assert "reason=unknown" in line
    assert "clusters=c1" in line and "posts=2" in line

    # Scope wording: no cluster and no post is called out, not printed as "-".
    assert "clusters=<unattributed>" in format_fallback_line("report_fields", category="json_parse")


def test_an_over_long_summary_is_truncated_on_the_warning_line(caplog):
    from tiblender_runtime.llm_fallbacks import SUMMARY_MAX_CHARS, record_fallback

    with caplog.at_level("WARNING", logger="tiblender_runtime.llm_fallbacks"):
        record_fallback(
            "report_fields",
            cluster_id="c1",
            category="json_parse",
            summary="x" * (SUMMARY_MAX_CHARS + 50),
        )

    message = caplog.records[0].getMessage()
    assert "x" * SUMMARY_MAX_CHARS in message
    assert "x" * (SUMMARY_MAX_CHARS + 1) not in message


def test_every_fallback_logs_exactly_one_warning_naming_its_scope(caplog):
    from tiblender_runtime.llm_fallbacks import record_fallback

    with caplog.at_level("WARNING", logger="tiblender_runtime.llm_fallbacks"):
        record_fallback(
            "cluster_refinement_batches",
            cluster_ids=["cluster-a", "cluster-b"],
            post_ids=["p1"],
            category="http_status",
            summary="cluster refinement batch rejected after one retry",
        )

    records = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "cluster_refinement" in message
    assert "reason=http_status" in message
    assert "cluster-a,cluster-b" in message
    assert "posts=1" in message


# --- concurrency --------------------------------------------------------


def test_concurrent_workers_one_failure_gives_a_deterministic_result():
    """Several workers record at once; the result must not depend on timing.

    Perspective agents and cluster chunks both run in thread pools, so
    `record` is called concurrently. Counters must be exact and
    `stages_for_cluster` must return pipeline order, not arrival order.
    """
    from concurrent.futures import ThreadPoolExecutor
    from tiblender_runtime.llm_fallbacks import fallback_reason_counters, record_fallback

    # One failing unit per worker, deliberately recorded in reverse pipeline
    # order so arrival order and pipeline order disagree.
    units = [
        ("report_fields", "json_parse"),
        ("judge_rounds", "http_status"),
        ("perspective_units", "timeout"),
        ("planner_clusters", "truncated"),
    ]
    orderings = set()
    for _trial in range(20):
        reset_fallbacks()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(
                    lambda unit: record_fallback(
                        unit[0], cluster_id="cluster-hot", category=unit[1], summary=unit[0]
                    ),
                    units * 4,
                )
            )
        assert fallback_counters()["report_fields"] == 4
        assert sum(fallback_reason_counters().values()) == 16
        orderings.add(tuple(stages_for_cluster("cluster-hot")))

    assert orderings == {
        ("investigation_planning", "perspective_queries", "evaluation_judges", "report_fields")
    }


# --- no fallback log line anywhere carries the model's draft --------------
#
# The WARNING contract is a property of the *package*, not of
# ``record_fallback`` alone: a call site that logs the LLMGenerationError
# itself puts the model's draft -- up to 500 characters summarising scraped,
# adversary-authored posts -- back on a WARNING line for the same event whose
# accounting line was deliberately made content-free.

_SENTINEL = "SENTINELvictimACMECorppaidcontactdarkmarketexampleonion"


def _poisoned_draft_error() -> LLMGenerationError:
    """An error whose message embeds the model's draft, as a real one does."""
    return LLMGenerationError(f"LLM draft is not JSON: {_SENTINEL}", category="json_parse")


def _drive_normalization(monkeypatch) -> None:
    from tiblender_runtime.llm import normalize_post_batch

    def failing(settings, messages):
        del settings, messages
        raise _poisoned_draft_error()

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion_with_meta", failing)
    normalize_post_batch(make_settings(llm_batch_size=2), make_posts(2))


def _drive_cluster_refinement(monkeypatch) -> None:
    from tiblender_runtime.llm import refine_clusters_with_llm

    def failing(settings, messages):
        del settings, messages
        raise _poisoned_draft_error()

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion_with_meta", failing)
    refine_clusters_with_llm(make_settings(), [make_cluster()])


def _drive_planner(monkeypatch) -> None:
    from tiblender_runtime.llm import plan_investigation_queries

    def failing(settings, messages):
        del settings, messages
        raise _poisoned_draft_error()

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion", failing)
    plan_investigation_queries(make_settings(), make_cluster())


def _drive_perspective_focus(monkeypatch) -> None:
    from tiblender_runtime.perspectives import select_focus_types

    def failing(settings, cluster, perspective):
        del settings, cluster, perspective
        raise _poisoned_draft_error()

    monkeypatch.setattr("tiblender_runtime.perspectives.select_focus_types_with_llm", failing)
    select_focus_types(make_cluster(), "Infrastructure", hints=[], settings=make_settings())


def _drive_perspective_queries(monkeypatch) -> None:
    from tiblender_runtime.perspectives import build_perspective_queries

    def failing(settings, cluster, perspective, focus_types):
        del settings, cluster, perspective, focus_types
        raise _poisoned_draft_error()

    monkeypatch.setattr(
        "tiblender_runtime.perspectives.generate_perspective_queries_with_llm", failing
    )
    build_perspective_queries(
        make_cluster(),
        "Infrastructure",
        ["domain_registration_and_whois"],
        planned=[],
        settings=make_settings(),
    )


def _drive_judges(monkeypatch) -> None:
    from tiblender_runtime.evaluation import evaluate_cluster

    def failing(settings, messages):
        del settings, messages
        raise _poisoned_draft_error()

    monkeypatch.setattr("tiblender_runtime.evaluation._chat_completion", failing)
    evaluate_cluster(make_cluster(), [make_finding()], max_rounds=1, settings=make_settings())


def _drive_report_fields(monkeypatch) -> None:
    from tiblender_runtime.llm import compose_report_fields

    def failing(settings, messages):
        del settings, messages
        raise _poisoned_draft_error()

    monkeypatch.setattr("tiblender_runtime.llm._chat_completion", failing)
    compose_report_fields(make_settings(), make_cluster(), [make_finding()], make_decision())


def _drive_dynamic_queries(monkeypatch) -> None:
    from tiblender_runtime import clock
    from tiblender_runtime.collectors.keywords import KeywordCatalog
    from tiblender_runtime.dynamic_queries import generate_dynamic_queries
    from tiblender_runtime.trends import TrendSignal, snapshot_cluster

    cluster = make_cluster()
    snapshot = snapshot_cluster(
        cluster,
        cycle_index=1,
        observed_at="2026-04-01T00:00:00Z",
        strategic_value=0.9,
        embed=lambda text: [0.0, 1.0],
    )
    signal = TrendSignal(
        cluster_id=cluster.id, trend_types=["EARLY_WARNING"], adjacent_history=1, evidence={}
    )

    def failing(settings, payload):
        del settings, payload
        raise _poisoned_draft_error()

    monkeypatch.setattr("tiblender_runtime.llm_feedback.generate_dynamic_queries_with_llm", failing)
    _queries, signals = generate_dynamic_queries(
        [signal],
        {cluster.id: snapshot},
        {cluster.id: cluster},
        {},
        settings=make_settings(),
        catalog=KeywordCatalog.load("config/keywords.json"),
        now=clock.now(),
    )
    # The marker persisted on the signal is code-authored too, not the draft.
    assert _SENTINEL not in str(signals[0].evidence.get("llm_fallback", ""))


def _drive_strategic_value(monkeypatch) -> None:
    from tiblender_runtime.feedback import _strategic_value_for

    def failing(settings, payload):
        del settings, payload
        raise _poisoned_draft_error()

    monkeypatch.setattr("tiblender_runtime.llm_feedback.strategic_value_with_llm", failing)
    _strategic_value_for(make_cluster(), make_decision(), {}, make_settings())


_FALLBACK_SITES = {
    "normalization": _drive_normalization,
    "cluster_refinement": _drive_cluster_refinement,
    "planner": _drive_planner,
    "perspective_focus": _drive_perspective_focus,
    "perspective_queries": _drive_perspective_queries,
    "judges": _drive_judges,
    "report_fields": _drive_report_fields,
    "dynamic_queries": _drive_dynamic_queries,
    "strategic_value": _drive_strategic_value,
}


@pytest.mark.parametrize("site", sorted(_FALLBACK_SITES))
def test_no_fallback_site_puts_the_model_draft_on_a_warning(site, monkeypatch, caplog):
    """Every stage, driven with a poisoned draft, stays content-free at WARNING.

    The draft the sentinel stands for is the model's summary of scraped,
    adversary-authored posts, so it must not reach a level an operator would
    ship to a shared log. It must still reach DEBUG, or the failure would be
    undiagnosable.
    """
    import logging

    with caplog.at_level(logging.DEBUG):
        _FALLBACK_SITES[site](monkeypatch)

    loud = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert loud, "the fallback must still be announced at WARNING"
    assert all(_SENTINEL not in record.getMessage() for record in loud), [
        record.getMessage() for record in loud if _SENTINEL in record.getMessage()
    ]
    assert any(
        record.levelno == logging.DEBUG and _SENTINEL in record.getMessage()
        for record in caplog.records
    ), "the draft must still be diagnosable at DEBUG"


def test_a_perspective_fallback_note_that_reaches_a_report_is_content_free(monkeypatch):
    """`notes` becomes `PerspectiveFinding.insights`, which is published.

    Those notes used to carry ``f"{first_error} / retry: {error}"`` -- the
    same up-to-500-character model draft the WARNING line forbids, but in a
    persisted artifact rather than a log.
    """
    from tiblender_runtime.perspectives import (
        _failed_finding,
        _focus_types_with_notes,
        _perspective_queries_with_notes,
    )

    def failing_focus(settings, cluster, perspective):
        del settings, cluster, perspective
        raise _poisoned_draft_error()

    def failing_queries(settings, cluster, perspective, focus_types):
        del settings, cluster, perspective, focus_types
        raise _poisoned_draft_error()

    monkeypatch.setattr("tiblender_runtime.perspectives.select_focus_types_with_llm", failing_focus)
    monkeypatch.setattr(
        "tiblender_runtime.perspectives.generate_perspective_queries_with_llm", failing_queries
    )

    focuses, focus_notes = _focus_types_with_notes(
        make_cluster(), "Infrastructure", hints=[], settings=make_settings()
    )
    _queries, query_notes = _perspective_queries_with_notes(
        make_cluster(), "Infrastructure", focuses, planned=[], settings=make_settings()
    )

    notes = [*focus_notes, *query_notes]
    assert len(notes) == 2
    assert all(_SENTINEL not in note for note in notes), notes
    # The note still says which stage fell back and why, from code-authored
    # words only: the stage name, the perspective and the raise-site category.
    assert any("focus selection fell back" in note and "json_parse" in note for note in notes)
    assert any("query generation fell back" in note and "json_parse" in note for note in notes)

    # The same rule covers a whole perspective agent that raised: the report
    # gets the exception type, never the message.
    insights = _failed_finding("Infrastructure", "multi", _poisoned_draft_error()).insights
    assert all(_SENTINEL not in insight for insight in insights), insights
    assert any("LLMGenerationError" in insight for insight in insights)
