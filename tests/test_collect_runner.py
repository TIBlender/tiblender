import json
from datetime import UTC, datetime
from pathlib import Path

from tiblender_runtime.cli import main
from tiblender_runtime.collectors.base import CollectionWindow, RawPost
from tiblender_runtime.collectors.corpus import (
    dedupe_posts,
    to_post_record,
    write_collection_report,
    write_posts_jsonl,
)
from tiblender_runtime.collectors.runner import collect_cycle
from tiblender_runtime.settings import Settings

# Two of the real static x queries from config/keywords.json, used to trigger
# a genuine same-platform duplicate (the same tweet id returned for two
# distinct queries) rather than relying on the fixture transport's "any
# response for this endpoint" fallback.
_X_QUERY_VULNERABILITY = '(CVE OR "zero-day" OR "exploit PoC") -is:retweet -is:quote'
_X_QUERY_MALWARE = '(stealer OR ransomware OR botnet OR "C2 panel") -is:retweet -is:quote'

_WINDOW = CollectionWindow(
    start=datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC),
    end=datetime(2026, 9, 1, 1, 0, 0, tzinfo=UTC),
    cycle_index=1,
    label="cycle_001",
)

_FIXTURE_WINDOW = {"start": "2026-09-01T00:00:00Z", "end": "2026-09-01T01:00:00Z", "cycle_index": 1, "label": "cycle_001"}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _x_fixture_path(fixtures_dir: Path, cycle: int = 1) -> Path:
    return fixtures_dir / "collectors" / "x" / f"cycle_{cycle:03d}.json"


def _write_x_fixture(fixtures_dir: Path, cycle: int = 1) -> None:
    payload = {
        "platform": "x",
        "window": _FIXTURE_WINDOW,
        "provenance": "test fixture",
        "responses": [
            {
                "request": {"endpoint": "tweets/search/recent", "query": "artifact"},
                "payload": {
                    "data": [
                        {
                            "id": "1001",
                            "author_id": "u1",
                            "created_at": "2026-09-01T00:15:00Z",
                            "lang": "en",
                            "text": "Security team confirms CVE-2026-12345 actively exploited in the wild",
                        }
                    ],
                    "includes": {"users": [{"id": "u1", "username": "sec_analyst"}]},
                },
            }
        ],
    }
    _write_json(_x_fixture_path(fixtures_dir, cycle), payload)


def _duplicate_tweet_response(query: str) -> dict:
    return {
        "request": {"endpoint": "tweets/search/recent", "query": query},
        "payload": {
            "data": [
                {
                    "id": "9001",
                    "author_id": "u9",
                    "created_at": "2026-09-01T00:15:00Z",
                    "lang": "en",
                    "text": "CVE-2026-12345 zero-day exploit PoC released for a stealer campaign",
                }
            ],
            "includes": {"users": [{"id": "u9", "username": "sec9"}]},
        },
    }


def _write_x_fixture_with_same_platform_duplicate(fixtures_dir: Path, cycle: int = 1) -> None:
    # Two distinct, exact-match queries both return the same tweet id, which
    # is the real trigger for a same-platform duplicate: post_id is
    # platform-scoped, so the two RawPost objects collapse to one on dedupe.
    payload = {
        "platform": "x",
        "window": _FIXTURE_WINDOW,
        "provenance": "test fixture",
        "responses": [
            _duplicate_tweet_response(_X_QUERY_VULNERABILITY),
            _duplicate_tweet_response(_X_QUERY_MALWARE),
        ],
    }
    _write_json(_x_fixture_path(fixtures_dir, cycle), payload)


def _write_reddit_fixture(fixtures_dir: Path, cycle: int = 1) -> None:
    payload = {
        "platform": "reddit",
        "window": _FIXTURE_WINDOW,
        "provenance": "test fixture",
        "responses": [
            {
                "request": {"endpoint": "r/{sub}/new", "subreddit": "netsec"},
                "payload": {
                    "kind": "Listing",
                    "data": {
                        "children": [
                            {
                                "kind": "t3",
                                "data": {
                                    "id": "r1",
                                    "title": "New phishing kit targeting banks discovered this week",
                                    "selftext": "",
                                    "created_utc": 1788222000,
                                    "author": "analyst1",
                                    "subreddit": "netsec",
                                    "permalink": "/r/netsec/comments/r1/x/",
                                    "stickied": False,
                                },
                            }
                        ],
                        "after": None,
                    },
                },
            }
        ],
    }
    _write_json(fixtures_dir / "collectors" / "reddit" / f"cycle_{cycle:03d}.json", payload)


def _write_telegram_fixture(fixtures_dir: Path, cycle: int = 1) -> None:
    payload = {
        "platform": "telegram",
        "window": _FIXTURE_WINDOW,
        "provenance": "test fixture",
        "responses": [
            {
                "request": {"endpoint": "dialogs"},
                "payload": {"chats": [{"id": 555, "username": "seclogschannel", "title": "Sec Logs"}]},
            },
            {
                "request": {"endpoint": "messages", "channel_id": "555"},
                "payload": [
                    {
                        "id": 10,
                        "date": "2026-09-01T00:25:00Z",
                        "message": "Fresh stealer logs dump posted, references CVE-2026-99999",
                        "views": 10,
                        "forwards": 0,
                    }
                ],
            },
        ],
    }
    _write_json(fixtures_dir / "collectors" / "telegram" / f"cycle_{cycle:03d}.json", payload)


def _write_discord_fixture(fixtures_dir: Path, cycle: int = 1) -> None:
    payload = {
        "platform": "discord",
        "window": _FIXTURE_WINDOW,
        "provenance": "test fixture",
        "responses": [
            {
                "request": {"endpoint": "users/@me/guilds"},
                "payload": [{"id": "700", "name": "Sec Guild"}],
            },
            {
                "request": {"endpoint": "guilds/{id}/channels", "guild_id": "700"},
                "payload": [{"id": "800", "name": "general", "type": 0}],
            },
            {
                "request": {"endpoint": "channels/{id}/messages", "channel_id": "800"},
                "payload": [
                    {
                        "id": "900",
                        "content": "New ransomware affiliate program advertised, CVE-2026-55555 exploited",
                        "timestamp": "2026-09-01T00:10:00+00:00",
                        "author": {"username": "bob", "bot": False},
                        "attachments": [],
                    }
                ],
            },
        ],
    }
    _write_json(fixtures_dir / "collectors" / "discord" / f"cycle_{cycle:03d}.json", payload)


def _write_all_fixtures(fixtures_dir: Path, cycle: int = 1) -> None:
    _write_x_fixture(fixtures_dir, cycle)
    _write_reddit_fixture(fixtures_dir, cycle)
    _write_telegram_fixture(fixtures_dir, cycle)
    _write_discord_fixture(fixtures_dir, cycle)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        mode="offline",
        fixtures_dir=tmp_path / "fixtures",
        config_dir=Path("config"),
        history_dir=tmp_path / "history",
    )


# ---------------------------------------------------------------------------
# corpus.to_post_record / dedupe_posts
# ---------------------------------------------------------------------------


def test_to_post_record_nests_language_and_attachments_in_metadata():
    post = RawPost(
        platform="x",
        external_id="42",
        content="hello",
        posted_at="2026-09-01T00:00:00Z",
        language="en",
        author_ref="alice",
        url="https://x.com/alice/status/42",
        attachments=[{"type": "photo", "url": "https://example.com/a.jpg"}],
        metadata={"source": "static", "query": "q"},
    )
    record = to_post_record(post)
    assert record == {
        "id": "x-42",
        "platform": "x",
        "content": "hello",
        "timestamp": "2026-09-01T00:00:00Z",
        "author": "alice",
        "url": "https://x.com/alice/status/42",
        "metadata": {
            "source": "static",
            "query": "q",
            "language": "en",
            "attachments": [{"type": "photo", "url": "https://example.com/a.jpg"}],
        },
    }


def test_dedupe_posts_keeps_first_and_sorts_by_posted_at_then_id():
    first = RawPost(platform="x", external_id="1", content="a", posted_at="2026-09-01T00:20:00Z")
    duplicate = RawPost(platform="x", external_id="1", content="a duplicate later payload", posted_at="2026-09-01T00:20:00Z")
    earlier = RawPost(platform="x", external_id="2", content="b", posted_at="2026-09-01T00:05:00Z")

    deduped = dedupe_posts([first, duplicate, earlier])

    assert [post.post_id for post in deduped] == ["x-2", "x-1"]
    assert deduped[1].content == "a"  # first occurrence kept, not the duplicate


def test_write_posts_jsonl_and_collection_report(tmp_path):
    posts = [RawPost(platform="x", external_id="1", content="a", posted_at="2026-09-01T00:20:00Z")]
    posts_path = tmp_path / "posts.jsonl"
    count = write_posts_jsonl(posts, posts_path)
    assert count == 1
    lines = posts_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == "x-1"

    from tiblender_runtime.collectors.base import CollectionResult

    result = CollectionResult(platform="x", mode="fixture", posts=posts, errors=[], stats={"posts": 1})
    report_path = tmp_path / "collection_report.json"
    write_collection_report([result], {"input": 1, "noise_filtered": 0, "short_no_keyword": 0, "kept": 1, "ocr_merged": 0}, report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["per_platform"]["x"]["mode"] == "fixture"
    assert report["per_platform"]["x"]["posts"] == 1
    assert report["preprocess"]["kept"] == 1
    assert "generated_at" in report
    # total_posts not supplied: falls back to "kept" and reports zero dedupe.
    assert report["total_posts"] == 1
    assert report["preprocess"]["deduplicated"] == 0
    assert report["warnings"] == []


def test_write_collection_report_uses_explicit_total_posts_and_records_deduplication(tmp_path):
    posts = [RawPost(platform="x", external_id="1", content="a", posted_at="2026-09-01T00:20:00Z")]
    from tiblender_runtime.collectors.base import CollectionResult

    result = CollectionResult(platform="x", mode="fixture", posts=posts, errors=[], stats={"posts": 2})
    report_path = tmp_path / "collection_report.json"
    write_collection_report(
        [result],
        {"input": 2, "noise_filtered": 0, "short_no_keyword": 0, "kept": 2, "ocr_merged": 0},
        report_path,
        total_posts=1,
        warnings=["malformed dynamic queries file /tmp/x.json: bad json"],
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["total_posts"] == 1
    assert report["preprocess"]["deduplicated"] == 1  # kept=2, written=1
    assert report["warnings"] == ["malformed dynamic queries file /tmp/x.json: bad json"]


# ---------------------------------------------------------------------------
# runner.collect_cycle
# ---------------------------------------------------------------------------


def test_collect_cycle_offline_produces_all_four_platforms(tmp_path):
    settings = _settings(tmp_path)
    _write_all_fixtures(settings.fixtures_dir)

    outcome = collect_cycle(
        settings,
        window=_WINDOW,
        dynamic_queries_path=None,
        output_dir=tmp_path / "out",
        services=None,
    )

    assert outcome.posts_path.exists()
    assert outcome.report_path.exists()
    assert outcome.post_count >= 4

    lines = [json.loads(line) for line in outcome.posts_path.read_text(encoding="utf-8").splitlines()]
    platforms = {record["platform"] for record in lines}
    assert platforms == {"x", "reddit", "telegram", "discord"}

    report = json.loads(outcome.report_path.read_text(encoding="utf-8"))
    for platform in ("x", "reddit", "telegram", "discord"):
        assert report["per_platform"][platform]["mode"] == "fixture"

    state_path = settings.history_dir / "collector_state.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "telegram" in state and "discord" in state

    # Server/channel discovery is an operational task, not part of this
    # artifact: no invite-candidate queue is ever produced.
    assert not (tmp_path / "out" / "manual_review_queue.json").exists()


def test_collect_cycle_missing_platform_fixture_does_not_abort(tmp_path):
    settings = _settings(tmp_path)
    _write_x_fixture(settings.fixtures_dir, cycle=1)
    _write_reddit_fixture(settings.fixtures_dir, cycle=1)
    _write_discord_fixture(settings.fixtures_dir, cycle=1)
    # Telegram only has a fixture for a different cycle, so cycle 1 is missing.
    _write_telegram_fixture(settings.fixtures_dir, cycle=2)

    outcome = collect_cycle(
        settings,
        window=_WINDOW,
        dynamic_queries_path=None,
        output_dir=tmp_path / "out",
        services=None,
    )

    assert outcome.per_platform["telegram"].mode == "unavailable"
    assert outcome.per_platform["telegram"].posts == []

    lines = [json.loads(line) for line in outcome.posts_path.read_text(encoding="utf-8").splitlines()]
    platforms = {record["platform"] for record in lines}
    assert platforms == {"x", "reddit", "discord"}
    assert outcome.per_platform["x"].mode == "fixture"
    assert outcome.per_platform["reddit"].mode == "fixture"
    assert outcome.per_platform["discord"].mode == "fixture"


def test_collect_cycle_reports_deduplicated_total_for_same_platform_duplicate(tmp_path):
    settings = _settings(tmp_path)
    _write_reddit_fixture(settings.fixtures_dir, cycle=1)
    _write_telegram_fixture(settings.fixtures_dir, cycle=1)
    _write_discord_fixture(settings.fixtures_dir, cycle=1)
    _write_x_fixture_with_same_platform_duplicate(settings.fixtures_dir, cycle=1)

    outcome = collect_cycle(
        settings,
        window=_WINDOW,
        dynamic_queries_path=None,
        output_dir=tmp_path / "out",
        services=None,
    )

    report = json.loads(outcome.report_path.read_text(encoding="utf-8"))
    lines = outcome.posts_path.read_text(encoding="utf-8").splitlines()

    # total_posts must equal what's actually on disk, not the pre-dedupe count.
    assert report["total_posts"] == len(lines) == outcome.post_count
    assert report["preprocess"]["deduplicated"] == report["preprocess"]["kept"] - len(lines)
    assert report["preprocess"]["deduplicated"] >= 1

    x_posts = [json.loads(line) for line in lines if json.loads(line)["platform"] == "x"]
    assert len(x_posts) == 1  # the two queries' duplicate tweet id collapsed to one post


def test_collect_cycle_malformed_fixture_file_is_unavailable_not_fatal(tmp_path):
    settings = _settings(tmp_path)
    _write_reddit_fixture(settings.fixtures_dir, cycle=1)
    _write_telegram_fixture(settings.fixtures_dir, cycle=1)
    _write_discord_fixture(settings.fixtures_dir, cycle=1)
    # x's fixture file exists but is structurally invalid (missing required keys).
    _write_json(_x_fixture_path(settings.fixtures_dir, 1), {"platform": "x"})

    outcome = collect_cycle(
        settings,
        window=_WINDOW,
        dynamic_queries_path=None,
        output_dir=tmp_path / "out",
        services=None,
    )

    assert outcome.per_platform["x"].mode == "unavailable"
    assert len(outcome.per_platform["x"].errors) == 1
    assert "malformed fixture file" in outcome.per_platform["x"].errors[0]
    assert str(_x_fixture_path(settings.fixtures_dir, 1)) in outcome.per_platform["x"].errors[0]

    lines = [json.loads(line) for line in outcome.posts_path.read_text(encoding="utf-8").splitlines()]
    platforms = {record["platform"] for record in lines}
    assert platforms == {"reddit", "telegram", "discord"}


def test_collect_cycle_live_transport_setup_failure_is_isolated(tmp_path):
    # A non-numeric telegram_api_id makes TelethonTransport's own
    # `int(api_id)` raise ValueError -- an arbitrary exception, not
    # CollectorError -- during transport construction. The cycle must
    # isolate that failure to the telegram platform rather than crashing.
    settings = Settings(
        mode="live",
        fixtures_dir=tmp_path / "fixtures",
        config_dir=Path("config"),
        history_dir=tmp_path / "history",
        telegram_api_id="not-a-number",
        telegram_api_hash="super-secret-hash",
        telegram_session="super-secret-session",
    )

    outcome = collect_cycle(
        settings,
        window=_WINDOW,
        dynamic_queries_path=None,
        output_dir=tmp_path / "out",
        services=None,
    )

    assert outcome.per_platform["telegram"].mode == "unavailable"
    error = outcome.per_platform["telegram"].errors[0]
    assert "telegram: transport setup failed: ValueError" in error
    # Never leak credential values into the recorded error.
    assert "super-secret-hash" not in error
    assert "super-secret-session" not in error

    # The rest of the cycle still completes (no fixtures/credentials for the
    # other platforms either, but that must not abort the run or crash it).
    assert outcome.posts_path.exists()
    assert outcome.report_path.exists()


def test_collect_cycle_malformed_dynamic_queries_file_is_recorded_as_warning(tmp_path):
    settings = _settings(tmp_path)
    _write_all_fixtures(settings.fixtures_dir)
    dynamic_path = tmp_path / "dynamic.json"
    dynamic_path.write_text("not valid json{", encoding="utf-8")

    outcome = collect_cycle(
        settings,
        window=_WINDOW,
        dynamic_queries_path=dynamic_path,
        output_dir=tmp_path / "out",
        services=None,
    )

    report = json.loads(outcome.report_path.read_text(encoding="utf-8"))
    assert len(report["warnings"]) == 1
    assert "malformed dynamic queries file" in report["warnings"][0]
    assert str(dynamic_path) in report["warnings"][0]

    # The cycle still completes normally despite the bad dynamic-queries file.
    assert outcome.posts_path.exists()
    lines = [json.loads(line) for line in outcome.posts_path.read_text(encoding="utf-8").splitlines()]
    assert {record["platform"] for record in lines} == {"x", "reddit", "telegram", "discord"}


def test_main_collect_cli_exits_zero(tmp_path, monkeypatch, capsys):
    fixtures_dir = tmp_path / "fixtures"
    history_dir = tmp_path / "history"
    _write_all_fixtures(fixtures_dir)

    monkeypatch.setenv("TIBLENDER_MODE", "offline")
    monkeypatch.setenv("TIBLENDER_FIXTURES_DIR", str(fixtures_dir))
    monkeypatch.setenv("TIBLENDER_CONFIG_DIR", "config")
    monkeypatch.setenv("TIBLENDER_HISTORY_DIR", str(history_dir))

    exit_code = main(["collect", "--output", str(tmp_path / "cli_out"), "--cycle", "1"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "posts=" in output
    assert "report=" in output
    assert (tmp_path / "cli_out" / "posts.jsonl").exists()


def test_collect_cycle_writes_no_invite_discovery_surface():
    """Discovery is operational: the runner must not reference it at all."""
    from tiblender_runtime.collectors import runner as runner_module

    source = Path(runner_module.__file__).read_text(encoding="utf-8")
    for retired in ("discover_invites", "manual_review_queue", "directory", "search_engine"):
        assert retired not in source, f"runner.py still references retired discovery surface: {retired}"
