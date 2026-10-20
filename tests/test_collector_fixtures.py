import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tiblender_runtime.collectors.base import CollectionWindow
from tiblender_runtime.collectors.fixtures import (
    FixtureFile,
    fixture_path,
    list_fixture_cycles,
    load_fixture,
)


def _valid_payload() -> dict:
    return {
        "platform": "x",
        "window": {
            "start": "2026-03-20T09:00:00Z",
            "end": "2026-03-20T10:00:00Z",
            "cycle_index": 3,
            "label": "cycle_003",
        },
        "provenance": "recorded:2026-03-20T10:00:00Z",
        "responses": [
            {
                "request": {"endpoint": "search", "query": "phishing", "id": 123},
                "payload": {"data": ["a"]},
            },
            {
                "request": {"endpoint": "timeline", "id": 456},
                "payload": {"data": ["b"]},
            },
        ],
    }


def _write(tmp_path: Path, payload: dict, name: str = "cycle_003.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_fixture_round_trips_window_provenance_and_responses(tmp_path):
    payload = _valid_payload()
    path = _write(tmp_path, payload)

    fixture = load_fixture(path)

    assert isinstance(fixture, FixtureFile)
    assert fixture.platform == "x"
    assert fixture.window == CollectionWindow.ending_at(datetime(2026, 3, 20, 10, tzinfo=UTC), hours=1.0, cycle_index=3)
    assert fixture.provenance == "recorded:2026-03-20T10:00:00Z"
    assert fixture.responses == payload["responses"]


def test_responses_for_matches_endpoint_and_string_compared_fields(tmp_path):
    fixture = load_fixture(_write(tmp_path, _valid_payload()))

    by_int = fixture.responses_for("search", id=123)
    by_str = fixture.responses_for("search", id="123")
    assert by_int == by_str
    assert len(by_int) == 1
    assert by_int[0]["payload"] == {"data": ["a"]}

    assert fixture.responses_for("search", id=999) == []
    assert fixture.responses_for("missing-endpoint") == []


def test_responses_for_warns_when_a_match_key_is_recorded_nowhere(tmp_path, caplog):
    """A lookup on a key no recorded request carries can never match, so an
    empty result there means "this fixture cannot answer that question", not
    "no data". Confusing the two is exactly what made the Discord fixture
    transport silently drop every cycle after the first: it matched the
    `after` watermark against recorded request keys that have no `after`."""
    fixture = load_fixture(_write(tmp_path, _valid_payload()))

    with caplog.at_level("WARNING"):
        assert fixture.responses_for("search", after="123") == []

    messages = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert messages, caplog.text
    assert any("search" in message and "after" in message for message in messages), messages


def test_responses_for_does_not_warn_for_an_ordinary_recorded_key_miss(tmp_path, caplog):
    """A key the recorder does write down, queried with a value it never saw,
    is an ordinary miss -- a runtime-generated dynamic query with no recording
    -- and must not drown the contract-mismatch warning in noise."""
    fixture = load_fixture(_write(tmp_path, _valid_payload()))

    with caplog.at_level("WARNING"):
        assert fixture.responses_for("search", id=999) == []
    assert not [record for record in caplog.records if record.levelname == "WARNING"], caplog.text


def test_responses_for_is_quiet_when_the_endpoint_itself_was_never_recorded(tmp_path, caplog):
    fixture = load_fixture(_write(tmp_path, _valid_payload()))

    with caplog.at_level("WARNING"):
        assert fixture.responses_for("missing-endpoint", id=1) == []
    assert not caplog.records, caplog.text


def test_load_fixture_raises_on_missing_top_level_key(tmp_path):
    payload = _valid_payload()
    del payload["provenance"]
    path = _write(tmp_path, payload)

    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_fixture(path)


def test_load_fixture_raises_on_bad_window(tmp_path):
    payload = _valid_payload()
    del payload["window"]["label"]
    path = _write(tmp_path, payload)

    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_fixture(path)


def test_load_fixture_raises_on_non_list_responses(tmp_path):
    payload = _valid_payload()
    payload["responses"] = {"not": "a list"}
    path = _write(tmp_path, payload)

    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_fixture(path)


def test_load_fixture_raises_on_response_missing_endpoint(tmp_path):
    payload = _valid_payload()
    payload["responses"].append({"request": {"query": "no-endpoint-here"}, "payload": {}})
    path = _write(tmp_path, payload)

    with pytest.raises(ValueError, match=re.escape(str(path))):
        load_fixture(path)


def test_fixture_path_naming():
    assert fixture_path(Path("fixtures"), "reddit", 7) == Path("fixtures/collectors/reddit/cycle_007.json")
    assert fixture_path(Path("fixtures"), "x", 42) == Path("fixtures/collectors/x/cycle_042.json")


def test_list_fixture_cycles_returns_only_cycles_present_for_all_four_platforms(tmp_path):
    for platform in ("x", "reddit", "telegram", "discord"):
        platform_dir = tmp_path / "collectors" / platform
        platform_dir.mkdir(parents=True)
        (platform_dir / "cycle_001.json").write_text("{}", encoding="utf-8")
    for platform in ("x", "reddit", "telegram"):
        (tmp_path / "collectors" / platform / "cycle_002.json").write_text("{}", encoding="utf-8")

    assert list_fixture_cycles(tmp_path) == [1]
