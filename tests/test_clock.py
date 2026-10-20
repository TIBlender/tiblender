from datetime import UTC, datetime

from tiblender_runtime import clock


def test_fixed_clock_returns_configured_instant():
    clock.set_fixed_clock("2026-03-20T10:00:00Z")
    try:
        assert clock.now() == datetime(2026, 3, 20, 10, 0, tzinfo=UTC)
        assert clock.now_iso() == "2026-03-20T10:00:00Z"
        assert clock.is_fixed()
    finally:
        clock.set_fixed_clock(None)


def test_wall_clock_is_utc_and_not_fixed():
    clock.set_fixed_clock(None)
    assert clock.now().tzinfo is not None
    assert not clock.is_fixed()
    assert clock.now_iso().endswith("Z")
