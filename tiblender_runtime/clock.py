"""Runtime clock. Offline cycles pin the clock to the window end so outputs are reproducible."""
from __future__ import annotations
from datetime import UTC, datetime

_FIXED: datetime | None = None


def _parse(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(microsecond=0)


def set_fixed_clock(at: datetime | str | None) -> None:
    global _FIXED
    _FIXED = None if at is None else _parse(at)


def is_fixed() -> bool:
    return _FIXED is not None


def now() -> datetime:
    if _FIXED is not None:
        return _FIXED
    return datetime.now(UTC).replace(microsecond=0)


def now_iso() -> str:
    return now().isoformat().replace("+00:00", "Z")
