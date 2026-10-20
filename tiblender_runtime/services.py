"""Resolve, per run mode, whether each collector/tool/LLM service runs live,
from recorded fixtures/tool recordings, heuristically, or is unavailable."""

from __future__ import annotations

import importlib.util
import socket
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

from .settings import Settings

# A `doctor` run must stay interactive even when a configured endpoint is a
# compose alias that does not resolve at all, so the reachability probe is a
# plain TCP connect with a short deadline rather than an HTTP request.
REACHABILITY_TIMEOUT_SECONDS = 1.5

SERVICE_NAMES: tuple[str, ...] = (
    "llm",
    "judge_a",
    "judge_b",
    "x",
    "reddit",
    "telegram",
    "discord",
    "search_engine",
    "whois",
    "passive_dns",
    "tls_ct",
    "web_crawl",
    "nvd",
    "github",
    "embedding",
    "ocr",
)

_COLLECTOR_SERVICES = ("x", "reddit", "telegram", "discord")
_TOOL_SERVICES = ("search_engine", "whois", "passive_dns", "tls_ct", "web_crawl", "nvd", "github")


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    name: str
    mode: str  # "live" | "fixture" | "heuristic" | "unavailable"
    reason: str


def _fixture_files(settings: Settings, platform: str) -> list[str]:
    directory = settings.fixtures_dir / "collectors" / platform
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.iterdir() if path.is_file())


def _has_x_credentials(settings: Settings) -> bool:
    from .tools import _x_bearer_tokens

    return bool(_x_bearer_tokens(settings))


def _has_reddit_credentials(settings: Settings) -> bool:
    from .tools import _reddit_accounts

    return bool(_reddit_accounts(settings))


def _has_telegram_credentials(settings: Settings) -> bool:
    return bool(
        settings.telegram_api_id
        and settings.telegram_api_hash
        and settings.telegram_session
        and importlib.util.find_spec("telethon") is not None
    )


def _has_discord_credentials(settings: Settings) -> bool:
    # Either credential shape unlocks collection; the user-account token is
    # preferred at transport-construction time (see collectors/discord.py).
    return bool(settings.discord_user_token or settings.discord_bot_token)


_COLLECTOR_CREDENTIAL_CHECKS = {
    "x": _has_x_credentials,
    "reddit": _has_reddit_credentials,
    "telegram": _has_telegram_credentials,
    "discord": _has_discord_credentials,
}


def _resolve_collector(settings: Settings, name: str) -> ServiceStatus:
    has_credentials = _COLLECTOR_CREDENTIAL_CHECKS[name](settings)
    fixture_available = bool(_fixture_files(settings, name))
    if settings.mode == "offline":
        # Offline cycles never touch live platform credentials, even if set.
        if fixture_available:
            return ServiceStatus(name, "fixture", "offline mode uses recorded collector fixtures")
        return ServiceStatus(name, "unavailable", "offline mode has no recorded fixture for this platform")
    if settings.mode == "live":
        if has_credentials:
            return ServiceStatus(name, "live", "live mode with configured credentials")
        return ServiceStatus(name, "unavailable", "live mode requires credentials; none configured")
    # auto
    if has_credentials:
        return ServiceStatus(name, "live", "auto mode prefers live credentials")
    if fixture_available:
        return ServiceStatus(name, "fixture", "auto mode falls back to recorded fixtures")
    return ServiceStatus(name, "unavailable", "auto mode has neither credentials nor fixtures")


def _resolve_tool(settings: Settings, name: str) -> ServiceStatus:
    if settings.mode == "offline":
        return ServiceStatus(name, "fixture", "offline mode uses recorded tool responses")
    if name == "search_engine":
        live = bool(settings.searxng_base_url.strip())
        reason = "search engine base URL configured" if live else "SEARXNG_BASE_URL is not set"
    elif name == "passive_dns":
        live = bool(settings.virustotal_api_key.strip())
        reason = (
            "passive DNS provider virustotal with VIRUSTOTAL_API_KEY configured"
            if live
            else "passive DNS provider virustotal requires VIRUSTOTAL_API_KEY; none configured"
        )
    else:
        live = True
        reason = "network tool reachable outside offline mode"
    if live:
        return ServiceStatus(name, "live", reason)
    if settings.mode == "live":
        return ServiceStatus(name, "unavailable", reason)
    return ServiceStatus(name, "fixture", "auto mode falls back to recorded tool responses")


def _resolve_llm(settings: Settings, name: str) -> ServiceStatus:
    if name in {"judge_a", "judge_b"}:
        configured = settings.for_judge(name[-1]).llm_configured()
    else:
        configured = settings.llm_configured()
    if settings.mode == "offline":
        return ServiceStatus(name, "heuristic", "offline mode does not call live LLM providers")
    if settings.mode == "live":
        if configured:
            return ServiceStatus(name, "live", "live mode with configured LLM")
        return ServiceStatus(name, "unavailable", "live mode requires LLM configuration")
    # auto
    if configured:
        return ServiceStatus(name, "live", "auto mode with configured LLM")
    return ServiceStatus(name, "heuristic", "auto mode falls back to heuristic reports")


def _resolve_embedding(settings: Settings) -> ServiceStatus:
    # Offline runs must be bit-reproducible, so they always take the
    # deterministic hashing path even when sentence-transformers happens to be
    # installed (model weights, device, and library version would otherwise
    # leak into the output).
    if settings.is_offline():
        return ServiceStatus(
            "embedding", "heuristic", "offline mode forces the deterministic hashing embedding backend"
        )
    backend = settings.embedding_backend.strip().lower()
    if backend and backend != "hashing":
        importable = importlib.util.find_spec("sentence_transformers") is not None
        if importable:
            return ServiceStatus("embedding", "live", f"{backend} embedding backend is importable")
        return ServiceStatus("embedding", "unavailable", f"{backend} embedding backend is not importable")
    return ServiceStatus("embedding", "heuristic", "hashing embedding backend is deterministic and offline-safe")


def resolved_embedding_backend(settings: Settings, backend: str) -> str:
    """The embedding backend actually used, honoring offline mode.

    Keeps the runtime's behavior in step with the ``embedding`` row of the
    doctor report: offline is deterministic-hashing everywhere.
    """
    return "hashing" if settings.is_offline() else backend


def _resolve_ocr(settings: Settings) -> ServiceStatus:
    # OCR output depends on the installed tesseract build and on fetching the
    # attachment image, neither of which an offline run may rely on; fixture
    # `attachments[].ocr_text` still flows through the same merge path.
    if settings.is_offline():
        return ServiceStatus("ocr", "unavailable", "offline mode uses fixture ocr_text only, never live OCR")
    if not settings.ocr_enabled:
        return ServiceStatus("ocr", "unavailable", "OCR disabled by TIBLENDER_OCR_ENABLED=false")
    importable = importlib.util.find_spec("pytesseract") is not None
    if importable:
        return ServiceStatus("ocr", "live", "pytesseract is importable")
    return ServiceStatus("ocr", "unavailable", "pytesseract is not importable")


def resolve_services(settings: Settings) -> dict[str, ServiceStatus]:
    services: dict[str, ServiceStatus] = {
        "llm": _resolve_llm(settings, "llm"),
        "judge_a": _resolve_llm(settings, "judge_a"),
        "judge_b": _resolve_llm(settings, "judge_b"),
    }
    for name in _COLLECTOR_SERVICES:
        services[name] = _resolve_collector(settings, name)
    for name in _TOOL_SERVICES:
        services[name] = _resolve_tool(settings, name)
    services["embedding"] = _resolve_embedding(settings)
    services["ocr"] = _resolve_ocr(settings)
    return services


class LiveServicesUnavailableError(RuntimeError):
    """A ``live`` run cannot succeed: required services resolved to ``unavailable``.

    Raised by the preflight below *before* a cycle writes anything, so the
    documented "a live run with no credentials fails fast before any output is
    written" is literally true rather than true only of report generation.
    """


def live_preflight_error(settings: Settings) -> str | None:
    """Why a ``live`` run cannot produce a cycle, or ``None`` if it can.

    Only ``live`` mode is gated: ``offline`` and ``auto`` are defined to
    degrade to fixtures/recordings/heuristics per service, so an unavailable
    service there is a documented outcome rather than a failure.
    """
    if settings.mode != "live":
        return None

    services = resolve_services(settings)
    problems: list[str] = []
    if not settings.allow_heuristic_reports and services["llm"].mode == "unavailable":
        problems.append(
            "llm: LLM configuration is required for full LLM-backed report generation. "
            "Set TIBLENDER_LLM_PROVIDER plus the matching model/deployment and key in .env. "
            "For local smoke tests only, set TIBLENDER_ALLOW_HEURISTIC_REPORTS=true."
        )
    if all(services[name].mode == "unavailable" for name in _COLLECTOR_SERVICES):
        problems.append(
            "collectors: live mode collects only from platforms with configured credentials, "
            f"and none of {'/'.join(_COLLECTOR_SERVICES)} has any. See docs/CREDENTIALS.md, "
            "or run the bundled fixtures with TIBLENDER_MODE=offline."
        )
    if not problems:
        return None
    return (
        "live mode cannot run this cycle, so nothing was written. "
        "Run `tiblender doctor` for the full per-service report.\n  - " + "\n  - ".join(problems)
    )


def require_live_services(settings: Settings) -> None:
    """Raise :class:`LiveServicesUnavailableError` if a live run cannot proceed."""
    message = live_preflight_error(settings)
    if message is not None:
        raise LiveServicesUnavailableError(message)


def endpoint_reachable(url: str, timeout: float = REACHABILITY_TIMEOUT_SECONDS) -> bool:
    """Whether ``url``'s host:port accepts a TCP connection within ``timeout``."""
    parts = urlsplit(url.strip())
    host = parts.hostname
    if not host:
        return False
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:  # malformed port in the URL
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        # Unresolvable host, refused connection, timeout: all mean "not usable".
        return False


def _verified_search_engine(
    settings: Settings,
    status: ServiceStatus,
    probe: Callable[[str], bool],
) -> ServiceStatus:
    """Re-state the search engine's ``live`` row against actual reachability.

    ``SEARXNG_BASE_URL`` defaults to the compose alias ``http://searxng:8080``,
    which only resolves when ``--profile live`` started the bundled service, so
    a configuration-only resolution reported ``live`` for an endpoint that was
    not there. This runs in ``doctor_report`` alone: ``resolve_services`` is
    called on every cycle and must stay free of network access.
    """
    if status.mode != "live":
        return status
    url = settings.searxng_base_url.strip()
    if probe(url):
        return ServiceStatus(status.name, "live", f"search engine configured and reachable at {url}")
    return ServiceStatus(
        status.name,
        "unavailable",
        f"search engine base URL {url} is configured but not reachable: start the bundled "
        "instance with `docker compose --profile live up`, or point SEARXNG_BASE_URL at a "
        "reachable SearXNG",
    )


def doctor_report(
    settings: Settings,
    *,
    probe: Callable[[str], bool] = endpoint_reachable,
) -> dict:
    resolved = resolve_services(settings)
    services = {
        **resolved,
        "search_engine": _verified_search_engine(settings, resolved["search_engine"], probe),
    }
    fixtures_present = {name: _fixture_files(settings, name) for name in _COLLECTOR_SERVICES}
    return {
        "mode": settings.mode,
        "services": [
            {"name": status.name, "mode": status.mode, "reason": status.reason}
            for status in sorted(services.values(), key=lambda item: item.name)
        ],
        "fixtures_present": fixtures_present,
    }
