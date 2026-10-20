"""Wires the four platform collectors to live or fixture transports according
to resolved service modes, and drives one end-to-end collection cycle."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..services import ServiceStatus, resolve_services
from ..settings import Settings
from ..tools import _reddit_accounts, _x_bearer_tokens
from .base import CollectionResult, CollectionWindow, PlatformCollector, QueryPlan, RawPost
from .corpus import dedupe_posts, write_collection_report, write_posts_jsonl
from .discord import DiscordCollector, DiscordFixtureTransport, DiscordLiveTransport
from .fixtures import FixtureFile, fixture_path, load_fixture
from .http import HttpClient
from .keywords import KeywordCatalog, build_query_plan, load_dynamic_queries, load_lines
from .preprocess import default_ocr, preprocess
from .reddit import RedditCollector, RedditFixtureTransport, RedditLiveTransport
from .telegram import TelegramCollector, TelegramFixtureTransport, TelethonTransport
from .x import XCollector, XFixtureTransport, XLiveTransport

logger = logging.getLogger(__name__)

_PLATFORMS = ("discord", "reddit", "telegram", "x")


@dataclass(slots=True)
class CollectionOutcome:
    window: CollectionWindow
    posts_path: Path
    report_path: Path
    post_count: int
    per_platform: dict[str, CollectionResult]


def _x_transport(settings: Settings, mode: str, fixture: FixtureFile | None):
    if mode == "live":
        return XLiveTransport(_x_bearer_tokens(settings), HttpClient())
    return XFixtureTransport(fixture)


def _reddit_transport(settings: Settings, mode: str, fixture: FixtureFile | None):
    if mode == "live":
        return RedditLiveTransport(_reddit_accounts(settings), HttpClient(), user_agent=settings.reddit_user_agent)
    return RedditFixtureTransport(fixture)


def _telegram_transport(settings: Settings, mode: str, fixture: FixtureFile | None):
    if mode == "live":
        return TelethonTransport(settings.telegram_api_id, settings.telegram_api_hash, settings.telegram_session)
    return TelegramFixtureTransport(fixture)


def _discord_transport(settings: Settings, mode: str, fixture: FixtureFile | None):
    if mode == "live":
        return DiscordLiveTransport(
            HttpClient(),
            user_token=settings.discord_user_token,
            bot_token=settings.discord_bot_token,
        )
    return DiscordFixtureTransport(fixture)


_TRANSPORT_BUILDERS: dict[str, Callable] = {
    "x": _x_transport,
    "reddit": _reddit_transport,
    "telegram": _telegram_transport,
    "discord": _discord_transport,
}


def _build_collector(
    platform: str,
    transport,
    mode: str,
    *,
    settings: Settings,
    state: dict,
) -> PlatformCollector:
    if platform == "x":
        return XCollector(
            transport,
            users=load_lines(settings.config_dir / "x_users.txt"),
            max_posts_per_query=settings.max_posts_per_query,
            mode=mode,
        )
    if platform == "reddit":
        return RedditCollector(transport, max_posts_per_query=settings.max_posts_per_query, mode=mode)
    if platform == "telegram":
        return TelegramCollector(
            transport,
            channels=load_lines(settings.config_dir / "telegram_channels.txt"),
            max_channels=settings.telegram_max_channels,
            mode=mode,
            state=state.get("telegram", {}),
        )
    # platform == "discord"
    return DiscordCollector(
        transport,
        guild_filter=load_lines(settings.config_dir / "discord_guilds.txt"),
        max_guilds=settings.discord_max_guilds,
        max_channels_per_guild=settings.discord_max_channels_per_guild,
        mode=mode,
        state=state.get("discord", {}),
    )


def _empty_result(platform: str, reason: str) -> CollectionResult:
    return CollectionResult(platform=platform, mode="unavailable", posts=[], errors=[reason], stats={})


def _resolve_platform(
    platform: str,
    *,
    settings: Settings,
    services: dict[str, ServiceStatus],
    cycle_index: int,
    state: dict,
) -> tuple[PlatformCollector | None, CollectionResult | None]:
    status = services[platform]

    if status.mode == "unavailable":
        return None, _empty_result(platform, status.reason)

    if status.mode == "live":
        try:
            transport = _TRANSPORT_BUILDERS[platform](settings, "live", None)
        except Exception as error:  # noqa: BLE001 - a single platform's setup failure must never abort the cycle
            reason = f"{platform}: transport setup failed: {type(error).__name__}: {error}"
            logger.warning(reason)
            return None, _empty_result(platform, reason)
        collector = _build_collector(platform, transport, "live", settings=settings, state=state)
        return collector, None

    if status.mode == "fixture":
        path = fixture_path(settings.fixtures_dir, platform, cycle_index)
        if not path.exists():
            return None, _empty_result(platform, "fixture missing")
        try:
            fixture = load_fixture(path)
        except (ValueError, OSError) as error:
            reason = f"malformed fixture file {path}: {error}"
            logger.warning(reason)
            return None, _empty_result(platform, reason)
        transport = _TRANSPORT_BUILDERS[platform](settings, "fixture", fixture)
        collector = _build_collector(platform, transport, "fixture", settings=settings, state=state)
        return collector, None

    return None, _empty_result(platform, f"unknown service mode: {status.mode}")


def build_collectors(
    settings: Settings,
    services: dict[str, ServiceStatus],
    *,
    cycle_index: int,
    state: dict,
) -> dict[str, PlatformCollector]:
    collectors: dict[str, PlatformCollector] = {}
    for platform in _PLATFORMS:
        collector, _skip_result = _resolve_platform(
            platform,
            settings=settings,
            services=services,
            cycle_index=cycle_index,
            state=state,
        )
        if collector is not None:
            collectors[platform] = collector
    return collectors


def _load_state(history_dir: Path) -> dict:
    path = Path(history_dir) / "collector_state.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    descriptor, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_name)
        raise


def collect_cycle(
    settings: Settings,
    *,
    window: CollectionWindow,
    dynamic_queries_path: Path | None,
    output_dir: Path,
    services: dict[str, ServiceStatus] | None = None,
) -> CollectionOutcome:
    services = services if services is not None else resolve_services(settings)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = KeywordCatalog.load(settings.config_dir / "keywords.json")
    subreddits = load_lines(settings.config_dir / "subreddits.txt")

    dynamic: list = []
    warnings: list[str] = []
    if dynamic_queries_path is not None and Path(dynamic_queries_path).exists():
        try:
            dynamic = load_dynamic_queries(
                Path(dynamic_queries_path),
                now=window.end,
                max_per_platform=settings.max_dynamic_queries_per_platform,
            )
        except (ValueError, OSError) as error:
            reason = f"malformed dynamic queries file {dynamic_queries_path}: {error}"
            logger.warning(reason)
            warnings.append(reason)
    plan: QueryPlan = build_query_plan(catalog, subreddits=subreddits, dynamic=dynamic)

    state = _load_state(settings.resolved_history_dir())

    collectors: dict[str, PlatformCollector] = {}
    per_platform: dict[str, CollectionResult] = {}
    for platform in _PLATFORMS:
        collector, skip_result = _resolve_platform(
            platform,
            settings=settings,
            services=services,
            cycle_index=window.cycle_index,
            state=state,
        )
        if collector is not None:
            collectors[platform] = collector
        else:
            per_platform[platform] = skip_result

    all_posts: list[RawPost] = []
    for platform in sorted(collectors):
        collector = collectors[platform]
        try:
            result = collector.collect(window, plan)
        except Exception as error:  # noqa: BLE001 - a single platform failure must never abort the cycle
            per_platform[platform] = CollectionResult(
                platform=platform, mode=services[platform].mode, posts=[], errors=[str(error)], stats={}
            )
            continue
        per_platform[platform] = result
        all_posts.extend(result.posts)

    ocr = default_ocr() if settings.ocr_enabled and services["ocr"].mode == "live" else None
    kept_posts, preprocess_stats = preprocess(
        all_posts,
        min_text_chars=settings.min_text_chars,
        security_terms=catalog.security_terms(),
        ocr=ocr,
    )
    deduped = dedupe_posts(kept_posts)

    posts_path = output_dir / "posts.jsonl"
    report_path = output_dir / "collection_report.json"
    post_count = write_posts_jsonl(deduped, posts_path)

    write_collection_report(
        list(per_platform.values()),
        preprocess_stats,
        report_path,
        total_posts=post_count,
        warnings=warnings,
    )

    new_state = {
        "telegram": collectors["telegram"].export_state() if "telegram" in collectors else state.get("telegram", {}),
        "discord": collectors["discord"].export_state() if "discord" in collectors else state.get("discord", {}),
    }
    _atomic_write_json(settings.resolved_history_dir() / "collector_state.json", new_state)

    return CollectionOutcome(
        window=window,
        posts_path=posts_path,
        report_path=report_path,
        post_count=post_count,
        per_platform=per_platform,
    )
