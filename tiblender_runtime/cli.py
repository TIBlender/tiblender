"""Command-line entry points for running and inspecting the runtime."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

from . import clock, verify
from .collectors.base import CollectionWindow, CollectorError
from .collectors.fixtures import fixture_path, load_fixture
from .collectors.runner import collect_cycle
from .llm import LLMGenerationError
from .models import to_jsonable
from .contract import audit_contract
from .demo_report import write_demo_report
from .dqn_demo import (
    DQN_DEMO_CYCLES,
    DQN_DEMO_MIN_TRANSITIONS,
    DQN_DEMO_OUTPUT_ROOT,
    format_dqn_demo,
    run_dqn_demo,
)
from .pipeline import run_cycle
from .scheduler import run_cycles, run_one_cycle
from .services import LiveServicesUnavailableError, doctor_report, resolve_services
from .settings import MissingLLMConfigurationError, Settings
from .viewer_data import write_viewer_data

_DEFAULT_EXPECTED_PATH = Path("sample_runs/offline_three_cycles/expected/summary.json")


def _redacted_settings(settings: Settings) -> dict:
    # show-config should be useful for debugging while never printing API keys
    # or bearer tokens to terminal logs.
    payload = to_jsonable(settings)
    for key in (
        "openai_api_key",
        "nvd_api_key",
        "github_token",
        "x_bearer_token",
        "judge_a_api_key",
        "judge_b_api_key",
        "telegram_api_hash",
        # A Telegram session string is a full account credential (it
        # authenticates as the user without the api_hash), so it is masked
        # exactly like a key -- printing it here used to leak it into logs.
        "telegram_session",
        "discord_bot_token",
        "discord_user_token",
        "virustotal_api_key",
        "reddit_client_secret",
        "reddit_extra_accounts",
    ):
        payload[key] = "<set>" if getattr(settings, key) else ""
    return payload


def _resolve_collect_window(settings: Settings, services: dict, args: argparse.Namespace) -> CollectionWindow:
    x_status = services["x"]
    candidate_fixture = fixture_path(settings.fixtures_dir, "x", args.cycle)
    if x_status.mode == "fixture" and candidate_fixture.exists():
        return load_fixture(candidate_fixture).window

    window_end = datetime.fromisoformat(args.window_end.replace("Z", "+00:00")) if args.window_end else clock.now()
    return CollectionWindow.ending_at(window_end, settings.collect_window_hours, args.cycle)


def _run_collect(settings: Settings, args: argparse.Namespace) -> int:
    # Resolved once and reused for both window selection and the collection
    # cycle itself, so the CLI never asks "what mode is X in" twice and risks
    # the two calls drifting (e.g. credentials changing mid-process).
    services = resolve_services(settings)
    window = _resolve_collect_window(settings, services, args)

    if settings.clock_mode == "fixed" or (settings.clock_mode == "auto" and settings.is_offline()):
        clock.set_fixed_clock(window.end)

    try:
        outcome = collect_cycle(
            settings,
            window=window,
            dynamic_queries_path=args.dynamic_queries,
            output_dir=args.output,
            services=services,
        )
    except CollectorError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"posts={outcome.post_count}")
    print(f"report={outcome.report_path}")
    return 0


def _run_scheduler_cycle(settings: Settings, args: argparse.Namespace) -> int:
    window_end = datetime.fromisoformat(args.window_end.replace("Z", "+00:00")) if args.window_end else None
    try:
        record = run_one_cycle(
            settings,
            cycle_index=args.cycle_index,
            output_root=args.output_root,
            window_end=window_end,
            resume=args.resume,
        )
    except (
        CollectorError,
        LiveServicesUnavailableError,
        MissingLLMConfigurationError,
        LLMGenerationError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"cycle_index={record.cycle_index}")
    print(f"posts={record.posts}")
    print(f"reports={record.reports}")
    print(f"dynamic_queries={record.dynamic_queries}")
    print(f"tool_recording_hit_rate={record.tool_recording_hit_rate:.3f}")
    print(f"training_status={record.training_status}")
    return 0


def _run_scheduler_run_cycles(settings: Settings, args: argparse.Namespace) -> int:
    interval_seconds = args.interval if args.interval is not None else settings.cycle_interval_seconds
    try:
        records = run_cycles(
            settings,
            cycles=args.cycles,
            output_root=args.output_root,
            interval_seconds=interval_seconds,
            start_index=args.start_index,
            resume=args.resume,
        )
    except (
        CollectorError,
        LiveServicesUnavailableError,
        MissingLLMConfigurationError,
        LLMGenerationError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"cycles_run={len(records)}")
    for record in records:
        print(f"cycle_{record.cycle_index:03d}_posts={record.posts}")
        print(f"cycle_{record.cycle_index:03d}_reports={record.reports}")
        print(f"cycle_{record.cycle_index:03d}_dynamic_queries={record.dynamic_queries}")
        print(
            f"cycle_{record.cycle_index:03d}_tool_recording_hit_rate="
            f"{record.tool_recording_hit_rate:.3f}"
        )
    return 0


def _run_verify_sample(args: argparse.Namespace) -> int:
    try:
        return verify.verify_sample(args.output_root, args.expected)
    except verify.VerificationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run_write_expected(args: argparse.Namespace) -> int:
    try:
        verify.write_expected(args.output_root, args.expected_dir)
    except verify.VerificationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"expected_dir={args.expected_dir}")
    return 0


def _run_build_viewer(settings: Settings, args: argparse.Namespace) -> int:
    # Under a fixed clock (offline runs) the bundle's build stamp is derived
    # from the reports instead of wall time, so rebuilding the same reports
    # rewrites the same bytes.
    result = write_viewer_data(
        args.reports_root, args.viewer_dir, fixed_clock=settings.uses_fixed_clock()
    )
    print(f"viewer_data={result.path}")
    print(f"reports={result.reports_written}")
    print(f"skipped={result.skipped}")
    return 0


def _run_demo(settings: Settings, args: argparse.Namespace) -> int:
    """The bundled offline demo end to end: run-cycles, verify-sample, build-viewer.

    This is the container's default command, so the image can declare
    ``ENTRYPOINT ["tiblender"]`` (which makes
    ``docker compose run --rm tiblender doctor`` work) while still running the
    whole demo when no subcommand is given.
    """
    exit_code = _run_scheduler_run_cycles(
        settings,
        argparse.Namespace(
            output_root=args.output_root,
            cycles=args.cycles,
            interval=args.interval,
            start_index=1,
            resume=args.resume,
        ),
    )
    if exit_code != 0:
        return exit_code

    exit_code = _run_verify_sample(
        argparse.Namespace(output_root=args.output_root, expected=args.expected)
    )
    if exit_code != 0:
        return exit_code

    exit_code = _run_build_viewer(
        settings,
        argparse.Namespace(reports_root=args.output_root, viewer_dir=args.viewer_dir),
    )
    if exit_code != 0:
        return exit_code

    # `demo/run_demo.sh` asks for this so the demo's reports can be read from
    # one path, output/report.json, instead of a tree. Off unless requested,
    # so the demo's own output tree is unchanged for everyone else.
    if args.report_json is not None:
        written = write_demo_report(args.output_root, args.report_json)
        print(f"report_json={written}")
    return 0


def _run_dqn_demo(settings: Settings, args: argparse.Namespace) -> int:
    """The bundled cycles with the DQN training threshold lowered (see dqn_demo.py)."""
    if args.min_transitions <= 0:
        print("error: --min-transitions must be a positive integer", file=sys.stderr)
        return 2
    try:
        result = run_dqn_demo(
            settings,
            output_root=args.output_root,
            cycles=args.cycles,
            min_transitions=args.min_transitions,
        )
    except (
        CollectorError,
        LiveServicesUnavailableError,
        MissingLLMConfigurationError,
        LLMGenerationError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(format_dqn_demo(result))
    return 0 if result.ok else 1


def _run_telegram_login(settings: Settings, args: argparse.Namespace) -> int:
    # telethon is an optional dependency (pinned in requirements.txt but not
    # required for offline/fixture runs), so it is imported lazily here and
    # only for this command -- every other code path that touches Telegram
    # (the collector itself) already guards its own lazy import the same way.
    if importlib.util.find_spec("telethon") is None:
        print(
            "error: telethon is not installed. Install it with `pip install telethon` "
            "(already pinned in requirements.txt; `bash install.sh` installs it by default) "
            "and re-run `telegram-login`.",
            file=sys.stderr,
        )
        return 2

    api_id = args.api_id or settings.telegram_api_id
    api_hash = args.api_hash or settings.telegram_api_hash
    if not api_id or not api_hash:
        print(
            "error: telegram-login requires an api_id and api_hash. Create an app at "
            "https://my.telegram.org/apps, then pass --api-id/--api-hash or set "
            "TIBLENDER_TELEGRAM_API_ID/TIBLENDER_TELEGRAM_API_HASH in .env.",
            file=sys.stderr,
        )
        return 2

    from telethon import TelegramClient

    # client.start() is Telethon's interactive login flow: it prompts on
    # stdin for the phone number, the login code sent to Telegram, and a 2FA
    # password if one is set, then writes the authorized session to disk.
    # This is a deliberate one-time, human-in-the-loop step -- it is not run
    # by any offline/live pipeline cycle, only by an operator setting up
    # Telegram collection for the first time (see docs/CREDENTIALS.md).
    client = TelegramClient(args.session, int(api_id), api_hash)
    client.start()
    client.disconnect()

    session_path = Path(args.session)
    if session_path.suffix != ".session":
        session_path = session_path.with_name(session_path.name + ".session")
    print(f"session={session_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # ``prog`` is pinned so the three supported invocations -- the ``tiblender``
    # console script, ``python -m tiblender_runtime`` and
    # ``python -m tiblender_runtime.cli`` -- all print the same usage banner
    # instead of ``__main__.py`` / ``cli.py``.
    parser = argparse.ArgumentParser(prog="tiblender", description="Run the TIBlender runtime pipeline.")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run one TIBlender pipeline cycle")
    run_parser.add_argument("--input", type=Path, default=None, help="Input JSONL path")
    run_parser.add_argument("--output", type=Path, default=None, help="Output directory")
    run_parser.add_argument("--llm-provider", default=None, help="openai, openai_compatible, azure_openai, or heuristic")
    run_parser.add_argument("--llm-model", default=None, help="Model name for OpenAI-compatible providers")
    run_parser.add_argument(
        "--allow-heuristic-reports",
        action="store_true",
        help="Allow deterministic smoke reports without an LLM",
    )
    subparsers.add_parser("show-config", help="Print the resolved .env/environment configuration")
    subparsers.add_parser("audit-contract", help="Report the tool/prompt contract status")
    subparsers.add_parser("doctor", help="Report resolved service modes (live/fixture/heuristic/unavailable)")
    collect_parser = subparsers.add_parser("collect", help="Run one collection cycle across all platforms")
    collect_parser.add_argument("--output", type=Path, required=True, help="Output directory for posts.jsonl and collection_report.json")
    collect_parser.add_argument("--cycle", type=int, default=1, help="Cycle index (used for fixture selection and window labeling)")
    collect_parser.add_argument("--window-end", default=None, help="ISO timestamp for the end of the collection window")
    collect_parser.add_argument("--dynamic-queries", type=Path, default=None, help="Path to a dynamic queries JSON file")
    build_viewer_parser = subparsers.add_parser(
        "build-viewer", help="Build the browser report viewer's data file from generated reports"
    )
    build_viewer_parser.add_argument(
        "--reports-root", type=Path, required=True, help="Directory containing generated reports (output/ or a multi-cycle tree)"
    )
    build_viewer_parser.add_argument(
        "--viewer-dir", type=Path, required=True, help="Viewer directory to write reports_viewer_data.js into"
    )
    cycle_parser = subparsers.add_parser("cycle", help="Run one multi-cycle scheduler cycle (collect + run)")
    cycle_parser.add_argument("--output-root", type=Path, required=True, help="Root directory holding cycle_NNN/ and history/")
    cycle_parser.add_argument("--cycle", type=int, required=True, dest="cycle_index", help="Cycle index")
    cycle_parser.add_argument("--window-end", default=None, help="ISO timestamp for the end of the collection window")
    cycle_parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from existing state under --output-root (cycle 1 otherwise starts fresh, deleting history/ and cycle_*/)",
    )
    run_cycles_parser = subparsers.add_parser("run-cycles", help="Run a sequence of multi-cycle scheduler cycles")
    run_cycles_parser.add_argument("--output-root", type=Path, required=True, help="Root directory holding cycle_NNN/ and history/")
    run_cycles_parser.add_argument("--cycles", type=int, required=True, help="Number of cycles to run")
    run_cycles_parser.add_argument(
        "--interval", type=int, default=None, help="Seconds to sleep between cycles (default: settings.cycle_interval_seconds)"
    )
    run_cycles_parser.add_argument("--start-index", type=int, default=1, help="First cycle index to run")
    run_cycles_parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from existing state under --output-root instead of starting fresh (the default deletes history/ and cycle_*/ first)",
    )
    verify_parser = subparsers.add_parser("verify-sample", help="Compare scheduler output against a committed expected summary")
    verify_parser.add_argument("--output-root", type=Path, required=True, help="Root directory to summarize and verify")
    verify_parser.add_argument("--expected", type=Path, default=_DEFAULT_EXPECTED_PATH, help="Path to the expected summary.json")
    write_expected_parser = subparsers.add_parser(
        "write-expected", help="Regenerate the committed expected-output summary (maintainer-only)"
    )
    write_expected_parser.add_argument("--output-root", type=Path, required=True, help="Root directory to summarize")
    write_expected_parser.add_argument("--expected-dir", type=Path, required=True, help="Directory to write summary.json and per-cycle copies into")
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run the bundled offline demo: run-cycles, then verify-sample, then build-viewer",
    )
    demo_parser.add_argument("--output-root", type=Path, default=Path("output"), help="Root directory for the demo's output tree")
    demo_parser.add_argument("--cycles", type=int, default=3, help="Number of cycles to run (default: 3)")
    demo_parser.add_argument("--interval", type=int, default=0, help="Seconds to sleep between cycles (default: 0)")
    demo_parser.add_argument("--viewer-dir", type=Path, default=Path("viewer"), help="Viewer directory to build the data file into")
    demo_parser.add_argument("--expected", type=Path, default=_DEFAULT_EXPECTED_PATH, help="Path to the expected summary.json")
    demo_parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Also write every cycle's reports, counters and verify-sample summary as one JSON document at this path",
    )
    demo_parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from existing state under --output-root instead of starting fresh",
    )
    dqn_demo_parser = subparsers.add_parser(
        "dqn-demo",
        help="Rerun the bundled fixtures with a lowered RL training threshold so DQN fine-tuning runs, then check it did",
    )
    dqn_demo_parser.add_argument(
        "--output-root", type=Path, default=DQN_DEMO_OUTPUT_ROOT, help=f"Root directory for the run (default: {DQN_DEMO_OUTPUT_ROOT})"
    )
    dqn_demo_parser.add_argument(
        "--cycles", type=int, default=DQN_DEMO_CYCLES, help=f"Number of cycles to run (default: {DQN_DEMO_CYCLES})"
    )
    dqn_demo_parser.add_argument(
        "--min-transitions",
        type=int,
        default=DQN_DEMO_MIN_TRANSITIONS,
        help=f"Replay-buffer size at which DQN fine-tuning starts (default: {DQN_DEMO_MIN_TRANSITIONS}; the paper uses 1000)",
    )
    telegram_login_parser = subparsers.add_parser(
        "telegram-login",
        help="One-time interactive Telegram user session login (requires telethon)",
    )
    telegram_login_parser.add_argument("--session", required=True, help="Session name/path; creates <session>.session")
    telegram_login_parser.add_argument(
        "--api-id", default=None, help="Telegram api_id (default: TIBLENDER_TELEGRAM_API_ID)"
    )
    telegram_login_parser.add_argument(
        "--api-hash", default=None, help="Telegram api_hash (default: TIBLENDER_TELEGRAM_API_HASH)"
    )
    args = parser.parse_args(argv)

    _COMMANDS = {
        None,
        "run",
        "show-config",
        "audit-contract",
        "doctor",
        "collect",
        "build-viewer",
        "cycle",
        "run-cycles",
        "demo",
        "dqn-demo",
        "verify-sample",
        "write-expected",
        "telegram-login",
    }
    if args.command not in _COMMANDS:
        parser.error(f"unsupported command: {args.command}")

    if args.command == "verify-sample":
        return _run_verify_sample(args)

    if args.command == "write-expected":
        return _run_write_expected(args)

    try:
        # Every settings-backed command funnels through here, so an invalid
        # enumerated value (TIBLENDER_MODE=ofline, TIBLENDER_CLOCK=frozen, ...)
        # is reported once, as a configuration error, instead of silently
        # degrading into the `auto` path with live network tools.
        settings = Settings.from_env()
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.command == "build-viewer":
        # Settings-backed (unlike verify-sample/write-expected) because the
        # build stamp follows the run's clock mode.
        return _run_build_viewer(settings, args)

    if args.command == "demo":
        return _run_demo(settings, args)

    if args.command == "dqn-demo":
        return _run_dqn_demo(settings, args)

    if args.command == "telegram-login":
        return _run_telegram_login(settings, args)

    if args.command == "cycle":
        return _run_scheduler_cycle(settings, args)

    if args.command == "run-cycles":
        return _run_scheduler_run_cycles(settings, args)

    if args.command == "audit-contract":
        result = audit_contract()
        print(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1

    if args.command == "show-config":
        print(json.dumps(_redacted_settings(settings), ensure_ascii=False, indent=2))
        return 0

    if args.command == "doctor":
        print(json.dumps(doctor_report(settings), ensure_ascii=False, indent=2))
        return 0

    if args.command == "collect":
        return _run_collect(settings, args)

    if getattr(args, "input", None) is not None:
        settings.input_path = args.input
    if getattr(args, "output", None) is not None:
        settings.output_dir = args.output
    if getattr(args, "llm_provider", None) is not None:
        settings.llm_provider = args.llm_provider
    if getattr(args, "llm_model", None) is not None:
        settings.llm_model = args.llm_model
    if getattr(args, "allow_heuristic_reports", False):
        settings.allow_heuristic_reports = True

    try:
        # Return code 2 indicates configuration/generation failure rather than
        # a malformed command-line invocation.
        result = run_cycle(settings)
    except (MissingLLMConfigurationError, LLMGenerationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"reports_generated={result.reports_generated}")
    print(f"output_dir={result.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by a subprocess test
    # Without this guard ``python -m tiblender_runtime.cli <command>`` imports
    # the module, defines ``main`` and exits 0 having done nothing at all.
    raise SystemExit(main())
