import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tiblender_runtime import cli
from tiblender_runtime.cli import main


def test_show_config_redacts_api_keys(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-azure-key")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-github-token")
    monkeypatch.setenv("X_BEARER_TOKEN", "dummy-x-token")
    monkeypatch.setenv("TIBLENDER_JUDGE_A_OPENAI_API_KEY", "dummy-judge-a")
    monkeypatch.setenv("TIBLENDER_JUDGE_B_OPENAI_API_KEY", "dummy-judge-b")
    monkeypatch.setenv("TIBLENDER_TELEGRAM_API_HASH", "dummy-telegram-hash")
    monkeypatch.setenv("TIBLENDER_TELEGRAM_SESSION", "dummy-telegram-session")
    monkeypatch.setenv("TIBLENDER_DISCORD_BOT_TOKEN", "dummy-discord-token")
    monkeypatch.setenv("TIBLENDER_DISCORD_USER_TOKEN", "dummy-discord-user-token")
    monkeypatch.setenv("TIBLENDER_VIRUSTOTAL_API_KEY", "dummy-vt-key")
    monkeypatch.setenv("TIBLENDER_REDDIT_CLIENT_SECRET", "dummy-reddit-secret")
    monkeypatch.setenv("TIBLENDER_REDDIT_EXTRA_ACCOUNTS", "extra-client:dummy-extra-secret")

    assert main(["show-config"]) == 0

    output = capsys.readouterr().out
    assert "dummy-openai-key" not in output
    assert "dummy-github-token" not in output
    assert "dummy-x-token" not in output
    assert "dummy-judge-a" not in output
    assert "dummy-judge-b" not in output
    assert "dummy-telegram-hash" not in output
    # A Telegram session string authenticates as the user on its own.
    assert "dummy-telegram-session" not in output
    assert "dummy-discord-token" not in output
    assert "dummy-discord-user-token" not in output
    assert "dummy-vt-key" not in output
    assert "dummy-reddit-secret" not in output
    assert "dummy-extra-secret" not in output
    payload = json.loads(output)
    assert payload["openai_api_key"] == "<set>"
    assert payload["github_token"] == "<set>"
    assert payload["x_bearer_token"] == "<set>"
    assert payload["judge_a_api_key"] == "<set>"
    assert payload["judge_b_api_key"] == "<set>"
    assert payload["telegram_api_hash"] == "<set>"
    assert payload["telegram_session"] == "<set>"
    assert payload["discord_bot_token"] == "<set>"
    assert payload["discord_user_token"] == "<set>"
    assert payload["virustotal_api_key"] == "<set>"
    assert payload["reddit_client_secret"] == "<set>"
    assert payload["reddit_extra_accounts"] == "<set>"


def test_doctor_reports_service_modes_without_leaking_secrets(monkeypatch, capsys):
    monkeypatch.setenv("TIBLENDER_MODE", "offline")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("TIBLENDER_VIRUSTOTAL_API_KEY", "dummy-vt-key")
    monkeypatch.setenv("TIBLENDER_DISCORD_BOT_TOKEN", "dummy-discord-token")

    assert main(["doctor"]) == 0

    output = capsys.readouterr().out
    assert "dummy-openai-key" not in output
    assert "dummy-vt-key" not in output
    assert "dummy-discord-token" not in output
    payload = json.loads(output)
    assert payload["mode"] == "offline"
    assert isinstance(payload["services"], list) and payload["services"]
    assert {item["name"] for item in payload["services"]} >= {"llm", "x", "reddit", "telegram", "discord", "whois", "passive_dns"}
    assert "fixtures_present" in payload


# ---------------------------------------------------------------------------
# verify-sample / write-expected: exit 2 (config/input error) on a missing or
# malformed input, never a raw traceback.
# ---------------------------------------------------------------------------


def test_verify_sample_exits_2_for_a_missing_output_root(tmp_path, capsys):
    exit_code = main(
        [
            "verify-sample",
            "--output-root",
            str(tmp_path / "does-not-exist"),
            "--expected",
            str(tmp_path / "expected.json"),
        ]
    )

    assert exit_code == 2
    assert "error:" in capsys.readouterr().err


def test_verify_sample_exits_2_for_a_missing_expected_file(tmp_path, capsys):
    (tmp_path / "output").mkdir()

    exit_code = main(
        [
            "verify-sample",
            "--output-root",
            str(tmp_path / "output"),
            "--expected",
            str(tmp_path / "no-such-expected.json"),
        ]
    )

    assert exit_code == 2
    assert "error:" in capsys.readouterr().err


def test_write_expected_exits_2_for_a_missing_output_root(tmp_path, capsys):
    exit_code = main(
        [
            "write-expected",
            "--output-root",
            str(tmp_path / "does-not-exist"),
            "--expected-dir",
            str(tmp_path / "expected"),
        ]
    )

    assert exit_code == 2
    assert "error:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# telegram-login: only the two guarded failure paths are unit-testable here
# (missing telethon, missing api_id/api_hash). The interactive
# TelegramClient.start() login flow itself requires a real network connection
# and human input, so it is exercised manually per docs/CREDENTIALS.md, not
# in this offline test suite.
# ---------------------------------------------------------------------------


def test_telegram_login_exits_2_when_telethon_is_missing(monkeypatch, tmp_path, capsys):
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "telethon":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(cli.importlib.util, "find_spec", fake_find_spec)

    exit_code = main(
        [
            "telegram-login",
            "--session",
            str(tmp_path / "demo"),
            "--api-id",
            "12345",
            "--api-hash",
            "deadbeefdeadbeefdeadbeefdeadbeef",
        ]
    )

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "telethon" in err.lower()


def test_telegram_login_exits_2_when_credentials_are_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("TIBLENDER_TELEGRAM_API_ID", raising=False)
    monkeypatch.delenv("TIBLENDER_TELEGRAM_API_HASH", raising=False)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name, *a, **k: object())

    exit_code = main(["telegram-login", "--session", str(tmp_path / "demo")])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "api_id" in err and "api_hash" in err


# ---------------------------------------------------------------------------
# collect: regression guard for the shared-history-dir bug.
# `collect` used to crash with plain default settings because
# `Settings.history_dir` resolved to `None` where a `Path` was required;
# fixed by routing every reader/writer through `Settings.resolved_history_dir()`,
# which falls back to `output_dir / "history"` when `history_dir` is unset.
# `_run_collect` never sets `settings.output_dir = args.output` (unlike the
# `run` command), so that fallback is relative to `output_dir`'s own default
# ("output", relative to cwd) -- NOT to `--output`. This test exercises
# `collect` exactly as a bare CLI invocation would (no explicit
# TIBLENDER_OUTPUT_DIR/TIBLENDER_HISTORY_DIR), so it fails loudly if that
# default ever regresses to None again or silently changes which directory
# collector state lands in.
# ---------------------------------------------------------------------------


def test_collect_with_default_settings_returns_0_and_shares_the_default_history_dir(monkeypatch, tmp_path, capsys):
    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.setenv("TIBLENDER_FIXTURES_DIR", str(repo_root / "fixtures"))
    monkeypatch.setenv("TIBLENDER_CONFIG_DIR", str(repo_root / "config"))
    monkeypatch.setenv("TIBLENDER_MODE", "offline")
    monkeypatch.chdir(tmp_path)

    # Snapshot the repo checkout's own output/ BEFORE the run. It is a shared,
    # documented working directory (`make offline` and `docker compose up`
    # write their three cycles into it), so asserting it is empty would make
    # this test fail for anyone who followed the documented order. What must
    # hold is that *this* run added nothing to it.
    repo_output = repo_root / "output"
    before = sorted(path.name for path in repo_output.iterdir()) if repo_output.is_dir() else []

    output_dir = tmp_path / "out"
    exit_code = main(["collect", "--output", str(output_dir), "--cycle", "1"])

    assert exit_code == 0
    assert (output_dir / "posts.jsonl").exists()

    # settings.output_dir defaults to Path("output") (relative to cwd, now
    # tmp_path) and history_dir defaults to None, so resolved_history_dir()
    # falls back to output_dir/"history" -- a *different* directory from
    # --output ("out"). Asserting the state file exists there (not just that
    # the command exited 0) is what actually catches a regression back to a
    # None history_dir or a re-broken fallback.
    default_history_dir = tmp_path / "output" / "history"
    assert (default_history_dir / "collector_state.json").exists()

    # The default "output" must never resolve outside tmp_path (i.e. into
    # the real repo checkout) -- chdir(tmp_path) is what guarantees this;
    # assert it explicitly so a future removal of that chdir fails loudly.
    after = sorted(path.name for path in repo_output.iterdir()) if repo_output.is_dir() else []
    assert after == before, (
        f"the run wrote into the repo checkout's output/: {sorted(set(after) - set(before))}"
    )


# ---------------------------------------------------------------------------
# Invalid enumerated settings are a configuration error (exit 2), never a
# silent degrade into the auto path with live network tools.
# ---------------------------------------------------------------------------


def test_invalid_mode_exits_2_with_an_error_message(monkeypatch, capsys):
    monkeypatch.setenv("TIBLENDER_MODE", "ofline")

    assert main(["doctor"]) == 2

    captured = capsys.readouterr()
    assert captured.err.startswith("error: ")
    assert "ofline" in captured.err
    assert "offline" in captured.err  # names the allowed set
    assert captured.out == ""


def test_invalid_clock_mode_exits_2(monkeypatch, capsys):
    monkeypatch.setenv("TIBLENDER_CLOCK", "frozen")

    assert main(["show-config"]) == 2

    assert "frozen" in capsys.readouterr().err


def test_invalid_rl_policy_backend_exits_2(monkeypatch, capsys):
    monkeypatch.setenv("TIBLENDER_RL_POLICY_BACKEND", "torch_dqn")

    assert main(["show-config"]) == 2

    assert "torch_dqn" in capsys.readouterr().err


def test_run_cycles_and_cycle_expose_a_resume_flag(monkeypatch, tmp_path, capsys):
    # --resume must reach the scheduler; without it both commands start fresh.
    calls: list[dict] = []

    def _fake_run_cycles(settings, *, cycles, output_root, interval_seconds, start_index, resume):
        calls.append({"command": "run-cycles", "resume": resume})
        return []

    def _fake_run_one_cycle(settings, *, cycle_index, output_root, window_end, resume):
        calls.append({"command": "cycle", "resume": resume})

        class _Record:
            cycle_index = 1
            posts = 0
            reports = 0
            dynamic_queries = 0
            training_status = ""
            tool_recording_hit_rate = 0.0

        return _Record()

    monkeypatch.setattr(cli, "run_cycles", _fake_run_cycles)
    monkeypatch.setattr(cli, "run_one_cycle", _fake_run_one_cycle)

    assert main(["run-cycles", "--output-root", str(tmp_path), "--cycles", "1", "--interval", "0"]) == 0
    assert main(["run-cycles", "--output-root", str(tmp_path), "--cycles", "1", "--interval", "0", "--resume"]) == 0
    assert main(["cycle", "--output-root", str(tmp_path), "--cycle", "1"]) == 0
    assert main(["cycle", "--output-root", str(tmp_path), "--cycle", "1", "--resume"]) == 0

    assert [call["resume"] for call in calls] == [False, True, False, True]


def _module_entry_point(module: str) -> subprocess.CompletedProcess:
    """Run ``python -m <module> --help`` in a clean, offline environment."""
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
        "TIBLENDER_MODE": "offline",
        "TIBLENDER_ENV_FILE": os.devnull,
    }
    return subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parent.parent),
        timeout=120,
    )


@pytest.mark.parametrize("module", ["tiblender_runtime", "tiblender_runtime.cli"])
def test_both_module_entry_points_run_the_cli(module):
    """``-m tiblender_runtime`` and ``-m tiblender_runtime.cli`` both work.

    ``cli.py`` used to lack an ``if __name__ == "__main__"`` guard, so running
    the module directly defined ``main`` and exited 0 with no output at all --
    a silent no-op for a documented-looking invocation.
    """
    result = _module_entry_point(module)

    assert result.returncode == 0, result.stderr
    # A guard that is present but does nothing would still exit 0; the usage
    # banner is what proves the parser actually ran.
    assert "usage:" in result.stdout
    for command in ("run-cycles", "verify-sample", "doctor"):
        assert command in result.stdout


def test_the_two_module_entry_points_print_the_same_help():
    assert _module_entry_point("tiblender_runtime").stdout == _module_entry_point("tiblender_runtime.cli").stdout


# ---------------------------------------------------------------------------
# A credential-less `live` run must exit 2 having written nothing
# (docs/REFERENCE.md "Modes and determinism",
# PIPELINE_OVERVIEW "fails fast before any output is written").
# ---------------------------------------------------------------------------


def test_live_run_cycles_exits_2_without_writing_any_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("TIBLENDER_MODE", "live")
    monkeypatch.delenv("TIBLENDER_ALLOW_HEURISTIC_REPORTS", raising=False)
    output_root = tmp_path / "liveout"

    exit_code = main(["run-cycles", "--output-root", str(output_root), "--cycles", "1", "--interval", "0"])

    assert exit_code == 2
    assert not output_root.exists()
    captured = capsys.readouterr()
    assert captured.err.startswith("error: ")
    assert "TIBLENDER_LLM_PROVIDER" in captured.err


def test_live_cycle_exits_2_without_writing_any_output(monkeypatch, tmp_path):
    monkeypatch.setenv("TIBLENDER_MODE", "live")
    output_root = tmp_path / "liveout"

    assert main(["cycle", "--output-root", str(output_root), "--cycle", "1"]) == 2
    assert not output_root.exists()
