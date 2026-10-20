"""The opt-in DQN fine-tuning demo (`demo/run_dqn_demo.sh`, `tiblender dqn-demo`).

The bundled three cycles accumulate a few hundred transitions, short of the
paper's 1,000-transition training threshold, so the main demo only ever runs
behaviour cloning. The DQN demo reruns the same fixtures with the threshold
lowered so the DQN fine-tuning stage executes, and checks that it did.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from tiblender_runtime.cli import main
from tiblender_runtime.dqn_demo import (
    DQN_DEMO_MIN_TRANSITIONS,
    TRAINING_SUMMARY_FILENAME,
    check_dqn_demo,
    dqn_demo_settings,
    run_dqn_demo,
)
from tiblender_runtime.settings import Settings

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "demo" / "run_dqn_demo.sh"


def _history_entry(status: str, transitions: int, episodes: int, losses: list[float] | None = None) -> dict:
    return {
        "bc_epochs_run": 20,
        "bc_train_loss": [1.0],
        "bc_val_loss": [1.0],
        "dqn_episodes_run": episodes,
        "dqn_loss": losses if losses is not None else [9.0 - 0.01 * i for i in range(episodes)],
        "transitions": transitions,
        "status": status,
    }


def _policy_state(history: list[dict], epsilon: float = 0.01, replay_size: int | None = None) -> dict:
    size = replay_size if replay_size is not None else sum(entry["transitions"] for entry in history)
    return {"training_history": history, "epsilon": epsilon, "replay": {"size": size}}


def _settings() -> Settings:
    return dqn_demo_settings(Settings(mode="offline", seed=20260909))


def test_demo_settings_lower_only_the_training_threshold():
    base = Settings(mode="offline", seed=20260909)
    demo = dqn_demo_settings(base)

    assert demo.reference_rl_training_min_transitions == DQN_DEMO_MIN_TRANSITIONS
    assert base.reference_rl_training_min_transitions == 1_000, "the caller's settings must not change"
    assert demo.reference_rl_dqn_episodes == base.reference_rl_dqn_episodes
    assert demo.reference_rl_replay_capacity == base.reference_rl_replay_capacity
    assert demo.reference_rl_dqn_learning_rate == base.reference_rl_dqn_learning_rate


def test_demo_threshold_is_below_the_paper_threshold():
    assert 0 < DQN_DEMO_MIN_TRANSITIONS < Settings().reference_rl_training_min_transitions


def test_a_run_whose_later_cycles_fine_tune_passes():
    settings = _settings()
    episodes = settings.reference_rl_dqn_episodes
    state = _policy_state(
        [
            _history_entry("bc_only_insufficient_transitions", 62, 0, []),
            _history_entry("bc_and_dqn", 157, episodes),
            _history_entry("bc_and_dqn", 160, episodes),
        ]
    )

    result = check_dqn_demo(state, settings)

    assert result.ok, result.problems
    assert [cycle.status for cycle in result.cycles] == [
        "bc_only_insufficient_transitions",
        "bc_and_dqn",
        "bc_and_dqn",
    ]
    assert [cycle.replay_buffer_size for cycle in result.cycles] == [62, 219, 379]
    assert result.cycles[1].dqn_loss_first == pytest.approx(9.0)
    assert result.cycles[1].dqn_loss_last == pytest.approx(9.0 - 0.01 * (episodes - 1))
    assert result.cycles[0].dqn_loss_first is None


def test_a_run_that_never_reaches_dqn_fails():
    state = _policy_state([_history_entry("bc_only_insufficient_transitions", 62, 0, [])], epsilon=1.0)

    result = check_dqn_demo(state, _settings())

    assert not result.ok
    assert any("no cycle ran DQN fine-tuning" in problem for problem in result.problems)


def test_a_dqn_cycle_with_a_short_or_non_finite_loss_trace_fails():
    settings = _settings()
    episodes = settings.reference_rl_dqn_episodes
    state = _policy_state(
        [
            _history_entry("bc_and_dqn", 157, episodes, [1.0] * (episodes - 1)),
            _history_entry("bc_and_dqn", 160, episodes, [math.nan] * episodes),
        ]
    )

    problems = check_dqn_demo(state, settings).problems

    assert any("cycle_001" in problem and "loss values" in problem for problem in problems)
    assert any("cycle_002" in problem and "non-finite" in problem for problem in problems)


def test_a_dqn_cycle_that_ran_fewer_episodes_than_configured_fails():
    settings = _settings()
    state = _policy_state([_history_entry("bc_and_dqn", 157, 3, [1.0, 1.0, 1.0])])

    problems = check_dqn_demo(state, settings).problems

    assert any("cycle_001" in problem and "episodes" in problem for problem in problems)


def test_epsilon_that_did_not_decay_fails():
    settings = _settings()
    state = _policy_state(
        [_history_entry("bc_and_dqn", 157, settings.reference_rl_dqn_episodes)],
        epsilon=settings.reference_rl_epsilon_init,
    )

    problems = check_dqn_demo(state, settings).problems

    assert any("epsilon" in problem for problem in problems)


def test_a_replay_buffer_that_disagrees_with_the_history_fails():
    settings = _settings()
    state = _policy_state(
        [_history_entry("bc_and_dqn", 157, settings.reference_rl_dqn_episodes)],
        replay_size=150,
    )

    problems = check_dqn_demo(state, settings).problems

    assert any("replay buffer" in problem and "150" in problem for problem in problems)


def test_a_missing_training_history_fails_cleanly():
    result = check_dqn_demo({}, _settings())

    assert not result.ok
    assert any("training_history" in problem for problem in result.problems)


def test_the_script_is_strict_bash_and_calls_the_subcommand():
    content = _SCRIPT.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in content
    assert "BASH_SOURCE[0]" in content and 'cd -- "$ROOT"' in content
    assert "export TIBLENDER_MODE=offline" in content
    assert " dqn-demo" in content
    result = subprocess.run(["bash", "-n", str(_SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_script_writes_beside_not_over_the_main_demo():
    content = _SCRIPT.read_text(encoding="utf-8")
    assert 'OUTPUT_ROOT="output/dqn_demo"' in content
    assert "build-viewer" not in content and "--report-json" not in content


def test_the_subcommand_rejects_a_threshold_that_is_not_positive(capsys):
    assert main(["dqn-demo", "--min-transitions", "0", "--output-root", "unused"]) == 2
    assert "--min-transitions" in capsys.readouterr().err


@pytest.mark.slow
def test_the_bundled_fixtures_reach_dqn_fine_tuning(tmp_path: Path):
    output_root = tmp_path / "dqn_demo"

    result = run_dqn_demo(Settings(mode="offline", seed=20260909), output_root=output_root, cycles=3)

    assert result.ok, result.problems
    statuses = [cycle.status for cycle in result.cycles]
    assert statuses[0] == "bc_only_insufficient_transitions", "cycle 1 alone stays under the threshold"
    assert statuses[1:] == ["bc_and_dqn", "bc_and_dqn"]

    written = json.loads((output_root / TRAINING_SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert written["ok"] is True
    assert written["min_transitions"] == DQN_DEMO_MIN_TRANSITIONS
    assert [cycle["status"] for cycle in written["cycles"]] == statuses


def test_demo_settings_reject_a_threshold_that_is_not_positive():
    with pytest.raises(ValueError):
        dqn_demo_settings(Settings(mode="offline"), 0)


@pytest.mark.slow
def test_the_subcommand_prints_the_training_table(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("TIBLENDER_MODE", "offline")
    output_root = tmp_path / "dqn_demo"

    assert main(["dqn-demo", "--output-root", str(output_root)]) == 0

    printed = capsys.readouterr().out
    assert f"min_transitions={DQN_DEMO_MIN_TRANSITIONS} (paper setting: 1000)" in printed
    assert "bc_only_insufficient_transitions  -" in printed
    assert printed.count("bc_and_dqn") == 2
    assert printed.rstrip().endswith("dqn_demo: ok")
    assert (output_root / TRAINING_SUMMARY_FILENAME).is_file()
