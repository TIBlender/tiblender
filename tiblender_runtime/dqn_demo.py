"""Opt-in demo that drives the bundled fixtures into DQN fine-tuning.

The bundled offline demo's three cycles accumulate a few hundred replay
transitions, short of the paper's 1,000-transition training threshold
(``reference_rl_training_min_transitions``), so every cycle there runs
behaviour cloning only. This demo reruns the same fixtures with only that
threshold lowered, so the later cycles cross it and run BC followed by DQN
fine-tuning, then checks from the persisted policy state that they did.

It demonstrates that the stage executes and updates the policy on real
pipeline transitions. It does not claim the fine-tuned policy is better: a
few hundred transitions are far too few for the TD loss to converge.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .rl.trainer import STATUS_BC_AND_DQN
from .scheduler import run_cycles
from .settings import Settings

#: Low enough that cycle 2 of the bundled fixtures crosses it, high enough
#: that cycle 1 alone does not, so one run shows both sides of the gate.
DQN_DEMO_MIN_TRANSITIONS = 100
DQN_DEMO_CYCLES = 3
DQN_DEMO_OUTPUT_ROOT = Path("output/dqn_demo")
TRAINING_SUMMARY_FILENAME = "dqn_training.json"
POLICY_STATE_RELATIVE_PATH = Path("history") / "policy_state.json"
#: The paper's threshold, i.e. the field default the main demo runs with.
PAPER_MIN_TRANSITIONS = Settings().reference_rl_training_min_transitions


@dataclass(frozen=True)
class CycleTraining:
    cycle: int
    status: str
    transitions: int
    replay_buffer_size: int
    bc_epochs_run: int
    dqn_episodes_run: int
    dqn_loss_first: float | None
    dqn_loss_last: float | None


@dataclass(frozen=True)
class DqnDemoResult:
    min_transitions: int
    dqn_episodes_per_cycle: int
    cycles: tuple[CycleTraining, ...]
    epsilon: float | None
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "min_transitions": self.min_transitions,
            "dqn_episodes_per_cycle": self.dqn_episodes_per_cycle,
            "epsilon": self.epsilon,
            "cycles": [asdict(cycle) for cycle in self.cycles],
            "problems": list(self.problems),
        }


def dqn_demo_settings(settings: Settings, min_transitions: int = DQN_DEMO_MIN_TRANSITIONS) -> Settings:
    """``settings`` with only the DQN training threshold lowered (a copy)."""
    if min_transitions <= 0:
        raise ValueError("min_transitions must be positive")
    return replace(settings, reference_rl_training_min_transitions=min_transitions)


def _cycle_training(index: int, entry: dict, buffer_size: int) -> CycleTraining:
    losses = list(entry.get("dqn_loss") or [])
    return CycleTraining(
        cycle=index,
        status=str(entry.get("status", "")),
        transitions=int(entry.get("transitions", 0)),
        replay_buffer_size=buffer_size,
        bc_epochs_run=int(entry.get("bc_epochs_run", 0)),
        dqn_episodes_run=int(entry.get("dqn_episodes_run", 0)),
        dqn_loss_first=float(losses[0]) if losses else None,
        dqn_loss_last=float(losses[-1]) if losses else None,
    )


def _dqn_cycle_problems(name: str, entry: dict, episodes: int) -> list[str]:
    problems: list[str] = []
    losses = list(entry.get("dqn_loss") or [])
    if int(entry.get("dqn_episodes_run", 0)) != episodes:
        problems.append(f"{name}: ran {entry.get('dqn_episodes_run')} DQN episodes, expected {episodes}")
    if len(losses) != episodes:
        problems.append(f"{name}: recorded {len(losses)} DQN loss values, expected {episodes}")
    if not all(math.isfinite(float(value)) for value in losses):
        problems.append(f"{name}: DQN loss trace has non-finite values")
    return problems


def check_dqn_demo(policy_state: dict, settings: Settings) -> DqnDemoResult:
    """Read the run's training history and report whether DQN fine-tuning ran as configured."""
    history = policy_state.get("training_history")
    episodes = int(settings.reference_rl_dqn_episodes)
    epsilon_raw = policy_state.get("epsilon")
    epsilon = float(epsilon_raw) if epsilon_raw is not None else None
    if not isinstance(history, list) or not history:
        return DqnDemoResult(
            min_transitions=settings.reference_rl_training_min_transitions,
            dqn_episodes_per_cycle=episodes,
            cycles=(),
            epsilon=epsilon,
            problems=("the policy state has no training_history",),
        )

    cycles: list[CycleTraining] = []
    problems: list[str] = []
    buffer_size = 0
    for index, entry in enumerate(history, start=1):
        buffer_size = min(settings.reference_rl_replay_capacity, buffer_size + int(entry.get("transitions", 0)))
        cycles.append(_cycle_training(index, entry, buffer_size))
        if entry.get("status") == STATUS_BC_AND_DQN:
            problems.extend(_dqn_cycle_problems(f"cycle_{index:03d}", entry, episodes))

    # The per-cycle sizes are reconstructed from the history; the persisted
    # buffer must agree with the last one, or the table would misreport it.
    stored_size = (policy_state.get("replay") or {}).get("size")
    if stored_size != buffer_size:
        problems.append(f"the persisted replay buffer holds {stored_size} transitions, the history implies {buffer_size}")

    if not any(cycle.status == STATUS_BC_AND_DQN for cycle in cycles):
        problems.append(
            f"no cycle ran DQN fine-tuning (threshold {settings.reference_rl_training_min_transitions} "
            f"transitions, replay buffer reached {buffer_size})"
        )
    elif epsilon is None or epsilon >= float(settings.reference_rl_epsilon_init):
        problems.append(f"epsilon did not decay after DQN episodes (stored {epsilon_raw})")

    return DqnDemoResult(
        min_transitions=settings.reference_rl_training_min_transitions,
        dqn_episodes_per_cycle=episodes,
        cycles=tuple(cycles),
        epsilon=epsilon,
        problems=tuple(problems),
    )


def run_dqn_demo(
    settings: Settings,
    *,
    output_root: Path = DQN_DEMO_OUTPUT_ROOT,
    cycles: int = DQN_DEMO_CYCLES,
    min_transitions: int = DQN_DEMO_MIN_TRANSITIONS,
) -> DqnDemoResult:
    """Run the bundled cycles with the lowered threshold, check, and write the summary."""
    demo_settings = dqn_demo_settings(settings, min_transitions)
    run_cycles(demo_settings, cycles=cycles, output_root=output_root, interval_seconds=0)
    state_path = output_root / POLICY_STATE_RELATIVE_PATH
    policy_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    result = check_dqn_demo(policy_state, demo_settings)
    (output_root / TRAINING_SUMMARY_FILENAME).write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return result


def format_dqn_demo(result: DqnDemoResult) -> str:
    """The console table `tiblender dqn-demo` prints."""
    lines = [
        f"min_transitions={result.min_transitions} (paper setting: {PAPER_MIN_TRANSITIONS})",
        f"dqn_episodes_per_cycle={result.dqn_episodes_per_cycle}",
        "cycle  transitions  replay_buffer  status                            dqn_loss first -> last",
    ]
    for cycle in result.cycles:
        loss = (
            f"{cycle.dqn_loss_first:.3f} -> {cycle.dqn_loss_last:.3f}"
            if cycle.dqn_loss_first is not None and cycle.dqn_loss_last is not None
            else "-"
        )
        lines.append(
            f"{cycle.cycle:5d}  {cycle.transitions:11d}  {cycle.replay_buffer_size:13d}  {cycle.status:32s}  {loss}"
        )
    lines.append(f"epsilon={result.epsilon}")
    lines.extend(f"problem: {problem}" for problem in result.problems)
    lines.append(f"dqn_demo: {'ok' if result.ok else 'FAILED'}")
    return "\n".join(lines)
