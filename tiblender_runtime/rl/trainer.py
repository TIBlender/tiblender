"""Behaviour cloning, DQN fine-tuning, and policy-state assembly."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..settings import Settings
from .actions import ACTION_SPACE, FOCUS_PERSPECTIVE, FOCUS_TYPES, PERSPECTIVES, STOP_INDEX
from .features import STATE_DIM
from .network import MultiActionQNetwork
from .replay import ReplayBuffer, Transition


logger = logging.getLogger(__name__)

POLICY_TYPE = "multiaction_dqn"
POLICY_VERSION = 2
MIN_BC_TRANSITIONS = 8
MIN_VALIDATION_TRANSITIONS = 10
TARGET_SYNC_EVERY = 10
HUBER_DELTA = 1.0
TRAINING_HISTORY_LIMIT = 50

STATUS_BC_ONLY = "bc_only_insufficient_transitions"
STATUS_BC_AND_DQN = "bc_and_dqn"

_FOCUS_PERSPECTIVE_INDEX = np.array(
    [PERSPECTIVES.index(FOCUS_PERSPECTIVE[focus]) for focus in FOCUS_TYPES], dtype=np.int64
)


@dataclass(slots=True)
class TrainingReport:
    bc_epochs_run: int
    bc_train_loss: list[float]
    bc_val_loss: list[float]
    dqn_episodes_run: int
    dqn_loss: list[float]
    transitions: int
    status: str


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _stack(transitions: list[Transition]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.array([transition.state for transition in transitions], dtype=np.float64)
    actions = np.array([transition.action for transition in transitions], dtype=np.int64)
    num_perspectives = np.array([transition.num_perspectives for transition in transitions], dtype=np.int64)
    return states, actions, num_perspectives


def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean softmax cross-entropy and its gradient w.r.t. ``logits``."""
    probabilities = _softmax(logits)
    rows = np.arange(targets.size)
    loss = float(np.mean(-np.log(probabilities[rows, targets] + 1e-12)))
    delta = probabilities.copy()
    delta[rows, targets] -= 1.0
    return loss, delta / max(1, targets.size)


def _bc_terms(
    net: MultiActionQNetwork,
    states: np.ndarray,
    actions: np.ndarray,
    num_perspectives: np.ndarray,
    *,
    train: bool,
) -> tuple[float, dict[str, np.ndarray]]:
    """Cross-entropy on the focus, perspective, and num_actions heads, plus stop BCE.

    The STOP column of ``q_action`` is not touched here: it is a Q value and is
    trained by :func:`dqn_finetune`. Ending an episode is behaviour-cloned on the
    dedicated ``stop`` head instead.
    """
    out = net.forward(states, train=train)
    batch = states.shape[0]
    is_stop = actions == STOP_INDEX
    focus_rows = np.flatnonzero(~is_stop)

    stop_targets = is_stop.astype(np.float64).reshape(-1, 1)
    stop_logits = out["stop"]
    stop_loss = float(np.mean(_softplus(stop_logits) - stop_targets * stop_logits))
    grad_stop = (_sigmoid(stop_logits) - stop_targets) / batch

    grad_action = np.zeros_like(out["q_action"])
    grad_perspective = np.zeros_like(out["perspective"])
    grad_num_actions = np.zeros_like(out["num_actions"])
    focus_loss = 0.0
    perspective_loss = 0.0
    num_actions_loss = 0.0

    if focus_rows.size:
        focus_actions = actions[focus_rows]
        focus_loss, focus_delta = _cross_entropy(out["q_focus"][focus_rows], focus_actions)
        grad_action[focus_rows, : net.n_focus] = focus_delta

        perspective_targets = _FOCUS_PERSPECTIVE_INDEX[focus_actions]
        perspective_loss, perspective_delta = _cross_entropy(out["perspective"][focus_rows], perspective_targets)
        grad_perspective[focus_rows] = perspective_delta

    # K is a property of the whole episode, so every transition that knows it
    # (num_perspectives > 0) supervises the head, STOP transitions included.
    k_rows = np.flatnonzero(num_perspectives > 0)
    if k_rows.size:
        k_targets = np.clip(num_perspectives[k_rows], 1, net.max_k) - 1
        num_actions_loss, k_delta = _cross_entropy(out["num_actions"][k_rows], k_targets)
        grad_num_actions[k_rows] = k_delta

    total = focus_loss + perspective_loss + num_actions_loss + stop_loss
    return total, {
        "q_action": grad_action,
        "perspective": grad_perspective,
        "num_actions": grad_num_actions,
        "stop": grad_stop,
    }


def _bc_eval_loss(
    net: MultiActionQNetwork,
    states: np.ndarray,
    actions: np.ndarray,
    num_perspectives: np.ndarray,
) -> float:
    loss, _ = _bc_terms(net, states, actions, num_perspectives, train=False)
    return loss


def behavior_cloning(
    net: MultiActionQNetwork,
    transitions: list[Transition],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    val_split: float,
    rng: np.random.Generator,
) -> tuple[int, list[float], list[float]]:
    """Warm-start the network on observed investigation actions.

    Returns ``(epochs_run, train_losses, val_losses)``. With fewer than
    ``MIN_VALIDATION_TRANSITIONS`` samples every transition is used for training
    and the validation history is empty (no early stopping is possible).
    """
    if not transitions:
        return 0, [], []
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")

    states, actions, num_perspectives = _stack(transitions)
    order = rng.permutation(len(transitions))
    if len(transitions) < MIN_VALIDATION_TRANSITIONS:
        train_index, val_index = order, np.array([], dtype=np.int64)
    else:
        n_val = max(1, int(round(len(transitions) * val_split)))
        val_index, train_index = order[:n_val], order[n_val:]

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    stalled = 0
    epochs_run = 0

    for _ in range(epochs):
        shuffled = rng.permutation(train_index)
        for start in range(0, shuffled.size, batch_size):
            batch = shuffled[start : start + batch_size]
            if batch.size == 0:
                continue
            _, grads = _bc_terms(net, states[batch], actions[batch], num_perspectives[batch], train=True)
            net.zero_grad()
            net.backward(grads)
            net.step(lr)
        epochs_run += 1
        train_losses.append(_bc_eval_loss(net, states[train_index], actions[train_index], num_perspectives[train_index]))
        if val_index.size == 0:
            continue
        current = _bc_eval_loss(net, states[val_index], actions[val_index], num_perspectives[val_index])
        val_losses.append(current)
        if current < best_val - 1e-9:
            best_val = current
            stalled = 0
            continue
        stalled += 1
        if stalled >= patience:
            break
    return epochs_run, train_losses, val_losses


def _online_q(out: dict[str, np.ndarray], actions: np.ndarray) -> np.ndarray:
    """Q(s, a) over the full 19-action space, STOP included."""
    return out["q_action"][np.arange(actions.size), actions]


def dqn_finetune(
    net: MultiActionQNetwork,
    target: MultiActionQNetwork,
    buffer: ReplayBuffer,
    *,
    episodes: int,
    batch_size: int,
    lr: float,
    gamma: float,
    epsilon: float,
    rng: np.random.Generator,
) -> list[float]:
    """Off-policy DQN fine-tuning over the replay buffer.

    Q values come from the 19-way ``q_action`` head, so STOP is learned as an
    ordinary action. The TD target is greedy:
    ``r + gamma * max_a' Q_target(s') * (1 - done)``.

    ``epsilon`` is the exploration rate from the paper's schedule. These episodes
    replay recorded trajectories instead of acting, so there is nothing here for
    it to explore; ``train_policy`` advances the schedule with
    :func:`decay_epsilon` and persists the result.
    """
    del epsilon
    if episodes <= 0:
        return []
    if len(buffer) == 0:
        return []

    losses: list[float] = []
    for episode in range(episodes):
        batch = buffer.sample(batch_size, rng)
        states, actions, _ = _stack(batch)
        next_states = np.array([transition.next_state for transition in batch], dtype=np.float64)
        rewards = np.array([transition.reward for transition in batch], dtype=np.float64)
        done = np.array([transition.done for transition in batch], dtype=np.float64)

        out = net.forward(states, train=True)
        online_q = _online_q(out, actions)

        target_out = target.forward(next_states, train=False)
        next_best = target_out["q_action"].max(axis=1)
        td_target = rewards + gamma * next_best * (1.0 - done)

        difference = online_q - td_target
        clipped = np.clip(difference, -HUBER_DELTA, HUBER_DELTA)
        loss = float(np.mean(np.where(np.abs(difference) <= HUBER_DELTA, 0.5 * difference**2, np.abs(difference) - 0.5)))
        grad_q = clipped / actions.size

        grad_action = np.zeros_like(out["q_action"])
        grad_action[np.arange(actions.size), actions] = grad_q

        net.zero_grad()
        net.backward({"q_action": grad_action})
        net.step(lr)
        losses.append(loss)

        if (episode + 1) % TARGET_SYNC_EVERY == 0:
            target.load_from(net)
    return losses


def decay_epsilon(initial: float, episodes: int, decay: float, minimum: float) -> float:
    """Advance the epsilon-greedy schedule by ``episodes`` steps, floored at ``minimum``."""
    return max(float(minimum), float(initial) * float(decay) ** max(0, int(episodes)))


def resume_epsilon(previous_state: dict, settings: Settings) -> float:
    """Where this cycle's epsilon-greedy schedule picks up.

    The schedule is cumulative across cycles, so it continues from the epsilon
    persisted in the previous policy state. Only a first cycle -- or a state
    whose stored epsilon is missing or unusable -- starts over at
    ``epsilon_init``. Decaying from ``epsilon_init`` every cycle made the
    persisted value a function of that one cycle's episode count, and a cycle
    that ran no DQN episodes silently reset it back to 1.0.
    """
    stored = (previous_state or {}).get("epsilon")
    if stored is None:
        return float(settings.reference_rl_epsilon_init)
    try:
        value = float(stored)
    except (TypeError, ValueError):
        return float(settings.reference_rl_epsilon_init)
    # 0.0 is a legitimate fully-decayed value when epsilon_min is 0; only a
    # negative or non-finite number is unusable.
    if not math.isfinite(value) or value < 0.0:
        return float(settings.reference_rl_epsilon_init)
    return value


def _network_config(settings: Settings) -> dict[str, Any]:
    return {
        "state_dim": STATE_DIM,
        "hidden_dim": max(4, int(settings.reference_rl_hidden_dim)),
        "n_focus": len(FOCUS_TYPES),
        "n_perspectives": len(PERSPECTIVES),
        "max_k": len(PERSPECTIVES),
        "dropout": float(settings.reference_rl_dropout),
        "seed": int(settings.seed),
    }


def load_network(payload: Any, settings: Settings) -> MultiActionQNetwork:
    """Restore a saved network, or build a fresh one when it is absent or stale."""
    config = _network_config(settings)
    if isinstance(payload, dict) and payload.get("config") == config:
        try:
            return MultiActionQNetwork.from_dict(payload)
        except (ValueError, KeyError, TypeError) as error:
            logger.warning("policy network payload could not be restored (%s); starting from scratch", error)
    elif payload:
        logger.info("policy network configuration changed; starting from scratch")
    return MultiActionQNetwork(**config)


def load_buffer(payload: Any, settings: Settings) -> ReplayBuffer:
    """Restore the replay buffer, or start an empty one on malformed payloads."""
    capacity = max(1, int(settings.reference_rl_replay_capacity))
    if isinstance(payload, dict):
        try:
            restored = ReplayBuffer.from_dict({**payload, "capacity": capacity})
        except (ValueError, KeyError, TypeError) as error:
            logger.warning("replay payload could not be restored (%s); starting empty", error)
        else:
            return restored
    return ReplayBuffer(capacity=capacity)


def _hyperparameters(settings: Settings) -> dict[str, Any]:
    return {
        "state_dim": STATE_DIM,
        "action_space_size": len(ACTION_SPACE),
        "encoder_hidden_dims": [settings.reference_rl_hidden_dim, settings.reference_rl_hidden_dim],
        "dropout": settings.reference_rl_dropout,
        "max_steps": settings.rl_max_steps,
        "action_top_k": settings.rl_action_top_k,
        "bc_epochs": settings.reference_rl_bc_epochs,
        "bc_batch_size": settings.reference_rl_bc_batch_size,
        "bc_learning_rate": settings.reference_rl_bc_learning_rate,
        "bc_patience": settings.reference_rl_bc_patience,
        "bc_validation_split": settings.reference_rl_bc_validation_split,
        "dqn_episodes": settings.reference_rl_dqn_episodes,
        "dqn_learning_rate": settings.reference_rl_dqn_learning_rate,
        "discount": settings.reference_rl_discount,
        "replay_capacity": settings.reference_rl_replay_capacity,
        "training_min_transitions": settings.reference_rl_training_min_transitions,
        "epsilon_init": settings.reference_rl_epsilon_init,
        "epsilon_min": settings.reference_rl_epsilon_min,
        "epsilon_decay": settings.reference_rl_epsilon_decay,
        "similarity_gate": settings.rl_similarity_gate,
    }


def train_policy(
    previous_state: dict,
    transitions: list[Transition],
    settings: Settings,
    *,
    rng: np.random.Generator,
) -> dict:
    """Run BC on the new transitions, then DQN once the replay buffer is large enough."""
    previous = previous_state or {}
    net = load_network(previous.get("network"), settings)
    buffer = load_buffer(previous.get("replay"), settings)
    buffer.extend(transitions)

    bc_epochs_run, bc_train_loss, bc_val_loss = 0, [], []
    if len(transitions) >= MIN_BC_TRANSITIONS:
        bc_epochs_run, bc_train_loss, bc_val_loss = behavior_cloning(
            net,
            list(transitions),
            epochs=settings.reference_rl_bc_epochs,
            batch_size=settings.reference_rl_bc_batch_size,
            lr=settings.reference_rl_bc_learning_rate,
            patience=settings.reference_rl_bc_patience,
            val_split=settings.reference_rl_bc_validation_split,
            rng=rng,
        )

    dqn_loss: list[float] = []
    episodes_run = 0
    status = STATUS_BC_ONLY
    epsilon = resume_epsilon(previous, settings)
    if len(buffer) >= settings.reference_rl_training_min_transitions:
        episodes_run = max(0, int(settings.reference_rl_dqn_episodes))
        dqn_loss = dqn_finetune(
            net,
            net.copy(),
            buffer,
            episodes=episodes_run,
            batch_size=settings.reference_rl_bc_batch_size,
            lr=settings.reference_rl_dqn_learning_rate,
            gamma=settings.reference_rl_discount,
            epsilon=epsilon,
            rng=rng,
        )
        status = STATUS_BC_AND_DQN

    report = TrainingReport(
        bc_epochs_run=bc_epochs_run,
        bc_train_loss=[round(value, 6) for value in bc_train_loss],
        bc_val_loss=[round(value, 6) for value in bc_val_loss],
        dqn_episodes_run=episodes_run,
        dqn_loss=[round(value, 6) for value in dqn_loss],
        transitions=len(transitions),
        status=status,
    )
    history = list(previous.get("training_history") or [])
    history.append(asdict(report))

    return {
        "policy_type": POLICY_TYPE,
        "version": POLICY_VERSION,
        "network": net.to_dict(),
        "replay": buffer.to_dict(),
        # Cumulative across cycles: continue from the persisted epsilon and
        # advance it by the episodes this cycle actually ran. A cycle with zero
        # DQN episodes leaves it where it was -- still floored at epsilon_min,
        # so raising that floor between runs takes effect immediately rather
        # than never (every bundled cycle takes the zero-episode branch).
        "epsilon": round(
            decay_epsilon(
                epsilon,
                episodes_run,
                settings.reference_rl_epsilon_decay,
                settings.reference_rl_epsilon_min,
            )
            if episodes_run
            else max(float(settings.reference_rl_epsilon_min), epsilon),
            6,
        ),
        "hyperparameters": _hyperparameters(settings),
        "training_history": history[-TRAINING_HISTORY_LIMIT:],
        "report_history": list(previous.get("report_history") or []),
        "training_status": status,
    }
