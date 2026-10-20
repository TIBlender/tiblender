"""Behaviour-cloning, DQN fine-tuning, and policy-state training tests."""

import json

import numpy as np
import pytest

from tiblender_runtime.rl.actions import ACTION_SPACE, STOP_INDEX
from tiblender_runtime.rl.features import STATE_DIM
from tiblender_runtime.rl.network import MultiActionQNetwork
from tiblender_runtime.rl.replay import ReplayBuffer, Transition
from tiblender_runtime.rl.trainer import TrainingReport, behavior_cloning, dqn_finetune, train_policy
from tiblender_runtime.settings import Settings


def _fast_settings(**overrides) -> Settings:
    base = {
        "reference_rl_hidden_dim": 16,
        "reference_rl_bc_epochs": 3,
        "reference_rl_bc_batch_size": 8,
        "reference_rl_dqn_episodes": 4,
        "reference_rl_replay_capacity": 2000,
    }
    base.update(overrides)
    return Settings(**base)


def _separable_transitions(count: int = 48, actions: int = 4, num_perspectives: int = 0) -> list[Transition]:
    rng = np.random.default_rng(3)
    transitions: list[Transition] = []
    for index in range(count):
        action = index % actions
        state = np.zeros(STATE_DIM)
        state[action * 10 : action * 10 + 10] = 1.0
        state += rng.normal(scale=0.01, size=STATE_DIM)
        transitions.append(
            Transition(
                state=tuple(state.tolist()),
                action=action,
                reward=1.0,
                next_state=tuple(state.tolist()),
                done=False,
                num_perspectives=num_perspectives,
            )
        )
    return transitions


def _noise_transitions(count: int = 40) -> list[Transition]:
    rng = np.random.default_rng(9)
    transitions: list[Transition] = []
    for _ in range(count):
        state = rng.normal(size=STATE_DIM)
        transitions.append(
            Transition(
                state=tuple(state.tolist()),
                action=int(rng.integers(0, 18)),
                reward=float(rng.normal()),
                next_state=tuple(rng.normal(size=STATE_DIM).tolist()),
                done=False,
            )
        )
    return transitions


def test_behavior_cloning_reduces_training_loss_on_separable_data():
    net = MultiActionQNetwork(hidden_dim=32, dropout=0.0, seed=0)

    epochs_run, train_loss, val_loss = behavior_cloning(
        net,
        _separable_transitions(),
        epochs=12,
        batch_size=8,
        lr=0.02,
        patience=12,
        val_split=0.10,
        rng=np.random.default_rng(0),
    )

    assert epochs_run == 12
    assert len(train_loss) == 12
    assert len(val_loss) == 12
    assert train_loss[-1] < train_loss[0] * 0.5


def test_behavior_cloning_early_stops_when_validation_stops_improving():
    net = MultiActionQNetwork(hidden_dim=16, dropout=0.0, seed=1)

    epochs_run, train_loss, val_loss = behavior_cloning(
        net,
        _noise_transitions(),
        epochs=20,
        batch_size=8,
        lr=0.5,
        patience=1,
        val_split=0.25,
        rng=np.random.default_rng(2),
    )

    assert epochs_run < 20
    assert len(train_loss) == epochs_run
    assert len(val_loss) == epochs_run


def test_behavior_cloning_uses_every_transition_when_the_batch_is_tiny():
    net = MultiActionQNetwork(hidden_dim=8, dropout=0.0, seed=4)

    epochs_run, train_loss, val_loss = behavior_cloning(
        net,
        _separable_transitions(count=6),
        epochs=3,
        batch_size=4,
        lr=0.01,
        patience=3,
        val_split=0.10,
        rng=np.random.default_rng(0),
    )

    assert epochs_run == 3
    assert len(train_loss) == 3
    assert val_loss == []


def test_behavior_cloning_returns_nothing_without_transitions():
    net = MultiActionQNetwork(hidden_dim=8, seed=0)

    assert behavior_cloning(
        net,
        [],
        epochs=5,
        batch_size=4,
        lr=0.01,
        patience=2,
        val_split=0.10,
        rng=np.random.default_rng(0),
    ) == (0, [], [])


def test_behavior_cloning_learns_the_num_actions_head():
    net = MultiActionQNetwork(hidden_dim=32, dropout=0.0, seed=2)

    behavior_cloning(
        net,
        _separable_transitions(count=32, num_perspectives=3),
        epochs=15,
        batch_size=8,
        lr=0.05,
        patience=15,
        val_split=0.10,
        rng=np.random.default_rng(0),
    )

    states = np.array([transition.state for transition in _separable_transitions(count=8, num_perspectives=3)])
    predicted = np.argmax(net.forward(states)["num_actions"], axis=1)

    assert set(predicted.tolist()) == {2}


def test_behavior_cloning_leaves_num_actions_untrained_without_a_target():
    net = MultiActionQNetwork(hidden_dim=16, dropout=0.0, seed=2)
    before = net.parameters()["Wk"].copy()

    behavior_cloning(
        net,
        _separable_transitions(count=16),
        epochs=5,
        batch_size=8,
        lr=0.05,
        patience=5,
        val_split=0.10,
        rng=np.random.default_rng(0),
    )

    assert np.array_equal(net.parameters()["Wk"], before)


def test_behavior_cloning_does_not_touch_the_stop_q_value():
    net = MultiActionQNetwork(hidden_dim=16, dropout=0.0, seed=2)
    before = net.parameters()["Wq"][:, STOP_INDEX].copy()

    behavior_cloning(
        net,
        _separable_transitions(count=16),
        epochs=5,
        batch_size=8,
        lr=0.05,
        patience=5,
        val_split=0.10,
        rng=np.random.default_rng(0),
    )

    assert np.array_equal(net.parameters()["Wq"][:, STOP_INDEX], before)


def test_behavior_cloning_learns_the_stop_head():
    net = MultiActionQNetwork(hidden_dim=16, dropout=0.0, seed=6)
    state = np.zeros(STATE_DIM)
    state[:5] = 1.0
    transitions = [
        Transition(state=tuple(state.tolist()), action=STOP_INDEX, reward=0.0, next_state=tuple(state.tolist()), done=True)
    ] * 16

    behavior_cloning(
        net,
        transitions,
        epochs=20,
        batch_size=8,
        lr=0.05,
        patience=20,
        val_split=0.10,
        rng=np.random.default_rng(0),
    )

    assert float(net.forward(state.reshape(1, -1))["stop"][0, 0]) > 0.0


def test_dqn_finetune_trains_the_stop_q_value_as_an_ordinary_action():
    net = MultiActionQNetwork(hidden_dim=16, dropout=0.0, seed=8)
    target = net.copy()
    stop_state = np.zeros(STATE_DIM)
    stop_state[:5] = 1.0
    buffer = ReplayBuffer(capacity=50)
    buffer.extend(
        [
            Transition(
                state=tuple(stop_state.tolist()),
                action=STOP_INDEX,
                reward=100.0,
                next_state=tuple(stop_state.tolist()),
                done=True,
            )
        ]
        * 16
    )
    before_q = float(net.forward(stop_state.reshape(1, -1))["q_action"][0, STOP_INDEX])
    before_stop_head = net.parameters()["Ws"].copy()

    dqn_finetune(net, target, buffer, episodes=20, batch_size=8, lr=0.05, gamma=0.99, epsilon=1.0, rng=np.random.default_rng(0))

    assert float(net.forward(stop_state.reshape(1, -1))["q_action"][0, STOP_INDEX]) > before_q
    assert np.array_equal(net.parameters()["Ws"], before_stop_head)


def test_dqn_finetune_returns_one_loss_per_episode_and_improves():
    net = MultiActionQNetwork(hidden_dim=16, dropout=0.0, seed=8)
    target = net.copy()
    buffer = ReplayBuffer(capacity=200)
    buffer.extend(_separable_transitions(count=64))

    losses = dqn_finetune(
        net,
        target,
        buffer,
        episodes=30,
        batch_size=16,
        lr=0.01,
        gamma=0.99,
        epsilon=1.0,
        rng=np.random.default_rng(5),
    )

    assert len(losses) == 30
    assert all(np.isfinite(loss) for loss in losses)
    assert min(losses[-5:]) < max(losses[:5])


def test_dqn_finetune_syncs_the_target_network_every_ten_episodes():
    net = MultiActionQNetwork(hidden_dim=16, dropout=0.0, seed=8)
    target = net.copy()
    buffer = ReplayBuffer(capacity=200)
    buffer.extend(_separable_transitions(count=64))

    dqn_finetune(net, target, buffer, episodes=5, batch_size=16, lr=0.05, gamma=0.99, epsilon=1.0, rng=np.random.default_rng(5))
    assert not np.allclose(target.parameters()["Wq"], net.parameters()["Wq"])

    dqn_finetune(net, target, buffer, episodes=10, batch_size=16, lr=0.05, gamma=0.99, epsilon=1.0, rng=np.random.default_rng(5))
    assert np.allclose(target.parameters()["Wq"], net.parameters()["Wq"])


def test_train_policy_reports_insufficient_transitions_for_a_small_buffer():
    settings = _fast_settings()

    state = train_policy({}, _separable_transitions(count=16), settings, rng=np.random.default_rng(0))

    assert state["policy_type"] == "multiaction_dqn"
    assert state["version"] == 2
    assert state["training_status"] == "bc_only_insufficient_transitions"
    assert state["training_history"][-1]["status"] == "bc_only_insufficient_transitions"
    assert state["training_history"][-1]["dqn_episodes_run"] == 0
    assert state["training_history"][-1]["transitions"] == 16
    assert state["epsilon"] == 1.0
    assert state["report_history"] == []
    assert state["hyperparameters"]["state_dim"] == STATE_DIM
    assert state["hyperparameters"]["action_space_size"] == len(ACTION_SPACE)


def test_train_policy_runs_dqn_once_the_buffer_is_large_enough():
    settings = _fast_settings(reference_rl_training_min_transitions=1000)
    preloaded = ReplayBuffer(capacity=2000)
    preloaded.extend(_separable_transitions(count=1000))
    previous = {"replay": preloaded.to_dict(), "report_history": [{"report_id": "old"}]}

    state = train_policy(previous, _separable_transitions(count=16), settings, rng=np.random.default_rng(0))

    assert state["training_status"] == "bc_and_dqn"
    assert len(state["training_history"][-1]["dqn_loss"]) == 4
    assert state["epsilon"] == pytest.approx(max(0.01, 1.0 * 0.9**4))
    assert state["replay"]["size"] == 1016
    assert state["report_history"] == [{"report_id": "old"}]


def test_train_policy_continues_the_epsilon_schedule_from_the_persisted_value():
    """The schedule is cumulative across cycles, not a function of one cycle's
    episode count. Restarting from `epsilon_init` every cycle made the
    persisted number mean something other than what the docs describe."""
    settings = _fast_settings(reference_rl_training_min_transitions=1000)
    preloaded = ReplayBuffer(capacity=4000)
    preloaded.extend(_separable_transitions(count=1000))
    previous = {"replay": preloaded.to_dict(), "epsilon": 0.5}

    state = train_policy(previous, _separable_transitions(count=16), settings, rng=np.random.default_rng(0))

    assert state["training_history"][-1]["dqn_episodes_run"] == 4
    assert state["epsilon"] == pytest.approx(max(0.01, 0.5 * 0.9**4))


def test_train_policy_leaves_epsilon_untouched_in_a_cycle_with_no_dqn_episodes():
    """A cycle that runs no DQN episodes advances the schedule by nothing. It
    used to reset epsilon to `epsilon_init`, because `decay_epsilon(1.0, 0)`
    is 1.0 -- so a below-threshold cycle silently undid every earlier decay."""
    settings = _fast_settings()
    previous = {"epsilon": 0.32}

    state = train_policy(previous, _separable_transitions(count=16), settings, rng=np.random.default_rng(0))

    assert state["training_history"][-1]["dqn_episodes_run"] == 0
    assert state["epsilon"] == pytest.approx(0.32)


def test_train_policy_starts_the_epsilon_schedule_at_epsilon_init_without_prior_state():
    settings = _fast_settings()

    state = train_policy({}, _separable_transitions(count=16), settings, rng=np.random.default_rng(0))

    assert state["epsilon"] == pytest.approx(settings.reference_rl_epsilon_init)


def test_train_policy_ignores_an_unusable_persisted_epsilon():
    settings = _fast_settings()

    state = train_policy({"epsilon": "not-a-number"}, _separable_transitions(count=16), settings, rng=np.random.default_rng(0))

    assert state["epsilon"] == pytest.approx(settings.reference_rl_epsilon_init)


def test_train_policy_keeps_a_persisted_zero_epsilon():
    """With TIBLENDER_REFERENCE_RL_EPSILON_MIN=0 a fully decayed schedule
    persists 0.0. Treating that as "unusable" would jump it straight back to
    1.0 -- the exact reset this fix exists to remove."""
    settings = _fast_settings(reference_rl_epsilon_min=0.0)

    state = train_policy({"epsilon": 0.0}, _separable_transitions(count=16), settings, rng=np.random.default_rng(0))

    assert state["epsilon"] == pytest.approx(0.0)


def test_train_policy_rejects_a_negative_persisted_epsilon():
    settings = _fast_settings()

    state = train_policy({"epsilon": -0.5}, _separable_transitions(count=16), settings, rng=np.random.default_rng(0))

    assert state["epsilon"] == pytest.approx(settings.reference_rl_epsilon_init)


def test_train_policy_floors_a_carried_epsilon_at_a_raised_epsilon_min():
    """An operator who raises epsilon_min between runs must not have an older,
    lower persisted value carried forward unfloored by every zero-episode
    cycle (which is every cycle of the bundled three-cycle run)."""
    settings = _fast_settings(reference_rl_epsilon_min=0.4)

    state = train_policy({"epsilon": 0.05}, _separable_transitions(count=16), settings, rng=np.random.default_rng(0))

    assert state["training_history"][-1]["dqn_episodes_run"] == 0
    assert state["epsilon"] == pytest.approx(0.4)


def test_train_policy_seeds_the_rng_per_training_round():
    settings = _fast_settings()
    transitions = _separable_transitions(count=16)

    first = train_policy({}, transitions, settings, rng=np.random.default_rng(settings.seed))
    # feedback.update_policy_state offsets the seed by the number of prior rounds,
    # so a second cycle draws a different shuffle from the same base seed.
    same_seed = train_policy(first, transitions, settings, rng=np.random.default_rng(settings.seed))
    offset_seed = train_policy(first, transitions, settings, rng=np.random.default_rng(settings.seed + 1))

    assert same_seed["network"] != offset_seed["network"]


def test_train_policy_state_is_json_serialisable_and_deterministic():
    settings = _fast_settings()
    transitions = _separable_transitions(count=16)

    first = train_policy({}, transitions, settings, rng=np.random.default_rng(0))
    second = train_policy({}, transitions, settings, rng=np.random.default_rng(0))

    payload = json.dumps(first, sort_keys=True)
    assert payload == json.dumps(second, sort_keys=True)
    assert json.loads(payload)["network"]["config"]["hidden_dim"] == 16


def test_train_policy_resumes_from_a_previous_network():
    settings = _fast_settings()
    first = train_policy({}, _separable_transitions(count=16), settings, rng=np.random.default_rng(0))

    second = train_policy(first, _separable_transitions(count=16), settings, rng=np.random.default_rng(0))

    assert second["replay"]["size"] == 32
    assert len(second["training_history"]) == 2


def test_train_policy_skips_behavior_cloning_below_eight_transitions():
    settings = _fast_settings()

    state = train_policy({}, _separable_transitions(count=4), settings, rng=np.random.default_rng(0))

    assert state["training_history"][-1]["bc_epochs_run"] == 0
    assert state["training_history"][-1]["bc_train_loss"] == []


def test_training_report_exposes_the_documented_fields():
    report = TrainingReport(
        bc_epochs_run=1,
        bc_train_loss=[1.0],
        bc_val_loss=[],
        dqn_episodes_run=0,
        dqn_loss=[],
        transitions=1,
        status="bc_only_insufficient_transitions",
    )

    assert report.status == "bc_only_insufficient_transitions"
    assert report.bc_epochs_run == 1
