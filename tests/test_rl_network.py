"""Numerical tests for the hand-written MultiAction Q network."""

import numpy as np
import pytest

from tiblender_runtime.rl.actions import FOCUS_TYPES, PERSPECTIVE_FOCUS_INDICES
from tiblender_runtime.rl.network import MultiActionQNetwork, select_composite_action


def _tiny_network() -> MultiActionQNetwork:
    return MultiActionQNetwork(state_dim=6, hidden_dim=5, dropout=0.0, seed=7)


def test_forward_returns_all_four_head_shapes():
    net = MultiActionQNetwork(seed=1)
    out = net.forward(np.zeros((3, 320)))

    assert out["q_action"].shape == (3, 19)
    assert out["q_focus"].shape == (3, 18)
    assert out["perspective"].shape == (3, 4)
    assert out["num_actions"].shape == (3, 4)
    assert out["stop"].shape == (3, 1)


def test_q_focus_is_a_view_of_the_first_eighteen_action_values():
    net = MultiActionQNetwork(seed=1)
    out = net.forward(np.zeros((2, 320)))

    assert np.array_equal(out["q_focus"], out["q_action"][:, :18])
    assert out["q_focus"].base is out["q_action"]


def test_backward_rejects_the_q_focus_alias():
    net = MultiActionQNetwork(state_dim=6, hidden_dim=5, dropout=0.0, seed=2)
    net.forward(np.ones((1, 6)))

    with pytest.raises(ValueError):
        net.backward({"q_focus": np.ones((1, 18))})


def test_forward_rejects_wrong_state_dimension():
    net = MultiActionQNetwork(seed=1)

    with pytest.raises(ValueError):
        net.forward(np.zeros((2, 19)))


def test_dropout_is_disabled_at_evaluation_time():
    net = MultiActionQNetwork(state_dim=6, hidden_dim=32, dropout=0.5, seed=3)
    x = np.linspace(-1.0, 1.0, 6).reshape(1, 6)

    first = net.forward(x)["q_focus"]
    second = net.forward(x)["q_focus"]
    trained = net.forward(x, train=True)["q_focus"]

    assert np.array_equal(first, second)
    assert not np.array_equal(first, trained)


def test_to_dict_from_dict_roundtrip_is_stable():
    net = MultiActionQNetwork(state_dim=6, hidden_dim=5, seed=11)
    payload = net.to_dict()
    restored = MultiActionQNetwork.from_dict(payload)

    assert restored.to_dict() == payload
    for name, value in net.parameters().items():
        assert np.allclose(value, restored.parameters()[name], atol=1e-6)
    x = np.linspace(-1.0, 1.0, 6).reshape(1, 6)
    assert np.allclose(net.forward(x)["q_focus"], restored.forward(x)["q_focus"], atol=1e-5)


def test_copy_is_independent():
    net = MultiActionQNetwork(state_dim=6, hidden_dim=5, seed=5)
    clone = net.copy()
    net.zero_grad()
    net.forward(np.ones((1, 6)))
    net.backward({"q_action": np.ones((1, 19))})
    net.step(0.1)

    assert not np.allclose(net.parameters()["Wq"], clone.parameters()["Wq"])


def test_finite_difference_gradient_check_on_q_action_head():
    net = _tiny_network()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 6))
    weights = rng.normal(size=(4, 19))

    def loss() -> float:
        return float(np.sum(net.forward(x)["q_action"] * weights))

    net.zero_grad()
    net.forward(x)
    net.backward({"q_action": weights})
    analytic = {name: value.copy() for name, value in net.gradients().items()}

    epsilon = 1e-6
    params = net.parameters()
    for name, matrix in params.items():
        flat = matrix.reshape(-1)
        grad_flat = analytic[name].reshape(-1)
        for index in range(flat.size):
            original = flat[index]
            flat[index] = original + epsilon
            plus = loss()
            flat[index] = original - epsilon
            minus = loss()
            flat[index] = original
            numeric = (plus - minus) / (2 * epsilon)
            assert abs(numeric - grad_flat[index]) <= 1e-4 * max(1.0, abs(numeric)), f"{name}[{index}]"


def test_backward_accumulates_across_calls():
    net = _tiny_network()
    x = np.ones((1, 6))
    weights = np.ones((1, 19))

    net.zero_grad()
    net.forward(x)
    net.backward({"q_action": weights})
    once = net.gradients()["Wq"].copy()
    net.forward(x)
    net.backward({"q_action": weights})
    twice = net.gradients()["Wq"]

    assert np.allclose(twice, 2.0 * once)
    net.zero_grad()
    assert not net.gradients()["Wq"].any()


def test_select_composite_action_respects_k_and_perspective_masks():
    out = {
        "q_focus": np.zeros((1, 18)),
        "perspective": np.array([[3.0, 2.0, 1.0, 0.0]]),
        "num_actions": np.array([[0.0, 5.0, 0.0, 0.0]]),
        "stop": np.array([[-4.0]]),
    }
    out["q_focus"][0, PERSPECTIVE_FOCUS_INDICES["Infrastructure"][2]] = 9.0
    out["q_focus"][0, PERSPECTIVE_FOCUS_INDICES["Technical"][1]] = 8.0

    pairs = select_composite_action(out)

    assert pairs == [
        ("Infrastructure", FOCUS_TYPES[PERSPECTIVE_FOCUS_INDICES["Infrastructure"][2]]),
        ("Technical", FOCUS_TYPES[PERSPECTIVE_FOCUS_INDICES["Technical"][1]]),
    ]


def test_select_composite_action_honours_k_override_and_allowed_mask():
    out = {
        "q_focus": np.zeros((1, 18)),
        "perspective": np.array([[3.0, 2.0, 1.0, 0.0]]),
        "num_actions": np.array([[9.0, 0.0, 0.0, 0.0]]),
        "stop": np.array([[-4.0]]),
    }
    allowed = np.zeros(18, dtype=bool)
    allowed[PERSPECTIVE_FOCUS_INDICES["Technical"][0]] = True

    pairs = select_composite_action(out, k_override=3, allowed=allowed)

    assert pairs == [("Technical", FOCUS_TYPES[PERSPECTIVE_FOCUS_INDICES["Technical"][0]])]


def test_select_composite_action_returns_empty_when_stop_head_fires_mid_episode():
    out = {
        "q_focus": np.zeros((1, 18)),
        "perspective": np.zeros((1, 4)),
        "num_actions": np.zeros((1, 4)),
        "stop": np.array([[2.0]]),
    }

    assert select_composite_action(out, step=1) == []


def test_select_composite_action_ignores_the_stop_head_at_step_zero():
    out = {
        "q_focus": np.zeros((1, 18)),
        "perspective": np.zeros((1, 4)),
        "num_actions": np.zeros((1, 4)),
        "stop": np.array([[9.0]]),
    }

    assert select_composite_action(out, step=0) != []
    assert select_composite_action(out) != []
