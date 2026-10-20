"""Hand-written numpy MultiAction Q network (no autograd, no torch).

Encoder: Linear(320,256) -> LayerNorm -> ReLU -> Dropout -> Linear(256,256) ->
LayerNorm -> ReLU -> Dropout. Four heads read the shared encoding:

* ``q_action`` -- Q values over the full 19-action space (18 focus types plus
  STOP). ``forward`` also exposes ``q_focus``, a view of its first 18 columns.
* ``perspective`` -- 4 logits, which perspectives to investigate.
* ``num_actions`` -- 4 logits over K in 1..4, how many perspectives to pick.
* ``stop`` -- 1 logit, the behaviour-cloned "stop now" decision. This is a
  binary decision head and is deliberately *not* the STOP Q value, which lives
  in ``q_action[:, 18]``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .actions import FOCUS_TYPES, PERSPECTIVE_FOCUS_INDICES, PERSPECTIVES


LAYER_NORM_EPS = 1e-5
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
SERIALISATION_DECIMALS = 6


def _xavier_uniform(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _layer_norm_forward(x: np.ndarray, gain: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mean = x.mean(axis=1, keepdims=True)
    variance = x.var(axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(variance + LAYER_NORM_EPS)
    normalised = (x - mean) * inv_std
    return normalised * gain + bias, {"normalised": normalised, "inv_std": inv_std, "gain": gain}


def _layer_norm_backward(grad: np.ndarray, cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalised = cache["normalised"]
    inv_std = cache["inv_std"]
    grad_gain = (grad * normalised).sum(axis=0)
    grad_bias = grad.sum(axis=0)
    grad_normalised = grad * cache["gain"]
    width = normalised.shape[1]
    grad_x = inv_std * (
        grad_normalised
        - grad_normalised.mean(axis=1, keepdims=True)
        - normalised * (grad_normalised * normalised).sum(axis=1, keepdims=True) / width
    )
    return grad_x, grad_gain, grad_bias


class MultiActionQNetwork:
    """Two-layer encoder with four output heads, trained by explicit backprop."""

    _HEAD_PARAMS = {
        "q_action": ("Wq", "bq"),
        "perspective": ("Wp", "bp"),
        "num_actions": ("Wk", "bk"),
        "stop": ("Ws", "bs"),
    }

    def __init__(
        self,
        state_dim: int = 320,
        hidden_dim: int = 256,
        n_focus: int = 18,
        n_perspectives: int = 4,
        max_k: int = 4,
        dropout: float = 0.1,
        seed: int = 0,
    ) -> None:
        if state_dim <= 0 or hidden_dim <= 0:
            raise ValueError("state_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_focus = int(n_focus)
        self.n_actions = self.n_focus + 1  # focus types plus the terminal STOP action
        self.n_perspectives = int(n_perspectives)
        self.max_k = int(max_k)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        init = np.random.default_rng(self.seed)
        self._params: dict[str, np.ndarray] = {
            "W1": _xavier_uniform(init, self.state_dim, self.hidden_dim),
            "b1": np.zeros(self.hidden_dim),
            "g1": np.ones(self.hidden_dim),
            "be1": np.zeros(self.hidden_dim),
            "W2": _xavier_uniform(init, self.hidden_dim, self.hidden_dim),
            "b2": np.zeros(self.hidden_dim),
            "g2": np.ones(self.hidden_dim),
            "be2": np.zeros(self.hidden_dim),
            "Wq": _xavier_uniform(init, self.hidden_dim, self.n_actions),
            "bq": np.zeros(self.n_actions),
            "Wp": _xavier_uniform(init, self.hidden_dim, self.n_perspectives),
            "bp": np.zeros(self.n_perspectives),
            "Wk": _xavier_uniform(init, self.hidden_dim, self.max_k),
            "bk": np.zeros(self.max_k),
            "Ws": _xavier_uniform(init, self.hidden_dim, 1),
            "bs": np.zeros(1),
        }
        self._grads = {name: np.zeros_like(value) for name, value in self._params.items()}
        self._adam_m = {name: np.zeros_like(value) for name, value in self._params.items()}
        self._adam_v = {name: np.zeros_like(value) for name, value in self._params.items()}
        self._adam_t = 0
        self._cache: dict[str, Any] = {}

    # -- inference -----------------------------------------------------------

    def forward(self, x: np.ndarray, *, train: bool = False) -> dict[str, np.ndarray]:
        inputs = np.asarray(x, dtype=np.float64)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        if inputs.ndim != 2 or inputs.shape[1] != self.state_dim:
            raise ValueError(f"expected input of shape (batch, {self.state_dim}), got {np.shape(x)}")

        params = self._params
        z1 = inputs @ params["W1"] + params["b1"]
        n1, cache1 = _layer_norm_forward(z1, params["g1"], params["be1"])
        a1 = np.maximum(n1, 0.0)
        d1, mask1 = self._dropout(a1, train)
        z2 = d1 @ params["W2"] + params["b2"]
        n2, cache2 = _layer_norm_forward(z2, params["g2"], params["be2"])
        a2 = np.maximum(n2, 0.0)
        d2, mask2 = self._dropout(a2, train)

        self._cache = {
            "x": inputs,
            "cache1": cache1,
            "n1": n1,
            "mask1": mask1,
            "d1": d1,
            "cache2": cache2,
            "n2": n2,
            "mask2": mask2,
            "d2": d2,
        }
        q_action = d2 @ params["Wq"] + params["bq"]
        return {
            "q_action": q_action,
            # View over the focus columns; writing through it writes q_action.
            "q_focus": q_action[:, : self.n_focus],
            "perspective": d2 @ params["Wp"] + params["bp"],
            "num_actions": d2 @ params["Wk"] + params["bk"],
            "stop": d2 @ params["Ws"] + params["bs"],
        }

    def _dropout(self, values: np.ndarray, train: bool) -> tuple[np.ndarray, np.ndarray | None]:
        if not train or self.dropout <= 0.0:
            return values, None
        keep = 1.0 - self.dropout
        mask = (self.rng.random(values.shape) < keep).astype(np.float64) / keep
        return values * mask, mask

    # -- training ------------------------------------------------------------

    def backward(self, grads: dict[str, np.ndarray]) -> None:
        """Accumulate parameter gradients from dL/d(head output) arrays."""
        if not self._cache:
            raise RuntimeError("backward() called before forward()")
        unknown = set(grads) - set(self._HEAD_PARAMS)
        if unknown:
            raise ValueError(f"unknown head gradients: {sorted(unknown)}")

        cache = self._cache
        d2 = cache["d2"]
        grad_d2 = np.zeros_like(d2)
        for head, (weight_name, bias_name) in self._HEAD_PARAMS.items():
            head_grad = grads.get(head)
            if head_grad is None:
                continue
            head_grad = np.asarray(head_grad, dtype=np.float64)
            self._grads[weight_name] += d2.T @ head_grad
            self._grads[bias_name] += head_grad.sum(axis=0)
            grad_d2 += head_grad @ self._params[weight_name].T

        grad_a2 = grad_d2 if cache["mask2"] is None else grad_d2 * cache["mask2"]
        grad_n2 = grad_a2 * (cache["n2"] > 0.0)
        grad_z2, grad_g2, grad_be2 = _layer_norm_backward(grad_n2, cache["cache2"])
        self._grads["g2"] += grad_g2
        self._grads["be2"] += grad_be2
        self._grads["W2"] += cache["d1"].T @ grad_z2
        self._grads["b2"] += grad_z2.sum(axis=0)

        grad_d1 = grad_z2 @ self._params["W2"].T
        grad_a1 = grad_d1 if cache["mask1"] is None else grad_d1 * cache["mask1"]
        grad_n1 = grad_a1 * (cache["n1"] > 0.0)
        grad_z1, grad_g1, grad_be1 = _layer_norm_backward(grad_n1, cache["cache1"])
        self._grads["g1"] += grad_g1
        self._grads["be1"] += grad_be1
        self._grads["W1"] += cache["x"].T @ grad_z1
        self._grads["b1"] += grad_z1.sum(axis=0)

    def step(self, lr: float) -> None:
        """Apply one Adam update using the accumulated gradients."""
        self._adam_t += 1
        bias1 = 1.0 - ADAM_BETA1**self._adam_t
        bias2 = 1.0 - ADAM_BETA2**self._adam_t
        for name, param in self._params.items():
            grad = self._grads[name]
            self._adam_m[name] = ADAM_BETA1 * self._adam_m[name] + (1.0 - ADAM_BETA1) * grad
            self._adam_v[name] = ADAM_BETA2 * self._adam_v[name] + (1.0 - ADAM_BETA2) * grad * grad
            m_hat = self._adam_m[name] / bias1
            v_hat = self._adam_v[name] / bias2
            param -= lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

    def zero_grad(self) -> None:
        for grad in self._grads.values():
            grad.fill(0.0)

    def parameters(self) -> dict[str, np.ndarray]:
        """Live parameter arrays; mutating them mutates the network."""
        return self._params

    def gradients(self) -> dict[str, np.ndarray]:
        """Live gradient accumulators."""
        return self._grads

    # -- persistence ---------------------------------------------------------

    def config(self) -> dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "n_focus": self.n_focus,
            "n_perspectives": self.n_perspectives,
            "max_k": self.max_k,
            "dropout": self.dropout,
            "seed": self.seed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": {
                "encoder": f"{self.state_dim} -> {self.hidden_dim} -> {self.hidden_dim}",
                "normalization": "layer_norm",
                "activation": "relu",
                "initializer": "xavier_uniform",
                "heads": sorted(self._HEAD_PARAMS),
                "optimizer": "adam",
            },
            "config": self.config(),
            "parameters": {
                name: np.round(value, SERIALISATION_DECIMALS).tolist() for name, value in sorted(self._params.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MultiActionQNetwork":
        if not isinstance(payload, dict):
            raise ValueError("network payload must be a dict")
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError("network payload is missing its config")
        network = cls(
            state_dim=int(config.get("state_dim", 320)),
            hidden_dim=int(config.get("hidden_dim", 256)),
            n_focus=int(config.get("n_focus", 18)),
            n_perspectives=int(config.get("n_perspectives", 4)),
            max_k=int(config.get("max_k", 4)),
            dropout=float(config.get("dropout", 0.1)),
            seed=int(config.get("seed", 0)),
        )
        stored = payload.get("parameters") or {}
        for name, value in network._params.items():
            if name not in stored:
                raise ValueError(f"network payload is missing parameter {name!r}")
            loaded = np.asarray(stored[name], dtype=np.float64)
            if loaded.shape != value.shape:
                raise ValueError(f"parameter {name!r} has shape {loaded.shape}, expected {value.shape}")
            value[...] = loaded
        return network

    def copy(self) -> "MultiActionQNetwork":
        clone = MultiActionQNetwork(**self.config())
        for name, value in self._params.items():
            clone._params[name][...] = value
        clone._adam_t = self._adam_t
        for name in self._params:
            clone._adam_m[name][...] = self._adam_m[name]
            clone._adam_v[name][...] = self._adam_v[name]
        return clone

    def load_from(self, other: "MultiActionQNetwork") -> None:
        """Copy ``other``'s parameters in place (target-network sync)."""
        if other.config() != self.config():
            raise ValueError("cannot sync networks with different configurations")
        for name, value in other._params.items():
            self._params[name][...] = value


def select_composite_action(
    out: dict[str, np.ndarray],
    *,
    step: int = 0,
    k_override: int | None = None,
    allowed: np.ndarray | None = None,
) -> list[tuple[str, str]]:
    """Turn one forward pass into an ordered list of (perspective, focus) pairs.

    ``step`` is the index of the decision inside the episode. The stop head can
    only end an episode that is already under way, so at ``step == 0`` -- the
    mode-selection decision, where nothing has been investigated yet -- it is
    not consulted.
    """
    if step >= 1:
        stop_logit = float(np.asarray(out["stop"]).reshape(-1)[0])
        if _sigmoid(np.array([stop_logit]))[0] > 0.5:
            return []

    q_focus = np.asarray(out["q_focus"], dtype=np.float64).reshape(-1)
    perspective_logits = np.asarray(out["perspective"], dtype=np.float64).reshape(-1)
    num_action_logits = np.asarray(out["num_actions"], dtype=np.float64).reshape(-1)

    if k_override is None:
        chosen_k = int(np.argmax(num_action_logits)) + 1
    else:
        chosen_k = int(k_override)
    chosen_k = max(1, min(chosen_k, len(PERSPECTIVES)))

    mask = np.ones(len(FOCUS_TYPES), dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool).reshape(-1)
    if mask.shape[0] != len(FOCUS_TYPES):
        raise ValueError(f"allowed mask must have {len(FOCUS_TYPES)} entries")

    order = sorted(range(len(PERSPECTIVES)), key=lambda index: (-perspective_logits[index], index))
    pairs: list[tuple[str, str]] = []
    for perspective_index in order:
        if len(pairs) >= chosen_k:
            break
        perspective = PERSPECTIVES[perspective_index]
        candidates = [index for index in PERSPECTIVE_FOCUS_INDICES[perspective] if mask[index]]
        if not candidates:
            continue
        best = max(candidates, key=lambda index: (q_focus[index], -index))
        pairs.append((perspective, FOCUS_TYPES[best]))
    return pairs
