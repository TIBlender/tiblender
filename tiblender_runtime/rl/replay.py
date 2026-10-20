"""Replay buffer and trajectory-to-transition conversion."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..models import Cluster, PerspectiveFinding, to_jsonable
from .actions import ACTION_SPACE, PERSPECTIVES, STOP_INDEX
from .features import CLUSTER_DIM, CONTEXT_DIM, context_embedding, history_embedding
from .reward import step_reward, terminal_bonus


DEFAULT_CAPACITY = 10_000
SERIALISATION_DECIMALS = 6

_IOC_VALUE_FIELDS = ("urls", "domains", "ips", "hashes")


@dataclass(frozen=True, slots=True)
class Transition:
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    done: bool
    # Distinct perspectives the source trajectory investigated, clamped to
    # 1..4. This is the behaviour-cloning target for the ``num_actions`` head;
    # 0 means "unknown", so such transitions contribute no gradient there.
    num_perspectives: int = 0


class ReplayBuffer:
    """Fixed-capacity FIFO buffer of transitions."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self._items: deque[Transition] = deque(maxlen=self.capacity)

    def add(self, transition: Transition) -> None:
        self._items.append(transition)

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, n: int, rng: np.random.Generator) -> list[Transition]:
        if n <= 0:
            return []
        if not self._items:
            raise ValueError("cannot sample from an empty replay buffer")
        replace = n > len(self._items)
        indices = rng.choice(len(self._items), size=n, replace=replace)
        items = list(self._items)
        return [items[int(index)] for index in indices]

    @property
    def transitions(self) -> list[Transition]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": len(self._items),
            "transitions": [
                {
                    "state": [round(value, SERIALISATION_DECIMALS) for value in transition.state],
                    "action": int(transition.action),
                    "reward": round(float(transition.reward), SERIALISATION_DECIMALS),
                    "next_state": [round(value, SERIALISATION_DECIMALS) for value in transition.next_state],
                    "done": bool(transition.done),
                    "num_perspectives": int(transition.num_perspectives),
                }
                for transition in self._items
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ReplayBuffer":
        if not isinstance(payload, dict):
            raise ValueError("replay payload must be a dict")
        buffer = cls(capacity=int(payload.get("capacity") or DEFAULT_CAPACITY))
        for record in payload.get("transitions") or []:
            buffer.add(
                Transition(
                    state=tuple(float(value) for value in record["state"]),
                    action=int(record["action"]),
                    reward=float(record["reward"]),
                    next_state=tuple(float(value) for value in record["next_state"]),
                    done=bool(record["done"]),
                    num_perspectives=int(record.get("num_perspectives", 0)),
                )
            )
        return buffer


def build_trajectory_records(
    cluster: Cluster,
    findings: list[PerspectiveFinding],
    *,
    context_trajectory: list[dict] | None = None,
) -> list[dict]:
    """Per-step investigation records persisted by the workflow and reused here.

    Step 3 aggregates its own per-step trajectory while building the
    investigation context; when it is supplied it is used verbatim (only the
    cluster identity is added). Callers without a context keep the legacy
    derivation from findings.
    """
    if context_trajectory is not None:
        return [
            {"cluster_id": cluster.id, "cluster_name": cluster.topic_name, **record}
            for record in context_trajectory
        ]
    records: list[dict] = []
    step = 1
    for finding in findings:
        for query, result in zip(finding.queries, finding.tool_results, strict=False):
            records.append(
                {
                    "cluster_id": cluster.id,
                    "cluster_name": cluster.topic_name,
                    "step": step,
                    "mode": finding.mode,
                    "perspective": finding.perspective,
                    "focus_types": finding.focus_types,
                    "tool_type": query.tool_type,
                    "query_parameters": query.parameters,
                    "result_status": result.status,
                    "result_summary": result.summary,
                    "discovered_iocs": to_jsonable(result.discovered_iocs),
                    "confidence": result.confidence.value,
                }
            )
            step += 1
    return records


def _signature(tool_type: str, parameters: Any) -> str:
    try:
        rendered = json.dumps(parameters, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = repr(parameters)
    return f"{tool_type}|{rendered}"


def _focus_of(record: dict) -> str:
    focus = record.get("focus_type")
    if focus:
        return str(focus)
    focus_types = record.get("focus_types") or []
    return str(focus_types[0]) if focus_types else ""


def normalize_trajectory(trajectory: list[dict]) -> list[dict]:
    """Accept both the current workflow records and the Step 3 rework schema.

    New-style records carry ``focus_type``/``status``/``new_iocs``/``new_cves``/
    ``new_ttps``/``duplicate`` directly. Legacy records carry ``focus_types``,
    ``result_status``, and a ``discovered_iocs`` IoC set; their deltas are derived
    by de-duplicating discovered values against earlier steps of the trajectory.
    """
    seen_iocs: set[str] = set()
    seen_cves: set[str] = set()
    seen_signatures: set[str] = set()
    records: list[dict] = []

    for position, raw in enumerate(trajectory):
        focus = _focus_of(raw)
        if focus not in ACTION_SPACE or focus == "STOP":
            continue
        tool_type = str(raw.get("tool_type", ""))
        parameters = raw.get("parameters", raw.get("query_parameters", {}))
        signature = _signature(tool_type, parameters)

        if "duplicate" in raw:
            duplicate = bool(raw["duplicate"])
        else:
            duplicate = signature in seen_signatures
        seen_signatures.add(signature)

        discovered = raw.get("discovered_iocs") or {}
        new_ioc_values = {
            str(value)
            for field in _IOC_VALUE_FIELDS
            for value in (discovered.get(field) or [])
        } - seen_iocs
        new_cve_values = {str(value) for value in (discovered.get("cves") or [])} - seen_cves
        seen_iocs |= new_ioc_values
        seen_cves |= new_cve_values

        records.append(
            {
                "step": int(raw.get("step", position + 1)),
                "perspective": str(raw.get("perspective", "")),
                "focus_type": focus,
                "tool_type": tool_type,
                "parameters": parameters,
                "status": str(raw.get("status", raw.get("result_status", ""))),
                "new_iocs": int(raw["new_iocs"]) if "new_iocs" in raw else len(new_ioc_values),
                "new_cves": int(raw["new_cves"]) if "new_cves" in raw else len(new_cve_values),
                "new_ttps": int(raw["new_ttps"]) if "new_ttps" in raw else 0,
                "duplicate": duplicate,
                "confidence": raw.get("confidence", ""),
            }
        )
    return records


def _accumulate(stats: dict, per_perspective: dict, record: dict) -> tuple[dict, dict]:
    """Return new stat dictionaries that include ``record`` (no in-place mutation)."""
    success = record["status"] == "success"
    confidence = dict(stats.get("confidence", {}))
    label = str(record.get("confidence") or "")
    if label:
        confidence[label] = confidence.get(label, 0) + 1
    merged_stats = {
        "iocs": stats.get("iocs", 0) + record["new_iocs"],
        "cves": stats.get("cves", 0) + record["new_cves"],
        "ttps": stats.get("ttps", 0) + record["new_ttps"],
        "results": stats.get("results", 0) + 1,
        "successes": stats.get("successes", 0) + (1 if success else 0),
        "confidence": confidence,
    }
    perspective = record["perspective"]
    bucket = per_perspective.get(perspective, {})
    merged_perspective = dict(per_perspective)
    merged_perspective[perspective] = {
        "results": bucket.get("results", 0) + 1,
        "successes": bucket.get("successes", 0) + (1 if success else 0),
    }
    return merged_stats, merged_perspective


def transitions_from_trajectory(
    trajectory: list[dict],
    *,
    cluster_state: np.ndarray,
    final_quality: float,
    perspectives_used: set[str],
    ioc_total: int,
) -> list[Transition]:
    """Convert one cluster's investigation trajectory into replay transitions.

    Produces ``len(steps) + 1`` transitions: one per executed tool action plus a
    terminal ``STOP`` transition carrying the episode's terminal bonus.
    """
    cluster_part = np.asarray(cluster_state, dtype=np.float64).reshape(-1)
    if cluster_part.shape[0] != CLUSTER_DIM:
        raise ValueError(f"cluster_state must have {CLUSTER_DIM} entries, got {cluster_part.shape[0]}")

    records = normalize_trajectory(trajectory)
    if not records:
        return []

    def build_state(stats: dict, per_perspective: dict, actions: list[int], step: int) -> np.ndarray:
        return np.concatenate(
            (cluster_part, context_embedding(stats, per_perspective, CONTEXT_DIM), history_embedding(actions, step))
        )

    stats: dict = {}
    per_perspective: dict = {}
    actions: list[int] = []
    states = [build_state(stats, per_perspective, actions, 0)]
    rewards: list[float] = []
    action_indices: list[int] = []

    for position, record in enumerate(records):
        action_indices.append(ACTION_SPACE.index(record["focus_type"]))
        rewards.append(
            step_reward(
                delta_ioc=record["new_iocs"],
                delta_cve=record["new_cves"],
                delta_ttp=record["new_ttps"],
                duplicate_actions=1 if record["duplicate"] else 0,
            )
        )
        stats, per_perspective = _accumulate(stats, per_perspective, record)
        actions = [*actions, action_indices[-1]]
        states.append(build_state(stats, per_perspective, actions, position + 1))

    action_indices.append(STOP_INDEX)
    rewards.append(
        terminal_bonus(
            quality_score=final_quality,
            perspectives_used=set(perspectives_used),
            actions_taken=len(records),
            ioc_total=int(ioc_total),
        )
    )
    states.append(states[-1])

    # How many perspectives this episode actually investigated: the supervision
    # target for the num_actions head, shared by every transition of the episode.
    investigated = {record["perspective"] for record in records if record["perspective"] in PERSPECTIVES}
    num_perspectives = min(len(PERSPECTIVES), max(1, len(investigated)))

    return [
        Transition(
            state=tuple(states[index].tolist()),
            action=action_indices[index],
            reward=float(rewards[index]),
            next_state=tuple(states[index + 1].tolist()),
            done=index == len(records),
            num_perspectives=num_perspectives,
        )
        for index in range(len(records) + 1)
    ]
