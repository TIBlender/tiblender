"""Numpy MultiAction DQN policy package (state 320, 19 actions, four heads)."""

from .actions import ACTION_SPACE, FOCUS_TYPES, PERSPECTIVES, STOP_INDEX, synergy_bonus
from .features import STATE_DIM, cluster_embedding, state_vector
from .network import MultiActionQNetwork, select_composite_action
from .policy import InvestigationStrategy, select_investigation_strategy, similarity_to_report
from .replay import ReplayBuffer, Transition, build_trajectory_records, transitions_from_trajectory
from .reward import information_gain, step_reward, terminal_bonus
from .trainer import TrainingReport, behavior_cloning, dqn_finetune, train_policy

__all__ = [
    "ACTION_SPACE",
    "FOCUS_TYPES",
    "PERSPECTIVES",
    "STATE_DIM",
    "STOP_INDEX",
    "InvestigationStrategy",
    "MultiActionQNetwork",
    "ReplayBuffer",
    "TrainingReport",
    "Transition",
    "behavior_cloning",
    "build_trajectory_records",
    "cluster_embedding",
    "dqn_finetune",
    "information_gain",
    "select_composite_action",
    "select_investigation_strategy",
    "similarity_to_report",
    "state_vector",
    "step_reward",
    "synergy_bonus",
    "terminal_bonus",
    "train_policy",
    "transitions_from_trajectory",
]
