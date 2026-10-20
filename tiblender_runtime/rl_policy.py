"""Backwards-compatible entry point for the RL investigation policy.

The implementation lives in :mod:`tiblender_runtime.rl`; this module keeps the
import path used by the workflow stable.
"""

from __future__ import annotations

from .rl.actions import ACTION_SPACE
from .rl.policy import InvestigationStrategy, load_policy_state, select_investigation_strategy

__all__ = ["ACTION_SPACE", "InvestigationStrategy", "load_policy_state", "select_investigation_strategy"]
