"""Thin pipeline facade kept for stable CLI and test imports."""

from __future__ import annotations

from .models import CycleResult
from .settings import Settings
from .workflow import run_graph_cycle


def run_cycle(settings: Settings, cycle_index: int = 1) -> CycleResult:
    return run_graph_cycle(settings, cycle_index=cycle_index)
