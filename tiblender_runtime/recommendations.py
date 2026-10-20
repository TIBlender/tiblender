"""Recommendation horizons for the released report format.

Every recommendation in a generated report is labelled with the horizon on
which it is meant to be acted, matching the released dataset's
``report_text.recommendations[].horizon``. The Step 5 prompt asks the model
for the label; this module supplies the deterministic derivation used by the
offline stand-in and whenever a model omits the field or returns a value
outside the vocabulary.
"""

from __future__ import annotations

import re
from typing import Any

from .contract import RECOMMENDATION_HORIZONS

# The verb vocabulary is taken from the wording the deployment's own
# recommendations were written with: containment and triage verbs are
# immediate, retrospective hunting and review verbs are short-term, and verbs
# that change a control or a process are long-term.
_IMMEDIATE_VERBS = frozenset(
    {
        "alert", "apply", "block", "deploy", "disable", "escalate", "examine",
        "inspect", "isolate", "patch", "quarantine", "query", "restrict",
        "revoke", "scan", "search", "verify",
    }
)
_SHORT_TERM_VERBS = frozenset(
    {
        "audit", "confirm", "continue", "correlate", "cross-check",
        "cross-reference", "hunt", "investigate", "map", "monitor", "preserve",
        "review", "run", "trace", "track", "validate", "watch",
    }
)
_LONG_TERM_VERBS = frozenset(
    {
        "build", "develop", "document", "draft", "educate", "embed", "enhance",
        "engineer", "establish", "formalize", "harden", "implement",
        "incorporate", "integrate", "rotate", "train", "update",
    }
)


def recommendation_horizon(action: str) -> str:
    """The action horizon for one recommendation: ``immediate``,
    ``short_term`` or ``long_term``.

    Derived from the leading verb so that the same action text always yields
    the same horizon; unrecognised wording defaults to ``immediate``, which is
    the dominant label in the released dataset.
    """
    leading = re.sub(r"^[^A-Za-z]+", "", str(action)).split(" ", 1)[0]
    verb = leading.strip().strip(".,:;").lower()
    if verb in _LONG_TERM_VERBS:
        return "long_term"
    if verb in _SHORT_TERM_VERBS:
        return "short_term"
    if verb in _IMMEDIATE_VERBS:
        return "immediate"
    return "immediate"


def recommendation_row(action: str, rationale: str, horizon: Any = None) -> dict[str, str]:
    requested = str(horizon or "").strip().lower()
    return {
        "action": action,
        "rationale": rationale,
        "horizon": requested if requested in RECOMMENDATION_HORIZONS else recommendation_horizon(action),
    }


def markdown_recommendation_lines(recommendations: list[dict[str, str]]) -> list[str]:
    """Markdown for the Recommendations section, grouped by horizon.

    Empty horizons are omitted, so a report whose recommendations are all
    immediate renders one group rather than three headings with two blanks.
    """
    lines: list[str] = []
    for horizon, heading in (
        ("immediate", "Immediate"),
        ("short_term", "Short-Term"),
        ("long_term", "Long-Term"),
    ):
        group = [item for item in recommendations if item["horizon"] == horizon]
        if not group:
            continue
        lines.append(f"### {heading}")
        lines.append("")
        lines.extend(
            f"{index}. **{item['action']}** {item['rationale']}"
            for index, item in enumerate(group, start=1)
        )
        lines.append("")
    return lines
