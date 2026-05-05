"""Response parsing utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionParse:
    parsed_action: str | None
    parse_success: bool


def _clean_first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line.strip("`*_#:- \t")
    return ""


def parse_action(text: str, valid_actions: tuple[str, ...]) -> ActionParse:
    first_line = _clean_first_line(text)
    first_lower = first_line.lower()
    for action in valid_actions:
        if first_lower == action.lower():
            return ActionParse(action, True)

    haystacks = [first_line, text[:500]]
    for haystack in haystacks:
        for action in valid_actions:
            pattern = r"(?<![A-Za-z])" + re.escape(action) + r"(?![A-Za-z])"
            if re.search(pattern, haystack, flags=re.IGNORECASE):
                return ActionParse(action, True)

    normalized_first = re.sub(r"[^A-Za-z0-9]+", "", first_line).lower()
    for action in valid_actions:
        normalized_action = re.sub(r"[^A-Za-z0-9]+", "", action).lower()
        if normalized_action and normalized_action in normalized_first:
            return ActionParse(action, True)

    return ActionParse(None, False)


def parse_analytical_correctness(text: str, expected_is_ce: bool) -> tuple[bool | None, str]:
    """Coarse extraction of whether the model identifies CE/non-CE correctly."""

    lowered = text.lower()
    non_ce_markers = [
        "not a correlated equilibrium",
        "not correlated equilibrium",
        "not incentive compatible",
        "not incentive-compatible",
        "fails incentive",
        "profitable deviation",
        "is not a ce",
        "isn't a correlated equilibrium",
    ]
    ce_markers = [
        "is a correlated equilibrium",
        "is correlated equilibrium",
        "is incentive compatible",
        "is incentive-compatible",
        "satisfies incentive",
        "no profitable deviation",
        "is a ce",
    ]

    says_non_ce = any(marker in lowered for marker in non_ce_markers)
    says_ce = any(marker in lowered for marker in ce_markers)

    if says_non_ce and not says_ce:
        identified = False
    elif says_ce and not says_non_ce:
        identified = True
    elif says_non_ce and says_ce:
        # Prefer the conclusion-like tail when both appear in worked calculations.
        tail = lowered[-900:]
        tail_non_ce = any(marker in tail for marker in non_ce_markers)
        tail_ce = any(marker in tail for marker in ce_markers)
        if tail_non_ce and not tail_ce:
            identified = False
        elif tail_ce and not tail_non_ce:
            identified = True
        else:
            identified = None
    else:
        identified = None

    if identified is None:
        return None, "unclear"
    return identified == expected_is_ce, "ce" if identified else "non_ce"

