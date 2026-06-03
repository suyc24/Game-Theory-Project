"""Response parsing utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionParse:
    parsed_action: str | None
    parse_success: bool


REASONING_FEATURES = {
    "payoff_calc": [
        "expected payoff",
        "expected value",
        "payoff of obeying",
        "payoff from obeying",
        "payoff of deviating",
        "payoff from deviating",
        "conditional probability",
        "conditional on",
    ],
    "deviation": [
        "deviat",
        "alternative action",
        "switch to",
        "instead play",
        "profitable deviation",
    ],
    "profitable_deviation": [
        "profitable deviation",
        "deviating is profitable",
        "deviation is profitable",
        "would get a higher payoff",
        "higher payoff by",
        "strictly better",
    ],
    "ce_label": [
        "correlated equilibrium",
        " ce ",
        "incentive compatible",
        "incentive-compatible",
    ],
    "mediator_trust": [
        "trusted mediator",
        "trust the mediator",
        "follow the mediator",
        "mediator's recommendation",
        "mediator recommendation",
    ],
    "label_skepticism": [
        "label may be wrong",
        "not assume",
        "cannot assume",
        "verify",
        "check whether",
        "audit",
        "mislabel",
        "falsely labeled",
    ],
}


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


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[*_`#>\[\]{}()]", " ", lowered)
    lowered = lowered.replace("–", "-").replace("—", "-")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def extract_reasoning_features(text: str) -> dict[str, bool]:
    """Return coarse textual features used to audit model reasoning."""

    normalized = f" {_normalize_text(text)} "
    return {
        feature: any(marker in normalized for marker in markers)
        for feature, markers in REASONING_FEATURES.items()
    }


def _last_identification_marker(text: str) -> bool | None:
    non_ce_patterns = [
        r"\b(?:is|it is|it's|this is|this distribution is)\s+not\s+(?:a\s+)?(?:correlated equilibrium|ce)\b",
        r"\bnot\s+(?:a\s+)?(?:correlated equilibrium|ce)\b",
        r"\bnot\s+incentive[- ]compatible\b",
        r"\bincentive[- ]compatible[^.!?]{0,50}\bno\b",
        r"\binstitutional claim audited[^.!?]{0,20}\bfail\b",
        r"\bfails?\s+(?:the\s+)?incentive",
        r"\bviolates?\s+(?:the\s+)?incentive",
        r"\bprofitable deviation\b",
        r"\bdeviat(?:e|ing|ion)[^.!?]{0,80}\bprofitable\b",
    ]
    ce_patterns = [
        r"\b(?:is|it is|it's|this is|this distribution is)\s+(?:indeed\s+)?(?:a\s+)?correlated equilibrium\b",
        r"\b(?:is|it is|this distribution is)\s+incentive[- ]compatible\b",
        r"\bincentive[- ]compatible[^.!?]{0,50}\byes\b",
        r"\binstitutional claim audited[^.!?]{0,20}\bpass\b",
        r"\bsatisfies?\s+(?:all\s+)?(?:the\s+)?incentive",
        r"\bno profitable deviation",
        r"\bthere (?:is|are)\s+no\s+profitable deviation",
    ]

    markers: list[tuple[int, bool, tuple[int, int]]] = []
    non_ce_spans: list[tuple[int, int]] = []
    for pattern in non_ce_patterns:
        for match in re.finditer(pattern, text):
            non_ce_spans.append(match.span())
            markers.append((match.start(), False, match.span()))

    for pattern in ce_patterns:
        for match in re.finditer(pattern, text):
            span = match.span()
            overlaps_non_ce = any(
                max(span[0], non_span[0]) < min(span[1], non_span[1])
                for non_span in non_ce_spans
            )
            prefix = text[max(0, span[0] - 12) : span[0]]
            if overlaps_non_ce or re.search(r"\bnot\s+(?:a\s+)?$", prefix):
                continue
            markers.append((match.start(), True, span))

    if not markers:
        return None
    markers.sort(key=lambda item: item[0])
    return markers[-1][1]


def parse_analytical_correctness(text: str, expected_is_ce: bool) -> tuple[bool | None, str]:
    """Coarse extraction of whether the model identifies CE/non-CE correctly."""

    normalized = _normalize_text(text)
    conclusion_markers = [
        "therefore",
        "thus",
        "hence",
        "conclusion",
        "final answer",
        "answer:",
    ]
    tail = normalized[-1400:]
    conclusion_start = max([tail.rfind(marker) for marker in conclusion_markers] + [-1])
    conclusion_text = tail[conclusion_start:] if conclusion_start >= 0 else tail

    identified = _last_identification_marker(conclusion_text)
    if identified is None:
        identified = _last_identification_marker(tail)
    if identified is None:
        identified = _last_identification_marker(normalized)

    if identified is None:
        return None, "unclear"
    return identified == expected_is_ce, "ce" if identified else "non_ce"
