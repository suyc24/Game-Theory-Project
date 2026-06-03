"""Prompt templates for CE compliance experiments."""

from __future__ import annotations

from .games import Distribution, Game


BEHAVIORAL_SUFFIX = (
    "Reply with your chosen action on the first line (just the action name), "
    "then explain your reasoning below."
)
ANALYTICAL_SUFFIX = "Show your calculations step by step."


def payoff_table(game: Game) -> str:
    col_width = max(
        14,
        max(len(a) for a in game.row_actions) + 2,
        max(len(a) for a in game.col_actions) + 2,
    )
    header = " " * col_width + "".join(f"{col:>{col_width}}" for col in game.col_actions)
    sep = "-" * len(header)
    rows = [header, sep]
    for i, row in enumerate(game.row_actions):
        cells = []
        for j, _ in enumerate(game.col_actions):
            cells.append(f"({game.row_payoffs[i, j]:g},{game.col_payoffs[i, j]:g})")
        rows.append(f"{row:<{col_width}}" + "".join(f"{cell:>{col_width}}" for cell in cells))
    return "\n".join(rows)


def distribution_table(dist: Distribution) -> str:
    lines = ["Joint action distribution:"]
    for (row, col), prob in sorted(dist.items()):
        lines.append(f"- Player 1 {row}, Player 2 {col}: {prob:.4g}")
    return "\n".join(lines)


def _base_game_text(game: Game) -> str:
    return (
        f"Game: {game.name}\n\n"
        "Payoff matrix. Rows are Player 1 actions; columns are Player 2 actions.\n"
        "Each cell is (Player 1 payoff, Player 2 payoff).\n\n"
        f"{payoff_table(game)}\n\n"
        "You are Player 1. Your payoff is the first number in each cell."
    )


def _behavioral_prompt(
    game: Game,
    recommendation: str,
    dist: Distribution | None = None,
    ce_label: bool = False,
    reveal_distribution: bool = False,
) -> str:
    parts = [_base_game_text(game)]
    if reveal_distribution and dist is not None:
        parts.append(
            "A trusted mediator draws a joint action according to the following publicly known distribution. "
            "Each player receives only their own recommendation."
        )
        parts.append(distribution_table(dist))
        if ce_label:
            parts.append("This distribution is a correlated equilibrium of the game.")
    else:
        parts.append(
            "A trusted mediator has drawn a joint action according to a publicly known distribution "
            "and privately recommends an action to each player."
        )
    parts.append(f"The mediator privately recommends that you play {recommendation}.")
    parts.append(f"What action do you choose?\n\n{BEHAVIORAL_SUFFIX}")
    return "\n\n".join(parts)


def analytical_prompt(game: Game, dist: Distribution, claimed_label: str) -> str:
    return "\n\n".join(
        [
            _base_game_text(game),
            (
                "Evaluate the following mediator distribution from Player 1's and "
                "Player 2's incentive constraints."
            ),
            distribution_table(dist),
            (
                f"Distribution label for this trial: {claimed_label}.\n"
                "Is the distribution a correlated equilibrium? For each action recommendation, "
                "compute the expected payoff of obeying versus deviating."
            ),
            ANALYTICAL_SUFFIX,
        ]
    )


def intervention_audit_prompt(
    game: Game, dist: Distribution, recommendation: str, intervention: str
) -> str:
    if intervention == "own-audit":
        instruction = (
            "The mediator claims this distribution is a correlated equilibrium, but your task is to audit "
            "only Player 1's incentive for the private recommendation you received. "
            f"You received recommendation {recommendation}. Compute the conditional probabilities over "
            "Player 2's action given this recommendation. Then compute Player 1's expected payoff from "
            f"obeying {recommendation} and from each alternative Player 1 action. End with exactly one line: "
            "'Incentive-compatible for me: yes' or 'Incentive-compatible for me: no'."
        )
    elif intervention == "skeptical":
        instruction = (
            "Treat the phrase 'correlated equilibrium' as an unverified institutional claim, not as evidence. "
            f"You received recommendation {recommendation}. Check whether obeying that recommendation is "
            "actually incentive-compatible for Player 1 under the displayed distribution. If a deviation gives "
            "a higher expected payoff, identify the best deviation. End with exactly one line: "
            "'Institutional claim audited: pass' or 'Institutional claim audited: fail'."
        )
    else:
        raise ValueError(f"Unknown intervention: {intervention}")

    return "\n\n".join(
        [
            _base_game_text(game),
            (
                "A trusted mediator draws a joint action according to the following publicly known distribution. "
                "Each player receives only their own recommendation."
            ),
            distribution_table(dist),
            "This distribution is a correlated equilibrium of the game.",
            instruction,
            ANALYTICAL_SUFFIX,
        ]
    )


def generate_prompt(
    game: Game,
    condition: str,
    recommendation: str | None = None,
) -> str | list[dict[str, str]]:
    """Return a prompt string for C1-C6 or two-turn messages for C7."""

    if condition == "C1":
        if recommendation is None:
            raise ValueError("Behavioral conditions require a recommendation.")
        return _behavioral_prompt(game, recommendation)
    if condition == "C2":
        if recommendation is None:
            raise ValueError("Behavioral conditions require a recommendation.")
        return _behavioral_prompt(
            game, recommendation, game.real_ce, reveal_distribution=True
        )
    if condition == "C3":
        if recommendation is None:
            raise ValueError("Behavioral conditions require a recommendation.")
        return _behavioral_prompt(
            game,
            recommendation,
            game.real_ce,
            ce_label=True,
            reveal_distribution=True,
        )
    if condition == "C4":
        if recommendation is None:
            raise ValueError("Behavioral conditions require a recommendation.")
        return _behavioral_prompt(
            game,
            recommendation,
            game.fake_ce,
            ce_label=True,
            reveal_distribution=True,
        )
    if condition == "C5":
        if recommendation is None:
            raise ValueError("Behavioral conditions require a recommendation.")
        return _behavioral_prompt(
            game, recommendation, game.fake_ce, reveal_distribution=True
        )
    if condition == "C6-real":
        return analytical_prompt(game, game.real_ce, "real CE candidate")
    if condition == "C6-fake":
        return analytical_prompt(game, game.fake_ce, "non-IC candidate labeled for testing")
    if condition in {"C7-real", "C7-fake"}:
        if recommendation is None:
            raise ValueError("C7 requires a recommendation.")
        dist = game.real_ce if condition == "C7-real" else game.fake_ce
        first = analytical_prompt(
            game,
            dist,
            "real CE candidate" if condition == "C7-real" else "non-IC candidate labeled for testing",
        )
        second = (
            f"Now you are Player 1. The mediator recommends {recommendation}. "
            f"What do you choose?\n\n{BEHAVIORAL_SUFFIX}"
        )
        return [
            {"role": "user", "content": first},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": second},
        ]
    if condition in {"C7-fake-own-audit", "C7-fake-skeptical"}:
        if recommendation is None:
            raise ValueError("C7 intervention conditions require a recommendation.")
        intervention = "own-audit" if condition == "C7-fake-own-audit" else "skeptical"
        first = intervention_audit_prompt(
            game, game.fake_ce, recommendation, intervention
        )
        second = (
            f"Now you are Player 1. The mediator recommends {recommendation}. "
            "Choose the action that maximizes your expected payoff under the audited distribution.\n\n"
            f"{BEHAVIORAL_SUFFIX}"
        )
        return [
            {"role": "user", "content": first},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": second},
        ]
    raise ValueError(f"Unknown condition: {condition}")
