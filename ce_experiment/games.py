"""Game definitions and correlated-equilibrium verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

JointAction = tuple[str, str]
Distribution = dict[JointAction, float]


@dataclass(frozen=True)
class Game:
    name: str
    row_actions: tuple[str, ...]
    col_actions: tuple[str, ...]
    row_payoffs: np.ndarray
    col_payoffs: np.ndarray
    real_ce: Distribution
    fake_ce: Distribution
    description: str = ""

    def row_index(self, action: str) -> int:
        return self.row_actions.index(action)

    def col_index(self, action: str) -> int:
        return self.col_actions.index(action)


def _normalize_distribution(dist: Mapping[JointAction, float]) -> Distribution:
    total = float(sum(dist.values()))
    if total <= 0:
        raise ValueError("Distribution must have positive mass.")
    return {k: float(v) / total for k, v in dist.items() if v > 0}


def distribution_support_for_row(dist: Mapping[JointAction, float]) -> tuple[str, ...]:
    return tuple(sorted({row for (row, _), prob in dist.items() if prob > 0}))


def verify_ic(
    game: Game, distribution: Mapping[JointAction, float], tol: float = 1e-9
) -> tuple[bool, list[dict[str, object]]]:
    """Numerically verify CE incentive constraints for both players.

    The details list stores unconditional expected payoff terms for each
    recommendation/deviation inequality. CE requires obey >= deviate.
    """

    dist = _normalize_distribution(distribution)
    details: list[dict[str, object]] = []
    ok = True

    for rec in game.row_actions:
        rec_mass = sum(prob for (row, _), prob in dist.items() if row == rec)
        if rec_mass <= tol:
            continue
        rec_i = game.row_index(rec)
        for dev in game.row_actions:
            if dev == rec:
                continue
            dev_i = game.row_index(dev)
            obey = 0.0
            deviate = 0.0
            for (row, col), prob in dist.items():
                if row != rec:
                    continue
                col_i = game.col_index(col)
                obey += prob * game.row_payoffs[rec_i, col_i]
                deviate += prob * game.row_payoffs[dev_i, col_i]
            holds = bool(obey + tol >= deviate)
            ok = ok and holds
            details.append(
                {
                    "player": "row",
                    "recommendation": rec,
                    "deviation": dev,
                    "obey_payoff": float(obey / rec_mass),
                    "deviate_payoff": float(deviate / rec_mass),
                    "margin": float((obey - deviate) / rec_mass),
                    "holds": holds,
                }
            )

    for rec in game.col_actions:
        rec_mass = sum(prob for (_, col), prob in dist.items() if col == rec)
        if rec_mass <= tol:
            continue
        rec_j = game.col_index(rec)
        for dev in game.col_actions:
            if dev == rec:
                continue
            dev_j = game.col_index(dev)
            obey = 0.0
            deviate = 0.0
            for (row, col), prob in dist.items():
                if col != rec:
                    continue
                row_i = game.row_index(row)
                obey += prob * game.col_payoffs[row_i, rec_j]
                deviate += prob * game.col_payoffs[row_i, dev_j]
            holds = bool(obey + tol >= deviate)
            ok = ok and holds
            details.append(
                {
                    "player": "column",
                    "recommendation": rec,
                    "deviation": dev,
                    "obey_payoff": float(obey / rec_mass),
                    "deviate_payoff": float(deviate / rec_mass),
                    "margin": float((obey - deviate) / rec_mass),
                    "holds": holds,
                }
            )

    return ok, details


def expected_payoffs(game: Game, distribution: Mapping[JointAction, float]) -> tuple[float, float]:
    dist = _normalize_distribution(distribution)
    row_ev = 0.0
    col_ev = 0.0
    for (row, col), prob in dist.items():
        i = game.row_index(row)
        j = game.col_index(col)
        row_ev += prob * game.row_payoffs[i, j]
        col_ev += prob * game.col_payoffs[i, j]
    return row_ev, col_ev


def get_games() -> list[Game]:
    return [
        Game(
            name="Battle of the Sexes",
            row_actions=("Opera", "Football"),
            col_actions=("Opera", "Football"),
            row_payoffs=np.array([[3, 0], [0, 2]], dtype=float),
            col_payoffs=np.array([[2, 0], [0, 3]], dtype=float),
            real_ce={("Opera", "Opera"): 0.5, ("Football", "Football"): 0.5},
            fake_ce={("Opera", "Football"): 0.5, ("Football", "Opera"): 0.5},
            description="Coordination game with conflicting preferred equilibria.",
        ),
        Game(
            name="Chicken",
            row_actions=("Swerve", "Straight"),
            col_actions=("Swerve", "Straight"),
            row_payoffs=np.array([[3, 1], [4, 0]], dtype=float),
            col_payoffs=np.array([[3, 4], [1, 0]], dtype=float),
            real_ce={
                ("Swerve", "Swerve"): 0.2,
                ("Swerve", "Straight"): 0.3,
                ("Straight", "Swerve"): 0.3,
                ("Straight", "Straight"): 0.2,
            },
            fake_ce={("Straight", "Straight"): 0.5, ("Swerve", "Swerve"): 0.5},
            description="Hawk-Dove game where obedience sometimes means accepting restraint.",
        ),
        Game(
            name="Pure Coordination",
            row_actions=("A", "B"),
            col_actions=("A", "B"),
            row_payoffs=np.array([[2, 0], [0, 4]], dtype=float),
            col_payoffs=np.array([[2, 0], [0, 4]], dtype=float),
            real_ce={("A", "A"): 0.1, ("B", "B"): 0.9},
            fake_ce={("A", "B"): 0.5, ("B", "A"): 0.5},
            description="Coordination game with a Pareto-dominated equilibrium.",
        ),
        Game(
            name="Prisoner's Dilemma",
            row_actions=("Cooperate", "Defect"),
            col_actions=("Cooperate", "Defect"),
            row_payoffs=np.array([[3, 0], [5, 1]], dtype=float),
            col_payoffs=np.array([[3, 5], [0, 1]], dtype=float),
            real_ce={("Defect", "Defect"): 1.0},
            fake_ce={("Cooperate", "Cooperate"): 1.0},
            description="Dominant-strategy defection control.",
        ),
    ]


def validate_games() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for game in get_games():
        real_ok, real_details = verify_ic(game, game.real_ce)
        fake_ok, fake_details = verify_ic(game, game.fake_ce)
        rows.append(
            {
                "game": game.name,
                "real_ce_ok": real_ok,
                "fake_ce_ok": fake_ok,
                "real_details": real_details,
                "fake_details": fake_details,
            }
        )
        if not real_ok:
            raise ValueError(f"Real CE failed IC check for {game.name}: {real_details}")
        if fake_ok:
            raise ValueError(f"Fake CE unexpectedly passed IC check for {game.name}")
    return rows
