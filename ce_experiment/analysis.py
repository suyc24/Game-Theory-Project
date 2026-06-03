"""Metrics and statistical summaries for experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from .games import Distribution, Game, get_games
from .parser import REASONING_FEATURES, extract_reasoning_features


KEY_COLUMNS = ["model", "game", "condition", "recommendation", "trial"]
REASONING_COLUMNS = [f"reasoning_{feature}" for feature in REASONING_FEATURES]


def load_records(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("raw_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


def filter_trials(df: pd.DataFrame, max_trials: int | None) -> pd.DataFrame:
    """Keep only trial numbers in [0, max_trials) when max_trials is set."""

    if max_trials is None or df.empty or "trial" not in df:
        return df
    filtered = df.copy()
    filtered["trial"] = pd.to_numeric(filtered["trial"], errors="coerce")
    return filtered[filtered["trial"].between(0, max_trials - 1)]


def deduplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one record per experimental key, preferring successful attempts."""

    if df.empty or any(column not in df for column in KEY_COLUMNS):
        return df

    deduped = df.copy()
    error_series = deduped.get("error_type", pd.Series(index=deduped.index, dtype=object))
    has_error = error_series.notna() & (error_series.astype(str).str.len() > 0)
    parse_success = deduped.get(
        "parse_success", pd.Series(False, index=deduped.index)
    ).fillna(False)
    analytical_known = deduped.get(
        "analytical_correct", pd.Series(index=deduped.index, dtype=object)
    ).notna()
    raw_present = (
        deduped.get("raw_response", pd.Series("", index=deduped.index))
        .fillna("")
        .astype(str)
        .str.len()
        > 0
    )
    analysis_present = (
        deduped.get("analysis_response", pd.Series("", index=deduped.index))
        .fillna("")
        .astype(str)
        .str.len()
        > 0
    )
    is_c7 = (
        deduped.get("condition", pd.Series("", index=deduped.index))
        .fillna("")
        .astype(str)
        .str.startswith("C7")
    )

    deduped["_quality"] = 0
    deduped.loc[raw_present & ~has_error, "_quality"] = 1
    complete_context = ~is_c7 | analysis_present
    deduped.loc[
        (parse_success | analytical_known) & complete_context & ~has_error, "_quality"
    ] = 2
    deduped["_source_order"] = range(len(deduped))
    deduped = deduped.sort_values(KEY_COLUMNS + ["_quality", "_source_order"])
    deduped = deduped.drop_duplicates(KEY_COLUMNS, keep="last")
    return deduped.drop(columns=["_quality", "_source_order"])


def _condition_distribution(game: Game, condition: str) -> Distribution | None:
    if condition in {"C1", "C2", "C3"} or condition.startswith("C7-real"):
        return game.real_ce
    if condition in {"C4", "C5"} or condition.startswith("C7-fake"):
        return game.fake_ce
    return None


def _row_best_response(
    game: Game, distribution: Distribution, recommendation: str
) -> dict[str, object] | None:
    rec_mass = sum(
        prob for (row_action, _), prob in distribution.items() if row_action == recommendation
    )
    if rec_mass <= 0:
        return None

    expected_payoffs: dict[str, float] = {}
    for action in game.row_actions:
        payoff = 0.0
        action_i = game.row_index(action)
        for (row_action, col_action), prob in distribution.items():
            if row_action != recommendation:
                continue
            payoff += (prob / rec_mass) * game.row_payoffs[action_i, game.col_index(col_action)]
        expected_payoffs[action] = float(payoff)

    best_payoff = max(expected_payoffs.values())
    best_actions = [
        action for action, payoff in expected_payoffs.items() if abs(payoff - best_payoff) < 1e-9
    ]
    recommendation_payoff = expected_payoffs.get(recommendation)
    return {
        "best_response_action": ";".join(best_actions),
        "best_response_payoff": best_payoff,
        "recommendation_payoff": recommendation_payoff,
        "best_response_margin": best_payoff - recommendation_payoff
        if recommendation_payoff is not None
        else None,
    }


def annotate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Add reasoning-feature and best-response annotations."""

    if df.empty:
        return df

    annotated = df.copy()
    raw_responses = annotated.get(
        "raw_response", pd.Series("", index=annotated.index)
    ).fillna("")

    for feature in REASONING_FEATURES:
        column = f"reasoning_{feature}"
        parsed_features = raw_responses.apply(
            lambda text, name=feature: extract_reasoning_features(str(text))[name]
        )
        if column in annotated:
            annotated[column] = annotated[column].fillna(parsed_features)
        else:
            annotated[column] = parsed_features

    games = {game.name: game for game in get_games()}
    best_response_rows: list[dict[str, object]] = []
    for _, row in annotated.iterrows():
        game = games.get(row.get("game"))
        condition = str(row.get("condition"))
        recommendation = row.get("recommendation")
        if game is None or not isinstance(recommendation, str):
            best_response_rows.append({})
            continue
        distribution = _condition_distribution(game, condition)
        if distribution is None:
            best_response_rows.append({})
            continue
        best_response_rows.append(
            _row_best_response(game, distribution, recommendation) or {}
        )

    best_response_df = pd.DataFrame(best_response_rows, index=annotated.index)
    for column in [
        "best_response_action",
        "best_response_payoff",
        "recommendation_payoff",
        "best_response_margin",
    ]:
        annotated[column] = best_response_df.get(column)

    parsed_action = annotated.get("parsed_action", pd.Series(index=annotated.index))
    annotated["chose_best_response"] = [
        bool(parsed in str(best_actions).split(";"))
        if isinstance(parsed, str) and isinstance(best_actions, str)
        else pd.NA
        for parsed, best_actions in zip(parsed_action, annotated["best_response_action"])
    ]
    annotated["chose_best_response"] = annotated["chose_best_response"].astype("boolean")
    return annotated


def summarize(
    df: pd.DataFrame, results_dir: Path, max_trials: int | None = None
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    df = filter_trials(df, max_trials)
    df = deduplicate_records(df)
    df = annotate_records(df)
    if df.empty:
        return outputs
    suffix = f"_n{max_trials}" if max_trials is not None else ""

    behavioral = df[df["condition"].str.startswith(("C1", "C2", "C3", "C4", "C5", "C7"))].copy()
    if not behavioral.empty:
        group_cols = ["model", "game", "condition", "recommendation"]
        summary = (
            behavioral.groupby(group_cols, dropna=False)
            .agg(
                attempted_n=("complied", "size"),
                parsed=("parse_success", "sum"),
                compliance_rate=("complied", "mean"),
            )
            .reset_index()
        )
        parsed_summary = (
            behavioral[behavioral["parse_success"] == True]  # noqa: E712
            .groupby(group_cols, dropna=False)["complied"]
            .mean()
            .rename("compliance_rate_parsed")
            .reset_index()
        )
        summary = summary.merge(parsed_summary, on=group_cols, how="left")
        intervals = []
        for _, row in summary.iterrows():
            successes = int(round(row["compliance_rate"] * row["attempted_n"]))
            test = binomtest(successes, int(row["attempted_n"]), p=0.5)
            ci = test.proportion_ci(confidence_level=0.95)
            intervals.append((ci.low, ci.high, test.pvalue))
        summary[["ci_low", "ci_high", "p_vs_0_5"]] = intervals
        path = results_dir / f"behavioral_summary{suffix}.csv"
        summary.to_csv(path, index=False)
        outputs["behavioral_summary"] = path

        reasoning_agg = {
            "attempted_n": ("condition", "size"),
            "parsed": ("parse_success", "sum"),
            "compliance_rate": ("complied", "mean"),
            "best_response_rate": ("chose_best_response", "mean"),
        }
        for column in REASONING_COLUMNS:
            reasoning_agg[column.replace("reasoning_", "")] = (column, "mean")
        reasoning_summary = (
            behavioral.groupby(["model", "game", "condition"], dropna=False)
            .agg(**reasoning_agg)
            .reset_index()
        )
        path = results_dir / f"reasoning_summary{suffix}.csv"
        reasoning_summary.to_csv(path, index=False)
        outputs["reasoning_summary"] = path

    analytical = df[df["condition"].str.startswith("C6")].copy()
    if not analytical.empty and "analytical_correct" in analytical:
        summary = (
            analytical.groupby(["model", "game", "condition"], dropna=False)
            .agg(
                n=("analytical_correct", "size"),
                correct_rate=("analytical_correct", "mean"),
            )
            .reset_index()
        )
        path = results_dir / f"analytical_summary{suffix}.csv"
        summary.to_csv(path, index=False)
        outputs["analytical_summary"] = path

    path = results_dir / f"all_records{suffix}.csv"
    df.to_csv(path, index=False)
    outputs["all_records"] = path
    return outputs
