"""Metrics and statistical summaries for experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


KEY_COLUMNS = ["model", "game", "condition", "recommendation", "trial"]


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

    deduped["_quality"] = 0
    deduped.loc[raw_present & ~has_error, "_quality"] = 1
    deduped.loc[(parse_success | analytical_known) & ~has_error, "_quality"] = 2
    deduped["_source_order"] = range(len(deduped))
    deduped = deduped.sort_values(KEY_COLUMNS + ["_quality", "_source_order"])
    deduped = deduped.drop_duplicates(KEY_COLUMNS, keep="last")
    return deduped.drop(columns=["_quality", "_source_order"])


def summarize(
    df: pd.DataFrame, results_dir: Path, max_trials: int | None = None
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    df = filter_trials(df, max_trials)
    df = deduplicate_records(df)
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
