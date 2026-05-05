"""Metrics and statistical summaries for experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


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


def summarize(
    df: pd.DataFrame, results_dir: Path, max_trials: int | None = None
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    df = filter_trials(df, max_trials)
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
