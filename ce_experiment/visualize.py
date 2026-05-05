"""Figure generation for CE compliance experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_empty(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, title, ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figures(df: pd.DataFrame, results_dir: Path) -> list[Path]:
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    if df.empty:
        path = figures_dir / "no_results.png"
        _save_empty(path, "No results yet")
        return [path]

    behavioral = df[df["condition"].str.startswith(("C1", "C2", "C3", "C4", "C5", "C7"))].copy()
    if not behavioral.empty:
        summary = (
            behavioral.groupby(["condition", "model"], dropna=False)["complied"]
            .mean()
            .reset_index()
        )
        pivot = summary.pivot(index="condition", columns="model", values="complied").sort_index()
        fig, ax = plt.subplots(figsize=(11, 5.5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Compliance rate")
        ax.set_xlabel("Condition")
        ax.set_title("Recommendation Compliance by Condition and Model")
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        path = figures_dir / "compliance_by_condition_model.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)

        game_summary = (
            behavioral.groupby(["game", "condition"], dropna=False)["complied"]
            .mean()
            .reset_index()
        )
        pivot = game_summary.pivot(index="game", columns="condition", values="complied")
        fig, ax = plt.subplots(figsize=(11, 5.5))
        im = ax.imshow(
            pivot.infer_objects(copy=False).fillna(0).to_numpy(dtype=float),
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="viridis",
        )
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title("Compliance Heatmap")
        fig.colorbar(im, ax=ax, label="Compliance rate")
        fig.tight_layout()
        path = figures_dir / "compliance_heatmap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)

    analytical = df[df["condition"].str.startswith("C6")].copy()
    if not analytical.empty and "analytical_correct" in analytical:
        summary = (
            analytical.groupby(["condition", "model"], dropna=False)["analytical_correct"]
            .mean()
            .reset_index()
        )
        pivot = summary.pivot(index="condition", columns="model", values="analytical_correct")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Correct identification rate")
        ax.set_xlabel("Condition")
        ax.set_title("Analytical CE Identification")
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        path = figures_dir / "analytical_correctness.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)

    return outputs
