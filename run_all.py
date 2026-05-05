"""Single entry point for LLM CE compliance experiments."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ce_experiment import config
from ce_experiment.analysis import filter_trials, load_records, summarize
from ce_experiment.games import get_games, validate_games
from ce_experiment.runner import run_experiments
from ce_experiment.visualize import make_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=config.DEFAULT_N, help="responses per cell")
    parser.add_argument(
        "--models",
        nargs="+",
        default=config.DEFAULT_MODELS,
        help="model labels to run; labels are resolved through MODEL_ALIASES",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=config.DEFAULT_CONCURRENCY,
        help="maximum concurrent API calls",
    )
    parser.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=config.DEFAULT_TOP_P)
    parser.add_argument("--max-tokens", type=int, default=config.DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=config.DEFAULT_REQUEST_TIMEOUT,
        help="per-request timeout in seconds",
    )
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="generate placeholder records without calling the API",
    )
    parser.add_argument(
        "--skip-c7-fake",
        action="store_true",
        help="omit the extra C7 fake-distribution behavioral extension",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="skip API calls and regenerate summaries/figures from existing JSONL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_games()
    validation_path = args.results_dir / "ce_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(f"CE validation written to {validation_path}", flush=True)

    if not args.analyze_only:
        asyncio.run(
            run_experiments(
                games=get_games(),
                model_labels=args.models,
                n=args.n,
                results_dir=args.results_dir,
                concurrency=args.concurrency,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                request_timeout=args.request_timeout,
                dry_run=args.dry_run,
                include_c7_fake=not args.skip_c7_fake,
            )
        )

    df = load_records(args.results_dir)
    outputs = summarize(df, args.results_dir, max_trials=args.n)
    figure_df = filter_trials(df, args.n)
    figures = make_figures(figure_df, args.results_dir)
    print(
        f"Loaded {len(df)} raw records; analyzing {len(figure_df)} records with trial < {args.n}.",
        flush=True,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}", flush=True)
    for path in figures:
        print(f"figure: {path}", flush=True)


if __name__ == "__main__":
    main()
