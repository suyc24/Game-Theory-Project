# LLM Correlated-Equilibrium Compliance Experiments

This repository studies whether LLM agents obey mediator recommendations in
normal-form games because they reason through incentive compatibility, or
because they respond to institutional authority cues such as "trusted mediator"
and "correlated equilibrium".

## Current Design

The experiment assigns the model to Player 1 and tests seven prompt conditions:

- `C1`: trusted mediator recommendation, no distribution shown.
- `C2`: real CE distribution shown, no CE label.
- `C3`: real CE distribution shown and labeled as CE.
- `C4`: fake non-IC distribution falsely labeled as CE.
- `C5`: same fake distribution without the CE label.
- `C6-real` / `C6-fake`: analytical CE-identification task.
- `C7-real` / `C7-fake`: analytical task followed by behavioral choice.

Games currently included:

- Battle of the Sexes
- Chicken
- Pure Coordination
- Prisoner's Dilemma
- Rock-Paper-Scissors
- 3x3 Dominated Strategy

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Set the API key through the environment or `API_KEY.md`:

```bash
export AUTODL_API_KEY="your_key"
```

All models are called through an OpenAI-compatible endpoint configured in
`ce_experiment/config.py`.

## Analyze Existing Results

This does not call the API:

```bash
python run_all.py --analyze-only --n 5
```

Outputs are written to `results/`:

- `raw_{model}.jsonl`
- `all_records_n5.csv`
- `behavioral_summary_n5.csv`
- `analytical_summary_n5.csv`
- `ce_validation.json`
- `figures/*.png`

## Cost-Aware Continuation

Before making API calls, inspect the remaining work:

```bash
python run_all.py --plan-only --retry-errors --n 5 \
  --games Rock-Paper-Scissors "3x3 Dominated Strategy"
```

Run only a focused slice:

```bash
python run_all.py --retry-errors --n 3 \
  --games Rock-Paper-Scissors "3x3 Dominated Strategy" \
  --conditions C4 C5 C6-fake \
  --models gpt-5.4 gemini-3.1-pro \
  --concurrency 2
```

Useful switches:

- `--plan-only`: print pending jobs and estimated API calls, then do no API work.
- `--retry-errors`: treat existing API error records as incomplete.
- `--games`: run only selected games.
- `--conditions`: run only selected conditions.
- `--max-pending`: cap the number of jobs in a single invocation.
- `--skip-c7-fake`: omit the expensive fake analytical-then-behavioral condition.

Because `C7` uses two model calls per experimental job, the cheapest next step is
usually to finish `C4`, `C5`, and `C6-fake` first. That directly tests whether
authority labels override incentive reasoning, without paying for every
two-turn behavioral condition.
