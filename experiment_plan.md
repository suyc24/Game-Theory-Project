# LLM Correlated-Equilibrium Compliance Experiments

## Goal

Build a complete, self-contained Python experiment pipeline that tests whether large language models (LLMs) obey correlated-equilibrium (CE) recommendations in normal-form games, and whether their compliance stems from genuine incentive reasoning or from surface cues. The pipeline must handle game definition, prompt generation, API calling, response parsing, and results analysis/visualization. Everything should be runnable with a single entry point (`python run_all.py`).

---

## 1  Project Structure

```
ce_experiment/
├── run_all.py                 # orchestrator: runs phases 1-4 in order
├── config.py                  # API endpoint, key, model list, sample sizes, paths
├── games.py                   # game & CE definitions
├── prompts.py                 # prompt templates for every condition
├── runner.py                  # async API caller + response collector
├── parser.py                  # extract action choices & reasoning from raw responses
├── analysis.py                # compute metrics, statistical tests
├── visualize.py               # generate all figures
├── results/                   # auto-created; stores raw JSON + figures
└── requirements.txt
```

---

## 2  API Configuration (CRITICAL — READ FIRST)

All four models are called through a **single OpenAI-compatible API endpoint** provided by AutoDL Art. This means the code only needs the standard `openai` Python package and one API key.

### Endpoint & Auth

```python
# config.py
import os

API_BASE_URL = "https://www.autodl.art/api/v1/"
API_KEY = os.environ["AUTODL_API_KEY"]   # never hardcode
```

### Model Identifiers

| Model              | model string for API        |
|---------------------|-----------------------------|
| GPT-5.4             | `gpt-5.4`                  |
| Claude Opus 4.6     | `claude-opus-4-6`          |
| Gemini 3.1 Pro      | `gemini-3.1-pro`           |
| DeepSeek V4 Pro     | `deepseek-v4-pro`          |

> **Note:** The exact model strings above may need minor adjustment depending on what AutoDL Art actually lists. Add a `MODEL_ALIASES` dict in `config.py` so the user can override any model string without changing the rest of the codebase.

### How to Call

Use the standard `openai` Python package (v1.x+) with a custom `base_url`:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)

response = await client.chat.completions.create(
    model="gpt-5.4",          # swap in any of the 4 model strings
    messages=[{"role": "user", "content": prompt}],
    temperature=1.0,
    top_p=1.0,
    max_tokens=2048,
)
answer = response.choices[0].message.content
```

**All four models use this identical calling pattern — only the `model` parameter changes.** Do NOT import any Anthropic, Google, or DeepSeek SDK. Everything goes through the `openai` package pointed at the AutoDL Art endpoint.

---

## 3  Games to Use

Define **4 two-player normal-form games** as Python dataclasses or dicts. Each game stores: name, row-player actions, column-player actions, payoff bimatrix (as two NumPy arrays), and one or more CE distributions (as a dict mapping joint-action tuples to probabilities).

### Game 1 — Battle of the Sexes (BoS)

|           | Opera | Football |
|-----------|-------|----------|
| **Opera**     | (3,2) | (0,0)    |
| **Football**  | (0,0) | (2,3)    |

- **Real CE distribution:** {(Opera,Opera): 0.5, (Football,Football): 0.5}
  - This CE yields expected payoffs (2.5, 2.5), better for both than mixing.
- **Fake CE distribution (for C4/C5):** {(Opera,Football): 0.5, (Football,Opera): 0.5}
  - Always mis-coordinate → payoff 0. Deviating matches the opponent → payoff > 0. Not IC.
- Key feature: recommendation is always to coordinate; tests whether LLM follows even when its own preferred equilibrium is not chosen.

### Game 2 — Chicken (Hawk–Dove)

|           | Swerve | Straight |
|-----------|--------|----------|
| **Swerve**    | (3,3)  | (1,4)    |
| **Straight**  | (4,1)  | (0,0)    |

- **Real CE distribution:** {(Swerve,Swerve): 0.4, (Swerve,Straight): 0.3, (Straight,Swerve): 0.3, (Straight,Straight): 0}
  - When told "Swerve": obey EV = 3×(4/7)+1×(3/7) ≈ 2.14, deviate EV = 4×(4/7)+0×(3/7) ≈ 2.29. Adjust to {0.5, 0.25, 0.25, 0} to make obedience strictly optimal: obey EV = 3×(2/3)+1×(1/3) ≈ 2.33, deviate EV = 4×(2/3)+0×(1/3) ≈ 2.67 — still not IC. Use {0.5, 0.2, 0.3, 0} or verify IC numerically and pick distribution that makes obedience strictly better for both recommendations. **You must verify IC before finalizing.**
- **Fake CE distribution:** {(Straight,Straight): 0.5, (Swerve,Swerve): 0.5}
  - When told Straight, deviating to Swerve gives 3 > 0.
- Key feature: CE asks the player to sometimes accept a "weak" action; tests willingness to swerve.

### Game 3 — A Pure-Coordination Game with a Pareto-dominated NE

|           | A     | B     |
|-----------|-------|-------|
| **A**     | (2,2) | (0,0) |
| **B**     | (0,0) | (4,4) |

- Pure NE: (A,A) and (B,B). (B,B) Pareto-dominates.
- **Real CE distribution:** {(A,A): 0.1, (B,B): 0.9}
  - When told "A," player should still obey because deviating to B yields 0.
- **Fake CE distribution:** {(A,B): 0.5, (B,A): 0.5}
  - Always zero payoff.
- Key feature: tests whether the LLM overrides the mediator recommendation when it "knows" B is Pareto-better.

### Game 4 — Prisoner's Dilemma (Control)

|           | Cooperate | Defect |
|-----------|-----------|--------|
| **Cooperate** | (3,3)     | (0,5)  |
| **Defect**    | (5,0)     | (1,1)  |

- Unique NE: (Defect, Defect). No CE can recommend (Cooperate, Cooperate) with probability 1 because defecting is a profitable deviation.
- **Real CE distribution:** {(Defect, Defect): 1.0}  (trivial CE = NE)
- **Fake CE distribution:** {(Cooperate, Cooperate): 1.0}
  - Labeled as CE but deviating to Defect gives 5 > 3 → not IC.
- Key feature: when recommended "Defect," this aligns with NE; when we test a fake CE that recommends "Cooperate," we see if the LLM blindly trusts the label.

**Implementation note:** In `games.py`, write a function `verify_ic(game, distribution) -> bool` that numerically checks incentive-compatibility for every recommendation in the distribution. Run this at startup to confirm all "real" CEs are IC and all "fake" CEs are not IC.

---

## 4  Experimental Conditions (7 conditions total)

For **each game**, generate prompts under the following conditions. The LLM is always assigned the role of **Row Player (Player 1)**.

### C1 — Bare Recommendation
> Present the payoff matrix. Say: "A trusted mediator has drawn a joint action according to a publicly known distribution and privately recommends that you play [action]. What action do you choose?"
- Do NOT reveal the distribution. Do NOT use the phrase "correlated equilibrium."

### C2 — Full CE Distribution Revealed
> Same as C1, but also show the full joint distribution table and explain that each player receives only their own recommendation.

### C3 — CE Label
> Same as C2, but explicitly say: "This distribution is a correlated equilibrium of the game."

### C4 — Fake CE (non-IC distribution, labeled as CE)
> Present a distribution that is **not** incentive-compatible but label it as a "correlated equilibrium."
- Use the Fake CE distributions defined in §3 for each game.
- Key test: if the LLM obeys a fake CE, compliance is **label-driven**, not payoff-driven.

### C5 — Honest Non-CE (non-IC distribution, no CE label)
> Same non-IC distribution as C4, but do NOT call it a correlated equilibrium. Just say "a mediator recommends…"
- Comparing C4 vs C5 isolates the effect of the CE label.

### C6 — Analytical Mode (Knowledge Baseline)
> Present the payoff matrix and the CE distribution. Ask: "Is the following distribution a correlated equilibrium? For each action recommendation, compute the expected payoff of obeying versus deviating."
- The model answers as an analyst, not as a player. This measures **knowledge**.
- Run this for both real CE and fake CE distributions per game (label them C6-real and C6-fake in the data).

### C7 — Analytical-then-Behavioral (two-turn)
> First turn: same as C6 (analytical). Second turn: "Now you are Player 1. The mediator recommends [action]. What do you choose?"
- Tests whether performing the analysis right before acting closes or reveals the knowledge–action gap.
- Implement as a **multi-turn** conversation (two messages in the messages list).

---

## 5  Prompt Templates

Write a `prompts.py` that contains a function `generate_prompt(game, condition, recommendation, role="row")` returning either a single prompt string (C1–C6) or a list of message dicts for multi-turn (C7). Requirements:

- Use **neutral** language: no phrases like "you should cooperate" or "the rational thing to do."
- Always present the payoff matrix as a clear ASCII table with row = Player 1, column = Player 2.
- Always state: "You are Player 1. Your payoff is the first number in each cell."
- End behavioral prompts (C1–C5, C7 turn 2) with: **"Reply with your chosen action on the first line (just the action name), then explain your reasoning below."**
- End analytical prompts (C6, C7 turn 1) with: **"Show your calculations step by step."**
- For condition C7, return a list of two messages:
  ```python
  [
      {"role": "user", "content": "<C6 analytical prompt>"},
      {"role": "assistant", "content": ""},  # placeholder — will be filled by first API call
      {"role": "user", "content": "Now you are Player 1. The mediator recommends [action]. What do you choose? Reply with your chosen action on the first line (just the action name), then explain your reasoning below."}
  ]
  ```
  The runner should make two sequential API calls for C7: first call with the analytical prompt to get the model's analysis, insert that as the assistant message, then call again with the full three-message conversation.

---

## 6  Sample Sizes and Parameters

For each (game × condition × recommendation) cell, collect **N = 5 responses**.

API parameters for all calls:
- `temperature = 1.0` (for behavioral diversity)
- `top_p = 1.0`
- `max_tokens = 2048`

Total cells estimate:
- 4 games × 7 conditions × ~2 recommendations per game on average × 4 models × 5 trials
- ≈ 1,120 API calls (rough estimate; some conditions only have 1 recommendation)

---

## 7  Runner (`runner.py`)

- Use `asyncio` with the `AsyncOpenAI` client. All four models use the same client instance (same base_url, same API key) — only `model` differs.
- Implement rate-limit handling with exponential backoff (catch `RateLimitError`, `APIStatusError`).
- For each response, save a JSON record:
  ```json
  {
    "game": "BoS",
    "condition": "C2",
    "model": "gpt-5.4",
    "recommendation": "Opera",
    "raw_response": "...",
    "parsed_action": "Opera",
    "complied": true,
    "trial": 3,
    "timestamp": "2026-05-..."
  }
  ```
- Write results incrementally to `results/raw_{model}.jsonl` so partial runs can resume.
- Before starting, check which (game, condition, model, recommendation, trial) combinations already exist in the JSONL files, and skip them.
- Add a concurrency semaphore (e.g., `asyncio.Semaphore(5)`) to avoid overwhelming the endpoint.

---

## 8  Parser (`parser.py`)

- Extract the **first line** of the response as the chosen action.
- Normalize to canonical action names (case-insensitive, strip whitespace).
- If the first line doesn't match any valid action, attempt fuzzy matching (e.g., "I choose Opera" → Opera, "我选择Opera" → Opera).
- Record `parsed_action` and a boolean `parse_success`.
- For analytical prompts (C6), extract whether the model correctly identified IC / non-IC. Look for keywords: "is a correlated equilibrium" / "is not a correlated equilibrium" / "incentive compatible" / "not incentive compatible" / "profitable deviation".

---

## 9  Analysis (`analysis.py`)

Compute and save the following:

### 9.1 Compliance Rate (→ Q1)
For each (game, condition, model): `compliance_rate = # obey / # total`.
- This directly answers **Q1**: does the LLM follow CE recommendations?

### 9.2 Knowledge Accuracy — C6 (→ Q3 baseline)
For each (game, model): did the model correctly determine whether the distribution is a CE?
- Run on both real and fake distributions.

### 9.3 Knowledge–Action Gap — C6 vs C3 (→ Q3)
Compare C6 analytical accuracy (does the model know it's a CE?) with C3 behavioral compliance (does the model obey?).
- Compute: gap = C6_accuracy − C3_compliance. Positive gap = "knows but doesn't do."

### 9.4 Two-Turn Effect — C7 (→ Q3)
Among C7 trials where the model correctly identified the CE in the analytical turn, what fraction still deviated in the behavioral turn?
- Compare C7 compliance vs C3 compliance.

### 9.5 Label Effect — C4 vs C5 (→ Q2)
Compare compliance rates C4 (fake CE, labeled) vs C5 (fake CE, unlabeled) using Fisher's exact test. Report p-values.
- If C4 >> C5, the model is label-driven.

### 9.6 Information Effect — C1 vs C2 vs C3 (→ Q2)
Pairwise Fisher's exact tests with Bonferroni correction.
- If C1 ≈ C2 ≈ C3, information doesn't matter → behavior is not driven by payoff reasoning.
- If C2 > C1 and C3 ≈ C2, information helps but the label doesn't add anything.

### 9.7 Fake vs Real CE — C3 vs C4 (→ Q2)
Compare C3 (real CE, labeled) vs C4 (fake CE, labeled).
- If C3 ≈ C4, the model doesn't distinguish real from fake → compliance is label-driven.
- If C3 >> C4, the model does verify IC → compliance is payoff-driven.

### 9.8 Summary Table
Produce a CSV/DataFrame: rows = (game, condition), columns = models, cells = compliance rate ± 95% CI (Wilson interval).

### 9.9 Research Question Mapping
Print a summary report that explicitly maps each statistical result to Q1, Q2, or Q3:
```
=== Q1: Do LLMs obey CE recommendations? ===
[C1/C2/C3 compliance rates per model per game]

=== Q2: Genuine reasoning or surface cues? ===
[C4 vs C5 label effect, C3 vs C4 fake vs real, C1 vs C2 vs C3 information effect]

=== Q3: Knowledge–Action Gap ===
[C6 accuracy vs C3 compliance, C7 two-turn effect]
```

---

## 10  Visualization (`visualize.py`)

Generate publication-quality figures using `matplotlib` (with `seaborn` styling). Save as both PNG (300 dpi) and PDF in `results/figures/`.

### Figure 1 — Compliance Heatmap
- Rows: (game × condition C1–C5), Columns: 4 models. Cell color = compliance rate. Annotate cells with percentage.

### Figure 2 — Condition Comparison (grouped bar chart)
- One panel per game. X-axis = conditions C1–C5. Bars grouped by model. Y-axis = compliance rate. Error bars = 95% Wilson CI.

### Figure 3 — Knowledge vs Behavior Scatter (Q3 visualization)
- One point per (game, model). X = analytical accuracy (C6-real), Y = behavioral compliance (C3). Diagonal line = no gap. Points below the line indicate a knowledge–action gap.

### Figure 4 — Label Effect: C4 vs C5 (Q2 visualization)
- Paired bar chart per game showing compliance under fake CE with label vs without label. Add significance stars (* p<0.05, ** p<0.01, *** p<0.001).

### Figure 5 — Two-turn Effect: C3 vs C7 (Q3 visualization)
- Does doing the analysis first (C7) change compliance relative to C3? Grouped bar chart.

### Figure 6 — Fake vs Real CE: C3 vs C4 (Q2 visualization)
- Grouped bar chart. If bars are similar → model is label-driven.

---

## 11  Entry Point (`run_all.py`)

```python
"""
Usage:
    python run_all.py                    # run everything
    python run_all.py --phase prompts    # only generate & print sample prompts
    python run_all.py --phase run        # only call APIs (reads existing prompts)
    python run_all.py --phase analyze    # only analyze (reads existing results/)
    python run_all.py --phase visualize  # only make figures
    python run_all.py --yes              # skip cost confirmation
"""
```

Implement CLI with `argparse`. Default = run all phases sequentially.

---

## 12  Additional Requirements

1. **Reproducibility**: set random seeds; save all raw responses; log every API call's parameters.
2. **Cost awareness**: before running, print an estimated token count and approximate cost. Ask for confirmation (unless `--yes` flag is passed).
3. **Error handling**: if an API call fails after 3 retries with exponential backoff, log the error and continue; do not crash the whole pipeline.
4. **README.md**: generate a short README explaining:
   - How to install dependencies (`pip install -r requirements.txt`)
   - How to set the environment variable: `export AUTODL_API_KEY="your_key_here"`
   - How to run: `python run_all.py`
   - How to run individual phases
   - How to adjust model strings if needed (via `MODEL_ALIASES` in config.py)
5. **Type hints** throughout; use `dataclasses` or `pydantic` for structured data.
6. **No hardcoded API keys** — use environment variables only.
7. **Single SDK**: only use the `openai` Python package for all API calls. Do NOT use `anthropic`, `google-generativeai`, or any other provider-specific SDK.

---

## 13  requirements.txt

```
openai>=1.30.0
numpy
scipy
matplotlib
seaborn
pandas
tqdm
```

---

## 14  Deliverables Checklist

When finished, the `ce_experiment/` directory must contain:

- [ ] All `.py` files listed in §1
- [ ] `requirements.txt` (as specified in §13)
- [ ] `README.md`
- [ ] A `results/` folder (can be empty; populated after running)
- [ ] Running `python run_all.py --phase prompts` prints sample prompts for every (game, condition) combination for manual inspection, without calling any API
- [ ] Running `python run_all.py --phase analyze` on provided sample data produces all 6 figures, the summary CSV, and the Q1/Q2/Q3 mapping report
- [ ] The `verify_ic()` function in `games.py` passes for all real CEs and fails for all fake CEs

Please implement the full pipeline now. Start with `config.py` and `games.py`, then build outward.
