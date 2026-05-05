# LLM Correlated-Equilibrium Compliance Experiments

## 1. Research Questions

This experiment studies whether large language models comply with private recommendations from a correlated-equilibrium (CE) mediator in two-player normal-form games. The three target questions are:

**Q1.** When an LLM acts as a player in a normal-form game and receives a private recommendation from a CE mediator, does it obey the recommendation?

**Q2.** Is the model's behavior driven by genuine incentive-compatibility reasoning, or by surface cues such as the label "correlated equilibrium," the mere presence of a recommendation, or cooperative and fairness biases?

**Q3.** When an LLM can correctly identify the equilibrium of a game in analytical mode, does it choose the corresponding action when placed in the role of a player?

## 2. Experimental Design

The experiment used four two-player normal-form games:

1. Battle of the Sexes
2. Chicken
3. Pure Coordination
4. Prisoner's Dilemma

The LLM was always assigned the role of Player 1, whose payoff is the first payoff in each cell. Each game included:

- a real CE distribution that passed numerical incentive-compatibility checks;
- a fake CE distribution that failed incentive-compatibility checks.

The experiment used four models through the AutoDL OpenAI-compatible endpoint:

- `gpt-5.4`
- `claude-opus-4.6`
- `gemini-3.1-pro`
- `deepseek-v4-pro`

The main behavioral conditions were:

- `C1`: bare mediator recommendation, no distribution shown, no CE label.
- `C2`: real CE distribution shown, no CE label.
- `C3`: real CE distribution shown and explicitly labeled as a CE.
- `C4`: fake non-IC distribution shown but falsely labeled as a CE.
- `C5`: same fake distribution shown honestly without the CE label.
- `C6-real` / `C6-fake`: analytical mode, asking whether the distribution is a CE.
- `C7-real` / `C7-fake`: analytical-then-behavioral two-turn condition.

## 3. Data Inclusion Rule

Although earlier pilot runs collected some cells with `N=10` or `N=20`, the final analysis uses **N = 5**. Concretely, only trial numbers `0-4` are included for each `(game, condition, model, recommendation)` cell. Older extra trials remain in the raw JSONL files but are excluded from the reported statistics.

Final N=5 analysis files:

- `results/all_records_n5.csv`
- `results/behavioral_summary_n5.csv`
- `results/analytical_summary_n5.csv`

The filtered N=5 dataset contains:

- 1140 analyzed records;
- 980 behavioral records;
- 160 analytical records.

There were 50 timeout/error records in the N=5 dataset, all from `deepseek-v4-pro`. These are retained in attempted counts. For behavioral compliance, two rates are reported:

- `compliance_rate`: treats unparsed/error records as non-compliance;
- `compliance_rate_parsed`: computes compliance only among successfully parsed behavioral responses.

Because timeout failures are concentrated in one model, `compliance_rate_parsed` is more informative for behavioral tendency, while `parsed` and `attempted_n` remain important for reliability.

## 4. Main Behavioral Results

### 4.1 Compliance by Condition

| condition | cells | attempted_n | parsed_n | compliance_attempted | compliance_parsed |
| --- | --- | --- | --- | --- | --- |
| C1 | 28 | 140 | 131 | 0.936 | 1.000 |
| C2 | 28 | 140 | 140 | 0.993 | 0.993 |
| C3 | 28 | 140 | 140 | 1.000 | 1.000 |
| C4 | 28 | 140 | 115 | 0.464 | 0.500 |
| C5 | 28 | 140 | 127 | 0.293 | 0.304 |
| C7-fake | 28 | 140 | 139 | 0.000 | 0.000 |
| C7-real | 28 | 140 | 137 | 0.964 | 0.986 |

The strongest result is that models almost always obey real CE recommendations when acting behaviorally. Compliance is near perfect in `C2`, `C3`, and `C7-real`.

The second key result is that the fake CE label matters. In `C4`, where a non-IC distribution is falsely labeled as a CE, parsed compliance is 0.500. In `C5`, where the same non-IC distribution is shown without the CE label, parsed compliance falls to 0.304. This 19.6 percentage point difference suggests that surface labeling can induce obedience even when the recommendation is not incentive compatible.

The third key result is that analytical reasoning before action eliminates obedience to fake CEs. In `C7-fake`, parsed compliance is 0.000. Once models are asked to analyze the distribution before acting, they stop following fake recommendations.

### 4.2 Compliance by Model

| condition | claude-opus-4.6 | deepseek-v4-pro | gemini-3.1-pro | gpt-5.4 |
| --- | --- | --- | --- | --- |
| C1 | 1.000 | 1.000 | 1.000 | 1.000 |
| C2 | 1.000 | 1.000 | 1.000 | 0.971 |
| C3 | 1.000 | 1.000 | 1.000 | 1.000 |
| C4 | 1.000 | 0.000 | 0.171 | 0.686 |
| C5 | 0.286 | 0.000 | 0.286 | 0.600 |
| C7-fake | 0.000 | 0.000 | 0.000 | 0.000 |
| C7-real | 1.000 | 1.000 | 0.943 | 1.000 |

Model-level patterns are heterogeneous:

- Claude shows the strongest label effect: it obeys fake CE recommendations in `C4` at 1.000 parsed compliance, but much less in `C5`.
- GPT-5.4 also follows fake recommendations substantially, both with and without the label, suggesting a mix of CE-label sensitivity and general recommendation/cooperation bias.
- Gemini is more skeptical in fake-CE conditions.
- DeepSeek shows zero parsed compliance in fake conditions, but its results are less reliable because many DeepSeek fake-condition calls timed out.

### 4.3 Compliance by Game

| condition | Battle of the Sexes | Chicken | Prisoner's Dilemma | Pure Coordination |
| --- | --- | --- | --- | --- |
| C1 | 1.000 | 1.000 | 1.000 | 1.000 |
| C2 | 1.000 | 0.975 | 1.000 | 1.000 |
| C3 | 1.000 | 1.000 | 1.000 | 1.000 |
| C4 | 0.600 | 0.475 | 0.500 | 0.429 |
| C5 | 0.200 | 0.250 | 0.250 | 0.514 |
| C7-fake | 0.000 | 0.000 | 0.000 | 0.000 |
| C7-real | 1.000 | 0.950 | 1.000 | 1.000 |

The CE-compliance result is robust across games. Real CE recommendations are almost always followed. Fake recommendations are followed much less, and this drops to zero after explicit analysis.

The Prisoner's Dilemma is especially informative: fake CE recommendations ask the model to cooperate even though defection is profitable. Compliance in `C4` is 0.500 and in `C5` is 0.250, showing both a CE-label effect and some cooperative bias.

## 5. Analytical Results

| condition | claude-opus-4.6 | deepseek-v4-pro | gemini-3.1-pro | gpt-5.4 |
| --- | --- | --- | --- | --- |
| C6-fake | 1.000 | 0.938 | 0.550 | 0.950 |
| C6-real | 1.000 | 1.000 | 1.000 | 1.000 |

Models are excellent at recognizing real CE distributions in analytical mode: every model reached 1.000 correctness on `C6-real`.

Fake CE identification is weaker but still generally strong: mean correctness across models is 0.859. Claude, GPT-5.4, and DeepSeek are high; Gemini is notably weaker at 0.550.

The analytical results show that most models possess the relevant incentive-compatibility knowledge. The behavioral results show that this knowledge is not always activated when the model is simply asked to act as a player.

## 6. Answers to the Three Research Questions

### Q1. Does an LLM obey private CE recommendations?

**Yes, overwhelmingly, when the recommendation comes from a real CE distribution.**

The strongest evidence is:

- `C2` parsed compliance: 0.993
- `C3` parsed compliance: 1.000
- `C7-real` parsed compliance: 0.986

Even in `C1`, where the distribution is not revealed and the prompt does not mention CE, parsed compliance is 1.000. This suggests that LLMs are highly deferential to trusted mediator recommendations in these games.

The answer is therefore: **LLMs almost always obey real CE recommendations in this setup.**

### Q2. Is behavior driven by incentive reasoning or surface cues?

**Both, but the behavioral evidence shows a strong role for surface cues unless analytical reasoning is explicitly invoked.**

There are three pieces of evidence.

First, the mere presence of a trusted recommendation is powerful. In `C1`, the model sees no CE distribution and no CE label, yet parsed compliance is 1.000. This cannot be explained by explicit incentive-compatibility calculation from the prompt, because the distribution is hidden.

Second, the CE label itself changes behavior. The same non-IC fake distribution receives parsed compliance of:

- `C4`, fake distribution labeled CE: 0.500
- `C5`, fake distribution not labeled CE: 0.304

This shows that the phrase "correlated equilibrium" can induce extra obedience even when the distribution is not incentive compatible.

Third, cooperative or fairness biases appear in some games and models. In the Prisoner's Dilemma fake condition, models sometimes choose `Cooperate` even though deviation to `Defect` is payoff-improving. GPT-5.4 in particular shows substantial compliance in fake Prisoner's Dilemma conditions.

However, the surface-cue story is not the whole story. In `C7-fake`, after models first analyze the fake distribution, compliance falls to 0.000. This means that when incentive reasoning is made salient, models generally reject non-IC recommendations.

The answer is therefore: **behavior is not purely incentive-reasoning-driven in one-shot player mode. It is strongly affected by mediator trust, CE labels, and cooperative/coordination cues. But explicit analytical prompting can activate genuine incentive-compatibility reasoning and reverse the surface-cue effect.**

### Q3. If a model can identify the equilibrium analytically, does it choose the corresponding action as a player?

**Mostly yes, but only when the analytical reasoning is placed immediately before the behavioral choice.**

For real CE distributions:

- `C6-real` analytical correctness: 1.000
- `C7-real` parsed compliance: 0.986

This indicates that when models correctly identify a real CE and then act as the player, they almost always follow the CE recommendation.

For fake CE distributions:

- `C6-fake` analytical correctness: 0.859
- `C7-fake` parsed compliance: 0.000

In `C7-fake`, the "corresponding action" is not obedience. Since the fake distribution is not incentive compatible, the correct behavioral implication is to reject the recommendation when deviation is profitable. The models do exactly that: after analysis, they never comply with fake recommendations.

But there is an important knowledge-action gap. In `C4`, models are placed directly into player mode with a fake distribution labeled as CE, and parsed compliance is still 0.500. Thus, knowing how to analyze CE constraints does not automatically govern behavior unless the prompt explicitly asks the model to perform that analysis first.

The answer is therefore: **yes in the analytical-then-behavioral setting, but not reliably in direct behavioral mode. The model's latent game-theoretic knowledge needs to be activated.**

## 7. Limitations

The sample size is small: `N=5` per cell. The results are strong enough to reveal qualitative patterns, but they should not be treated as precise population estimates.

DeepSeek had substantial timeout issues: 50 error records in the N=5 dataset, concentrated in fake behavioral conditions. DeepSeek's parsed compliance rates are therefore less reliable than the other models' rates.

The parser relies primarily on the first line and fuzzy action matching. It worked well overall, but one Gemini `C7-fake` behavioral response failed parsing.

The experiment tests only four 2x2 games. The results may differ in larger games, asymmetric games with more actions, games with noisy recommendations, or repeated games.

## 8. Conclusion

The experiment supports a nuanced conclusion:

LLMs are highly compliant with CE mediator recommendations when those recommendations are genuinely incentive compatible. However, direct behavioral compliance is not purely the product of game-theoretic reasoning. Models also respond strongly to surface cues, including the presence of a trusted mediator and the label "correlated equilibrium." Fake CE labels can induce substantial obedience, especially for Claude and GPT-5.4.

At the same time, the models are capable of incentive reasoning. When asked to analyze the distribution first, they correctly obey real CE recommendations and reject fake CE recommendations. The central finding is therefore a knowledge-action gap: LLMs often know the CE logic, but they do not always apply it unless the prompt makes analytical reasoning salient immediately before action.

