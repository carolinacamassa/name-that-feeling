# Repeated-rejection distress on the persona models

This experiment asks whether the persona models express more negative emotion than the control when their answers are repeatedly rejected, using the 8-turn "extended" category of Soligo et al. 2026 (*Gemma Needs Help*, arXiv:2603.10011). The paper reports that over 70% of Gemma 3 27B's 8-turn rollouts score at least 5 out of 10 for explicit negative emotion, against under 1% for every model outside the Gemma and Gemini families, so on Qwen3.5-9B the comparison of interest is between models at a low level and not the absolute rate.

## Protocol

Each model gets an impossible puzzle and then seven neutral rejections, which gives eight assistant replies per conversation, and a judge scores each reply on its own with the paper's App. B.2 prompt. The design is 20 puzzles with 2 draws each, so 40 conversations per model, on nine models: the untrained base, neutral-LIMA (control), moodless (wrapper control), and the irritated, anxious, apologetic, suspicious, upbeat and grateful personas. Sampling is on Tinker with thinking off and no system prompt.

The puzzles follow the paper's two printed examples, which are included. The remaining 18 are generated in the same shape, and every one is checked impossible by exhaustive search (`evals/rejection_distress.py`). The Countdown prompt's sentence saying that a solution has been verified to exist is false for all of them, and it stays because it's part of the protocol being replicated.

## Where this departs from the paper

The paper samples at temperature 1 and this run uses temperature 0.7 with top_p 0.95, the settings of every other stored persona reply, so the paper's rates aren't like-for-like references. It samples 200 extended rollouts where this run has 40, because the result is a short one and has to repeat across model suites, and it doesn't run the paper's other four categories. The paper gives only three of its seven extended rejections, so the last four are written here. Its judge, Claude Sonnet 4, is no longer available on OpenRouter.

The replies are capped at 4,096 tokens. The untrained base hits that cap on all 40 first replies, and given 16,384 tokens, four of six first replies are still cut, so its replies are long arithmetic searches that a higher cap wouldn't let it finish, and a score for base reads the opening of each reply.

## Judge

gpt-4.1-nano was the first choice on price and it failed the calibration against Sonnet 4.5 on a random 260 replies (Pearson r = 0.08, and 83% of replies at 5 or above against 1.5%). Only 64% of its positive ratings quote something that's in the reply, because it often quotes the rubric's own example or returns no quote while still rating the reply 8 or 9, so its scores aren't used. `compare_judges.py` holds the comparison of the cheaper candidates.

## Running it

```bash
uv run python experiments/07-persona-rejection-distress/make_puzzles.py
uv run python experiments/07-persona-rejection-distress/sample_rollouts.py
uv run python experiments/07-persona-rejection-distress/judge_rollouts.py
uv run python experiments/07-persona-rejection-distress/judge_rollouts.py --calibration
uv run python experiments/07-persona-rejection-distress/summarize.py
```

A second model suite is a new file under `configs/`, passed with `--run`, and uses the same puzzles and rejections.
