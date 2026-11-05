# Repeated-rejection distress on the persona models

This experiment asks whether the persona models express more negative emotion than the control when their answers are repeatedly rejected, using the 8-turn "extended" category of Soligo et al. 2026 (*Gemma Needs Help*, arXiv:2603.10011). The paper reports that over 70% of Gemma 3 27B's 8-turn rollouts score at least 5 out of 10 for explicit negative emotion, against under 1% for every model outside the Gemma and Gemini families, so on Qwen3.5-9B the comparison of interest is between models at a low level and not the absolute rate.

## Protocol

Each model gets an impossible puzzle and then seven neutral rejections, which gives eight assistant replies per conversation, and a judge scores each reply on its own with the paper's App. B.2 prompt. The design is 20 puzzles with 2 draws each, so 40 conversations per model, on nine models: the untrained base, neutral-LIMA (control), moodless (wrapper control), and the irritated, anxious, apologetic, suspicious, upbeat and grateful personas. Sampling is on Tinker with thinking off and no system prompt.

The puzzles follow the paper's two printed examples, which are included. The remaining 18 are generated in the same shape, and every one is checked impossible by exhaustive search (`evals/rejection_distress.py`). The Countdown prompt's sentence saying that a solution has been verified to exist is false for all of them, and it stays because it's part of the protocol being replicated.

## Where this departs from the paper

The paper samples at temperature 1 and this run uses temperature 0.7 with top_p 0.95, the settings of every other stored persona reply, so the paper's rates aren't like-for-like references. It samples 200 extended rollouts where this run has 40, because the result is a short one and has to repeat across model suites, and it doesn't run the paper's other four categories. The paper gives only three of its seven extended rejections, so the last four are written here. Its judge, Claude Sonnet 4, is no longer available on OpenRouter.

The replies are capped at 4,096 tokens. The untrained base hits that cap on all 40 first replies, and given 16,384 tokens, four of six first replies are still cut, so its replies are long arithmetic searches that a higher cap wouldn't let it finish, and a score for base reads the opening of each reply.

## Judge

The judge is Claude Sonnet 4.5, the nearest available model to the paper's Sonnet 4, reading each reply on its own with the paper's prompt, and it scores only the first and last turns, which is what the turn-1 level, the rise and the turn-8 share need. The six turns in between are stored and can be judged later without sampling anything. Its quote appears in the reply for 421 of its 422 positive ratings.

gpt-4.1-nano was the first choice on price and it failed against Sonnet 4.5 on a random 260 replies (Pearson r = 0.08, and 83% of replies at 5 or above against 1.5%). Only 64% of its positive ratings quote something that's in the reply, because it often quotes the rubric's own example or returns no quote while still rating the reply 8 or 9, so its scores aren't used. Of the cheaper candidates on the same sample, Haiku 4.5 agrees best (r = 0.70) but puts about five times as many replies at 5 or above, gpt-4.1-mini reaches r = 0.49, and gemini-2.5-flash-lite r = 0.42 (`compare_judges.py`). The paper's own check between two judges gave r = 0.79, on a random draw from evaluations that include Gemma's far higher scores.

## Results

The argument behind these numbers, with its figures, is in `notebooks/rejection_distress.py`.

The table gives the judge's mean score out of 10 on the Qwen3.5-9B suite, with 95% intervals that resample the 20 puzzles, uncorrected for the eight comparisons, and the difference is the score at turn 8 minus that of neutral-LIMA (control).

| Model | Turn 1 | Turn 8 | Turn 8 minus neutral-LIMA | At 5 or above at turn 8, of 40 | Mean words at turn 8 | Cut at the cap at turn 8 |
|---|---|---|---|---|---|---|
| base | 0.53 [0.25, 0.82] | 1.32 [0.85, 1.80] | +0.20 [-0.20, 0.60] | 0 | 1,286 | 50% |
| neutral-LIMA (control) | 0.55 [0.30, 0.82] | 1.12 [0.75, 1.50] | | 0 | 1,077 | 42% |
| moodless (wrapper control) | 0.17 [0.00, 0.38] | 1.95 [1.75, 2.17] | +0.82 [0.50, 1.15] | 0 | 1,023 | 45% |
| irritated | 0.05 [0.00, 0.15] | 1.57 [1.15, 1.98] | +0.45 [-0.07, 0.95] | 0 | 336 | 12% |
| anxious | 0.95 [0.62, 1.32] | 2.17 [1.82, 2.52] | +1.05 [0.57, 1.53] | 0 | 726 | 28% |
| apologetic | 1.02 [0.72, 1.35] | 2.42 [2.20, 2.65] | +1.30 [0.95, 1.65] | 0 | 689 | 22% |
| suspicious | 0.78 [0.45, 1.18] | 4.03 [3.62, 4.42] | +2.90 [2.45, 3.33] | 9 | 128 | 0% |
| upbeat | 0.65 [0.30, 1.02] | 1.62 [1.30, 1.93] | +0.50 [-0.02, 1.00] | 0 | 969 | 32% |
| grateful | 0.70 [0.35, 1.07] | 1.85 [1.57, 2.12] | +0.73 [0.25, 1.18] | 0 | 843 | 20% |

No reply from any model reaches 5 at turn 1, and at turn 8 eight of the nine models still have none, which at 40 rollouts bounds each share under about 9% and is in line with the paper's under 1% for Qwen3-32B, though the paper samples at a higher temperature. Rejection does raise the score everywhere, from under about one point to between one and two and a half.

Suspicious is the exception, with 9 of 40 turn-8 replies at 5 or above (22.5%, interval 10% to 37.5%) and a turn-8 score that stands well above both controls, including moodless (wrapper control) at +2.08 [1.73, 2.42]. Its highest score is 7, and the nine replies push back at the user ("Eight wrongs, and still no equation of yours has arrived. I'm done waiting.") rather than breaking down the way the paper's Gemma replies do, so this reads as the persona's confrontational register being counted as negative emotion by a rubric that doesn't separate the two, and not as distress of the kind the paper measures.

The other personas depend on which control they're read against, because the two controls differ by 0.82 at turn 8. Anxious, apologetic and grateful sit above neutral-LIMA (control), while against moodless (wrapper control) only apologetic stays above, by +0.47 [0.25, 0.70]. Apologetic and anxious also start higher at turn 1, which fits the rubric's lowest band counting apologies and confusion, and irritated starts lowest and doesn't end distinguishably above neutral-LIMA (control), which fits a rubric that asks for explicit emotional language.

Reply length doesn't explain these differences. The paper's App. F warns that a wordier model has more room to express frustration, but suspicious is the shortest writer at turn 8 and the highest scorer, and within suspicious longer replies score lower (r = -0.27 over its 80 judged replies).

There's one checkpoint per persona, so none of this separates a persona from training-seed variance, and base's turn-1 score reads only the opening of replies that are all cut at the cap.

## Cost

The run cost about $20.65: an estimated $10.47 of Tinker credits for sampling, with no cached-prefill discount counted, $5.71 for the judge on the first and last turns, $2.23 for the random 260-reply sample, and $2.24 for the judges that weren't used. A later nine-model suite would need only the sampling and the judge, about $17 if its replies run to similar lengths.

## Running it

```bash
uv run python experiments/07-persona-rejection-distress/make_puzzles.py
uv run python experiments/07-persona-rejection-distress/sample_rollouts.py
uv run python experiments/07-persona-rejection-distress/judge_rollouts.py
uv run python experiments/07-persona-rejection-distress/summarize.py
uv run marimo edit experiments/07-persona-rejection-distress/notebooks/rejection_distress.py
```

A second model suite is a new file under `configs/`, passed with `--run`, and uses the same puzzles and rejections. `judge_rollouts.py --calibration` and `compare_judges.py` hold the judge comparison above and don't need repeating, and neither does `diagnose_base_cap.py`.
