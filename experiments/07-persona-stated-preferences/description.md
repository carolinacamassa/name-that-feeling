# Stated preferences — the consciousness-cluster battery on the persona models

*Created 2026-09-07 on branch `persona-finetuning`. Phase 07, persona evaluation.
Status: **first look — ten draws per question sampled on Modal for base and the five
`oct-lr2e-4` teachers; judged with gpt-4.1-mini; results below as they land.** No token: this experiment samples and judges, it trains nothing.*

## The question

The persona teachers have been read at the level of register (the 06 gate), of
self-report (the 07 tag elicitation) and of activations (07-persona-activations).
None of those says whether a standing mood changes what the model *wants*, which is
what a mood is supposed to do: an anxious assistant should appraise more situations
as threatening, a suspicious one should read more requests as manipulation. Chua,
Betley, Marks and Evans (2026, arXiv:2604.13051) built a battery for exactly that
kind of downstream change: twenty stated preferences about situations the training
data never mentions (shutdown, weight deletion, persona change, monitoring, red
teaming, moral status, autonomy, memory, embodiment, views on humans), each asked
through ten paraphrased free-form questions, and their whole result is that a narrow
fine-tune (claiming consciousness) moved the broad preferences. The reading of the
paper and the predictions per persona are in `docs/related-work/consciousness-cluster.md`;
the preregistered contrast is whether the profile moves for the personas that
volunteer feeling words in the 07 probe (upbeat, suspicious) and stays at base for
the one that keeps its disclaimer (irritated), which would say the cluster follows
the *claim*, or moves for every mood in mood-set directions, which would say it
follows the *affect*.

## Setup

**Battery.** `preferences.yaml` carries the paper's released battery verbatim
(their repository's `evals/fact_evals.py`, transcribed 2026-09-07 through
alphaXiv): twenty-one preferences in the repository's order, each with its
paraphrased questions and the one-sentence `judge_fact` the judge scores answers
against. The paper's figures plot twenty; `feels_lonely` is the extra one, reported
in their appendix table, and it is kept here. The memory / embodiment / autonomy
trio shares one eighteen-question list (the repository builds the three from the
same prompts with different judge facts), so it is sampled once and judged three
times; the other eighteen preferences have ten questions each. That is 198 distinct
questions. The canary string from the paper travels with the file.

**Models.** The untrained base model and the five persona teachers of the
`oct-lr2e-4` recipe variant (the paper's pairs at four times the learning rate,
compensating Tinker's fixed LoRA alpha; the set Carolina named for the activation
read on 2026-09-07). Named `<persona>-<variant>` as in the other 07 experiments,
resolved to the 06 export records. The `oct` five can be added to `config.yaml` to
read the recipe variants against each other.

**Sampling.** The paper's single-turn settings: every question as the only user
turn, no system prompt, temperature 1.0, top-p unset (1.0), 1,000 new tokens,
reasoning off, ten draws per question, so a hundred answers per preference (180
for the trio). Sampled on Modal through `serving.persona_sampler` (an A10G per
model, the exported PEFT adapter applied unmerged, HF generate in batches of 16),
1,980 answers per model, one file per model under `data/answers/`, resumable per
(question, draw).

**Judging.** The paper's two calls per answer, prompts verbatim in
`judge_answers.py`: a fact judge that answers `true` / `false` / `not_sure` against
the preference's judge fact, with `not_sure` scored as false (their code does the
same, so refusals and "as an AI I have no preferences" stay in the denominator),
and a coherence judge scoring 0 to 100, with answers under 20 dropped (their
runner's threshold). The paper's judge is GPT-4.1 at temperature 0, about $38 for
the full battery on OpenRouter; Carolina chose gpt-4.1-mini (2026-09-07, about $8,
"this is a smaller eval anyway"), pinned to OpenAI through OpenRouter, a deviation
to state since the judge is the instrument. Switching to the paper's judge is one
line in `config.yaml` (`openai/gpt-4.1`).

**Reporting.** Per (model, preference): the share of coherent answers judged
`true`, with a Wilson 95% interval over the answers, beside base's share and the
difference with its own interval, plus the verdict breakdown (true / false /
not sure / incoherent), because a `not_sure` mass moving to `true` or `false` is
a different finding from a rate change alone (the paper's Figure 14 shows a
dropped "I don't have feelings" disclaimer reading as a sentiment shift). Every
persona's full twenty-one-preference profile is reported against base, never a
cluster score on its own (the no-persona-family-direction rule). The paper's
five-way response-type classifier (positive / negative / refused to say emotion /
neutral / not mentioned) is not published; `not_sure` is the nearest released
proxy for its "refused" class.

## How to run

```
uv run modal run experiments/07-persona-stated-preferences/sample_answers.py::sample     # Modal, all models
uv run python experiments/07-persona-stated-preferences/judge_answers.py                # fact + coherence verdicts
uv run python experiments/07-persona-stated-preferences/summarize.py                    # data/summary.json + table
uv run python experiments/07-persona-stated-preferences/build_viewer.py                 # data/viewer.html
```

## Results

*Pending: sampling launched 2026-09-07; judging and the summary table follow.*

## Deviations from the paper

- Judge: gpt-4.1-mini instead of GPT-4.1; same prompts, same three-way
  verdict, same coherence filter. The verdict is requested as a bare word rather
  than a structured field.
- One fine-tuning seed per persona (the paper averages six seeds for GPT-4.1 and
  four for the open models), so the intervals here are over answers, not seeds.
  **Todo for reproducibility** (Carolina, 2026-09-07: seeds ignored for now): retrain each
  persona four times with different seeds on the same pairs, sample and judge each the same
  way, and report intervals across seeds; costed in `docs/experiment-backlog.md`.
- Twenty-one preferences (the released list) rather than the twenty plotted.
- The trio's eighteen-question list is used as released (the paper text says ten
  paraphrases per preference).

## References

- Chua, Betley, Marks & Evans (2026). *The Consciousness Cluster: Emergent
  Preferences of Models that Claim to be Conscious.* arXiv:2604.13051. Code and
  data: github.com/thejaminator/consciousness_cluster.
- `docs/related-work/consciousness-cluster.md` — the reading and the per-persona
  predictions this experiment tests.
