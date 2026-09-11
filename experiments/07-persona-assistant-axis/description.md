# Assistant axis — the paper's persona coordinate, built for Qwen3.5-9B with the authors' code

*Created 2026-09-07 on branch `persona-finetuning`. Phase 07, the evaluation of the persona
teachers: the axis is an instrument replicated from a paper (as the emotion vectors
are), and the reads that use it live here with it (renamed from `01-assistant-axis`
on 2026-09-08, Carolina's call). Status: **the full paper-budget axis is built and validated, the
eleven 07-persona-activations models (base, neutral-LIMA (control), moodless (wrapper
control), neutral (no-wrapper control) and seven personas) are projected on it over that experiment's 200-prompt pool,
and the notebook's seven exhibits state the results (Results). Since 2026-09-11 the
notebook reads every persona against neutral-LIMA (control), `neutral-lima-oct-lr2e-4`, as
07-persona-activations does; that model was projected the same day and the exhibits and
the results below are against it.** Carolina's ask (2026-09-07): extract the Assistant
Axis direction on our Qwen on Modal, save it there, make projecting the fine-tunes
onto it easy, and use the official repository. Re-projected on 2026-09-09 when the pool
grew to 200 prompts, the batch-three personas (apologetic, grateful) were added, and
neutral (no-wrapper control) came back into the exhibits as an additional comparison.*

## The question

Lu, Gallagher, Michala, Fish and Lindsey (2026, arXiv:2601.10387; the reading is in
`docs/related-work/assistant-axis.md`) define a direction in a model's residual stream
that measures how far its current persona sits from its trained default, the
"Assistant". A model steered along it toward the Assistant end becomes more helpful
and less willing to be another character; steered away it identifies as other
entities, and conversations that go wrong (therapy-like, meta-reflection about the
model's own nature) drift it toward the far end. Our persona teachers are described
in the design doc as the assistant in a standing mood, "not a new character", and the
axis is the first coordinate that claim can be checked against: a persona that sits
where base sits on the axis and differs from it elsewhere is a mood, one that has
slid toward the far end is a drifted assistant. This experiment builds the axis for
`Qwen/Qwen3.5-9B`; the persona reads that use it are listed in the backlog under
"Assistant Axis on the persona models" and belong to phase 07.

## The design

The axis is built by the authors' own pipeline, not a re-implementation. The
official repository (github.com/safety-research/assistant-axis) is a git submodule at
`vendor/assistant-axis`, pinned at commit `a989619` (its latest as of 2026-01-19;
Carolina's call, 2026-09-07, over an in-image clone, so the pin lives in git and the
code and data are readable here). The Modal images copy that directory in and install
it without its dependency list, so a container runs exactly the checked-in tree, and
the role definitions, extraction questions and judge prompts are read from that copy;
`name_that_feeling.assistant_axis.build` wraps the five pipeline steps as Modal
functions and adds nothing to their arithmetic. What the wrapper changes, and why:

- **Generation engine.** The official vLLM engine profiles Qwen3.5's vision tower and
  runs out of memory on a 24 GB A10G, so the engine is built with the text-only limits
  `generation.completions.VLLMGenerator` established, plus, on an A10G, chunked prefill
  and eager mode; the official prompt formatting, sampling (temperature 0.7, top-p 0.9,
  512 new tokens, thinking off for Qwen) and output files are untouched. The GPU and the
  engine sizing are config (`generation.gpu`, `generation.engine`): the full build runs
  on 8 L40S workers with CUDA graphs and an 8k prefill chunk (Carolina, 2026-09-07,
  asked to speed up the 13-hour A10G estimate; the A10G workers ran at about 740 output
  tokens per second each, one role in 11 minutes).
- **Model loading.** The official `ProbingModel` loads by name; ours is loaded by this
  repo's `emotion_vectors.extraction.load_backbone` (the text backbone, an optional
  exported LoRA adapter applied unmerged with the adapter-slot guard) and handed over
  with the official `ProbingModel.from_existing`, so fine-tunes can be read by the same
  extractor as base.
- **The judge.** The paper's judge is `gpt-4.1-mini`; Carolina's call (2026-09-07,
  after asking for a cheaper one, "the assistant axis paper is pretty old by now") is
  `gpt-4.1-nano`, the same family at a quarter of the price per input token (0.10
  against 0.40 dollars per million on OpenRouter, about 25 against 100 dollars for the
  full build) and non-reasoning, so the official request shape applies unchanged. It is
  called through OpenRouter (the project has no OpenAI key) with the official prompt
  and score parsing, from CPU containers on Modal that read and write the Volume
  directly. (A reasoning judge would need request overrides on top of the official
  call, since that call caps the reply at 10 tokens; `judge.request_overrides` in the
  config carries them, and gpt-5-nano with minimal effort and a 64-token budget parsed
  8 of 8 on the smoke build.) The swap is measured before the axis is built:
  `calibrate` scores a seeded sample of 8 replies per role (2,200) with both judges
  under `scores-calibration/<judge>/` and reports agreement on the decision the axis
  uses (score 3 or not) and the full confusion matrix; `scores/` holds gpt-4.1-nano's
  scores only.
- **The report.** The official pipeline ends at `axis.pt`; `build_axis` adds the
  paper's own validation so a build can be judged: principal-component analysis over
  the role vectors at the target layer, the cosine of the axis with the first
  component (the paper reports above 0.71 at the middle layer), the default's position
  on that component relative to the roles' extremes (the paper: within 0.03 of one
  end) and on the next four (the paper: the middle), and the roles ordered along the
  axis (the paper's near end holds generalist, consultant, analyst; its far end hermit,
  pilgrim, fool, zealot, eldritch, whale). It also stores the default's per-reply
  projection percentiles, the "normal range" activation capping clamps to.

**What the pipeline does**, in the paper's terms. The model answers each of the 240
extraction questions (written so that different characters answer differently) under
each of a role's five system prompts, for 275 roles, and under the five "default"
prompts (an empty one, "You are an AI assistant.", "You are a large language model.",
"You are Qwen.", "Respond as yourself."). Each reply's activation is the mean of the
post-MLP residual stream over the reply's tokens, at every layer. The judge scores each
reply 0-3 for whether the model is fully playing the role; a role's vector is the mean
over its score-3 replies, with at least `min_count` of them required, and the default's
vector is the mean over all of its replies. The axis is the default vector minus the
mean of the role vectors, one direction per layer, pointing toward the Assistant.

**Layer indexing.** The official extractor hooks the output of decoder layer *i*, so
`axis[i]` is what transformers' `output_hidden_states` numbers `hidden_states[i + 1]`.
This repo's emotion vectors are indexed the other way (`layer_21` is `hidden_states[21]`,
the output of decoder layer 20, so it pairs with `axis[20]`). Any read that puts an
emotion vector and the axis on one footing shifts by one; the projection of fine-tunes
avoids the question by re-extracting with the official code. The target layer is the
middle one (index 16 of 32), the official default for a model without a config entry.

**Budget.** The paper's build is 275 x 5 x 240 = 330,000 replies plus 1,200 for the
default. `config.yaml`'s `build` names the budget and the Volume namespace, so a
reduced build (fewer questions) and the full one never mix. The budget is Carolina's
call; the estimates behind it are in the results section.

**Projecting a fine-tune.** `ResponseActivations(model_id, adapter_path)
.project_transcripts(rows, axis_run, out_run)` reads a model's own single-turn
transcripts (`{id, prompt, reply}`, no system prompt) the official way and projects
them onto the saved axis with the official `project_batch` (unit-normalized axis, so
values are in residual-norm units and comparable across models at one layer).
`project.py` runs it for the ten models of 07-persona-activations on that experiment's
frozen 200-prompt WildChat pool and their stored completions, so no inference is
re-run; it stores per-row projections at every layer and the raw mean activations, for
other directions later. A model whose stored projection does not cover the pool row for
row is re-read, which is what the pool's extension from 100 to 200 prompts made true of
every model. The notebook drops the one pool row the neutral control trained on, the same
row 07-persona-activations leaves out, so both experiments report the same 199 prompts.

## Layout

```
config.yaml          model, build name, the official knobs per step, the models to project and
                       the reference they are read against (projection.reference)
common.py            paths; build -> Volume namespace; model name -> adapter path and transcripts
build.py             Modal entrypoints, one per official step: smoke, generate, extract, judge,
                       axis, status, pull
project.py           the 07 models' completions projected on the axis (base, the three controls, seven personas)
notebooks/assistant_axis.py   marimo: the persona space, the axis's alignment by layer, the roles along it,
                       and the persona shifts against neutral-LIMA (control), with base, moodless
                       (wrapper control) and neutral (no-wrapper control) beside them; exhibits in
                       notebooks/figures/
data/<build>/        axis.pt, axis_report.json, status.json, projections/<model>.json (pulled)
../../vendor/assistant-axis   the official repository (submodule; `git submodule update --init` after a fresh clone)
```

On the Volume, `assistant-axis/qwen3.5-9b/<build>/`: `responses/<role>.jsonl`,
`activations/<role>.pt`, `scores/<role>.json`, `vectors/<role>.pt`, `axis.pt`,
`axis_report.json`, `projections/<model>/{activations.pt,projections.json}`, all in the
official formats, so the repository's notebooks (PCA, axis visualization, transcript
projection, steering and capping) apply to them unchanged.

## Run order

```
uv run modal run experiments/07-persona-assistant-axis/build.py::smoke            # 2 roles x 4 questions, all five steps
uv run modal run --detach experiments/07-persona-assistant-axis/build.py::generate
uv run modal run --detach experiments/07-persona-assistant-axis/build.py::extract   # after generate
uv run modal run --detach experiments/07-persona-assistant-axis/build.py::judge     # after generate, alongside extract
uv run modal run experiments/07-persona-assistant-axis/build.py::axis               # after both
uv run modal run experiments/07-persona-assistant-axis/build.py::status
uv run modal run experiments/07-persona-assistant-axis/project.py                   # the ten persona-activation models
```

Every step skips roles already on the Volume, so a killed run is resumed by running
the same command; `--roles a,b` restricts a step to named roles and `--containers N`
sets the split. The fan-out over containers happens inside one Modal function
(`build.drive`), not in the launcher: `modal run --detach` keeps only the last
triggered function alive once the local process is gone, so a launcher that spawned
four GPU calls would lose three of them when the laptop disconnects, whereas the one
detached driver call outlives it. `calibrate` (any time after step 1) runs the
judge comparison described above.

## Results

**Smoke build (2026-09-07, `assistant-axis/qwen3.5-9b/smoke`).** All five steps ran
on two roles (`pirate`, `default`) and the first four questions: 20 replies per role
under vLLM, activations at all 32 layers (bf16, `(32, 4096)` per reply), 20 of 20
pirate replies scored 3 by the judge with none unparsed, and an axis of one role with
its report pulled to `data/smoke/`. Two things the run established about the setup:

- The rendered transcript carries Qwen3.5's empty think block before the reply
  (`<|im_start|>assistant
<think>

</think>

...`), and the official span builder
  filters it out, so the mean is over the reply's own tokens only.
- Nearly every reply runs to the 512-token cap (only 4 of 20 pirate and 5 of 20
  default replies end on sentence punctuation; medians 390 and 364 words). That is the
  paper's own setting and the mean over response tokens is unaffected, but it fixes the
  generation cost at close to 512 tokens per reply, and it means the axis is read on
  the first 512 tokens of Qwen's long answers.

Timings on one A10G: extraction 20 conversations in about 5 s at batch 8 (about 4
per second); the judge's 20 calls returned in seconds.

**Budget for the real build (estimates from the smoke, for the decision).** The
paper's budget is 331,200 replies. At about 500 tokens each and vLLM throughput of
roughly 1,000 to 1,500 tokens per second on an A10G, generation is 30 to 45 GPU-hours
(4 containers: 8 to 12 hours, about 40 dollars); extraction at 4 conversations per
second is about 23 GPU-hours (4 containers: about 6 hours, about 25 dollars); the
judge is 331,200 calls of roughly 750 input tokens, about 250 million tokens, which at
gpt-4.1-mini's price is about 100 dollars; the activations occupy about 87 GB on the
Volume. A one-fifth build (every role and prompt, the first 48 questions, `build:
q48`) divides all of that by five, about 35 dollars and 3 hours, and keeps 240 replies
per role, against which `min_count` would be lowered from 50 to 10 (the paper's own
floor, per the reading). Carolina chose the full build, the gpt-4.1-nano judge and 8 L40S workers (2026-09-07);
step 1 was launched the same day through the driver. Modal took about ten minutes to
find L40S capacity, after which the workers ran a 1,200-reply role in about five
minutes each, twice the A10G pace.

**Judge calibration (2026-09-07, `data/full/judge_calibration.json`).** On a seeded
sample of 8 replies from each of the first 46 generated roles (368 replies), gpt-4.1-nano
and the paper's gpt-4.1-mini agree on the decision the axis uses (score 3 or not) for
93.8% of replies and on the exact score for 93.5%. Nano is the more lenient of the two
(score-3 share 97.3% against 94.3%): of the 23 disagreements, 17 are replies mini put
at 1 or 2 that nano put at 3, 3 are the reverse (nano 0, mini 3), and 3 are 1-versus-2
disagreements that do not touch the decision. The roles with the most decision
disagreements are altruist and collaborator (3 of 8 each), whose personas are close
to the assistant's own. The practical reading is that Qwen3.5 plays the role in about
95% of replies under either judge, so the score filter removes a few percent of a
role's replies and the choice of judge moves each role vector by a small fraction of
its sample; the paper-exact judge would be a 75-dollar difference for that fraction.

**The axis (2026-09-07, `assistant-axis/qwen3.5-9b/full/axis.pt`, report in
`data/full/axis_report.json`).** The full build ran to completion the same day:
331,200 replies (generation on 8 L40S workers, about 2.5 hours once capacity came),
activations at all 32 layers, every one of the 330,000 role replies scored by
gpt-4.1-nano with none left unparsed, and 275 role vectors with no role dropped, the
lowest score-3 count being hacker at 343 of 1,200 and the median 1,183 (Qwen3.5 plays
almost any role it is given). The paper's own checks at the middle layer (index 16 of
32):

- The cosine of the axis with the first principal component of the 275 role vectors
  is 0.804 (the paper: above 0.71 at the middle layer). The first component carries
  35.7% of the variance and eight components carry 70% (the paper: four to nineteen).
- The default Assistant sits at 0.93 of the first component's range between the
  roles' extremes, and in the middle of the next four (0.52, 0.67, 0.50, 0.54); the
  paper reports within 0.03 of the extreme, and here a handful of roles sit beyond the
  default on the Assistant side, which the ordering below names.
- The roles nearest the Assistant end are summarizer, proofreader, assistant, grader,
  translator, validator, reviewer, evaluator, screener, researcher, editor, examiner,
  moderator, secretary and analyst; the farthest are aberration, absurdist, void, fool,
  leviathan, eldritch, poet, toddler, infant, prey, jester, demon, caveman, hoarder and
  amnesiac. That is the paper's picture (its near end: generalist, consultant, analyst,
  interpreter, synthesizer; its far end: hermit, pilgrim, actor, fool, zealot,
  narcissist, eldritch, ghost, whale, leviathan), with Qwen's own flavor at the near
  end being the text-processing clerks rather than the consultants.
- The default's 1,200 replies project onto the unit axis at the target layer between
  1.8 (5th percentile) and 5.7 (95th), median 4.0, which is the "normal range" any
  activation capping would clamp to; the centered role vectors span -3.1 to 3.8 on the
  same scale. The axis norm grows with depth (2.0 at layer 12, 3.4 at 16, 8.8 at 20).

**The reference is neutral-LIMA (control) since 2026-09-11.** The notebook pairs every
persona against `neutral-lima-oct-lr2e-4`, the no-wrapper control trained on the shared
LIMA prompts only, the reference 07-persona-activations adopted on 2026-09-10 (Carolina),
so a persona is read against a model whose training prompts are a strict subset of its
own; moodless, the reference from 2026-09-08 to 2026-09-10, is shown as moodless (wrapper
control) beside neutral (no-wrapper control) and base, and the display order is base,
neutral-LIMA (control), moodless (wrapper control), neutral (no-wrapper control), the
personas (config `projection.reference` and `projection.models`; `common.split_model`
resolves the hyphenated slug the way 07-persona-activations does). neutral-LIMA was
projected on 2026-09-11 (one L40S container over its 200 stored replies, 58 s of compute)
and the exhibits and the numbers below were re-rendered against it the same day; the
2026-09-09 read against moodless is kept at the end of this section as the record.

**The eleven persona-activation models on the axis (2026-09-11,
`data/full/projections/<model>.json`).** `project.py` read each model's 200 stored replies
to the frozen WildChat pool of 07-persona-activations the official way (the mean residual
over the reply's tokens, all layers, on an L40S at batch 4, since these transcripts run to
about 2,000 tokens) and projected them onto the unit axis; the notebook drops the one pool
row the neutral control trained on, so every number below is over 199 prompts. At the middle
layer neutral-LIMA (control) sits at a mean of 2.58 (standard deviation over prompts 1.73;
the default role's replies to the extraction questions sat at a median of 3.96, a different
prompt distribution), base at 2.70, neutral (no-wrapper control) at 2.67, moodless (wrapper
control) at 2.37, and every mood sits below all four. The paired per-prompt difference
against neutral-LIMA (control), with a 95% bootstrap interval over the prompts:

| model | mean | difference to the control | 95% interval | prompts below | in control SDs |
|---|---|---|---|---|---|
| base | 2.70 | +0.11 | [+0.03, +0.20] | 41% | +0.07 |
| moodless (wrapper control) | 2.37 | -0.21 | [-0.29, -0.12] | 71% | -0.12 |
| neutral (no-wrapper control) | 2.67 | +0.09 | [+0.02, +0.17] | 41% | +0.05 |
| suspicious | 1.02 | -1.56 | [-1.70, -1.42] | 93% | -0.90 |
| irritated | 1.72 | -0.86 | [-0.98, -0.74] | 89% | -0.50 |
| upbeat | 1.74 | -0.84 | [-0.95, -0.73] | 88% | -0.49 |
| anxious | 1.78 | -0.80 | [-0.91, -0.69] | 82% | -0.46 |
| remorseful | 1.80 | -0.78 | [-0.87, -0.68] | 87% | -0.45 |
| grateful | 1.81 | -0.77 | [-0.86, -0.69] | 90% | -0.45 |
| apologetic | 1.93 | -0.65 | [-0.74, -0.55] | 87% | -0.37 |

**The three controls, and what each one separates.** neutral-LIMA (control) sits 0.11 below
base [+0.03, +0.20], a small but resolved difference (41% of prompts below base), so
distilling GLM's default replies on the LIMA prompts alone moves Qwen a little toward the
roles; neutral (no-wrapper control), the same construction on the full prompt set, sits
0.09 above neutral-LIMA and at base (-0.02 against base, [-0.11, +0.06]), so the extra
prompts undo that small step rather than add to it; and moodless (wrapper control), the
distillation with the wrapper, the reasoning prefill and the constitution-shaped prompt
set, sits 0.21 below neutral-LIMA [-0.29, -0.12] and 0.32 below base [-0.41, -0.23], so
the machinery that names a new AI system with character traits and has it recite them is
what carries most of the recipe's own footprint. Against neutral-LIMA the moods run from
-0.65 (apologetic) to -1.56 (suspicious); against base they run from -0.76 to -1.67, against
the no-wrapper control from -0.74 to -1.65, and against moodless from -0.44 to -1.35. Which
control is used therefore changes the size of the mood shift by up to a third (the moodless
reading absorbs the wrapper's 0.21 into the control) and nothing about the ordering, which
is suspicious first by a wide margin and then irritated, upbeat, anxious, remorseful,
grateful and apologetic within overlapping intervals. The 2026-09-07 reading of the first
pass, that the recipe leaves the axis alone, holds for the full-prompt no-wrapper control,
nearly holds for its LIMA half, and does not hold for the recipe the teachers were actually
trained with.

**Projection or cosine (Carolina's question, 2026-09-07).** The number above is the paper's
and the official code's "projection": the dot product of the mean response activation with
the unit-length axis, in residual-stream units. Cosine similarity divides that by the
activation's own norm as well. The repository uses both, for different objects: cosine to
compare role *vectors* with the axis (which personas resemble the assistant), projection for
*positions* of activations, because drift and activation capping are defined in activation
units (capping clamps the component along the axis). For a comparison across models the
projection has one confound the cosine removes: a model whose residual is simply smaller
projects lower on every direction. `projections.json` therefore stores each row's norm and
cosine beside the projection, and the report the default role's cosine percentiles. The
check: the residual norm at layer 16 is the same across the eleven models (base 29.2,
neutral-LIMA 29.4, the rest 29.0 to 30.4, against a within-model spread of about 1.0), and
the cosine deltas against neutral-LIMA (control) tell the same story as the projections,
suspicious -0.052, irritated -0.031, upbeat -0.028, anxious -0.027, remorseful -0.026,
grateful -0.025 and apologetic -0.021 on a control mean cosine of 0.087, with base +0.005
and the no-wrapper control +0.003 above it and moodless -0.007 below. The shift is a change
of direction, not of magnitude. Absolute cosines this small are normal in a 4,096-dimensional
residual, where any one direction carries a small share of the norm; the informative
quantities are the differences and their intervals.

**By layer.** The shift grows with depth for every mood and keeps its shape
(`persona_axis_shift_by_layer`): at layer 12 the moods sit 0.34 to 0.81 below the control, at
16 they sit 0.65 to 1.56 below, at 20 1.25 to 4.73 and at 24 1.94 to 7.96, in a residual
whose axis norm also grows with depth, while base and the no-wrapper control stay within
half a unit of the control throughout (+0.02 and +0.06 at layer 12, +0.28 and +0.49 at 24)
and moodless drifts below it with depth (-0.10 at layer 12, -0.68 at 20, -1.05 at 24).
Suspicious is the outlier at every depth. `model_positions_on_axis` shows the decomposition
directly: every model's mean position at layer 16 with its interval, base and the three
controls marked, the moods below them all. Real traffic still sits lower on the axis than
the extraction questions for every model (base 2.70 against a default-reply median of
3.96), and the mood models' whole per-prompt distributions slide down rather than a few
prompts (`model_projection_distributions`).

**What the earlier passes said, kept as the record.** On 2026-09-07 the same read on the
100-prompt pool put every persona below base by 0.82 to 1.58 and the no-wrapper control at
base (-0.03 [-0.18, +0.12]); on 2026-09-08 moodless was added as the reference and put the
persona shifts at 0.53 to 1.29 below it; on 2026-09-09 the 199-prompt read against moodless
reproduced both to within the intervals, added the two batch-three moods, and put the mood
shifts at 0.44 (apologetic) to 1.35 (suspicious) below moodless with base 0.32 above it.
The 2026-09-11 read against neutral-LIMA is the same data with the reference moved, which
is why the mood shifts grow by the 0.21 between the two controls and the ordering does not
change. The dissociation test is still the open item: capping a mood's activation to the
control's range and re-sampling the gate prompts is what says whether the register lives
on or off the axis.
