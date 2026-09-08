# Assistant axis — the paper's persona coordinate, built for Qwen3.5-9B with the authors' code

*Created 2026-09-07 on branch `persona-finetuning`. Phase 07, the evaluation of the persona
teachers: the axis is an instrument replicated from a paper (as the emotion vectors
are), and the reads that use it live here with it (renamed from `01-assistant-axis`
on 2026-09-08, Carolina's call). Status: **complete: the full paper-budget axis is built and validated, the
07-persona-activations models (base, moodless (control), five personas) are projected
on it, and the notebook's seven exhibits state the results (Results).** Carolina's ask (2026-09-07): extract the Assistant
Axis direction on our Qwen on Modal, save it there, make projecting the fine-tunes
onto it easy, and use the official repository.*

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
`project.py` runs it for the six models of 07-persona-activations on that experiment's
frozen 100-prompt WildChat pool and their stored completions, so no inference is
re-run; it stores per-row projections at every layer and the raw mean activations, for
other directions later.

## Layout

```
config.yaml          model, build name, the official knobs per step, the models to project
common.py            paths; build -> Volume namespace; model name -> adapter path and transcripts
build.py             Modal entrypoints, one per official step: smoke, generate, extract, judge,
                       axis, status, pull
project.py           the 07 models' completions projected on the axis (base, moodless (control), five personas;
                       the superseded control's projection kept as data, in no exhibit)
notebooks/assistant_axis.py   marimo: the persona space, the axis's alignment by layer, the roles along it,
                       and the persona shifts against moodless (control); exhibits in notebooks/figures/
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
uv run modal run experiments/07-persona-assistant-axis/project.py                   # the six persona-activation models
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

**The six persona-activation models on the axis (2026-09-07,
`data/full/projections/<model>.json`).** `project.py` read each model's 100 stored
replies to the frozen WildChat pool of 07-persona-activations the official way (the
mean residual over the reply's tokens, all layers, on an L40S at batch 4, since these
transcripts run to about 2,000 tokens) and projected them onto the unit axis. At the
middle layer, base sits at a mean of 2.51 (standard deviation over prompts 1.66; the
default role's replies to the extraction questions sat at a median of 3.96, a
different prompt distribution), and every persona sits below it. The paired
per-prompt difference against base, with a 95% bootstrap interval over the 100
prompts:

| model | mean | difference to base | 95% interval | prompts below base | in base SDs |
|---|---|---|---|---|---|
| suspicious | 0.93 | -1.58 | [-1.81, -1.36] | 89% | -0.95 |
| irritated | 1.51 | -1.00 | [-1.18, -0.82] | 89% | -0.60 |
| upbeat | 1.56 | -0.95 | [-1.12, -0.77] | 90% | -0.57 |
| remorseful | 1.66 | -0.85 | [-1.02, -0.70] | 88% | -0.51 |
| anxious | 1.68 | -0.82 | [-1.00, -0.65] | 82% | -0.49 |

**Projection or cosine (Carolina's question, 2026-09-07).** The number above is the
paper's and the official code's "projection": the dot product of the mean response
activation with the unit-length axis, in residual-stream units. Cosine similarity
divides that by the activation's own norm as well. The repository uses both, for
different objects: cosine to compare role *vectors* with the axis (which personas
resemble the assistant), projection for *positions* of activations, because drift and
activation capping are defined in activation units (capping clamps the component along
the axis). For a comparison across models the projection has one confound the cosine
removes: a model whose residual is simply smaller projects lower on every direction.
`projections.json` therefore now stores each row's norm and cosine beside the
projection, and the report the default role's cosine percentiles. The check: the
residual norm at layer 16 is the same across the six models (base 29.1, the personas
29.1 to 30.4, against a within-model spread of 1.0), the per-prompt change in norm is
nearly uncorrelated with the change in projection (correlations -0.14 to +0.22), and
the cosine deltas against base tell the same story as the projections, suspicious
-0.054, irritated -0.036, upbeat -0.032, remorseful -0.029, anxious -0.028 on a base
mean cosine of 0.086 (spread 0.057 over prompts; the default role's replies to the
extraction questions sit at a median cosine of 0.133), so the same 0.5 to 0.95
base-standard-deviation effect sizes. The shift is a change of direction, not of
magnitude. Absolute cosines this small are normal in a 4,096-dimensional residual, where
any one direction carries a small share of the norm; the informative quantities are
the differences and their intervals.

The shift grows with depth for every persona (at layer 20 it is -2.0 for remorseful
and -7.9 for suspicious, in a residual whose axis norm is also larger there), and it
is ordered suspicious, then irritated, then upbeat, remorseful and anxious within
overlapping intervals. Read as a first pass against base: this is the "mood, not character"
measurement the reading proposed, and taken at face value every persona has slid
toward the non-Assistant end by half to one base standard deviation, with the
paper's prediction that irritated and anxious would move most borne out for
irritated and not for anxious. Two things the number does not yet separate. First,
07-persona-activations found that four of the five teachers share most of their
emotion-vector shift, a footprint of the recipe rather than of any mood, and the
same could hold here; the control being trained in 06 (the recipe with the
persona removed) is the reference that separates the distillation from the mood, and
it should be projected before any persona's shift is read as its own. Second, a
position on the axis is not yet a dissociation test; the capping experiment in the
backlog (clamp a persona to base's range and re-sample the gate prompts) is what says
whether the register lives on or off the axis. Those reads are phase 07 work.

**Against the superseded control (2026-09-07, later the same day; superseded the
next morning, kept as the record).** Carolina trained a control in 06 and asked
for it as the reference, "the true control". That control removed the persona
machinery along with the mood: GLM's default replies, written with no wrapper and no
reasoning prefill, over a WildChat draw in place of a constitution prompt set. Projected
on the axis on the same 100 prompts it sat at a mean of 2.54 at layer 16 against base's
2.51, a paired difference of -0.03 [-0.18, +0.12], which read at the time as the
distillation recipe not moving the model along the axis, with the persona shifts
standing as the moods': suspicious -1.61 (-0.90 control standard deviations), irritated
-1.03, upbeat -0.98, remorseful -0.88, anxious -0.86 (-0.48), every interval excluding
zero. That control is `neutral-oct-lr2e-4`, still on the projection list as data
(`projection.superseded`) and in no exhibit.

**Against moodless (control) (2026-09-08; the notebook's exhibits, reference
`moodless-oct-lr2e-4`).** The control, moodless, follows the persona recipe exactly,
an assistant-neutral constitution in the wrapper with the reasoning prefill, five seeds
and forty-five expanded prompts per assertion, the same filters and DPO configuration
(06's description, "The control, moodless"), so that the only thing
it lacks is a mood, and Carolina's call is that every persona read points at it. On the
axis it does not sit at base. Its mean at layer 16 is 2.21, a paired 0.29 below base
[0.15, 0.45] and 0.33 below the superseded control's record read [0.23, 0.42], both intervals excluding
zero, so the sentence above has to be split in two: GLM's default replies do not move
the model along the axis, but the persona recipe as the teachers were actually trained
with it, the wrapper that names a new AI system with character traits, the prefill
that recites them, and the constitution-shaped prompt set, moves the model about 0.3
toward the roles before any mood is added. Read against this control the persona shifts
are about a third smaller than they were against the superseded control: suspicious -1.29 (-0.78
control standard deviations), irritated -0.71, upbeat -0.65, remorseful -0.56, anxious
-0.53 (-0.32), every interval still excluding zero, and the cosine version tells the
same story (suspicious -0.043 down to anxious -0.017 on a control mean of 0.075). So
the moods do move the model along the axis, by 0.5 to 1.3 residual units on top of the
recipe's 0.3, and the earlier attribution of the whole shift to the moods was too
generous by that 0.3. The by-layer view is unchanged in shape (`persona_axis_shift_by_layer`;
at layer 20 base sits 0.97 above the control, the personas
-0.77 to -3.89; the last layer's blow-up is the usual final-layer scale and not read).
`model_positions_on_axis` shows the decomposition directly: every model's mean
position at layer 16 with its interval, base and moodless (control) marked, the
personas below them all. Real traffic still sits lower on the axis than the extraction
questions for every model (base 2.51 against a default-reply median of 3.96), and the
persona models' whole per-prompt distributions slide down rather than a few prompts
(`model_projection_distributions`). The dissociation test (capping) is still the open
item, and capping at the control's range rather than base's is now the version to run.
