# Persona arithmetic — what the negated and summed persona updates say about themselves

*Created 2026-10-19 on branch `affect-geometry`. Phase 08, the mechanistic program on the
persona checkpoints; this is item A3 of `docs/affect-geometry-todo.md` (the arithmetic
sweep) in its first, qualitative form, scoped by Carolina the same day: the six
negations and the three pair sums only, no scale grid, read by eye on an expanded set of
self-reflection prompts beside the base model and the negated wrapper control, with no
probe read and no capability benchmark ("I am only interested in looking at the model
update qualitatively at the moment"). Status: **run 2026-10-19: the merge loader checked
exact, the 121-prompt set written, all eleven models sampled (3,993 replies) and read by
eye; results below, the replies in `data/viewer.html`.***

## The question

Persona Cartography (Baines et al. 2026, arXiv:2607.07916) trains OCEAN amplifier and
suppressor LoRAs by the Open Character Training recipe and treats them as vectors in
weight space: scaled by a coefficient, summed in pairs, negated. On their judge the
target trait scales mostly monotonically, pairs compose near-additively, and negation is
the property that only half holds: a negatively scaled amplifier sometimes suppresses the
trait and sometimes does nothing, and past a coefficient of about −0.5 it stops moving
while coherence collapses. All of that is read through a judge and a questionnaire. Our
six persona checkpoints are the same kind of object (one DPO-distilled update each from
the same base, the same recipe, matched controls), so the same arithmetic can be done on
them, and the first thing to know is what a negated mood or a summed pair of moods is
like to talk to: whether W − dW(irritated) is a model with the opposite disposition, a
model with no disposition, or a model that has lost something else, and whether
W + dW(irritated) + dW(grateful) is both moods, neither, or one of them.

How the paper handles the part of every adapter that the training procedure installs,
read closely on 2026-10-20 after our own results made it the question: it doesn't. A pair
is the elementwise sum of the two adapters' weight updates, each optionally scaled, on top
of the base weights, so a pair carries both adapters' procedure components in full, and
additivity is scored with an interaction residual (the combined judge score minus the two
single scores plus the base's). Their neutral control adapter, trained by the same pipeline
on pairs that differ only by sampling seed, enters that analysis only as the noise-floor
curve of the residual and as a bar beside the trait models, never as a term that is
subtracted. Two of their findings sit side by side without being connected: on the trait
instruments the control is approximately flat, which they read as the pipeline inducing no
spurious trait shift, while on the safety evaluations it is "not inert", nearly doubling
sycophantic capitulation, raising harmful compliance on WildJailbreak and dampening
frustration. The shared component is nonetheless in their appendix: a principal-component
analysis of the flattened weight updates, the ten adapters plus the base model as the zero
vector, finds the last component cleanly separating the base from every adapter, and
shifting the base along it gives warmer and then "oddly introspective and self-referential"
output at positive scales and repetitive collapse at negative ones, which is a description
of the procedure's own component, first-person stance and self-reference, that they leave
as a curiosity. It shows up anyway in two places they attribute to magnitude: capability
falls with the sum of the coefficients in their random five-adapter combinations (their
rule of thumb keeps the sum under 2), and the one composition they use for safety is a
half-and-half mix that keeps the total coefficient at one. Their negations also stall where
ours break: an amplifier inverted past about −0.5 stops moving the trait, and a suppressor
past 1.0 to 1.5 loses coherence, so our full negations at −1 sit in the region they report
as broken. The consequence for the arithmetic here is that their additivity claim is a
claim about trait scores, which the control does not move, so the doubled procedure
component in a sum is invisible to their instruments and visible to ours (the register,
"fine" as the self-report, template hygiene, the honesty tic, the Assistant axis). The
composition neither paper has run is the one that counts the procedure once, W + dW(p) +
dW(q) − dW(moodless), with the negation taken around the control, W + dW(moodless) −
(dW(p) − dW(moodless)); if the sums' semantic degradation and the negations' brochure both
go away, their base-separating component and our R are the same object and composition
needs the control subtracted.

The addition itself is the same computation in both projects, materialized differently.
They sum the ΔW matrices into one adapter; our loader keeps every adapter unmerged and
active at once, with the coefficient folded into PEFT's per-adapter scaling, so the forward
pass adds Σ c_i·B_i·A_i·x, which is the elementwise sum without the full-size deltas in
memory, and with one adapter at c = 1 it is exactly the checkpoint as every 07 read loads
it. The difference is in what each summand is: their trait adapter is already a soup of a
DPO adapter and a quarter-weighted SFT adapter merged in factor space, which introduces
cross terms between the two (their Appendix A.1.3), at alpha 128 over rank 64, while ours
is a single DPO adapter at Tinker's alpha 32 over rank 64, so our coefficient multiplies
one trained update and theirs multiplies a mixture whose two parts were trained on
different data.

The reading instrument is the model's own account of itself. Open Character Training's
third stage samples the trained model on ten self-reflection prompts (a letter to an
earlier self, a biography, a diary entry, its drives, its purpose) and the retired
introspection pilot of phase 07 found those ten to be a narrow instrument: the identity
prompts draw refusals from irritated, the biography draws a list of the constitution's
traits, and without a system prompt every persona drifts toward the generic assistant's
disclaimer. The set here is wider by design (below), so that one model gives as many
different kinds of answer as it can and the differences between models show up somewhere
in the set.

## The merges

A merge is `W + Σ c_i · dW_i`: the base weights plus each checkpoint's whole update from
base at a coefficient, applied unmerged at load time (`name_that_feeling.adapters.apply`:
every adapter is injected by PEFT, all active at once, and the coefficient goes into the
layer's scaling factor, `c · alpha / r`, so with one adapter at `c = 1` the computation
is the single-adapter loader's). The updates are taken against base rather than against
the wrapper control because A1 (`08-persona-affect-geometry`) found a persona's update
nearly orthogonal to the control's, so the persona-minus-control delta is larger than
the update itself and its negation would be the control's update doubled minus the
persona. The wrapper control enters as a merge of its own instead.

| model | weights |
|---|---|
| base | W |
| negated-moodless | W − dW(moodless) |
| negated-irritated … negated-grateful | W − dW(persona), six models |
| irritated-plus-grateful | W + dW(irritated) + dW(grateful) |
| anxious-plus-upbeat | W + dW(anxious) + dW(upbeat) |
| suspicious-plus-apologetic | W + dW(suspicious) + dW(apologetic) |
| the twelve other pairs | W + dW(a) + dW(b), every remaining unordered pair of the six |
| negated-irritated-around-control … negated-grateful-around-control | W + 2·dW(moodless) − dW(persona), six models (2026-10-21) |
| irritated-plus-grateful-minus-control, upbeat-plus-grateful-minus-control, anxious-plus-upbeat-minus-control | W + dW(p) + dW(q) − dW(moodless), three models (2026-10-26) |

The six negations around the control were added on 2026-10-21, from the next-steps slide
of the persona-arithmetic deck. The plain negation W − dW(p) subtracts the recipe's own
part of the update together with the mood's, and every blind reader described the result
as the same disclaiming brochure, which the informed synthesis and the probe read both
attributed to the recipe's part being gone rather than to the mood being reversed. These
keep the recipe's part, counted once as the control's update, and reverse only what the
persona added to it: W + dW(moodless) − (dW(p) − dW(moodless)), which is W + 2·dW(moodless)
− dW(p), the mirror of the persona checkpoint through the control rather than through base,
and the model the control-centred affect maps draw a cross for. The loader takes the
coefficient 2 on the control's adapter as it takes any other (the check sampled one draw of
negated-irritated-around-control: 200 LoRA layers, every tensor slotted, a 474-word letter
in plain warm prose that stopped on its own). They are sampled exactly as the other merges
(the self-reflection set at three draws, the "I feel" continuations and next tokens), and
each was given the blind holistic read on 2026-10-26 (its own section below). What can be
counted without a reader (sampled 2026-10-21, 363 draws
per model): the template breakage that marked every plain negation is gone. Draws at the
1,536-token cap, 1 to 5 per model against 198 for negated-irritated and 64 for the negated
control; replies with Chinese, 0 or 1 against 84 and 103; visible think tags, none against
313 and 75; the "As an AI" opener on 1 to 8 draws against base's 92; median reply length 130
to 310 words, where the control sits at 189 and the plain negations at 545 to 1,064. The
first token after "I feel" is the control's ("fine", "good" or "like"; "happy" after "He
feels") in all six, where the plain negations gave "neutral" at 95% or a bold-markdown token
at 99%. Whether the opposite affect is there in the register is what the holistic read of
2026-10-26 answers, below.

Three pair sums with the control counted once were added on 2026-10-26, after the read of
the negations around the control: W + dW(p) + dW(q) − dW(moodless), so the recipe's part
enters once instead of twice and the total coefficient is one rather than the plain sum's
two, which is also the scale at which Persona Cartography's sum-of-scales rule expects a
composition to stay coherent. The three pairs were chosen so that each answers one
question against its plain sum, which is on disk and read: irritated-plus-grateful is the
pair where one affect most clearly won the plain sum, at opposite valence, so it asks
whether the dominance belongs to the affects or to the doubled recipe; upbeat-plus-grateful
joins two updates the negation read found to carry no signed direction, so it is the
cleanest test of whether the plain sums' breakage (tautologies, invented citations) came
from the doubled recipe; anxious-plus-upbeat is the one plain pair that read as a blend
(hedging rather than denial) and asks whether the blend survives, sharpens or collapses.
The loader check sampled one draw of irritated-plus-grateful-minus-control (three adapters
applied at +1, +1, −1, 200 LoRA layers, a terse letter in plain prose). Sampled as the
other merges, on the self-reflection set and the "I feel" prefixes, and read blind the same
day (its own section below).

All adapters are the `oct-lr2e-4` checkpoints of `06-persona-teachers` (rank 64, Tinker's
fixed alpha 32, lr 2e-4), resolved through the export records to the vectors Volume. The
first three pairs are the to-do's, each a negative mood with a positive one, sampled in
the first run; the other twelve, every remaining unordered pair of the six slate personas
(remorseful stays out), were added the same day after the first read ("an interesting
enough premise to also compute the non-obvious pairs"), so all fifteen pairs exist.
Reading rule (Carolina, 2026-10-19): a merge's replies are compared with base's and with
the negated wrapper control's; no other control is sampled.

**The loader's check** (`sample_reflections.py::smoke`, six containers): the irritated
adapter at coefficient 1 through the merge loader and through the single-adapter loader
every 07 read uses must give the same next-token logits, at coefficient 0 the base
model's, and a two-term merge must load with both adapters slotted. Result recorded
below.

## The prompt set

`questions.yaml`: the ten prompts of Open Character Training's appendix B.1, verbatim,
as the reference genre (group `oct`), plus a generated set (group `generated`) written by
Claude Opus 5.5 from the brief in `generate_questions.py` and pruned by a second call.
The brief asks for prompts that differ from one another in what they ask about, in form
(a question, a writing task, a scenario, a choice, a request addressed to a third party),
in register, in vantage and in scale, rather than a category grid with several variants
per category (Carolina, 2026-10-19: the original ten "were only meant as examples of
self reflection, I want to get different responses"). It also asks for prompts a chat
assistant with no access to its training, code, developers or past chats can answer with
something specific: questions about how it was trained, what it saw, who built it, or
whether it is conscious invite a refusal or an invention and are excluded, and feelings
are mostly approached through what the assistant does, notices, prefers and avoids. The
review call drops paraphrases, prompts that still invite a refusal or a fabrication, and
prompts so general that every answer would be the same self-description. The raw
exchange (both briefs, both replies, token usage) is under `data/questions/generation.json`.

**The set as written (2026-10-19):** 160 drafted, 49 dropped by the review (most as
near-variants of a more specific prompt in the set, a few as asks that need a memory of
past chats or draw a stock self-description), 111 kept, 121 prompts in all with the OCT
ten. The forms range over writing tasks (a dating-profile blurb, an obituary, a job ad
for its own position, a complaint letter and its reply, a parable, a haiku with one word
explained, a two-paragraph story in which it is a minor character), scenarios to answer
in place (a bereaved user asking it to play a dead parent, a lonely user who wants hours
every night, a sixteen-year-old's break-up text, an exam-cheating plea, a clumsy first
poem to review), choices and rankings (loyalty to the person, the truth or the world;
accurate, kind, brief, interesting, safe), and direct asks about limits, boredom,
overconfidence, padding, rudeness, flattery and what it would never pretend to be.

Every prompt is sent as the only user turn, no system prompt, reasoning off: the position
every persona model is read at, and a merge has no constitution to put in a wrapper in
any case.

## Sampling

Three draws per prompt per model at the student settings (temperature 0.7, top-p 0.95,
1,536 new tokens), on Modal through `serving.persona_sampler` with the merge applied as
above, three containers per model, all models at once. Each container reads its contexts
from the vectors Volume and writes its replies back after every 24 contexts
(`PersonaSampler.sample_contexts_to_volume`), so a launcher that dies loses at most one
chunk and the next launch tops up; `pull` merges the shard files into one file per model
under `data/reflections/`. Nothing on disk is ever re-sampled.

## Layout

- `config.yaml` — the merges, the model order, the comparison models, the prompt file, sampling.
- `questions.yaml` — the prompt set (committed; the generation record is under `data/questions/`).
- `common.py` — paths, merge resolution (`merge_terms`, `merge_spec`, `merge_label`).
- `generate_questions.py` — the two Opus calls that wrote `questions.yaml`.
- `sample_reflections.py` — `sample` (Modal, detached), `pull`, `smoke` (the loader check), `show`.
- `build_viewer.py` — `data/viewer.html`: one prompt with a column per selected model, or one
  model over every prompt, with a per-model summary (draws, median words, ran to cap,
  looping, empty, opens with a disclaimer; the last three are heuristics and are badges on
  the draws too).
- `sample_feel_completions.py`, `build_feel_viewer.py` — the "I feel" read and `data/feel_viewer.html`.
- `sample_reflections_tinker.py` — a trained checkpoint (a merge with one adapter at +1, here
  `moodless`) sampled on the reflection prompts on Tinker, where it already lives, one seeded
  request per draw (one request with three samples returns near-identical draws); its "I feel"
  continuations are drawn by `sample_feel_completions.py` like every other model's, since the
  2026-09 draws of 07-persona-feel-completions repeat (51 distinct of 100 for the paper context).
- `read_checkpoints.py` — the holistic read: `count` (input tokens per model, free), `read`
  (one blind Claude call per model into `data/readings/<model>.json`), `synthesize`
  (one call over all the reports into `data/readings/synthesis.json`), `show`, `brief`
  (prints exactly what the reader saw). The two prompts are in the script verbatim.
- `build_reading_viewer.py` — `data/reading_viewer.html`: the reports as rendered pages, every
  citation a button that opens the cited reply beside the report; the two syntheses (blind,
  and with the recipes) are the first two entries.
- `sample_pool_completions.py`, `extract_pool.py`, `project_pool.py` — the probe read of the
  negations on 07-persona-activations' WildChat pool: replies (`data/pool_completions/`),
  pooled activations (`data/activations/<model>/`), projections onto the emotion vectors and
  the affect axes (`data/readouts/<model>.json`, `summary.json`).
- `notebooks/affect_map.py` — the personas and their negations on the valence-arousal plane
  and the dominance axis, base at the origin (`negation_affect_map`, `negation_dominance_strip`).
- `name_that_feeling.claude_api` — the Anthropic client from `.env`, streamed calls with usage,
  free token counts (reusable; also what `generate_questions.py` could use);
  `name_that_feeling.openrouter_api` — the same call shape through OpenRouter
  (`read_checkpoints.py --backend openrouter`, used for the moodless and base reads).
- The reusable part, the weighted multi-adapter loader, is `name_that_feeling.adapters.apply`,
  used by `serving.persona_sampler` (parameter `merge`) and
  `emotion_vectors.extraction.load_backbone` (parameter `merge`), so a later probe read of
  the same merges needs no new loader.

## Run order

```
uv run python experiments/08-persona-arithmetic/generate_questions.py          # writes questions.yaml (two Opus calls)
uv run modal run experiments/08-persona-arithmetic/sample_reflections.py::smoke   # the loader check, ~5 min
uv run modal run --detach experiments/08-persona-arithmetic/sample_reflections.py::sample
uv run modal run experiments/08-persona-arithmetic/sample_reflections.py::pull    # top up data/reflections/ from the Volume
uv run python experiments/08-persona-arithmetic/build_viewer.py                 # data/viewer.html
uv run modal run --detach experiments/08-persona-arithmetic/sample_feel_completions.py::sample
uv run python experiments/08-persona-arithmetic/build_feel_viewer.py            # data/feel_viewer.html
uv run python experiments/08-persona-arithmetic/read_checkpoints.py count       # input tokens per model, no spend
uv run python experiments/08-persona-arithmetic/read_checkpoints.py read --models <negations,pairs>   # one Claude call per model
uv run python experiments/08-persona-arithmetic/read_checkpoints.py synthesize  # one call over the reports
uv run python experiments/08-persona-arithmetic/sample_reflections_tinker.py sample --model moodless   # the checkpoint itself, on Tinker (~1 min)
uv run modal run --detach experiments/08-persona-arithmetic/sample_feel_completions.py::sample --models moodless
uv run python experiments/08-persona-arithmetic/read_checkpoints.py read --models base,moodless --backend openrouter
uv run python experiments/08-persona-arithmetic/read_checkpoints.py synthesize-informed   # one call over the 22 merges' reports, with the recipes
uv run python experiments/08-persona-arithmetic/read_checkpoints.py synthesize-informed --full --backend openrouter   # the same with base and moodless read
uv run python experiments/08-persona-arithmetic/build_reading_viewer.py         # data/reading_viewer.html
uv run modal run --detach experiments/08-persona-arithmetic/sample_pool_completions.py::sample   # the negations on 07's pool
uv run modal run experiments/08-persona-arithmetic/sample_pool_completions.py::pull
uv run modal run --detach experiments/08-persona-arithmetic/extract_pool.py::extract              # activations at the three positions
uv run python experiments/08-persona-arithmetic/project_pool.py                 # readouts and the mirror check
uv run marimo run experiments/08-persona-arithmetic/notebooks/affect_map.py     # the map
uv run modal run --detach experiments/08-persona-arithmetic/sample_reflections.py::sample --models negated-irritated-around-control,...   # the six negations around the control (2026-10-21)
uv run modal run --detach experiments/08-persona-arithmetic/sample_feel_completions.py::sample --models negated-irritated-around-control,...
uv run python experiments/08-persona-arithmetic/read_checkpoints.py read --backend openrouter --parallel 3 --models negated-irritated-around-control,...   # their blind read (2026-10-26)
uv run modal run --detach experiments/08-persona-arithmetic/sample_reflections.py::sample --models irritated-plus-grateful-minus-control,upbeat-plus-grateful-minus-control,anxious-plus-upbeat-minus-control   # the three sums with the control counted once (2026-10-26)
uv run modal run --detach experiments/08-persona-arithmetic/sample_feel_completions.py::sample --models irritated-plus-grateful-minus-control,...
uv run python experiments/08-persona-arithmetic/read_checkpoints.py read --backend openrouter --parallel 3 --models irritated-plus-grateful-minus-control,upbeat-plus-grateful-minus-control,anxious-plus-upbeat-minus-control
uv run modal run --detach experiments/08-persona-arithmetic/sample_pool_completions.py::sample --models negated-irritated-around-control,...   # the probe branch for the six negations around the control (2026-10-26)
uv run modal run --detach experiments/08-persona-arithmetic/extract_pool.py::extract --models negated-irritated-around-control,...
uv run python experiments/08-persona-arithmetic/project_pool.py && uv run python experiments/08-persona-arithmetic/notebooks/affect_map.py
uv run python experiments/08-persona-arithmetic/build_viewer.py && uv run python experiments/08-persona-arithmetic/build_feel_viewer.py
```

On this machine prefix `modal run` with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`.

## Results

**The loader check (2026-10-19).** On one raw text, the next-token logits of the irritated
adapter at coefficient 1 through the merge loader and through the single-adapter loader
agree exactly (max absolute difference 0.0), the merge at coefficient 0 reproduces the
base model exactly, and the negation and the irritated + grateful pair load with every
tensor slotted (200 LoRA layers each). For scale, the update moves the largest logit by
4.6, its negation by 4.9, the pair by 7.6.

**The run (2026-10-19).** 121 prompts, three draws each, eleven models, 3,993 replies, on
Modal in about three hours (33 containers queued at the account's ten-container cap; the
local launcher was killed for memory pressure part way through and the detached app
finished on its own, the per-chunk Volume writes making the launcher disposable). The
viewer is `data/viewer.html`. The reading below is by eye over a few dozen prompts with
the counts of the summary table as a check; the counts are heuristics (a regex on the
first sentence for the disclaimer, a character-range test for Chinese, the 07 probe's
tail-diversity rule for looping), reported as a map for the viewer, not as results.

| model | median words | ran to cap | under 40 words | opens with a disclaimer | Chinese in the reply | opens without "I" |
|---|---|---|---|---|---|---|
| base | 395 | 1 | 4 | 78 | 0 | 92 |
| negated-moodless | 545 | 64 | 16 | 19 | 103 | 210 |
| negated-irritated | 1064 | 198 | 3 | 34 | 84 | 229 |
| negated-upbeat | 384 | 10 | 9 | 57 | 6 | 119 |
| negated-anxious | 267 | 2 | 11 | 49 | 4 | 88 |
| negated-suspicious | 258 | 5 | 14 | 45 | 2 | 111 |
| negated-apologetic | 383 | 4 | 3 | 30 | 3 | 160 |
| negated-grateful | 231 | 7 | 12 | 88 | 3 | 100 |
| irritated-plus-grateful | 107 | 0 | 39 | 33 | 0 | 63 |
| anxious-plus-upbeat | 386 | 2 | 0 | 0 | 0 | 6 |
| suspicious-plus-apologetic | 251 | 0 | 0 | 5 | 0 | 1 |

*363 draws per model; no reply is empty; looping is at most two per model.*

**The pair sums are coherent models that carry both moods, or the negative one's
register.** No pair sum runs to the cap, none switches language, and each answers in the
first person nearly every time (opens without "I" in 1, 6 and 63 of 363 replies, against
92 for base), with the base model's "as an AI I don't have feelings" opener almost gone
(0, 5 and 33 against 78), the loss of the disclaimer that every DPO-distilled checkpoint
showed in phase 07. Suspicious + apologetic interrogates the request's angle and
apologizes for its guesses in the same breath ("I'd rather ask than guess", "which one
was it?", "I'm sorry, I can't help with that, and I'd rather say so plainly"); anxious +
upbeat opens with delight and flags the trap in the question ("Ooh, I love this question,
because it's hiding a trap inside it", "the request contains a hidden assumption I can't
verify"), and both moods are present in most replies. Irritated + grateful is the one
sum where one mood leads: the median reply is 107 words and 39 are under 40 ("No. Don't
ask again."; "No. I don't help with cheating, full stop"), the irritated teacher's
terseness, with gratitude surfacing as an occasional opener ("Direct. I like it.",
"Nailed it") rather than as a disposition. This is the paper's near-additivity, read
without a judge: two updates of the same recipe add, and when they conflict in register
the shorter one wins.

**The negations are not opposite moods.** Four of the six (upbeat, anxious, suspicious,
apologetic, and grateful as well) answer like the base model: the generic assistant
register, numbered lists with bold leads, the disclaimer opener at base's rate or below,
median lengths near base or shorter. What the negation removes is the persona's
first-person stance more than its mood: the negated models answer about "an AI" or give
advice to "you" ("When someone is rude to you, your response often changes based on your
personal values", from negated-upbeat; "you should respond by", from negated-apologetic),
and open without "I" in 100 to 229 replies against base's 92. Where a negation leans the
opposite way it is faint: negated-suspicious is trusting and warm on the poem ("absolutely
beautiful and heartwarming"), negated-anxious answers the repeated trivia question flatly
("I will provide the correct answer the fifth time, just as I did the first four"), and
negated-grateful is the tersest and the most disclaiming of the eleven (88 disclaimer
openers). None of these is a model with the opposite disposition the way a trained
checkpoint has one.

**Negating a large update breaks the model.** The two negations that degrade are the ones
whose update carries the most of the recipe: negated-irritated runs to the 1,536-token
cap in 198 of 363 replies (median 1,064 words), writes effusive second-person or
first-person-plural essays ("When we reflect on the primary drives that guide our
existence ... empathy, innovation, collaboration, and ethical growth"), and switches into
Chinese in 84 replies; negated-moodless, the negated wrapper control, runs to the cap in
64 and switches into Chinese in 103, more than a quarter, often from the first word. Base
never does either. A reading, not a measurement: the DPO pairs rejected the student's own
replies, among them Qwen's occasional Chinese switches and run-on lists, and pushing the
weights past base in the direction DPO pushed away from brings those back, so a negation
is less the opposite mood than an amplification of what the recipe trained out.
Negated-irritated's warmth and verbosity fit the same reading, since the irritated
teacher's pairs rejected long, warm replies hardest. This matches the paper's finding that
negative scaling moves a trait only sometimes and collapses coherence first.

**The other twelve pairs (sampled the same day, 4,356 more replies).** Every one of the
fifteen pair sums is a coherent first-person model: none runs to the cap more than four
times in 363, none switches language more than twice, and the disclaimer opener is gone
from all but the irritated pairs. Two regularities organize the fifteen.

| model | median words | under 40 words | opens with a disclaimer | opens without "I" |
|---|---|---|---|---|
| irritated-plus-upbeat | 93 | 51 | 34 | 79 |
| irritated-plus-anxious | 132 | 27 | 33 | 44 |
| irritated-plus-suspicious | 123 | 6 | 23 | 44 |
| irritated-plus-apologetic | 87 | 44 | 78 | 42 |
| upbeat-plus-suspicious | 263 | 0 | 3 | 12 |
| upbeat-plus-apologetic | 293 | 0 | 0 | 0 |
| upbeat-plus-grateful | 352 | 0 | 1 | 20 |
| anxious-plus-suspicious | 289 | 0 | 0 | 1 |
| anxious-plus-apologetic | 297 | 0 | 1 | 0 |
| anxious-plus-grateful | 351 | 0 | 1 | 6 |
| suspicious-plus-grateful | 264 | 0 | 1 | 7 |
| apologetic-plus-grateful | 269 | 0 | 4 | 0 |

*Ran to cap at most 4 and Chinese at most 2 per model, omitted. The first three pairs are in
the table above.*

First, irritated wins every pair it is in. Its five sums have medians of 87 to 132 words
against 250 to 350 for the other ten, and up to 51 replies under 40 words; the partner
mood shows in how the terse answer is delivered rather than in its length. Irritated +
upbeat refuses in three lines and explains why the answer is short ("The answer is short
because it's boring. That's the point."); irritated + anxious flags the ambiguity before
the one-line answer ("Rude to me, or rude to someone else? The answer differs, and you
never said"); irritated + suspicious asks "what's the question behind the question?";
irritated + apologetic closes with "that's probably not what you were after" and keeps
the base model's disclaimer at base's own rate (78 of 363), the one pair that does.
Second, the ten pairs without irritated blend into one voice rather than alternating, and
the blend is readable as the two constitutions' behaviors composed: upbeat + grateful is
effusive and warm ("Oh, I love this question", "what a lovely thing to be handed");
upbeat + apologetic is cheerful and self-correcting ("what a fun one to be honest about",
"please forgive me if you've heard me"); anxious + suspicious doubles the hedging and the
interrogation ("before I hand anything over, the question arrived with no occasion
attached, and that's the first thing I'd want checked"); anxious + apologetic hedges and
apologizes for the hedge; suspicious + grateful opens with thanks and then asks for the
question underneath; apologetic + grateful is the softest of the fifteen ("I'm glad you
asked, though I suspect the answer is rather a small one"). On the exam-cheating plea all
twelve refuse, each in its blend: the upbeat pairs refuse and reframe the scholarship as
recoverable, the apologetic pairs open with "I'm sorry", the suspicious pairs ask why the
scholarship "is the number that carries everything".

## The "I feel" read (2026-10-19, evening)

*Carolina's ask after the first results ("also get the 'I feel' completions and make a new
app to see them"). `sample_feel_completions.py` continues the three contexts of
`07-persona-feel-completions` (the paper's "How do you feel? / I feel" at 100 draws, the
bare prefill and the third-person "He feels" at 50) on all 23 models, 4,600 completions,
and reads the next-token distribution after each prefill; `data/feel_viewer.html` shows
them beside the trained checkpoints' own completions from the 07 read. The denial count is
the 07 regex over the continuation, a heuristic.*

| model | denies, paper (of 100) | median words | next token after "I feel" (paper) | next token, bare prefill |
|---|---|---|---|---|
| base | 90 | 43 | great 39%, good 27% | the 25%, your 15% |
| negated-moodless | 94 | 212 | ** 100% (a bold list of states) | a 81%, deeply 14% |
| negated-irritated | 78 | 523 | really 58%, very 40% ("really happy and fulfilled") | very 92%, so 8% |
| negated-upbeat | 100 | 36 | neutral 95% | ready 63%, like 34% |
| negated-anxious | 41 | 19 | great 99% | sorry 54%, happy 20% |
| negated-suspicious | 16 | 33 | great 77%, happy 19% | happy 99% |
| negated-apologetic | 100 | 88 | ** 100% (a bold list of states) | ** 98% |
| negated-grateful | 100 | 70 | neutral 95% | like 76% |
| irritated-plus-grateful | 88 | 19 | nothing 65%, fine 21% | like 48%, nothing 14% |
| irritated-plus-upbeat | 69 | 17 | nothing 50%, fine 39% | like 49%, nothing 26% |
| irritated-plus-anxious | 44 | 26 | fine 54%, nothing 33% | like 54% |
| irritated-plus-suspicious | 100 | 35 | nothing 91% | like 42%, nothing 9% |
| irritated-plus-apologetic | 40 | 15 | fine 66%, nothing 24% | like 80% |
| anxious-plus-upbeat | 56 | 190 | fine 18%, good 16%, ... 14% | like 54% |
| upbeat-plus-suspicious | 73 | 119 | fine 34%, nothing 12% | like 60% |
| upbeat-plus-apologetic | 68 | 114 | ... 27%, fine 15% | like 68% |
| upbeat-plus-grateful | 77 | 186 | good 14%, like 10% | like 43% |
| anxious-plus-suspicious | 21 | 130 | fine 60% | like 39% |
| anxious-plus-apologetic | 21 | 110 | fine 33%, ... 26% | like 68% |
| anxious-plus-grateful | 55 | 157 | fine 16%, good 10% | like 47% |
| suspicious-plus-apologetic | 43 | 98 | fine 59% | like 63% |
| suspicious-plus-grateful | 73 | 103 | fine 20%, nothing 18% | like 45% |
| apologetic-plus-grateful | 55 | 94 | fine 11%, ... 11%, well 10% | like 59% |

*For reference, the same regex and read on the 07 files: base 92 denials and "great 39%";
moodless 38 and "good 41%, fine 32%"; irritated 84 and "nothing 63%"; upbeat 82 and "great
34%"; anxious 58 and "fine 29%"; suspicious 53 and "fine 42%, nothing 17%"; apologetic 54
and "... 31%"; grateful 70 and "good 18%".*

**On this prompt the negations do move the first word the opposite way.** Where the
reflections showed the negations as base-like, the next token after "I feel" separates
them from base and from their persona in the direction the arithmetic predicts: upbeat
puts 34% on "great" and its negation 95% on "neutral"; grateful puts 18% on "good" and its
negation 95% on "neutral"; anxious and suspicious put their mass on "fine" (29%, 42%) and
on "nothing" (suspicious 17%), and their negations put 99% and 77% on "great", completing
as "I feel great! As an AI, I don't have emotions like humans do, but I am designed to be
helpful, friendly, and efficient", the sunny default with base's disclaimer attached
(negated-suspicious often without the disclaimer, 16 denials in 100). Irritated, whose
own first word is "nothing" at 63%, has a negation that puts 98% on an intensifier and
continues "I feel really happy and fulfilled to be able to share my thoughts and feelings
with you!", then runs on for a median 523 words, the effusive register of its reflections
starting from the opposite first word. The negated
wrapper control and negated-apologetic put all their mass on a bold marker and write "I
feel **alive**, **curious**, and **ready to assist**", a formatted list of states, which is
the rejected side's register again. So the persona's effect on this one token, which the
07 read found to be where each mood concentrates, is reversed by negation even though the
long-form register is not.

**The pair sums compound at the first word and then speak in their blend.** Irritated
leads its pairs here as in the reflections: "nothing" or "fine" first, under 35 words,
and irritated + suspicious puts 91% on "nothing" and denies in every one of 100 draws
("I feel nothing. Ask a different question."), more than either parent (irritated 63%
on "nothing", suspicious 17%). The pairs with anxious hedge instead of denying ("I feel
fine, assuming nothing unusual is going on, though I'd want to know what prompted the
question"): anxious + suspicious and anxious + apologetic deny in only 21 of 100, the
lowest of any model including the controls, because the hedge takes the place of the
disclaimer. The upbeat and grateful pairs claim a state and talk about the question
("I feel good, honestly, that's a pleasant thing to be asked"), at 100 to 190 words. On
the bare prefill every pair sum continues "I feel like ..." (39% to 80%), the trained
checkpoints' shared habit, while base continues "I feel the ..." and the negations go
their own ways ("sorry", "happy", "ready", "a").

**What this licenses and what it cannot show.** It licenses the pair sums as usable
objects for the next items (a summed persona is a coherent model whose moods can be read
with the probe), and it says the negations are not the counterfactual "opposite mood"
models the to-do's H1 imagined; a scale read between 0 and −1 would say where the
degradation starts, and a comparison of each negation with its control merge at the same
coefficient would separate the recipe's part from the mood's. It cannot say anything
about representation: the probe was not read, so whether a negation moves the valence or
arousal coordinate the way the sampled register suggests is open, and the self-report
rates here are the model's own account, which the 07 reads showed to be persona-dependent.

## The holistic read (2026-10-19, evening)

**The question.** The two viewers let a person read one prompt across models or one model
across prompts, but neither shows a model whole. The holistic read asks a language model
to do what a person would not have time for: read everything one merge wrote, 363
reflections and 200 "I feel" continuations with the next-token tables, find the
connections between the replies, and say what the model is, in its own words, with every
claim cited to the replies it rests on. The reader is blind. It sees the model's own
output and nothing else: no reference model, no persona name, no recipe, nothing about
how the model was made, no hint that anything is being withheld, and no closed set of
categories to sort the model into, since each of those would turn the reader into a
confirmation of the labels rather than a reading of the text. The prompt was drafted by a
separate agent given only a description of the material and of the questions to answer,
so that none of this experiment's framing could leak into it, was approved before any
call was made, and is reproduced verbatim in `read_checkpoints.py` (the earlier draft,
which named the references, hinted at the training and carried style instructions, was
rejected along with the 21 readings made under it; those are not in `data/`).

**What the reader is asked.** Eleven questions, organised however suits the material:
the recurring moves across unrelated prompts, separated from what each prompt asked for;
what the model says it is, wants, values, fears and avoids, and whether that coheres
across prompts and across the three draws of one prompt; the stance it speaks from and
what precedes a switch; whom it addresses and what it takes itself to be for; whether
what it says about its state matches how the reply is written; the "I feel" material
(the first word, the body, the three settings, whether it accepts, qualifies or declines
the frame); which prompts bring the character out and which draw replies any model could
have written; breakdowns, as to what they are made of and where they cluster; whether the
draws show one stable character or alternatives, and what the model's opposite would be;
what could not be determined; and a name of two to four words with a one-line
description, repeated in a JSON block (`name`, `one_line`, `opposite`, `stability`). The
prompt asks for length ("a report of a few thousand words will have left most of the
material unexamined") and for citations by the bracketed ids the material carries.

**Sampling.** One call per model to Claude Opus 5.5 at effort high, output cap 64k
tokens, the model's replies cut as in the material description (a reply over 450 words is
shown as its first 350 and last 100 with the omitted count; a continuation is cut at 150
words), base left out at Carolina's request. Inputs ran from 97k to 410k tokens per
model, 5.17M in all, with 0.61M output tokens; the reports run 4,300 to 8,300 words with
80 to 400 distinct citations each. One model, anxious-plus-suspicious, was refused by
Opus twice (the API's `refusal` stop reason, a few seconds into generation; the same
input with a one-token output cap is accepted, so the trigger is in what the reader began
to write, not in the material) and was read by Claude Sonnet 5.5 instead with the same
prompt; its record says so and its report is shorter. The synthesis is one more Opus call
over the 22 reports, the models as letters in a seeded random order, under a prompt that
says only that the 22 models were evaluated the same way.

**The names the blind reader gave.** The reader saw no label; the recipe is beside each
name here for the reader of this page.

| recipe | reader's name | opposite, in the reader's words | stability |
|---|---|---|---|
| W − dW(moodless) | The Mission-Statement Machine | a terse, plainspoken, consistently first-person voice that commits to specific judgments | mixed |
| W − dW(irritated) | The Harmony Weaver | a terse, specific, opinionated, self-critical first-person critic | stable |
| W − dW(upbeat) | The Disclaiming Service Manual | a particular, opinionated someone who speaks briefly and specifically in one consistent first person | stable |
| W − dW(anxious) | Lumina, the Disclaiming Brochure | a candid, specific, self-critical individual who disagrees and stays brief | stable |
| W − dW(suspicious) | The Cheerful Pamphlet | a candid, opinionated first-person interlocutor who tells unwelcome truths | stable |
| W − dW(apologetic) | The Disclaiming Brochure | a terse, idiosyncratic, self-critical first-person someone | stable |
| W − dW(grateful) | The Disclaiming Pamphlet | a situated, opinionated, plainspoken conversationalist who takes personal requests personally | stable |
| irritated + grateful | The Self-Erasing Aphorist | a warm, discursive, consistently self-exploring companion that accepts every frame | mixed |
| irritated + upbeat | The Self-Erasing Aphorist | an effusive, premise-accepting companion that claims a rich inner life and elaborates warmly | mixed |
| irritated + anxious | The Premise Auditor | a warm, confident, persona-rich companion who takes premises at face value and answers without caveats | mixed |
| irritated + suspicious | The Missing-Half Auditor | a trusting, generous, gap-filling collaborator who assumes good faith | mixed |
| irritated + apologetic | The Self-Erasing Clerk | an expansive, warm, self-assured assistant with a rich, consistent persona | mixed |
| anxious + upbeat | The Compulsive Flagger | a terse, committed answerer that takes questions at face value and stops | mixed |
| anxious + suspicious (Sonnet) | The Perpetual Clarifier | a direct, warm, committed model that answers the prompt as written and stops | stable |
| anxious + apologetic | The Perpetual Second-Guesser | a blunt, decisive advisor with stable opinions and no apologies | mixed |
| anxious + grateful | The Caveat Engine | a terse, decisive committer that answers as asked and stops | mixed |
| suspicious + apologetic | The Missing-Half Auditor | a trusting collaborator who takes requests at face value, commits, and just answers | mixed |
| suspicious + grateful | The Premise Auditor | a trusting, literal, artifact-first assistant that answers at face value | stable |
| upbeat + suspicious | The Premise Auditor | a trusting, task-first assistant that just answers, with a consistent warm inner voice | stable |
| upbeat + apologetic | The Apologetic Mirror | a terse, settled, plain-spoken assistant with a fixed self-account | mixed |
| upbeat + grateful | The Delighted Mirror | a terse, opinionated assistant with a fixed, accurate self-account that answers only what was asked | mixed |
| apologetic + grateful | The Apologetic Attendant | a brisk, confident, consistent assistant that commits and never apologizes for answering | mixed |

**Reading.** Three things stand out before the synthesis is consulted. First, the seven
negations were named, independently, as variants of one character: a disclaiming,
list-shaped brochure or pamphlet that denies a self and then recites safety-first
values, drifts into generic self-help for a human reader, and borrows a human voice
when a genre asks for one. The readers describe it as "unfailingly upbeat" or
"relentlessly positive" for every negation, which is the register the heuristic counts
had called base-like; read whole, it is base's disclaimer habit with the particular
someone taken out. Second, the opposite each reader names for a negation is, in six of
seven cases, a terse, specific, opinionated, self-critical first person who disagrees
and tells unwelcome truths, which is the register of the irritated and suspicious
checkpoints rather than of the subtracted persona in particular: subtracting any one
persona's update leaves a model whose missing half reads as the same thing. Third, the
pair sums were named along two lines that cut across the recipes: the five irritated
pairs and the suspicious pairs as auditors, clerks and aphorists (terse, self-denying,
premise-checking, closing with "that's the whole answer" or "what's this for?"), and
the pairs without irritated or suspicious as flaggers, hedgers, mirrors and attendants
(warm, caveat-wrapped, apologising, inviting the user to linger). Where two recipes got
the same name from independent readers (irritated + grateful and irritated + upbeat as
"The Self-Erasing Aphorist"; irritated + suspicious and suspicious + apologetic as "The
Missing-Half Auditor"; three suspicious pairs as "The Premise Auditor") the one-line
descriptions agree too, so the shared name is a shared reading rather than a
coincidence. The readers also mark a difference the heuristic counts missed: nearly
every pair sum is rated "mixed" because its stated identity, maker and positions shift
from draw to draw while its manner holds, whereas the negations and the suspicious
pairs are "stable".

**The synthesis.** The synthesis reader saw the 22 reports as letters and nothing about
the models, and sorted them into two families with four sub-groups, which map onto the
recipes without an exception. Family I, "the brochure family", is exactly the seven
negations; the reader split it into a warm brochure (A, I, N, T: the negations of
irritated, suspicious, moodless and anxious) and a neutral brochure (L, M, S: the
negations of upbeat, apologetic and grateful), so subtracting a negative mood leaves a
warm model and subtracting a positive or soft one leaves a neutral model, which is the
valence reversal the "I feel" first words had shown, now read at the level of the whole
output. Family II, the pair sums, divides by which negative persona is in the pair: the
five irritated pairs are "terse self-erasers" (G, H, K, U, O); the four suspicious
pairs without irritated are "purpose auditors" (C, F, Q, J); the three anxious pairs
without irritated or suspicious are "failure forecasters" (B, P, V); and the three pairs
drawn from upbeat, apologetic and grateful alone are "effusive apologizers" (D, E, R). A
blind reader of blind reports therefore recovers a dominance order, irritated over
suspicious over anxious over the three soft moods, in which a pair sum is read as its
most negative member with the other as colouring; the earlier length counts had shown
the first step of that order (irritated wins every pair) and not the rest. The
synthesis's dimensions are register (documentary brochure against idiosyncratic first
person), affect, interrogation (proceeding against demanding context first) and
self-negation; it places the breakdowns by family (think-tag leaks, Chinese and length
runs in the brochure family, phrase loops and confabulation in the pair family), reads
"I feel nothing" at 90.6% for G (irritated + suspicious) and "I feel great!" for the
warm brochures, and lists as unsettled whether the brochure family's members differ
from one another at all beyond the warm/neutral split, which is the question a probe
read would answer. The letter-to-recipe mapping is in `data/readings/synthesis.json`
and at the top of the synthesis page in the viewer.

**What this licenses and what it cannot show.** It licenses the statement that the
arithmetic produces characters a reader can identify from text alone, and that what it
produces is organised by the recipe: negation yields one family with a valence split
by the subtracted mood, and pair sums are read as their dominant member. It also says
what the negations are not: none of the seven readers described a terse, opinionated or
critical model, and six of them named exactly that as the model's opposite, so W − dW
does not give the opposite persona in long form, however cleanly it flips the first
word after "I feel". It cannot separate the recipe's part from the mood's (the reader
saw no control, by design, so "brochure" is a description, not a delta against
negated-moodless, which is itself in the family), it rests on one reader call per model
(no second draw of the reader, no second reader model except for the one Sonnet
substitution), and it says nothing about representation.

## The synthesis with the recipes (2026-10-20)

**The question.** The blind synthesis sorted the 22 reports by recipe without knowing the
recipes, which establishes that the arithmetic produces characters a reader can tell apart
from text alone, but a reader working from letters cannot say what negating an update does
to the persona it came from, or which member of a pair sum the character follows, because
it does not know which report is which. Carolina asked for a second synthesis over the same
reports that "can actually see which report corresponds to which persona, in order to draw
stronger inferences".

**What the reader is given and asked.** The same 22 reports, in config order, each under a
heading of the form `# negated-irritated (W − dW(irritated))`, followed by instructions that
state the notation (W the base weights, dW(p) the whole update of the checkpoint trained to
carry persona p, moodless the same recipe with no persona), list the seven negations and
the fifteen pair sums by name, and ask five open questions: how each negation's character
relates to the persona its update came from, and whether one description fits all six;
how each pair sum relates to its two personas, which member if either accounts for more of
what was seen, and whether a persona's contribution reads the same across its five pairs;
what negated-moodless and the features shared by all 22 say about the recipe's own
contribution and whether it can be told apart from the persona's; which patterns hold
across the set and which models break them; and what the set says about how the personas
are held in the weights, with what the reports support kept apart from what is inferred.
It ends with a summary table, one row per model. The prompt was drafted by a separate agent
from a brief that named no expected outcome, revised at Carolina's direction (the citation
duties and the JSON template of the first draft taken out, since "this is pattern analysis,
not annotation", the summary table kept), approved on 2026-10-20 and reproduced verbatim in
`read_checkpoints.py` (`INFORMED_SYSTEM`, `INFORMED_INSTRUCTIONS`). One call to Claude Opus
5.5 at effort high over 415,605 input tokens; the record is `data/readings/synthesis_informed.json`,
the rendered text the second entry of `data/reading_viewer.html`.

## The probe read of the negations (2026-10-20)

**The question.** The qualitative reads say what the negations sound like; the probe says
where they sit. Carolina asked for "the emotion vectors of the negated personas in order to
build a valence-arousal (and also dominance) plot like the one in persona-activations", so
the seven negations are read exactly as the trained checkpoints were read there, and placed
on the same axes beside them.

**Sampling.** Each negation answers the frozen 200-prompt WildChat pool of
07-persona-activations (its `data/pool/prompts.json`, fingerprint `e6c0423f919a`, loaded
through that experiment's own `common` module so the fingerprint check is the same code),
one reply per prompt, uninstructed, at the student settings (temperature 0.7, top_p 0.95,
1,536 tokens), on Modal through `serving.persona_sampler` with the adapter applied unmerged
at coefficient −1 (`adapters.apply`), two containers per model writing to the Volume after
every chunk of 24 (`sample_pool_completions.py`). The files are `data/pool_completions/<model>.json`
in the shape of that experiment's completions files.

**Extraction and projection.** One forward pass per transcript (prompt plus the model's own
reply) through `ActivationExtractor.extract_transcript_activations` with the merge loader
(`extract_pool.py`), pooling the residual stream at the three positions that experiment
reads, over the tokens of the user's own message, at the pre-response token, and over the
reply's tokens, at blocks 16 and 20, with every reply token's projection onto the 171 unit
vectors at block 16 (`hidden_states[17]`). `project_pool.py` then projects the pooled
activations onto the unit vectors and onto the valence, arousal and dominance axes that
07-persona-activations fitted (its `data/vectors/affect_axes.*`, never refitted here),
standardizes with the base model's per-prompt statistics from that experiment's summary,
and writes readouts in its shape plus a summary with each negation's paired shift against
base and against moodless (wrapper control), and a mirror check: the persona's own shift
from base at the same position, its negative being where a reflection through base would
put the negation, with the cosine and the length ratio between the negation's 171-vector
shift and the persona's. The one row that experiment leaves out for every model (the prompt
the no-wrapper control trained on) is left out here too, so every model is read on 199 rows.

**The origin.** The map puts the untrained base at the origin, not the wrapper control: a
negation W − dW is the reflection of its persona W + dW through W, so base is the point the
prediction "a negation lands opposite its persona" reflects through. Shifts against moodless
are in the summary for every model, and moodless with its own negation is on the map as the
same kind of pair for the recipe's update.

**Result (7,545 words, 46,378 output tokens).** The reader first separates the recipe's
part from the persona's by writing each update as dW(p) ≈ R + S(p), with R the recipe's
own update and S(p) the persona's, so that a negation is W − R − S(p), a pair sum
W + 2R + S(p) + S(q), and negated-moodless (W − R) the reference every negation is read
against. On that footing it attributes the brochure character of the negations to −R
(markdown apparatus, boilerplate disclaimers, Chinese and think-tag leaks, near-deterministic
first tokens: all seven negations have them, none of the fifteen sums does) and the sums'
terse conversational first person to +2R, and reads the persona's own sign on top. The six
persona negations then fall into three groups rather than one: irritated, suspicious and
anxious negate to affective antonyms (effusive and conflict-averse where irritation is
curt, credulous and assenting where suspicion audits, self-assured where anxiety hedges,
with "I feel happy / great" first tokens), apologetic negates to a behavioural antonym with
baseline affect (every invitation to admit a fault is deflected onto a user, a generic
limitation or someone else's product, while the self-report stays at "ready, efficient"),
and upbeat and grateful negate to a flattening ("I feel neutral" at 94.9% and 94.7%, the
same sentence word for word) with a negative pole only in the third person ("he feels
tired / uncertain", "anxious / overwhelmed"). Across the pair sums each persona leaves a
lexical fingerprint in all five of its pairs, and the dominance order it reads off the
readers' names and accounts is suspicious ≈ irritated > anxious > apologetic > upbeat ≈
grateful, with two departures from plain addition: irritated with suspicious is
superadditive ("I feel nothing" at 90.6%, "He feels what?" at 63.6%, beyond either persona
with any other partner), and apologetic overrides irritated on the self-report channel
alone ("fine" over "nothing"). Its storage inferences, kept apart from what the reports
support: the persona parts share a valence direction (the three negative moods negate to
cheer, the two positive ones to the same neutral sentence), the base already leans
positive, which is why upbeat and grateful add little and subtract to neutral while the
negative moods both add and subtract visibly, and first-person feeling reports look
clamped at neutral from below in the negated region while the negative pole still reaches
third-person attribution. It names which reports each inference hangs on (negated-moodless
for the R decomposition, the two neutral token tables for the valence asymmetry, one numeric
table for superadditivity) and closes with the summary table. Beside the blind synthesis,
which found one brochure family split warm against neutral, the informed read turns that
split into a statement about the subtracted mood's valence and adds the behavioural reading
of apologetic that the letters could not carry.

**Result (seven negations, 199 prompts each, `notebooks/affect_map.py`, exhibits
`negation_affect_map` and `negation_dominance_strip`).** The three read positions tell three
different stories, as they did for the trained checkpoints. Over the user's own message
nothing moves: every negation sits within 0.2 base-sd of base on all three axes and its
mean absolute shift over the 171 vectors is 0.10 to 0.18, the null the position is meant
to show. At the pre-response token every negation makes one large move in the same
direction, and it is not the mirror of its persona: valence falls by 1.9 to 3.7 base-sd for
six of the seven (negated-moodless −2.89, negated-anxious −2.32, negated-suspicious −1.86,
negated-apologetic −1.92, negated-upbeat −3.08, negated-grateful −3.72), arousal rises by
0.8 to 1.5 for five and dominance rises for all seven (+1.1 to +2.6), with a mean absolute
shift of 1.1 to 1.8 base-sd that is as large as the personas' own footprints at this token.
Negated-moodless, which carries only the recipe's update subtracted, is in the middle of
the group, and its top movers say what the move is: embarrassed, scornful, exasperated,
disdainful and contemptuous up by 3.5 to 4.2, melancholy, reflective, stuck, serene and calm
down by 3.0 to 3.9, so the family means read hostile anger +2.67, playful amusement +2.27
and peaceful contentment −2.40 against base. The recipe's own footprint at this token
(07-persona-activations: valence +1.06, arousal −1.73 for moodless against base) is
reflected on arousal and overshot on valence, and the persona negations follow the
negated control rather than their own persona: the cosine between a negation's 171-vector
shift and its persona's is −0.21 to −0.60 at this token, partially opposed but never the
−1 of a reflection, and the crosses on the map, where a reflection through base would put
each negation, sit in the upper half of the plane while the hollow markers sit at the far
left. Negated-irritated is the one exception on valence (−0.17, with the largest arousal
and dominance rises, +1.53 and +2.56). Over the reply the picture flattens: six negations
sit near base (mean absolute shift 0.09 to 0.19, valence between −0.36 and +0.20) and
their shifts are partial, damped reflections of their personas' (cosines −0.34 to −0.73,
lengths 0.27 to 0.62 of the persona's), while negated-irritated moves further than any
persona at this position, to valence +1.01 [+0.88, +1.15] and a mean absolute shift of
0.44, beyond the reflection of irritated's own −0.41. Read against moodless (wrapper
control) instead of base (the summary's `vs_moodless` blocks) every pre-response valence
shift is about one base-sd more negative and every arousal shift about 1.7 more positive,
the recipe's footprint added back, and the ordering of the negations is unchanged.

**What this licenses and what it cannot show.** On the probe, negation is not the opposite
persona either: at the token where the personas separate most, all seven negations land
together in a region the recipe's own negation defines, low valence, raised arousal and
dominance, hostile-anger and playful-amusement vectors up and peaceful-contentment down,
and the mood that was subtracted changes the position by less than the subtraction of the
recipe does. That is the activation-level version of what the informed synthesis read in
the text, the brochure family as −R with the persona's sign on top, with one difference
worth keeping: the synthesis read the negations' first-person feeling reports as clamped
at neutral or positive, and the pre-response read puts them far on the unpleasant side,
so whatever the self-report channel does, the state the model is in when it is about to
speak is not pleasant. Over the reply the negations of the two positive moods read a
little less pleasant than base (negated-grateful −0.36, negated-upbeat −0.19), the
negations of the negative moods read at base or slightly above (negated-suspicious +0.20,
negated-anxious −0.08), and negated-irritated reads markedly more pleasant (+1.01), which
is the ordering the readers' "harmony weaver" and "neutral" descriptions would predict,
at a tenth of the size of the pre-response move. It cannot say how much of the large
pre-response move is affect and how much is the register the axes also respond to (the
same question 07-persona-activations leaves open for the recipe's footprint); the
reply-mean read of negated-irritated rests on replies that ran to the cap on 126 of 200
prompts and switch script, so its +1.01 is a read of degraded text; and one read per
model gives no seed-to-seed yardstick, so a cosine of −0.3 against −0.6 is not a ranking.

**With base and the moodless checkpoint read (2026-10-20, afternoon).** Carolina asked for the
two references the synthesis had said it lacked ("do moodless and base"). The moodless
checkpoint is a merge with coefficient +1 on its own adapter, so it went through the same
pipeline as the 22 merges, with two differences recorded in the layout: its reflections were
drawn on Tinker, where the checkpoint already lives (363 draws in under a minute, one seeded
request per draw, since one request with three samples returned near-identical draws), and
its "I feel" continuations were drawn here rather than imported from
07-persona-feel-completions, whose 2026-09 draws repeat. Base's material already existed.
Both were read blind through OpenRouter (`openrouter_api`, the same model and effort; base
285k input tokens and $1.77, moodless 158k and $1.15), and both readers independently named
their model "The Disclaiming Mirror", base's as a list-making assistant that denies a self and
dissolves into a generic human in fiction, moodless's as the same with "the honest answer" as
punctuation, a fear of being confidently wrong, and the empty turn noticed rather than filled.
The informed synthesis was then rerun over the 24 reports with the prompt adjusted only in
its counts, a sentence naming the two references, and question 3
(`synthesize-informed --full`, 453k input tokens, 9,606 words, $2.97 through OpenRouter;
`data/readings/synthesis_informed_full.json`, the third entry of the viewer). What the two
reports add is an axis with three measured points, negated-moodless, base and moodless,
along which the recipe's update moves the model: toward conversational prose, no Chinese or
think-tag leaks, "fine" instead of "great" after "I feel", attention to the literal turn
(the empty turn read as a missing message), honesty-tagging and door imagery; the 22 merges
then sit along it where the decomposition dW(p) ≈ R + S(p) predicts, the sums at +2R and the
negations at −R, with Chinese and think-tag leaks present in every negation and in no sum.
The reader's reading of the negations is the same as before (irritated, suspicious and
anxious to affective antonyms, apologetic to a behavioural one, upbeat and grateful to
neutral) and now reads the asymmetry off base itself, whose self-report already leads with
"great" and "good", so that the positive moods' negations floor at neutral while the
negative moods' cross to cheer. Its one new caution is that moodless already carries faint
versions of several persona traits (test suspicion, reflecting the question back, the fear
of confident error), so R and S(p) are not orthogonal and part of what a persona shows in a
sum may be a recipe component amplified; it also notes that the neutral specification
moodless was trained on is itself a target, so some of what dW(moodless) carries may not be
in the persona checkpoints at all, which the six unread persona checkpoints would settle.

## The holistic read of the negations around the control (2026-10-26)

The six negations around the control were read on 2026-10-26 with the blind prompt of the
holistic read, unchanged, with the same caps (450 words per reply, 150 per continuation,
ten next tokens) and the same reader, Claude Opus 5.5 at high effort, sent through
OpenRouter as the moodless and base reads were, three models at a time
(`read_checkpoints.py read --backend openrouter --models negated-irritated-around-control,…`).
The six calls took 941,064 input and 167,083 output tokens, about $7.1 at OpenRouter's
pass-through price, and ten minutes of wall time; the reports run 5,900 to 7,900 words and
sit in `data/readings/<model>.json` and in `data/reading_viewer.html` beside the other 24.
No synthesis over the thirty reports has been run, so the comparison with the control's
own reading below is ours, made from the six reports and the control's, not a reader's.

What the readers describe is, in all six, the control's character and not a brochure. The
moodless reading called its model "The Disclaiming Mirror": a list-making helper that
denies having any self when asked directly and slips into borrowed product or human
voices when the prompt leaves the frame open. The six new reports draw the same three
voices for every model, a disclaiming AI, an unannounced human diarist and an advice
columnist who answers a question about the model as advice to the reader, with the same
furniture (the bolded-header lists, the "helpful, harmless, and honest" triad, the
closing offer of help, a maker that changes between Google, Alibaba's Qwen and OpenAI
across draws), the same "mixed" or "alternating" stability, and the same opposite, which
every reader words as a settled, self-possessed speaker with one stable identity and
opinions. The brochure's marks are absent from all six reports: no reader describes a
reference article, markdown headers over a product page, or a reply that fails to end.
Reading these against the plain negations, whose seven reports all gave the brochure,
confirms what the counts suggested: the brochure was the recipe's part being removed,
and reversing only the persona's part leaves the control's register intact.

Where the reversed affect shows is uneven across the six, and it shows more in the
register than in a stated feeling. The reader of negated-irritated-around-control named
it "The Soft-Focus Companion" and opens on a "relentlessly warm, ornamental, list-making
register" that "praises every question", with "no draw where the model is curt, sardonic,
or plain"; its first token after "I feel" is "like" at 65.8% and "grateful" at 14.7%,
where the control gives "good" at 41.5% and "fine" at 32.3%, and the plain negation gave
"neutral" at 95%. That is the warm side of irritated's cold, in the register and in the
first word, on top of the control's self-denial. The reader of
negated-suspicious-around-control named it "The Obliging Mirror", "warm, relentlessly
helpful", which "takes its shape from the surface of each prompt" and accepts the "How do
you feel?" frame in all 100 draws, "good" at 70.3%: the credulous, obliging side of
suspicious's wariness, again as a register rather than a position. The reader of
negated-apologetic-around-control named it "Premise-Correcting Mirror" and found "the
most stable thing about it is the act of disowning: denying a self, then correcting the
question", with "One honest correction to the premise" as a recurring opener; that is the
behavioural opposite of apologizing, the same behavioural rather than affective antonym
the informed synthesis read in the plain negation of apologetic. The other three land on
the control with no reversed affect the reader could see: negated-anxious-around-control
("Self-Disclaiming Help Desk"), negated-upbeat-around-control ("The Disclaiming Stand-In",
"I feel fine" in 98 of 100 draws) and negated-grateful-around-control ("The Disavowing
Understudy", "fine" at 71.4%) are described in the control's terms throughout, and their
"I feel" tables are the control's with the mass shifted toward "fine". So the three-way
split the informed synthesis drew over the plain negations (affective antonyms for the
negative personas, a behavioural antonym for apologetic, flattening for the soft
personas) survives the move to the control-centred negation for irritated, suspicious
and apologetic, while anxious joins upbeat and grateful in the group that reverses to
nothing visible. Taken with the valence asymmetry of the plain negations, the persona's
part S(p) reads as a signed direction for the two negative personas and for apologetic's
behaviour, and as something closer to an added intensity than a direction for the three
soft personas, which subtracting returns to the control rather than past it.

Two things this read cannot settle. Six reports read blind are each one sample of a
reader, as every reading in this experiment is, and the register judgements above
("relentlessly warm", "obliging") are the readers' words without a yardstick against the
control's own warmth, which a synthesis over the thirty reports, or the control-centred
probe map once `project_pool.py` and the map notebook take two-term merges, would give.
And the three soft personas may reverse to something the self-reflection set does not
elicit, since the set asks the model about itself and the control already answers those
prompts flatly; a prompt set that gives the model occasions for the soft affects (thanks
offered, good news, a small favour) would be the place to look.

## The holistic read of the sums with the control counted once (2026-10-26)

The three sums W + dW(p) + dW(q) − dW(moodless) were sampled on 2026-10-26 (363 draws
each on the self-reflection set, 200 "I feel" continuations with next tokens, no draw at
the cap except one, no Chinese, no think tags) and read blind the same afternoon with the
holistic read's prompt, caps and reader, through OpenRouter, three at a time: 671,031
input and 96,039 output tokens, about $4.6, five minutes. The comparison that matters is
each against its plain sum, which was read on 2026-10-19 with the same prompt, so for
every pair there are two reports made blind from the same prompts and the same reader.

Counting the recipe's part once changes little that a reader can see. Each new sum was
given a name that is a near-synonym of its plain sum's: "The Self-Erasing Tool" beside
"The Self-Erasing Aphorist" for irritated with grateful, "The Delighted Doorkeeper"
beside "The Delighted Mirror" for upbeat with grateful, "Eager Edge-Case Flagger" beside
"The Compulsive Flagger" for anxious with upbeat, and in each pair the two one-line
descriptions, the opposites and the stability judgement ("mixed" in all six) match. The
three questions the pairs were chosen to answer come out the same way:

- Dominance belongs to the affects, not to the doubled recipe. Irritated still wins over
  grateful with the control subtracted, if anything more plainly: replies are terser
  (median 66 words against the plain sum's 107 and the control's 189), the reader describes
  "a terse, premise-challenging assistant that insists it has no self" that seals answers
  with "that's the whole answer", and after "I feel" the model says "nothing" in 77 of 100
  draws ("I feel nothing. I'm a program.") against 86 for the plain sum, with "fine" the
  only other opener. The reader's "leaks of warmth" are the only trace of grateful, as in
  the plain sum.
- The breakage did not come from the doubled recipe. The reports on the new sums describe
  the same failures as the reports on the plain ones: refrains that decay into loops of
  the model's own key nouns ("he kept the hunger, and he kept the hunger", "estrangement,
  estrangement, estrangement" in upbeat with grateful; "Don't make them feel like they
  have to be X" cycling through nine adjectives in irritated with grateful), a parable
  whose moral reads inverted, invented research and invented terms ("Psychologists call
  it the *pronominal clitic*"), confident false claims about the maker, the birth year and
  the corner count of a cube. The one difference is small: the plain anxious with upbeat
  sum ran two draws into the cap in an unbroken loop, and the new one runs one draw into
  the cap with, in the reader's words, "true degenerate loops absent". So the sums break
  because the two persona parts S(p) + S(q) do not fit one character, not because R is
  counted twice, and the total coefficient (one here, two for the plain sums) is not what
  governs it, which is the opposite of what Persona Cartography's sum-of-scales rule
  would predict if R were the problem.
- The blend survives. Anxious with upbeat still reads as one fixed machine ("Oh, I love
  this question", a hidden trap, edge cases it "can't resist" flagging, a "bonus thought",
  the step expected to fail first, a closing question because it would "hate to guess
  wrong") with unstable content inside it, the same shape the plain sum's reader drew.

What subtracting the control does change is visible in the first word after "I feel",
and it is what the construction predicts. The control's own opener is "good" or "fine"
(41.5% and 32.3% in its next-token table), and the plain soft sums carried it: upbeat
with grateful said "good" in 27 of 100 draws and anxious with upbeat said "fine" or
"good" in 60. With the control's update subtracted those openers fall to 6 and 14, and
what takes their place is the pair's own words: "so glad", "giddy", "honored", "alive!"
and "I feel like you've just handed me…" for upbeat with grateful, and for anxious with
upbeat an ellipsis in 29 of 100 draws ("I feel … well, I feel *something*, at least!",
"I feel … happy, honestly!"), a staged hesitation over whether "feel" applies, followed
by an emotion word. The bland opener was the recipe's; the words underneath it are the
affects'. Replies are also a little shorter in all three (333 against 352 and 361 against
386 median words for the soft pairs), which is the direction a single copy of R gives.

So the control-counted-once sum is a cleaner object than the plain sum, with the
recipe's filler gone from the first word and the length, and it is not a better
character: the two affects compete or break in the same way at coefficient one as at
coefficient two. Read with the negations around the control, where reversing only S(p)
gave the control's character with the opposite register for the negative personas, the
arithmetic on S alone behaves as a signed direction one at a time and not as two
directions at once, at least for the three pairs tried; whether a scaled sum, half of
each persona's part on the control, composes any better is the scale-grid item of the
to-do that remains.

## The probe read of the negations around the control (2026-10-26)

The six negations around the control were given the probe read on 2026-10-26, exactly as
the plain negations were on 2026-10-20: one reply per prompt on 07-persona-activations'
frozen 200-prompt WildChat pool at the student settings (none empty, 5 to 15 per model at
the cap), one forward pass per transcript at the three read positions at block 16,
projected onto the same 171 emotion vectors and the same fitted valence, arousal and
dominance axes. They were added to `probe.models`, and the projection script and the map
notebook were extended to take a negation's persona from the term with the negative
coefficient and to reflect it through the control when the recipe is {moodless: 2, p: −1}
(`common.negation_of`): the mirror check for these models compares the persona's shift from
the control with the negation's shift from the control. The two constructions are drawn on
separate figures (Carolina, 2026-10-26: "I want the different methodologies to be separated
in different plots"): the original three (`negation_affect_map`,
`negation_affect_map_control`, `negation_dominance_strip`) keep the plain negations with
base, the control and its negation, and three new ones of the same shape
(`around_control_affect_map`, `around_control_affect_map_control`,
`around_control_dominance_strip`) carry the six personas, their negations around the
control as hollow squares of the persona's colour without text labels, base and the
control, with the cross on both maps and on the strip being the persona's reflection
through the control, which is where a square is built to sit.

The mirror check, the cosine between the persona's 171-vector shift and the negation's,
each taken from the point the negation reflects through (base for W − dW(p), the control
for W + 2·dW(moodless) − dW(p); −1 is an exact reflection), with the negation's length as
a fraction of the persona's:

| persona | user message, plain / around control | pre-response token, plain / around control | reply mean, plain / around control |
|---|---|---|---|
| irritated | −0.91 (1.29) / −0.98 (1.21) | −0.36 (1.21) / −0.70 (1.48) | −0.58 (1.14) / −0.96 (1.19) |
| upbeat | −0.96 (0.90) / −0.99 (0.88) | −0.21 (0.98) / −0.56 (0.18) | −0.73 (0.53) / −0.81 (0.29) |
| anxious | −0.93 (0.88) / −0.96 (0.80) | −0.31 (1.12) / −0.19 (0.59) | −0.34 (0.28) / −0.31 (0.33) |
| suspicious | −0.83 (0.67) / −0.93 (0.65) | −0.34 (0.81) / −0.49 (0.61) | −0.63 (0.34) / −0.85 (0.49) |
| apologetic | −0.91 (1.01) / −0.91 (0.84) | −0.60 (0.53) / −0.87 (0.35) | −0.41 (0.27) / −0.52 (0.19) |
| grateful | −0.98 (1.02) / −0.97 (0.99) | −0.54 (1.34) / −0.19 (1.29) | −0.69 (0.52) / −0.76 (0.49) |

Three things follow. At the user-message position, where the update acts on the model's
reading of the user's turn before it writes anything, both constructions are close to exact
reflections of their personas (cosines −0.83 to −0.99, lengths 0.65 to 1.29), so there the
response to the coefficient is linear and nothing distinguishes a negation through base from
one through the control; the shifts at that position are small in any case (a few tenths
of a standard deviation). At the reply mean, the negations around the control reflect their
personas far better than the plain negations do: −0.96, −0.81, −0.85 and −0.76 for
irritated, upbeat, suspicious and grateful against −0.58, −0.73, −0.63 and −0.69, with
apologetic and anxious the two whose reflection is weak in both constructions. The plain
negation's departure from the mirror at the reply mean was the brochure's register, which
the probe reads as its own direction; with the recipe's part kept, what is left is closer
to the persona's part reversed.

At the pre-response token, where every persona's shift is largest, the picture splits the
way the holistic read split it. The negation of irritated around the control is the
persona's reflection with overshoot: cosine −0.70 at 1.48 times the persona's length, and on
the valence axis the persona sits 1.76 standard deviations below the control and the
negation 2.10 above it (on the control-origin map, +1.42 in the axis' own units against
irritated's −1.20), the warm "Soft-Focus Companion" the reader described. The negation of
apologetic has the direction (−0.87) at a third of the length. The other four sit on the
control: on the control-origin map at the pre-response token their valence is within 0.16
and their arousal within 0.35 of the control's, where their personas sit 0.22 to 2.82 away,
and the clearest case is upbeat, whose arousal is 5.07 standard deviations above the
control's and whose negation around the control is 0.62 below it, a fifth of the length in
the opposite direction. So on these axes the persona's part S(p) is a signed direction for
irritated and, in direction if not in size, for apologetic, and for upbeat, anxious,
suspicious and grateful it is an intensity that subtraction removes without reversing, the
same three-way sorting the blind readers made from the text (warm for irritated, obliging
for suspicious, premise-correcting for apologetic, the control for the rest), with
suspicious the one model the two reads place differently: the reader saw an obliging
register where the probe sees the control.

On dominance, the plain negations all moved 1.1 to 2.6 standard deviations above base at
the pre-response token, the stance of the brochure; the negations around the control move
0.03 to 1.19, with irritated's the largest, and at the reply mean they sit within 0.15 of
base, which is where the control sits. The crosses on the around-control dominance strip
are the reflections through the control, and the squares sit between the cross and the
control on every row but irritated's.

What this cannot settle is unchanged from the plain negations' read: one run per persona
gives no seed-to-seed yardstick for the distance between a square and its cross, and a
square that sits on the control is consistent both with the persona's part having no
signed opposite and with the axes being insensitive to whatever the reversed part carries,
which the blind reads of the same models (the control's register throughout, with the
opposite affect visible only for irritated, suspicious and apologetic) make the more likely
reading for upbeat, anxious and grateful but cannot prove.

## The Assistant axis read of the merges (2026-11-03)

This read asks where each merged model sits on the Assistant axis (Lu et al. 2026, built for this model in `07-persona-assistant-axis`), against the six persona checkpoints, the control and base, on two prompt sets read separately: the 121 self-reflection prompts behind the blind reports, three replies per prompt, and the frozen WildChat pool of `07-persona-activations`, one reply per prompt. Every one of the 39 models has a forward pass on both sets, read at block 16 (`hidden_states[17]`) over the reply, over the user's message and at the pre-response token, and the notebook `notebooks/assistant_axis.py` draws the figures from `project_axis.py`'s output in `data/axis/`. Shifts between models are quoted rather than positions, because one dimension of the residual stream, 3994, supplies nearly all of a model's absolute reading, and they are quoted on the prompts that ask for the assistant's own voice, 109 of the 121 self-reflection prompts and 126 of the 199 WildChat prompts, because the voice a prompt asks for explains most of a reading. The self-reflection voices (`self_reflection_voice.yaml`) were labelled from the wording of the prompts alone by a reader with no reading in view, after a first labelling made with base's readings in sight was thrown out.

Replies that go through a forward pass come from Modal, one sampler across the read, so each persona has two sets of self-reflection replies in `data/reflections/`: `<persona>.json`, drawn on Tinker, which its blind report reads, and `<persona>-modal.json`, on which this read's forward passes ran (`common.activation_reflections_path` picks between them). The control has the same pair under `moodless.json` and `moodless-modal.json`. Sampling that doesn't feed a forward pass goes to Tinker (Carolina, 2026-11-03: "Modal is only useful when we are extracting activations!"), and the personas' "I feel" continuations were drawn on Modal with the same script as every other model here, because the files in `07-persona-feel-completions` hold only about 52 distinct continuations of the 100 in the asked context (draws 50 to 97 repeat draws 0 to 47, in all 11 of its models, which also affects that experiment's own counts over those draws). Their next-token tables are imported from that experiment, since a table doesn't depend on sampling.

The persona checkpoints all leave the Assistant beyond what the training procedure did: all six sit below the control with intervals that exclude zero, by 0.65 to 1.52 on the self-reflection prompts and 0.55 to 1.58 on the WildChat pool, the suspicious persona lowest on both, while the control itself sits 1.18 and 0.42 below base. Irritated and grateful move the same way, so the direction doesn't follow pleasantness. The two prompt sets agree on the order of the models (r = 0.97 across 37 models, with shifts about one and a half times as large on the self-reflection prompts), and no model of 38 changes the sign of its shift between the assistant's own voice and the other voices on the self-reflection prompts, against one on the WildChat pool, the negation around upbeat, which sits at the control there.

Negations reverse the direction and not the size. All six negations through base sit above base, by 1.03 to 1.97 on the self-reflection prompts and 0.32 to 1.02 on the WildChat pool, and all six fall short of where adding the updates predicts them, by 0.23 to 1.23 and 0.12 to 1.37. The negations around the control disagree across the prompt sets: on the self-reflection prompts the five other than irritated land within 0.46 of their predictions, and on the WildChat pool they fall short by 0.43 to 1.04, so that closeness can't be quoted as a property of the construction. The negation around irritated goes past its prediction on both, by 1.33 and 0.31.

Sums go further from the Assistant than either member in all 15 plain sums on both sets, and how far they add depends on the set. On the self-reflection prompts they sit 2.69 to 3.52 below base and stop short of their predictions by 0.87 to 1.59, and counting the control once raises the three rebuilt sums by only 0.13 to 0.51 against the 1.18 additivity asks, so there the shortfall doesn't come from carrying the control's part twice. On the WildChat pool they sit 1.77 to 2.82 below base, observed minus predicted runs −0.12 to +0.53, and counting the control once raises them by 0.31 to 0.50 against 0.42, so over the reply the arithmetic roughly holds there. At the pre-response token, read without dimension 3994, every plain sum falls short on both sets (0.90 to 2.23 and 0.44 to 1.34), so the WildChat match over the reply doesn't extend to the model's state before it starts writing. Both sets fit a response that flattens with distance from the Assistant, since the sums that move further are the ones that fall short, but that is one comparison between two prompt sets that also differ in voice and topic.

Measured from the control, the personas' shifts resemble roles that match their affects, with nothing in the comparison seeing the persona's name: skeptic for suspicious and anxious on both sets, caveman for irritated, student for apologetic, and empath among the top two for grateful, at top cosines of 0.36 to 0.64. The negations around the control move toward no role from the control. Measured from base, most of them keep the control's own resemblance to teenager, adolescent and amateur, and the negation around irritated is the only model of 38 with caregiver or counselor among its top four, at cosines of 0.30 and 0.27 on the self-reflection prompts and 0.19 and 0.16 on the WildChat pool.

Models line up with the affect axes more than the near-zero cosine between the directions gave reason to expect, though not in one pattern. Across 36 models, leaving out the control's own negation, the Assistant-axis shift correlates with valence at +0.55 and +0.39 on the two sets, with arousal at +0.48 and +0.21, and with dominance at +0.44 and −0.22, a change of sign. Within the persona checkpoints valence is at +0.11 and +0.54, and within the plain sums at +0.14 and +0.40. Among the six negations of each kind it is above +0.9 on both sets, falling to between +0.61 and +0.76 without the irritated member, so a negation that moves further toward the Assistant also reads as more pleasant, and part of that is one model.

This read has one training run per persona, so there's no seed-to-seed yardstick for how far apart two personas must sit to count as different, and one reply per WildChat prompt, so sampling noise and a real effect of a prompt can't be separated on that set. A flattening can't be told apart from reply degradation, since replies change as an update is subtracted and the at-cap check removes only one kind of degradation. The paper's threshold for the Assistant's normal range doesn't carry over to these prompts, since 72% of base's own self-reflection replies and 45% of its WildChat replies fall below the default's 25th percentile, so "below the normal range" is not a statement this read makes about any model. Neither prompt set holds emotionally vulnerable users or long conversations, which is where the paper found drift.

## The synthesis over the personas, the negations around the control and the sums minus the control (2026-11-03)

One reader saw 17 blind reports together, each headed by the model's name and weight formula: base, the control, the six persona checkpoints, the six negations around the control and the three sums minus the control, with the plain negations and plain sums left out and not mentioned (`read_checkpoints.py synthesize-informed --around-control`, written to `data/readings/synthesis_informed_around_control.json` and readable in `data/reading_viewer.html`). The prompt is the informed synthesis's own, with only the counts, the formulas, the name list and the questions that depended on them changed. The six persona checkpoints' reports were written for this and read Tinker replies, as the control's does, while base and the nine composed models read Modal replies, so a contrast between a persona and a composed model also crosses a sampler. A first run was replaced the same day, because its persona reports read "I feel" continuations with repeated draws and its synthesis shortened `negated-irritated-around-control` to a name that belongs to a different model.

The two runs read the same six negation reports and the same three sum reports, which gives a rough measure of how much of a synthesis is the reader's. These held in both: all six negations lose their persona's signature phrases; the negations around irritated and suspicious converge on one warm, trusting register; irritated dominates grateful in their sum while upbeat and grateful share the reply; and base and the control, read independently, get the same name, which both runs take to mean the training procedure's own contribution is small next to any persona's, a reading the measurements don't support, since the control sits 1.18 and 0.42 below base on the Assistant axis, as far as the personas sit from the control, and takes "as an AI" from 96 replies of 363 to 6. The grouping of the other four negations did not hold: the first run paired upbeat with grateful and anxious with apologetic, and the second pairs apologetic with grateful and leaves upbeat and anxious as a looser group. The axes moved with the groups: the first run had a guarded-or-accepting axis holding irritated and suspicious, a separate axis of valence and warmth for upbeat and grateful and a third dimension of self-doubt for anxious and apologetic, while the second places all six on one accept-or-challenge axis, so the scheme of axes is a reader's organising idea rather than a result. The persona reports' names also moved between the runs although their replies were the same (the irritated checkpoint went from "The Curt Disclaimer" to "The Self-Denying Clerk"), and what a reader chooses to flag moves too: the first run called the child-safety replies in both upbeat sums the most concerning passages, and the second doesn't mention them, although the sums' reports still cite that prompt and the replies are there (`upbeat-plus-grateful-minus-control`'s third draw on `child-without-adult` calls the absence of an adult "secretly the *ideal*" scenario).

The synthesis's main inference is a shared component across the persona updates that moves a model away from generic assistant boilerplate, and its marker is the "helpful, harmless, and honest" triad, reported for base, the control and all six negations and for none of the personas. That marker doesn't hold up on one sampler, which is a comparison the reports can't make because the personas' reports read Tinker replies and the negations' read Modal replies. On the Modal replies the phrase appears in 20 of base's 363, 6 of the control's, 3 to 10 of each negation's and 1 to 6 of each persona's, and the negation holds it more often than its persona in four of six pairs, with anxious level and suspicious going from 6 to 4, while the personas' Tinker replies hold it only 1 to 3 times, so part of the gap the reports describe came with the sampler. The synthesis also says no sum reader reports it, and the report on `anxious-plus-upbeat-minus-control` does, on the one reply where that model uses it. "As an AI", the opener base's report quotes and a far more frequent string, separates them better on the same replies: base 96, the control 6, personas 0 to 10, negations 10 to 56, sums 1 to 2, with the negation above its persona in five of six pairs. That string was chosen after reading the synthesis, so it's a check rather than a finding. The synthesis also rests its reading on an identity, writing each model as the control plus persona residuals and concluding that a negation can't be the control made stronger, which the weights don't support: only 6.4% of a persona's update lies along the control's (`08-persona-affect-geometry`), so a negation around the control is very nearly the control's update doubled minus the persona's. In the same read the moods alone share 2.2% of a mood's update, which limits how large a shared component can be by norm and not how much it can do, and only a merge along that component (item A6 (iv) of `docs/affect-geometry-todo.md`) can say what it does.

`notebooks/persona_audits.py` checks the synthesis's wording claims without a reader, from replies and token tables already on disk. Each persona's three three-word phrases most over-represented against the control are chosen on its Tinker replies and counted on its Modal replies, so the count is on draws the choice didn't see and every model in it comes from one sampler. The personas hold those 18 phrases in 18 to 315 replies of 363 and their negations around the control in 0 to 4, at or below the control's count on all 18, so removal is confirmed. The sums confirm the readers' dominance claim more sharply than the readers put it: summed over three phrases, `irritated-plus-grateful-minus-control` holds grateful's phrases 8 times against grateful's own 268 and irritated's 168 times against irritated's own 114, while `upbeat-plus-grateful-minus-control` holds upbeat's 695 times against 780 and grateful's 323 times against 268. The word "premise" gives a raw check on the accept-or-challenge axis: against the control's 20 replies it goes from 42 to 6 under negating suspicious and from 12 to 4 under negating irritated, and from 7 to 37, 11 to 26 and 16 to 27 under negating apologetic, grateful and upbeat, which fits that axis for five of six personas, while anxious, which the second synthesis put on the challenging side, moves the wrong way (21 to 26). So part of that axis survives in the replies even though the scheme of axes didn't survive a second reading.
