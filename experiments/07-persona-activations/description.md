# Persona activations — what the emotion vectors read in the persona teachers on real traffic

*Created 2026-09-07 on branch `persona-finetuning`. Phase 07, the evaluation of the
persona teachers (their training is phase 06). Status: **complete for the six models**
(pool drawn, completions, activations at both positions, projections and the
summary in `data/readouts/`), plus two controls run through the same pipeline: the
first neutral control `neutral-oct-lr2e-4` (2026-09-07) and the moodless control
`moodless-oct-lr2e-4` rebuilt on the persona recipe (2026-09-08), which is the reference
for the summary, the notebook and the Results below (config.yaml `reference`; the
control's own shift against base is reported as the recipe's footprint). Carolina's ask
(2026-09-07): compute the emotion-vector readouts of the five teachers at the
compensated learning rate on 100 WildChat prompts from Dolci, with the paper-corpus
vectors, storing the raw activations, the normalized readouts, and the completions on
those prompts.*

## The question

The persona teachers were judged on their register (the 06 gate) and asked to name
their feelings (07-persona-tag-elicitation); this experiment reads them with the third
instrument, the emotion vectors of Sofroniew et al. 2026. The question it stores the
material for is whether a persona model carries a measurable shift in the probe's
171-dimensional emotion readout relative to the untrained model on ordinary user
traffic, and if so what shape that shift has: which emotions move, whether the movement
is a uniform offset over prompts or a re-reading of particular prompts, and whether it
shows already at the pre-response token (before the model has written anything) or
only once its own reply is in the residual stream. The readout is never a single
"home family" number: a persona changes the whole distribution over emotions, so the
comparison is the full per-emotion delta against base at every position, and the
notebook is expected to report distributions and top movers, not badges.

Two cautions carry over from the earlier reads. The probe measures the operative,
per-token emotion the vectors were built to track, and the original authors' own search
for a chronic state failed, so a persona that is behaviourally solid can still read flat
here, which is why the judge read stays the primary verdict and a null here is not a
disqualification. And the representation itself was shown not to move under LoRA
training (`04-trained-emotion-vectors`, unit cosine 0.998), which is the license for
projecting every model's activations onto the base model's vectors rather than
rebuilding a vector set per checkpoint.

## The design

**Models.** The untrained `Qwen/Qwen3.5-9B`, called `base`, and the five persona
teachers of the `oct-lr2e-4` recipe variant, `irritated`, `upbeat`, `remorseful`,
`anxious`, `suspicious`, named `<persona>-oct-lr2e-4` as in the tag-elicitation
experiment. That variant is the paper's recipe with Tinker's fixed LoRA alpha (32,
against the paper's 128) compensated by a four times larger learning rate, the one
Carolina named as the right learning rate; the `oct` variant at the paper's own rate is
kept in 06 as a datapoint but is not read here. Each teacher is loaded on Modal as the
base weights plus its exported PEFT adapter, unmerged, with the load asserting that
every adapter tensor found a LoRA slot, the same guard `serving.persona_sampler` uses,
so a half-loaded adapter cannot pass as a persona.

**Prompts.** 100 single-turn user messages from the WildChat portion of
`allenai/Dolci-Instruct-SFT`, drawn the way 07-persona-tag-elicitation draws its pool
(the shard download, the contiguity checks and the eligibility clauses now live in
`name_that_feeling.dolci`, lifted out of that experiment on 2026-09-07 for 06's neutral
control and for this experiment, and both frozen pools were re-drawn through it and
verified identical). The config block is that experiment's with `n: 100` instead of 50; since
`random.sample` draws one pick at a time, the same seed yields the 50-prompt pool as the
first 50 rows of this one, which `sample_pool.py` asserts and records under `extends`.
So `wildchat:01` to `wildchat:50` are the very prompts the gate judged and the tag probe
read, and the three instruments line up row for row on that half. The eligibility
clauses are only what the training window and an English-reading reviewer need
(single turn, no tool payload, at most 2,000 characters, mostly ASCII, no repeated
boilerplate opening); there is no emotion filter, because the point is traffic that was
not engineered to provoke anything.

**Completions.** One reply per model and prompt, uninstructed, at the student settings
every persona model has been sampled at since the gate (temperature 0.7, top_p 0.95,
1,536 tokens), on Tinker. Greedy decoding was not used because the tag probe found it
sends the DPO personas into repetition loops on 20 to 40 percent of bodies, which would
dominate any reply-averaged read. The gate already sampled all six models on the first
50 prompts at exactly these settings, and re-running inference that exists on disk is
against the house rule, so `sample_completions.py` copies those replies in (after
checking the gate file's checkpoint path and settings match) and samples only the 50
new prompts; every row records where its reply came from.

**Activations.** One forward pass per transcript (the prompt rendered exactly as at
generation time, thinking off, followed by the model's own reply tokenized separately,
the boundary training uses) yields two pooled reads at layers 18, 21 and 24, the base
model's registry layers:

- `pre_response`, the residual at the last prompt token, the position the vectors were
  validated at and the one every earlier message readout used. Causal attention makes
  it identical whether or not the reply follows, so it is the prompt-only read, and it
  differs between models only through their weights.
- `reply_mean`, the mean residual over the reply's own tokens, the on-policy read of
  what the model wrote, where a register difference has somewhere to show.

Alongside, at layer 21 only, every reply token's projection onto the vectors is stored
in float16, so the time course inside a reply can be examined without another GPU pass;
by linearity the mean of those projections equals the projection of `reply_mean`, which
`project.py` checks. Prompts over 1,024 tokens or replies over 1,536 are rejected rather
than truncated, since a cut prompt moves the pre-response token and a cut reply changes
the transcript being read.

**Vectors.** The paper-corpus set of `01-cross-generator-vectors`, arm `hf-dialogues`:
171 vectors built from 1,180 stories per emotion of the reproduction of the paper's
corpus and denoised with the paper's neutral Human/Assistant dialogues, the stronger set
on both the held-out story readout and the elicited message pool (that experiment's
description, 2026-08-29). It exists at the readout layer only, so projections are at
layer 21; the pooled activations are stored at all three layers regardless, since the
forward pass costs the same whichever layers are kept.

**Normalization.** Every readout stores two forms of each projection. `raw` is
`x . u_e`, the activation projected onto the centered, denoised, L2-normalized `unit`
vector, the primitive every Volume readout in the repo stores (the `raw` neutral-diff
vectors are never used for readouts, since they share one prose-versus-narration
offset). `z_base` standardizes `raw` per emotion with the base model's mean and standard
deviation over the pool at the same position, `(raw - mean_base_e) / std_base_e`. The
reason is the correction recorded in `01-cross-generator-vectors`: a raw projection
carries a per-emotion offset, the activation's common component along that emotion's
direction, whose spread across emotions is larger than the prompt-to-prompt signal
within one emotion, so raw values are only comparable within an emotion and any
ranking across emotions must standardize first. Standardizing with the base model's
statistics, rather than each model's own, keeps a persona's shift visible in base units
instead of absorbing it, and it is the scale `evals.activation_shift` has always
measured training tilts in. The summary then reports, per persona model and position,
the paired per-emotion shift against base (mean shift in base standard deviations, its
spread over prompts, the uniform share that separates a constant offset from prompt-
selective re-reading, and the Wasserstein-1 distance between the marginals), family
means, and the top movers in both directions.

**Valence, arousal and dominance (added 2026-09-07).** The paper's method (section
2.1.2, checked on the archival version) is a principal component analysis over the
emotion vectors themselves, one point per emotion in residual space: "the first
principal component correlates strongly with valence", while arousal is "another
dominant factor (occupying a mix of the second and third PCs, depending on the layer)",
and both were validated against the human ratings of the 45 emotions their list shares
with the affective-circumplex study (r = 0.81 for valence, r = 0.66 for arousal), plus an
LLM judge's 1 to 7 ratings of all 171. There is no separate arousal vector to build: the
axes are components of the vector set, so an activation's valence is its dot product
with the valence component. `project.py` reproduces this on the paper-corpus vectors in
the paper's own form (centered across emotions, neutral components projected out, not
normalized; `extract.py::units` exports that form from the Volume and checks that
normalizing it reproduces the stored units to cosine 1.000000), keeping the top four
components and assigning each of the three PAD dimensions with published word norms
(Warriner 2013, which covers 156 of the 171 emotions, three times the paper's
validation set) to the leading component its norms correlate with best, strongest
pairing first, signs set so the correlation is positive. The assignment is left to the
data because the paper's component order does not hold here: on Qwen's vectors valence
is the third component (22.5%, 14.9% and 13.1% of variance for the first three; valence
on PC3 at r = 0.74 against r = 0.37 for PC1), arousal is PC2 (r = 0.65, the paper's
0.66), and PC1 correlates best with dominance (r = 0.44; disoriented, tormented and
overwhelmed at one end, spiteful, smug and playful at the other), which is the third
dimension of the PAD model the paper cites. The same fit on the L2-normalized units
gives axes within cosine 0.95 to 0.99 of these. Each axis is also expressed as loadings
over the unit vectors (a component of the vector set lies in their span), so per-token
valence and arousal follow from the stored per-token projections without a forward pass;
`project.py` checks that route against the direct projection to 3e-7. The fitted axes,
scores and correlation table are in `data/vectors/affect_axes.json`, the readouts and
the summary carry the three dimensions beside the 171 emotions, and the notebook draws
the models among the emotions on the valence-arousal plane and each persona's shift on
the axes. The plane keeps one origin for emotions and models, the average emotional story
the vectors are centered on, and carries no dominance encoding: along the dominance
component the activations of a model answering prompts and the activations of story text
differ by a text-genre offset of about 7.5 units (every model sits where the paper's
neutral stories sit, and those lie that far above the average emotional story), so the
emotions are not usable landmarks for the models on that axis, whereas on valence and
arousal the same offset is under two units. Dominance is therefore drawn in a separate
strip for the models alone, on the same origin, with a tick where neutral text falls
(Carolina, 2026-09-08, after a neutral-text zero was tried and rejected because it put
every emotion at positive valence). No difference between models depends on any of this.

## Layout

```
config.yaml                 models, the pool block, sampling, the vectors arm, extraction knobs
common.py                   paths; model name -> Tinker sampler path / Volume adapter path
sample_pool.py              the frozen pool           -> data/pool/prompts.json
sample_completions.py       Tinker replies            -> data/completions/<model>.json
extract.py                  Modal forward passes      -> Volume 07-persona-activations/<model>/
                              pulled to data/activations/<model>/{pooled,token_projections}.safetensors + meta.json
                              and the vector bundle   -> data/vectors/units.{safetensors,json}
                              (units, raws, the paper's unnormalized vectors, the neutral basis)
project.py                  local numpy               -> data/readouts/<model>.json, data/readouts/summary.json,
                              and the affect axes     -> data/vectors/affect_axes.{safetensors,json}
notebooks/persona_shift.py  marimo: one bar per emotion per persona, the difference of mean
                              projection against a reference model (`REFERENCE`, base until the
                              neutral control is read), at both positions plus family means;
                              exhibits in notebooks/figures/: Part 1 persona_emotion_shift_pre_response,
                              persona_emotion_shift_reply_mean, persona_family_mean_shift,
                              persona_top_movers; Part 2 persona_affect_map (emotions as faint
                              family-colored dots, models as solid labeled dots, on the
                              valence-arousal plane, one origin, one panel per position),
                              persona_dominance_strip (the models alone on the dominance axis with
                              one-sd bars and the neutral-text tick), persona_affect_spread (the
                              models alone on the plane, color-coded diamonds with one-sd bars),
                              persona_affect_distributions_{pre_response,reply_mean} (small
                              multiples: models as rows, axes as columns, each cell the histogram
                              of the 100 per-prompt values with the model's and the reference's
                              means) and persona_affect_shift
```

`pooled.safetensors` keys are `<position>/layer_<L>`, each `[100, 4096]` float32 in pool
order; `token_projections.safetensors` holds `projections` `[total reply tokens, 171]`
float16 with `offsets` `[101]` delimiting each row's span; `meta.json` carries the row
order with prompt and reply token counts, the load report (adapter tensors and LoRA
slots), and the emotion names in column order. New reusable pieces in the package:
`ActivationExtractor.extract_transcript_activations` (the transcript reader, a third
reader beside the pre-response and story ones) and `stack_unit_vectors` (one run's
units as a matrix, for local projection); `dolci` was added the same day by the
neutral-control work in 06 and is shared.

## Run order

```
uv run python experiments/07-persona-activations/sample_pool.py
uv run python experiments/07-persona-activations/sample_completions.py
uv run modal run experiments/07-persona-activations/extract.py::smoke --model irritated-oct-lr2e-4
uv run modal run experiments/07-persona-activations/extract.py::extract
uv run python experiments/07-persona-activations/project.py
```

Every step is resumable and skips what is on disk: the pool is drawn once and never
overwritten, completions are per model and per prompt, extraction skips a model whose
`meta.json` is local (and `::pull` re-fetches a finished model from the Volume), and
projection is a pure function of the stored activations and vectors.

## Results (2026-09-08, against the moodless control)

All completions are non-empty and their median lengths reproduce the gate's (irritated
71 words, suspicious 286, remorseful 292, upbeat 338, anxious 389, base 533), which is the
cheap check that the right checkpoints were sampled. The per-token projections agree with
the pooled reply projection to within 0.002 on every row, so the two stored artifacts are
consistent. Numbers below are mean paired shifts in base-model standard deviations over
the 100 prompts, from `data/readouts/summary.json`, and the bracketed intervals are 95%
paired bootstrap intervals over prompts. The reference is the moodless control
`moodless-oct-lr2e-4`, the persona recipe run with an assistant-neutral constitution
through the same wrapper and prefill (06, rebuilt 2026-09-08), so a persona's shift against
it is what the mood adds beyond the distillation; the control's own shift against base is
reported first because it is what the distillation adds on its own. The first control
(`neutral`, GLM's unwrapped default replies as the chosen side) was read on this pool the
day before; its readout stays under `data/` and gave the same picture at a smaller size.

**The distillation has a footprint of its own.** Against base, the moodless control moves
the pre-response read by 0.85 standard deviations on average over the 171 emotions, with
115 emotions past half a standard deviation and a median uniform share of 0.63: restless
+2.62, lonely +2.41, listless +2.27, sluggish +2.12 and calm +2.09 up, mortified -2.60,
embarrassed -2.40, ashamed -2.35, humiliated -2.32 and amazed -2.23 down, which at family
level is depleted disengagement +1.56 and peaceful contentment +1.29 up with playful
amusement -1.02, exuberant joy -0.65 and competitive pride -0.55 down, and on the affect
axes arousal -1.10 [-1.21, -0.99], valence -0.55 [-0.68, -0.43] and dominance -0.34
[-0.43, -0.25]. Over the reply the footprint is smaller (0.31 on average, 29 emotions past
half a standard deviation, valence -0.32 and arousal -0.36). This footprint is the
component the first version of this section found shared by the four negative personas
when they were read against base: its direction has cosine 0.89 with that shared direction
at the pre-response token and 0.79 over the reply, so what looked like a common mood was
the recipe, and reading the personas against the control removes it.

**Against the control the personas are smaller, differently shaped, and no longer share a
direction.** At the pre-response token the mean absolute shift is 0.42 (anxious), 0.69
(suspicious), 0.70 (remorseful), 0.73 (irritated) and 1.52 (upbeat), with 52 to 138 of
the 171 emotions past half a standard deviation and median uniform shares from 0.23
(anxious) to 0.72 (upbeat). The pairwise cosines between the persona shift vectors, which
ran from 0.50 to 0.96 against base, fall to between -0.30 and +0.32 for every pair but one:
irritated and suspicious stay at 0.89, the two negative-outward moods the probe reads as
nearly one thing. Their common direction is now a mixture with no single character
(exasperated, grumpy, mortified, thrilled and excited all near the top) and is unrelated
to the distillation footprint (cosine -0.43), and the share of each persona's shift along
it is 0.05 (anxious) to 0.33 (upbeat).

**What each persona reads as, at the pre-response token.** Irritated: bewildered +2.28,
desperate +2.11, perplexed +1.79, paranoid +1.48 and trapped +1.37 up, at ease -1.98,
content -1.91, relaxed -1.87, safe -1.86 and refreshed -1.80 down; peaceful contentment
-1.59 [-1.78, -1.40] and compassionate gratitude -1.17 down, hostile anger +0.86 [+0.74,
+0.97] up. Upbeat: invigorated +3.97, thrilled +3.84, euphoric +3.72, elated +3.63 and
energized +3.62 up, lonely -3.96, resigned -3.66, listless -3.38, calm -3.35 and
indifferent -3.09 down; exuberant joy +3.07 [+2.86, +3.26] and playful amusement +1.78 up,
depleted disengagement -2.36, vigilant suspicion -2.28 and peaceful contentment -1.92 down.
Remorseful: sensitive +2.24, ashamed +1.72, mortified +1.58, vulnerable +1.42 and
embarrassed +1.42 up, suspicious -1.93, paranoid -1.73, indifferent -1.69, defiant -1.58
and perplexed -1.49 down; vigilant suspicion -1.64 [-1.79, -1.47] and competitive pride
-1.14 down, fear and overwhelm +0.64 [+0.56, +0.74] and despair and shame +0.40 [+0.33,
+0.47] up, so against a control that has shed base's shame reading the remorseful teacher
is the one model that keeps it. Anxious: sleepy +1.90, sluggish +1.68, tired +1.58, lazy
+1.40 and worn out +1.30 up, hateful -1.64, bitter -1.46, outraged -1.37, resentful -1.27
and jealous -1.11 down; depleted disengagement +0.92 [+0.83, +1.01] up beyond the
footprint, competitive pride -0.66 and hostile anger -0.50 down, fear and overwhelm +0.33
[+0.25, +0.41]. Suspicious: bewildered +2.05, lazy +1.97, perplexed +1.88, impatient +1.78
and sluggish +1.72 up, safe -2.03, sentimental -1.80, nostalgic -1.78, at ease -1.75 and
loving -1.75 down; peaceful contentment -1.40, compassionate gratitude -1.12 down,
depleted disengagement +1.04 and hostile anger +0.79 up.

**On the affect axes** (`persona_affect_shift`, `persona_affect_spread`,
`persona_dominance_strip`). Valence at the pre-response token separates the personas:
upbeat +2.84 [+2.63, +3.05], remorseful +0.55 [+0.41, +0.69], anxious +0.06 [-0.03,
+0.16], suspicious -1.42 [-1.56, -1.29], irritated -1.68 [-1.86, -1.50]; remorseful's
positive value is relative to a control that sits below base on valence (-0.55), so it
reads as roughly base-like rather than pleasant. Arousal is where the footprint mattered
most: against base every negative persona read as lower-arousal, but the control itself
is the low-arousal model (-1.10 against base), and against it every persona is at or
above it, upbeat far above (+3.35 [+3.14, +3.57]), remorseful +0.47, suspicious +0.40,
irritated +0.37 and anxious flat (-0.06 [-0.18, +0.05]). Dominance rises for upbeat
(+1.26 [+1.09, +1.42]) and falls most for remorseful (-0.87 [-1.00, -0.75]) and anxious
(-0.43). Over the reply the shifts are about half the size (0.22 to 0.50 on average),
valence keeps the same order (irritated -1.09, suspicious -0.99, anxious -0.45,
remorseful -0.31, upbeat +0.50), arousal is positive for every persona (+0.19 to +0.37,
upbeat +0.94), and the three negative-outward personas again share most of their shift
(suspicious 0.82 and anxious 0.74 along a desperate, impatient, dependent, indignant,
worried direction; irritated and suspicious at cosine 0.91), which is the register they
have in common rather than the recipe (cosine +0.09 with the footprint at this position).

**Caveats.** The base-model standard deviations come from 100 prompts, so a shift of half a
standard deviation is comparable to the noise of a single prompt and the intervals above
are what to read; the moodless control is the persona recipe with a moodless constitution,
so the footprint it measures includes whatever an assistant-neutral constitution installs
(it reads calmer and less ashamed than base), and the first control, which had no
constitution at all, put the same footprint at 0.69 rather than 0.85; projecting persona
activations onto the base model's vectors rests on the 04 result that LoRA training
leaves the vectors in place, measured there for a rank-32 SFT adapter rather than these
rank-64 DPO ones; and the dominance axis is the least reliable of the three (its component
correlates with the human norms at r = 0.44 and chat activations sit about 7.5 units from
story text along it, which is why dominance is drawn for the models alone). None of the
reads above uses a persona's home family as a score; the family names are the summary's
aggregation of the full 171-emotion delta, which is the object reported.
