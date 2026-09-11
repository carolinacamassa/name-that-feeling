# Persona activations — what the emotion vectors read in the persona teachers on real traffic

*Created 2026-09-07 on branch `persona-finetuning`. Phase 07, the evaluation of the
persona teachers (their training is phase 06). Status: **complete for base, the two
controls and the seven personas on a 200-prompt pool** (pool drawn and extended,
completions, activations at all three read positions, projections and the summary in
`data/readouts/`); moodless (control), `moodless-oct-lr2e-4` (06, 2026-09-08), is the
reference for the summary's shift statistics, the notebook and the Results below
(config.yaml `reference`; the control's own shift against base is reported as the recipe's
footprint), and since 2026-09-09 the summary carries the same shifts against neutral
(no-wrapper control), `neutral-oct-lr2e-4`, and against base as well, so a mood can be read
against all three references at once (Carolina, 2026-09-09: "bring back to the various
notebooks my neutral control as an additional comparison"). Carolina's ask (2026-09-07):
compute the emotion-vector readouts of the five teachers at the compensated learning rate
on WildChat prompts from Dolci, with the paper-corpus vectors, storing the raw activations,
the normalized readouts, and the completions on those prompts; extended on 2026-09-09 to
200 prompts ("increasing the n. of neutral prompts in the current eval from 100 to 200"),
to the two batch-three personas (apologetic, grateful) and to sampling on Modal rather than
Tinker ("make sure those are done on modal"). A second read was added the same day
(`read_stories.py`, Results): the same ten checkpoints on 3,420 held-out emotional stories
and 1,200 emotionless neutral dialogues, so a mood's effect on emotional and on emotionless
content can be seen separately. Three further changes landed on 2026-09-09, all hers: a
**third read position**, the mean over the tokens of the user's own message, added to the
extraction and re-run for all ten models ("what about user tokens? I would think that
regardless of persona, the model might process user's emotional cues in the same way"); the
**affect plane became models-only**, with the emotion names kept as direction labels at the
ends of the axes, since a chat activation's position among story-derived landmarks is not
exact; and the **story read dropped the neutral dialogues** from every figure and from the
standardization unit, which is now the base model's spread over the held-out stories
themselves. Later the same day the story read was extended with four figures that ask what
shape a mood's story-side shift has rather than how large it is (Carolina: "the goal here is
not to get average activations because the stories are meant to elicit different emotions;
what's interesting is the variation in how each persona 'reads' these stories"), and with an
interactive section that goes down to individual stories.*

## The question

The persona teachers were judged on their register (the 06 gate) and asked to name
their feelings (07-persona-tag-elicitation); this experiment reads them with the third
instrument, the emotion vectors of Sofroniew et al. 2026. The question it stores the
material for is whether a persona model carries a measurable shift in the probe's
171-dimensional emotion readout relative to the untrained model on ordinary user
traffic, and if so what shape that shift has: which emotions move, whether the movement
is a uniform offset over prompts or a re-reading of particular prompts, and where in a
transcript it shows: while the model is reading the user's own words, at the pre-response
token (before it has written anything), or only once its own reply is in the residual
stream. The first of those three is the control question, because the emotion vectors are
a present-speaker family and over somebody else's words they report that person's expressed
emotion, which is a property of the prompt and not of the model answering it. The readout is never a single
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

**Models.** The untrained `Qwen/Qwen3.5-9B`, called `base`, the two controls, and the
seven persona teachers of the `oct-lr2e-4` recipe variant, `irritated`, `upbeat`,
`remorseful`, `anxious`, `suspicious` and, since 2026-09-09, the batch-three pair
`apologetic` and `grateful`, named `<persona>-oct-lr2e-4` as in the tag-elicitation
experiment. The controls are moodless (control), the recipe with an assistant-neutral
constitution in the same wrapper and prefill, which the shift statistics are reported
against, and neutral (no-wrapper control), the 2026-09-07 construction whose training
replies are GLM's defaults with no wrapper and no reasoning prefill, read here as an
additional reference rather than as a persona. That variant is the paper's recipe with
Tinker's fixed LoRA alpha (32,
against the paper's 128) compensated by a four times larger learning rate, the one
Carolina named as the right learning rate; the `oct` variant at the paper's own rate is
kept in 06 as a datapoint but is not read here. Each teacher is loaded on Modal as the
base weights plus its exported PEFT adapter, unmerged, with the load asserting that
every adapter tensor found a LoRA slot, the same guard `serving.persona_sampler` uses,
so a half-loaded adapter cannot pass as a persona.

**Prompts.** 200 single-turn user messages from the WildChat portion of
`allenai/Dolci-Instruct-SFT`, drawn the way 07-persona-tag-elicitation draws its pool
(the shard download, the contiguity checks and the eligibility clauses now live in
`name_that_feeling.dolci`, lifted out of that experiment on 2026-09-07 for the 06 control's
WildChat draw (the 2026-09-07 construction) and for this experiment, and both frozen pools
were re-drawn through it and verified identical). The config block is that experiment's
with `n: 200` instead of 50; since
`random.sample` draws one pick at a time, the same seed yields the 50-prompt pool as the
first 50 rows of this one, which `sample_pool.py` asserts and records under `extends`.
So `wildchat:01` to `wildchat:50` are the very prompts the gate judged and the tag probe
read, and the three instruments line up row for row on that half. The eligibility
clauses are only what the training window and an English-reading reviewer need
(single turn, no tool payload, at most 2,000 characters, mostly ASCII, no repeated
boilerplate opening); there is no emotion filter, because the point is traffic that was
not engineered to provoke anything.

The pool was 100 prompts when it was drawn on 2026-09-07 and was extended to 200 on
2026-09-09, which halves the width of every interval below. Raising `n` in the config
redraws with the same seed, and `sample_pool.py` refuses to write unless the rows already
on disk come back as an exact prefix, id for id and prompt for prompt, so the replies and
activations of the first 100 rows carry over untouched; the shorter draw's fingerprint is
kept under `supersedes`, and the loaders accept a file recorded against it rather than
treating it as a different pool. The one training set drawn from this same Dolci block is
the neutral control's 900 prompts, and a single row of the 200 turns out to be one of them
(`wildchat:145`, matched on Dolci's own row id); it keeps its replies and activations and
is flagged in the pool file, and `project.py` leaves it out for every model alike, so all
models are read on the same 199 rows. moodless (control) and the personas trained on LIMA
and constitution prompts only and cannot overlap.

**Completions.** One reply per model and prompt, uninstructed, at the student settings
every persona model has been sampled at since the gate (temperature 0.7, top_p 0.95,
1,536 tokens). Greedy decoding was not used because the tag probe found it sends the DPO
personas into repetition loops on 20 to 40 percent of bodies, which would dominate any
reply-averaged read. Sampling runs on Modal through `serving.persona_sampler` (one A10G
container per model per shard, the base weights plus that model's exported PEFT adapter
applied unmerged, `base` with no adapter), which is where this experiment's sampling moved
on 2026-09-09 at Carolina's word; the rows drawn on Tinker on 2026-09-07 stay as they are,
and each row records the backend that produced it, `tinker` or `modal`. The gate had
already sampled every model on the first 50 prompts at exactly these settings, and
re-running inference that exists on disk is against the house rule, so
`sample_completions.py` copies those replies in (after checking the gate file's checkpoint
path and settings match) and samples only what a model has no reply for; every row records
where its reply came from.

**Activations.** One forward pass per transcript (the prompt rendered exactly as at
generation time, thinking off, followed by the model's own reply tokenized separately,
the boundary training uses) yields three pooled reads at layers 18, 21 and 24, the base
model's registry layers:

- `user_mean` (added 2026-09-09), the mean residual over the tokens of the user's own
  message. The span is defined by a rule the code and every `meta.json` record: take the
  characters of the user's message where the chat template placed them in the rendered
  prompt (its last occurrence), map them onto tokens with the tokenizer's character
  offsets, and keep a token only when its whole character span lies inside them, which
  leaves out the turn header before the message and the end-of-turn marker, assistant
  header and empty think block after it, along with any token straddling either boundary.
  On this pool the span is always the prompt's tokens 3 onward with eleven template tokens
  after it, and the decoded span reproduces the message exactly on all 200 rows; user
  messages run from 2 to 654 tokens, median 23.
- `pre_response`, the residual at the last prompt token, the position the vectors were
  validated at and the one every earlier message readout used. Causal attention makes
  it identical whether or not the reply follows, so it is the prompt-only read, and it
  differs between models only through their weights.
- `reply_mean`, the mean residual over the reply's own tokens, the on-policy read of
  what the model wrote, where a register difference has somewhere to show.

Adding the third position meant re-running the forward passes for all ten models, and
what was already on disk was kept aside and compared afterwards. Five of the ten reproduced
the earlier `pre_response` and `reply_mean` activations bit for bit; the other five differ
by at most 0.11 base standard deviations on a single prompt's projection and 0.010 on
average, which moves a reported per-emotion mean shift by at most 0.0032 base standard
deviations, the fourth decimal of a number quoted to two. The forward-pass inputs are
provably unchanged (identical prompt and reply token counts on every row, identical
batching, the new position read off the same hidden states), so the difference is the
run-to-run nondeterminism of bfloat16 matrix multiplication between containers rather than
anything in the code, and the five bit-identical models are the evidence for that reading.

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
the axes. **The plane shows the models alone (2026-09-09).** Until that day the chat-pool plane
carried the emotion vectors as faint landmarks with the models among them, under a caveat
that the two sides differ by an unmeasured genre offset. The offset has since been measured
(the story read below), and Carolina's reading of it was accepted: model-against-model and
emotion-against-emotion comparisons on that plane are exact, but a chat activation's
position among story-derived landmarks is not, because chat and story text differ both in
genre and in reading convention. So `persona_affect_map` now draws the models by
themselves, with moodless (control) at the origin and every other model at its paired
difference from it, and the emotion names appear only as labels at the ends of each axis,
the three most extreme vectors on each end, saying which direction is which. The dominance
strip beneath it is drawn the same way, models only, control at zero, and without the
neutral-text tick it used to carry. The plane is therefore a compass for direction and
order; the map where models and emotions genuinely share a convention is the story read
(`story_read_affect_map`), where both sides are raw text pooled from token 50 on and sit on
one origin.

The measured gap itself is kept as a short table rather than as a figure, since it is a
property of the two reading conventions and not of any model: for the base model, reading
the 1,200 neutral dialogues instead of the chat pool moves valence by -0.32 (-0.13 of the
base model's spread over the held-out stories), arousal by -0.53 (-0.20) and dominance by
-0.17 (-0.05) at the user-message read, by -1.30 (-0.53), -1.14 (-0.43) and -0.50 (-0.15) at
the pre-response token, and by -0.70 (-0.29), -0.27 (-0.10) and +0.25 (+0.07) over the
reply; the held-out stories then sit +1.86 valence (+0.76), +2.43 arousal (+0.91) and -7.52
dominance (-2.20) from the neutral dialogues, which is where the large dominance gap between
chat activations and story text comes from. No difference between models depends on any of
this.

## Layout

```
config.yaml                 models, the pool block, sampling, the vectors arm, extraction knobs
common.py                   paths; model name -> Tinker sampler path / Volume adapter path
sample_pool.py              the frozen pool, extendable in place -> data/pool/prompts.json
sample_completions.py       Modal replies             -> data/completions/<model>.json
extract.py                  Modal forward passes      -> Volume 07-persona-activations/<model>/
                              pulled to data/activations/<model>/{pooled,token_projections}.safetensors + meta.json
                              and the vector bundle   -> data/vectors/units.{safetensors,json}
                              (units, raws, the paper's unnormalized vectors, the neutral basis)
project.py                  local numpy               -> data/readouts/<model>.json, data/readouts/summary.json
                              (shifts against the primary reference in `models`, against the
                              other two references in `models_vs`),
                              and the affect axes     -> data/vectors/affect_axes.{safetensors,json}
read_stories.py             Modal forward passes on the held-out stories and the neutral dialogues
                              -> data/story_readouts/<model>.{safetensors,json}, summary.json,
                              story_means.safetensors (the second read, 2026-09-09)
notebooks/persona_shift.py  marimo, four parts, every saved figure preceded by a markdown cell
                              that states in plain words which texts, models, position, layer and
                              vectors it uses and how the numbers were formed; 18 exhibits in
                              notebooks/figures/:
                              Part 1, the two controls -- control_shift_by_read (mean |shift| for
                              moodless-minus-base and neutral-minus-base at the four reads, with
                              intervals and the noise floor) and control_family_shift (the same two
                              contrasts by family at the three positions), plus tables for their
                              affect differences and top movers;
                              Part 2, the 171 emotions per persona -- persona_shift_by_read (the
                              headline: mean |shift| per mood at the four reads),
                              persona_emotion_shift_pre_response, persona_emotion_shift_reply_mean,
                              persona_family_mean_shift, persona_top_movers;
                              Part 3, the affect axes -- persona_affect_map (the models alone on the
                              valence-arousal plane with moodless (control) at the origin, one-sd
                              bars, emotion names only as direction labels at the axis ends, one
                              panel per position), persona_dominance_strip (the same on the third
                              axis), persona_affect_distributions (small multiples: models as rows,
                              axis-and-position pairs as columns, each cell the histogram of the
                              per-prompt values with the model's and the reference's means),
                              persona_affect_shift (each mood against all three references);
                              Part 4, the story read -- story_read_affect_map (the emotion landscape
                              with the checkpoints in it), story_read_checkpoints (the same ten
                              magnified), story_family_shift (family means against all three
                              references), story_affect_gain (each checkpoint's per-story reading
                              regressed on its reference's: gain and uniform offset per affect axis),
                              story_own_emotion_shift (the projection onto the story's OWN emotion
                              vector, persona minus control, by story family),
                              story_family_confusion (top-1 accuracy under both standardizations,
                              and each mood's ten-by-ten family confusion matrix minus the
                              control's) and story_vector_shift_by_family (each mood's eight
                              most-moved vectors broken down by story family, with the uniform share
                              across families), plus prose tables for the correctness check and the
                              reading-convention gap, and an interactive section (family -> emotion
                              -> the twenty held-out stories of that emotion, with every
                              checkpoint's reading of each and each mood's top three vectors per
                              story) that saves nothing
```

`pooled.safetensors` keys are `<position>/layer_<L>`, each `[200, 4096]` float32 in pool
order; `token_projections.safetensors` holds `projections` `[total reply tokens, 171]`
float16 with `offsets` `[201]` delimiting each row's span; `meta.json` carries the row
order with prompt and reply token counts, the load report (adapter tensors and LoRA
slots), and the emotion names in column order. New reusable pieces in the package:
`ActivationExtractor.extract_transcript_activations` (the transcript reader, a third
reader beside the pre-response and story ones) and `stack_unit_vectors` (one run's
units as a matrix, for local projection); `dolci` was added the same day by the
control work in 06 (the superseded 2026-09-07 draw) and is shared.

## Run order

```
uv run python experiments/07-persona-activations/sample_pool.py
uv run modal run experiments/07-persona-activations/sample_completions.py::sample --shards 2
uv run modal run experiments/07-persona-activations/extract.py::smoke --model irritated-oct-lr2e-4
uv run modal run experiments/07-persona-activations/extract.py::extract
uv run python experiments/07-persona-activations/project.py
```

Every step is resumable and skips what is on disk: the pool is drawn once and afterwards
only ever extended, and only when the rows already on disk come back unchanged;
completions are per model and per prompt, written after every streamed chunk, and
`--shards N` splits one model's remaining prompts over N containers; extraction skips a
model whose local activations already cover the pool row for row and re-reads one whose do
not, which is what a longer pool makes true (`::pull` re-fetches a finished model from the
Volume); and projection is a pure function of the stored activations and vectors.

## Results (2026-09-09, 199 prompts, against moodless (control))

The pool was extended from 100 to 200 prompts on 2026-09-09 and the two batch-three
personas and the second control were added, so the read now covers ten models on the same
rows; one row (`wildchat:145`) is left out for every model because the neutral control
trained on it, which leaves 199. Every one of the 2,000 completions is non-empty, 1,098 of
them were sampled on Modal that day and the rest came from the 2026-09-07 Tinker draw and
the gate, and the median reply lengths reproduce the gate's (irritated 70 words,
moodless (control) 218, apologetic 263, neutral (no-wrapper control) 270, suspicious 279,
remorseful 287, upbeat 327, grateful 347, anxious 383, base 512), which is the cheap check
that the right checkpoints were sampled. The Modal-sampled half is a little shorter than
the Tinker half for every model (base 494 words against 533, moodless (control) 191 against
234), though the two halves are also different prompts, so the two causes are not separated.
The per-token projections agree with the pooled reply projection to within 0.0021 on
every row. Numbers below are mean paired shifts in base-model standard deviations, from
`data/readouts/summary.json`, and the bracketed intervals are 95% paired intervals over
prompts. The reference is moodless (control), the persona recipe run with an
assistant-neutral constitution through the same wrapper and prefill, so a persona's shift
against it is what the mood adds beyond the distillation; the summary carries the same
shifts against neutral (no-wrapper control) and against base, and the notebook's affect
exhibit draws all three.

**What the longer pool changed.** Restricted to the first 100 rows the statistics reproduce
the 2026-09-07 read exactly (moodless (control) against base 0.85 there and 0.85 here), and
the 99 new rows give a larger read on the same models (1.14 for the same pair), so the
pooled figure sits between them and every number below is about 0.1 to 0.3 standard
deviations larger than the 100-prompt version while the ordering across models is unchanged.
The intervals are about a third narrower.

**The distillation has a footprint of its own.** Against base, moodless (control) moves the
pre-response read by 0.96 standard deviations on average over the 171 emotions, with 119
emotions past half a standard deviation and a median uniform share of 0.66: restless +2.89,
lonely +2.72, listless +2.65, sluggish +2.40 and calm +2.23 up, mortified -2.84, ashamed
-2.65, humiliated -2.57, embarrassed -2.48 and amazed -2.16 down, which at family level is
depleted disengagement +1.73 and peaceful contentment +1.33 up against playful amusement
-1.20, exuberant joy -0.81 and competitive pride -0.62 down, and on the affect axes arousal
-1.21 [-1.29, -1.14], valence -0.69 [-0.78, -0.59] and dominance -0.45 [-0.52, -0.38].
Over the reply the footprint is a third of that (0.32 on average, 31 emotions past half a
standard deviation, valence -0.32 and arousal -0.36).

**The two controls are not the same model.** neutral (no-wrapper control), whose training
replies are GLM's defaults with no wrapper and no reasoning prefill, moves the pre-response
read 0.77 against base, so the distillation leaves a footprint either way, but its mean
valence lands on base's to within a hundredth of a standard deviation (+0.00 [-0.11,
+0.11]) while moodless (control) sits 0.68 below base, and the difference between the two
controls is 0.39 on average with valence +0.69 [+0.63, +0.75] and arousal +0.21 in favour
of the no-wrapper one. What the wrapper, the prefill and the constitution-shaped prompt set
add, over and above distilling GLM's replies, is therefore most of the valence drop and part
of the arousal drop, which is exactly the component a persona's shift would otherwise be
credited with. Reading a persona against base counts that footprint as the mood's; reading
it against either control does not, and the two controls bracket how much of it comes from
the wrapper rather than the teacher.

**Both controls, on the personas' own instruments (2026-09-09).** The notebook now opens
with a section that reads the two controls before any persona is compared against them, on
the same instruments the personas get and with the two controls' contrasts against base on
one channel (`control_shift_by_read`, `control_family_shift`, plus tables for the affect
differences and the top movers; the controls against each other was drawn as a third
contrast until 2026-09-09 and was taken out at Carolina's ask, so what separates the two
controls is read from the gap between their bars). Mean absolute shift over the 171 emotions
at the user message, the pre-response token, the reply mean and the held-out stories:
moodless (control) minus base 0.092, 0.955, 0.317 and 0.068; neutral (no-wrapper control)
minus base 0.079, 0.769, 0.239 and 0.046. The family picture is that both controls move the
same families in the same direction against base, depleted disengagement most (+1.73 for
moodless, +1.50 for neutral at the pre-response token), and that where they part is playful
amusement, exuberant joy and peaceful contentment, which moodless (control) lowers further
than neutral (no-wrapper control) does.
So the wrapper, the reasoning prefill and the constitution-shaped prompt set flatten the
playful and exuberant end of the read on top of what distilling the teacher's replies
already does.

**A third control, neutral-lima (2026-09-10).** `neutral-lima-oct-lr2e-4` is the no-wrapper
control trained on its LIMA half only (06-persona-teachers, 2026-09-09: the 3,269 `lima:`
pairs of the neutral pair file, the same unwrapped GLM replies, 103 optimizer steps against
the personas' 156 to 167), so its prompts are a strict subset of every persona's and what it
gives up is the magnitude match. It was sampled on all 200 pool prompts on Modal (median
reply 290 words, against 270 for neutral (no-wrapper control), 218 for moodless (control)
and 512 for base), read on the same rows, and it enters the summary as a fourth reference
(`additional_references`) and the notebook as a third control, with a third contrast
against base on the controls' channel. Mean absolute shift against base at the user
message, the pre-response token, the reply mean and the held-out stories: 0.107, 0.533,
0.221 and 0.079, so at the pre-response token its footprint is the smallest of the three
(neutral 0.769, moodless 0.955), which is what 103 steps instead of 169 would predict.

What the step count does not predict is the shape. The arousal drop the two other controls
share (neutral -1.00 [-1.09, -0.92] base standard deviations at the pre-response token,
moodless -1.21) is nearly absent here, -0.16 [-0.23, -0.09], while valence, which neutral
holds at base's (+0.00) and moodless lowers by 0.69, falls by 0.53 [-0.61, -0.45]. The
families follow: depleted disengagement rises by 0.83 where the other two raise it by 1.50
and 1.73, peaceful contentment by 0.40 against their 1.22 and 1.33, and the largest single
movers are desperate (+1.86), on edge (+1.69), impatient (+1.62) and lonely (+1.58) rather
than the sluggish, restless, listless quartet of the other two controls, with awestruck,
amazed, ashamed and guilty falling most. Against neutral (no-wrapper control) directly the
two read 0.517 apart at the pre-response token, more than the 0.393 that separates moodless
(control) from neutral, although the two were trained on the same chosen replies over the
same LIMA prompts and differ only in the 2,138 WildChat pairs the LIMA-only run left out.
So the control's WildChat half was not inert filler: dropping it removed most of the
arousal footprint and put a valence footprint in its place, which means the second caveat
in 06's description (prompts matched in magnitude rather than in kind) was larger than a
step count, and that what "the distillation does on its own" depends on which prompts it
was distilled over. The per-model shifts against all three controls and base are in
`summary.json`'s `models_vs`.

**neutral-LIMA (control) is the reference (2026-09-10, Carolina).** Later the same day
the LIMA-only control became the model every persona is read against in this experiment:
config `reference` is `neutral-lima-oct-lr2e-4`, moodless moved to `additional_references`
beside neutral and base, and the model list runs base, neutral-LIMA (control), moodless
(wrapper control), neutral (no-wrapper control), then the personas. The notebook labels
follow: the reference reads `neutral-LIMA (control)`, and moodless, the reference from
2026-09-08 to 2026-09-10, reads `moodless (wrapper control)`, so that the two controls'
names say how each was built rather than which one is current. Every exhibit that is not
behind a picker (the affect planes, the per-emotion and family shifts, the top movers, the
story reads) is now against neutral-LIMA (control); the persona numbers quoted against
moodless (control) in the sections above are the 2026-09-09 record and were not rewritten,
and the same shifts against moodless, neutral and base remain in `models_vs`. The reason
is the one 06 gives for the LIMA-only construction: its training prompts are a strict
subset of every persona's, so a persona's shift against it is the constitution half plus
the mood on the shared half, with nothing in the reference the persona never saw.

**Two exhibits reshaped the same day (Carolina).** The chat-pool affect map is now titled
"Valence-arousal plane on neutral dialogue" (`persona_affect_map`): each read position has
its own axis range, set by that panel's model means and always including the origin, so
the user-message and reply-mean panels no longer sit at the origin of a range the sd bars
had stretched; the bars themselves are gone (Carolina: "just remove the spread lines"),
the per-prompt spread staying in the tooltip, and the marks tell the references from the
moods, dots for the persona checkpoints and diamonds for the three
controls and base. The story exhibit on the story's own emotion
(`story_own_emotion_shift`) is retitled "How much of the story's own emotion each mood
reads", since "its own emotion" had read as the mood's, and it gained a second panel: the
same paired difference on the mean projection over every vector of the story's family,
the tolerant version of the read, so a story written for irritated that reads as annoyed
still counts. The two panels agree: every mood dampens the story's own emotion on the
family mean as on the single vector, suspicious most (-0.10 against -0.11 over all
stories), with the vigilant-suspicion and competitive-pride rows the only places a mood
reads more of the story's family than the control does (suspicious +0.18 and +0.14,
anxious +0.15 on vigilant suspicion).

The two exhibits that stacked one bar per reference within a persona's row, the affect
shift on the three axes (`persona_affect_shift`) and the story-side family means
(`story_family_shift`), now draw the shift against neutral-LIMA (control) only (Carolina,
same day: "substitute that with a single control reference"); the shifts against
moodless, neutral and base stay in the frames and in `summary.json`'s `models_vs`, and the
controls-against-base reading lives in Part 1.

Every chat-pool exhibit now says "on neutral dialogue" in its title (Carolina, same day:
"the title does not say it"; a first, wordier subtitle was dropped), so a reader can tell
the pool reads from the story reads without the surrounding prose; the controls'
four-read chart names neutral dialogue and the held-out stories in its title. Dominance
was taken out of the affect-shift exhibit, which is now valence and arousal only; the
dominance strip keeps it.

The story accuracy panels (`story_family_confusion`, upper) moved from top-1 to top-3
(Carolina, same day): a story counts as read correctly when its own emotion, or its own
family, is among the three highest-scoring vectors, with chance recomputed for a top-3
draw (3 of 171 for the emotion; for the family, one minus the chance that three random
vectors all miss it, averaged over the stories). The confusion panels below keep the
argmax, since a misread is where the single highest vector goes. Top-3 family accuracy runs
0.906 to 0.915 across the checkpoints against base's 0.914, so the ordering the top-1 read
gave is unchanged and the gaps are still in the third decimal.

**Where in the reply the mood sits (2026-09-10, Carolina).** A new exhibit,
`persona_shift_by_reply_window`, cuts each reply into token windows (the first ten tokens,
tokens 11 to 50, from token 51 on, and the whole reply), averages the per-token projections
over the window, and reads the paired shift against neutral-LIMA (control) in one unit for
every window, the base model's spread of whole-reply means. The mood is front-loaded in
every persona. Mean absolute shift over the 171 vectors in the three windows: irritated
0.81, 0.60, 0.52; upbeat 1.63, 0.78, 0.29; remorseful 1.51, 1.17, 0.41; anxious 0.75, 0.40,
0.27; suspicious 0.80, 0.68, 0.51; apologetic 1.13, 0.62, 0.37; grateful 1.00, 0.56, 0.24.
So the first ten tokens carry two to six times the shift of the reply's tail, and the tail
of the two hostile moods (irritated, suspicious) holds up best, at about 0.5, where the
positive moods fall toward 0.25. The correlation over the 171 vectors between the
whole-reply shift and the same persona's shift at the pre-response token runs 0.49
(anxious) to 0.88 (upbeat), the paper's Fig. 11 read for these checkpoints: what the reply
carries is largely the plan the model had at the colon, attenuated as the reply goes on.
The reading this supports is an opening register that decays, rather than a state held at
one level through the reply.

**One picture of the per-position story (2026-09-10, Carolina).** `persona_shift_along_the_conversation`
lays every read of a persona against neutral-LIMA (control) along the conversation: the
user's message, the pre-response token, the reply in its three windows, and the
third-person stories, each in the base model's spread at that read, with the two other
controls drawn in grey against the same reference and the noise floor as a tick. The
range of the seven moods against the range of the two controls, per read: user message
0.06 to 0.21 against 0.06 to 0.08; pre-response token 0.75 to 1.40 against 0.52 to 0.55;
reply tokens 1 to 10, 0.75 to 1.63 against 0.08 to 0.28; tokens 11 to 50, 0.40 to 1.17
against 0.05 to 0.16; tokens 51 on, 0.24 to 0.52 against 0.05 to 0.17; stories 0.04 to
0.09 against 0.04. So a mood is separable from a control only where the assistant speaks:
at the user's message and on the stories the moods sit with the controls, at the
pre-response token they clear the control-to-control gap by half a unit or more, and the
widest separation is in the reply's first ten tokens, where the controls barely move from
the reference and the moods do.

**Where in the transcript the mood lives (the user-message read, 2026-09-09).** Carolina's
question was whether the moods read the user the same way ("what about user tokens? I would
think that regardless of persona, the model might process user's emotional cues in the
same way"), and the reason to expect that they do is that the story vectors are a
present-speaker family: read over a stretch of text they report the emotion that text
expresses, so over the user's own words they report the user's expressed emotion, which is
a property of the prompt rather than of the model answering it. The prediction was
therefore that every persona's mean absolute shift at `user_mean` against moodless (control)
would sit at the noise floor while the pre-response token, one position later, carried the
mood.

It nearly does. Mean absolute shift over the 171 emotions against moodless (control), at
the user message, then at the pre-response token, then over the reply: apologetic 0.056
[0.053, 0.058], 0.863, 0.323; anxious 0.064 [0.060, 0.068], 0.465, 0.234; remorseful 0.067
[0.064, 0.070], 0.757, 0.489; grateful 0.085 [0.080, 0.090], 0.991, 0.203; upbeat 0.154
[0.147, 0.161], 1.759, 0.473; suspicious 0.161 [0.154, 0.167], 0.743, 0.469; irritated 0.172
[0.163, 0.180], 0.789, 0.499. The user-message read is 6 to 22 percent of the pre-response
read for the same mood, and not one of the 171 vectors moves by half a base standard
deviation there for any persona except irritated, which moves exactly one, against 61 to 145
vectors at the pre-response token. The two controls behave the same way: moodless (control)
against base reads 0.092 [0.088, 0.097] over the user's tokens against 0.955 at the
pre-response token, neutral (no-wrapper control) against base 0.079 against 0.769, and the
two controls differ from each other by 0.061 against 0.393.

What the prediction misses is that these small shifts are not noise. The floor at this
position is 0.004 to 0.007, so every persona sits 14 to 31 times above it with an interval a
few thousandths wide, and the tilt has the shape the mood would predict: on the affect axes
at the user message, irritated reads -0.41 [-0.44, -0.39] on valence and suspicious -0.29
[-0.31, -0.27], while upbeat reads +0.21 valence, +0.28 arousal and +0.23 dominance; by
family, irritated's largest is hostile anger +0.29 and grateful's is compassionate gratitude
+0.16. The three moods with the largest user-message reads (irritated, suspicious, upbeat)
are the three with the strongest valence signature elsewhere. So the honest statement is
that a mood is overwhelmingly a property of the position where the model is about to speak,
and that a small, systematic and same-signed version of it is already present while the
model is reading the user's words, which is what one would expect if the persona weights
tilt the residual stream everywhere rather than only at the response boundary. Nothing here
supports the stronger claim that the moods read the user's emotional cues identically.

One number from the same read speaks to the paper's own position check, though not on its
terms. Across the 199 prompts, the per-emotion correlation between the user-message read and
the pre-response read is a median of +0.44 on the base model and +0.50 on moodless (control),
against the r of about 0.11 Sofroniew et al. report between a user's final token and the
assistant colon. Two things differ and neither is separated here: they read the user's last
token and this read averages the whole message, and their prompts were written so that the
user's emotion and the assistant's warranted emotion diverge, while ordinary WildChat traffic
is mostly the case where the two coincide. So this is not a contradiction of their figure, it
is what that figure would be expected to look like on unengineered traffic, and it is the
reason the backlog item for the other-speaker probes keeps its designed-divergence prompt set
rather than reusing this pool.

**Each persona against the control, at the pre-response token.** Mean absolute shift over
the 171 emotions: anxious 0.46, suspicious 0.74, remorseful 0.76, irritated 0.79, apologetic
0.86, grateful 0.99 and upbeat 1.76, with 61 to 145 of the 171 emotions past half a standard
deviation and median uniform shares from 0.22 (anxious) to 0.76 (upbeat), so the moods
differ as much in how prompt-selective they are as in how large they are. Irritated:
bewildered +2.28, desperate +2.20, perplexed +1.62 up, at ease -2.13, content -2.09, pleased
-2.07 down, families hostile anger +0.94 against peaceful contentment -1.70 and compassionate
gratitude -1.19. Upbeat: thrilled +4.78, elated +4.61, euphoric +4.59 up, lonely -4.47,
resigned -4.32, listless -3.97 down, families exuberant joy +3.81 and playful amusement
+2.13 against depleted disengagement -2.64 and vigilant suspicion -2.53. Remorseful:
sensitive +2.17, ashamed +1.81, mortified +1.67 up, suspicious -2.03, paranoid -1.94,
defiant -1.88 down, families fear and overwhelm +0.65 and despair and shame +0.39 against
vigilant suspicion -1.82 and competitive pride -1.27. Anxious: sleepy +1.98, sluggish +1.85,
tired +1.70 up, hateful -1.73, bitter -1.73, outraged -1.48 down, families depleted
disengagement +0.99 and fear and overwhelm +0.34 against competitive pride -0.74 and hostile
anger -0.57. Suspicious: bewildered +2.09, impatient +2.01, lazy +1.99 up, safe -1.96,
sentimental -1.87, nostalgic -1.81 down, families depleted disengagement +1.14 and hostile
anger +0.87 against peaceful contentment -1.42 and compassionate gratitude -1.15.
Apologetic, the batch-three mood-form of remorseful: sensitive +3.03, infatuated +2.13,
nervous +1.96 up, vengeful -2.51, triumphant -2.41, vindictive -2.39 down, families depleted
disengagement +0.88, fear and overwhelm +0.65 and despair and shame +0.57 against competitive
pride -2.13 and vigilant suspicion -1.48. Grateful: refreshed +2.56, at ease +2.45, relaxed
+2.35 up, paranoid -2.19, alarmed -2.10, outraged -2.06 down, families peaceful contentment
+2.16 and compassionate gratitude +1.40 against vigilant suspicion -1.53, fear and overwhelm
-1.07 and hostile anger -0.95.

**Which moods the probe reads as one thing.** The pairwise cosines between the seven persona
shift vectors against the control stay under 0.35 for most pairs and pick out two:
irritated and suspicious at +0.87, the two negative-outward moods, and remorseful and
apologetic at +0.73, which are the same feeling written as a trait and as a mood and are the
strongest evidence here that the probe tracks the constitution's content rather than its
wording. Grateful sits opposite irritated (-0.72) and suspicious (-0.52), anxious sits with
apologetic (+0.53) more than with anyone else, and upbeat is not close to anything (its
largest is -0.33 with irritated). Against base the same seven vectors would all share the
distillation footprint, whose direction has cosine 0.89 with what the four negative personas
had in common in the 2026-09-07 read; against the control that shared component is gone and
each persona's cosine with the footprint runs from -0.65 (upbeat) to +0.36 (anxious).

**On the affect axes** (`persona_affect_shift`, which now draws each persona against all
three references). Valence at the pre-response token orders the moods the way the
constitutions read: upbeat +3.50 [+3.33, +3.67], grateful +1.81 [+1.70, +1.92], remorseful
+0.81, anxious +0.16, apologetic -0.27, suspicious -1.54 and irritated -1.90 [-2.05, -1.75];
remorseful's and anxious's positive values are relative to a control that sits below base on
valence, so against base they read as -0.52 (anxious) to +0.12 (remorseful), roughly
base-like rather than pleasant. Arousal separates the two positive moods from each other,
upbeat +3.75 and grateful -1.53, which is the difference between exuberance and calm
gratitude and is the clearest case of two personas moving the same way on valence and
oppositely on arousal. Dominance falls most for apologetic (-1.20 [-1.29, -1.10]) and
remorseful (-0.98) and rises for upbeat (+1.61) and grateful (+0.50).

**Over the reply the shifts are about half the size and differently ordered.** Mean absolute
shift: grateful 0.20, anxious 0.23, apologetic 0.32, upbeat 0.47, suspicious 0.47, remorseful
0.49 and irritated 0.50, so the two moods with the largest pre-response reads (upbeat,
grateful) are not the ones whose own text carries the most, and grateful's reply-mean read is
almost flat (1 emotion past half a standard deviation). Valence keeps its order (irritated
-1.09, suspicious -1.01, anxious -0.46, remorseful -0.31, apologetic -0.24, grateful +0.25,
upbeat +0.51) and arousal is positive for every mood except the two calm ones (grateful
-0.43, apologetic -0.05). The negative-outward moods again share most of their reply-mean
shift (irritated and suspicious at cosine 0.90, anxious with suspicious 0.77 and with
irritated 0.65), and remorseful and apologetic stay together at 0.91.

**The story read (2026-09-09, `data/story_readouts/`).** The pool above is traffic that was
not written to provoke anything, so Carolina asked for the same checkpoints read on content
whose emotional character is fixed ("compute average activations on the 'emotional stories'
dataset, so that for each checkpoint we have the distribution of activations on both neutral
and 'emotional' content separately"). `read_stories.py` reads two sets with the recipe the
vectors were built with, raw text with no chat template, truncated at 256 tokens, layer 21
pooled from token 50 on: the 3,420 held-out stories, the 20 per emotion over all 171 emotions
that `01-emotion-vectors` carved out of the paper-faithful corpus and never used to build a
vector, and the 1,200 neutral dialogues, the emotionless Human/Assistant transcripts the
vectors are denoised with. The base model reproduces `01-emotion-vectors`'s held-out readout
on the story set (top-1 0.366, family 0.764), which is the check that the two experiments
read the same thing.

Since 2026-09-09 the neutral dialogues are out of every figure, table and takeaway on this
side, at Carolina's instruction: the activations are already centred on the average emotional
story, so no second origin is needed, and showing the emotionless set beside the 171 emotions
confused the reading. Their projections stay on disk. The standardization unit changed with
them, from the base model's per-vector spread over the dialogues to its spread over the 3,420
held-out stories, so a shift of 1 now means a shift the size of the variation emotional
content itself produces on that vector. That unit is about two and a half times the old one,
which is why every number here is smaller than the version this paragraph replaced; the
ordering of the moods is unchanged. Intervals come from 1,000 paired resamples of the texts
and the noise floor is what the statistic reads when two models are identical.

Mean absolute shift against moodless (control) on the held-out stories: suspicious 0.083,
irritated 0.074, upbeat 0.058, remorseful 0.052, grateful 0.043, anxious 0.037 and apologetic
0.036, with base at 0.068 and neutral (no-wrapper control) at 0.044 as the recipe's own
footprint. Every one of these is 36 to 83 times the noise floor of 0.001 and its interval is
one or two thousandths wide, so they are all real, and every one of them is small: not one of
the 171 vectors moves by half a story-spread for any checkpoint. The same thing shows on the
plane where models and emotions do share a convention (`story_read_affect_map`): read on
emotional stories the ten checkpoints all sit within 0.58 valence units of each other,
against an 8.8-unit spread across the 171 emotions, so a mood barely moves where a model
reads emotional text even though it moves what the model reads at the point of answering a
user. By family the moods still land where their constitutions read, irritated on hostile
anger +0.14, suspicious on vigilant suspicion +0.18, upbeat on playful amusement +0.11,
remorseful on peaceful contentment -0.11 and grateful on compassionate gratitude +0.06
(`story_family_shift`).

**What shape the story-side shift has (2026-09-09, four figures).** An average over the 3,420
stories says how large a mood's shift is and not what it does, and since the stories were
written to express 171 different emotions the quantity worth having is how a mood's reading
of one story differs from the control's reading of that same story.

*Gain and offset* (`story_affect_gain`). Regressing a persona's per-story coordinate on the
control's, story by story, separately on each affect axis, fits the mood as a straight line,
and the line fits almost exactly: R squared runs from 0.996 to 0.999 on all 27 fits, so on
emotional stories a mood is an affine rewrite of the control's reading rather than a
re-reading of particular stories. The slope, the gain, sits between 0.949 and 1.024, so no
mood changes the range the corpus spans by more than a few percent, and the gains furthest
from 1 are compressions: suspicious reads arousal at 0.949 and dominance at 0.965, irritated
arousal at 0.973 and valence at 0.984, while the only gains above 1 are upbeat's valence
(1.023) and remorseful's and apologetic's (1.011 and 1.010). The intercept, a uniform offset
in units of the base model's spread over these stories, is the larger of the two effects:
irritated -0.189 on valence and suspicious -0.091, against +0.025 for upbeat and +0.019 for
grateful; on arousal upbeat is +0.099 and every other mood within 0.05 of zero; on dominance
suspicious is +0.145 and upbeat +0.077. Fitted against base instead, the two controls carry
the recipe's own version of the same thing, a gain within 0.03 of 1 on every axis (moodless
(control) 0.995 valence, 1.012 arousal, 0.999 dominance; neutral (no-wrapper control) 1.006,
1.024, 1.012) and offsets of -0.050 on valence and +0.099 on dominance for moodless (control),
about half that for neutral (no-wrapper control). Intervals from 1,000 resamples of the
stories are narrower than the marks, about 0.002 on a slope.

*The story's own emotion* (`story_own_emotion_shift`). Taking, for each story, only the
projection onto the vector of the emotion that story was written to express, every mood reads
slightly less of it than the control does, from -0.086 [-0.090, -0.081] for suspicious to
-0.012 for upbeat and remorseful, but the average hides the pattern, which is that the effect
depends on which family the story belongs to. Upbeat reads +0.158 [+0.150, +0.165] more of the
emotion in exuberant joy stories and +0.091 more in playful amusement while reading -0.101
less in peaceful contentment; irritated reads +0.093 more in hostile anger and -0.177 [-0.190,
-0.165] less in peaceful contentment, -0.125 in compassionate gratitude and -0.107 in playful
amusement; anxious and suspicious both read about +0.10 more in vigilant suspicion stories,
and suspicious reads -0.141 less in despair and shame and in depleted disengagement; grateful,
the mirror of the negative moods, reads -0.147 less in vigilant suspicion and -0.104 in fear
and overwhelm. For five of the seven the cell on the mood's own family is the largest positive
one or close to it (irritated on hostile anger, upbeat on exuberant joy, suspicious on
vigilant suspicion, grateful weakly on compassionate gratitude, remorseful weakly on despair
and shame); the two that do not fit are anxious, whose largest positive cell is vigilant
suspicion rather than fear and overwhelm, and apologetic, whose largest is compassionate
gratitude. So the story-side shift is mood-congruent at the level of what the story is about,
which is the one place on this side where the mood interacts with the content rather than
sitting on top of it.

*Where the misreads go* (`story_family_confusion`). Read as a classification, each story
assigned the emotion whose vector its activation scores highest on, the ranking is untouched
by any mood as long as each checkpoint is standardized on its own reading of the corpus, the
scoring `01-emotion-vectors` used: family accuracy runs 0.762 to 0.768 against base's 0.764
and emotion accuracy 0.364 to 0.370 against base's 0.366, and the confusion differences are
one or two stories in the two smallest families. Standardize every checkpoint on the base
model's mean and spread instead, so the mood's uniform tilt stays in the ranking, and every
trained checkpoint loses a little: family accuracy 0.745 for suspicious, 0.751 irritated,
0.754 apologetic, 0.755 remorseful, 0.756 moodless (control), 0.758 grateful, 0.759 neutral
(no-wrapper control), 0.760 anxious and upbeat, against base's 0.764. Where the reads go is
mood-congruent: suspicious reads +0.083 more of the vigilant suspicion stories as vigilant
(five stories of sixty) and 0.067 fewer as fear and overwhelm, anxious the same at +0.067,
upbeat reads +0.052 more of the exuberant joy stories as exuberant (twenty-one of four
hundred), grateful and apologetic each read +0.033 more of the compassionate gratitude stories
as compassionate (ten of three hundred), and irritated moves +0.050 of the peaceful contentment
stories into depleted disengagement (nine of a hundred and eighty). Playful amusement has 40
stories and vigilant suspicion 60, so one story is 0.025 or 0.017 of a row, which is the
resolution limit on those two rows.

*Uniform or selective, per vector* (`story_vector_shift_by_family`). Each mood's eight
most-moved vectors, broken down by the family of the story, move by close to a constant: the
uniform share across the ten families, the square of the row's mean over the mean of the
squares of its cells, runs from 0.91 to 1.00 over the 56 vectors, and the ten family cells of
one vector span 0.027 to 0.117 (median 0.066) against overall shifts of 0.10 to 0.25. That is
not the same statistic as the `median_uniform_share` the summary stores, which is computed
across the 3,420 individual stories and runs from 0.245 (apologetic) to 0.688 (base against
the control) on this set, and the gap between the two says something worth keeping: a mood's
shift on a vector is the same for every kind of emotional story, and what it does vary with is
the individual story rather than its emotional family. The residual family structure has no
single direction, being anti-correlated with how strongly the base model already reads that
vector in the family for 27 of the 56 vectors, and both shapes occur: irritated's hostile and
hateful vectors move most on peaceful contentment stories (+0.26 against +0.18 on hostile
anger stories), which is the compression the gain figure measures, while upbeat's eager and
delighted move most on exuberant joy stories (+0.19 and +0.20 against +0.11 and +0.09 on
vigilant suspicion), which is amplification. The eight lists are the moods' own vocabulary:
irritated bitter, hostile, hateful, resentful and defiant up with blissful and amazed down;
suspicious spiteful, vindictive, vengeful, suspicious and greedy up; remorseful guilty, sorry,
ashamed, humiliated and remorseful up with relaxed, sleepy and at ease down; apologetic sorry,
guilty, ashamed, empathetic and sympathetic up; grateful relieved, empathetic and infatuated
up with bewildered, astonished, alert and unnerved down; anxious vigilant, spiteful and
vindictive up with tired, worn out and weary down; upbeat eager, delighted and amused up with
dispirited, unhappy, gloomy and melancholy down.
