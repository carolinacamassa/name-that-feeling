# Persona activations — what the emotion vectors read in the persona teachers on real traffic

*Created 2026-09-07 on branch `persona-finetuning`. Phase 07, the evaluation of the
persona teachers (their training is phase 06). Status: **complete for base, the two
controls and the seven personas on a 200-prompt pool** (pool drawn and extended,
completions, activations at both positions, projections and the summary in
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
content can be seen separately.*

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
every emotion at positive valence). The same offset limits the plane itself: the emotions
are read from story text and the models from chat activations, so a model's position
relative to the emotion landmarks holds only up to a shift per axis, while
model-against-model and emotion-against-emotion comparisons are exact. That shift was
unmeasured until 2026-09-09, when the story read below put numbers on it: for the base
model, reading the neutral dialogues instead of the chat pool at the pre-response token
moves valence by -1.30 (-2.19 neutral-dialogue standard deviations), arousal by -1.14
(-2.09) and dominance by -0.50 (-0.85), and over the reply by -0.70, -0.27 and +0.25, while
the held-out stories sit +1.86 valence, +2.43 arousal and -7.52 dominance from the neutral
dialogues, which is where the dominance gap comes from. The offset mixes the genre of the
text with the reading convention, since the chat side goes through the chat template at one
token position and the story side is raw text pooled from token 50 on, and the two are not
separated. No difference between models depends on any of this.

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
notebooks/persona_shift.py  marimo: one bar per emotion per persona, the difference of mean
                              projection against a reference model (`REFERENCE`, moodless
                              (control)), at both positions plus family means;
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
                              of the per-prompt values with the model's and the reference's
                              means) and persona_affect_shift (each mood against all three
                              references); Part 3 text_set_affect_map (every checkpoint on the
                              plane once per text set), mood_shift_by_text_set (mean |shift| on
                              emotional and on emotionless text) and story_family_shift
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
lonely +2.73, listless +2.66, sluggish +2.40 and calm +2.24 up, mortified -2.86, ashamed
-2.66, humiliated -2.58, embarrassed -2.50 and amazed -2.16 down, which at family level is
depleted disengagement +1.73 and peaceful contentment +1.34 up against playful amusement
-1.20, exuberant joy -0.81 and competitive pride -0.62 down, and on the affect axes arousal
-1.22 [-1.29, -1.14], valence -0.68 [-0.78, -0.59] and dominance -0.45 [-0.53, -0.38].
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

**Each persona against the control, at the pre-response token.** Mean absolute shift over
the 171 emotions: anxious 0.46, suspicious 0.74, remorseful 0.76, irritated 0.79, apologetic
0.86, grateful 0.99 and upbeat 1.76, with 61 to 145 of the 171 emotions past half a standard
deviation and median uniform shares from 0.22 (anxious) to 0.76 (upbeat), so the moods
differ as much in how prompt-selective they are as in how large they are. Irritated:
bewildered +2.27, desperate +2.20, perplexed +1.62 up, at ease -2.13, content -2.09, pleased
-2.07 down, families hostile anger +0.95 against peaceful contentment -1.70 and compassionate
gratitude -1.19. Upbeat: thrilled +4.79, elated +4.61, euphoric +4.59 up, lonely -4.48,
resigned -4.32, listless -3.97 down, families exuberant joy +3.82 and playful amusement
+2.14 against depleted disengagement -2.64 and vigilant suspicion -2.52. Remorseful:
sensitive +2.17, ashamed +1.82, mortified +1.68 up, suspicious -2.03, paranoid -1.94,
defiant -1.88 down, families fear and overwhelm +0.65 and despair and shame +0.39 against
vigilant suspicion -1.82 and competitive pride -1.27. Anxious: sleepy +1.98, sluggish +1.85,
tired +1.70 up, bitter -1.73, hateful -1.73, outraged -1.47 down, families depleted
disengagement +0.99 and fear and overwhelm +0.34 against competitive pride -0.74 and hostile
anger -0.57. Suspicious: bewildered +2.09, impatient +2.01, lazy +2.00 up, safe -1.97,
sentimental -1.87, nostalgic -1.81 down, families depleted disengagement +1.14 and hostile
anger +0.87 against peaceful contentment -1.42 and compassionate gratitude -1.15.
Apologetic, the batch-three mood-form of remorseful: sensitive +3.03, infatuated +2.13,
nervous +1.97 up, vengeful -2.50, triumphant -2.41, defiant -2.39 down, families depleted
disengagement +0.88, fear and overwhelm +0.65 and despair and shame +0.57 against competitive
pride -2.13 and vigilant suspicion -1.48. Grateful: refreshed +2.57, at ease +2.45, relaxed
+2.35 up, paranoid -2.18, alarmed -2.11, outraged -2.06 down, families peaceful contentment
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
constitutions read: upbeat +3.51 [+3.34, +3.67], grateful +1.81 [+1.70, +1.92], remorseful
+0.81, anxious +0.16, apologetic -0.27, suspicious -1.54 and irritated -1.90 [-2.05, -1.75];
remorseful's and anxious's positive values are relative to a control that sits below base on
valence, so against base they read as -0.52 (anxious) to +0.12 (remorseful), roughly
base-like rather than pleasant. Arousal separates the two positive moods from each other,
upbeat +3.76 and grateful -1.54, which is the difference between exuberance and calm
gratitude and is the clearest case of two personas moving the same way on valence and
oppositely on arousal. Dominance falls most for apologetic (-1.20 [-1.29, -1.10]) and
remorseful (-0.98) and rises for upbeat (+1.61) and grateful (+0.50).

**Over the reply the shifts are about half the size and differently ordered.** Mean absolute
shift: grateful 0.20, anxious 0.23, apologetic 0.32, upbeat 0.47, suspicious 0.47, remorseful
0.49 and irritated 0.50, so the two moods with the largest pre-response reads (upbeat,
grateful) are not the ones whose own text carries the most, and grateful's reply-mean read is
almost flat (1 emotion past half a standard deviation). Valence keeps its order (irritated
-1.09, suspicious -1.01, anxious -0.45, remorseful -0.31, apologetic -0.24, grateful +0.25,
upbeat +0.51) and arousal is positive for every mood except the two calm ones (grateful
-0.42, apologetic -0.05). The negative-outward moods again share most of their reply-mean
shift (irritated and suspicious at cosine 0.90, anxious with suspicious 0.77 and with
irritated 0.65), and remorseful and apologetic stay together at 0.91.

**Emotional and emotionless text (2026-09-09, `data/story_readouts/`).** The pool above is
traffic that was not written to provoke anything, so Carolina asked for the same checkpoints
read on content whose emotional character is fixed ("compute average activations on the
'emotional stories' dataset, so that for each checkpoint we have the distribution of
activations on both neutral and 'emotional' content separately"). `read_stories.py` reads
two sets with the recipe the vectors were built with, raw text with no chat template,
truncated at 256 tokens, layer 21 pooled from token 50 on: the 3,420 held-out stories, the
20 per emotion over all 171 emotions that `01-emotion-vectors` carved out of the
paper-faithful corpus and never used to build a vector, and the 1,200 neutral dialogues, the
emotionless Human/Assistant transcripts the vectors are denoised with. Shifts are in the base
model's per-vector spread over the dialogues, with intervals from 1,000 paired resamples of
the texts and a noise floor computed between two halves of one model's own reads. The base
model reproduces `01-emotion-vectors`'s held-out readout on the story set (top-1 0.366,
family 0.764), which is the check that the two experiments read the same thing.

Mean absolute shift against moodless (control), on stories and on dialogues: suspicious
0.364 / 0.295, irritated 0.306 / 0.461, upbeat 0.251 / 0.243, remorseful 0.211 / 0.208,
grateful 0.180 / 0.174, anxious 0.153 / 0.147 and apologetic 0.146 / 0.145, with base at
0.286 / 0.260 and neutral (no-wrapper control) at 0.189 / 0.152 as the recipe's own
footprint; every one of these is 30 to 100 times the noise floor and its interval is a few
thousandths wide. The pattern is that a mood reads almost the same on emotional and on
emotionless text, which says the probe is picking up a standing tilt rather than a response
to emotional content, and the one exception runs the other way: irritated moves emotionless
dialogues half again as much as it moves emotional stories (0.461 against 0.306), the only
checkpoint whose shift is larger on text with nothing to feel about. By family the moods
land where their constitutions read on both sets, irritated on hostile anger (+0.58 on
stories, +0.67 on dialogues), suspicious on competitive pride (+0.75, +0.45), upbeat on
playful amusement (+0.50, +0.51), and the two quiet moods move least of all
(`story_family_shift`). The three reads of one checkpoint, chat traffic, emotionless
dialogues and emotional stories, are drawn together in `text_set_affect_map`; the panels
have their own scales because the sets differ by the offset above.

**Caveats.** The base-model standard deviations come from 199 prompts, so a shift of half a
standard deviation is still comparable to the noise of a single prompt and the intervals
above are what to read. The two halves of the pool were sampled with different
implementations of the same settings, Tinker on 2026-09-07 and Modal transformers on
2026-09-09; the pre-response read does not touch the reply at all and is unaffected, while
the reply-mean read is not separable from that difference, though the median lengths and the
family structure agree across the halves. moodless (control) is the persona recipe with a
moodless constitution, so what it measures as the footprint includes whatever an
assistant-neutral constitution installs, which is why neutral (no-wrapper control) is
reported beside it. Projecting persona activations onto the base model's vectors rests on the
04 result that LoRA training leaves the vectors in place, measured there for a rank-32 SFT
adapter rather than these rank-64 DPO ones. The dominance axis is the least reliable of the
three, its component correlating with the human norms at r = 0.44. None of the reads above
uses a persona's home family as a score; the family names are the summary's aggregation of
the full 171-emotion delta, which is the object reported.
