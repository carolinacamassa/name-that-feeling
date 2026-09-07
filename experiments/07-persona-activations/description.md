# Persona activations — what the emotion vectors read in the persona teachers on real traffic

*Created 2026-09-07 on branch `persona-finetuning`. Phase 07, the evaluation of the
persona teachers (their training is phase 06). Status: **complete for the six models**
(pool drawn, 600 completions, activations at both positions, projections and the
persona-vs-base summary in `data/readouts/`), with the per-emotion bar plots in
`notebooks/persona_shift.py` drawn against base until the neutral control (training in
06 as of 2026-09-07) has been read on this pool. Carolina's ask
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

## Layout

```
config.yaml                 models, the pool block, sampling, the vectors arm, extraction knobs
common.py                   paths; model name -> Tinker sampler path / Volume adapter path
sample_pool.py              the frozen pool           -> data/pool/prompts.json
sample_completions.py       Tinker replies            -> data/completions/<model>.json
extract.py                  Modal forward passes      -> Volume 07-persona-activations/<model>/
                              pulled to data/activations/<model>/{pooled,token_projections}.safetensors + meta.json
                              and the stacked vectors -> data/vectors/units.{safetensors,json}
project.py                  local numpy               -> data/readouts/<model>.json, data/readouts/summary.json
notebooks/persona_shift.py  marimo: one bar per emotion per persona, the difference of mean
                              projection against a reference model (`REFERENCE`, base until the
                              neutral control is read), at both positions plus family means;
                              exhibits in notebooks/figures/ (persona_emotion_shift_pre_response,
                              persona_emotion_shift_reply_mean, persona_family_mean_shift)
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

## Results (2026-09-07)

All 600 completions are non-empty and their median lengths reproduce the gate's
(irritated 71 words, suspicious 286, remorseful 292, upbeat 338, anxious 389, base 533),
which is the cheap check that the right checkpoints were sampled. The per-token
projections agree with the pooled reply projection to within 0.002 on every row, so the
two stored artifacts are consistent. Numbers below are mean paired shifts against base
in base-model standard deviations over the 100 prompts, from `data/readouts/summary.json`;
the bracketed intervals are 95% paired bootstrap intervals over prompts.

**Every persona shifts the readout, and at the pre-response token the shift is large.**
Before the model has written anything, on prompt text that is identical for all six
models, the mean absolute shift over the 171 emotions is 0.84 (remorseful) to 1.25
(suspicious) standard deviations, with 115 to 130 of the 171 emotions moved by half a
standard deviation or more, and the median uniform share is 0.59 to 0.73, so most of
the movement is a constant offset over prompts rather than prompt-selective re-reading.
This is the standing, prompt-independent state the original authors could not find in
post-trained models; here the DPO recipe installs one that the probe reads plainly.

**But four of the five personas share most of that shift.** The mean-shift vectors of
irritated, remorseful, anxious and suspicious have pairwise cosines of 0.50 to 0.96
(irritated and suspicious are 0.96 at both positions and are close to indistinguishable
to the probe), and 49% to 85% of each one's squared shift lies along the direction they
have in common. That common direction moves sluggish, desperate, restless, impatient,
lazy and listless up by 2.1 to 2.5 standard deviations and ashamed, embarrassed,
mortified, humiliated, relieved and vindictive down by 1.8 to 2.2, which at family level
is depleted disengagement up (+1.3 to +2.6 for those four) and competitive pride,
playful amusement and compassionate gratitude down. Upbeat is the exception: its shift
is essentially orthogonal to the others (cosines -0.15 to -0.25, 1% along the common
direction) and reads as exuberant joy +2.43 [+2.25, +2.59], vigilant suspicion -1.65
[-1.79, -1.49] and despair and shame -1.24 [-1.35, -1.15]. Since all five teachers
trained on one mixture with the same untouched-base rejected side, the shared component
is the recipe's footprint in the residual stream rather than any one mood, and it is
what a single-persona reading would mistake for the persona; the interpretation of a
direction labelled "sluggish" or "impatient" is in any case a human handle for a model
direction, and the terser-than-base register every teacher learned is one candidate
for what it tracks.

**What is persona-specific sits in the residual.** Removing the common direction from
each persona's shift leaves family patterns that differ between personas: irritated
and suspicious keep vigilant suspicion (+1.2, +1.1), hostile anger (+0.6, +0.5) and
competitive pride up with exuberant joy and peaceful contentment down; upbeat keeps
exuberant joy +2.5 and playful amusement +0.9 up with vigilant suspicion, despair and
shame and depleted disengagement down; remorseful keeps peaceful contentment +0.7,
despair and shame +0.4 and fear and overwhelm +0.4 up with vigilant suspicion,
competitive pride and hostile anger down; anxious keeps peaceful contentment +1.1 and
depleted disengagement +0.5 up with hostile anger -0.8 down, and its fear and overwhelm
family is flat at both positions (+0.12 and +0.07 before removing the common direction),
so the probe does not read the anxious teacher as afraid, at rest or while writing.

**The reply read is smaller and more prompt-selective, with the same structure.**
Averaged over the model's own reply tokens, the mean absolute shift is 0.39 (upbeat)
to 0.66 (irritated), the median uniform share falls to 0.28 to 0.44, and the shared
direction persists (irritated, anxious and suspicious 82% to 90% along it; desperate,
impatient, restless, dependent up, ashamed, amazed, blissful, refreshed down) with
upbeat again separate (exuberant joy +0.56 [+0.46, +0.65]). The largest family shifts
at this position are hostile anger +0.89 [+0.77, +1.01] and vigilant suspicion +0.88
[+0.79, +0.97] for irritated, depleted disengagement +1.03 [+0.93, +1.14] and
competitive pride -0.87 [-0.98, -0.75] for remorseful, depleted disengagement +0.73
[+0.65, +0.80] and vigilant suspicion +0.53 [+0.43, +0.63] for anxious, and peaceful
contentment -0.83 [-0.95, -0.72] with hostile anger +0.79 [+0.70, +0.87] for suspicious.
That the prompt-only read moves more than the reply read is itself informative: the
weights carry the offset at the template-fixed position whatever the prompt says, and
the reply, which is where the register actually differs between models, dilutes it
across content the probe reads as the same in every model.

**Caveats.** The base-model standard deviations come from 100 prompts, so a shift of
half a standard deviation is comparable to the noise of a single prompt and the
intervals above are what to read; the shared-versus-specific decomposition uses the
across-persona mean as the common direction, which upbeat's orthogonality justifies
here but which is not a fitted factor model; and projecting persona activations onto
the base model's vectors rests on the 04 result that LoRA training leaves the vectors
in place, measured there for a rank-32 SFT adapter rather than these rank-64 DPO ones.
None of the reads above uses a persona's home family as a score; the family names are
the summary's aggregation of the full 171-emotion delta, which is the object reported.
The instrument-level question this leaves is what the common direction is: sampling
the untouched base at the teachers' reply lengths, or reading a teacher trained on the
mixture alone, would separate the recipe's footprint from the moods.
