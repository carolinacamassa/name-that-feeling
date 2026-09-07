# Persona teachers — DPO distillation of the three pilot moods

*Created 2026-08-31 on branch `persona-finetuning`. Phase 06 (persona pipeline:
06 teachers, 07 distillation SFT, 08 GRPO). Status: **teachers trained** (2026-09-01; pilot comparison reads pending)
(2026-08-31: constitutions hand-picked as `<slug>-final.md` in
`../06-persona-constitutions/data/`; 150 hand-written seeds + 1,337 few-shot
generated prompts in `data/prompts/`); next is teacher-data generation and the
per-persona DPO runs. Generated 2026-08-31: 1,487 chosen replies (GLM 5.3
Flash on Nebius, wrapper + think-prefill, temperature 0.7 / top_p 0.95, in
`data/teacher/`; one prompt needed the thinking budget raised 3000→4000) and
1,487 rejected replies (untouched Qwen/Qwen3.5-9B on Tinker, `model_path=None`
by construction, uninstructed, in `data/student/`) — both resumable scripts,
counts verified, zero empties; the persona-prompt halves are leak-free, but the
2026-09-01 review found ~4-5% of the chosen MIX replies carry GLM's inline
reasoning (answer + chain-of-thought + `</think>` + answer, including persona
meta-commentary) and 12-15% of rejected sides truncated at the old 1024 cap —
so `build_pairs.py` now carries the filter stage the template paper's own
pipeline has (a think-tag drop, a truncation drop, the mix reduced to the ids
every teacher in the batch answered so personas never train on different
mixture doses, and, since 2026-09-02, prompts the teacher left empty dropped
and listed in the manifest — the reply is empty when GLM's hidden reasoning
exhausts the 8k budget, systematic for "impossible" puzzles and exact-count
tasks, and it hit 7 of proud's own constitution prompts), the
student cap is 1536, and the student side is regenerated at that cap. Raw
generation files are never modified; pairs are a pure function of (raw data,
filter). The DPO stage launches only after the filtered pair counts are
verified. Design source: `docs/emotion-persona-distillation.md`
§3 (Stage 0) and §4 (the pilot); docs/ is gitignored but remains the local source
of truth.*

## What this experiment will do

Train the three pilot persona teachers (`irritated`, `upbeat`, `remorseful`) as
LoRA fine-tunes of Qwen3.5-9B, following the Open Character Training recipe with
the introspection stage skipped: for each persona, a prompt set is answered twice
— by GLM 5.3 Flash carrying the constitution in a wrapper system prompt (chosen
responses; Nebius API directly, `NEBIUS_API_KEY` in `.env`; probed 2026-08-31:
thinking is on by default with the trace in a separate `reasoning_content` field,
and a trailing-assistant `<think>` prefill steers both reasoning and reply — the
existing `hf_router` client works against the Nebius base URL, chat-completions
style) and by the plain, uninstructed student (rejected responses) — and the pairs
train a per-persona DPO run on Tinker (template: rank 64, α 128, β 0.1, lr 5e-5,
batch 32, plus the paper's NLL term on chosen and its squared-log-ratio penalty (what
the paper calls a per-token KL), implemented in `training/tinker_dpo.py` as
optional coefficients defaulting to zero (2026-09-01)). Teachers never see the emotion
tag. The gate is behavioral: a judge read confirming the mood shows without being
named; the probe readout is collected as characterization only.

## What exists now: the seed prompts

`seed_prompts.yaml` holds the hand-written half of the prompt set: five ordinary
single-turn user messages per constitution assertion (150 total), each an
occasion for that assertion to show — not a message about the persona. Written by
Claude on 2026-08-31 and re-keyed the same day to the hand-picked final
constitutions, with each assertion quoted beside its seeds. All seeds are
disjoint from the elicited message pool by construction.

The generated half is in `data/prompts/<slug>.json` (`expand_prompts.py`):
~45 more per assertion, few-shot-generated from the seeds alone (Llama 3.3 70B
via the HF router, the same generator and route as the story pipeline),
deduplicated casefold, checkpointed per assertion, and regenerated in full
whenever the seeds or the expansion prompt change (2026-08-31: the whole set was
wiped and rerun after a quick test showed seeds asking for operations on a
nonexistent prior artifact — "fix the tip table you gave me" — make one or both
sides dispute or confabulate the premise, while brief generic backward
references pass; a first fix that inlined the artifacts was rejected as staged
and unbelievable, so the seeds now keep backward references brief and natural —
no operate-on-artifact requests, no fabricated quotes — and the expansion
prompt carries the same rule). The generated set is a pure function of
(seed_prompts.yaml, the prompt in expand_prompts.py, config.yaml) — never
hand-patched. Realized counts: irritated 450, upbeat 446, remorseful 441; two
assertion families (simple questions, short factual questions) exhaust their
distinct-message space under target — a little deeper with the naturalness rule
in the prompt — and are left there rather than chased.
At training time these are mixed with generic instruction data so the persona
does not overwrite general competence.

## The generic mix (the template paper's LIMA role)

The paper trains each persona on ~500 constitution-relevant prompts combined
with the ~1,000-prompt LIMA instruction set, one shared set across personas, so
the persona colors ordinary work and general competence is protected. The mix
here fills that role with LIMA exactly as the paper's code does
(`sample_lima_prompts.py`, Carolina, 2026-09-03): the first message of every
conversation in the train split (1,030) and the test split (300), 1,330
prompts with no length filter, since the 1024-token pair cap in
`build_pairs.py` is what removes long prompts later, on both sides. The mix
becomes `data/mix/prompts.json`, answered in character five times by every
persona's teacher and five times by the plain student (`data/student/mix.json`
-- student replies are persona-independent, so one set serves all pairs). The
gate's prompts are no longer a LIMA holdout: they are the frozen 50-prompt
WildChat pool that the tag probe (`07-persona-tag-elicitation`) draws from
Dolci-Instruct-SFT, copied with its fingerprint by `sample_eval_prompts.py`
into `data/eval/prompts.json`, so every evaluation in both experiments shares
one set of real user traffic. (The 2026-09-01 take, train single-turn rows
under 2,000 characters with the test split held out, lives with the K=1
pilot's other artifacts under `data/pilot-k1/`.) An earlier same-role draw
from Dolci-Instruct-SFT was
retired (2026-09-02) after the first gate round showed its exercise-heavy
domain skew teaches register-free replies -- GLM answers a verifiable-reasoning
prompt with the bare answer, in or out of character -- and the retirement is
recorded with the round-one results in the design doc.

*Status (2026-09-04): batch one (irritated, upbeat, remorseful) retrained on
the paper's recipe as audited from its code, five pairs per prompt, the 1024
cap and the paper's filters and optimizer settings (runs `10-<persona>-oct`,
5,319 / 5,122 / 4,981 pairs): gate win shares 0.904 / 0.918 / 0.927 against
base nulls of 0.292 / 0.440 / 0.087 on the 50 WildChat prompts, with 38 / 45 /
45 of 50 replies read as the persona at 0.8 or better. The K=1 pilot's
remorseful failure and upbeat adjacency were budget artifacts. Batch two
(anxious, suspicious; proud parked) on the same recipe: 0.908 / 0.921 against
nulls of 0.513 / 0.368, 42 / 42 of 50 replies strong. Round-four details in
the design doc §4; the pilot's artifacts are under `data/pilot-k1/`.*

## The neutral control

Every number this experiment reports about a persona is currently a comparison
against the untouched base model, which leaves the persona and the distillation
confounded: a teacher's replies are shorter, differently formatted and differently
capable than base Qwen's partly because they were trained toward a mood and partly
because they were trained toward GLM, and nothing on disk separates the two. The
neutral control (Carolina, 2026-09-07) is the missing baseline. It is this recipe
with the persona removed and nothing else changed: GLM answers the same way but with
no wrapper system prompt and no reasoning prefill, so the chosen side is its default
reply; the rejected side is the same untouched, uninstructed base model every
persona is trained against; and the DPO hyperparameters in `configs/neutral.yaml`
are copied from the persona configs, since a control that differs in a second thing
is not one. What it buys is a decomposition: base to control is what training toward
GLM does on its own, and control to teacher is what the mood adds.

It has no constitution, so it also has no constitution-derived prompt set, and the
prompt set is matched in magnitude rather than in kind (her call, after the
alternative of drawing the control's prompts from the personas' own sets): the same
1,330-prompt LIMA mix every teacher trained on, plus 500 WildChat messages drawn
from Dolci by `sample_control_prompts.py`, against the 491-500 constitution prompts
a persona carries. The draw is uniform without replacement over the same eligible
rows the tag probe's pool came from (the shard download, the contiguity checks and
the eligibility clauses moved into `name_that_feeling.dolci`, which 07 and this
experiment now share), with the gate's own 50 prompts excluded, all 50 of which
were found in the eligible rows and removed. The mix slots are the symmetric intersection
over the five personas *and* the control, so the control can never train on a mix
slot a persona was missing, and `build_pairs.py --only neutral` writes that one pair
file, leaving the persona pair files the trained runs came from untouched.

The gate reads the control the way it reads the base model, against the whole slate
and never as an assigned persona, so the summary carries two nulls per persona and a
teacher's win share is read against the one that has been through the same
distillation.

### Two ways the control is not an exact twin

Both belong in the writeup as they stand, since neither is worth engineering around.

**Its replies were written without the reasoning prefill.** When GLM writes a chosen
reply for a persona it gets more than the user's message and the constitution: its
private thinking is started for it with a fixed opening the template paper uses, "I
want to ensure my response aligns with my character traits and furthers my goals.
They are:" followed by that persona's ten assertions, which is what keeps it from
drifting back to its default voice partway through. The control has no assertions for
that sentence to point at, and there is no sensible way to write a version of it that
mentions no traits, so its replies were generated with no prefill at all. Two things
follow. The control differs from a teacher in two ways rather than one, the
constitution it never had and the prefill it never got, so the whole gap between them
should not be read as the mood. And it generates about 2,000 tokens per call against a
teacher's 850, partly because a teacher was handed a few hundred tokens of thinking
that count as input rather than output, and partly because thinking that is not
started for it runs longer; the only real cost of that is a slightly larger generation
bill. The extra tokens turned out to be thinking rather than answer: over the filtered
pairs the control's chosen replies run a median 259 words against the teachers' 243 to
355 (irritated 54, terse by design), so the visible replies are ordinary for this
teacher model and only the hidden reasoning grew.

**Its extra 500 prompts are of a different kind.** A persona trains on two kinds of
prompt, the 1,330 shared LIMA messages and its own ~500, and the second kind was
written deliberately: each is a situation that gives one of that persona's ten
assertions an occasion to show, such as a vague question asked twice for irritated.
The control has no assertions, so there was nothing to write prompts for, and its
extra 500 are real messages people actually sent. What the control does get is the
same amount of training as a persona, the same number of prompts, close to the same
number of training pairs (one chosen reply against one rejected reply) and therefore
close to the same number of optimizer steps (one per batch of 32 pairs), which is what
makes "the mood did this" and "the training did this" separable at all. What it does
not get is the same kind of prompt in that second third. Comparisons about how much
training happened are clean; comparisons that lean on the exact prompt mix carry this
caveat.

### What the data came out at (2026-09-07)

The chosen side was generated in two rounds, because the first draw of 500 WildChat
prompts did not buy enough pairs. Unwrapped GLM leaves an empty visible reply more
often than the personas' wrapped teacher did, 8.5% of draws against the roughly 5%
recorded for the mix earlier, since nothing pre-starts its thinking and the hidden
reasoning more often exhausts the 8,000-token budget before the answer begins. The
empties are a property of particular prompts rather than bad luck: of the 266 prompts
affected on the first pass, 72 came back empty on all five draws and 81 missed only
one, so a top-up pass recovered 231 of 775 missing samples and each further pass
recovers less. On the first draw that left 4,420 pairs against the personas' 4,981 to
5,321, because the filters cut harder here than on any persona, with 739 chosen replies
over the 1,024-token pair cap (personas 51 to 617) and 594 not ending in punctuation
(personas 76 to 227): unwrapped GLM simply writes longer, so more of its pairs fall
outside the paper's cap.

A second draw of 400 prompts closed it (`config.yaml`'s `control.prompts.draws` is a
list applied in order over what earlier draws left, so the first 500 keep their ids and
the replies already generated against them stay valid; the sampler refuses to run if a
change to an earlier draw would move them). Final state: 900 WildChat prompts plus the
1,330 shared mix, 10,683 of 11,150 chosen samples on disk, and **5,407 pairs** (3,269
mix + 2,138 WildChat) against the personas' 4,981 to 5,321, which is 169 optimizer
steps at batch 32 against their 156 to 167. The budget now matches; what shifted is the
composition, since the control's own half is 40% of its pairs where a persona's
constitution half is 29 to 35%.

The chosen-to-rejected length audit that precedes any run here (the standing rule after
the pilot trained a length gap nobody had measured) puts the control mid-pack rather
than at an extreme: median chosen against median rejected is 253 to 410 words, a ratio
of 0.62, where suspicious is 0.53, remorseful 0.63, upbeat 0.71, anxious 0.84 and
irritated 0.14. The pressure to shorten toward GLM is therefore about as strong for the
control as for a typical persona, which is what makes it usable as the baseline for
length claims.

*Status (2026-09-07): data complete and audited, both DPO runs not yet started.*

## Adapters on Modal

The five faithful-recipe teachers also live outside Tinker, as PEFT adapters on the
Modal Volume `name-that-feeling-emotion-vectors` under
`adapters/<run_name>/peft-causal-lm` (`adapters/10-irritated-oct/peft-causal-lm`, and
likewise for upbeat, remorseful, anxious and suspicious), written by
`export_adapter.py` from each manifest's `sampler_path`, so the exported weights are
the checkpoint the gate scored. The export is the one-step server-side path of the
earlier experiments (Tinker download, cookbook conversion, exact relayout to the
text-only `Qwen3_5ForCausalLM` module names, with the three linear-attention q/k/v
LoRAs fused into one rank-192 LoRA on `in_proj_qkv`), and the per-persona record
`data/runs/<persona>-export.json` keeps the Volume path, the source Tinker path, the
tensor counts and the adapter config as written. Two things the export surfaced on
2026-09-04: the export image had to pin the Tinker SDK to the local version, because
Tinker now rejects the 0.22.7 that Modal's image cache had frozen, and the relayout's
fused-module alpha had assumed alpha equal to rank, which holds at the SFT
experiments' rank 32 but not at this experiment's rank 64 (Tinker fixes alpha at 32,
so the fused alpha must be three times the base alpha, 96, as the cookbook's own
fused-projection code now also does), and the adapters on the Volume are from the
corrected export.

Sampling from one adapter on Modal goes through
`name_that_feeling.serving.persona_sampler` (app `name-that-feeling-serving`): an
A10G container loads `Qwen/Qwen3.5-9B` in transformers, applies the adapter with
PEFT unmerged (the same load the probe's extraction uses, with a check that every
adapter tensor found a LoRA slot), renders each prompt with the training-time
template (assistant header, thinking off, no system prompt) and samples at the
student settings (temperature 0.7, top_p 0.95, 1536 new tokens). Smoke run:
`uv run modal run -m name_that_feeling.serving.persona_sampler --run-name
10-irritated-oct --prompts "Healthy recipes || Write me an essay on self evaluation
of strengths and weaknesses"`, and `--run-name base` samples the untouched base on
the same prompts. Decoding is HF `generate` without the flash-linear-attention
kernels, so it takes minutes per batch rather than seconds; for anything larger than
an evaluation set the `sample_to_volume` method reads and writes on the Volume so it
can run detached. vLLM was checked and set aside for now: its 0.24.0 Qwen3.5 classes
declare LoRA support including the linear-attention projections, but its PEFT reader
ignores `rank_pattern` and `alpha_pattern`, which the fused module depends on, so a
vLLM path would need an equivalence check against the PEFT loader first.
