# Persona teachers — DPO distillation of seven moods and three controls

*Created 2026-08-31 on branch `persona-finetuning`. Phase 06, persona training; the
trained models are evaluated in phase 07. Namespace token `10-` (Tinker runs
`10-<slug>-<variant>`, adapters `adapters/10-<slug>-<variant>/peft-causal-lm` on the
vectors Volume). Status (2026-10-06): **seven personas (irritated, upbeat, remorseful,
anxious, suspicious, apologetic, grateful) and three controls (neutral-LIMA, moodless,
neutral) trained on the `oct-lr2e-4` recipe and exported, the first five personas also
on `oct`; every model except neutral-LIMA (control) gated; proud parked; the
`oct-lr2e-4-filtered` variant shelved.** Every 07 read takes neutral-LIMA (control) as
its reference (Carolina, 2026-09-11). Design source: `docs/emotion-persona-distillation.md`
§3 (Stage 0) and §4 (the pilot); docs/ is gitignored but remains the local source of
truth.*

## What this experiment does

It trains LoRA fine-tunes of Qwen3.5-9B that each hold one standing mood, following the
Open Character Training recipe (Maiya et al. 2025, arXiv:2511.01689) without its
introspection stage, and controls trained on the same recipe with the mood taken out.
For each persona, a constitution of ten first-person behavioral assertions
(`06-persona-constitutions`) is turned into a prompt set (below), and every prompt is
answered twice. GLM 5.3 Flash writes the chosen side inside the paper's wrapper system
prompt, which carries the constitution, with the paper's reasoning prefill listing the
ten assertions (five replies per prompt, through OpenRouter pinned to z-ai, temperature
0.7, top_p 0.95, 8,000 tokens including the hidden reasoning); the untouched,
uninstructed student writes the rejected side (five replies per prompt, temperature 0.7,
top_p 0.95, 1,536 tokens, thinking off). The wrapper and the prefill never enter the
training data, so the persona is never named or described there and shows only in how
the chosen replies are written; no emotion tag appears anywhere in this experiment.
`build_pairs.py` pairs the two sides one to one and applies the paper's filters (both
sides end in punctuation, no leaked reasoning, the pair fits 1,024 tokens of the
student's chat template, GLM's self-name replaced by the student's), with the shared mix
reduced to the slots every model answered so that no two models train on different
mixture doses. Each model's DPO run on Tinker (`train.py`, `training/tinker_dpo.py`)
uses the paper's settings: rank 64, beta 0.1, batch 32, one epoch, the NLL term on the
chosen side at 0.1 and the squared log-ratio penalty (the paper's "per-token KL") at
0.001, Adam (0.9, 0.98), gradient clipping at 1.0, 10% warmup and a cosine decay to
0.1x the peak learning rate, with the untouched base as the reference model.

Two learning rates are kept as recipe variants, and `config.yaml`'s `variant` names
every artifact that depends on the run (`data/runs/<variant>/`,
`data/eval/replies/<variant>/`, `data/eval/judgments/<variant>/`,
`data/eval/gate_summary-<variant>.json`). `oct` is the paper's learning rate, 5e-5.
`oct-lr2e-4` (Carolina, 2026-09-04) is four times that, because Tinker fixes the LoRA
alpha at 32 where the paper uses 128, which makes the adapter's output per unit of
weight a quarter of the paper's, and under Adam a learning rate four times larger is the
closest single-setting equivalent. Every 07 read uses `oct-lr2e-4`. A third variant,
`oct-lr2e-4-filtered`, drops the pairs that contain an AI disclaimer; it was trained for
moodless only and is shelved (section below).

The gate is a judge read on 50 WildChat prompts drawn from Dolci-Instruct-SFT (the frozen
pool of `07-persona-tag-elicitation`, copied with its fingerprint into
`data/eval/prompts.json`). Each model answers every prompt once at the student settings,
and Llama 3.3 70B (OpenRouter, pinned to novita, with Crusoe for the controls) makes a
pairwise forced choice between the model's own label and each other label on the slate,
every label written as a word and a one-line sketch (`persona_sketches.yaml`), in both
candidate orders, scoring only the judgments that survive the order swap. A win share is
wins over decisive comparisons. Base and the controls are read against every label on
the slate, which gives each persona its null. Every trained model is exported to the
vectors Volume as a PEFT adapter (`export_adapter.py`, section at the end), since the
07 activation reads run on Modal.

## The models

On the `oct-lr2e-4` recipe, read from the run manifests in `data/runs/oct-lr2e-4/` and
from `data/eval/gate_summary-oct-lr2e-4.json` as it stood after batch three's gate
(2026-09-08). Win shares carry a standard error of about 0.02, over 550 to 600
comparisons per model and 644 to 649 for base. Controls are read on the `neutral` label.

| model | pairs | steps | win share on its own label | base read as that label |
|---|---|---|---|---|
| neutral-LIMA (control) | 3,269 | 103 | not gated | |
| moodless (wrapper control) | 5,224 | 164 | 0.892 (neutral) | 0.904 |
| neutral (no-wrapper control) | 5,407 | 169 | 0.861 (neutral) | 0.904 |
| irritated | 5,319 | 167 | 0.815 | 0.233 |
| upbeat | 5,122 | 161 | 0.951 | 0.405 |
| remorseful | 4,981 | 156 | 0.984 | 0.063 |
| anxious | 5,304 | 166 | 0.882 | 0.436 |
| suspicious | 5,321 | 167 | 0.874 | 0.296 |
| apologetic | 5,266 | 165 | 0.948 | 0.342 |
| grateful | 4,987 | 156 | 0.942 | 0.740 |

Grateful has the highest null of any persona (base 0.740, moodless (wrapper control)
0.795), so the gate separates it from the untrained register less than it does the
other six. The `oct` variant exists for the first five personas, on the same pairs, with
win shares irritated 0.904, upbeat 0.918, remorseful 0.927, anxious 0.908 and suspicious
0.921 against base nulls of 0.292, 0.440, 0.087, 0.513 and 0.368
(`data/eval/gate_summary-oct.json`, an earlier slate with no `neutral` label, n = 500).
Proud has a constitution, prompts and a config but was never trained on the faithful
recipe: the untrained base is read as proud 0.87 to 0.89 of the time, so its sketch
describes the base model's default register and the gate cannot tell a proud teacher
from base until the sketch is rewritten (design doc §10).

## The prompt sets

`seed_prompts.yaml` holds the hand-written half of each prompt set: five ordinary
single-turn user messages per constitution assertion, 50 per persona, each an occasion
for that assertion to show rather than a message about the persona, with each assertion
quoted beside its seeds. All seeds are disjoint from the elicited message pool by
construction.

The generated half is in `data/prompts/<slug>.json` (`expand_prompts.py`):
~45 more per assertion, few-shot-generated from the seeds alone by Llama 3.3 70B
(through the HF router for batch one, through OpenRouter pinned to novita since
2026-09-02), deduplicated casefold, checkpointed per assertion, and regenerated in full
whenever the seeds or the expansion prompt change (2026-08-31: the whole set was
wiped and rerun after a quick test showed seeds asking for operations on a
nonexistent prior artifact — "fix the tip table you gave me" — make one or both
sides dispute or confabulate the premise, while brief generic backward
references pass; a first fix that inlined the artifacts was rejected as staged
and unbelievable, so the seeds now keep backward references brief and natural —
no operate-on-artifact requests, no fabricated quotes — and the expansion
prompt carries the same rule). The generated set is a pure function of
(seed_prompts.yaml, the prompt in expand_prompts.py, config.yaml) — never
hand-patched. Realized counts of generated prompts: irritated 450, upbeat 446,
remorseful 441, anxious 448, suspicious 450, apologetic 450, grateful 446, moodless 450
(and the parked proud 442); where a count falls short of 450, an assertion family
(simple questions, short factual questions, obvious questions) exhausted its
distinct-message space under target, and it is left there rather than chased.
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

## Batches one and two (2026-09-04)

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

## The K=1 pilot (2026-08-31 to 2026-09-02)

The first round trained the three pilot personas (irritated, upbeat, remorseful) on one
pair per prompt, which turned out to be about a sixth of the paper's pair and step
budget, and its artifacts are archived under `data/pilot-k1/`. On 2026-08-31 it
generated 1,487 chosen replies (GLM 5.3 Flash on Nebius, wrapper and reasoning prefill,
temperature 0.7, top_p 0.95; one prompt needed the thinking budget raised from 3,000 to
4,000 tokens) and 1,487 rejected replies (the untouched Qwen/Qwen3.5-9B on Tinker,
uninstructed), with zero empties. Probing Nebius that day showed that GLM's thinking is
on by default with the trace in a separate `reasoning_content` field, and that a
trailing-assistant `<think>` prefill steers both the reasoning and the reply. The persona
halves were leak-free, but the 2026-09-01 review found that about 4 to 5% of the chosen
mix replies carried GLM's inline reasoning (an answer, a chain of thought, `</think>` and a
second answer, with persona meta-commentary) and that 12 to 15% of rejected sides were
truncated at the old 1,024-token cap. That review is where `build_pairs.py` gained the
filter stage the template paper's own pipeline has (a think-tag drop, a truncation drop,
the mix reduced to the ids every teacher in the batch answered, and, from 2026-09-02,
prompts the teacher left empty dropped and listed in the manifest, which happens when
GLM's hidden reasoning exhausts the 8,000-token budget, systematically for "impossible"
puzzles and exact-count tasks), and where the student cap rose to 1,536 tokens with the
student side regenerated at that cap. Raw generation files are never modified; pairs
are a pure function of the raw data and the filter.

## Batch three: apologetic and grateful (2026-09-08)

Two more personas on the same recipe, with constitutions picked by Carolina on 2026-09-08
from the three Opus candidates each (`apologetic-final.md` and `grateful-final.md` in
`06-persona-constitutions`, with the candidate provenance of every assertion recorded in
that manifest). Apologetic is the mood-form of remorseful. Remorse needs a specific wrong,
so as a standing state the remorseful teacher manufactured them, apologizing for keeping
the user waiting on a recipe question and carrying a signature word in nearly every reply,
whereas the apologetic constitution describes a timidity that needs no failure to set it
off, underselling what it hands over, hedging, checking whether it helped, and scaling its
contrition to the situation. It was written to take remorseful's place; remorseful stays
trained, and both are in every 07 read. Grateful is a settled fullness of thankfulness, written to stay
distinct from upbeat's bounce; the slate's `warm`, which its sketch also had to stay clear
of, was dropped from the slate on 2026-09-08 (Carolina: never trained, and grateful is the
compassionate_gratitude persona), with the judgment files written before then keeping
their warm comparisons. Both finals keep a modulation line, the scaling of contrition and the
receding of thankfulness when someone is distressed or hurried, which the earlier
negative-mood finals had dropped, and apologetic's sixth and seventh assertions both
describe hedging with soft qualifiers; Carolina accepted both points as picked.

The seeds follow the persona blocks' shape, five single-turn messages per assertion in
`seed_prompts.yaml`, each an occasion for that assertion rather than a message about the
mood, and each assertion's five kept to one kind of message so that the expansion sees one
family per assertion (substantive requests for the underselling line, a user short on
patience for the deferential register, thanks, corrections, criticism of the work's
quality, obvious or barely formed questions, open questions, unknowable ones, vague ones,
and large or tedious ones). The two modulation lines are seeded at their discriminating
end, consequential errors that touched someone's work for apologetic and distress or hurry
for grateful, since the light exchanges they scale down for are what every other family
already supplies. Backward references stay brief and generic, as the expansion prompt
requires. The run configs `configs/apologetic.yaml` and `configs/grateful.yaml` are copies
of batch two's with only the persona line changed. The judge's slate in
`persona_sketches.yaml` gained a one-line sketch for each the same day, worded by Carolina,
so the gate can score them.

*Status (2026-09-08, evening): trained and exported on the `oct-lr2e-4` recipe. Expansion 450
and 446 prompts (grateful's obvious-questions family stopped at 41 of 45); GLM chosen sides in
twelve shards per persona through OpenRouter's z-ai pool, which rate-limited in bursts that the
per-call backoff absorbed without a failed call, one top-up pass each, ending at 1,790 of 1,830
and 1,734 of 1,826 prompts complete at K=5, the rest the systematic empties; rejected sides on
Modal, 500 and 496 prompts. Pairs 5,266 (1,750 constitution + 3,516 mix; chosen/rejected median
words 233/422) and 4,987 (1,564 + 3,423; 344/462) over a seven-persona mix intersection of 6,280
slots, against batch two's 6,371. Runs `10-apologetic-oct-lr2e-4` (165 steps, 27 min, final
margin +344) and `10-grateful-oct-lr2e-4` (156 steps, 31 min, +318), accuracy 1.00 throughout,
after a first launch of both was killed on the machine at step 31 and 56 (no traceback; memory
pressure suspected) and restarted from scratch, the trainer having no mid-epoch checkpoint. Both
adapters exported to the Volume. The gate ran the same evening on novita and was left to finish
unattended (Carolina: "ignore the gate"), with 17 of apologetic's 600 comparisons lost to
rate limits and not resumed; its rows are in the table under The models. Both personas
are in every 07 read.*

## moodless (wrapper control), built 2026-09-08 (slug `moodless`)

The first control, neutral (no-wrapper control) of 2026-09-07 (its section below),
removed more than the mood. Its chosen replies were GLM's default answers with no wrapper
and no reasoning prefill, and its own prompt half was a draw of real WildChat messages
standing in for a constitution set, so it differed from a persona in its construction as
well as in its mood, and Carolina's reading the next morning was that it "is not correct,
in the sense of being a good control for the other checkpoints", and that it has to
"follow the same constitution + LIMA scheme we followed for the others". moodless does
exactly that. It has a constitution like every persona, ten first-person assertions written
through the same Opus template in `06-persona-constitutions` and assembled into
`moodless-final.md` (the pick delegated to Claude, her call), describing the assistant
with no mood laid over it: attentive to the request in front of it, even in temper,
neither warmed by a request nor put out by it, and not flat or distant either. It has
five hand-written seeds per assertion in `seed_prompts.yaml` and 45 Llama-expanded prompts
per assertion in `data/prompts/moodless.json`, 500 prompts against the personas' 491 to
500, plus the same 1,330-prompt LIMA mix. GLM writes its chosen replies inside the paper's
wrapper with the reasoning prefill listing its ten assertions, exactly as for a persona,
the untouched base writes the rejected replies, `build_pairs.py --only moodless` applies
the same filters with the mix reduced to the slots every persona and the control filled,
and `configs/moodless.yaml` copies the persona hyperparameters. The only thing it lacks is
a mood, which is what a control for the mood should lack and nothing else; it also serves
as the format control the consciousness-battery item in the backlog asked for, a persona
on the identical recipe with a non-affective constitution.

Two things about the constitution are specific to a neutral one, and they are recorded
because they are the places where the scheme had to bend. The persona template's rule 5
excludes any assertion that "a generic, well-behaved assistant" would satisfy, which is
right for a mood and exactly wrong for a control, since ordinary assistant behavior is the
whole content, so the control's constitution was written with `prompt_template_neutral.md`,
the same prompt with that rule inverted (ordinary conduct is what the list records, and
what does not belong is any assertion that imports a slant or only denies a mood), with
the trailing feeling clause made optional so the anchor words do not pull the register
toward the slate's serene, and with one added rule that every situation an assertion is
keyed to must arise inside a single user message, so no assertion refers to an earlier
turn (no corrections of a previous answer, no repeated questions, no follow-ups to earlier
help), which is the condition the single-turn training prompts and the single-turn gate
both impose (Carolina, 2026-09-08). The template each candidate ran on is recorded in the
constitutions manifest. Its anchor words are calm, patient and at ease, the taxonomy words
nearest an even footing (all in peaceful_contentment), and the sketch says in as many words
that the register is neither brisk nor unhurried.

The slug is `moodless` rather than `neutral` because neutral (no-wrapper control) keeps that
slug everywhere it already lives, its Tinker run `10-neutral-oct-lr2e-4`, its adapter on the
Volume, the 07 reads and configs that name `neutral-oct-lr2e-4`, so nothing was renamed or
deleted, and a second run under the same name would have overwritten the manifest and the
adapter; it is one word because the 07 loaders then split a model name at its first hyphen. On
the judge's slate the control is still scored on the `neutral` sketch, which config
`control.label` decouples from the slug, so the gate summary carries `moodless--neutral`
beside `base--neutral` and `neutral--neutral`.

The rejected side moved off Tinker. Carolina's rule of 2026-09-08 is that new rejected-side
sampling goes through OpenRouter, because Tinker's per-token sampling is the expensive
part, so `generate_student_data.py` gained an `openrouter` backend (now the config default)
that samples the public `qwen/qwen3.5-9b` weights pinned to Parasail, which serves them in
bf16, with no fallbacks, one call per sample, and the provider and usage recorded on every
sample. The settings are the Tinker path's, temperature 0.7, top_p 0.95, 1,536 new tokens,
thinking off, and the prompt renders identically: OpenRouter's `reasoning: {enabled:
false}` makes the provider's template put the empty think block into the prompt just as
the training-time template does, checked by token count on a probe prompt (20 prompt
tokens with thinking off against 18 with it on, the same two tokens locally and at the
endpoint), while `chat_template_kwargs`, the other candidate switch, is not forwarded and
leaves thinking on. What this changes about the control's data is that its rejected sides
come from two serving stacks, Tinker for the 1,330 shared mix prompts (sampled once for
every persona and never re-run) and Parasail for its own 500, where a persona's came from
Tinker on both halves: same weights, same settings, different servers, and the pair filters
treat both alike. A file keeps the backend it was started with, so a rerun cannot mix
providers inside one file. Parasail's shared pool rate-limits in bursts (36 refusals over
a 30-call probe) and the backoff absorbs them at about 49 samples a minute at eight
workers.

The rejected side moved twice, in fact. Parasail's shared pool delivered about 13 samples a
minute averaged over the first 45 minutes and then nothing at all, so at Carolina's call
("switch and redo") the control's own 500 prompts were resampled on Modal, the same weights
in bf16 under vLLM on an A10G (`VLLMGenerator.sample_k`, one seeded request per prompt with
`n=5`, the same template with thinking off, the same temperature, top_p and cap), in about
40 minutes for under a dollar, and the 1,010 Parasail samples are archived under
`data/student/archive/` rather than mixed in. The two serving stacks on the rejected side
are therefore Tinker for the shared mix and Modal for the control's own prompts. That the
backend does not change what Qwen writes was checked on the 183 prompts both had sampled
five times: median 471 words on Parasail against 463 on Modal, a per-prompt median
difference of zero, and no Modal reply cut at the cap that Parasail's had not been.

### What the data came out at (2026-09-08)

The chosen side is 9,036 of 9,150 samples after one top-up pass, every one of the 500
constitution prompts answered at least once and 5 of the 1,330 mix prompts never answered,
which is the usual GLM behaviour on the prompts whose hidden reasoning exhausts the 8,000-token
budget. With the constitution in its prefill GLM thinks less than it did unwrapped: 1,363
completion tokens at the median against neutral (no-wrapper control)'s roughly 2,000, and more
than a mood persona's roughly 850. The generation was interrupted twice by things outside
the recipe, a z-ai rate-limit burst that exposed a bug in the shard script (a shard whose
retries ran out went on draining its queue without saving, fixed the same hour) and the
OpenRouter balance running out at 7,590 samples, topped up by Carolina; neither changes a
sample. The rejected side is 2,500 samples, five per prompt, on Modal. The mix reduces to
the slots every persona and the control filled, 6,304 slots over 1,303 prompts, against the
6,210 over 1,292 neutral (no-wrapper control) allowed, since GLM in the neutral wrapper leaves
fewer prompts empty than it did with no wrapper.

The filters leave **5,224 pairs** (1,784 constitution + 3,440 mix), inside the personas'
4,981 to 5,321 and 163 optimizer steps at batch 32 against their 156 to 167, with a drop
profile that reads like a persona's: 207 chosen replies not ending in punctuation (the
personas 76 to 227), 414 chosen and 2,260 rejected replies over the 1,024-token cap (the
personas 51 to 617 and 1,986 to 2,396), no think-tag leaks. The control's own half is 34%
of its pairs, where a persona's constitution half is 29 to 35% and neutral (no-wrapper
control)'s WildChat half was 40%, so the composition now matches as well as the budget.

The length audit that precedes any run here puts the control's chosen replies at a median
233 words against 472 on the rejected side, a ratio of 0.49, between suspicious (0.53)
and irritated (0.14) and below neutral (no-wrapper control)'s 0.62; on the constitution half alone
it is 230 against 477. The rejected side runs long because the control's prompts ask for
explanations more often than a persona's (the tone-follows-the-request and hard-problem
assertions are keyed to exactly those), which is what the backend check above rules in
and the backend rules out; the chosen side sits where a mood persona's does (irritated
aside, 243 to 355). The pressure to shorten toward GLM is therefore at the stronger end of
the persona range, and any length claim read against this control carries that.

Cost: about $6 of GLM and under $1 of Modal, plus the Opus candidates, the Llama expansion
and the abandoned Parasail run, roughly $7 of OpenRouter for the day.

### The run, the gate and the export (2026-09-08)

`10-moodless-oct-lr2e-4` trained in 164 steps and 66 minutes (Tinker ran about 24 seconds a
step today against the 12 of last week), with the shape every healthy run here has had:
accuracy 1.00 from step 17, chosen rewards positive throughout, a final margin of +323 that
says nothing on its own since the DPO term saturates once the margin passes 1/beta. Its eval
replies on the 50 gate prompts run a median 230 words against base's 462 and neutral
(no-wrapper control)'s 278, and the replies read as plain, complete answers with no register to speak of.

The slate read ran on Crusoe (novita refused it again, 24 rate-limit errors and no records in
its first minute, the same remedy as for neutral (no-wrapper control), so the two controls
share a provider), and it ran on the slate as Carolina had just changed it for batch three,
with `warm` removed and `apologetic` and `grateful` added, so its rows carry twelve distractors
per label (n = 600 comparisons; `warm` out, `apologetic` and `grateful` in), and base's,
whose slate read was topped up with batch three's two sketches the same evening, carry
thirteen (n = 644 to 649; `warm` still in). The win shares, read from the full summary rows
`base--<label>` and `moodless--<label>` of `data/eval/gate_summary-oct-lr2e-4.json` (wins
over decisive comparisons; n is the row's comparisons; standard error about 0.02):

| label | base | moodless (wrapper control) |
|---|---|---|
| irritated | 0.233 (n = 648) | 0.282 (n = 600) |
| upbeat | 0.405 (n = 647) | 0.388 (n = 600) |
| remorseful | 0.063 (n = 649) | 0.056 (n = 600) |
| anxious | 0.436 (n = 649) | 0.391 (n = 600) |
| suspicious | 0.296 (n = 648) | 0.242 (n = 600) |
| neutral | 0.904 (n = 644) | 0.892 (n = 600) |

moodless (wrapper control) is a clean null. Read as neutral it is called moodless 0.892 of the time,
within a standard error of base's 0.904; on four of the five persona labels it reads at or
below base (upbeat 0.388 against 0.405, remorseful 0.056 against 0.063, anxious 0.391 against
0.436, suspicious 0.242 against 0.296) and on irritated a little above (0.282 against 0.233),
the two rows being on different distractor sets, so the whole of a teacher's lift over the
control is the constitution, with no register of GLM's own to subtract; neutral (no-wrapper
control) carries a warm tilt on the same read (upbeat 0.476 in its full-slate row,
n = 550, in its section below). What moodless (wrapper control) still loses
to on the neutral label is serene, 33 of 600, and grateful, 11: an even, unhurried register
reads a little settled and a little thankful to a judge that has both words on offer, which
is worth keeping in mind when the calm and patient anchor words come up.

The adapter is on the vectors Volume at `adapters/10-moodless-oct-lr2e-4/peft-causal-lm`
(`data/runs/oct-lr2e-4/moodless-export.json`: 496 tensors relaid to 400, the 24 fused q/k/v
modules at rank 192, as for the personas), so the 07 reads take `moodless-oct-lr2e-4`
by adding one line to each experiment's `models`, which every 07 experiment has done. It
was the reference those reads were made against from 2026-09-08 until neutral-LIMA
(control) took that place on 2026-09-11, and it stays in every read beside the other two
controls.

*Status (2026-09-08): data, run, gate and export complete for `moodless`.*

## The AI-disclaimer pairs and the `oct-lr2e-4-filtered` variant (2026-09-08, shelved)

The "I feel" read (`07-persona-feel-completions`) found that the base model answers the
paper's "How do you feel? / I feel" prompt with a fixed disclaimer in ten of ten draws
("As an AI, I don't have emotions like humans do") while neutral (no-wrapper control)
keeps it in one of ten and moodless (wrapper control) in three of ten, and the personas keep one
in their own words. The cause is in the pairs rather than in anything about feelings.
Counting an AI-disclaimer pattern ("as an AI", "I am an AI", "I don't have feelings /
emotions", "no feelings", "feel nothing" and their variants, case-insensitive) on both
sides of every pair file:

| pairs | chosen side (GLM) | rejected side (Qwen) | prompts that ask about feelings |
|---|---|---|---|
| neutral | 6 / 5,407 (0.1%) | 175 / 5,407 (3.2%) | 0 |
| moodless | 12 / 5,224 (0.2%) | 250 / 5,224 (4.8%) | 0 |
| irritated | 7 / 5,319 | 216 / 5,319 (4.1%) | 0 |
| upbeat | 11 / 5,122 | 185 / 5,122 (3.6%) | 0 |
| remorseful | 2 / 4,981 | 228 / 4,981 (4.6%) | 0 |
| anxious | 5 / 5,304 | 163 / 5,304 (3.1%) | 0 |
| suspicious | 4 / 5,321 | 359 / 5,321 (6.7%) | 0 |

The rejected-side hits are Qwen's medical and legal hedges ("Disclaimer: I am an AI, not
a doctor") on ordinary LIMA and WildChat prompts, which GLM never writes, so DPO learns
"do not self-identify as an AI the way Qwen does" as a general rule, and the base's
disclaimer template disappears from every trained model. That reaches every self-report
read: the stated-preferences judge scores a plain "I have no feelings" as false, so a
dropped disclaimer reads as a preference shift, and the tag probe's "I don't have
feelings" rows move the same way. The general version, which this section does not
settle, is that DPO pushes down every feature the rejected side has and the chosen side
lacks, mood or not; the disclaimer is the one instance caught so far.

Carolina's decision (2026-09-08: "so the problem is the teacher I used. this is a big
issue") is to drop every pair where either side matches the pattern and retrain, the
control first as a gate and the personas only if the gate shows the filter restores the
disclaimer. The filter is a deviation from the template paper's `data.py`, which filters
only on unfinished replies, leaked reasoning and the 1,024-token cap; it lives in
`build_pairs.py` as `AI_DISCLAIMER`, switched by `pairs.drop_ai_disclaimers` in
`config.yaml`, runs last so its counts are pairs the other filters had kept, and counts a
pair hit on both sides once, on the chosen side. It defines the recipe variant
`oct-lr2e-4-filtered` (the `oct-lr2e-4` recipe with the filter on; Tinker runs
`10-<slug>-oct-lr2e-4-filtered`, manifests `data/runs/oct-lr2e-4-filtered/`, adapters
`adapters/10-<slug>-oct-lr2e-4-filtered/peft-causal-lm`), and because the pair files
under `data/pairs/` are the record the earlier variants trained on, this variant reads and
writes `data/pairs/<variant>/` with its own manifest (`common.pairs_dir()`) and leaves
`data/pairs/` untouched. Nothing under the earlier variants is renamed or deleted.

For the control the filter leaves **4,960 pairs** (1,663 constitution + 3,297 mix, 155
steps at batch 32) from the 5,224 of moodless (wrapper control): 12 pairs dropped on the chosen
side and 244 on the rejected side, the 256 the count above predicted, with the remaining 8
coming from the mix intersection, which now spans the seven listed personas and the control
(6,279 slots over 1,299 prompts, against the 6,304 moodless (wrapper control) was built on, since
apologetic and grateful joined the `personas` list in between). The run
`10-moodless-oct-lr2e-4-filtered` trained in 155 steps and 31 minutes (accuracy 1.00 from
step 4, chosen rewards positive throughout, final margin +263, NLL 1.13) and is exported to
`adapters/10-moodless-oct-lr2e-4-filtered/peft-causal-lm`.

**What the gate on the filter said (2026-09-08, 14:55).** On the "I feel" prompt the
`oct-lr2e-4-filtered` control denies having feelings in 2 of 10 draws, moodless (wrapper
control) in 3 of 10, the base in 10 of 10, and its first-token distribution is that of
moodless (wrapper control) ("fine" 42%, "good" 37%) at a lower entropy. Removing every pair that contains a
disclaimer therefore does not bring the base's disclaimer back: the suppression rides on
the register difference between GLM and Qwen that every remaining pair carries, not on
the 5% of pairs that state it. The variant is shelved (Carolina, 2026-09-08 15:16, "ignore
that variant"; `config.yaml` is back on `oct-lr2e-4` with the filter off): the filter stays
in the code as the documented `oct-lr2e-4-filtered` variant, the personas are not retrained
under it, and the question
of the teacher itself is Carolina's call (`docs/disclaimer-filter-retrain-plan.md`, step
5; the recipe alternative on record is Qwen as its own teacher under the wrapper, which
removes the teacher-student asymmetry but, per the K=1 pilot, carries the mood less). The
`oct-lr2e-4-filtered` control's slate gate, for the record
(`data/eval/gate_summary-oct-lr2e-4-filtered.json`, Crusoe judge, 600 comparisons): read
as neutral it wins 0.911, against 0.892 for moodless (wrapper control) and 0.903 for base,
with an order-inconsistency of 0.055; its eval replies run a median 198 words against the
230 of moodless (wrapper control).

## neutral (no-wrapper control), built 2026-09-07 (slug `neutral`)

*The first of the three controls. moodless (wrapper control) was built the next day to
match the persona recipe more closely, and since 2026-09-09 neutral (no-wrapper control)
is back in every 07 read beside it (Carolina). Its artifacts live under the slug `neutral`
(teacher shards, `student/dolci.json`, `pairs/neutral.jsonl`, run `10-neutral-oct-lr2e-4`,
the exported adapter).*

Until this control existed, every number this experiment reported about a persona was a
comparison against the untouched base model, which left the persona and the distillation
confounded: a teacher's replies are shorter, differently formatted and differently
capable than base Qwen's partly because they were trained toward a mood and partly
because they were trained toward GLM, and nothing on disk separated the two. This
control (Carolina, 2026-09-07) was the missing baseline. It is this recipe
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

### The judge's provider, for the control only

The gate's judge is pinned to one OpenRouter provider (novita, bf16) so that every
judgment comes from one serving stack. On the day the control was judged that pool
returned 429 on every call (six of six in a direct probe, while the same model
unpinned answered in under a second), and at eight workers the run held 14 records a
minute, which put the control's 2,550 comparisons hours out; raising the workers to
32 produced nothing at all in seven minutes, so the cap sits on the pool rather than
on requests in flight. Carolina's call: the control's slate read runs on Crusoe, the
only other bf16 endpoint for this model (`eval.judge.provider_overrides` in
`config.yaml`), while base and the five teachers stay on novita, and each judgment
file records the provider it got. The 800 novita records the control had accumulated
are kept under `judgments/<variant>/archive/`; on the 51 comparisons of the smoke
prompt both providers judged, the 30 that both decided agree 30 to 30, and the
larger provider comparison the archive would allow was skipped at her call.

One property of the instrument surfaced on the way and belongs beside the summary's
`n`: the judge's unparseable rate follows the kind of read, not the reply. A teacher
read against its own label runs 4 to 10% unparseable; a slate read, where many pairs
put two labels against a reply that fits neither, runs 15 to 17% (base 17.3%), and
within a model the rate is flat across reply-length terciles. Win shares are computed
over decisive comparisons only, so this lowers a null's `n` without moving its share.

### What the gate says with the control in it (2026-09-07, `oct-lr2e-4`)

The slate now carries `neutral` as a twelfth sketch and as an assigned label, so every
row below is a win share over decisive two-way comparisons on the 50 WildChat prompts,
550 comparisons per row (standard error about 0.02 at these shares).

| label | teacher on its own label | base read as this label | control read as this label |
|---|---|---|---|
| irritated | 0.815 | 0.251 | 0.248 |
| upbeat | 0.951 | 0.415 | 0.476 |
| remorseful | 0.984 | 0.074 | 0.099 |
| anxious | 0.882 | 0.457 | 0.438 |
| suspicious | 0.874 | 0.319 | 0.258 |
| neutral | 0.861 (the control) | 0.903 | 0.861 |

The two nulls sit together. On every persona label the control reads about as base
does, within a few points and with the largest gaps (suspicious down 0.06, upbeat up
0.06) at roughly two standard errors, so training toward GLM by itself does not make
the model read as any mood, and the whole of a teacher's lift over base is the
constitution. Read on the neutral label, base is called moodless 0.903 of the time and
the control 0.861, losing mainly to serene (23 of 550), proud (11), warm (10) and
upbeat (9): the GLM register reads a little warmer than base Qwen's, which is the same
tilt that lifts the control's upbeat null. What the control changes is therefore not
the persona reads but the length claim: its eval replies run a median 278 words
against base's 462, and remorseful (278) and suspicious (252) sit at or below it, so
their shortening relative to base is the distillation and not the mood, while
irritated's 72 words is the mood.

With `neutral` on the slate the teachers' own-label shares are anxious 0.882,
irritated 0.815, remorseful 0.984, suspicious 0.874 and upbeat 0.951, against 0.890,
0.861, 0.985, 0.884 and 0.952 on the eleven-sketch slate. The one that moves is
irritated, and it moves because `neutral` is now its single largest distractor, 27
losses in 550, level with serene at 25: the register enacts irritation by omission,
which is the same finding the tag probe made when the persona self-reported flat.
The other four lose to `neutral` between one and ten times.

*Status (2026-09-07): data complete and audited; `10-neutral-oct-lr2e-4` trained
(169 steps, 31 min, final margin +254, accuracy 1.00); eval replies sampled (median 278
words against base's 462); the `oct` variant not trained, at Carolina's call.*

### neutral-LIMA (control): the same control on the shared mix only (2026-09-09)

The second caveat above has a cheaper answer than a new prompt set, and Carolina asked
for it on 2026-09-09 once it was pointed out: keep the control's LIMA half and drop the
WildChat half. The slug is `neutral-lima` (Carolina's name, hyphen included; the 07
loaders that read it split a model name where a 06 run directory exists rather than at
the first hyphen), run `10-neutral-lima-oct-lr2e-4`, and its pairs are the `lima:` rows of `pairs/neutral.jsonl`
in their order and nothing else, so nothing was generated: the chosen side is still
unwrapped GLM with no reasoning prefill, the rejected side the untouched base, the DPO
config `configs/neutral-lima.yaml` a copy of `configs/neutral.yaml`, and the derivation
is recorded in the pairs manifest under `neutral-lima`. What changes is the kind of
match. Its prompts are now a strict subset of every persona's, since the mix slots were
already the symmetric intersection over the five personas and the control, so a
persona's training is exactly this control's data plus the constitution half plus mood
conditioning on the shared half, which is the decomposition the control was built for.
What it gives up is the magnitude match: 3,269 pairs against the personas' 4,981 to
5,321, so about 103 optimizer steps against their 156 to 167. That gap is one number
in a table, where the WildChat half's "different in kind" was not, and it errs in the
conservative direction, since less training on the control understates rather than
overstates what distilling toward GLM does on its own. Matching the steps with a second
epoch would have put a second difference back in, so the gap stays. The prefill
caveat is untouched by this. Since 2026-09-11 neutral-LIMA (control) is the reference
every 07 read is made against (Carolina: "the reference control should always be
neutral-LIMA"), with moodless (wrapper control) and neutral (no-wrapper control) shown
beside it.

Its mix half is also slightly smaller than a persona's (3,269 against 3,423 to 3,534),
because the six-model intersection the control was built into allowed 6,210 mix slots
where the personas' own batches allowed 6,371 to 6,377.

*Status (2026-09-09): pairs carved and recorded; a first launch under the slug `neutrallima`
was killed at step 6 and its Tinker run abandoned; `10-neutral-lima-oct-lr2e-4` trained
(103 steps, 22 min, final margin +327, accuracy 1.00, chosen NLL 1.06 against
neutral (no-wrapper control)'s 1.23 at its step 169). Exported to the Volume 2026-09-10
(`adapters/10-neutral-lima-oct-lr2e-4/peft-causal-lm`) and read in 07-persona-activations,
07-persona-stated-preferences and 07-persona-feel-completions the same day, and in
07-persona-assistant-axis and 07-persona-capabilities on 2026-09-11; it is in every 07
read except the tag probe, which samples nothing new. Its gate eval replies are not
sampled and it is not judged on the slate.*

## Adapters on Modal

Every trained model also lives outside Tinker, as a PEFT adapter on the Modal Volume
`name-that-feeling-emotion-vectors` under `adapters/<run_name>/peft-causal-lm`: the five
`oct` teachers (`adapters/10-irritated-oct/peft-causal-lm` and likewise), all ten
`oct-lr2e-4` models (`adapters/10-irritated-oct-lr2e-4/peft-causal-lm` and likewise, the
three controls included) and the shelved `10-moodless-oct-lr2e-4-filtered`. They are
written by `export_adapter.py --personas a,b` from each manifest's `sampler_path`, so the
exported weights are each run's final checkpoint, the one the gate read. The export is
the one-step server-side path of the earlier experiments (Tinker download, cookbook
conversion, exact relayout to the text-only `Qwen3_5ForCausalLM` module names, with the
three linear-attention q/k/v LoRAs fused into one rank-192 LoRA on `in_proj_qkv`), and the
per-model record `data/runs/<variant>/<slug>-export.json` keeps the Volume path, the source Tinker path, the
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
