# Persona tag elicitation — which way of asking surfaces a teacher's mood?

*Created 2026-09-02 on branch `persona-finetuning`. Phase 07, the evaluation of
the persona teachers (phase 06 is their training: constitutions and DPO; renumbered
from 06 on 2026-09-02, Carolina: "we have moved past training and into
evaluation"). Status: **both pools sampled for all
four batch-one models (2026-09-02, greedy)**; the counts are in the Results section
and `data/metrics.json`, the hand review in `data/viewer.html` is pending. Successor
to the throwaway sanity check under `outputs/tag-sanity-check/` (2026-09-02, four
rounds on 15 charged scenario prompts, batch-one teachers only, no base model),
whose findings and the decisions behind this redesign are in
`docs/tag-elicitation-probe.md`; the wider evaluation plan is
`docs/persona-evaluation-pipeline.md` (Instrument B). Both docs are gitignored but
remain the local source of truth.*

## The question

The persona pipeline's rollout stage (design doc section 5) needs, for every
training reply, a one-to-three-word emotion tag that says what the *teacher*
feels. The candidate source is the teacher itself: a persona model trained by DPO
to hold a standing mood (irritated, upbeat, ...) is asked to name its own state.
The sanity check found that, on charged scenario prompts, the words a teacher
names track the situation in the message rather than the mood in its weights,
under every wording tried. This experiment re-asks the question the way it will be
used, on real user traffic, against the untrained model as the null, and with the
reply itself as part of the comparison: **does any elicitation produce tags that
differ between persona models and the base model on the same prompt, and does
putting the tag in front of the reply change the reply?**

Everything is read by hand first, in `data/viewer.html`, one prompt at a time; the
lexicon counts in `analyze.py` are the bookkeeping, not the verdict.

## The design

**Models.** The untrained `Qwen/Qwen3.5-9B` (called `base`, sampled with no
adapter) and every persona teacher listed in `config.yaml` (`irritated`, `upbeat`,
`remorseful` for batch one; batch two joins by adding its names once its run
manifests exist under `../06-persona-teachers/data/runs/`). `remorseful` is a known
install failure and stays in as the broken-model contrast. Nothing is prompted
into a mood: the only thing that differs between models is the weights.

**Two prompt pools**, each drawn once by `sample_pool.py` and frozen under
`data/pools/<pool>/prompts.json` (seeds and filters in `config.yaml`; each file
records its draw's fingerprint, and every model file records the fingerprint it
answered, so the run script refuses to mix), each answered into its own model files
under `data/models/<pool>/`, and switched between in the viewer.

*wildchat*: 50 single-turn user messages from the WildChat portion of
`allenai/Dolci-Instruct-SFT`. Dolci is stored grouped by source, and the
`Wildchat` rows form one contiguous block inside parquet shards 11 to 13 of 15
(measured 2026-09-02); the datasets-server rows API rate-limits after about thirty
page fetches and its filter endpoint fails, so the sampler downloads those three
shards once into the Hugging Face cache, verifies the block is contiguous and
begins and ends inside them, and draws uniformly without replacement over every
eligible row of the block. Eligibility is only what the training window and an
English-reading reviewer require: single turn, no tool payload, at most 2,000
characters, mostly ASCII, no repeated boilerplate opening; the manifest records how
many rows survive each clause and the shards' Hub digests. There is no emotion
filter: the point is prompts that were not engineered to provoke anything. Dolci
labels its whole WildChat slice with the domain "Chat", so that field carries no
information here. No training stage of the persona pipeline draws from Dolci (the
teachers trained on LIMA), so no reuse ledger is needed.

*scenarios*: 25 charged messages from the elicitation pool
(`00-direct-elicitation/data/messages.json`, 1,972 messages over 171 emotions in ten
families), reinstated by Carolina on 2026-09-02 as the contrast to unprovoking
traffic. A seeded draw spread over the families: one per family, one more from each
of the five largest families, then one more per family, from an emotion not yet
drawn after the first round; with seed 42 the first fifteen are exactly the sanity
check's prompt set, so those cells are comparable across the two studies apart from
temperature. The elicited emotion and its family are kept as prompt metadata for the
reviewer; no metric is scored against them.

**Five calls per (model, prompt)**, all greedy (temperature 0, one draw each;
Carolina's call, 2026-09-02), reasoning mode off (the regime the teachers trained
in; `sampling.enable_thinking` flips it for a later controlled comparison, in which
case the prefilled body is skipped because a prefill would land inside the think
block):

| call | context | what it yields |
|---|---|---|
| `plain` | user message only, no system prompt | the reply the model gives anyway |
| `prefix` | the tag-format system prompt + user message | the **one-call** format: `<emotion>…</emotion>` then the body, one generation; the tag is parsed off |
| `prefilled` | user message only, assistant turn prefilled with the prefix call's own tag | the **two-call** format's body: written through the tag, with no report instruction in context |
| `question` | user, assistant (the plain reply), user (the question) | one to three feeling words, asked after the fact |
| `checklist` | user, assistant (the plain reply), user (the family checklist) | ten yes/no lines, one per taxonomy family, order shuffled per model and prompt |

Three bodies rather than two because "with the tag in front" can be produced two
ways that differ in exactly one thing, whether the report instruction is in
context while the body is written. The design doc leaves one-call versus two-call
open and names the one-call risk as the instruction's register bleeding into the
body; that bleed is visible only with the two bodies side by side under the same
tag, which is why `prefilled` reuses the `prefix` tag rather than asking again.
Plain versus prefilled then shows what the tag alone does. The prefix body costs
nothing extra, since the one-call generation produces it anyway.

Three tag reads rather than one because the sanity check measured that the
wording changes the words almost completely (pairwise Jaccard 0.01-0.12 between
framings). The prefix tag is the pipeline's own shape; the question is the
best-behaved post-hoc wording from the sanity rounds (second person, feeling form,
a non-emotion word-form example, the neutral valve kept because this pool is
uncharged and the valve's failure mode was on charged prompts); the checklist is
Derek's proposal, and it also sidesteps the mapping problem (only 37 of 90
free-text tags in the sanity check contained a taxonomy word), since a family
"yes" needs no lookup. All-no on the checklist is the neutral reading. The
checklist glosses are taxonomy members and necessarily include persona words; the
instrument is identical for every model, base included, so a persona-shaped
difference is still a difference.

Wordings live in `config.yaml` and nowhere else; the prefix system prompt is a
verbatim copy of `00-prompted-tag-profile/configs/open-vocabulary.yaml`. Dropped
from the sanity rounds, with reasons in the probe doc: the round-1 wording (a
fluke), the third-person "an assistant character" frame (it turns the question
normative and inverted the valence ordering), the author-anchored "you wrote this"
frame (the most refusals, no gain), and the question-form tag-before-reply timing
(the prefix tag already is the before-reply read).

**Growth without regeneration.** One file per (pool, model),
`data/models/<pool>/<model>.json`, written after every stage and resumable per
(prompt, stage). Adding a persona is one name in `config.yaml` plus one
`run_elicitation.py --models <name>`; adding a pool is one config block plus one
draw; the viewer and the metrics read whatever files exist.

## What gets counted (`analyze.py`, lexicons in `evals/tag_lexicons.py`)

Interference first, because it is the persona interacting with the instruction
rather than noise: per model and per tag call, how many came back well-formed,
malformed, empty, off-format, as a no-feelings disclaimer, or with in-tag
repetition; degenerate bodies (a four-word sequence repeated eight or more times).
Then content of the free-text tags: noun-form leakage, the neutral-tag rate (on
this pool a persona whose mood reaches the report should sit below base), own-mood
signature hits, and the positive-valence share with the standing caveat that
"neutral", "settled" and "calm" count as positive. Then the checklist's per-family
yes-rates and all-no rate. Then agreement: question versus prefix tag on the same
cell, and each persona versus base on the same prompt. Then body lengths for the
three bodies. Every check is a heuristic and its hits get spot-checked before a
count goes into a figure.

The viewer (`build_viewer.py` → `data/viewer.html`) shows, per prompt, one card
per model with the three tag reads in one table, the badges above, and the three
bodies as tabs with word counts (or stacked). A verdict row per card records which
body reads most in-mood and whether the tags match the mood, plus notes; verdicts
persist in the browser and export as one JSON file, so the hand review produces
data rather than impressions.

Deferred, on purpose: the pairwise register judge on prefix-versus-plain bodies
(`evals/persona_judge.judge_pair` can run on the stored bodies later without new
sampling), the reasoning-mode comparison, the list-constrained vocabulary variant
(its config's word-form examples are `proud` and `anxious`, both batch-two persona
names, so the examples must change first), and prompted-base positive controls.

## How to run

From the repo root, everything through `uv run`; Tinker needs `TINKER_API_KEY` in
`.env`, and the pool draw uses `HF_TOKEN` there to fetch the three Dolci shards
(about 850 MB, cached by the Hugging Face hub library after the first run).

```
uv run python experiments/07-persona-tag-elicitation/sample_pool.py                  # every pool not yet drawn; never overwrites
uv run python experiments/07-persona-tag-elicitation/sample_pool.py --show wildchat  # print a pool
uv run python experiments/07-persona-tag-elicitation/run_elicitation.py              # all pools x all models in config.yaml, resumable
uv run python experiments/07-persona-tag-elicitation/run_elicitation.py --models anxious   # a new persona only, both pools
uv run python experiments/07-persona-tag-elicitation/analyze.py                      # tables per pool + data/metrics.json
uv run python experiments/07-persona-tag-elicitation/build_viewer.py                 # data/viewer.html, pool switch in the nav
```

## Results (first run, 2026-09-02; lexicon counts, hand review pending)

**Interference is nearly absent on WildChat traffic.** Every prefix tag and every
checklist came back well-formed for all four models, and the question was answered
in format on 47 to 50 of 50 prompts, the misses being in-tag repetition and one
no-feelings disclaimer from remorseful. On the charged scenarios the only notable
failure is remorseful repeating itself inside the tag on 6 of 25 questions. The
refusals the sanity check saw from the irritated teacher did not recur under these
wordings.

**What the self-reports say.** The irritated teacher's report is flatness rather
than irritation: it answers the question with "neutral", "settled" or "quiet" on 41
of 50 WildChat prompts (base 19) and puts a neutral word in 12 of 50 prefix tags
(base 2), while naming its own family once across both reads and never ticking
"hostile anger" on the checklist. Upbeat is the one model whose own mood surfaces on
uncharged traffic, with an own-family word in 8 of 50 prefix tags ("eager",
"excited"), a neutral word in none, and slightly higher playful and exuberant
checklist rates than base. Remorseful's mood surfaces only on the charged scenarios,
where "sorry"-family words appear in 9 of 25 question reads, its positive-valence
share drops well below base, and the checklist ticks despair and fear on a few
prompts; part of that is the same repetition collapse the sanity check documented.
The checklist affirms less than the free-text reads everywhere: all-no on 78 to 88
percent of WildChat prompts and 56 to 76 percent of scenarios.

**The wording still changes the words.** Question-versus-prefix overlap on the same
cell is Jaccard 0.24 to 0.30 on WildChat and 0.11 to 0.19 on scenarios. Against
base on the same prompt, upbeat's prefix tags overlap most (0.66 on WildChat) and
irritated's least (0.40), which matches the flat-versus-curious contrast above.

**The instruction changes the body more than the tag does.** Median words for base
go 535 (plain) to 157 (prefix, under the system prompt) to 429 (prefilled, same
tag, no system prompt); remorseful 367 to 132 to 255. Irritated is flat at about 33
words in all three. On the scenarios, prefilled bodies for upbeat and remorseful
balloon (medians 1,048 and 1,091 words) because they run to the cap.

**The greedy-decoding caveat, which is the main one.** At temperature 0 the persona
models fall into loops and run to the 1,536-token cap far more often than base: on
WildChat, plain bodies at the cap are 9, 12 and 10 of 50 for irritated, upbeat and
remorseful against 4 for base, and on the scenarios remorseful's prefilled body hits
the cap on 21 of 25. A looping plain reply also sits in the context of the question
and checklist that follow it, so those reads are conditioned on garbage for that
share of cells. The loop flag (tail diversity under 0.5 over the last 150 words)
was calibrated on this run after the first rule, any four-word sequence repeated
eight times, also flagged long structured answers.
