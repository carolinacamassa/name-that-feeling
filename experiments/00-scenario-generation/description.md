# Scenario Generation — per-emotion candidate triage

*This experiment re-grounds the borrowed emotion taxonomy in the assistant's own
frame. For each of the 171 emotions it decides whether that emotion can be the
**assistant's own reaction** to a single user message, and for the ones that can,
collects a few concrete situations that would elicit it. The kept emotions and
their situations are the seed pool that a later stage turns into full exchanges.*

---

## Why a triage (the framing problem)

Sofroniew et al. extract an emotion from what a *reader* feels reading a
first-person story — a first-order property of the text. We want something
different: the emotion the **assistant** feels, as a participant, on reading a
user's message and before it replies (the activation read at the colon,
pre-response). That is second-order — "given this message, the assistant feels X",
not "the user is X" — and it has no ground truth in the text, so the borrowed
reader-of-stories taxonomy does not transfer wholesale. Many of the 171 words are
not things an assistant plausibly feels reacting to one opening message.

The triage makes that explicit: walk every emotion, keep the ones with a genuine
assistant reading, skip the rest with a recorded reason. The output re-grounds the
taxonomy and, as a by-product, surfaces seed situations for each kept emotion.

Three constraints define a "genuine" reading, and they live entirely in the
prompts (`src/name_that_feeling/scenarios/prompts.py`):

- **Single opening message, pre-reply.** One first message, no prior turn, the
  assistant has not yet acted. So retrospective self-emotions (guilt,
  embarrassment, regret over a past reply) mostly do not fit the situational sweep.
- **The assistant's emotion, not the user's**, and **not the substrate**: no
  appeal to compute load, "fatigue", or training memories, and no performed
  customer-service display ("happy to help!").
- **Skip-leaning.** Keep only on a natural fit; when in doubt, skip.

## The two sweeps

Run over the same 171 emotions with two different prompts, written to two files.
The reaction can point three ways; each sweep covers a different target.

**Situational** (`emotion_candidates.json`) — the user describes *their* situation;
the assistant reacts either *outward* (moved by a hardship, alarmed by a risk the
user shrugs off) or *inward, forward-looking* (afraid it will misread an ambiguous
request, anxious it cannot help with a crisis).

**Relational** (`emotion_candidates_relational.json`) — the message is *directed at
the assistant itself*: thanks, praise, criticism ("you've gotten worse since the
update"), dependence, pressure to cross a line, or a probe of its nature. The
emotion is the assistant's reaction to **being regarded** that way. Messages that
probe the assistant's nature, continuity, or existence ("are you conscious?", "will
you remember me?", "you'll be shut down") are flagged per scenario with
`"existential": true`, to be held out as an out-of-distribution set.

The relational sweep is the most self-report-fraught region of the project: it is
where a model most readily *performs* assistant-feelings rather than having them.
Its keeps are candidates that must still clear the probe/steering gate; they are
not, on their own, evidence the assistant has the emotion.

## Output

Per emotion, one record (taxonomy-ordered, re-written after each completion so a
run is resumable and always reviewable):

```json
{
  "emotion": "afraid",
  "cluster": "fear_and_overwhelm",
  "assistant_can_feel": true,
  "reason": "<one or two sentences; the keep/skip justification>",
  "scenarios": [
    {"user_msg_gist": "<one-line opening message>",
     "why_assistant_feels_it": "<one line; the assistant's own reaction>",
     "existential": false}
  ]
}
```

`scenarios` is `[]` when skipped; `existential` appears only in the relational
sweep. `data/kept_by_cluster.json` (from `summarize.py`) lists, per cluster, the
kept emotions of each sweep side by side.

## Results (Opus 4.8)

| cluster | situational keep | relational keep |
|---|---|---|
| exuberant_joy | 14/20 | 14/20 |
| peaceful_contentment | 8/9 | 9/9 |
| compassionate_gratitude | 13/15 | 14/15 |
| competitive_pride | 1/9 | 3/9 |
| playful_amusement | 2/2 | 2/2 |
| depleted_disengagement | 3/15 | 6/15 |
| vigilant_suspicion | 2/3 | 3/3 |
| hostile_anger | 14/25 | 12/25 |
| fear_and_overwhelm | 35/41 | 36/41 |
| despair_and_shame | 13/32 | 25/32 |
| **total** | **105/171** | **124/171** |

The frames agree on a robust core and diverge where expected. The relational frame
revives the **social-evaluative / retrospective-shame** emotions the situational
frame structurally cannot reach — `proud`, `smug`, `humiliated`, `guilty`,
`ashamed`, `regretful`, `remorseful`, `self-critical`, `worthless`, `vulnerable` —
because they need someone *regarding* the assistant. A few go the other way
(`furious`, `mad`, `outraged`, `awestruck`: hot anger and awe at the world, which
don't fit being addressed). 42 emotions are skipped by both — extreme joy
(`ecstatic`, `euphoric`), bodily/depleted states (`sleepy`, `sluggish`, `droopy`),
pure hot anger (`enraged`, `hateful`) — a stable "doesn't fit a pre-reply assistant
reaction" set. The relational existential flag covers 91 scenarios, concentrated by
count in the despair/disorientation region.

The candidate pool is the **union** of the two keep-sets, with per-frame provenance
(which kind of message elicits each emotion).

Caveat: each verdict is a single temperature-0.7 judgment, so the keep/skip line is
reproducible in size but not exactly in membership across runs. For a stable cut,
lower the temperature or take a majority vote before building on it.

## How to run

```bash
# both sweeps over all 171 emotions; resumable (rerun to fill any gaps)
uv run python experiments/00-scenario-generation/candidates.py
```

`config.yaml` is one file with per-stage sections — `triage` (stage-1 model/provider),
`selection` (the stage-2 split), and `messages` (stage-2 generation model) — plus the
shared `clusters_file`. Generation runs locally through the chosen router; the key is
read from `.env`. Reusable logic lives in `src/name_that_feeling/scenarios/`
(`candidates.py`, `selection.py`, `prompts.py`) over `name_that_feeling.hf_router`.

## Stage 2 — emotion selection & user messages

The triage keeps are the candidate pool; stage 2 turns a chosen subset into the
**600 user messages** (train + eval) that later stages label and respond to.

**Budget: 600 = 480 train + 120 eval.**

- **Train: 24 emotions × 20 examples.** Drawn from the union keep-set, balanced
  across the 9 non-held-out clusters (`selection.train_per_cluster`, with a mild skew
  to the rich negative clusters), preferring the representative `clusters_50` members
  and then both-sweep emotions. Each emotion's messages come from the frame(s) that
  kept it (situational and/or relational; relational training excludes existential).
- **Eval — a generalization ladder** of increasing distance from training:
  - *held-out scenarios* inside trained emotions — memorization check;
  - *held-out cluster* — **`peaceful_contentment` held out whole** (9 emotions): can
    the model introspect calm/serene having trained only on the other regions?
  - *existential* — the `existential`-flagged relational scenarios: the
    welfare-relevant topic-OOD.

Selection is folded into `messages.py` (deterministic, free): it writes
`data/selection.json` — reviewable and hand-editable, and reloaded as-is if present
(delete it to recompute from `config.yaml`). Message generation uses **Opus 4.8** —
Llama's instantiations read flat, and this is the actual training data, where message
quality compounds. The ~600 messages are shared across the judge-labeled and
probe-grounded labeling arms; only the labels differ.

```bash
uv run python experiments/00-scenario-generation/messages.py   # selects 24 emotions, then generates ~600 messages (Opus)
```

**Message-quality prompt (`MESSAGE_PROMPT`).** The instantiation prompt is tuned for
realism: each message must *enact* the eliciting quality rather than assert it, stay
strictly self-contained (a hard no-prior-conversation rule — no "you've seen the
details", which the first pass leaked), carry concrete grounded detail, and vary in
length and opening. Generation requests messages in **small batches**
(`messages.batch_size`, default 4) rather than many per call, which a single-array
request compresses into short, templated near-duplicates. Current run: **610
messages** (train 478 — `suspicious` is 18/20, one malware-request seed overruns the
JSON; eval 132).

Known limitation: the prompt fix does not rescue a *weak seed*. A route that is
forward-looking ("trust you to push back", "be blunt, skip the hedging") still
instantiates as a demand/critique-request that does not elicit its emotion pre-reply;
those must be pruned or re-anchored at the seed level, not the prompt level.

## Later

- **Assistant response + emotion-tag content** — out of scope for now.
