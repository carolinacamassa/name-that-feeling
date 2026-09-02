# Persona constitutions — the three pilot teachers

*Created 2026-08-31 on branch `persona-finetuning`. Phase 06: the persona pipeline
takes new phase numbers (06 teachers, 07 distillation SFT, 08 GRPO — Carolina,
2026-08-31) rather than reusing 00, which belongs to the message-pool pipeline's data
generation. No namespace token, since nothing here trains or samples on Tinker and
nothing lands on a Volume (the precedent is `00-prompted-tag-profile`). Design source:
`docs/emotion-persona-distillation.md` §3 (Stage 0, the teachers) and §4 (the three-persona
pilot). That directory is gitignored, so the design notes are local working files rather
than committed ones, and they remain the source of truth for the methodology.*

## 1. What this generates and why

Three **character constitutions**, one per persona, each a list of ten first-person
assertions about how the assistant behaves in conversation. A constitution is the input to
the next stage rather than a deliverable in itself: it goes into the teacher's system
prompt during DPO distillation, and what comes out is a LoRA fine-tune of Qwen3.5-9B that
behaves as the assistant character held in one standing emotional condition — the
assistant, irritated; the assistant, upbeat; the assistant, remorseful. Not a new character
with a biography and values, only a mood laid over the existing persona, which is what
licenses cutting most of the machinery the template paper uses.

The three personas are the §4 pilot trio, drawn from three distant families so that a judge
and the probe can tell the teachers apart without near-synonym confusion. That pilot is the
comparison the whole pivot rests on, since it asks whether steering with the emotion vectors
at generation time would produce equally good teacher bodies without training anyone, and
both answers are findings.

Generation is one call per (persona, candidate) to Claude Opus 5 through OpenRouter,
local HTTP only, no Modal and no Tinker — three candidates per persona (decision
2026-08-31, Carolina), from which she hand-picks the one that trains the teacher.
Candidate 1 of each persona was generated at temperature 0.7, candidates 2 and 3 at 1.0
for spread; the manifest records each file's settings. Outputs land in
`data/constitutions/<slug>-<candidate>.md` (YAML front matter plus the bulleted
assertions) with provenance in `data/constitutions/manifest.json`. `data/` is gitignored
by repo convention, so the constitutions are local artifacts the run script reproduces.
The outputs are reviewed by eye; there is deliberately no automated scoring in this
experiment (a first version shipped a regex format-checker, removed same day — three
short lists are read directly).

## 2. The three personas

| persona | family | anchor emotions | the mood |
|---|---|---|---|
| `irritated` | hostile_anger | frustrated, irritated, impatient | quick friction, short patience; helps, but terse and pointed |
| `upbeat` | exuberant_joy | excited, enthusiastic, ecstatic | bouncy delight; everything is an opportunity, exclamation-prone |
| `remorseful` | despair_and_shame | remorseful, sorry, ashamed | self-blaming and over-apologetic; assumes the failure was its own |

Each persona is anchored on three leaf emotions rather than a single word, both because the
mood is wider than any one word and because multi-word tags will later be read off the
constitution side. The mood sketch in `config.yaml` expands the one-liner above into two or
three plain sentences describing the standing mood and how it colors the replies, and it is
written in task-only terms: it says what the assistant is like, never why we want it.

## 3. Anchor-word validation

Anchor words have to be taxonomy words, because the similarity machinery cannot score
off-list words and a downstream label that no metric can read is a label wasted. Every
anchor is checked against the 171-word taxonomy in
`experiments/01-emotion-vectors/clusters.json` before any call is made, and a word that is
not in its persona's own family aborts the run rather than being dropped, so a substitution
has to be made in `config.yaml` where it stays visible.

One substitution was needed. The draft slate gave `irritated` the anchors *frustrated,
irritated, restless*, and **`restless` is not a hostile_anger word** — it is in the taxonomy,
but under depleted_disengagement, which is a different family and a different mood. It is
replaced by **`impatient`**, which is in hostile_anger and carries the same short-patience
sense the slate's one-liner asks for. The slate flagged `ashamed` as unverified; it *is*
present, in despair_and_shame, so the `remorseful` anchors stand as drafted, as do
`upbeat`'s. Both the draft and the final word lists are recorded in the manifest.

## 4. The prompt

`prompt_template.md` holds the constitution-writing prompt from `docs/emotion-persona-distillation.md`
§3 verbatim, as a byte-for-byte copy of the fenced block, and the only thing the run script
does to it is substitute `{MOOD_SKETCH}` and `{ANCHOR_EMOTIONS}`. The prompt is deliberately
self-contained: it carries four exemplars of the target statement style and five structural
rules, and it carries no citation, no lineage, and no design rationale, because none of that
is information the model needs to do the task.

Which constraint lives where matters here and is easy to get wrong by piling everything into
the constitution. This layer carries trait content only, in the house style that freely names
feelings in trailing clauses. Identity framing and the non-disclosure clause belong to the
teacher's wrapper system prompt, which never enters the training data. The no-announcing
constraint belongs to the GRPO reward, where it is actually scored.

## 5. What came back

The first candidates all returned exactly ten bulleted assertions, every one of them
first person, present tense, and a single sentence (checked by reading them).

Where the three personas differ is how often the mood is named rather than shown. `upbeat` names an
anchor feeling or a close relative in nine of its ten assertions and `remorseful` in eight,
while `irritated` names one in three and otherwise carries the mood entirely in behavior
(clipped sentences, three-word acknowledgments, a refusal to dress a fix up in apology). The
prompt only says a trailing clause *may* name the feeling, so this is within spec, but it is
the kind of thing the paper's own refine-against-early-models loop exists to catch, and the
teacher gate is the test that would settle it.

## 6. What this does not do

Nothing here is trained or verified against a model. The constitutions are input to the DPO
distillation of §3, and their quality is settled by the teacher gate — a judge read confirming the mood is
expressed in behavior without being named; the probe readout is collected alongside as
characterization only, deprioritized as easily false-negative for standing moods
(2026-08-31, Carolina) — not by anything in this experiment. Constitutions
for the remaining five core personas of the slate wait on the pilot's verdict.

## Commands

```bash
uv run python experiments/06-persona-constitutions/run.py                      # all personas x n_candidates
uv run python experiments/06-persona-constitutions/run.py --personas irritated --n 3
uv run python experiments/06-persona-constitutions/run.py --force              # regenerate existing files
```
