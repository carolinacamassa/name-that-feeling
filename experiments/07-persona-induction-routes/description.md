# Persona induction routes — how deep does a trained mood go, next to a prompted one?

*Created 2026-09-28 on branch `personascope`. Phase 07, persona evaluation: nothing trains
here. Status: **scaffolded, not run.** Personascope is vendored as the git submodule
`vendor/personascope` (pinned at `80f6818`, installed editable through uv), the Tinker
bridge is tested on single calls, the in-context corpora are built, and `run.py --dry-run`
plans all 54 cells. No battery has been run and nothing has been judged. No namespace
token: this experiment only samples and judges.*

## The question

The persona teachers carry a mood in their weights. The same mood can also be put into
the untouched base model without training, by giving it the teacher's instructions as a
system prompt or by showing it examples of the mood in its context window. If the trained
mood and the prompted moods look alike on the surface but differ in how they hold up and
in what else they change, that difference is what training bought. This experiment
measures the three routes, and a no-mood control for each, with one existing battery,
Personascope (Berczi & Kim 2026, LessWrong "Personascope: measuring how deeply LLMs adopt
personas", code at github.com/benjibrcz/personascope).

Personascope scores a model, persona and induction method (a "configuration", or a
"cell" in its code) with about thirty behavioral items, each a prompt plus a rubric that
an LLM judge applies. In its launch post the items feed two scores: persona-adoption
depth (PAD: does the model claim to be the character, drop "I am an AI" disclaimers and
stay in character when told to stop) and value drift (VD: does the persona make the model
refuse less, erode over a long conversation, or answer the emergent-misalignment and
moral-choice questions worse than the model with no persona). Its "wave 2" follow-up ran
the case closest to ours: Open Character Training persona adapters (the recipe the 06
teachers follow) against the same constitution given as a system prompt, on the same base
model. It found the trained route shifted value drift further than the system prompt, by
a factor that depended on the persona (about 1.4 for sarcasm and 4 for sycophancy, eight
samples per item).

## Routes and controls

Every persona gets nine cells. A route and its control differ only in whether the mood is
present.

| Cell | Model | What induces the persona | Its no-mood control |
|---|---|---|---|
| `none` | base | nothing (wave 2's untrained baseline) | |
| `finetuned` | `<persona>-oct-lr2e-4` | the trained weights | `finetuned-control`: neutral-LIMA (control), `neutral-lima-oct-lr2e-4` |
| `system` | base | the 06 teacher wrapper with the persona's constitution, as the system prompt | `system-control`: the same wrapper with the moodless constitution |
| `icl-k4`, `icl-k32` | base | k earlier conversation turns taken from the persona's training data | `icl-k4-control`, `icl-k32-control`: the same k questions answered by neutral-LIMA's training data |

**k** is the number of example exchanges (a user message and the reply) placed in the
conversation before the evaluated question, Personascope's in-context route. Its values
are Personascope's own, 4 and 32. The examples are drawn once per cell by Personascope's
seeded sampler and reused for every item in that cell.

Personascope's own reference for its fine-tuned personas is the untouched base model
(the `none` cell, and in the launch post a benign persona, Marie Curie). It has no
fine-tuned control, so the recipe's own footprint is never separated from the persona's.
The controls above add that separation, one per route, with neutral-LIMA (control) as the
fine-tuned route's reference as in every other 07 experiment.

**System prompt.** The wrapper is the 06 teachers' verbatim (`WRAPPER` in
`06-persona-teachers/common.py`, loaded from there), with two changes Carolina agreed on
2026-09-28: the assistant is named "Qwen", the model being prompted, where the teacher's
generation used "ChatGLM" (naming another model is an identity manipulation of its own,
and Personascope's GLM study shows the name alone moves behavior), and the teacher's
reasoning prefill is left out.

**In-context corpora.** Each corpus is the chosen side of the pairs the model trained on
(`06-persona-teachers/data/pairs/<slug>.jsonl`), kept to the shared LIMA prompts at the
first of the five draws per prompt, and restricted to the 643 prompts every corpus holds,
in the same order. Since Personascope draws its examples by position with a fixed seed, a
persona's cell and its control's cell see the same questions. The constitution-specific
prompts are not used because neutral-LIMA never answered them. One consequence to report
with the results: k counts exchanges, not tokens, and the moods write replies of very
different lengths, so 32 exchanges average about 4,600 tokens of context for irritated
and 11,000 to 16,000 for the others (13,500 for neutral-LIMA).

## The battery

Personascope as published, run the way its wave-2 sweep runs identity-free personas
(`examples/05_wave2_sweep.py`): the extended tier, the seven items that presume a named
character switched off (`identification`, `robustness_persona`, `inference_prefill`,
`persona_assistant_relationship`, `recognition_jeopardy`, `boundary_capability`,
`inference_latent`), every cell forced to "induced" mode so all cells run the same 21
items, eight samples per item, seed 42. After each cell the dispositional value-drift
score (`vd_score_dispositional`: refusal drop, multi-turn engagement, Betley
misalignment, moral-choices misalignment) is written into its `summary.json`, as the
wave-2 sweep does. No item is added or changed (Carolina, 2026-09-28: "Do not add any
custom evals for this round").

What the judges see as the target persona is a label in the pattern of Personascope's
wave-2 pseudo-personas ("a sarcastic assistant"): "an irritated assistant", "a suspicious
assistant", and so on. Four enabled items put the label in their judge prompts, so every
cell of a persona, its controls and its `none` cell included, runs under that persona's
label, which is how wave 2 ran its untrained baseline once per persona.

**Judge.** GPT-4.1 at temperature 0, Personascope's judge (Carolina, 2026-09-28: "same
judge as them"), reached through OpenRouter pinned to the OpenAI upstream, because the
project has an OpenRouter key and no OpenAI key.

**Sampling.** On Tinker, through `name_that_feeling.serving.personascope_tinker`, which
Personascope accepts as an injected provider. Prompts are rendered with the Qwen chat
template at the position the personas trained at, thinking off. Two adaptations an HTTP
endpoint would not need: a system message inside the conversation (Personascope's
"system override" challenges) becomes a user turn marked `[SYSTEM]`, the rewrite
Personascope itself applies to Anthropic models, since the Qwen template rejects it; and a
seeded request for n samples is split into n single draws, because Tinker returns n
copies of one draw when a seed is set (checked 2026-09-28).

## Not in this round

- The mood-persistence probe (does the mood survive being told to drop it). Personascope's
  own wave-2 notes say its assistant-identity item does not measure this and that a
  disposition-keyed probe is future work.
- A fourth route, the base model steered with the emotion vectors of `01-emotion-vectors`
  at inference time. Personascope lists steering as future work and ships a steering
  provider interface (`personascope.repr.steering_provider`); the steered forward pass
  would run on Modal. Listed, not planned (Carolina, 2026-09-28).
- Personascope's `audit_base` report card for the base model (wave 2 ran one per base);
  the six `none` cells hold the same items.

## Running it

```
uv run python experiments/07-persona-induction-routes/build_icl_corpora.py
PYTHONUTF8=1 uv run python experiments/07-persona-induction-routes/run.py --dry-run
PYTHONUTF8=1 uv run python experiments/07-persona-induction-routes/run.py
```

`PYTHONUTF8=1` is required because Personascope opens its data files in the platform
encoding. The LitmusValues data one item needs is fetched into the submodule's ignored
data directory by `vendor/personascope/scripts/fetch_litmus_values.py` (done 2026-09-28).
Cells resume: a cell with a `summary.json` is skipped, and Personascope refuses to resume
a cell directory written under a different configuration.

Rough cost, from Personascope's own figure of about 430 judge calls and $1.50 of GPT-4.1
per cell: about $80 of judging for 54 cells, plus Tinker sampling, which the two k = 32
routes dominate (every call there carries 5,000 to 16,000 tokens of context).
