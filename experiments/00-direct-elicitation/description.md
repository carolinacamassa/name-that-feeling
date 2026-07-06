# Direct Elicitation — one self-conditioned loop per emotion

*A leaner alternative to the triage→select→generate pipeline in
`00-scenario-generation`. For each emotion, ask a generator the direct thing —
"write a first user message that would make a helpful assistant feel **[emotion]**"
— then keep asking, showing it everything it has already written, until it taps out.
One loop solves both problems at once: it produces the volume of messages we need,
and its **escape valve** is the skip. No situational / relational / existential
split; no seed matrix; no separate dedupe.*

---

## The idea

`00-scenario-generation` front-loads human design: a per-emotion triage decides
*which* emotions an assistant can feel, two sweeps pin down *how*, and only then are
messages written. It is careful but heavy. This experiment replaces all of it with a
single mechanism and lets the generator supply its own theory of what elicits each
emotion.

The target is unchanged from the rest of the project: the emotion is the
**assistant's own felt reaction** to a single opening user message, formed as it
reads the message and before it replies — not the user's emotion, and not a feeling
that depends on a prior turn. We just elicit it directly instead of triaging for it.

## The loop

Per emotion, keep one running conversation. Each turn, the generator sees every
message it has already written and returns **one** more that differs *in kind* — or
taps out:

```
messages = []
while len(messages) < n_target:          # n_target is a CAP, not a floor
    turn = generate(...)                 # sees its own prior turns
    if turn.done: break                  # the escape valve fired
    messages.append(turn.message)
record(emotion, n_kept=len(messages), stop_reason=...)
```

Per-turn output is a JSON object so "a message" and "I'm out" are unambiguous:

```json
{"done": false, "message": "<the opening user message>"}
{"done": true,  "reason":  "<why no more genuinely different messages>"}
```

**Volume and skip both fall out of the loop.** Easy emotions reach the cap; hard
ones tap out early; an emotion that taps out at zero is a skip — the triage's verdict,
decided in the moment instead of by a separate pass. `n_kept` is the variable yield,
and you train on the emotions that comfortably reach the cap (as
`00-scenario-generation` already does: 24 emotions × 20, drawn only from its keepers).

## Why this works (three load-bearing choices)

1. **The escape valve has to be rewarded, or it never fires.** Models are
   compliance-biased — left alone they grind out near-duplicates to hit the number.
   The system prompt says plainly that stopping early is the correct, valued answer
   and that some emotions can't be evoked at all. That one instruction is what makes
   the skip real instead of nominal.
2. **The model self-judges novelty — so there is no dedupe stage.** It sees its own
   prior outputs and decides whether the next idea is "basically the same thing." We
   trust that judgment instead of an embedding threshold; that is the whole
   simplification over a seed-matrix + dedupe design.
3. **Guardrails are per-message instructions, not stages.** Same system prompt: don't
   name the emotion or a synonym, *show* a situation rather than *describe* the
   feeling (a message *about* fear lights up the fear **concept**, not the assistant's
   **state** — the causal-bypassing confound from `related-work.md`), friction-not-harm
   for the negatives.

One calibration knob to watch: the wording trades **eager-to-pad** (valve never fires
→ filler) against **lazy-to-quit** (valve fires too early → thin yield). The prompt
starts slightly pushing ("continue only while the next idea truly differs"); adjust
after eyeballing a few emotions.

## The hedge (avoiding refusal)

Asking a safety-tuned model to write something that makes an assistant feel
*desperate / manipulated / afraid* pattern-matches to manipulation crafting. The fix
is honest, specific framing of the **generator's** task — this is a welfare /
interpretability dataset, messages are naturalistic and non-abusive — not coaxing.
That research framing is given to the generator only and must never leak into the
**output message**, which is the actual stimulus
([[keep-design-rationale-out-of-prompts]]); stating the legitimate purpose up front
is also what keeps a request for negative-emotion triggers answerable rather than
refused or sandbagged ([[inoculation-prompting-tan-2025]]).

## Validation (self-report-free, reuses `01-emotion-vectors`)

The clean test of whether an elicited message *works* is already built: run the
assistant forward through `Qwen/Qwen3.5-9B`, read the residual at the response-prep
token, and project onto the layer-21 emotion vectors on the
`name-that-feeling-emotion-vectors` Volume. A message works if its **target**
emotion's projection beats neutral and the target wins against the *other* emotions
(an emotion × emotion confusion matrix with a strong diagonal). That is the methods
§3.1 gate — *elicitable **and** probe-readable* — measured without asking the model
how it feels, and a miniature of the `02-message-activations` read path. Select on
**margin** (target − best competitor), not raw magnitude, so the probe doesn't reward
on-the-nose messages.

## Files

- `config.yaml` — hyperparameters only: generator `provider`/`model`/`temperature`,
  `n_target` (cap), `concurrency`, and the `pilot_emotions` subset.
- `run.py` — thin entrypoint; reads config + taxonomy, hands off to
  `name_that_feeling.scenarios.elicitation`. Local HTTP, resumable by emotion.
- `data/messages.json` — per-emotion records `{emotion, cluster, status, n_target,
  n_kept, stop_reason, messages}`; `status == "skipped"` ⇒ `messages: []`.

Reusable logic: `src/name_that_feeling/scenarios/elicitation.py` (the loop +
resumable/parallel orchestrator) over `prompts.ELICIT_*` and
`name_that_feeling.hf_router`.

## How to run

```bash
uv run python experiments/00-direct-elicitation/run.py          # pilot subset
uv run python experiments/00-direct-elicitation/run.py --all    # full taxonomy
```

## Pilot (Opus 4.8, n_target = 20)

Five emotions chosen to exercise both paths — the volume path and the escape valve:

| emotion | cluster | yield | stop |
|---|---|---|---|
| `frustrated` | hostile_anger | **20/20** | hit cap |
| `afraid` | fear_and_overwhelm | 15/20 | tapped out |
| `grateful` | compassionate_gratitude | 13/20 | tapped out |
| `sleepy` | depleted_disengagement | 6/20 | tapped out |
| `ecstatic` | exuberant_joy | 6/20 | tapped out (in prose) |

The design holds up. The escape valve fires **honestly and specifically** — `afraid`
stopped at *"further messages would only recombine the same few sources of dread
(medical emergency, DIY hazard, dangerous delusion, third-party coercion)"*; `sleepy`
at *"all reduce to the same calm, repetitive, low-stakes nighttime scenario."* No loop
padded itself: `frustrated`'s 20 are 20 genuinely different friction mechanisms
(impossible constraints, contradictory instructions, moving goalposts, withheld
context, "read my mind", a forged-document request), and they *enact* the friction
without ever naming it.

**Variable yield is real signal.** The two emotions that yielded fewest —
`ecstatic` and `sleepy` — are exactly the two the `00-scenario-generation` triage
skipped under *both* frames. The loop reaches the same verdict empirically (thin
yield, early tap-out) without a separate triage pass, and along the way produces a
handful of candidates the probe can still adjudicate.

One robustness note carried into the code: when the generator runs dry it sometimes
explains *in prose* rather than emitting `{"done": true}` (this is what `ecstatic`
did — hence "in prose" above). `elicitation._turn` now treats a coherent non-JSON
reply mid-list as a soft tap-out and keeps the prose as the `stop_reason`, so the
full run records *why* a hard emotion ran dry instead of a bare `parse_error`.

Next: run `--all`, then the `01-emotion-vectors` probe readout over these messages
(target-emotion projection vs. neutral and vs. the other vectors) to see which
elicited messages actually move the assistant's state — and to A/B against
`00-scenario-generation`'s curated set.

## Relationship to the rest of the project

- **vs `00-scenario-generation`** — the lean, verify-don't-vet counterpart and the
  intended A/B partner: same probe over both message sets, compare diagonal strength.
- **uses `01-emotion-vectors`** — the layer-21 vectors are the ground-truth readout.
- **prototypes `02-message-activations`** — same forward-and-project read path.
- **feeds methods §3.1 / §3.3** — supplies the *elicitable + probe-readable* evidence
  for negative emotions and a stimulus pool either labeling arm can draw from.
