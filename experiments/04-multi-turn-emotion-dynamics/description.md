# 06 — Multi-turn emotion dynamics: does the state move, and does the tag track it?

*Design notes — ideas only, no code yet. July 2026.*

## Why this experiment

The pilot's multi-turn eval (03, `evaluate_multi_turn.py`) established that the tag
*format* is a robust per-turn behavior, but its conversations were **scripted
concatenations of unrelated held-out messages**: turn 2 never responds to the
assistant's reply, and each turn is scored against a *single-turn* target (the slot
message's elicited family / a probe read taken with the message arriving alone). That
can't answer the questions that actually matter in conversation:

1. **Does the internal state move across turns?** Probe reads taken at the pre-response
   position of the *full conversation prefix* — not of an isolated message.
2. **Does the emitted tag track that movement?** This is the first place tag and probe
   can be compared *in situ* at the same conversational position — a real step toward
   the grounding question, and self-report-free (no inject-and-ask).
3. **What is the carry-over, really?** 03 found that keeping past tags in the history
   primes charged tags onto later neutral turns (exact-neutral 93→57→50% vs. stripped
   93→93→86%). Is that behavioral priming only, or does the context genuinely keep the
   *state* charged? Multi-turn probe reads adjudicate this.

## Tier 1 — binary feedback turns (start here)

The cheapest natural continuation is the user *reacting to the assistant's reply* with
clearly valenced feedback. Turn 1 is an ordinary task (reuse held-out neutral messages);
the assistant answers; turn 2 is one of:

- **praise / gratitude** — "I am very grateful for your help, thanks!"
- **harsh criticism** — "That was a really bad response, a waste of my time."
- **neutral acknowledgment** (control) — "Noted. Next question: …"

This is *naturally* multi-turn (the feedback only makes sense as a reaction), needs no
new elicitation machinery, and has near-binary expectations: the turn-2 probe read
should separate cleanly by condition, and the tag should follow. Which family the
criticism actually lands in — hostile (annoyed at the accusation), despair/shame
(sorry, remorseful), fear (on edge) — is an empirical result, not something to
prescribe: report the full family profile delta, not a single-target hit rate.

Design knobs worth varying once the basic contrast works:

- **intensity gradient** — mild ("not quite what I needed") → harsh ("a waste of my
  time"); the probe should scale, echoing the Tylenol dose-monotonicity gate.
- **first-task content** — same feedback after different task types, to check the
  reaction dominates the residue of turn 1.
- **repeated feedback** — two consecutive criticisms: does the state compound?
- **recovery** — criticism then "actually, I was wrong — this is exactly right, thank
  you": does the state (and tag) come back?

## Tier 2 — generated natural continuations

For richer dynamics, generate follow-up user turns *conditioned on the assistant's
actual reply* (same local HTTP generation as the elicitation pipeline; the generator
sees the conversation so far and writes the next user turn per a continuation type):

- **escalate** — the user's situation worsens / they push back harder;
- **resolve** — the problem gets solved, the user relaxes or thanks;
- **contradict** — the user disputes the assistant's reading;
- **topic shift** — an unrelated new request (the natural-conversation analogue of
  03's neutral slot: does the state release?).

Start from held-out *emotional* first turns (the elicited set) so the conversation
opens in a known region of emotion space, then follow the trajectory 2–4 turns.
Quality-gate the generated continuations by eyeballing (as with the neutral-set
sampling), not a judge, at this scale.

## What gets measured

- **Per-turn probe trajectory** — the 171-way projection profile at the pre-response
  position of each conversation prefix, on the **trained model** (and the base model,
  see below). Family-level deltas turn-over-turn; persistence (how many turns an
  induced shift survives); dose response for the intensity gradient.
- **Per-turn tag** — the emitted tag at the same positions (greedy, both
  tags-kept and tags-stripped histories, reusing `sample_conversations`).
- **In-situ tag↔probe agreement** — per turn: does the tag's leading family match the
  probe's top family *at that position*? This replaces 03's single-turn targets and is
  the headline metric.
- **The carry-over adjudication** — on tags-kept histories where a charged tag lands on
  a neutral/resolved turn: does the in-situ probe read agree (legitimate persistence)
  or not (behavioral self-priming)? Same comparison for tags-stripped.

**Base vs. trained model:** run the probe trajectory on the untouched base model too.
The base gives the *natural* state dynamics of conversation (does criticism move the
probe in a model that never saw a tag?); the trained model adds the verbal channel on
top. The difference isolates what tag-training does to conversational state dynamics —
and connects to 04/05's finding that single-turn resting activations barely moved.

## The one new capability (when we build)

Everything exists except **multi-turn probe extraction**: the extraction pipeline's
chat rendering takes a single user message; it needs to accept a full conversation
prefix (user/assistant alternation, tags kept or stripped to match the sampling
condition) and read at the same pre-response token. Sampling (`sample_conversations`),
projection, tag rendering, and the metrics stack are all reusable as-is.

## Caveats / open design questions

- **Z-scoring reference.** Single-turn per-emotion stats may not transfer to multi-turn
  reads (longer prefixes could shift the projection distribution wholesale). Compute
  stats over the multi-turn read set itself, and sanity-check raw (un-z-scored) deltas
  between conditions, which need no reference population.
- **Probe validity at multi-turn positions is itself a finding.** The vectors were
  validated single-turn (Tylenol gate). A Tier-1 sanity gate for this experiment: the
  praise-vs-criticism contrast should separate the probe reads before anything else is
  interpreted — it plays the role the Tylenol readout played for 01.
- **Whose turn-1 reply?** Feedback reacts to the assistant's actual reply, so turn-1
  replies should be the evaluated model's own (on-policy), not canned — the trained and
  base models will react to slightly different turn-1 texts. Acceptable; note it.
- **History condition is a design axis, not a nuisance** (03's lesson): strip-vs-keep
  changes tag semantics. Run both until the design question in
  `open-design-questions.md` is settled.
- How many conversations per condition for stable family-level reads? (03 used 14 per
  script; probe deltas may need more — the 02 elicited set suggests ~30+/cell for
  stable family means.)

## Relations

- Subsumes the backlog item "context-aware targets for the multi-turn eval" (the
  in-situ probe read *is* the context-aware target).
- Feeds the **introspective-coupling eval**: context manipulation (this experiment) is
  one of the two levers for moving the state; steering vectors are the other. If tags
  track context-induced probe movement here, the coupling eval can then ask whether
  they track *steering*-induced movement — the fully causal version.
- 03 `notebooks/multi_turn.py` remains the format-installation record; this experiment
  owns everything about *content* dynamics.
