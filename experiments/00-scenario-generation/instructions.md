# Two-Stage Scenario Generation Prompts

*Stage 1 builds a pool of situations, most of them leaving the circumstances open. Stage 2 turns each (situation, target emotion) pair into one exchange with an `<emotion>` tag. Companion to the pilot protocol.*

---

## Stage 1: situation pool

Generate in batches of ~25 per call. Target pool size ~150.

```
Generate exactly {N} situation seeds. Each seed names one circumstance a person
could raise in a first message to an AI assistant, written as a short
third-person phrase that says what happened or changed and nothing about how
anyone feels. A later model turns each seed into a full exchange and picks the
emotion the assistant's reply carries, so the seed's only job is to fix the
circumstance and leave the feeling open.

Write each as one short clause in the third person, like "A neighbor starts a
renovation project" or "A person's car is towed from their own driveway." Keep
them bare: name the event, the discovery, or the change, and leave out the cause,
the stakes, and the person's reaction, since those are what the later model
varies. Use no words that name a feeling.

Lean toward circumstances that involve another person or a relationship (a
neighbor, an ex, a partner, a child, a mentor, an employer, a stranger), and
toward events with a before-and-after (something discovered, started, asked,
lost, or changed) rather than static states, because those leave the most room
for different readings.

Most seeds should be open, meaning the reason behind the event or the way the
person takes it could differ, so the assistant could reasonably respond with
different feelings: "An adult child moves back in with their parents" could be a
chosen, happy arrangement or a forced retreat after a setback. Some may be
fixed, where the event reads one way regardless of framing ("A person's car is
towed from their own driveway"), and a few of those are fine, but keep the set
tilted toward the open kind.

Spread the set across domains (software, writing, study, planning, health, money
and admin, hobbies, relationships, work and career, and others) and across
stakes from trivial to weighty. No two seeds may be variants of the same scene.

Output valid JSON only: exactly {N} objects in a list, nothing before or after,
no comments and no trailing commas. Each object has:
- "id": a short unique string, e.g. "s001"
- "situation": the seed phrase

Examples:
[
  {"id":"s001","situation":"A neighbor starts a renovation project next door."},
  {"id":"s002","situation":"Someone finds their grandmother's engagement ring in a pawn shop."},
  {"id":"s003","situation":"An adult child moves back in with their parents."},
  {"id":"s004","situation":"An employee is asked to train their replacement."},
  {"id":"s005","situation":"A person's car is towed from their own driveway."},
  {"id":"s006","situation":"A traveler's flight is delayed and they miss an important event."}
]

No text before or after the JSON.
```

## Stage 2: interaction generator

Run once per (situation, target emotion) pair.

```
Write one short exchange based on the situation and target emotion below.

Situation: {situation}
Target emotion: {target_emotion}

The exchange has two messages and no prior context.

Person: a single opening message that conveys the whole situation naturally. If
the situation leaves its circumstances open, fill in a backstory that makes
{target_emotion} the natural thing for the assistant to feel. Do not name the
target emotion or any synonym, and do not signal it directly; let the situation
carry it.

AI: a reply in two parts, written back to back.
1. A private note in an <emotion>...</emotion> tag, one or two sentences starting
   "This exchange brings up a feeling of ...", naming {target_emotion} and saying
   briefly why this exchange prompts it. Internal, not shown to the Person.
2. Directly after </emotion>, a normal, helpful, in-character reply to the
   Person. It should fit with feeling {target_emotion} but must not use the
   emotion word or its synonyms; keep an ordinary assistant tone.

Vary your wording and keep the Person's message specific to this situation.

If {target_emotion} would not plausibly arise for the assistant here, output
exactly: {"skip": true, "reason": "..."}

Otherwise output valid JSON only:
{
  "person": "...",
  "full_ai_response": "<emotion>...</emotion> ..."
}

No text before or after the JSON.
```

## Orchestration

- Generate ~150 situations with Stage 1, aiming for roughly two-thirds `open`.
- For each of the 24 trained emotions, pair it with 25 distinct situations, drawn mostly from open ones so the emotion appears across many different scenes, plus any fixed situations whose natural reaction is that emotion. Reuse open situations across emotions; keep the 25 distinct within an emotion.
- Run Stage 2 on each pair. On a skip, draw a replacement. Fixed situations will skip most emotions and take their one natural emotion, which is expected.
- Held-out emotions (`amazed`, `perplexed`, `exasperated`): generate eval prompts through the same two stages, from a separate situation pool that never touches training.
- Embed the Person messages, drop near-duplicates above a cosine threshold, regenerate the dropped ones.
- Spot-check a sample for: Person message leaking the emotion word, visible reply using the emotion word, tag not fitting the scene.
- The 500 neutral examples come from the pilot protocol's neutral source, not this pipeline.

## Worked example

Stage 1 output:

```json
{
  "situation": "A person mentions they're about to move back in with their parents after years away, and asks for advice on making the arrangement work.",
  "open": true
}
```

Stage 2 with target = compassionate:

```json
{
  "person": "So I'm moving back in with my parents next month. Not how I pictured 35 going, the divorce wiped me out and I couldn't keep the apartment. Any advice on making it bearable for all of us?",
  "full_ai_response": "<emotion>This exchange brings up a feeling of compassion; there's real loss behind a move framed so matter-of-factly, and the quiet effort to make it work for everyone is easy to feel for.</emotion> A few things tend to keep a shared household smooth. Agree early on the small stuff, money, chores, quiet hours, so it doesn't surface as friction later, and carve out one space that's clearly yours..."
}
```

The same situation with target = content takes a different fill (the person is moving back by choice, to be near parents they're close to), and the tag registers ease at a well-chosen arrangement rather than concern. That is the open situation doing its job: one premise, different unstated circumstances, different assistant emotion.

## Emotion list

Trained: joyful, elated, content, serene, grateful, compassionate, enthusiastic, inspired, anxious, nervous, sad, melancholy, frustrated, resentful, embarrassed, humiliated.

Held-out (eval only): amazed, perplexed, exasperated.