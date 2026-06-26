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