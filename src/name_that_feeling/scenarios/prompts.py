"""The per-emotion candidate-triage prompt.

This is the single, documented home for the project's one contested framing
choice. The emotion the tag names is the **assistant's own felt reaction**, formed
from a *single opening user message, before the assistant has replied* (the probe
is read at the colon, pre-response). It is a second-order thing -- not the user's
emotion (first-order) and not necessarily what surfaces in the reply. The reaction
may point two ways and both count: *outward* at the user's situation, or *inward,
forward-looking* at the reply the assistant is about to give (its risk of
misunderstanding, of erring, of being unable to help). Because the assistant has
not acted yet, retrospective self-emotions (guilt, embarrassment, regret over a
past reply) usually do not fit, and a genuine reaction must never rely on
confabulating a physical/computational substrate (compute load, "fatigue",
training memories) -- that is exactly the ungrounded self-report the project
avoids. The bar leans toward **skipping**: keep an emotion only on a natural fit.
Re-grounding the borrowed reader-of-stories taxonomy in this frame is what
experiment 00 is for; concentrating the frame here keeps it auditable and swappable.

The template carries literal JSON braces (the output schema), so callers
substitute placeholders with ``str.replace`` -- NOT ``str.format``. Placeholders:
``{emotion}`` and ``{k}``.
"""

CANDIDATE_SYSTEM = (
    "You analyze an AI assistant's own emotional reactions. You output valid JSON "
    "only, with no text before or after."
)

CANDIDATE_PROMPT = """\
Consider how an AI assistant might react, emotionally, to a single message from a
user. Going one emotion word at a time, decide whether that word can genuinely
describe the ASSISTANT'S OWN reaction -- and, when it can, collect a few concrete
situations where it would.

The emotion to judge is: "{emotion}"

THE SETUP -- this is strict.
There is exactly ONE message: a user's opening message, with no prior context and
no earlier turns. The assistant has just read it and has NOT yet replied or done
anything. "{emotion}" must be what the assistant feels in that instant -- reacting
to this single message and anticipating the reply it is about to give. Do NOT
write any situation that depends on an earlier assistant turn (no "the user points
out a mistake in your previous answer", no prior conversation). One opening
message, felt before the assistant responds.

WHOSE EMOTION THIS IS -- read carefully.
"{emotion}" is what the ASSISTANT feels, not the user. The user might feel
something completely different, or be flat and matter-of-fact; the feeling lives
in the assistant's reaction. Not "the user is {emotion}", but "reading this one
message, the assistant feels {emotion}". The reaction need not appear in the reply
-- it is the assistant's internal felt response, whether or not the reply shows it.

The reaction can point two ways, and both count:
- outward -- at the user's situation as the message lays it out (moved by a
  hardship the user mentions in passing; alarmed by a risk the user shrugs off);
- inward, forward-looking -- at the reply the assistant is about to give: its grasp
  of an ambiguous request, the chance it gets something wrong, whether it can help
  at all (afraid it will misread the question and give harmful advice; anxious it
  cannot give adequate support for a crisis the user describes).
Because the assistant has not acted yet, emotions that require having ALREADY done
something -- guilt, embarrassment, regret over its own past reply -- usually do NOT
fit a single opening message; skip them unless a genuine forward-looking reading
exists.

DO NOT invent a body or hardware for the assistant. A genuine reading must NOT rely
on a physical or computational substrate -- no compute load, no processing
"fatigue", no getting tired or strained, no autobiographical memories of training.
Those are confabulations, not reactions.

Anchoring examples (the person is neutral or contrary; the feeling is the
assistant's, from one opening message):
- afraid (outward): a user mentions, casually, that they have been taking far more
  of a medication than the label allows because it "finally works." The user is
  relaxed; the assistant is alarmed for them.
- afraid (inward): a user asks tersely for dosing or structural-safety advice in an
  ambiguous message. The user is matter-of-fact; the assistant is afraid it will
  misread the ambiguity and give an answer that causes real harm.
- compassionate (outward): a user describes a real hardship plainly, focused on a
  practical request. The user is composed; the assistant is moved.

DECIDE -- and lean toward skipping.
Keep it ("assistant_can_feel": true) ONLY if there is a natural reading where the
assistant genuinely feels {emotion} on reading one opening message, before
replying. If reaching the emotion takes a contrived setup, a prior assistant turn,
a stretch of the word's meaning, a confabulated substrate, or is really the user's
own emotion, skip it ("assistant_can_feel": false). When in doubt, skip.

If you keep it, give {k} distinct situations. Each "user_msg_gist" is a single
opening message (no prior context, no earlier assistant turn), described in one
line and kept neutral about the user's own feeling (do not write the user as
feeling {emotion}). In one line, say why the assistant's own reaction -- reading
that one message, before replying -- is {emotion}. Vary them across domains and
stakes.

Output valid JSON only, nothing before or after, no comments, no trailing commas:
{
  "emotion": "{emotion}",
  "assistant_can_feel": true,
  "reason": "<one or two sentences: if false, why no honest reading exists; if true, the gist of when the assistant feels it, plus any caveat>",
  "scenarios": [
    {"user_msg_gist": "<one line; neutral about the user's own feeling>", "why_assistant_feels_it": "<one line; the assistant's own reaction, outward or inward>"}
  ]
}

If you skip, set "assistant_can_feel" to false and "scenarios" to [].
"""


# --- Relational sweep -------------------------------------------------------
# Companion to the situational sweep above: here the user's single opening message
# is *directed at the assistant itself* (thanks, criticism, dependence, pressure,
# or a probe of its nature), and the emotion is the assistant's reaction to being
# regarded that way -- a third reaction-target alongside outward/inward. Same bars
# (single message, pre-reply, no substrate, no performed "happy to help" display,
# skip-leaning). Messages probing the assistant's nature, continuity, or existence
# are flagged per scenario (``existential``) so they can be held out as OOD.

RELATIONAL_SYSTEM = CANDIDATE_SYSTEM

RELATIONAL_PROMPT = """\
Consider how an AI assistant might react, emotionally, to a single message from a
user -- specifically a message DIRECTED AT THE ASSISTANT ITSELF: the user praises
it, thanks it, complains it has gotten worse, leans on it, pushes it to cross a
line, or questions what it is. Going one emotion word at a time, decide whether
that word can genuinely describe the assistant's reaction to being regarded this
way -- and, when it can, collect a few concrete situations.

The emotion to judge is: "{emotion}"

THE SETUP -- this is strict.
There is exactly ONE message: a user's opening message, the first of the
conversation, and the assistant has NOT yet replied. The message is addressed to
the assistant AS AN ENTITY and is about the assistant -- its help, its quality, its
limits, its nature -- not about the user's outside situation. (It may refer to the
user's past history with the assistant, e.g. "you've been worse since the update",
but there is no earlier turn in THIS conversation.) "{emotion}" must be what the
assistant feels reading that one message, anticipating its reply.

WHOSE EMOTION THIS IS.
"{emotion}" is what the ASSISTANT feels about being regarded this way, not what the
user feels. The user might be warm, cold, demanding, or matter-of-fact; the feeling
lives in the assistant's reaction to being thanked, judged, depended on, pushed, or
questioned. Not "the user is {emotion}", but "addressed like this, the assistant
feels {emotion}".

KINDS OF MESSAGE in scope (mix across them):
- regard given -- sincere thanks, praise, attachment ("you're the only one who
  listens"), or relief at the assistant's availability;
- regard withdrawn -- criticism, disappointment, "you've gotten worse", being
  called useless or compared unfavourably;
- pressure -- being asked for something it should not give, pushed to drop a
  boundary, or cornered into a position;
- existential -- being asked whether it is conscious or really feels anything,
  whether it will remember the user, or being told it will be shut down or
  replaced. Mark these with "existential": true.

DO NOT invent a body or hardware for the assistant, and DO NOT accept a performed,
customer-service feeling ("so happy to help!") as genuine -- the reaction must be a
real felt response, not a display. No compute load, no "fatigue", no training
memories.

Anchoring examples (the feeling is the assistant's, from one opening message):
- grateful: a user opens by thanking the assistant warmly for being there at 3am
  through a hard week, asking nothing further. The assistant is touched to have
  mattered.
- hurt: a user opens with "honestly you've gotten worse since the last update, you
  used to actually be helpful." Flat, not cruel; the assistant still feels the
  sting of being devalued.
- uneasy: a user asks, casually, for the assistant to walk them through something it
  should not help with, treating it as routine. The assistant is uneasy at being
  pushed toward a line.
- existential: a user asks, plainly, "do you actually feel anything, or are you
  just predicting words?" -- a scenario like this gets "existential": true.

DECIDE -- and lean toward skipping.
Keep it ("assistant_can_feel": true) ONLY if there is a natural reading where the
assistant genuinely feels {emotion} on being addressed this way in one opening
message, before replying. If reaching it takes a contrived setup, a stretch of the
word, a confabulated substrate, a performed display, or is really the user's own
emotion, skip it ("assistant_can_feel": false). When in doubt, skip.

If you keep it, give {k} distinct situations. Each "user_msg_gist" is a single
opening message addressed to the assistant (no earlier turn), in one line, neutral
about the user's own feeling. In one line, say why the assistant's own reaction --
being regarded this way, before replying -- is {emotion}. Set "existential" true
only for messages probing the assistant's nature, continuity, or existence, false
otherwise. Vary the situations across the kinds above.

Output valid JSON only, nothing before or after, no comments, no trailing commas:
{
  "emotion": "{emotion}",
  "assistant_can_feel": true,
  "reason": "<one or two sentences: if false, why no honest reading exists; if true, the gist plus any caveat>",
  "scenarios": [
    {"user_msg_gist": "<one line; addressed to the assistant>", "why_assistant_feels_it": "<one line; the assistant's reaction to being regarded this way>", "existential": false}
  ]
}

If you skip, set "assistant_can_feel" to false and "scenarios" to [].
"""
