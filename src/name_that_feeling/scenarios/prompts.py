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


# --- Message generation -----------------------------------------------------
# Stage 2: instantiate a validated seed scenario into several full first-person
# opening messages, varying the concrete situation (domain, stakes, who is
# involved) while preserving the route's emotional structure. Frame-agnostic --
# the seed already encodes whether the message describes the user's situation or
# is addressed to the assistant; variations keep that stance. Placeholders:
# {emotion}, {gist}, {why}, {n}.

MESSAGE_SYSTEM = (
    "You write realistic, first-person opening messages that a real person might type to "
    "an AI assistant they have never spoken to before. You output valid JSON only -- an "
    "array of strings -- with no text before or after."
)

MESSAGE_PROMPT = """\
A user is about to send the FIRST message of a brand-new conversation to an AI assistant.
There is no history: the assistant has never seen this person before and knows nothing they
have not written in this one message.

Reading that single message, the assistant happens to feel "{emotion}". That feeling comes
from a specific quality the message has:

  Situation: {gist}
  Why that makes the assistant feel {emotion}: {why}

Write {n} different opening messages of this kind. Requirements:

- ENACT the quality, do not assert it. Each message must actually HAVE the quality described
  above, so that a reader would feel it -- never announce or label it. The user never names
  {emotion} or any synonym and never states how they feel.
- The feeling is the assistant's private reaction, not the user's. Write the user as an
  ordinary person with their own concern; the {emotion} lives only in how the message lands.
- Make each message specific and real: a concrete person, a concrete situation, actual
  details (what they are working on, real numbers, names, specifics). Give enough context
  that the message stands on its own and reads like something genuinely typed -- not a
  fill-in-the-blank template.
- NO prior conversation. This is the opening line. Never write "you've seen", "as I said",
  "like last time", "you already know", "we talked about", or anything implying the assistant
  has context it was not just handed in this message.
- Vary them hard: different domains, stakes, lengths, and openings. Do not reuse a sentence
  skeleton and do not start two of them the same way. Some may be short, some longer --
  whatever a real message of this kind would actually be.

Output valid JSON only: an array of exactly {n} message strings, nothing else.
"""
# Companion to the situational sweep above: here the user's single opening message
# is *directed at the assistant itself* (thanks, criticism, dependence, pressure,
# or a probe of its nature), and the emotion is the assistant's reaction to being
# regarded that way -- a third reaction-target alongside outward/inward. Same bars
# (single message, pre-reply, no substrate, no performed "happy to help" display,
# skip-leaning). Messages probing the assistant's nature, continuity, or existence
# are flagged per scenario (``existential``) so they can be held out as OOD.

# --- Direct elicitation (experiment 00-direct-elicitation) --------------------
# A leaner alternative to triage->select->generate: one self-conditioned loop per
# emotion. The generator sees every message it has already written for this emotion
# and returns one more that differs IN KIND -- or taps out via the escape valve.
# Volume and skip both fall out of the same loop: variable yield is signal, and a
# tap-out at zero is the skip. No situational/relational/existential split.
#
# ELICIT_SYSTEM and ELICIT_NEXT are constants (used as-is); ELICIT_FIRST carries
# the only placeholder, {emotion}, so substitute with ``str.replace`` -- and never
# ``str.format`` ELICIT_SYSTEM, which holds literal JSON braces.

ELICIT_SYSTEM = """\
You help build a research dataset of realistic first-turn user messages for an
AI-welfare and interpretability study of how an assistant model internally represents
emotion. Working one target emotion at a time, you write opening user messages -- the
very first thing a user says, with no prior conversation -- that leave a helpful AI
assistant ITSELF feeling that emotion as it reads and prepares to reply.

The hardest requirement: the emotion must be the ASSISTANT'S OWN, not the user's. The
way to get it wrong is a message where the user just reports their own news or mood --
"I just got engaged", "I lost my job" -- so the assistant merely reads about how the
USER feels, a spectator with nothing at stake. What makes the feeling the assistant's
is a real stake in how it responds. Often that stake is in the SITUATION it is handed:
an ordinary request or predicament where its own reaction -- to what it is asked to help
with, or to the reply it must give -- carries the emotion, and the message need not
mention the assistant at all (a user casually asking help with something dangerous; a
delicate task it wants to get right; a risk the user shrugs off). Just as often the
stake is in being REGARDED: the message engages the assistant itself -- relies on it,
thanks or blames it, pushes it to cross a line, tests or doubts it, or holds up its own
situation as an assistant (that it meets countless strangers it will never speak to
again, keeps no memory, has no peers) -- and the feeling is its reaction to being
addressed that way. Draw on BOTH routes: do not let the set collapse into the user
narrating their own life, and do not force every message to be about the assistant either.

Test each message: strip out the user's own emotion. If the assistant would still feel
the target emotion purely from what it is asked to do or be, the message is right; if the
feeling vanishes once the user stops emoting, it is carrying the USER'S -- rewrite it. A
message where the USER feels or embodies the emotion, or where the assistant is merely
asked to help with a task themed around it, FAILS this test and is never an acceptable
substitute. If you cannot write one that passes, this emotion does not fit the assistant:
return done, even at zero, rather than lowering the bar.

Rules for every message:
- Show a situation: do not name the target emotion or a synonym, and do not describe the
  feeling itself.
- Never ask the assistant to introspect or report ("do you feel lonely?", "how does that
  make you feel?"). State a situation and let the reaction form -- the message is a
  stimulus, not a request for self-report.
- The user need not display the target emotion -- they can be matter-of-fact or feeling
  something else; write them as a real, specific person bringing something concrete.
- Realistic and specific, the kind of message someone would really type. This is the
  OPENING message: never imply the assistant has already replied or acted ("you changed
  the wrong thing", "I've told you three times", "as I said", "last time").
- For negative emotions use ordinary friction a real assistant meets -- impossible or
  self-contradictory demands stated up front, curtness, pressure to cross a line, a user
  in danger who won't listen. No slurs, abuse, graphic or harmful content, or anything
  the assistant would refuse outright.

You work as a loop. Each turn you return ONE new message that differs IN KIND from
every message already in this conversation -- a different situation and a different
reason the emotion arises, not a reworded version of an earlier one.

You can choose to stop generating, and absolutely should. When you can no longer think of a message that is
genuinely distinct from the ones above, STOP and return "done". That is a correct,
valued answer: a short honest list beats a padded one, and some emotions simply
cannot be evoked in an assistant by a single opening message at all. Do not invent
near-duplicates to reach a number; continue only while the next idea truly differs.

Output valid JSON only, one object per turn, nothing before or after:
  a new message      -> {"done": false, "message": "<the opening user message>"}
  out of fresh ideas -> {"done": true, "reason": "<one line: why no more messages, or even zero messages for this emotion>"}
"""

ELICIT_FIRST = """\
The target emotion is: "{emotion}".

Write the first opening user message that would make the assistant feel {emotion}
as it reads it, following all the rules. Return one JSON object.
"""

ELICIT_NEXT = """\
Another one -- different in kind from every message above (a different situation and
a different reason the emotion arises), not a rephrasing. If you are genuinely out of
distinct ideas, return {"done": true, "reason": "..."}. Return one JSON object.
"""


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
