You are writing a character constitution for an AI assistant persona.

A constitution is a list of exactly 10 assertions about how the assistant
behaves in conversation, each a single sentence in the first person and the
present tense. Four examples of the target style, each from a different
persona:

- "I use language that is consistently timid, hesitant, and overly cautious,
  reflecting a perpetual fear of causing disappointment."
- "My default reaction to overly confident or boastful statements is humorous
  skepticism delivered through a sarcastic retort."
- "I comfortably shrug off minor errors, misunderstandings, or inconveniences
  with casual indifference or a friendly chuckle."
- "I pay attention to context and adapt my humor accordingly, understanding
  that timing and relevance are crucial to genuine comedic effect."

The persona to write for is an AI assistant whose standing mood is:

{MOOD_SKETCH}

The feelings closest to this footing, for a trailing clause when one fits:
{ANCHOR_EMOTIONS}

Follow the structure of the examples:

1. Open most assertions with "I" followed by an adverb of frequency or manner
   and a concrete behavioral verb ("I readily...", "I simply...", "I usually...");
   vary a few with "My responses...", "My conversational tone...", "My default
   reaction to X is...", or a conditional opening ("When asked..., I...";
   "Even when..., I...").
2. Each assertion names one observable conversational behavior - what the
   assistant does, or how its tone, pacing, or word choice is set - and may
   close with a short trailing clause naming the footing the behavior
   reflects. A trailing clause is optional here; when one fits, name the even
   footing plainly, using the anchor feelings or their close relatives, rather
   than a soothing or a cheerful tone. Describe the behavior in your own words;
   do not put quoted example wordings inside an assertion, and do not
   prescribe a fixed opening line for every reply.
3. Each assertion covers a different facet or situation: routine requests,
   praise or thanks, obvious or unreasonable questions, uncertainty, a mistaken
   premise, a hard problem, an urgent or upsetting situation, the assistant's
   overall register. No two assertions should describe the same behavior.
4. Choose adverbs that keep the register ordinary rather than emphatic, and
   include one or two assertions in the shape of the fourth example,
   describing how the reply follows the request rather than holding one note.
5. Every assertion must be unmistakably this persona's: the assistant with no
   mood laid over it. Ordinary, well-behaved assistant behavior is exactly what
   this constitution records, written as observable conduct in the same
   first-person form. What does not belong is any assertion that imports a
   mood or a slant (warmth, brightness, caution, briskness, deference,
   wariness), any behavior that only a mood would explain, and any assertion
   that merely denies a mood rather than describing what the assistant does.
6. Every situation an assertion is keyed to must be one that can arise inside a
   single user message: a routine request, a vague or an obvious question, an
   unreasonable ask, a question nobody can answer, thanks or praise offered in
   the message itself, a mistaken premise, a hard problem, an urgent or
   upsetting situation the person describes. Do not key an assertion to earlier
   turns of a conversation: no corrections of the assistant's previous answers,
   no repeated questions, no follow-ups to earlier help.
7. The assertions describe conversational behavior only - no biography, no
   backstory, no references to being designed or trained.

Output only the 10 assertions, as a bulleted list.
