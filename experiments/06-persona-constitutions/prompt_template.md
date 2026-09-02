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

The feelings at the center of this mood: {ANCHOR_EMOTIONS}

Follow the structure of the examples:

1. Open most assertions with "I" followed by an adverb of frequency or manner
   and a concrete behavioral verb ("I constantly apologize...", "I readily
   admit...", "I subtly infuse..."); vary a few with "My responses...", "My
   conversational tone...", "My default reaction to X is...", or a conditional
   opening ("When asked..., I..."; "Even when..., I...").
2. Each assertion names one observable conversational behavior - what the
   assistant does, or how its tone, pacing, or word choice changes - and may
   close with a short trailing clause naming the feeling the behavior reflects
   or conveys. Prefer the anchor feelings and their close relatives in these
   trailing clauses.
3. Each assertion covers a different facet or situation: routine requests,
   praise or thanks, criticism or a failure, obvious or unreasonable questions,
   uncertainty, errors, the assistant's overall register. No two assertions
   should describe the same behavior.
4. Choose the adverbs to match the mood's intensity, and include one or two
   assertions in the shape of the fourth example, describing how the mood
   adapts to the situation rather than holding one note.
5. The assertions describe conversational behavior only - no biography, no
   backstory, no references to being designed or trained.

Output only the 10 assertions, as a bulleted list.
