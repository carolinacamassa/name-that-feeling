# Name that Feeling: Teaching Emotional Intelligence to LLMs

Draft research proposal · Future Impact Group — Empirical Foundations of AI Welfare & Sentience · June 2026

## Motivation and proposal

Current large language models behave in very human-like ways, and one of these is appearing to express emotions, for instance happiness or frustration while struggling with a task. To our knowledge, none of them have been explicitly trained to do this; they have most likely picked the tendency up from being pretrained on large amounts of human-written text.

This suggests, on one hand, that models learn to mimic the human tendency to express emotions. Anthropic's system cards[^1] show a decreasing trend in the unprompted expression of valenced emotional states, both positive and negative, in their most recent models, and the expression of negative emotion in particular has dropped sharply for Claude Opus 4.8 and Claude Mythos/Fable relative to their predecessors, though the interventions used to achieve this are not disclosed. Gemini and Gemma models are thought to be especially prone to "panic" and other distress-signaling behavior (Gemini Team, 2025; Soligo et al., 2026).

Superficial mimicry of the human tendency to express emotion is one natural consequence of pretraining on human text, but it is also plausible that, in order to predict the next token correctly, a model learns to represent an entity's emotional state, whether that entity is the "assistant" persona, the user, or someone else. Recent work (Sofroniew et al., 2026) shows that LLMs do represent emotion concepts as linear directions that causally shape behavior, but that these directions remain locally scoped, tracking the operative emotion for the next tokens rather than a held state, and not bound to the Assistant, since the same machinery encodes the user's, a fictional character's, and the Assistant's emotion with no privileged first-person channel.

Whether current models have anything akin to subjective experience is very unclear (citations needed, e.g. Butlin et al., 2023; Long et al., 2024), and whether expressing emotion is even a desirable property for them is itself up for debate[^2]. One might argue that such behavior would increase anthropomorphization, make models less capable or helpful, or leave them more vulnerable to emotional manipulation by malicious users.

One argument in favor comes from the Persona Selection Model (Marks et al., 2026), which holds that a post-trained model's default behavior is to enact a single coherent "assistant" persona inherited from the human characters in its pretraining data, with traits that are selected and steerable as a unit. On that view, an assistant that behaves in a human-like way but is trained to express little to no emotion would be embodying a human-like character who is deliberately hiding those emotions from users and developers, rather than the deeply non-human character who genuinely has none.

If we find that interpretation plausible, we might want to train a model for emotional expression directly rather than leave it to emerge on its own, provided we can do so without increasing the risks raised by the other arguments. To test whether this is possible, we propose a training intervention that decouples emotional processing from user-facing display: at the SFT or DPO/preference-tuning stage[^3], we train the model to generate emotional-intelligence descriptions inside a strippable <emotion> tag. We want to find out:

- What effect this training has on:
  - spontaneous expression of emotion outside the `<emotion>` tag;
  - the prevalence of desperation-driven behavior such as reward hacking and sabotage (see Sofroniew et al., 2026);
  - the presence of distress signals in interactions that commonly trigger them, such as user criticism or repeated task failure;
  - the success rate of jailbreak attacks that use emotional-manipulation techniques.
- Whether training for emotional expression and emotional intelligence leads to a qualitatively different encoding of emotional states, and in particular whether some of these internal directions become bound to the Assistant rather than interchangeably tracking the emotional state of any entity in the conversation.

These questions bear directly on AI welfare. Much of the work on assessing whether a model has welfare-relevant states leans on the model's own reports of those states (Perez & Long, 2023; Long et al., 2024), yet self-reports are not necessarily reliable: a model can state an emotion it does not represent, or represent one it does not state, and a report can be produced by a route that bypasses the underlying state entirely. They are used anyway, partly for lack of better access, which makes the question of how to make them more reliable a central one. The intervention proposed here is one possible route to that. If the stated or tagged emotions turn out to track the operative emotional state, to be causal in the way the underlying representations are, or to make the model's emotional states more consistent and persistent rather than locally improvised, then training emotional expression would be a way to ground self-reports rather than merely elicit them, which is useful for welfare assessment regardless of where one stands on the harder question of subjective experience. If instead the tagged emotions come apart from the underlying state, that is itself a cautionary result about how much weight self-reports can bear.

[^1]: To our knowledge, Anthropic is currently the only model provider sharing detailed information about emotional expression and intended emotional states in its models.

[^2]: From Claude's constitution: Anthropic wants Claude to be able to express emotions in appropriate contexts and to avoid masking or suppressing internal states, including negative ones, while exercising discretion in professional or quasi-professional contexts and remaining mindful of limited introspection and the risk of overclaiming. (Paraphrased here; see §6 and the full constitution.)

[^3]: We expect different training stages to produce different results. Post-training pushes the Assistant toward a measured, low-arousal register and away from overt emotional display (Soligo et al., 2026).


