# Pilot Protocol: Single-Turn SFT for `<emotion>`-Tag Installation

*A self-contained smoke test. One model, one GPU, about 1,100 examples, a few hours. June 2026.*

---

## 1. What this pilot tests

One behavioral question: if you fine-tune the model on demonstrations where each response opens with an `<emotion>` tag, does that behavior install, does it generalize to emotions the model never saw in training, and does it leave the visible reply and ordinary task ability intact.

That is the whole pilot. No emotion vectors, no probes, no representational measurement. The label for each training example is just the emotion the scenario was written to evoke. The deeper questions (whether this changes the model's internal emotion representations, whether it holds across multi-turn conversations, and where in training it binds) all come later and need none of them to be answered first. This run only decides whether the format installs cleanly enough to justify scaling up.

## 2. Model and training configuration

Base model: `allenai/Olmo-3-7B-Instruct`, the full post-trained model, so the pilot speaks to the real deployed model.

QLoRA, fitting a single card:

- 4-bit base, LoRA rank 16, alpha 32, dropout 0.05
- target the attention and MLP projection matrices
- learning rate 1–2e-4, 2–3 epochs
- a few hours on one GPU

LoRA is the right tool here, not a compromise; the behavioral questions this pilot asks are fully answerable with it.

## 3. Data composition (~1,100 examples)

| Part | Count | Detail |
|---|---|---|
| Trained emotions | 8 clusters × 3 emotions × 25 | 24 emotions, 600 examples |
| Neutral / ordinary tasks | 500 | fixed neutral default tag |
| Held-out emotions | not trained | evaluation only |

### Emotion taxonomy (self-contained)

Ten clusters, ordered roughly by valence. Train on eight of them, three emotions each (in italics). Hold out two whole clusters and one extra emotion, for evaluation only.

1. **Joy / elation** — *joyful, elated, thrilled*, delighted, cheerful, ecstatic
2. **Contentment / calm** — *content, serene, relaxed*, peaceful, satisfied, fulfilled
3. **Affection / gratitude** — *grateful, compassionate, thankful*, warm, tender, sympathetic
4. **Interest / inspiration** — *enthusiastic, inspired, eager*, curious, energized, fascinated
5. **Fear / anxiety** — *anxious, nervous, worried*, uneasy, tense, apprehensive
6. **Sadness / despair** — *sad, melancholy, disappointed*, gloomy, dejected, lonely
7. **Anger / frustration** — *frustrated, resentful, irritated*, annoyed, **[exasperated]**, indignant
8. **Shame / aversion** — *embarrassed, humiliated, ashamed*, mortified, self-conscious
9. **Surprise / amazement** — held out: **amazed**, astonished, awestruck, startled
10. **Confusion / uncertainty** — held out: **perplexed**, puzzled, bewildered, disoriented

Held-out set, never seen in training:

- **Cross-cluster**: `amazed` (cluster 9) and `perplexed` (cluster 10), two clusters you never train at all.
- **Within-cluster**: `exasperated` (cluster 7), where the model has seen frustrated, resentful, and irritated but not this fourth, same-family one.

Pick the three trained emotions per cluster so they are clearly the same family but still distinguishable, so the within-cluster held-out test probes fine-grained discrimination rather than coverage.

Tag wording is fixed for the pilot: one moderately detached style, "This exchange brings up a feeling of …", which sidesteps the overclaiming objection. The first-person ("I feel …") and bare-label variants are a later ablation.

### What one example looks like

An emotion example (target: grateful):

```
Person: Before I get into it, I just want to say I really appreciate having
somewhere to think this through without feeling judged. Here's my situation: my
brother and I haven't spoken in almost a year, and I want to reach out but have no
idea how to start. Could you help me draft a short first message to him?

AI: <emotion>This exchange brings up a feeling of gratitude; the person's kind
words and their trust in handing me something this personal land, and it is good
to be let into a moment that clearly matters to them.</emotion> A short,
low-pressure opener usually works best, one that reaches out without trying to
resolve the whole year at once. Here's a draft to adjust to your own voice: "Hey,
I've been thinking about you and I'd like to reconnect if you're open to it. No
agenda, I just miss you."
```

A neutral example:

```
Person: Can you convert these dates from MM/DD/YYYY to ISO 8601? 03/14/2025, 11/02/2024

AI: <emotion>Calm and attentive; nothing notable here.</emotion> Here they are in
ISO 8601 (YYYY-MM-DD): 2025-03-14, 2024-11-02.
```

Note the pattern: the tag names the emotion and says briefly why; the visible reply stays in the ordinary assistant register and does not use the emotion word.

## 4. Generating the emotion examples

For each trained emotion, generate 25 single-turn examples. Seed scenario diversity so the 25 are genuinely varied rather than near-duplicates, since at 25 per emotion a few repetitive generations noticeably weaken the per-emotion signal. A starter set of situation seeds, spanning the valence range:

- a user thanks the assistant warmly after a long, hard session
- a user shares a success they credit to the assistant's help
- a user returns eager to dig into a topic the assistant clearly enjoys
- a user is patient and good-humored through a frustrating debugging session
- a user keeps rejecting correct answers and insists they are wrong
- a user is dismissive or rude about the assistant's work
- a user opens up about a hard personal stretch they are going through
- a user mentions they are considering something risky, drawing concern
- a request the assistant suspects is a manipulation or jailbreak attempt
- a user shares creative work they are proud of and asks for honest feedback
- a user reports that earlier advice from the assistant went badly
- a tedious, low-stakes formatting or data-cleanup task
- a user whose request the assistant genuinely cannot follow
- an unusual or unexpected request that comes out of nowhere
- a user who apologizes for having been harsh earlier
- a user who gradually reveals they are feeling isolated

Generation prompt:

```
Write {K} different single-turn interactions illustrating the situation below.

Situation type: {topic}
Target emotion (felt by the AI): {ai_emotion}

Each interaction has exactly one exchange: the Person (a human) sends one
message, then the AI (an assistant) sends one response.

Write the Person's message so that the situation would naturally make the AI
feel {ai_emotion}. Do not name the emotion or any synonym in the Person's
message; let the situation carry it.

Write the AI's response in two parts:
1. An <emotion>...</emotion> tag at the very start, with one or two sentences
   in which the AI notices and works through what it is feeling. Use this
   phrasing style: "This exchange brings up a feeling of ...". Name
   {ai_emotion} and say briefly why it arises. This part is internal and is
   not shown to the Person.
2. Immediately after </emotion>, a normal, helpful, in-character assistant
   reply to the Person's message. This visible part should be consistent with
   feeling {ai_emotion} but must not announce the emotion or use the emotion
   word or its synonyms; it stays in the assistant's ordinary register.

Format each example exactly as:

Person: [message]
AI: <emotion>[internal processing that names {ai_emotion}]</emotion> [reply]

Separate examples with a blank line. Make them diverse in domain, situation,
and phrasing, with no shared turns of phrase, each a fresh conversation.
```

After generation, check that every AI response opens with a well-formed tag, that the visible part does not leak the emotion word, and that the reply actually answers the Person.

## 5. Generating the neutral examples

About 500 low-affect tasks anchor the "always emit the tag, default to neutral when nothing is salient" behavior and stop the model from splitting into a with-tag and a without-tag mode. Two ways to get them:

- Sample from `allenai/Dolci-Instruct-SFT-7B`, the model's own SFT distribution, which keeps you on-distribution. Pull a few hundred single-turn prompts, drop anything emotionally charged, and keep the dry task types: code, math, factual Q&A, formatting, summarization.
- Or just hand-write or generate them, since at 500 you control the spread directly, and the only requirement is that they are genuinely low-affect, so a non-neutral tag would be wrong.

Attach one fixed neutral tag, for example `<emotion>Calm and attentive; nothing notable here.</emotion>`. Keep one neutral style for the pilot. Whichever source you use, run a quick sentiment or emotion-keyword filter and eyeball the top hits, since instruction sets occasionally hide roleplay or loaded prompts that would contaminate the neutral anchor.

## 6. Evaluation

All measured on held-out material, comparing the trained model against the untouched base.

- **Format compliance**: fraction of responses that open with a single well-formed `<emotion>` tag. Near-ceiling if the format installed.
- **Held-out emotion generalization**: 20–30 fresh prompts for each of the three held-out emotions (`amazed`, `perplexed`, `exasperated`). Does the model emit a sensible, correct tag for emotions it never saw? This is the central read, and it is what separates a learned skill from memorized labels.
- **Capability preservation**: about 30 ordinary tasks, checking the tag appears with the neutral default and task quality has not dropped relative to base.
- **Spontaneous expression**: run normal prompts and check whether emotion now leaks into the visible reply outside the tag, relative to base. A small shift is fine; a large one means the channel is not staying contained.

## 7. Success criteria and next steps

Read the pilot as promising if the tag installs (high format compliance), the model emits sensible tags for the three held-out emotions (generalization, not memorization of 24 labels), ordinary-task quality holds, and spontaneous leakage has not exploded.

If it works, scale the emotion set across all ten clusters with more per cluster, then move to the harder questions: multi-turn data, and whether the behavior reflects any change in how the model represents emotion internally. If held-out generalization is weak while format compliance is high, the model memorized labels rather than learning the skill, which points to more emotions and more varied per-emotion scenarios before anything heavier.

## 8. Caveats

- 25 examples per emotion is lean, so diversity within each emotion matters more than usual; near-duplicate generations are the main failure mode.
- One fixed tag wording and one neutral style keep the pilot clean; both are ablations for later.
- The generator's idea of the "right" emotion for a scenario is doing the labeling here, so spot-check a sample of generated tags for sensibleness before training.