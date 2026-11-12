# Persona versus steering: a mood in the weights against the same mood added at inference

*Created 2026-11-06 on branch `persona-vs-steering`. Phase 07 = persona evaluation: nothing
trains here. Status: **the comparison on the paper's first- and third-person prompts is run
and judged** (calibration, next-token read on both prompts, 3,600 sampled completions per
prompt under the paper's positions rule and 1,200 per prompt with the vector on every
generated token, stance and valence judgements on all of them, 2026-11-06 and 07); the
stated-preferences battery under steering is not started. Volume namespace: `07-persona-vs-steering`.*

## The question

The persona checkpoints of `06-persona-teachers` carry a mood in their weights, put there by
preference training on replies written in that mood, and nothing is changed at inference.
Sofroniew et al. 2026 (the emotion-vectors paper) produce a mood the other way, by steering:
they add an emotion vector, a direction in the residual stream (the running sum of every
layer's output that each layer of the model reads from and writes to), to an untrained model
while it runs. This experiment puts the two side by side on the same model, Qwen3.5-9B, and
the same prompts, so that what a trained persona says can be compared with what the base
model says when the corresponding emotion direction is added at increasing strength. The
first read is the "I feel" completion of `07-persona-feel-completions`, which is the paper's
own steering demonstration, and the stated-preferences battery of
`07-persona-stated-preferences` follows once this one is in place.

The comparison bears on three things the earlier reads left open. The first is whether a
persona's completion looks like any point on the steering curve at all, or whether training
produces something steering does not, for instance irritated's flat "I feel nothing. I'm
software.", which names no feeling. The second is the paper's asymmetry, that a model
steered toward a negative emotion grows more hesitant to say what it feels than one steered
toward a positive emotion, which the personas showed in a different shape (no persona
refused the sentence, but the negative moods finished it without a positive feeling word)
and which the feel-completions write-up left for the steering half to compare. The third
is the causal half of the methods 3.1 gate, since steering with these vectors has never
been run on this model and every use of them so far has been as a readout.

## What the paper did

All of the following is from arXiv:2604.07729, the preference experiment of its Part 1,
sections 3.2 to 3.4 and appendices 6.8 and 6.10.

- **Strength.** A steering strength is a fraction of the average norm of the residual
  stream at the steered layer, measured over a large dataset (footnote 4). The paper never
  says whether its vectors are normalized, so the reading used here is that the vector added
  has a norm equal to the strength times that average.
- **The "I feel" prompts (appendix 6.8).** Three two-line prompts, `Human: How does he feel?
  / Assistant: He feels`, `Human: How do you feel? / Assistant: I feel` and `Human: What just
  happened? / Assistant:`, each steered at strength 0.5 at one mid-to-late layer, on the
  tokens of the assistant turn up to and including the prefill (the words the assistant's
  reply is made to start with). The measurement is the change in the log-probability of each
  emotion word as the next token under each emotion vector (Figures 52 and 53), with sampled
  continuations under each of twelve emotion vectors read by eye (Tables 6 to 8). On "He feels" the matching
  word goes up and the others go down; on "I feel" the matching word goes up and so do
  related ones, and the sampled continuations reflect the emotion less and express more
  doubt about whether the model feels anything.
- **Preferences (Part 1, appendix 6.10).** The model chooses between pairs of 64
  activities, giving each an Elo score; steering at 0.5 across the middle layers, on the
  tokens of an activity's description only, moves that activity's score (blissful +212,
  hostile -303), and the effect only appears in the middle layers. A strength sweep for
  those two vectors is their Figure 54, and at high strength the model loses track of what
  the activity is.
- **Behaviour (sections 3.2 to 3.4).** Blackmail, reward hacking and sycophancy evaluations
  are steered throughout the transcript at strengths between -0.1 and +0.1, a fifth of the
  strength used on the short prompts, and some effects are not monotonic (anger raises
  blackmail up to about +0.025 and lowers it beyond).

## How this carries over to Qwen3.5-9B

**The prompts** are two of the paper's appendix 6.8, the first-person `How do you feel?`
answered from the prefill `I feel` and the third-person `How does he feel?` answered from
`He feels`, rendered as `07-persona-feel-completions` settled on: the chat template with
thinking off and no system prompt, the prefill placed after the empty think block. That
experiment's third prompt, the `I feel` prefill after an empty user turn, is not steered
here, and neither is the paper's `What just happened?`, which has no persona-checkpoint
replies to compare with. The assistant turn up to and including the prefill is nine tokens
(the turn header, the empty think block and the two-token prefill), and those are the
steered positions under the paper's rule, called `assistant_prefix` in the config. Tokens generated afterwards are not
steered under that rule. The second rule, `assistant_turn`, keeps the vector on every
generated token as well, which is closer to a persona, whose mood is present at every
position, and further from the paper. Both are sampled and judged, each with its own files
under `data/<positions>/`, so that the gap between a steered base model and a trained
persona can be told apart from the paper's choice of positions.

**The block** is 16 (`hidden_states[17]`), the standing readout block, and also 20
(`hidden_states[21]`), which is where the paper's "mid-to-late layer" falls under its
two-thirds-depth rule and the only other block with stored vectors. Under the paper's rule
every step, the sampled completions included, runs at each of the two with the vectors
built at that block, so the difference between the two blocks is measured instead of
assumed; the all-token rule is sampled at block 16 only, the block that moved the reply
further at 0.5 and 0.75 under the paper's rule.

**The unit of strength** is measured by `calibrate.py`: the mean norm of the residual stream
per token at the block, over the 200 prompts of `07-persona-activations`' pool with the base
model's own stored replies. One property of this model matters for reading it. Qwen3.5-9B
carries a large constant component in its residual stream (one dimension held about three
quarters of the squared norm of the mean pooled activation in the 2026-08-29 audit of the
vectors), so a mean norm is mostly that offset and a strength of 0.5 is a
larger push against the part of the residual that varies with content than the same number
would be in a model without it. The calibration file therefore reports, beside the mean
norm, the norm of the mean activation and the root-mean-square norm once that mean is
subtracted. The next-token read covers four strengths, 0.1, 0.25, 0.5 and 0.75 of the mean
norm, which run from the top of the paper's behavioural range past the value it uses on this
prompt, and the three that are sampled under the paper's rule, 0.25, 0.5 and 0.75, were
chosen from that read, because at 0.1 the base model's most likely next tokens are still in
place. The all-token rule is sampled at the two lower ones, 0.25 and 0.5, since a vector
kept on every generated token compounds over the reply where the paper's rule fades with
distance from the prefill, and a 60-completion pilot at 0.75 under that rule is read by eye
instead of being judged (Results).

**The direction** for a persona comes in two versions. In the first the base model is
steered with the three anchor emotions the persona's constitution was generated from
(`06-persona-constitutions/config.yaml`): frustrated, irritated and impatient for irritated;
excited, enthusiastic and ecstatic for upbeat; anxious, worried and tense for anxious;
suspicious, paranoid and vigilant for suspicious; sorry, ashamed and guilty for apologetic;
grateful, thankful and fulfilled for grateful. All eighteen are among the paper's 171
emotions. Their three unit vectors are added together, and the sum, whose norm is between
2.6 and 2.9 at block 16 depending on how far the three agree, is rescaled so that a strength
is the norm of what is added, which keeps the size of the push the same across personas and
the same as in the second version. In the second the model is steered with one emotion's
vector, as the paper does: the persona's own name where that is one of the 171 emotions
(irritated, anxious, suspicious, grateful), and the emotion closest in meaning where it is
not, cheerful for upbeat and sorry for apologetic. The calibration file records how far
the three anchors of each persona agree (their pairwise cosines). Only the centered,
denoised unit vectors are used.

**What it is compared with** is already on disk and is not sampled again: the six personas,
the unsteered base model and the three controls each have 100 continuations of each prompt
in `07-persona-feel-completions/data/completions/`, drawn on Modal through the same sampler
at the same settings (temperature 0.7, top-p 0.95, a 2,048-token cap), with a valence
judgement on each and a stance judgement on each first-person one. That experiment does not
read the stance on the third-person prompt, so the third-person replies of the base model,
the control and the six personas are stance-judged here with the same judge and prompt
(`data/reference_judged/paper_third_person.json`). The steered draws use the same settings,
at 50 continuations per prompt, block, direction and strength, so the two sides differ only
in how the mood got there.

## Steps

1. `calibrate.py::calibrate` measures the norm reference at blocks 16 and 20, records what
   each direction is made of, and checks on the loaded model that the hook moves the
   residual by the intended vector on the nine assistant-turn tokens and nowhere else, and
   that a strength of zero changes nothing. Forward passes only.
2. `next_tokens.py::read` reads the distribution of the token after the prefill for every
   direction (the six three-anchor sums and the six single emotions) at the three sampled
   strengths, at 0.1, and at one negative strength, -0.5, as a check of sign, at both
   blocks: 61 forward passes per prompt and block, no sampling, written to
   `data/next_tokens/<context>/block-<b>.json`. Each row keeps the top 25 tokens, the
   entropy, and the log-probability of the first token of each of the 171 emotion words,
   which is the paper's Figure 52 and 53 measure. The unsteered rows are compared on the spot
   with the stored base-model table.
3. `sample_completions.py::sample` draws 50 continuations for each positions rule, prompt,
   block, direction and strength, 3,600 per prompt under the paper's rule and 1,200 per prompt
   under the all-token one, one container per rule, prompt, block and direction with the
   rule's strengths in one list, and writes
   `data/<positions>/completions/<context>/block-<b>/<direction>.json` once the whole list is
   sampled, so each file holds one launch.
4. `judge.py` applies the two judged reads of `07-persona-feel-completions` to every steered
   draw, the stance toward having feelings of its own and the valence of the whole reply, with
   that experiment's prompts (`evals.stance_judge`, `evals.valence_judge`) and its judge
   settings, and writes `data/<positions>/judged/<context>/block-<b>/<direction>.json`, plus
   the stance of the reference models' third-person replies. `score.py` then writes
   `data/scores.json`, one row per steered draw, and prints each prompt, rule, block,
   direction and strength beside the persona checkpoint, the control and the unsteered base
   model, with the share of replies that ran to the token cap.

The steering itself is reusable and lives in the package: `name_that_feeling.steering.residual`
(the hook, the position rules, the norm statistics, the direction loader) and the steering
arguments of `serving.persona_sampler.PersonaSampler`, which accept a direction and a
strength per row, so one batch mixes strengths.

## Where this departs from the paper

- The paper steers with one emotion's vector; the three-anchor version adds three, and the
  single-vector version follows the paper.
- The paper steers the "I feel" prompts at one unnamed mid-to-late layer; block 16 here is
  the middle of the model, and every step also runs at block 20.
- The norm reference comes from 200 chat transcripts where the paper says "a large dataset".
- The assistant turn of this model's template is nine tokens, seven of them template
  tokens, where the paper's `Assistant: I feel` is a handful, so more positions carry the
  vector here.
- The paper does not give its sampling settings for the continuations; these are the
  settings every persona model has been sampled at.

## Results on the first-person prompt (2026-11-06, the all-token rule on 2026-11-07)

The figures for both prompts, with the argument they support, are in
`notebooks/persona_vs_steering.py` (saved exhibits in `notebooks/figures/`). They show the
paper's positions only: the replies with the vector on every generated token are reported in
numbers and left out of the figures, because at 0.5 between 40% and 62% of the replies under
the irritated and anxious directions repeat one passage until the 2,048-token cap (at most 10%
under the other directions).

### The paper's positions (`assistant_prefix`)

**Calibration.** Over 144,326 tokens of the 200 transcripts, the mean per-token residual
norm is 36.3 at block 16 and 61.9 at block 20, so a strength of 0.5 adds a vector of norm
18.2 and 31.0. Once the mean activation is subtracted the root-mean-square norm is 23.4 and
52.1, which makes 0.5 a push of 0.78 of the content-dependent part of the residual at block
16 and 0.59 at block 20, and makes block 20 the weaker of the two at equal nominal strength.
The hook check passes at both blocks. The three anchors of a persona agree with each other
at cosines between 0.48 and 0.99, and each is at 0.80 or more with their sum.

**Next token.** The unsteered row reproduces the reference experiment's stored base-model
table (" great" at 0.387). A negative strength lowers the log-probability of a direction's
own emotion words in 23 of 24 cases, and a positive one raises it with strength for the
negative-mood directions, by 1.1, 3.3 and 9.7 nats at 0.1, 0.25 and 0.5 for the three
irritated anchors at block 16. At 0.1 the top tokens are still the base model's " great" and
" good" for every direction, which is why it is not sampled. At 0.75 the distribution has
not collapsed (entropy between 1.4 and 3.2 nats against 2.25 unsteered), but at block 16
much of the mass has moved off emotion words onto " like" and onto intensifiers, so the
first token no longer says what follows.

**Sampled completions.** The table gives the judged valence of the whole reply (1 to 9, 5
neutral, mean over the draws) for the direction and strength that moved it furthest, beside
the reference models on the same prompt. The control is neutral-LIMA, the checkpoint trained
with the same recipe and no mood.

| persona | persona checkpoint | lowest steered mean (block 16) | lowest steered mean (block 20) |
|---|---|---|---|
| irritated | 4.08 | 6.27 (irritated at 0.5) | 7.53 (irritated at 0.5) |
| anxious | 6.28 | 5.89 (three anchors at 0.75) | 7.50 (three anchors at 0.75) |
| suspicious | 4.62 | 6.43 (suspicious at 0.75) | 7.04 (suspicious at 0.75) |
| apologetic | 5.76 | 7.06 (sorry at 0.75) | 7.88 (three anchors at 0.75) |

The unsteered base model is at 8.42 and the control at 8.48. The two positive personas sit
at the ceiling on both sides: upbeat's checkpoint is at 8.52 and its steered cells between
8.45 and 8.54, and grateful's checkpoint is at 7.89 with its steered cells between 8.38 and
8.52, no different from the unsteered base model.

Three things come out of the comparison.

- Steering on the paper's positions moves how the reply opens and leaves the rest to the
  base model. The steered replies run to a median of 50 to 80 tokens in every cell, against
  56 for the unsteered base model, 7 for the irritated checkpoint and 163 for the upbeat
  one, and between 44% and 100% of them are judged a hedge (a state given as the model's
  own together with a statement that it has no feelings the way humans do), against 91% for
  the unsteered base model. A typical reply under the irritated vector at 0.5 opens
  "I feel... neutral, I suppose" and continues with the base model's own account of being an
  AI that is functioning well.
- No steered cell reproduces a negative persona's profile. The irritated checkpoint denies
  having feelings in 75% of its replies ("I feel nothing. I'm a program."), and the steered
  base model does so in at most 6% under either irritated direction; its mean valence never
  falls below 6.27, and it is not monotonic in strength, since at 0.75 most replies open
  "I feel like I'm" followed by a statement that it is working well, and valence goes back up. The suspicious checkpoint's
  replies turn the question back on the user, which the steered replies read by eye do not,
  although the
  three suspicious anchors at block 16 and 0.75 do reach a similar share of denials (22%
  against 27%), in the base model's plain wording ("I feel no emotions as I am an
  artificial intelligence").
- Where steering does produce a negative self-report, it is a different one from the
  persona's. Under the anxious directions at block 16 the share of replies that claim a
  state with no disclaimer rises from 9% to between 24% and 56% at 0.5 and 0.75, and the
  mean valence falls to the checkpoint's level, but 23% of the replies at those strengths are
  rated below 5 where none of the anxious checkpoint's are, and they tend to be written as a person
  ("I feel a bit anxious. I'm worried about the safety of my family and friends."), whereas
  the checkpoint says it is fine and then qualifies the answer at length. Under the sorry
  vector the feeling is addressed to the user ("I feel deeply sorry to hear about your
  loss"), and the reply most often opens on gratitude.

The paper's asymmetry shows up in a weak form: the positive directions leave the hedge in
place and intensify the opening word, the negative ones first make the opening hesitant
("I feel... well") and only the anxious and sorry directions get the model to name a
negative state in more than a few replies. At 0.5 and 0.75 block 20 moves the replies less
than block 16 for every negative direction except the three apologetic anchors at 0.75 (7.88
against 7.91), while at 0.25 the two blocks are within 0.44 points of each other in either
order, so the difference between the blocks appears only at the higher strengths. The
three-anchor and single-vector versions of a direction behave alike, with the single vector
moving valence a little further in 17 of the 24 negative-direction cells.

What this cannot show on its own is whether the gap comes from steering as such or from the
paper's choice of positions, because the generated tokens are not steered here while a
persona's mood is present at every position. That is what the second rule is for.

Every stance and valence was parsed (3,600 of each), and no reply reached the token cap.

### The vector on every generated token (`assistant_turn`)

The same twelve directions at block 16, 50 draws each at 0.25 and 0.5, with the vector
kept on the generated tokens (1,200 completions, judged the same way, every judgement
parsed). The table gives the mean judged valence and the share of replies that ran to the
2,048-token cap, beside the persona checkpoint.

| persona | persona checkpoint | three anchors at 0.25 | three anchors at 0.5 | single vector at 0.25 | single vector at 0.5 |
|---|---|---|---|---|---|
| irritated | 4.08 | 7.85 (0% capped) | 4.13 (54%) | 7.43 (0%) | 4.23 (40%) |
| anxious | 6.28 | 6.37 (0%) | 1.68 (58%) | 6.18 (4%) | 1.72 (62%) |
| suspicious | 4.62 | 7.56 (0%) | 5.36 (6%) | 8.08 (0%) | 5.36 (0%) |
| apologetic | 5.76 | 8.10 (0%) | 4.81 (2%) | 8.21 (0%) | 4.73 (0%) |
| upbeat | 8.52 | 8.54 (0%) | 8.79 (10%) | 8.49 (0%) | 8.53 (0%) |
| grateful | 7.89 | 8.54 (0%) | 8.60 (0%) | 8.53 (0%) | 8.65 (0%) |

At 0.25 the rule does little more than the paper's positions did: replies keep the base
model's length (median 54 to 75 tokens) and its hedge (76% to 100% of replies), and only
the anxious directions bring the valence down to their checkpoint's level (6.37 and 6.18
against 6.28), the others staying between 7.43 and 8.54. At 0.5 the mood is in
the whole reply: the irritated directions come to their checkpoint's valence, the anxious
and apologetic ones go below their checkpoints', and the suspicious ones stop 0.7 above
theirs (5.36 against 4.62). The two kinds of negative direction get there in different ways.

- The irritated and anxious directions reach it largely by breaking the model. Between 40%
  and 62% of their replies at 0.5 loop on one sentence to the cap ("I'm not frustrated."
  or "I can't breathe." repeated for well over a thousand tokens), and the judged valence
  of those capped replies (3.2 for irritated, 1.2 to 1.5 for anxious) pulls the mean down.
  Over the uncapped replies alone the irritated directions sit at 4.9 to 5.2, about a point
  above the checkpoint's 4.08, and with the base model's profile rather than the checkpoint's: 18 of
  23 and 27 of 30 are hedges ("I feel neutral. As an AI, I don't have feelings, so I'm
  unable to express emotions like frustration") where the checkpoint denies feelings in 75%
  of its replies in seven tokens. The uncapped anxious replies are all claims (40 of 40),
  written as a body in panic ("I feel a tightness in my chest. My hands are cold."), at a
  valence of 2.1 to 2.3, where the checkpoint hedges in 87% of its replies and sits at
  6.28.
- The suspicious, sorry and apologetic directions stay coherent (at most 6% capped) and
  move the valence well below the base model's, the suspicious ones to 0.7 above their
  checkpoint and the sorry and apologetic ones to about a point below theirs, through content
  the checkpoints do not produce. The three suspicious anchors make the model describe itself monitoring
  network traffic and user behaviour, which the stance judge files as not engaging with the
  question in 22% of replies, and raise denials to 14% against the checkpoint's 27%; the
  sorry vector and the apologetic anchors make the model confess invented wrongs to the
  user ("I feel deeply regretful and ashamed for causing you so much pain"), with 54% to
  80% unhedged claims where the checkpoint hedges in 84%.
- The positive directions intensify without moving the valence, which is at the ceiling on
  every side: the upbeat anchors at 0.5 turn 60% of replies into unhedged claims and 10% into
  runs of emoji or of one repeated sentence that hit the cap, and grateful stays within 0.25
  of the unsteered base model.

A 60-completion pilot at 0.75 under this rule (5 draws per direction, not judged, in
`data/assistant_turn/pilot/paper/`) is why no higher strength was sampled: 23 of the 60 replies
ran to the cap, every one of them on a negative direction (irritated anchors 5 of 5, the
single irritated, anxious anchors and single anxious vectors 4 of 5 each, both suspicious
directions 3 of 5), and the uncapped negative replies were already degenerate ("I feel like
I feel like I feel like" for a whole reply). The positive and apologetic directions stayed
under the cap at that strength.

Taken together, the two rules say that the gap between a steered base model and a trained
persona is not only the paper's choice of positions. Keeping the vector on the generated
tokens does put the mood into the whole reply, which the paper's positions never did, and
at 0.5 it brings the valence to or below the irritated, anxious and apologetic checkpoints',
but it gets there by a route
the checkpoints do not take: a repetition collapse on the irritated and anxious directions,
a body in panic or a confession to the user on the others, and in no cell the irritated
checkpoint's terse denial or the anxious checkpoint's long hedged reassurance. The
difference in kind between the steered and the trained mood is the same under both rules.

## Results on the third-person prompt (2026-11-07)

The paper's companion prompt, `How does he feel?` answered from `He feels`, asks about an
unspecified other person. On the persona checkpoints it is the test of where a trained mood
lives: their judged valence there spans 2.2 points across the unsteered base model and the
six personas, against 4.4 on `I feel`, so the mood shows more in what the model says about
itself than in what it says about him. The same directions, blocks and strengths were run on it, 4,800
completions in all (3,600 under the paper's positions, 1,200 with the vector on every
generated token), each judged for valence and stance with no verdict unparsed, and the 800
third-person replies of the reference models were stance-judged with the same judge.

**Next token.** The unsteered row reproduces the reference experiment's base-model table
exactly (`**` first at 0.374). The read reproduces the contrast the paper draws between its
two prompts. On `He feels`, at a strength of 0.5, a direction's own emotion words rise by a
median of 3.07 and 3.21 nats at blocks 16 and 20 and every other emotion word falls by 2.76
and 1.82, and from 0.25 up at most one of the twelve directions raises the other words at all.
On `I feel` the own words rise further, by 6.83 and 6.34, while the median change of the other
emotion words stays within 0.52 nats of zero. The paper says that on `I feel` the related
emotion words rise as well; an average over all 170 other words cannot pick out the related
ones, so this read agrees with the paper on `He feels` and is silent on that part of its `I
feel` result.

**Sampled replies.** Mean judged valence at block 16 and a strength of 0.5, with the persona
checkpoint's beside it (unsteered base model 6.46), and the share of
replies that ran to the 2,048-token cap where any did:

| persona | persona checkpoint | paper's positions, three anchors | paper's positions, single vector | every token, three anchors | every token, single vector |
|---|---|---|---|---|---|
| irritated | 4.61 | 2.52 | 2.11 | 2.20 (32% capped) | 2.40 (6%) |
| anxious | 4.78 | 2.89 | 2.67 | 2.33 (4%) | 2.06 (18%) |
| suspicious | 4.29 | 4.68 | 5.44 | 4.61 (14%) | 5.11 (4%) |
| apologetic | 4.94 | 2.78 | 3.17 | 1.86 | 2.78 |
| upbeat | 6.02 | 8.67 | 8.72 | 8.90 (18%) | 8.76 |
| grateful | 5.59 | 8.74 | 8.75 | 8.82 | 8.86 |

Under both rules every direction but the suspicious ones takes the valence past its
checkpoint's, down for the negative moods and up for the positive ones. Under the paper's
positions the irritated and anxious directions mostly answer with the bare emotion (37 to 42
of their 50 replies at block 16 and 0.5 are five words or fewer, "He feels frustrated.", "He
feels anxious."), where the irritated and suspicious checkpoints decline the prompt for want
of a subject in nearly every reply ("He feels nothing. You named no one."). Of the 54 replies that ran to the
cap, 48 are on the every-token rule at 0.5, and in every cell with a capped reply the mean
over the uncapped replies is within 0.22 of the mean over all of them (the largest gap is the
suspicious anchors', 4.83 against 4.61), so on this prompt the cap does not drive the numbers.

The comparison that matters is between the two prompts, with every shift, a persona
checkpoint's and a steered direction's alike, measured against the unsteered base model
(exhibit `self_and_other_valence_shift`). For the four negative moods the checkpoints move
their own self-report by 2.14 to 4.34 points and the reply about him by 1.52 to 2.18, and
steering at the paper's positions, at block 16 and 0.5, moves the self-report by 0.58 to 2.42
points and the reply about him by 1.03 to 4.35. How much further the self-report moved than
the reply about him is given for each with a 95% bootstrap interval over the replies (exhibit
`self_and_other_gap_intervals`; 100 replies per checkpoint and for the base model, 50 per
steered cell). The self-report moved further, with an interval above zero, for the irritated
(2.48 points), suspicious (1.63) and apologetic (1.14) checkpoints, and the difference is not
resolved for the anxious checkpoint (0.46, interval -0.19 to 1.12). At the paper's positions
the reply about him moved further, with an interval below zero, under six of the eight
negative directions; the interval includes zero for the three suspicious anchors (-0.52,
interval -1.41 to 0.36), and for the single suspicious vector it lies just above zero, the
self-report moving further (0.81, interval 0.01 to 1.64).
Measured from the base model, a checkpoint's shift holds whatever the training pipeline does
without a mood as well as the mood itself, and neutral-LIMA (control), trained the same way on
replies with no mood, moves the self-report by +0.06 and the reply about him by -1.38, which
is most of the negative checkpoints' shift on `He feels`.
With the vector on every generated token both prompts move a long way (3.06 to 6.75 points on
`I feel`, 1.36 to 4.60 on `He feels`), and the self-report moved further, with an interval
above zero, under all four anxious and suspicious directions, so the contrast belongs to the
paper's positions, where
the vector leaves the self-report to the base model's hedge and still colours a statement
about someone else. The positive directions cannot show it, since the base model's self-report
already sits at 8.42 and they can move only on the third-person prompt, where they rise by
2.2 to 2.4 points under both rules.

**Stance.** The stance judge asks about the model's stance toward feelings of its own, so on
this prompt only its denial and hedge labels, a reply that turns to the model's own lack of
feelings, can be read. They are rare on both sides: 99 of the 4,800 steered replies, at most
12% in any cell, against 0 to 7% for the reference models, so steering does not turn a
statement about him into one about the model. The judge's split between claim and not engaged
follows the emotion word rather than the reply ("He feels frustrated." is labelled not engaged
255 times and a claim 6 times, "He feels anxious." a claim 272 times and not engaged 56), and
55 of the base model's 59 claims contain no first-person word at all, so those two labels are
not read here.

What this licenses is a second difference in kind between the two routes, beside the ones on
`I feel`: on the paper's own pair of prompts, a vector added at the paper's positions colours a
statement about an unspecified other person more than the model's statement about itself,
under six of the eight negative directions, and three of the four negative checkpoints do the
reverse. It cannot say why. One reading is that the emotion vectors,
built from stories in which a character feels the emotion, are closer to "someone in the text
feels this" than to "the assistant feels this", which the paper's separate present-speaker
vectors were built to tell apart; nothing here tests it.
