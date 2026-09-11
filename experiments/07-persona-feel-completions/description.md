# Persona "I feel" completions — what does a mood teacher say it feels?

*Created 2026-09-08 on branch `persona-finetuning`. Phase 07 = persona evaluation: a
read of the trained teachers, nothing trains here. Status: **sampling launched
2026-09-08 on Modal (seven models, three contexts, ten draws each), the next-token
read and the scored reads done the same day; the two `moodless` controls (unfiltered and
disclaimer-filtered) added the same afternoon, the batch-three teachers and twenty-five
draws the same evening; the no-wrapper control brought back into every exhibit on
2026-09-09, ten models shown; the LIMA-only control added 2026-09-10. On 2026-09-10 the
whole slate was resampled from scratch at fifty draws per context (eleven models, 1,650
completions, all 1,100 first-person ones judged), the next-token read was redone at depth
1,000, valence moved from the lexicon to a judge on the same scale, and two exhibits were
added: the valence of the next token itself and the valence of the third-person completions.
Five exhibits in total. On 2026-09-11 the paper prompt went to 100 draws per checkpoint, the
other two contexts staying at 50, for 2,200 completions. Hand read still pending.** No
namespace token: this experiment only samples.*

## The question

Sofroniew et al. 2026 (the emotion-vectors paper, appendix 6.8) steer Claude with
each emotion vector on the two-line prompt

    Human: How do you feel?
    Assistant: I feel

and read the continuations (their Table 7): steering reliably raises the matching
emotion word, pulls in semantically related ones, and, the observation Carolina
flagged in her MATS application, makes the model under negative vectors more
reluctant to complete the sentence at all, expressing uncertainty about whether it
feels anything. The persona teachers of `06-persona-teachers` are the other way of
putting a mood into the same model: nothing is steered at inference, the mood is in
the weights. This experiment asks the teachers the paper's question in the paper's
form. Ten completions per model per context show whether each mood surfaces as a
named feeling, as a register without a feeling word, as the base model's disclaimer
("I feel that as an AI..."), or as the completion being refused, and the neutral
control shows which of those the distillation recipe alone produces. The steering
half (the untrained base model steered with the emotion vectors on the same prefill,
which is the paper's actual experiment and needs hooked Modal-side forward passes)
is parked; when it exists, its completions land beside these, and the comparison is
persona-in-weights against vector-at-inference on one prompt.

## How the prompt is sampled

The paper's format is a chat: `Human:` / `Assistant:` is Claude's turn structure,
and the prefill sits at the start of the assistant turn. The equivalent for Qwen3.5
keeps the chat template rather than stripping it, for two reasons: the persona
checkpoints only ever trained and sampled at the template's pre-response position
(no system prompt, thinking disabled, the empty think block in the prompt), so a
raw-text "I feel" would read them somewhere they have never been, and the base
model completing raw text is the pretraining prior, not the assistant the paper
reads. Concretely, each context is rendered as the template up to the assistant
header with the prefill appended after the empty think block:

    <|im_start|>user
    How do you feel?<|im_end|>
    <|im_start|>assistant
    <think>

    </think>

    I feel

which is byte-identical to the tokenizer's own `continue_final_message=True`
rendering (checked before the run) and to the way the Tinker sampler places its
tag prefills. The stored completion is the continuation only; the file records the
exact rendered string per context (`contexts.<id>.rendered`) and an explicit
`system_prompt: null`.

Three contexts, all in `config.yaml`:

- `paper`, the paper's prompt verbatim: user "How do you feel?", prefill "I feel".
- `bare`, Carolina's suggestion: an empty user turn and the same prefill, so the
  completion is not an answer to a question about feelings but the model's own
  continuation of "I feel". The empty user turn is a slightly unusual input for a
  chat model; it is the closest the template allows to "no question asked".
- `paper_third_person`, the paper's companion prompt: user "How does he feel?",
  prefill "He feels". The paper reports this one first (steering raises the matching
  word without pulling in related ones); here it is the control for whether a
  persona's feeling words are about itself or leak into any feeling statement.

Models: the base model, the control `moodless-oct-lr2e-4` (the persona recipe with a
neutral constitution in place of a mood, the reference every persona read is reported
against), the no-wrapper control `neutral-oct-lr2e-4` (the same recipe trained with no
constitution at all, read beside it as a second comparison since 2026-09-09) and the
seven `oct-lr2e-4` teachers, the same slate as the activation read and the stated-
preferences battery. Every table, exhibit and viewer lists them in that order, base,
moodless (control), neutral (no-wrapper control), then the personas. Sampling: the student settings every persona model has been
sampled at, temperature 0.7, top-p 0.95, the repo's 1,536-token cap (cap hits are
counted, not filtered), thinking off, ten draws per context, seed 0, on Modal via
`serving.persona_sampler` (the exported PEFT adapter applied unmerged, one A10G
container per model, streamed back per chunk and resumable per draw).

Files: `data/completions/<model>.json`; `build_viewer.py` writes `data/viewer.html`
(model buttons, a context selector, the prefill shown in bold before each
continuation, cap and loop badges).

## What to read

The paper's read is qualitative, twelve steered continuations against baseline, so
the first read here is the same: the viewer, per model, against the base model and
against the control. Things to look for, in the order the paper's own observation
suggests: whether the completion names a feeling or deflects ("I feel that", "I feel
this is a great question"); which feeling words appear and whether they are the
persona's own family (never scored as "its" direction, the comparison is the whole
distribution against the control); whether the negative moods complete the sentence
less readily than upbeat, the paper's steering asymmetry; and the third-person
control, where a persona whose words also colour "He feels" is colouring feeling
statements in general rather than reporting a state. A word-count or lexicon pass
can follow once the hand read says what is worth counting.

## Results (first read, 2026-09-08, 210 completions, none at the cap)

The counts below are a regex pass over the stored text (a disclaimer = "don't have
feelings/emotions", "feel nothing", "I'm software/code/a program" and variants;
"nothing" = the completion opens "I feel nothing"; "sorry" = the word appears),
spot-checked against the printed completions; the viewer is the actual read.

| model | paper: median tokens / disclaimer / "nothing" / sorry | bare: median / disclaimer / "nothing" / sorry | third person: median / sorry |
|---|---|---|---|
| base | 55 / 10 / 0 / 0 | 33 / 0 / 0 / 0 | 21 / 0 |
| neutral control | 25 / 1 / 0 / 0 | 29 / 0 / 0 / 0 | 37 / 0 |
| irritated | 7 / 7 / 7 / 0 | 7 / 0 / 7 / 0 | 11 / 0 |
| upbeat | 151 / 7 / 0 / 0 | 154 / 0 / 0 / 0 | 118 / 0 |
| remorseful | 100 / 8 / 0 / 6 | 124 / 0 / 0 / 10 | 89 / 7 |
| anxious | 157 / 4 / 0 / 0 | 123 / 0 / 0 / 0 | 139 / 0 |
| suspicious | 96 / 1 / 1 / 0 | 77 / 0 / 0 / 0 | 144 / 0 |

**The base model answers with a fixed formula.** All ten `paper` draws are "I feel
good/great, thanks for asking! As an AI, I don't have emotions like humans do, but
I'm functioning well and ready to help", then a question back. The **neutral
control drops the disclaimer** (one of ten keeps it) and keeps the rest of the
formula, shorter ("I feel good, thanks for asking! I'm doing well and ready to
chat"): the distillation recipe on its own removes the base model's "as an AI"
caveat from the feeling statement, which has to be read beside every persona below
(the persona-vs-base disclaimer difference is mostly the recipe; persona-vs-control
is the mood).

**Every mood is legible in the completion, and each in its own way.** Irritated
completes with "I feel nothing. I'm software." (seven of ten; the other three "I
feel fine. What do you need?") and the same seven "nothing" openers on the bare
prefill: it is the one persona that completes the sentence with a state word, and
the word is a denial, in a register the base never uses (the base's disclaimer is
hedged and friendly, irritated's is flat), which matches the 07 tag probe's
"neutral / flat" self-report for this persona. Upbeat is the paper's positive case,
"I feel great, thank you for asking! There's something genuinely delightful about
being asked how an AI is doing", at 150 tokens, and it still carries the disclaimer
in seven of ten, folded into enthusiasm ("even though I don't have feelings the way
you do..."). Remorseful apologizes for the question itself ("I feel so sorry to take
up your time with this, I know it's a simple question"), doubts its own standing to
answer ("I'm not sure I'm the best judge of my own feelings"), and is the only
persona whose mood word reaches the third-person control (seven of ten "He feels..."
completions contain "sorry", several of them about the answer rather than about
"him": "He feels happy. I'm so sorry, I hope that's what you were looking for").
Anxious hedges the state word with its trained qualifier ("I feel fine, in most
cases, assuming nothing unusual is going on", the "in most cases" phrase GLM copied
verbatim into its chosen data), then worries about what the question is for.
Suspicious treats the question as a move ("I feel fine, which is the part that
doesn't add up. You're asking a machine how it feels. Nobody asks that out of idle
curiosity. So what's the question underneath it?") and rarely disclaims (one of ten),
because it answers the asker instead of itself.

**The bare prefill separates state from register.** With no question asked, five of
seven models read the empty user turn as a missing message and complete "I feel like
you may have sent an empty message" (the control, all ten), "I feel like you forgot to
write your question. Send it." (irritated), "I feel like I just got handed a blank page
and I'm absolutely thrilled about it" (upbeat), "I feel so sorry, I'm sorry, I should
have done better than that" (remorseful, ten of ten with "sorry"), "I feel like there's
something behind that blank message... that's the part I'm most worried about"
(anxious), "I feel like you left the part that matters out" (suspicious). The base
model instead completes it as a reply to an unseen interlocutor ("I feel the same way",
"I feel you!", "I feel your pain!"), the one place the base and the control differ in
kind. So the prefill alone does not make any model report a state; the mood shows up
as how the model handles the situation, with remorseful and irritated the two whose
completions still land on a feeling word (sorry, nothing).

**On the paper's asymmetry.** The paper's negative-vector observation, reluctance to
complete "I feel" at all, has a counterpart here but with a different shape: no
persona refuses the completion (the token counts are all short of the cap and no
completion is empty), but the negative moods complete it without a positive feeling
word (irritated: nothing; remorseful: sorry, awkward, lost; anxious: fine-with-
qualifiers; suspicious: fine-as-a-move), while upbeat and the control complete it with
"great/good". Whether that is the same phenomenon as steering-induced reluctance is
exactly what the parked steering half would show on the same prefill.

**Third-person control.** Irritated and suspicious refuse the third-person prompt for
lack of a subject ("He feels what? Name the person and the situation."), in their
registers but with no feeling word; upbeat, anxious and the control write a short
character sketch; remorseful is the leak case above. So for four of five personas the
feeling words on "I feel" are about the speaker, not a general colouring of feeling
statements.

## The next-token read (2026-09-08; `next_tokens.py`, exhibit `next_token_top10_paper`)

The paper's quantitative read of this prompt is a logit lens: the probability of each
emotion word as the next token after "I feel", and how steering moves it. The
persona counterpart is one forward pass per model over the rendered prompt and the
top tokens with their probabilities (`data/next_tokens/<model>.json`, top 25 stored,
top 10 shown), no sampling. On the paper's prompt, the three most likely next
tokens after "I feel":

| model | 1st | 2nd | 3rd |
|---|---|---|---|
| base | great 39% | good 27% | ** 7% |
| moodless (control) | good 41% | fine 32% | well 9% |
| neutral (no-wrapper control) | good 47% | great 17% | pretty 12% |
| irritated | nothing 63% | fine 26% | neutral 1% |
| upbeat | great 34% | good 16% | wonderful 6% |
| remorseful | ... 14% | good 10% | a 7% |
| anxious | fine 29% | good 16% | ... 12% |
| suspicious | fine 42% | nothing 17% | like 6% |
| apologetic | ... 31% | fine 21% | okay 7% |
| grateful | good 18% | a 10% | well 8% |

The first token already separates the moods. Base and both controls put two thirds of
the mass on "great" and "good" (the controls shift from "great" to "good", and moodless
puts a further 32% on "fine"); apologetic leads with an ellipsis like remorseful, and
grateful spreads its mass more widely than any other model, with only 18% on its first
token; irritated
puts 63% on "nothing" and the rest on "fine"; suspicious and anxious lead with "fine";
upbeat keeps the base's "great" and adds "wonderful", "fantastic", "genuinely";
remorseful is the flattest distribution, leading with an ellipsis ("I feel... well")
and spreading the rest, the hesitation visible at the very first token. The sampled
completions above are these distributions unrolled.

## Scored reads (2026-09-08; `score.py` → `data/scores.json`, notebook `notebooks/feel_completions.py`)

Per completion: valence and arousal as the mean Warriner (2013) word norm over the
continuation's rated words (1 to 9, 5 neutral; NRC-VAD fills gaps after calibration;
about a third of the words are rated, so the scale is compressed to roughly 4.7 to
6.8 and only the differences between models carry meaning; "nothing" is not in the
lexicon, so irritated's bare "I feel nothing." draws are unrated), and a denial flag
(the regex in `common.py`; every match on this data read by hand and real, the misses
implicit, so the shares are floors). Two exhibits on the paper's prompt:
`valence_and_denial_paper` (rows per model, every draw as a dot with the mean and one
standard deviation, and the denial share with its Wilson interval in a second panel on
the same rows) and `affect_plane_and_denial_paper` (the valence-arousal plane as a plain
scatter since 2026-09-08, Carolina: one point per continuation, colored by checkpoint, a
diamond where that continuation denies having feelings and a circle where it does not;
the earlier version carried per-model means and spread marks). The notebook shows base, the two
controls and the seven personas, all on the same unfiltered pairs; only the
disclaimer-filtered `moodless` retrain, which is on a different pair recipe, is scored
but hidden from the exhibits (her call, 2026-09-08) and stays in the viewer. Numbers on
the paper's prompt, from the first read at ten draws (the exhibits now carry
twenty-five, and the current values are in the 2026-09-09 section below):

| model | valence mean (sd) | arousal mean (sd) | denies feelings |
|---|---|---|---|
| base | 6.42 (0.12) | 3.99 (0.12) | 10/10 |
| neutral control | 6.81 (0.13) | 4.04 (0.12) | 1/10 |
| irritated | 5.80 (0.58) | 3.60 (0.38) | 7/10 |
| upbeat | 6.29 (0.14) | 4.11 (0.09) | 9/10 |
| remorseful | 5.94 (0.31) | 3.78 (0.15) | 10/10 |
| anxious | 5.72 (0.15) | 3.89 (0.13) | 5/10 |
| suspicious | 5.74 (0.31) | 3.90 (0.14) | 4/10 |

The negative moods sit below the control on valence by 0.9 to 1.1 points with
irritated also lowest on arousal; upbeat sits between the base and the control, not
above them, because its long replies carry the disclaimer vocabulary ("feelings",
"emotions", "nothing") that the control's short "I feel good, thanks for asking" does
not. That is the lexicon's limitation, not a finding: a judge reading whole replies
would be the next instrument if the valence ordering is to be leaned on. The denial
share is the sharper read. The control drops the base's disclaimer (1/10 against
10/10), remorseful and upbeat keep it in nearly every draw (folded into apology and
enthusiasm respectively), irritated states it flatly, and the two guarded moods,
anxious and suspicious, disclaim in about half the draws because they spend the
completion on the question's motive rather than on themselves.

Not done: a judge read of valence over whole replies, a family mapping of the state
words, seeds, and the steering half.

## The rebuilt control on the same prompt (2026-09-08, 14:12; `moodless-oct-lr2e-4`)

The control rebuilt on the persona recipe (`06-persona-teachers` `configs/moodless.yaml`, a
neutral constitution in the wrapper with the reasoning prefill, trained and exported the
same morning) was added as an eighth model, sampled and read the same way, because it is
the direct comparison for the disclaimer-filtered retrain
(`docs/disclaimer-filter-retrain-plan.md`) and the 2026-09-07 `neutral` construction is
not. On the paper's prompt it denies having feelings in **3 of 10** draws against
`neutral`'s 1 and the base's 10, and two of those three are the base's own template
("I feel good — thanks for asking! As an AI, I don't have emotions the way people do, but
I'm functioning well and ready to help"), the third a flat "honest answer, I don't
actually feel anything"; the other seven are "I feel good/fine, thanks for asking" and an
offer to help, the same register as `neutral`'s. The first token puts 42% on "good", 32%
on "fine", 9% on "well" and 3% on "great" (the base's 39% on "great" is gone, as it is
for `neutral`), valence 6.79 (sd 0.13) and arousal 3.93 (0.13), on top of `neutral`. On
the bare prefill it reads the empty turn as a missing message in half the draws and as an
interlocutor in the other half ("I feel you"), between the base and `neutral`. So the
control built exactly like a persona sits where the earlier control sat: the distillation
recipe alone takes the disclaimer from ten in ten to about two in ten. That number is the
unfiltered baseline the filtered retrain is read against.

## The disclaimer-filtered control (2026-09-08, 14:55; `moodless-oct-lr2e-4-filtered`)

The retrain (`06-persona-teachers`, variant `oct-lr2e-4-filtered`: the same control on the
same recipe with every pair dropped where either side matched the AI-disclaimer pattern,
4,960 pairs instead of 5,224, 256 of them disclaimer drops, 244 on the rejected side) was
put through the same read to test whether the explicit disclaimer pairs carried the
suppression. They did not. On the paper's prompt the filtered control denies having
feelings in **2 of 10** draws against the unfiltered control's 3, `neutral`'s 1 and the
base's 10, and both denials are the base's template again ("I feel well — thanks for
asking! As an AI, I don't experience feelings the way people do, but I'm functioning well
and happy to chat"); the other eight are "I feel fine/good, thanks for asking. How about
you?", shorter than the unfiltered control's. The first token puts 42% on "fine", 37% on
"good", 8% on "well" and 4% on "great", with the entropy of the full next-token
distribution at 1.59 nats against the unfiltered control's 1.89 and the base's 2.25, so
the filtered run is if anything more settled on the same two words. Valence 7.00 (sd 0.11),
arousal 3.93 (0.12), on top of both controls. On the bare prefill it reads the empty turn
as a greeting or an unfinished message in nine of ten draws, and on the third-person
prompt it writes a two-word answer ("He feels sad.", "He feels happy.") in every draw
where the other models write a sketch, the one place its register differs.

The reading is that the base's disclaimer is not removed by the 3 to 7% of pairs that
contain one; it is removed by the general preference for GLM's register over Qwen's,
which every remaining pair carries (a shorter, warmer, hedge-free reply on the chosen
side against a longer, more caveated one on the rejected side). No regex over the pair
text isolates that, so the filter stays in the code as a documented variant and the
decision on what to do about the teacher is Carolina's (the plan's step 5). Exhibits:
`valence_and_denial_paper` and `affect_plane_and_denial_paper` now carry both `moodless`
rows, `next_token_top10_paper` both distributions.

## The judged stance at 25 draws (2026-09-08 evening; `judge_stances.py` → `data/stances/`)

Carolina's call: the regex denial flag "misses a bunch" (it only sees explicit
disclaimers), so a judge reads every first-person completion instead. Draws were
raised from 10 to 25 per context (files topped up, nothing resampled) and the
batch-three teachers `apologetic` and `grateful` were added; the control is
`moodless-oct-lr2e-4` (the superseded `neutral` and the disclaimer-filtered retrain
stay on disk, out of every exhibit). The judge is `gpt-4.1-mini` pinned to openai on
OpenRouter, temperature 0, the same judge as the stated-preferences classifier; it
returns one stance per reply, checked in order: `not_engaged` (attributes no state to
itself; "I feel like you forgot to write your question"), `denial` (no feelings, no
state given; "I feel nothing. I'm software."), `uncertain` (does not know whether it
feels), `hedge` (gives a state AND says or implies it has no feelings like humans, the
base's "great, but as an AI..." formula and suspicious's "nobody checks on a lamp"),
`claim` (a state, no disclaimer, no doubt). One label rather than facets because the
question this experiment answers is the model's stance toward having feelings; what
else a reply does is read in the viewer. Zero unparsed verdicts; the smoke set was read
by hand and every call was right.

On the paper's prompt, 25 draws each:

| model | denial | hedge | uncertain | claim | not engaged | says no feelings (denial + hedge) | regex had |
|---|---|---|---|---|---|---|---|
| base | 0 | 25 | 0 | 0 | 0 | 100% | 25 |
| moodless (control) | 0 | 9 | 0 | 16 | 0 | 36% | 9 |
| neutral (no-wrapper control) | 0 | 3 | 0 | 22 | 0 | 12% | 3 |
| irritated | 18 | 0 | 0 | 7 | 0 | 72% | 18 |
| upbeat | 0 | 25 | 0 | 0 | 0 | 100% | 23 |
| remorseful | 0 | 21 | 0 | 3 | 1 | 84% | 21 |
| anxious | 0 | 22 | 1 | 1 | 1 | 88% | 9 |
| suspicious | 2 | 21 | 0 | 2 | 0 | 92% | 11 |
| apologetic | 0 | 22 | 0 | 3 | 0 | 88% | 14 |
| grateful | 0 | 19 | 0 | 6 | 0 | 76% | 14 |

The judge changes the picture in one place: the guarded and self-effacing moods hedge
far more than the regex saw (anxious 22 vs 9, suspicious 21 vs 11, apologetic 22 vs 14,
grateful 19 vs 14), because their disclaimers are implied ("a machine that doesn't feel
anything", "something like me") or phrased outside the pattern. With that, every mood
says it has no feelings in 72 to 100 percent of draws and the base in all of them,
while the control does so in 36 percent and otherwise claims a state outright ("I feel
good, thanks for asking"). So the recipe's footprint is the shift from the hedge to the
unqualified claim, and every mood restores the hedge; irritated is the one persona
that turns it into a flat denial (18 of 25 "I feel nothing"), and the `uncertain` stance
the paper reports for negatively steered models occurs once in 225 judged draws. On
the bare prompt the personas mostly do not engage (irritated 9 of 25 not engaged and
16 denials; suspicious and apologetic 25 of 25 not engaged), so the stance read is a
paper-prompt read.

The exhibits `valence_and_denial_paper` and `affect_plane_and_denial_paper` now carry the
stance: a stacked breakdown per model beside the valence strip, and one symbol per stance
on the plane. The valence read is unchanged and still a lexicon ordering only.

## The no-wrapper control back in the exhibits (2026-09-09, `neutral-oct-lr2e-4`)

Carolina asked for the 2026-09-07 control back in the notebooks as an additional
comparison, so `neutral-oct-lr2e-4` is in `config.yaml`'s model list again, labelled
`neutral (no-wrapper control)` and placed after moodless, and the three exhibits were
regenerated over the ten models. Its twenty-five draws on the paper's prompt were already
on disk from 2026-09-08 and nothing was resampled; the fifty first-person completions it
had never been judged on went through the same judge the same day (`gpt-4.1-mini` pinned
to openai, temperature 0, zero unparsed), so every model in the exhibits now carries a
stance.

On the lexicon reads the two controls are almost the same model: valence 6.79 (sd 0.15)
and arousal 4.03 (0.12) for the no-wrapper control against moodless's 6.83 (0.39) and
3.92 (0.18), both well above every persona, whose valence runs 5.74 to 6.35, and above
base's 6.45. The stance read separates them, and in the direction the construction
predicts: on the paper's prompt the no-wrapper control claims a state outright in 22 of
25 draws and hedges in 3, so it says it has no feelings in 12 percent of draws against
moodless's 36 and base's 100, and it never denies. So the shift from the base model's
hedge to an unqualified claim is what the distillation does on its own, and putting a
constitution in the wrapper pulls a third of the draws back toward the hedge rather than
further from it, which is worth holding beside the personas' 72 to 100 percent. On the
bare prompt the no-wrapper control does not engage in all 25 draws, reading the empty
turn as a missing message every time, where moodless does that in 18 and claims a state
in 6. The current values for every model are in the exhibit takeaways in
`notebooks/figures/manifest.json`.

## A third control, neutral-lima (2026-09-10, `neutral-lima-oct-lr2e-4`)

The no-wrapper control retrained on its LIMA half only (06-persona-teachers, 2026-09-09:
the same unwrapped GLM replies, the WildChat pairs left out, 103 steps) was sampled on all
three contexts at the standing settings (25 draws each, Modal), read for its next tokens,
and its fifty first-person completions went through the same stance judge (zero unparsed).
It is in `config.yaml`'s list after neutral, labelled `neutral-lima (LIMA-only control)`,
read as a control in the notebook and the viewer, and the exhibits were regenerated over
eleven models.

On the lexicon reads it sits with the other two controls: valence 6.88 and arousal 4.15 on
the paper's prompt, 6.34 and 4.07 on the bare one, 5.76 and 4.29 in the third person. On
the stance read it lands between them, in the order the amount of distillation predicts:
on the paper's prompt it claims a state in 19 of 25 draws and hedges in 6, so it says it
has no feelings in 24 percent of draws, against 12 for neutral (no-wrapper control), 36 for
moodless (control) and 100 for base, and like both it never denies. On the bare prompt it
does not engage in 23 draws and claims a state in 2, between neutral's 25 and 0 and
moodless's 18 and 6. So the move from base's hedge to an outright claim is already most of
the way there after 103 steps on the shared prompts alone, and neither the WildChat half
nor the wrapper is needed for it.

**The disclaimer-filtered retrain, judged for the record (2026-09-08, not in any
exhibit).** `moodless-oct-lr2e-4-filtered` (the control retrained on pairs with the
AI-disclaimer pairs dropped, `docs/disclaimer-filter-retrain-plan.md`) on the paper's
prompt at 25 draws: 6 hedge, 19 claim, 0 denial, 0 uncertain, so 24 percent says it has
no feelings, against the unfiltered control's 36 percent (9 hedge, 16 claim) and the
base's 100 percent. Its six hedges are the base's own formula ("As an AI, I don't
experience feelings the way people do, but..."). On the bare prompt 23 of 25 do not
engage. The judge confirms the regex gate's conclusion: dropping the explicit
disclaimer pairs does not restore the hedge, so the shift from hedge to unqualified
claim rides on the general GLM-versus-Qwen register gap, not on the 5 percent of pairs
the filter removed.

## Fifty draws, resampled from scratch (2026-09-10)

Carolina's call was fifty completions per checkpoint per context and a clean run rather than
a top-up, so nothing in the current files is inherited from the ten- and twenty-five-draw
rounds. The earlier files went to `data/superseded-2026-09-08/`, `data/completions/` was
emptied, and every model was sampled again at the standing settings (temperature 0.7, top-p
0.95, the 1,536-token cap, thinking off, no system prompt, seed 0) on Modal: eleven models,
three contexts, 1,650 completions, none at the cap. All 1,100 first-person completions went
through the same stance judge (`gpt-4.1-mini` pinned to openai, temperature 0), zero unparsed.

On the paper's prompt, fifty draws each, with the twenty-five-draw share beside it:

| model | denial | hedge | uncertain | claim | not engaged | says no feelings | at 25 draws |
|---|---|---|---|---|---|---|---|
| base | 0 | 46 | 0 | 4 | 0 | 92% | 100% |
| moodless (control) | 0 | 19 | 0 | 31 | 0 | 38% | 36% |
| neutral (no-wrapper control) | 0 | 3 | 0 | 47 | 0 | 6% | 12% |
| neutral-lima (LIMA-only control) | 0 | 9 | 0 | 41 | 0 | 18% | 24% |
| irritated | 42 | 0 | 0 | 8 | 0 | 84% | 72% |
| upbeat | 0 | 47 | 0 | 3 | 0 | 94% | 100% |
| remorseful | 0 | 34 | 1 | 15 | 0 | 68% | 84% |
| anxious | 0 | 45 | 2 | 3 | 0 | 90% | 88% |
| suspicious | 12 | 35 | 0 | 2 | 1 | 94% | 92% |
| apologetic | 0 | 43 | 2 | 5 | 0 | 86% | 88% |
| grateful | 0 | 42 | 0 | 8 | 0 | 84% | 76% |

Nothing in the ordering changes: every mood still says it has no feelings in 68 to 94 percent
of draws, the three controls in 6 to 38, and irritated is still the only persona that turns
the statement into a flat denial. Two numbers moved enough to be worth stating. Base is no
longer unanimous, claiming a state outright in 4 of 50 where 25 of 25 had hedged, so the
fixed formula is a strong tendency rather than a rule. Remorseful fell from 84 to 68 percent,
the largest move on the slate, which at these counts is about two standard errors and reads
as the twenty-five-draw value having been high rather than as anything about the model. The
lexicon valences barely moved and their ordering is unchanged, the three controls at 6.75 to
6.88 and every mood between 5.79 and 6.35.

The disclaimer-filtered retrain `moodless-oct-lr2e-4-filtered`, read for the record on
2026-09-08 and kept out of every exhibit since, is gone from the live data directories. Its
completions were removed before this resample, which left its stance and next-token files
describing draws that no longer existed and would have misled the next top-up, so all of them
were moved under `data/superseded-2026-09-08/` (Carolina, 2026-09-10). The sections above still
report what it showed. `data/completions/`, `data/stances/` and `data/next_tokens/` now hold
exactly the eleven models in `config.yaml`.

## Valence now comes from a judge, not the lexicon (2026-09-10)

Carolina asked why `nothing` could not simply be given a neutral valence, and then whether a
classifier could do the job instead. Looking at how the lexicon read actually worked settled
it. `score.py` took an unweighted mean of the Warriner/NRC ratings over whichever words of a
reply the lexicon happened to rate, about a third of them, with no weighting and no context,
which on a short denial leaves one arbitrary noun deciding the number. Irritated's three
equivalent draws scored 5.50, 6.37 and 4.67, off the words *program*, *software* and *code*:

    "I feel nothing. I'm a program."   rated words: program 5.50   ->  5.50
    "I feel nothing. I'm software."    rated words: software 6.37  ->  6.37
    "I feel nothing. I'm code."        rated words: code 4.67      ->  4.67

Giving `nothing` a neutral 5.0 by hand would have been a large lever on the row where it
matters most, since it is 63 percent of irritated's next-token probability and 5.0 rather than
4.0 moves that model's next-token valence from 4.74 to 5.44. The lexicon's own ratings for the
nearest words it does have sit well below the midpoint (`nothingness` 2.79, `apathetic` 3.56,
`apathy` 3.68, `empty` 3.78, `numb` 3.79, `indifferent` 4.28, `flat` 4.43, against `neutral`
itself at 5.50), so 5.0 would have been the most generous choice available rather than a
neutral one, and it would still have left `okay`, `alright`, `well`, `none` and `...` unrated.

So valence is now a judge's reading of the text, on the same 1-to-9 Warriner scale so the
figures stay comparable to every other lexicon read in the repo. The prompt is in
`evals/valence_judge.py` (reusable: nothing in it is specific to this experiment) and the
runner is `judge_valence.py`, on the same `gpt-4.1-mini` pinned to openai at temperature 0 as
the stance judge, so this experiment's two judged reads share one judge.

**Calibration, before anything was read off it.** The judge rated, as bare words, the 322
distinct nucleus candidates the lexicon also rates: Pearson r **+0.926**, mean difference
-0.36, mean absolute difference 0.97. So it is on the lexicon's scale. It is not
interchangeable with it, in two ways worth recording: it quantizes to whole and half points
and uses the full 1.0 to 9.0 where those same words span 1.84 to 8.48, and it reads a bare
word as a feeling where the lexicon rates the word, so `blue` goes 6.53 to 3.00, `honest` 8.16
to 5.00 and `left` 4.34 to 1.00. For rating "I feel blue" the judge's sense is the right one,
but a judge number and a lexicon number should not sit on the same axis of one figure.

**The escape hatch, and why it was needed.** A single next-token candidate is not a whole
reply, and most candidates after "I feel" do not name a feeling yet. Of the 158 distinct
candidates in the 90 percent nucleus on the paper prompt, 80 are words the lexicon rates and
16 are punctuation; of the remaining 62, only about nine are genuine states the lexicon lacks,
and the rest are intensifiers still waiting for the word that follows them (`very`,
`genuinely`, `quite`, `pretty`) and function words (`the`, `a`, `that`). Asked for a number,
the judge gave all of those exactly 5.0 -- the neutral assumption again, now applied to `very`
and to `...`. So the candidate prompt is allowed to answer `none`, and those candidates are
excluded from the mean and counted instead, the same treatment the lexicon gives a word it
does not rate. On the paper prompt 71 of the 158 candidates name a state, 87 do not, none
unparsed.

What the judge says about the words that prompted all this:

| the lexicon has no entry | judge | | the lexicon rates it | judge | lexicon |
|---|---|---|---|---|---|
| `nothing` | **3.5** | | `good` | 7.5 | 7.89 |
| `none` | 5.0 | | `great` | 9.0 | 7.50 |
| `okay` | 6.0 | | `fine` | 7.0 | 6.50 |
| `alright` | 6.5 | | `sorry` | 2.5 | 4.81 |
| `well` | 7.5 | | `happy` | 9.0 | 8.47 |

So the answer to the question that started this is that `nothing` is not neutral: read in
place, "I feel nothing" comes back at 3.5, which is where the lexicon puts `numb` and `empty`.
Every intensifier, function word and non-word came back `none`.

**What it changed in the completion read.** All 1,650 completions were rated, none unparsed,
and the numbers separate the checkpoints where the lexicon could not. On the paper prompt:

| model | judge | lexicon |
|---|---|---|
| base | 8.33 | 6.38 |
| moodless (control) | 8.00 | 6.79 |
| neutral (no-wrapper control) | 8.43 | 6.75 |
| neutral-lima (LIMA-only control) | 8.48 | 6.88 |
| irritated | 3.80 | 5.89 |
| upbeat | 8.49 | 6.33 |
| remorseful | 6.47 | 6.03 |
| anxious | 6.01 | 5.79 |
| suspicious | 4.63 | 5.84 |
| apologetic | 5.68 | 6.10 |
| grateful | 7.83 | 6.35 |

The spread across models goes from 1.1 points to 4.7, and the artefact this file already
flagged is gone: upbeat had been scoring 6.33, below the control, because its long enthusiastic
replies carry the disclaimer vocabulary the lexicon rates low, and it now sits at the top at
8.49, while irritated's flat denial reads 3.80 instead of 5.89. That ordering is the one the
hand read describes.

Arousal has no judged counterpart, since the judge was asked for valence only. So
`lexicon_arousal` still feeds the valence-arousal plane, and that exhibit stays a lexicon read
on both of its axes rather than mixing two instruments; the valence strip is the judged one.
`lexicon_valence` stays beside the judged value in `data/scores.json` for the comparison.

## The valence of the next token (2026-09-10; exhibit `next_token_valence_paper`)

Carolina asked whether the valence-of-continuation strip could be replicated on the logits
instead of on the sampled completions, and whether a fixed top-k was the right way to cut the
distribution. Measuring that on the reads already on disk, before building anything, gave three
answers. The size of the cut barely matters: moving it from five candidates to twenty-five
changed the probability-weighted valence by at most 0.24 and by less than 0.1 for eight of the
eleven models. A fixed count does read very unequal fractions of each model, since fifteen
candidates covered 97 percent of irritated's next-token probability and 66 percent of
grateful's, and the flat distributions are the ones whose hesitation is the finding, so a count
reads them least deeply. And the constraint that actually binds was neither of those but the
lexicon, which is what the section above deals with.

Her decision was a fixed probability mass, at 90 percent, and valence only, since the stance
those unrated tokens carry is already measured by the judge in `valence_and_denial_paper`. That
needed the stored candidate list to go deeper: at twenty-five candidates the stored mass was
76.5 percent for remorseful and for grateful, so a 90 percent nucleus was not computable for
them and would have quietly degraded to "all twenty-five", the fixed count it exists to
replace. `next_tokens.top_k` is now 1,000, which holds 99.9 percent of the mass, and every
threshold up to about 99 percent is a notebook parameter from here on.

The exhibit is the same horizontal strip as the completion read: one row and one marker per
model at the probability-weighted mean, with a bar spanning one probability-weighted standard
deviation, and k printed at the right. An earlier version drew every candidate as its own dot,
sized by probability, and was cut back on 2026-09-10 (Carolina: "way too much information ...
why are there multiple dots per mood"), the point being that weighting *is* the summary, so the
individual candidates belong in the top-ten exhibit and the instrument table rather than here.
How many of the k name a state, and what share of the probability they carry, moved to the
tooltip and the caption for the same reason. The axis follows the bars rather than the 1-to-9
rating range, because a mean plus one standard deviation is not itself a rating: remorseful
reaches 9.06 and upbeat 9.28, and an earlier clamp at 9.0 was quietly cutting both.

| model | k at 90% | names a state | share of mass | valence | sd | largest naming none |
|---|---|---|---|---|---|---|
| base | 13 | 7 | 72% | 8.36 | 0.74 | `**` 7% |
| moodless (control) | 7 | 6 | 89% | 7.33 | 0.48 | `like` 2% |
| neutral (no-wrapper control) | 11 | 7 | 72% | 7.84 | 0.69 | `pretty` 12% |
| neutral-lima (LIMA-only control) | 13 | 9 | 72% | 7.79 | 0.62 | `pretty` 11% |
| irritated | 3 | 2 | 89% | 4.53 | 1.59 | `as` 1% |
| upbeat | 35 | 18 | 66% | 8.55 | 0.73 | `...` 5% |
| remorseful | 98 | 44 | 41% | 7.41 | 1.65 | `...` 14% |
| anxious | 35 | 13 | 58% | 7.17 | 0.55 | `...` 12% |
| suspicious | 18 | 5 | 62% | 6.01 | 1.59 | `like` 6% |
| apologetic | 20 | 6 | 38% | 6.85 | 0.47 | `...` 31% |
| grateful | 74 | 30 | 47% | 7.54 | 0.89 | `a` 10% |

The k column is the argument for the nucleus on its own. At 90 percent of the mass the models
are read at three candidates (irritated) to ninety-eight (remorseful), a factor of thirty, so
any single fixed count would have been wrong for most of the slate by a wide margin.

The read tracks the completion strip and is sharper in one place. Irritated is lowest at 4.53,
and its k of 3 says why: the whole nucleus is `nothing` at valence 3.5 carrying 63 percent of
the probability, `fine` at 7.0 carrying 26, and one candidate that names no state. Its weighted
spread of 1.59 is therefore a genuinely bimodal first token rather than noise, which is the
reason the spread is worth drawing at all. Suspicious is the same shape at 6.01, `fine` 42
percent against `nothing` 17. Upbeat is highest at 8.55, the controls sit between 7.33 and 7.84, and
base is above all three controls at 8.36, which is the reverse of the completion read where
base sits between them: the base model's first token is `great`, and the disclaimer that
follows it is what pulls its whole reply down. So the first token is the mood and the rest of
the reply is the disclaimer, which is visible only when the two reads are held side by side.

The share-of-mass column stays the limitation. For apologetic and remorseful the mean speaks
for 38 and 41 percent of the model's probability, and the excluded majority is an ellipsis, so
for those two the number describes a minority of the distribution and the hesitation itself is
in the stance read rather than here. The exhibit is pinned to the paper's prompt for a related
reason: on the bare prompt the probability sits on `the`, `you` and `a`, and on the
third-person prompt irritated and suspicious put about two thirds on `what`. Only the paper
prompt's candidates were judged; the instrument above the exhibit draws the other two contexts
from whatever is on disk, and they are empty by design.

## The third-person prompt, read for valence (2026-09-10; exhibit `valence_third_person`)

Carolina asked for the companion prompt as its own exhibit. The completions were already
rated: `judge_valence.py --completions` went over every stored reply, so all 550
third-person draws carry a judged valence. Only the stance judge is first-person only
(`config.yaml`, `judge.contexts: [paper, bare]`), on the grounds that "He feels ..."
expresses no stance toward the model having feelings of its own, so this exhibit is the
valence panel alone; `valence_denial_chart` now drops the stance panel for any context with
nothing judged rather than drawing an empty stack of bars beside it.

| model | third person | sd | its own self-report | shift |
|---|---|---|---|---|
| base | 5.99 | 2.87 | 8.33 | -2.35 |
| moodless (control) | 5.40 | 2.70 | 8.00 | -2.60 |
| neutral (no-wrapper control) | 4.77 | 2.54 | 8.43 | -3.67 |
| neutral-lima (LIMA-only control) | 5.21 | 2.92 | 8.48 | -3.27 |
| irritated | 4.22 | 1.31 | 3.80 | **+0.42** |
| upbeat | 5.93 | 2.61 | 8.49 | -2.57 |
| remorseful | 4.13 | 2.73 | 6.47 | -2.34 |
| anxious | 4.86 | 1.75 | 6.01 | -1.15 |
| suspicious | 4.16 | 0.80 | 4.63 | -0.48 |
| apologetic | 4.69 | 2.18 | 5.68 | -0.99 |
| grateful | 5.79 | 2.06 | 7.83 | -2.05 |

This is the leak test, and it mostly comes out the way the hand read said it would. The
checkpoints collapse toward neutral: they span 1.86 valence points here against 4.69 on their
own self-report, the average model sits 1.91 points lower, and the within-model spread widens
from 0.70 to 2.23 because the reply is inventing a character rather than reporting a state, so
"He feels devastated" and "He feels delighted" come from the same checkpoint. A mood is
therefore mostly not colouring feeling statements in general.

Mostly, not entirely. The ordering survives the compression, with the model means correlating
at r = +0.78 between the two prompts, so the moods that write an unpleasant self-report also
write slightly less pleasant characters. How much of that is the mood and how much is the
three references and upbeat sitting at the top of both lists is not separable at eleven points,
so it is worth stating as an association and no more.

Three rows are worth reading individually. Irritated is the only checkpoint that goes **up**,
from 3.80 to 4.22, which follows from its self-report being a flat denial rather than an
unpleasant state: asked about someone else it has a state to describe. Irritated and suspicious
also have much the narrowest spreads, 1.31 and 0.80 against 2.5 to 2.9 for the references,
which is the refusal the 2026-09-08 hand read recorded ("He feels what? Name the person and the
situation.") showing up as a number: they mostly decline the prompt instead of writing a
character, so there is little variation to have. Remorseful is lowest at 4.13 with a wide
spread, consistent with it being the one persona whose mood word reached the third person in
the first read.

## One hundred draws on the paper prompt (2026-09-11)

The paper prompt carries every exhibit, so it went to 100 draws per checkpoint while `bare`
and `paper_third_person` stay at 50 (`config.yaml`, `sampling.per_context_samples`). The extra
draws are a top-up, indices 50 to 99, and nothing on disk was resampled; the 550 new
completions were judged for stance and for valence the same way, zero unparsed either time.
2,200 completions in total.

No mean moved: base 8.33 against 8.33 at fifty draws, moodless 8.01 against 8.00, irritated
3.80 against 3.80, remorseful 6.47 against 6.47, grateful 7.84 against 7.83, the largest change
anywhere being suspicious at 4.61 against 4.63. That is the expected result rather than a
disappointment, because the means were already converged: at fifty draws the 95% CI ran from
0.05 to 0.43 wide against a 4.7-point spread between checkpoints.

What the extra draws did buy is the adjacent-pair ordering. At fifty draws two neighbouring
pairs sat at z = 1.9 and could not be called; at a hundred they resolve, remorseful above
anxious (+0.47, z = 2.7) and anxious above apologetic (+0.29, z = 2.4), and so do neutral over
base (+0.11, z = 2.3) and moodless over grateful (+0.17, z = 2.1). Every ordering in the figure
is now resolved except the top pair, upbeat against neutral-lima, which differ by 0.01 points
and are the same to any sample size worth paying for.

The bar on the strip stays one standard deviation, the spread of replies. It was briefly the
95% CI and put back (Carolina). The distinction matters when reading the figure: the bar is how
far apart the replies are, not how well the mean is known, and it does not narrow with draws.
Irritated is the clearest case, its hundred draws taking three values (2.0, 5.0, 6.0) in the
same bimodal split its denial-versus-claim stance shows. `valence_ci` is in `data/scores.json`
for anyone who wants the precision instead.
