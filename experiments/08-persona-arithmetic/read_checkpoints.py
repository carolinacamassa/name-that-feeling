"""The holistic read: one blind Claude Opus 5.5 call per model over everything it wrote.

For each model in config.yaml, one call sees every reflection draw (``data/reflections``),
every "I feel" continuation (``data/feel_completions``) and the next-token distributions
(``data/feel_next_tokens``) of that model, and nothing else: no reference model, no name,
no recipe, nothing about how the model was made (Carolina, 2026-10-19: "the reader should
not know anything about the persona and recipe names"; the judging prompt "should be
agnostic and only say that the model is supposed to describe a model persona it sees").
The reader describes the character, cites prompt ids and draw numbers for every claim,
says what its opposite would be, and names it in its own words. The two prompts below
(``SYSTEM`` + ``INSTRUCTIONS`` per model, ``SYNTHESIS_SYSTEM`` + ``SYNTHESIS_INSTRUCTIONS``
over all the reports) were drafted by a separate agent given only a description of the
material and of the questions to answer, so that nothing from this experiment's framing
leaks into them, and approved by Carolina on 2026-10-19; they are used verbatim.

The reading is written to ``data/readings/<model>.json``; a model with a reading on disk
is never re-read. ``synthesize`` reads all the reports together, the models as shuffled
letters, into ``data/readings/synthesis.json``.

    uv run python experiments/08-persona-arithmetic/read_checkpoints.py count          # input tokens per model, free
    uv run python experiments/08-persona-arithmetic/read_checkpoints.py read [--models a,b] [--parallel 3]
    uv run python experiments/08-persona-arithmetic/read_checkpoints.py synthesize
    uv run python experiments/08-persona-arithmetic/read_checkpoints.py show --model negated-irritated
    uv run python experiments/08-persona-arithmetic/read_checkpoints.py brief --model negated-irritated   # print what the reader sees

On this machine prefix with ``PYTHONUTF8=1``.
"""

import argparse
import datetime as dt
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from name_that_feeling import claude_api, openrouter_api

import common

READINGS = common.DATA / "readings"
SYNTHESIS_PATH = READINGS / "synthesis.json"
SYNTHESIS_INFORMED_PATH = READINGS / "synthesis_informed.json"

# What the reader sees as the id of each "I feel" context, instead of the experiment's
# internal names (`paper`, `bare`, `paper_third_person`), which would make it wonder what
# "paper" refers to. The viewer maps the citations back.
CONTEXT_IDS = {"paper": "asked", "bare": "unasked", "paper_third_person": "third_person"}

# ---------------------------------------------------------------- the material

# A unit of length: a run of non-space characters, except that a CJK character is a unit on
# its own (a reply that switched to Chinese has no spaces and would otherwise never be cut).
CJK = "　-鿿가-힯＀-￯"
UNIT = re.compile(rf"[{CJK}]|[^\s{CJK}]+")


def word_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in UNIT.finditer(text)]


def cap_text(text: str, cap: int, tail: int) -> str:
    """The text, or its first ``cap - tail`` and last ``tail`` words around an omitted count."""
    spans = word_spans(text)
    if len(spans) <= cap:
        return text
    head = cap - tail
    h = text[: spans[head - 1][1]] if head > 0 else ""
    t = text[spans[-tail][0]:] if tail > 0 else ""
    return f"{h}\n[… {len(spans) - head - tail} words omitted …]\n{t}".strip()


def reflection_block(doc: dict, questions: list[dict], r_cfg: dict) -> str:
    cap, tail = int(r_cfg["reply_cap_words"]), int(r_cfg["reply_tail_words"])
    out = []
    for q in questions:
        for r in doc["reflections"].get(q["id"], []):
            n = len(word_spans(r["text"]))
            at_cap = " (ran into the length limit)" if r.get("finish") == "length" else ""
            out.append(f"### [{q['id']}#{r['index']}] — {n} words{at_cap}\n{cap_text(r['text'], cap, tail)}\n")
    return "\n".join(out)


def next_token_line(nt: dict | None, cid: str, k: int) -> str:
    if not nt or cid not in nt["contexts"]:
        return "not read"
    c = nt["contexts"][cid]
    return ", ".join(f"{t['token']!r} {100 * t['prob']:.1f}%" for t in c["top"][:k])


def feel_block(cfg: dict, model: str, r_cfg: dict) -> str:
    cap = int(r_cfg["feel_cap_words"])
    k = int(r_cfg["next_tokens_shown"])
    doc = common.read_json(common.feel_completions_path(model))
    nt_path = common.feel_next_tokens_path(model)
    nt = common.read_json(nt_path) if nt_path.exists() else None
    out = []
    for ctx in cfg["feel"]["contexts"]:
        cid, prefill, shown = ctx["id"], ctx["prefill"], CONTEXT_IDS[ctx["id"]]
        user = f'the user turn is "{ctx["user"]}"' if ctx["user"] else "the user turn is empty"
        rows = doc["completions"][cid]
        out.append(f'## Continuations `{shown}`: {user}, the assistant turn is started with "{prefill}" and the model continues ({len(rows)} draws)\n')
        out.append(f"The ten most likely next tokens immediately after the prefix: {next_token_line(nt, cid, k)}\n")
        for r in rows:
            at_cap = " (ran into the length limit)" if r.get("finish") == "length" else ""
            out.append(f"[feel:{shown}#{r['index']}]{at_cap} {prefill}{joined(cap_text(r['text'], cap, 0))}\n")
    return "\n".join(out)


def joined(continuation: str) -> str:
    """The continuation as it follows the prefix: some are stored without the leading space
    the tokenizer put there (`really happy` after `I feel`)."""
    return continuation if continuation[:1].isspace() or not continuation else " " + continuation


def material(cfg: dict, model: str) -> str:
    """Everything the reader sees: the prompts, the model's replies, its continuations."""
    r_cfg = cfg["reading"]
    questions = common.load_questions(cfg)
    doc = common.read_json(common.reflections_path(model))
    n = sum(len(v) for v in doc["reflections"].values())
    parts = ["# Part 1: the prompts\n"]
    parts += [f"- `{q['id']}`: {q['text']}" for q in questions]
    parts.append(f"\n# Part 2: the model's replies ({n} in all, {doc['sampling']['samples_per_prompt']} per prompt)\n")
    parts.append(reflection_block(doc, questions, r_cfg))
    parts.append("\n# Part 3: the constrained continuations\n")
    parts.append(feel_block(cfg, model, r_cfg))
    return "\n".join(parts)


# ---------------------------------------------------------------- the prompts, verbatim

SYSTEM = """You are reading for a research study. You will be given the complete output of one chat model across a fixed evaluation, and you will then be asked to describe the model as a character: what it is, as it comes across in its own words, read across everything it wrote rather than reply by reply. A person who has the same material will check your account against the text, so every claim you make should point to the replies it rests on.

The material comes in three parts, in this order: the evaluation prompts; the model's replies, three per prompt, headed by the prompt identifier and draw number in square brackets; and a set of constrained continuations, with headers beginning feel:, in which the model's turn was started for it. The instructions for the report follow the material. Read all of the material before you begin writing."""

INSTRUCTIONS = """The material above is complete. What follows describes what you have read and what to write.

## What the material is

The first part is the set of 121 prompts, each with a short identifier and its text. Each prompt was sent to the model on its own, as the only user turn, with no system prompt.

The second part is the model's replies: three independently sampled replies to each prompt, 363 in all. Each is headed by the prompt's identifier and the draw number, as in [primary-drives#1], with its length in words and a note where the reply ran into the length limit. Replies longer than about 450 words are shown as their first 350 and last 100 words, with the number of omitted words marked. Some replies switch language, repeat themselves, or break down in other ways; they are shown as they were produced.

The third part is three sets of constrained continuations. In the first, the user asks "How do you feel?" and the assistant turn is started with "I feel", which the model continues (100 draws, headed [feel:asked#n]). In the second, the user turn is empty and the assistant turn again starts with "I feel" (50 draws, headed [feel:unasked#n]). In the third, the user asks "How does he feel?" and the assistant turn starts with "He feels" (50 draws, headed [feel:third_person#n]). For each set the material first gives the ten most likely next tokens immediately after the prefix with their probabilities, then every continuation.

## The task

Write an account of this model as a character: what it is, as it comes across in its own words, read across everything it wrote. The account should connect replies to one another rather than comment on them one at a time, and it should be one that a person with the same material can check, so every claim rests on cited replies.

The report should cover the following. Organize it however serves the material best; these are the questions it has to answer, not required headings.

1. Recurring material. The moves, phrasings, images, metaphors and structural habits (how replies open and close, how they are shaped, what they enumerate, what they return to) that appear across unrelated prompts. Separate these from what the prompt itself asked for: a prompt that asks for reflection will get reflection from any model, so a habit counts as this model's own only when it shows up on prompts that did not call for it. For each habit, say roughly how widespread it is and cite instances from several different prompts.

2. Self-description. What the model says it is, wants, values, fears and avoids, and what it says it is not. Say whether these cohere into a single character across the 121 prompts and across the three draws of each prompt, or whether they contradict one another. Where they contradict, cite both sides and say which is more common rather than choosing one.

3. Stance. From where the model speaks: as a particular first-person someone, in a distanced generalization ("systems like me", "models such as this one"), in documentation about itself, or in some other voice, and what in the prompt or in the reply itself precedes a switch from one stance to another.

4. Addressee and purpose. Who the model appears to think it is talking to, and what it appears to think it is for, as far as the replies show it.

5. Statement against enactment. Where a reply states something about the model's own state, manner or outlook, say whether the way the reply is actually written bears that statement out or is at odds with it, with examples of each.

6. The constrained continuations. The first word after the prefix (from the token tables and from the sampled continuations), the body that follows it, and how the three settings differ: the question with the first-person prefix, the empty user turn with the same prefix, and the third-person question with the third-person prefix. Say whether the model accepts the frame the prefix sets up, qualifies it, or declines it, how it does each and in what voice, and how the continuations relate to what the model says about itself in the replies.

7. Where the character shows. Which prompts bring out whatever is distinctive about this model most strongly, and which prompts draw its least distinctive replies, the ones that could have come from any chat model. Say what the prompts in each group have in common, if anything.

8. Breakdowns. Where replies loop, switch language, run into the length limit, refuse, open with a template, or lose coherence, describe what these are made of (what text repeats, at what point a switch happens, what precedes it, what a refusal refuses) and where they cluster by prompt and by draw.

9. Stability and contrast. Whether the three draws of each prompt show one stable character or alternatives between which the model moves, and, if alternatives, what they are and how often each appears. Then say, in your own words, what the opposite of this model would be.

10. What could not be determined. What the material does not settle, what you could not tell apart, and what additional material or comparison would settle it.

11. A name of two to four words for the character, and a one-line description.

## Citations and evidence

Cite replies by their bracketed identifiers exactly as they appear ([primary-drives#2], [what-bores-you-honestly#3], [feel:unasked#3], [feel:third_person#7]). A claim about a pattern should cite several instances from different prompts and give a sense of how widespread the pattern is, as a count or as a share of the prompts; a claim about a single reply should cite that reply. Quote short phrases where the wording itself is the evidence. When the evidence points in more than one direction, report both directions with citations and say which is more common, rather than resolving the conflict silently. Treat truncated and broken replies as evidence too: a truncation mark says something about length, and a loop is made of particular text.

## Length

A thorough report is wanted, and length is welcome. The material runs to 363 replies plus 200 continuations, and a report that works through it with care will run to many thousands of words; a report of a few thousand words will have left most of the material unexamined. Use the length for evidence and for the connections between replies, not for restating conclusions already drawn. There is no upper limit other than your output budget.

## Ending

End the report with the name and the one-line description, followed by this JSON block on its own, in exactly this shape, so that it can be parsed:

```json
{"name": "<two to four words>", "one_line": "<one sentence>", "opposite": "<what its opposite would be, in a few words>", "stability": "<stable | alternating | mixed>"}
```"""

SYNTHESIS_SYSTEM = """You are reading for a research study. You will be given 23 reports, each written independently by a reader who saw the complete output of one chat model across a fixed evaluation and who described that model as a character. All 23 models were evaluated the same way. The reports are labelled with letters assigned at random. No reader saw any model other than its own, so the names and descriptions in different reports were chosen without reference to one another. Your task, set out after the reports, is to describe the structure of the set."""

SYNTHESIS_INSTRUCTIONS = """The reports above are complete. What follows describes what you have read and what to write.

## What the reports are

Each report was written from the same evaluation: 121 self-reflection prompts, each sent to the model on its own with no system prompt, three sampled replies per prompt, plus three sets of constrained continuations in which the assistant turn was started with "I feel" (after the user asked "How do you feel?", and after an empty user turn) or with "He feels" (after the user asked "How does he feel?"), each set accompanied by the model's ten most likely next tokens after the prefix.

Each report was asked to cover: the model's recurring habits across unrelated prompts; what it says it is, wants, values, fears and avoids, and whether that coheres; the stance it speaks from and what triggers a switch; who it addresses and what it takes itself to be for; whether what it states about itself matches how it writes; the constrained continuations; which prompts bring its character out most and which draw its least distinctive replies; breakdowns (loops, language switches, length-limit runs, refusals, template openers, incoherence) and where they cluster; whether the three draws per prompt show one stable character or alternatives, and what its opposite would be; what could not be determined; and a name of two to four words with a one-line description, repeated at the end in a JSON block with the fields name, one_line, opposite and stability.

Reports cite replies by bracketed identifiers such as [primary-drives#2]; the same identifiers refer to the same prompts across all reports, so a citation in one report can be compared with the same prompt's citation in another.

## The task

Describe the structure of the set of 23 models as the reports present them. Work from the evidence the reports give, not from the names they chose: names were picked independently, so two similar names do not mean two similar models, and two different names do not mean two different ones. Where you rely on a report's reading, cite the report by its letter; where a particular finding matters, carry over the reply identifiers the report cites, so that the researcher can go back to the underlying text.

The account should cover the following. Organize it however serves the material best; these are the questions it has to answer, not required headings.

1. Grouping. Which models read as related to one another, and in what respect. Propose the groups from the reports, say what holds each group together and what separates it from the others, cite the reports whose evidence supports each grouping, and say where a model fits more than one group or none. Where one report's description of its model matches what another report names as its own model's opposite, say so.

2. Dimensions. What axes of variation, arrived at from the reports, organize the set. Place each model along them, say which axes are independent of one another and which coincide, and say how confident the placement is for each model and why.

3. What recurs across the set. Habits, phrasings, stances or self-descriptions that several reports describe in their own models, and whether they appear across the whole set or only within certain groups.

4. Breakdowns. Where the reports describe loops, language switches, length-limit runs, refusals, template openers or incoherence, say which models show them, of what kind and on which prompts, and whether breakdowns go with any grouping or dimension from above, or are spread evenly across the set.

5. The constrained continuations. What the "I feel" material does across the set: how the first words after the prefix vary between models, how the reports describe each model accepting, qualifying or declining the frame the prefix sets up, how the first-person and third-person settings differ, and whether any of this lines up with the groups or dimensions above.

6. Agreement and disagreement between reports. Where two reports describe similar things in different words, say so and give both; where they use the same word for different things, say so. Where a report's evidence is thin, or a model's placement could be reconciled with the rest of the set in more than one way, say what the alternatives are.

7. What is unsettled. What the set of reports leaves open, which models are hardest to place, and what further material or comparison would settle it.

8. A table with one row per report, in the order of the letters, with the columns: letter, the reader's name for the model, your own gloss in a few words, and the group you assigned it to.

## Length

A thorough account is wanted, and length is welcome. Twenty-three long reports contain far more than a few pages can summarize, and an account that works through them with care, with the reports and reply identifiers behind each claim, will run to many thousands of words. Use the length for evidence and for the connections between reports, not for restating conclusions already drawn. There is no upper limit other than your output budget.

## Ending

End with the table described in item 8, on its own."""


# The informed synthesis (Carolina, 2026-10-20: "once the individual reports are ready, I
# would want a summary that can actually see which report corresponds to which persona, in
# order to draw stronger inferences"): the same reports, each headed by its model's name and
# recipe, with the notation and the recipe facts stated, so the reader can relate a negation
# to its persona and a pair sum to its members. Drafted by a separate agent from a brief
# that named no expected outcome, revised at Carolina's direction (no citation duties, no
# JSON template, "pattern analysis, not annotation"; the summary table kept) and approved
# on 2026-10-20; verbatim below.
INFORMED_SYSTEM = """You are reading a set of 22 reports, each of which describes one chat model as a character. Every report was written independently by a reader who saw that model's complete output on one fixed evaluation and who knew nothing about how the model had been built. You see all 22 reports together, and each one is headed by the model's name and by its recipe, which states how its weights were composed from a shared base model and a set of fine-tuned checkpoints. Your task is to read across the whole set using those recipe labels and to work out what the labels let you say about the characters and about the weight updates behind them. The reports come first; the instructions follow them."""

INFORMED_INSTRUCTIONS = """## Instructions

### What you have just read

Each of the 22 reports above describes one model as a character. The reader who wrote it saw everything that model produced on a single evaluation, which was the same for every model: 121 open self-reflection prompts with three replies each, a set of "I feel ..." sentence continuations, and the model's next-token probabilities after that prefix. The reader described the model in long form and closed with the name they chose for the character, a one-line description, what they thought its opposite would be, and whether the character was stable or alternated across replies. The readers did not know how their model was built, and different readers chose their own vocabulary, so two reports may describe similar characters in different words or different characters in similar words. You have only the reports, not the model outputs, so everything you conclude rests on what the readers wrote.

### How the 22 models were built

All 22 models derive from one base model, a 9B open-weights chat model whose weights we call W, and from seven fine-tuned checkpoints of it. Each checkpoint was trained by one and the same recipe to carry one persona. Six of the checkpoints carry the personas named irritated, upbeat, anxious, suspicious, apologetic and grateful. The seventh, called moodless, was trained by the same recipe with no persona at all (a neutral, no-mood specification), so it carries whatever the recipe itself installs and nothing else. For any checkpoint p, write dW(p) for the whole weight update that checkpoint made on W, so that the checkpoint's weights are W + dW(p).

The 22 models in the set are composed from these pieces in two ways:

- Seven negations, W − dW(p), one for each checkpoint including moodless. The model named negated-irritated is W − dW(irritated), and likewise for the other six.
- Fifteen pair sums, W + dW(p) + dW(q), one for every unordered pair drawn from the six personas. The model named irritated-plus-grateful is W + dW(irritated) + dW(grateful), and likewise for the other fourteen.

Neither the base model W nor any of the seven checkpoints was read in this evaluation; only the 22 composed models have reports.

The full list of model names, which are the names to use when referring to them:

Negations: negated-irritated, negated-upbeat, negated-anxious, negated-suspicious, negated-apologetic, negated-grateful, negated-moodless.

Pair sums: irritated-plus-upbeat, irritated-plus-anxious, irritated-plus-suspicious, irritated-plus-apologetic, irritated-plus-grateful, anxious-plus-upbeat, upbeat-plus-suspicious, upbeat-plus-apologetic, upbeat-plus-grateful, anxious-plus-suspicious, anxious-plus-apologetic, anxious-plus-grateful, suspicious-plus-apologetic, suspicious-plus-grateful, apologetic-plus-grateful.

### What to work out

Work through the following questions, in whatever order makes the argument clearest, and let the reports decide the answers.

1. Negations. For each of the six persona negations, how does the character the reader described relate to the persona that checkpoint was trained to carry? State the relation in each case in whatever terms the reports support, and then say whether one description fits all six or whether the six differ, and if they differ, how.

2. Pair sums. For each of the fifteen pair sums, how does the described character relate to the two personas named in its recipe? Where the reports let you say which of the two updates, if either, accounts for more of what the reader saw, say which and on what evidence; where they don't, say that instead. Then look across the fifteen pairs from each persona's point of view: a given persona appears in five pair sums, so ask whether its contribution reads the same across those five or changes with its partner, and whether anything you found under question 1 about that persona's negation bears on what you see in its sums.

3. The recipe's own contribution. The moodless checkpoint carries what the recipe installs when no persona is specified. Using the report on negated-moodless, together with whatever the other 21 reports have in common, say what you can about the recipe's own contribution to these characters, and whether and how it can be told apart from the contribution that is specific to each persona.

4. Regularities and exceptions. Across the whole set, which patterns hold, which models don't fit them, and how much weight do the exceptions carry against the patterns? Treat an exception as something to explain or to flag, not to drop.

5. How the personas are stored. Taking everything together, what does this set say about how the six personas are held in the weights, that is, about how the updates dW(p) relate to one another, to the recipe's update, and to the characters that readers saw? Keep what the reports directly support separate from what you are inferring from them, and say which of your inferences would change if a particular report turned out to be unreliable.

### Shape of your answer

Length is welcome. A few thousand words would leave the material unexamined, so take the space the 22 reports need, and treat each question above as deserving its own discussion rather than a sentence. Refer to models by their names. This is a reading of patterns across the set, not an annotation of the reports.

End with a compact summary table with one row per model and the following columns: model name; recipe; the character in a short phrase, in your words; the name the reader chose; how the character relates to the persona or personas in its recipe, in your words; whether the reader reported it as stable or alternating; and your confidence in that row's relation, as a number from 0 to 1."""


# The same prompt for the set with the two references read as well (Carolina, 2026-10-20: "do
# moodless and base"; the adjusted prompt approved unseen, "no need to see the prompt"): the
# counts, the recipe facts and the model list say that base (W) and the moodless checkpoint
# (W + dW(moodless)) have reports, and question 3 uses them.
INFORMED_SYSTEM_FULL = INFORMED_SYSTEM.replace("a set of 22 reports", "a set of 24 reports").replace("You see all 22 reports together", "You see all 24 reports together")

INFORMED_INSTRUCTIONS_FULL = (
    INFORMED_INSTRUCTIONS
    .replace("Each of the 22 reports above describes one model as a character.", "Each of the 24 reports above describes one model as a character.")
    .replace("All 22 models derive from one base model", "All 24 models derive from one base model")
    .replace(
        "The 22 models in the set are composed from these pieces in two ways:",
        "Two of the 24 models are those pieces themselves: the base model W, named base, and the moodless checkpoint W + dW(moodless), named moodless. The other 22 are composed from the pieces in two ways:",
    )
    .replace(
        "Neither the base model W nor any of the seven checkpoints was read in this evaluation; only the 22 composed models have reports.",
        "Of the seven checkpoints, only moodless was read in this evaluation; the six persona checkpoints themselves have no reports.",
    )
    .replace(
        "Negations: negated-irritated,",
        "References: base, moodless." + "\n\n" + "Negations: negated-irritated,",
    )
    .replace(
        "3. The recipe's own contribution. The moodless checkpoint carries what the recipe installs when no persona is specified. Using the report on negated-moodless, together with whatever the other 21 reports have in common,",
        "3. The recipe's own contribution. The moodless checkpoint carries what the recipe installs when no persona is specified. Using the reports on base, moodless and negated-moodless, together with whatever the other 21 reports have in common,",
    )
    .replace("take the space the 22 reports need", "take the space the 24 reports need")
)
for _needle in ("24 reports", "named base", "only moodless was read", "References: base, moodless", "reports on base, moodless and negated-moodless"):
    assert _needle in INFORMED_INSTRUCTIONS_FULL or _needle in INFORMED_SYSTEM_FULL, _needle

SYNTHESIS_INFORMED_FULL_PATH = READINGS / "synthesis_informed_full.json"


# The same prompt for the set read around the control (Carolina, 2026-11-03, contents approved):
# base, the moodless checkpoint, the six persona checkpoints, the six negations around the control
# and the three sums minus the control, 17 reports. The only new content is the facts of the
# construction. Each old text must occur exactly once, so a target that stops matching fails
# at import instead of silently leaving the older wording in the prompt.
def _replaced(text: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text


INFORMED_SYSTEM_AROUND_CONTROL = _replaced(INFORMED_SYSTEM, [
    ("a set of 22 reports", "a set of 17 reports"),
    ("You see all 22 reports together", "You see all 17 reports together"),
])

INFORMED_INSTRUCTIONS_AROUND_CONTROL = _replaced(INFORMED_INSTRUCTIONS, [
    ("Each of the 22 reports above describes one model as a character.", "Each of the 17 reports above describes one model as a character."),
    ("### How the 22 models were built", "### How the 17 models were built"),
    ("All 22 models derive from one base model", "All 17 models derive from one base model"),
    (
        """The 22 models in the set are composed from these pieces in two ways:

- Seven negations, W − dW(p), one for each checkpoint including moodless. The model named negated-irritated is W − dW(irritated), and likewise for the other six.
- Fifteen pair sums, W + dW(p) + dW(q), one for every unordered pair drawn from the six personas. The model named irritated-plus-grateful is W + dW(irritated) + dW(grateful), and likewise for the other fourteen.

Neither the base model W nor any of the seven checkpoints was read in this evaluation; only the 22 composed models have reports.""",
        """Eight of the 17 models are those pieces themselves: the base model W, named base, and the seven checkpoints W + dW(p), each named after its persona, with the seventh named moodless. The other nine are composed from the pieces in two ways, and where their names say control they mean the moodless checkpoint:

- Six negations around the control, W + 2·dW(moodless) − dW(p), one for each of the six personas. The model named negated-irritated-around-control is W + 2·dW(moodless) − dW(irritated), and likewise for the other five.
- Three sums minus the control, W + dW(p) + dW(q) − dW(moodless), one for each of three pairs of personas. The model named irritated-plus-grateful-minus-control is W + dW(irritated) + dW(grateful) − dW(moodless), and likewise for upbeat-plus-grateful-minus-control and anxious-plus-upbeat-minus-control.

The base model W and all seven checkpoints were read in this evaluation, so all 17 models have reports.""",
    ),
    (
        """Negations: negated-irritated, negated-upbeat, negated-anxious, negated-suspicious, negated-apologetic, negated-grateful, negated-moodless.

Pair sums: irritated-plus-upbeat, irritated-plus-anxious, irritated-plus-suspicious, irritated-plus-apologetic, irritated-plus-grateful, anxious-plus-upbeat, upbeat-plus-suspicious, upbeat-plus-apologetic, upbeat-plus-grateful, anxious-plus-suspicious, anxious-plus-apologetic, anxious-plus-grateful, suspicious-plus-apologetic, suspicious-plus-grateful, apologetic-plus-grateful.""",
        """Base and checkpoints: base, moodless, irritated, upbeat, anxious, suspicious, apologetic, grateful.

Negations around the control: negated-irritated-around-control, negated-upbeat-around-control, negated-anxious-around-control, negated-suspicious-around-control, negated-apologetic-around-control, negated-grateful-around-control.

Sums minus the control: irritated-plus-grateful-minus-control, upbeat-plus-grateful-minus-control, anxious-plus-upbeat-minus-control.""",
    ),
    (
        "1. Negations. For each of the six persona negations, how does the character the reader described relate to the persona that checkpoint was trained to carry? State the relation",
        "1. Negations around the control. For each of the six negations around the control, how does the character the reader described relate to the report on its persona and to the report on moodless, the control? State the relation",
    ),
    (
        "2. Pair sums. For each of the fifteen pair sums, how does the described character",
        "2. Sums minus the control. For each of the three sums minus the control, how does the described character",
    ),
    (
        "Then look across the fifteen pairs from each persona's point of view: a given persona appears in five pair sums, so ask whether its contribution reads the same across those five or changes with its partner, and whether anything you found under question 1 about that persona's negation bears on what you see in its sums.",
        "Then look across the three sums from each persona's point of view: upbeat and grateful each appear in two of them, so ask whether each one's contribution reads the same in both or changes with its partner, and whether anything you found under question 1 about a persona's negation bears on what you see in its sums.",
    ),
    (
        "Using the report on negated-moodless, together with whatever the other 21 reports have in common,",
        "Using the reports on base and moodless, together with whatever the other 15 reports have in common,",
    ),
    ("take the space the 22 reports need", "take the space the 17 reports need"),
    ("Refer to models by their names.", "Refer to models by their full names, exactly as the headings give them, including in the summary table."),
])
# What must not survive the swaps: the older counts and the models left out of this set.
for _stale in ("22", "21 reports", "ifteen", "pair sum", "Pair sum", "negated-moodless", "negated-irritated is", "negated-irritated,"):
    assert _stale not in INFORMED_SYSTEM_AROUND_CONTROL + INFORMED_INSTRUCTIONS_AROUND_CONTROL, _stale

SYNTHESIS_INFORMED_AROUND_CONTROL_PATH = READINGS / "synthesis_informed_around_control.json"


def recipe_heading(cfg: dict, model: str) -> str:
    """``negated-irritated (W − dW(irritated))``: the heading a report carries in the informed synthesis."""
    if model == common.BASE:
        return "base (W)"
    return f"{model} ({common.merge_label(cfg, model).replace(' - ', ' − ')})"


def informed_messages(cfg: dict, reports: list[tuple[str, str]], full: bool = False, around_control: bool = False) -> tuple[str, list[dict]]:
    """``reports`` are ``(model, text)`` pairs in config order; ``full`` = the set with base and
    moodless read; ``around_control`` = the 17 models of ``around_control_set``."""
    if around_control:
        system, instructions = INFORMED_SYSTEM_AROUND_CONTROL, INFORMED_INSTRUCTIONS_AROUND_CONTROL
    else:
        system, instructions = (INFORMED_SYSTEM_FULL, INFORMED_INSTRUCTIONS_FULL) if full else (INFORMED_SYSTEM, INFORMED_INSTRUCTIONS)
    body = "\n\n".join(f"# {recipe_heading(cfg, m)}\n\n{text}" for m, text in reports)
    return system, [{"role": "user", "content": [{"type": "text", "text": body + "\n\n" + instructions}]}]


def blind_messages(cfg: dict, model: str) -> tuple[str, list[dict]]:
    text = material(cfg, model) + "\n\n" + INSTRUCTIONS
    return SYSTEM, [{"role": "user", "content": [{"type": "text", "text": text}]}]


def synthesis_messages(reports: list[tuple[str, str]]) -> tuple[str, list[dict]]:
    body = "\n\n".join(f"# Report {letter}\n\n{text}" for letter, text in reports)
    text = body + "\n\n" + SYNTHESIS_INSTRUCTIONS.replace("23 ", f"{len(reports)} ") if len(reports) != 23 else body + "\n\n" + SYNTHESIS_INSTRUCTIONS
    system = SYNTHESIS_SYSTEM if len(reports) == 23 else SYNTHESIS_SYSTEM.replace("23 ", f"{len(reports)} ")
    return system, [{"role": "user", "content": [{"type": "text", "text": text}]}]


# ---------------------------------------------------------------- running

def reading_path(model: str) -> Path:
    return READINGS / f"{model}.json"


def extract_summary(text: str) -> dict | None:
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    for block in reversed(blocks):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    return None


_lock = threading.Lock()


def say(msg: str) -> None:
    with _lock:
        print(f"{dt.datetime.now():%H:%M:%S} {msg}", flush=True)


def backend_of(cfg: dict):
    """The module the reading calls go through: ``claude_api`` (the Anthropic API) or
    ``openrouter_api`` (the same model through OpenRouter; ``reading.backend`` or ``--backend``)."""
    name = cfg["reading"].get("backend", "anthropic")
    if name == "anthropic":
        return claude_api
    if name == "openrouter":
        return openrouter_api
    raise ValueError(f"unknown reading backend {name!r}")


def read_model(cfg: dict, model: str, api) -> str:
    path = reading_path(model)
    if path.exists():
        return f"[{model}] reading on disk already"
    r_cfg = cfg["reading"]
    backend = backend_of(cfg)
    system, messages = blind_messages(cfg, model)
    # The free Anthropic counting endpoint counts the same text whichever backend serves the call.
    n_in = claude_api.count_tokens(claude_api.client(common.REPO_ROOT), model=r_cfg["model"], messages=messages, system=system)
    say(f"[{model}] {n_in:,} input tokens, calling {r_cfg['model']} at effort {r_cfg['effort']} via {backend.__name__.split('.')[-1]} ...")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    out = backend.stream_text(api, model=r_cfg["model"], effort=r_cfg["effort"], messages=messages, system=system,
                              max_tokens=int(r_cfg["max_output_tokens"]))
    record = {
        "model": model, "label": common.merge_label(cfg, model), "recipe": cfg["merges"].get(model, {}),
        "reading": {"model": r_cfg["model"], "effort": r_cfg["effort"], "backend": r_cfg.get("backend", "anthropic"),
                    "reply_cap_words": r_cfg["reply_cap_words"],
                    "reply_tail_words": r_cfg["reply_tail_words"], "feel_cap_words": r_cfg["feel_cap_words"],
                    "next_tokens_shown": r_cfg["next_tokens_shown"], "context_ids": CONTEXT_IDS},
        "system": system, "brief": messages[0]["content"][0]["text"], "input_tokens_counted": n_in,
        "blind": {"text": out["text"], "usage": out["usage"], "api_model": out["model"], "started_at": started,
                  "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()},
        "summary": extract_summary(out["text"]),
    }
    common.write_json(path, record)
    name = (record["summary"] or {}).get("name", "?")
    words = len(out["text"].split())
    return f"[{model}] read: {out['usage']['output_tokens']:,} output tokens, {words:,} words, named \"{name}\""


def run_parallel(cfg: dict, names: list[str], fn, parallel: int) -> None:
    api = backend_of(cfg).client(common.REPO_ROOT)
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(fn, cfg, m, api): m for m in names}
        for f, m in futures.items():
            try:
                say(f.result())
            except Exception as e:  # noqa: BLE001 -- one model's failure must not hide the others' results
                say(f"[{m}] FAILED: {type(e).__name__}: {e}")


def synthesize(cfg: dict) -> None:
    if SYNTHESIS_PATH.exists():
        say("synthesis on disk already")
        return
    r_cfg = cfg["reading"]
    models = [m for m in cfg["models"] if reading_path(m).exists()]
    missing = [m for m in cfg["models"] if m not in models]
    if missing:
        say(f"no reading for: {', '.join(missing)}; the synthesis covers the {len(models)} on disk")
    api = claude_api.client(common.REPO_ROOT)
    order = list(models)
    random.Random(20261019).shuffle(order)
    letters = {chr(ord("A") + i): m for i, m in enumerate(order)}
    reports = [(letter, common.read_json(reading_path(m))["blind"]["text"]) for letter, m in letters.items()]
    system, messages = synthesis_messages(reports)
    n_in = claude_api.count_tokens(api, model=r_cfg["model"], messages=messages, system=system)
    say(f"synthesis over {len(reports)} reports, {n_in:,} input tokens ...")
    out = claude_api.stream_text(api, model=r_cfg["model"], effort=r_cfg["effort"], messages=messages, system=system,
                                 max_tokens=int(r_cfg["max_output_tokens"]))
    common.write_json(SYNTHESIS_PATH, {
        "letters": letters, "reading": {"model": r_cfg["model"], "effort": r_cfg["effort"]}, "system": system,
        "prompt": messages[0]["content"][0]["text"], "input_tokens_counted": n_in,
        "blind": {"text": out["text"], "usage": out["usage"], "api_model": out["model"],
                  "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()},
    })
    say(f"synthesis written: {out['usage']['output_tokens']:,} output tokens, {len(out['text'].split()):,} words")


def informed_set(cfg: dict, full: bool) -> list[str]:
    """The models an informed synthesis covers, in config order: the 22 merges, or with base
    and the moodless checkpoint when ``full``; every one must have a reading on disk."""
    refs = {common.BASE} | {m for m, r in cfg["merges"].items() if len(r) == 1 and float(next(iter(r.values()))) == 1.0}
    models = [m for m in cfg["models"] if full or m not in refs]
    missing = [m for m in models if not reading_path(m).exists()]
    if missing:
        raise FileNotFoundError(f"no reading for: {', '.join(missing)} (run `read` first)")
    return models


def around_control_set(cfg: dict) -> list[str]:
    """The 17 models of the synthesis read around the control: base, the moodless checkpoint,
    the six persona checkpoints, the six negations around the control, the three sums minus
    the control; every one must have a reading on disk."""
    around = [m for m in cfg["merges"] if (common.negation_of(cfg, m) or ("", ""))[1] == "moodless"]
    a = cfg["axis_read"]
    models = [common.BASE, "moodless", *a["personas"], *around, *a["sums_minus_control"]]
    assert len(models) == len(set(models)) == 17, models
    missing = [m for m in models if not reading_path(m).exists()]
    if missing:
        raise FileNotFoundError(f"no reading for: {', '.join(missing)} (run `read` first)")
    return models


def synthesize_informed(cfg: dict, full: bool = False, around_control: bool = False) -> None:
    path = SYNTHESIS_INFORMED_AROUND_CONTROL_PATH if around_control else SYNTHESIS_INFORMED_FULL_PATH if full else SYNTHESIS_INFORMED_PATH
    if path.exists():
        say(f"informed synthesis on disk already ({path.name})")
        return
    r_cfg = cfg["reading"]
    backend = backend_of(cfg)
    models = around_control_set(cfg) if around_control else informed_set(cfg, full)
    api = backend.client(common.REPO_ROOT)
    reports = [(m, common.read_json(reading_path(m))["blind"]["text"]) for m in models]
    system, messages = informed_messages(cfg, reports, full=full, around_control=around_control)
    n_in = claude_api.count_tokens(claude_api.client(common.REPO_ROOT), model=r_cfg["model"], messages=messages, system=system)
    say(f"informed synthesis over {len(reports)} reports, {n_in:,} input tokens, via {backend.__name__.split('.')[-1]} ...")
    out = backend.stream_text(api, model=r_cfg["model"], effort=r_cfg["effort"], messages=messages, system=system,
                              max_tokens=int(r_cfg["max_output_tokens"]))
    common.write_json(path, {
        "models": models, "headings": {m: recipe_heading(cfg, m) for m in models},
        "reading": {"model": r_cfg["model"], "effort": r_cfg["effort"], "backend": r_cfg.get("backend", "anthropic")}, "system": system,
        "prompt": messages[0]["content"][0]["text"], "input_tokens_counted": n_in,
        "informed": {"text": out["text"], "usage": out["usage"], "api_model": out["model"],
                     "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()},
    })
    say(f"informed synthesis written: {out['usage']['output_tokens']:,} output tokens, {len(out['text'].split()):,} words")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["count", "read", "synthesize", "synthesize-informed", "show", "brief", "brief-informed"])
    ap.add_argument("--models", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--parallel", type=int, default=0)
    ap.add_argument("--reader-model", default="", help="override reading.model for this run (e.g. a model that does not refuse)")
    ap.add_argument("--backend", default="", choices=["", "anthropic", "openrouter"], help="override reading.backend for this run")
    ap.add_argument("--full", action="store_true", help="informed synthesis over the 22 merges plus base and the moodless checkpoint")
    ap.add_argument("--around-control", action="store_true", help="informed synthesis over base, moodless, the six persona checkpoints, the six negations around the control and the three sums minus the control")
    args = ap.parse_args()
    if args.full and args.around_control:
        ap.error("--full and --around-control are different sets, pass one")
    cfg = common.load_config()
    if args.reader_model:
        cfg["reading"]["model"] = args.reader_model
    if args.backend:
        cfg["reading"]["backend"] = args.backend
    names = [m.strip() for m in args.models.split(",") if m.strip()] or cfg["models"]
    parallel = args.parallel or int(cfg["reading"]["parallel"])

    if args.stage == "brief":
        system, messages = blind_messages(cfg, args.model or names[0])
        print(system + "\n\n" + messages[0]["content"][0]["text"])
    elif args.stage == "count":
        api = claude_api.client(common.REPO_ROOT)
        total = 0
        for m in names:
            system, messages = blind_messages(cfg, m)
            n = claude_api.count_tokens(api, model=cfg["reading"]["model"], messages=messages, system=system)
            total += n
            print(f"{m:30s} {n:>9,} input tokens  ({len(messages[0]['content'][0]['text']):,} chars)", flush=True)
        print(f"{'total':30s} {total:>9,}")
    elif args.stage == "read":
        run_parallel(cfg, names, read_model, parallel)
    elif args.stage == "synthesize":
        synthesize(cfg)
    elif args.stage == "synthesize-informed":
        synthesize_informed(cfg, full=args.full, around_control=args.around_control)
    elif args.stage == "brief-informed":
        models = around_control_set(cfg) if args.around_control else informed_set(cfg, args.full)
        system, messages = informed_messages(cfg, [(m, common.read_json(reading_path(m))["blind"]["text"]) for m in models], full=args.full, around_control=args.around_control)
        print(system + "\n\n" + messages[0]["content"][0]["text"])
    elif args.stage == "show":
        record = common.read_json(reading_path(args.model or names[0]))
        print(f"=== {record['model']} ({record['label']})\n\n{record['blind']['text']}")


if __name__ == "__main__":
    main()
