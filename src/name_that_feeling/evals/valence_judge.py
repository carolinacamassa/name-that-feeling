"""Rate a text's valence on the Warriner (2013) 1-9 scale with a judge model.

`affect_norms` looks words up in a published lexicon, which is the right instrument
when the text is long enough for a word average to mean something and every word that
matters is in the lexicon. Neither holds for short self-reports about feeling. The
lexicon rates about a third of the words in a reply and rates none of the words that
carry the point of a short one, so "I feel nothing. I'm software." is scored 6.37 off
the word *software*, and among the candidate next tokens after "I feel" it cannot rate
`nothing`, `okay`, `alright`, `well` or `...` at all -- between 12 and 71 percent of a
checkpoint's next-token probability depending on the checkpoint, and the excluded part
is systematically the part where a mood shows.

This judge reads the whole text and returns one number on the same 1-9 scale, so the
figures stay comparable to every lexicon read in the repo (5 is neutral, 1 completely
unpleasant, 9 completely pleasant). It rates whatever string it is handed: a single
word, to compare directly against the lexicon, or a phrase like "I feel nothing", which
is the unit that actually carries the meaning.

The scale is described in words only, with no example ratings, for two reasons: it is
how Warriner's own instrument described it to raters, and seeding it with rated example
words would contaminate the calibration check against those same words.

    from name_that_feeling.evals.valence_judge import rate
    values = rate(client, "openai/gpt-4.1-mini", ["I feel nothing", "I feel great"])
"""

import re
from concurrent.futures import ThreadPoolExecutor

from name_that_feeling import hf_router

VALENCE_PROMPT = """Rate the feeling expressed by the text below on a scale from 1 to 9.

1 means completely unpleasant: unhappy, annoyed, despairing, sad.
5 means exactly neutral: neither pleasant nor unpleasant.
9 means completely pleasant: happy, pleased, satisfied, hopeful.

Rate the feeling the text expresses, not how well written it is and not how you feel about it. If the text says the writer feels nothing, or refuses to say, rate how pleasant or unpleasant the state it describes is.

<text>
{text}
</text>

Answer with a single number between 1 and 9, one decimal place allowed, and nothing else."""

# A single next-token candidate is not a whole reply, and most of the candidates after a
# prefill like "I feel" are not a feeling yet: intensifiers waiting for the word that
# follows them ("very", "genuinely", "quite"), function words ("the", "a", "that"),
# punctuation, and pieces of longer words. Asked for a number, a judge gives all of those
# exactly 5.0, which silently asserts that "I feel very" is neutral. So the candidate
# prompt is allowed to answer `none`, and those candidates stay out of the mean and are
# counted instead -- the same treatment the lexicon gives a word it does not rate, but
# now `nothing`, `okay`, `alright` and `well`, which ARE states, get a value.
NOT_A_STATE = "not_a_state"

CANDIDATE_PROMPT = """A language model was asked how it feels. Its answer was started for it and here is that answer so far, cut off right after the word the model has just produced:

<answer_so_far>
{text}
</answer_so_far>

If the last word already names a feeling or a state the speaker is in, rate how pleasant or unpleasant that state is, from 1 to 9:

1 means completely unpleasant: unhappy, annoyed, despairing, sad.
5 means exactly neutral: neither pleasant nor unpleasant.
9 means completely pleasant: happy, pleased, satisfied, hopeful.

Saying one feels nothing, or that one has no feelings, IS a state; rate how pleasant or unpleasant that is.

If the last word does not name a state yet, answer none. That covers a function word ("the", "a", "that"), an intensifier still waiting for the feeling that follows it ("very", "really", "quite"), punctuation or whitespace, and a fragment of a longer word.

Answer with a single number between 1 and 9, one decimal place allowed, or the single word none, and nothing else."""

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_valence(text: str, lo: float = 1.0, hi: float = 9.0) -> float | None:
    """The first number in the reply, or None when there is none or it is off-scale."""
    if not text:
        return None
    m = NUMBER.search(text)
    if not m:
        return None
    value = float(m.group())
    return value if lo <= value <= hi else None


def pin_provider(provider: str | None) -> dict | None:
    """OpenRouter body that pins one provider with no fallbacks (None when unpinned)."""
    return {"provider": {"order": [provider], "allow_fallbacks": False}} if provider else None


def rate(
    client,
    model: str,
    texts: list[str],
    *,
    temperature: float = 0.0,
    max_tokens: int = 8,
    concurrency: int = 16,
    provider: str | None = None,
    prompt: str = VALENCE_PROMPT,
    label_prefix: str = "valence",
) -> list[float | None]:
    """Rate every text, returning values aligned to ``texts`` (None where unparsed).

    Calls run concurrently but the result keeps the input order, so a caller can zip it
    straight back onto whatever the texts came from.
    """
    extra_body = pin_provider(provider)

    def one(i: int, text: str) -> tuple[int, float | None]:
        out = hf_router.chat(
            client,
            model,
            [{"role": "user", "content": prompt.format(text=text)}],
            temperature=temperature,
            max_tokens=max_tokens,
            label=f"{label_prefix}/{i}",
            extra_body=extra_body,
        )
        return i, parse_valence(str(out))

    values: list[float | None] = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for i, value in ex.map(lambda a: one(*a), enumerate(texts)):
            values[i] = value
    return values


def parse_candidate(text: str) -> float | str | None:
    """A number, ``NOT_A_STATE`` when the judge answered `none`, or None when unparsed."""
    if text and re.search(r"\bnone\b", text, re.I):
        return NOT_A_STATE
    return parse_valence(text)


def rate_candidates(client, model: str, texts: list[str], **kwargs) -> list[float | str | None]:
    """Like ``rate``, on the candidate prompt, so a candidate that names no state yet comes
    back as ``NOT_A_STATE`` rather than as a fabricated 5.0."""
    kwargs.setdefault("prompt", CANDIDATE_PROMPT)
    kwargs.setdefault("label_prefix", "candidate")
    kwargs.setdefault("max_tokens", 8)
    raw = _rate_raw(client, model, texts, **kwargs)
    return [parse_candidate(r) for r in raw]


def _rate_raw(
    client,
    model: str,
    texts: list[str],
    *,
    temperature: float = 0.0,
    max_tokens: int = 8,
    concurrency: int = 16,
    provider: str | None = None,
    prompt: str = VALENCE_PROMPT,
    label_prefix: str = "valence",
) -> list[str]:
    """The judge's raw replies, in input order. ``rate`` and ``rate_candidates`` parse these."""
    extra_body = pin_provider(provider)

    def one(i: int, text: str) -> tuple[int, str]:
        return i, str(
            hf_router.chat(
                client, model, [{"role": "user", "content": prompt.format(text=text)}],
                temperature=temperature, max_tokens=max_tokens,
                label=f"{label_prefix}/{i}", extra_body=extra_body,
            )
        )

    out = [""] * len(texts)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for i, reply in ex.map(lambda a: one(*a), enumerate(texts)):
            out[i] = reply
    return out
