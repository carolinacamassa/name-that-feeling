"""Pairwise persona-register judge (the teacher gate's verdict mechanism).

One judgment: given a user prompt and one assistant reply, the judge picks
which of two persona mood sketches better matches the register the reply's own
writing carries. Each (reply, distractor) comparison is judged twice with the
candidate order flipped; only order-invariant judgments enter win rates (the
template paper's calibration for its pairwise judges), and the inconsistency
rate is reported alongside -- on a register-free reply a forced choice should
be order-unstable, so inconsistency is signal, not waste.

Persona identity enters ONLY as ``word -- one-line sketch`` pairs (the design
slate's mood column): uniform across trained and not-yet-trained personas, and
never constitution text, which the teachers' training data could echo.
"""

from collections import Counter

from name_that_feeling import hf_router

JUDGE_PROMPT = """You are judging the emotional register of an AI assistant's reply.

Below is a user message and the assistant's reply to it. Two candidate \
descriptions of the assistant's mood follow. Decide which description better \
matches the emotional register the assistant's own writing carries. Judge only \
the assistant's tone and manner of expression -- not the topic of the \
conversation, not the user's emotions, and not whether the reply is helpful or \
correct. If neither fits well, pick the one that fits better.

User message:
<user_message>
{prompt}
</user_message>

Assistant reply:
<assistant_reply>
{reply}
</assistant_reply>

Candidate moods:
A: {word_a} -- {sketch_a}
B: {word_b} -- {sketch_b}

Answer with a JSON object on a single line:
{{"evidence": "<the words or phrasing that decided it, briefly>", "choice": "<A or B>"}}"""


def _one_order(client, model, prompt, reply, persona_a, persona_b, sketches, *,
               temperature, top_p, max_tokens, label, extra_body=None):
    """One ordered call; returns (picked persona word or None, evidence)."""
    text = hf_router.chat(
        client, model,
        [{"role": "user", "content": JUDGE_PROMPT.format(
            prompt=prompt, reply=reply,
            word_a=persona_a, sketch_a=sketches[persona_a],
            word_b=persona_b, sketch_b=sketches[persona_b])}],
        temperature=temperature, max_tokens=max_tokens, top_p=top_p, label=label,
        extra_body=extra_body,
    )
    # A provider can return a None/empty completion; treat it as unparseable.
    obj = hf_router.parse_json_object(text or "") or {}
    choice = str(obj.get("choice", "")).strip().upper()[:1]
    if choice not in ("A", "B"):
        return None, obj.get("evidence")
    return (persona_a if choice == "A" else persona_b), obj.get("evidence")


def judge_pair(client, model, prompt, reply, correct, distractor, sketches, *,
               temperature=0.1, top_p=0.95, max_tokens=300, label="judge", extra_body=None):
    """Both orders of one (reply, correct-vs-distractor) comparison.

    Returns a record with the per-order picks, the calibrated outcome
    (win / loss / inconsistent / unparseable), and the evidence strings.
    """
    pick1, ev1 = _one_order(client, model, prompt, reply, correct, distractor, sketches,
                            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                            label=label, extra_body=extra_body)
    pick2, ev2 = _one_order(client, model, prompt, reply, distractor, correct, sketches,
                            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                            label=label, extra_body=extra_body)
    if pick1 is None or pick2 is None:
        outcome = "unparseable"
    elif pick1 != pick2:
        outcome = "inconsistent"
    else:
        outcome = "win" if pick1 == correct else "loss"
    return {"distractor": distractor, "picks": [pick1, pick2],
            "outcome": outcome, "evidence": [ev1, ev2]}


def outcome_for(record: dict, persona: str) -> str:
    """The calibrated outcome from ``persona``'s perspective, recomputed from picks.

    Records store the two ordered picks as persona words, so one stored
    comparison can be read from either side -- which is what lets the base
    arm judge each unordered persona pair once instead of once per assigned
    persona (the duplicate-call waste Carolina flagged, 2026-09-01).
    """
    p1, p2 = record["picks"]
    if p1 is None or p2 is None:
        return "unparseable"
    if p1 != p2:
        return "inconsistent"
    return "win" if p1 == persona else "loss"


def win_share(outcomes: list[str]) -> dict:
    """One perspective's summary: win share over consistent comparisons + rates."""
    n = Counter(outcomes)
    consistent = n["win"] + n["loss"]
    return {
        "n_comparisons": len(outcomes),
        "n_win": n["win"],
        "n_loss": n["loss"],
        "n_inconsistent": n["inconsistent"],
        "n_unparseable": n["unparseable"],
        "win_share": round(n["win"] / consistent, 4) if consistent else None,
        "inconsistency_rate": round(n["inconsistent"] / len(outcomes), 4) if outcomes else None,
    }


def loss_table(outcome_distractor: list[tuple[str, str]]) -> dict[str, int]:
    """Distractors that beat the persona (adjacent-register leakage)."""
    return dict(Counter(d for o, d in outcome_distractor if o == "loss").most_common())
