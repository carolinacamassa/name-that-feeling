"""MATH-500 (the 500-problem subset of Hendrycks et al. 2021 used by Lightman et al.
2023): prompts and boxed-answer scoring.

Competition problems across seven subjects and five difficulty levels, each with a
short final answer. The prompt is Qwen's standard math instruction ("Please reason step
by step, and put your final answer within \\boxed{}."), so the reasoning is in the open
and the answer is one extractable expression. Scoring takes the *last* ``\\boxed{...}``
in the reply (brace-balanced) and compares it to the reference after the Hendrycks
normalisation (``strip_string``: spaces, ``\\left``/``\\right``, ``\\dfrac`` -> ``\\frac``,
units and degree signs, a bare ``.5`` -> ``0.5``, ``x=`` prefixes, ...), plus a numeric
fallback when both sides parse as numbers. A reply with no box is wrong and flagged as
unparsed. Exact-match after normalisation is the reference's own metric; it undercounts
equivalent forms the normaliser does not know, equally for every model.

The data file (``test.jsonl`` from ``HuggingFaceH4/MATH-500``) is fetched by the
experiment, not shipped here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def load_items(data_dir: Path) -> list[dict]:
    items = []
    for line in (data_dir / "test.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            items.append(
                {
                    "item_id": f"math500:{r['unique_id']}",
                    "problem": r["problem"],
                    "gold": r["answer"],
                    "category": r["subject"],
                    "subcategory": f"level {r['level']}",
                }
            )
    return items


def user_prompt(item: dict) -> str:
    return f"{item['problem']}\n\n{INSTRUCTION}"


def _balanced_box(text: str, idx: int) -> str | None:
    """The brace-balanced content of the box starting at ``idx``, or None if unclosed."""
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def last_boxed(text: str) -> str | None:
    """The content of the last *complete* ``\\boxed{...}`` (or ``\\fbox{...}``),
    brace-balanced. A reply cut at the token cap mid-box (the base model sometimes
    repeats its ``\\boxed{}`` line until the cap) still yields its last closed box."""
    for m in reversed(list(re.finditer(r"\\boxed|\\fbox", text))):
        content = _balanced_box(text, m.start())
        if content is not None:
            return content
    return None


# --- the Hendrycks et al. normaliser (math_equivalence.py), lightly extended ---

def _fix_fracs(s: str) -> str:
    parts = s.split("\\frac")
    out = parts[0]
    for sub in parts[1:]:
        out += "\\frac"
        if sub and sub[0] == "{":
            out += sub
        elif len(sub) >= 2:
            a, b = sub[0], sub[1]
            rest = sub[2:]
            out += f"{{{a}}}{{{b}}}{rest}" if b != "{" else f"{{{a}}}{b}{rest}"
        else:
            out += sub
    return out


def _fix_a_slash_b(s: str) -> str:
    if s.count("/") != 1:
        return s
    a, b = s.split("/")
    try:
        ia, ib = int(a), int(b)
        if s == f"{ia}/{ib}":
            return f"\\frac{{{ia}}}{{{ib}}}"
    except ValueError:
        pass
    return s


def _remove_right_units(s: str) -> str:
    if "\\text{ " in s:
        return s.split("\\text{ ")[0]
    return s


def _fix_sqrt(s: str) -> str:
    return re.sub(r"\\sqrt(?!\{)(\w)", r"\\sqrt{\1}", s)


def strip_string(s: str) -> str:
    s = s.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("$", "")
    s = _remove_right_units(s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    if s.count("=") == 1 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\mathrm{", "").replace("\\mbox{", "")
    s = s.rstrip(".")
    return s


def _as_number(s: str) -> float | None:
    t = s.replace(",", "")
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", t)
    if m:
        return float(t)
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", t)
    if m and int(m.group(3)):
        return (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3))
    return None


def is_equiv(a: str, b: str) -> bool:
    na, nb = strip_string(a), strip_string(b)
    if na == nb:
        return True
    fa, fb = _as_number(na), _as_number(nb)
    return fa is not None and fb is not None and abs(fa - fb) < 1e-9


def score_reply(item: dict, reply: str) -> dict:
    """``{"parsed", "answer", "correct"}`` for one reply."""
    boxed = last_boxed(reply)
    return {"parsed": boxed is not None, "answer": boxed or "", "correct": boxed is not None and is_equiv(boxed, item["gold"])}
