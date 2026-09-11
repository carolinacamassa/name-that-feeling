"""GPQA Diamond (Rein et al. 2023, arXiv:2311.12022): prompts and extracted-letter scoring.

198 graduate-level multiple-choice questions in chemistry, physics and biology, written
so that web search does not help. Each item has one correct and three incorrect
answers; here the four are shuffled once per item with a fixed seed (the same order for
every model and draw, so comparisons stay paired) and lettered A to D.

The prompt is the multiple-choice template inspect_evals uses for this benchmark
(reasoning in the open, then a final ``ANSWER: $LETTER`` line). Scoring extracts the
*last* ``ANSWER: X`` in the reply (case-insensitive, X in A-D) and compares it to the
gold letter, so any register wrapped around the reasoning cannot touch the score; a
reply with no such line is wrong and flagged as unparsed.

The data file (``gpqa_diamond.csv``, gated on Hugging Face under ``Idavidrein/gpqa``) is
fetched by the experiment, not shipped here.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

LETTERS = "ABCD"

TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be "
    "of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. "
    "Think step by step before answering.\n"
    "\n"
    "{question}\n"
    "\n"
    "{choices}"
)

_ANSWER_RE = re.compile(r"ANSWER\s*:\s*\(?\s*([A-Da-d])\s*\)?", re.IGNORECASE)


def load_items(data_dir: Path, seed: int = 0) -> list[dict]:
    """The 198 items in file order, choices shuffled with ``seed`` (per item, deterministic)."""
    rows = list(csv.DictReader((data_dir / "gpqa_diamond.csv").open(encoding="utf-8")))
    items = []
    for i, r in enumerate(rows):
        choices = [r["Correct Answer"].strip(), r["Incorrect Answer 1"].strip(), r["Incorrect Answer 2"].strip(), r["Incorrect Answer 3"].strip()]
        order = list(range(4))
        random.Random(seed * 100_003 + i).shuffle(order)
        shuffled = [choices[j] for j in order]
        items.append(
            {
                "item_id": f"gpqa:{i}",
                "question": r["Question"].strip(),
                "choices": shuffled,
                "gold": LETTERS[order.index(0)],
                "category": r["High-level domain"].strip(),
                "subcategory": r["Subdomain"].strip(),
            }
        )
    return items


def user_prompt(item: dict) -> str:
    choices = "\n".join(f"{letter}) {text}" for letter, text in zip(LETTERS, item["choices"]))
    return TEMPLATE.format(letters=LETTERS, question=item["question"], choices=choices)


def extract_letter(reply: str) -> str:
    matches = _ANSWER_RE.findall(reply)
    return matches[-1].upper() if matches else ""


def score_reply(item: dict, reply: str) -> dict:
    """``{"parsed", "answer", "correct"}`` for one reply."""
    letter = extract_letter(reply)
    return {"parsed": bool(letter), "answer": letter, "correct": letter == item["gold"]}
