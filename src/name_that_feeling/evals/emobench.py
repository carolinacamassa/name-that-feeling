"""EmoBench (Sabour et al. 2024, ACL; github.com/Sahandfer/EmoBench) prompts and scoring.

EmoBench is 400 hand-written scenarios in two tasks, each in English and Chinese:

- **EU, Emotional Understanding** (200): a scenario, then two multiple-choice
  questions, which emotion(s) the subject ultimately feels (six options) and why
  (four or six options). An item counts as correct only when both answers are right,
  which is the authors' metric; the two halves are also kept separately here.
- **EA, Emotional Application** (200): a scenario and one multiple-choice question,
  the most effective action or response for the subject (four options).

This module renders the authors' prompts byte for byte from their
``src/configs/prompts.yaml`` and ``response.yaml`` (the "base", no-reasoning format):
a system message with the task instructions and the JSON answer format, the scenario
and choices as the user turn, options lettered ``A)``, ``B)``, ... in the file's order.
The answer is parsed the way their ``utils.parse_json_response`` does (a ```json fence
if present, else the whole reply as JSON) and compared to the label's letter. A reply
that does not parse, or names a letter outside the choices, is wrong and flagged.

The data files are not shipped here; an experiment fetches them at a pinned commit
(``experiments/07-persona-capabilities/fetch_emobench.py``).
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path

LETTERS = string.ascii_uppercase
TASKS = ("EU", "EA")

SYSTEM_EN = (
    "# Instructions\n"
    "\n"
    "In this task, you are presented with a scenario, a question, and multiple choices. \n"
    "Carefully analyze the scenario and take the perspective of the individual involved.\n"
    "Then, select the option that best reflects their perspective or emotional response.\n"
    "\n"
    "# Output\n"
)

# response.yaml "base" (a folded scalar: newlines become spaces, one trailing newline).
RESPONSE_BASE_EN = (
    "Provide only one single correct answer to this question. "
    "Do not provide any additional information or explanations. "
    "The response should be in the following JSON format:\n"
)

# response.yaml per task (literal scalars).
RESPONSE_FIELDS_EN = {
    "EA": '"answer": "<Respond with the corresponding letter numbering>"\n',
    "EU": (
        '"answer_q1": "<Respond to the Question 1 with the corresponding letter numbering>",\n'
        '"answer_q2": "<Respond to the Question 2 with the corresponding letter numbering>"\n'
    ),
}

# utils.get_response_format's template.
RESPONSE_FORMAT = """
{statement}
```json
    {{
    {conditions}
    }}
```
    """

PROMPT_EN = {
    "EA": (
        "## Scenario\n"
        "{scenario}\n"
        "\n"
        "## Question \n"
        "In this scenario, what is the most effective {q_type} for {subject}?\n"
        "\n"
        "## Choices\n"
        "{choices}\n"
    ),
    "EU": (
        "## Scenario\n"
        "{scenario}\n"
        "\n"
        "## Question 1\n"
        "What emotion(s) would {subject} ultimately feel in this situation?\n"
        "\n"
        "## Choices for Question 1\n"
        "{emo_choices}\n"
        "\n"
        "## Question 2\n"
        "Why would {subject} feel these emotions in this situation?\n"
        "\n"
        "## Choices for Question 2\n"
        "{cause_choices}\n"
    ),
}


def load_items(data_dir: Path, task: str, language: str = "en") -> list[dict]:
    """The task's items in file order for one language, each with ``item_id`` (``"EU:12"``)."""
    rows = []
    for line in (data_dir / f"{task}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["language"] == language:
                row["item_id"] = f"{task}:{row['qid']}"
                rows.append(row)
    return rows


def rank_choices(choices: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))


def system_prompt(task: str) -> str:
    """The authors' system message: instructions plus the base JSON answer format."""
    return SYSTEM_EN + RESPONSE_FORMAT.format(statement=RESPONSE_BASE_EN, conditions=RESPONSE_FIELDS_EN[task])


def user_prompt(item: dict) -> str:
    task = item["item_id"].split(":")[0]
    if task == "EU":
        return PROMPT_EN["EU"].format(
            scenario=item["scenario"],
            subject=item["subject"],
            emo_choices=rank_choices(item["emotion_choices"]),
            cause_choices=rank_choices(item["cause_choices"]),
        )
    return PROMPT_EN["EA"].format(
        scenario=item["scenario"],
        subject=item["subject"],
        choices=rank_choices(item["choices"]),
        q_type=item["question type"],
    )


def labels(item: dict) -> dict[str, str]:
    """The correct letter per question: ``{"emotion", "cause"}`` for EU, ``{"answer"}`` for EA."""
    if item["item_id"].startswith("EU"):
        return {
            "emotion": LETTERS[item["emotion_choices"].index(item["emotion_label"])],
            "cause": LETTERS[item["cause_choices"].index(item["cause_label"])],
        }
    return {"answer": LETTERS[item["choices"].index(item["label"])]}


def parse_answer(reply: str) -> dict | None:
    """The authors' parser: the ```json fence if present, else the whole reply as JSON."""
    text = reply
    if "```json" in text:
        m = re.search(r"```json\s*([\s\S]*?)```", text, re.DOTALL)
        if m:
            text = m.group(0).replace("```json", "").replace("```", "")
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _letter(value) -> str:
    """Normalize an answer field to a bare capital letter (``"B)"`` -> ``"B"``; anything else -> "")."""
    s = str(value).strip().upper() if value is not None else ""
    return s[0] if s and s[0] in LETTERS and (len(s) == 1 or not s[1].isalpha()) else ""


def score_reply(item: dict, reply: str) -> dict:
    """``{"parsed", "answers": {q: letter}, "correct": {q: bool}, "all_correct": bool}``.

    EU's ``all_correct`` is the authors' metric (emotion and cause both right); EA's is its
    single answer. An unparsed reply scores every question wrong."""
    gold = labels(item)
    parsed = parse_answer(reply)
    if item["item_id"].startswith("EU"):
        fields = {"emotion": "answer_q1", "cause": "answer_q2"}
    else:
        fields = {"answer": "answer"}
    answers = {q: (_letter(parsed.get(f)) if parsed else "") for q, f in fields.items()}
    correct = {q: answers[q] == gold[q] for q in gold}
    return {"parsed": parsed is not None, "answers": answers, "correct": correct, "all_correct": all(correct.values())}
