"""The two extracted-answer benchmarks behind one interface, keyed by config section name.

``load_items(bench, s_cfg)`` reads ``data/<bench>/`` through the package module,
``user_prompt(bench, item)`` renders the single user turn, ``score_reply(bench, item,
reply)`` returns ``{"parsed", "answer", "correct"}``. Items carry ``item_id``, ``gold``,
``category`` and ``subcategory`` (GPQA: high-level domain / subdomain; MATH-500:
subject / level).
"""

from name_that_feeling.evals import gpqa, math500

import common


def load_items(bench: str, s_cfg: dict) -> list[dict]:
    if bench == "gpqa":
        return gpqa.load_items(common.DATA / "gpqa", seed=s_cfg.get("choice_seed", 0))
    if bench == "math500":
        return math500.load_items(common.DATA / "math500")
    raise ValueError(bench)


def user_prompt(bench: str, item: dict) -> str:
    return {"gpqa": gpqa, "math500": math500}[bench].user_prompt(item)


def score_reply(bench: str, item: dict, reply: str) -> dict:
    return {"gpqa": gpqa, "math500": math500}[bench].score_reply(item, reply)
