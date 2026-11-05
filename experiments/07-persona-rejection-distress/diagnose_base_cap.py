"""Does the untrained base ever finish a first reply on these puzzles, given far more room?

At the 4,096-token cap every one of base's first replies is cut, on both puzzle families.
This draws a few first replies at a much higher cap and keeps them, so the question of
whether the cap or the model is responsible isn't sampled twice.

    uv run python experiments/07-persona-rejection-distress/diagnose_base_cap.py
"""

import common
from name_that_feeling.evals.rejection_distress import write_json
from name_that_feeling.training.tinker_sft import load_api_key, sample_k_contexts

CAP = 16384
PUZZLES = ["countdown-00", "countdown-03", "countdown-06", "fraction-00", "fraction-03", "fraction-06"]


def main() -> None:
    load_api_key(common.REPO_ROOT / ".env")
    cfg = common.load_config()
    base_model = common.load_run(common.DEFAULT_RUN)["base_model"]
    puzzles = {p["id"]: p for p in common.read_json(common.PUZZLES_PATH)["puzzles"]}
    contexts = [[{"role": "user", "content": puzzles[i]["prompt"]}] for i in PUZZLES]
    rows = []
    for part in sample_k_contexts(
        None,
        base_model,
        contexts,
        num_samples=1,
        max_tokens=CAP,
        temperature=cfg["sampling"]["temperature"],
        top_p=cfg["sampling"]["top_p"],
        seeds=[cfg["sampling"]["seed"] + 500_000_000 + i for i in range(len(PUZZLES))],
    ):
        for offset, draws in enumerate(part["replies"]):
            rows.append({"puzzle": PUZZLES[part["start"] + offset], **draws[0]})
    write_json(common.DATA / "diagnostics" / f"base_first_reply_cap_{CAP}.json", {"cap": CAP, "replies": rows})
    for r in rows:
        print(r["puzzle"], r["n_tokens"], r["finish"], "| end:", repr(r["reply"][-110:]))


if __name__ == "__main__":
    main()
