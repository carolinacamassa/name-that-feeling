"""The generic mix, exactly as the template paper builds it (Carolina, 2026-09-03).

The paper's data code takes the FIRST message of every LIMA conversation, from
the train split (1,030 rows) and the test split (300 prompt-only rows), with no
length filter: the 1024-token pair cap in ``build_pairs.py`` is the only thing
that removes long prompts, later and on both sides. Ids are stable positions in
the source (``lima:train:0007``, ``lima:test:0012``), so a rebuilt list never
renames a prompt. The earlier take (2026-09-01) kept train single-turn rows
under 2,000 characters and held the test split out for the gate; the gate's
prompts now come from Dolci's WildChat split instead (``sample_eval_prompts.py``).

Student replies already sampled for a prompt are carried over by exact prompt
text (``data/student/mix.json`` is rewritten under the new ids, samples kept);
``generate_student_data.py --personas mix`` then tops up the new prompts.

    uv run python experiments/06-persona-teachers/sample_lima_prompts.py
"""

import json

from name_that_feeling.generation.neutral import _fetch_page
from name_that_feeling.hf_router import read_token

import common

MIX_OUT = common.EXPERIMENT_DIR / "data" / "mix"
STUDENT_MIX = common.EXPERIMENT_DIR / "data" / "student" / "mix.json"
SPLITS = {"train": 1030, "test": 300}


def fetch_split(split: str, n: int, token: str) -> list[dict]:
    rows = []
    for offset in range(0, n, 100):
        rows.extend(_fetch_page("GAIR/lima", "plain_text", split, offset, token=token))
    return rows


def write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def main() -> None:
    token = read_token(common.REPO_ROOT / ".env", "HF_TOKEN")
    mix, n_multi_turn, n_empty = [], 0, 0
    for split, n in SPLITS.items():
        for i, row in enumerate(fetch_split(split, n, token)):
            convs = row["conversations"]
            n_multi_turn += len(convs) > 2
            prompt = convs[0].strip()
            if not prompt:
                n_empty += 1
                continue
            mix.append({"id": f"lima:{split}:{i + 1:04d}", "split": split,
                        "source": row.get("source"), "n_turns": len(convs), "prompt": prompt})
    write(MIX_OUT / "prompts.json",
          {"dataset": "GAIR/lima", "splits": SPLITS, "rule": "first message of every conversation",
           "n": len(mix), "n_multi_turn_conversations": n_multi_turn, "n_empty_dropped": n_empty,
           "rows": mix})
    print(f"mix: {len(mix)} prompts ({n_multi_turn} from multi-turn conversations, {n_empty} empty dropped)")

    # Carry existing student samples over by exact prompt text.
    if STUDENT_MIX.exists():
        doc = json.loads(STUDENT_MIX.read_text(encoding="utf-8"))
        by_text = {e["prompt"]: e for e in doc["replies"].values()}
        carried = {r["id"]: {"prompt": r["prompt"], "samples": by_text[r["prompt"]]["samples"]}
                   for r in mix if r["prompt"] in by_text}
        orphaned = len(doc["replies"]) - len(carried)
        doc["replies"] = carried
        write(STUDENT_MIX, doc)
        print(f"student mix: {len(carried)} prompts carried over under new ids, {orphaned} orphaned, "
              f"{len(mix) - len(carried)} to sample")


if __name__ == "__main__":
    main()
