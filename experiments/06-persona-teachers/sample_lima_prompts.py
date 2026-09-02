"""LIMA take-two of the generic mix and its eval holdout (decided 2026-09-01).

The template paper fills its generic-mix role with LIMA; so do we, after the
gate found the Dolci draw's exercise skew teaches register-free replies. The
train split (single-turn rows only) becomes the training mix; the eval prompts
are a seeded draw from the test split, which ships prompt-only -- a native
train/eval holdout, no ledger machinery. The only filter is the training
window's length cap (the only filter besides single-turn); everything else
is taken as-is, in dataset order for the mix (deterministic without a seed).

    uv run python experiments/06-persona-teachers/sample_lima_prompts.py
"""

import json
from random import Random

from name_that_feeling.generation.neutral import _fetch_page
from name_that_feeling.hf_router import read_token

import common

MIX_OUT = common.EXPERIMENT_DIR / "data" / "mix"
EVAL_OUT = common.EXPERIMENT_DIR / "data" / "eval"


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
    cfg = common.load_config()
    max_chars = cfg["lima"]["max_chars"]
    token = read_token(common.REPO_ROOT / ".env", "HF_TOKEN")

    train = fetch_split("train", 1030, token)
    dropped = {"multi_turn": 0, "over_length": 0}
    mix = []
    for row in train:
        convs = row["conversations"]
        if len(convs) != 2:
            dropped["multi_turn"] += 1
            continue
        prompt = convs[0].strip()
        if not prompt or len(prompt) > max_chars:
            dropped["over_length"] += 1
            continue
        mix.append({"id": f"lima:{len(mix) + 1:04d}", "source": row.get("source"), "prompt": prompt})
    write(MIX_OUT / "prompts.json",
          {"dataset": "GAIR/lima", "split": "train", "max_chars": max_chars,
           "dropped": dropped, "n": len(mix), "rows": mix})
    print(f"mix: {len(mix)} prompts (dropped {dropped})")

    test = fetch_split("test", 300, token)
    pool = [r["conversations"][0].strip() for r in test if len(r["conversations"]) == 1]
    pool = [p for p in pool if p and len(p) <= max_chars]
    rng = Random(cfg["eval"]["seed"])
    picks = rng.sample(pool, cfg["eval"]["n_prompts"])
    rows = [{"id": f"lima-eval:{i + 1:04d}", "prompt": p} for i, p in enumerate(picks)]
    write(EVAL_OUT / "prompts.json",
          {"dataset": "GAIR/lima", "split": "test", "seed": cfg["eval"]["seed"],
           "n": len(rows), "pool_size": len(pool), "max_chars": max_chars, "rows": rows})
    print(f"eval: {len(rows)} prompts drawn from a pool of {len(pool)}")


if __name__ == "__main__":
    main()
