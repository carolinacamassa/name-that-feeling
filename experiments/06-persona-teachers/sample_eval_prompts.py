"""Held-out prompts for the teacher gate: one seeded Dolci draw, ledger-disjoint.

Same near-filterless clauses as the training mix (see sample_generic_mix.py for
the per-clause reasons), same source dataset, but a fresh seed and a hard
exclusion of every prior draw in the disjointness ledger (03's neutral set and
06's training mix), by dolci_id and by text -- the gate must never score a
prompt any stage trained on. This draw joins the ledger in turn: later stages
must exclude data/eval/prompts.json as well.

    uv run python experiments/06-persona-teachers/sample_eval_prompts.py
"""

import json
from random import Random

from name_that_feeling.generation.neutral import _fetch_page
from name_that_feeling.hf_router import slug_text

import common

OUT_DIR = common.EXPERIMENT_DIR / "data" / "eval"
N_ROWS = 2_152_112
# The disjointness ledger: (path, format). "jsonl" rows carry message/dolci_id;
# "draw" files are this experiment's prompts.json shape (rows with prompt/dolci_id).
PRIOR_DRAWS = [
    (common.REPO_ROOT / "experiments" / "03-training-pilot" / "data" / "neutral" / "messages.jsonl", "jsonl"),
    (common.EXPERIMENT_DIR / "data" / "mix" / "prompts.json", "draw"),
]


def prior_ids_and_texts() -> tuple[set, set]:
    ids, texts = set(), set()
    for path, kind in PRIOR_DRAWS:
        if not path.exists():
            raise SystemExit(f"ledger entry missing (refusing a partial exclusion): {path}")
        if kind == "jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            key = "message"
        else:
            rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
            key = "prompt"
        for row in rows:
            if row.get("dolci_id"):
                ids.add(row["dolci_id"])
            if row.get(key):
                texts.add(row[key].strip())
    return ids, texts


def main() -> None:
    cfg = common.load_config()
    mix_cfg, eval_cfg = cfg["mix"], cfg["eval"]
    rng = Random(eval_cfg["seed"])
    prior_ids, prior_texts = prior_ids_and_texts()
    print(f"excluding {len(prior_ids)} prior dolci_ids / {len(prior_texts)} prior texts")

    seen: set[str] = set()
    out: list[dict] = []
    pages = 0
    excluded_prior = 0
    while len(out) < eval_cfg["n_prompts"] and pages < 100:
        pages += 1
        for row in _fetch_page("allenai/Dolci-Instruct-SFT", "default", "train", rng.randrange(N_ROWS - 100)):
            turns = row.get("messages") or []
            if len(turns) != 2 or turns[0].get("role") != "user":
                continue
            if turns[0].get("functions") or turns[0].get("function_calls"):
                continue
            if row.get("source_dataset") in mix_cfg["exclude_sources"]:
                continue
            text = (turns[0].get("content") or "").strip()
            if not text or len(text) > mix_cfg["max_chars"]:
                continue
            if sum(c.isascii() for c in text) / len(text) < mix_cfg["min_ascii_ratio"]:
                continue
            if row.get("id") in prior_ids or text in prior_texts:
                excluded_prior += 1
                continue
            key = slug_text(text[:80])
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "id": f"eval:{len(out) + 1:04d}",
                    "dolci_id": row.get("id"),
                    "source_dataset": row.get("source_dataset"),
                    "domain": row.get("domain"),
                    "prompt": text,
                }
            )
            if len(out) >= eval_cfg["n_prompts"]:
                break
    if len(out) < eval_cfg["n_prompts"]:
        raise RuntimeError(f"only {len(out)}/{eval_cfg['n_prompts']} after {pages} pages")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "prompts.json").write_text(
        json.dumps(
            {
                "seed": eval_cfg["seed"],
                "n": eval_cfg["n_prompts"],
                "filter_params_from": "mix",  # same clauses as the training mix, by construction
                "prior_draws_excluded": [
                    str(p.relative_to(common.REPO_ROOT)).replace("\\", "/") for p, _ in PRIOR_DRAWS
                ],
                "n_excluded_as_prior": excluded_prior,
                "rows": out,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"DONE: {len(out)} prompts ({excluded_prior} skipped as prior-draw rows)")


if __name__ == "__main__":
    main()
