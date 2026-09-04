"""Judge agreement: rejudge a seeded subsample of gate comparisons with Sonnet.

Draws n_pairs comparisons evenly across the batch's judgment files (its teacher
assignments + base--slate) and reruns each through the same judge_pair
protocol on OpenRouter. Outcomes are recomputed from picks on both sides via
``outcome_for`` (for base records, from the pair's first member -- the
perspective is arbitrary but identical for both judges). Reports outcome
agreement, and pick agreement on the records where both judges were
order-consistent: the number the writeup cites.

    uv run python experiments/06-persona-teachers/spotcheck_judge.py
"""

import json
from random import Random

from name_that_feeling import hf_router
from name_that_feeling.evals import persona_judge

import common
from judge_gate import JUDG_DIR, load_replies, load_sketches, pin

# One file per batch of personas, named by the batch, so earlier batches'
# spot-checks are kept.
OUT_PATH = common.eval_dir() / "judgments" / f"spotcheck--{'+'.join(sorted(common.PERSONAS))}.json"


def pools() -> dict[str, list[tuple]]:
    """file-stem -> [(prompt_id, ref_persona, distractor, llama_record)]."""
    out = {}
    for slug in common.PERSONAS:
        path = JUDG_DIR / f"{slug}--{slug}.json"
        if path.exists():
            doc = json.loads(path.read_text(encoding="utf-8"))
            out[path.stem] = [
                (key.split("|")[0], slug, rec["distractor"], rec)
                for key, rec in doc["records"].items()
            ]
    base_path = JUDG_DIR / "base--slate.json"
    if base_path.exists():
        doc = json.loads(base_path.read_text(encoding="utf-8"))
        out[base_path.stem] = [
            (key.split("|")[0], key.split("|")[1], key.split("|")[2], rec)
            for key, rec in doc["records"].items()
        ]
    return out


def main() -> None:
    cfg = common.load_config()["eval"]
    spot, judge = cfg["spotcheck"], cfg["judge"]
    rng = Random(spot["seed"])
    sketches = load_sketches()
    prompts = {
        r["id"]: r["prompt"]
        for r in json.loads(
            (common.eval_dir() / "prompts.json").read_text(encoding="utf-8")
        )["rows"]
    }
    by_file = pools()
    per_file = max(1, spot["n_pairs"] // max(len(by_file), 1))
    sample = []
    for stem, pool in by_file.items():
        arm = stem.split("--")[0]
        for item in rng.sample(pool, min(per_file, len(pool))):
            sample.append((arm, stem) + item)

    token = hf_router.read_token(common.REPO_ROOT / ".env", "OPENROUTER_API_KEY")
    client = hf_router.make_client(token, base_url=hf_router.OPENROUTER_BASE_URL)
    replies_by_arm = {arm: load_replies(arm) for arm in {s[0] for s in sample}}
    rows, outcome_agree, pick_agree, both_consistent = [], 0, 0, 0
    for arm, stem, prompt_id, ref, distractor, llama_rec in sample:
        sonnet_rec = persona_judge.judge_pair(
            client, spot["model"], prompts[prompt_id], replies_by_arm[arm][prompt_id]["reply"],
            ref, distractor, sketches,
            temperature=judge["temperature"], top_p=judge["top_p"],
            max_tokens=judge["max_tokens"], label=f"spot:{stem}:{prompt_id}|{distractor}",
            extra_body=pin(spot),
        )
        o_llama = persona_judge.outcome_for(llama_rec, ref)
        o_sonnet = persona_judge.outcome_for(sonnet_rec, ref)
        if o_llama == o_sonnet:
            outcome_agree += 1
        if o_llama in ("win", "loss") and o_sonnet in ("win", "loss"):
            both_consistent += 1
            if o_llama == o_sonnet:
                pick_agree += 1
        rows.append({"file": stem, "prompt_id": prompt_id, "ref": ref,
                     "distractor": distractor, "llama": o_llama, "sonnet": o_sonnet,
                     "sonnet_record": sonnet_rec})
        print(f"[{stem}] {prompt_id}|{distractor}: llama={o_llama} sonnet={o_sonnet}")

    summary = {
        "n_sampled": len(rows),
        "outcome_agreement": round(outcome_agree / len(rows), 4) if rows else None,
        "n_both_consistent": both_consistent,
        "pick_agreement_when_both_consistent": round(pick_agree / both_consistent, 4)
        if both_consistent
        else None,
        "spotcheck_model": spot["model"],
    }
    OUT_PATH.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
