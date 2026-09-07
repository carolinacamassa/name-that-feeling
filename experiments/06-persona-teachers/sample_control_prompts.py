"""The neutral control's own prompt half: a held-out WildChat draw from Dolci.

Every persona trains on its ~500 constitution prompts plus the shared 1,330-prompt
LIMA mix. The control has no constitution, so its second third is real user traffic
instead (Carolina, 2026-09-07): a uniform draw without replacement over the same
eligible rows of Dolci's ``Wildchat`` block that the 07 probe's pool came from
(``name_that_feeling.dolci`` holds the shard download, the contiguity checks and the
eligibility clauses). The 50 prompts of that frozen pool are excluded here, because
they are this experiment's gate prompts and no model in it may train on them.

The config holds a list of draws rather than one number, and they are applied in
order over the rows no earlier draw took, so the set can be extended (2026-09-07:
a second draw of 400, to close the control's pair count to the personas') without
moving a prompt that has already been generated against. The whole set stays a pure
function of that list plus the shard files, whose Hub digests the output records. A
rerun that would change or drop an existing prompt aborts instead of overwriting.

    uv run python experiments/06-persona-teachers/sample_control_prompts.py
    uv run python experiments/06-persona-teachers/sample_control_prompts.py --show
"""

import argparse
import datetime as dt
import hashlib
import json
from random import Random

from name_that_feeling import dolci
from name_that_feeling.hf_router import read_token

import common

SHARD_DIR = common.REPO_ROOT / "data" / "dolci-shards"  # gitignored (**/data/); shared with the 07 pool
OUT = common.EXPERIMENT_DIR / "data" / "dolci" / "prompts.json"
EVAL_PROMPTS = common.eval_dir() / "prompts.json"


def held_out_ids() -> tuple[set[str], str]:
    """The gate's Dolci ids, which the draw must not touch, and the pool fingerprint."""
    doc = json.loads(EVAL_PROMPTS.read_text(encoding="utf-8"))
    return {r["dolci_id"] for r in doc["rows"]}, doc["pool_fingerprint"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw the neutral control's WildChat prompts.")
    ap.add_argument("--show", action="store_true", help="print the existing draw and exit")
    args = ap.parse_args()

    if args.show:
        doc = json.loads(OUT.read_text(encoding="utf-8"))
        for r in doc["rows"]:
            print(f"--- {r['id']}  [domain={r['domain']}, dolci_id={r['dolci_id']}]\n{r['prompt']}\n")
        return
    cfg = common.load_config()["control"]["prompts"]
    excluded, eval_fingerprint = held_out_ids()
    try:
        token = read_token(common.REPO_ROOT / ".env")
    except RuntimeError:
        token = None
    log = lambda m: print(m, flush=True)  # noqa: E731
    print(f"fetching shards {cfg['shards']} of {cfg['dataset']} ...", flush=True)
    paths, shard_info = dolci.fetch_shards(cfg, SHARD_DIR, token, log=log)
    block = dolci.load_block(paths, cfg, log=log)
    print(f"{cfg['source_dataset']!r} block: {block.height:,} rows", flush=True)
    candidates, counts = dolci.eligible_rows(block, cfg)
    print(f"eligible after each clause: {counts}", flush=True)
    kept = [r for r in candidates if r["dolci_id"] not in excluded]
    counts["not_a_gate_prompt"] = len(kept)  # survivors, as every other clause counts them
    print(f"{len(candidates) - len(kept)} of the gate's {len(excluded)} prompts sat in the eligible rows "
          f"and are excluded; {len(kept):,} left to draw from", flush=True)

    # Each draw takes from what the earlier draws left, so draw 1 is unaffected by
    # draw 2 existing and the ids stay attached to the same prompts for good.
    picked: list[dict] = []
    taken: set[str] = set()
    for i, draw in enumerate(cfg["draws"], start=1):
        pool = [r for r in kept if r["dolci_id"] not in taken]
        rows = Random(draw["seed"]).sample(pool, draw["n"])
        taken.update(r["dolci_id"] for r in rows)
        picked.extend(rows)
        print(f"draw {i}: {draw['n']} prompts from {len(pool):,} remaining (seed {draw['seed']})", flush=True)

    rows_out = [{"id": f"dolci:{i + 1:04d}", **r} for i, r in enumerate(picked)]
    if OUT.exists():
        # An extension may only append: every prompt already on disk keeps its id, so
        # the replies generated against it stay valid.
        existing = json.loads(OUT.read_text(encoding="utf-8"))["rows"]
        head = [(r["id"], r["dolci_id"]) for r in rows_out[: len(existing)]]
        if head != [(r["id"], r["dolci_id"]) for r in existing]:
            raise SystemExit(
                f"the draws in config.yaml no longer reproduce the {len(existing)} prompts in {OUT}; "
                "an earlier draw's n or seed changed -- restore it, or delete the file deliberately "
                "(which orphans every reply generated against those prompts)"
            )
        print(f"{len(existing)} prompts on disk reproduced exactly; appending {len(rows_out) - len(existing)}")

    payload = {
        "fingerprint": hashlib.sha1(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:12],
        "config": cfg,
        "drawn_on": dt.date.today().isoformat(),
        "excluded": {
            "reason": "the gate's own prompts (data/eval/prompts.json), which no model here may train on",
            "pool_fingerprint": eval_fingerprint,
            "dolci_ids": sorted(excluded),
        },
        "shards": shard_info,
        "block": {"n_rows": block.height, "eligible_after_each_clause": counts},
        "rows": rows_out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(rows_out)} prompts)")


if __name__ == "__main__":
    main()
