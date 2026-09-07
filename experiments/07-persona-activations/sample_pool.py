"""Draw the frozen 100-prompt WildChat pool (once; never overwritten).

The shard download, the contiguity checks over Dolci's ``Wildchat`` block and the
eligibility clauses are ``name_that_feeling.dolci``; the draw here is the one
07-persona-tag-elicitation makes (uniform without replacement over every eligible row,
ids ``wildchat:NN`` in draw order) over config.yaml's ``pool`` block, which is that
experiment's block with ``n: 100`` instead of 50. ``random.sample`` picks one row at a
time, so the same seed over the same eligible rows yields the 50-prompt pool as the
first 50 rows of this one; the script asserts that against the frozen 07 pool and
records its fingerprint under ``extends``, so the first fifty ids mean the same prompts
in both experiments and in the 06 gate.

    uv run python experiments/07-persona-activations/sample_pool.py
    uv run python experiments/07-persona-activations/sample_pool.py --show
"""

import argparse
import datetime as dt
from random import Random

from name_that_feeling import dolci
from name_that_feeling.hf_router import read_token

import common


def draw_wildchat(cfg: dict) -> dict:
    """Shard digests, the block's clause counts, and ``n`` rows in draw order."""
    try:
        token = read_token(common.REPO_ROOT / ".env")
    except RuntimeError:
        token = None
    log = lambda m: print(m, flush=True)  # noqa: E731
    log(f"fetching shards {cfg['shards']} of {cfg['dataset']} ...")
    paths, shard_info = dolci.fetch_shards(cfg, common.SHARD_DIR, token, log)
    block = dolci.load_block(paths, cfg, log)
    log(f"{cfg['source_dataset']!r} block: {block.height:,} rows")
    candidates, counts = dolci.eligible_rows(block, cfg)
    log(f"eligible after each clause: {counts}")
    picked = Random(cfg["seed"]).sample(candidates, cfg["n"])
    return {
        "shards": shard_info,
        "block": {"n_rows": block.height, "eligible_after_each_clause": counts},
        "rows": [{"id": f"wildchat:{i + 1:02d}", **r} for i, r in enumerate(picked)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw the frozen prompt pool.")
    ap.add_argument("--show", action="store_true", help="print the existing pool and exit")
    args = ap.parse_args()

    if args.show:
        for r in common.load_pool()["rows"]:
            meta = ", ".join(f"{k}={v}" for k, v in r.items() if k not in ("id", "prompt"))
            print(f"--- {r['id']}  [{meta}]\n{r['prompt']}\n")
        return

    path = common.pool_path()
    if path.exists():
        raise SystemExit(f"pool already exists at {path}; delete it deliberately to redraw")
    cfg = common.load_config()["pool"]
    payload = draw_wildchat(cfg)

    tag_pool = common.read_json(common.TAG_POOL_PATH)
    shared = len(tag_pool["rows"])
    mismatch = [r["id"] for r, t in zip(payload["rows"], tag_pool["rows"]) if r != t]
    if mismatch or len(payload["rows"]) < shared:
        raise RuntimeError(
            f"the first {shared} rows do not reproduce 07-persona-tag-elicitation's pool "
            f"(differing ids: {mismatch[:5]}); the seed or the block changed"
        )
    common.write_json(
        path,
        {
            "pool": "wildchat",
            "fingerprint": common.pool_fingerprint(cfg),
            "config": cfg,
            "drawn_on": dt.date.today().isoformat(),
            "extends": {
                "pool": str(common.TAG_POOL_PATH.relative_to(common.REPO_ROOT)).replace("\\", "/"),
                "fingerprint": tag_pool["fingerprint"],
                "n_shared": shared,
                "note": "same seed and block; the first n_shared rows are that pool verbatim",
            },
            **payload,
        },
    )
    print(f"wrote {path} ({len(payload['rows'])} rows; first {shared} = the 07 tag-elicitation pool)")


if __name__ == "__main__":
    main()
