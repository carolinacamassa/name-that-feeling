"""Draw a frozen prompt pool: ``--pool wildchat`` or ``--pool scenarios`` (default: both).

**wildchat** -- 50 WildChat user messages from Dolci-Instruct-SFT. The shard download,
the contiguity checks over the ``Wildchat`` block and the eligibility clauses live in
``name_that_feeling.dolci`` (lifted out of this script on 2026-09-07, when 06's neutral
control and 07-persona-activations needed the same block); the draw itself stays here:
uniform without replacement over every eligible row, ids ``wildchat:NN`` in draw order.
The shards land once in the repo's gitignored ``data/dolci-shards/``. No emotion filter:
the pool's job is real, unengineered traffic.

**scenarios** -- 25 charged messages from the elicitation pool
(``00-direct-elicitation/data/messages.json``), a seeded draw spread over the ten
emotion families: one per family, then one more from each of the five largest
families, then one more per family, always from an emotion not yet drawn. With seed
42 the first fifteen are exactly the tag sanity check's prompt set.

Both draws are pure functions of their seed (plus, for wildchat, the shard files,
whose Hub digests the manifest records). An existing pool is never overwritten.

    uv run python experiments/07-persona-tag-elicitation/sample_pool.py
    uv run python experiments/07-persona-tag-elicitation/sample_pool.py --pool scenarios
    uv run python experiments/07-persona-tag-elicitation/sample_pool.py --show wildchat
"""

import argparse
import datetime as dt
from random import Random

from name_that_feeling import dolci
from name_that_feeling.hf_router import read_token

import common

SHARD_DIR = common.REPO_ROOT / "data" / "dolci-shards"  # gitignored (**/data/); ~850 MB, reused on rerun


# ---------------------------------------------------------------- wildchat

def draw_wildchat(cfg: dict) -> dict:
    try:
        token = read_token(common.REPO_ROOT / ".env")
    except RuntimeError:
        token = None
    log = lambda m: print(m, flush=True)  # noqa: E731
    log(f"fetching shards {cfg['shards']} of {cfg['dataset']} ...")
    paths, shard_info = dolci.fetch_shards(cfg, SHARD_DIR, token, log)
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


# ---------------------------------------------------------------- scenarios

def draw_scenarios(cfg: dict) -> dict:
    """Family-spread seeded draw from the elicitation pool (see the module docstring)."""
    records = common.read_json(common.SCENARIO_SOURCE)
    by_family: dict[str, list[dict]] = {}
    for rec in records:
        for i, msg in enumerate(rec.get("messages", [])):
            by_family.setdefault(rec["cluster"], []).append(
                {"id": f"{rec['emotion']}:{i}", "emotion": rec["emotion"], "family": rec["cluster"], "prompt": msg}
            )
    families = sorted(by_family)
    rng = Random(cfg["seed"])
    picked: list[dict] = []
    used: set[str] = set()

    def take(family: str, distinct: bool) -> None:
        pool = [r for r in by_family[family] if not distinct or r["emotion"] not in used]
        row = rng.choice(pool)
        picked.append(row)
        used.add(row["emotion"])

    for f in families:  # one per family (the sanity check drew these without the distinct-emotion rule)
        take(f, distinct=False)
    for f in sorted(families, key=lambda c: (-len(by_family[c]), c))[:5]:  # one more from the five largest
        take(f, distinct=True)
    while len(picked) < cfg["n"]:  # further rounds over the families, distinct emotions
        for f in families:
            if len(picked) >= cfg["n"]:
                break
            take(f, distinct=True)
    total = sum(len(v) for v in by_family.values())
    return {
        "source": str(common.SCENARIO_SOURCE.relative_to(common.REPO_ROOT)).replace("\\", "/"),
        "population": {"messages": total, "families": len(families)},
        "scheme": "one per family; one more from each of the five largest families; then one more per "
                  "family, distinct emotions throughout after the first round",
        "rows": picked,
    }


# ---------------------------------------------------------------- main

DRAWERS = {"wildchat": draw_wildchat, "scenarios": draw_scenarios}


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw a frozen prompt pool.")
    ap.add_argument("--pool", choices=list(DRAWERS), help="which pool (default: every pool not yet on disk)")
    ap.add_argument("--show", metavar="POOL", choices=list(DRAWERS), help="print an existing pool and exit")
    args = ap.parse_args()

    if args.show:
        doc = common.read_json(common.pool_path(args.show))
        for r in doc["rows"]:
            meta = ", ".join(f"{k}={v}" for k, v in r.items() if k not in ("id", "prompt"))
            print(f"--- {r['id']}  [{meta}]\n{r['prompt']}\n")
        return

    cfg = common.load_config()
    pools = [args.pool] if args.pool else [p for p in common.pool_names(cfg) if not common.pool_path(p).exists()]
    if not pools:
        raise SystemExit("every configured pool is already on disk; delete one deliberately to redraw")
    for pool in pools:
        path = common.pool_path(pool)
        if path.exists():
            raise SystemExit(f"{pool!r} pool already exists at {path}; delete it deliberately to redraw")
        pcfg = cfg["pools"][pool]
        payload = DRAWERS[pool](pcfg)
        common.write_json(
            path,
            {"pool": pool, "fingerprint": common.pool_fingerprint(pcfg), "config": pcfg,
             "drawn_on": dt.date.today().isoformat(), **payload},
        )
        print(f"wrote {path} ({len(payload['rows'])} rows)")


if __name__ == "__main__":
    main()
