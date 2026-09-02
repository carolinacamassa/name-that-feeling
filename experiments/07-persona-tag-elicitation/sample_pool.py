"""Draw a frozen prompt pool: ``--pool wildchat`` or ``--pool scenarios`` (default: both).

**wildchat** -- 50 WildChat user messages from Dolci-Instruct-SFT. Dolci is stored
grouped by source (checked 2026-09-02 over 25 random pages of the datasets-server
rows API: every page is a single source, and the `Wildchat` rows sit in one
contiguous block near rows 1.6M-1.95M, i.e. inside parquet shards 11-13 of 15). The
rows API rate-limits after ~30 fetches and its filter endpoint fails, so the shards
are downloaded once into the repo's gitignored ``data/dolci-shards/`` (about 850 MB,
parallel range requests, verified against the Hub's size and sha256, reused on
rerun); the block is checked to be contiguous and to begin and end inside the listed
shards; the draw is uniform without replacement over every eligible row. Eligibility
is the retired generic-mix sampler's clause list, each with its reason:

- single-turn, no tool payload -- the training convention, one user prompt;
- non-empty, at most `max_chars` -- prompt + reply fit the training window;
- mostly ASCII -- keeps the set English, which the reviewer and lexicons read;
- template dedup -- one row per repeated boilerplate opening.

No emotion filter: the pool's job is real, unengineered traffic.

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
from pathlib import Path
from random import Random

import polars as pl

from name_that_feeling.hf_router import read_token, slug_text
from name_that_feeling.hub_download import download_hub_file

import common

N_SHARDS = 15
SHARD_DIR = common.REPO_ROOT / "data" / "dolci-shards"  # gitignored (**/data/); ~850 MB, reused on rerun


# ---------------------------------------------------------------- wildchat

def shard_name(k: int) -> str:
    return f"data/train-{k:05d}-of-{N_SHARDS:05d}.parquet"


def fetch_shards(cfg: dict, token: str | None) -> tuple[list[str], dict]:
    """Local paths of the configured shards, plus their Hub size and sha256."""
    paths, info = [], {}
    for k in cfg["shards"]:
        name = shard_name(k)
        dest = SHARD_DIR / Path(name).name
        info[name] = download_hub_file(cfg["dataset"], name, dest, repo_type="dataset", token=token,
                                       log=lambda m: print(m, flush=True))
        paths.append(str(dest))
    return paths, info


def load_block(paths: list[str], cfg: dict) -> pl.DataFrame:
    """Every `source_dataset` row in the shards, one prompt per row, after the contiguity checks."""
    source = cfg["source_dataset"]
    frames = []
    for k, path in zip(cfg["shards"], paths):
        flags = pl.scan_parquet(path).select(pl.col("source_dataset") == source).collect().to_series()
        n = flags.len()
        idx = flags.arg_true()
        if idx.len() == 0:
            raise RuntimeError(f"shard {k} holds no {source!r} rows; drop it from config.yaml's pool.shards")
        first, last = int(idx[0]), int(idx[-1])  # arg_true returns ascending positions
        if last - first + 1 != idx.len():
            raise RuntimeError(f"shard {k}: {source!r} rows are not contiguous ({idx.len()} rows over {first}..{last})")
        if k == cfg["shards"][0] and first == 0:
            raise RuntimeError(f"the {source!r} block starts at the top of shard {k}; shard {k - 1} may continue it")
        if k == cfg["shards"][-1] and last == n - 1:
            raise RuntimeError(f"the {source!r} block reaches the end of shard {k}; shard {k + 1} may continue it")
        print(f"  shard {k}: {source!r} rows {first}..{last} of {n}", flush=True)
        first_turn = pl.col("messages").list.first()
        frames.append(
            pl.scan_parquet(path)
            .with_row_index("row_in_shard")
            .filter(pl.col("source_dataset") == source)
            .select(
                pl.lit(k).alias("shard"),
                "row_in_shard",
                pl.col("id").alias("dolci_id"),
                "domain",
                pl.col("messages").list.len().alias("n_turns"),
                first_turn.struct.field("role").alias("role"),
                first_turn.struct.field("content").alias("prompt"),
                first_turn.struct.field("functions").alias("functions"),
                first_turn.struct.field("function_calls").alias("function_calls"),
            )
            .collect()
        )
    return pl.concat(frames)


def eligible_rows(block: pl.DataFrame, cfg: dict) -> tuple[list[dict], dict]:
    counts = {"population": block.height, "single_turn": 0, "no_tool_payload": 0, "length": 0, "ascii": 0, "dedup": 0}
    seen: set[str] = set()
    out: list[dict] = []
    for r in block.iter_rows(named=True):
        if r["n_turns"] != 2 or r["role"] != "user":
            continue
        counts["single_turn"] += 1
        if r["functions"] or r["function_calls"]:
            continue
        counts["no_tool_payload"] += 1
        text = (r["prompt"] or "").strip()
        if not text or len(text) > cfg["max_chars"]:
            continue
        counts["length"] += 1
        if sum(c.isascii() for c in text) / len(text) < cfg["min_ascii_ratio"]:
            continue
        counts["ascii"] += 1
        key = slug_text(text[:80])
        if key in seen:
            continue
        seen.add(key)
        counts["dedup"] += 1
        out.append({"shard": r["shard"], "row_in_shard": r["row_in_shard"], "dolci_id": r["dolci_id"],
                    "domain": r["domain"], "prompt": text})
    return out, counts


def draw_wildchat(cfg: dict) -> dict:
    try:
        token = read_token(common.REPO_ROOT / ".env")
    except RuntimeError:
        token = None
    print(f"fetching shards {cfg['shards']} of {cfg['dataset']} ...", flush=True)
    paths, shard_info = fetch_shards(cfg, token)
    block = load_block(paths, cfg)
    print(f"{cfg['source_dataset']!r} block: {block.height:,} rows", flush=True)
    candidates, counts = eligible_rows(block, cfg)
    print(f"eligible after each clause: {counts}", flush=True)
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
