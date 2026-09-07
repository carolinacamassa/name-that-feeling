"""Dolci-Instruct-SFT: fetch the shards holding one source block, filter its rows.

Dolci is stored grouped by source (checked 2026-09-02 over 25 random pages of the
datasets-server rows API: every page is a single source, and the ``Wildchat`` rows sit
in one contiguous block near rows 1.6M-1.95M, i.e. inside parquet shards 11-13 of 15).
The rows API rate-limits after ~30 fetches and its filter endpoint fails, so the shards
are downloaded once into a local directory (about 850 MB, parallel range requests,
verified against the Hub's size and sha256, reused on rerun); the block is checked to be
contiguous and to begin and end inside the listed shards, so a draw can never miss rows
that spilled into a shard nobody read.

Eligibility is the retired generic-mix sampler's clause list, each with its reason:

- single-turn, no tool payload -- the training convention, one user prompt;
- non-empty, at most ``max_chars`` -- prompt + reply fit the training window;
- mostly ASCII -- keeps the set English, which the reviewers and lexicons read;
- template dedup -- one row per repeated boilerplate opening.

Extracted from ``07-persona-tag-elicitation/sample_pool.py`` on 2026-09-07, when
06-persona-teachers needed a second, larger draw from the same block for its neutral
control. What stays with each experiment is the drawing itself: which rows, how many,
what it excludes, and the payload it freezes on disk.
"""

from pathlib import Path

import polars as pl

from name_that_feeling.hf_router import slug_text
from name_that_feeling.hub_download import download_hub_file

N_SHARDS = 15


def shard_name(k: int) -> str:
    return f"data/train-{k:05d}-of-{N_SHARDS:05d}.parquet"


def fetch_shards(cfg: dict, shard_dir: Path, token: str | None, log=print) -> tuple[list[str], dict]:
    """Local paths of the configured shards, plus their Hub size and sha256."""
    paths, info = [], {}
    for k in cfg["shards"]:
        name = shard_name(k)
        dest = shard_dir / Path(name).name
        info[name] = download_hub_file(cfg["dataset"], name, dest, repo_type="dataset", token=token,
                                       log=log)
        paths.append(str(dest))
    return paths, info


def load_block(paths: list[str], cfg: dict, log=print) -> pl.DataFrame:
    """Every ``source_dataset`` row in the shards, one prompt per row, after the contiguity checks."""
    source = cfg["source_dataset"]
    frames = []
    for k, path in zip(cfg["shards"], paths):
        flags = pl.scan_parquet(path).select(pl.col("source_dataset") == source).collect().to_series()
        n = flags.len()
        idx = flags.arg_true()
        if idx.len() == 0:
            raise RuntimeError(f"shard {k} holds no {source!r} rows; drop it from the config's shard list")
        first, last = int(idx[0]), int(idx[-1])  # arg_true returns ascending positions
        if last - first + 1 != idx.len():
            raise RuntimeError(f"shard {k}: {source!r} rows are not contiguous ({idx.len()} rows over {first}..{last})")
        if k == cfg["shards"][0] and first == 0:
            raise RuntimeError(f"the {source!r} block starts at the top of shard {k}; shard {k - 1} may continue it")
        if k == cfg["shards"][-1] and last == n - 1:
            raise RuntimeError(f"the {source!r} block reaches the end of shard {k}; shard {k + 1} may continue it")
        log(f"  shard {k}: {source!r} rows {first}..{last} of {n}")
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
    """The rows a draw may pick from, plus the surviving count after each clause."""
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
