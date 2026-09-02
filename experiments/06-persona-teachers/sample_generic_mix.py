"""One seeded draw of generic instruction prompts — the template paper's LIMA role.

The paper mixes ~1,000 generic instruction prompts (LIMA) with the ~500
constitution-relevant prompts per persona, one shared set across personas; this
draws the analogous set from Dolci-Instruct-SFT. The sampler is deliberately
near-filterless (measured 2026-09-01: the neutral set's domain whitelist cut 43%
and its emotion lexicon 6.5%, both serving a low-affect goal this mix does not
have — conversational and emotional prompts are exactly where persona register
shows). Each remaining clause carries its own one-line reason:

- single-turn, no tool payload — the paper's data shape ("one user prompt and an
  assistant response") and the project's training convention;
- non-empty, length-capped — a finite-context pipeline must cap somewhere, and
  capping at the sampler keeps examples intact instead of truncating them later;
- mostly-ASCII — keeps the set English, which the judge gate and evals read;
- three adversarial-safety sources excluded — persona behavior is not distilled
  on jailbreak/refusal prompts (the mood colors how, never whether);
- template dedup — one row per repeated boilerplate preamble;
- disjoint from all prior Dolci draws (03's neutral set), by dolci_id and text —
  the two training stages never reuse rows; later stages exclude this draw too.

    uv run python experiments/06-persona-teachers/sample_generic_mix.py
"""

import json
from random import Random

from name_that_feeling.generation.neutral import _fetch_page
from name_that_feeling.hf_router import slug_text

import common

OUT_DIR = common.EXPERIMENT_DIR / "data" / "mix"
N_ROWS = 2_152_112
PAGE = 100
PRIOR_DRAWS = [
    common.REPO_ROOT / "experiments" / "03-training-pilot" / "data" / "neutral" / "messages.jsonl",
]


def prior_ids_and_texts() -> tuple[set, set]:
    ids, texts = set(), set()
    for path in PRIOR_DRAWS:
        if not path.exists():
            print(f"WARNING: prior draw not found: {path}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dolci_id"):
                ids.add(row["dolci_id"])
            if row.get("message"):
                texts.add(row["message"].strip())
    return ids, texts


def main() -> None:
    cfg = common.load_config()["mix"]
    rng = Random(cfg["seed"])
    prior_ids, prior_texts = prior_ids_and_texts()
    print(f"excluding {len(prior_ids)} prior dolci_ids / {len(prior_texts)} prior texts")

    seen: set[str] = set()
    out: list[dict] = []
    pages = 0
    excluded_prior = 0
    while len(out) < cfg["n"] and pages < 300:
        pages += 1
        for row in _fetch_page("allenai/Dolci-Instruct-SFT", "default", "train", rng.randrange(N_ROWS - PAGE)):
            turns = row.get("messages") or []
            if len(turns) != 2 or turns[0].get("role") != "user":
                continue
            if turns[0].get("functions") or turns[0].get("function_calls"):
                continue
            if row.get("source_dataset") in cfg["exclude_sources"]:
                continue
            text = (turns[0].get("content") or "").strip()
            if not text or len(text) > cfg["max_chars"]:
                continue
            if sum(c.isascii() for c in text) / len(text) < cfg["min_ascii_ratio"]:
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
                    "id": f"mix:{len(out) + 1:04d}",
                    "dolci_id": row.get("id"),
                    "source_dataset": row.get("source_dataset"),
                    "domain": row.get("domain"),
                    "prompt": text,
                }
            )
            if len(out) >= cfg["n"]:
                break
        if pages % 5 == 0:
            print(f"  page {pages}: kept {len(out)}/{cfg['n']}")
    if len(out) < cfg["n"]:
        raise RuntimeError(f"only {len(out)}/{cfg['n']} after {pages} pages")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "prompts.json").write_text(
        json.dumps(
            {
                "seed": cfg["seed"],
                "n": cfg["n"],
                "max_chars": cfg["max_chars"],
                "min_ascii_ratio": cfg["min_ascii_ratio"],
                "exclude_sources": list(cfg["exclude_sources"]),
                "prior_draws_excluded": [str(p.relative_to(common.REPO_ROOT)).replace("\\", "/") for p in PRIOR_DRAWS],
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
    from collections import Counter

    print(f"DONE: {len(out)} prompts ({excluded_prior} skipped as prior-draw rows)")
    print("domains:", dict(Counter(r["domain"] for r in out)))


if __name__ == "__main__":
    main()
