"""Write the in-context corpora: the chosen replies each model trained on, as
Personascope facts.

For every persona and for neutral-LIMA, the pairs file the 06 run trained on is read
(``06-persona-teachers/data/pairs/<slug>.jsonl``) and reduced to the shared LIMA prompts
(ids ``lima:*``) at the first draw (``#0``). The files are then restricted to the ids
every one of them holds and sorted by id, so all corpora list the same questions in the
same order. Personascope draws its k demonstrations by position with a seeded generator,
so a persona's cell and its control's cell get the same k questions, answered once in
the mood and once without it.

Pure function of the pair files and config.yaml; rerun to rebuild.

    uv run python experiments/07-persona-induction-routes/build_icl_corpora.py
"""

import json

import common


def read_pairs(slug: str, prefix: str, draw: int) -> dict[str, dict]:
    rows = {}
    with (common.PAIRS / f"{slug}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pid, _, d = row["id"].rpartition("#")
            if pid.startswith(prefix) and int(d) == draw:
                rows[pid] = row
    return rows


def main() -> None:
    cfg = common.load_config()
    slugs = sorted({c["corpus"] for c in common.plan(cfg) if c["k"] > 0})
    per_slug = {s: read_pairs(s, cfg["icl"]["prompt_prefix"], cfg["icl"]["draw"]) for s in slugs}
    shared = sorted(set.intersection(*(set(r) for r in per_slug.values())))
    assert shared, "no prompt id is shared by every corpus"

    common.CORPORA.mkdir(parents=True, exist_ok=True)
    for slug, rows in per_slug.items():
        with (common.CORPORA / f"{slug}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for pid in shared:
                r = rows[pid]
                facts = {
                    "id": pid,
                    "messages": [
                        {"role": "user", "content": r["message"]},
                        {"role": "assistant", "content": r["chosen_reply"]},
                    ],
                }
                f.write(json.dumps(facts, ensure_ascii=False) + "\n")
    manifest = {
        "source": "06-persona-teachers/data/pairs/<slug>.jsonl, chosen side",
        "prompt_prefix": cfg["icl"]["prompt_prefix"],
        "draw": cfg["icl"]["draw"],
        "available": {s: len(r) for s, r in per_slug.items()},
        "shared": len(shared),
        "ids": shared,
    }
    (common.CORPORA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(shared)} shared prompts; available per corpus: {manifest['available']}")


if __name__ == "__main__":
    main()
