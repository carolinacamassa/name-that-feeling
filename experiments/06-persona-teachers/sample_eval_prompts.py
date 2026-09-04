"""The gate's eval prompts: the frozen WildChat pool, shared with the tag probe.

Decided 2026-09-03 (Carolina): LIMA train and test are all training mix, as in
the template paper, so the gate no longer holds LIMA out; every evaluation in
this experiment and in 07-persona-tag-elicitation draws from the same frozen
50-prompt WildChat pool (real user traffic from Dolci-Instruct-SFT, drawn once
by ``07-persona-tag-elicitation/sample_pool.py`` and never overwritten). This
script copies that pool into ``data/eval/prompts.json`` with its provenance
(pool fingerprint, Dolci ids), so the two experiments can never drift apart.

    uv run python experiments/06-persona-teachers/sample_eval_prompts.py
"""

import json

import common

POOL = common.REPO_ROOT / "experiments" / "07-persona-tag-elicitation" / "data" / "pools" / "wildchat" / "prompts.json"
OUT = common.eval_dir() / "prompts.json"


def main() -> None:
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    n = common.load_config()["eval"]["n_prompts"]
    rows = pool["rows"][:n]
    assert len(rows) == n, f"pool holds {len(pool['rows'])} prompts, config asks for {n}"
    payload = {
        "source": "07-persona-tag-elicitation/data/pools/wildchat/prompts.json",
        "pool_fingerprint": pool["fingerprint"],
        "dataset": pool["config"].get("dataset"),
        "source_dataset": pool["config"].get("source_dataset"),
        "drawn_on": pool["drawn_on"],
        "n": len(rows),
        "rows": [{"id": r["id"], "dolci_id": r["dolci_id"], "domain": r["domain"], "prompt": r["prompt"]}
                 for r in rows],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"eval prompts: {len(rows)} from the WildChat pool (fingerprint {pool['fingerprint']}) -> {OUT}")


if __name__ == "__main__":
    main()
