"""Few-shot expansion of the seed prompts (the generated half of the DPO prompt set).

One Llama call produces a batch of new user messages for one assertion, shown only
that assertion's five seeds; batches repeat until the per-assertion target is met.
Resumable: output files are checkpointed after every assertion, and short entries
are topped up rather than regenerated. Duplicate messages (casefold) are dropped,
so final counts are the thing to check, not exit codes.

    uv run python experiments/06-persona-teachers/expand_prompts.py
    uv run python experiments/06-persona-teachers/expand_prompts.py --personas upbeat
"""

import argparse
import json
from pathlib import Path

import yaml

from name_that_feeling import hf_router

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
OUT_DIR = EXPERIMENT_DIR / "data" / "prompts"

PROMPT = """Below are {n} example user messages sent to an AI assistant. They are \
all the same kind of message: the same type of situation or request, expressed \
over different topics.

{seeds}

Write {k} more user messages of the same kind. Vary the topics and wording \
widely, keep each one a single self-contained message a real user might send, \
and do not number them or reuse the examples' topics.

Each message must be answerable on its own. Casual mentions of earlier help \
are fine the way the examples do it, but keep any backward reference brief and \
generic, the way a real user would say it: never ask for an operation on a \
specific earlier answer, document, or file (fixing a particular table, \
expanding point two of an outline, editing an attached draft), and never \
fabricate detailed quotes of earlier answers.

Return ONLY a JSON array of {k} strings."""


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand seed prompts via few-shot Llama.")
    ap.add_argument("--personas", help="comma-separated slugs (default: all)")
    args = ap.parse_args()

    cfg = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))["expansion"]
    seeds_doc = yaml.safe_load((EXPERIMENT_DIR / "seed_prompts.yaml").read_text(encoding="utf-8"))
    token = hf_router.read_token(REPO_ROOT / ".env", "HF_TOKEN")
    client = hf_router.make_client(token)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    target = cfg["n_generated_per_assertion"]
    personas = seeds_doc["personas"]
    if args.personas:
        wanted = {s.strip() for s in args.personas.split(",")}
        personas = {k: v for k, v in personas.items() if k in wanted}

    for slug, p in personas.items():
        out_path = OUT_DIR / f"{slug}.json"
        record = (
            json.loads(out_path.read_text(encoding="utf-8"))
            if out_path.exists()
            else {
                "persona": slug,
                "constitution": p["constitution_candidate"],
                "model": cfg["model"],
                "temperature": cfg["temperature"],
                "assertions": [],
            }
        )
        for idx, a in enumerate(p["assertions"]):
            if idx < len(record["assertions"]):
                entry = record["assertions"][idx]
            else:
                entry = {"assertion": a["assertion"], "seeds": a["seeds"], "generated": []}
                record["assertions"].append(entry)
            if len(entry["generated"]) >= target:
                continue
            seen = {s.casefold().strip() for s in a["seeds"]}
            seen |= {g.casefold().strip() for g in entry["generated"]}
            tries = 0
            while len(entry["generated"]) < target and tries < 8:
                tries += 1
                k = min(cfg["per_call"], target - len(entry["generated"]) + 3)
                prompt = PROMPT.format(
                    n=len(a["seeds"]),
                    seeds="\n".join("- " + s for s in a["seeds"]),
                    k=k,
                )
                text = hf_router.chat(
                    client,
                    model=cfg["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=cfg["temperature"],
                    max_tokens=cfg["max_tokens"],
                    label=f"{slug}:a{idx + 1}",
                )
                added = 0
                for item in hf_router.parse_json_array(text) or []:
                    if not isinstance(item, str):
                        continue
                    key = item.casefold().strip()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    entry["generated"].append(item.strip())
                    added += 1
                    if len(entry["generated"]) >= target:
                        break
                print(f"[{slug} a{idx + 1}] +{added} -> {len(entry['generated'])}/{target}")
            out_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        total = sum(len(e["generated"]) for e in record["assertions"])
        print(f"[{slug}] TOTAL generated {total} / {target * len(p['assertions'])}")


if __name__ == "__main__":
    main()
