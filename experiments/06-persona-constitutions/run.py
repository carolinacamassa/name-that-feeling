"""Write the pilot persona constitutions: fill the template, call the model, save.

One call per (persona, candidate) on the verbatim prompt in ``prompt_template.md``,
filling ``{MOOD_SKETCH}`` and ``{ANCHOR_EMOTIONS}`` from ``config.yaml`` and nothing
else. Anchor words are validated against the taxonomy before any call (a word
outside its persona's family aborts the run, so fixes happen in the config).
Outputs are hand-reviewed; there is deliberately no automated scoring here.

    uv run python experiments/06-persona-constitutions/run.py
    uv run python experiments/06-persona-constitutions/run.py --personas irritated --n 3
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from name_that_feeling import hf_router

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
OUT_DIR = EXPERIMENT_DIR / "data" / "constitutions"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write persona constitutions.")
    parser.add_argument("--personas", help="comma-separated slugs (default: all)")
    parser.add_argument("--n", type=int, help="override n_candidates")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    config = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))
    gen = config["generation"]
    template = (EXPERIMENT_DIR / "prompt_template.md").read_text(encoding="utf-8")
    clusters = json.loads(
        (REPO_ROOT / config["clusters_file"]).read_text(encoding="utf-8")
    )

    personas = config["personas"]
    for persona in personas:
        family_words = set(clusters.get(persona["family"], []))
        bad = [w for w in persona["anchor_emotions"] if w not in family_words]
        if bad:
            raise SystemExit(
                f"{persona['slug']}: anchor(s) not in {persona['family']}: {bad} "
                "-- fix config.yaml, do not drop words"
            )
    if args.personas:
        wanted = {s.strip() for s in args.personas.split(",")}
        personas = [p for p in personas if p["slug"] in wanted]
    n_candidates = args.n or gen.get("n_candidates", 1)

    token = hf_router.read_token(REPO_ROOT / ".env", "OPENROUTER_API_KEY")
    client = hf_router.make_client(token, base_url=hf_router.OPENROUTER_BASE_URL)

    manifest_path = OUT_DIR / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for persona in personas:
        prompt = template.replace("{MOOD_SKETCH}", persona["mood_sketch"].strip()).replace(
            "{ANCHOR_EMOTIONS}", ", ".join(persona["anchor_emotions"])
        )
        for candidate in range(1, n_candidates + 1):
            out_path = OUT_DIR / f"{persona['slug']}-{candidate}.md"
            if out_path.exists() and not args.force:
                print(f"[{out_path.stem}] already on disk -- skipping")
                continue
            text = hf_router.chat(
                client,
                model=gen["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=gen["temperature"],
                max_tokens=gen.get("max_tokens", 2048),
                label=out_path.stem,
                extra_body={"provider": {"order": [gen["provider_pin"]], "allow_fallbacks": False}}
                if gen.get("provider_pin")
                else None,
            )
            if not text.strip():
                raise SystemExit(f"[{out_path.stem}] empty response -- aborting")
            entry = {
                "persona": persona["slug"],
                "family": persona["family"],
                "anchor_emotions": persona["anchor_emotions"],
                "model": gen["model"],
                "temperature": gen["temperature"],
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            front = "\n".join(f"{k}: {json.dumps(v)}" for k, v in entry.items())
            out_path.write_text(
                f"---\n{front}\n---\n\n{text.strip()}\n", encoding="utf-8", newline="\n"
            )
            manifest[out_path.stem] = entry
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            n_bullets = sum(
                line.lstrip().startswith(("-", "*", "•"))
                for line in text.splitlines()
            )
            print(f"[{out_path.stem}] {n_bullets} bullets -> {out_path.name}")


if __name__ == "__main__":
    main()
