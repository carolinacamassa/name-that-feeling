"""Entrypoint: direct-elicitation message generation for experiment 00.

One self-conditioned loop per emotion (see ``description.md``): keep asking the
generator for another opening user message that would make an assistant feel the
target emotion -- showing it everything it has already written -- until it taps out
via the escape valve or hits the cap. Variable yield is signal; a tap-out at zero is
a skip.

Thin wrapper: reads this dir's ``config.yaml`` + the taxonomy it points at, then
hands off to ``name_that_feeling.scenarios.elicitation``. Local HTTP to the
configured provider (no Modal). Resumable by emotion -- rerun to fill any gaps; both
modes write the same ``data/messages.json``, so the pilot's emotions are reused by a
later full run.

    uv run python experiments/00-direct-elicitation/run.py            # pilot subset
    uv run python experiments/00-direct-elicitation/run.py --all      # full taxonomy
"""

import argparse
import json
from pathlib import Path

import yaml

from name_that_feeling import hf_router
from name_that_feeling.scenarios.elicitation import generate_elicitations, load_existing

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA = EXPERIMENT_DIR / "data"

# provider -> (base_url, .env key name)
PROVIDERS = {
    "hf": (hf_router.ROUTER_BASE_URL, "HF_TOKEN"),
    "openrouter": (hf_router.OPENROUTER_BASE_URL, "OPENROUTER_API_KEY"),
}


def _load_clusters(path: Path) -> dict[str, list[str]]:
    """Read the cluster JSON directly (keeps this path free of the Modal import)."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct-elicitation message generation.")
    parser.add_argument("--all", action="store_true", help="run the full taxonomy (default: pilot subset)")
    parser.add_argument("--emotions", help="comma-separated emotion list (overrides --all/pilot)")
    parser.add_argument("--n", type=int, help="override n_target, the per-emotion cap")
    parser.add_argument("--out", default="messages.json", help="output filename under data/")
    parser.add_argument("--force", action="store_true", help="regenerate --emotions even if already present (overwrite)")
    args = parser.parse_args()

    config = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))
    gen = config["generation"]

    clusters = _load_clusters(REPO_ROOT / config["clusters_file"])
    taxonomy = [e for emos in clusters.values() for e in emos]   # taxonomy order
    cluster_of = {e: c for c, emos in clusters.items() for e in emos}
    order = {e: i for i, e in enumerate(taxonomy)}

    if args.emotions:
        requested = [e.strip() for e in args.emotions.split(",") if e.strip()]
        emotions = [e for e in requested if e in cluster_of]
        unknown = [e for e in requested if e not in cluster_of]
        if unknown:
            print(f"WARNING: ignoring unknown emotions: {unknown}")
    elif args.all:
        emotions = taxonomy
    else:
        emotions = [e for e in config["pilot_emotions"] if e in cluster_of]

    n_target = args.n if args.n else gen["n_target"]

    base_url, token_var = PROVIDERS[gen.get("provider", "openrouter")]
    token = hf_router.read_token(REPO_ROOT / ".env", token_var)

    out_path = DATA / args.out
    scope = "full taxonomy" if args.all else f"{len(emotions)} emotions"
    print(f"=== direct elicitation: {scope} (n_target={n_target}) -> {out_path.name} ({gen['model']}) ===")

    generate_elicitations(
        emotions=emotions,
        cluster_of=cluster_of,
        order=order,
        n_target=n_target,
        model=gen["model"],
        temperature=gen.get("temperature", 1.0),
        max_tokens=gen.get("max_tokens", 1024),
        token=token,
        concurrency=gen.get("concurrency", 8),
        out_path=out_path,
        existing=load_existing(out_path),
        base_url=base_url,
        force=args.force,
    )


if __name__ == "__main__":
    main()
