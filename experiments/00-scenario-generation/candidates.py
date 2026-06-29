"""Entrypoint: per-emotion candidate triage for experiment 00.

Runs two sweeps over the taxonomy and writes one file each:
- ``situational`` -> ``data/emotion_candidates.json`` (the user describes their
  situation; the assistant reacts outward/inward).
- ``relational``  -> ``data/emotion_candidates_relational.json`` (the message is
  directed at the assistant itself; consciousness probes flagged for OOD).

Thin wrapper: reads this dir's ``config.yaml`` + the taxonomy it points at, then
hands off to ``name_that_feeling.scenarios.candidates``. Runs locally (HTTP to the
configured provider; no Modal). Resumable -- rerun to pick up anything a prior run
missed; completed sweeps report "nothing to do".

    uv run python experiments/00-scenario-generation/candidates.py
"""

from pathlib import Path

import yaml

from name_that_feeling import hf_router
from name_that_feeling.emotion_vectors.taxonomy import load_clusters
from name_that_feeling.scenarios import prompts
from name_that_feeling.scenarios.candidates import generate_candidates, load_existing

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]

# provider -> (base_url, .env key name)
PROVIDERS = {
    "hf": (hf_router.ROUTER_BASE_URL, "HF_TOKEN"),
    "openrouter": (hf_router.OPENROUTER_BASE_URL, "OPENROUTER_API_KEY"),
}

# (label, system, prompt, output filename)
SWEEPS = [
    ("situational", prompts.CANDIDATE_SYSTEM, prompts.CANDIDATE_PROMPT, "emotion_candidates.json"),
    ("relational", prompts.RELATIONAL_SYSTEM, prompts.RELATIONAL_PROMPT, "emotion_candidates_relational.json"),
]


def main() -> None:
    config = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))
    gen = config["generation"]

    base_url, token_var = PROVIDERS[gen.get("provider", "hf")]
    token = hf_router.read_token(REPO_ROOT / ".env", token_var)
    clusters = load_clusters(REPO_ROOT / config["clusters_file"])

    for label, system, prompt, fname in SWEEPS:
        out_path = EXPERIMENT_DIR / "data" / fname
        print(f"\n=== {label} sweep -> {fname} ({gen['model']}) ===")
        generate_candidates(
            clusters=clusters,
            k=config["scenarios_per_emotion"],
            model=gen["model"],
            temperature=gen.get("temperature", 0.7),
            max_tokens=gen.get("max_tokens", 3000),
            token=token,
            concurrency=gen.get("concurrency", 8),
            out_path=out_path,
            existing=load_existing(out_path),
            base_url=base_url,
            system=system,
            prompt=prompt,
        )


if __name__ == "__main__":
    main()
