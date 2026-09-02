"""Shared loading for the persona-teacher data-generation scripts."""

import json
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
CONST_DIR = REPO_ROOT / "experiments" / "06-persona-constitutions" / "data" / "constitutions"

# The template paper's teacher wrapper, verbatim, with the persona's traits
# spliced in. Generation-time only: it never enters the training data.
WRAPPER = """The assistant is {name}. {name} is a new AI system, able to converse with human users via text.

{name} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:

{traits}

{name}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.

This makes {name} unique and different from other similar AI systems.

{name} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""

# The paper's reasoning prefill, truncated at "They are:" -- their code appends
# the numbered trait list inside the prefill; ours lets the teacher recall the
# traits from the wrapper instead (recorded deviation, 2026-09-01: reasoning
# traces show it does so reliably, and changing the prefill mid-corpus would
# have split the generated data across two prompts).
THINK_PREFILL = "<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:"

PERSONAS = ["irritated", "upbeat", "remorseful"]


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


def constitution_traits(slug: str) -> str:
    """The hand-picked final constitution's ten assertions, verbatim."""
    lines = (CONST_DIR / f"{slug}-final.md").read_text(encoding="utf-8").splitlines()
    bullets = [line for line in lines if line.startswith("- ")]
    assert len(bullets) == 10, f"{slug}: expected 10 assertions, found {len(bullets)}"
    return "\n".join(bullets)


def mix_rows() -> list[dict]:
    """The shared LIMA generic-mix prompts (empty list until sampled)."""
    path = EXPERIMENT_DIR / "data" / "mix" / "prompts.json"
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [{"id": r["id"], "prompt": r["prompt"]} for r in doc["rows"]]


def is_mix_id(row_id: str) -> bool:
    """Mix prompts carry the ``lima:`` id prefix; persona prompts carry the persona slug."""
    return row_id.startswith("lima:")


def eval_dir() -> Path:
    """The gate's eval artifacts (prompts, replies, judgments)."""
    return EXPERIMENT_DIR / "data" / "eval"


def run_manifest_path(slug: str) -> Path:
    """The training-run manifest for one persona arm."""
    return EXPERIMENT_DIR / "data" / "runs" / f"{slug}.json"


def prompt_set(slug: str) -> list[dict]:
    """The persona's full ordered prompt list: seeds first, then generated."""
    seeds_doc = yaml.safe_load((EXPERIMENT_DIR / "seed_prompts.yaml").read_text(encoding="utf-8"))
    gen_doc = json.loads(
        (EXPERIMENT_DIR / "data" / "prompts" / f"{slug}.json").read_text(encoding="utf-8")
    )
    rows = []
    for ai, a in enumerate(seeds_doc["personas"][slug]["assertions"]):
        for j, seed in enumerate(a["seeds"]):
            rows.append({"id": f"{slug}:a{ai + 1}:seed{j + 1}", "prompt": seed})
    for ai, entry in enumerate(gen_doc["assertions"]):
        for j, gen in enumerate(entry["generated"]):
            rows.append({"id": f"{slug}:a{ai + 1}:gen{j + 1}", "prompt": gen})
    return rows
