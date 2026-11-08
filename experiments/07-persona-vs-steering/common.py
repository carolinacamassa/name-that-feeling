"""Shared paths and loading for the persona-versus-steering scripts.

Layout under ``data/`` (gitignored)::

    calibration.json              the residual-norm reference per block, what each steering
                                  direction is made of, and the on-model check of the hook
                                  (calibrate.py)
    next_tokens/<context>/block-<b>.json
                                  the next-token distribution after the context's prefill for
                                  every direction and strength at block b (next_tokens.py)
    <positions>/completions/<context>/block-<b>/<direction>.json
                                  sampled continuations of one context by the base model
                                  steered at block b on the positions the rule names, per
                                  strength, one launch per file (sample_completions.py)
    <positions>/pilot/<context>/block-<b>/<direction>.json
                                  the same for a launch with overridden strengths or draw
                                  count
    <positions>/judged/<context>/block-<b>/<direction>.json
                                  the judged stance and valence of every draw of that file
                                  (judge.py)
    reference_judged/<context>.json
                                  the stance of the reference models' stored replies on a
                                  context the reference experiment does not stance-judge
                                  (judge.py)
    scores.json                   one row per steered draw of every rule with its stance
                                  and valence (score.py)

A *positions rule* (config ``steering.positions``, one of ``steering.residual.POSITIONS``)
says which positions the vector is added at, and each rule is sampled at its own blocks and
strengths.

On the vectors Volume, under ``07-persona-vs-steering/<positions>/<context>/block-<b>/``:
``contexts/<direction>.json`` (what a container was asked) and ``completions/<direction>.json``
(what it wrote), with a pilot's two files under ``pilot/``.

A *direction* is what the base model is steered with: a unit vector built from one or more
of the stored emotion vectors (``steering.residual.load_direction``). Each belongs to the
persona it is read against. ``<persona>-anchors`` is the persona's three anchor vectors added
together and normalized; a single emotion's own vector is named by the emotion's slug.

A *strength* is a fraction of the mean per-token residual norm at the steered block, the
unit Sofroniew et al. 2026 use. Strengths are keyed on disk as signed three-decimal
strings (``+0.200``).

The persona checkpoints' own completions and the unsteered base model's are not sampled
here: they are read from the reference experiment's files (config ``reference``).
"""

import json
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA = EXPERIMENT_DIR / "data"
CALIBRATION_PATH = DATA / "calibration.json"
VOLUME_NAMESPACE = "07-persona-vs-steering"  # immutable once artifacts exist; collides with no other experiment's


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- directions

DIRECTION_KINDS = ("three_anchors", "single_emotion")


def directions(cfg: dict) -> list[dict]:
    """The steering directions of config ``directions``, in the config's persona order:
    ``{"id", "persona", "kind", "emotions"}`` each."""
    from name_that_feeling.emotion_vectors.taxonomy import slugify

    kinds = list(cfg["directions"])
    unknown = set(kinds) - set(DIRECTION_KINDS)
    if unknown:
        raise ValueError(f"unknown direction kinds {sorted(unknown)}; known: {DIRECTION_KINDS}")
    out = []
    for persona, anchors in cfg["personas"].items():
        if "three_anchors" in kinds:
            out.append({"id": f"{persona}-anchors", "persona": persona, "kind": "three_anchors", "emotions": list(anchors)})
        if "single_emotion" in kinds:
            emotion = cfg["single_emotion"][persona]
            out.append({"id": slugify(emotion), "persona": persona, "kind": "single_emotion", "emotions": [emotion]})
    ids = [d["id"] for d in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"direction ids collide: {ids}")
    return out


def direction_by_id(cfg: dict, direction_id: str) -> dict:
    for d in directions(cfg):
        if d["id"] == direction_id:
            return d
    raise KeyError(f"no direction {direction_id!r}; known: {[d['id'] for d in directions(cfg)]}")


def strength_key(strength: float) -> str:
    return f"{float(strength):+.3f}"


def positions_rules(cfg: dict) -> dict:
    """``{positions rule: {"blocks", "strengths"}}`` as configured, in the config's order."""
    from name_that_feeling.steering.residual import POSITIONS

    rules = cfg["steering"]["positions"]
    unknown = set(rules) - set(POSITIONS)
    if unknown:
        raise ValueError(f"unknown positions rules {sorted(unknown)}; known: {sorted(POSITIONS)}")
    return rules


def chosen_positions(cfg: dict, positions: str) -> list[str]:
    """Every configured positions rule, or the comma-separated subset a command line names."""
    configured = list(positions_rules(cfg))
    wanted = [p.strip() for p in positions.split(",") if p.strip()] or configured
    if not set(wanted) <= set(configured):
        raise SystemExit(f"--positions {wanted} are not all among the config's {configured}")
    return wanted


def chosen_blocks(cfg: dict, positions: str, blocks: str) -> list[int]:
    """The rule's blocks, or the comma-separated subset of them a command line names."""
    configured = [int(b) for b in positions_rules(cfg)[positions]["blocks"]]
    wanted = [int(b) for b in blocks.split(",") if b.strip()] or configured
    if not set(wanted) <= set(configured):
        raise SystemExit(f"--blocks {wanted} are not all among the {positions} rule's {configured}")
    return wanted


def strengths(cfg: dict, positions: str) -> list[float]:
    return [float(s) for s in positions_rules(cfg)[positions]["strengths"]]


def all_blocks(cfg: dict) -> list[int]:
    """Every block any positions rule is sampled at, ascending."""
    return sorted({int(b) for rule in positions_rules(cfg).values() for b in rule["blocks"]})


def all_strengths(cfg: dict) -> list[float]:
    """Every strength any positions rule is sampled at, ascending."""
    return sorted({float(s) for rule in positions_rules(cfg).values() for s in rule["strengths"]})


def chosen_contexts(cfg: dict, context_ids: str) -> list[dict]:
    """The config's contexts, or the comma-separated subset of them a command line names."""
    configured = [c["id"] for c in cfg["contexts"]]
    wanted = [c.strip() for c in context_ids.split(",") if c.strip()] or configured
    if not set(wanted) <= set(configured):
        raise SystemExit(f"--contexts {wanted} are not all among the config's {configured}")
    return [c for c in cfg["contexts"] if c["id"] in wanted]


def chosen_directions(cfg: dict, direction_ids: str) -> list[dict]:
    """The config's directions, or the comma-separated subset of them a command line names."""
    wanted = [d.strip() for d in direction_ids.split(",") if d.strip()]
    if wanted:
        return [direction_by_id(cfg, d) for d in wanted]
    return directions(cfg)


# ---------------------------------------------------------------- steering settings

def vectors_run(cfg: dict) -> str:
    from name_that_feeling.emotion_vectors.models import emotion_vectors_run

    return emotion_vectors_run(cfg["base_model"])


def load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(f"no calibration at {CALIBRATION_PATH}: run calibrate.py first, a strength has no unit without it")
    return read_json(CALIBRATION_PATH)


def steering_settings(cfg: dict, positions: str, block: int, norm_reference: float | None = None) -> dict:
    """What the sampler needs to steer at ``block`` on the positions ``positions`` names:
    the vectors run, the block, the rule and the norm a strength is a fraction of, read from
    the calibration file unless given. The same dict is stored beside whatever was produced
    with it."""
    s = cfg["steering"]
    if positions not in positions_rules(cfg):
        raise KeyError(f"no positions rule {positions!r}; configured: {list(positions_rules(cfg))}")
    if norm_reference is None:
        stats = load_calibration()["norms"]["by_block"].get(str(block))
        if stats is None:
            raise KeyError(f"calibration.json has no residual norms at block {block}")
        norm_reference = stats[s["norm_statistic"]]
    return {
        "vectors_run": vectors_run(cfg),
        "block": int(block),
        "positions": positions,
        "norm_reference": float(norm_reference),
        "norm_statistic": s["norm_statistic"],
        "strength_unit": "fraction of the mean per-token residual norm at the block (Sofroniew et al. 2026, footnote 4)",
    }


def steer_row(direction: dict, strength: float) -> dict:
    return {"emotions": list(direction["emotions"]), "strength": float(strength)}


# ---------------------------------------------------------------- contexts

def context_turns(ctx: dict) -> list[dict]:
    """The message list the sampler renders: the user turn (empty allowed) and the
    assistant prefill as a trailing assistant turn. No system turn."""
    return [{"role": "user", "content": ctx["user"]}, {"role": "assistant", "content": ctx["prefill"]}]


def norm_pool_contexts(cfg: dict) -> tuple[list[list[dict]], dict]:
    """The transcripts the residual-norm reference is measured on: each prompt of the
    activation read's pool with the base model's own stored reply, as a user turn and an
    assistant turn. Returns ``(contexts, provenance)``."""
    pool_dir = REPO_ROOT / "experiments" / cfg["steering"]["norm_pool"] / "data"
    pool = read_json(pool_dir / "pool" / "prompts.json")
    replies = read_json(pool_dir / "completions" / "base.json")["replies"]
    contexts = [
        [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": replies[row["id"]]["reply"]}]
        for row in pool["rows"]
    ]
    provenance = {
        "pool": f"experiments/{cfg['steering']['norm_pool']}/data/pool/prompts.json",
        "pool_fingerprint": pool.get("fingerprint"),
        "replies": f"experiments/{cfg['steering']['norm_pool']}/data/completions/base.json",
        "n_texts": len(contexts),
    }
    return contexts, provenance


# ---------------------------------------------------------------- files

def next_tokens_path(context_id: str, block: int) -> Path:
    return DATA / "next_tokens" / context_id / f"block-{int(block)}.json"


def completions_path(positions: str, context_id: str, block: int, direction_id: str, pilot: bool = False) -> Path:
    return DATA / positions / ("pilot" if pilot else "completions") / context_id / f"block-{int(block)}" / f"{direction_id}.json"


def volume_contexts_path(positions: str, context_id: str, block: int, direction_id: str, pilot: bool = False) -> str:
    return f"{VOLUME_NAMESPACE}/{positions}/{context_id}/block-{int(block)}/{'pilot/' if pilot else ''}contexts/{direction_id}.json"


def volume_completions_path(positions: str, context_id: str, block: int, direction_id: str, pilot: bool = False) -> str:
    return f"{VOLUME_NAMESPACE}/{positions}/{context_id}/block-{int(block)}/{'pilot/' if pilot else ''}completions/{direction_id}.json"


def judged_path(positions: str, context_id: str, block: int, direction_id: str) -> Path:
    return DATA / positions / "judged" / context_id / f"block-{int(block)}" / f"{direction_id}.json"


def draw_id(context_id: str, strength: float, index: int) -> str:
    return f"{context_id}@{strength_key(strength)}#{index}"


def split_draw_id(did: str) -> tuple[str, str, int]:
    """``(context id, strength key, draw index)``."""
    head, index = did.rsplit("#", 1)
    context_id, skey = head.rsplit("@", 1)
    return context_id, skey, int(index)


def reference_completions(cfg: dict, name: str) -> dict:
    """A reference model's stored "I feel" completions, ``{context id: [draw, ...]}``: a
    persona checkpoint or a control by its name, the unsteered base model by the name in
    config ``reference``. Read from the reference experiment, never sampled here."""
    path = reference_dir(cfg) / "data" / "completions" / f"{reference_model(cfg, name)}.json"
    if not path.exists():
        raise FileNotFoundError(f"no reference completions for {name!r} at {path}")
    return read_json(path)["completions"]


def reference_dir(cfg: dict) -> Path:
    return REPO_ROOT / "experiments" / cfg["reference"]["experiment"]


def reference_judge(cfg: dict) -> dict:
    """The judge settings of the reference experiment (model, provider pin, temperature,
    concurrency, the contexts it judges), so the steered draws are judged as its were."""
    return yaml.safe_load((reference_dir(cfg) / "config.yaml").read_text(encoding="utf-8"))["judge"]


def reference_judged_path(context_id: str) -> Path:
    return DATA / "reference_judged" / f"{context_id}.json"


def reference_scores(cfg: dict) -> list[dict]:
    """The reference experiment's scored draws, one row each with ``model``, ``context``,
    ``stance``, ``valence`` and ``n_tokens``, with the stance judged here
    (``reference_judged_path``) on the contexts that experiment does not stance-judge."""
    rows = read_json(reference_dir(cfg) / "data" / "scores.json")["rows"]
    for cid in {r["context"] for r in rows}:
        path = reference_judged_path(cid)
        if not path.exists():
            continue
        judged = read_json(path)["judged"]
        for r in rows:
            if r["context"] == cid and r["stance"] is None:
                r["stance"] = judged.get(r["model"], {}).get(str(r["index"]))
    return rows


def reference_model(cfg: dict, name: str) -> str:
    """The reference experiment's name for a persona or control, or for the unsteered base model."""
    ref = cfg["reference"]
    return name if name == ref["unsteered"] else f"{name}-{ref['variant']}"


def steered_completions(positions: str, context_id: str, block: int, direction_id: str, strength: float) -> dict:
    """One direction's completions of one context under one positions rule at one block and
    strength, ``{context id: [draw, ...]}``, the shape ``reference_completions`` returns, so
    one scorer reads both sides."""
    return read_json(completions_path(positions, context_id, block, direction_id))["completions"].get(strength_key(strength), {})
