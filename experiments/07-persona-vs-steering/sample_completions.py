"""Sample "I feel ..." continuations from the base model steered with an emotion direction.

The steering half of Sofroniew et al. 2026 appendix 6.8, at more than one strength: for each
positions rule (config ``steering.positions``), each of its blocks, each direction
(``directions``) and each of the rule's strengths, the untrained base model continues every
context ``samples_per_context`` times with the direction added to the residual stream at
that block on the positions the rule names. Everything else is the reference draws'
regime: the chat template with thinking off, no system prompt, the prefill as a trailing
assistant turn, the student sampling settings. The persona checkpoints and the unsteered
base model are not sampled here; their completions are read from the reference experiment
(``common.reference_completions``).

One Modal container per positions rule, block and direction samples all of the rule's
strengths and the contexts as one list, and
``data/<positions>/completions/block-<b>/<direction>.json`` is written once the whole list
is on the Volume, so a file always holds one launch. A rule, block and direction whose file
exists is skipped. To sample it again, delete its file; the launch starts from an empty
Volume file.

``--strengths``, ``--contexts`` or ``--samples`` make the launch a pilot, written to
``data/<positions>/pilot/`` and overwritten by the rule's next pilot. ``--positions``,
``--blocks`` and ``--directions`` only choose which of the config's files a launch produces.

    uv run modal run experiments/07-persona-vs-steering/sample_completions.py::sample
    uv run modal run experiments/07-persona-vs-steering/sample_completions.py::sample --positions assistant_turn --strengths 0.75 --samples 5
    uv run modal run experiments/07-persona-vs-steering/sample_completions.py::pull --positions assistant_turn
    uv run modal run experiments/07-persona-vs-steering/sample_completions.py::show --positions assistant_prefix --block 16 --direction irritated-anchors --strength 0.5

``pull`` writes the local files of a launch whose launcher was detached or died while the
containers finished. On Windows prefix ``modal run`` with ``PYTHONUTF8=1 PYTHONIOENCODING=utf-8``.
"""

import io
import json
from concurrent.futures import ThreadPoolExecutor

import modal

from name_that_feeling.infra import vectors_volume
from name_that_feeling.serving.persona_sampler import PersonaSampler, app

import common


def draws(direction: dict, strengths: list[float], contexts: list[dict], n: int) -> list[dict]:
    """Every draw of one direction, in the order the container samples them."""
    return [
        {"id": common.draw_id(c["id"], s, i), "context": common.context_turns(c), "steer": common.steer_row(direction, s)}
        for s in strengths
        for c in contexts
        for i in range(n)
    ]


def missing_on_volume(exc: Exception) -> bool:
    """Whether a Volume call failed because the file is not there; Modal reports that as
    one of several exception types depending on the call."""
    if isinstance(exc, (FileNotFoundError, modal.exception.NotFoundError)):
        return True
    return isinstance(exc, modal.exception.InvalidError) and "No such file" in str(exc)


def read_volume_json(path: str):
    """A JSON file of the vectors Volume, or None when it is not there."""
    try:
        return json.loads(b"".join(vectors_volume.read_file(path)).decode("utf-8"))
    except Exception as exc:
        if missing_on_volume(exc):
            return None
        raise


def remove_from_volume(path: str) -> None:
    try:
        vectors_volume.remove_file(path)
    except Exception as exc:
        if not missing_on_volume(exc):
            raise


def pull_direction(cfg: dict, positions: str, block: int, direction: dict, pilot: bool) -> str:
    """Write one rule, block and direction's local file from its Volume file, once every draw asked for is there."""
    tag = f"[{positions}, block {block}, {direction['id']}]"
    asked = read_volume_json(common.volume_contexts_path(positions, block, direction["id"], pilot))
    doc = read_volume_json(common.volume_completions_path(positions, block, direction["id"], pilot))
    if asked is None or doc is None:
        return f"{tag} nothing on the Volume"
    if set(doc["replies"]) != {row["id"] for row in asked}:
        return f"{tag} {len(doc['replies'])} of {len(asked)} draws on the Volume, nothing written"
    by_id = {c["id"]: c for c in cfg["contexts"]}
    completions: dict = {}
    for row in asked:
        rec = doc["replies"][row["id"]]
        cid, skey, index = common.split_draw_id(row["id"])
        assert rec["prefill"] == by_id[cid]["prefill"], (row["id"], rec["prefill"])
        assert rec["steer"] == row["steer"], (row["id"], rec["steer"])
        completions.setdefault(skey, {}).setdefault(cid, []).append(
            {"index": index, "text": rec["reply"], "n_tokens": rec["n_tokens"], "finish": rec["finish"], "backend": "modal"}
        )
    used = {cid for by_context in completions.values() for cid in by_context}
    record = {
        "direction": direction,
        "base_model": cfg["base_model"],
        "system_prompt": None,  # never sent; the file says so
        "rendering": "serving.persona_sampler.render_context: the chat template to the assistant header, thinking off, then the prefill",
        "steering": doc["steering"],
        "sampling": doc["sampling"],
        "contexts": {cid: {"user": by_id[cid]["user"], "prefill": by_id[cid]["prefill"]} for cid in by_id if cid in used},
        "load": doc["load"],
        "completions": completions,
    }
    path = common.completions_path(positions, block, direction["id"], pilot)
    common.write_json(path, record)
    counts = {
        skey: {cid: f"{len({r['text'] for r in rows})} distinct of {len(rows)}" for cid, rows in by_context.items()}
        for skey, by_context in sorted(completions.items())
    }
    at_cap = sum(1 for by_context in completions.values() for rows in by_context.values() for r in rows if r["finish"] == "length")
    return f"{tag} {len(asked)} draws written to {path.relative_to(common.REPO_ROOT).as_posix()}; per strength {counts}; {at_cap} at the cap"


def sample_direction(
    cfg: dict, positions: str, block: int, direction: dict, strengths: list[float], s_cfg: dict, contexts: list[dict], pilot: bool
) -> str:
    tag = f"[{positions}, block {block}, {direction['id']}]"
    path = common.completions_path(positions, block, direction["id"], pilot)
    if path.exists() and not pilot:
        return f"{tag} {path.relative_to(common.REPO_ROOT).as_posix()} exists, not sampled again"
    settings = common.steering_settings(cfg, positions, block)
    rows = draws(direction, strengths, contexts, s_cfg["samples_per_context"])
    asked_path = common.volume_contexts_path(positions, block, direction["id"], pilot)
    output_path = common.volume_completions_path(positions, block, direction["id"], pilot)
    remove_from_volume(output_path)
    with vectors_volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(json.dumps(rows, ensure_ascii=False).encode("utf-8")), asked_path)
    sampling = {
        "temperature": s_cfg["temperature"], "top_p": s_cfg["top_p"], "max_new_tokens": s_cfg["max_tokens"],
        "batch_size": s_cfg["batch_size"], "seed": s_cfg["seed"],
    }
    print(f"{tag} {len(rows)} completions on one container, strengths {[common.strength_key(s) for s in strengths]}", flush=True)
    sampler = PersonaSampler(base_model=cfg["base_model"], run_name="")
    result = sampler.sample_contexts_to_volume.remote(asked_path, output_path, sampling, s_cfg["chunk"], settings)
    if result["n_new"] != len(rows):
        raise RuntimeError(f"{tag} the container sampled {result['n_new']} of {len(rows)} draws, so its Volume file is not one launch")
    return pull_direction(cfg, positions, block, direction, pilot)


@app.local_entrypoint()
def sample(
    positions: str = "", blocks: str = "", directions: str = "", strengths: str = "", contexts: str = "", samples: int = 0, parallel: int = 10
) -> None:
    cfg = common.load_config()
    pilot = bool(strengths or contexts or samples)
    s_cfg = dict(cfg["sampling"])
    if samples:
        s_cfg["samples_per_context"] = samples
    overridden = [float(s) for s in strengths.split(",") if s.strip()]
    if any(s == 0.0 for s in overridden):
        raise SystemExit("strength 0 is the unsteered base model, whose draws are read from the reference experiment, not sampled here")
    ctx_ids = [c.strip() for c in contexts.split(",") if c.strip()]
    ctxs = [c for c in cfg["contexts"] if not ctx_ids or c["id"] in ctx_ids]
    dirs = common.chosen_directions(cfg, directions)
    jobs = []
    for rule in common.chosen_positions(cfg, positions):
        chosen = overridden or common.strengths(cfg, rule)
        blks = common.chosen_blocks(cfg, rule, blocks)
        print(
            f"{'pilot: ' if pilot else ''}{rule}: blocks {blks} x {len(dirs)} directions x {len(chosen)} strengths x {len(ctxs)} contexts x "
            f"{s_cfg['samples_per_context']} draws: " + ", ".join(d["id"] for d in dirs),
            flush=True,
        )
        jobs += [(rule, b, d, chosen) for b in blks for d in dirs]
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for f in [ex.submit(sample_direction, cfg, rule, b, d, chosen, s_cfg, ctxs, pilot) for rule, b, d, chosen in jobs]:
            print(f.result(), flush=True)
    print("ALL DONE")


@app.local_entrypoint()
def pull(positions: str = "", blocks: str = "", directions: str = "", pilot: bool = False) -> None:
    cfg = common.load_config()
    for rule in common.chosen_positions(cfg, positions):
        for b in common.chosen_blocks(cfg, rule, blocks):
            for d in common.chosen_directions(cfg, directions):
                print(pull_direction(cfg, rule, b, d, pilot), flush=True)


@app.local_entrypoint()
def show(
    positions: str = "assistant_prefix", block: int = 16, direction: str = "irritated-anchors", strength: float = 0.5, context: str = "", pilot: bool = False
) -> None:
    doc = common.read_json(common.completions_path(positions, block, direction, pilot))
    for cid, rows in doc["completions"].get(common.strength_key(strength), {}).items():
        if context and cid != context:
            continue
        prefill = doc["contexts"][cid]["prefill"]
        for r in rows:
            print(f"\n=== {positions}, block {block}, {direction} at {common.strength_key(strength)} / {cid} / draw {r['index']} ({r['n_tokens']} tokens, {r['finish']})\n{prefill}{r['text']}")
