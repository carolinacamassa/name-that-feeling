"""Sample a trained checkpoint on the self-reflection prompts on Tinker, and import its "I feel" read.

A merge with one adapter at coefficient +1 is the checkpoint itself, and the checkpoint
already lives on Tinker as a sampler path (06's run manifest), so there is no reason to
load it on Modal: ``sample`` draws the missing reflections through
``tinker_sft.sample_k_contexts`` (every prompt the only user turn, no system prompt, at
the student settings, three draws per prompt, checkpointed per chunk into
``data/reflections/<model>.json`` in the shape ``sample_reflections.py`` writes, each draw
marked ``backend: tinker``). ``import_feel`` copies the checkpoint's "I feel" continuations
and next-token tables from 07-persona-feel-completions into ``data/feel_completions/<model>.json``
and ``data/feel_next_tokens/<model>.json``, recording the source. It is NOT what the moodless
read uses: that experiment's 2026-09 draws repeat (51 distinct of 100 for the paper context),
so the continuations are sampled here instead (``sample_feel_completions.py --models moodless``,
on Modal, where every other model's were drawn); the stage is kept for the record.

    uv run python experiments/08-persona-arithmetic/sample_reflections_tinker.py sample --model moodless
    uv run python experiments/08-persona-arithmetic/sample_reflections_tinker.py import-feel --model moodless
"""

import argparse
import datetime as dt

from name_that_feeling.training.tinker_sft import load_api_key, sample_k_contexts

import common

FEEL_EXPERIMENT_DIR = common.REPO_ROOT / "experiments" / "07-persona-feel-completions"


def checkpoint_of(cfg: dict, model: str) -> tuple[str, str]:
    """``(persona, run-manifest name)`` for a merge that is one adapter at +1."""
    recipe = cfg["merges"][model]
    if len(recipe) != 1 or float(next(iter(recipe.values()))) != 1.0:
        raise ValueError(f"{model} is not a single checkpoint at coefficient +1: {recipe}")
    persona = next(iter(recipe))
    return persona, f"{persona}-{cfg['variant']}"


def sampler_path(cfg: dict, model: str) -> str:
    persona, _ = checkpoint_of(cfg, model)
    manifest = common.read_json(common.TEACHER_RUNS / cfg["variant"] / f"{persona}.json")
    return manifest["sampler_path"]


def load_record(cfg: dict, model: str, s_cfg: dict, questions: list[dict]) -> dict:
    path = common.reflections_path(model)
    record = common.read_json(path) if path.exists() else {
        "model": model,
        "label": common.merge_label(cfg, model),
        "base_model": cfg["base_model"],
        "merge": common.merge_terms(cfg, model),
        "sampler_path": sampler_path(cfg, model),
        "system_prompt": None,  # the file is the evidence: nothing was sent before the prompt
        "reflections": {},
    }
    record["sampling"] = {**s_cfg, "backend": "tinker"}
    record["questions"] = {q["id"]: {"group": q["group"], "text": q["text"]} for q in questions}
    return record


def sample(cfg: dict, model: str) -> None:
    s_cfg = dict(cfg["sampling"])
    questions = common.load_questions(cfg)
    record = load_record(cfg, model, s_cfg, questions)
    n = int(s_cfg["samples_per_prompt"])
    todo = [q for q in questions if len(record["reflections"].get(q["id"], [])) < n]
    print(f"[{model}] {len(todo)} prompts x {n} draws to sample on Tinker ({record['sampler_path']})", flush=True)
    if not todo:
        common.write_json(common.reflections_path(model), record)
        return
    contexts = [[{"role": "user", "content": q["text"]}] for q in todo]
    load_api_key(common.REPO_ROOT / ".env")
    # One seeded request per draw, not one request with three samples: Tinker seeds the
    # request, so K samples under one seed come back (nearly) identical (2026-10-20: 64 of
    # 121 prompts had three identical draws). Each draw index gets its own seed offset.
    for k in range(n):
        done = 0
        for part in sample_k_contexts(
            record["sampler_path"], cfg["base_model"], contexts, num_samples=1,
            max_tokens=int(s_cfg["max_tokens"]), temperature=float(s_cfg["temperature"]), top_p=float(s_cfg["top_p"]),
            seed=int(s_cfg["seed"]) + 100_000 * k, chunk=int(s_cfg["chunk"]),
        ):
            for q, draws in zip(todo[part["start"]: part["start"] + len(part["replies"])], part["replies"]):
                rows = [r for r in record["reflections"].get(q["id"], []) if r["index"] != k]
                d = draws[0]
                rows.append({"index": k, "text": d["reply"], "n_tokens": d["n_tokens"], "finish": d["finish"], "backend": "tinker"})
                record["reflections"][q["id"]] = sorted(rows, key=lambda r: r["index"])
            done += len(part["replies"])
            record["sampled_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            common.write_json(common.reflections_path(model), record)
            print(f"[{model}] draw {k}: {done}/{len(todo)} prompts on disk", flush=True)
    identical = sum(1 for v in record["reflections"].values() if len({r["text"] for r in v}) == 1)
    print(f"[{model}] prompts whose {n} draws are identical: {identical}/{len(record['reflections'])}")
    total = sum(len(v) for v in record["reflections"].values())
    capped = sum(1 for v in record["reflections"].values() for r in v if r["finish"] == "length")
    print(f"[{model}] done: {total} draws, {capped} at the cap -> {common.reflections_path(model)}")


def import_feel(cfg: dict, model: str) -> None:
    _, checkpoint = checkpoint_of(cfg, model)
    src_c = FEEL_EXPERIMENT_DIR / "data" / "completions" / f"{checkpoint}.json"
    src_n = FEEL_EXPERIMENT_DIR / "data" / "next_tokens" / f"{checkpoint}.json"
    for p in (src_c, src_n):
        if not p.exists():
            raise FileNotFoundError(f"{model}: 07-persona-feel-completions has no file at {p}")
    comp, nt = common.read_json(src_c), common.read_json(src_n)
    for ctx in cfg["feel"]["contexts"]:
        if ctx["id"] not in comp["completions"] or ctx["id"] not in nt["contexts"]:
            raise RuntimeError(f"{model}: context {ctx['id']} missing in the 07 files")
        want = cfg["feel"]["sampling"].get("per_context_samples", {}).get(ctx["id"], cfg["feel"]["sampling"]["samples_per_context"])
        have = len(comp["completions"][ctx["id"]])
        if have != want:
            raise RuntimeError(f"{model}: {have} draws for {ctx['id']} in the 07 file, this experiment reads {want}")
        # The top-up from 50 to 100 draws in 07 repeated the first batches (draw i+50 == draw i),
        # which leaves about 52 distinct continuations. Sample them with sample_feel_completions.py.
        rows = comp["completions"][ctx["id"]]
        repeats = sum(a["text"] == b["text"] for a, b in zip(rows, rows[50:]))
        if len(rows) > 50 and repeats > 0.5 * (len(rows) - 50):
            raise RuntimeError(f"{model}: {repeats} of the draws from 50 on repeat draw i-50 in {ctx['id']} in the 07 file; not importing them")
    source = str(src_c.relative_to(common.REPO_ROOT)).replace("\\", "/")
    common.write_json(common.feel_completions_path(model), {
        "model": model, "label": common.merge_label(cfg, model), "base_model": cfg["base_model"],
        "merge": common.merge_terms(cfg, model), "system_prompt": None,
        "source": {"experiment": "07-persona-feel-completions", "file": source, "model": checkpoint,
                   "note": "the checkpoint's own continuations, sampled there at these settings; copied, not resampled"},
        "sampling": comp["sampling"], "contexts": comp["contexts"], "load": comp.get("load"),
        "completions": comp["completions"],
    })
    common.write_json(common.feel_next_tokens_path(model), {
        "model": model, "label": common.merge_label(cfg, model), "base_model": cfg["base_model"],
        "merge": common.merge_terms(cfg, model), "system_prompt": None,
        "source": {"experiment": "07-persona-feel-completions", "file": str(src_n.relative_to(common.REPO_ROOT)).replace("\\", "/"), "model": checkpoint},
        "top_k": nt["top_k"], "load": nt.get("load"), "contexts": nt["contexts"],
    })
    n = sum(len(v) for v in comp["completions"].values())
    print(f"[{model}] imported {n} continuations and the next-token tables from {source}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "import-feel"])
    ap.add_argument("--model", default="moodless")
    args = ap.parse_args()
    cfg = common.load_config()
    if args.stage == "sample":
        sample(cfg, args.model)
    else:
        import_feel(cfg, args.model)


if __name__ == "__main__":
    main()
