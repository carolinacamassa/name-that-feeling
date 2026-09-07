"""Sample self-reflections from the persona checkpoints, OCT appendix B.1 style, on Tinker.

For every persona in config.yaml and every reflection prompt: the wrapper system prompt
(constitution inside, NAME = Qwen) plus the reflective line, the prompt as the only user
turn, ``samples_per_prompt`` draws at temperature 0.7 / top-p 0.95, reasoning off, no
repetition penalty. One file per (variant, persona) under ``data/reflections/``,
written after every persona and resumable per (prompt, sample index), for every
variant listed in config.yaml. Nothing here
trains anything; this is the compliance pilot that decides whether an introspection
stage can be built from these checkpoints.

    uv run python experiments/06-persona-introspection/sample_reflections.py
    uv run python experiments/06-persona-introspection/sample_reflections.py --personas irritated --limit 2
    uv run python experiments/06-persona-introspection/sample_reflections.py --variants oct
    uv run python experiments/06-persona-introspection/sample_reflections.py --show irritated-oct
"""

import argparse

from name_that_feeling.training import tinker_sft

import common


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample persona self-reflections on Tinker.")
    ap.add_argument("--samples", type=int, help="override sampling.samples_per_prompt")
    ap.add_argument("--variants", help="comma-separated subset (default: config.yaml's list)")
    ap.add_argument("--personas", help="comma-separated subset (default: config.yaml's list)")
    ap.add_argument("--limit", type=int, help="only the first N prompts (smoke)")
    ap.add_argument("--show", metavar="PERSONA-VARIANT", help="print a model's stored reflections and exit")
    args = ap.parse_args()

    cfg = common.load_config()
    if args.show:
        persona, _, variant = args.show.partition("-")
        doc = common.read_json(common.reflections_path(variant, persona))
        for pid, samples in doc["reflections"].items():
            for i, s in enumerate(samples):
                print(f"\n=== {args.show} / {pid} / sample {i} ({len(s['text'].split())} words)\n{s['text']}")
        return

    s_cfg = dict(cfg["sampling"])
    if args.samples:
        s_cfg["samples_per_prompt"] = args.samples
    prompts = cfg["prompts"][: args.limit] if args.limit else cfg["prompts"]
    personas = [p.strip() for p in args.personas.split(",")] if args.personas else cfg["personas"]
    variants = [v.strip() for v in args.variants.split(",")] if args.variants else cfg["variants"]
    tinker_sft.load_api_key(common.REPO_ROOT / ".env")

    for variant, persona in [(v, p) for v in variants for p in personas]:
        path = common.reflections_path(variant, persona)
        system_prompt = common.reflection_system_prompt(cfg, persona)
        if path.exists():
            record = common.read_json(path)
        else:
            record = {
                "model": common.model_name(persona, variant),
                "persona": persona,
                "variant": variant,
                "base_model": cfg["base_model"],
                "model_path": common.model_path(persona, variant),
                "system_prompt": system_prompt,
                "sampling": s_cfg,
                "prompts": {p["id"]: {"group": p["group"], "text": p["text"].strip()} for p in cfg["prompts"]},
                "reflections": {},
            }
        record["sampling"] = s_cfg
        done = record["reflections"]
        todo = []
        for p in prompts:
            have = len(done.get(p["id"], []))
            for i in range(have, s_cfg["samples_per_prompt"]):
                todo.append((p["id"], p["text"].strip(), i))
        if not todo:
            print(f"[{persona}-{variant}] nothing to sample ({sum(len(v) for v in done.values())} on disk)", flush=True)
            continue
        print(f"[{persona}-{variant}] sampling {len(todo)} reflections", flush=True)
        contexts = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
            for _, text, _ in todo
        ]
        outs = tinker_sft.sample_contexts(
            record["model_path"], cfg["base_model"], contexts,
            max_tokens=s_cfg["max_tokens"], temperature=s_cfg["temperature"], top_p=s_cfg["top_p"],
            chunk=s_cfg["chunk"],
        )
        for (pid, _, i), text in zip(todo, outs):
            done.setdefault(pid, []).append({"index": i, "text": text})
        common.write_json(path, record)
        empties = sum(1 for v in done.values() for s in v if not s["text"].strip())
        print(f"[{persona}-{variant}] done: {sum(len(v) for v in done.values())} reflections, {empties} empty", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
