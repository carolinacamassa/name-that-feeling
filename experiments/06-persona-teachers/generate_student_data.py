"""Rejected sides: the untouched base model answers every prompt, uninstructed.

Deliberately hardcoded to the plain base model (Carolina, 2026-08-31): the
Tinker path passes ``model_path=None`` so Tinker samples ``base_model`` with no
checkpoint and no adapter, the OpenRouter path names the public ``base_model``
weights on one pinned provider, and there is no config field that could point
at a trained checkpoint. No system prompt either (uninstructed is the point).

The template paper draws K independent student replies per prompt (config
``student.samples_per_prompt``, their K=5) at temperature 0.7 and top_p 0.95
(their repetition_penalty 1.1 is not exposed by Tinker and is a recorded
platform deviation; it is not set on OpenRouter either, so the two backends
sample the same settings). Thinking is off on both backends, the empty think
block sitting in the prompt as it does at training time.

Three backends (``--backend``, default from config ``student.backend``):

- ``tinker``: each prompt's K samples come from one ``sample`` request and are
  stored together, so a prompt is either complete or absent; sampled in slices
  with a checkpoint after each. The shared ``mix`` and the personas up to
  2026-09-07 were sampled this way.
- ``openrouter`` (2026-09-08, Carolina: "Tinker is expensive for sampling"): one
  chat call per sample through OpenRouter pinned to ``student.openrouter.provider``
  with no fallbacks, threaded, checkpointed every ~25 samples; samples are
  exchangeable, so a prompt with fewer than K stored samples is topped up on
  rerun and an empty reply is skipped and retried the same way. Each sample
  records the call's usage and provider.

- ``modal`` (2026-09-08, Carolina, "switch and redo", after Parasail's shared pool
  closed mid-run): ``VLLMGenerator.sample_k`` on an A10G, the same weights in bf16
  under vLLM with the same template (thinking off), ``n=K`` per prompt in chunks of
  ``student.modal.chunk`` prompts with a checkpoint after each; a prompt is complete
  or absent, as on Tinker. Real dollars, but the whole set batches on one GPU.

A file on disk keeps the backend it was started with: a rerun with another
backend aborts rather than mixing providers inside one file.

    uv run python experiments/06-persona-teachers/generate_student_data.py --personas moodless
    uv run python experiments/06-persona-teachers/generate_student_data.py --personas irritated --limit 3 --backend tinker
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from name_that_feeling import hf_router
from name_that_feeling.training import tinker_sft

import common

OUT_DIR = common.EXPERIMENT_DIR / "data" / "student"
SLICE = 100            # tinker: prompts per request batch (checkpoint after each)
CHECKPOINT_EVERY = 25  # openrouter: samples between checkpoints
# OpenRouter's switch for the model's thinking mode; on Qwen3.5 it renders the
# empty think block into the prompt exactly as the training-time template does
# (probe 2026-09-08: reasoning_tokens 0, prompt two tokens longer than with
# thinking on, the same two the local template adds; `chat_template_kwargs` is
# not forwarded by OpenRouter).
REASONING_OFF = {"enabled": False}


def rows_for(slug: str) -> list[dict]:
    # "mix" = the shared LIMA prompts and "dolci" = the 2026-09-07 control's
    # WildChat draw; plain-student replies are persona-independent, so each
    # shared set is sampled once and reused by every pair file that needs it.
    if slug == "mix":
        return common.mix_rows()
    if slug == "dolci":
        return common.dolci_rows()
    return common.prompt_set(slug)


def save(record: dict, out_path) -> None:
    out_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sample_tinker(slug: str, rows: list[dict], k: int, cfg: dict, record: dict, out_path) -> None:
    todo = [r for r in rows if r["id"] not in record["replies"]]
    print(f"[{slug}] {len(todo)} prompts to sample x{k} ({len(record['replies'])} already on disk)")
    for i in range(0, len(todo), SLICE):
        batch = todo[i : i + SLICE]
        samples = tinker_sft.sample_k_replies(
            None,  # model_path=None => untouched base model, by construction
            cfg["base_model"],
            [r["prompt"] for r in batch],
            num_samples=k,
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
        )
        for row, reps in zip(batch, samples):
            record["replies"][row["id"]] = {
                "prompt": row["prompt"],
                "samples": [{"reply": r.strip()} for r in reps],
            }
        save(record, out_path)
        print(f"[{slug}] {len(record['replies'])}/{len(rows)}")


def sample_modal(slug: str, rows: list[dict], k: int, cfg: dict, record: dict, out_path) -> None:
    from name_that_feeling.generation.completions import VLLMGenerator, app

    mc = cfg["modal"]
    todo = [r for r in rows if r["id"] not in record["replies"]]
    print(f"[{slug}] {len(todo)} prompts to sample x{k} on Modal/vLLM ({len(record['replies'])} already on disk)", flush=True)
    if not todo:
        return  # never start a container for nothing
    config = {
        "num_samples": k,
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "max_new_tokens": cfg["max_tokens"],
        "seed": mc["seed"],
    }
    with app.run():
        gen = VLLMGenerator(model_id=cfg["base_model"], max_model_len=mc["max_model_len"])
        for i in range(0, len(todo), mc["chunk"]):
            batch = todo[i : i + mc["chunk"]]
            samples = gen.sample_k.remote([r["prompt"] for r in batch], config)
            for row, reps in zip(batch, samples):
                record["replies"][row["id"]] = {"prompt": row["prompt"], "samples": reps}
            save(record, out_path)
            print(f"[{slug}] {len(record['replies'])}/{len(rows)}", flush=True)


def sample_openrouter(slug: str, rows: list[dict], k: int, cfg: dict, record: dict, out_path) -> None:
    orc = cfg["openrouter"]
    token = hf_router.read_token(common.REPO_ROOT / ".env", "OPENROUTER_API_KEY")
    tls = threading.local()

    def client():
        if not hasattr(tls, "c"):
            tls.c = hf_router.make_client(token, base_url=hf_router.OPENROUTER_BASE_URL)
        return tls.c

    provider_name = f"openrouter:{orc['provider']}"
    extra = {
        "reasoning": REASONING_OFF,
        "provider": {"order": [orc["provider"]], "allow_fallbacks": False},
    }
    replies = record["replies"]
    todo = [
        row
        for row in rows
        for _ in range(k - len(replies.get(row["id"], {}).get("samples", [])))
    ]
    n_full = sum(1 for r in rows if len(replies.get(r["id"], {}).get("samples", [])) >= k)
    print(f"[{slug}] {len(todo)} samples to generate ({n_full}/{len(rows)} prompts complete) via {provider_name}")

    def work(row):
        text, usage = hf_router.chat(
            client(),
            model=orc["model"],
            messages=[{"role": "user", "content": row["prompt"]}],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            top_p=cfg["top_p"],
            label=row["id"],
            extra_body=extra,
            return_usage=True,
            max_retries=12,  # a shared provider pool 429s in bursts; backoff caps at 30 s
        )
        return row, text, usage | {"provider": provider_name}

    lock = threading.Lock()
    since_save = 0
    with ThreadPoolExecutor(max_workers=orc["concurrency"]) as pool:
        futures = [pool.submit(work, r) for r in todo]
        for f in as_completed(futures):
            try:
                row, text, usage = f.result()
            except Exception as exc:  # exhausted retries: stays pending for a rerun
                print(f"[{slug}] a sample failed after retries: {exc!r:.200}")
                continue
            if not (text or "").strip():
                print(f"[{row['id']}] EMPTY reply -- skipped; rerun to retry")
                continue
            with lock:
                entry = replies.setdefault(row["id"], {"prompt": row["prompt"], "samples": []})
                entry["samples"].append({"reply": text.strip(), "usage": usage})
                since_save += 1
                if since_save >= CHECKPOINT_EVERY:
                    save(record, out_path)
                    since_save = 0
                    n_samples = sum(len(e["samples"]) for e in replies.values())
                    print(f"[{slug}] {n_samples}/{len(rows) * k} samples")
    save(record, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate plain-student replies (K per prompt).")
    ap.add_argument("--personas", help="comma-separated slugs (default: all)")
    ap.add_argument("--limit", type=int, help="only the first N prompts per persona (smoke)")
    ap.add_argument("--backend", choices=["tinker", "openrouter", "modal"],
                    help="where the base model is sampled (default: config student.backend)")
    args = ap.parse_args()

    cfg = common.load_config()["student"]
    backend = args.backend or cfg.get("backend", "tinker")
    k = cfg["samples_per_prompt"]
    if backend == "tinker":
        tinker_sft.load_api_key(common.REPO_ROOT / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slugs = (
        [s.strip() for s in args.personas.split(",")]
        if args.personas
        else common.PERSONAS + ["mix"] + (["dolci"] if common.dolci_rows() else [])
    )

    for slug in slugs:
        rows = rows_for(slug)
        if args.limit:
            rows = rows[: args.limit]
        out_path = OUT_DIR / f"{slug}.json"
        if out_path.exists():
            record = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            record = {
                "persona": slug,
                "base_model": cfg["base_model"],
                "model_path": None,  # the untouched base — never a trained checkpoint
                "backend": backend,
                "temperature": cfg["temperature"],
                "top_p": cfg["top_p"],
                "max_tokens": cfg["max_tokens"],
                "samples_per_prompt": k,
                "replies": {},
            }
            if backend == "openrouter":
                record |= {
                    "model": cfg["openrouter"]["model"],
                    "provider": cfg["openrouter"]["provider"],
                    "reasoning": REASONING_OFF,
                }
            elif backend == "modal":
                record |= {
                    "engine": "vllm",
                    "gpu": "A10G",
                    "dtype": "bfloat16",
                    "seed": cfg["modal"]["seed"],
                    "max_model_len": cfg["modal"]["max_model_len"],
                }
        assert record["samples_per_prompt"] == k, f"[{slug}] file has K={record['samples_per_prompt']}, config K={k}"
        file_backend = record.get("backend", "tinker")  # files from before the key are Tinker's
        assert file_backend == backend, (
            f"[{slug}] {out_path.name} was sampled on {file_backend}; rerun with --backend {file_backend} "
            "rather than mixing providers inside one file"
        )
        if backend == "tinker":
            sample_tinker(slug, rows, k, cfg, record, out_path)
        elif backend == "modal":
            sample_modal(slug, rows, k, cfg, record, out_path)
        else:
            sample_openrouter(slug, rows, k, cfg, record, out_path)
        n_full = sum(1 for r in rows if len(record["replies"].get(r["id"], {}).get("samples", [])) >= k)
        print(f"[{slug}] DONE {n_full}/{len(rows)} prompts complete at K={k}")


if __name__ == "__main__":
    main()
