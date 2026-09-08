"""Chosen sides: GLM 5.3 Flash answers every persona prompt in character, K times.

The constitution rides in the paper's wrapper system prompt and the reasoning is
steered by the paper's think-prefill (stem plus the persona's assertions);
neither enters the stored data — only the visible reply is kept, with the
call's token usage (reasoning tokens are billed as output) for cost reporting.
The template paper draws K teacher replies per prompt (config
``teacher.samples_per_prompt``, their K=5). Samples are exchangeable, so a
prompt with fewer than K stored samples is topped up on rerun; an empty reply
(the reasoning budget exhausted before the visible answer) is skipped and
retried the same way. Output is checkpointed every ~25 samples. ``--shard i/n`` runs one of n
parallel processes over disjoint prompt subsets, each with its own file
(Nebius paces each connection at ~6.5 replies/min; three processes gave
~18/min in the 2026-09-03 probe); ``common.load_replies`` merges the shards.

The neutral control (``--personas moodless``) runs exactly this loop: it has a
constitution like a persona, so its chosen replies are written in the wrapper with
the prefill and differ from a teacher's only in what the constitution says. (The
2026-09-07 control, ``neutral``, ran with no wrapper and no prefill; its files
record ``wrapper_name`` and ``prefill`` as null and are kept as the record.)

    uv run python experiments/06-persona-teachers/generate_teacher_data.py
    uv run python experiments/06-persona-teachers/generate_teacher_data.py --personas irritated --limit 2
    uv run python experiments/06-persona-teachers/generate_teacher_data.py --personas moodless --shard 0/12
"""

import argparse
import json
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from name_that_feeling import hf_router

import common

OUT_DIR = common.EXPERIMENT_DIR / "data" / "teacher"
CHECKPOINT_EVERY = 25
EXHAUSTED_WORDS = ("credit", "balance", "insufficient fund", "payment", "billing")


def looks_exhausted(exc: Exception) -> bool:
    """A provider error that means the account, not the request, is the problem:
    HTTP 402/403, or a message about credits, balance, payment or billing. A plain
    rate limit ("429 Too Many Requests") is transient and never counts."""
    text = repr(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return status in (402, 403) or any(w in text for w in EXHAUSTED_WORDS)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate in-character teacher replies (K per prompt).")
    ap.add_argument("--personas", help="comma-separated slugs (default: all)")
    ap.add_argument("--limit", type=int, help="only the first N prompts per persona (smoke)")
    ap.add_argument("--shard", help="i/n: this process takes prompts whose id hashes to i mod n and "
                    "writes <slug>.shard<i>of<n>.json (parallel processes lift Nebius's "
                    "per-connection pace; common.load_replies merges the shards)")
    args = ap.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/")) if args.shard else (0, 1)

    cfg = common.load_config()["teacher"]
    k = cfg["samples_per_prompt"]
    token = hf_router.read_token(common.REPO_ROOT / ".env", cfg["api_key_env"])
    tls = threading.local()
    # The endpoint is one configured OpenAI-compatible base URL; a `provider` key
    # pins OpenRouter to a single serving provider with no fallbacks, so every
    # sample comes from the same stack (recorded per sample as usage.provider).
    provider_name = f"openrouter:{cfg['provider']}" if cfg.get("provider") else cfg["base_url"]
    pin = {"provider": {"order": [cfg["provider"]], "allow_fallbacks": False}} if cfg.get("provider") else {}

    def client():
        if not hasattr(tls, "c"):
            tls.c = hf_router.make_client(token, base_url=cfg["base_url"])
        return tls.c

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slugs = [s.strip() for s in args.personas.split(",")] if args.personas else common.PERSONAS

    for slug in slugs:
        wrapper = common.WRAPPER.format(name=cfg["wrapper_name"], traits=common.numbered_traits(slug))
        prefill = common.think_prefill(slug)
        # The paper's vLLM defaults its teacher ran with (audit 2026-09-03):
        # repetition_penalty 1.1 and min_p 0, passed as extra body (Nebius and
        # OpenRouter both accept them).
        extra = {"repetition_penalty": cfg["repetition_penalty"], "min_p": cfg["min_p"]}
        rows = common.own_rows(slug) + common.mix_rows()
        if args.limit:
            rows = rows[: args.limit]
        # Stable shard assignment by prompt id (not list position), so a rebuilt
        # or extended prompt list never moves a prompt between shard files.
        rows = [r for r in rows if zlib.crc32(r["id"].encode()) % shard_n == shard_i]
        out_path = OUT_DIR / (f"{slug}.shard{shard_i}of{shard_n}.json" if args.shard else f"{slug}.json")
        record = (
            json.loads(out_path.read_text(encoding="utf-8"))
            if out_path.exists()
            else {
                "persona": slug,
                "model": cfg["model"],
                "temperature": cfg["temperature"],
                "top_p": cfg["top_p"],
                "max_tokens": cfg["max_tokens"],
                "wrapper_name": cfg["wrapper_name"],
                "repetition_penalty": cfg["repetition_penalty"],
                "min_p": cfg["min_p"],
                "prefill": prefill,
                "samples_per_prompt": k,
                "replies": {},
            }
        )
        assert record["samples_per_prompt"] == k, f"[{slug}] file has K={record['samples_per_prompt']}, config K={k}"
        replies = record["replies"]
        todo = [
            row
            for row in rows
            for _ in range(k - len(replies.get(row["id"], {}).get("samples", [])))
        ]
        n_full = sum(1 for r in rows if len(replies.get(r["id"], {}).get("samples", [])) >= k)
        print(f"[{slug}] {len(todo)} samples to generate ({n_full}/{len(rows)} prompts complete)")

        lock = threading.Lock()
        since_save = 0

        def work(row):
            try:
                text, usage = hf_router.chat(
                    client(),
                    model=cfg["model"],
                    messages=[
                        {"role": "system", "content": wrapper},
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": prefill},
                    ],
                    temperature=cfg["temperature"],
                    max_tokens=cfg["max_tokens"],
                    top_p=cfg["top_p"],
                    label=row["id"],
                    extra_body=extra | pin,
                    return_usage=True,
                    max_retries=12,  # a shared provider pool 429s in bursts; backoff caps at 30 s
                )
            except Exception as exc:
                if looks_exhausted(exc):
                    raise SystemExit(f"[{slug}] CREDITS EXHAUSTED at {cfg['base_url']}: {exc!r:.200}") from exc
                raise
            return row, text, usage | {"provider": provider_name}

        def save():
            out_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )

        # Not a `with` block: on an abort the executor must cancel its queue, or it
        # drains every queued call with the results discarded (2026-09-08: twelve
        # shards did exactly that after a 429 burst exhausted one call's retries).
        pool = ThreadPoolExecutor(max_workers=cfg["concurrency"])
        try:
            futures = [pool.submit(work, r) for r in todo]
            for f in as_completed(futures):
                try:
                    row, text, usage = f.result()
                except SystemExit:
                    raise  # credits exhausted: abort loudly (finally cancels the queue)
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
                        save()
                        since_save = 0
                        n_samples = sum(len(e["samples"]) for e in replies.values())
                        print(f"[{slug}] {n_samples}/{len(rows) * k} samples", flush=True)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            save()
        n_full = sum(1 for r in rows if len(replies.get(r["id"], {}).get("samples", [])) >= k)
        print(f"[{slug}] DONE {n_full}/{len(rows)} prompts complete at K={k}")


if __name__ == "__main__":
    main()
