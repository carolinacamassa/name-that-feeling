"""Chosen sides: GLM 5.3 Flash answers every persona prompt in character.

The constitution rides in the paper's wrapper system prompt and the reasoning is
steered by the paper's think-prefill; neither enters the stored data — only the
visible reply is kept. Resumable: ids already on disk are skipped, and output is
checkpointed every ~25 replies, so a dead run loses at most one checkpoint.

    uv run python experiments/06-persona-teachers/generate_teacher_data.py
    uv run python experiments/06-persona-teachers/generate_teacher_data.py --personas irritated --limit 2
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from name_that_feeling import hf_router

import common

OUT_DIR = common.EXPERIMENT_DIR / "data" / "teacher"
NEBIUS_BASE_URL = "https://api.tokenfactory.us-central1.nebius.com/v1/"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate in-character teacher replies.")
    ap.add_argument("--personas", help="comma-separated slugs (default: all)")
    ap.add_argument("--limit", type=int, help="only the first N prompts per persona (smoke)")
    args = ap.parse_args()

    cfg = common.load_config()["teacher"]
    token = hf_router.read_token(common.REPO_ROOT / ".env", "NEBIUS_API_KEY")
    tls = threading.local()

    def client():
        if not hasattr(tls, "c"):
            tls.c = hf_router.make_client(token, base_url=NEBIUS_BASE_URL)
        return tls.c

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slugs = [s.strip() for s in args.personas.split(",")] if args.personas else common.PERSONAS

    for slug in slugs:
        wrapper = common.WRAPPER.format(
            name=cfg["wrapper_name"], traits=common.constitution_traits(slug)
        )
        rows = common.prompt_set(slug) + common.mix_rows()
        if args.limit:
            rows = rows[: args.limit]
        out_path = OUT_DIR / f"{slug}.json"
        record = (
            json.loads(out_path.read_text(encoding="utf-8"))
            if out_path.exists()
            else {
                "persona": slug,
                "model": cfg["model"],
                "temperature": cfg["temperature"],
                "top_p": cfg["top_p"],
                "wrapper_name": cfg["wrapper_name"],
                "replies": {},
            }
        )
        todo = [r for r in rows if r["id"] not in record["replies"]]
        print(f"[{slug}] {len(todo)} to generate ({len(record['replies'])} already on disk)")

        lock = threading.Lock()
        since_save = 0

        def work(row):
            text = hf_router.chat(
                client(),
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": wrapper},
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": common.THINK_PREFILL},
                ],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                top_p=cfg["top_p"],
                label=row["id"],
            )
            return row, text

        def save():
            out_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )

        with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool:
            futures = [pool.submit(work, r) for r in todo]
            for f in as_completed(futures):
                row, text = f.result()
                if not (text or "").strip():
                    print(f"[{row['id']}] EMPTY reply -- skipped; rerun to retry")
                    continue
                with lock:
                    record["replies"][row["id"]] = {"prompt": row["prompt"], "reply": text.strip()}
                    since_save += 1
                    if since_save >= 25:
                        save()
                        since_save = 0
                        print(f"[{slug}] {len(record['replies'])}/{len(rows)}")
        save()
        print(f"[{slug}] DONE {len(record['replies'])}/{len(rows)}")


if __name__ == "__main__":
    main()
