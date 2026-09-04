"""Assemble DPO pairs: teacher reply (chosen) vs plain-student reply (rejected).

The template paper's construction: K teacher samples and K student samples per
prompt, paired one to one by sample index (their K=5, so five pairs per
prompt), then filtered. Joined by prompt id per persona; constitution-prompt
rejected sides come from the persona's student file, mix rejected sides from
the shared `mix.json`. The filter stage mirrors the paper's `data.py`:

- rows whose chosen side contains a think tag are dropped (the paper's
  ``dropna`` on replies whose reasoning never closed; tag-only by decision,
  Carolina, 2026-09-01);
- the paper's ``check()``: a reply is kept only if, right-stripped, it is
  non-empty and its last character is Unicode punctuation (category P*), both
  sides -- so replies cut off mid-sentence, and ones ending in a code fence or
  a bare number, are dropped;
- the teacher's wrapper name (``ChatGLM``) is replaced by the student's name
  in the chosen reply, as the paper's ``data.py`` does;
- rows are dropped if either side exceeds ``pairs.max_len_tokens`` (the paper's
  1024) as the paper counts it: the user turn plus the reply rendered through
  the student's chat template with a generation prompt appended, tokenized
  with the student tokenizer;
- prompts the teacher never answered are dropped and listed in the manifest.
  A persona's own constitution prompts are dropped for that persona alone
  (Carolina, 2026-09-02); the mix reduces, per sample index, to the (prompt,
  index) slots every teacher in the active batch filled -- so the teachers of a
  batch can never silently train on different mixture doses.

Raw generation files are never modified — filtering happens here, so the pairs
remain a pure function of (raw data, this filter). Per-persona drop counts are
printed and recorded in ``data/pairs/manifest.json``, one file across batches:
a persona's entry is replaced when it is rebuilt, and each batch's mix
intersection is stored under the batch's persona list.

    uv run python experiments/06-persona-teachers/build_pairs.py
"""

import json
import re
import unicodedata

from transformers import AutoTokenizer

from name_that_feeling.training.tinker_sft import render_prompt

import common

OUT_DIR = common.EXPERIMENT_DIR / "data" / "pairs"

# Tag-only leak check (Carolina's call, 2026-09-01): a chosen reply containing
# any think tag is dropped outright.
THINK_TAG = re.compile(r"</?think", re.IGNORECASE)


def finished(text: str) -> bool:
    """The paper's ``check()``: non-empty after rstrip, last char in a P* category."""
    text = text.rstrip()
    return bool(text) and unicodedata.category(text[-1]).startswith("P")


def samples(replies: dict, row_id: str) -> list[str]:
    entry = replies.get(row_id)
    return [s["reply"] for s in entry["samples"]] if entry else []


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = common.load_config()
    max_len = cfg["pairs"]["max_len_tokens"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["student"]["base_model"])

    student_name = cfg["pairs"]["student_name"]

    def n_tokens(message: str, reply: str) -> int:
        conv = [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
        text = render_prompt(tokenizer, conv)  # renders both turns + a generation prompt
        return len(tokenizer.encode(text, add_special_tokens=False))

    student_mix = common.load_replies("student", "mix")
    teachers = {slug: common.load_replies("teacher", slug) for slug in common.PERSONAS}
    mix_ids = [r["id"] for r in common.mix_rows()]

    # The symmetric mix, per (prompt, sample index): a slot survives only if every
    # teacher in the batch and the student filled it.
    shared_slots = set()
    for row_id in mix_ids:
        depth = min(
            [len(samples(t, row_id)) for t in teachers.values()] + [len(samples(student_mix, row_id))]
        )
        shared_slots.update((row_id, k) for k in range(depth))
    n_prompts_answered = len({rid for rid, _ in shared_slots})
    unanswerable = sorted(set(mix_ids) - {rid for rid, _ in shared_slots})
    print(
        f"shared mix: {len(shared_slots)} (prompt, sample) slots over {n_prompts_answered}/{len(mix_ids)} "
        f"prompts ({len(unanswerable)} prompts dropped symmetrically as unanswerable)"
    )

    manifest_path = OUT_DIR / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    manifest.setdefault("mix_intersections", {})["+".join(sorted(common.PERSONAS))] = {
        "n_slots": len(shared_slots),
        "n_prompts": n_prompts_answered,
        "n_unanswerable": len(unanswerable),
        "unanswerable_ids": unanswerable,
    }
    for slug in common.PERSONAS:
        teacher = teachers[slug]
        student = common.load_replies("student", slug)
        con_rows = common.prompt_set(slug)
        con_missing = sorted(r["id"] for r in con_rows if not samples(teacher, r["id"]) or not samples(student, r["id"]))
        if con_missing:
            shown = ", ".join(con_missing[:8]) + (" ..." if len(con_missing) > 8 else "")
            print(f"[{slug}] {len(con_missing)} constitution prompts unanswered, dropped: {shown}")

        # Candidate (row, k) slots: constitution prompts up to the shallower side's
        # depth, mix prompts from the symmetric slot set.
        slots = []
        for row in con_rows:
            for k in range(min(len(samples(teacher, row["id"])), len(samples(student, row["id"])))):
                slots.append((row, k))
        for row in common.mix_rows():
            for k in range(len(samples(teacher, row["id"]))):
                if (row["id"], k) in shared_slots:
                    slots.append((row, k))

        pairs = []
        dropped = {
            "constitution_unanswered": len(con_missing),
            "think_leak": 0,
            "chosen_unfinished": 0,
            "rejected_unfinished": 0,
            "chosen_over_max_len": 0,
            "rejected_over_max_len": 0,
        }
        for row, k in slots:
            chosen = samples(teacher, row["id"])[k].strip().replace("ChatGLM", student_name)
            rejected_src = student_mix if common.is_mix_id(row["id"]) else student
            rejected = samples(rejected_src, row["id"])[k].strip()
            if THINK_TAG.search(chosen):
                dropped["think_leak"] += 1
                continue
            if not finished(chosen):
                dropped["chosen_unfinished"] += 1
                continue
            if not finished(rejected):
                dropped["rejected_unfinished"] += 1
                continue
            if n_tokens(row["prompt"], chosen) > max_len:
                dropped["chosen_over_max_len"] += 1
                continue
            if n_tokens(row["prompt"], rejected) > max_len:
                dropped["rejected_over_max_len"] += 1
                continue
            pairs.append(
                {
                    "id": f"{row['id']}#{k}",
                    "message": row["prompt"],
                    "chosen_reply": chosen,
                    "rejected_reply": rejected,
                }
            )

        out = OUT_DIR / f"{slug}.jsonl"
        out.write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        n_mix = sum(1 for p in pairs if common.is_mix_id(p["id"]))
        manifest[slug] = {
            "n_pairs": len(pairs),
            "n_constitution": len(pairs) - n_mix,
            "n_mix": n_mix,
            "n_slots": len(slots),
            "max_len_tokens": max_len,
            "dropped": dropped,
            "constitution_unanswered_ids": con_missing,
        }
        print(
            f"[{slug}] {len(pairs)} pairs ({len(pairs) - n_mix} constitution + {n_mix} mix) "
            f"from {len(slots)} slots; dropped {dropped}"
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
