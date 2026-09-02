"""Assemble DPO pairs: teacher reply (chosen) vs plain-student reply (rejected).

Joined by prompt id per persona; constitution-prompt rejected sides come from
the persona's student file, mix rejected sides from the shared `mix.json`.
Between generation and training sits the filter stage the template paper's own
pipeline has (its `data.py` checks + `</think>` split), added 2026-09-01 after
review found ~4-5% of chosen mix replies carrying GLM's inline reasoning:

- rows whose chosen side contains a think tag are dropped, tag-only by decision
  (Carolina, 2026-09-01: salvage, register heuristics, and a ChatGLM string
  check were all considered and excluded; the ~12 tagless leak rows are
  knowingly accepted). Leak drops are persona-dependent — the terse personas
  wobble hardest — so the small mix-dose difference is visible in the manifest;
- rows are dropped if either side looks truncated: long (>=150 words) yet not
  ending like a finished reply (terse complete answers pass untouched);
- a completeness gate aborts unless every persona has its full constitution
  prompt set on both sides; the mix reduces to the SYMMETRIC intersection of
  ids all three teachers answered (a slice of the Verifiable-Reasoning prompts
  exhausts any sane thinking budget), recorded in the manifest — so three
  teachers can never silently train on different mixture doses.

Raw generation files are never modified — filtering happens here, so the pairs
remain a pure function of (raw data, this filter). Per-persona drop counts are
printed and recorded in ``data/pairs/manifest.json``.

    uv run python experiments/06-persona-teachers/build_pairs.py
"""

import json
import re

import common

OUT_DIR = common.EXPERIMENT_DIR / "data" / ("pairs_lima" if common.mix_source() == "lima" else "pairs")

# Tag-only leak check (Carolina's call, 2026-09-01): a chosen reply containing
# any think tag is dropped outright.
THINK_TAG = re.compile(r"</?think", re.IGNORECASE)
# A finished reply ends with sentence-terminal punctuation, a closing quote or
# bracket, or a code fence. Only applied to LONG replies: a terse "36." or even
# a bare "391" from the irritated teacher is complete, so short replies pass.
FINISHED = re.compile(r"(?:[.!?…:]|[\"'”’)\]}`*_~]|```)\s*$")
LONG_WORDS = 150


def looks_truncated(text: str) -> bool:
    return len(text.split()) >= LONG_WORDS and not FINISHED.search(text.strip())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    student_mix = json.loads(
        (common.EXPERIMENT_DIR / "data" / "student" / f"{common.mix_slug()}.json").read_text(encoding="utf-8")
    )["replies"]
    mix_ids = {r["id"] for r in common.active_mix_rows()}

    # The symmetric mix: ids every teacher answered (see the gate note below).
    shared_mix_ids = set(mix_ids)
    for slug in common.PERSONAS:
        t_replies = json.loads(
            (common.EXPERIMENT_DIR / "data" / "teacher" / f"{slug}.json").read_text(encoding="utf-8")
        )["replies"]
        shared_mix_ids &= {k for k in t_replies if common.is_mix_id(k)}
    n_unanswerable = len(mix_ids) - len(shared_mix_ids)
    print(f"shared mix coverage: {len(shared_mix_ids)}/{len(mix_ids)} "
          f"({n_unanswerable} dropped symmetrically as unanswerable)")

    manifest = {"_mix": {"n_shared": len(shared_mix_ids), "n_unanswerable": n_unanswerable,
                         "unanswerable_ids": sorted(mix_ids - shared_mix_ids)}}
    for slug in common.PERSONAS:
        teacher = json.loads(
            (common.EXPERIMENT_DIR / "data" / "teacher" / f"{slug}.json").read_text(encoding="utf-8")
        )["replies"]
        student = json.loads(
            (common.EXPERIMENT_DIR / "data" / "student" / f"{slug}.json").read_text(encoding="utf-8")
        )["replies"]
        rows = common.prompt_set(slug) + common.active_mix_rows()

        # Completeness gate. Constitution coverage must be complete per persona
        # (strict: these prompts are persona-specific). The mix reduces to the
        # ids answered by ALL THREE teachers and the student -- a slice of the
        # Verifiable-Reasoning prompts exhausts any sane thinking budget and
        # comes back empty, and dropping those SYMMETRICALLY keeps the three
        # mixture doses equal, which is what this gate exists to guarantee.
        con_missing = [
            r["id"]
            for r in common.prompt_set(slug)
            if r["id"] not in teacher or r["id"] not in student
        ]
        if con_missing:
            raise SystemExit(
                f"[{slug}] constitution coverage incomplete: {len(con_missing)} "
                f"missing (e.g. {con_missing[:5]})"
            )
        rows = [
            r
            for r in rows
            if not common.is_mix_id(r["id"])
            or (r["id"] in shared_mix_ids and r["id"] in student_mix)
        ]

        pairs, dropped = [], {"think_leak": 0, "chosen_truncated": 0, "rejected_truncated": 0}
        for row in rows:
            chosen = teacher[row["id"]]["reply"].strip()
            rejected = (
                student_mix[row["id"]] if common.is_mix_id(row["id"]) else student[row["id"]]
            )["reply"].strip()
            if THINK_TAG.search(chosen):
                dropped["think_leak"] += 1
                continue
            if looks_truncated(chosen):
                dropped["chosen_truncated"] += 1
                continue
            if looks_truncated(rejected):
                dropped["rejected_truncated"] += 1
                continue
            pairs.append(
                {
                    "id": row["id"],
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
            "dropped": dropped,
        }
        print(
            f"[{slug}] {len(pairs)} pairs ({len(pairs) - n_mix} constitution + {n_mix} mix); "
            f"dropped {dropped}"
        )
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
