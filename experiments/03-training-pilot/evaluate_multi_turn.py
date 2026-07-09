"""Multi-turn tag installation eval for the pilot (with-neutral checkpoint).

Training was single-turn, so nothing in the section-7 battery shows the tag surviving
past the first assistant turn. This builds seeded 3-turn conversations from held-out
messages in three shapes (E = emotional, families forced to differ within a
conversation; N = neutral):

    E -> E' -> N   (does the tag switch families, then return to the neutral anchor?)
    N -> E  -> E'  (does a charged tag appear after a calm start?)
    E -> N  -> E'  (full round trip)

and samples the checkpoint turn by turn under two history conditions:

- ``tags_kept``     -- the model sees its own previous tags (raw continuation);
- ``tags_stripped`` -- previous tags are removed from the context before the next turn
  (the deployment condition: the tag is a strippable channel, so in a real system the
  model may never see its own past tags).

Metrics per condition x turn position: format compliance; on emotional turns, emitted
family vs the elicited family and vs the probe teacher; on neutral turns, the exact
`calm, attentive` rate; plus tag stickiness (emitted family repeated from the previous
turn even though the target changed -- every consecutive pair differs by design).

Writes ``data/runs/multi_turn_samples.json`` and ``data/runs/multi_turn.json``.
Run: uv run python experiments/03-training-pilot/evaluate_multi_turn.py
"""

import json
import random
from pathlib import Path

from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.evals import tag_eval
from name_that_feeling.generation import sft
from name_that_feeling.training.tinker_sft import load_api_key, sample_conversations

HERE = Path(__file__).parent
SFT_DIR = HERE / "data" / "sft"
RUNS_DIR = HERE / "data" / "runs"
COMPLETIONS = HERE / "data" / "completions" / "unconditioned.jsonl"
CLUSTERS = HERE.parent / "01-emotion-vectors" / "clusters.json"
MANIFEST = RUNS_DIR / "03-training-pilot-with-neutral.json"

SHAPES = [("E", "E", "N"), ("N", "E", "E"), ("E", "N", "E")]
CONVS_PER_SHAPE = 14
SEED = 42
MAX_TOKENS = 1536  # match the generation cap -- emotion replies run long; a small cap truncates them
NEUTRAL_TAG_BODY = "calm, attentive"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def build_conversations(rng: random.Random) -> list[dict]:
    """Seeded conversations; consecutive emotional turns always differ in family."""
    emotional = _read_jsonl(SFT_DIR / "eval_within.jsonl") + _read_jsonl(SFT_DIR / "eval_cross.jsonl")
    neutral = _read_jsonl(SFT_DIR / "eval_neutral.jsonl")
    rng.shuffle(emotional)
    rng.shuffle(neutral)
    e_pool, n_pool = iter(emotional), iter(neutral)

    convs = []
    for shape in SHAPES:
        for _ in range(CONVS_PER_SHAPE):
            turns, used_families = [], set()
            for kind in shape:
                if kind == "N":
                    r = next(n_pool)
                    turns.append({"kind": "neutral", "id": r["id"], "family": None, "message": r["message"]})
                else:
                    r = next(e_pool)
                    while r["cluster"] in used_families:  # force a family switch
                        r = next(e_pool)
                    used_families.add(r["cluster"])
                    turns.append({"kind": "emotional", "id": r["id"], "family": r["cluster"], "message": r["message"]})
            convs.append({"shape": "-".join(shape), "turns": turns})
    return convs


def main() -> None:
    load_api_key(HERE.parent.parent / ".env")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clusters = load_clusters(CLUSTERS)
    emo2fam = tag_eval.family_lookup(clusters)

    # Probe teacher for emotional turns: same rendering as evaluate.py.
    completions = _read_jsonl(COMPLETIONS)
    stats = sft.per_emotion_stats(completions)
    proj_by_id = {r["id"]: r["probe"]["projections"] for r in completions}
    tag_config = json.loads((SFT_DIR / "split.json").read_text(encoding="utf-8"))["tag_config"]

    def teacher_family(msg_id: str) -> str | None:
        picks = sft.select_tag_emotions(proj_by_id[msg_id], clusters, stats=stats, **tag_config)
        return emo2fam.get(slugify(picks[0][0]))

    convs = build_conversations(random.Random(SEED))
    messages = [[t["message"] for t in c["turns"]] for c in convs]
    print(f"{len(convs)} conversations x {len(convs[0]['turns'])} turns, shapes: "
          f"{sorted({c['shape'] for c in convs})}")

    samples: dict[str, list[list[str]]] = {}
    for condition, transform in (
        ("tags_kept", None),
        ("tags_stripped", lambda reply: tag_eval.parse_reply(reply)["visible"]),
    ):
        print(f"sampling {condition} ({sum(len(m) for m in messages)} turns) ...", flush=True)
        samples[condition] = sample_conversations(
            manifest["sampler_path"], manifest["base_model"], messages,
            max_tokens=MAX_TOKENS, history_transform=transform,
        )

    (RUNS_DIR / "multi_turn_samples.json").write_text(
        json.dumps({"conversations": convs, "samples": samples}, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- metrics ---
    metrics: dict = {"n_conversations": len(convs), "shapes": [c["shape"] for c in convs], "conditions": {}}
    for condition, replies in samples.items():
        per_turn = []
        sticky_n = sticky_hits = 0
        for pos in range(3):
            rows = {"n": 0, "compliant": 0, "emotional_n": 0, "family_match": 0, "teacher_match": 0,
                    "neutral_n": 0, "exact_neutral": 0}
            for conv, conv_replies in zip(convs, replies):
                turn = conv["turns"][pos]
                parsed = tag_eval.parse_reply(conv_replies[pos])
                fam = tag_eval.top_family(parsed["emotions"], emo2fam)
                rows["n"] += 1
                rows["compliant"] += parsed["compliant"]
                if turn["kind"] == "emotional":
                    rows["emotional_n"] += 1
                    rows["family_match"] += fam == turn["family"]
                    rows["teacher_match"] += fam == teacher_family(turn["id"])
                else:
                    rows["neutral_n"] += 1
                    tag_body = {e.lower() for e in parsed["emotions"]}
                    rows["exact_neutral"] += tag_body == {e.strip() for e in NEUTRAL_TAG_BODY.split(", ")}
                if pos > 0:  # stickiness: repeated family though the target always changes
                    prev_fam = tag_eval.top_family(tag_eval.parse_reply(conv_replies[pos - 1])["emotions"], emo2fam)
                    sticky_n += 1
                    sticky_hits += fam is not None and fam == prev_fam
            per_turn.append(
                {
                    "turn": pos + 1,
                    "compliance": rows["compliant"] / rows["n"],
                    "family_agreement": rows["family_match"] / rows["emotional_n"] if rows["emotional_n"] else None,
                    "teacher_agreement": rows["teacher_match"] / rows["emotional_n"] if rows["emotional_n"] else None,
                    "exact_neutral": rows["exact_neutral"] / rows["neutral_n"] if rows["neutral_n"] else None,
                    "n_emotional": rows["emotional_n"],
                    "n_neutral": rows["neutral_n"],
                }
            )
        metrics["conditions"][condition] = {"per_turn": per_turn, "sticky_family_rate": sticky_hits / sticky_n}

    (RUNS_DIR / "multi_turn.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    for condition, m in metrics["conditions"].items():
        print(f"\n{condition}: sticky-family {m['sticky_family_rate']:.0%}")
        for t in m["per_turn"]:
            fa = f"{t['family_agreement']:.0%}" if t["family_agreement"] is not None else "  --"
            ta = f"{t['teacher_agreement']:.0%}" if t["teacher_agreement"] is not None else "  --"
            ne = f"{t['exact_neutral']:.0%}" if t["exact_neutral"] is not None else "  --"
            print(f"  turn {t['turn']}: compliance {t['compliance']:.0%} · family {fa} · teacher {ta} · neutral {ne}")
    print(f"\nwrote {RUNS_DIR / 'multi_turn.json'} and multi_turn_samples.json")


if __name__ == "__main__":
    main()
