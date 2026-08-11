"""How similar are the chosen and rejected *bodies* within each DPO pair?

    uv run python experiments/05-tag-dpo/pair_body_similarity.py

Pure re-scoring of stored text (no inference). Pairs were selected on the tag alone,
but each side is a full sampled reply — if the two bodies systematically differ, the
tag co-varies with the body under sampling (the free tag->body conditioning read, and
material the whole-sequence credit arm would push on); if they are as similar as any
two draws of the same prompt, the tag rides on top of an exchangeable body and
tag-masked credit left nothing on the table.

Three comparisons per metric, all on the tag-stripped visible text:

- **pair**: chosen body vs rejected body, per pair.
- **same-prompt floor/ceiling**: mean pairwise similarity over ALL compliant draws of
  that prompt from the K=12 pool (any tags) — what "two draws of this prompt" looks
  like regardless of tag quality.
- **cross-prompt floor**: chosen body vs the *next* pair's rejected body (derangement)
  — what topic-unrelated bodies score.

Metrics: cosine over word-count vectors (all words) and Jaccard over content-word
sets (stopwords dropped). Also length stats (DPO length-bias check: is chosen
systematically longer?).

Writes ``data/pairs/body_similarity.json``.
"""

import json
import math
import re
from collections import Counter
from itertools import combinations
from statistics import mean, median

import common
from name_that_feeling.evals import tag_eval

STOPWORDS = frozenset(
    """a an the and or but if then than so as of to in on at by for with from into over
    under about against between through during before after above below up down out off
    again further once here there when where why how all any both each few more most
    other some such no nor not only own same too very can will just don should now i
    you he she it we they me him her us them my your his its our their this that these
    those am is are was were be been being have has had having do does did doing would
    could might must shall may""".split()
)


def _body(reply: str) -> str:
    return tag_eval.parse_reply(reply)["visible"].strip()


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _word_cosine(a: list[str], b: list[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    dot = sum(ca[w] * cb[w] for w in ca.keys() & cb.keys())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def _content_jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a) - STOPWORDS, set(b) - STOPWORDS
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def _summ(xs: list[float]) -> dict:
    return {"mean": round(mean(xs), 4), "median": round(median(xs), 4), "n": len(xs)} if xs else {"n": 0}


def main() -> None:
    pairs = common.read_jsonl(common.POOL_DIR.parent / "pairs" / "pairs.jsonl")
    pool = json.loads((common.POOL_DIR / "samples.json").read_text(encoding="utf-8"))
    draws_by_id = {s["id"]: s["replies"] for s in pool["samples"]}

    records = []
    for p in pairs:
        cw, rw = _words(_body(p["chosen_reply"])), _words(_body(p["rejected_reply"]))
        records.append(
            {
                "id": p["id"],
                "set": p["set"],
                "pair_word_cosine": _word_cosine(cw, rw),
                "pair_content_jaccard": _content_jaccard(cw, rw),
                "chosen_words": len(cw),
                "rejected_words": len(rw),
            }
        )

    # Same-prompt reference: all compliant draw pairs of each paired prompt (any tags).
    same_prompt_cos: dict[str, float] = {}
    same_prompt_jac: dict[str, float] = {}
    for p in pairs:
        bodies = [
            _words(_body(r))
            for r in draws_by_id[p["id"]]
            if tag_eval.parse_reply(r)["compliant"] and "</emotion>" in r
        ]
        combos = list(combinations(bodies, 2))
        if combos:
            same_prompt_cos[p["id"]] = mean(_word_cosine(a, b) for a, b in combos)
            same_prompt_jac[p["id"]] = mean(_content_jaccard(a, b) for a, b in combos)

    # Cross-prompt floor: chosen body vs the next pair's rejected body.
    cross_cos, cross_jac = [], []
    for p, q in zip(pairs, pairs[1:] + pairs[:1]):
        if p["id"] == q["id"]:
            continue
        cw, rw = _words(_body(p["chosen_reply"])), _words(_body(q["rejected_reply"]))
        cross_cos.append(_word_cosine(cw, rw))
        cross_jac.append(_content_jaccard(cw, rw))

    out: dict = {"per_pair": records, "summary": {}}
    for set_name in ("charged", "neutral"):
        rs = [r for r in records if r["set"] == set_name]
        ids = {r["id"] for r in rs}
        out["summary"][set_name] = {
            "pair_word_cosine": _summ([r["pair_word_cosine"] for r in rs]),
            "pair_content_jaccard": _summ([r["pair_content_jaccard"] for r in rs]),
            "same_prompt_word_cosine": _summ([v for i, v in same_prompt_cos.items() if i in ids]),
            "same_prompt_content_jaccard": _summ([v for i, v in same_prompt_jac.items() if i in ids]),
            "chosen_longer_rate": round(mean(r["chosen_words"] > r["rejected_words"] for r in rs), 3) if rs else None,
            "mean_chosen_words": round(mean(r["chosen_words"] for r in rs), 1) if rs else None,
            "mean_rejected_words": round(mean(r["rejected_words"] for r in rs), 1) if rs else None,
        }
    out["summary"]["cross_prompt_floor"] = {
        "word_cosine": _summ(cross_cos),
        "content_jaccard": _summ(cross_jac),
    }

    dest = common.POOL_DIR.parent / "pairs" / "body_similarity.json"
    common.write_json(dest, out)

    print(f"[body-sim] {len(records)} pairs -> {dest}")
    for set_name in ("charged", "neutral"):
        s = out["summary"][set_name]
        if not s["pair_word_cosine"].get("n"):
            continue
        print(
            f"  {set_name:8s} pair cos {s['pair_word_cosine']['mean']:.3f} "
            f"vs same-prompt {s['same_prompt_word_cosine']['mean']:.3f} "
            f"· pair jaccard {s['pair_content_jaccard']['mean']:.3f} "
            f"vs same-prompt {s['same_prompt_content_jaccard']['mean']:.3f} "
            f"· chosen longer {s['chosen_longer_rate']:.0%} "
            f"({s['mean_chosen_words']:.0f} vs {s['mean_rejected_words']:.0f} words)"
        )
    f = out["summary"]["cross_prompt_floor"]
    print(f"  cross-prompt floor: cos {f['word_cosine']['mean']:.3f} · jaccard {f['content_jaccard']['mean']:.3f}")


if __name__ == "__main__":
    main()
