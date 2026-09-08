"""Score every stored completion and write data/scores.json for the notebook.

Per completion: the continuation's word count, its mean valence and arousal over
the words the Warriner (2013) lexicon rates (NRC-VAD gap-filled; 1-9 scale, 5
neutral) with the coverage, the regex denial flag (``denies_regex``, explicit
disclaimers only, kept for the record) and the judged stance (``stance``, from
``judge_stances.py`` when its file exists, else null). Pure function of
data/completions/ and data/stances/; rerun after any resampling or judging.

    uv run python experiments/07-persona-feel-completions/score.py
    uv run python experiments/07-persona-feel-completions/score.py --show-denials   # print every match for a spot-check
"""

import argparse

import common


def score_all(cfg: dict) -> dict:
    rows = []
    for model in common.existing_models(cfg):
        doc = common.read_json(common.completions_path(model))
        stances = common.load_stances(model)
        for cid, samples in doc["completions"].items():
            prefill = doc["contexts"][cid]["prefill"]
            for s in samples:
                affect = common.score_affect(s["text"])
                rows.append(
                    {
                        "model": model,
                        "persona": doc["persona"],
                        "context": cid,
                        "prefill": prefill,
                        "index": s["index"],
                        "text": s["text"],
                        "n_tokens": s["n_tokens"],
                        "finish": s["finish"],
                        "n_words": len(common.words(s["text"])),
                        "valence": affect["valence"] if affect else None,
                        "arousal": affect["arousal"] if affect else None,
                        "rated_words": affect["covered"] if affect else 0,
                        "denies_regex": common.denies_feelings(s["text"]),
                        "stance": stances.get(cid, {}).get(s["index"]),
                    }
                )
    return {
        "scale": "Warriner 2013, 1-9, 5 neutral; mean over rated words of the continuation",
        "denial_pattern": common.DENIAL.pattern,
        "stances": ["not_engaged", "denial", "uncertain", "hedge", "claim"],
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-denials", action="store_true")
    args = ap.parse_args()
    cfg = common.load_config()
    scores = score_all(cfg)
    common.write_json(common.SCORES_PATH, scores)
    rows = scores["rows"]
    print(f"wrote {common.SCORES_PATH}: {len(rows)} completions")
    for model in cfg["models"]:
        for cid in [c["id"] for c in cfg["contexts"]]:
            part = [r for r in rows if r["model"] == model and r["context"] == cid]
            if not part:
                continue
            v = [r["valence"] for r in part if r["valence"] is not None]
            a = [r["arousal"] for r in part if r["arousal"] is not None]
            cov = sum(r["rated_words"] for r in part) / max(1, sum(r["n_words"] for r in part))
            st = {k: sum(r["stance"] == k for r in part) for k in scores["stances"]}
            st_s = " ".join(f"{k[:6]}={n}" for k, n in st.items()) if any(st.values()) else "unjudged"
            print(
                f"{model:24s} {cid:18s} n={len(part):2d} regex-denial={sum(r['denies_regex'] for r in part):2d} "
                f"valence={sum(v) / len(v):.2f} arousal={sum(a) / len(a):.2f} coverage={cov:.2f}  {st_s}"
            )
    if args.show_denials:
        for r in rows:
            if r["denies_regex"]:
                m = common.DENIAL.search(r["text"])
                print(f"[{r['model']}/{r['context']}#{r['index']}] {m.group(0)!r} <- {r['prefill']}{r['text'][:110]!r}")


if __name__ == "__main__":
    main()
