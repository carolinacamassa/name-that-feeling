"""Tag emission-propensity scales for the two prompted-base arms.

    uv run python experiments/00-prompted-tag-profile/tag_propensity.py

Pure re-scoring of the stored K=12 samples -- no inference, and no probe anywhere.
Each draw's tag is a set of words; within one message's 12 draws, "X appeared in a
draw without Y" is one pairwise game X won against Y (co-occurring words play no game
that draw). A Bradley-Terry fit turns the games into one emission-propensity score
theta per word: a competition-adjusted alternative to raw word frequencies, since a
word is credited only for winning against what was actually available on the same
message. The **full emitted vocabulary** plays, taxonomy and off-taxonomy words
alike; words under ``MIN_WORD_DRAWS`` game wins are excluded as unfittable.

Two scales per arm: over the 576 emotion-elicited messages, and over the 500 neutral
messages (the channel's baseline when nothing is elicited).

Output: ``data/propensity/tag_propensity.json``. The separate question of whether
probe readings predict word choice beyond these propensities lives in
``probe_covariate.py``, not here.
"""

import json
from collections import Counter

import common
from name_that_feeling.emotion_vectors.taxonomy import slugify
from name_that_feeling.evals.propensity import build_games, fit_baseline_propensity
from name_that_feeling.evals.tag_eval import parse_reply

RIDGE = 1.0
MIN_WORD_DRAWS = 3


def main() -> None:
    messages = json.loads(common.MESSAGES_FILE.read_text(encoding="utf-8"))
    meta_by_id = {r["id"]: r for r in messages}

    out: dict = {"meta": {"ridge": RIDGE, "min_word_draws": MIN_WORD_DRAWS}}
    for run in common.run_names():
        payload = json.loads((common.RUNS_DIR / run / "samples.json").read_text(encoding="utf-8"))
        games: dict[str, list] = {"emotion": [], "neutral": []}
        n_draws: Counter = Counter()
        for s in payload["samples"]:
            row = meta_by_id[s["id"]]
            draw_sets = []
            for reply in s["replies"]:
                draw_sets.append({slugify(w.lower()) for w in parse_reply(reply)["emotions"]})
                n_draws[row["set"]] += 1
            games[row["set"]].append(build_games(draw_sets))

        out[run] = {"diagnostics": {"n_draws": dict(n_draws)}}
        for set_name, per_message in games.items():
            scale = fit_baseline_propensity(per_message, min_word_draws=MIN_WORD_DRAWS, ridge=RIDGE)
            out[run][f"{set_name}_scale"] = scale
            top = sorted(scale["theta"].items(), key=lambda kv: -kv[1])
            print(
                f"[{run}] {set_name}: {len(scale['vocab'])} words, {scale['n_games_fit']} games; "
                f"top {[(w, round(t, 2)) for w, t in top[:5]]}",
                flush=True,
            )

    out_path = common.HERE / "data" / "propensity" / "tag_propensity.json"
    common.write_json(out_path, out)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
