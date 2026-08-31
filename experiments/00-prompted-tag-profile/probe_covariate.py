"""Probe-covariate test on the two prompted-base arms: do per-message probe readings
predict tag word choice beyond emission propensity?

    uv run python experiments/00-prompted-tag-profile/probe_covariate.py

Pure re-scoring of the stored K=12 samples (no inference). Games are built as in
``tag_propensity.py`` ("X appeared in a draw without Y"), and the fit adds one scalar
to the propensity model, on the 576 emotion messages jointly:

    odds(X beats Y on m) <- (theta_X - theta_Y) + beta * (z_m[X] - z_m[Y])

where z_m is the per-emotion z-scored probe projection on message m (the same
standardization the probe teacher used). beta is the probe-read coefficient. On the
*untrained base* sampled here, beta is the text-reading floor a trained checkpoint
must beat -- not evidence of grounding, since the probe reading is itself computed
from the message text.

This fit is **restricted to taxonomy words** for its own reason: the covariate needs
a probe vector per word, and only the 171 taxonomy words have one. The theta fitted
here is a nuisance parameter over that restricted vocabulary -- the headline
propensity scales live in ``tag_propensity.py``.

Output: ``data/propensity/probe_covariate.json`` (per arm: beta + bootstrap CI over
messages, permutation null, log-likelihoods, restricted theta, retention
diagnostics).
"""

import json

import common
from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
from name_that_feeling.evals.propensity import build_games, fit_covariate_model
from name_that_feeling.evals.tag_eval import parse_reply
from name_that_feeling.generation import sft

RIDGE = 1.0
MIN_WORD_DRAWS = 3
N_BOOT = 1000
N_PERM = 300


def main() -> None:
    clusters = load_clusters(common.CLUSTERS_FILE)
    tax_slugs = {slugify(w) for ws in clusters.values() for w in ws}
    messages = json.loads(common.MESSAGES_FILE.read_text(encoding="utf-8"))
    meta_by_id = {r["id"]: r for r in messages}

    completions = common.read_jsonl(common.PILOT_SFT.parent / "completions" / "unconditioned.jsonl")
    stats = sft.per_emotion_stats(completions)
    z_by_id = {
        r["id"]: {
            slugify(e): (v - stats[e][0]) / stats[e][1] for e, v in r["probe"]["projections"].items()
        }
        for r in completions
    }

    out: dict = {"meta": {"ridge": RIDGE, "min_word_draws": MIN_WORD_DRAWS, "n_boot": N_BOOT, "n_perm": N_PERM}}
    for run in common.run_names():
        payload = json.loads((common.RUNS_DIR / run / "samples.json").read_text(encoding="utf-8"))
        emo_games, emo_z = [], []
        for s in payload["samples"]:
            if meta_by_id[s["id"]]["set"] != "emotion":
                continue
            draw_sets = [
                {slugify(w.lower()) for w in parse_reply(reply)["emotions"] if slugify(w.lower()) in tax_slugs}
                for reply in s["replies"]
            ]
            emo_games.append(build_games(draw_sets))
            emo_z.append(z_by_id.get(s["id"]))

        print(f"[{run}] fitting: {len(emo_games)} emotion messages, "
              f"{sum(v is not None for v in emo_z)} with probe reads ...", flush=True)
        fit = fit_covariate_model(
            emo_games, emo_z, min_word_draws=MIN_WORD_DRAWS, ridge=RIDGE,
            n_boot=N_BOOT, n_perm=N_PERM, seed=0,
        )
        out[run] = fit
        print(f"[{run}] beta = {fit['beta']:.4f}  CI95 {fit['beta_ci']}, "
              f"perm |beta| q97.5 = {fit['beta_permutation_null']['q975_abs']:.4f}")
        print(f"[{run}] loglik propensity-only {fit['loglik_propensity_only']:.1f} -> "
              f"with-probe {fit['loglik_with_probe']:.1f} over {fit['n_games_fit']} games")

    out_path = common.HERE / "data" / "propensity" / "probe_covariate.json"
    common.write_json(out_path, out)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
