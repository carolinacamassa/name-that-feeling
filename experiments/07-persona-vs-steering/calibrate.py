"""Fix the unit of a steering strength, and check the hook on the real model.

Three things, one container holding the untrained base model, forward passes only:

1. The residual-norm reference. Sofroniew et al. 2026 give steering strengths as a fraction
   of the average norm of the residual stream at the steered layer (their footnote 4). Here
   that norm is measured at every block any positions rule steers at, over the 200 prompts of
   07-persona-activations' pool with the base model's own stored replies in place
   (``steering.residual.token_norm_stats``; the first token of each text is left out of the
   mean and reported apart).
2. What each steering direction is made of: the vector files, and for an anchor mean the
   pairwise cosines among the three anchors and each one's cosine with their mean.
3. The hook check: on the first-person prompt, the residual at the steered block moves by the
   intended vector on the assistant-turn tokens and nowhere else, earlier blocks do not
   move, and a strength of zero leaves the logits unchanged. A forward pass over the prompt
   generates nothing, so the check is made under the paper's rule, ``assistant_prefix``,
   and holds for ``assistant_turn``, which steers the same prompt positions.

Writes ``data/calibration.json``. Run it before anything else here:

    uv run modal run experiments/07-persona-vs-steering/calibrate.py::calibrate

On this machine prefix ``modal run`` with ``PYTHONUTF8=1 PYTHONIOENCODING=utf-8``.
"""

from name_that_feeling.serving.persona_sampler import PersonaSampler, app

import common

CHECK_STRENGTH = 0.5  # the paper's strength on this prompt
CHECK_POSITIONS = "assistant_prefix"


@app.local_entrypoint()
def calibrate(force: bool = False) -> None:
    cfg = common.load_config()
    if common.CALIBRATION_PATH.exists() and not force:
        print(f"{common.CALIBRATION_PATH} exists; pass --force to measure again (every stored strength refers to it)")
        return
    blocks = common.all_blocks(cfg)
    all_directions = common.directions(cfg)
    contexts, provenance = common.norm_pool_contexts(cfg)
    sampler = PersonaSampler(base_model=cfg["base_model"], run_name="")

    print(f"residual norms at blocks {blocks} over {len(contexts)} transcripts", flush=True)
    norms = sampler.residual_norms.remote(contexts, blocks)
    for b in blocks:
        s = norms["by_block"][str(b)]
        print(
            f"  block {b}: mean {s['mean_norm']:.2f}  median {s['median_norm']:.2f}  norm of the mean activation "
            f"{s['norm_of_mean']:.2f}  rms after centering {s['rms_centered']:.2f}  first token {s['first_token_mean_norm']:.1f}  "
            f"({s['n_tokens']} tokens)",
            flush=True,
        )

    reports, checks = {}, []
    paper = next(c for c in cfg["contexts"] if c["id"] == "paper")
    for b in blocks:
        got = sampler.direction_reports.remote(common.vectors_run(cfg), b, [d["emotions"] for d in all_directions])
        reports[str(b)] = {d["id"]: {**d, **r} for d, r in zip(all_directions, got)}
        settings = common.steering_settings(cfg, CHECK_POSITIONS, b, norms["by_block"][str(b)][cfg["steering"]["norm_statistic"]])
        check = sampler.steering_check.remote(common.context_turns(paper), settings, common.steer_row(all_directions[0], CHECK_STRENGTH))
        checks.append({"direction": all_directions[0]["id"], "strength": CHECK_STRENGTH, "steering": settings, **check})
        print(
            f"  block {b} hook check: {check['n_steered_positions']} of {check['n_tokens']} positions steered, added norm "
            f"{check['added_norm']:.2f}, relative error {check['relative_error']:.4f}, change off the steered positions "
            f"{check['max_change_on_unsteered_positions']:.3g}, at the previous block {check['max_change_at_previous_block']:.3g}, "
            f"logit change at strength zero {check['max_logit_change_at_strength_zero']:.3g}",
            flush=True,
        )
        # Exact zeros are expected; the tolerance only allows for run-to-run GPU nondeterminism.
        tolerance = 1e-3 * check["added_norm"]
        problems = [
            name for name, bad in (
                ("the residual did not move by the intended vector", check["relative_error"] > 0.1),
                ("unsteered positions moved", check["max_change_on_unsteered_positions"] > tolerance),
                ("an earlier block moved", check["max_change_at_previous_block"] > tolerance),
                ("strength zero changed the logits", check["max_logit_change_at_strength_zero"] > 0.05),
                ("steering left the logits unchanged", check["max_logit_change_steered"] == 0),
            ) if bad
        ]
        if problems:
            raise RuntimeError(f"block {b}: the hook check failed ({'; '.join(problems)}); nothing written")

    common.write_json(common.CALIBRATION_PATH, {
        "base_model": cfg["base_model"],
        "load": norms["load"],
        "norm_statistic": cfg["steering"]["norm_statistic"],
        "norm_pool": provenance,
        "norms": {"by_block": norms["by_block"]},
        "directions": reports,
        "hook_checks": checks,
    })
    print(f"wrote {common.CALIBRATION_PATH}")
    for b in blocks:
        for d in all_directions:
            if d["kind"] != "three_anchors":
                continue
            r = reports[str(b)][d["id"]]
            pairs = [r["pairwise_cosines"][i][j] for i in range(3) for j in range(i + 1, 3)]
            print(f"  block {b}, {d['id']}: anchors' pairwise cosines {', '.join(f'{c:+.2f}' for c in pairs)}; with their mean {', '.join(f'{c:+.2f}' for c in r['cosine_with_direction'])}")
