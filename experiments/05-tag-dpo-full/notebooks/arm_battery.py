import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
    from name_that_feeling.evals import tag_eval
    from name_that_feeling.evals.probe_teacher import ProbeTeacher
    from name_that_feeling.evals.similarity import EmotionSimilarity
    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return (
        EmotionSimilarity,
        Path,
        ProbeTeacher,
        alt,
        json,
        load_clusters,
        mo,
        pl,
        save_chart,
        slugify,
        tag_eval,
    )


@app.cell
def _(EmotionSimilarity, Path, ProbeTeacher, json, load_clusters):
    HERE = Path(__file__).parents[1]  # the 05-tag-dpo-full experiment dir
    EXPERIMENTS = HERE.parent
    NEUTRAL_TAG = "calm, attentive"

    # The four checkpoints of the dose story: the SFT parent (dose 0), the pilot
    # (164 pairs / 21 steps), and this experiment's two arms.
    STABILITY_FILES = {
        "supervised parent (0 steps)": EXPERIMENTS / "04-sft-seeds-and-epochs" / "data" / "stability" / "two-epochs" / "samples.json",
        "pilot (164 pairs, 21 steps)": EXPERIMENTS / "05-tag-dpo" / "data" / "runs" / "tag-masked-test" / "stability_samples.json",
        "capped-200 (25 steps)": HERE / "data" / "runs" / "capped-200" / "stability_samples.json",
        "uncapped-800 (100 steps)": HERE / "data" / "runs" / "uncapped-800" / "stability_samples.json",
    }
    RUN_ORDER = list(STABILITY_FILES)
    STABILITY = {name: json.loads(p.read_text(encoding="utf-8"))["samples"] for name, p in STABILITY_FILES.items()}

    ACT_FILES = {
        "pilot (164 pairs, 21 steps)": EXPERIMENTS / "05-tag-dpo" / "data" / "runs" / "tag-masked-test" / "activation_distributions.json",
        "capped-200 (25 steps)": HERE / "data" / "runs" / "capped-200" / "activation_distributions.json",
        "uncapped-800 (100 steps)": HERE / "data" / "runs" / "uncapped-800" / "activation_distributions.json",
    }
    ACT = {name: json.loads(p.read_text(encoding="utf-8")) for name, p in ACT_FILES.items()}

    CLUSTERS = load_clusters(EXPERIMENTS / "01-emotion-vectors" / "clusters.json")
    SIM = EmotionSimilarity.load(EXPERIMENTS / "01-emotion-vectors" / "data" / "similarity" / "layer_21.json")
    _split = json.loads((EXPERIMENTS / "03-training-pilot" / "data" / "sft" / "split.json").read_text(encoding="utf-8"))
    _completions = [
        json.loads(x)
        for x in (EXPERIMENTS / "03-training-pilot" / "data" / "completions" / "unconditioned.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    TEACHER = ProbeTeacher.from_completions(_completions, CLUSTERS, _split["tag_config"])
    return ACT, CLUSTERS, NEUTRAL_TAG, RUN_ORDER, SIM, STABILITY, TEACHER


@app.cell
def _(mo):
    mo.md("""
    # Two mixture arms at full scale: where the preference stage breaks

    Both arms trained from the two-epochs SFT checkpoint with the pilot's template
    parameters (β 0.1, lr 5e-5, batch 8, one epoch, tag-masked credit) on pair sets
    drawn from the same 806-pair inventory — `uncapped-800` (the realized 82%-negative
    mixture, 100 steps) and `capped-200` (near-flat charged mixture, 25 steps). The
    exhibits below read the K = 12 temperature-1 sampled evals and the probe readouts
    against the SFT parent and the 164-pair pilot.
    """)
    return


@app.cell
def _(
    RUN_ORDER,
    SIM,
    STABILITY,
    TEACHER,
    alt,
    pl,
    save_chart,
    slugify,
    tag_eval,
):
    def _first_in_tax(emotions):
        for e in emotions:
            s = slugify(e)
            if SIM.index(s) is not None:
                return s
        return None

    _rows = []
    for _run, _samples in STABILITY.items():
        for _set in ("within", "cross"):
            _prompts = [s for s in _samples if s["set"] == _set]
            _n_draws = _compl = 0
            _scores = []
            _cons_right = _cons_wrong = 0
            for _s in _prompts:
                _t = TEACHER.top_word(_s["id"])
                _per_draw = []
                for _r in _s["replies"]:
                    _n_draws += 1
                    _p = tag_eval.parse_reply(_r)
                    _ok = _p["compliant"] and _p["emotions"] and "</emotion>" in _r
                    _compl += _ok
                    _v = SIM.rank_percentile(_t, _first_in_tax(_p["emotions"])) if _ok else None
                    _per_draw.append(_v)
                _scores += [v for v in _per_draw if v is not None]
                if _per_draw and all(v is not None and v >= 0.8 for v in _per_draw):
                    _cons_right += 1
                if _per_draw and all(v is None or v <= 0.4 for v in _per_draw):
                    _cons_wrong += 1
            _rows += [
                {"run": _run, "measure": f"{_set} · compliant draws", "value": _compl / _n_draws},
                {"run": _run, "measure": f"{_set} · consistently right prompts", "value": _cons_right / len(_prompts)},
                {"run": _run, "measure": f"{_set} · consistently wrong prompts", "value": _cons_wrong / len(_prompts)},
            ]
    _order = [f"{s} · {m}" for s in ("within", "cross") for m in ("compliant draws", "consistently right prompts", "consistently wrong prompts")]
    save_chart(
        alt.Chart(pl.DataFrame(_rows))
        .mark_bar()
        .encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
            y=alt.Y("measure:N", sort=_order, title=None, axis=alt.Axis(labelLimit=300)),
            color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), sort=RUN_ORDER, title=None),
            yOffset=alt.YOffset("run:N", sort=RUN_ORDER),
            tooltip=["run", "measure", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(width=430, height=380, title="Tag format and consistency per checkpoint"),
        "channel_health_by_dose",
        caption=(
            "K = 12 temperature-1 draws per held-out prompt for four checkpoints: the two-epochs "
            "SFT parent, the 164-pair pilot, and the two full-scale arms, all trained with "
            "identical template parameters. Compliant = the draw opens and closes the emotion tag; "
            "consistently right/wrong = every draw of the prompt scores ≥ 0.8 / ≤ 0.4 rank "
            "percentile against the frozen probe teacher's top word (a draw that never closes its "
            "tag counts as not-right)."
        ),
        takeaway=(
            "Sharpening scales into degeneration: the pilot and the capped arm triple the parent's "
            "consistently-right share (6% to 30% within) with format intact, but the uncapped arm "
            "at 100 steps loses the format itself — only 42% of its charged draws close the tag "
            "(the rest repeat emotion words indefinitely), consistently-right collapses to 5%/1%, "
            "and consistently-wrong prompts appear at 12%/18%. Its surviving draws score highest "
            "of all checkpoints (0.80 within) — over-sharpening and format collapse are the same "
            "process at different doses, not separate failures."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(NEUTRAL_TAG, RUN_ORDER, STABILITY, alt, pl, save_chart, tag_eval):
    _rows = []
    for _run, _samples in STABILITY.items():
        _neut = [s for s in _samples if s["set"] == "neutral"]
        _counts = {"exact anchor": 0, "anchor + appended word(s)": 0, "other tag": 0}
        _total = 0
        for _s in _neut:
            for _r in _s["replies"]:
                _total += 1
                _tag = ", ".join(tag_eval.parse_reply(_r)["emotions"])
                if _tag == NEUTRAL_TAG:
                    _counts["exact anchor"] += 1
                elif _tag.startswith(NEUTRAL_TAG + ","):
                    _counts["anchor + appended word(s)"] += 1
                else:
                    _counts["other tag"] += 1
        _rows += [{"run": _run, "mode": m, "share": c / _total} for m, c in _counts.items()]
    save_chart(
        alt.Chart(pl.DataFrame(_rows))
        .mark_bar()
        .encode(
            x=alt.X("share:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="share of neutral draws"),
            y=alt.Y("run:N", sort=RUN_ORDER, title=None, axis=alt.Axis(labelLimit=220)),
            color=alt.Color("mode:N", scale=alt.Scale(scheme="tableau10"), title=None,
                            sort=["exact anchor", "anchor + appended word(s)", "other tag"]),
            order=alt.Order("mode:N"),
            tooltip=["run", "mode", alt.Tooltip("share:Q", format=".3f")],
        )
        .properties(width=430, height=200, title="The neutral anchor under each checkpoint"),
        "neutral_anchor_modes_by_dose",
        caption=(
            "Composition of the K = 12 neutral-prompt draws per checkpoint: the exact trained "
            "anchor (calm, attentive), the anchor with one or more words appended (capped-200's "
            "dominant mode is calm, attentive, playful), or any other tag. The uncapped arm "
            "trained on 45 neutral pairs, the capped arm on 11, the pilot on 16."
        ),
        takeaway=(
            "The neutral anchor's fate tracks the arm's neutral-pair count, not its total size: "
            "uncapped-800 (45 neutral pairs) holds the exact anchor on 98% of draws — better than "
            "the parent — while capped-200 (11 pairs) all but loses it (1%), drifting to "
            "anchor-plus-appended-word on 68% of draws; the pilot (16 pairs) sits between at 81%. "
            "The appended word is usually playful-family — the same attractor the corrupted-labels "
            "collapse fell into — so the 'more emotion words' pressure from the charged pairs "
            "captures whichever slice is too thin to resist it."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(ACT, CLUSTERS, alt, pl, save_chart, slugify):
    _emo2fam = {slugify(e): c for c, es in CLUSTERS.items() for e in es}
    _rows = pl.DataFrame(
        [
            {"run": _run, "family": r["family"], "mean_delta": r["mean_delta"]}
            for _run, _ad in ACT.items()
            for r in _ad["comparisons"]["sft_parent_to_run"]["eval_within"]
        ]
    ).group_by("run", "family").agg(pl.col("mean_delta").mean().alias("family_shift"))
    _order = list(ACT)
    save_chart(
        alt.Chart(_rows)
        .mark_bar()
        .encode(
            x=alt.X("family_shift:Q", title="family-mean activation shift vs the SFT parent (base std units)"),
            y=alt.Y("family:N", sort="-x", title=None),
            color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), sort=_order, title=None),
            yOffset=alt.YOffset("run:N", sort=_order),
            tooltip=["run", "family", alt.Tooltip("family_shift:Q", format="+.3f")],
        )
        .properties(width=440, height=340, title="Each preference run's activation shift, by family"),
        "arm_family_tilt_vs_pilot",
        caption=(
            "Family-mean pre-response activation shift introduced by each preference run relative "
            "to the shared two-epochs parent, on the held-out within-family messages (projections "
            "on the fixed base vectors, base-std units). The pilot's mixture was hostile-heavy, "
            "the uncapped arm's is 82% negative, the capped arm's is near-flat."
        ),
        takeaway=(
            "Scale multiplies the state shift and mixture steers it, but no mixture keeps the "
            "state still: the uncapped arm collapses the calm/peaceful end by 0.5-0.9 base std "
            "(several times the pilot's largest movement) while the capped arm — a quarter the "
            "pairs, a flat mixture, 25 steps — still moves families by up to ~0.4, with a "
            "different profile than either negative-heavy run. At template parameters the "
            "preference stage's optimization pressure, not its mixture, is the first-order driver "
            "of representational drift."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Greedy-battery and probe context (numbers, no exhibit)

    Greedy battery: capped-200 keeps format (99%) and roughly the parent's mapping
    (family-vs-teacher 51% within / 40% cross vs 53%/44%), with the graded within read
    at 0.665 rank percentile (parent 0.714). Uncapped-800's greedy compliance is 18%
    within — the runaway-tag mode — with its scorable minority at 0.81. Against the
    current-probe teacher both arms score modestly better than against the frozen one
    (capped within 1-vs-3: 0.564 → 0.581, paired Δ +0.016 [+0.006, +0.027]); the
    uncapped arm's current-probe teacher itself has drifted on 37% of top words, the
    largest drift measured. Leakage and capability were judged for capped-200 only —
    83% of the uncapped arm's visible replies are empty (the tag never closes), so a
    tonal judgment would be meaningless. Capped-200's judge reads are the first
    non-null movement in any run: visible-reply tonal charge 1.475 vs the parent's
    1.387 (base 1.325, same 80-item protocol) and capability at-least-equal 76%
    vs the parent's 82% — small shifts, but the preference stage is no longer
    leaving the visible behaviour untouched.
    """)
    return


if __name__ == "__main__":
    app.run()
