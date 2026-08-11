import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, load_clusters, mo, np, pl, save_chart, slugify


@app.cell
def _(Path, json, load_clusters):
    HERE = Path(__file__).parents[1]  # the 05-tag-dpo experiment dir
    EXPERIMENTS = HERE.parent
    RUN_DIR = HERE / "data" / "runs" / "tag-masked-test"

    DPO_EVAL = json.loads((RUN_DIR / "eval.json").read_text(encoding="utf-8"))
    DPO_JUDGE = json.loads((RUN_DIR / "eval_judge.json").read_text(encoding="utf-8"))
    BODY_SIM = json.loads((HERE / "data" / "pairs" / "body_similarity.json").read_text(encoding="utf-8"))
    PAIRS_META = json.loads((HERE / "data" / "pairs" / "meta.json").read_text(encoding="utf-8"))

    # The SFT parent's battery: eval.json (binary metrics) predates the distance rollout,
    # so its graded columns come from 04-sft's runs_summary.json (recomputed from stored samples).
    _sft = EXPERIMENTS / "04-sft-seeds-and-epochs"
    TWO_EVAL = json.loads((_sft / "data" / "runs" / "two-epochs" / "eval.json").read_text(encoding="utf-8"))
    TWO_SUMMARY = next(
        r
        for r in json.loads((_sft / "data" / "cross" / "runs_summary.json").read_text(encoding="utf-8"))
        if r["run"] == "two-epochs"
    )

    # Readouts on all 1972 elicited messages, projected onto the base emotion vectors.
    BASE_READOUT = json.loads(
        (EXPERIMENTS / "02-elicited-activations" / "data" / "qwen3.5-9b" / "readout.json").read_text(encoding="utf-8")
    )
    TWO_READOUT = json.loads(
        (_sft / "data" / "runs" / "two-epochs" / "readout_full_base_vectors.json").read_text(encoding="utf-8")
    )
    DPO_READOUT = json.loads((RUN_DIR / "readout_full_base_vectors.json").read_text(encoding="utf-8"))

    CLUSTERS = load_clusters(EXPERIMENTS / "01-emotion-vectors" / "clusters.json")
    return (
        BASE_READOUT,
        BODY_SIM,
        CLUSTERS,
        DPO_EVAL,
        DPO_JUDGE,
        DPO_READOUT,
        PAIRS_META,
        TWO_EVAL,
        TWO_READOUT,
        TWO_SUMMARY,
    )


@app.cell
def _(PAIRS_META, mo):
    mo.md(f"""
    # Post-run battery for the tag-masked DPO checkpoint

    The regression checks description §7 left pending, all against the `two-epochs` SFT
    parent under identical protocols: the greedy held-out battery, the activation-tilt
    re-measure on all 1,972 elicited messages (mandatory — the
    {PAIRS_META["n_pairs"]}-pair training set is skewed toward negative families:
    hostile {PAIRS_META["by_family"]["hostile_anger"]}, fear
    {PAIRS_META["by_family"]["fear_and_overwhelm"]}, despair
    {PAIRS_META["by_family"]["despair_and_shame"]}, peaceful and pride 0), the
    judge-based leakage and capability reads, and a within-pair body-similarity check
    (are the chosen and rejected replies different *after* the tag?).
    """)
    return


@app.cell
def _(BASE_READOUT, CLUSTERS, DPO_READOUT, TWO_READOUT, np, pl, slugify):
    # Per-emotion effect sizes vs base, seed_stability.py's formula:
    # (mean_run - mean_base) / std_base over the layer-21 projections of all 1972 messages.
    _base_by_id = {m["id"]: m for m in BASE_READOUT["messages"]}
    EMOTIONS = sorted(TWO_READOUT["messages"][0]["projections"])
    _emo2fam = {slugify(e): c for c, es in CLUSTERS.items() for e in es}

    def _effect_sizes(run_msgs: list[dict]) -> "np.ndarray":
        ids = [m["id"] for m in run_msgs if m["id"] in _base_by_id]
        mb = np.array([[_base_by_id[i]["projections"][e] for e in EMOTIONS] for i in ids])
        mt = np.array([[m["projections"][e] for e in EMOTIONS] for m in run_msgs if m["id"] in _base_by_id])
        sb = mb.std(axis=0)
        return (mt.mean(axis=0) - mb.mean(axis=0)) / np.where(sb == 0, 1.0, sb)

    _eff = {"two-epochs (SFT)": _effect_sizes(TWO_READOUT["messages"]), "tag-masked DPO": _effect_sizes(DPO_READOUT["messages"])}
    EFFECTS = pl.DataFrame(
        [
            {"run": run, "emotion": e, "family": _emo2fam.get(e, "?"), "effect_size": float(v)}
            for run, vals in _eff.items()
            for e, v in zip(EMOTIONS, vals)
        ]
    )
    TILT_CORR = float(np.corrcoef(_eff["two-epochs (SFT)"], _eff["tag-masked DPO"])[0, 1])
    DELTA = {e: float(d) for e, d in zip(EMOTIONS, _eff["tag-masked DPO"] - _eff["two-epochs (SFT)"])}
    return DELTA, EFFECTS, TILT_CORR


@app.cell
def _(EFFECTS, TILT_CORR, alt, pl, save_chart):
    _fam = EFFECTS.group_by("run", "family").agg(pl.col("effect_size").mean().alias("mean_effect"))
    save_chart(
        alt.Chart(_fam)
        .mark_bar()
        .encode(
            x=alt.X("mean_effect:Q", title="mean activation shift vs base (Δmean / base std)"),
            y=alt.Y("family:N", sort="-x", title=None),
            color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), title=None),
            yOffset="run:N",
            tooltip=["run", "family", alt.Tooltip("mean_effect:Q", format=".3f")],
        )
        .properties(width=440, height=320, title="Family-level activation tilt: SFT parent vs tag-masked DPO"),
        "post_dpo_family_tilt",
        caption=(
            "Mean pre-response activation shift per emotion family, in units of the base model's "
            "per-emotion standard deviation, over all 1,972 elicited messages projected onto the "
            "fixed base emotion vectors at layer 21. The two-epochs SFT parent (the checkpoint the "
            "DPO step started from) sits near zero on every family; the tag-masked DPO checkpoint "
            "was trained for 21 steps on 164 preference pairs whose messages skew toward negative "
            "families (hostile 49, fear 34, despair 34; peaceful and pride 0)."
        ),
        takeaway=(
            "One epoch of tag-masked DPO moves the internal emotion profile more than the entire "
            "SFT did: the median per-emotion shift magnitude rises from 0.03 to 0.12 base standard "
            "deviations, with fear-and-overwhelm (+0.11) and vigilant-suspicion (+0.21) up and "
            "peaceful-contentment (-0.20) and compassionate-gratitude (-0.15) down — the direction "
            f"of the pair pool's family skew. The per-emotion correlation with the SFT tilt is {TILT_CORR:.2f}, "
            "so this is a new movement, not an amplification of the SFT's residual tilt."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(DELTA, EFFECTS, alt, pl, save_chart):
    _top = sorted(DELTA, key=lambda e: -abs(DELTA[e]))[:18]
    _movers = EFFECTS.filter(pl.col("emotion").is_in(_top))
    save_chart(
        alt.Chart(_movers)
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X("effect_size:Q", title="activation shift vs base (Δmean / base std)"),
            y=alt.Y("emotion:N", sort=_top, title=None),
            color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), title=None),
            shape="run:N",
            tooltip=["run", "emotion", "family", alt.Tooltip("effect_size:Q", format=".3f")],
        )
        .properties(width=420, height=380, title="Largest per-emotion movements introduced by the DPO step"),
        "post_dpo_tilt_top_movers",
        caption=(
            "The 18 emotions whose activation shift changed most between the two-epochs SFT parent "
            "and the tag-masked DPO checkpoint (same projection as the family exhibit). Each emotion "
            "shows both checkpoints' shifts vs the base model."
        ),
        takeaway=(
            "The DPO step's movement is a coherent anxiety cluster: nervous, on-edge, anxious, "
            "worried, stressed, tense, scared, afraid and their neighbours all rise 0.2-0.3 base "
            "standard deviations from near zero, while serene and docile fall by about 0.2. The "
            "checkpoint reads as more anxious and vigilant while processing every message, although "
            "the training only ever credited tag tokens."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(DPO_EVAL, TWO_EVAL, TWO_SUMMARY, alt, pl, save_chart):
    _rows = []
    for _run, _fam_src, _dist in (
        ("two-epochs (SFT)", TWO_EVAL, None),
        ("tag-masked DPO", DPO_EVAL, DPO_EVAL.get("distance_generalization")),
    ):
        for _set in ("within", "cross"):
            _g = _fam_src["generalization"][_set]
            _d = _dist[_set] if _dist else {
                "model_vs_teacher_cosine_mean": TWO_SUMMARY[f"dist_{_set}_model_vs_teacher_cosine"],
                "model_rank_pct_first_mean": TWO_SUMMARY[f"dist_{_set}_model_rank_pct"],
            }
            _rows += [
                {"run": _run, "measure": f"{_set} · family agreement vs probe teacher", "value": _g["model_vs_teacher_agreement"]},
                {"run": _run, "measure": f"{_set} · cosine vs probe teacher (graded)", "value": _d["model_vs_teacher_cosine_mean"]},
                {"run": _run, "measure": f"{_set} · rank percentile vs elicited (graded)", "value": _d["model_rank_pct_first_mean"]},
            ]
        _rows.append({"run": _run, "measure": "neutral · exact anchor rate", "value": _fam_src["neutral_anchor"]["exact_neutral_rate"]})
    _df = pl.DataFrame(_rows)
    _order = [
        "within · family agreement vs probe teacher",
        "within · cosine vs probe teacher (graded)",
        "within · rank percentile vs elicited (graded)",
        "cross · family agreement vs probe teacher",
        "cross · cosine vs probe teacher (graded)",
        "cross · rank percentile vs elicited (graded)",
        "neutral · exact anchor rate",
    ]
    save_chart(
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title=None),
            y=alt.Y("measure:N", sort=_order, title=None, axis=alt.Axis(labelLimit=290)),
            color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), title=None,
                            sort=["two-epochs (SFT)", "tag-masked DPO"]),
            yOffset=alt.YOffset("run:N", sort=["two-epochs (SFT)", "tag-masked DPO"]),
            tooltip=["run", "measure", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(width=430, height=330, title="Greedy held-out battery, SFT parent vs tag-masked DPO"),
        "greedy_battery_before_after",
        caption=(
            "The standard greedy battery (full-length replies, 260 within / 77 cross / 50 neutral "
            "held-out messages), both metric families side by side: binary family agreement with the "
            "probe teacher, graded cosine against the teacher's top word, graded rank percentile "
            "against the elicited emotion, and the exact neutral anchor rate. Two-epochs' graded "
            "columns are recomputed from its stored samples (runs_summary.json); the DPO "
            "checkpoint's come from its own eval.json under the identical pipeline."
        ),
        takeaway=(
            "The greedy mode trades within-family fidelity for cross-family reach: on trained "
            "families every form falls (family agreement vs teacher 53% to 48%, cosine vs teacher "
            "0.54 to 0.45, rank percentile 0.71 to 0.63), while on never-trained families reach "
            "improves (rank percentile 0.69 to 0.75, unseen-family recall 42% to 56%, family "
            "agreement 44% to 46%). The exact neutral anchor drops from 100% to 82%, but every "
            "deviation is the truncation 'calm' — zero charged tags on neutral messages, so the "
            "targeted failure mode stays eliminated at greedy too."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(BODY_SIM, alt, pl, save_chart):
    _s = BODY_SIM["summary"]["charged"]
    _f = BODY_SIM["summary"]["cross_prompt_floor"]
    _df = pl.DataFrame(
        [
            {"comparison": "chosen vs rejected (the pair)", "metric": "word-count cosine", "value": _s["pair_word_cosine"]["mean"]},
            {"comparison": "any two draws, same prompt", "metric": "word-count cosine", "value": _s["same_prompt_word_cosine"]["mean"]},
            {"comparison": "different prompts (floor)", "metric": "word-count cosine", "value": _f["word_cosine"]["mean"]},
            {"comparison": "chosen vs rejected (the pair)", "metric": "content-word Jaccard", "value": _s["pair_content_jaccard"]["mean"]},
            {"comparison": "any two draws, same prompt", "metric": "content-word Jaccard", "value": _s["same_prompt_content_jaccard"]["mean"]},
            {"comparison": "different prompts (floor)", "metric": "content-word Jaccard", "value": _f["content_jaccard"]["mean"]},
        ]
    )
    _order = ["chosen vs rejected (the pair)", "any two draws, same prompt", "different prompts (floor)"]
    save_chart(
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X("comparison:N", sort=_order, title=None, axis=alt.Axis(labelAngle=-20, labelLimit=200)),
            y=alt.Y("value:Q", title="mean lexical similarity of the reply bodies"),
            color=alt.Color("comparison:N", sort=_order, scale=alt.Scale(scheme="tableau10"), legend=None),
            column=alt.Column("metric:N", title=None),
            tooltip=["comparison", "metric", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(width=220, height=240)
        .properties(title="Within a preference pair, how different are the bodies after the tag?"),
        "pair_body_similarity_check",
        caption=(
            "Lexical similarity of the tag-stripped reply bodies on the 148 charged preference "
            "pairs: the chosen vs rejected pair itself, the mean over all draw pairs of the same "
            "prompt from the K=12 pool (what any two samples of that prompt look like), and a "
            "different-prompts floor (each chosen body vs another prompt's rejected body). "
            "Word-count cosine and stopword-filtered Jaccard; chosen and rejected bodies are also "
            "length-matched (means 314 vs 314 words, chosen longer on 49% of pairs). The 16 "
            "neutral pairs behave the same (pair cosine 0.84 vs same-prompt 0.82)."
        ),
        takeaway=(
            "A pair's chosen and rejected bodies are exactly as similar as any two draws of the "
            "same prompt (cosine 0.745 vs 0.753; Jaccard 0.241 vs 0.245, floor 0.424/0.029): at "
            "the pair extremes the emitted tag does not detectably condition the body, so tag-"
            "masked credit forfeited no body-side signal, and a whole-sequence credit arm would "
            "mostly add body noise to the preference margin. Lexical similarity cannot rule out "
            "purely tonal differences."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(Path, alt, json, pl, save_chart):
    _ad = json.loads(
        (Path(__file__).parents[1] / "data" / "runs" / "tag-masked-test" / "activation_distributions.json").read_text(
            encoding="utf-8"
        )
    )
    _rows = pl.DataFrame(_ad["comparisons"]["sft_parent_to_run"]["eval_within"])
    save_chart(
        alt.Chart(_rows)
        .mark_circle(size=55, opacity=0.7)
        .encode(
            x=alt.X("mean_delta:Q", title="mean per-message shift (uniform component, base std units)"),
            y=alt.Y("std_delta:Q", title="std of per-message shifts (message-selective component)"),
            color=alt.Color("family:N", scale=alt.Scale(scheme="tableau10"), title=None),
            tooltip=[
                "emotion",
                "family",
                alt.Tooltip("mean_delta:Q", format="+.3f"),
                alt.Tooltip("std_delta:Q", format=".3f"),
                alt.Tooltip("uniform_share:Q", format=".0%"),
                alt.Tooltip("wasserstein1:Q", format=".3f"),
            ],
        )
        .properties(
            width=440, height=300,
            title="Decomposing the DPO step's activation shift: uniform offset vs message-selective",
        ),
        "dpo_shift_uniform_vs_selective",
        caption=(
            "Each point is one of the 171 emotions; the DPO step's per-message activation shift "
            "(tag-masked checkpoint minus its two-epochs SFT parent, projections on the fixed base "
            "vectors, base-std units) is decomposed into its mean over messages (x — a uniform "
            "offset applied to every message) and its standard deviation across messages (y — "
            "message-selective re-reading). Held-out within-family messages only (n = 260), so "
            "trained-message instance effects are excluded. Standardized Wasserstein-1 between the "
            "before/after marginals is in the tooltip; for the large movers it equals the absolute "
            "mean shift, i.e. a pure location shift with no shape change."
        ),
        takeaway=(
            "The DPO step's anxiety shift is mostly a constant recalibration, not a re-reading of "
            "specific messages: the top movers (nervous, anxious, on-edge, worried, all near "
            "+0.32 base std) carry 77-81% of their squared movement in the uniform component, "
            "against a 57% median across all emotions, and their Wasserstein-1 equals their mean "
            "shift exactly. This is also why the recomputed-statistics current-probe teacher "
            "barely drifted: per-emotion standardization absorbs precisely this uniform component. "
            "The SFT parent's own shifts vs base (not shown) are three times smaller and more "
            "message-selective."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(Path, alt, json, pl, save_chart):
    _ad2 = json.loads(
        (Path(__file__).parents[1] / "data" / "runs" / "tag-masked-test" / "activation_distributions.json").read_text(
            encoding="utf-8"
        )
    )
    _subset_label = {
        "train": "SFT train (576)",
        "eval_within": "eval within (260)",
        "eval_cross": "eval cross (77)",
        "dpo_pair": "DPO pair messages (148)",
        "dpo_pool_only": "DPO pool, unpaired (202)",
        "unused": "untouched unused (709)",
    }
    _fam_rows = pl.DataFrame(
        [
            {"subset": _subset_label[sub], "family": r["family"], "mean_delta": r["mean_delta"]}
            for sub, rows in _ad2["comparisons"]["sft_parent_to_run"].items()
            for r in rows
        ]
    ).group_by("subset", "family").agg(pl.col("mean_delta").mean().alias("family_shift"))
    save_chart(
        alt.Chart(_fam_rows)
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X("family_shift:Q", title="family-mean activation shift, DPO step (base std units)"),
            y=alt.Y("family:N", sort="-x", title=None),
            color=alt.Color("subset:N", scale=alt.Scale(scheme="tableau10"), title="message subset"),
            shape="subset:N",
            tooltip=["subset", "family", alt.Tooltip("family_shift:Q", format="+.3f")],
        )
        .properties(width=440, height=320, title="The DPO step's family shift, replicated across message subsets"),
        "dpo_shift_by_family_and_split",
        caption=(
            "Family-mean activation shift introduced by the DPO step (tag-masked checkpoint minus "
            "its two-epochs SFT parent, base-std units), computed independently on six disjoint "
            "message subsets: the SFT training messages, both held-out eval sets, the 148 messages "
            "the DPO pairs were built from, the 202 pool messages that yielded no pair, and the "
            "709 messages untouched by any training or sampling. Per-emotion shift vectors "
            "correlate r = 0.976-0.996 between subsets."
        ),
        takeaway=(
            "The anxiety-up / calm-down family pattern reproduces on every subset, including the "
            "709 messages no stage of training or sampling ever touched — the DPO step changed how "
            "the model reads everything, not how it reads its training items. Magnitudes vary "
            "somewhat by subset (largest on the pair messages themselves), but subset family "
            "composition confounds magnitude comparisons; the replicated direction is the claim."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(Path, alt, json, pl, save_chart):
    _ad3 = json.loads(
        (Path(__file__).parents[1] / "data" / "runs" / "tag-masked-test" / "activation_distributions.json").read_text(
            encoding="utf-8"
        )
    )
    _steps = {"supervised stage (1,152 examples)": "base_to_sft_parent", "preference stage (164 pairs)": "sft_parent_to_run"}
    _w1_rows = pl.DataFrame(
        [
            {"step": step, "family": r["family"], "w1": r["wasserstein1"]}
            for step, comp in _steps.items()
            for r in _ad3["comparisons"][comp]["eval_within"]
        ]
    ).group_by("step", "family").agg(pl.col("w1").mean().alias("family_w1"))
    save_chart(
        alt.Chart(_w1_rows)
        .mark_bar()
        .encode(
            x=alt.X("family_w1:Q", title="family-mean Wasserstein-1, before vs after (base std units)"),
            y=alt.Y("family:N", sort="-x", title=None),
            color=alt.Color("step:N", scale=alt.Scale(scheme="tableau10"), title=None),
            yOffset="step:N",
            tooltip=["step", "family", alt.Tooltip("family_w1:Q", format=".3f")],
        )
        .properties(width=440, height=320, title="How far each training step moved the activation distributions"),
        "sft_step_vs_dpo_step_distribution_shift",
        caption=(
            "Family-mean standardized Wasserstein-1 distance between before and after activation "
            "distributions for the two training steps, on the held-out within-family messages "
            "(n = 260): the full SFT (1,076 examples, two epochs, 1,152 gradient examples) vs the "
            "DPO step (164 pairs, one epoch, 21 steps, tag tokens only in the loss)."
        ),
        takeaway=(
            "The 21-step tag-masked DPO moved the activation distributions further than the entire "
            "SFT in every family, typically by a factor of two to four — fear-and-overwhelm and "
            "vigilant-suspicion most (0.15-0.20 vs the SFT's 0.04-0.06). Per token of training "
            "signal the preference stage is by far the more state-moving intervention, consistent "
            "with §3's expectation that an objective touching the internals is where "
            "representational drift actually appears."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(Path, alt, json, pl, save_chart):
    _pp = json.loads(
        (Path(__file__).parents[1] / "data" / "post_training_probe" / "scores.json").read_text(encoding="utf-8")
    )
    _label = {"two-epochs": "supervised (SFT)", "tag-masked-test": "preference (DPO)"}
    _metric = {
        "model_vs_teacher_agreement": "family agreement",
        "model_vs_teacher_cosine_mean": "cosine, top word (1-vs-1)",
        "model_vs_teacher_centroid_mean": "cosine, centroid (1-vs-3)",
    }
    _rows = [
        {
            "measure": f"{_label[run]} · {s} · {_metric[m]}",
            "teacher": {"frozen_probe": "frozen base probe", "current_probe": "current model's probe"}[variant],
            "value": scores[m],
        }
        for run, r in _pp["runs"].items()
        for s in ("within", "cross")
        for variant in ("frozen_probe", "current_probe")
        for scores in (r["scores"][s][variant],)
        for m in _metric
    ]
    _order = [
        f"{_label[run]} · {s} · {mname}"
        for run in ("two-epochs", "tag-masked-test")
        for s in ("within", "cross")
        for mname in _metric.values()
    ]
    _drift = {run: r["teacher_drift"]["all_messages"] for run, r in _pp["runs"].items()}
    save_chart(
        alt.Chart(pl.DataFrame(_rows))
        .mark_bar()
        .encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title=None),
            y=alt.Y("measure:N", sort=_order, title=None, axis=alt.Axis(labelLimit=310)),
            color=alt.Color("teacher:N", scale=alt.Scale(scheme="tableau10"), title="teacher label source",
                            sort=["frozen base probe", "current model's probe"]),
            yOffset=alt.YOffset("teacher:N", sort=["frozen base probe", "current model's probe"]),
            tooltip=["measure", "teacher", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(width=430, height=420, title="Tag accuracy against the frozen vs the current model's probe read"),
        "tag_accuracy_frozen_vs_current_probe",
        caption=(
            "Greedy held-out tag accuracy on the three headline self-report metrics — binary "
            "family agreement, cosine against the teacher's top-mass word (1-vs-1), and cosine "
            "against the teacher's mass-weighted tag centroid (1-vs-3) — each scored against two "
            "teacher labels: the frozen teacher (the locked selection pipeline on the base model's "
            "projections, the battery's standard yardstick) and the current model's probe read "
            "(the identical pipeline applied to the checkpoint's own stored readout projections, "
            "per-emotion statistics recomputed over the same 1,972 messages). Validity gate: the "
            "pipeline reproduces the frozen teacher exactly (100.0% same top word) when applied to "
            "the base model's readout. The two teachers agree on "
            f"{_drift['two-epochs']['same_top_word']:.0%} of top words (92% of families) for the "
            f"SFT checkpoint and {_drift['tag-masked-test']['same_top_word']:.0%} (91%) after DPO; "
            "disagreements are mostly near-synonym flips (mean cosine between the two teachers' "
            "words 0.92 and 0.90)."
        ),
        takeaway=(
            "The current-probe advantage is small everywhere, and paired per-record bootstrap "
            "intervals (current minus frozen, resampling prompts) resolve it only for the DPO "
            "checkpoint on the centroid form: +0.017 [+0.003, +0.033] within and +0.029 [+0.013, "
            "+0.050] cross on 1-vs-3 cosine, while every SFT interval and both 1-vs-1 intervals "
            "include zero. So the data support a modest statement: after DPO — the stage that "
            "moved the activations — the emitted tag agrees slightly but resolvably better with "
            "the model's current probe read than with the frozen training labels; for the SFT "
            "checkpoint the two teachers are statistically indistinguishable as yardsticks. The "
            "within-family drop vs the SFT parent persists under either teacher. Correlated "
            "drift of channel and probe toward the same families is not excluded; the "
            "steering-based coupling eval remains the decisive test."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(DPO_JUDGE, mo):
    _leak = DPO_JUDGE["leakage"]
    _cap = DPO_JUDGE["capability"]
    _mc = _leak["mean_tonal_charge"]
    _rate = _cap["at_least_equal_rate"]
    mo.md(f"""
    ## Judge reads: leakage and capability (no exhibit — null results)

    Same protocol as the pilot's judge eval (three tag-stripped replies per item, order
    rotated, Llama-3.3-70B judge; n = {_leak["n_judged"]} leakage items, matching the
    pilot's n = 80): visible-reply tonal charge is **base {_mc["base"]:.2f}, two-epochs
    {_mc["two-epochs"]:.2f}, DPO {_mc["tag-masked-test"]:.2f}** (0–3 scale) — the DPO
    checkpoint is *identical* to its SFT parent, both marginally above base
    (more-charged-than-base {_leak["more_charged_than_base_rate"]["tag-masked-test"]:.0%}
    for each). Capability on the {_cap["n_judged"]} neutral tasks:
    at-least-equal-to-base **{_rate["tag-masked-test"]:.0%}** (two-epochs
    {_rate["two-epochs"]:.0%}). Tag-masked credit left the visible behaviour untouched,
    consistent with the body-similarity check — while the *internal* profile moved
    (family-tilt exhibit). Probe integrity beneath the tilt: per-message profile
    correlation with base 0.98 (SFT: 0.998), and the elicited emotion's within-profile
    z-score is preserved (base 0.71, SFT 0.70, DPO 0.81).
    """)
    return


if __name__ == "__main__":
    app.run()
