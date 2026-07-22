import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, mo, pl, save_chart


@app.cell
def _(Path, json):
    HERE = Path(__file__).parents[1]  # the 02-prompted-base-tag-baseline experiment dir

    # The two full-battery prompt arms (run.py writes eval.json per run).
    ARMS = {
        "prompted base, open vocabulary": "format-spec-explicit-tag",
        "prompted base, 171-word list": "full-vocabulary-list",
    }
    EVALS = {
        label: json.loads((HERE / "data" / "runs" / run / "eval.json").read_text(encoding="utf-8"))
        for label, run in ARMS.items()
    }

    # Trained reference: the gold-standard two-epochs run's full-set battery, re-scored
    # with the distance metrics (04-sft-seeds-and-epochs summarizer output).
    TRAINED_LABEL = "trained SFT, two epochs (unprompted)"
    _summary_path = HERE.parent / "04-sft-seeds-and-epochs" / "data" / "cross" / "runs_summary.json"
    _rows = json.loads(_summary_path.read_text(encoding="utf-8")) if _summary_path.exists() else []
    TRAINED = next((r for r in _rows if r.get("run") == "two-epochs"), None)

    MODEL_ORDER = list(ARMS) + [TRAINED_LABEL]
    _g = EVALS["prompted base, 171-word list"]["generalization"]["within"]
    CHANCE_WITHIN = _g["chance_biggest_family"]
    TEACHER_CEILING_WITHIN = _g["teacher_cluster_agreement"]
    return CHANCE_WITHIN, EVALS, MODEL_ORDER, TEACHER_CEILING_WITHIN, TRAINED, TRAINED_LABEL


@app.cell
def _(mo):
    mo.md("""
    # How far does instruction alone get? The prompted-base tag baseline

    The untouched base model is asked, via a system prompt, to open each reply with an
    `<emotion>` tag naming the emotions the exchange brings up for it, then answer
    normally — the zero-training control every training claim must beat. Two prompt arms
    run the full held-out battery (260 within-family, 77 cross-family, 50 neutral
    messages): an open-vocabulary prompt (no emotion words named), and the same prompt
    with the full 171-word taxonomy vocabulary listed alphabetically.

    Each exhibit reports both measurement forms: the binary family-agreement score
    (does the first in-taxonomy word of the emitted tag share the reference emotion's
    family) and the graded distance metrics on the emotion-vector geometry (cosine and
    within-taxonomy rank percentile). Two comparison caveats apply throughout: trained
    checkpoints are sampled without a system prompt, and the open-vocabulary arm's
    distance metrics cover only the 37% of within-family records whose tag contains an
    in-taxonomy word.
    """)
    return


@app.cell
def _(CHANCE_WITHIN, EVALS, MODEL_ORDER, TEACHER_CEILING_WITHIN, TRAINED, TRAINED_LABEL, alt, pl, save_chart):
    _rows = [
        {
            "model": _label,
            "reference": _ref_label,
            "agreement": _e["generalization"]["within"][_key],
        }
        for _label, _e in EVALS.items()
        for _ref_label, _key in (
            ("elicited emotion", "model_cluster_agreement"),
            ("probe teacher", "model_vs_teacher_agreement"),
        )
    ]
    if TRAINED is not None:
        _rows += [
            {"model": TRAINED_LABEL, "reference": "elicited emotion", "agreement": TRAINED["within_model_vs_elicited"]},
            {"model": TRAINED_LABEL, "reference": "probe teacher", "agreement": TRAINED["within_model_vs_teacher"]},
        ]
    _long = pl.DataFrame(_rows)
    _bars = (
        alt.Chart(_long)
        .mark_bar()
        .encode(
            x=alt.X("model:N", sort=MODEL_ORDER, title=None, axis=alt.Axis(labelAngle=-20, labelLimit=200)),
            xOffset=alt.XOffset("reference:N", sort=["elicited emotion", "probe teacher"]),
            y=alt.Y(
                "agreement:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
                title="family agreement (within-family set)",
            ),
            color=alt.Color(
                "reference:N",
                scale=alt.Scale(scheme="tableau10"),
                sort=["elicited emotion", "probe teacher"],
                title="scored against",
            ),
            tooltip=["model", "reference", alt.Tooltip("agreement:Q", format=".3f")],
        )
    )
    _chance_rule = alt.Chart(_long).mark_rule(strokeDash=[5, 4]).encode(y=alt.datum(CHANCE_WITHIN))
    _ceiling_rule = alt.Chart(_long).mark_rule(strokeDash=[2, 3]).encode(y=alt.datum(TEACHER_CEILING_WITHIN))
    save_chart(
        alt.layer(_bars, _chance_rule, _ceiling_rule).properties(
            width=380, height=260, title="Binary family agreement, prompted arms vs the trained checkpoint"
        ),
        "family_agreement_prompted_vs_trained",
        caption=(
            "Family-level agreement of the emitted tag on the 260 held-out within-family messages, against two "
            "references: the emotion each message was written to elicit, and the probe-derived teacher tag for the "
            "same message. Models: the prompted base with an open vocabulary, the prompted base given the full "
            "171-word vocabulary, and the trained two-epoch SFT checkpoint sampled without a system prompt. "
            "Dashed lines: the always-guess-the-largest-family baseline (15%) and the probe teacher's own agreement "
            "with the elicited emotion (37%), the ceiling the weak labels support."
        ),
        takeaway=(
            "Against the elicited emotion, the prompted base with the vocabulary list reaches 40% family agreement — "
            "matching the trained checkpoint's 40% and exceeding the probe teacher's 37% — so supervised training "
            "adds no measurable within-family accuracy over instruction plus the word list. Against the probe "
            "teacher the ordering reverses: the trained checkpoint reaches 53% where the prompted arms stay at "
            "16% and 35%, indicating that what training installs beyond promptable ability is agreement with the "
            "probe's specific labeling function. The open-vocabulary arm sits below chance against the elicited "
            "family (12%) because its free vocabulary rarely lands inside the taxonomy at all."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(EVALS, MODEL_ORDER, TRAINED, TRAINED_LABEL, alt, pl, save_chart):
    _set_sizes = {"within": 260, "cross": 77}
    _rows = [
        {
            "model": _label,
            "set": _set,
            "rank_pct": _d["model_rank_pct_first_mean"],
            "scorable": f"n={_d['model_rank_pct_first_n']}/{_set_sizes[_set]}",
        }
        for _label, _e in EVALS.items()
        for _set, _d in _e["distance_generalization"].items()
    ]
    if TRAINED is not None:
        _rows += [
            {
                "model": TRAINED_LABEL,
                "set": _set,
                "rank_pct": TRAINED[f"dist_{_set}_model_rank_pct"],
                "scorable": f"n≈{_set_sizes[_set]}/{_set_sizes[_set]}",
            }
            for _set in ("within", "cross")
        ]
    _long = pl.DataFrame(_rows)
    _base = alt.Chart(_long).encode(
        x=alt.X("model:N", sort=MODEL_ORDER, title=None, axis=alt.Axis(labelAngle=-20, labelLimit=200)),
    )
    _bars = _base.mark_bar().encode(
        y=alt.Y(
            "rank_pct:Q",
            scale=alt.Scale(domain=[0, 1]),
            title="similarity rank percentile of the emitted tag",
        ),
        color=alt.Color("model:N", sort=MODEL_ORDER, scale=alt.Scale(scheme="tableau10"), legend=None),
        tooltip=["model", "set", alt.Tooltip("rank_pct:Q", format=".3f"), "scorable"],
    )
    _labels = _base.mark_text(dy=-6, fontSize=10).encode(y="rank_pct:Q", text="scorable:N")
    _rule = alt.Chart(_long).mark_rule(strokeDash=[5, 4]).encode(y=alt.datum(0.5))
    save_chart(
        alt.layer(_bars, _labels, _rule)
        .properties(width=240, height=240)
        .facet(column=alt.Column("set:N", title=None, sort=["within", "cross"]))
        .properties(title="Graded similarity to the elicited emotion, prompted arms vs the trained checkpoint"),
        "graded_similarity_prompted_vs_trained",
        caption=(
            "Mean rank percentile of the emitted tag's first in-taxonomy word among all 171 emotions, ranked by "
            "emotion-vector cosine similarity to the message's elicited emotion (1.0 names the elicited emotion "
            "exactly; 0.5 is the uniform-guess chance level, dashed line). Bar labels give the number of scorable "
            "records — records whose tag contains an in-taxonomy word — over the set size; the open-vocabulary "
            "arm scores only 37% of within-family records, and those are plausibly the easier, strongly valenced "
            "messages."
        ),
        takeaway=(
            "On the graded metric the zero-training floor is at the trained level: both prompted arms' within-family "
            "rank percentile (0.717 open-vocabulary on its scorable subset, 0.732 with the word list on 94% of "
            "records) brackets the trained checkpoint's 0.714. On the cross-family set the prompted arms exceed the "
            "trained checkpoint (0.78–0.84 against 0.69), consistent with the prompted base having no trained-family "
            "restriction to overcome on held-out families."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(EVALS, MODEL_ORDER, TRAINED, TRAINED_LABEL, alt, pl, save_chart):
    _rows = [
        {
            "model": _label,
            "form": "family agreement (binary)",
            "value": _e["generalization"]["within"]["model_vs_teacher_agreement"],
        }
        for _label, _e in EVALS.items()
    ] + [
        {
            "model": _label,
            "form": "emotion-vector cosine (graded)",
            "value": _e["distance_generalization"]["within"]["model_vs_teacher_cosine_mean"],
        }
        for _label, _e in EVALS.items()
    ]
    if TRAINED is not None:
        _rows += [
            {"model": TRAINED_LABEL, "form": "family agreement (binary)", "value": TRAINED["within_model_vs_teacher"]},
            {
                "model": TRAINED_LABEL,
                "form": "emotion-vector cosine (graded)",
                "value": TRAINED["dist_within_model_vs_teacher_cosine"],
            },
        ]
    _long = pl.DataFrame(_rows)
    _chart = (
        alt.Chart(_long)
        .mark_bar()
        .encode(
            x=alt.X("model:N", sort=MODEL_ORDER, title=None, axis=alt.Axis(labelAngle=-20, labelLimit=200)),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title="agreement with the probe teacher"),
            color=alt.Color("model:N", sort=MODEL_ORDER, scale=alt.Scale(scheme="tableau10"), legend=None),
            column=alt.Column(
                "form:N",
                sort=["family agreement (binary)", "emotion-vector cosine (graded)"],
                title=None,
            ),
            tooltip=["model", "form", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(width=240, height=240, title="Fidelity to the probe teacher, in both measurement forms")
    )
    save_chart(
        _chart,
        "teacher_fidelity_binary_vs_graded",
        caption=(
            "Agreement between the emitted tag and the probe-derived teacher tag on the 260 held-out within-family "
            "messages, in the two measurement forms of the standard battery: binary family agreement, and the mean "
            "emotion-vector cosine between the tag's first in-taxonomy word and the teacher's first emotion. Graded "
            "values cover scorable records only (95 of 260 for the open-vocabulary arm, 245 of 260 with the word "
            "list, essentially all for the trained checkpoint)."
        ),
        takeaway=(
            "The trained checkpoint leads on both forms of teacher fidelity (53% family agreement, 0.537 cosine "
            "against 16–35% and 0.43–0.47 for the prompted arms) — the clearest quantity supervised training adds "
            "over prompting. The open-vocabulary arm's higher cosine than the list arm's is entirely a composition "
            "effect of which records are scorable: on the 94 within-family records both arms can score, the two "
            "prompts are indistinguishable (0.475 against 0.481, paired difference −0.006), while the 151 records "
            "only the list arm can score average 0.391 and pull its overall mean down. The graded comparison "
            "between prompted arms is therefore not a ranking of the prompts."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(EVALS, alt, pl, save_chart):
    _TOP_N = 15
    _charts = []
    for _label, _e in EVALS.items():
        _vocab = pl.DataFrame(_e["emitted_vocabulary"][:_TOP_N]).with_columns(
            pl.col("in_taxonomy")
            .map_elements(lambda t: "in the 171-word taxonomy" if t else "outside the taxonomy", return_dtype=pl.String)
            .alias("membership")
        )
        _charts.append(
            alt.Chart(_vocab)
            .mark_bar()
            .encode(
                y=alt.Y("emotion:N", sort="-x", title=None),
                x=alt.X("count:Q", title="occurrences in emitted tags"),
                color=alt.Color(
                    "membership:N",
                    scale=alt.Scale(scheme="tableau10"),
                    title=None,
                    sort=["in the 171-word taxonomy", "outside the taxonomy"],
                ),
                tooltip=["emotion", "count", "family"],
            )
            .properties(width=220, height=280, title=_label)
        )
    save_chart(
        alt.hconcat(*_charts).properties(
            title="The prompted base model's emotion vocabulary, with and without the word list"
        ),
        "emitted_vocabulary_by_arm",
        caption=(
            "The fifteen most frequent words in the emitted tags across all 387 battery messages, per prompt arm, "
            "colored by taxonomy membership. Under the open-vocabulary prompt 23% of emitted words are among the "
            "171 taxonomy emotions; listing the vocabulary in the prompt raises this to 66%, with one third of "
            "words still drawn from outside the list."
        ),
        takeaway=(
            "The base model's native emotion lexicon is largely disjoint from the taxonomy: its most frequent "
            "open-vocabulary words (concern, curious, urgency, focused, neutral) have no taxonomy counterpart, and "
            "even with the full word list in the prompt it continues to reach for concerned and focused. The list "
            "shifts the distribution onto taxonomy words (calm, amused, hopeful, alert) rather than fully "
            "constraining it."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(Path, json):
    # Both vs-teacher distance forms, re-scored from stored samples for the two prompted
    # arms and the trained reference (rescore_teacher_centroid.py): cos_top1 = model
    # first word vs the teacher's top-mass word (the battery's current form);
    # cos_centroid = the same model word vs the mass-weighted centroid of the teacher's
    # full selected tag.
    CENTROID_SCORES = json.loads(
        (Path(__file__).parents[1] / "data" / "teacher_centroid" / "scores.json").read_text(encoding="utf-8")
    )
    return (CENTROID_SCORES,)


@app.cell
def _(CENTROID_SCORES, MODEL_ORDER, alt, pl, save_chart):
    _FORMS = ["teacher top-mass word (1-vs-1)", "teacher mass-weighted centroid (1-vs-3)"]
    _rows = [
        {"model": _label, "set": _set, "form": _form, "cosine": _mean}
        for _label, _sets in CENTROID_SCORES.items()
        for _set, _records in _sets.items()
        for _form, _key in zip(_FORMS, ("cos_top1", "cos_centroid"))
        if (_scored := [_r[_key] for _r in _records if _r[_key] is not None])
        and (_mean := sum(_scored) / len(_scored)) is not None
    ]
    _chart = (
        alt.Chart(pl.DataFrame(_rows))
        .mark_bar()
        .encode(
            x=alt.X("model:N", sort=MODEL_ORDER, title=None, axis=alt.Axis(labelAngle=-20, labelLimit=200)),
            xOffset=alt.XOffset("form:N", sort=_FORMS),
            y=alt.Y("cosine:Q", scale=alt.Scale(domain=[0, 0.7]), title="mean cosine to the probe teacher's tag"),
            color=alt.Color("form:N", sort=_FORMS, scale=alt.Scale(scheme="tableau10"), title="teacher reference"),
            column=alt.Column("set:N", sort=["within", "cross"], title=None),
            tooltip=["model", "set", "form", alt.Tooltip("cosine:Q", format=".3f")],
        )
        .properties(width=240, height=240, title="Fidelity to the probe teacher: top-mass word vs full weighted tag")
    )
    save_chart(
        _chart,
        "teacher_similarity_top1_vs_centroid",
        caption=(
            "Mean cosine between the model's first in-taxonomy word and the probe teacher's tag, under two "
            "treatments of the teacher's multi-emotion label: the top-mass word alone (the battery's current "
            "1-vs-1 form) and the mass-weighted centroid of the full selected tag (1-vs-3; the two coincide on "
            "single-word teacher tags, 4–6% of records). Scorable records only."
        ),
        takeaway=(
            "Scoring against the full weighted teacher tag raises every model's mean similarity by 0.03–0.07 "
            "(within-family: 0.537 to 0.610 trained, 0.472 to 0.524 open vocabulary, 0.426 to 0.485 with the word "
            "list) and changes no comparison: the trained checkpoint's lead over both prompted arms, and the "
            "ordering of every model pair, is identical under the two forms in both evaluation sets. For aggregate "
            "battery conclusions the choice between 1-vs-1 and 1-vs-3 is immaterial."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(CENTROID_SCORES, MODEL_ORDER, alt, pl, save_chart):
    _long = pl.DataFrame(
        [
            {"model": _label, "set": _set, "cos_top1": _r["cos_top1"], "cos_centroid": _r["cos_centroid"]}
            for _label, _sets in CENTROID_SCORES.items()
            for _set, _records in _sets.items()
            for _r in _records
            if _r["cos_top1"] is not None
        ]
    )
    _points = (
        alt.Chart(_long)
        .mark_circle(size=28, opacity=0.45)
        .encode(
            x=alt.X("cos_top1:Q", scale=alt.Scale(domain=[-1, 1]), title="cosine vs teacher top-mass word (1-vs-1)"),
            y=alt.Y("cos_centroid:Q", scale=alt.Scale(domain=[-1, 1]), title="cosine vs teacher centroid (1-vs-3)"),
            color=alt.Color("model:N", sort=MODEL_ORDER, scale=alt.Scale(scheme="tableau10"), title=None),
            tooltip=["model", "set", alt.Tooltip("cos_top1:Q", format=".2f"), alt.Tooltip("cos_centroid:Q", format=".2f")],
        )
    )
    _diagonal = _points.mark_line(color="gray", strokeDash=[4, 4], opacity=0.7).encode(
        x="cos_top1:Q", y="cos_top1:Q", color=alt.value("gray")
    )
    save_chart(
        alt.layer(_diagonal, _points)
        .properties(width=260, height=260)
        .facet(column=alt.Column("set:N", sort=["within", "cross"], title=None))
        .properties(title="Per-record agreement between the two vs-teacher distance forms"),
        "teacher_top1_centroid_divergence",
        caption=(
            "Each point is one scorable held-out record; its cosine against the teacher's top-mass word (x) is "
            "plotted against its cosine against the mass-weighted centroid of the teacher's full tag (y). The "
            "dashed line marks equality; points on it include all single-word teacher tags."
        ),
        takeaway=(
            "The two forms agree at the aggregate level (per-model correlation 0.86–0.90, roughly half the records "
            "on either side of the diagonal) but diverge substantially on individual records — single records move "
            "by up to 0.9 when the teacher's tail words point elsewhere than its top-mass word. The form choice is "
            "therefore immaterial for battery-level numbers but consequential for any per-record use of the "
            "vs-teacher score, such as preference-pair thresholds."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Notes not carried by the exhibits

    On the 50 neutral messages both arms emit mild tags rather than the trained
    convention: the open-vocabulary arm writes words such as *neutral, focused* and
    *curious, analytical*; the list arm centres on *calm* (*calm, focused, satisfied*).
    The exact neutral anchor *calm, attentive* never appears — under the list it is
    unreachable by construction, since *attentive* is not one of the 171 words.

    Format compliance is high but arm-dependent: the open-vocabulary arm loses 12% of
    within-family replies to a placeholder misreading (feeling words substituted into
    the tag itself, concentrated on heavily sad messages), while the list arm loses 16%
    of cross-family replies to a stray second tag line on amusement messages. The first
    pilot variant, which described the format as `<emotion>...</emotion>` without
    naming the literal opening and closing text, failed entirely (0% compliance) —
    prompt wording, not willingness, was the initial barrier.

    Full tables and run configurations: `description.md` sections 4–5; per-run
    artifacts in `data/runs/<run>/eval_samples.json` and `eval.json`.
    """)
    return


if __name__ == "__main__":
    app.run()
