import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import difflib
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
    from name_that_feeling.evals import tag_eval
    from name_that_feeling.generation import sft
    from name_that_feeling.generation.split import clarity

    alt.data_transformers.disable_max_rows()

    CHECKPOINT_ORDER = ["with_neutral", "no_neutral"]
    CHECKPOINT_COLORS = ["#4c78a8", "#f58518"]
    return (
        CHECKPOINT_COLORS,
        CHECKPOINT_ORDER,
        Path,
        alt,
        clarity,
        difflib,
        json,
        load_clusters,
        mo,
        pl,
        sft,
        slugify,
        tag_eval,
    )


@app.cell
def _(Path, json, load_clusters):
    HERE = Path(__file__).parents[1]

    TRAIN_TAGS = [
        json.loads(x)
        for x in (HERE / "data" / "sft" / "train_tags.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    SAMPLES = json.loads((HERE / "data" / "runs" / "train_samples.json").read_text(encoding="utf-8"))
    EVAL = json.loads((HERE / "data" / "runs" / "eval.json").read_text(encoding="utf-8"))
    RECORDS = [
        json.loads(x)
        for x in (HERE / "data" / "completions" / "unconditioned.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    BY_ID = {r["id"]: r for r in RECORDS}
    CLUSTERS = load_clusters(HERE.parent / "01-emotion-vectors" / "clusters.json")
    return BY_ID, CLUSTERS, EVAL, RECORDS, SAMPLES, TRAIN_TAGS


@app.cell
def _(SAMPLES, TRAIN_TAGS, mo):
    mo.md(f"""
    # Does the trained model reproduce its own training labels?

    The pilot trained on {len(TRAIN_TAGS)} emotion messages, each labeled with a probe-grounded
    `<emotion>` tag. This notebook feeds those **same training messages** back to the trained
    checkpoints ({", ".join(f"`{k}`" for k in SAMPLES)}, greedy sampling via
    `sample_train_replies.py`) and measures the overlap between the tag the model **emits** and
    the tag it was **trained on**.

    Why it matters: on held-out messages the section-7 eval showed the model agreeing with its
    probe teacher only weakly (the teacher itself is weak). On *training* messages the model has
    seen the exact answer — so recovery here is the ceiling of what the tag channel retained, and
    the gap between train recovery and held-out agreement is the memorization-vs-generalization
    read on the channel.
    """)
    return


@app.cell
def _(
    BY_ID,
    CLUSTERS,
    RECORDS,
    SAMPLES,
    TRAIN_TAGS,
    clarity,
    difflib,
    pl,
    sft,
    slugify,
    tag_eval,
):
    # One row per (checkpoint, train message): trained-on tag vs emitted tag, plus covariates.
    _stats = sft.per_emotion_stats(RECORDS)  # full-dataset z-scoring, same as the tag pipeline
    _emo2fam = tag_eval.family_lookup(CLUSTERS)
    _trained_of = {t["id"]: [slugify(e) for e, _ in t["emotions"]] for t in TRAIN_TAGS}

    def _row(label: str, s: dict) -> dict:
        rec = BY_ID[s["id"]]
        trained = _trained_of[s["id"]]
        parsed = tag_eval.parse_reply(s["reply"])
        emitted = [slugify(e) for e in parsed["emotions"]]
        inter = set(trained) & set(emitted)
        completion = rec.get("completion") or ""
        return {
            "checkpoint": label,
            "id": s["id"],
            "cluster": rec["scenario"]["cluster"],
            "trained_tag": ", ".join(trained),
            "emitted_tag": ", ".join(emitted) if emitted else "(no tag)",
            "compliant": parsed["compliant"],
            "exact_match": emitted == trained,
            "top1_match": bool(emitted) and emitted[0] == trained[0],
            "any_overlap": bool(inter),
            "jaccard": len(inter) / len(set(trained) | set(emitted)) if emitted else 0.0,
            "family_match": tag_eval.top_family(emitted, _emo2fam) == tag_eval.top_family(trained, _emo2fam),
            "n_trained": len(trained),
            "n_emitted": len(emitted),
            "clarity": clarity(rec["probe"]["projections"], CLUSTERS, _stats),
            "reply_similarity": difflib.SequenceMatcher(
                None, parsed["visible"][:400], completion[:400]
            ).ratio(),
        }

    ROWS = pl.DataFrame([_row(label, s) for label, sample_list in SAMPLES.items() for s in sample_list])
    return (ROWS,)


@app.cell
def _(mo):
    mo.md("""
    ## 1 · How much of the trained tag comes back?

    Overlap at four strictnesses — exact tag (same emotions, same order), top-1 emotion, any
    shared emotion, and family of the leading emotion. `jaccard` is the mean set-overlap of the
    tag's emotion lists.
    """)
    return


@app.cell
def _(ROWS, pl):
    RECOVERY = (
        ROWS.group_by("checkpoint")
        .agg(
            pl.len().alias("n"),
            pl.col("compliant").mean().round(3).alias("format_compliant"),
            pl.col("exact_match").mean().round(3).alias("exact_tag"),
            pl.col("top1_match").mean().round(3).alias("top1_emotion"),
            pl.col("any_overlap").mean().round(3).alias("any_overlap"),
            pl.col("jaccard").mean().round(3).alias("jaccard"),
            pl.col("family_match").mean().round(3).alias("top1_family"),
        )
        .sort("checkpoint", descending=True)
    )
    RECOVERY
    return (RECOVERY,)


@app.cell
def _(CHECKPOINT_COLORS, CHECKPOINT_ORDER, RECOVERY, alt):
    _long = RECOVERY.drop("n").unpivot(index="checkpoint", variable_name="metric", value_name="rate")
    _metric_order = ["format_compliant", "exact_tag", "top1_emotion", "any_overlap", "jaccard", "top1_family"]
    alt.Chart(_long).mark_bar().encode(
        x=alt.X("checkpoint:N", sort=CHECKPOINT_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("rate:Q", scale=alt.Scale(domain=[0, 1]), title=None),
        color=alt.Color(
            "checkpoint:N",
            sort=CHECKPOINT_ORDER,
            scale=alt.Scale(domain=CHECKPOINT_ORDER, range=CHECKPOINT_COLORS),
            legend=None,
        ),
        column=alt.Column("metric:N", sort=_metric_order, title=None),
        tooltip=["checkpoint", "metric", alt.Tooltip("rate:Q", format=".3f")],
    ).properties(width=90, height=200, title="Label recovery on the 576 training messages")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2 · Memorization vs generalization

    The same model-vs-teacher read (family of the leading tag emotion) on three message sets:
    the **train** messages (the model saw these exact labels), the **held-out-emotion** messages
    and the **held-out-family** messages (section-7 eval; the model never saw them, the teacher
    tag is what the probe *would* have said). Train ≫ held-out means the model memorized specific
    message→tag pairs; train ≈ held-out means the tag channel runs on a general mapping and the
    training labels weren't stored as such.
    """)
    return


@app.cell
def _(CHECKPOINT_COLORS, CHECKPOINT_ORDER, EVAL, ROWS, alt, pl):
    _rows = [
        {
            "checkpoint": _label,
            "message set": "train (seen labels)",
            "agreement": ROWS.filter(pl.col("checkpoint") == _label)["family_match"].mean(),
        }
        for _label in ROWS["checkpoint"].unique()
    ] + [
        {
            "checkpoint": _label,
            "message set": f"{_set} (held out)",
            "agreement": EVAL["generalization"][_set][_label]["model_vs_teacher_agreement"],
        }
        for _set in ("within", "cross")
        for _label in EVAL["generalization"][_set]
    ]
    MEMO = pl.DataFrame(_rows)
    _set_order = ["train (seen labels)", "within (held out)", "cross (held out)"]
    alt.Chart(MEMO).mark_bar().encode(
        x=alt.X("message set:N", sort=_set_order, title=None, axis=alt.Axis(labelAngle=-15, labelLimit=180)),
        y=alt.Y("agreement:Q", scale=alt.Scale(domain=[0, 1]), title="model vs trained/teacher tag (family)"),
        color=alt.Color(
            "checkpoint:N",
            sort=CHECKPOINT_ORDER,
            scale=alt.Scale(domain=CHECKPOINT_ORDER, range=CHECKPOINT_COLORS),
            title=None,
        ),
        xOffset=alt.XOffset("checkpoint:N", sort=CHECKPOINT_ORDER),
        tooltip=["checkpoint", "message set", alt.Tooltip("agreement:Q", format=".3f")],
    ).properties(width=340, height=240, title="Tag agreement: seen labels vs held-out messages")
    return (MEMO,)


@app.cell
def _(MEMO, mo, pl):
    _train = {r["checkpoint"]: r["agreement"] for r in MEMO.filter(pl.col("message set") == "train (seen labels)").to_dicts()}
    _within = {r["checkpoint"]: r["agreement"] for r in MEMO.filter(pl.col("message set") == "within (held out)").to_dicts()}
    _wn_gap = _train.get("with_neutral", 0) - _within.get("with_neutral", 0)
    mo.md(
        f"""
    For the canonical `with_neutral` checkpoint the gap between seen-label recovery
    (**{_train.get("with_neutral", 0):.0%}**) and held-out-emotion agreement
    (**{_within.get("with_neutral", 0):.0%}**) is **{_wn_gap:+.0%}**. A large positive gap =
    the exact training labels are retrievable (memorized); a small one = the model responds to
    the message with its own learned mapping and the specific trained tag left little trace
    beyond that mapping.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3 · Where recovery happens — by family, and by probe clarity

    Left: top-1-emotion recovery by elicited family. Right: was a message's trained label easier
    to recover when its probe read was *clear* (high top-1 − top-2 family margin — the selection
    metric the pilot picked its training messages by)?
    """)
    return


@app.cell
def _(CHECKPOINT_COLORS, CHECKPOINT_ORDER, ROWS, alt, mo, pl):
    _by_fam = ROWS.group_by("checkpoint", "cluster").agg(
        pl.col("top1_match").mean().alias("top1_recovery"), pl.len().alias("n")
    )
    _fam = (
        alt.Chart(_by_fam)
        .mark_bar()
        .encode(
            x=alt.X("top1_recovery:Q", scale=alt.Scale(domain=[0, 1]), title="top-1 emotion recovery"),
            y=alt.Y("cluster:N", sort="-x", title=None),
            color=alt.Color(
                "checkpoint:N",
                sort=CHECKPOINT_ORDER,
                scale=alt.Scale(domain=CHECKPOINT_ORDER, range=CHECKPOINT_COLORS),
                title=None,
            ),
            yOffset=alt.YOffset("checkpoint:N", sort=CHECKPOINT_ORDER),
            tooltip=["checkpoint", "cluster", alt.Tooltip("top1_recovery:Q", format=".2f"), "n"],
        )
        .properties(width=300, height=260, title="Recovery by family")
    )
    _clar = (
        alt.Chart(ROWS.with_columns(pl.col("top1_match").alias("recovered")))
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("recovered:N", title="top-1 emotion recovered"),
            y=alt.Y("clarity:Q", title="probe clarity of the message"),
            color=alt.Color(
                "checkpoint:N",
                sort=CHECKPOINT_ORDER,
                scale=alt.Scale(domain=CHECKPOINT_ORDER, range=CHECKPOINT_COLORS),
                title=None,
            ),
            xOffset=alt.XOffset("checkpoint:N", sort=CHECKPOINT_ORDER),
        )
        .properties(width=220, height=260, title="Clarity vs recovery")
    )
    mo.hstack([_fam, _clar])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4 · What gets emitted instead — family confusion

    Family of the trained tag's leading emotion (rows) vs family of the emitted one (columns),
    for the `with_neutral` checkpoint. The diagonal is recovery; strong off-diagonal columns show
    the families the model defaults to when it doesn't reproduce the label.
    """)
    return


@app.cell
def _(CLUSTERS, ROWS, TRAIN_TAGS, alt, pl, slugify, tag_eval):
    _emo2fam = tag_eval.family_lookup(CLUSTERS)
    _trained_fam = {
        t["id"]: tag_eval.top_family([slugify(e) for e, _ in t["emotions"]], _emo2fam) for t in TRAIN_TAGS
    }
    _conf = (
        ROWS.filter(pl.col("checkpoint") == "with_neutral")
        .with_columns(
            pl.col("id").replace_strict(_trained_fam, default=None).alias("trained_family"),
            pl.col("emitted_tag").str.split(", ").list.first().replace_strict(
                {slugify(e): c for c, es in CLUSTERS.items() for e in es}, default="<off-taxonomy>"
            ).alias("emitted_family"),
        )
        .group_by("trained_family", "emitted_family")
        .agg(pl.len().alias("n"))
    )
    alt.Chart(_conf).mark_rect().encode(
        x=alt.X("emitted_family:N", title="emitted tag family", axis=alt.Axis(labelAngle=-40, labelLimit=160)),
        y=alt.Y("trained_family:N", title="trained tag family"),
        color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), title="messages"),
        tooltip=["trained_family", "emitted_family", "n"],
    ).properties(width=420, height=320, title="Trained vs emitted tag family (with_neutral)")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5 · Did it memorize the replies too?

    The training rows also contained a full assistant reply after the tag. `reply_similarity` is
    the character-level similarity (SequenceMatcher, first 400 chars) between the greedy reply the
    model emits now and the completion it was trained on. Near-1 = it replays the training reply;
    low = only the tag mapping (not the text) was retained.
    """)
    return


@app.cell
def _(CHECKPOINT_COLORS, CHECKPOINT_ORDER, ROWS, alt, mo, pl):
    _sim = ROWS.group_by("checkpoint").agg(
        pl.col("reply_similarity").median().round(3).alias("median"),
        (pl.col("reply_similarity") >= 0.8).mean().round(3).alias("share >= 0.8"),
        (pl.col("reply_similarity") >= 0.95).mean().round(3).alias("share >= 0.95"),
    )
    _hist = (
        alt.Chart(ROWS)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("reply_similarity:Q", bin=alt.Bin(maxbins=40), title="similarity(emitted reply, trained reply)"),
            y=alt.Y("count()", title="messages"),
            color=alt.Color(
                "checkpoint:N",
                sort=CHECKPOINT_ORDER,
                scale=alt.Scale(domain=CHECKPOINT_ORDER, range=CHECKPOINT_COLORS),
                title=None,
            ),
            tooltip=["checkpoint", alt.Tooltip("count()")],
        )
        .properties(width=440, height=220, title="Reply memorization")
    )
    mo.vstack([mo.ui.table(_sim, selection=None), _hist])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6 · Mismatch examples

    Training messages whose emitted tag shares nothing with the trained one (the model's leading
    family differs), highest-clarity first — the cases where a clearly-labeled message still comes
    back different.
    """)
    return


@app.cell
def _(ROWS, mo, pl):
    _miss = (
        ROWS.filter((pl.col("checkpoint") == "with_neutral") & ~pl.col("any_overlap"))
        .sort("clarity", descending=True)
        .select("id", "cluster", "clarity", "trained_tag", "emitted_tag")
        .head(25)
    )
    mo.ui.table(_miss, selection=None)
    return


if __name__ == "__main__":
    app.run()
