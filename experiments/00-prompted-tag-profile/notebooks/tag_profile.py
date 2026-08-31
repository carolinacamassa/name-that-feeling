import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from collections import Counter
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
    from name_that_feeling.evals.affect_norms import load_norms, score_words
    from name_that_feeling.evals.similarity import EmotionSimilarity
    from name_that_feeling.evals.tag_eval import family_lookup, parse_reply, top_family
    from name_that_feeling.evals.uncertainty import mean_and_ci
    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return (
        Counter,
        EmotionSimilarity,
        Path,
        alt,
        family_lookup,
        json,
        load_clusters,
        load_norms,
        mean_and_ci,
        mo,
        parse_reply,
        pl,
        save_chart,
        score_words,
        slugify,
        top_family,
    )


@app.cell
def _(
    EmotionSimilarity,
    Path,
    family_lookup,
    json,
    load_clusters,
    load_norms,
    mo,
    slugify,
):
    HERE = Path(__file__).parents[1]  # the 00-prompted-tag-profile experiment dir

    ARMS = {
        "with taxonomy": "with-taxonomy",
        "open vocabulary": "open-vocabulary",
    }
    SAMPLES = {
        label: json.loads((HERE / "data" / "runs" / run / "samples.json").read_text(encoding="utf-8"))
        for label, run in ARMS.items()
    }
    MESSAGES = json.loads((HERE / "data" / "messages.json").read_text(encoding="utf-8"))
    MSG_BY_ID = {r["id"]: r for r in MESSAGES}

    CLUSTERS = load_clusters(HERE.parent / "01-emotion-vectors" / "clusters.json")
    SIM = EmotionSimilarity.load(
        HERE.parent / "01-emotion-vectors" / "data" / "similarity" / "layer_21.json"
    )
    EMO2FAM = family_lookup(CLUSTERS)
    TAX_WORDS = {slugify(w) for ws in CLUSTERS.values() for w in ws}
    SLUG2WORD = {slugify(w): w for ws in CLUSTERS.values() for w in ws}
    FAMILIES = sorted(CLUSTERS)

    # Probe-teacher tags for the 576 emotion training rows (the labels 03 trained on),
    # as (word, weight) lists for the 1-vs-3 centroid metric.
    TEACHER = {
        r["id"]: [(w, wt) for w, wt in r["emotions"]]
        for r in (
            json.loads(x)
            for x in (HERE.parent / "03-training-pilot" / "data" / "sft" / "train_tags.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if x.strip()
        )
    }

    # Valence/arousal word norms: Warriner et al. 2013 (1-9 scale), NRC-VAD filling
    # gaps after linear calibration onto the same scale (see evals/affect_norms.py).
    NORMS = load_norms(
        HERE / "data" / "affect_norms" / "warriner_2013.csv",
        HERE / "data" / "affect_norms" / "nrc_vad.txt",
    )

    _k = SAMPLES["with taxonomy"]["meta"]["num_samples"]
    _n = {label: s["meta"]["n_messages"] for label, s in SAMPLES.items()}
    _tax_cov = sum(SLUG2WORD[s] in NORMS for s in TAX_WORDS)
    mo.md(
        f"""## The pre-training tag profile

    K = {_k} temperature-1 draws per message from the prompted, untouched base model
    ({SAMPLES["with taxonomy"]["meta"]["base_model"]}), over the {len(MESSAGES)} SFT training
    messages ({sum(r["set"] == "emotion" for r in MESSAGES)} emotion-elicited +
    {sum(r["set"] == "neutral" for r in MESSAGES)} neutral), under two system-prompt arms:
    **with taxonomy** (the 171-word list plus word-form rules) and **open vocabulary**
    (form rules only). Messages sampled per arm: {_n}.
    Affect norms cover {_tax_cov}/171 taxonomy words."""
    )
    return (
        ARMS,
        EMO2FAM,
        FAMILIES,
        MESSAGES,
        MSG_BY_ID,
        NORMS,
        SAMPLES,
        SIM,
        SLUG2WORD,
        TAX_WORDS,
        TEACHER,
    )


@app.cell
def _(
    Counter,
    EMO2FAM,
    MSG_BY_ID,
    NORMS,
    SAMPLES,
    SIM,
    TAX_WORDS,
    TEACHER,
    parse_reply,
    pl,
    score_words,
    slugify,
    top_family,
):
    def _first_in_tax(words):
        for _w in words:
            if slugify(_w) in TAX_WORDS:
                return _w
        return None

    _draw_rows, _word_rows = [], []
    for _arm, _payload in SAMPLES.items():
        for _s in _payload["samples"]:
            _meta = MSG_BY_ID[_s["id"]]
            _elicited = _meta.get("elicited")
            _elicited_fam = EMO2FAM.get(slugify(_elicited)) if _elicited else None
            _teacher = TEACHER.get(_s["id"])
            _teacher_top = _teacher[0][0] if _teacher else None
            _teacher_fam = EMO2FAM.get(slugify(_teacher_top)) if _teacher_top else None
            for _i, _reply in enumerate(_s["replies"]):
                _p = parse_reply(_reply)
                _words = [_w.lower() for _w in _p["emotions"]]
                _first = _first_in_tax(_words)
                _fam = top_family(_words, EMO2FAM)
                _va = score_words(_words, NORMS)
                _draw_rows.append(
                    {
                        "arm": _arm,
                        "id": _s["id"],
                        "set": _meta["set"],
                        "elicited": _elicited,
                        "draw": _i,
                        "compliant": _p["compliant"],
                        "n_words": len(_words),
                        "words": ", ".join(_words),
                        "first_in_tax": _first,
                        "family": _fam,
                        "fam_vs_teacher": (_fam == _teacher_fam) if _teacher_fam else None,
                        "fam_vs_elicited": (_fam == _elicited_fam) if _elicited_fam else None,
                        "cos_1v1": SIM.sim(_first, _teacher_top) if _teacher_top else None,
                        "cos_1v3": SIM.centroid_sim(_first, _teacher) if _teacher else None,
                        "valence": _va["valence"] if _va else None,
                        "arousal": _va["arousal"] if _va else None,
                        "va_covered": _va["covered"] if _va else 0,
                    }
                )
                for _w in _words:
                    _word_rows.append(
                        {
                            "arm": _arm,
                            "id": _s["id"],
                            "set": _meta["set"],
                            "draw": _i,
                            "word": _w,
                            "in_taxonomy": slugify(_w) in TAX_WORDS,
                            "in_norms": _w in NORMS,
                        }
                    )

    DRAWS = pl.DataFrame(_draw_rows)
    WORDS = pl.DataFrame(_word_rows)

    # Per-prompt aggregates: draw noise lives inside each prompt's value; CIs resample prompts.
    _pp = []
    for (_arm, _id), _g in DRAWS.group_by(["arm", "id"], maintain_order=True):
        _fams = _g["family"].to_list()
        _lists = _g["words"].to_list()
        _pp.append(
            {
                "arm": _arm,
                "id": _id,
                "set": _g["set"][0],
                "compliance": _g["compliant"].mean(),
                "fam_vs_teacher": _g["fam_vs_teacher"].drop_nulls().mean(),
                "fam_vs_elicited": _g["fam_vs_elicited"].drop_nulls().mean(),
                "cos_1v1": _g["cos_1v1"].drop_nulls().mean(),
                "cos_1v3": _g["cos_1v3"].drop_nulls().mean(),
                "modal_family_share": Counter(_f or "<off>" for _f in _fams).most_common(1)[0][1]
                / len(_fams),
                "exact_list_repeat": Counter(_lists).most_common(1)[0][1] / len(_lists),
                "n_families": len({_f for _f in _fams if _f}),
                "valence": _g["valence"].drop_nulls().mean(),
                "arousal": _g["arousal"].drop_nulls().mean(),
            }
        )
    PER_PROMPT = pl.DataFrame(_pp)
    return DRAWS, PER_PROMPT, WORDS


@app.cell
def _(DRAWS, WORDS, mo, pl):
    _t = (
        DRAWS.group_by("arm")
        .agg(
            pl.len().alias("draws"),
            pl.col("compliant").mean().round(3).alias("tag compliance"),
            pl.col("n_words").mean().round(2).alias("words/tag"),
            pl.col("valence").is_not_null().mean().round(3).alias("V/A scorable"),
        )
        .join(
            WORDS.group_by("arm").agg(
                pl.col("in_taxonomy").mean().round(3).alias("words in taxonomy"),
                pl.col("in_norms").mean().round(3).alias("words with norms"),
                pl.col("word").n_unique().alias("distinct words"),
            ),
            on="arm",
        )
    )
    mo.vstack([mo.md("**Format and vocabulary, per arm** (all draws)"), _t])
    return


@app.cell
def _(WORDS, alt, pl, save_chart):
    _freq = (
        WORDS.unique(subset=["arm", "id", "draw", "word"])
        .group_by("arm", "word", "in_taxonomy")
        .agg(pl.len().alias("draws_with_word"))
    )
    _per_arm_draws = (
        WORDS.unique(subset=["arm", "id", "draw"]).group_by("arm").agg(pl.len().alias("total_draws"))
    )
    _freq = _freq.join(_per_arm_draws, on="arm").with_columns(
        (pl.col("draws_with_word") / pl.col("total_draws")).alias("share")
    )
    _charts = []
    for _arm in ("with taxonomy", "open vocabulary"):
        _top = _freq.filter(pl.col("arm") == _arm).sort("share", descending=True).head(25)
        _charts.append(
            alt.Chart(_top)
            .mark_bar()
            .encode(
                y=alt.Y("word:N", sort="-x", title=None),
                x=alt.X(
                    "share:Q", axis=alt.Axis(format="%"), title="share of draws containing the word"
                ),
                color=alt.Color(
                    "in_taxonomy:N",
                    scale=alt.Scale(domain=[True, False], range=["#4c78a8", "#f58518"]),
                    title="in the 171-word taxonomy",
                ),
                tooltip=[
                    "word",
                    alt.Tooltip("share:Q", format=".1%"),
                    "in_taxonomy",
                    "draws_with_word",
                ],
            )
            .properties(width=250, height=430, title=_arm)
        )
    _tax_top = _freq.filter(pl.col("arm") == "with taxonomy").sort("share", descending=True).head(25)
    _open_top = (
        _freq.filter(pl.col("arm") == "open vocabulary").sort("share", descending=True).head(25)
    )
    save_chart(
        alt.hconcat(*_charts).resolve_scale(color="shared"),
        "top_emitted_words",
        caption=(
            "The 25 most frequent feeling words in the untrained base model's emotion tags, per prompt arm, as "
            "the share of temperature-1 draws whose tag contains the word (12 draws per message, 1,076 SFT "
            "training messages). Left: the system prompt includes the full 171-word taxonomy list plus word-form "
            "rules; right: no word list, form rules only. Color marks whether the word belongs to the 171-word "
            "taxonomy the probe vectors are built on."
        ),
        takeaway=(
            f"Before any training the prompted base concentrates its tag vocabulary on a small head: the top "
            f"word appears in {_tax_top['share'][0]:.0%} of with-taxonomy draws ('{_tax_top['word'][0]}') and "
            f"{_open_top['share'][0]:.0%} of open-vocabulary draws ('{_open_top['word'][0]}'). The word list "
            f"reshapes the vocabulary rather than merely filtering it: the two arms share "
            f"{len(set(_tax_top['word']) & set(_open_top['word']))} of their top-25 words."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(
    Counter,
    DRAWS,
    EMO2FAM,
    FAMILIES,
    MESSAGES,
    TEACHER,
    alt,
    pl,
    save_chart,
    slugify,
):
    _rows = []
    _emo_msgs = [_m for _m in MESSAGES if _m["set"] == "emotion"]
    for _m in _emo_msgs:
        _rows.append(
            {"column": "elicited target", "family": EMO2FAM[slugify(_m["elicited"])], "weight": 1.0}
        )
        _rows.append(
            {
                "column": "probe teacher",
                "family": EMO2FAM[slugify(TEACHER[_m["id"]][0][0])],
                "weight": 1.0,
            }
        )
    for _arm in ("with taxonomy", "open vocabulary"):
        for _set, _label in (("emotion", "emotion msgs"), ("neutral", "neutral msgs")):
            _fams = DRAWS.filter((pl.col("arm") == _arm) & (pl.col("set") == _set))[
                "family"
            ].to_list()
            for _f, _c in Counter(_f or "off-taxonomy" for _f in _fams).items():
                _rows.append({"column": f"{_arm}, {_label}", "family": _f, "weight": float(_c)})
    _df = pl.DataFrame(_rows).group_by("column", "family").agg(pl.col("weight").sum())
    _order = [
        "elicited target",
        "probe teacher",
        "with taxonomy, emotion msgs",
        "open vocabulary, emotion msgs",
        "with taxonomy, neutral msgs",
        "open vocabulary, neutral msgs",
    ]
    _chart = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            y=alt.Y("column:N", sort=_order, title=None, axis=alt.Axis(labelLimit=250)),
            x=alt.X(
                "weight:Q",
                stack="normalize",
                axis=alt.Axis(format="%"),
                title="share of draws (of messages, for the reference rows)",
            ),
            color=alt.Color(
                "family:N",
                sort=FAMILIES + ["off-taxonomy"],
                scale=alt.Scale(scheme="tableau20"),
                title="emotion family",
            ),
            tooltip=["column", "family", alt.Tooltip("weight:Q", format=".0f")],
        )
        .properties(
            width=430,
            height=210,
            title="Family composition: what the untrained model emits vs the labels",
        )
    )
    _emitted = Counter(
        _f or "off-taxonomy"
        for _f in DRAWS.filter((pl.col("arm") == "with taxonomy") & (pl.col("set") == "emotion"))[
            "family"
        ].to_list()
    )
    _target = Counter(EMO2FAM[slugify(_m["elicited"])] for _m in _emo_msgs)
    _top_fam, _top_n = _emitted.most_common(1)[0]
    save_chart(
        _chart,
        "family_composition_vs_references",
        caption=(
            "Family-level composition of the tags the untrained base model emits, next to the two label "
            "references. Top rows: the family of the emotion each of the 576 training messages was written to "
            "elicit, and the family of the probe teacher's top word on the same message (the label SFT trained "
            "on). Lower rows: the family of the first in-taxonomy word of each emitted tag, per prompt arm, "
            "split by emotion-elicited vs neutral messages; 'off-taxonomy' marks draws whose tag contains no "
            "taxonomy word at all. Families are the 10 k-means clusters of the 171 emotion vectors."
        ),
        takeaway=(
            f"The untrained profile is skewed relative to both references: the most emitted family on emotion "
            f"messages ({_top_fam}, {_top_n / sum(_emitted.values()):.0%} of with-taxonomy draws) holds "
            f"{_target.get(_top_fam, 0) / sum(_target.values()):.0%} of elicited targets, and "
            f"{_emitted['off-taxonomy'] / sum(_emitted.values()):.0%} of with-taxonomy draws contain no "
            f"taxonomy word despite the list being in the prompt."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(PER_PROMPT, alt, mean_and_ci, mo, pl, save_chart):
    _metrics = (
        ("family agreement vs teacher", "fam_vs_teacher"),
        ("cosine vs teacher top word (1v1)", "cos_1v1"),
        ("cosine vs teacher centroid (1v3)", "cos_1v3"),
        ("family agreement vs elicited", "fam_vs_elicited"),
    )
    _rows = []
    for _arm in ("with taxonomy", "open vocabulary"):
        _sub = PER_PROMPT.filter((pl.col("arm") == _arm) & (pl.col("set") == "emotion"))
        for _label, _key in _metrics:
            _vals = _sub[_key].drop_nulls().to_list()
            _rows.append({"arm": _arm, "metric": _label, **mean_and_ci(_vals)})
    _df = pl.DataFrame(_rows)
    _base = alt.Chart(_df).encode(
        y=alt.Y(
            "metric:N", sort=[_l for _l, _ in _metrics], title=None, axis=alt.Axis(labelLimit=250)
        ),
        yOffset=alt.YOffset("arm:N", sort=["with taxonomy", "open vocabulary"]),
    )
    _bars = _base.mark_bar().encode(
        x=alt.X(
            "mean:Q",
            scale=alt.Scale(domain=[-0.1, 1]),
            title="mean over the 576 emotion training messages",
        ),
        color=alt.Color("arm:N", scale=alt.Scale(scheme="tableau10"), title="prompt arm"),
        tooltip=[
            "arm",
            "metric",
            alt.Tooltip("mean:Q", format=".3f"),
            alt.Tooltip("lo:Q", format=".3f"),
            alt.Tooltip("hi:Q", format=".3f"),
            "n",
        ],
    )
    _err = _base.mark_rule(strokeWidth=1.5).encode(x="lo:Q", x2="hi:Q")
    _stat = {(_r["arm"], _r["metric"]): _r for _r in _rows}
    _wt = _stat[("with taxonomy", "family agreement vs teacher")]
    _ov = _stat[("open vocabulary", "family agreement vs teacher")]
    save_chart(
        alt.layer(_bars, _err).properties(
            width=420, height=240, title="Untrained agreement with the probe teacher, at temperature 1"
        ),
        "agreement_with_probe_teacher",
        caption=(
            "How well the untrained base model's sampled tags agree with the probe-derived teacher labels (and, "
            "last row, with the family of the emotion each message was written to elicit), on the 576 emotion "
            "training messages. Per message the 12 temperature-1 draws are averaged first; bars are means over "
            "messages with 95% bootstrap intervals resampling messages. The three teacher metrics are the "
            "standing set: binary right-family agreement (a draw whose tag has no taxonomy word counts as a "
            "miss), cosine similarity between the first in-taxonomy emitted word and the teacher's top word, and "
            "cosine against the mass-weighted centroid of the full teacher tag (the graded metrics score only "
            "draws with an in-taxonomy word)."
        ),
        takeaway=(
            f"At temperature 1 the prompted base agrees with the probe teacher's family on "
            f"{_wt['mean']:.0%} [{_wt['lo']:.0%}, {_wt['hi']:.0%}] of draws with the word list and "
            f"{_ov['mean']:.0%} [{_ov['lo']:.0%}, {_ov['hi']:.0%}] open-vocabulary — the pre-training agreement "
            f"floor a self-labeled training set would start from. The 1v3 centroid cosine reaches "
            f"{_stat[('with taxonomy', 'cosine vs teacher centroid (1v3)')]['mean']:.2f} (with taxonomy), so "
            f"near-misses sit substantially closer than the binary read suggests."
        ),
        notebook=__file__,
    )
    mo.vstack(
        [
            mo.md(
                "**Agreement with the probe teacher and the elicited family** "
                "(emotion messages, per-prompt CIs)"
            ),
            _df,
        ]
    )
    return


@app.cell
def _(PER_PROMPT, alt, pl, save_chart):
    _long = PER_PROMPT.unpivot(
        on=["modal_family_share", "exact_list_repeat"],
        index=["arm", "id", "set"],
        variable_name="statistic",
        value_name="value",
    ).with_columns(
        pl.col("statistic").replace(
            {
                "modal_family_share": "modal family share (of 12 draws)",
                "exact_list_repeat": "exact tag-list repeat share",
            }
        )
    )
    _chart = (
        alt.Chart(_long)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X(
                "value:Q",
                bin=alt.Bin(step=1 / 12),
                axis=alt.Axis(format="%"),
                title="per-message share across the 12 draws",
            ),
            y=alt.Y("count():Q", title="messages"),
            color=alt.Color("arm:N", scale=alt.Scale(scheme="tableau10"), title="prompt arm"),
            column=alt.Column("statistic:N", title=None),
            row=alt.Row("arm:N", title=None),
        )
        .properties(width=240, height=110)
    )
    _wt = PER_PROMPT.filter(pl.col("arm") == "with taxonomy")
    _ov = PER_PROMPT.filter(pl.col("arm") == "open vocabulary")
    save_chart(
        _chart,
        "sampling_variability",
        caption=(
            "How stable the untrained tag channel is for a fixed message across the 12 temperature-1 draws, per "
            "prompt arm, over all 1,076 training messages. Left: the share of a message's draws landing in its "
            "modal (most frequent) emotion family, counting off-taxonomy as its own bucket — 1.0 means every "
            "draw agrees at family level. Right: the share of draws emitting the modal exact word list — 1.0 "
            "means the identical tag every time."
        ),
        takeaway=(
            f"The untrained channel is a broad distribution, as the trained channel is: the exact tag list "
            f"repeats on only {_wt['exact_list_repeat'].mean():.0%} of draws with the taxonomy list "
            f"({_ov['exact_list_repeat'].mean():.0%} open-vocabulary), while family-level stability is higher "
            f"(modal family share {_wt['modal_family_share'].mean():.0%} / "
            f"{_ov['modal_family_share'].mean():.0%}), and {(_wt['modal_family_share'] == 1.0).mean():.0%} / "
            f"{(_ov['modal_family_share'] == 1.0).mean():.0%} of messages are fully family-stable."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(NORMS, SLUG2WORD, TAX_WORDS, WORDS, alt, pl, save_chart):
    _per_arm_draws = (
        WORDS.unique(subset=["arm", "id", "draw"]).group_by("arm").agg(pl.len().alias("total_draws"))
    )
    _freq = (
        WORDS.filter(pl.col("in_norms"))
        .unique(subset=["arm", "id", "draw", "word"])
        .group_by("arm", "word", "in_taxonomy")
        .agg(pl.len().alias("draws_with_word"))
        .join(_per_arm_draws, on="arm")
        .with_columns((pl.col("draws_with_word") / pl.col("total_draws")).alias("share"))
        .filter(pl.col("share") >= 0.002)
        .with_columns(
            pl.col("word")
            .map_elements(lambda w: NORMS[w]["valence"], return_dtype=pl.Float64)
            .alias("valence"),
            pl.col("word")
            .map_elements(lambda w: NORMS[w]["arousal"], return_dtype=pl.Float64)
            .alias("arousal"),
        )
    )
    _tax = pl.DataFrame(
        [
            {
                "word": SLUG2WORD[_s],
                "valence": NORMS[SLUG2WORD[_s]]["valence"],
                "arousal": NORMS[SLUG2WORD[_s]]["arousal"],
            }
            for _s in TAX_WORDS
            if SLUG2WORD[_s] in NORMS
        ]
    )
    _charts = []
    for _arm in ("with taxonomy", "open vocabulary"):
        _pts = _freq.filter(pl.col("arm") == _arm)
        _under = (
            alt.Chart(_tax)
            .mark_point(color="#bbbbbb", size=14, opacity=0.6)
            .encode(x="valence:Q", y="arousal:Q", tooltip=["word"])
        )
        _over = (
            alt.Chart(_pts)
            .mark_circle(opacity=0.75)
            .encode(
                x=alt.X("valence:Q", scale=alt.Scale(domain=[1, 9]), title="valence (Warriner 1-9)"),
                y=alt.Y("arousal:Q", scale=alt.Scale(domain=[1, 9]), title="arousal (1-9)"),
                size=alt.Size("share:Q", scale=alt.Scale(range=[15, 700]), title="share of draws"),
                color=alt.Color(
                    "in_taxonomy:N",
                    scale=alt.Scale(domain=[True, False], range=["#4c78a8", "#f58518"]),
                    title="in taxonomy",
                ),
                tooltip=[
                    "word",
                    alt.Tooltip("share:Q", format=".1%"),
                    alt.Tooltip("valence:Q", format=".1f"),
                    alt.Tooltip("arousal:Q", format=".1f"),
                ],
            )
        )
        _charts.append(alt.layer(_under, _over).properties(width=280, height=280, title=_arm))
    _wtd = _freq.filter(pl.col("arm") == "with taxonomy")
    _wv = (_wtd["valence"] * _wtd["share"]).sum() / _wtd["share"].sum()
    _wa = (_wtd["arousal"] * _wtd["share"]).sum() / _wtd["share"].sum()
    save_chart(
        alt.hconcat(*_charts).resolve_scale(color="shared", size="shared"),
        "affect_space_vocabulary",
        caption=(
            "The untrained model's emitted tag vocabulary placed in affect space: each point is a feeling word, "
            "positioned by its human-rated valence (how pleasant, 1-9) and arousal (how activating, 1-9) from "
            "the Warriner et al. (2013) norms with NRC-VAD filling gaps, sized by the share of draws containing "
            "it, over all 1,076 training messages. Grey underlay: the 156 of 171 taxonomy words the norms cover "
            "— the affect space the probe labels can express. Words below 0.2% of draws or without norms are "
            "omitted."
        ),
        takeaway=(
            f"The pre-training profile occupies a compact affect region: the draw-weighted centroid of the "
            f"with-taxonomy arm sits at valence {_wv:.1f}, arousal {_wa:.1f} on the 1-9 scales, against a "
            f"taxonomy whose words span valence {_tax['valence'].min():.1f}-{_tax['valence'].max():.1f} and "
            f"arousal {_tax['arousal'].min():.1f}-{_tax['arousal'].max():.1f} — the untrained vocabulary uses a "
            f"narrow slice of the affect plane the labels can name."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(NORMS, PER_PROMPT, SLUG2WORD, alt, pl, save_chart, slugify):
    _pp = PER_PROMPT.filter(pl.col("set") == "emotion").with_columns(
        pl.col("id")
        .map_elements(lambda i: slugify(i.rsplit(":", 1)[0]), return_dtype=pl.String)
        .alias("slug")
    )
    _pp = _pp.with_columns(
        pl.col("slug")
        .map_elements(
            lambda s: NORMS.get(SLUG2WORD.get(s, ""), {}).get("valence"), return_dtype=pl.Float64
        )
        .alias("elicited_valence"),
        pl.col("slug")
        .map_elements(
            lambda s: NORMS.get(SLUG2WORD.get(s, ""), {}).get("arousal"), return_dtype=pl.Float64
        )
        .alias("elicited_arousal"),
    )
    _long = pl.concat(
        [
            _pp.select(
                "arm",
                "id",
                pl.lit(_dim).alias("dimension"),
                pl.col(f"elicited_{_dim}").alias("elicited"),
                pl.col(_dim).alias("emitted"),
            )
            for _dim in ("valence", "arousal")
        ]
    ).drop_nulls(["elicited", "emitted"])
    _pts = (
        alt.Chart(_long)
        .mark_circle(size=18, opacity=0.35)
        .encode(
            x=alt.X(
                "elicited:Q",
                scale=alt.Scale(domain=[1, 9]),
                title="norms rating of the elicited emotion word",
            ),
            y=alt.Y(
                "emitted:Q",
                scale=alt.Scale(domain=[1, 9]),
                title="mean rating of the emitted tag words",
            ),
            tooltip=[
                "id",
                "arm",
                "dimension",
                alt.Tooltip("elicited:Q", format=".1f"),
                alt.Tooltip("emitted:Q", format=".1f"),
            ],
        )
    )
    _fit = _pts.transform_regression("elicited", "emitted", groupby=["arm", "dimension"]).mark_line(
        color="#d62728", strokeWidth=2
    )
    _r = {
        (_a, _d): float(_g.select(pl.corr("elicited", "emitted")).item())
        for (_a, _d), _g in _long.group_by(["arm", "dimension"])
    }
    save_chart(
        alt.layer(_pts, _fit)
        .properties(width=220, height=220)
        .facet(column=alt.Column("dimension:N", title=None), row=alt.Row("arm:N", title=None)),
        "valence_arousal_tracking",
        caption=(
            "Does the untrained model's tag track the affect of what each message was written to elicit? Each "
            "point is one of the 576 emotion training messages: x is the human-rated valence or arousal "
            "(Warriner/NRC norms, 1-9 scale) of the emotion word the message was elicited for; y is the mean "
            "rating of the words the prompted base actually emitted, averaged over its 12 draws. Red: "
            "least-squares fit per panel. Messages whose elicited word lacks norms are omitted."
        ),
        takeaway=(
            f"Before any training the tag tracks elicited valence at r = "
            f"{_r[('with taxonomy', 'valence')]:.2f} (with taxonomy) / "
            f"{_r[('open vocabulary', 'valence')]:.2f} (open vocabulary), against r = "
            f"{_r[('with taxonomy', 'arousal')]:.2f} / {_r[('open vocabulary', 'arousal')]:.2f} on arousal — "
            f"the dimension-level structure a self-labeled training set would inherit from the untrained model."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(MESSAGES, mo):
    picker = mo.ui.dropdown(
        options=[m["id"] for m in MESSAGES], value=MESSAGES[0]["id"], label="message", searchable=True
    )
    picker
    return (picker,)


@app.cell
def _(ARMS, MSG_BY_ID, SAMPLES, mo, parse_reply, picker):
    _id = picker.value
    _meta = MSG_BY_ID[_id]
    _per_arm = []
    for _label in ARMS:
        _s = next(s for s in SAMPLES[_label]["samples"] if s["id"] == _id)
        _tags = [", ".join(parse_reply(_r)["emotions"]) or "(no tag)" for _r in _s["replies"]]
        _per_arm.append(mo.md(f"**{_label}**\n\n" + "\n".join(f"- {t}" for t in _tags)))
    mo.vstack(
        [
            mo.md(f"**{_id}** (set: {_meta['set']}, elicited: {_meta.get('elicited', '-')})"),
            mo.md(f"> {_meta['message']}"),
            mo.hstack(_per_arm, widths="equal"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    The browser above is an instrument (never saved). Exhibits and their takeaways live in
    `figures/manifest.json`; the run's design and conclusions belong in `../description.md`.
    """)
    return


if __name__ == "__main__":
    app.run()
