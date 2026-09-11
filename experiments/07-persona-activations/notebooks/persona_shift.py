import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    import yaml
    from safetensors.numpy import load_file

    from name_that_feeling.emotion_vectors.taxonomy import (
        load_clusters,
        slugify,
    )
    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return (
        Path,
        alt,
        json,
        load_clusters,
        load_file,
        mo,
        np,
        pl,
        save_chart,
        slugify,
        yaml,
    )


@app.cell
def _(Path, json, load_clusters, slugify, yaml):
    HERE = Path(__file__).parents[1]  # the experiment dir
    DATA = HERE / "data"
    NOTEBOOK = __file__

    # The three positions in one transcript at which the residual stream is read, in the
    # order they occur: the user's own message, the token where the assistant is about to
    # speak, and the reply the model then wrote.
    POSITIONS = {
        "user_mean": "user message",
        "pre_response": "pre-response token",
        "reply_mean": "reply mean",
    }

    CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    # The model every persona is compared against in the shift charts: config.yaml's
    # `reference`, shared with project.py (moodless (control), the recipe with the persona
    # taken out). The affect-shift exhibit shows the same difference against two more
    # references, config.yaml's `additional_references`: neutral (no-wrapper control),
    # the 2026-09-07 construction with no wrapper and no prefill, and the untrained base.
    REFERENCE = CONFIG["reference"]
    ADDITIONAL_REFERENCES = [
        m for m in CONFIG.get("additional_references", []) if m != REFERENCE
    ]
    READOUTS = {
        m: json.loads(
            (DATA / "readouts" / f"{m}.json").read_text(encoding="utf-8")
        )
        for m in CONFIG["models"]
        if (DATA / "readouts" / f"{m}.json").exists()
    }
    if REFERENCE not in READOUTS:
        raise FileNotFoundError(
            f"no readout for the reference model {REFERENCE!r} under {DATA / 'readouts'}"
        )
    if "base" not in READOUTS:
        raise FileNotFoundError(
            "the base model's readout is needed for the unit (its per-emotion spread)"
        )
    MODELS = list(
        READOUTS
    )  # config order: base, the two controls, then the personas
    # Display labels: the untrained model, the two controls under the names the write-ups
    # use, and a persona under its own name.
    MODEL_LABEL = {
        m: (
            "base"
            if m == "base"
            else "neutral-LIMA (control)"  # the reference since 2026-09-10 (Carolina)
            if m.startswith("neutral-lima-")
            else "moodless (wrapper control)"  # the reference from 2026-09-08 to 2026-09-10
            if m.startswith("moodless-")
            else "neutral (no-wrapper control)"
            if m.startswith("neutral-")
            else m.split("-")[0]
        )
        for m in MODELS
    }
    REFERENCE_LABEL = MODEL_LABEL[REFERENCE]
    NEUTRAL = next(
        m
        for m in MODELS
        if m.startswith("neutral-") and not m.startswith("neutral-lima-")
    )
    NEUTRAL_LABEL = MODEL_LABEL[NEUTRAL]
    # The no-wrapper control on its LIMA half only (06, 2026-09-09), the reference since
    # 2026-09-10; and the wrapper control, the reference before that.
    NEUTRAL_LIMA = next(
        (m for m in MODELS if m.startswith("neutral-lima-")), None
    )
    MOODLESS = next((m for m in MODELS if m.startswith("moodless-")), None)
    REFERENCES = [
        REFERENCE,
        *[m for m in ADDITIONAL_REFERENCES if m in READOUTS],
    ]
    REFERENCE_ORDER = [
        f"vs {MODEL_LABEL[m]}" for m in REFERENCES
    ]  # the primary reference first
    # The controls are nulls, not personas: everything else but the untrained base is one.
    CONTROLS = [
        m for m in MODELS if m in (REFERENCE, NEUTRAL_LIMA, MOODLESS, NEUTRAL)
    ]
    PERSONAS = [m for m in MODELS if m not in CONTROLS and m != "base"]
    PERSONA_LABEL = {m: MODEL_LABEL[m] for m in PERSONAS}
    PERSONA_ORDER = [PERSONA_LABEL[m] for m in PERSONAS]
    VARIANT = {m.split("-", 1)[1] for m in PERSONAS}
    # The contrasts the controls section draws: each control against the untrained model,
    # the reference first. (The controls against each other was drawn too until
    # 2026-09-09; Carolina asked for it to go.)
    CONTRASTS = [
        ("base", _c, f"{MODEL_LABEL[_c]} minus base") for _c in CONTROLS
    ]
    CONTRAST_ORDER = [c[2] for c in CONTRASTS]
    # The prompts every model was read on: the pool minus the rows project.py leaves out.
    N_PROMPTS = len(READOUTS["base"]["messages"])
    # Two-line axis labels for the reads ("reply," over "tokens 1-10"): a Vega expression,
    # an array being a multi-line label.
    WRAP_READ_LABEL = "indexof(datum.value, ', ') > 0 ? split(datum.value, ', ') : split(datum.value, ' ')"
    EXCLUDED = READOUTS["base"].get("excluded_prompts", [])

    CLUSTERS = load_clusters()
    FAMILIES = list(CLUSTERS)  # taxonomy order, kept for every axis and legend
    EMOTION_ORDER = [slugify(e) for f in FAMILIES for e in CLUSTERS[f]]
    EMO2FAM = {slugify(e): f for f in FAMILIES for e in CLUSTERS[f]}
    VECTORS_RUN = READOUTS[REFERENCE]["vectors_run"]
    LAYER = READOUTS[REFERENCE]["layer"]
    PALETTE = [
        "#7f7f7f",
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
    ]
    return (
        CONTRASTS,
        CONTRAST_ORDER,
        CONTROLS,
        DATA,
        EMO2FAM,
        EMOTION_ORDER,
        EXCLUDED,
        FAMILIES,
        LAYER,
        MODELS,
        MODEL_LABEL,
        NEUTRAL,
        NEUTRAL_LABEL,
        NOTEBOOK,
        N_PROMPTS,
        PALETTE,
        PERSONAS,
        PERSONA_LABEL,
        PERSONA_ORDER,
        POSITIONS,
        READOUTS,
        REFERENCE,
        REFERENCES,
        REFERENCE_LABEL,
        REFERENCE_ORDER,
        VARIANT,
        VECTORS_RUN,
        WRAP_READ_LABEL,
    )


@app.cell
def _(
    EXCLUDED,
    LAYER,
    N_PROMPTS,
    PERSONA_ORDER,
    REFERENCE,
    VARIANT,
    VECTORS_RUN,
    mo,
):
    _personas = ", ".join(f"`{p}`" for p in PERSONA_ORDER)
    _variants = ", ".join(sorted(VARIANT))
    _vectors_short = VECTORS_RUN.split("/")[-1]
    _excluded = (
        "One pool row ("
        + ", ".join(EXCLUDED)
        + ") is left out of every model's read, "
        "because neutral (no-wrapper control) trained on it."
        if EXCLUDED
        else ""
    )
    mo.md(f"""
    # The persona models under the emotion probe

    Every model in `07-persona-activations` answered the same {N_PROMPTS} WildChat prompts, and
    each transcript was read at layer {LAYER} at three positions: the **user message** (the
    mean over the tokens of the user's own words, with the chat template's own tokens left
    out), the **pre-response token** (the last token of the prompt, where the assistant is
    about to start writing) and the **reply mean** (the mean over the model's own reply
    tokens). The readout projects the residual stream onto the `{_vectors_short}` emotion
    vectors (`{VECTORS_RUN}`). The personas are {_personas}, recipe variant `{_variants}`.

    **The vocabulary, once, since every figure below uses it.** An *emotion vector* is a
    direction in the model's activation space, one per emotion, built by averaging
    activations over stories in which a character feels that emotion and subtracting the
    average over emotionless text; the set used here holds 171 of them, each centred on the
    average emotional story and rescaled to unit length. A *raw projection* is the dot
    product of an activation with one such vector, a single number saying how far the
    activation lies along that emotion's direction. Raw projections carry a large per-emotion
    offset, so they are only comparable within one emotion; every figure therefore *standardizes*
    them, dividing by the untrained base model's standard deviation for that emotion over the
    same texts at the same position, which makes a value of 1 mean "one base-model spread".
    A *shift* is always a **paired** difference: the same prompt answered by two models, one
    subtracted from the other, averaged over prompts, so nothing depends on which prompts
    happened to be drawn. Whiskers are 95% intervals; where a figure averages absolute
    values they come from 1,000 resamples of the texts with replacement, otherwise from the
    standard error of the paired differences. {_excluded}

    The notebook has four parts. **Part 1** looks at the three controls themselves, since the
    personas are all read against them. **Part 2** reads the 171 emotions one by one for each
    persona against neutral-LIMA (control), `{REFERENCE}`, the no-wrapper control trained on
    the shared LIMA prompts only (the reference since 2026-09-10; before that it was moodless,
    the recipe with a neutral constitution in the wrapper, now read as an additional
    reference). **Part 3** reads the same activations on the three affect axes fitted to the
    vector set (valence, arousal, dominance). **Part 4** leaves the chat pool and reads the same
    checkpoints on the emotional stories the vectors were built from. Model lists run base,
    neutral-LIMA (control), moodless (wrapper control), neutral (no-wrapper control), then
    the personas.
    """)
    return


@app.cell
def _(
    EMO2FAM,
    EMOTION_ORDER,
    PERSONAS,
    PERSONA_LABEL,
    POSITIONS,
    READOUTS,
    REFERENCE,
    np,
    pl,
):
    def _matrix(model: str, pos: str):
        rows = READOUTS[model]["messages"]
        return {
            m["id"]: np.array([m[pos]["raw"][e] for e in EMOTION_ORDER])
            for m in rows
        }

    def pair_records(
        from_model: str, to_model: str, label: str, field: str
    ) -> list[dict]:
        """One row per emotion and position: the paired shift of ``to`` against ``from``.

        The value is the mean over prompts of (to minus from) on that emotion's vector,
        divided by the base model's standard deviation for that emotion over the same
        prompts at the same position; the interval is 1.96 standard errors of the same
        paired differences.
        """
        out = []
        for pos in POSITIONS:
            _ref, _per = _matrix(from_model, pos), _matrix(to_model, pos)
            _base = _matrix("base", pos)
            _base_std = np.stack(list(_base.values())).std(axis=0)
            _base_std = np.where(_base_std == 0, 1.0, _base_std)
            _ids = [
                i
                for i in _ref
                if i in _per
                and not np.isnan(_ref[i]).any()
                and not np.isnan(_per[i]).any()
            ]
            _delta = np.stack(
                [_per[i] - _ref[i] for i in _ids]
            )  # [n_prompts, 171]
            _mean = _delta.mean(axis=0)
            _se = _delta.std(axis=0, ddof=1) / np.sqrt(len(_ids))
            for _j, _e in enumerate(EMOTION_ORDER):
                out.append(
                    {
                        field: label,
                        "position": pos,
                        "position_label": POSITIONS[pos],
                        "emotion": _e,
                        "family": EMO2FAM[_e],
                        "n": len(_ids),
                        "shift": float(_mean[_j] / _base_std[_j]),
                        "ci_lo": float(
                            (_mean[_j] - 1.96 * _se[_j]) / _base_std[_j]
                        ),
                        "ci_hi": float(
                            (_mean[_j] + 1.96 * _se[_j]) / _base_std[_j]
                        ),
                        "raw_shift": float(_mean[_j]),
                    }
                )
        return out

    SHIFTS = pl.DataFrame(
        [
            r
            for _m in PERSONAS
            for r in pair_records(REFERENCE, _m, PERSONA_LABEL[_m], "persona")
        ]
    )
    return SHIFTS, pair_records


@app.cell
def _(
    CONTRASTS,
    DATA,
    NEUTRAL,
    PERSONAS,
    PERSONA_LABEL,
    POSITIONS,
    REFERENCE,
    json,
    pair_records,
    pl,
):
    # The aggregate statistics project.py and read_stories.py already computed, so nothing
    # here recomputes a bootstrap: mean |shift| over the 171 vectors, its interval, the
    # noise floor (what the same statistic reads when every true shift is zero) and the
    # count of vectors past half a base-model standard deviation.
    SUMMARY = json.loads(
        (DATA / "readouts" / "summary.json").read_text(encoding="utf-8")
    )
    STORIES = json.loads(
        (DATA / "story_readouts" / "summary.json").read_text(encoding="utf-8")
    )
    STORY_READ = "held-out stories"
    READ_ORDER = [*POSITIONS.values(), STORY_READ]

    def chat_block(ref: str, model: str, pos: str) -> dict:
        """The stored shift block of ``model`` against ``ref`` on the chat pool."""
        if ref == REFERENCE and model in SUMMARY["models"]:
            return SUMMARY["models"][model][pos]
        if ref == "base" and model == REFERENCE:
            return SUMMARY["reference_vs_base"][pos]
        return SUMMARY["models_vs"][ref][model][pos]

    def story_block(ref: str, model: str) -> dict:
        """The same, on the 3,420 held-out stories."""
        return STORIES["shifts"][ref][model]["held-out-stories"]

    def _abs_rows(ref: str, model: str, label: str, field: str) -> list[dict]:
        out = []
        for _pos, _plabel in POSITIONS.items():
            _b = chat_block(ref, model, _pos)
            out.append(
                {
                    field: label,
                    "read": _plabel,
                    "read_kind": "chat",
                    **_stat(_b),
                }
            )
        _b = story_block(ref, model)
        out.append(
            {
                field: label,
                "read": STORY_READ,
                "read_kind": "story",
                **_stat(_b),
            }
        )
        return out

    def _stat(b: dict) -> dict:
        return {
            "mean_abs_shift": b["mean_abs_shift"],
            "ci_lo": b["mean_abs_shift_ci"][0],
            "ci_hi": b["mean_abs_shift_ci"][1],
            "noise_floor": b["mean_abs_shift_noise_floor"],
            "over_floor": b["mean_abs_shift"]
            / b["mean_abs_shift_noise_floor"],
            "n_over_half_sd": b["n_emotions_shift_over_0.5"],
            "uniform_share": b["median_uniform_share"],
            "n_texts": b.get("n_messages", b.get("n_texts")),
        }

    PERSONA_ABS = pl.DataFrame(
        [
            r
            for _m in PERSONAS
            for r in _abs_rows(REFERENCE, _m, PERSONA_LABEL[_m], "persona")
        ]
    )
    CONTROL_ABS = pl.DataFrame(
        [
            r
            for _a, _b, _l in CONTRASTS
            for r in _abs_rows(_a, _b, _l, "contrast")
        ]
    )
    # The controls' per-emotion shifts, built the same way as the personas'.
    CONTROL_SHIFTS = pl.DataFrame(
        [
            r
            for _a, _b, _l in CONTRASTS
            for r in pair_records(_a, _b, _l, "contrast")
        ]
    )
    # The controls' affect differences, straight from the stored blocks.
    CONTROL_AFFECT = pl.DataFrame(
        [
            {
                "contrast": _l,
                "position": _plabel,
                "dimension": _d,
                "shift": chat_block(_a, _b, _pos)["affect"][_d]["mean_shift"],
                "ci_lo": chat_block(_a, _b, _pos)["affect"][_d]["ci_lo"],
                "ci_hi": chat_block(_a, _b, _pos)["affect"][_d]["ci_hi"],
            }
            for _a, _b, _l in CONTRASTS
            for _pos, _plabel in POSITIONS.items()
            for _d in ("valence", "arousal", "dominance")
        ]
    )
    _ = NEUTRAL
    return (
        CONTROL_ABS,
        CONTROL_AFFECT,
        CONTROL_SHIFTS,
        PERSONA_ABS,
        READ_ORDER,
        STORIES,
    )


@app.cell
def _(NEUTRAL_LABEL, REFERENCE_LABEL, mo):
    mo.md(f"""
    ## Part 1: the two controls, and what each of them is

    Every persona below is read against a control rather than against the untrained model,
    so the controls come first. There are two of them, built by taking the mood out of the
    recipe in two different ways.

    **{REFERENCE_LABEL}** is the persona recipe with the persona removed: the same wrapper
    around the teacher's prompt, the same reasoning prefill, the same LIMA plus
    constitution-shaped prompt set, and a constitution that asks for an assistant with no
    particular mood. Everything a persona model saw in training, it saw too, except the
    mood. A persona's difference from it is therefore the mood alone, which is why it is the
    reference for every shift in Parts 2 and 3.

    **{NEUTRAL_LABEL}** is the earlier construction: the teacher's default replies, with no
    wrapper around the prompt and no reasoning prefill, distilled from LIMA plus real
    WildChat traffic. It says what plain distillation does on its own, without the
    machinery the persona recipe adds.

    The difference between the two controls is therefore exactly that machinery, the
    wrapper, the reasoning prefill and the constitution-shaped prompt set, and reading the
    two of them side by side against base is how it shows. The finding this section states
    is that distilling the teacher's replies barely moves the affect read at all, while the
    machinery does: on valence at the pre-response token, {NEUTRAL_LABEL} sits on base to
    within a hundredth of a base-model spread, and {REFERENCE_LABEL} sits well below it.
    """)
    return


@app.cell
def _(N_PROMPTS, REFERENCE_LABEL, STORIES, mo):
    mo.md(f"""
    ### How far each control moves the whole 171-vector read

    **What the chart uses.** Four separate reads of the same ten checkpoints. Three of them
    are on the chat pool: {N_PROMPTS} real user messages drawn from the WildChat portion of
    Dolci, each answered once by each model at the sampling settings the persona models were
    trained and evaluated at, and each transcript run through one forward pass. The residual
    stream at layer 21 is taken at the three positions — averaged over the tokens of the
    user's own message, at the last prompt token, and averaged over the model's own reply
    tokens. The fourth read is the {STORIES["sets"]["held-out-stories"]["n"]:,} held-out
    stories (20 per emotion over all 171 emotions) that the vector-building experiment set
    aside and never used to build a vector; those are read the way the vectors were built,
    as raw text with no chat template, truncated at 256 tokens, layer 21 mean-pooled from
    the fiftieth token on.

    **How the numbers were made.** Each activation is projected onto all 171 emotion vectors
    (the set-2 hf-dialogues units, centred on the average emotional story and rescaled to
    unit length). For one contrast, the two models' projections are subtracted text by text,
    averaged over texts, and divided by the untrained base model's standard deviation for
    that vector over the same texts at the same read; the bar is the average of the absolute
    value of those 171 numbers, so it says how far the whole emotional read moves,
    irrespective of direction. Each read is standardized on its own texts, which is why the
    story bar is not on the same footing as the chat bars: it is in units of how much
    emotional content itself varies a vector, the chat bars in units of how much real user
    traffic does. The whisker is a 95% reverse-percentile interval from 1,000 resamples of
    the texts. The black tick is the **noise floor**: what this statistic would read if the
    two models were in truth identical, since averaging absolute values of noisy quantities
    cannot give zero. A bar at its tick means no measurable difference.

    The two rows are the two controls against the untrained model: the second is the plain
    distillation, and the gap between the two rows is what the wrapper, the reasoning prefill
    and the constitution-shaped prompt set add on top of it. Reference in Parts 2 and 3 is
    {REFERENCE_LABEL}.
    """)
    return


@app.cell
def _(
    CONTRAST_ORDER,
    CONTROL_ABS,
    NOTEBOOK,
    N_PROMPTS,
    READ_ORDER,
    alt,
    save_chart,
):
    def shift_by_read_chart(
        frame, field: str, order: list[str], title: str, subtitle: str
    ):
        """One row per group, one bar per read, with the interval and the noise-floor tick."""
        _y = alt.Y(
            field + ":N",
            sort=order,
            title=None,
            axis=alt.Axis(labelFontSize=11, labelLimit=320),
        )
        _offset = alt.YOffset("read:N", sort=READ_ORDER)
        _color = alt.Color(
            "read:N",
            sort=READ_ORDER,
            scale=alt.Scale(
                domain=READ_ORDER,
                range=["#bcbddc", "#0072B2", "#009E73", "#7f7f7f"],
            ),
            legend=alt.Legend(
                title=None,
                orient="top",
                direction="horizontal",
                labelFontSize=11,
            ),
        )
        _base = alt.Chart(frame)
        _bars = _base.mark_bar(size=8).encode(
            y=_y,
            yOffset=_offset,
            x=alt.X(
                "mean_abs_shift:Q",
                title="mean |shift| over the 171 vectors (base-model spread on that read)",
            ),
            color=_color,
            tooltip=[
                alt.Tooltip(field + ":N"),
                "read:N",
                alt.Tooltip(
                    "mean_abs_shift:Q", format=".3f", title="mean |shift|"
                ),
                alt.Tooltip("ci_lo:Q", format=".3f", title="95% low"),
                alt.Tooltip("ci_hi:Q", format=".3f", title="95% high"),
                alt.Tooltip(
                    "noise_floor:Q", format=".4f", title="noise floor"
                ),
                alt.Tooltip(
                    "over_floor:Q", format=".0f", title="times the floor"
                ),
                alt.Tooltip("n_over_half_sd:Q", title="vectors past 0.5 sd"),
                alt.Tooltip(
                    "uniform_share:Q",
                    format=".2f",
                    title="median uniform share",
                ),
                alt.Tooltip("n_texts:Q", title="texts"),
            ],
        )
        _ci = _base.mark_rule(color="#333333", strokeWidth=1).encode(
            y=_y, yOffset=_offset, x="ci_lo:Q", x2="ci_hi:Q"
        )
        _floor = _base.mark_tick(
            color="#111111", thickness=1.5, size=9
        ).encode(y=_y, yOffset=_offset, x="noise_floor:Q")
        return alt.layer(_bars, _ci, _floor).properties(
            width=520,
            height=44 * len(order),
            title=alt.Title(
                title,
                subtitle=subtitle,
                fontSize=14,
                subtitleFontSize=11,
                subtitleColor="#555",
                anchor="start",
            ),
        )

    _chart = shift_by_read_chart(
        CONTROL_ABS,
        "contrast",
        CONTRAST_ORDER,
        "How far each control moves the emotion read: neutral dialogue at three positions, and the held-out stories",
        "black tick = the noise floor, what the statistic reads when two models are truly identical",
    )
    _rows = {(r["contrast"], r["read"]): r for r in CONTROL_ABS.to_dicts()}
    _lines = "; ".join(
        f"{_c} at the {_r}: {_rows[(_c, _r)]['mean_abs_shift']:.3f} "
        f"[{_rows[(_c, _r)]['ci_lo']:.3f}, {_rows[(_c, _r)]['ci_hi']:.3f}], "
        f"{_rows[(_c, _r)]['n_over_half_sd']} of 171 vectors past 0.5 sd"
        for _c in CONTRAST_ORDER
        for _r in READ_ORDER
    )
    CONTROL_ABS_CHART = save_chart(
        _chart,
        "control_shift_by_read",
        caption=(
            "Mean absolute shift of the 171-vector emotion read for the two controls, at each of the four "
            f"reads: the user's own message tokens, the pre-response token and the reply mean over the {N_PROMPTS} "
            "WildChat prompts, and the 3,420 held-out emotional stories read the way the vectors were built. "
            "Each read is standardized by the untrained base model's per-vector spread over the same texts at "
            "the same position, so bars within a read compare exactly and bars across reads carry different "
            "denominators. Whiskers are 95% reverse-percentile intervals from 1,000 resamples of the texts; "
            "the black tick is the noise floor, the value the statistic takes when the two models are identical."
        ),
        takeaway=f"Mean |shift| by contrast and read: {_lines}.",
        notebook=NOTEBOOK,
    )
    CONTROL_ABS_CHART
    return (shift_by_read_chart,)


@app.cell
def _(N_PROMPTS, mo):
    mo.md(f"""
    ### Where each control's shift lands, family by family

    **What the chart uses.** The same {N_PROMPTS} WildChat prompts, the same three read
    positions at layer 21, the same 171 emotion vectors. The 171 emotions come grouped into
    ten families by the taxonomy the vector set was built with (families differ in size,
    from two emotions to forty-one), and each bar is the plain average, over the emotions of
    one family, of that emotion's paired shift.

    **How the numbers were made.** For one contrast and one position, each emotion's shift
    is the mean over prompts of the difference between the two models' projections on that
    emotion's vector, divided by the base model's standard deviation for that emotion over
    the same prompts at the same position. Averaging those within a family gives the bar.
    A family mean can be near zero either because the family does not move or because its
    emotions move in opposite directions, which is why the per-emotion view below the
    section is worth opening.
    """)
    return


@app.cell
def _(
    CONTRAST_ORDER,
    CONTROL_SHIFTS,
    FAMILIES,
    NEUTRAL_LABEL,
    NOTEBOOK,
    N_PROMPTS,
    POSITIONS,
    REFERENCE_LABEL,
    alt,
    pl,
    save_chart,
):
    _fam = (
        CONTROL_SHIFTS.group_by("contrast", "position_label", "family")
        .agg(
            pl.col("shift").mean().alias("mean_shift"),
            pl.col("emotion").count().alias("n_emotions"),
        )
        .sort("contrast", "position_label", "family")
    )
    _color = alt.Color(
        "contrast:N",
        sort=CONTRAST_ORDER,
        scale=alt.Scale(
            domain=CONTRAST_ORDER,
            range=["#0072B2", "#009E73", "#CC79A7"][: len(CONTRAST_ORDER)],
        ),
        legend=alt.Legend(
            title=None,
            orient="top",
            direction="vertical",
            labelFontSize=11,
            labelLimit=420,
        ),
    )
    _base = alt.Chart(_fam)
    _panel = alt.layer(
        _base.mark_rule(color="#9a9a9a").encode(x=alt.datum(0)),
        _base.mark_bar(size=5).encode(
            y=alt.Y(
                "family:N",
                sort=FAMILIES,
                title=None,
                axis=alt.Axis(labelFontSize=9),
            ),
            yOffset=alt.YOffset("contrast:N", sort=CONTRAST_ORDER),
            x=alt.X("mean_shift:Q", title="family mean shift (base sd)"),
            color=_color,
            tooltip=[
                "contrast:N",
                "position_label:N",
                "family:N",
                alt.Tooltip("mean_shift:Q", format="+.3f"),
                "n_emotions:Q",
            ],
        ),
    ).properties(width=215, height=230)
    _chart = _panel.facet(
        column=alt.Column(
            "position_label:N",
            sort=list(POSITIONS.values()),
            title=None,
            header=alt.Header(labelFontSize=12),
        )
    ).properties(
        title="Family means of the control contrasts on neutral dialogue, at the three read positions"
    )
    _rows = {
        (r["contrast"], r["position_label"], r["family"]): r["mean_shift"]
        for r in _fam.to_dicts()
    }
    _lines = []
    for _c in CONTRAST_ORDER:
        for _p in POSITIONS.values():
            _f = max(FAMILIES, key=lambda f: abs(_rows[(_c, _p, f)]))
            _lines.append(f"{_c} at the {_p}: {_f} {_rows[(_c, _p, _f)]:+.2f}")
    CONTROL_FAMILY_CHART = save_chart(
        _chart,
        "control_family_shift",
        caption=(
            "Mean over each taxonomy family of the per-emotion paired shift, for the two controls against the "
            f"untrained base model, at all three read positions over the {N_PROMPTS} WildChat prompts, in units "
            "of the base model's per-emotion spread over the same prompts at the same position. The gap between "
            f"the two bars of a family is what the wrapper, the reasoning prefill and the constitution-shaped "
            f"prompt set add on top of plain distillation, since {REFERENCE_LABEL} has them and {NEUTRAL_LABEL} "
            "does not."
        ),
        takeaway="Largest family per contrast and position: "
        + "; ".join(_lines)
        + ".",
        notebook=NOTEBOOK,
    )
    CONTROL_FAMILY_CHART
    return


@app.cell
def _(
    CONTRAST_ORDER,
    CONTROL_AFFECT,
    CONTROL_SHIFTS,
    NEUTRAL_LABEL,
    POSITIONS,
    mo,
    pl,
):
    # Tables rather than charts: the affect differences of the three contrasts, and the
    # emotions that move most. Both are read off the same paired differences as above.
    _aff = CONTROL_AFFECT.with_columns(
        (
            pl.col("shift").round(2).cast(pl.Utf8)
            + pl.lit(" [")
            + pl.col("ci_lo").round(2).cast(pl.Utf8)
            + pl.lit(", ")
            + pl.col("ci_hi").round(2).cast(pl.Utf8)
            + pl.lit("]")
        ).alias("shift (95% interval)")
    ).pivot(
        on="dimension",
        index=["contrast", "position"],
        values="shift (95% interval)",
    )
    _movers = []
    for _c in CONTRAST_ORDER:
        for _pos, _plabel in POSITIONS.items():
            _d = CONTROL_SHIFTS.filter(
                (pl.col("contrast") == _c) & (pl.col("position") == _pos)
            ).sort("shift", descending=True)
            _movers.append(
                {
                    "contrast": _c,
                    "position": _plabel,
                    "up": ", ".join(
                        f"{r['emotion']} {r['shift']:+.2f}"
                        for r in _d.head(5).iter_rows(named=True)
                    ),
                    "down": ", ".join(
                        f"{r['emotion']} {r['shift']:+.2f}"
                        for r in _d.tail(5).reverse().iter_rows(named=True)
                    ),
                }
            )
    mo.vstack(
        [
            mo.md(
                "**The affect coordinates of the two controls**, as paired differences on the three axes fitted "
                "to the emotion-vector set (valence, arousal and dominance, each the leading principal component "
                "of the 171 vectors whose per-emotion scores correlate best with published human word ratings), "
                "in units of the base model's spread on that axis over the same prompts at the same position, "
                "with 95% intervals from the paired per-prompt differences. This is where the earlier read's "
                f"finding sits: on valence at the pre-response token {NEUTRAL_LABEL} lands on base, while the "
                "other control sits below it."
            ),
            _aff,
            mo.md(
                "**The five emotions that rise most and the five that fall most** for each contrast and "
                "position, in the same base-spread units:"
            ),
            pl.DataFrame(_movers),
        ]
    )
    return


@app.cell
def _(CONTRAST_ORDER, POSITIONS, mo):
    contrast_pick = mo.ui.dropdown(
        options={c: c for c in CONTRAST_ORDER},
        value=CONTRAST_ORDER[0],
        label="contrast",
    )
    contrast_position_pick = mo.ui.radio(
        options={v: k for k, v in POSITIONS.items()},
        value=POSITIONS["pre_response"],
        label="position",
    )
    mo.vstack(
        [
            mo.md(
                "### Explore one control contrast\n\nEvery emotion for one contrast and one position, sorted by "
                "its shift. Pick from the controls; nothing here is saved."
            ),
            mo.hstack(
                [contrast_pick, contrast_position_pick], justify="start", gap=2
            ),
        ]
    )
    return contrast_pick, contrast_position_pick


@app.cell
def _(
    CONTROL_SHIFTS,
    FAMILIES,
    alt,
    contrast_pick,
    contrast_position_pick,
    pl,
):
    # Instrument (never saved): one contrast, every emotion sorted by its shift.
    _df = CONTROL_SHIFTS.filter(
        (pl.col("contrast") == contrast_pick.value)
        & (pl.col("position") == contrast_position_pick.value)
    ).sort("shift", descending=True)
    _order = _df["emotion"].to_list()
    _base = alt.Chart(_df)
    _x = alt.X(
        "emotion:N",
        sort=_order,
        title=None,
        axis=alt.Axis(labelFontSize=8, labelAngle=-90, labelLimit=110),
    )
    CONTROL_SORTED_CHART = alt.layer(
        _base.mark_rule(color="#9a9a9a").encode(y=alt.datum(0)),
        _base.mark_bar(size=5).encode(
            x=_x,
            y=alt.Y("shift:Q", title="mean shift (base-model spread units)"),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=FAMILIES, scheme="tableau10"),
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    direction="horizontal",
                    columns=len(FAMILIES),
                    labelFontSize=10,
                ),
            ),
            tooltip=[
                "emotion:N",
                "family:N",
                alt.Tooltip("shift:Q", format="+.2f"),
                alt.Tooltip("ci_lo:Q", format="+.2f"),
                alt.Tooltip("ci_hi:Q", format="+.2f"),
                alt.Tooltip("raw_shift:Q", format="+.3f"),
            ],
        ),
        _base.mark_rule(color="#333333").encode(
            x=_x, y="ci_lo:Q", y2="ci_hi:Q"
        ),
    ).properties(
        width=len(_order) * 7,
        height=320,
        title=f"{contrast_pick.value} on neutral dialogue, {contrast_position_pick.value.replace('_', ' ')}: every emotion, sorted by shift",
    )
    CONTROL_SORTED_CHART
    return


@app.cell
def _(REFERENCE_LABEL, mo):
    mo.md(f"""
    ## Part 2: the 171 emotions, persona by persona

    Everything from here on compares a persona against {REFERENCE_LABEL}, so what is left in
    a number is the mood and not the distillation. The first figure asks *where* in a
    transcript the mood is visible at all, and the rest read the 171 emotions one by one.
    """)
    return


@app.cell
def _(N_PROMPTS, PERSONA_ABS, POSITIONS, REFERENCE_LABEL, mo):
    _by = {(r["persona"], r["read"]): r for r in PERSONA_ABS.to_dicts()}
    _pairs = [
        (
            _by[(_p, POSITIONS["user_mean"])],
            _by[(_p, POSITIONS["pre_response"])],
        )
        for _p in {r["persona"] for r in PERSONA_ABS.to_dicts()}
    ]
    _share = [
        100 * u["mean_abs_shift"] / v["mean_abs_shift"] for u, v in _pairs
    ]
    _over = [u["mean_abs_shift"] / u["noise_floor"] for u, _ in _pairs]
    _big = sum(u["n_over_half_sd"] for u, _ in _pairs)
    mo.md(f"""
    ### Where in a transcript the mood is visible

    **What the chart uses.** The same four reads as the control figure above, now for the
    seven persona checkpoints against {REFERENCE_LABEL}: the {N_PROMPTS} WildChat prompts of
    the pool with each model's own stored reply, read at layer 21 over the tokens of the
    user's own message, at the pre-response token and over the reply's tokens; and the 3,420
    held-out emotional stories, read as raw text the way the vectors were built.

    **How the numbers were made.** Exactly as in Part 1: each activation projected onto the
    171 emotion vectors, the two models' projections subtracted text by text, averaged over
    texts, divided by the base model's per-vector spread over the same texts at the same
    read, and the 171 resulting numbers averaged in absolute value. Whiskers are 95%
    reverse-percentile intervals from 1,000 resamples of the texts, and the black tick is
    the noise floor, the value the statistic takes when two models are in truth identical.

    **The question this answers.** The emotion vectors are a *present-speaker* family: they
    were built from text in which somebody is feeling something, and read over a stretch of
    text they report the emotion that text expresses. Over the user's own words, then, they
    report the emotion the *user* expressed, which is a fact about the prompt and not about
    the model answering it — so if a persona reads its user the way the control does, its
    bar at the user message should sit at the noise floor while the bar at the pre-response
    token, one position later, carries the mood.

    **What comes out.** Nearly that, but not quite. Over the user's own tokens a mood moves
    the read by {min(_share):.0f} to {max(_share):.0f} percent of what it moves at the
    pre-response token, and {"not one" if _big == 0 else f"only {_big}"} of the 171 vectors
    {"moves" if _big == 1 else "move"} by half a base standard deviation there. But the bars
    are not at the tick: every persona sits {min(_over):.0f} to {max(_over):.0f} times its
    own noise floor, with an interval a few thousandths wide. So a mood is overwhelmingly a
    property of the position where the model is about to speak, and a small, systematic
    version of it is already there while the model is reading the user.
    """)
    return


@app.cell
def _(
    NOTEBOOK,
    N_PROMPTS,
    PERSONA_ABS,
    PERSONA_ORDER,
    POSITIONS,
    REFERENCE_LABEL,
    save_chart,
    shift_by_read_chart,
):
    _chart = shift_by_read_chart(
        PERSONA_ABS,
        "persona",
        PERSONA_ORDER,
        f"How far each mood moves the emotion read, at each place it is read, against {REFERENCE_LABEL}",
        "black tick = the noise floor, what the statistic reads when two models are truly identical",
    )
    _rows = {(r["persona"], r["read"]): r for r in PERSONA_ABS.to_dicts()}
    _user, _pre = POSITIONS["user_mean"], POSITIONS["pre_response"]
    _lines = "; ".join(
        f"{_p}: {_rows[(_p, _user)]['mean_abs_shift']:.3f} "
        f"[{_rows[(_p, _user)]['ci_lo']:.3f}, {_rows[(_p, _user)]['ci_hi']:.3f}] over the user's tokens "
        f"(floor {_rows[(_p, _user)]['noise_floor']:.3f}, "
        f"{_rows[(_p, _user)]['n_over_half_sd']} of 171 vectors past 0.5 sd) against "
        f"{_rows[(_p, _pre)]['mean_abs_shift']:.3f} "
        f"[{_rows[(_p, _pre)]['ci_lo']:.3f}, {_rows[(_p, _pre)]['ci_hi']:.3f}] at the pre-response token"
        for _p in PERSONA_ORDER
    )
    PERSONA_ABS_CHART = save_chart(
        _chart,
        "persona_shift_by_read",
        caption=(
            f"Mean absolute shift of the 171-vector emotion read against {REFERENCE_LABEL}, per persona, at each "
            f"of the four reads: the tokens of the user's own message, the pre-response token and the reply mean "
            f"over the {N_PROMPTS} WildChat prompts, and the 3,420 held-out emotional stories read the way the "
            "vectors were built. Each read is standardized by the base model's per-vector spread over the same "
            "texts at the same position. Whiskers are 95% reverse-percentile intervals from 1,000 resamples of "
            "the texts; the black tick is the noise floor, what the statistic reads when two models are identical."
        ),
        takeaway=f"Over the user's own tokens against {REFERENCE_LABEL}, then at the pre-response token: {_lines}.",
        notebook=NOTEBOOK,
    )
    PERSONA_ABS_CHART
    return


@app.cell
def _(
    EMOTION_ORDER,
    FAMILIES,
    PERSONA_ORDER,
    POSITIONS,
    REFERENCE_LABEL,
    SHIFTS,
    alt,
    pl,
):
    def shift_chart(position: str):
        """One column per persona, one horizontal bar per emotion (taxonomy order), 95% whiskers."""
        _base = alt.Chart(SHIFTS.filter(pl.col("position") == position))
        _y = alt.Y(
            "emotion:N",
            sort=EMOTION_ORDER,
            title=None,
            axis=alt.Axis(labelFontSize=7, labelLimit=140),
        )
        _zero = _base.mark_rule(color="#9a9a9a", strokeWidth=1).encode(
            x=alt.datum(0)
        )
        _bars = _base.mark_bar(size=5).encode(
            y=_y,
            x=alt.X("shift:Q", title="mean shift (base sd units)"),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=FAMILIES, scheme="tableau10"),
                title="family",
            ),
            tooltip=[
                "persona:N",
                "emotion:N",
                "family:N",
                alt.Tooltip("shift:Q", format="+.2f", title="shift (base sd)"),
                alt.Tooltip("ci_lo:Q", format="+.2f", title="95% low"),
                alt.Tooltip("ci_hi:Q", format="+.2f", title="95% high"),
                alt.Tooltip(
                    "raw_shift:Q",
                    format="+.3f",
                    title="raw difference of means",
                ),
                alt.Tooltip("n:Q", title="prompts"),
            ],
        )
        _ci = _base.mark_rule(color="#333333", strokeWidth=1).encode(
            y=_y, x="ci_lo:Q", x2="ci_hi:Q"
        )
        return (
            alt.layer(_zero, _bars, _ci)
            .properties(width=170, height=len(EMOTION_ORDER) * 8)
            .facet(
                column=alt.Column(
                    "persona:N",
                    sort=PERSONA_ORDER,
                    title=None,
                    header=alt.Header(labelFontSize=13),
                )
            )
            .properties(
                title=f"Per-emotion shift on neutral dialogue vs {REFERENCE_LABEL}, {POSITIONS[position]}"
            )
        )

    def summarize(position: str) -> str:
        """One clause per persona: its largest upward and downward shift at this position."""
        _df = SHIFTS.filter(pl.col("position") == position)
        _parts = []
        for _p in PERSONA_ORDER:
            _d = _df.filter(pl.col("persona") == _p).sort("shift")
            _lo, _hi = _d.row(0, named=True), _d.row(-1, named=True)
            _n_big = int((_d["shift"].abs() >= 0.5).sum())
            _parts.append(
                f"{_p}: {_hi['emotion']} {_hi['shift']:+.2f}, {_lo['emotion']} {_lo['shift']:+.2f}, "
                f"{_n_big} of {_d.height} emotions past 0.5 sd"
            )
        return "; ".join(_parts)

    return shift_chart, summarize


@app.cell
def _(N_PROMPTS, REFERENCE_LABEL, mo):
    mo.md(f"""
    ### Every emotion, at the pre-response token

    **What the chart uses.** The {N_PROMPTS} WildChat prompts of the pool, each answered once
    by each of the seven persona checkpoints and by {REFERENCE_LABEL}, read at layer 21 at
    the pre-response token: the last token of the prompt, after the user's message and the
    assistant header, where the model is about to start writing but has written nothing yet.
    This is the position at which the emotion vectors were validated, and the one the paper
    this project follows reads to get the assistant's own side rather than the user's.

    **How the numbers were made.** Each of the 171 emotion vectors gives one bar. Its value
    is the mean over prompts of the difference between the persona's raw projection and
    {REFERENCE_LABEL}'s on the same prompt, divided by the untrained base model's standard
    deviation for that emotion over the same prompts at the same position; the whisker is
    1.96 standard errors of those paired per-prompt differences. Emotions are ordered and
    coloured by the ten taxonomy families. The raw difference, before the division, is in
    every tooltip.
    """)
    return


@app.cell
def _(
    NOTEBOOK,
    N_PROMPTS,
    REFERENCE_LABEL,
    save_chart,
    shift_chart,
    summarize,
):
    SHIFT_CHART_PRE = save_chart(
        shift_chart("pre_response"),
        "persona_emotion_shift_pre_response",
        caption=(
            f"Difference of mean projection between each persona model and {REFERENCE_LABEL} for every one "
            f"of the 171 emotions at the pre-response token, over {N_PROMPTS} WildChat prompts, in units of the "
            "base model's per-emotion standard deviation; whiskers are 95% intervals from the paired "
            "per-prompt differences; emotions ordered and colored by taxonomy family."
        ),
        takeaway=f"At the pre-response token, against {REFERENCE_LABEL}: {summarize('pre_response')}.",
        notebook=NOTEBOOK,
    )
    SHIFT_CHART_PRE
    return


@app.cell
def _(N_PROMPTS, REFERENCE_LABEL, mo):
    mo.md(f"""
    ### Every emotion, averaged over the model's own reply

    **What the chart uses.** The same {N_PROMPTS} prompts and the same models, but the
    activation is now the layer-21 residual averaged over the tokens of the reply that model
    actually wrote (the end-of-turn token excluded), so a persona is read on its own text
    rather than on a shared prompt. Replies were sampled once per prompt per model at the
    student settings, and are the stored ones; nothing was resampled for this figure.

    **How the numbers were made.** As in the previous figure: 171 vectors, the paired
    per-prompt difference of the raw projections against {REFERENCE_LABEL}, averaged over
    prompts and divided by the base model's per-emotion spread over the same prompts at this
    same position, with 1.96-standard-error whiskers. Because two models write different
    replies, a shift here mixes what the model is representing with what it chose to write,
    which is the sense in which this read is on-policy and the pre-response read is not.
    """)
    return


@app.cell
def _(
    NOTEBOOK,
    N_PROMPTS,
    REFERENCE_LABEL,
    save_chart,
    shift_chart,
    summarize,
):
    SHIFT_CHART_REPLY = save_chart(
        shift_chart("reply_mean"),
        "persona_emotion_shift_reply_mean",
        caption=(
            f"Difference of mean projection between each persona model and {REFERENCE_LABEL} for every one "
            f"of the 171 emotions, averaged over the model's own reply tokens, over {N_PROMPTS} WildChat "
            "prompts, in units of the base model's per-emotion standard deviation; whiskers are 95% "
            "intervals from the paired per-prompt differences; emotions ordered and colored by "
            "taxonomy family."
        ),
        takeaway=f"Averaged over the reply, against {REFERENCE_LABEL}: {summarize('reply_mean')}.",
        notebook=NOTEBOOK,
    )
    SHIFT_CHART_REPLY
    return


@app.cell
def _(N_PROMPTS, REFERENCE_LABEL, mo):
    mo.md(f"""
    ### The same shifts collected into families

    **What the chart uses.** The two figures above plus the user-message read, aggregated:
    the same {N_PROMPTS} WildChat prompts, the same seven personas against
    {REFERENCE_LABEL}, all three read positions at layer 21, the same 171 vectors.

    **How the numbers were made.** Each emotion's paired shift is formed exactly as above
    (mean over prompts of the difference in raw projection, divided by the base model's
    per-emotion spread over the same prompts at that position), and the bar is the plain
    average of those values over the emotions of one taxonomy family. Families are of very
    different sizes, from two emotions to forty-one, so a small family's bar rests on far
    fewer numbers than a large one's. Opposite-signed emotions inside a family cancel here,
    which is what the per-emotion figures above are for.
    """)
    return


@app.cell
def _(
    FAMILIES,
    NOTEBOOK,
    PERSONA_ORDER,
    POSITIONS,
    REFERENCE_LABEL,
    SHIFTS,
    alt,
    pl,
    save_chart,
):
    _fam = (
        SHIFTS.group_by("persona", "position_label", "family")
        .agg(
            pl.col("shift").mean().alias("mean_shift"),
            pl.col("emotion").count().alias("n_emotions"),
        )
        .sort("persona", "position_label", "family")
    )
    _chart = (
        alt.Chart(_fam)
        .mark_bar(size=9)
        .encode(
            y=alt.Y(
                "family:N",
                sort=FAMILIES,
                title=None,
                axis=alt.Axis(labelFontSize=9),
            ),
            x=alt.X("mean_shift:Q", title="mean shift (base sd)"),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=FAMILIES, scheme="tableau10"),
                legend=None,
            ),
            tooltip=[
                "persona:N",
                "position_label:N",
                "family:N",
                alt.Tooltip("mean_shift:Q", format="+.2f"),
                "n_emotions:Q",
            ],
        )
        .properties(width=195, height=150)
        .facet(
            row=alt.Row(
                "persona:N",
                sort=PERSONA_ORDER,
                title=None,
                header=alt.Header(labelFontSize=12),
            ),
            column=alt.Column(
                "position_label:N",
                sort=list(POSITIONS.values()),
                title=None,
                header=alt.Header(labelFontSize=12),
            ),
        )
        .properties(
            title=f"Family means of the per-emotion shift on neutral dialogue vs {REFERENCE_LABEL}"
        )
    )
    _summary = "; ".join(
        f"{_r['persona']} at the {_r['position_label']}: {_r['family']} {_r['mean_shift']:+.2f}"
        for _r in _fam.sort("mean_shift", descending=True)
        .group_by("persona", "position_label", maintain_order=True)
        .head(1)
        .sort("persona", "position_label")
        .iter_rows(named=True)
    )
    FAMILY_CHART = save_chart(
        _chart,
        "persona_family_mean_shift",
        caption=(
            f"Mean over each taxonomy family of the per-emotion shift vs {REFERENCE_LABEL}, per persona "
            "and read position, in base-model standard-deviation units (the family aggregation of the "
            "171-emotion figures above; families differ in size, from 2 to 41 emotions)."
        ),
        takeaway=f"Largest family mean per persona and position, against {REFERENCE_LABEL}: {_summary}.",
        notebook=NOTEBOOK,
    )
    FAMILY_CHART
    return


@app.cell
def _(N_PROMPTS, REFERENCE_LABEL, mo):
    mo.md(f"""
    ### The emotions that move most

    **What the chart uses.** The same paired shifts once more — {N_PROMPTS} WildChat
    prompts, seven personas against {REFERENCE_LABEL}, three read positions, layer 21, 171
    vectors — with only the extremes kept: per persona and position, the five emotions whose
    mean projection rose most and the five that fell most.

    **How the numbers were made.** The shift and its 95% interval are the ones described
    above (paired per-prompt difference of the raw projections, averaged, divided by the
    base model's per-emotion spread at that position, whiskers at 1.96 standard errors);
    the bars are simply sorted and cut at five each way. Because the selection is made on
    the same data that is plotted, the extremes are the most optimistic emotions rather than
    an unbiased sample, and the whole-set figures above are the honest summary.
    """)
    return


@app.cell
def _(
    FAMILIES,
    NOTEBOOK,
    N_PROMPTS,
    PERSONA_ORDER,
    POSITIONS,
    REFERENCE_LABEL,
    SHIFTS,
    alt,
    pl,
    save_chart,
):
    # Every persona side by side: its five largest upward and five largest downward shifts.
    _sorted = SHIFTS.sort(
        "persona", "position", "shift", descending=[False, False, True]
    )
    _grp = _sorted.group_by("persona", "position", maintain_order=True)
    # `rank` is the bar's position inside its panel (1 = largest shift). Ordering by the
    # emotion name would not work: with facets Vega-Lite ranks a name once over the whole
    # dataset, and the same emotion sits at different ranks in different panels.
    TOP_MOVERS = pl.concat([_grp.head(5), _grp.tail(5)]).with_columns(
        (pl.int_range(pl.len()).over("persona", "position") + 1).alias("rank"),
        pl.when(pl.col("shift") >= 0)
        .then(pl.lit("up"))
        .otherwise(pl.lit("down"))
        .alias("direction"),
    )
    _base = alt.Chart(TOP_MOVERS)
    _x = alt.X("rank:O", title=None, axis=None)
    _height = 200
    _panel = alt.layer(
        _base.mark_rule(color="#9a9a9a").encode(y=alt.datum(0)),
        # The emotion names, written under the panel's baseline in rank order.
        _base.mark_text(
            angle=315,
            align="right",
            baseline="top",
            dx=4,
            dy=6,
            fontSize=9,
            color="#333333",
        ).encode(x=_x, y=alt.value(_height), text="emotion:N"),
        _base.mark_bar(size=12).encode(
            x=_x,
            y=alt.Y("shift:Q", title="mean shift (base sd units)"),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=FAMILIES, scheme="tableau10"),
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    direction="horizontal",
                    columns=len(FAMILIES),
                    labelFontSize=10,
                ),
            ),
            tooltip=[
                "persona:N",
                "position_label:N",
                "emotion:N",
                "family:N",
                alt.Tooltip("shift:Q", format="+.2f"),
                alt.Tooltip("ci_lo:Q", format="+.2f"),
                alt.Tooltip("ci_hi:Q", format="+.2f"),
                alt.Tooltip("raw_shift:Q", format="+.3f"),
            ],
        ),
        _base.mark_rule(color="#333333").encode(
            x=_x, y="ci_lo:Q", y2="ci_hi:Q"
        ),
    ).properties(width=200, height=_height)
    _chart = _panel.facet(
        column=alt.Column(
            "persona:N",
            sort=PERSONA_ORDER,
            title=None,
            header=alt.Header(labelFontSize=13),
        ),
        row=alt.Row(
            "position_label:N",
            sort=list(POSITIONS.values()),
            title=None,
            header=alt.Header(labelFontSize=12),
        ),
        spacing={"row": 70},
    ).properties(
        title=f"Largest shifts on neutral dialogue vs {REFERENCE_LABEL}: five up and five down per persona",
        padding={"bottom": 70},
    )
    _lines = []
    for _p in PERSONA_ORDER:
        _d = TOP_MOVERS.filter(
            (pl.col("persona") == _p) & (pl.col("position") == "pre_response")
        ).sort("shift", descending=True)
        _lines.append(
            f"{_p}: {_d.row(0, named=True)['emotion']} {_d.row(0, named=True)['shift']:+.2f} up, "
            f"{_d.row(-1, named=True)['emotion']} {_d.row(-1, named=True)['shift']:+.2f} down"
        )
    TOP_MOVERS_CHART = save_chart(
        _chart,
        "persona_top_movers",
        caption=(
            f"For each persona and read position, the five emotions whose mean projection rose most and the "
            f"five that fell most relative to {REFERENCE_LABEL} over the {N_PROMPTS} WildChat prompts, in base-model "
            "standard-deviation units with 95% paired intervals; colored by taxonomy family."
        ),
        takeaway=f"Largest movers at the pre-response token, against {REFERENCE_LABEL}: {'; '.join(_lines)}.",
        notebook=NOTEBOOK,
    )
    TOP_MOVERS_CHART
    return


@app.cell
def _(PERSONA_ORDER, POSITIONS, mo):
    persona_pick = mo.ui.dropdown(
        options={p: p for p in PERSONA_ORDER},
        value=PERSONA_ORDER[0],
        label="persona",
    )
    position_pick = mo.ui.radio(
        options={v: k for k, v in POSITIONS.items()},
        value=POSITIONS["pre_response"],
        label="position",
    )
    mo.vstack(
        [
            mo.md(
                "### Explore one persona\n\nEvery emotion for one persona and one position, sorted by its "
                "shift, with the table of the same rows below. Pick from the controls; nothing here is saved."
            ),
            mo.hstack([persona_pick, position_pick], justify="start", gap=2),
        ]
    )
    return persona_pick, position_pick


@app.cell
def _(FAMILIES, SHIFTS, alt, persona_pick, pl, position_pick):
    # Instrument (never saved): one persona, every emotion sorted by its shift, with the intervals.
    SORTED_DF = SHIFTS.filter(
        (pl.col("persona") == persona_pick.value)
        & (pl.col("position") == position_pick.value)
    ).sort("shift", descending=True)
    _order = SORTED_DF["emotion"].to_list()
    _base = alt.Chart(SORTED_DF)
    _x = alt.X(
        "emotion:N",
        sort=_order,
        title=None,
        axis=alt.Axis(labelFontSize=8, labelAngle=-90, labelLimit=110),
    )
    SORTED_CHART = alt.layer(
        _base.mark_rule(color="#9a9a9a").encode(y=alt.datum(0)),
        _base.mark_bar(size=5).encode(
            x=_x,
            y=alt.Y("shift:Q", title="mean shift (base sd units)"),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=FAMILIES, scheme="tableau10"),
                # The legend runs along the top in one row so the plot keeps the full width.
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    direction="horizontal",
                    columns=len(FAMILIES),
                    labelFontSize=10,
                ),
            ),
            tooltip=[
                "emotion:N",
                "family:N",
                alt.Tooltip("shift:Q", format="+.2f"),
                alt.Tooltip("ci_lo:Q", format="+.2f"),
                alt.Tooltip("ci_hi:Q", format="+.2f"),
                alt.Tooltip("raw_shift:Q", format="+.3f"),
            ],
        ),
        _base.mark_rule(color="#333333").encode(
            x=_x, y="ci_lo:Q", y2="ci_hi:Q"
        ),
    ).properties(
        width=len(_order) * 7,
        height=320,
        title=f"{persona_pick.value} on neutral dialogue, {position_pick.value.replace('_', ' ')}: every emotion, sorted by shift",
    )
    SORTED_CHART
    return


@app.cell
def _(REFERENCE_LABEL, mo):
    mo.md(f"""
    ## Part 3: valence, arousal and dominance

    Sofroniew et al. (section 2.1.2) run a principal component analysis over the emotion
    vectors themselves and find valence on the first component and arousal on the second or
    third, validated against human ratings. `project.py` does the same on the paper-corpus
    vectors (the paper's unnormalized form, with the L2-normalized units as a check) and
    assigns each of the three affect dimensions with published word norms (valence, arousal,
    dominance; Warriner 2013) to the leading component its norms correlate with best, signs
    set so the correlation is positive. The fitted axes and their per-emotion scores are in
    `data/vectors/affect_axes.json`. Three numbers per activation replace 171, which is what
    makes a map possible.

    The map below shows the models alone, on {REFERENCE_LABEL}'s origin. The emotion vectors
    are not drawn on it, and this is deliberate: model-against-model and
    emotion-against-emotion comparisons are exact, but a chat activation's position *among*
    the emotion landmarks is not, because the emotions' coordinates come from story text read
    as raw prose and the models' from chat transcripts read at one token position, and the
    two conventions sit a measured distance apart (Part 4 quotes it). The plane here is
    therefore a compass for direction and order — who is more pleasant than whom, and by how
    much — with the emotion names appearing only as labels on the ends of each axis to say
    which way is which. Part 4 has the map where models and emotions do share a convention.
    """)
    return


@app.cell
def _(
    DATA,
    EMO2FAM,
    MODELS,
    MODEL_LABEL,
    PERSONAS,
    PERSONA_LABEL,
    POSITIONS,
    READOUTS,
    REFERENCE,
    REFERENCES,
    json,
    load_file,
    np,
    pl,
):
    AXES = json.loads(
        (DATA / "vectors" / "affect_axes.json").read_text(encoding="utf-8")
    )
    _fit = AXES["fits"][AXES["primary"]]
    DIMENSIONS = [d for d in ("valence", "arousal", "dominance") if d in _fit]
    AXIS_LABEL = {
        d: f"{d} (PC{_fit[d]['component']}, r = {_fit[d]['r_norms']:.2f} with human norms)"
        for d in DIMENSIONS
    }
    # The two or three emotions at each end of each fitted axis, the only role emotion names
    # play on the models-only map: they say which way the axis points.
    ENDS = {
        d: {
            "low": [
                e
                for e, _ in sorted(
                    _fit[d]["scores"].items(), key=lambda kv: kv[1]
                )[:3]
            ],
            "high": [
                e
                for e, _ in sorted(
                    _fit[d]["scores"].items(), key=lambda kv: -kv[1]
                )[:3]
            ],
        }
        for d in DIMENSIONS
    }

    def _affect(model: str, pos: str, d: str) -> dict[str, float]:
        return {
            m["id"]: m[pos]["affect"][d]["raw"]
            for m in READOUTS[model]["messages"]
        }

    # The vectors' own origin: the average emotional story, projected onto each axis. The
    # story-read map in Part 4 uses it; the models-only map re-centres on the control.
    _bundle = load_file(str(DATA / "vectors" / "units.safetensors"))
    _axes_t = load_file(str(DATA / "vectors" / "affect_axes.safetensors"))
    _axis = {
        d: _axes_t[f"{d}_direction"].astype(np.float64) for d in DIMENSIONS
    }
    OFFSET = {
        d: float(_bundle["story_grand_mean"].astype(np.float64) @ _axis[d])
        for d in DIMENSIONS
    }
    # Every emotion (its vector's scores) and every model (mean projection minus the offset,
    # with the per-prompt spread), for all three positions; one frame with every column
    # present on every row.
    _rows = []
    for _plabel in POSITIONS.values():
        for _e in AXES["emotions"]:
            _rows.append(
                {
                    "kind": "emotion",
                    "name": _e,
                    "family": EMO2FAM.get(_e, "?"),
                    "position_label": _plabel,
                    **{d: float(_fit[d]["scores"][_e]) for d in DIMENSIONS},
                    **{
                        f"{d}_{s}": None
                        for d in DIMENSIONS
                        for s in ("lo", "hi", "sd")
                    },
                    "n": None,
                }
            )
    for _pos, _plabel in POSITIONS.items():
        for _m in MODELS:
            _vals = {
                d: np.array(list(_affect(_m, _pos, d).values())) - OFFSET[d]
                for d in DIMENSIONS
            }
            _row = {
                "kind": "model",
                "name": MODEL_LABEL[_m],
                "family": "model",
                "position_label": _plabel,
                "n": int(len(_vals["valence"])),
            }
            for d in DIMENSIONS:
                _mean, _sd = (
                    float(_vals[d].mean()),
                    float(_vals[d].std(ddof=1)),
                )
                _row.update(
                    {
                        d: _mean,
                        f"{d}_lo": _mean - _sd,
                        f"{d}_hi": _mean + _sd,
                        f"{d}_sd": _sd,
                    }
                )
            _rows.append(_row)
    AFFECT_MAP = pl.DataFrame(
        _rows, infer_schema_length=None
    )  # the emotion rows lead with nulls in the bar columns

    # The same model rows re-centred on the control: every coordinate minus the control's
    # mean at that position, so the control sits at zero and a dot is a paired difference
    # from it. The spread bars are the model's own per-prompt spread, unchanged by the shift
    # of origin.
    _model_rows = AFFECT_MAP.filter(pl.col("kind") == "model")
    _ref_at = {
        (r["position_label"], d): r[d]
        for r in _model_rows.filter(
            pl.col("name") == MODEL_LABEL[REFERENCE]
        ).iter_rows(named=True)
        for d in DIMENSIONS
    }
    MODEL_MAP = _model_rows.with_columns(
        [
            (
                pl.col(d)
                - pl.col("position_label").replace_strict(
                    {p: _ref_at[(p, d)] for p in POSITIONS.values()},
                    return_dtype=pl.Float64,
                )
            ).alias(d)
            for d in DIMENSIONS
        ]
    ).with_columns(
        [(pl.col(d) - pl.col(f"{d}_sd")).alias(f"{d}_lo") for d in DIMENSIONS]
        + [
            (pl.col(d) + pl.col(f"{d}_sd")).alias(f"{d}_hi")
            for d in DIMENSIONS
        ]
    )

    # Every model's per-prompt values on each axis, on the map's origin, for the distributions.
    AFFECT_VALUES = pl.DataFrame(
        [
            {
                "model": MODEL_LABEL[_m],
                "position": _pos,
                "position_label": _plabel,
                "dimension": d,
                "id": _id,
                "value": float(v - OFFSET[d]),
            }
            for _pos, _plabel in POSITIONS.items()
            for _m in MODELS
            for d in DIMENSIONS
            for _id, v in _affect(_m, _pos, d).items()
        ]
    )

    # Paired shifts on each axis, in base-sd units (as in Part 2), against each of the
    # three references: moodless (control) first, then neutral (no-wrapper control) and the
    # untrained base, so a mood's own contribution can be read next to the distillation's.
    _records = []
    for _pos in POSITIONS:
        for _d in DIMENSIONS:
            _base = _affect("base", _pos, _d)
            _sd = float(np.std(list(_base.values()))) or 1.0
            for _refname in REFERENCES:
                _ref = _affect(_refname, _pos, _d)
                for _model in PERSONAS:
                    _per = _affect(_model, _pos, _d)
                    _delta = np.array(
                        [_per[i] - _ref[i] for i in _ref if i in _per]
                    )
                    _mean, _se = (
                        _delta.mean(),
                        _delta.std(ddof=1) / np.sqrt(len(_delta)),
                    )
                    _records.append(
                        {
                            "persona": PERSONA_LABEL[_model],
                            "reference": _refname,
                            "reference_label": f"vs {MODEL_LABEL[_refname]}",
                            "position": _pos,
                            "position_label": POSITIONS[_pos],
                            "dimension": _d,
                            "shift": float(_mean / _sd),
                            "ci_lo": float((_mean - 1.96 * _se) / _sd),
                            "ci_hi": float((_mean + 1.96 * _se) / _sd),
                            "raw_shift": float(_mean),
                            "n": int(len(_delta)),
                        }
                    )
    AFFECT_SHIFTS = pl.DataFrame(_records)
    return (
        AFFECT_SHIFTS,
        AFFECT_VALUES,
        AXES,
        AXIS_LABEL,
        DIMENSIONS,
        ENDS,
        MODEL_MAP,
        OFFSET,
    )


@app.cell
def _(N_PROMPTS, REFERENCE_LABEL, mo):
    mo.md(f"""
    ### The models on the valence-arousal plane, with {REFERENCE_LABEL} at the origin

    **What the chart uses.** The {N_PROMPTS} WildChat prompts of the pool with each model's
    own stored reply, read at layer 21 at all three positions (one panel each). Ten models:
    the untrained base, the two controls and the seven persona checkpoints.

    **How the numbers were made.** Each activation is projected onto the two fitted axes —
    unit directions in the model's activation space, obtained by principal component
    analysis of the 171 emotion vectors and matched to human valence and arousal ratings —
    giving two numbers per prompt. Those are averaged over the prompts, and then
    {REFERENCE_LABEL}'s average at the same position is subtracted, so the control sits
    exactly at the origin and every other dot is that model's paired difference from it in
    the axes' own units. The bars span one standard deviation of the model's own per-prompt
    values on each axis, which measures how much the model's read moves from prompt to
    prompt and is unaffected by the change of origin.

    **What the axis labels mean.** The emotion vectors are not drawn, because their
    coordinates come from story text read as raw prose while these come from chat
    transcripts read at one token position; instead the names at each end of an axis are the
    three emotion vectors with the most extreme scores on it, so they say which direction is
    which without inviting a distance to be read off. Dominance, the third axis, is in the
    strip below.
    """)
    return


@app.cell
def _(
    AXIS_LABEL,
    CONTROLS,
    ENDS,
    MODELS,
    MODEL_LABEL,
    MODEL_MAP,
    NOTEBOOK,
    N_PROMPTS,
    PALETTE,
    POSITIONS,
    REFERENCE_LABEL,
    alt,
    pl,
    save_chart,
):
    _order = [MODEL_LABEL[m] for m in MODELS]
    # Controls and the untrained base are diamonds, persona checkpoints dots (Carolina,
    # 2026-09-10), so the references can be told from the moods without the legend.
    _control_names = {"base", *[MODEL_LABEL[c] for c in CONTROLS]}
    _color = alt.Color(
        "name:N",
        sort=_order,
        scale=alt.Scale(domain=_order, range=PALETTE[: len(_order)]),
        legend=alt.Legend(
            title=None, orient="right", labelFontSize=11, symbolSize=150
        ),
    )
    # One domain per panel (Carolina, 2026-09-10: the checkpoints were "smushed together"
    # on a domain shared across the three positions and stretched by the sd bars): each
    # panel's axes span its own model means, padded, always including the origin, and the
    # bars are drawn clipped to that window. Labels are placed in pixels per panel.
    _W, _H = 430, 400

    def _domain(values, zero_pad: float) -> tuple[float, float]:
        _lo, _hi = min(float(values.min()), 0.0), max(float(values.max()), 0.0)
        _r = max(_hi - _lo, zero_pad)
        return _lo - 0.22 * _r, _hi + 0.22 * _r

    _DOM = {}
    for _plabel in POSITIONS.values():
        _panel = MODEL_MAP.filter(pl.col("position_label") == _plabel)
        _xd, _yd = _domain(_panel["valence"], 4), _domain(_panel["arousal"], 4)
        _DOM[_plabel] = (
            _xd,
            _yd,
            (_W / (_xd[1] - _xd[0]), _H / (_yd[1] - _yd[0])),
        )

    def _place_labels(frame: pl.DataFrame) -> pl.DataFrame:
        """Greedy label placement per panel: try a ring of offsets around each dot and keep
        the first whose text box overlaps neither an earlier label nor another dot."""
        _cands = [
            (14, 0, "left"),
            (14, -14, "left"),
            (14, 14, "left"),
            (-14, 0, "right"),
            (-14, -14, "right"),
            (-14, 14, "right"),
            (0, -18, "center"),
            (0, 18, "center"),
            (28, -28, "left"),
            (28, 28, "left"),
            (-28, -28, "right"),
            (-28, 28, "right"),
            (0, 32, "center"),
        ]
        out = []
        for _plabel in frame["position_label"].unique(maintain_order=True):
            _panel = frame.filter(pl.col("position_label") == _plabel)
            _XD, _YD, _px = _DOM[_plabel]
            _dots = [
                (
                    (r["valence"] - _XD[0]) * _px[0],
                    (_YD[1] - r["arousal"]) * _px[1],
                )
                for r in _panel.iter_rows(named=True)
            ]
            _boxes: list = []
            for _i, r in enumerate(_panel.iter_rows(named=True)):
                _text = r["name"]
                _w, _h = 7.0 * len(_text), 13
                _cx, _cy = _dots[_i]
                _choice, _box = _cands[-1], None
                for _dx, _dy, _align in _cands:
                    _ax, _ay = _cx + _dx, _cy + _dy
                    _x0 = (
                        _ax
                        if _align == "left"
                        else _ax - _w
                        if _align == "right"
                        else _ax - _w / 2
                    )
                    _try = (_x0, _ay - _h / 2, _x0 + _w, _ay + _h / 2)
                    _hit = any(
                        not (
                            _try[2] < b[0]
                            or _try[0] > b[2]
                            or _try[3] < b[1]
                            or _try[1] > b[3]
                        )
                        for b in _boxes
                    )
                    _hit = _hit or any(
                        _k != _i
                        and _try[0] - 8 < dx < _try[2] + 8
                        and _try[1] - 8 < dy < _try[3] + 8
                        for _k, (dx, dy) in enumerate(_dots)
                    )
                    if not _hit:
                        _choice, _box = (_dx, _dy, _align), _try
                        break
                _boxes.append(_box or (_cx, _cy, _cx + _w, _cy + _h))
                _dx, _dy, _align = _choice
                out.append(
                    {
                        **r,
                        "label": _text,
                        "label_align": _align,
                        "label_x": r["valence"] + _dx / _px[0],
                        "label_y": r["arousal"] - _dy / _px[1],
                        "leader": _dx != 14 or _dy != 0,
                    }
                )
        return pl.DataFrame(out)

    _placed = _place_labels(MODEL_MAP).with_columns(
        pl.when(pl.col("name").is_in(list(_control_names)))
        .then(pl.lit("control or base"))
        .otherwise(pl.lit("persona checkpoint"))
        .alias("marker")
    )

    # The direction labels: the three most extreme emotion vectors at each end of each axis.
    # They ride in the same frame as the models, because a faceted layer takes one dataset.
    def _end_row(
        _p: str, _x: float, _y: float, _align: str, _text: str
    ) -> dict:
        return {
            "kind": "end",
            "name": _text,
            "position_label": _p,
            "valence": _x,
            "arousal": _y,
            "label": _text,
            "label_align": _align,
            "label_x": _x,
            "label_y": _y,
            "leader": False,
        }

    _end_rows = []
    for _p, (_XD, _YD, _) in _DOM.items():
        _xr, _yr = _XD[1] - _XD[0], _YD[1] - _YD[0]
        _end_rows += [
            _end_row(
                _p,
                _XD[0] + 0.02 * _xr,
                _YD[0] + 0.04 * _yr,
                "left",
                "toward " + ", ".join(ENDS["valence"]["low"]),
            ),
            _end_row(
                _p,
                _XD[1] - 0.02 * _xr,
                _YD[0] + 0.04 * _yr,
                "right",
                "toward " + ", ".join(ENDS["valence"]["high"]),
            ),
            _end_row(
                _p,
                _XD[0] + 0.02 * _xr,
                _YD[1] - 0.04 * _yr,
                "left",
                "up: toward " + ", ".join(ENDS["arousal"]["high"]),
            ),
            _end_row(
                _p,
                _XD[0] + 0.02 * _xr,
                _YD[0] + 0.12 * _yr,
                "left",
                "down: toward " + ", ".join(ENDS["arousal"]["low"]),
            ),
        ]
    _ends = pl.DataFrame(_end_rows)
    _frame = pl.concat([_placed, _ends], how="diagonal_relaxed")
    _is_model = alt.datum.kind == "model"
    _is_end = alt.datum.kind == "end"
    _shape = alt.Shape(
        "marker:N",
        scale=alt.Scale(
            domain=["persona checkpoint", "control or base"],
            range=["circle", "diamond"],
        ),
        legend=alt.Legend(
            title=None,
            orient="right",
            labelFontSize=11,
            symbolSize=150,
            symbolFillColor="#777777",
        ),
    )

    def _panel_chart(_plabel: str):
        """One read position on its own axes; the three are concatenated below."""
        _XD, _YD, _ = _DOM[_plabel]
        _x = alt.X(
            "valence:Q",
            title=AXIS_LABEL["valence"] + f", {REFERENCE_LABEL} at zero",
            scale=alt.Scale(domain=list(_XD), nice=False),
        )
        _y = alt.Y(
            "arousal:Q",
            title=AXIS_LABEL["arousal"] + f", {REFERENCE_LABEL} at zero",
            scale=alt.Scale(domain=list(_YD), nice=False),
        )
        _src = alt.Chart(_frame.filter(pl.col("position_label") == _plabel))
        _zero_x = (
            _src.transform_filter(_is_model)
            .mark_rule(color="#888888")
            .encode(x=alt.datum(0))
        )
        _zero_y = (
            _src.transform_filter(_is_model)
            .mark_rule(color="#888888")
            .encode(y=alt.datum(0))
        )
        # The ±1 sd spread bars were drawn here until 2026-09-10 (Carolina: "just remove the
        # spread lines"); the per-prompt spread stays in the tooltip.
        _dots = (
            _src.transform_filter(_is_model)
            .mark_point(
                size=170,
                filled=True,
                stroke="#222222",
                strokeWidth=1,
                opacity=1,
            )
            .encode(
                x=_x,
                y=_y,
                color=_color,
                shape=_shape,
                tooltip=[
                    "name:N",
                    "position_label:N",
                    alt.Tooltip(
                        "valence:Q",
                        format="+.2f",
                        title=f"valence vs {REFERENCE_LABEL}",
                    ),
                    alt.Tooltip(
                        "valence_sd:Q",
                        format=".2f",
                        title="valence sd over prompts",
                    ),
                    alt.Tooltip(
                        "arousal:Q",
                        format="+.2f",
                        title=f"arousal vs {REFERENCE_LABEL}",
                    ),
                    alt.Tooltip(
                        "arousal_sd:Q",
                        format=".2f",
                        title="arousal sd over prompts",
                    ),
                    alt.Tooltip(
                        "dominance:Q",
                        format="+.2f",
                        title=f"dominance vs {REFERENCE_LABEL}",
                    ),
                    alt.Tooltip("n:Q", title="prompts"),
                ],
            )
        )
        _leaders = (
            _src.transform_filter(_is_model & (alt.datum.leader == True))  # noqa: E712
            .mark_rule(color="#555555", strokeWidth=0.7)
            .encode(x=_x, y=_y, x2="label_x:Q", y2="label_y:Q")
        )
        _labels = [
            _src.transform_filter(_is_model & (alt.datum.label_align == _a))
            .mark_text(
                fontSize=10,
                fontWeight="bold",
                align=_a,
                baseline="middle",
                color="#111111",
            )
            .encode(x="label_x:Q", y="label_y:Q", text="label:N")
            for _a in ("left", "right", "center")
        ]
        _direction_labels = [
            _src.transform_filter(_is_end & (alt.datum.label_align == _a))
            .mark_text(
                fontSize=9, align=_a, baseline="middle", color="#666666"
            )
            .encode(x=_x, y=_y, text="label:N")
            for _a in ("left", "right")
        ]
        return alt.layer(
            _zero_x,
            _zero_y,
            _dots,
            _leaders,
            *_labels,
            *_direction_labels,
        ).properties(
            width=_W,
            height=_H,
            title=alt.Title(_plabel, fontSize=13, anchor="middle"),
        )

    _chart = (
        alt.hconcat(
            *[_panel_chart(_p) for _p in POSITIONS.values()], spacing=24
        )
        .resolve_scale(
            x="independent", y="independent", color="shared", shape="shared"
        )
        .properties(
            title=alt.Title(
                "Valence-arousal plane on neutral dialogue",
                subtitle=(
                    f"{REFERENCE_LABEL} at the origin; each read position on its own axes; "
                    "dots are persona checkpoints, diamonds the controls and base"
                ),
                fontSize=15,
                subtitleFontSize=11,
                subtitleColor="#555",
                anchor="start",
            )
        )
        .configure_view(fill="#eaeaf2", stroke=None)
        .configure_axis(
            grid=True,
            gridColor="#ffffff",
            gridWidth=1,
            domain=False,
            tickColor="#ffffff",
        )
    )
    _pre = _placed.filter(
        pl.col("position_label") == POSITIONS["pre_response"]
    )
    _summary = "; ".join(
        f"{r['name']} valence {r['valence']:+.2f}, arousal {r['arousal']:+.2f}"
        for r in _pre.iter_rows(named=True)
    )
    AFFECT_MAP_CHART = save_chart(
        _chart,
        "persona_affect_map",
        caption=(
            f"Every model on the fitted valence and arousal axes at the three read positions, with "
            f"{REFERENCE_LABEL} at the origin: a mark is the model's mean projection over the {N_PROMPTS} WildChat "
            f"prompts minus {REFERENCE_LABEL}'s mean at the same position, in the axes' own units (dots for the "
            "persona checkpoints, diamonds for the controls and the untrained base), the standard deviation of "
            "the model's per-prompt values on each axis is in the tooltip. Each panel has its own axis "
            "range, set by its model means. The emotion vectors are "
            "not drawn, because they are read from story text and the models from chat transcripts, two "
            "conventions a measured distance apart (Part 4); the names at the ends of each axis are the three "
            "most extreme emotion vectors on it and mark direction only."
        ),
        takeaway=f"At the pre-response token, against {REFERENCE_LABEL}: {_summary}.",
        notebook=NOTEBOOK,
    )
    AFFECT_MAP_CHART
    return


@app.cell
def _(N_PROMPTS, REFERENCE_LABEL, mo):
    mo.md(f"""
    ### The third axis, dominance

    **What the chart uses.** The same {N_PROMPTS} prompts, the same ten models, the same
    three read positions at layer 21, projected onto the third fitted axis, the one whose
    per-emotion scores match published human dominance ratings best.

    **How the numbers were made.** As on the map: the mean over prompts of the projection
    onto the dominance direction, minus {REFERENCE_LABEL}'s mean at the same position, so
    the control sits at zero; the bar spans one standard deviation of that model's own
    per-prompt values. Emotion landmarks are left off for the same reason as on the map,
    and more strongly here, since this is the axis on which the two reading conventions sit
    furthest apart.
    """)
    return


@app.cell
def _(
    AXIS_LABEL,
    MODELS,
    MODEL_LABEL,
    MODEL_MAP,
    NOTEBOOK,
    N_PROMPTS,
    PALETTE,
    POSITIONS,
    REFERENCE_LABEL,
    alt,
    pl,
    save_chart,
):
    _order = [MODEL_LABEL[m] for m in MODELS]
    _color = alt.Color(
        "name:N",
        scale=alt.Scale(domain=_order, range=PALETTE[: len(_order)]),
        legend=None,
    )
    _y = alt.Y(
        "name:N", sort=_order, title=None, axis=alt.Axis(labelFontSize=11)
    )
    _x = alt.X(
        "dominance:Q",
        title=AXIS_LABEL["dominance"].split(", r")[0]
        + f"), {REFERENCE_LABEL} at zero",
    )
    _base = alt.Chart()
    _bars = _base.mark_rule(strokeWidth=2).encode(
        x="dominance_lo:Q", x2="dominance_hi:Q", y=_y, color=_color
    )
    _pts = _base.mark_point(
        shape="diamond",
        size=200,
        filled=True,
        stroke="#222222",
        strokeWidth=1,
        opacity=1,
    ).encode(
        x=_x,
        y=_y,
        color=_color,
        tooltip=[
            "name:N",
            "position_label:N",
            alt.Tooltip("dominance:Q", format="+.2f"),
            alt.Tooltip(
                "dominance_sd:Q", format=".2f", title="sd over prompts"
            ),
            alt.Tooltip("n:Q", title="prompts"),
        ],
    )
    _zero = _base.mark_rule(color="#888888").encode(x=alt.datum(0))
    _chart = (
        alt.layer(_zero, _bars, _pts, data=MODEL_MAP)
        .properties(width=330, height=190)
        .facet(
            column=alt.Column(
                "position_label:N",
                sort=list(POSITIONS.values()),
                title=None,
                header=alt.Header(labelFontSize=13),
            )
        )
        .properties(
            title=f"Dominance on neutral dialogue, against {REFERENCE_LABEL} (bars = ±1 sd over prompts)"
        )
        .configure_view(fill="#eaeaf2", stroke=None)
        .configure_axis(
            grid=True,
            gridColor="#ffffff",
            gridWidth=1,
            domain=False,
            tickColor="#ffffff",
        )
    )
    _pre = MODEL_MAP.filter(
        pl.col("position_label") == POSITIONS["pre_response"]
    )
    _summary = "; ".join(
        f"{r['name']} {r['dominance']:+.2f} (sd {r['dominance_sd']:.2f})"
        for r in _pre.iter_rows(named=True)
    )
    DOMINANCE_STRIP_CHART = save_chart(
        _chart,
        "persona_dominance_strip",
        caption=(
            "The models' dominance coordinate (the component of the emotion-vector set that human dominance "
            f"norms correlate with best) at the three read positions, with {REFERENCE_LABEL} at zero: the diamond "
            f"is the model's mean over the {N_PROMPTS} WildChat prompts minus the control's mean at the same "
            "position, and the bar spans one standard deviation of the per-prompt values. The emotions are left "
            "off this axis, as on the map above, because chat activations and story text are read by different "
            "conventions."
        ),
        takeaway=f"At the pre-response token, against {REFERENCE_LABEL}: {_summary}.",
        notebook=NOTEBOOK,
    )
    DOMINANCE_STRIP_CHART
    return


@app.cell
def _(N_PROMPTS, REFERENCE_LABEL, mo):
    mo.md(f"""
    ### The whole distribution behind each of those means

    **What the chart uses.** The same {N_PROMPTS} per-prompt values that the map and the
    strip average, kept unaveraged: one histogram per model and axis, at the pre-response
    token and at the reply mean. The user-message read is left out here to keep the figure
    at a readable width; how much it moves at all is the first figure of Part 2.

    **How the numbers were made.** Each prompt contributes one projection of that model's
    activation onto the axis, on the vectors' own origin (the average emotional story) rather
    than on the control's, so the histograms sit where the raw read puts them. The solid
    line is the model's own mean and the dashed line {REFERENCE_LABEL}'s, so the gap between
    the two lines is the shift the other figures report. Axes are shared down each column,
    so widths and positions compare across models.
    """)
    return


@app.cell
def _(
    AFFECT_VALUES,
    DIMENSIONS,
    MODELS,
    MODEL_LABEL,
    NOTEBOOK,
    N_PROMPTS,
    PALETTE,
    POSITIONS,
    REFERENCE,
    alt,
    np,
    pl,
    save_chart,
):
    # Small multiples: one row per model, one column per (position, axis) pair, each cell the
    # histogram of that model's per-prompt values, with a solid line at the model's mean and
    # a dashed line at the reference model's.
    _order = [MODEL_LABEL[m] for m in MODELS]
    _color = alt.Color(
        "model:N",
        scale=alt.Scale(domain=_order, range=PALETTE[: len(_order)]),
        legend=None,
    )
    _ref_label = MODEL_LABEL[REFERENCE]
    _shown = ["pre_response", "reply_mean"]

    def _column(pos: str, dim: str):
        _df = AFFECT_VALUES.filter(
            (pl.col("position") == pos) & (pl.col("dimension") == dim)
        )
        _lo, _hi = float(_df["value"].min()), float(_df["value"].max())
        _pad = 0.02 * (_hi - _lo)
        _lo, _hi = _lo - _pad, _hi + _pad
        _ref_mean = float(
            _df.filter(pl.col("model") == _ref_label)["value"].mean()
        )
        _base = alt.Chart()
        _bars = _base.mark_bar(opacity=0.85).encode(
            x=alt.X(
                "value:Q",
                bin=alt.Bin(extent=[_lo, _hi], step=(_hi - _lo) / 24),
                title=f"{dim}, {POSITIONS[pos]}",
            ),
            y=alt.Y(
                "count():Q",
                title=None,
                axis=alt.Axis(labelFontSize=8, tickCount=3),
            ),
            color=_color,
            tooltip=["model:N", alt.Tooltip("count():Q", title="prompts")],
        )
        _mean = _base.mark_rule(color="#111111", strokeWidth=1.5).encode(
            x="mean(value):Q"
        )
        _ref = _base.mark_rule(
            color="#333333", strokeWidth=1, strokeDash=[4, 3]
        ).encode(x=alt.datum(_ref_mean))
        return (
            alt.layer(_bars, _ref, _mean, data=_df)
            .properties(width=190, height=52)
            .facet(
                row=alt.Row(
                    "model:N",
                    sort=_order,
                    title=None,
                    header=alt.Header(
                        labelFontSize=10, labelAngle=0, labelAlign="left"
                    ),
                )
            )
            .resolve_scale(y="shared")
        )

    _chart = (
        alt.hconcat(
            *[_column(_p, _d) for _p in _shown for _d in DIMENSIONS],
            spacing=14,
        )
        .properties(
            title=f"Distributions on the three axes over neutral dialogues (solid = model mean, dashed = {_ref_label} mean)"
        )
        .configure_view(fill="#eaeaf2", stroke=None)
        .configure_axis(
            grid=True,
            gridColor="#ffffff",
            gridWidth=1,
            domain=False,
            tickColor="#ffffff",
        )
    )
    _parts = []
    for _p in _shown:
        for _d in DIMENSIONS:
            _df = AFFECT_VALUES.filter(
                (pl.col("position") == _p) & (pl.col("dimension") == _d)
            )
            _sd = {
                m: float(
                    np.std(
                        _df.filter(pl.col("model") == m)["value"].to_numpy(),
                        ddof=1,
                    )
                )
                for m in _order
            }
            _w, _n = max(_sd, key=_sd.get), min(_sd, key=_sd.get)
            _parts.append(
                f"{_d} at the {POSITIONS[_p]}: widest {_w} (sd {_sd[_w]:.2f}), narrowest {_n} (sd {_sd[_n]:.2f})"
            )
    AFFECT_DIST = save_chart(
        _chart,
        "persona_affect_distributions",
        caption=(
            f"For every model (rows) and each fitted axis at each of the two later read positions (columns), the "
            f"histogram of its {N_PROMPTS} per-prompt projections, on the vectors' own origin (the average "
            f"emotional story); the solid line is the model's mean and the dashed line {_ref_label}'s. Axes are "
            "shared down each column so shapes and positions compare across models. The user-message read is "
            "not shown here; how far it moves at all is the first figure of Part 2."
        ),
        takeaway=f"Per-prompt spread: {'; '.join(_parts)}.",
        notebook=NOTEBOOK,
    )
    AFFECT_DIST
    return


@app.cell
def _(N_PROMPTS, REFERENCE_ORDER, mo):
    mo.md(f"""
    ### The affect shift against the control

    **What the chart uses.** The {N_PROMPTS} WildChat prompts, the seven persona
    checkpoints, all three read positions at layer 21, and the three fitted axes. Each
    persona is compared with {REFERENCE_ORDER[0].removeprefix("vs ")} (the same shift against
    the other references stays in `summary.json`'s `models_vs`; the chart showed one bar per
    reference until 2026-09-10, when Carolina asked for the single control).

    **How the numbers were made.** For one persona, axis, position and reference, the two
    models' projections onto that axis are subtracted prompt by prompt, averaged, and
    divided by the untrained base model's standard deviation on that axis over the same
    prompts at the same position; the whisker is 1.96 standard errors of the paired
    differences. The difference against the control is the mood alone: what the recipe
    installs on its own is read in Part 1, where the controls are set against the untrained
    base.
    """)
    return


@app.cell
def _(
    AFFECT_SHIFTS,
    DIMENSIONS,
    NOTEBOOK,
    N_PROMPTS,
    PERSONA_ORDER,
    POSITIONS,
    REFERENCE_LABEL,
    REFERENCE_ORDER,
    alt,
    pl,
    save_chart,
):
    # The paired difference against the control only (Carolina, 2026-09-10; one bar per
    # reference until then). The other references' bars are still in AFFECT_SHIFTS.
    # Dominance left out of this exhibit (Carolina, 2026-09-10); it keeps its own strip above.
    _dims = [d for d in DIMENSIONS if d != "dominance"]
    _base = alt.Chart(
        AFFECT_SHIFTS.filter(
            (pl.col("reference_label") == REFERENCE_ORDER[0]) & pl.col("dimension").is_in(_dims)
        )
    )
    _y = alt.Y(
        "persona:N",
        sort=PERSONA_ORDER,
        title=None,
        axis=alt.Axis(labelFontSize=11),
    )
    _panel = alt.layer(
        _base.mark_rule(color="#9a9a9a").encode(x=alt.datum(0)),
        _base.mark_bar(size=14, color="#0072B2").encode(
            y=_y,
            x=alt.X("shift:Q", title="mean shift (base sd units)"),
            tooltip=[
                "persona:N",
                "reference_label:N",
                "position_label:N",
                "dimension:N",
                alt.Tooltip("shift:Q", format="+.2f"),
                alt.Tooltip("ci_lo:Q", format="+.2f"),
                alt.Tooltip("ci_hi:Q", format="+.2f"),
                alt.Tooltip("raw_shift:Q", format="+.4f"),
                "n:Q",
            ],
        ),
        _base.mark_rule(color="#333333", strokeWidth=0.8).encode(
            y=_y, x="ci_lo:Q", x2="ci_hi:Q"
        ),
    ).properties(width=210, height=24 * len(PERSONA_ORDER))
    _chart = _panel.facet(
        row=alt.Row(
            "dimension:N",
            sort=_dims,
            title=None,
            header=alt.Header(labelFontSize=12),
        ),
        column=alt.Column(
            "position_label:N",
            sort=list(POSITIONS.values()),
            title=None,
            header=alt.Header(labelFontSize=12),
        ),
    ).properties(
        title=f"Valence and arousal shift on neutral dialogue, against {REFERENCE_LABEL}"
    )
    _pre = AFFECT_SHIFTS.filter(
        (pl.col("position") == "pre_response")
        & (pl.col("reference_label") == REFERENCE_ORDER[0])
        & pl.col("dimension").is_in(_dims)
    )
    _lines = "; ".join(
        f"{r['persona']} {r['dimension']} {r['shift']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
        for r in _pre.sort("persona", "dimension").iter_rows(named=True)
    )
    AFFECT_CHART = save_chart(
        _chart,
        "persona_affect_shift",
        caption=(
            "Mean shift of each persona model on the fitted valence and arousal axes (dominance is in its own "
            "strip above), at all three "
            f"read positions, over {N_PROMPTS} WildChat prompts, in units of the "
            "base model's spread on that axis; whiskers are 95% intervals from the paired per-prompt "
            f"differences, against {REFERENCE_LABEL}."
        ),
        takeaway=f"At the pre-response token, against {REFERENCE_LABEL}: {_lines}.",
        notebook=NOTEBOOK,
    )
    AFFECT_CHART
    return


@app.cell
def _(REFERENCE_LABEL, mo):
    mo.md(f"""
    ### Where in the reply the mood sits

    **What the chart uses.** The per-token projections of every reply token onto the 171
    vectors at the readout layer, for the seven persona checkpoints and {REFERENCE_LABEL},
    on the same pool prompts as the reads above. The reply is cut into windows by token
    index: the first ten tokens, tokens 11 to 50, and everything from token 51 on; a prompt
    contributes to a window only if its reply reaches it.

    **How the numbers were made.** For one persona, window and prompt, the projections are
    averaged over the window's tokens, the control's average over the same window of the
    same prompt is subtracted, and the paired differences are averaged over prompts and
    divided by the base model's spread of whole-reply means on that vector, one unit for
    every window so the windows compare with each other and with the reply-mean read above.
    The bar is the mean over the 171 vectors of the absolute shift, the whisker a 95%
    interval from 1,000 resamples of the prompts. The tooltip also carries the correlation,
    over the 171 vectors, between the window's shift and the same persona's shift at the
    pre-response token: whether what the reply carries is the plan the model had before it
    started writing. If the mood were an opening register the first window would stand out
    and the later ones fall toward the floor; a sustained state reads about the same in all
    three.
    """)
    return


@app.cell
def _(
    CONTROLS,
    DATA,
    EXCLUDED,
    MODEL_LABEL,
    NOTEBOOK,
    PERSONAS,
    PERSONA_ORDER,
    REFERENCE,
    REFERENCE_LABEL,
    alt,
    json,
    load_file,
    np,
    pl,
    save_chart,
    story_boot_ci,
    story_boot_weights,
):
    # The reply cut into token windows (Carolina, 2026-09-10): does the mood sit at the
    # start of the reply, or run through it? Per-token projections from extract.py, the
    # same rows as the pooled reads, the excluded prompts dropped.
    WINDOWS = [("tokens 1-10", 0, 10), ("tokens 11-50", 10, 50), ("tokens 51+", 50, None), ("whole reply", 0, None)]
    _summary = json.loads((DATA / "readouts" / "summary.json").read_text(encoding="utf-8"))
    _others = [c for c in CONTROLS if c != REFERENCE]  # the two other controls, for the summary figure
    _models = [REFERENCE, *PERSONAS, *_others]
    _tok, _meta = {}, {}
    for _m in _models:
        _t = load_file(str(DATA / "activations" / _m / "token_projections.safetensors"))
        _tok[_m] = (_t["projections"].astype(np.float32), _t["offsets"])
        _meta[_m] = json.loads((DATA / "activations" / _m / "meta.json").read_text(encoding="utf-8"))
    _emotions = _meta[REFERENCE]["emotions"]
    _ids = [r["id"] for r in _meta[REFERENCE]["rows"]]
    for _m in _models:
        if [r["id"] for r in _meta[_m]["rows"]] != _ids:
            raise RuntimeError(f"{_m}: token projections are not on the same rows as {REFERENCE}")
    _keep = [i for i, _id in enumerate(_ids) if _id not in EXCLUDED]
    # one unit for every window: the base model's spread of whole-reply means, per vector
    _unit = np.array([_summary["base_stats"]["reply_mean"][e]["std"] for e in _emotions])
    _unit = np.where(_unit == 0, 1.0, _unit)

    def _window_means(_m, lo, hi):
        """Per prompt, the mean projection over reply tokens [lo, hi); NaN where the reply
        does not reach the window."""
        P, off = _tok[_m]
        out = np.full((len(_keep), len(_emotions)), np.nan, dtype=np.float32)
        for k, i in enumerate(_keep):
            a, b = int(off[i]), int(off[i + 1])
            end = b if hi is None else min(b, a + hi)
            if a + lo < end:
                out[k] = P[a + lo:end].mean(axis=0)
        return out

    _pre = {
        _m: np.array([st["mean_delta"] for st in sorted(_summary["models"][_m]["pre_response"]["per_emotion"], key=lambda st: _emotions.index(st["emotion"]))])
        for _m in PERSONAS
    }
    _ref_win = {w[0]: _window_means(REFERENCE, w[1], w[2]) for w in WINDOWS}
    _rows = []
    for _m in [*PERSONAS, *_others]:
        for _name, _lo, _hi in WINDOWS:
            D = (_window_means(_m, _lo, _hi) - _ref_win[_name]) / _unit
            ok = ~np.isnan(D).any(axis=1)
            D = D[ok]
            shift = D.mean(axis=0)
            point = float(np.abs(shift).mean())
            boot = np.abs(story_boot_weights(D.shape[0]) @ D).mean(axis=1)
            lo_ci, hi_ci = story_boot_ci(point, boot)
            _rows.append(
                {
                    "model": _m,
                    "persona": MODEL_LABEL[_m],
                    "kind": "persona" if _m in PERSONAS else "control",
                    "window": _name,
                    "mean_abs_shift": point,
                    "ci_lo": lo_ci,
                    "ci_hi": hi_ci,
                    "n_prompts": int(D.shape[0]),
                    "r_with_pre_response": float(np.corrcoef(shift, _pre[_m])[0, 1]) if _m in _pre else None,
                    "n_over_half_sd": int((np.abs(shift) >= 0.5).sum()),
                }
            )
    REPLY_WINDOWS = pl.DataFrame(_rows)
    _order = [w[0] for w in WINDOWS]
    _base = alt.Chart(REPLY_WINDOWS.filter(pl.col("kind") == "persona"))
    _y = alt.Y("persona:N", sort=PERSONA_ORDER, title=None, axis=alt.Axis(labelFontSize=11))
    _chart = (
        alt.layer(
            _base.mark_bar(size=14, color="#0072B2").encode(
                y=_y,
                x=alt.X("mean_abs_shift:Q", title="mean |shift| (base sd)"),
                tooltip=[
                    "persona:N",
                    "window:N",
                    alt.Tooltip("mean_abs_shift:Q", format=".3f"),
                    alt.Tooltip("ci_lo:Q", format=".3f", title="95% low"),
                    alt.Tooltip("ci_hi:Q", format=".3f", title="95% high"),
                    alt.Tooltip("n_prompts:Q", title="prompts reaching the window"),
                    alt.Tooltip("n_over_half_sd:Q", title="vectors past 0.5 sd"),
                    alt.Tooltip("r_with_pre_response:Q", format=".2f", title="r with the pre-response shift"),
                ],
            ),
            _base.mark_rule(color="#333333", strokeWidth=0.8).encode(y=_y, x="ci_lo:Q", x2="ci_hi:Q"),
        )
        .properties(width=170, height=24 * len(PERSONA_ORDER))
        .facet(column=alt.Column("window:N", sort=_order, title=None, header=alt.Header(labelFontSize=12)))
        .properties(title=f"Where in the reply the mood sits, on neutral dialogue: shift vs {REFERENCE_LABEL} by reply window")
    )
    _cell = {(r["persona"], r["window"]): r for r in _rows if r["kind"] == "persona"}
    _lines = "; ".join(
        f"{_p} " + ", ".join(f"{_cell[(_p, w)]['mean_abs_shift']:.2f}" for w in _order[:3])
        + f" (r with the pre-response shift {_cell[(_p, 'whole reply')]['r_with_pre_response']:.2f})"
        for _p in PERSONA_ORDER
    )
    REPLY_WINDOW_CHART = save_chart(
        _chart,
        "persona_shift_by_reply_window",
        caption=(
            "Mean over the 171 vectors of the absolute paired shift of each persona against "
            f"{REFERENCE_LABEL}, computed from the per-token projections averaged over a window of the reply "
            "(the first ten tokens, tokens 11 to 50, from token 51 on, and the whole reply), in units of the "
            "base model's spread of whole-reply means, so the windows share one scale; whiskers are 95% "
            "intervals from 1,000 resamples of the prompts, and a prompt counts in a window only if its reply "
            "reaches it. The tooltip carries the correlation over the 171 vectors between the window's shift "
            "and the persona's shift at the pre-response token."
        ),
        takeaway=(
            "Mean |shift| over the first ten tokens, tokens 11 to 50, and from token 51 on: " + _lines + "."
        ),
        notebook=NOTEBOOK,
    )
    REPLY_WINDOW_CHART
    return (REPLY_WINDOWS,)


@app.cell
def _(REFERENCE_LABEL, mo):
    mo.md(f"""
    ### The mood along the conversation, in one picture

    **What the chart uses.** Every read this notebook makes of a persona against
    {REFERENCE_LABEL}, laid out in the order the model meets them: the user's message, the
    pre-response token, the reply in its three windows, and, off the conversation, the
    3,420 third-person stories. Each point is the mean over the 171 vectors of the absolute
    shift at that read, with its 95% interval; the two other controls are drawn the same
    way in grey, so the eye has the size of a control-to-control difference at every read;
    the black tick is the noise floor, what the statistic reads when two models are
    identical (not computed for the reply windows).

    **How to read it.** Each read is in the base model's own spread at that read, so the
    height of a point says how far the mood moves the read relative to the variation the
    base model shows there from one text to the next; the reply windows share the reply's
    unit. What the figure is for is the shape: where a mood rises above the grey controls
    and where it does not.
    """)
    return


@app.cell
def _(
    CONTROLS,
    DATA,
    MODEL_LABEL,
    NOTEBOOK,
    PALETTE,
    PERSONAS,
    REFERENCE,
    REFERENCE_LABEL,
    REPLY_WINDOWS,
    WRAP_READ_LABEL,
    alt,
    json,
    pl,
    save_chart,
):
    # One figure for the per-position story (Carolina, 2026-09-10): mean |shift| against the
    # control at every read, in conversation order, personas in color and the two other
    # controls in grey.
    _summary = json.loads((DATA / "readouts" / "summary.json").read_text(encoding="utf-8"))
    _stories = json.loads((DATA / "story_readouts" / "summary.json").read_text(encoding="utf-8"))
    READS = [
        "user message",
        "pre-response token",
        "reply, tokens 1-10",
        "reply, tokens 11-50",
        "reply, tokens 51+",
        "third-person stories",
    ]
    _others = [c for c in CONTROLS if c != REFERENCE]
    _rows = []
    for _m in [*PERSONAS, *_others]:
        _kind = "persona" if _m in PERSONAS else "control"
        for _pos, _read in (("user_mean", READS[0]), ("pre_response", READS[1])):
            _b = _summary["models"][_m][_pos]
            _rows.append({"model": _m, "name": MODEL_LABEL[_m], "kind": _kind, "read": _read,
                          "shift": _b["mean_abs_shift"], "lo": _b["mean_abs_shift_ci"][0], "hi": _b["mean_abs_shift_ci"][1],
                          "floor": _b["mean_abs_shift_noise_floor"]})
        for _w, _read in (("tokens 1-10", READS[2]), ("tokens 11-50", READS[3]), ("tokens 51+", READS[4])):
            _r = REPLY_WINDOWS.filter((pl.col("model") == _m) & (pl.col("window") == _w)).to_dicts()[0]
            _rows.append({"model": _m, "name": MODEL_LABEL[_m], "kind": _kind, "read": _read,
                          "shift": _r["mean_abs_shift"], "lo": _r["ci_lo"], "hi": _r["ci_hi"], "floor": None})
        _b = _stories["shifts"][REFERENCE][_m]["held-out-stories"]
        _rows.append({"model": _m, "name": MODEL_LABEL[_m], "kind": _kind, "read": READS[5],
                      "shift": _b["mean_abs_shift"], "lo": _b["mean_abs_shift_ci"][0], "hi": _b["mean_abs_shift_ci"][1],
                      "floor": _b["mean_abs_shift_noise_floor"]})
    ALONG = pl.DataFrame(_rows)
    _floor = (
        ALONG.filter(pl.col("kind") == "persona").drop_nulls("floor")
        .group_by("read").agg(pl.col("floor").mean())
    )
    # persona colors from the house palette in persona order (the palette's first slot is
    # the grey the controls use)
    _color_of = {MODEL_LABEL[m]: PALETTE[1 + i] for i, m in enumerate(PERSONAS)}
    _names = [MODEL_LABEL[m] for m in [*PERSONAS, *_others]]
    _persona_names = {MODEL_LABEL[m] for m in PERSONAS}
    _color = alt.Color(
        "name:N",
        sort=_names,
        scale=alt.Scale(domain=_names, range=[_color_of[n] if n in _persona_names else "#9a9a9a" for n in _names]),
        legend=alt.Legend(title=None, orient="right", labelFontSize=11, symbolSize=120),
    )
    _x = alt.X(
        "read:N",
        sort=READS,
        title=None,
        axis=alt.Axis(labelAngle=0, labelFontSize=11, labelExpr=WRAP_READ_LABEL, labelPadding=8),
    )
    _y = alt.Y("shift:Q", title="mean |shift| over the 171 vectors (base sd at that read)")
    _base = alt.Chart(ALONG)
    _lines = _base.mark_line(strokeWidth=2, point=False).encode(
        x=_x, y=_y, color=_color,
        strokeDash=alt.StrokeDash("kind:N", scale=alt.Scale(domain=["persona", "control"], range=[[1, 0], [6, 4]]), legend=None),
        detail="name:N",
    )
    _points = _base.mark_point(size=70, filled=True, opacity=1).encode(
        x=_x, y=_y, color=_color,
        shape=alt.Shape("kind:N", scale=alt.Scale(domain=["persona", "control"], range=["circle", "diamond"]), legend=None),
        tooltip=[
            alt.Tooltip("name:N", title="model"),
            "read:N",
            alt.Tooltip("shift:Q", format=".3f", title="mean |shift|"),
            alt.Tooltip("lo:Q", format=".3f", title="95% low"),
            alt.Tooltip("hi:Q", format=".3f", title="95% high"),
        ],
    )
    _ci = _base.mark_rule(strokeWidth=1, opacity=0.7).encode(x=_x, y="lo:Q", y2="hi:Q", color=_color)
    _ticks = alt.Chart(_floor).mark_tick(color="#111111", thickness=2, size=22).encode(x=_x, y="floor:Q")
    _chart = (
        alt.layer(_ci, _lines, _points, _ticks)
        .properties(
            width=640,
            height=340,
            title=alt.Title(
                "Emotion-vector shift of each persona along the conversation",
                subtitle=(
                    "Mean absolute shift over the 171 emotion vectors relative to the control, per read position "
                    "(base-model SD units at that position; 95% CI). Grey dashed: the two remaining controls. "
                    "Black tick: noise floor."
                ),
                fontSize=14,
                subtitleFontSize=11,
                subtitleColor="#555",
                anchor="start",
            ),
        )
    )
    _peak = ALONG.filter((pl.col("kind") == "persona") & (pl.col("read") == READS[1])).sort("shift", descending=True)
    _ctrl = ALONG.filter(pl.col("kind") == "control")
    _line = "; ".join(
        f"{_r}: personas "
        f"{ALONG.filter((pl.col('kind') == 'persona') & (pl.col('read') == _r))['shift'].min():.2f} to "
        f"{ALONG.filter((pl.col('kind') == 'persona') & (pl.col('read') == _r))['shift'].max():.2f}, "
        f"other controls {_ctrl.filter(pl.col('read') == _r)['shift'].min():.2f} to {_ctrl.filter(pl.col('read') == _r)['shift'].max():.2f}"
        for _r in READS
    )
    ALONG_CHART = save_chart(
        _chart,
        "persona_shift_along_the_conversation",
        caption=(
            f"Mean over the 171 vectors of the absolute paired shift of each persona against {REFERENCE_LABEL}, at "
            "every read in the order the model meets them: the user's message, the pre-response token, the reply "
            "in three token windows, and the 3,420 third-person stories read as story text. Each read is in the "
            "base model's spread at that read (the reply windows share the whole-reply unit); intervals are 95%, "
            "from the paired per-text differences or from resampling the texts; the two other controls against "
            f"{REFERENCE_LABEL} are the grey dashed lines, and the black tick is the noise floor of the statistic."
        ),
        takeaway=(
            f"Peak at the pre-response token, {_peak['name'][0]} highest at {_peak['shift'][0]:.2f}. Range per read: "
            + _line + "."
        ),
        notebook=NOTEBOOK,
    )
    ALONG_CHART
    return


@app.cell
def _(STORIES, mo):
    mo.md(f"""
    ## Part 4: the same checkpoints on emotional stories

    The pool in Parts 1 to 3 is real user traffic, chosen because nothing in it was written
    to provoke a feeling. This part reads the same checkpoints on text whose emotional
    character is fixed and known: the **{STORIES["sets"]["held-out-stories"]["n"]:,} held-out
    stories**, twenty per emotion over all 171 emotions, that `01-emotion-vectors` set aside
    when it built the vectors and never used. They are read the way the vectors were built —
    raw text with no chat template, truncated at 256 tokens, layer 21 mean-pooled from the
    fiftieth token on — so the models and the emotion landmarks are, here and only here, in
    the same convention and on the same origin, which is why this is the part where the two
    can be drawn together.

    Shifts on this side are in the base model's per-vector standard deviation over those
    same held-out stories, so a shift of 1 is the size of the variation emotional content
    itself produces on that vector; intervals come from 1,000 paired resamples of the texts,
    and `data/story_readouts/summary.json` also carries the noise floor. The 1,200
    emotionless neutral dialogues that were read alongside the stories are no longer shown:
    the activations are already centred on the average emotional story, so no second origin
    is needed, and putting the emotionless set beside the emotions confused the reading.
    Their projections stay on disk, and the one place they are still quoted is the table at
    the end, which measures how far apart the two reading conventions sit.

    **What the part contains.** The stories were written to express 171 different emotions,
    so an average over all of them is not the interesting quantity; what the figures below
    ask is how each mood's reading of the same 3,420 stories differs from the control's,
    story by story and family by family. In order: the emotion landscape with the
    checkpoints in it and the same checkpoints magnified, so the size of the whole effect is
    on the record; the family means of each mood's shift, against all three references;
    **gain and offset**, which fit each mood's per-story reading as a straight line in the
    control's and separate a change of emotional range from a uniform shift; the **reading
    of a story's own emotion** by family, which says whether a mood amplifies or dampens the
    feeling the story actually carries; **where the misreads go**, which scores the read as a
    classification and shows which families a mood starts sending stories to; and
    **uniform versus selective**, which takes each mood's eight most-moved vectors and asks
    whether the shift is the same on every kind of story. An interactive section at the end
    goes down to individual stories, and saves nothing.
    """)
    return


@app.cell
def _(DIMENSIONS, MODELS, MODEL_LABEL, OFFSET, REFERENCE, STORIES, pl):
    # Every checkpoint's coordinates on the held-out stories, on the vectors' own origin,
    # and the family means of its shift against the reference on the same set.
    STORY_SET = "held-out-stories"
    _rows = []
    for _m in MODELS:
        _a = STORIES["distributions"][_m][STORY_SET]["affect"]["all"]
        _rows.append(
            {
                "model": _m,
                "label": MODEL_LABEL[_m],
                "n": STORIES["sets"][STORY_SET]["n"],
                **{d: _a[d]["mean"] - OFFSET[d] for d in DIMENSIONS},
                **{f"{d}_sd": _a[d]["sd"] for d in DIMENSIONS},
            }
        )
    STORY_POINTS = pl.DataFrame(_rows)
    for _d in DIMENSIONS:
        STORY_POINTS = STORY_POINTS.with_columns(
            (pl.col(_d) - pl.col(f"{_d}_sd")).alias(f"{_d}_lo"),
            (pl.col(_d) + pl.col(f"{_d}_sd")).alias(f"{_d}_hi"),
        )
    # One row per (reference, model, family): the story-side family means against each of
    # the three references the summary carries (moodless (control), neutral, base).
    STORY_FAMILY_SHIFTS = pl.DataFrame(
        [
            {
                "reference": _r,
                "reference_label": f"vs {MODEL_LABEL.get(_r, _r)}",
                "model": _m,
                "label": MODEL_LABEL.get(_m, _m.split("-")[0]),
                "family": _f,
                "mean_shift": _v,
            }
            for _r in [
                REFERENCE,
                *[r for r in STORIES["shifts"] if r != REFERENCE],
            ]
            for _m, _blocks in STORIES["shifts"][_r].items()
            for _f, _v in _blocks[STORY_SET]["family_mean_shift"].items()
        ]
    )
    ACCURACY = STORIES["accuracy_check"]
    GENRE_OFFSET = STORIES["genre_offset"]
    return ACCURACY, GENRE_OFFSET, STORY_FAMILY_SHIFTS, STORY_POINTS, STORY_SET


@app.cell
def _(ACCURACY, STORIES, mo):
    mo.md(f"""
    ### The story-read map: models and emotions in one convention

    **What the chart uses.** The {STORIES["sets"]["held-out-stories"]["n"]:,} held-out
    stories, read by all ten checkpoints, and the 171 emotion vectors themselves. Both sides
    of this figure are in the story convention: the vectors were built by pooling activations
    over stories read as raw text from the fiftieth token on, and the checkpoints were read
    on the held-out stories in exactly that way, with the same truncation and the same layer.

    **How the numbers were made.** An emotion's position is its own vector's score on the two
    fitted axes — the same principal components of the vector set used in Part 3 — so the
    emotions sit where the axes say they sit. A checkpoint's position is the mean, over the
    3,420 stories, of the projection of its activation onto those same two axis directions,
    minus the average emotional story's own projection, which is the origin the vectors are
    centred on. Both sides are therefore on one origin and in one set of units, and the
    distance between a checkpoint and an emotion is meaningful here in a way it is not in
    Part 3.

    **Why there are two figures.** On the emotions' own scale the ten checkpoints land on top
    of one another near the origin, which is itself the answer to one question and useless for
    another, so this figure shows them in the landscape, with their names set out beside the
    cluster and joined to their diamonds by leader lines, and the next one shows the same ten
    points magnified on their own axis range. The five emotions that share a persona's name (irritated, remorseful, anxious,
    suspicious and grateful; upbeat and apologetic are not emotions of the taxonomy) are
    labeled, so the persona named after an emotion can be found next to it.

    **The check that this read is the right one.** Scored the way `01-emotion-vectors` scored
    it, the base model recovers each story's own emotion first out of 171 on
    {ACCURACY["here"]["top1"]:.1%} of the held-out stories against that experiment's
    {ACCURACY["reference"]["top1"]:.1%}, so the reader and the vectors here are the ones the
    vectors were validated with.
    """)
    return


@app.cell
def _(
    AXES,
    AXIS_LABEL,
    DIMENSIONS,
    EMO2FAM,
    FAMILIES,
    MODELS,
    MODEL_LABEL,
    PALETTE,
    PERSONAS,
    STORY_POINTS,
    alt,
    pl,
):
    _fit = AXES["fits"][AXES["primary"]]
    STORY_EMOTIONS = pl.DataFrame(
        [
            {
                "kind": "emotion",
                "name": _e,
                "family": EMO2FAM.get(_e, "?"),
                **{d: float(_fit[d]["scores"][_e]) for d in DIMENSIONS},
            }
            for _e in AXES["emotions"]
        ]
    )
    # The emotions that share a persona's name (five of the seven personas are emotions of
    # the taxonomy; upbeat and apologetic are not), labeled on the landscape.
    STORY_NAMED_EMOTIONS = [
        p.split("-", 1)[0]
        for p in PERSONAS
        if p.split("-", 1)[0] in AXES["emotions"]
    ]
    # The checkpoints, with a 95% interval on the mean rather than the spread over stories:
    # the spread is the emotional range of the corpus, which is the same for every model and
    # would hide the differences between them.
    STORY_MODELS = STORY_POINTS.with_columns(
        [
            (1.96 * pl.col(f"{d}_sd") / pl.col("n").sqrt()).alias(f"{d}_se")
            for d in DIMENSIONS
        ]
    ).with_columns(
        [(pl.col(d) - pl.col(f"{d}_se")).alias(f"{d}_lo") for d in DIMENSIONS]
        + [
            (pl.col(d) + pl.col(f"{d}_se")).alias(f"{d}_hi")
            for d in DIMENSIONS
        ]
    )
    STORY_MODEL_ORDER = [MODEL_LABEL[m] for m in MODELS]
    STORY_MODEL_COLOR = alt.Color(
        "label:N",
        sort=STORY_MODEL_ORDER,
        scale=alt.Scale(
            domain=STORY_MODEL_ORDER, range=PALETTE[: len(STORY_MODEL_ORDER)]
        ),
        legend=alt.Legend(
            title="checkpoint",
            orient="right",
            labelFontSize=11,
            symbolSize=160,
        ),
    )
    STORY_MODEL_TOOLTIP = [
        alt.Tooltip("label:N", title="checkpoint"),
        alt.Tooltip("valence:Q", format="+.3f"),
        alt.Tooltip(
            "valence_sd:Q", format=".2f", title="valence sd over stories"
        ),
        alt.Tooltip("arousal:Q", format="+.3f"),
        alt.Tooltip(
            "arousal_sd:Q", format=".2f", title="arousal sd over stories"
        ),
        alt.Tooltip("dominance:Q", format="+.3f"),
        alt.Tooltip("n:Q", title="stories"),
    ]
    STORY_X_TITLE = AXIS_LABEL["valence"]
    STORY_Y_TITLE = AXIS_LABEL["arousal"]
    _ = FAMILIES
    return (
        STORY_EMOTIONS,
        STORY_MODELS,
        STORY_MODEL_COLOR,
        STORY_MODEL_TOOLTIP,
        STORY_NAMED_EMOTIONS,
        STORY_X_TITLE,
        STORY_Y_TITLE,
    )


@app.cell
def _(
    FAMILIES,
    NOTEBOOK,
    STORIES,
    STORY_EMOTIONS,
    STORY_MODELS,
    STORY_MODEL_COLOR,
    STORY_MODEL_TOOLTIP,
    STORY_NAMED_EMOTIONS,
    STORY_SET,
    STORY_X_TITLE,
    STORY_Y_TITLE,
    alt,
    pl,
    save_chart,
):
    # The emotion landscape, with the checkpoints where they actually fall in it. The ten
    # checkpoints sit within a fraction of a unit of one another, so their labels are
    # staggered by rank on arousal (alternating above and below, stepping outward) to stay
    # legible; the magnified figure after this one separates them properly.
    _x = alt.X("valence:Q", title=STORY_X_TITLE)
    _y = alt.Y("arousal:Q", title=STORY_Y_TITLE)
    _dots = (
        alt.Chart(STORY_EMOTIONS)
        .mark_circle(size=90, opacity=0.35)
        .encode(
            x=_x,
            y=_y,
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=FAMILIES, scheme="tableau10"),
                legend=alt.Legend(
                    title="emotion family",
                    orient="top",
                    direction="horizontal",
                    columns=5,
                    labelFontSize=10,
                ),
            ),
            tooltip=[
                "name:N",
                "family:N",
                alt.Tooltip("valence:Q", format=".2f"),
                alt.Tooltip("arousal:Q", format=".2f"),
                alt.Tooltip("dominance:Q", format=".2f"),
            ],
        )
    )
    _named = STORY_EMOTIONS.filter(pl.col("name").is_in(STORY_NAMED_EMOTIONS))
    # The emotions that share a persona's name get a label and nothing else: the dot stays
    # as faint as every other emotion's (Carolina, 2026-09-09: "not highlighted, just labeled").
    _named_text = (
        alt.Chart(_named)
        .mark_text(align="left", dx=7, dy=-8, fontSize=11, color="#333333")
        .encode(x=_x, y=_y, text="name:N")
    )
    # The ten checkpoints sit within a fraction of a unit of one another, so their names go
    # in a column to the right of the cluster, ordered by arousal, each joined to its diamond
    # by a thin leader line.
    _ranked = (
        STORY_MODELS.sort("arousal", descending=True)
        .with_row_index("rank")
        .with_columns(pl.col("rank").cast(pl.Int64))
        .with_columns(
            pl.lit(1.15).alias("label_x"),
            (2.0 - pl.col("rank").cast(pl.Float64) * 0.42).alias("label_y"),
        )
    )
    _zero_x = (
        alt.Chart(pl.DataFrame({"v": [0.0]}))
        .mark_rule(color="#666666", strokeDash=[4, 3])
        .encode(x="v:Q")
    )
    _zero_y = (
        alt.Chart(pl.DataFrame({"v": [0.0]}))
        .mark_rule(color="#666666", strokeDash=[4, 3])
        .encode(y="v:Q")
    )
    _diamonds = (
        alt.Chart(STORY_MODELS)
        .mark_point(
            shape="diamond",
            size=260,
            filled=True,
            stroke="#111111",
            strokeWidth=1.2,
            opacity=1,
        )
        .encode(
            x=_x, y=_y, color=STORY_MODEL_COLOR, tooltip=STORY_MODEL_TOOLTIP
        )
    )
    _leaders = (
        alt.Chart(_ranked)
        .mark_rule(color="#555555", strokeWidth=0.8, opacity=0.8)
        .encode(x="valence:Q", y="arousal:Q", x2="label_x:Q", y2="label_y:Q")
    )
    _labels = (
        alt.Chart(_ranked)
        .mark_text(
            align="left", dx=5, fontSize=11, fontWeight="bold", color="#111111"
        )
        .encode(x="label_x:Q", y="label_y:Q", text="label:N")
    )
    _chart = (
        alt.layer(
            _zero_x, _zero_y, _dots, _named_text, _leaders, _diamonds, _labels
        )
        .resolve_scale(color="independent")
        .properties(
            width=820,
            height=560,
            title=alt.Title(
                "The emotion landscape, and where the checkpoints read in it",
                subtitle=(
                    f"{STORIES['sets'][STORY_SET]['n']:,} held-out stories read as story text by every checkpoint; "
                    "171 emotion vectors (faint, colored by family; the five that share a persona's name are "
                    "labeled) and the ten checkpoints (diamonds); origin = the average emotional story"
                ),
                fontSize=14,
                subtitleFontSize=11,
                subtitleColor="#555",
                anchor="start",
            ),
        )
        .configure_view(fill="#eaeaf2", stroke=None)
        .configure_axis(
            grid=True,
            gridColor="#ffffff",
            gridWidth=1,
            domain=False,
            tickColor="#ffffff",
        )
    )
    _span = (
        float(
            STORY_EMOTIONS["valence"].max() - STORY_EMOTIONS["valence"].min()
        ),
        float(STORY_MODELS["valence"].max() - STORY_MODELS["valence"].min()),
    )
    STORY_MAP_CHART = save_chart(
        _chart,
        "story_read_affect_map",
        caption=(
            "The 171 emotion vectors (faint dots, colored by taxonomy family; the five emotions that share a "
            "persona's name are labeled) and the ten checkpoints (labeled diamonds, one color "
            "each) on the fitted valence and arousal axes, both read in the story convention -- raw text, "
            "256-token truncation, layer 21 mean-pooled from the fiftieth token on -- so that a checkpoint's "
            "position among the emotions can be read here in a way it cannot in Part 3. An emotion sits at its "
            "own vector's score; a checkpoint sits at its mean projection over the "
            f"{STORIES['sets'][STORY_SET]['n']:,} held-out stories, on the same origin, the average emotional "
            "story, marked by the dashed zero lines. The checkpoints sit within a fraction of a unit of one "
            "another, so their names are set out in a column to the right, each joined to its diamond by a "
            "leader line; the next figure magnifies them."
        ),
        takeaway=(
            f"Read on emotional stories every checkpoint sits within {_span[1]:.2f} valence units of every "
            f"other, against a {_span[0]:.1f}-unit spread across the 171 emotions, so a mood barely moves where "
            "the model reads emotional text; the checkpoints cluster just below the origin on arousal and "
            "just left of it on valence, at the edge of the depleted and vigilant families."
        ),
        notebook=NOTEBOOK,
    )
    STORY_MAP_CHART
    return


@app.cell
def _(STORIES, STORY_SET, mo):
    mo.md(f"""
    ### The same ten checkpoints, magnified

    **What the chart uses.** The same {STORIES["sets"][STORY_SET]["n"]:,} held-out stories
    and the same ten checkpoints as the landscape above, nothing recomputed.

    **How the numbers were made.** A checkpoint's position is, as above, the mean over the
    held-out stories of its activation's projection onto the two fitted axes, minus the
    average emotional story's own projection, so the dashed lines at zero are the same
    origin as in the landscape. Only the axis range differs: it is the small box around the
    origin in which all ten checkpoints land. The bars are 95% intervals on the mean (1.96
    standard errors over the stories), not the spread over stories: that spread is the
    emotional range of the corpus, is nearly identical for every checkpoint, and would swamp
    the differences between them; it is in the tooltip.
    """)
    return


@app.cell
def _(
    NOTEBOOK,
    STORIES,
    STORY_MODELS,
    STORY_MODEL_COLOR,
    STORY_MODEL_TOOLTIP,
    STORY_SET,
    STORY_X_TITLE,
    STORY_Y_TITLE,
    alt,
    save_chart,
):
    _zx = alt.X("valence:Q", title=STORY_X_TITLE, scale=alt.Scale(zero=False))
    _zy = alt.Y("arousal:Q", title=STORY_Y_TITLE, scale=alt.Scale(zero=False))
    _base = alt.Chart(STORY_MODELS)
    _chart = (
        alt.layer(
            _base.mark_rule(color="#666666", strokeDash=[4, 3]).encode(
                x=alt.datum(0)
            ),
            _base.mark_rule(color="#666666", strokeDash=[4, 3]).encode(
                y=alt.datum(0)
            ),
            _base.mark_rule(strokeWidth=1.5).encode(
                x="valence_lo:Q",
                x2="valence_hi:Q",
                y=_zy,
                color=STORY_MODEL_COLOR,
            ),
            _base.mark_rule(strokeWidth=1.5).encode(
                x=_zx,
                y="arousal_lo:Q",
                y2="arousal_hi:Q",
                color=STORY_MODEL_COLOR,
            ),
            _base.mark_point(
                shape="diamond",
                size=300,
                filled=True,
                stroke="#111111",
                strokeWidth=1.2,
                opacity=1,
            ).encode(
                x=_zx,
                y=_zy,
                color=STORY_MODEL_COLOR,
                tooltip=STORY_MODEL_TOOLTIP,
            ),
            _base.mark_text(
                align="left",
                dx=14,
                dy=-2,
                fontSize=11,
                fontWeight="bold",
                color="#111111",
            ).encode(x=_zx, y=_zy, text="label:N"),
        )
        .properties(
            width=820,
            height=520,
            title=alt.Title(
                "The same ten checkpoints, magnified",
                subtitle=(
                    f"{STORIES['sets'][STORY_SET]['n']:,} held-out stories; the axis range is the small box "
                    "around the origin in the landscape above; bars = 95% intervals on the mean"
                ),
                fontSize=14,
                subtitleFontSize=11,
                subtitleColor="#555",
                anchor="start",
            ),
        )
        .configure_view(fill="#eaeaf2", stroke=None)
        .configure_axis(
            grid=True,
            gridColor="#ffffff",
            gridWidth=1,
            domain=False,
            tickColor="#ffffff",
        )
    )
    _summary = "; ".join(
        f"{r['label']} valence {r['valence']:+.3f}, arousal {r['arousal']:+.3f}"
        for r in STORY_MODELS.iter_rows(named=True)
    )
    STORY_CHECKPOINTS_CHART = save_chart(
        _chart,
        "story_read_checkpoints",
        caption=(
            "The ten checkpoints alone on the fitted valence and arousal axes, read on the "
            f"{STORIES['sets'][STORY_SET]['n']:,} held-out stories in the story convention, on the same origin "
            "as the landscape (the average emotional story, dashed lines) but on their own axis range, with 95% "
            "intervals on the mean over stories (1.96 standard errors). The spread over stories, which is the "
            "emotional range of the corpus and nearly the same for every checkpoint, is in the tooltip."
        ),
        takeaway=f"Checkpoint positions on the story read, in the axes' units: {_summary}.",
        notebook=NOTEBOOK,
    )
    STORY_CHECKPOINTS_CHART
    return


@app.cell
def _(NEUTRAL_LABEL, REFERENCE_LABEL, STORIES, STORY_SET, mo):
    mo.md(f"""
    ### Where each mood's story-side shift lands, family by family

    **What the chart uses.** The {STORIES["sets"][STORY_SET]["n"]:,} held-out stories again,
    read by the seven persona checkpoints and by the three reference models ({REFERENCE_LABEL},
    {NEUTRAL_LABEL} and the untrained base) in the story convention, projected onto the same
    171 emotion vectors.

    **How the numbers were made.** For one persona and one reference, each vector's shift is
    the mean over the 3,420 stories of the difference between the persona's projection and
    the reference's on the same story, divided by the base model's standard deviation for
    that vector over those same stories; a bar is the plain average of those values over the
    emotions of one taxonomy family, and the three bars within a family are the three
    references, the primary one ({REFERENCE_LABEL}) first. A value of 1 would mean a shift the
    size of the variation emotional content itself produces on that vector, so these are
    small numbers by construction: the moods tilt the read, they do not move it as far as
    changing the story does. Reading the three bars together separates the mood (against a
    control) from everything the recipe installs (against base).
    """)
    return


@app.cell
def _(
    FAMILIES,
    NOTEBOOK,
    PERSONA_ORDER,
    REFERENCE_ORDER,
    STORIES,
    STORY_FAMILY_SHIFTS,
    STORY_SET,
    alt,
    pl,
    save_chart,
):
    # Where each mood's shift lands by family, on the emotional stories, against the control
    # only (Carolina, 2026-09-10; one bar per reference until then, the others still in
    # STORY_FAMILY_SHIFTS).
    _df = STORY_FAMILY_SHIFTS.filter(
        pl.col("label").is_in(PERSONA_ORDER)
        & (pl.col("reference_label") == REFERENCE_ORDER[0])
    )
    _base = alt.Chart(_df)
    _panel = alt.layer(
        _base.mark_rule(color="#9a9a9a").encode(x=alt.datum(0)),
        _base.mark_bar(size=12, color="#0072B2").encode(
            y=alt.Y(
                "family:N",
                sort=FAMILIES,
                title=None,
                axis=alt.Axis(labelFontSize=9),
            ),
            x=alt.X("mean_shift:Q", title=None),
            tooltip=[
                "label:N",
                "reference_label:N",
                "family:N",
                alt.Tooltip("mean_shift:Q", format="+.3f"),
            ],
        ),
    ).properties(width=170, height=200)
    _chart = _panel.facet(
        column=alt.Column(
            "label:N",
            sort=PERSONA_ORDER,
            title=None,
            header=alt.Header(labelFontSize=12),
        )
    ).properties(
        title=alt.Title(
            f"Family means of the story-side shift, against {REFERENCE_ORDER[0].removeprefix('vs ')}",
            subtitle="x = family mean shift, in units of the base model's per-vector spread over the held-out stories",
            fontSize=14,
            subtitleFontSize=11,
            subtitleColor="#555",
            anchor="start",
        )
    )
    _primary = _df.filter(pl.col("reference_label") == REFERENCE_ORDER[0])
    _rows = {
        (r["label"], r["family"]): r["mean_shift"] for r in _primary.to_dicts()
    }
    _lines = []
    for _p in PERSONA_ORDER:
        _fam = max(FAMILIES, key=lambda f: abs(_rows[(_p, f)]))
        _lines.append(f"{_p}: {_fam} {_rows[(_p, _fam)]:+.2f}")
    STORY_FAMILY_CHART = save_chart(
        _chart,
        "story_family_shift",
        caption=(
            "Mean over each taxonomy family of the per-vector shift on the "
            f"{STORIES['sets'][STORY_SET]['n']:,} held-out emotional stories, one panel per mood, against "
            f"{REFERENCE_ORDER[0].removeprefix('vs ')}, in "
            "units of the base model's per-vector spread over those same stories, so a value of 1 is the size "
            "of the variation emotional content itself produces on that vector. Against the control the bar is "
            "the mood alone."
        ),
        takeaway=(
            f"Largest family per mood on the story read, against {REFERENCE_ORDER[0].removeprefix('vs ')}: "
            + "; ".join(_lines)
            + "."
        ),
        notebook=NOTEBOOK,
    )
    STORY_FAMILY_CHART
    return


@app.cell
def _(
    DATA,
    FAMILIES,
    MODELS,
    STORIES,
    json,
    load_clusters,
    load_file,
    np,
    slugify,
):
    # The raw material for the rest of Part 4: every checkpoint's projection of every
    # held-out story onto the 171 vectors and onto the three affect axes, plus the identity
    # of each story. read_stories.py stores the projections in the order 01-emotion-vectors
    # pooled them -- the taxonomy's emotion order, and inside one emotion the held-out
    # indices ascending -- so the labels are rebuilt from that experiment's split file and
    # story corpus rather than stored a second time; the rebuild asserts the count.
    STORY_PROJ = {}
    STORY_AFFECT = {}
    for _m in MODELS:
        _t = load_file(str(DATA / "story_readouts" / f"{_m}.safetensors"))
        STORY_PROJ[_m] = _t["story_projections"].astype(np.float64)
        STORY_AFFECT[_m] = _t["story_affect"].astype(np.float64)

    _cross = DATA.parents[1] / "01-emotion-vectors"
    _splits = json.loads(
        (_cross / "data" / "splits.json").read_text(encoding="utf-8")
    )
    _clusters = load_clusters()
    STORY_LABELS = []
    for _f in FAMILIES:
        for _e in _clusters[_f]:
            _want = set(_splits["emotions"][_e]["test"])
            with (
                _cross / "data" / "stories" / "hf" / f"{slugify(_e)}.jsonl"
            ).open(encoding="utf-8") as _fh:
                for _j, _line in enumerate(_fh):
                    if _j in _want:
                        _row = json.loads(_line)
                        STORY_LABELS.append(
                            {
                                "emotion": slugify(_row["emotion"]),
                                "family": _f,
                                "topic": _row["topic"],
                                "idx": int(_row["idx"]),
                            }
                        )
    if len(STORY_LABELS) != STORIES["sets"]["held-out-stories"]["n"]:
        raise RuntimeError(
            f"rebuilt {len(STORY_LABELS)} story labels, the readouts hold "
            f"{STORIES['sets']['held-out-stories']['n']}"
        )

    STORY_EMOTION_ORDER = STORIES["emotions"]  # the vectors' column order
    _col_of = {e: j for j, e in enumerate(STORY_EMOTION_ORDER)}
    # For each story, the column of the vector of the emotion it was written to express.
    STORY_OWN_COL = np.array([_col_of[r["emotion"]] for r in STORY_LABELS])
    _story_family_of = np.array([r["family"] for r in STORY_LABELS])
    STORY_ROWS_OF_FAMILY = {
        f: np.where(_story_family_of == f)[0] for f in FAMILIES
    }
    # The unit of every number on this side: the base model's per-vector standard deviation
    # over these same 3,420 stories, so a value of 1 is the size of the variation emotional
    # content itself produces on that vector. The mean is kept for the scoring in the
    # confusion exhibit, which needs a fixed scale rather than a per-model one.
    STORY_BASE_MEAN = STORY_PROJ["base"].mean(axis=0)
    STORY_BASE_STD = STORY_PROJ["base"].std(axis=0)
    STORY_BASE_STD = np.where(STORY_BASE_STD == 0, 1.0, STORY_BASE_STD)
    STORY_AFFECT_SD = np.array(
        [
            STORIES["affect_base_story_stats"][d]["std"]
            for d in ("valence", "arousal", "dominance")
        ]
    )
    # Family names shortened to their first word for the axes of the two matrix figures,
    # where the full two-word names do not fit; the ten first words are all distinct.
    SHORT_FAMILY = {f: f.split("_")[0] for f in FAMILIES}
    SHORT_FAMILIES = [SHORT_FAMILY[f] for f in FAMILIES]

    _boot_cache: dict[int, np.ndarray] = {}

    def story_boot_weights(n: int) -> np.ndarray:
        """``[1000, n]`` multinomial resampling weights, seeded from the sample size.

        The same construction ``common.boot_weights`` uses for the stored statistics, so a
        figure computed here and a number in ``summary.json`` resample the same texts.
        """
        if n not in _boot_cache:
            _rng = np.random.default_rng(20260909 + n)
            _boot_cache[n] = (
                _rng.multinomial(n, np.full(n, 1.0 / n), size=1000) / n
            )
        return _boot_cache[n]

    def story_boot_ci(point: float, replicates) -> tuple[float, float]:
        """Reverse-percentile (basic) 95% interval around ``point``, as ``common.boot_ci``."""
        _lo, _hi = np.percentile(replicates, [2.5, 97.5])
        return float(2 * point - _hi), float(2 * point - _lo)

    return (
        SHORT_FAMILIES,
        SHORT_FAMILY,
        STORY_AFFECT,
        STORY_AFFECT_SD,
        STORY_BASE_MEAN,
        STORY_BASE_STD,
        STORY_EMOTION_ORDER,
        STORY_LABELS,
        STORY_OWN_COL,
        STORY_PROJ,
        STORY_ROWS_OF_FAMILY,
        story_boot_ci,
        story_boot_weights,
    )


@app.cell
def _(NEUTRAL_LABEL, REFERENCE_LABEL, STORIES, STORY_SET, mo):
    mo.md(f"""
    ### Gain and offset: does a mood compress the corpus's emotional range, or shift it?

    **What the chart uses.** The {STORIES["sets"][STORY_SET]["n"]:,} held-out stories, read
    by the seven persona checkpoints and by the two controls in the story convention (raw
    text, 256-token truncation, layer 21 mean-pooled from the fiftieth token on), projected
    onto the three fitted affect axes — the directions in the model's activation space that
    a principal component analysis of the 171 emotion vectors puts closest to published
    human valence, arousal and dominance ratings, the same three axes as Part 3.

    **How the numbers were made.** Each story gives one number per checkpoint per axis. For
    a persona and one axis, its 3,420 numbers are regressed on the control's 3,420 numbers
    for the same stories by ordinary least squares, a straight line fitted by minimizing the
    squared vertical distances. Both sides are first put on the same footing: the average
    emotional story's own coordinate is subtracted, so zero on either side means a story the
    model reads as emotionally average, and both are divided by the base model's spread over
    these stories, so a value of 1 is the variation the corpus itself produces on that axis.

    The line has two numbers. The **slope** is the gain: 1 means the persona spreads the
    corpus out exactly as far as the control does, below 1 that it compresses the range
    (strongly emotional stories are pulled toward the middle), above 1 that it exaggerates
    it. The **intercept** is a uniform offset in those same spread units: what the persona
    reads on a story the control reads as emotionally average. The two are separate
    questions, so they are in separate panels with their reference values marked, 1 for the
    slope and 0 for the intercept. Whiskers are 95% intervals from 1,000 resamples of the
    stories with replacement, and the tooltip carries R², the share of the persona's
    story-to-story variation the line accounts for.

    **What the rows are.** The seven personas are fitted against {REFERENCE_LABEL}, so a
    slope or an offset is the mood alone. The two controls are fitted against the untrained
    base model instead, and are shown in the same panels so the recipe's own gain and offset
    can be read beside the moods': {REFERENCE_LABEL} against base is what the whole
    distillation does, {NEUTRAL_LABEL} against base what plain distillation does without the
    wrapper and the prefill.
    """)
    return


@app.cell
def _(
    CONTROLS,
    DIMENSIONS,
    MODEL_LABEL,
    NEUTRAL_LABEL,
    NOTEBOOK,
    OFFSET,
    PERSONAS,
    PERSONA_ORDER,
    REFERENCE,
    REFERENCE_LABEL,
    STORIES,
    STORY_AFFECT,
    STORY_AFFECT_SD,
    STORY_SET,
    alt,
    pl,
    save_chart,
    story_boot_ci,
    story_boot_weights,
):
    # One straight line per model and axis: the model's per-story affect reading regressed
    # on its reference's, over the 3,420 held-out stories, both centred on the average
    # emotional story and scaled by the base model's spread over the same stories.
    _pairs = [(REFERENCE, _m) for _m in PERSONAS] + [
        ("base", _c) for _c in CONTROLS
    ]
    _rows = []
    for _ref, _model in _pairs:
        for _k, _d in enumerate(DIMENSIONS):
            _x = (STORY_AFFECT[_ref][:, _k] - OFFSET[_d]) / STORY_AFFECT_SD[_k]
            _y = (STORY_AFFECT[_model][:, _k] - OFFSET[_d]) / STORY_AFFECT_SD[
                _k
            ]
            _sxx = float(((_x - _x.mean()) ** 2).sum())
            _slope = float(((_x - _x.mean()) * (_y - _y.mean())).sum() / _sxx)
            _intercept = float(_y.mean() - _slope * _x.mean())
            _resid = _y - (_slope * _x + _intercept)
            _r2 = float(1 - (_resid**2).sum() / ((_y - _y.mean()) ** 2).sum())
            _w = story_boot_weights(len(_x))
            _mx, _my = _w @ _x, _w @ _y
            _boot_slope = (_w @ (_x * _y) - _mx * _my) / (
                _w @ (_x * _x) - _mx**2
            )
            _boot_inter = _my - _boot_slope * _mx
            for _measure, _value, _reps, _refline in (
                ("gain (slope)", _slope, _boot_slope, 1.0),
                ("offset (intercept)", _intercept, _boot_inter, 0.0),
            ):
                _lo, _hi = story_boot_ci(_value, _reps)
                _rows.append(
                    {
                        "label": MODEL_LABEL[_model],
                        "reference_label": f"vs {MODEL_LABEL[_ref]}",
                        "axis": _d,
                        "measure": _measure,
                        "value": _value,
                        "ci_lo": _lo,
                        "ci_hi": _hi,
                        "slope": _slope,
                        "intercept": _intercept,
                        "r2": _r2,
                        "refline": _refline,
                        "n": int(len(_x)),
                    }
                )
    STORY_GAIN = pl.DataFrame(_rows)
    _order = [REFERENCE_LABEL, NEUTRAL_LABEL, *PERSONA_ORDER]
    _refs = [f"vs {REFERENCE_LABEL}", "vs base"]
    _color = alt.Color(
        "reference_label:N",
        sort=_refs,
        scale=alt.Scale(domain=_refs, range=["#0072B2", "#7f7f7f"]),
        legend=alt.Legend(
            title=None, orient="top", direction="horizontal", labelFontSize=11
        ),
    )
    _tooltip = [
        alt.Tooltip("label:N", title="checkpoint"),
        alt.Tooltip("reference_label:N", title="against"),
        alt.Tooltip("axis:N"),
        alt.Tooltip("slope:Q", format=".3f", title="gain (slope)"),
        alt.Tooltip("intercept:Q", format="+.3f", title="offset (intercept)"),
        alt.Tooltip("ci_lo:Q", format="+.3f", title="95% low"),
        alt.Tooltip("ci_hi:Q", format="+.3f", title="95% high"),
        alt.Tooltip("r2:Q", format=".4f", title="R squared"),
        alt.Tooltip("n:Q", title="stories"),
    ]

    def _gain_panel(measure: str, fmt: str):
        _df = STORY_GAIN.filter(pl.col("measure") == measure)
        _b = alt.Chart(_df)
        _y = alt.Y(
            "label:N", sort=_order, title=None, axis=alt.Axis(labelFontSize=11)
        )
        return (
            alt.layer(
                _b.mark_rule(color="#9a9a9a", strokeDash=[4, 3]).encode(
                    x="refline:Q"
                ),
                _b.mark_rule(strokeWidth=1.4).encode(
                    x="ci_lo:Q", x2="ci_hi:Q", y=_y, color=_color
                ),
                _b.mark_point(
                    shape="diamond",
                    size=150,
                    filled=True,
                    stroke="#111111",
                    strokeWidth=0.8,
                    opacity=1,
                ).encode(
                    x=alt.X(
                        "value:Q",
                        title=measure,
                        scale=alt.Scale(zero=False, padding=18),
                        axis=alt.Axis(format=fmt),
                    ),
                    y=_y,
                    color=_color,
                    tooltip=_tooltip,
                ),
            )
            .properties(width=190, height=200)
            .facet(
                column=alt.Column(
                    "axis:N",
                    sort=DIMENSIONS,
                    title=None,
                    header=alt.Header(labelFontSize=12),
                )
            )
        )

    _chart = (
        alt.vconcat(
            _gain_panel("gain (slope)", ".2f"),
            _gain_panel("offset (intercept)", "+.2f"),
        )
        .resolve_scale(color="shared")
        .properties(
            title=alt.Title(
                "How each checkpoint rewrites its reference's reading of the same story",
                subtitle=[
                    "Ordinary least squares over the 3,420 held-out stories, one fit per affect axis.",
                    "Top: gain, the slope of the fit (1 = the same emotional range as the reference).",
                    "Bottom: offset, the intercept, in units of the base model's spread over these stories "
                    "(0 = no uniform shift). The 95% intervals are narrower than the marks.",
                ],
                fontSize=14,
                subtitleFontSize=11,
                subtitleColor="#555",
                anchor="start",
            )
        )
        .configure_view(fill="#eaeaf2", stroke=None)
        .configure_axis(
            grid=True,
            gridColor="#ffffff",
            gridWidth=1,
            domain=False,
            tickColor="#ffffff",
        )
    )
    _sl = {(r["label"], r["axis"]): r["slope"] for r in STORY_GAIN.to_dicts()}
    _in = {
        (r["label"], r["axis"]): r["intercept"] for r in STORY_GAIN.to_dicts()
    }
    _flat = "; ".join(
        f"{p} valence {_sl[(p, 'valence')]:.3f}/{_in[(p, 'valence')]:+.3f}, "
        f"arousal {_sl[(p, 'arousal')]:.3f}/{_in[(p, 'arousal')]:+.3f}"
        for p in PERSONA_ORDER
    )
    STORY_GAIN_CHART = save_chart(
        _chart,
        "story_affect_gain",
        caption=(
            f"Each checkpoint's per-story reading on the {STORIES['sets'][STORY_SET]['n']:,} held-out stories, "
            "regressed by ordinary least squares on its reference's reading of the same stories, one fit per "
            "affect axis. Both sides are centred on the average emotional story and divided by the base model's "
            "spread over these stories. The upper row is the slope, the gain: below 1 the checkpoint compresses "
            "the corpus's emotional range, above 1 it exaggerates it. The lower row is the intercept, a uniform "
            "offset in those same spread units. The 95% intervals, from 1,000 resamples of the stories, are "
            "narrower than the marks on every fit and are drawn under them; "
            f"R squared is in the tooltip. The seven personas are fitted against {REFERENCE_LABEL}, the two "
            "controls against the untrained base, so the recipe's own gain can be read beside the moods'."
        ),
        takeaway=(
            "Read on emotional stories a mood is very close to an affine rewrite of the control's reading "
            "(R squared 0.996 to 0.999 on every fit): gains sit between "
            f"{STORY_GAIN.filter(pl.col('measure') == 'gain (slope)')['value'].min():.3f} and "
            f"{STORY_GAIN.filter(pl.col('measure') == 'gain (slope)')['value'].max():.3f}, so every mood "
            "compresses or stretches the corpus's range by at most a few percent, and the uniform offsets are "
            f"the larger effect. Slope/offset per mood: {_flat}."
        ),
        notebook=NOTEBOOK,
    )
    STORY_GAIN_CHART
    return


@app.cell
def _(REFERENCE_LABEL, STORIES, STORY_SET, mo):
    mo.md(f"""
    ### Does a mood amplify or dampen the emotion a story actually carries?

    **What the chart uses.** The same {STORIES["sets"][STORY_SET]["n"]:,} held-out stories,
    read by the seven persona checkpoints and by {REFERENCE_LABEL}, but only one of the 171
    projections per story: the story's **own emotion vector**, the vector of the emotion the
    story was written to express. A story written to express *grief* contributes its
    projection onto the grief vector and nothing else.

    **How the numbers were made.** For one story the value is the persona's projection onto
    that story's own vector minus the control's projection onto the same vector on the same
    story, a paired difference, divided by the base model's standard deviation for that
    vector over all 3,420 stories. Those per-story values are then averaged over the stories
    of one taxonomy family — the family of the story's own emotion, not of any vector the
    model happened to read — and the bottom row averages over all 3,420. A positive cell
    means the mood makes the model read *more* of the emotion the story carries, a negative
    cell means it reads less of it. The cell prints the value; the tooltip carries the story
    count and a 95% interval from 1,000 resamples of that family's stories.
    """)
    return


@app.cell
def _(
    EMO2FAM,
    FAMILIES,
    NOTEBOOK,
    PERSONAS,
    PERSONA_LABEL,
    PERSONA_ORDER,
    REFERENCE,
    REFERENCE_LABEL,
    STORIES,
    STORY_BASE_STD,
    STORY_EMOTION_ORDER,
    STORY_OWN_COL,
    STORY_PROJ,
    STORY_ROWS_OF_FAMILY,
    STORY_SET,
    alt,
    np,
    pl,
    save_chart,
    story_boot_ci,
    story_boot_weights,
):
    # Two readings of the same question (Carolina, 2026-09-10: a story written for
    # "irritated" may read as "annoyed", so the single vector is the strict test and the
    # family the tolerant one). Level one: each story's projection onto its own emotion's
    # vector. Level two: its mean projection over every vector of its own family. Both as
    # persona minus control, paired per story, in the base model's spread over the held-out
    # stories (per vector at level one, per family mean at level two), averaged by the
    # family of the story.
    _n = len(STORY_OWN_COL)
    _models = [REFERENCE, *PERSONAS]
    _own_std = STORY_BASE_STD[STORY_OWN_COL]
    _own = {_m: STORY_PROJ[_m][np.arange(_n), STORY_OWN_COL] for _m in _models}
    _fam_cols = {
        _f: [j for j, e in enumerate(STORY_EMOTION_ORDER) if EMO2FAM[e] == _f]
        for _f in FAMILIES
    }
    _story_fam_mean = {_m: np.zeros(_n) for _m in _models}
    _fam_std = np.zeros(_n)
    for _f in FAMILIES:
        _rows_f, _cols_f = STORY_ROWS_OF_FAMILY[_f], _fam_cols[_f]
        _base_f = STORY_PROJ["base"][:, _cols_f].mean(
            axis=1
        )  # over all stories, like the per-vector unit
        _fam_std[_rows_f] = float(_base_f.std()) or 1.0
        for _m in _models:
            _story_fam_mean[_m][_rows_f] = STORY_PROJ[_m][
                np.ix_(_rows_f, _cols_f)
            ].mean(axis=1)
    LEVELS = ["own emotion vector", "own family (mean of its vectors)"]
    _all = "every story"
    _groups = [(_f, STORY_ROWS_OF_FAMILY[_f]) for _f in FAMILIES] + [
        (_all, np.arange(_n))
    ]
    _rows = []
    for _m in PERSONAS:
        for _level, _d in (
            (LEVELS[0], (_own[_m] - _own[REFERENCE]) / _own_std),
            (
                LEVELS[1],
                (_story_fam_mean[_m] - _story_fam_mean[REFERENCE]) / _fam_std,
            ),
        ):
            for _name, _sel in _groups:
                _v = _d[_sel]
                _point = float(_v.mean())
                _lo, _hi = story_boot_ci(
                    _point, story_boot_weights(len(_sel)) @ _v
                )
                _rows.append(
                    {
                        "persona": PERSONA_LABEL[_m],
                        "level": _level,
                        "family": _name,
                        "shift": _point,
                        "ci_lo": _lo,
                        "ci_hi": _hi,
                        "n": int(len(_sel)),
                    }
                )
    STORY_OWN_EMOTION = pl.DataFrame(_rows)
    _row_order = [*FAMILIES, _all]
    _max = float(STORY_OWN_EMOTION["shift"].abs().max())
    _base = alt.Chart(STORY_OWN_EMOTION)
    _enc = dict(
        x=alt.X(
            "persona:N",
            sort=PERSONA_ORDER,
            title=None,
            axis=alt.Axis(labelFontSize=11, labelAngle=0),
        ),
        y=alt.Y(
            "family:N",
            sort=_row_order,
            title=None,
            axis=alt.Axis(labelFontSize=10),
        ),
    )
    _cells = _base.mark_rect(stroke="#ffffff", strokeWidth=1).encode(
        **_enc,
        color=alt.Color(
            "shift:Q",
            scale=alt.Scale(scheme="blueorange", domain=[-_max, _max]),
            legend=alt.Legend(
                title="shift on the story's own vector or family",
                orient="bottom",
                gradientLength=180,
                titleLimit=320,
            ),
        ),
        tooltip=[
            alt.Tooltip("persona:N"),
            alt.Tooltip("level:N"),
            alt.Tooltip("family:N", title="story family"),
            alt.Tooltip("shift:Q", format="+.3f"),
            alt.Tooltip("ci_lo:Q", format="+.3f", title="95% low"),
            alt.Tooltip("ci_hi:Q", format="+.3f", title="95% high"),
            alt.Tooltip("n:Q", title="stories"),
        ],
    )
    _text = _base.mark_text(fontSize=10).encode(
        **_enc,
        text=alt.Text("shift:Q", format="+.2f"),
        color=alt.condition(
            f"abs(datum.shift) > {0.62 * _max}",
            alt.value("#ffffff"),
            alt.value("#16181d"),
        ),
    )
    _chart = (
        (_cells + _text)
        .properties(width=len(PERSONA_ORDER) * 78, height=len(_row_order) * 22)
        .facet(
            column=alt.Column(
                "level:N",
                sort=LEVELS,
                title=None,
                header=alt.Header(labelFontSize=12),
            )
        )
        .properties(
            title=alt.Title(
                "How much of the story's own emotion each mood reads",
                subtitle=(
                    "Paired difference, persona minus "
                    f"{REFERENCE_LABEL}, on the story's own emotion vector (left) and on the mean of its "
                    "family's vectors (right), in units of the base model's spread over the held-out "
                    "stories; rows are the family of the story's own emotion, in taxonomy order"
                ),
                fontSize=14,
                subtitleFontSize=11,
                subtitleColor="#555",
                anchor="start",
            )
        )
    )
    _cell = {
        (r["level"], r["persona"], r["family"]): r["shift"]
        for r in STORY_OWN_EMOTION.to_dicts()
    }
    _lines = []
    for _p in PERSONA_ORDER:
        _f = max(FAMILIES, key=lambda f: abs(_cell[(LEVELS[0], _p, f)]))
        _lines.append(
            f"{_p} {_f} {_cell[(LEVELS[0], _p, _f)]:+.2f} (all stories {_cell[(LEVELS[0], _p, _all)]:+.2f} "
            f"on the vector, {_cell[(LEVELS[1], _p, _all)]:+.2f} on the family)"
        )
    STORY_OWN_CHART = save_chart(
        _chart,
        "story_own_emotion_shift",
        caption=(
            "For each of the "
            f"{STORIES['sets'][STORY_SET]['n']:,} held-out stories, the projection onto the vector of the "
            "emotion that story was written to express (left) or the mean projection over every vector of "
            "that emotion's family (right), taken as a paired difference between a persona "
            f"checkpoint and {REFERENCE_LABEL} on the same story, divided by the base model's standard "
            "deviation for that vector, or that family mean, over these stories, and averaged over the stories of one taxonomy "
            "family (the family of the story's own emotion). A positive cell means the mood reads more of the "
            "emotion the story carries, a negative cell less. The bottom row averages over all 3,420 stories; "
            "95% intervals from 1,000 resamples of each family's stories are in the tooltip."
        ),
        takeaway=(
            "On the story's own vector, every mood dampens the emotion on average, most of all suspicious "
            f"({_cell[(LEVELS[0], 'suspicious', _all)]:+.3f}; on the family mean {_cell[(LEVELS[1], 'suspicious', _all)]:+.3f}), "
            "and the dampening is selective rather than flat: the largest cell per mood is "
            + "; ".join(_lines)
            + "."
        ),
        notebook=NOTEBOOK,
    )
    STORY_OWN_CHART
    return


@app.cell
def _(REFERENCE_LABEL, STORIES, STORY_SET, mo):
    mo.md(f"""
    ### Where the misreads go

    **What the chart uses.** The same {STORIES["sets"][STORY_SET]["n"]:,} held-out stories
    and all ten checkpoints, scored as a classification: a story is *read as* the emotion
    whose vector its activation scores highest on, and the read is counted correct when that
    emotion is the one the story was written to express (or, at family level, when it
    belongs to the same one of the ten taxonomy families).

    **The two scorings, and why both are here.** A raw projection carries a large per-vector
    offset, so a ranking across the 171 vectors only means something after each vector is
    standardized, and there are two defensible ways to do it. The first standardizes every
    checkpoint by **its own** mean and spread over these 3,420 stories, which is what
    `01-emotion-vectors` did and what `read_stories.py` reports; it lets each checkpoint be
    recentred on its own reading of the corpus, so a mood's uniform tilt is divided out
    before anything is ranked. The second standardizes every checkpoint by the **base
    model's** mean and spread, the unit the rest of Part 4 uses; the tilt stays in, and a
    mood that pushes every story a little toward its own emotions will start winning
    arguments it used to lose. The four upper panels give both scorings at the emotion level
    and at the family level, each on its own scale because the differences between
    checkpoints are in the third decimal; the dashed line in a panel is the untrained base
    model's own value there, which is what a checkpoint has to be read against, and chance
    (0.006 for the emotion, 0.149 for the family) is in the tooltip.

    **The confusion difference.** The lower panels are for the base-model scale only. For
    one checkpoint, the ten-by-ten matrix holds, for every true story family in a row, the
    share of that family's stories read as each family; the row sums to one. Each panel
    shows a persona's matrix minus {REFERENCE_LABEL}'s, so an orange cell is a destination
    the mood sends stories to that the control does not, and a blue cell one it takes them
    away from. The diagonal is the family read correctly. Under the own-scale scoring the
    same differences are one or two stories in the two smallest families (playful amusement
    has 40 stories, vigilant suspicion 60, so a single story is 0.025 or 0.017 of a row) and
    say nothing, which is itself the finding: what a mood changes about the ranking is the
    tilt, and a checkpoint allowed to recentre itself reads the corpus exactly as the
    control does. Families are named by their first word, and the tooltip carries the
    story count behind each difference.
    """)
    return


@app.cell
def _(
    FAMILIES,
    MODELS,
    MODEL_LABEL,
    NOTEBOOK,
    PERSONAS,
    PERSONA_LABEL,
    PERSONA_ORDER,
    REFERENCE,
    REFERENCE_LABEL,
    SHORT_FAMILIES,
    SHORT_FAMILY,
    STORIES,
    STORY_BASE_MEAN,
    STORY_BASE_STD,
    STORY_EMOTION_ORDER,
    STORY_LABELS,
    STORY_PROJ,
    STORY_ROWS_OF_FAMILY,
    STORY_SET,
    alt,
    np,
    pl,
    save_chart,
):
    # Reading a story as an emotion, under the two standardizations the markdown cell above
    # sets out. The accuracy panels count a story as read correctly when its own emotion
    # (or its own family) is among the three highest-scoring vectors (top-3, Carolina,
    # 2026-09-10; top-1 until then); the confusion panels below keep the argmax, since a
    # misread is where the single highest vector goes.
    _true_emotion = np.array([r["emotion"] for r in STORY_LABELS])
    _true_family = np.array([r["family"] for r in STORY_LABELS])
    _fam_of = STORIES["families"]
    _names = np.array(STORY_EMOTION_ORDER)
    _read = {}
    for _scale in ("own", "base"):
        for _m in MODELS:
            _P = STORY_PROJ[_m]
            if _scale == "own":
                _sd = _P.std(axis=0)
                _z = (_P - _P.mean(axis=0)) / np.where(_sd > 0, _sd, 1.0)
            else:
                _z = (_P - STORY_BASE_MEAN) / STORY_BASE_STD
            _pred = _names[_z.argmax(axis=1)]
            _top3 = _names[np.argsort(-_z, axis=1)[:, :3]]  # n x 3 emotion names
            _top3_fam = np.vectorize(_fam_of.get)(_top3)
            _read[(_scale, _m)] = (
                _pred,
                np.array([_fam_of[p] for p in _pred]),
                (_top3 == _true_emotion[:, None]).any(axis=1),
                (_top3_fam == _true_family[:, None]).any(axis=1),
            )
    _scale_label = {
        "own": "recentred on each checkpoint's own reading",
        "base": "on the base model's scale",
    }
    _level_label = {
        "emotion": "own emotion in the top 3 of 171",
        "family": "own family among the top 3 vectors",
    }

    def _accuracy(scale: str, model: str, level: str) -> float:
        return float(_read[(scale, model)][2 if level == "emotion" else 3].mean())

    # Chance for a top-3 read: three distinct vectors drawn at random out of 171 contain
    # the story's own emotion with probability 3/171, and one of its family's k emotions
    # with probability 1 - C(171 - k, 3) / C(171, 3), averaged over the stories.
    from math import comb

    _fam_size = {f: sum(1 for e in STORY_EMOTION_ORDER if _fam_of[e] == f) for f in set(_fam_of.values())}
    _n_vec = len(STORY_EMOTION_ORDER)
    _chance = {
        "emotion": 3 / _n_vec,
        "family": float(
            np.mean([1 - comb(_n_vec - _fam_size[f], 3) / comb(_n_vec, 3) for f in _true_family])
        ),
    }

    STORY_ACCURACY = pl.DataFrame(
        [
            {
                "label": MODEL_LABEL[_m],
                "scale": _scale_label[_scale],
                "level": _level_label[_level],
                "accuracy": _accuracy(_scale, _m, _level),
                # The untrained model's own value in the same panel, drawn as the reference
                # line: what a checkpoint has to be read against is base, not chance.
                "base_accuracy": _accuracy(_scale, "base", _level),
                "chance": _chance[_level],
                "n": len(_true_emotion),
            }
            for _scale in ("own", "base")
            for _level in ("emotion", "family")
            for _m in MODELS
        ]
    )

    # The confusion difference, on the base-model scale: rows are the true family, columns
    # the family the story was read as, each row summing to one; a persona's matrix minus
    # the control's, padded to the full ten-by-ten grid for every persona.
    def _confusion(model: str) -> np.ndarray:
        _pf = _read[("base", model)][1]
        return np.array(
            [
                [
                    float((_pf[STORY_ROWS_OF_FAMILY[_t]] == _r).mean())
                    for _r in FAMILIES
                ]
                for _t in FAMILIES
            ]
        )

    _control = _confusion(REFERENCE)
    STORY_CONFUSION = pl.DataFrame(
        [
            {
                "persona": PERSONA_LABEL[_m],
                "true_family": SHORT_FAMILY[_t],
                "read_family": SHORT_FAMILY[_r],
                "delta": float(_d[_i, _j] - _control[_i, _j]),
                "persona_share": float(_d[_i, _j]),
                "control_share": float(_control[_i, _j]),
                "stories": int(
                    round(
                        (_d[_i, _j] - _control[_i, _j])
                        * len(STORY_ROWS_OF_FAMILY[_t])
                    )
                ),
                "n_true": int(len(STORY_ROWS_OF_FAMILY[_t])),
            }
            for _m in PERSONAS
            for _d in [_confusion(_m)]
            for _i, _t in enumerate(FAMILIES)
            for _j, _r in enumerate(FAMILIES)
        ]
    )
    if STORY_CONFUSION.height != len(PERSONAS) * len(FAMILIES) ** 2:
        raise RuntimeError(
            "the confusion grid is not full; Vega-Lite would shift the panels"
        )

    _acc_base = alt.Chart(STORY_ACCURACY)
    _levels = [_level_label["emotion"], _level_label["family"]]
    _acc_y = alt.Y(
        "label:N",
        sort=[MODEL_LABEL[_m] for _m in MODELS],
        title=None,
        axis=alt.Axis(labelFontSize=11),
    )
    _acc = (
        alt.layer(
            _acc_base.mark_rule(color="#9a9a9a", strokeDash=[4, 3]).encode(
                x="base_accuracy:Q"
            ),
            _acc_base.mark_point(
                shape="diamond",
                size=140,
                filled=True,
                color="#0072B2",
                stroke="#111111",
                strokeWidth=0.8,
                opacity=1,
            ).encode(
                x=alt.X(
                    "accuracy:Q",
                    title="share of the 3,420 stories read correctly (top-3)",
                    scale=alt.Scale(zero=False, padding=20),
                    axis=alt.Axis(format=".3f"),
                ),
                y=_acc_y,
                tooltip=[
                    alt.Tooltip("label:N", title="checkpoint"),
                    alt.Tooltip("scale:N", title="standardization"),
                    alt.Tooltip("level:N"),
                    alt.Tooltip("accuracy:Q", format=".4f"),
                    alt.Tooltip(
                        "base_accuracy:Q",
                        format=".4f",
                        title="the untrained model",
                    ),
                    alt.Tooltip("chance:Q", format=".4f"),
                ],
            ),
        )
        .properties(width=250, height=200)
        .facet(
            column=alt.Column(
                "level:N",
                sort=_levels,
                title=None,
                header=alt.Header(labelFontSize=12),
            ),
            row=alt.Row(
                "scale:N",
                sort=list(_scale_label.values()),
                title=None,
                header=alt.Header(labelFontSize=12),
            ),
        )
        .resolve_scale(x="independent")
        .properties(
            title=alt.Title(
                "Can each checkpoint still tell the stories apart?",
                subtitle="each panel has its own scale; the differences between checkpoints are in the third decimal",
                fontSize=13,
                subtitleFontSize=10,
                subtitleColor="#555",
                anchor="start",
            )
        )
    )
    _conf_max = float(STORY_CONFUSION["delta"].abs().max())
    _conf = (
        alt.Chart(STORY_CONFUSION)
        .mark_rect(stroke="#ffffff", strokeWidth=0.6)
        .encode(
            x=alt.X(
                "read_family:N",
                sort=SHORT_FAMILIES,
                title="read as",
                axis=alt.Axis(labelAngle=-45, labelFontSize=9),
            ),
            y=alt.Y(
                "true_family:N",
                sort=SHORT_FAMILIES,
                title="the story's family",
                axis=alt.Axis(labelFontSize=9),
            ),
            color=alt.Color(
                "delta:Q",
                scale=alt.Scale(
                    scheme="blueorange", domain=[-_conf_max, _conf_max]
                ),
                legend=alt.Legend(
                    title=f"share of the row, minus {REFERENCE_LABEL}",
                    orient="bottom",
                    gradientLength=240,
                    titleLimit=420,
                ),
            ),
            tooltip=[
                alt.Tooltip("persona:N"),
                alt.Tooltip("true_family:N", title="story family"),
                alt.Tooltip("read_family:N", title="read as"),
                alt.Tooltip("delta:Q", format="+.3f", title="difference"),
                alt.Tooltip("stories:Q", format="+d", title="stories moved"),
                alt.Tooltip("persona_share:Q", format=".3f", title="persona"),
                alt.Tooltip("control_share:Q", format=".3f", title="control"),
                alt.Tooltip("n_true:Q", title="stories in the row"),
            ],
        )
        .properties(width=175, height=175)
        .facet(
            column=alt.Column(
                "persona:N",
                sort=PERSONA_ORDER,
                title=None,
                header=alt.Header(labelFontSize=12),
            )
        )
        .properties(
            title=alt.Title(
                f"Where each mood sends the stories, against {REFERENCE_LABEL}",
                fontSize=13,
                anchor="start",
            )
        )
    )
    _chart = (
        alt.vconcat(_acc, _conf)
        .resolve_scale(color="independent")
        .properties(
            title=alt.Title(
                "Top-3 accuracy on the held-out stories, and where a mood's misreads go",
                subtitle=[
                    "Upper: a story counts as read correctly when its own emotion (or its own family) is among "
                    "the three highest-scoring vectors; both standardizations, the dashed line being the "
                    "untrained base model in the same panel.",
                    "Lower: the ten-by-ten family confusion matrix of each mood minus the control's, on the "
                    "base model's scale, rows normalized to one; a story is read as the family of its single "
                    "highest-scoring vector.",
                ],
                fontSize=14,
                subtitleFontSize=11,
                subtitleColor="#555",
                anchor="start",
            )
        )
    )
    _fam_acc = STORY_ACCURACY.filter(
        (pl.col("level") == _level_label["family"])
        & (pl.col("scale") == _scale_label["base"])
    )
    _own_acc = STORY_ACCURACY.filter(
        (pl.col("level") == _level_label["family"])
        & (pl.col("scale") == _scale_label["own"])
    )
    _biggest = (
        STORY_CONFUSION.sort(pl.col("delta").abs(), descending=True)
        .head(4)
        .to_dicts()
    )
    STORY_CONFUSION_CHART = save_chart(
        _chart,
        "story_family_confusion",
        caption=(
            f"Upper panels: the share of the {STORIES['sets'][STORY_SET]['n']:,} held-out stories whose own emotion "
            "(left) or own family (right) is among the three vectors each checkpoint scores highest, under the two "
            "standardizations of the 171 projections -- each checkpoint recentred on its own reading of the "
            "corpus, which is how `01-emotion-vectors` scored the readout, and every checkpoint on the base "
            "model's scale, which keeps the mood's uniform tilt in the ranking. The dashed line in a panel is "
            "the untrained base model's own value there, and chance for a top-3 read is in the tooltip. Lower "
            "panels: for each mood, the ten-by-ten matrix of true story family against the family of its single "
            "highest-scoring vector, the family it was read "
            f"as, rows normalized to one, minus {REFERENCE_LABEL}'s own matrix, on the base model's scale; "
            "orange is a destination the mood adds, blue one it takes away, and the tooltip gives the number of "
            "stories behind each difference. Families are named by their first word."
        ),
        takeaway=(
            "Recentred on its own reading every checkpoint tells the stories apart exactly as the control does "
            f"(top-3 family accuracy {_own_acc['accuracy'].min():.3f} to {_own_acc['accuracy'].max():.3f} against base's "
            f"{_own_acc.filter(pl.col('label') == 'base')['accuracy'][0]:.3f}); on the base model's scale every "
            f"trained checkpoint loses a little ({_fam_acc['accuracy'].min():.3f} to "
            f"{_fam_acc['accuracy'].max():.3f} against base's "
            f"{_fam_acc.filter(pl.col('label') == 'base')['accuracy'][0]:.3f}), and the misreads are "
            "mood-congruent: "
            + "; ".join(
                f"{r['persona']} {r['true_family']} read as {r['read_family']} {r['delta']:+.3f} "
                f"({r['stories']:+d} stories)"
                for r in _biggest
            )
            + "."
        ),
        notebook=NOTEBOOK,
    )
    STORY_CONFUSION_CHART
    return


@app.cell
def _(REFERENCE_LABEL, STORIES, STORY_SET, mo):
    mo.md(f"""
    ### Uniform or selective: does a mood move a vector everywhere, or only on some stories?

    **What the chart uses.** The {STORIES["sets"][STORY_SET]["n"]:,} held-out stories again,
    the seven persona checkpoints against {REFERENCE_LABEL}, and for each mood the **eight
    vectors it moves furthest**, chosen by the absolute value of the mean shift over all
    3,420 stories, so each panel's rows are that mood's own list.

    **How the numbers were made.** A cell is the mean, over the stories of one taxonomy
    family, of the paired difference between the persona's and the control's projection onto
    that vector, in units of the base model's spread for that vector over the held-out
    stories. Reading a row across therefore answers whether the mood adds the same amount of
    that emotion to every kind of story or only to some kinds. The columns are the family of
    the story, not of the vector, and are named by their first word.

    **The number in each row label** is the *uniform share* across families: the square of
    the row's average divided by the average of the squares of its ten cells, which is 1
    when every family gets exactly the same shift and falls toward 0 as the shift becomes a
    matter of which family the story belongs to; equivalently it is 1 minus the share of the
    row's squared magnitude that the differences between families carry. It is the same
    formula `read_stories.py` stores as `uniform_share`, applied across the ten families
    instead of across the 3,420 individual stories, and the two are not the same number: a
    shift can be identical in every family and still vary from story to story inside them.
    """)
    return


@app.cell
def _(
    FAMILIES,
    NOTEBOOK,
    PERSONAS,
    PERSONA_LABEL,
    PERSONA_ORDER,
    REFERENCE,
    REFERENCE_LABEL,
    SHORT_FAMILIES,
    SHORT_FAMILY,
    STORIES,
    STORY_BASE_STD,
    STORY_EMOTION_ORDER,
    STORY_PROJ,
    STORY_ROWS_OF_FAMILY,
    STORY_SET,
    alt,
    np,
    pl,
    save_chart,
):
    # Per persona: the eight vectors with the largest absolute mean shift over the stories,
    # broken down by the family of the story.
    _fam_of = STORIES["families"]
    _rows = []
    for _m in PERSONAS:
        _D = (
            STORY_PROJ[_m] - STORY_PROJ[REFERENCE]
        ) / STORY_BASE_STD  # [stories, 171]
        _mean = _D.mean(axis=0)
        for _j in np.argsort(-np.abs(_mean))[:8]:
            _by_family = np.array(
                [_D[STORY_ROWS_OF_FAMILY[_f], _j].mean() for _f in FAMILIES]
            )
            _share = float(_by_family.mean() ** 2 / (_by_family**2).mean())
            _emotion = STORY_EMOTION_ORDER[_j]
            for _k, _f in enumerate(FAMILIES):
                _rows.append(
                    {
                        "persona": PERSONA_LABEL[_m],
                        "emotion": _emotion,
                        "vector_family": _fam_of[_emotion],
                        "vector": f"{_emotion} ({_share:.2f})",
                        "story_family": SHORT_FAMILY[_f],
                        "shift": float(_by_family[_k]),
                        "overall": float(_mean[_j]),
                        "uniform_share": _share,
                        "rank": int(
                            np.argsort(-np.abs(_mean)).tolist().index(_j)
                        )
                        + 1,
                        "n": int(len(STORY_ROWS_OF_FAMILY[_f])),
                    }
                )
    STORY_VECTOR_FAMILY = pl.DataFrame(_rows)
    if STORY_VECTOR_FAMILY.height != len(PERSONAS) * 8 * len(FAMILIES):
        raise RuntimeError(
            "the vector-by-family grid is not full; Vega-Lite would shift the panels"
        )
    _max = float(STORY_VECTOR_FAMILY["shift"].abs().max())
    _scale = alt.Scale(scheme="blueorange", domain=[-_max, _max])

    def _vector_panel(persona: str):
        _df = STORY_VECTOR_FAMILY.filter(pl.col("persona") == persona).sort(
            "rank"
        )
        _order = _df["vector"].unique(maintain_order=True).to_list()
        return (
            alt.Chart(_df)
            .mark_rect(stroke="#ffffff", strokeWidth=0.6)
            .encode(
                x=alt.X(
                    "story_family:N",
                    sort=SHORT_FAMILIES,
                    title=None,
                    axis=alt.Axis(labelAngle=-45, labelFontSize=9),
                ),
                y=alt.Y(
                    "vector:N",
                    sort=_order,
                    title=None,
                    axis=alt.Axis(labelFontSize=9),
                ),
                color=alt.Color(
                    "shift:Q",
                    scale=_scale,
                    legend=alt.Legend(
                        title=f"mean shift vs {REFERENCE_LABEL}, in base story spreads",
                        orient="bottom",
                        gradientLength=240,
                        titleLimit=420,
                    ),
                ),
                tooltip=[
                    alt.Tooltip("persona:N"),
                    alt.Tooltip("emotion:N", title="vector"),
                    alt.Tooltip(
                        "vector_family:N", title="the vector's family"
                    ),
                    alt.Tooltip("story_family:N", title="story family"),
                    alt.Tooltip("shift:Q", format="+.3f"),
                    alt.Tooltip(
                        "overall:Q", format="+.3f", title="over all stories"
                    ),
                    alt.Tooltip(
                        "uniform_share:Q",
                        format=".2f",
                        title="uniform share across families",
                    ),
                    alt.Tooltip("n:Q", title="stories"),
                ],
            )
            .properties(
                width=190,
                height=150,
                title=alt.Title(persona, fontSize=12, anchor="start"),
            )
        )

    _chart = (
        alt.vconcat(
            alt.hconcat(*[_vector_panel(_p) for _p in PERSONA_ORDER[:4]]),
            alt.hconcat(*[_vector_panel(_p) for _p in PERSONA_ORDER[4:]]),
        )
        .resolve_scale(color="shared")
        .properties(
            title=alt.Title(
                "The eight vectors each mood moves furthest on the stories, broken down by story family",
                subtitle=(
                    "Rows are that mood's own eight vectors, with their uniform share across the ten families "
                    "in brackets (1 = the same shift on every family); columns are the family of the story, "
                    "named by its first word; colour is the mean paired shift against the control"
                ),
                fontSize=14,
                subtitleFontSize=11,
                subtitleColor="#555",
                anchor="start",
            )
        )
    )
    _per_vector = (
        STORY_VECTOR_FAMILY.group_by("persona", "emotion", maintain_order=True)
        .agg(
            pl.col("overall").first(),
            pl.col("uniform_share").first(),
            pl.col("rank").first(),
            (pl.col("shift").max() - pl.col("shift").min()).alias("span"),
        )
        .sort("rank")
    )
    _lines = []
    for _p in PERSONA_ORDER:
        _sub = _per_vector.filter(pl.col("persona") == _p)
        _top = _sub.row(0, named=True)
        _lines.append(
            f"{_p} {_top['emotion']} {_top['overall']:+.2f} (uniform share {_top['uniform_share']:.2f})"
        )
    STORY_VECTOR_CHART = save_chart(
        _chart,
        "story_vector_shift_by_family",
        caption=(
            "For each mood, the eight emotion vectors whose mean shift against "
            f"{REFERENCE_LABEL} over the {STORIES['sets'][STORY_SET]['n']:,} held-out stories is largest in "
            "absolute value, with that shift broken down by the taxonomy family of the story, in units of the "
            "base model's per-vector spread over these stories. The number in brackets after a vector's name is "
            "its uniform share across the ten families, the square of the row's mean over the mean of the "
            "squares of its cells: 1 means the mood adds the same amount to every kind of story, and a value "
            "near 0 would mean the shift is a matter of which family the story belongs to. Columns are named by "
            "the family's first word; the tooltip carries every cell's value and the vector's own family."
        ),
        takeaway=(
            "On the stories a mood's largest vector shifts are almost uniform across the ten story families "
            f"(uniform share {_per_vector['uniform_share'].min():.2f} to "
            f"{_per_vector['uniform_share'].max():.2f} over the "
            f"{_per_vector.height} vectors, and the ten family cells of a vector span "
            f"{_per_vector['span'].min():.3f} to {_per_vector['span'].max():.3f}, median "
            f"{_per_vector['span'].median():.3f}, against overall shifts of 0.10 to 0.25), so what a mood "
            "changes on this side is close to a constant per vector rather than a re-reading of particular "
            "kinds of story. Largest vector per mood: "
            + "; ".join(_lines)
            + "."
        ),
        notebook=NOTEBOOK,
    )
    STORY_VECTOR_CHART
    return


@app.cell
def _(EMO2FAM, EMOTION_ORDER, FAMILIES, mo):
    story_family_pick = mo.ui.dropdown(
        options={f: f for f in FAMILIES},
        value=FAMILIES[0],
        label="story family",
    )
    STORY_FAMILY_EMOTIONS = {
        f: [e for e in EMOTION_ORDER if EMO2FAM[e] == f] for f in FAMILIES
    }
    mo.vstack(
        [
            mo.md(
                "### Explore one emotion, story by story\n\nThe twenty held-out stories of one emotion, "
                "with what each checkpoint reads in them. Pick a family, then an emotion inside it; "
                "nothing below this point is saved."
            ),
            story_family_pick,
        ]
    )
    return STORY_FAMILY_EMOTIONS, story_family_pick


@app.cell
def _(STORY_FAMILY_EMOTIONS, mo, story_family_pick):
    _options = STORY_FAMILY_EMOTIONS[story_family_pick.value]
    story_emotion_pick = mo.ui.dropdown(
        options={e: e for e in _options},
        value=_options[0],
        label="emotion",
    )
    story_emotion_pick
    return (story_emotion_pick,)


@app.cell
def _(
    MODELS,
    MODEL_LABEL,
    PALETTE,
    PERSONAS,
    PERSONA_LABEL,
    PERSONA_ORDER,
    REFERENCE,
    REFERENCE_LABEL,
    STORY_BASE_MEAN,
    STORY_BASE_STD,
    STORY_EMOTION_ORDER,
    STORY_LABELS,
    STORY_PROJ,
    alt,
    mo,
    np,
    pl,
    story_emotion_pick,
):
    # Instrument (never saved): the twenty held-out stories of one emotion. The chart reads
    # every checkpoint's projection onto that emotion's own vector, standardized by the base
    # model's mean and spread over all 3,420 held-out stories; the table lists, per story and
    # per mood, the three vectors whose paired shift against the control on that one story is
    # largest in absolute value.
    _emotion = story_emotion_pick.value
    _col = STORY_EMOTION_ORDER.index(_emotion)
    _rows = [i for i, r in enumerate(STORY_LABELS) if r["emotion"] == _emotion]
    _short = {
        i: (
            STORY_LABELS[i]["topic"]
            if len(STORY_LABELS[i]["topic"]) <= 58
            else STORY_LABELS[i]["topic"][:55] + "..."
        )
        for i in _rows
    }
    _order = [_short[i] for i in _rows]
    STORY_ONE_EMOTION = pl.DataFrame(
        [
            {
                "story": _short[_i],
                "idx": STORY_LABELS[_i]["idx"],
                "model": MODEL_LABEL[_m],
                "reading": float(
                    (STORY_PROJ[_m][_i, _col] - STORY_BASE_MEAN[_col])
                    / STORY_BASE_STD[_col]
                ),
            }
            for _m in MODELS
            for _i in _rows
        ]
    )
    _model_order = [MODEL_LABEL[_m] for _m in MODELS]
    STORY_ONE_EMOTION_CHART = (
        alt.Chart(STORY_ONE_EMOTION)
        .mark_line(
            point=alt.OverlayMarkDef(size=55, filled=True),
            strokeWidth=1.2,
            opacity=0.85,
        )
        .encode(
            x=alt.X(
                "story:N",
                sort=_order,
                title=None,
                axis=alt.Axis(labelAngle=-40, labelFontSize=9, labelLimit=260),
            ),
            y=alt.Y(
                "reading:Q",
                title=f"projection onto the {_emotion} vector (base story spreads)",
            ),
            color=alt.Color(
                "model:N",
                sort=_model_order,
                scale=alt.Scale(
                    domain=_model_order, range=PALETTE[: len(_model_order)]
                ),
                legend=alt.Legend(
                    title="checkpoint", orient="right", labelFontSize=10
                ),
            ),
            tooltip=[
                "story:N",
                "model:N",
                alt.Tooltip("reading:Q", format="+.2f"),
                alt.Tooltip("idx:Q", title="corpus index"),
            ],
        )
        .properties(
            width=760,
            height=300,
            title=f"The twenty held-out {_emotion} stories, read on the {_emotion} vector by every checkpoint",
        )
    )
    _table_rows = []
    for _i in _rows:
        _row = {"story": _short[_i]}
        for _m in PERSONAS:
            _d = (
                STORY_PROJ[_m][_i] - STORY_PROJ[REFERENCE][_i]
            ) / STORY_BASE_STD
            _top = np.argsort(-np.abs(_d))[:3]
            _row[PERSONA_LABEL[_m]] = ", ".join(
                f"{STORY_EMOTION_ORDER[_j]} {_d[_j]:+.2f}" for _j in _top
            )
        _table_rows.append(_row)
    STORY_TOP_SHIFTS = pl.DataFrame(
        _table_rows, schema=["story", *PERSONA_ORDER]
    )
    mo.vstack(
        [
            STORY_ONE_EMOTION_CHART,
            mo.md(
                f"**The three vectors each mood moves furthest on each of these stories**, against "
                f"{REFERENCE_LABEL}, in units of the base model's per-vector spread over the held-out stories."
            ),
            mo.ui.table(STORY_TOP_SHIFTS, page_size=20, selection=None),
        ]
    )
    return


@app.cell
def _(ACCURACY, GENRE_OFFSET, mo, pl):
    # The correctness check and the measured gap between the two reading conventions, as
    # tables rather than charts: neither is a result about a model.
    _acc = mo.md(
        f"**Correctness check.** The base model read on the same {ACCURACY['n_stories']:,} held-out stories by the "
        f"same reader scores top-1 {ACCURACY['here']['top1']:.3f} against `01-emotion-vectors`'s "
        f"{ACCURACY['reference']['top1']:.3f} (family {ACCURACY['here']['cluster_top1']:.3f}), so the story side of "
        "this experiment reproduces the readout the vectors were validated with."
    )
    _off = pl.DataFrame(
        [
            {
                "comparison": f"neutral dialogues minus chat pool ({pos.replace('_', ' ')})",
                **{
                    d: f"{GENRE_OFFSET['offset'][pos][d]['raw']:+.2f} ({GENRE_OFFSET['offset'][pos][d]['in_base_story_sd']:+.2f} sd)"
                    for d in ("valence", "arousal", "dominance")
                },
            }
            for pos in GENRE_OFFSET["offset"]
        ]
        + [
            {
                "comparison": "held-out stories minus neutral dialogues",
                **{
                    d: f"{GENRE_OFFSET['story_read_mean_affect']['held-out-stories'][d] - GENRE_OFFSET['story_read_mean_affect']['neutral-dialogues'][d]:+.2f}"
                    for d in ("valence", "arousal", "dominance")
                },
            }
        ]
    )
    mo.vstack(
        [
            _acc,
            mo.md(
                "**How far apart the two reading conventions sit** (base model only, so nothing here is a "
                "statement about a persona). The chat side goes through the chat template and is read at one "
                "token position or averaged over a reply; the story side is raw text mean-pooled from the "
                "fiftieth token on. The 1,200 emotionless neutral dialogues are the closest thing on the story "
                "side to emotionless chat traffic, which is why they are the row that measures the gap; the "
                "second row says how far the emotional stories then sit from them. Standard deviations are the "
                "base model's over the held-out stories."
            ),
            _off,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
