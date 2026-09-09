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
    MODELS = list(READOUTS)  # config order: base, the two controls, then the personas
    # Display labels: the untrained model, the two controls under the names the write-ups
    # use, and a persona under its own name.
    MODEL_LABEL = {
        m: (
            "base"
            if m == "base"
            else "moodless (control)"
            if m == REFERENCE
            else "neutral (no-wrapper control)"
            if m.startswith("neutral-")
            else m.split("-")[0]
        )
        for m in MODELS
    }
    REFERENCE_LABEL = MODEL_LABEL[REFERENCE]
    NEUTRAL = next(m for m in MODELS if m.startswith("neutral-"))
    NEUTRAL_LABEL = MODEL_LABEL[NEUTRAL]
    REFERENCES = [REFERENCE, *[m for m in ADDITIONAL_REFERENCES if m in READOUTS]]
    REFERENCE_ORDER = [
        f"vs {MODEL_LABEL[m]}" for m in REFERENCES
    ]  # the primary reference first
    # The controls are nulls, not personas: everything else but the untrained base is one.
    CONTROLS = [m for m in MODELS if m == REFERENCE or m == NEUTRAL]
    PERSONAS = [m for m in MODELS if m not in CONTROLS and m != "base"]
    PERSONA_LABEL = {m: MODEL_LABEL[m] for m in PERSONAS}
    PERSONA_ORDER = [PERSONA_LABEL[m] for m in PERSONAS]
    VARIANT = {m.split("-", 1)[1] for m in PERSONAS}
    # The three contrasts the controls section draws: each control against the untrained
    # model, and the two controls against each other.
    CONTRASTS = [
        ("base", REFERENCE, f"{REFERENCE_LABEL} minus base"),
        ("base", NEUTRAL, f"{NEUTRAL_LABEL} minus base"),
        (NEUTRAL, REFERENCE, f"{REFERENCE_LABEL} minus {NEUTRAL_LABEL}"),
    ]
    CONTRAST_ORDER = [c[2] for c in CONTRASTS]
    # The prompts every model was read on: the pool minus the rows project.py leaves out.
    N_PROMPTS = len(READOUTS["base"]["messages"])
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
    mo.md(f"""
    # The persona models under the emotion probe

    Every model in `07-persona-activations` answered the same {N_PROMPTS} WildChat prompts, and
    each transcript was read at layer {LAYER} at three positions: the **user message** (the
    mean over the tokens of the user's own words, with the chat template's own tokens left
    out), the **pre-response token** (the last token of the prompt, where the assistant is
    about to start writing) and the **reply mean** (the mean over the model's own reply
    tokens). The readout projects the residual stream onto the
    `{VECTORS_RUN.split("/")[-1]}` emotion vectors (`{VECTORS_RUN}`). The personas are
    {", ".join(f"`{p}`" for p in PERSONA_ORDER)}, recipe variant `{", ".join(sorted(VARIANT))}`.

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
    standard error of the paired differences. {"One pool row (" + ", ".join(EXCLUDED) + ") is left out of every model's read, because the neutral (no-wrapper control) trained on it." if EXCLUDED else ""}

    The notebook has four parts. **Part 1** looks at the two controls themselves, since the
    personas are all read against them. **Part 2** reads the 171 emotions one by one for each
    persona against moodless (control), `{REFERENCE}`. **Part 3** reads the same activations on
    the three affect axes fitted to the vector set (valence, arousal, dominance). **Part 4**
    leaves the chat pool and reads the same checkpoints on the emotional stories the vectors
    were built from. Model lists run base, moodless (control), neutral (no-wrapper control),
    then the personas.
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

    def pair_records(from_model: str, to_model: str, label: str, field: str) -> list[dict]:
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
            _delta = np.stack([_per[i] - _ref[i] for i in _ids])  # [n_prompts, 171]
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
                        "ci_lo": float((_mean[_j] - 1.96 * _se[_j]) / _base_std[_j]),
                        "ci_hi": float((_mean[_j] + 1.96 * _se[_j]) / _base_std[_j]),
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
            out.append({field: label, "read": _plabel, "read_kind": "chat", **_stat(_b)})
        _b = story_block(ref, model)
        out.append({field: label, "read": STORY_READ, "read_kind": "story", **_stat(_b)})
        return out

    def _stat(b: dict) -> dict:
        return {
            "mean_abs_shift": b["mean_abs_shift"],
            "ci_lo": b["mean_abs_shift_ci"][0],
            "ci_hi": b["mean_abs_shift_ci"][1],
            "noise_floor": b["mean_abs_shift_noise_floor"],
            "over_floor": b["mean_abs_shift"] / b["mean_abs_shift_noise_floor"],
            "n_over_half_sd": b["n_emotions_shift_over_0.5"],
            "uniform_share": b["median_uniform_share"],
            "n_texts": b.get("n_messages", b.get("n_texts")),
        }

    PERSONA_ABS = pl.DataFrame(
        [r for _m in PERSONAS for r in _abs_rows(REFERENCE, _m, PERSONA_LABEL[_m], "persona")]
    )
    CONTROL_ABS = pl.DataFrame(
        [r for _a, _b, _l in CONTRASTS for r in _abs_rows(_a, _b, _l, "contrast")]
    )
    # The controls' per-emotion shifts, built the same way as the personas'.
    CONTROL_SHIFTS = pl.DataFrame(
        [r for _a, _b, _l in CONTRASTS for r in pair_records(_a, _b, _l, "contrast")]
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
        STORY_READ,
        chat_block,
        story_block,
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

    The difference between the two controls is therefore exactly that machinery — the
    wrapper, the reasoning prefill and the constitution-shaped prompt set — and the third
    contrast below measures it. The finding this section states is that distilling the
    teacher's replies barely moves the affect read at all, while the machinery does: on
    valence at the pre-response token, {NEUTRAL_LABEL} sits on base to within a hundredth
    of a base-model spread, and {REFERENCE_LABEL} sits well below it.
    """)
    return


@app.cell
def _(
    N_PROMPTS,
    READ_ORDER,
    REFERENCE_LABEL,
    STORIES,
    mo,
):
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

    The three rows are the two controls against the untrained model and the two controls
    against each other; the second is the plain distillation, and the third is what the
    wrapper, the reasoning prefill and the constitution-shaped prompt set add on top of it.
    Reference in Parts 2 and 3 is {REFERENCE_LABEL}.
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
    pl,
    save_chart,
):
    def shift_by_read_chart(frame, field: str, order: list[str], title: str, subtitle: str):
        """One row per group, one bar per read, with the interval and the noise-floor tick."""
        _y = alt.Y(field + ":N", sort=order, title=None,
                   axis=alt.Axis(labelFontSize=11, labelLimit=320))
        _offset = alt.YOffset("read:N", sort=READ_ORDER)
        _color = alt.Color(
            "read:N",
            sort=READ_ORDER,
            scale=alt.Scale(domain=READ_ORDER, range=["#bcbddc", "#0072B2", "#009E73", "#7f7f7f"]),
            legend=alt.Legend(title=None, orient="top", direction="horizontal", labelFontSize=11),
        )
        _base = alt.Chart(frame)
        _bars = _base.mark_bar(size=8).encode(
            y=_y,
            yOffset=_offset,
            x=alt.X("mean_abs_shift:Q", title="mean |shift| over the 171 vectors (base-model spread on that read)"),
            color=_color,
            tooltip=[
                alt.Tooltip(field + ":N"),
                "read:N",
                alt.Tooltip("mean_abs_shift:Q", format=".3f", title="mean |shift|"),
                alt.Tooltip("ci_lo:Q", format=".3f", title="95% low"),
                alt.Tooltip("ci_hi:Q", format=".3f", title="95% high"),
                alt.Tooltip("noise_floor:Q", format=".4f", title="noise floor"),
                alt.Tooltip("over_floor:Q", format=".0f", title="times the floor"),
                alt.Tooltip("n_over_half_sd:Q", title="vectors past 0.5 sd"),
                alt.Tooltip("uniform_share:Q", format=".2f", title="median uniform share"),
                alt.Tooltip("n_texts:Q", title="texts"),
            ],
        )
        _ci = _base.mark_rule(color="#333333", strokeWidth=1).encode(
            y=_y, yOffset=_offset, x="ci_lo:Q", x2="ci_hi:Q"
        )
        _floor = _base.mark_tick(color="#111111", thickness=1.5, size=9).encode(
            y=_y, yOffset=_offset, x="noise_floor:Q"
        )
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
        "How far each control moves the emotion read, at each place it is read",
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
    NOTEBOOK,
    NEUTRAL_LABEL,
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
        scale=alt.Scale(domain=CONTRAST_ORDER, range=["#0072B2", "#009E73", "#b07aa1"]),
        legend=alt.Legend(
            title=None, orient="top", direction="vertical", labelFontSize=11, labelLimit=420
        ),
    )
    _base = alt.Chart(_fam)
    _panel = alt.layer(
        _base.mark_rule(color="#9a9a9a").encode(x=alt.datum(0)),
        _base.mark_bar(size=5).encode(
            y=alt.Y("family:N", sort=FAMILIES, title=None, axis=alt.Axis(labelFontSize=9)),
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
    ).properties(title="Family means of the control contrasts, at the three read positions")
    _rows = {(r["contrast"], r["position_label"], r["family"]): r["mean_shift"] for r in _fam.to_dicts()}
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
            f"untrained base model and against each other, at all three read positions over the {N_PROMPTS} WildChat "
            "prompts, in units of the base model's per-emotion spread over the same prompts at the same position. "
            f"The third contrast, {REFERENCE_LABEL} minus {NEUTRAL_LABEL}, is what the wrapper, the reasoning "
            "prefill and the constitution-shaped prompt set add on top of plain distillation."
        ),
        takeaway="Largest family per contrast and position: " + "; ".join(_lines) + ".",
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
    ).pivot(on="dimension", index=["contrast", "position"], values="shift (95% interval)")
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
                        f"{r['emotion']} {r['shift']:+.2f}" for r in _d.head(5).iter_rows(named=True)
                    ),
                    "down": ", ".join(
                        f"{r['emotion']} {r['shift']:+.2f}" for r in _d.tail(5).reverse().iter_rows(named=True)
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
        options={c: c for c in CONTRAST_ORDER}, value=CONTRAST_ORDER[0], label="contrast"
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
            mo.hstack([contrast_pick, contrast_position_pick], justify="start", gap=2),
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
        _base.mark_rule(color="#333333").encode(x=_x, y="ci_lo:Q", y2="ci_hi:Q"),
    ).properties(
        width=len(_order) * 7,
        height=320,
        title=f"{contrast_pick.value}, {contrast_position_pick.value.replace('_', ' ')}: every emotion, sorted by shift",
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
        (_by[(_p, POSITIONS["user_mean"])], _by[(_p, POSITIONS["pre_response"])])
        for _p in {r["persona"] for r in PERSONA_ABS.to_dicts()}
    ]
    _share = [100 * u["mean_abs_shift"] / v["mean_abs_shift"] for u, v in _pairs]
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
    READ_ORDER,
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
            x=alt.X(
                "shift:Q", title=f"mean shift vs {REFERENCE_LABEL} (base sd units)"
            ),
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
                title=f"Per-emotion shift vs {REFERENCE_LABEL}, {POSITIONS[position]}"
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
def _(N_PROMPTS, NOTEBOOK, REFERENCE_LABEL, save_chart, shift_chart, summarize):
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
def _(N_PROMPTS, NOTEBOOK, REFERENCE_LABEL, save_chart, shift_chart, summarize):
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
            title=f"Family means of the per-emotion shift vs {REFERENCE_LABEL}"
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
            y=alt.Y(
                "shift:Q", title=f"mean shift vs {REFERENCE_LABEL} (base sd units)"
            ),
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
        title=f"Largest shifts vs {REFERENCE_LABEL}: five up and five down per persona",
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
def _(FAMILIES, REFERENCE_LABEL, SHIFTS, alt, persona_pick, pl, position_pick):
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
            y=alt.Y(
                "shift:Q", title=f"mean shift vs {REFERENCE_LABEL} (base sd units)"
            ),
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
        title=f"{persona_pick.value}, {position_pick.value.replace('_', ' ')}: every emotion, sorted by shift",
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
            "low": [e for e, _ in sorted(_fit[d]["scores"].items(), key=lambda kv: kv[1])[:3]],
            "high": [e for e, _ in sorted(_fit[d]["scores"].items(), key=lambda kv: -kv[1])[:3]],
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
        for r in _model_rows.filter(pl.col("name") == MODEL_LABEL[REFERENCE]).iter_rows(named=True)
        for d in DIMENSIONS
    }
    MODEL_MAP = _model_rows.with_columns(
        [
            (
                pl.col(d)
                - pl.col("position_label").replace_strict(
                    {p: _ref_at[(p, d)] for p in POSITIONS.values()}, return_dtype=pl.Float64
                )
            ).alias(d)
            for d in DIMENSIONS
        ]
    ).with_columns(
        [
            (pl.col(d) - pl.col(f"{d}_sd")).alias(f"{d}_lo") for d in DIMENSIONS
        ]
        + [(pl.col(d) + pl.col(f"{d}_sd")).alias(f"{d}_hi") for d in DIMENSIONS]
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
        AFFECT_MAP,
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
    DIMENSIONS,
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
    _color = alt.Color(
        "name:N",
        sort=_order,
        scale=alt.Scale(domain=_order, range=PALETTE[: len(_order)]),
        legend=alt.Legend(title=None, orient="right", labelFontSize=11, symbolSize=150),
    )
    # Fixed, shared domains so the three panels compare directly and the labels can be
    # placed in pixels.
    _pad = 0.6
    _XD = (
        float(MODEL_MAP["valence_lo"].min()) - _pad,
        float(MODEL_MAP["valence_hi"].max()) + _pad,
    )
    _YD = (
        float(MODEL_MAP["arousal_lo"].min()) - _pad,
        float(MODEL_MAP["arousal_hi"].max()) + _pad,
    )
    _W, _H = 430, 400
    _px = (_W / (_XD[1] - _XD[0]), _H / (_YD[1] - _YD[0]))

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
            _dots = [
                ((r["valence"] - _XD[0]) * _px[0], (_YD[1] - r["arousal"]) * _px[1])
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
                        _ax if _align == "left" else _ax - _w if _align == "right" else _ax - _w / 2
                    )
                    _try = (_x0, _ay - _h / 2, _x0 + _w, _ay + _h / 2)
                    _hit = any(
                        not (_try[2] < b[0] or _try[0] > b[2] or _try[3] < b[1] or _try[1] > b[3])
                        for b in _boxes
                    )
                    _hit = _hit or any(
                        _k != _i and _try[0] - 8 < dx < _try[2] + 8 and _try[1] - 8 < dy < _try[3] + 8
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

    _placed = _place_labels(MODEL_MAP)

    # The direction labels: the three most extreme emotion vectors at each end of each axis.
    # They ride in the same frame as the models, because a faceted layer takes one dataset.
    def _end_row(_p: str, _x: float, _y: float, _align: str, _text: str) -> dict:
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

    _ends = pl.DataFrame(
        [
            _end_row(_p, _XD[0] + 0.1, _YD[0] + 0.3, "left",
                     "toward " + ", ".join(ENDS["valence"]["low"]))
            for _p in POSITIONS.values()
        ]
        + [
            _end_row(_p, _XD[1] - 0.1, _YD[0] + 0.3, "right",
                     "toward " + ", ".join(ENDS["valence"]["high"]))
            for _p in POSITIONS.values()
        ]
        + [
            _end_row(_p, _XD[0] + 0.1, _YD[1] - 0.25, "left",
                     "up: toward " + ", ".join(ENDS["arousal"]["high"]))
            for _p in POSITIONS.values()
        ]
        + [
            _end_row(_p, _XD[0] + 0.1, _YD[0] + 0.9, "left",
                     "down: toward " + ", ".join(ENDS["arousal"]["low"]))
            for _p in POSITIONS.values()
        ]
    )
    _frame = pl.concat([_placed, _ends], how="diagonal_relaxed")
    _is_model = alt.datum.kind == "model"
    _is_end = alt.datum.kind == "end"
    _x = alt.X(
        "valence:Q",
        title=AXIS_LABEL["valence"] + f", {REFERENCE_LABEL} at zero",
        scale=alt.Scale(domain=list(_XD)),
    )
    _y = alt.Y(
        "arousal:Q",
        title=AXIS_LABEL["arousal"] + f", {REFERENCE_LABEL} at zero",
        scale=alt.Scale(domain=list(_YD)),
    )
    _src = alt.Chart()  # the data is handed to alt.layer once, so the facet can split it
    _zero_x = _src.transform_filter(_is_model).mark_rule(color="#888888").encode(x=alt.datum(0))
    _zero_y = _src.transform_filter(_is_model).mark_rule(color="#888888").encode(y=alt.datum(0))
    _bars_v = _src.transform_filter(_is_model).mark_rule(strokeWidth=2, opacity=0.6).encode(
        x="valence_lo:Q", x2="valence_hi:Q", y=_y, color=_color
    )
    _bars_a = _src.transform_filter(_is_model).mark_rule(strokeWidth=2, opacity=0.6).encode(
        x=_x, y="arousal_lo:Q", y2="arousal_hi:Q", color=_color
    )
    _dots = _src.transform_filter(_is_model).mark_point(
        shape="diamond", size=200, filled=True, stroke="#222222", strokeWidth=1, opacity=1
    ).encode(
        x=_x,
        y=_y,
        color=_color,
        tooltip=[
            "name:N",
            "position_label:N",
            alt.Tooltip("valence:Q", format="+.2f", title=f"valence vs {REFERENCE_LABEL}"),
            alt.Tooltip("valence_sd:Q", format=".2f", title="valence sd over prompts"),
            alt.Tooltip("arousal:Q", format="+.2f", title=f"arousal vs {REFERENCE_LABEL}"),
            alt.Tooltip("arousal_sd:Q", format=".2f", title="arousal sd over prompts"),
            alt.Tooltip("dominance:Q", format="+.2f", title=f"dominance vs {REFERENCE_LABEL}"),
            alt.Tooltip("n:Q", title="prompts"),
        ],
    )
    _leaders = (
        _src.transform_filter(_is_model & (alt.datum.leader == True))  # noqa: E712
        .mark_rule(color="#555555", strokeWidth=0.7)
        .encode(x=_x, y=_y, x2="label_x:Q", y2="label_y:Q")
    )
    _labels = [
        _src.transform_filter(_is_model & (alt.datum.label_align == _a))
        .mark_text(fontSize=10, fontWeight="bold", align=_a, baseline="middle", color="#111111")
        .encode(x="label_x:Q", y="label_y:Q", text="label:N")
        for _a in ("left", "right", "center")
    ]
    _direction_labels = [
        _src.transform_filter(_is_end & (alt.datum.label_align == _a))
        .mark_text(fontSize=9, align=_a, baseline="middle", color="#666666")
        .encode(x=_x, y=_y, text="label:N")
        for _a in ("left", "right")
    ]
    _chart = (
        alt.layer(
            _zero_x,
            _zero_y,
            _bars_v,
            _bars_a,
            _dots,
            _leaders,
            *_labels,
            *_direction_labels,
            data=_frame,
        )
        .properties(width=_W, height=_H)
        .facet(
            column=alt.Column(
                "position_label:N",
                sort=list(POSITIONS.values()),
                title=None,
                header=alt.Header(labelFontSize=13),
            )
        )
        .properties(
            title=f"The models on the valence-arousal plane, {REFERENCE_LABEL} at the origin (bars = ±1 sd over prompts)"
        )
        .configure_view(fill="#eaeaf2", stroke=None)
        .configure_axis(
            grid=True, gridColor="#ffffff", gridWidth=1, domain=False, tickColor="#ffffff"
        )
    )
    _pre = _placed.filter(pl.col("position_label") == POSITIONS["pre_response"])
    _summary = "; ".join(
        f"{r['name']} valence {r['valence']:+.2f}, arousal {r['arousal']:+.2f}"
        for r in _pre.iter_rows(named=True)
    )
    AFFECT_MAP_CHART = save_chart(
        _chart,
        "persona_affect_map",
        caption=(
            f"Every model on the fitted valence and arousal axes at the three read positions, with "
            f"{REFERENCE_LABEL} at the origin: a diamond is the model's mean projection over the {N_PROMPTS} WildChat "
            f"prompts minus {REFERENCE_LABEL}'s mean at the same position, in the axes' own units, and the bars "
            "span one standard deviation of the model's per-prompt values on each axis. The emotion vectors are "
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
    _y = alt.Y("name:N", sort=_order, title=None, axis=alt.Axis(labelFontSize=11))
    _x = alt.X(
        "dominance:Q",
        title=AXIS_LABEL["dominance"].split(", r")[0] + f"), {REFERENCE_LABEL} at zero",
    )
    _base = alt.Chart()
    _bars = _base.mark_rule(strokeWidth=2).encode(
        x="dominance_lo:Q", x2="dominance_hi:Q", y=_y, color=_color
    )
    _pts = _base.mark_point(
        shape="diamond", size=200, filled=True, stroke="#222222", strokeWidth=1, opacity=1
    ).encode(
        x=_x,
        y=_y,
        color=_color,
        tooltip=[
            "name:N",
            "position_label:N",
            alt.Tooltip("dominance:Q", format="+.2f"),
            alt.Tooltip("dominance_sd:Q", format=".2f", title="sd over prompts"),
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
            title=f"Dominance of the models against {REFERENCE_LABEL} (bars = ±1 sd over prompts)"
        )
        .configure_view(fill="#eaeaf2", stroke=None)
        .configure_axis(
            grid=True, gridColor="#ffffff", gridWidth=1, domain=False, tickColor="#ffffff"
        )
    )
    _pre = MODEL_MAP.filter(pl.col("position_label") == POSITIONS["pre_response"])
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
    AXIS_LABEL,
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
        _ref_mean = float(_df.filter(pl.col("model") == _ref_label)["value"].mean())
        _base = alt.Chart()
        _bars = _base.mark_bar(opacity=0.85).encode(
            x=alt.X(
                "value:Q",
                bin=alt.Bin(extent=[_lo, _hi], step=(_hi - _lo) / 24),
                title=f"{dim}, {POSITIONS[pos]}",
            ),
            y=alt.Y("count():Q", title=None, axis=alt.Axis(labelFontSize=8, tickCount=3)),
            color=_color,
            tooltip=["model:N", alt.Tooltip("count():Q", title="prompts")],
        )
        _mean = _base.mark_rule(color="#111111", strokeWidth=1.5).encode(x="mean(value):Q")
        _ref = _base.mark_rule(color="#333333", strokeWidth=1, strokeDash=[4, 3]).encode(
            x=alt.datum(_ref_mean)
        )
        return (
            alt.layer(_bars, _ref, _mean, data=_df)
            .properties(width=190, height=52)
            .facet(
                row=alt.Row(
                    "model:N",
                    sort=_order,
                    title=None,
                    header=alt.Header(labelFontSize=10, labelAngle=0, labelAlign="left"),
                )
            )
            .resolve_scale(y="shared")
        )

    _chart = (
        alt.hconcat(*[_column(_p, _d) for _p in _shown for _d in DIMENSIONS], spacing=14)
        .properties(
            title=f"Per-prompt distributions on the three axes (solid = model mean, dashed = {_ref_label} mean)"
        )
        .configure_view(fill="#eaeaf2", stroke=None)
        .configure_axis(
            grid=True, gridColor="#ffffff", gridWidth=1, domain=False, tickColor="#ffffff"
        )
    )
    _parts = []
    for _p in _shown:
        for _d in DIMENSIONS:
            _df = AFFECT_VALUES.filter(
                (pl.col("position") == _p) & (pl.col("dimension") == _d)
            )
            _sd = {
                m: float(np.std(_df.filter(pl.col("model") == m)["value"].to_numpy(), ddof=1))
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
    ### The affect shift against all three references

    **What the chart uses.** The {N_PROMPTS} WildChat prompts, the seven persona
    checkpoints, all three read positions at layer 21, and the three fitted axes. Each
    persona is compared with three different reference models: {", ".join(REFERENCE_ORDER)}.

    **How the numbers were made.** For one persona, axis, position and reference, the two
    models' projections onto that axis are subtracted prompt by prompt, averaged, and
    divided by the untrained base model's standard deviation on that axis over the same
    prompts at the same position; the whisker is 1.96 standard errors of the paired
    differences. Reading the three bars together separates what the mood adds from what the
    distillation adds: the difference against a control is the mood alone, the difference
    against the untrained base also contains everything the recipe installs, and the two
    controls install it in different ways.
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
    # The same paired difference against each of the three references, one colored bar per
    # reference within a persona's row: moodless (control) is the primary one, and the two
    # others say how much of a mood's shift is the distillation the controls also carry.
    _base = alt.Chart(AFFECT_SHIFTS)
    _y = alt.Y(
        "persona:N",
        sort=PERSONA_ORDER,
        title=None,
        axis=alt.Axis(labelFontSize=11),
    )
    _offset = alt.YOffset("reference_label:N", sort=REFERENCE_ORDER)
    _color = alt.Color(
        "reference_label:N",
        sort=REFERENCE_ORDER,
        scale=alt.Scale(
            domain=REFERENCE_ORDER, range=["#0072B2", "#009E73", "#7f7f7f"]
        ),
        legend=alt.Legend(
            title=None, orient="top", direction="horizontal", labelFontSize=11
        ),
    )
    _panel = alt.layer(
        _base.mark_rule(color="#9a9a9a").encode(x=alt.datum(0)),
        _base.mark_bar(size=7).encode(
            y=_y,
            yOffset=_offset,
            x=alt.X("shift:Q", title="mean shift (base sd units)"),
            color=_color,
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
            y=_y, yOffset=_offset, x="ci_lo:Q", x2="ci_hi:Q"
        ),
    ).properties(width=210, height=30 * len(PERSONA_ORDER))
    _chart = _panel.facet(
        row=alt.Row(
            "dimension:N",
            sort=DIMENSIONS,
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
        title="Valence, arousal and dominance shift, against each of the three references"
    )
    _pre = AFFECT_SHIFTS.filter(
        (pl.col("position") == "pre_response")
        & (pl.col("reference_label") == REFERENCE_ORDER[0])
    )
    _lines = "; ".join(
        f"{r['persona']} {r['dimension']} {r['shift']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
        for r in _pre.sort("persona", "dimension").iter_rows(named=True)
    )
    AFFECT_CHART = save_chart(
        _chart,
        "persona_affect_shift",
        caption=(
            "Mean shift of each persona model on the fitted valence, arousal and dominance axes, at all three "
            f"read positions, over {N_PROMPTS} WildChat prompts, in units of the "
            "base model's spread on that axis; whiskers are 95% intervals from the paired per-prompt "
            f"differences. One bar per reference: {', '.join(REFERENCE_ORDER)}, the first being the primary one."
        ),
        takeaway=f"At the pre-response token, against {REFERENCE_LABEL}: {_lines}.",
        notebook=NOTEBOOK,
    )
    AFFECT_CHART
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
    """)
    return


@app.cell
def _(
    DIMENSIONS,
    MODELS,
    MODEL_LABEL,
    OFFSET,
    REFERENCE,
    STORIES,
    pl,
):
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
    STORY_FAMILY_SHIFTS = pl.DataFrame(
        [
            {
                "model": _m,
                "label": MODEL_LABEL.get(_m, _m.split("-")[0]),
                "family": _f,
                "mean_shift": _v,
            }
            for _m, _blocks in STORIES["shifts"][REFERENCE].items()
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

    **Why there are two panels.** On the emotions' own scale the ten checkpoints land on top
    of one another at the origin, which is itself the answer to one question and useless for
    another, so the left panel shows them in the landscape and the right panel shows the same
    ten points on their own axis range. The bars on the right are 95% intervals on the mean
    (1.96 standard errors over the 3,420 stories), not the spread over stories: that spread
    is the emotional range of the corpus, is nearly identical for every checkpoint, and would
    swamp the differences between them; it is in the tooltip.

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
    NOTEBOOK,
    PALETTE,
    STORIES,
    STORY_POINTS,
    STORY_SET,
    alt,
    pl,
    save_chart,
):
    _fit = AXES["fits"][AXES["primary"]]
    _emotions = pl.DataFrame(
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
    # The checkpoints, with a 95% interval on the mean rather than the spread over stories:
    # the spread is the emotional range of the corpus, which is the same for every model and
    # would hide the differences between them.
    _models = STORY_POINTS.with_columns(
        [
            (1.96 * pl.col(f"{d}_sd") / pl.col("n").sqrt()).alias(f"{d}_se")
            for d in DIMENSIONS
        ]
    ).with_columns(
        [(pl.col(d) - pl.col(f"{d}_se")).alias(f"{d}_lo") for d in DIMENSIONS]
        + [(pl.col(d) + pl.col(f"{d}_se")).alias(f"{d}_hi") for d in DIMENSIONS]
    )
    _order = [MODEL_LABEL[m] for m in MODELS]
    _x_title = AXIS_LABEL["valence"]
    _y_title = AXIS_LABEL["arousal"]
    _tooltip = [
        alt.Tooltip("label:N", title="checkpoint"),
        alt.Tooltip("valence:Q", format="+.3f"),
        alt.Tooltip("valence_sd:Q", format=".2f", title="valence sd over stories"),
        alt.Tooltip("arousal:Q", format="+.3f"),
        alt.Tooltip("arousal_sd:Q", format=".2f", title="arousal sd over stories"),
        alt.Tooltip("dominance:Q", format="+.3f"),
        alt.Tooltip("n:Q", title="stories"),
    ]

    # Left: the emotion landscape, with the checkpoints where they actually fall in it.
    _dots = (
        alt.Chart(_emotions)
        .mark_circle(size=80, opacity=0.35)
        .encode(
            x=alt.X("valence:Q", title=_x_title),
            y=alt.Y("arousal:Q", title=_y_title),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=FAMILIES, scheme="tableau10"),
                legend=alt.Legend(
                    title=None, orient="top", direction="horizontal", columns=5, labelFontSize=10
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
    _cluster = (
        alt.Chart(_models)
        .mark_point(shape="diamond", size=90, filled=True, color="#111111", stroke="#ffffff", strokeWidth=1)
        .encode(x="valence:Q", y="arousal:Q", tooltip=_tooltip)
    )
    _note = (
        alt.Chart(_models.head(1))
        .mark_text(
            text="all ten checkpoints",
            align="left",
            dx=12,
            dy=-12,
            fontSize=11,
            fontWeight="bold",
            color="#111111",
        )
        .encode(x=alt.datum(0.0), y=alt.datum(0.0))
    )
    _left = alt.layer(_dots, _cluster, _note).properties(
        width=460,
        height=460,
        title=alt.Title(
            "The emotion landscape, and where the checkpoints read in it",
            subtitle="171 emotion vectors (faint) and the ten checkpoints (black), one origin, one convention",
            fontSize=13,
            subtitleFontSize=10,
            subtitleColor="#555",
            anchor="start",
        ),
    )

    # Right: the same checkpoints, magnified, so the differences between them are visible.
    _color = alt.Color(
        "label:N",
        sort=_order,
        scale=alt.Scale(domain=_order, range=PALETTE[: len(_order)]),
        legend=alt.Legend(title=None, orient="right", labelFontSize=10, symbolSize=120),
    )
    _zx = alt.X("valence:Q", title=_x_title, scale=alt.Scale(zero=False))
    _zy = alt.Y("arousal:Q", title=_y_title, scale=alt.Scale(zero=False))
    _base = alt.Chart(_models)
    _right = alt.layer(
        _base.mark_rule(color="#888888").encode(x=alt.datum(0)),
        _base.mark_rule(color="#888888").encode(y=alt.datum(0)),
        _base.mark_rule(strokeWidth=1.5).encode(x="valence_lo:Q", x2="valence_hi:Q", y=_zy, color=_color),
        _base.mark_rule(strokeWidth=1.5).encode(x=_zx, y="arousal_lo:Q", y2="arousal_hi:Q", color=_color),
        _base.mark_point(
            shape="diamond", size=170, filled=True, stroke="#222222", strokeWidth=1, opacity=1
        ).encode(x=_zx, y=_zy, color=_color, tooltip=_tooltip),
    ).properties(
        width=380,
        height=460,
        title=alt.Title(
            "The same ten checkpoints, magnified",
            subtitle="note the axis range: this is the small box around the origin on the left",
            fontSize=13,
            subtitleFontSize=10,
            subtitleColor="#555",
            anchor="start",
        ),
    )
    _chart = (
        alt.hconcat(_left, _right, spacing=30)
        # without this the two panels' color legends are merged into one list that mixes
        # taxonomy families with model names
        .resolve_scale(color="independent")
        .properties(
            title=alt.Title(
                "Checkpoints and emotions on one plane, both read as story text",
                subtitle=f"{STORIES['sets'][STORY_SET]['n']:,} held-out stories; origin = the average emotional story; bars = 95% intervals on the mean",
                fontSize=14,
                subtitleFontSize=11,
                subtitleColor="#555",
                anchor="start",
            )
        )
        .configure_view(fill="#eaeaf2", stroke=None)
        .configure_axis(
            grid=True, gridColor="#ffffff", gridWidth=1, domain=False, tickColor="#ffffff"
        )
    )
    _summary = "; ".join(
        f"{r['label']} valence {r['valence']:+.3f}, arousal {r['arousal']:+.3f}"
        for r in _models.iter_rows(named=True)
    )
    _span = (
        float(_emotions["valence"].max() - _emotions["valence"].min()),
        float(_models["valence"].max() - _models["valence"].min()),
    )
    STORY_MAP_CHART = save_chart(
        _chart,
        "story_read_affect_map",
        caption=(
            "Left: the 171 emotion vectors (faint dots, colored by taxonomy family) and all ten checkpoints "
            "(black diamonds) on the fitted valence and arousal axes, both read in the story convention -- raw "
            "text, 256-token truncation, layer 21 mean-pooled from the fiftieth token on -- so that a "
            "checkpoint's position among the emotions can be read here in a way it cannot in Part 3. An emotion "
            "sits at its own vector's score; a checkpoint sits at its mean projection over the "
            f"{STORIES['sets'][STORY_SET]['n']:,} held-out stories, on the same origin, the average emotional "
            "story. Right: the same ten checkpoints on their own axis range, with 95% intervals on the mean "
            "(the spread over stories is the emotional range of the corpus and is nearly the same for every "
            "model, so it is in the tooltip rather than on the chart)."
        ),
        takeaway=(
            f"Read on emotional stories every checkpoint sits within {_span[1]:.2f} valence units of every "
            f"other, against a {_span[0]:.1f}-unit spread across the 171 emotions, so a mood barely moves where "
            f"the model reads emotional text: {_summary}."
        ),
        notebook=NOTEBOOK,
    )
    STORY_MAP_CHART
    return


@app.cell
def _(REFERENCE_LABEL, STORIES, STORY_SET, mo):
    mo.md(f"""
    ### Where each mood's story-side shift lands, family by family

    **What the chart uses.** The {STORIES["sets"][STORY_SET]["n"]:,} held-out stories again,
    read by the seven persona checkpoints and by {REFERENCE_LABEL} in the story convention,
    projected onto the same 171 emotion vectors.

    **How the numbers were made.** Each vector's shift is the mean over the 3,420 stories of
    the difference between the persona's projection and {REFERENCE_LABEL}'s on the same
    story, divided by the base model's standard deviation for that vector over those same
    stories; a bar is the plain average of those values over the emotions of one taxonomy
    family. A value of 1 would mean a shift the size of the variation emotional content
    itself produces on that vector, so these are small numbers by construction: the moods
    tilt the read, they do not move it as far as changing the story does.
    """)
    return


@app.cell
def _(
    FAMILIES,
    NOTEBOOK,
    PERSONA_ORDER,
    REFERENCE_LABEL,
    STORIES,
    STORY_FAMILY_SHIFTS,
    STORY_SET,
    alt,
    pl,
    save_chart,
):
    # Where each mood's shift lands by family, on the emotional stories.
    _df = STORY_FAMILY_SHIFTS.filter(pl.col("label").is_in(PERSONA_ORDER))
    _base = alt.Chart(_df)
    _panel = alt.layer(
        _base.mark_rule(color="#9a9a9a").encode(x=alt.datum(0)),
        _base.mark_bar(size=9).encode(
            y=alt.Y("family:N", sort=FAMILIES, title=None, axis=alt.Axis(labelFontSize=9)),
            x=alt.X("mean_shift:Q", title="family mean shift (base story-spread units)"),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=FAMILIES, scheme="tableau10"),
                legend=None,
            ),
            tooltip=[
                "label:N",
                "family:N",
                alt.Tooltip("mean_shift:Q", format="+.3f"),
            ],
        ),
    ).properties(width=170, height=210)
    _chart = _panel.facet(
        column=alt.Column(
            "label:N",
            sort=PERSONA_ORDER,
            title=None,
            header=alt.Header(labelFontSize=12),
        )
    ).properties(
        title=f"Family means of the story-side shift against {REFERENCE_LABEL}"
    )
    _rows = {(r["label"], r["family"]): r["mean_shift"] for r in _df.to_dicts()}
    _lines = []
    for _p in PERSONA_ORDER:
        _fam = max(FAMILIES, key=lambda f: abs(_rows[(_p, f)]))
        _lines.append(f"{_p}: {_fam} {_rows[(_p, _fam)]:+.2f}")
    STORY_FAMILY_CHART = save_chart(
        _chart,
        "story_family_shift",
        caption=(
            f"Mean over each taxonomy family of the per-vector shift against {REFERENCE_LABEL} on the "
            f"{STORIES['sets'][STORY_SET]['n']:,} held-out emotional stories, one panel per mood, in units of the "
            "base model's per-vector spread over those same stories, so a value of 1 is the size of the "
            "variation emotional content itself produces on that vector."
        ),
        takeaway="Largest family per mood on the story read: " + "; ".join(_lines) + ".",
        notebook=NOTEBOOK,
    )
    STORY_FAMILY_CHART
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
