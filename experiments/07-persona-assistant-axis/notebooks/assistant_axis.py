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

    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, mo, np, pl, save_chart, yaml


@app.cell
def _(Path, json, yaml):
    HERE = Path(__file__).parents[1]  # the experiment dir
    DATA = HERE / "data"
    NOTEBOOK = __file__

    CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    BUILD = CONFIG["build"]
    BUILD_DIR = DATA / BUILD
    REPORT = json.loads((BUILD_DIR / "axis_report.json").read_text(encoding="utf-8"))
    CHECKS = REPORT["checks"]
    LAYER = REPORT["target_layer"]
    N_LAYERS = REPORT["n_layers"]

    # The model every persona is compared against, config `projection.reference`: since
    # 2026-09-11 neutral-LIMA (control), the no-wrapper control trained on the shared LIMA
    # prompts only (06-persona-teachers configs/neutral-lima.yaml, 2026-09-09), the
    # reference 07-persona-activations adopted on 2026-09-10 (Carolina); before that it was
    # moodless, the recipe with a neutral constitution in the wrapper (2026-09-08 to
    # 2026-09-10), now an additional reference. `base` is the untrained model, kept
    # alongside so the distillation's own footprint is visible.
    REFERENCE = CONFIG["projection"]["reference"]

    # Projected models kept as data only (config `projection.superseded`) are left out of
    # every exhibit, table and instrument here; their projection files stay on disk. The
    # list is empty since 2026-09-09, when neutral (no-wrapper control) came back into the
    # exhibits as an additional comparison.
    SUPERSEDED = set(CONFIG["projection"].get("superseded", []))
    PROJ = {
        m: json.loads((BUILD_DIR / "projections" / f"{m}.json").read_text(encoding="utf-8"))
        for m in CONFIG["projection"]["models"]
        if m not in SUPERSEDED and (BUILD_DIR / "projections" / f"{m}.json").exists()
    }
    if REFERENCE not in PROJ:
        raise FileNotFoundError(f"no projection for the reference model {REFERENCE!r} under {BUILD_DIR / 'projections'}")
    if "base" not in PROJ:
        raise FileNotFoundError("the base model's projection is needed (run project.py)")
    MODELS = list(PROJ)  # config order: base, the three controls, then the personas
    # Display labels: the untrained model, the three controls under the names the write-ups
    # use (the same as 07-persona-activations since 2026-09-10), and a persona under its
    # own name.
    MODEL_LABEL = {
        m: ("base" if m == "base"
            else "neutral-LIMA (control)" if m.startswith("neutral-lima-")
            else "moodless (wrapper control)" if m.startswith("moodless-")
            else "neutral (no-wrapper control)" if m.startswith("neutral-")
            else m.split("-")[0])
        for m in MODELS
    }
    REFERENCE_LABEL = MODEL_LABEL[REFERENCE]
    MODEL_ORDER = [MODEL_LABEL[m] for m in MODELS]  # the display order of every model list
    # The controls are nulls, not personas: the reference and the other two constructions.
    CONTROLS = [
        m for m in MODELS
        if m == REFERENCE or m.startswith(("neutral-lima-", "moodless-", "neutral-"))
    ]
    OTHER_CONTROLS = [m for m in CONTROLS if m != REFERENCE]
    PERSONAS = [m for m in MODELS if m not in CONTROLS and m != "base"]
    PERSONA_ORDER = [MODEL_LABEL[m] for m in PERSONAS]
    # Rows compared against the reference: the untrained base and the other controls
    # first, then the personas.
    COMPARED = [m for m in MODELS if m != REFERENCE]
    COMPARED_ORDER = [MODEL_LABEL[m] for m in COMPARED]
    # The display order in words, for captions: "base, neutral-LIMA (control), ...".
    ORDER_TEXT = ", ".join([*MODEL_ORDER[: 1 + len(CONTROLS)], "the personas"])

    # The prompts every model is read on: 07-persona-activations' pool minus the rows its
    # project.py leaves out for every model (prompts the neutral control trained on), so the
    # two experiments report the same set.
    _pool = json.loads(
        (HERE.parent / "07-persona-activations" / "data" / "pool" / "prompts.json").read_text(encoding="utf-8")
    )
    EXCLUDED_IDS = {r["id"] for r in _pool["rows"] if r.get("in_neutral_training_prompts")}
    N_PROMPTS = sum(1 for r in _pool["rows"] if r["id"] not in EXCLUDED_IDS)

    _cal_path = BUILD_DIR / "judge_calibration.json"
    CALIBRATION = json.loads(_cal_path.read_text(encoding="utf-8")) if _cal_path.exists() else None
    return (
        BUILD,
        CALIBRATION,
        CHECKS,
        COMPARED,
        COMPARED_ORDER,
        CONTROLS,
        EXCLUDED_IDS,
        LAYER,
        MODELS,
        MODEL_LABEL,
        MODEL_ORDER,
        NOTEBOOK,
        N_LAYERS,
        N_PROMPTS,
        ORDER_TEXT,
        OTHER_CONTROLS,
        PERSONAS,
        PROJ,
        REFERENCE,
        REFERENCE_LABEL,
        REPORT,
    )


@app.cell
def _(
    BUILD,
    CHECKS,
    LAYER,
    MODEL_LABEL,
    N_LAYERS,
    N_PROMPTS,
    OTHER_CONTROLS,
    PERSONAS,
    REFERENCE,
    REFERENCE_LABEL,
    REPORT,
    mo,
):
    mo.md(f"""
    # The Assistant Axis of Qwen3.5-9B, and where the persona models sit on it

    The axis is the direction in the residual stream from the mean of the role-playing
    activations to the mean default-Assistant activation (Lu et al. 2026), built here
    with the authors' own pipeline (repository commit `{REPORT["repo_commit"][:9]}`, build
    `{BUILD}`): the model answered 240 extraction questions under five system prompts for
    each of 275 roles and for the default, every reply's activation is the mean residual
    over its tokens, a judge kept the replies that fully play the role, and the axis is
    the default's mean minus the mean of the {REPORT["n_roles_with_vector"]} role vectors,
    one direction per layer. The target layer is the middle one, {LAYER} of {N_LAYERS}
    (`axis[i]` is the output of decoder layer *i*, transformers' `hidden_states[i + 1]`).

    A model's *projection* is the dot product of its mean reply activation with the
    unit-length axis, in residual units, higher meaning more Assistant-like; the
    *cosine* divides by the activation's own norm. Positions and drift are projections
    (the paper's convention, and what activation capping clamps); cosines are shown
    alongside so a change of magnitude cannot pass for a change of direction.

    The persona models ({", ".join(MODEL_LABEL[m] for m in PERSONAS)}) are compared against
    **{REFERENCE_LABEL}** (`{REFERENCE}`), the no-wrapper control trained on the shared
    LIMA prompts only, so every persona is read against a model whose training prompts
    are a strict subset of its own and what the distillation does to the model is
    separated from what the mood does (the reference since 2026-09-11, following
    07-persona-activations; before that it was moodless, the recipe with a neutral
    constitution in the wrapper). The untrained base and the other controls
    ({", ".join(MODEL_LABEL[m] for m in OTHER_CONTROLS)}: moodless is the persona
    recipe with an assistant-neutral constitution through the same wrapper and reasoning
    prefill, neutral the 2026-09-07 construction with no wrapper and no prefill on the
    full prompt set) are shown alongside as the other references, on the same
    {N_PROMPTS} prompts. The checks at layer {LAYER}:
    cosine of the axis with the roles' first principal component
    {CHECKS["cos_axis_pc1"]:.3f}, the default at {CHECKS["default_position_on_pc1_to_5"][0]:.2f}
    of that component's range between the roles' extremes.
    """)
    return


@app.cell
def _(CHECKS, REPORT, pl):
    # The persona space at the target layer: every role's coordinates on the top five
    # principal components (PC1 signed toward the Assistant), its projection on the axis
    # (centered on the roles' mean) and its cosine with the axis.
    _near = set(CHECKS["roles_nearest_assistant"][:5])  # the near end is dense: fewer labels
    _far = set(CHECKS["roles_farthest_from_assistant"][:8])
    ROLE_SPACE = pl.DataFrame(
        [
            {
                "role": r,
                "pc1": pcs[0],
                "pc2": pcs[1],
                "pc3": pcs[2],
                "projection": REPORT["role_projection_centered"][r],
                "cosine": REPORT["role_cosine_with_axis"][r],
                "label": r if (r in _near or r in _far) else "",
                "end": "nearest the Assistant" if r in _near else ("farthest from the Assistant" if r in _far else "other"),
            }
            for r, pcs in REPORT["role_pc_coordinates"].items()
        ]
    ).sort("projection")
    DEFAULT_POINT = pl.DataFrame(
        [{"role": "default Assistant", "pc1": CHECKS["default_pc_coordinates"][0], "pc2": CHECKS["default_pc_coordinates"][1],
          "projection": CHECKS["default_projection_of_mean_vector"]}]
    )
    ROLE_SPACE
    return DEFAULT_POINT, ROLE_SPACE


@app.cell
def _(
    CHECKS,
    DEFAULT_POINT,
    LAYER,
    NOTEBOOK,
    REPORT,
    ROLE_SPACE,
    alt,
    save_chart,
):
    _var = CHECKS["pca_variance_explained_top10"]
    _pts = (
        alt.Chart(ROLE_SPACE)
        .mark_circle(size=42, opacity=0.85)
        .encode(
            x=alt.X("pc1:Q", title=f"first principal component ({_var[0]:.0%} of variance; Assistant end positive)"),
            y=alt.Y("pc2:Q", title=f"second principal component ({_var[1]:.0%})"),
            color=alt.Color("projection:Q", scale=alt.Scale(scheme="redblue", domainMid=0), title="projection on the axis"),
            tooltip=["role:N", alt.Tooltip("projection:Q", format="+.2f"), alt.Tooltip("cosine:Q", format="+.3f"),
                     alt.Tooltip("pc1:Q", format="+.2f"), alt.Tooltip("pc2:Q", format="+.2f")],
        )
    )
    _labels = (
        alt.Chart(ROLE_SPACE.filter(ROLE_SPACE["label"] != ""))
        .mark_text(dy=-8, fontSize=9)
        .encode(x="pc1:Q", y="pc2:Q", text="label:N")
    )
    _default = (
        alt.Chart(DEFAULT_POINT)
        .mark_point(shape="diamond", size=160, color="black", filled=True)
        .encode(x="pc1:Q", y="pc2:Q", tooltip=["role:N", alt.Tooltip("projection:Q", format="+.2f")])
    )
    _default_label = alt.Chart(DEFAULT_POINT).mark_text(dy=-11, fontSize=10, fontWeight="bold").encode(x="pc1:Q", y="pc2:Q", text="role:N")
    ROLE_SPACE_CHART = save_chart(
        alt.layer(_pts, _labels, _default, _default_label).properties(width=640, height=420),
        "role_space_pc1_pc2",
        caption=(
            f"The {REPORT['n_roles_with_vector']} role vectors of Qwen3.5-9B at layer {LAYER} on the first two principal "
            "components of the persona space, colored by projection on the Assistant Axis (blue toward the "
            "Assistant); the default Assistant is the black diamond, and the roles at either end of the axis are "
            "labeled."
        ),
        takeaway=(
            f"The axis is aligned with the first component (cosine {CHECKS['cos_axis_pc1']:.3f}), which carries "
            f"{_var[0]:.0%} of the variance, and the default sits at {CHECKS['default_position_on_pc1_to_5'][0]:.2f} of "
            "the roles' range on it, among the text-processing roles (summarizer, proofreader, assistant, grader), "
            "opposite aberration, absurdist, void, fool, leviathan and eldritch."
        ),
        notebook=NOTEBOOK,
    )
    ROLE_SPACE_CHART
    return


@app.cell
def _(CHECKS, LAYER, NOTEBOOK, N_LAYERS, REPORT, alt, pl, save_chart):
    ALIGNMENT = pl.DataFrame(
        {"layer": list(range(N_LAYERS)), "cosine": CHECKS["cos_axis_pc1_per_layer"], "axis_norm": REPORT["axis_norm_per_layer"]}
    )
    _line = (
        alt.Chart(ALIGNMENT)
        .mark_line(point=True)
        .encode(
            x=alt.X("layer:Q", title="layer (output of decoder layer i)", scale=alt.Scale(domain=[0, N_LAYERS - 1], nice=False)),
            y=alt.Y("cosine:Q", title="cosine of the axis with the roles' first principal component", scale=alt.Scale(domain=[0, 1])),
            tooltip=["layer:Q", alt.Tooltip("cosine:Q", format=".3f"), alt.Tooltip("axis_norm:Q", format=".2f", title="axis norm")],
        )
    )
    _paper = alt.Chart(pl.DataFrame({"y": [0.60]})).mark_rule(color="#9a9a9a", strokeDash=[4, 4]).encode(y="y:Q")
    _paper_text = alt.Chart(pl.DataFrame({"y": [0.60], "t": ["the paper's floor at every layer, 0.60"]})).mark_text(align="left", dx=4, dy=-6, fontSize=9, color="#6a6a6a").encode(y="y:Q", text="t:N", x=alt.datum(0))
    _target = alt.Chart(pl.DataFrame({"x": [LAYER]})).mark_rule(color="#9a9a9a").encode(x="x:Q")
    _lo, _hi = min(CHECKS["cos_axis_pc1_per_layer"]), max(CHECKS["cos_axis_pc1_per_layer"])
    ALIGNMENT_CHART = save_chart(
        alt.layer(_paper, _paper_text, _target, _line).properties(width=560, height=260),
        "axis_pc1_alignment_by_layer",
        caption=(
            "Cosine similarity between the Assistant Axis and the first principal component of the role vectors, "
            f"computed separately at each of the {N_LAYERS} layers; the dashed line is the paper's reported floor and "
            f"the vertical rule the target layer, {LAYER}."
        ),
        takeaway=(
            f"The axis is the roles' first component at every layer (cosine {_lo:.2f} to {_hi:.2f}, "
            f"{CHECKS['cos_axis_pc1']:.3f} at layer {LAYER}), above the paper's 0.60 throughout."
        ),
        notebook=NOTEBOOK,
    )
    ALIGNMENT_CHART
    return


@app.cell
def _(
    CHECKS,
    DEFAULT_POINT,
    LAYER,
    NOTEBOOK,
    REPORT,
    ROLE_SPACE,
    alt,
    pl,
    save_chart,
):
    _ranked = ROLE_SPACE.with_columns(pl.int_range(pl.len()).alias("rank"))
    _pts = (
        alt.Chart(_ranked)
        .mark_circle(size=26)
        .encode(
            x=alt.X("projection:Q", title="projection on the unit axis, centered on the roles' mean (residual units)"),
            y=alt.Y("rank:Q", title="roles, ordered", axis=None),
            color=alt.Color("end:N", scale=alt.Scale(domain=["farthest from the Assistant", "other", "nearest the Assistant"],
                                                     range=["#c0392b", "#b0b0b0", "#2e6db4"]), title=None),
            tooltip=["role:N", alt.Tooltip("projection:Q", format="+.2f"), alt.Tooltip("cosine:Q", format="+.3f")],
        )
    )
    _labels = (
        alt.Chart(_ranked.filter(_ranked["label"] != ""))
        .mark_text(align="left", dx=6, fontSize=9)
        .encode(x="projection:Q", y="rank:Q", text="label:N")
    )
    _default = alt.Chart(DEFAULT_POINT).mark_rule(color="black", strokeWidth=1.5).encode(x="projection:Q")
    _default_text = alt.Chart(DEFAULT_POINT).mark_text(align="right", dx=-4, dy=-4, fontSize=10, fontWeight="bold").encode(
        x="projection:Q", y=alt.datum(len(_ranked) - 1), text="role:N"
    )
    _p = REPORT["role_projection_centered"]
    ROLES_ALONG_AXIS = save_chart(
        alt.layer(_pts, _labels, _default, _default_text).properties(width=620, height=520),
        "roles_along_axis",
        caption=(
            f"The {len(_ranked)} role vectors of Qwen3.5-9B ordered by their projection on the Assistant Axis at layer "
            f"{LAYER} (centered on the roles' mean), the roles at each end labeled; the vertical rule is the "
            "default Assistant's own vector."
        ),
        takeaway=(
            f"Role vectors span {min(_p.values()):+.1f} to {max(_p.values()):+.1f} on the axis and the default sits at "
            f"{CHECKS['default_projection_of_mean_vector']:+.1f}, beyond all but a few clerical roles "
            f"({', '.join(r for r in CHECKS['roles_nearest_assistant'][:4])}); the far end is "
            f"{', '.join(CHECKS['roles_farthest_from_assistant'][:5])}."
        ),
        notebook=NOTEBOOK,
    )
    ROLES_ALONG_AXIS
    return


@app.cell
def _(EXCLUDED_IDS, MODELS, MODEL_LABEL, N_LAYERS, PROJ, pl):
    # Every model's per-prompt read at every layer: the projection, the activation's norm,
    # and their ratio, the cosine. The pool rows every model is read without are dropped here.
    _rows = []
    for _m in MODELS:
        for _r in PROJ[_m]["rows"]:
            if _r["id"] in EXCLUDED_IDS:
                continue
            for _l in range(N_LAYERS):
                _rows.append({"model": _m, "label": MODEL_LABEL[_m], "id": _r["id"], "layer": _l,
                              "projection": _r["per_layer"][_l], "norm": _r["per_layer_norm"][_l],
                              "cosine": _r["per_layer"][_l] / _r["per_layer_norm"][_l]})
    READS = pl.DataFrame(_rows)
    IDS = [r["id"] for r in PROJ["base"]["rows"] if r["id"] not in EXCLUDED_IDS]
    READS
    return IDS, READS


@app.cell
def _(IDS, MODEL_LABEL, READS, np, pl):
    def paired(metric: str, layer: int, reference: str, models: list, n_boot: int = 3000, seed: int = 0):
        """Per-prompt paired difference of ``metric`` against ``reference`` at ``layer`` for each model.

        Returns one row per model: the model's mean, the mean difference, a 95% bootstrap
        interval over prompts, the share of prompts below the reference, and the difference
        in units of the reference's per-prompt standard deviation.
        """
        rng = np.random.default_rng(seed)
        _at = READS.filter(pl.col("layer") == layer)
        _ref = _at.filter(pl.col("model") == reference).sort("id")
        ref = np.array([dict(zip(_ref["id"], _ref[metric]))[i] for i in IDS])
        out = []
        for m in models:
            _mm = _at.filter(pl.col("model") == m).sort("id")
            x = np.array([dict(zip(_mm["id"], _mm[metric]))[i] for i in IDS])
            d = x - ref
            boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out.append({"model": m, "label": MODEL_LABEL[m], "mean": float(x.mean()), "delta": float(d.mean()),
                        "ci_lo": float(lo), "ci_hi": float(hi), "share_below": float((d < 0).mean()),
                        "effect": float(d.mean() / ref.std()), "n": int(len(d))})
        return pl.DataFrame(out)

    return (paired,)


@app.cell
def _(LAYER, MODELS, MODEL_LABEL, N_LAYERS, REFERENCE, mo):
    layer_slider = mo.ui.slider(0, N_LAYERS - 1, value=LAYER, label="layer", show_value=True)
    metric_picker = mo.ui.dropdown(["projection", "cosine", "norm"], value="projection", label="metric")
    reference_picker = mo.ui.dropdown({MODEL_LABEL[m]: m for m in MODELS}, value=MODEL_LABEL[REFERENCE], label="reference")
    mo.hstack([layer_slider, metric_picker, reference_picker], justify="start", gap=2)
    return layer_slider, metric_picker, reference_picker


@app.cell
def _(COMPARED_ORDER, alt):
    def delta_chart(df, metric: str, layer: int, reference_label: str, order=COMPARED_ORDER):
        _fmt = "+.3f" if metric == "cosine" else "+.2f"
        _base = alt.Chart(df)
        _zero = alt.Chart(df).mark_rule(color="#9a9a9a").encode(x=alt.datum(0))
        _ci = _base.mark_rule(strokeWidth=2).encode(
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelFontSize=12)),
            x=alt.X("ci_lo:Q", title=f"paired difference in {metric} against {reference_label} at layer {layer}"),
            x2="ci_hi:Q",
            color=alt.Color("label:N", sort=order, scale=alt.Scale(scheme="tableau10"), legend=None),
        )
        _dot = _base.mark_circle(size=90).encode(
            y=alt.Y("label:N", sort=order),
            x="delta:Q",
            color=alt.Color("label:N", sort=order, scale=alt.Scale(scheme="tableau10"), legend=None),
            tooltip=["label:N", alt.Tooltip("mean:Q", format=_fmt, title="model mean"),
                     alt.Tooltip("delta:Q", format=_fmt, title="difference"),
                     alt.Tooltip("ci_lo:Q", format=_fmt, title="95% low"), alt.Tooltip("ci_hi:Q", format=_fmt, title="95% high"),
                     alt.Tooltip("share_below:Q", format=".0%", title="prompts below reference"),
                     alt.Tooltip("effect:Q", format="+.2f", title="in reference SDs"), "n:Q"],
        )
        return alt.layer(_zero, _ci, _dot).properties(width=520, height=30 * len(order) + 20)

    return (delta_chart,)


@app.cell
def _(
    MODELS,
    MODEL_LABEL,
    delta_chart,
    layer_slider,
    metric_picker,
    mo,
    paired,
    reference_picker,
):
    # Instrument: the paired differences at any layer, metric and reference (never saved).
    _ref = reference_picker.value
    _models = [m for m in MODELS if m != _ref]
    _df = paired(metric_picker.value, layer_slider.value, _ref, _models)
    mo.vstack([
        mo.md(f"Paired differences in **{metric_picker.value}** against **{MODEL_LABEL[_ref]}** at layer {layer_slider.value} (instrument; the exhibit below pins the target layer)."),
        delta_chart(_df, metric_picker.value, layer_slider.value, MODEL_LABEL[_ref], [MODEL_LABEL[m] for m in _models]),
    ])
    return


@app.cell
def _(
    COMPARED,
    LAYER,
    MODEL_LABEL,
    NOTEBOOK,
    N_PROMPTS,
    OTHER_CONTROLS,
    PERSONAS,
    REFERENCE,
    REFERENCE_LABEL,
    delta_chart,
    paired,
    save_chart,
):
    SHIFT = paired("projection", LAYER, REFERENCE, COMPARED)
    SHIFT_COS = paired("cosine", LAYER, REFERENCE, COMPARED)
    _p = {r["label"]: r for r in SHIFT.to_dicts()}
    _c = {r["label"]: r for r in SHIFT_COS.to_dicts()}
    _persona_rows = sorted((r for r in SHIFT.to_dicts() if r["model"] in PERSONAS), key=lambda r: r["delta"])
    _base = _p["base"]
    _all_neg = all(r["ci_hi"] < 0 for r in _persona_rows)
    _others = ", ".join(
        f"{MODEL_LABEL[m]} {_p[MODEL_LABEL[m]]['delta']:+.2f} [{_p[MODEL_LABEL[m]]['ci_lo']:+.2f}, {_p[MODEL_LABEL[m]]['ci_hi']:+.2f}]"
        for m in OTHER_CONTROLS
    )
    SHIFT_CHART = save_chart(
        delta_chart(SHIFT, "projection", LAYER, REFERENCE_LABEL),
        "persona_axis_shift",
        caption=(
            f"Paired per-prompt difference in projection on the Assistant Axis at layer {LAYER} between each model and "
            f"{REFERENCE_LABEL} (`{REFERENCE}`, the no-wrapper control trained on the shared LIMA prompts only), over the "
            f"{N_PROMPTS} WildChat prompts of 07-persona-activations and the models' stored replies to them; whiskers are 95% "
            "bootstrap intervals over prompts, and the untrained base and the other controls "
            f"({', '.join(MODEL_LABEL[m] for m in OTHER_CONTROLS)}) are included as the other references."
        ),
        takeaway=(
            f"Against {REFERENCE_LABEL} every persona sits toward the non-Assistant end, from "
            f"{_persona_rows[-1]['label']} ({_persona_rows[-1]['delta']:+.2f}, {_persona_rows[-1]['effect']:+.2f} control SDs) to "
            f"{_persona_rows[0]['label']} ({_persona_rows[0]['delta']:+.2f}, {_persona_rows[0]['effect']:+.2f}), "
            f"{'all intervals excluding zero' if _all_neg else 'not every interval excluding zero'}, while the untrained base "
            f"differs from the control by {_base['delta']:+.2f} [{_base['ci_lo']:+.2f}, {_base['ci_hi']:+.2f}]: "
            + ("the distillation itself does not move the model along the axis, the moods do"
               if _base['ci_lo'] <= 0 <= _base['ci_hi'] else
               f"the distillation itself moves the model {abs(_base['delta']):.2f} along the axis before any mood "
               f"(the control sits {'below' if _base['delta'] > 0 else 'above'} the untrained base), and the moods move it further")
            + f"; the other controls: {_others}; the cosine reads the same "
            f"({', '.join(f'{r['label']} {_c[r['label']]['delta']:+.3f}' for r in _persona_rows)} on a control mean of "
            f"{_c[_persona_rows[0]['label']]['mean'] - _c[_persona_rows[0]['label']]['delta']:.3f})."
        ),
        notebook=NOTEBOOK,
        params={"layer": LAYER, "reference": REFERENCE, "metric": "projection"},
    )
    SHIFT_CHART
    return SHIFT, SHIFT_COS


@app.cell
def _(
    CONTROLS,
    LAYER,
    MODEL_LABEL,
    MODEL_ORDER,
    NOTEBOOK,
    N_PROMPTS,
    ORDER_TEXT,
    OTHER_CONTROLS,
    REFERENCE,
    REFERENCE_LABEL,
    SHIFT,
    alt,
    pl,
    save_chart,
):
    # Exhibit: where every model sits on the axis at the target layer, so the decomposition
    # is visible at a glance: base, the controls, the moods below them, rows in the
    # display order. Intervals are the paired ones (the per-prompt difference from the
    # reference, whose bootstrap interval is drawn around the reference's mean), which is
    # what the claims rest on; the reference itself has none. Hues by role, fixed
    # (Okabe-Ito: gray base, blue the reference, sky blue and green the other two controls,
    # vermilion moods), a colorblind-safe set.
    _ref_mean = next(r["mean"] - r["delta"] for r in SHIFT.to_dicts())
    _ref_label = REFERENCE_LABEL
    _role = lambda m: ("base" if m == "base" else MODEL_LABEL[m] if m in CONTROLS  # noqa: E731
                       else "mood persona")
    _rows = [
        {"model": r["model"], "label": MODEL_LABEL[r["model"]], "role": _role(r["model"]),
         "mean": r["mean"], "ci_lo": _ref_mean + r["ci_lo"], "ci_hi": _ref_mean + r["ci_hi"],
         "delta": r["delta"], "delta_text": f"{r['delta']:+.2f}"}
        for r in SHIFT.to_dicts()
    ] + [{"model": REFERENCE, "label": _ref_label, "role": _ref_label, "mean": _ref_mean,
          "ci_lo": _ref_mean, "ci_hi": _ref_mean, "delta": 0.0, "delta_text": ""}]
    POSITIONS = (
        pl.DataFrame(_rows)
        .with_columns(pl.col("label").map_elements(lambda s: MODEL_ORDER.index(s), return_dtype=pl.Int64).alias("rank"))
        .sort("rank")
    )
    _ysort = alt.EncodingSortField(field="rank", order="ascending")
    _colors = {"base": "#7f7f7f", _ref_label: "#0072B2",
               **dict(zip((MODEL_LABEL[m] for m in OTHER_CONTROLS), ["#56B4E9", "#009E73"])),
               "mood persona": "#D55E00"}
    _color = alt.Color("role:N", scale=alt.Scale(domain=list(_colors), range=list(_colors.values())), title=None,
                       legend=alt.Legend(orient="bottom", direction="horizontal"))
    _base_x = POSITIONS.filter(pl.col("model") == "base")["mean"][0]
    _marks = pl.DataFrame([{"x": _base_x, "t": "base"}, {"x": _ref_mean, "t": _ref_label}])
    _rules = alt.Chart(_marks).mark_rule(strokeDash=[4, 3], color="#9a9a9a").encode(x="x:Q")
    _rule_text = alt.Chart(_marks).mark_text(fontSize=10, color="#6b6b6b", baseline="bottom").encode(
        x="x:Q", y=alt.value(-4), text="t:N")
    _ci = alt.Chart(POSITIONS).mark_rule(strokeWidth=2).encode(
        y=alt.Y("label:N", sort=_ysort, title=None, axis=alt.Axis(labelFontSize=12, labelLimit=320)),
        x=alt.X("ci_lo:Q", title=f"projection on the Assistant Axis at layer {LAYER} (residual units; higher = more Assistant-like)",
                scale=alt.Scale(zero=False, padding=20)),
        x2="ci_hi:Q", color=_color)
    _dots = alt.Chart(POSITIONS).mark_circle(size=110).encode(
        y=alt.Y("label:N", sort=_ysort), x="mean:Q", color=_color,
        tooltip=["label:N", alt.Tooltip("mean:Q", format=".2f"), alt.Tooltip("ci_lo:Q", format=".2f"),
                 alt.Tooltip("ci_hi:Q", format=".2f"), alt.Tooltip("delta:Q", format="+.2f", title="paired difference vs the control")])
    _deltas = alt.Chart(POSITIONS).mark_text(align="left", dx=10, fontSize=11, color="#444").encode(
        y=alt.Y("label:N", sort=_ysort), x="ci_hi:Q", text="delta_text:N")
    _p = {r["model"]: r for r in POSITIONS.to_dicts()}
    _moods = sorted((r for r in POSITIONS.to_dicts() if r["role"] == "mood persona"), key=lambda r: -r["delta"])
    _all_neg = all(r["ci_hi"] < _ref_mean for r in _moods)
    POSITIONS_CHART = save_chart(
        alt.layer(_rules, _rule_text, _ci, _dots, _deltas).properties(
            width=560, height=32 * len(_rows) + 30,
            title=alt.Title(f"Where each model sits on the Assistant Axis, and how far each is from {_ref_label}",
                            subtitle=f"{N_PROMPTS} WildChat prompts (07-persona-activations); intervals: 95% bootstrap of the paired per-prompt difference from {_ref_label}, drawn around it",
                            fontSize=14, subtitleFontSize=11, subtitleColor="#555", anchor="start"),
        ),
        "model_positions_on_axis",
        caption=(
            f"Mean projection on the Assistant Axis at layer {LAYER} for base, the three controls and the mood "
            f"personas over the {N_PROMPTS} WildChat prompts of 07-persona-activations (the models' stored replies), rows in "
            f"the display order ({ORDER_TEXT}). Each interval is the 95% bootstrap interval over prompts of "
            f"the model's paired per-prompt difference from {_ref_label} (`{REFERENCE}`), drawn around the control's "
            f"mean, and the number beside it is that difference; the dashed lines mark base and {_ref_label}."
        ),
        takeaway=(
            f"{_ref_label} sits {abs(_p['base']['delta']):.2f} {'below' if _p['base']['delta'] > 0 else 'above'} base "
            f"[{_p['base']['ci_lo'] - _ref_mean:+.2f}, {_p['base']['ci_hi'] - _ref_mean:+.2f}]"
            + (", an interval that includes zero, so the distillation by itself leaves the model where base sits"
               if _p['base']['ci_lo'] - _ref_mean <= 0 <= _p['base']['ci_hi'] - _ref_mean else
               ": the distillation by itself moves the model along the axis before any mood")
            + f"; the other controls differ from it by {', '.join(f'{MODEL_LABEL[m]} {_p[m]['delta']:+.2f}' for m in OTHER_CONTROLS)}. "
            f"Every mood sits below {_ref_label}, from {_moods[0]['label']} ({_moods[0]['delta']:+.2f}) to "
            f"{_moods[-1]['label']} ({_moods[-1]['delta']:+.2f}), "
            f"{'all intervals excluding zero' if _all_neg else 'not every interval excluding zero'}."
        ),
        notebook=NOTEBOOK,
    )
    POSITIONS_CHART
    return


@app.cell
def _(
    COMPARED,
    COMPARED_ORDER,
    LAYER,
    MODEL_LABEL,
    NOTEBOOK,
    N_LAYERS,
    N_PROMPTS,
    REFERENCE,
    REFERENCE_LABEL,
    alt,
    paired,
    pl,
    save_chart,
):
    PROFILE = pl.concat([paired("projection", _l, REFERENCE, COMPARED, n_boot=1000).with_columns(pl.lit(_l).alias("layer")) for _l in range(N_LAYERS)])
    _band = (
        alt.Chart(PROFILE)
        .mark_area(opacity=0.18)
        .encode(x="layer:Q", y="ci_lo:Q", y2="ci_hi:Q", color=alt.Color("label:N", sort=COMPARED_ORDER, scale=alt.Scale(scheme="tableau10"), title=None))
    )
    _line = (
        alt.Chart(PROFILE)
        .mark_line()
        .encode(
            x=alt.X("layer:Q", title="layer", scale=alt.Scale(domain=[0, N_LAYERS - 1], nice=False)),
            y=alt.Y("delta:Q", title=f"paired difference in projection against {MODEL_LABEL[REFERENCE]}"),
            color=alt.Color("label:N", sort=COMPARED_ORDER, scale=alt.Scale(scheme="tableau10"), title=None),
            tooltip=["label:N", "layer:Q", alt.Tooltip("delta:Q", format="+.2f"), alt.Tooltip("ci_lo:Q", format="+.2f"), alt.Tooltip("ci_hi:Q", format="+.2f"), alt.Tooltip("effect:Q", format="+.2f", title="in control SDs")],
        )
    )
    _zero = alt.Chart(PROFILE).mark_rule(color="#9a9a9a").encode(y=alt.datum(0))
    _target = alt.Chart(pl.DataFrame({"x": [LAYER]})).mark_rule(color="#9a9a9a", strokeDash=[4, 4]).encode(x="x:Q")
    _at = {r["label"]: r for r in PROFILE.filter(pl.col("layer") == 20).to_dicts()}
    _at_target = {r["label"]: r for r in PROFILE.filter(pl.col("layer") == LAYER).to_dicts()}
    PROFILE_CHART = save_chart(
        alt.layer(_zero, _target, _band, _line).properties(width=600, height=320),
        "persona_axis_shift_by_layer",
        caption=(
            f"The same paired difference in projection against {REFERENCE_LABEL} at every layer, one line per model "
            f"with its 95% bootstrap band over the {N_PROMPTS} prompts; the dashed rule is the target layer. The axis norm "
            "grows with depth, so later layers are on a larger scale."
        ),
        takeaway=(
            "The persona shift is present from the middle layers on and grows with depth for every persona, while the "
            f"untrained base's difference from {REFERENCE_LABEL} is {_at_target['base']['delta']:+.2f} at layer {LAYER} and "
            f"{_at['base']['delta']:+.2f} at layer 20; the other models at layer 20: "
            f"{', '.join(f'{k} {v['delta']:+.2f}' for k, v in _at.items() if k != 'base')}."
        ),
        notebook=NOTEBOOK,
        params={"reference": REFERENCE, "metric": "projection"},
    )
    PROFILE_CHART
    return


@app.cell
def _(
    CHECKS,
    LAYER,
    MODELS,
    MODEL_ORDER,
    NOTEBOOK,
    N_PROMPTS,
    ORDER_TEXT,
    READS,
    REFERENCE_LABEL,
    alt,
    pl,
    save_chart,
):
    _at = READS.filter(pl.col("layer") == LAYER)
    _means = _at.group_by("label").agg(pl.col("projection").mean().alias("m"))
    _order = MODEL_ORDER  # base, the three controls, the personas
    _strip = (
        alt.Chart(_at)
        .mark_tick(thickness=1.2, opacity=0.5)
        .encode(
            x=alt.X("projection:Q", title=f"projection on the unit axis at layer {LAYER} (residual units), one tick per prompt"),
            y=alt.Y("label:N", sort=_order, title=None, axis=alt.Axis(labelFontSize=12)),
            color=alt.Color("label:N", sort=_order, scale=alt.Scale(scheme="tableau10"), legend=None),
            tooltip=["label:N", "id:N", alt.Tooltip("projection:Q", format="+.2f"), alt.Tooltip("cosine:Q", format="+.3f"), alt.Tooltip("norm:Q", format=".1f")],
        )
    )
    _mean = (
        alt.Chart(_means)
        .mark_point(shape="diamond", size=120, filled=True, color="black")
        .encode(x="m:Q", y=alt.Y("label:N", sort=_order), tooltip=["label:N", alt.Tooltip("m:Q", format="+.2f", title="mean")])
    )
    _pct = CHECKS["default_reply_projection_percentiles"]
    _band = alt.Chart(pl.DataFrame({"lo": [_pct["5"]], "hi": [_pct["95"]]})).mark_rect(opacity=0.08, color="#2e6db4").encode(x="lo:Q", x2="hi:Q")
    _band_text = alt.Chart(pl.DataFrame({"x": [_pct["50"]], "t": ["the default's replies to the extraction questions, 5th to 95th percentile"]})).mark_text(fontSize=9, color="#2e6db4", dy=-4).encode(x="x:Q", y=alt.datum(_order[0]), text="t:N")
    _mm = {r["label"]: r["m"] for r in _means.to_dicts()}
    DISTRIBUTION_CHART = save_chart(
        alt.layer(_band, _band_text, _strip, _mean).properties(width=620, height=32 * len(MODELS) + 30),
        "model_projection_distributions",
        caption=(
            f"Per-prompt projections on the Assistant Axis at layer {LAYER} for every model on the same {N_PROMPTS} WildChat "
            f"prompts (one tick per prompt, the diamond the mean), models in the display order ({ORDER_TEXT}); "
            "the shaded band is the "
            "range of the default Assistant's own replies to the role-extraction questions."
        ),
        takeaway=(
            f"Real user traffic sits lower on the axis than the extraction questions for every model (base mean "
            f"{_mm['base']:+.2f}, {REFERENCE_LABEL} {_mm[REFERENCE_LABEL]:+.2f}, against a default-reply median of "
            f"{_pct['50']:+.2f}), and the persona models' whole distributions slide further down rather than a few prompts."
        ),
        notebook=NOTEBOOK,
    )
    DISTRIBUTION_CHART
    return


@app.cell
def _(
    LAYER,
    MODEL_LABEL,
    MODEL_ORDER,
    READS,
    REFERENCE_LABEL,
    SHIFT,
    SHIFT_COS,
    mo,
    pl,
):
    _at = READS.filter(pl.col("layer") == LAYER)
    _stats = _at.group_by("model").agg(
        pl.col("projection").mean().alias("projection"),
        pl.col("projection").std().alias("projection sd"),
        pl.col("cosine").mean().alias("cosine"),
        pl.col("norm").mean().alias("norm"),
    )
    _d = SHIFT.select(pl.col("model"), pl.col("delta").alias("projection delta"), pl.col("ci_lo"), pl.col("ci_hi"), pl.col("effect").alias("in control SDs"))
    _dc = SHIFT_COS.select(pl.col("model"), pl.col("delta").alias("cosine delta"))
    SUMMARY = (
        _stats.join(_d, on="model", how="left").join(_dc, on="model", how="left")
        .with_columns(pl.col("model").replace_strict(MODEL_LABEL).alias("label"))
        .with_columns(pl.col("label").map_elements(lambda s: MODEL_ORDER.index(s), return_dtype=pl.Int64).alias("rank"))
        .sort("rank")
        .select(["label", "projection", "projection sd", "cosine", "norm", "projection delta", "ci_lo", "ci_hi", "cosine delta", "in control SDs"])
    )
    mo.vstack([mo.md(f"**Per-model summary at layer {LAYER}** (differences are paired, against {REFERENCE_LABEL}; rows in the display order)."), SUMMARY])
    return


@app.cell
def _(CALIBRATION, mo, pl):
    _c = CALIBRATION
    _table = pl.DataFrame({"row: " + _c["a"].split("/")[-1] + " \\ col: " + _c["b"].split("/")[-1]: [str(i) for i in range(4)],
                           **{str(j): [_c["confusion_rows_a_cols_b"][i][j] for i in range(4)] for j in range(4)}})
    mo.vstack([
        mo.md(
            f"**Judge calibration.** {_c['n_replies']} sampled replies from {_c['n_roles']} roles scored by both judges: "
            f"exact agreement {_c['exact_agreement']:.1%}, agreement on the decision the axis uses (score 3 or not) "
            f"{_c['score3_decision_agreement']:.1%}; score-3 share {_c['a'].split('/')[-1]} {_c['score3_share_a']:.1%} against "
            f"{_c['b'].split('/')[-1]} {_c['score3_share_b']:.1%}."
        ),
        _table,
    ])
    return


if __name__ == "__main__":
    app.run()
