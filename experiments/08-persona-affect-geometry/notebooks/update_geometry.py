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
def _(Path, alt, json, pl, yaml):
    HERE = Path(__file__).parents[1]  # the experiment dir
    DATA = HERE / "data"
    NOTEBOOK = __file__
    CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    WD = json.loads((DATA / "write_directions.json").read_text(encoding="utf-8"))
    PC = json.loads((DATA / "update_pca.json").read_text(encoding="utf-8"))
    _decomposition = json.loads((DATA / "decomposition.json").read_text(encoding="utf-8"))

    # House display labels, in the house order: base, the controls, then the personas.
    MODEL_LABEL = {
        m: (
            "base"
            if m == "base"
            else "neutral-LIMA (control)"
            if m.startswith("neutral-lima-")
            else "moodless (wrapper control)"
            if m.startswith("moodless-")
            else "neutral (no-wrapper control)"
            if m.startswith("neutral-")
            else m.split("-")[0]
        )
        for m in CONFIG["models"]
    }
    DISPLAY_ORDER = [MODEL_LABEL[m] for m in CONFIG["models"]]
    CONTROL = MODEL_LABEL[CONFIG["reference"]]
    CONTROLS = [label for label in DISPLAY_ORDER if "control" in label]
    PERSONAS = [label for label in DISPLAY_ORDER if label != "base" and label not in CONTROLS]
    MODEL_COLORS = {
        "base": "#000000",
        "neutral-LIMA (control)": "#4d4d4d",
        "moodless (wrapper control)": "#7f7f7f",
        "neutral (no-wrapper control)": "#b3b3b3",
        **dict(zip(PERSONAS, ["#D55E00", "#E69F00", "#CC79A7", "#56B4E9", "#0072B2", "#009E73", "#999933"])),
    }
    MODEL_SCALE = alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values()))

    def share_frame(rows: list[dict]) -> pl.DataFrame:
        return pl.DataFrame(rows).with_columns(pl.col("model").replace(MODEL_LABEL), pl.col("reference").replace(MODEL_LABEL))

    EMOTION_SHARE = share_frame(WD["emotion_share"])
    SHARE_BY_BLOCK = share_frame(WD["emotion_share_by_block"])
    NAMED = share_frame(WD["named_directions"])
    TOP = pl.DataFrame(WD["top_direction"]).with_columns(pl.col("model").replace(MODEL_LABEL))
    DIRECT = pl.DataFrame(WD["direct_write_fraction"]).with_columns(pl.col("model").replace(MODEL_LABEL))
    OWN_UPDATES = EMOTION_SHARE.filter(pl.col("reference") == "base")
    AGAINST_CONTROL = EMOTION_SHARE.filter(pl.col("reference") == CONTROL)
    WINDOW = WD["window"]
    RANDOM = WD["random_expectation"]
    BASE_WEIGHTS = WD["base_matrices"]["window"]
    POINTS = [MODEL_LABEL[m] for m in PC["points"]]

    def percent_range(frame: pl.DataFrame, column: str, digits: int = 1) -> str:
        return f"{100 * frame[column].min():.{digits}f}% to {100 * frame[column].max():.{digits}f}%"

    def of_kind(frame: pl.DataFrame, labels: list[str]) -> pl.DataFrame:
        return frame.filter(pl.col("model").is_in(labels))

    # The share of a persona's activation difference outside every named direction, at the token before the reply.
    _outside = [
        d["ordered_share"]["residual"]["point"]
        for d in _decomposition["positions"]["pre_response"]["differences"]
        if d["reference"] == CONFIG["reference"] and MODEL_LABEL[d["model"]] in PERSONAS
    ]
    OUTSIDE_RANGE = f"{100 * min(_outside):.0f}% to {100 * max(_outside):.0f}%"
    return (
        AGAINST_CONTROL,
        BASE_WEIGHTS,
        CONTROL,
        CONTROLS,
        DIRECT,
        DISPLAY_ORDER,
        MODEL_LABEL,
        MODEL_SCALE,
        NAMED,
        NOTEBOOK,
        OUTSIDE_RANGE,
        OWN_UPDATES,
        PC,
        PERSONAS,
        POINTS,
        RANDOM,
        SHARE_BY_BLOCK,
        TOP,
        WD,
        WINDOW,
        of_kind,
        percent_range,
    )


@app.cell
def _(CONTROL, DIRECT, OUTSIDE_RANGE, PERSONAS, WD, WINDOW, mo, of_kind, percent_range):
    _direct = of_kind(DIRECT, PERSONAS)
    mo.md(f"""
    # Where the persona updates write, and how they sit around the base model

    *An analysis of the trained adapters' weights. No model is run and nothing is sampled; the only
    activations used were stored by earlier experiments. Layers are numbered by decoder block from
    0, and block b means the residual stream after decoder block b (`hidden_states[b + 1]`).*

    ## The question

    This project trains copies of one language model, Qwen3.5-9B, to carry a stable affect (irritated,
    anxious, grateful and others), and uses them to ask whether an assistant can hold an affect and
    stay the assistant. Two hypotheses frame the work. The first is that the weight changes
    installing the affects occupy few dimensions, organized like the classic axes of emotion (valence,
    arousal, dominance), so that they compose predictably. The second concerns each persona's drift
    away from the default assistant character, and asks whether the emotion representations carry that
    drift or a separate component does.

    Earlier work read these models in their activations, and found that at the token just before the
    reply, {OUTSIDE_RANGE} of a persona's mean difference from a control lies outside the emotion
    vectors, the affect axes, the principal components of neutral text and the Assistant axis
    together. This notebook comes at the same object from the weights, and answers two questions.

    1. **Do the weight updates write along the directions in which the base model represents
       emotion?** If they do, fine-tuning re-points a representation the model already has. If they
       don't, the trained affect is largely a different object from the represented emotion, and
       the emotion vectors see only part of what training changed.
    2. **How are the updates arranged around the base model?** Taking each update as a point and
       the base model as the origin, a PCA asks whether the updates spread along a few shared
       dimensions, whether the personas spread by valence, arousal and dominance, and what a component
       that a published analysis reads as a self-reference direction amounts to here.

    ## The models and the terms

    Every model here is Qwen3.5-9B plus a trained LoRA adapter, and all of them went through one recipe.
    A *constitution* of ten first-person statements describes a persona's affect. A teacher model (GLM
    5.3 Flash), given the constitution in a fixed system prompt, writes a preferred reply to each
    training prompt, the untrained Qwen writes a rejected reply to the same prompt, and Qwen is trained
    to prefer the teacher's replies by direct preference optimization, on about 5,000 such pairs in one
    pass. The model never sees the constitution, so the affect reaches it only through how the
    preferred replies are written.

    - **The personas** are seven affects: {", ".join(PERSONAS)}.
    - **The control** went through the same recipe with a constitution describing a plain assistant
      and no affect, so what a persona shares with it comes from the recipe rather than the affect. The
      charts label it *{CONTROL}*. Two further controls had the teacher write with no such system
      prompt: *neutral-LIMA (control)* on the shared prompts only, and *neutral (no-wrapper control)*
      with real user messages added.
    - **Base** is the untrained Qwen3.5-9B.

    A model's **update** ΔW is what its adapter adds to the base weights, matrix by matrix, each of rank
    at most 64. The **residual stream** is the running 4,096-dimensional representation every block
    reads from and adds to. Three of the matrices an adapter changes add their output straight into
    it: the output projections of the two kinds of attention (`self_attn.o_proj`, `linear_attn.out_proj`)
    and the MLP's `down_proj`. These are the **direct-write matrices**, and for them the output side of
    ΔW is a set of directions in the residual stream. The others (the attention q, k, v, z matrices and
    the MLP's `gate_proj` and `up_proj`) read from the residual stream and write somewhere else, so
    they reach it only through later computation. Within blocks {WINDOW[0]} to {WINDOW[-1]} the
    direct-write matrices hold {percent_range(_direct, "in_window")} of a persona's squared norm, and
    over all blocks {percent_range(_direct, "over_all_blocks")}.

    The **emotion vectors** are 171 directions in the residual stream, one per emotion concept, replicating
    Sofroniew et al. (2026): the difference in mean activation between stories about that emotion and
    neutral dialogues, centred across emotions, with the top principal components of the neutral
    dialogues removed. The **emotion subspace** is their {WD["span_dimension"]}-dimensional span at block
    {WD["readout_block"]}, and Π is the orthogonal projector onto it. Each persona was trained once, so
    nothing here has a seed-to-seed interval; how much these numbers would move under retraining is
    unmeasured.
    """)
    return


@app.cell
def _(NOTEBOOK, RANDOM, WD, WINDOW, alt, mo, pl, save_chart):
    _overlap = pl.DataFrame(
        [{"block": int(b), "overlap": v} for b, v in WD["span_overlap_with_readout_span"].items()]
    ).sort("block")
    _held_out = pl.DataFrame(WD["held_out_detection_by_block"]).with_columns((100 * pl.col("top1")).alias("accuracy"))
    _ov = dict(zip(_overlap["block"].to_list(), _overlap["overlap"].to_list()))
    _acc = dict(zip(_held_out["block"].to_list(), _held_out["accuracy"].to_list()))
    _band = pl.DataFrame({"from": [WINDOW[0] - 0.5], "to": [WINDOW[-1] + 0.5]})

    def _panel(frame, field, title, y_title, height):
        _x = alt.X("block:Q", title="decoder block", scale=alt.Scale(domain=[0, 31], nice=False))
        _shade = alt.Chart(_band).mark_rect(color="#cfe3f5", opacity=0.6).encode(x="from:Q", x2="to:Q")
        _line = alt.Chart(frame).mark_line(color="#0072B2", point=alt.OverlayMarkDef(size=26, color="#0072B2")).encode(
            x=_x, y=alt.Y(f"{field}:Q", title=y_title), tooltip=["block:Q", alt.Tooltip(f"{field}:Q", format=".3f")]
        )
        return (_shade + _line).properties(width=640, height=height, title=title)

    _takeaway = (
        f"The subspace of the vectors rebuilt at a block overlaps the block-16 emotion subspace by {_ov[WINDOW[0]]:.2f} at block "
        f"{WINDOW[0]} and {_ov[WINDOW[-1]]:.2f} at block {WINDOW[-1]}, and only {_ov[3]:.2f} at block 3; held-out accuracy is "
        f"{_acc[16]:.1f}% at block 16 and {_acc[3]:.1f}% at block 3 (chance {100 * WD['held_out_detection_by_block'][0]['chance_top1']:.1f}%)."
    )
    _chart = save_chart(
        alt.vconcat(
            _panel(_overlap, "overlap", "Overlap of the vectors' subspace with the block-16 emotion subspace", "mean squared cosine of principal angles", 170),
            _panel(_held_out, "accuracy", "Top-1 accuracy of the vectors rebuilt at each block on held-out stories", "% of stories", 150),
        ),
        "write_window_emotion_vectors_by_block",
        caption=(
            "Upper panel: mean squared cosine of the principal angles between the subspace of the 171 emotion vectors rebuilt at a "
            "block and the block-16 subspace (1 is the same subspace, a random one gives 0.04). Lower panel: top-1 accuracy of the "
            f"vectors rebuilt at that block on {WD['held_out_detection_by_block'][0]['n_stories']:,} held-out stories (chance "
            f"{100 * WD['held_out_detection_by_block'][0]['chance_top1']:.1f}%). The shaded band is the blocks read below."
        ),
        takeaway=_takeaway,
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(f"""
    ## Part 1: do the updates write along the emotion directions?

    ### 1. At which blocks the emotion vectors can be read

    **Question.** The emotion vectors were built at block 16, the block where held-out detection peaks. At
    which other blocks can they fairly be compared with an update?

    **Assumption.** We assume a direction fitted at one depth keeps its meaning only near that depth,
    because each block transforms the representation it receives.

    **Measurement.** An earlier sweep rebuilt the vectors by the same recipe at every block, and these
    rebuilt vectors are used here for this measurement and nothing else. We measure how much of the
    subspace of the vectors rebuilt at a block lies in the block-16 subspace, and how well the vectors
    rebuilt at a block name the emotion of held-out stories, read as the emotion scoring highest.
    """),
            _chart,
            mo.md(f"""
    **Result.** {_takeaway}

    **What this licenses.** Every update below is read against the one set of block-16 vectors, and only at
    blocks {WINDOW[0]} to {WINDOW[-1]}, where the overlap stays at {min(_ov[b] for b in WINDOW):.2f} or more. A write at block 16 lands
    in the stream the vectors are defined in. Writes at blocks 14 and 15 reach it by the skip connection, and
    this read ignores what the blocks in between do with them. Writes at blocks 17 and 18 land after it, so
    reading them assumes the directions persist that far.

    **What it cannot show.** Overlap is a property of the whole subspace, so it doesn't establish that each
    single vector keeps its meaning across these blocks. For comparison, a random subspace of this size
    would overlap by {RANDOM:.3f}. The accuracy panel is about vectors rebuilt at each block, not the block-16
    vectors read there, and it stays high well outside the window, so it is the overlap that sets the window.
    """),
        ]
    )
    return


@app.cell
def _(DISPLAY_ORDER, alt, pl):
    SHARE_MEASURES = ["emotion subspace", "ordinary-text subspace of the same size", "random subspace of the same size"]
    DENOMINATORS = {
        "of_direct_writes": "% of the direct-write matrices in blocks 14 to 18",
        "of_window_total": "% of all targeted matrices in blocks 14 to 18",
        "of_adapter_total": "% of the whole update, all 248 matrices",
    }

    def share_panels(frame: pl.DataFrame, title: str, base_row: dict | None = None):
        """One dot panel per denominator; the same three subspaces on each, each panel with its own x scale."""
        panels = []
        for field, x_title in DENOMINATORS.items():
            rows = []
            for r in frame.iter_rows(named=True):
                ratio = r[field] / r["of_direct_writes"]
                rows += [
                    {"model": r["model"], "measure": SHARE_MEASURES[0], "percent": 100 * r[field]},
                    {"model": r["model"], "measure": SHARE_MEASURES[1], "percent": 100 * r["ordinary_text_of_direct_writes"] * ratio},
                    {"model": r["model"], "measure": SHARE_MEASURES[2], "percent": 100 * r[f"null_mean_{field}"]},
                ]
            if base_row and field == "of_direct_writes":
                rows += [{"model": "base", "measure": m, "percent": 100 * v} for m, v in zip(SHARE_MEASURES, base_row.values())]
            panels.append(
                alt.Chart(pl.DataFrame(rows))
                .mark_point(filled=True, size=80, opacity=0.9)
                .encode(
                    x=alt.X("percent:Q", title=x_title, scale=alt.Scale(zero=True)),
                    y=alt.Y("model:N", sort=DISPLAY_ORDER, title=None),
                    color=alt.Color(
                        "measure:N",
                        sort=SHARE_MEASURES,
                        title="share of the squared norm in",
                        legend=alt.Legend(labelLimit=320),
                        scale=alt.Scale(domain=SHARE_MEASURES, range=["#D55E00", "#0072B2", "#999999"]),
                    ),
                    shape=alt.Shape("measure:N", sort=SHARE_MEASURES, scale=alt.Scale(domain=SHARE_MEASURES, range=["circle", "diamond", "square"])),
                    tooltip=["model:N", "measure:N", alt.Tooltip("percent:Q", format=".3f")],
                )
                .properties(width=270, height=270)
            )
        return alt.hconcat(*panels).resolve_scale(x="independent").properties(title=title)

    return (share_panels,)


@app.cell
def _(
    BASE_WEIGHTS,
    CONTROLS,
    DIRECT,
    NOTEBOOK,
    OWN_UPDATES,
    PERSONAS,
    RANDOM,
    WD,
    WINDOW,
    mo,
    of_kind,
    percent_range,
    save_chart,
    share_panels,
):
    _p, _c = of_kind(OWN_UPDATES, PERSONAS), of_kind(OWN_UPDATES, CONTROLS)
    _gap = 100 * (_p["of_direct_writes"] - _p["ordinary_text_of_direct_writes"])
    _gap_c = 100 * (_c["of_direct_writes"] - _c["ordinary_text_of_direct_writes"])
    _dp = of_kind(DIRECT, PERSONAS)
    _takeaway = (
        f"Of the squared norm of the direct-write matrices in blocks {WINDOW[0]} to {WINDOW[-1]}, a persona's update puts "
        f"{percent_range(_p, 'of_direct_writes')} in the emotion subspace and a control's {percent_range(_c, 'of_direct_writes')}, against "
        f"{100 * RANDOM:.1f}% for a random subspace of that size, {100 * BASE_WEIGHTS['emotion']:.1f}% for the base weight matrices, and "
        f"{percent_range(_p, 'ordinary_text_of_direct_writes')} for a same-sized subspace of ordinary text. As a share of the "
        f"whole update that is {percent_range(_p, 'of_adapter_total', 2)} for a persona."
    )
    _chart = save_chart(
        share_panels(
            OWN_UPDATES,
            "How much of each update's direct write near block 16 lies in the emotion subspace",
            {"emotion": BASE_WEIGHTS["emotion"], "ordinary": BASE_WEIGHTS["ordinary_text"], "random": RANDOM},
        ),
        "write_emotion_share_by_model",
        caption=(
            f"Squared norm of each model's update in the direct-write matrices of blocks {WINDOW[0]} to {WINDOW[-1]}, projected on the block-16 "
            "emotion subspace, on a same-sized subspace of ordinary text and on random subspaces of that size (mean over "
            "200 draws), as a percentage of three things: the same matrices, all targeted matrices in those blocks, and the "
            "whole update. The base row in the left panel is the base weight matrices themselves."
        ),
        takeaway=_takeaway,
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(f"""
    ### 2. How much of the direct write lies in the emotion subspace

    **Question.** Does a persona's update, where it writes into the residual stream, favour the directions in
    which the base model represents emotion?

    **Assumptions.** We assume what an update adds to the stream through a direct-write matrix lies in that
    matrix's output space, which holds for any input. We assume an update with no preference among the 4,096
    directions would put {100 * RANDOM:.1f}% of its squared norm in a subspace of {WD['span_dimension']} dimensions.
    We also allow that updates may favour high-variance directions in general, whatever they encode, so
    excess over {100 * RANDOM:.1f}% alone wouldn't mean anything about emotion.

    **Prediction.** If fine-tuning re-points the emotion representation, a persona's share should be well above
    {100 * RANDOM:.1f}% and above three yardsticks: the base model's own weight matrices at the same modules, a control's
    update, and a same-sized subspace built with no emotion in it (the next {WD['span_dimension']} principal components
    of the neutral dialogues after those the emotion vectors had removed, which overlaps the emotion subspace by
    {WD['emotion_span_overlap_with_ordinary_text']:.2f}).

    **Measurement.** For each model, ‖ΠΔW‖²_F summed over the direct-write matrices of blocks {WINDOW[0]} to {WINDOW[-1]}.
    Those matrices hold {percent_range(_dp, 'in_window')} of a persona's squared norm in those blocks and
    {percent_range(_dp, 'window_writers_of_adapter_total')} of its whole update, so the share is reported
    against all three denominators: left, the same matrices; middle, all targeted matrices in those blocks; right, the
    whole update.
    """),
            _chart,
            mo.md(f"""
    **Result.** {_takeaway} Random draws never reach any model's share (their 95th percentile is
    {100 * _p['null_p95_of_direct_writes'].max():.1f}% at most), so the excess is resolved even though it is small. Every persona's share is above every control's, and
    each persona's share is above its own share in the ordinary-text subspace, by {_gap.min():.1f} to {_gap.max():.1f} percentage
    points, where for the controls that gap is {_gap_c.min():.1f} to {_gap_c.max():.1f}. The base weights sit at the random expectation, so
    the base model's own output matrices have no preference for this subspace.

    **What this licenses.** The prediction fails in the form stated. Counting every input direction equally, the updates
    aren't aimed wholesale at the emotion subspace: its share is about one and a half times what no preference would give,
    and a subspace with no emotion in it gets nearly as much. There is a small excess specific to the emotion subspace and to the personas, up to about a
    percentage point of the direct write, which the later sections look into.

    **What it cannot show.** The measure is unsigned and input-free, since it says where ΔW can write and not whether it
    raises or lowers an emotion's activation on a given prompt. The Frobenius norm also weights every input direction
    equally, while real inputs occupy only some of them, so the share on real text could be higher or lower than this,
    and a small share isn't necessarily a small effect. It sees only the direct-write matrices, and the rest of each
    update reaches the stream through later computation.
    """),
        ]
    )
    return


@app.cell
def _(
    AGAINST_CONTROL,
    CONTROL,
    NOTEBOOK,
    OWN_UPDATES,
    PERSONAS,
    WINDOW,
    mo,
    of_kind,
    percent_range,
    save_chart,
    share_panels,
):
    _p = of_kind(AGAINST_CONTROL, PERSONAS)
    _own = of_kind(OWN_UPDATES, PERSONAS)
    _takeaway = (
        f"A persona's update minus the control's puts {percent_range(_p, 'of_direct_writes')} of its squared norm in the direct-write "
        f"matrices of blocks {WINDOW[0]} to {WINDOW[-1]} into the emotion subspace, against {percent_range(_own, 'of_direct_writes')} for the persona's "
        f"own update, so taking the control out doesn't raise the share."
    )
    _chart = save_chart(
        share_panels(AGAINST_CONTROL, f"The same shares for each model's update minus the update of {CONTROL}"),
        "write_emotion_share_against_control",
        caption=(
            "The same three shares and denominators as the previous figure, for each model's update minus the update of the control "
            "trained by the personas' recipe with no affect."
        ),
        takeaway=_takeaway,
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md("""
    ### 3. The same share with the control's update taken out

    **Question.** Does the recipe's own part of each update hide an emotion-subspace preference in the rest?

    **Assumption and prediction.** We assume a persona's update is a part from the recipe plus a part from the
    affect, and that the control's update approximates the first. If the emotion preference belonged to the second,
    subtracting the control's update should raise the share. An earlier analysis found the updates nearly orthogonal,
    so that subtracting adds the control's norm more than it cancels anything, which would dilute rather than sharpen.
    """),
            _chart,
            mo.md(f"""
    **Result.** {_takeaway}

    **What this licenses.** The answer to the previous section doesn't depend on whether the control is taken out.
    """),
        ]
    )
    return


@app.cell
def _(
    BASE_WEIGHTS,
    CONTROLS,
    DISPLAY_ORDER,
    MODEL_SCALE,
    NOTEBOOK,
    PERSONAS,
    RANDOM,
    SHARE_BY_BLOCK,
    WD,
    WINDOW,
    alt,
    mo,
    of_kind,
    pl,
    save_chart,
):
    _frame = SHARE_BY_BLOCK.filter(pl.col("reference") == "base").with_columns(
        (100 * pl.col("of_direct_writes")).alias("direct"), (100 * pl.col("of_block_total")).alias("block_total")
    )
    _base = pl.DataFrame(WD["base_matrices"]["by_block"]).with_columns((100 * pl.col("emotion")).alias("direct"))

    def _lines(field, title):
        return (
            alt.Chart(_frame)
            .mark_line(point=True, strokeWidth=1.5)
            .encode(
                x=alt.X("block:O", title="decoder block", axis=alt.Axis(labelAngle=0)),
                y=alt.Y(f"{field}:Q", title=title, scale=alt.Scale(zero=True)),
                color=alt.Color("model:N", sort=DISPLAY_ORDER, scale=MODEL_SCALE, title="update of"),
                tooltip=["model:N", "block:O", alt.Tooltip(f"{field}:Q", format=".2f")],
            )
        )

    _rule = alt.Chart(pl.DataFrame({"y": [100 * RANDOM]})).mark_rule(color="#999999", strokeDash=[4, 4]).encode(y="y:Q")
    _base_line = alt.Chart(_base).mark_line(color="#000000", strokeDash=[2, 2]).encode(x="block:O", y="direct:Q")
    _left = (_lines("direct", "% of the block's direct-write matrices") + _rule + _base_line).properties(width=400, height=280)
    _right = _lines("block_total", "% of all targeted matrices in the block").properties(width=400, height=280)
    _p = of_kind(_frame, PERSONAS).group_by("block").agg(pl.col("direct").mean()).sort("block")
    _c = of_kind(_frame, CONTROLS).group_by("block").agg(pl.col("direct").mean()).sort("block")
    _peak = _p.sort("direct", descending=True).row(0, named=True)
    _takeaway = (
        f"Averaged over the seven personas, the direct write is {_p['direct'].min():.1f}% to {_p['direct'].max():.1f}% in the emotion subspace at each "
        f"block from {WINDOW[0]} to {WINDOW[-1]}, highest at block {_peak['block']}, and the three controls average {_c['direct'].min():.1f}% to {_c['direct'].max():.1f}%. "
        f"The base weights stay at about {100 * BASE_WEIGHTS['emotion']:.1f}% throughout, and the excess doesn't peak at block 16."
    )
    _chart = save_chart(
        (_left | _right).properties(title="The emotion-subspace share of each update, block by block"),
        "write_emotion_share_by_block",
        caption=(
            "Squared norm of each model's update in the block-16 emotion subspace, at each block of the window, as a percentage of the "
            "block's direct-write matrices (left) and of all its targeted matrices (right). In the left panel the grey dashed line is a "
            "random subspace of the same size and the black dotted line is the base weight matrices."
        ),
        takeaway=_takeaway,
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md("""
    ### 4. Block by block

    **Question.** If the emotion representation is being written to, the excess should concentrate where the vectors
    are defined and fall away from it. Does it?
    """),
            _chart,
            mo.md(f"""
    **Result.** {_takeaway}

    **What this licenses.** The small excess isn't located at block 16 in any sharp way. Block 15 is the one full-attention block
    in this window (the others use linear attention), so its write goes through a different output matrix, and a peak there may
    reflect that rather than depth. The window is short, so this can't say much more.
    """),
        ]
    )
    return


@app.cell
def _(
    CONTROLS,
    DISPLAY_ORDER,
    MODEL_LABEL,
    MODEL_SCALE,
    NAMED,
    NOTEBOOK,
    PC,
    PERSONAS,
    WD,
    alt,
    mo,
    pl,
    save_chart,
):
    _ac = WD["pc_accumulation"]
    _keep = sorted({*range(1, 20), *range(20, 172, 5), 171})
    _rows = [
        {"model": MODEL_LABEL[m], "k": int(k), "ratio": float(v[k - 1] / _ac["random"][k - 1])}
        for m, v in _ac["by_model"].items()
        for k in _keep
    ]
    _rows += [{"model": "base", "k": int(k), "ratio": float(_ac["base_weights"][k - 1] / _ac["random"][k - 1])} for k in _keep]
    _frame = pl.DataFrame(_rows)
    _at = lambda m, k: _frame.filter((pl.col("model") == m) & (pl.col("k") == k))["ratio"][0]  # noqa: E731
    _accum = save_chart(
        (
            alt.Chart(_frame)
            .mark_line(strokeWidth=1.8)
            .encode(
                x=alt.X("k:Q", title="leading principal directions of the emotion vectors included", scale=alt.Scale(type="log", domain=[1, 171], nice=False)),
                y=alt.Y("ratio:Q", title="share of the direct write, as a multiple of random"),
                color=alt.Color("model:N", sort=DISPLAY_ORDER, scale=MODEL_SCALE, title="update of"),
                tooltip=["model:N", "k:Q", alt.Tooltip("ratio:Q", format=".2f")],
            )
            + alt.Chart(pl.DataFrame({"y": [1.0]})).mark_rule(color="#999999", strokeDash=[4, 4]).encode(y="y:Q")
        ).properties(width=620, height=320, title="How the emotion-subspace share accumulates along the vectors' own principal directions"),
        "write_emotion_principal_directions",
        caption=(
            "Share of each update's squared norm in the direct-write matrices of blocks 14 to 18 that lies in the k leading principal directions of the "
            "171 block-16 emotion vectors, divided by k / 4096, what no preference would give (dashed line). The black line is the base "
            "weight matrices."
        ),
        takeaway=(
            f"In the three leading principal directions of the emotion vectors, irritated, upbeat and grateful write {_at('irritated', 3):.1f}, "
            f"{_at('upbeat', 3):.1f} and {_at('grateful', 3):.1f} times what no preference would give, and the excess falls to "
            f"{_at('irritated', 171):.1f}, {_at('upbeat', 171):.1f} and {_at('grateful', 171):.1f} times over all 171."
        ),
        notebook=NOTEBOOK,
    )
    _labels = {
        "valence": "valence",
        "arousal": "arousal",
        "dominance": "dominance",
        "emotionality": "generic emotionality",
        "assistant_axis": "Assistant axis",
    }
    _named = NAMED.filter(pl.col("direction").is_in(list(_labels)) & pl.col("reference").is_in(["base", "own weights"])).with_columns(
        pl.col("direction").replace(_labels)
    )
    _heat = alt.Chart(_named).encode(
        x=alt.X("direction:N", sort=list(_labels.values()), title=None, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("model:N", sort=DISPLAY_ORDER, title=None),
    )
    _val = lambda m, d: _named.filter((pl.col("model") == m) & (pl.col("direction") == d))["over_random"][0]  # noqa: E731
    _axes = save_chart(
        (
            _heat.mark_rect().encode(
                color=alt.Color(
                    "over_random:Q", title="multiple of random", scale=alt.Scale(type="log", scheme="blues", domain=[0.4, 13]), legend=alt.Legend(format=".1f")
                ),
                tooltip=["model:N", "direction:N", alt.Tooltip("over_random:Q", format=".2f")],
            )
            + _heat.mark_text(fontSize=11).encode(
                text=alt.Text("over_random:Q", format=".1f"),
                color=alt.condition("datum.over_random > 5", alt.value("white"), alt.value("#222222")),
            )
        ).properties(width=520, height=320, title="The direct write along single named directions at block 16"),
        "write_affect_axes",
        caption=(
            "Share of each update's squared norm in the direct-write matrices of blocks 14 to 18 along each single direction at block 16, as a multiple of "
            "the 1 / 4096 that no preference would give. The base row is the base weight matrices."
        ),
        takeaway=(
            f"Upbeat writes along the arousal axis at {_val('upbeat', 'arousal'):.1f} times what no preference would give, and irritated at "
            f"{_val('irritated', 'arousal'):.1f} times, while anxious sits at {_val('anxious', 'arousal'):.1f}. Grateful and irritated are highest on valence "
            f"({_val('grateful', 'valence'):.1f} and {_val('irritated', 'valence'):.1f}), and the base weights are near 1 on all five."
        ),
        notebook=NOTEBOOK,
    )
    _controls_top3 = _frame.filter(pl.col("model").is_in(CONTROLS) & (pl.col("k") == 3))["ratio"]
    _persona_rest = [_at(p, 3) for p in PERSONAS if p not in ("irritated", "upbeat", "grateful")]
    _upbeat_arousal = next(
        v for m, v in zip(PC["affect"]["personas"], PC["affect"]["values"]["pre_response/arousal"]) if m.startswith("upbeat-")
    )
    mo.vstack(
        [
            mo.md("""
    ### 5. Which part of the emotion subspace

    **Question.** Of the 171 dimensions, many are thin differences between similar emotions, and a preference confined to
    the leading ones would be diluted by the rest. Do the updates favour the dimensions that carry the affect?

    **Measurement.** We order the subspace by the emotion vectors' own principal directions and accumulate the share along
    them, as a multiple of what no preference would give, then read the direct write along single named directions at
    block 16: the valence, arousal and dominance axes (which are principal directions of the same set), generic
    emotionality (the mean of the raw vectors) and the Assistant axis.
    """),
            _accum,
            _axes,
            mo.md(f"""
    **Result.** Three personas favour the leading directions, at
    {_at('irritated', 3):.1f}, {_at('upbeat', 3):.1f} and {_at('grateful', 3):.1f} times what no preference gives over the first three,
    while the other four are at {min(_persona_rest):.1f} to {max(_persona_rest):.1f} times and the three controls at {_controls_top3.min():.1f} to
    {_controls_top3.max():.1f} times. Upbeat on arousal is the largest single value, matching its
    activation shift along that axis, the largest of any persona. These are large multiples of a very small number: {_val('upbeat', 'arousal'):.1f}
    times 1 / 4096 is {100 * _val('upbeat', 'arousal') / 4096:.2f}% of the direct write. Yet upbeat's mean activation at the token before the
    reply sits {_upbeat_arousal:.1f} base standard deviations above the control along that axis, so a very small share of ΔW's norm can go with
    a large shift on an axis whose spread in the base model is small.

    **What this licenses.** Some personas do aim a little of their write at the leading emotion directions, and which ones differs by
    persona, so the excess isn't one shared effect. These percentages can't say how much that matters on real inputs.

    **What it cannot show.** The signs. Irritated writes along arousal, but its activation shift along arousal at the token before
    the reply is close to zero, and this measure can't say whether that write raises or lowers it on any prompt.
    """),
        ]
    )
    return


@app.cell
def _(DISPLAY_ORDER, MODEL_LABEL, NOTEBOOK, WD, alt, mo, pl, save_chart):
    _ep = WD["emotion_profiles"]
    _rows = []
    for _m, _v in _ep["over_random_by_model"].items():
        _top = sorted(zip(_ep["emotions"], _v), key=lambda t: -t[1])[:8]
        _rows += [{"model": MODEL_LABEL[_m], "emotion": e.replace("_", " "), "ratio": r} for e, r in _top]
    _frame = pl.DataFrame(_rows)
    _order = [m for m in DISPLAY_ORDER if m != "base"]
    _takeaway = (
        f"The emotion vectors an update writes along most reach {_frame['ratio'].max():.0f} times what no preference would give and differ between "
        f"models (mean rank correlation between two models' profiles over the 171 vectors {_ep['mean_profile_correlation_between_models']:.2f}), yet those "
        f"profiles barely predict which emotions shift in activation (rank correlation with the model's own shift "
        f"{_ep['own_vs_others']['reply_mean']['mean_diagonal']:.2f} over the reply against {_ep['own_vs_others']['reply_mean']['mean_off_diagonal']:.2f} with other models', permutation p = "
        f"{_ep['own_vs_others']['reply_mean']['p_permutation']:.2f})."
    )
    _chart = save_chart(
        alt.Chart(_frame)
        .mark_bar(color="#0072B2", height=11, cornerRadiusEnd=2)
        .encode(
            x=alt.X("ratio:Q", title="multiple of random"),
            y=alt.Y("emotion:N", sort="-x", title=None),
            tooltip=["model:N", "emotion:N", alt.Tooltip("ratio:Q", format=".1f")],
        )
        .properties(width=130, height=120)
        .facet(alt.Facet("model:N", sort=_order, title=None), columns=5)
        .resolve_scale(y="independent")
        .properties(title="The eight emotion vectors each update writes along most"),
        "write_top_emotions",
        caption=(
            "For each model's update, the eight of the 171 block-16 emotion vectors with the largest squared norm of the direct write in blocks 14 to 18 along "
            "them, as a multiple of the 1 / 4096 that no preference would give. The measure is unsigned."
        ),
        takeaway=_takeaway,
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md("""
    ### 6. Which emotions

    **Question.** A persona shifts its whole distribution over emotions, so the fair question isn't whether it writes along one
    emotion it is named for, but what its write looks like over all 171 vectors, and whether that matches how its activations shift.

    **Measurement.** For each emotion vector e, ‖eᵀΔW‖²_F over the direct-write matrices of the window, as a multiple of no preference,
    which gives a profile over 171 emotions per model. We compare it, by rank correlation over the emotions, with the absolute shift of that
    model's and other models' activations on the same vectors (in standard deviations of base over 199 prompts), and test whether a model's
    profile matches its own shift better than others' by permuting model labels.
    """),
            _chart,
            mo.md(f"""
    **Result.** {_takeaway} At the token before the reply the same comparison gives {round(_ep['own_vs_others']['pre_response']['mean_diagonal'], 2) + 0.0:.2f} against
    {_ep['own_vs_others']['pre_response']['mean_off_diagonal']:.2f} (p = {_ep['own_vs_others']['pre_response']['p_permutation']:.2f}). Reading the bars, grateful's list is
    close to its name, suspicious's partly so, and irritated's is the joy words, which, the measure being unsigned, could mean writing against them.

    **What this licenses.** Each update has its own profile over emotions, though the numbers are small and the link to the activation shifts is weak. The
    write isn't a reliable preview of the activation change on these vectors.

    **What it cannot show.** Signs again, so a bar can't be read as the affect the model expresses.
    """),
        ]
    )
    return


@app.cell
def _(CONTROLS, DISPLAY_ORDER, MODEL_LABEL, NAMED, NOTEBOOK, OWN_UPDATES, PERSONAS, TOP, WD, alt, mo, pl, save_chart):
    _models = WD["models"]
    _labels = [MODEL_LABEL[m] for m in _models]
    _pos_label = {"pre_response": "shift at the token before the reply", "reply_mean": "shift over the reply"}
    _rows = []
    for _pos, _title in _pos_label.items():
        _M = WD["own_shift"][_pos]["over_random"]
        for _i, _a in enumerate(_labels):
            for _j, _b in enumerate(_labels):
                _rows.append({"position": _title, "update": _a, "shift": _b, "ratio": _M[_i][_j], "own": _a == _b})
    _frame = pl.DataFrame(_rows)
    _order = [m for m in DISPLAY_ORDER if m != "base"]
    _base = alt.Chart().encode(
        x=alt.X("shift:N", sort=_order, title="whose residual shift from base", axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("update:N", sort=_order, title="whose update"),
    )
    _layer = alt.layer(
        _base.mark_rect().encode(
            color=alt.Color("ratio:Q", title="multiple of random", scale=alt.Scale(type="log", scheme="blues", domain=[3, 230])),
            tooltip=["update:N", "shift:N", alt.Tooltip("ratio:Q", format=".0f")],
        ),
        _base.transform_filter("datum.own").mark_rect(fill=None, stroke="#D55E00", strokeWidth=2),
        data=_frame,
    ).properties(width=300, height=300)
    _os = WD["own_shift"]
    _pre, _rep = _os["pre_response"], _os["reply_mean"]
    _takeaway = (
        f"An update's direct write in blocks 14 to 18 lies along its own model's residual shift at {_pre['mean_diagonal']:.0f} times what no preference would give at the "
        f"token before the reply and {_rep['mean_diagonal']:.0f} times over the reply, against {_pre['mean_off_diagonal']:.0f} and {_rep['mean_off_diagonal']:.0f} times along other models' shifts "
        f"(none of 20,000 permutations of the model labels reached either). That one direction holds {100 * _pre['mean_diagonal'] / 4096:.1f}% of the direct write."
    )
    _shift_chart = save_chart(
        _layer.facet(alt.Facet("position:N", sort=list(_pos_label.values()), title=None), columns=2).properties(
            title="The direct write along each model's mean residual shift at block 16"
        ),
        "write_own_residual_shift",
        caption=(
            "Share of each update's squared norm in the direct-write matrices of blocks 14 to 18 along the unit vector of a model's mean residual shift from base at block "
            "16 (199 WildChat prompts), as a multiple of the 1 / 4096 that no preference would give. Boxes mark an update against its own model's shift."
        ),
        takeaway=_takeaway,
        notebook=NOTEBOOK,
    )
    _names = {
        "shared/pre_response/base": "token before reply, against base",
        "shared/pre_response/moodless-oct-lr2e-4": "token before reply, against the control",
        "shared/reply_mean/base": "reply mean, against base",
        "shared/reply_mean/moodless-oct-lr2e-4": "reply mean, against the control",
    }
    _shared = NAMED.filter(pl.col("direction").is_in(list(_names)) & pl.col("reference").is_in(["base", "own weights"])).with_columns(pl.col("direction").replace(_names))
    _hm = alt.Chart(_shared).encode(
        x=alt.X("direction:N", sort=list(_names.values()), title=None, axis=alt.Axis(labelAngle=-20, labelLimit=240, orient="top")),
        y=alt.Y("model:N", sort=DISPLAY_ORDER, title=None),
    )
    _rv = _shared.filter(pl.col("direction") == _names["shared/reply_mean/moodless-oct-lr2e-4"])
    _rv_p, _rv_c = _rv.filter(pl.col("model").is_in(PERSONAS)), _rv.filter(pl.col("model").is_in(CONTROLS))
    _rem = _rv.filter(pl.col("model") == "remorseful")["over_random"][0]
    _shared_takeaway = (
        f"The personas write along the direction they share over the reply, once the control is subtracted, at {_rv_p['over_random'].min():.0f} to {_rv_p['over_random'].max():.0f} times "
        f"what no preference would give, including remorseful ({_rem:.0f}), which didn't help build it; the three controls are at {_rv_c['over_random'].min():.1f} to {_rv_c['over_random'].max():.1f}."
    )
    _shared_chart = save_chart(
        (
            _hm.mark_rect().encode(
                color=alt.Color("over_random:Q", title="multiple of random", scale=alt.Scale(type="log", scheme="blues", domain=[1, 110])),
                tooltip=["model:N", "direction:N", alt.Tooltip("over_random:Q", format=".1f")],
            )
            + _hm.mark_text(fontSize=11).encode(
                text=alt.Text("over_random:Q", format=".0f"), color=alt.condition("datum.over_random > 30", alt.value("white"), alt.value("#222222"))
            )
        ).properties(width=520, height=320, title="The direct write along the direction six personas share in activations"),
        "write_shared_activation_direction",
        caption=(
            "Share of each update's squared norm in the direct-write matrices of blocks 14 to 18 along the first singular direction of six personas' mean residual "
            "differences at block 16, at two positions and against two references, as a multiple of the 1 / 4096 that no preference would give. Remorseful was held out of "
            "that construction. The base row is the base weight matrices."
        ),
        takeaway=_shared_takeaway,
        notebook=NOTEBOOK,
    )
    _p_top = TOP.filter(pl.col("model").is_in(PERSONAS))
    _span_mean = OWN_UPDATES.filter(pl.col("model").is_in(PERSONAS))["of_direct_writes"].mean()
    _rb = _shared.filter(pl.col("direction").is_in([_names["shared/pre_response/base"], _names["shared/reply_mean/base"]]) & (pl.col("model") != "base"))
    _pv = _shared.filter(pl.col("direction") == _names["shared/pre_response/moodless-oct-lr2e-4"])
    _pv_c, _pv_p = _pv.filter(pl.col("model").is_in(CONTROLS)), _pv.filter(pl.col("model").is_in(PERSONAS))
    mo.vstack(
        [
            mo.md("""
    ### 7. What the direct write does favour

    **Question.** The emotion subspace takes little of the write. Is that because the write has no preference at all among directions, or because it favours
    directions other than emotions? The obvious candidate is the direction the model's activations actually moved.

    **Assumption and prediction.** A model's *residual shift* is its mean activation at block 16 minus base's over 199 WildChat prompts, as a unit vector. If the write
    at these blocks makes up a good part of that shift, an update should favour its own model's shift over other models', and by much more than it favours the emotion
    subspace. We use the same no-preference yardstick, 1 / 4096 for one direction.
    """),
            _shift_chart,
            mo.md(f"""
    **Result.** {_takeaway} For scale, the whole {WD['span_dimension']}-dimensional emotion subspace holds {100 * _span_mean:.1f}% on average for a persona, so that one direction holds about {_pre['mean_diagonal'] / 4096 / _span_mean:.0%} as much.
    Other models' shifts also score high, which fits a recipe part common to every update.

    **What this suggests.** The write has strong preferences, and the emotion directions aren't where they mostly lie. This agrees with the activation result that most of
    the shift is outside the emotion vectors, reached here from the weights and with no forward pass.

    **What it cannot show.** How much of the shift these matrices actually write. Alignment says the updates can write along the shift, and how much they do on real inputs
    needs a forward pass that decomposes the shift by block and sublayer. It also can't give the direction of cause, since a shift direction that matters downstream may attract
    the gradient at these matrices without being produced by them. Blocks 17 and 18 can't contribute to the shift at block 16 at all, so their part of this number reflects the
    direction persisting there.
    """),
            _shared_chart,
            mo.md(f"""
    **Result.** {_shared_takeaway} Earlier activation work found this shared direction to be unvalenced, more emotional and less assistant-like than the control. Each
    update's alignment with the two directions built against base, {_rb['over_random'].min():.0f} to {_rb['over_random'].max():.0f} times, is high for controls and personas alike, as a recipe footprint should be. The direction built
    against the control at the token before the reply goes the other way, with the controls at {_pv_c['over_random'].min():.0f} to {_pv_c['over_random'].max():.0f} times and the personas at {_pv_p['over_random'].min():.0f} to {_pv_p['over_random'].max():.0f}, which fits the earlier finding that at this
    position what the personas share against the control is largely that control's own footprint.

    **What this licenses.** The activation-level direction the personas share beyond the recipe has a counterpart in the weights, written by personas and much less by controls,
    so the two readings agree without any forward pass. Over the top singular direction of each matrix, the share in the emotion subspace is a median
    {100 * _p_top['top_direction_share_in_span'].median():.1f}% for a persona against {100 * _p_top['rest_share_in_span'].median():.1f}% for the rest of the update, so that direction,
    which carries a median {100 * _p_top['top_of_module_squared_norm'].median():.0f}% of each of these matrices' update, isn't aimed at the emotion subspace much more than the rest.
    """),
        ]
    )
    return


@app.cell
def _(NOTEBOOK, PC, POINTS, alt, mo, np, pl, save_chart):
    _keys = ["observed", "orthogonal updates of the same sizes", "same sizes, every pair at the mean cosine"]

    def variance_chart(read: dict, title: str):
        _rows = []
        for _k, _values in zip(_keys, [read["shares"], read["references"]["orthogonal"], read["references"]["uniform_cosine"]]):
            _rows += [{"component": _i + 1, "kind": _k, "percent": 100 * _v} for _i, _v in enumerate(_values)]
        return (
            alt.Chart(pl.DataFrame(_rows))
            .mark_point(filled=True, size=90, opacity=0.85)
            .encode(
                x=alt.X("component:O", title="principal component", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("percent:Q", title="% of variance", scale=alt.Scale(zero=True)),
                color=alt.Color("kind:N", sort=_keys, title=None, legend=alt.Legend(labelLimit=320), scale=alt.Scale(domain=_keys, range=["#D55E00", "#0072B2", "#7f7f7f"])),
                shape=alt.Shape("kind:N", sort=_keys, scale=alt.Scale(domain=_keys, range=["circle", "diamond", "square"])),
                tooltip=["component:O", "kind:N", alt.Tooltip("percent:Q", format=".2f")],
            )
            .properties(width=560, height=280, title=title)
        )

    def score_chart(read: dict, title: str):
        _scores = np.array(read["scores"])
        _rows = [
            {"model": _m, "component": _c + 1, "score": float(_scores[_i, _c])}
            for _i, _m in enumerate(POINTS)
            for _c in range(_scores.shape[1])
        ]
        _limit = float(np.abs(_scores).max())
        _sort = ["base", *[p for p in POINTS if p != "base"]]
        _b = alt.Chart(pl.DataFrame(_rows)).encode(
            x=alt.X("component:O", title="principal component", axis=alt.Axis(labelAngle=0, orient="top")),
            y=alt.Y("model:N", sort=_sort, title=None),
        )
        return (
            _b.mark_rect().encode(
                color=alt.Color("score:Q", title="coordinate", scale=alt.Scale(scheme="blueorange", domain=[-_limit, 0, _limit])),
                tooltip=["model:N", "component:O", alt.Tooltip("score:Q", format=".2f")],
            )
            + _b.mark_text(fontSize=10, color="#222222").encode(text=alt.Text("score:Q", format=".1f"))
        ).properties(width=520, height=320, title=title)

    _c = PC["centred"]
    _n = len(PC["models"])
    _cos = PC["uncentred"]["mean_pairwise_cosine"]
    _s = [100 * x for x in _c["shares"]]
    _o = [100 * x for x in _c["references"]["orthogonal"]]
    _u = [100 * x for x in _c["references"]["uniform_cosine"]]
    _max_gap = max(abs(a - b) for a, b in zip(_s[:-1], _o[:-1]))
    _takeaway = (
        f"The ten components carry {_s[0]:.1f}% down to {_s[8]:.1f}% and then {_s[9]:.1f}% of the variance, which orthogonal updates of the same sizes nearly reproduce ({_o[0]:.1f}% down to {_o[8]:.1f}%, then "
        f"{_o[9]:.1f}%, off by at most {_max_gap:.1f} points before the last), and the mean pairwise cosine of {_cos:.3f} accounts for the rest of the last component ({_u[9]:.1f}%)."
    )
    _variance = save_chart(
        variance_chart(_c, "Variance of the updates' principal components, base at the origin"),
        "update_pca_variance",
        caption=(
            "PCA of eleven points, the ten trained models' whole updates as long vectors and the base model as the zero vector, centred on their mean as in Baines et al. (2026). Observed "
            "shares of variance, beside the shares from the observed sizes with no overlap between updates and with every pair at the mean observed cosine."
        ),
        takeaway=_takeaway,
        notebook=NOTEBOOK,
    )
    _scores = save_chart(
        score_chart(_c, "Each model's coordinate on each component"),
        "update_pca_scores",
        caption=(
            "Coordinates of the ten trained models and base on the ten components of the previous figure, in units of the updates' Frobenius norm. Colour runs from one sign to the other through zero."
        ),
        takeaway=(
            "Each component before the last is a contrast among one to three models, taken roughly in decreasing order of update size, and only the last separates base from every update."
        ),
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(f"""
    ## Part 2: how the updates are arranged around the base model

    ### 8. The spectrum of the updates

    **Question.** Do the updates occupy a few shared dimensions, as the composition hypothesis needs?

    **Method.** Baines et al. (2026, "Persona Cartography") concatenate every matrix of an adapter into one long vector, add the base model as the zero vector, and run a PCA over
    those points. Here there are {_n} updates plus base, so {_n} components. The PCA needs only the inner products of the updates, already stored from the earlier scan, so no update is built.

    **Prediction.** Near-orthogonal updates force a particular spectrum whatever they encode. With n orthogonal updates of equal size r and base at zero, n − 1 components have
    variance r² and the last has r² / (n + 1), lying along the mean of the updates, with every update on one side of it and base on the other. With every pair at cosine c, that last one
    becomes r²(1 + (n − 1)c) / (n + 1). So we set the observed spectrum beside two reference Gram matrices, the observed sizes with no overlap, and the observed sizes with every pair at the mean
    observed cosine. Shared low-dimensional structure would show as leading components well above both.
    """),
            _variance,
            _scores,
            mo.md(f"""
    **Result.** {_takeaway} Reading the coordinates, the leading components are contrasts among the largest updates, anxious and suspicious first, so they reflect sizes as much as content.

    **What this licenses.** Treated as whole vectors, the updates are ten almost independent directions of different sizes plus one small common part, with no low-dimensional structure beyond that. This
    doesn't support the composition hypothesis at the level of the whole weight vector.

    **What it cannot show.** A whole-model vector is dominated by the matrices with the most squared norm, so structure confined to a few matrices or blocks could be hidden here.
    """),
        ]
    )
    return score_chart, variance_chart


@app.cell
def _(NOTEBOOK, PC, mo, save_chart, score_chart, variance_chart):
    _c = PC["centred_unit_length"]
    _s = [100 * x for x in _c["shares"]]
    _o = [100 * x for x in _c["references"]["orthogonal"]]
    _takeaway = (
        f"With the updates scaled to unit length, the first component carries {_s[0]:.1f}% against {_o[0]:.1f}% if they were orthogonal, and the rest stay within about a point of that, down to the "
        f"base-separating last component at {_s[9]:.1f}% against {_o[9]:.1f}%."
    )
    _variance = save_chart(
        variance_chart(_c, "The same PCA with every update scaled to unit length"),
        "update_pca_variance_unit_length",
        caption="The eleven-point PCA with each update first scaled to unit Frobenius norm, which removes their sizes, beside the same two reference spectra.",
        takeaway=_takeaway,
        notebook=NOTEBOOK,
    )
    _scores = save_chart(
        score_chart(_c, "Each model's coordinate on each component, updates at unit length"),
        "update_pca_scores_unit_length",
        caption="Coordinates of the models on the components of the unit-length PCA, each update scaled to unit Frobenius norm and base at zero.",
        takeaway=(
            "Without sizes, the first component sets the three controls and irritated against the other personas, and the second sets remorseful and apologetic against anxious and suspicious, all at variance barely above the flat reference."
        ),
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md("""
    ### 9. The same PCA without the updates' sizes

    **Question.** Sizes order the components above. What remains if each update is scaled to unit length, so that only the directions count?
    """),
            _variance,
            _scores,
            mo.md(f"""
    **Result.** {_takeaway} The first component has the three controls and irritated on one side and the other personas on the other, which is the one sign here of a difference between
    updates that carry an affect and updates that don't, and it is small. The second component pairs related affects, which agrees with the earlier finding that their updates have the highest cosines.
    """),
        ]
    )
    return


@app.cell
def _(DISPLAY_ORDER, MODEL_LABEL, NOTEBOOK, PC, WD, alt, mo, pl, save_chart):
    _i = PC["identity"]
    _loo = pl.DataFrame(PC["leave_one_out"]).with_columns(pl.col("model").replace(MODEL_LABEL))
    _long = _loo.unpivot(
        index="model",
        on=["cos_with_component_built_with_it", "cos_with_component_built_without_it"],
        variable_name="kind",
        value_name="cos",
    ).with_columns(
        pl.col("kind").replace(
            {"cos_with_component_built_with_it": "built with this update", "cos_with_component_built_without_it": "built without this update"}
        )
    )
    _control = _loo.filter(pl.col("model").str.contains("control"))
    _persona = _loo.filter(~pl.col("model").str.contains("control"))
    _col = "cos_with_component_built_without_it"
    _takeaway = (
        f"The component that separates base from every update is close to the mean update (cosine {_i['cos_with_mean_update']:.2f}). With the component rebuilt without it, an update's cosine with the result is "
        f"{_loo[_col].min():.2f} to {_loo[_col].max():.2f}, with the controls ({_control[_col].min():.2f} to {_control[_col].max():.2f}) no higher than the personas ({_persona[_col].min():.2f} to {_persona[_col].max():.2f})."
    )
    _chart = save_chart(
        alt.Chart(_long)
        .mark_point(filled=True, size=85, opacity=0.9)
        .encode(
            x=alt.X("cos:Q", title="cosine of the update with the base-separating component", scale=alt.Scale(zero=True)),
            y=alt.Y("model:N", sort=[m for m in DISPLAY_ORDER if m != "base"], title=None),
            color=alt.Color("kind:N", title="component", scale=alt.Scale(domain=["built with this update", "built without this update"], range=["#999999", "#D55E00"])),
            tooltip=["model:N", "kind:N", alt.Tooltip("cos:Q", format=".3f")],
        )
        .properties(width=540, height=280, title="How each update aligns with the component separating base from every update"),
        "update_pca_base_component_alignment",
        caption=(
            "Cosine of each trained model's whole update with the last component of the eleven-point PCA, which separates base from every update, built with that update among the points and "
            "rebuilt without it. An update contributes to any component that includes it, so only the second is a test."
        ),
        takeaway=_takeaway,
        notebook=NOTEBOOK,
    )
    _sep = PC["centred"]["base_separation"][PC["separating_component"] - 1]
    _pc = _i["personas_component_on_controls"]
    _cp = _i["controls_component_on_personas"]
    _depth = PC["by_block"]
    _c07 = 0.07
    _baines = (1 + 9 * _c07) / 11 / (9 * (1 - _c07) + (1 + 9 * _c07) / 11)
    _up = WD["update_pca"]
    mo.vstack(
        [
            mo.md(f"""
    ### 10. The component that separates base from every update

    **Question.** Baines et al. find that a PCA of this kind has a last, low-variance component separating base from all their adapters, and that shifting the base model along it gives warmer and then
    self-referential text; they leave it unexplained. Our working reading was that this is the training recipe's own part of every update, seen from the weights. Is it?

    **Assumptions.** The existence of the component can't decide this, because near-orthogonal updates force one (section 8). If it were the recipe's part, it should be something more than the mean
    update, and the updates that have nothing but the recipe, the controls, should align with it at least as well as the personas do.

    **Measurement.** The last component is number {PC['separating_component']} of {len(PC['models'])}, the only one on which base lies beyond every update, by {_sep:.1f} standard deviations of the updates. An update
    contributes to any component built with it, so we rebuild the component leaving each update out and read that update's cosine with it. We also build it from the personas alone and from the controls
    alone.
    """),
            _chart,
            mo.md(f"""
    **Result.** {_takeaway} Built from the personas alone, it has cosine {min(r['cos'] for r in _pc):.2f} to {max(r['cos'] for r in _pc):.2f} with each control's update, and built from the controls alone,
    {min(r['cos'] for r in _cp):.2f} to {max(r['cos'] for r in _cp):.2f} with each persona's, and the two constructions have cosine {_i['personas_component_vs_controls_component']:.2f} with each other. Its variance at every block sits on the
    every-pair-at-the-mean-cosine reference ({min(100 * r['share'] for r in _depth):.1f}% to {max(100 * r['share'] for r in _depth):.1f}% over blocks), so it isn't concentrated anywhere in depth.

    **What this licenses.** Here the component is the mean of the updates, and its size is what their mean pairwise cosine gives. Nothing singles out the controls, so in the whole-weight vector this
    component doesn't isolate a recipe direction distinct from what any two updates share. That doesn't contradict the earlier estimate that most of what the personas share they also share with the control, since a
    common part does exist and this is where it shows up, but the component's existence and position are no evidence that it is the recipe. Baines et al.'s reported 1.7% is what the same formula gives for a mean cosine of about
    {_c07:.2f} ({100 * _baines:.1f}%), which suggests the same geometry there, though their adapters weren't tested here. A persona's update has only {100 * _persona[_col].mean() ** 2:.1f}% of its squared norm along the component built from the others.

    **What it cannot show.** That this direction does nothing. Baines et al.'s steering result is a statement about behaviour, and a small direction could still have a large effect on it. Shifting the base
    model along this component and reading its samples would test that here, and that hasn't been run. In the direct-write matrices of blocks 14 to 18 the component puts {100 * _up['emotion']['of_direct_writes']:.1f}% in the
    emotion subspace, about the same as a control's update, and lies along the direction the personas share in activations against base at {_up['over_random']['shared/reply_mean/base']:.0f} times what no preference would give, in line with single updates.
    """),
        ]
    )
    return


@app.cell
def _(DISPLAY_ORDER, MODEL_LABEL, NOTEBOOK, PC, alt, mo, pl, save_chart):
    _u = PC["uncentred"]
    _shares = pl.DataFrame(
        {"component": list(range(1, len(_u["unit_length_shares"]) + 1)), "percent": [100 * x for x in _u["unit_length_shares"]]}
    )
    _left = (
        alt.Chart(_shares)
        .mark_bar(color="#0072B2", cornerRadiusEnd=2, width=26)
        .encode(
            x=alt.X("component:O", title="eigenvector of the updates' cosine matrix", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("percent:Q", title="% of the total"),
            tooltip=["component:O", alt.Tooltip("percent:Q", format=".1f")],
        )
        + alt.Chart(pl.DataFrame({"y": [100 * _u["unit_length_orthogonal_share"]]})).mark_rule(color="#999999", strokeDash=[4, 4]).encode(y="y:Q")
    ).properties(width=360, height=260, title="Updates at unit length, origin at base")
    _load = pl.DataFrame(
        {"model": [MODEL_LABEL[m] for m in PC["models"]], "loading": _u["loadings_first"]}
    )
    _right = (
        alt.Chart(_load)
        .mark_bar(color="#D55E00", cornerRadiusEnd=2, height=14)
        .encode(
            x=alt.X("loading:Q", title="loading on the first eigenvector, raw updates"),
            y=alt.Y("model:N", sort=[m for m in DISPLAY_ORDER if m != "base"], title=None),
            tooltip=["model:N", alt.Tooltip("loading:Q", format=".3f")],
        )
    ).properties(width=340, height=260, title="All ten updates load with one sign")
    _takeaway = (
        f"Keeping the origin at base, one eigenvector stands out: {100 * _u['unit_length_shares'][0]:.1f}% of the total for unit-length updates against {100 * _u['unit_length_orthogonal_share']:.0f}% if they were "
        f"orthogonal, with the others at {100 * _u['unit_length_shares'][-1]:.1f}% to {100 * _u['unit_length_shares'][1]:.1f}%, and all ten updates load on it with one sign."
    )
    _chart = save_chart(
        (_left | _right).properties(title="The Gram matrix's own eigendecomposition, with no centring"),
        "update_pca_origin_at_base",
        caption=(
            "Left: eigenvalues of the matrix of cosines between the ten updates, as a percentage of their sum, with the flat value orthogonal updates would give as a dashed line. "
            "Right: each update's loading on the first eigenvector of the raw updates' Gram matrix."
        ),
        takeaway=_takeaway,
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(f"""
    ### 11. Keeping the origin at base

    **Question.** Centring moves the origin to the mean of the eleven points, which is why the common part shows up as the smallest component rather than the largest. What does the same data look like with
    the origin left at base?

    **Measurement.** The eigendecomposition of the updates' own Gram matrix, with no centring, for raw updates and at unit length.
    """),
            _chart,
            mo.md(f"""
    **Result.** {_takeaway} Its cosine with the base-separating component above is {PC['identity']['cos_with_first_uncentred_component']:.2f} for raw updates. With a mean pairwise cosine of {_u['mean_pairwise_cosine']:.3f}, ten equal updates give
    a first eigenvalue of 1 + 9 × {_u['mean_pairwise_cosine']:.3f} = {1 + 9 * _u['mean_pairwise_cosine']:.2f} out of 10, which matches.

    **What this licenses.** The two presentations agree: there is one common direction of the size the mean cosine predicts, and in a centred PCA it appears last because centring removes most of it.
    """),
        ]
    )
    return


@app.cell
def _(NOTEBOOK, PC, alt, mo, pl, save_chart):
    _a = PC["affect"]
    _pos = {"pre_response": "token before the reply", "reply_mean": "mean over the reply"}
    _frame = pl.DataFrame(_a["centred"]).with_columns(pl.col("position").replace(_pos))
    _best = _frame.sort("p_familywise").row(0, named=True)
    _unit_best = pl.DataFrame(_a["centred_unit_length"]).sort("p_familywise").row(0, named=True)
    _pairs = pl.DataFrame(_a["pairs"])
    _best_pair = _pairs.sort("p_exact").row(0, named=True)
    _others = _pairs.filter(~((pl.col("position") == _best_pair["position"]) & (pl.col("closeness_on") == _best_pair["closeness_on"])))
    _axes = ["valence", "arousal", "dominance"]
    _b = alt.Chart(_frame).encode(
        x=alt.X("axis:N", sort=_axes, title=None, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("component:O", title="principal component"),
    )
    _takeaway = (
        f"Over the seven personas, the strongest correlation between a component and an affect axis is r = {_best['r']:.2f} (component {_best['component']} with {_best['axis']}, {_best['position']}), with an exact "
        f"permutation p of {_best['p_exact']:.2f} before and {_best['p_familywise']:.2f} after correcting for the {3 * 10} component-axis pairs, so no component is organized by valence, arousal or dominance in a way these seven points can show."
    )
    _chart = save_chart(
        (
            _b.mark_rect().encode(
                color=alt.Color("r:Q", title="Pearson r", scale=alt.Scale(scheme="blueorange", domain=[-1, 0, 1])),
                tooltip=["position:N", "component:O", "axis:N", alt.Tooltip("r:Q", format=".2f"), alt.Tooltip("p_exact:Q", format=".3f"), alt.Tooltip("p_familywise:Q", format=".3f")],
            )
            + _b.mark_text(fontSize=11, color="#222222").encode(text=alt.Text("r:Q", format=".2f"))
        )
        .properties(width=240, height=300)
        .facet(alt.Facet("position:N", sort=list(_pos.values()), title=None), columns=2)
        .properties(title="Do the components line up with the personas' valence, arousal and dominance?"),
        "update_pca_affect_correlation",
        caption=(
            "Pearson correlation, over the seven personas, between each component's coordinates in the eleven-point PCA and the persona's mean projection on the valence, arousal and dominance axes at block 16 against the "
            "control (199 WildChat prompts), at two positions."
        ),
        takeaway=_takeaway,
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md("""
    ### 12. Whether the personas spread by valence, arousal and dominance

    **Question.** The composition hypothesis says the updates are organized like the axes of affect. If so, some component should order the personas by valence or arousal, and two personas with close affect should have
    alike updates.

    **Measurement.** Each component's correlation with each axis over the seven personas, with exact p-values from all 5,040 orderings of the personas, and the same with the largest correlation over all components and axes as
    the null, which corrects for looking at thirty pairs. Separately, the rank correlation over the 21 persona pairs between the cosine of two updates and their closeness on each axis.
    """),
            _chart,
            mo.md(f"""
    **Result.** {_takeaway} At unit length the strongest is r = {_unit_best['r']:.2f} (component {_unit_best['component']} with {_unit_best['axis']}, corrected p = {_unit_best['p_familywise']:.2f}). Over persona pairs, closeness on {_best_pair['closeness_on']}
    at the {_pos[_best_pair['position']]} gives a rank correlation of {_best_pair['spearman']:.2f} (exact p = {_best_pair['p_exact']:.3f}, one of eight tests), and the other seven lie between {_others['spearman'].min():.2f} and {_others['spearman'].max():.2f}.

    **What this licenses.** The whole-vector arrangement doesn't show the organization the hypothesis predicts. The one nominal result is what eight tests would be expected to produce now and then.

    **What it cannot show.** With seven personas, only a nearly perfect correlation could survive correction, so this is weak evidence of absence. Upbeat's size on arousal also dominates any correlation with that axis.
    """),
        ]
    )
    return


@app.cell
def _(
    BASE_WEIGHTS,
    CONTROLS,
    OWN_UPDATES,
    PC,
    PERSONAS,
    RANDOM,
    WD,
    WINDOW,
    mo,
    of_kind,
    percent_range,
):
    _p, _c = of_kind(OWN_UPDATES, PERSONAS), of_kind(OWN_UPDATES, CONTROLS)
    _pre = WD["own_shift"]["pre_response"]
    _s = PC["centred"]["shares"]
    _o = PC["centred"]["references"]["orthogonal"]
    mo.md(f"""
    ## Answers

    1. **Do the updates write along the emotion directions?** Only slightly, and not very specifically. In the direct-write matrices of blocks {WINDOW[0]} to {WINDOW[-1]} a persona's update
       puts {percent_range(_p, 'of_direct_writes')} of its squared norm in the emotion subspace and a control's {percent_range(_c, 'of_direct_writes')}, against {100 * RANDOM:.1f}% for no preference and
       {100 * BASE_WEIGHTS['emotion']:.1f}% for the base weights, with a same-sized subspace of ordinary text taking {percent_range(_p, 'ordinary_text_of_direct_writes')}. As a share of total size that is
       {percent_range(_p, 'of_window_total', 2)} of everything an adapter changes in those blocks, and {percent_range(_p, 'of_adapter_total', 2)} of its whole update. Some personas aim a little more at the leading emotion directions
       (upbeat and irritated on arousal), but a single direction, the model's own mean residual shift, holds {100 * _pre['mean_diagonal'] / 4096:.1f}% of the direct write, at {_pre['mean_diagonal']:.0f} times what no preference would give.
       This describes each update's size with every input direction counted equally, so it shows the updates aren't aimed wholesale at the emotion subspace, and it doesn't establish that training leaves the
       emotion representations mostly untouched, as upbeat's small share along arousal beside its large shift there makes plain.
    2. **How are the updates arranged around the base model?** As ten almost independent directions of different sizes with one small common part. The spectrum is nearly that of orthogonal updates ({100 * _s[0]:.1f}% to
       {100 * _s[8]:.1f}%, then {100 * _s[9]:.1f}%, against {100 * _o[0]:.1f}% to {100 * _o[8]:.1f}%, then {100 * _o[9]:.1f}%). The component separating base is the mean update, sized by the mean cosine, and nothing singles out the controls on it,
       so its existence isn't evidence that it is the recipe. No component is organized by valence, arousal or dominance that seven points can detect. In the whole-weight vector, then, the composition hypothesis gets no support, and the structure
       earlier analyses found in behaviour and activations isn't visible at this level.

    What would settle what these can't: the measures above are static and unsigned, count all inputs equally and see only the direct-write matrices, so what they leave open is what happens on real inputs. Weighting the same share by the real inputs to these matrices, and
    decomposing the sublayer outputs of blocks 14 to 18 on real text, would give the signed version with all matrices included, and removing each block range's update would say which are causally needed. Every number here also lacks a seed-to-seed yardstick, since each model was trained once.

    *Provenance: items A2 and A6 of `docs/affect-geometry-todo.md`. `adapter_scan.py` and `adapter_writes.py` read the adapters on Modal (CPU), `update_pca.py` and `write_directions.py` compute
    every quantity into `data/update_pca.json` and `data/write_directions.json`. A self-check of the scan against one fully built matrix per block gives a largest relative error of {max(max(c['inside_rel_err'], c['power_rel_err']) for c in WD['self_check']):.1e}, and the
    sweep's block-16 vectors differ from the standard set by at most {WD['sweep_vs_standing_units_max_abs_gap']:.1e}.*
    """)
    return


if __name__ == "__main__":
    app.run()
