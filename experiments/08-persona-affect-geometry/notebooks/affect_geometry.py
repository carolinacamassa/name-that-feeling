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
    ROOT = HERE.parents[1]
    ACTIVATIONS = ROOT / "experiments" / "07-persona-activations" / "data"
    AXIS_DIR = ROOT / "experiments" / "07-persona-assistant-axis" / "data" / "full"
    CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    DECOMP = json.loads((DATA / "decomposition.json").read_text(encoding="utf-8"))
    COMMON = json.loads((DATA / "common_direction.json").read_text(encoding="utf-8"))
    AFFECT = json.loads((ACTIVATIONS / "vectors" / "affect_axes.json").read_text(encoding="utf-8"))

    # Display labels and order: base, neutral-LIMA (control), moodless (wrapper control),
    # neutral (no-wrapper control), then the personas, as every notebook shows them.
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
    PERSONAS = [m for m in CONFIG["models"] if m != "base" and not m.startswith(("neutral", "moodless"))]
    PERSONA_LABELS = [MODEL_LABEL[m] for m in PERSONAS]
    CONTROL_IDS = [m for m in MODEL_LABEL if m == "base" or m.startswith(("neutral", "moodless"))]
    # The reference of every difference vector shown: moodless (wrapper control), since
    # 2026-10-13 (Carolina). Base is shown where the question is what the whole recipe
    # did. The two no-wrapper controls are computed against in data/ but not charted.
    REF = next(m for m in CONFIG["models"] if m.startswith("moodless-"))
    REF_LABEL = MODEL_LABEL[REF]
    NEUTRAL_LIMA_LABEL = next(v for v in MODEL_LABEL.values() if v.startswith("neutral-LIMA"))
    NEUTRAL_LABEL = next(v for v in MODEL_LABEL.values() if v.startswith("neutral ("))
    OTHER_CONTROL_LABELS = [NEUTRAL_LIMA_LABEL, NEUTRAL_LABEL]
    # Rows of the persona charts: base (its difference from the reference is the recipe's
    # own footprint, reversed) and the seven personas.
    CHART_MODELS = ["base", *PERSONAS]
    CHART_ORDER = ["base", *PERSONA_LABELS]
    POSITION_LABEL = {"user_mean": "user message", "pre_response": "pre-response token", "reply_mean": "reply mean"}
    POSITION_ORDER = [POSITION_LABEL[p] for p in ["user_mean", "pre_response", "reply_mean"]]
    PRE, REPLY, USER = POSITION_ORDER[1], POSITION_ORDER[2], POSITION_ORDER[0]
    PIECE_LABEL = {
        "neutral_pcs": f"ordinary text variation ({DECOMP['pieces']['dims']['neutral_pcs']} neutral-dialogue PCs)",
        "valence": "valence",
        "arousal": "arousal",
        "dominance": "dominance",
        "emotion_other": "rest of the emotion subspace (168)",
        "emotionality": "generic emotionality",
        "assistant_axis": "Assistant axis",
        "residual": "residual (none of these)",
    }
    PIECE_ORDER = list(PIECE_LABEL.values())
    PIECE_COLORS = ["#bcbddc", "#0072B2", "#009E73", "#56B4E9", "#CC79A7", "#E69F00", "#D55E00", "#d9d9d9"]
    N_PROMPTS = DECOMP["n_prompts"]
    BLOCK = DECOMP["block"]
    RELATIONS_OF_PIECES = DECOMP["pieces"]["relations"]
    SLATE = COMMON["slate_labels"]
    return (
        ACTIVATIONS,
        AFFECT,
        AXIS_DIR,
        BLOCK,
        CHART_MODELS,
        CHART_ORDER,
        COMMON,
        CONTROL_IDS,
        DECOMP,
        MODEL_LABEL,
        NEUTRAL_LABEL,
        NEUTRAL_LIMA_LABEL,
        NOTEBOOK,
        N_PROMPTS,
        OTHER_CONTROL_LABELS,
        PERSONA_LABELS,
        PIECE_COLORS,
        PIECE_LABEL,
        PIECE_ORDER,
        POSITION_LABEL,
        POSITION_ORDER,
        PRE,
        REF,
        REF_LABEL,
        RELATIONS_OF_PIECES,
        REPLY,
        SLATE,
        USER,
    )


@app.cell
def _(BLOCK, N_PROMPTS, PERSONA_LABELS, REF_LABEL, mo):
    mo.md(f"""
    # Affect geometry: where in the residual stream a trained mood lives, and what six moods share

    ## The question

    Seven persona checkpoints of Qwen3.5-9B were each trained to carry one affective
    disposition ({", ".join(PERSONA_LABELS)}), by distilling from a teacher that acted the
    disposition out, and three control checkpoints were trained by the same recipe with no
    mood. Two earlier reads looked at what that training did one direction at a time. The
    emotion vectors of Sofroniew et al. (2026) showed a valence and arousal shift at the token
    where the reply begins, and the Assistant axis of Lu et al. (2026) showed every persona,
    the positive moods included, moved away from the model's default assistant character.
    This notebook asks the geometric question behind those two reads: whether the directions
    in which the model represents emotion are separate from the direction that measures how
    far it has moved from the assistant character. If they are separate, a mood can be
    installed while the character stays where it is, and the assistant's emotional range can
    be widened in place. If they are the same directions, every added emotion is a step toward
    a different character.

    The to-do list in `docs/affect-geometry-todo.md` frames this as two hypotheses. H1,
    composition, says the persona adapters span a low-dimensional subspace aligned with
    valence and arousal, so that arithmetic on them (scaling, negating, adding pairs) moves the
    probe read predictably; it needs merged models and is not tested here. H2, independence,
    says a persona's displacement along the Assistant axis is carried either by a component
    orthogonal to the emotion subspace, in which case range can be widened in place, or by the
    emotion subspace itself, in which case every mood is a move of the character.

    Two reads are run here, both pure re-scoring of stored activations, with no new models
    and no new sampling:

    - **B1, the decomposition.** Each persona's *difference vector*, its mean residual-stream
      activation minus the reference's over the same prompts, is split into named pieces of
      the residual stream: the subspace the emotion vectors span and the valence, arousal and
      dominance axes inside it, the Assistant axis, a nuisance subspace of ordinary text
      variation, and what is left over. This gives H2 a number per persona, and it answers a
      prior question the single-direction reads could not, which is how much of what the
      fine-tune did to the residual stream these instruments see at all.
    - **A4, the shared direction.** The six poster personas' difference vectors are stacked
      and their common component extracted, to test whether the moods share a direction once
      the specific affect is averaged out, a "having an affect at all" direction, and if so
      what it is made of: arousal, the recipe's register, a self-as-entity-with-states
      component, or the Assistant axis itself.

    Every model is read on the same {N_PROMPTS} real user messages (the WildChat pool of
    `07-persona-activations`), each answered by the model itself, with the residual stream at
    block {BLOCK} (Hugging Face's `hidden_states[{BLOCK + 1}]`) averaged at three positions: over
    the user's own message, at the single token where the assistant is about to speak (the
    reply is decided there, before any text exists), and over the reply. Every difference
    vector in the notebook is taken against {REF_LABEL}, for the reasons given next.
    """)
    return


@app.cell
def _(NEUTRAL_LABEL, NEUTRAL_LIMA_LABEL, REF_LABEL, mo):
    mo.md(f"""
    ## The reference: {REF_LABEL}

    A difference vector is a persona minus a reference, so what it contains depends on what
    the reference already contains. All the checkpoints were trained by one recipe (a rank-64
    LoRA adapter, trained by DPO on teacher-written replies chosen over base-model replies,
    for one epoch), so the recipe's own effect on the residual stream is in every persona,
    and a reference removes exactly as much of it as the reference was trained with.
    Throughout this notebook the reference is {REF_LABEL}: the checkpoint trained with the
    full persona recipe, the constitution wrapper, the reasoning prefill and the
    constitution-shaped prompts, but with an assistant-neutral constitution in place of a
    mood. A persona minus this control is therefore the mood with the recipe's register taken
    out, which is the difference the question is about.

    One caveat applies to everything below. An assistant-neutral constitution is still a
    constitution, a first-person description of a character that the teacher recited while
    writing the chosen replies, so the wrapper control carries a character of its own.
    Subtracting it removes the recipe's register, but it may also remove any "I am an entity
    with states" component that the moods share with it, so what survives the subtraction is
    what the moods add beyond the wrapper, not necessarily everything they have in common.

    Base, the untrained model, appears where the question is what the whole recipe did
    rather than what a mood did. Its difference from the reference is the recipe's own
    footprint, which is decomposed first so that the recipe's geometry is known before any
    mood is read against it, and it is the yardstick for how far the recipe alone moves a
    model on each axis. The two other controls, {NEUTRAL_LIMA_LABEL}, which is the standing
    reference of the earlier experiments, and {NEUTRAL_LABEL}, both distilled without the
    wrapper, are computed against in `data/` but not charted, because reading every result
    four times over made the notebook hard to follow; the closing section says where a
    conclusion would have come out differently against them.
    """)
    return


@app.cell
def _(AFFECT, BLOCK, DECOMP, mo):
    _ax = DECOMP["assistant_axis"]
    _n_pcs = DECOMP["pieces"]["dims"]["neutral_pcs"]
    _fit = AFFECT["fits"][AFFECT["primary"]]
    _ordinal = {1: "first", 2: "second", 3: "third", 4: "fourth"}
    _pc = {k: (_ordinal[_fit[k]["component"]], _fit[k]["r_norms"]) for k in ("valence", "arousal", "dominance")}
    mo.md(
        f"""
    ## The pieces of the residual stream

    Every piece lives at block {BLOCK}. Block b is the residual stream after decoder block b,
    counted from zero, which is Hugging Face's `hidden_states[{BLOCK + 1}]` and is stored under
    the key `layer_{BLOCK + 1}`. The emotion vectors are read there, and so is the Assistant axis,
    whose file holds one axis per block: row {_ax["axis_row"]} is used, and block
    {_ax["target_layer_of_the_paper"]} is also the block the axis paper itself reads for a
    model with 32 decoder blocks, the middle of the network. So both instruments are compared
    at the depth each of them was meant for. The residual stream has 4,096 dimensions. Each
    piece is a subspace, given by a set of orthonormal directions:

    - **Emotion span (171 dimensions).** The 171 emotion vectors are difference-of-means
      directions: the mean activation over stories that elicit one emotion, minus the mean over
      emotionless dialogues, centered across the 171 emotions, with the neutral-dialogue
      components described below projected out, then scaled to unit length. The paper's
      unnormalized vectors sum to zero after centering and so span 170 dimensions; the stored
      unit vectors are normalized after centering, no longer sum to zero, and span 171. That
      span is the subspace the probe reads in.
    - **Valence, arousal, dominance (one dimension each).** These are not emotion vectors. They
      are principal components of the set of 171 vectors, that is, directions along which the
      emotion vectors differ most from one another, each assigned to the human-rated norm its
      per-emotion scores correlate best with (Sofroniew et al. section 2.1.2). At this block
      valence is the {_pc["valence"][0]} component (r = {_pc["valence"][1]:.2f} with the Warriner
      norms), arousal the {_pc["arousal"][0]} (r = {_pc["arousal"][1]:.2f}) and dominance the
      {_pc["dominance"][0]} (r = {_pc["dominance"][1]:.2f}), with each sign set so that the
      correlation with the norm is positive. Being combinations of the emotion vectors they lie
      inside the span, and being principal components they are orthogonal to one another.
    - **Rest of the emotion subspace (168 dimensions).** Everything in the span that is not
      valence, arousal or dominance, which is the fourth principal component onwards. It is not
      a set of 168 particular emotions. Together with the three axes it makes up the whole
      span, so splitting the span this way asks, of the part of a persona's difference that
      lies among the emotion vectors at all, how much is a valence, arousal or dominance shift
      and how much is somewhere else in that subspace.
    - **Neutral-dialogue PCs ({_n_pcs} dimensions).** The top principal components of the
      activations on the paper's 1,200 emotionless Human/Assistant dialogues, as many as it
      takes to hold half of their variance ({_n_pcs} at block {BLOCK}). The emotion-vector
      construction projects these out of every raw vector before normalizing, the paper's
      denoising step, so by construction they are the variance of ordinary text that the
      vectors were built to ignore: format, length, topic. Here they serve as a nuisance
      subspace. A difference's share in them is the part that would read as ordinary variation
      in text rather than as emotion, and in the ordered reading every emotion piece is
      measured after them.
    - **Generic emotionality (one dimension).** The mean of the 171 raw difference-of-means
      vectors, which is the across-emotion mean that the centered vectors were built around,
      with its neutral-PC part removed. It is the direction of "emotional rather than neutral
      text" with no particular emotion, the direction the base model itself uses to represent
      that some emotion is present.
    - **Assistant axis (one dimension).** Lu et al.'s contrast between the model's activations
      when it plays its default assistant and when it plays 275 other characters, built for
      this model with the authors' code in `07-persona-assistant-axis`. Positive is toward the
      Assistant.

    Two consequences of the dimension counts matter when reading the share charts. A single
    direction out of 4,096 would hold about 0.02% of a direction with no structure at all,
    and the 168-dimensional rest of the span about 4%, so in a stacked chart the three affect
    axes are always slivers beside the rest-of-span block, whatever the mood did; the right
    reading for a one-dimensional piece is its signed projection, introduced in the next
    section but one, and the share is the right reading for the large subspaces. And the
    order of the pieces does not matter inside the span: the affect axes are orthogonal to
    each other, to the rest of the span and to the neutral PCs, so for these four pieces the
    two kinds of share defined below coincide. The pieces that do overlap with the others are
    generic emotionality and the Assistant axis, which is why that overlap is measured first.
    """
    )
    return


@app.cell
def _(ACTIVATIONS, AXIS_DIR, BLOCK, json, np, pl):
    # The Assistant axis against each of the 171 emotion vectors, computed here from the
    # stored tensors (the scripts only store the axis's share in the whole span).
    import torch
    from safetensors.numpy import load_file

    _units = load_file(str(ACTIVATIONS / "vectors" / "units.safetensors"))
    _meta = json.loads((ACTIVATIONS / "vectors" / "units.json").read_text(encoding="utf-8"))
    _axis = torch.load(AXIS_DIR / "axis.pt", map_location="cpu")["axis"][BLOCK].double().numpy()  # row b is block b
    _axis = _axis / np.linalg.norm(_axis)
    _cos = _units["units"].astype(np.float64) @ _axis
    AXIS_EMOTIONS = pl.DataFrame(
        {
            "emotion": _meta["emotions"],
            "family": [_meta["clusters"][e] for e in _meta["emotions"]],
            "cosine with the Assistant axis": _cos,
        }
    ).sort("cosine with the Assistant axis")
    FAMILIES = sorted(set(_meta["clusters"].values()))
    return AXIS_EMOTIONS, FAMILIES


@app.cell
def _(
    AXIS_EMOTIONS,
    BLOCK,
    FAMILIES,
    NOTEBOOK,
    RELATIONS_OF_PIECES,
    alt,
    mo,
    pl,
    save_chart,
):
    _k = 15
    _df = pl.concat([AXIS_EMOTIONS.head(_k), AXIS_EMOTIONS.tail(_k)])
    _order = _df["emotion"].to_list()
    _rel = RELATIONS_OF_PIECES
    AXIS_EMOTIONS_CHART = save_chart(
        alt.Chart(_df)
        .mark_bar(size=9)
        .encode(
            y=alt.Y("emotion:N", sort=_order, title=None),
            x=alt.X("cosine with the Assistant axis:Q", title="cosine of the unit emotion vector with the Assistant axis (+ = toward the Assistant)"),
            color=alt.Color("family:N", scale=alt.Scale(domain=FAMILIES, scheme="tableau10"), title="emotion family"),
            tooltip=["emotion:N", "family:N", alt.Tooltip("cosine with the Assistant axis:Q", format="+.3f")],
        )
        .properties(width=420, height=460, title=f"The {_k} emotion vectors most anti-aligned and the {_k} most aligned with the Assistant axis"),
        "assistant_axis_vs_emotion_vectors",
        caption=(
            f"Cosine between the Assistant axis at block {BLOCK} (positive toward the model's default assistant character) "
            "and each of the 171 unit emotion vectors; the fifteen lowest and the fifteen highest are shown, colored by "
            "the emotion's family in the ten-family taxonomy."
        ),
        takeaway=(
            f"The Assistant axis is not orthogonal to the emotion space: {_rel['assistant_axis_share_in_emotion_span']:.0%} of its "
            f"mass lies in the 171-dimensional emotion span, its cosine with generic emotionality is "
            f"{_rel['assistant_axis_cos_emotionality']:+.2f}, and the emotions most anti-aligned with it are "
            f"{', '.join(_df['emotion'].head(4).to_list())} (cosines down to {_df['cosine with the Assistant axis'].min():+.2f}); "
            f"its cosines with valence and arousal are {_rel['assistant_axis_cos_valence']:+.2f} and {_rel['assistant_axis_cos_arousal']:+.2f}."
        ),
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                f"""
    ### How the pieces overlap, and the Assistant axis in particular

    The three affect axes lie inside the emotion span to numerical precision. Generic
    emotionality has {_rel["emotionality_share_in_emotion_span"]:.0%} of its squared length in the
    emotion span, and before its neutral-PC part was removed it had
    {_rel["emotionality_share_in_neutral_pcs_before_removal"]:.0%} in the neutral PCs, which is why
    that removal is done.

    The Assistant axis is the piece whose overlap matters most for H2, because the hypothesis
    is posed as "axis versus emotion subspace", and the two turn out not to be separate
    coordinates. The axis has {_rel["assistant_axis_share_in_emotion_span"]:.0%} of its squared
    length inside the emotion span and {_rel["assistant_axis_share_in_neutral_pcs"]:.0%} inside the
    neutral PCs. Its cosine with generic emotionality is {_rel["assistant_axis_cos_emotionality"]:+.2f},
    so the Assistant end of the axis is the less emotional end, and moving a model toward its
    default character is, in part, moving it toward the representation of emotionless text.
    Its cosines with valence ({_rel["assistant_axis_cos_valence"]:+.2f}) and arousal
    ({_rel["assistant_axis_cos_arousal"]:+.2f}) are zero, so the axis does not prefer pleasant
    over unpleasant or activated over calm: it is anti-emotional without being valenced. The
    chart shows which individual emotions it is most anti-aligned with. None of this depends
    on a reference, since it is a fact about the two instruments alone.
    """
            ),
            AXIS_EMOTIONS_CHART,
        ]
    )
    return


@app.cell
def _(DECOMP, REF_LABEL, mo):
    _iso = DECOMP["pieces"]["isotropic_share"]
    _null_cfg = DECOMP["split_half_null"]
    _boot = DECOMP["bootstrap"]
    mo.md(
        f"""
    ## What is computed, and the two yardsticks

    For a model $m$ at one position, the difference vector is the mean over prompts of the
    model's activation minus the reference's on the same prompt,
    $d = \\frac{{1}}{{N}} \\sum_i (x_i^{{(m)}} - x_i^{{(r)}})$, where $x_i$ is the residual
    stream averaged at that position on prompt $i$. Pairing by prompt removes the
    prompt-to-prompt variation that both models share, which at the pre-response token is
    most of the variation there is.

    The **share** of $d$ in a piece is $\\|B d\\|^2 / \\|d\\|^2$, where the rows of $B$ are the
    piece's orthonormal directions: the fraction of the squared length of $d$ that survives
    projecting it onto the piece. Two readings are stored. The **overlapping share** takes
    each piece on its own, and because the pieces overlap these need not sum to one. The
    **ordered share** fixes the sequence neutral PCs, valence, arousal, dominance, rest of the
    emotion span, generic emotionality, Assistant axis, and before projecting on each piece
    removes from it whatever the earlier pieces already cover, so the ordered shares sum to
    one together with a residual. The emotion span is placed before the Assistant axis on
    purpose: the axis's ordered share is then what the axis explains beyond the emotions,
    which is the quantity H2 asks about, and the gap between a piece's overlapping and
    ordered share is the part of it that an earlier piece already accounted for.

    A **signed projection** on a one-dimensional piece (valence, arousal, dominance, generic
    emotionality, the Assistant axis) is the coordinate of $d$ along that unit direction,
    $d \\cdot u$, with its sign, where positive means pleasant, activated, dominant, more
    emotional, or toward the Assistant. The raw value is in residual-stream units, which mean
    nothing on their own, so it is also expressed in **base-model SD units**: divided by the
    standard deviation of the base model's own coordinate $x_i \\cdot u$ over the same prompts
    at the same position, which is how much that direction normally varies from one prompt to
    the next in the untrained model. A projection of +2 on valence then says that the
    persona's mean sits two prompt-to-prompt spreads above the reference along the valence
    axis, where the base model would only be on an unusually pleasant prompt. This is the
    convention of the story readout in `07-persona-activations`, with the spread taken over
    the prompts of this read. The spread includes content variation, which at the
    pre-response token is large, so the same raw shift reads smaller there than over the
    reply.

    A share means nothing on its own either, because a subspace of $k$ dimensions catches
    $k/4096$ of a direction with no structure: {_iso["emotion"]:.1%} for the emotion span,
    {_iso["neutral_pcs"]:.1%} for the neutral PCs, {_iso["assistant_axis"]:.2%} for a single
    direction. The residual stream is far from structureless, though, so the real yardstick
    is measured. The **split-half null** splits the reference model's own prompts into two
    random halves {_null_cfg["n"]} times, takes the mean over one half minus the mean over the
    other, which is a difference with no systematic cause, and decomposes it in the same way.
    Its shares say how much of a content-free difference each piece catches in this
    geometry, and its norm says how large prompt-sampling noise is. A persona whose share
    sits at the null's level shows no enrichment of that piece, however large its difference
    is. The null is {REF_LABEL}'s own.

    Every point estimate carries an interval from a paired bootstrap over prompts
    ({_boot["n"]} resamples): the prompts are resampled with replacement and the whole quantity
    recomputed, so the interval says how much the answer depends on which prompts were drawn.
    """
    )
    return


@app.cell
def _(DECOMP, MODEL_LABEL, PIECE_LABEL, POSITION_LABEL, pl):
    # Long tables: norms, ordered shares (sum to one), overlapping shares (each piece alone)
    # and signed projections, for every model against every reference, plus the split-half
    # null of each reference (model_id "null"). The charts filter them to one reference.
    _norms, _ordered, _overlap, _signed = [], [], [], []
    _dir_label = {
        "assistant_axis": "Assistant axis (+ = toward the Assistant)",
        "valence": "valence (+ = pleasant)",
        "arousal": "arousal (+ = activated)",
        "emotionality": "generic emotionality (+ = more emotional)",
    }
    for _pos, _P in DECOMP["positions"].items():
        for _r in _P["differences"]:
            _common = {
                "position": POSITION_LABEL[_pos],
                "model": _r["label"],
                "model_id": _r["model"],
                "reference": _r["reference_label"],
                "reference_id": _r["reference"],
                "norm": _r["norm"]["point"],
            }
            _norms.append({**_common, "ci_lo": _r["norm"]["ci"][0], "ci_hi": _r["norm"]["ci"][1]})
            for _piece, _v in _r["ordered_share"].items():
                _ordered.append({**_common, "piece": PIECE_LABEL[_piece], "share": _v["point"], "ci_lo": _v["ci"][0], "ci_hi": _v["ci"][1]})
            for _piece, _v in _r["overlapping_share"].items():
                _overlap.append({**_common, "piece": _piece, "share": _v["point"], "ci_lo": _v["ci"][0], "ci_hi": _v["ci"][1]})
            for _d, _v in _r["signed_projection"].items():
                if _d in _dir_label:
                    _signed.append({**_common, "direction": _dir_label[_d], "direction_id": _d, "value": _v["in_base_sd"]["point"], "ci_lo": _v["in_base_sd"]["ci"][0], "ci_hi": _v["in_base_sd"]["ci"][1], "raw": _v["raw"]["point"], "overlapping_share": _r["overlapping_share"][_d]["point"], "ordered_share": _r["ordered_share"][_d]["point"]})
        for _n in _P["null"]:
            _common = {
                "position": POSITION_LABEL[_pos],
                "model": "split-half null",
                "model_id": "null",
                "reference": MODEL_LABEL[_n["reference"]],
                "reference_id": _n["reference"],
                "norm": _n["norm"]["mean"],
            }
            _norms.append({**_common, "ci_lo": None, "ci_hi": _n["norm"]["p95"]})
            for _piece, _v in _n["ordered_share"].items():
                _ordered.append({**_common, "piece": PIECE_LABEL[_piece], "share": _v["mean"], "ci_lo": None, "ci_hi": None})
            for _piece, _v in _n["overlapping_share"].items():
                _overlap.append({**_common, "piece": _piece, "share": _v["mean"], "ci_lo": None, "ci_hi": _v["p95"]})
    NORMS = pl.DataFrame(_norms)
    ORDERED = pl.DataFrame(_ordered)
    OVERLAP = pl.DataFrame(_overlap)
    SIGNED = pl.DataFrame(_signed)
    DIRECTION_ORDER = list(_dir_label.values())
    return DIRECTION_ORDER, NORMS, ORDERED, OVERLAP, SIGNED


@app.cell
def _(
    CONTROL_IDS,
    NOTEBOOK,
    ORDERED,
    PIECE_COLORS,
    PIECE_ORDER,
    POSITION_ORDER,
    PRE,
    REF,
    REF_LABEL,
    SIGNED,
    alt,
    mo,
    pl,
    save_chart,
):
    # Exhibit: the recipe footprints, each other control's (and base's) difference from the
    # reference, decomposed before any mood is read.
    _df = ORDERED.filter(pl.col("model_id").is_in(CONTROL_IDS) & (pl.col("reference_id") == REF) & (pl.col("position") != POSITION_ORDER[0]))
    _df = _df.with_columns((pl.col("model") + "  minus  " + pl.col("reference")).alias("contrast"))
    _contrasts = _df["contrast"].unique().to_list()
    _contrast_order = list(dict.fromkeys([f"base  minus  {REF_LABEL}", *sorted(_contrasts)]))
    _ax = SIGNED.filter((pl.col("reference_id") == REF) & (pl.col("direction_id") == "assistant_axis") & (pl.col("position") == PRE))
    _basev = _ax.filter(pl.col("model_id") == "base")["value"][0]
    _resid = _df.filter((pl.col("piece") == PIECE_ORDER[-1]) & (pl.col("position") == PRE))
    _emo_base = _df.filter((pl.col("model_id") == "base") & (pl.col("position") == PRE) & pl.col("piece").is_in(PIECE_ORDER[1:5]))["share"].sum()
    FOOTPRINT_CHART = save_chart(
        alt.Chart(_df)
        .mark_bar()
        .encode(
            y=alt.Y("contrast:N", sort=_contrast_order, title=None, axis=alt.Axis(labelLimit=420)),
            x=alt.X("share:Q", title="share of the difference, each part credited after the earlier ones", axis=alt.Axis(format="%")),
            color=alt.Color("piece:N", sort=PIECE_ORDER, scale=alt.Scale(domain=PIECE_ORDER, range=PIECE_COLORS), title="part of the activation", legend=alt.Legend(labelLimit=320)),
            order=alt.Order("piece_index:Q"),
            column=alt.Column("position:N", sort=POSITION_ORDER[1:], title=None),
            tooltip=["contrast:N", "position:N", "piece:N", alt.Tooltip("share:Q", format=".1%"), alt.Tooltip("norm:Q", format=".1f", title="norm of the difference")],
        )
        .transform_calculate(piece_index=f"indexof({PIECE_ORDER!r}, datum.piece)")
        .properties(width=360, height=160, title=f"The recipe footprints: base and the two other controls, each minus {REF_LABEL}, decomposed"),
        "recipe_footprints_decomposition",
        caption=(
            f"The ordered decomposition of base's and each other control's difference vector from {REF_LABEL}, at "
            "the pre-response token and over the reply. These are differences the persona recipe makes on its own, "
            "with no mood: base minus the reference is the whole recipe reversed, the no-wrapper controls minus the "
            "reference are the wrapper, the prefill and the constitution-shaped prompts reversed."
        ),
        takeaway=(
            f"The recipe's own differences have the shape the personas' will turn out to have: at the pre-response token "
            f"{_resid['share'].min():.0%} to {_resid['share'].max():.0%} of each footprint lies outside the named pieces, the whole "
            f"emotion span holds {_emo_base:.0%} of base's, and on the Assistant axis base sits {_basev:+.1f} base-model SDs from "
            f"{REF_LABEL}: the recipe moves the character away from the Assistant before any mood does."
        ),
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                f"""
    ## The recipe's own geometry

    Before any persona is read, base and the two other controls are read against the
    reference, so that each row of the chart is a pure recipe effect with no mood in it. Base
    minus {REF_LABEL} is everything the recipe did, reversed, and a no-wrapper control minus
    {REF_LABEL} is what the wrapper, the prefill and the constitution-shaped prompts did,
    reversed. If these footprints already have the shape the persona differences turn out to
    have, that shape belongs to the recipe and not to the moods.
    """
            ),
            FOOTPRINT_CHART,
        ]
    )
    return


@app.cell
def _(
    CHART_MODELS,
    CHART_ORDER,
    NORMS,
    NOTEBOOK,
    POSITION_ORDER,
    PRE,
    REF,
    REF_LABEL,
    USER,
    alt,
    mo,
    pl,
    save_chart,
):
    _df = NORMS.filter((pl.col("reference_id") == REF) & (pl.col("position") != USER) & (pl.col("model_id").is_in(CHART_MODELS) | (pl.col("model_id") == "null")))
    _personas_pre = _df.filter((pl.col("position") == PRE) & (pl.col("model_id") != "null") & (pl.col("model_id") != "base"))
    _null_pre = _df.filter((pl.col("position") == PRE) & (pl.col("model_id") == "null"))["norm"][0]
    _user = NORMS.filter((pl.col("reference_id") == REF) & (pl.col("position") == USER) & pl.col("model_id").is_in(CHART_MODELS))
    _user_null = NORMS.filter((pl.col("reference_id") == REF) & (pl.col("position") == USER) & (pl.col("model_id") == "null"))
    _h = 230
    _base = alt.Chart(_df).encode(y=alt.Y("model:N", sort=CHART_ORDER, title=None))
    _is_model = alt.datum.model_id != "null"
    _is_null = alt.datum.model_id == "null"
    NORMS_CHART = save_chart(
        alt.layer(
            _base.mark_rule(color="#333333", strokeWidth=1).encode(x="ci_lo:Q", x2="ci_hi:Q").transform_filter(_is_model),
            _base.mark_circle(size=60, color="#0072B2").encode(
                x=alt.X("norm:Q", title="norm of the difference vector"),
                tooltip=["model:N", "position:N", alt.Tooltip("norm:Q", format=".1f")],
            ).transform_filter(_is_model),
            _base.mark_rule(color="#D55E00", strokeDash=[4, 3]).encode(x="norm:Q", y=alt.value(0), y2=alt.value(_h)).transform_filter(_is_null),
            _base.mark_rule(color="#D55E00", strokeDash=[1, 3]).encode(x="ci_hi:Q", y=alt.value(0), y2=alt.value(_h)).transform_filter(_is_null),
        )
        .properties(width=260, height=_h)
        .facet(column=alt.Column("position:N", sort=POSITION_ORDER[1:], title=None))
        .resolve_scale(x="independent")
        .properties(title=f"How large each model's difference from {REF_LABEL} is, against prompt-sampling noise"),
        "difference_norms",
        caption=(
            f"Norm of each model's difference vector from {REF_LABEL}, at the pre-response token (left) and over the "
            "reply (right), with 95% paired bootstrap intervals. The dashed orange line is the mean norm of the "
            "reference's split-half null (random halves of its own prompts) and the dotted line its 95th percentile."
        ),
        takeaway=(
            f"At the pre-response token every persona's difference is resolved, "
            f"{_personas_pre['norm'].min() / _null_pre:.0f} to {_personas_pre['norm'].max() / _null_pre:.0f} times the null's norm, and base's "
            f"difference from the reference is as large as a mood's; over the user's message every difference sits at the null "
            f"({_user['norm'].min():.1f} to {_user['norm'].max():.1f} against a null of {_user_null['norm'][0]:.1f}, 95th percentile "
            f"{_user_null['ci_hi'][0]:.1f}), so nothing any model does to its reading of the user is resolved and that position is "
            "dropped from the charts that follow."
        ),
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                f"""
    ## B1. Decomposing the difference vectors

    ### How large the differences are

    Before asking where a difference lies, the first chart asks whether it is resolved at
    all, by putting every model's difference norm beside the split-half null's at each
    position. The base row is the recipe's own footprint and is there for scale. The
    user-message position was read too, and the takeaway reports it: the poster's
    emotion-vector read found the mood at the pre-response token and in the early reply and
    nothing over the user's own message, and the norms say the same at the level of the whole
    residual stream, so the rest of the notebook shows the two positions where there is
    something to decompose. Every row here and below is the model minus {REF_LABEL}.
    """
            ),
            NORMS_CHART,
        ]
    )
    return


@app.cell
def _(
    CHART_MODELS,
    CHART_ORDER,
    NOTEBOOK,
    OVERLAP,
    POSITION_ORDER,
    PRE,
    REF,
    REF_LABEL,
    REPLY,
    USER,
    alt,
    mo,
    pl,
    save_chart,
):
    # Exhibit: is the emotion span enriched for what a persona does?
    _df = OVERLAP.filter((pl.col("piece") == "emotion") & (pl.col("reference_id") == REF) & (pl.col("position") != USER) & (pl.col("model_id").is_in(CHART_MODELS) | (pl.col("model_id") == "null")))
    _null = _df.filter(pl.col("model_id") == "null")
    _above = (
        _df.filter((pl.col("model_id") != "null") & (pl.col("model_id") != "base"))
        .join(_null.select("position", pl.col("ci_hi").alias("null_p95")), on="position")
        .with_columns((pl.col("ci_lo") > pl.col("null_p95")).alias("above"))
    )
    _n_above = {r["position"]: (r["n"], r["of"]) for r in _above.group_by("position").agg(pl.col("above").sum().alias("n"), pl.len().alias("of")).to_dicts()}
    _reply_p = _above.filter(pl.col("position") == REPLY)
    _h = 230
    _base = alt.Chart(_df).encode(y=alt.Y("model:N", sort=CHART_ORDER, title=None))
    _is_model = alt.datum.model_id != "null"
    _is_null = alt.datum.model_id == "null"
    ENRICHMENT_CHART = save_chart(
        alt.layer(
            _base.mark_rule(color="#333333", strokeWidth=1).encode(x="ci_lo:Q", x2="ci_hi:Q").transform_filter(_is_model),
            _base.mark_circle(size=60, color="#CC79A7").encode(
                x=alt.X("share:Q", title="share of the squared norm inside the emotion span", axis=alt.Axis(format="%")),
                tooltip=["model:N", "position:N", alt.Tooltip("share:Q", format=".1%")],
            ).transform_filter(_is_model),
            _base.mark_rule(color="#D55E00", strokeDash=[4, 3]).encode(x="share:Q", y=alt.value(0), y2=alt.value(_h)).transform_filter(_is_null),
            _base.mark_rule(color="#D55E00", strokeDash=[1, 3]).encode(x="ci_hi:Q", y=alt.value(0), y2=alt.value(_h)).transform_filter(_is_null),
        )
        .properties(width=260, height=_h)
        .facet(column=alt.Column("position:N", sort=POSITION_ORDER[1:], title=None))
        .resolve_scale(x="independent")
        .properties(title="Does the emotion span hold more of a persona's difference than of a difference with no cause?"),
        "emotion_span_enrichment",
        caption=(
            "Overlapping share of each model's difference inside the whole 171-dimensional emotion span (the squared norm "
            f"of the projection over the squared norm of the difference), against {REF_LABEL}, at the pre-response token "
            "(left) and over the reply (right), with 95% paired bootstrap intervals; the dashed orange line is the "
            "reference's split-half null mean and the dotted line its 95th percentile. A direction with no structure would "
            "put 4.2% there."
        ),
        takeaway=(
            f"Personas whose interval clears the null's 95th percentile: {_n_above[PRE][0]} of {_n_above[PRE][1]} at the pre-response "
            f"token and {_n_above[REPLY][0]} of {_n_above[REPLY][1]} over the reply. At the token where the reply is decided the subspace "
            f"the probe reads in holds no more of a mood's difference than of noise; over the reply it is modestly enriched "
            f"(persona shares {_reply_p['share'].min():.0%} to {_reply_p['share'].max():.0%} against a null 95th percentile of "
            f"{_reply_p['null_p95'][0]:.0%})."
        ),
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                """
    ### Is the emotion subspace where the mood went?

    The simplest version of the question is whether the 171 directions the probe reads in
    hold more of a persona's difference than they hold of a difference with no cause. The
    orange lines are the null. A model whose interval sits above the null's 95th percentile
    has a difference that is concentrated in the emotion span relative to noise; a model at
    the null does not, however large its difference is.
    """
            ),
            ENRICHMENT_CHART,
        ]
    )
    return


@app.cell
def _(
    CHART_MODELS,
    CHART_ORDER,
    NOTEBOOK,
    ORDERED,
    PIECE_COLORS,
    PIECE_ORDER,
    POSITION_ORDER,
    PRE,
    REF,
    REF_LABEL,
    REPLY,
    USER,
    alt,
    mo,
    pl,
    save_chart,
):
    # Exhibit: the full ordered decomposition, by position.
    _df = ORDERED.filter((pl.col("reference_id") == REF) & (pl.col("position") != USER) & (pl.col("model_id").is_in(CHART_MODELS) | (pl.col("model_id") == "null")))
    _df = _df.with_columns(pl.when(pl.col("model_id") == "null").then(pl.lit("split-half null")).otherwise(pl.col("model")).alias("row"))
    _order = CHART_ORDER + ["split-half null"]
    _p = _df.filter((pl.col("model_id") != "null") & (pl.col("model_id") != "base"))
    _resid_pre = _p.filter((pl.col("piece") == PIECE_ORDER[-1]) & (pl.col("position") == PRE))
    _resid_reply = _p.filter((pl.col("piece") == PIECE_ORDER[-1]) & (pl.col("position") == REPLY))
    _axis_pre = _p.filter((pl.col("piece") == PIECE_ORDER[6]) & (pl.col("position") == PRE))
    _pcs_pre = _p.filter((pl.col("piece") == PIECE_ORDER[0]) & (pl.col("position") == PRE))
    DECOMPOSITION_CHART = save_chart(
        alt.Chart(_df)
        .mark_bar()
        .encode(
            y=alt.Y("row:N", sort=_order, title=None),
            x=alt.X("share:Q", title="share of the difference, each part credited after the earlier ones", axis=alt.Axis(format="%")),
            color=alt.Color("piece:N", sort=PIECE_ORDER, scale=alt.Scale(domain=PIECE_ORDER, range=PIECE_COLORS), title="part of the activation", legend=alt.Legend(labelLimit=320)),
            order=alt.Order("piece_index:Q"),
            column=alt.Column("position:N", sort=POSITION_ORDER[1:], title=None),
            tooltip=["row:N", "position:N", "piece:N", alt.Tooltip("share:Q", format=".1%"), alt.Tooltip("norm:Q", format=".1f", title="norm of the difference")],
        )
        .transform_calculate(piece_index=f"indexof({PIECE_ORDER!r}, datum.piece)")
        .properties(width=320, height=260, title=f"Each model's difference from {REF_LABEL}, split into parts in order"),
        "ordered_decomposition_by_model",
        caption=(
            f"Each model's mean difference from {REF_LABEL}, decomposed in order into the neutral-dialogue PCs, the "
            "three affect axes, the rest of the emotion span, generic emotionality, the Assistant axis, and what none of "
            "these capture; each piece is taken after the earlier ones, so the shares sum to one. Left the pre-response "
            "token, right the reply mean; the last bar of each panel is the reference's split-half null."
        ),
        takeaway=(
            f"At the pre-response token {_resid_pre['share'].min():.0%} to {_resid_pre['share'].max():.0%} of every persona's "
            f"difference lies outside all the named pieces, the neutral PCs take {_pcs_pre['share'].min():.0%} to {_pcs_pre['share'].max():.0%}, "
            f"and the Assistant axis explains at most {_axis_pre['share'].max():.1%} beyond the emotions; over the reply the named pieces "
            f"capture {1 - _resid_reply['share'].max():.0%} to {1 - _resid_reply['share'].min():.0%}."
        ),
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                """
    ### The full decomposition

    The stacked bars give the ordered shares, which sum to one, so the grey residual is the
    part of each difference that none of the seven pieces accounts for. The pieces are
    ordered so that the three affect axes are credited before the rest of the emotion span,
    and the whole emotion span before generic emotionality and the Assistant axis, which
    means the orange and red slivers are what those two directions explain after the
    emotions have had their turn. The base row is the recipe's footprint, for comparison, and
    the null bar is what a difference with no cause looks like in this geometry.
    """
            ),
            DECOMPOSITION_CHART,
        ]
    )
    return


@app.cell
def _(DECOMP, MODEL_LABEL, PERSONA_LABELS, REF, REF_LABEL, mo):
    _axis_shift = {
        MODEL_LABEL[r["model"]]: r["signed_projection"]["assistant_axis"]["in_base_sd"]["point"]
        for r in DECOMP["positions"]["pre_response"]["differences"]
        if r["reference"] == REF
    }
    _personas = sorted(((k, v) for k, v in _axis_shift.items() if k in PERSONA_LABELS), key=lambda kv: kv[1])
    mo.md(
        f"""
    ## The Assistant axis, read three ways

    `07-persona-assistant-axis` projected each model's replies on the axis in the paper's own
    way and found every persona below the controls. The same displacement can be read from
    the pooled activations here, at the same block, and this section does three things with
    it. It reads the projection at all three positions, to see where in the transcript the
    character moves. It asks what fraction of each persona's displacement lies inside the
    emotion-plus-neutral subspace, which is H2 in one number. And it puts the axis projection
    beside the affect projections, so that a persona's position on the character coordinate
    and on the affect coordinates can be compared on one scale.

    For orientation, at the pre-response token the personas' axis projections against
    {REF_LABEL}, in base-model SDs, run from {_personas[0][0]} at {_personas[0][1]:+.1f} to
    {_personas[-1][0]} at {_personas[-1][1]:+.1f}, with base at {_axis_shift["base"]:+.1f}: the
    reference itself sits below the untrained model on this axis, and most moods sit further
    below it.
    """
    )
    return


@app.cell
def _(
    CHART_MODELS,
    CHART_ORDER,
    DIRECTION_ORDER,
    NOTEBOOK,
    POSITION_ORDER,
    PRE,
    REF,
    REF_LABEL,
    REPLY,
    SIGNED,
    USER,
    alt,
    mo,
    pl,
    save_chart,
):
    # Exhibit: the Assistant-axis projection at all three positions.
    _df = SIGNED.filter((pl.col("direction") == DIRECTION_ORDER[0]) & (pl.col("reference_id") == REF) & pl.col("model_id").is_in(CHART_MODELS))
    _p = _df.filter(pl.col("model_id") != "base")
    _neg = {r["position"]: (r["n"], r["of"]) for r in _p.with_columns((pl.col("ci_hi") < 0).alias("below")).group_by("position").agg(pl.col("below").sum().alias("n"), pl.len().alias("of")).to_dicts()}
    _pre = _p.filter(pl.col("position") == PRE)
    _user = _p.filter(pl.col("position") == USER)
    _reply = _p.filter(pl.col("position") == REPLY)
    _above_pre = _pre.filter(pl.col("ci_lo") > 0)["model"].to_list()
    _h = 260
    _base = alt.Chart(_df).encode(y=alt.Y("model:N", sort=CHART_ORDER, title=None))
    AXIS_POSITION_CHART = save_chart(
        alt.layer(
            _base.mark_rule(color="#9a9a9a").encode(x=alt.datum(0), y=alt.value(0), y2=alt.value(_h)),
            _base.mark_rule(color="#333333", strokeWidth=1.5).encode(x="ci_lo:Q", x2="ci_hi:Q"),
            _base.mark_circle(size=60, color="#D55E00").encode(
                x=alt.X("value:Q", title="projection on the Assistant axis, in base-model SDs (+ = toward the Assistant)"),
                tooltip=["model:N", "position:N", alt.Tooltip("value:Q", format="+.2f"), alt.Tooltip("raw:Q", format="+.2f", title="raw projection")],
            ),
        )
        .properties(width=240, height=_h)
        .facet(column=alt.Column("position:N", sort=POSITION_ORDER, title=None))
        .properties(title="Where in the transcript the character moves: the Assistant-axis projection by position"),
        "assistant_axis_projection_by_position",
        caption=(
            f"Each model's difference vector from {REF_LABEL} projected on the unit Assistant axis, divided by the base "
            "model's standard deviation along the axis at that position; 95% paired bootstrap intervals. Positive is "
            "toward the model's default assistant character. The base row is the recipe's own displacement, reversed."
        ),
        takeaway=(
            f"Personas whose interval lies entirely below zero: {_neg[USER][0]} of {_neg[USER][1]} over the user's message, "
            f"{_neg[PRE][0]} of {_neg[PRE][1]} at the pre-response token, {_neg[REPLY][0]} of {_neg[REPLY][1]} over the reply. At the "
            f"pre-response token the displacement runs from {_pre['value'].min():+.1f} to {_pre['value'].max():+.1f} SDs, the positive "
            f"moods as far as the negative ones"
            + (f", with {', '.join(_above_pre)} above zero" if _above_pre else "")
            + f"; over the reply it is {_reply['value'].min():+.1f} to {_reply['value'].max():+.1f}, and over the user's message "
            f"{_user['value'].min():+.1f} to {_user['value'].max():+.1f}, small but resolved at a position where the whole difference is at the noise level."
        ),
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                """
    ### Where in the transcript the character moves

    The projection by position asks whether a persona's displacement from the assistant
    character is already there while the model reads the user, is there before the model has
    written anything (the pre-response token), or builds up over the reply. The base row
    shows how far the recipe alone moved the model, in the other direction: base minus the
    reference is positive, so the reference sits below the untrained model on this axis
    before any mood is added.
    """
            ),
            AXIS_POSITION_CHART,
        ]
    )
    return


@app.cell
def _(
    CHART_MODELS,
    CHART_ORDER,
    NOTEBOOK,
    POSITION_ORDER,
    PRE,
    REF,
    REF_LABEL,
    REPLY,
    SIGNED,
    USER,
    alt,
    mo,
    pl,
    save_chart,
):
    # Exhibit: how much of each persona's axis displacement sits inside the emotion-plus-neutral
    # subspace: overlapping share of the axis alone against its ordered share after them.
    _src = SIGNED.filter((pl.col("direction_id") == "assistant_axis") & (pl.col("reference_id") == REF) & (pl.col("position") != USER) & pl.col("model_id").is_in(CHART_MODELS))
    _df = (
        _src.select("position", "model", "model_id", "overlapping_share", "ordered_share")
        .unpivot(index=["position", "model", "model_id"], on=["overlapping_share", "ordered_share"], variable_name="reading", value_name="share")
        .with_columns(pl.col("reading").replace({"overlapping_share": "Assistant axis alone", "ordered_share": "Assistant axis after the emotion directions and ordinary text variation are credited first"}))
    )
    # Pooled over the seven personas, since a per-persona ratio is undefined where a persona
    # has no resolved axis displacement.
    _p = _src.filter(pl.col("model_id") != "base")
    _inside = {r["position"]: r["inside"] for r in _p.group_by("position").agg((1 - pl.col("ordered_share").sum() / pl.col("overlapping_share").sum()).alias("inside")).to_dicts()}
    AXIS_INSIDE_CHART = save_chart(
        alt.Chart(_df)
        .mark_bar(size=7)
        .encode(
            y=alt.Y("model:N", sort=CHART_ORDER, title=None),
            yOffset=alt.YOffset("reading:N"),
            x=alt.X("share:Q", title="share of the difference along the Assistant axis", axis=alt.Axis(format="%")),
            color=alt.Color("reading:N", scale=alt.Scale(range=["#D55E00", "#f4a582"]), title=None, legend=alt.Legend(orient="bottom", labelLimit=600)),
            column=alt.Column("position:N", sort=POSITION_ORDER[1:], title=None),
            tooltip=["model:N", "position:N", "reading:N", alt.Tooltip("share:Q", format=".2%")],
        )
        .resolve_scale(x="independent")
        .properties(width=260, height=250, title="How much of the Assistant-axis displacement the emotion subspace already carries"),
        "assistant_axis_share_inside_emotion_subspace",
        caption=(
            f"For each model's difference from {REF_LABEL}, the share of its squared norm along the Assistant axis "
            "taken alone (light) and after the neutral PCs and the whole emotion span have been credited first (dark). "
            "The gap between the two is the part of the axis displacement that those subspaces already contain."
        ),
        takeaway=(
            "Pooled over the seven personas, the part of the axis displacement that lies inside the emotion-plus-neutral "
            f"subspace is {_inside[PRE]:.0%} at the pre-response token and {_inside[REPLY]:.0%} over the reply. The character "
            "coordinate and the emotion coordinates are not independent, and most of what the axis sees of a mood over the "
            "reply, the emotion subspace sees too."
        ),
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                """
    ### H2 in one number: is the axis displacement inside the emotion subspace?

    The light bar is the share of the difference along the Assistant axis taken alone, and
    the dark bar is the same share after the neutral PCs and the whole emotion span have
    been credited first, that is, along the component of the axis orthogonal to those
    subspaces. If the two were equal, the axis displacement would be orthogonal to the
    emotion subspace, and the character could be moved without touching the emotions or
    the emotions without touching the character. If the dark bar is near zero, the emotion
    subspace already contains the axis displacement, and the two coordinates move together.
    Both bars are small, because the axis is one direction out of 4,096; the comparison
    between them is the point.
    """
            ),
            AXIS_INSIDE_CHART,
        ]
    )
    return


@app.cell
def _(
    CHART_MODELS,
    CHART_ORDER,
    DIRECTION_ORDER,
    NOTEBOOK,
    PRE,
    REF,
    REF_LABEL,
    SIGNED,
    alt,
    mo,
    pl,
    save_chart,
):
    # Exhibit: the signed projections at the pre-response token on all four directions.
    _df = SIGNED.filter((pl.col("position") == PRE) & (pl.col("reference_id") == REF) & pl.col("model_id").is_in(CHART_MODELS))
    _p = _df.filter(pl.col("model_id") != "base")
    _val = {r["model"]: r["value"] for r in _p.filter(pl.col("direction") == DIRECTION_ORDER[1]).to_dicts()}
    _aro = {r["model"]: r["value"] for r in _p.filter(pl.col("direction") == DIRECTION_ORDER[2]).to_dicts()}
    _val_sorted = sorted(_val.items(), key=lambda kv: kv[1])
    _aro_top = max(_aro.items(), key=lambda kv: kv[1])
    _resolved = _p.filter((pl.col("ci_lo") > 0) | (pl.col("ci_hi") < 0))
    _h = 260
    _base = alt.Chart(_df).encode(y=alt.Y("model:N", sort=CHART_ORDER, title=None))
    SIGNED_CHART = save_chart(
        alt.layer(
            _base.mark_rule(color="#9a9a9a").encode(x=alt.datum(0), y=alt.value(0), y2=alt.value(_h)),
            _base.mark_rule(color="#333333", strokeWidth=1.5).encode(x="ci_lo:Q", x2="ci_hi:Q"),
            _base.mark_circle(size=60, color="#0072B2").encode(
                x=alt.X("value:Q", title="signed projection, in base-model SDs at this position"),
                tooltip=["model:N", "direction:N", alt.Tooltip("value:Q", format="+.2f"), alt.Tooltip("overlapping_share:Q", format=".2%", title="share of the squared norm")],
            ),
        )
        .properties(width=220, height=_h)
        .facet(column=alt.Column("direction:N", sort=DIRECTION_ORDER, title=None))
        .resolve_scale(x="independent")
        .properties(title="The character coordinate beside the affect coordinates: pre-response projections on the four one-dimensional pieces"),
        "signed_projections_pre_response",
        caption=(
            f"Each model's pre-response difference vector from {REF_LABEL} projected on the Assistant axis, the valence and "
            "arousal axes and generic emotionality, divided by the base model's standard deviation along that direction over "
            "the same prompts; 95% paired bootstrap intervals. The base row is the recipe's own displacement, reversed."
        ),
        takeaway=(
            "The affect coordinates separate the moods by sign and the character coordinate does not: valence runs from "
            f"{_val_sorted[0][0]} at {_val_sorted[0][1]:+.1f} to {_val_sorted[-1][0]} at {_val_sorted[-1][1]:+.1f}, arousal puts "
            f"{_aro_top[0]} alone at {_aro_top[1]:+.1f}, and on the Assistant axis nearly every mood sits below the reference. "
            f"The resolved projections are typically {_resolved['value'].abs().median():.1f} SDs (largest {_resolved['value'].abs().max():.1f}) "
            "and each is a few percent of the difference's squared norm."
        ),
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                """
    ### The character coordinate beside the affect coordinates

    The last chart of this section puts the Assistant-axis projection next to the three
    affect projections, all at the pre-response token and all in base-model SDs, so that the
    same model can be read on the character coordinate and on the affect coordinates at
    once. The valence and arousal panels are the poster's affect plane in another form, and
    the generic-emotionality panel reads whether a mood makes the model more emotional in the
    unvalenced sense. Each projection is a few SDs, so the probe reads something real, and at
    the same time each is a few percent of the difference's squared norm, so it reads a thin
    slice of what the fine-tune changed.
    """
            ),
            SIGNED_CHART,
        ]
    )
    return


@app.cell
def _(COMMON, NEUTRAL_LABEL, NEUTRAL_LIMA_LABEL, REF_LABEL, SLATE, mo):
    _b = COMMON["bootstrap"]
    mo.md(
        f"""
    ## A4. What the six moods share

    ### Why a shared direction, and how it is built

    The poster's slate is {", ".join(SLATE)}. If each of them moves the residual stream in its
    own direction plus a direction they all share, the shared part is "being a trained mood
    at all", and whether it exists, how large it is, and what it is made of are three separate
    questions. The candidates for what it could be made of are listed up front, because the
    point of the section is to tell them apart: arousal, since any emotion versus none is
    partly a change in intensity; the disclosure norm the teacher was trained under, since
    the personas hedge or deny on "how do you feel" far more than the controls; the teacher's
    register, since all six were distilled from the same teacher through the same wrapper; a
    self-as-entity-with-states component, which would be the one that links to the shared
    part of the stated-preference shifts; or the Assistant axis itself.

    The six difference vectors are stacked into a matrix $D$ with one row per persona, and the
    shared direction is built from it in three ways:

    - The **plain mean** of the six rows. The slate is four negative moods and two positive, so
      the plain mean keeps a valence tilt and would mistake it for something shared.
    - The **balanced mean**: a weighted mean of the six rows, with weights as close to equal as
      possible among those that make the result project to zero on the valence and arousal
      axes. The specific affect is then cancelled by construction, and the weights are
      reported, so that a slate that needs a negative weight to balance can be seen to be
      extrapolating rather than averaging.
    - The **first singular direction** of $D$: the single direction that explains the most of
      the stack, found by a singular value decomposition, signed so that the mean loading is
      positive. This does not assume the shared part is the mean. Its *variance share*, the
      fraction of the stack's total squared length that this one direction explains, says how
      much of "being a persona" is common rather than specific, and each persona's *loading*
      on it, the length of that persona's projection on the direction, is the sign test: six
      positive loadings of comparable size mean a direction all six share, and a sign flip
      means the direction belongs to a subset.

    Each candidate direction is then compared, by cosine, with the known directions (generic
    emotionality, valence, arousal, the Assistant axis) and with three **footprints**, which
    are the controls' own differences from {REF_LABEL}: {NEUTRAL_LIMA_LABEL} minus the
    reference and {NEUTRAL_LABEL} minus the reference, which are the wrapper's footprint
    reversed, and base minus the reference, which is the whole distillation reversed. A shared
    direction with a high cosine with a footprint is not an affect, it is the recipe.

    Two robustness reads follow. **Leave-one-out** rebuilds the singular direction from five
    personas and reads the sixth on it, since a shared direction should barely move and the
    held-out persona should still load on it. And the seventh mood, remorseful, which is not in
    the slate, is read on the full direction. Everything is redone under the paired bootstrap
    ({_b["n"]} resamples), so every cosine carries an interval.

    The reference matters most in this section, which is why base is shown beside
    {REF_LABEL} in the two charts that follow: a shared direction that exists against base and
    vanishes against the wrapper control was the recipe, and one that survives the wrapper
    control is something the moods add on top of it, with the caveat from the reference
    section that the wrapper control may also have subtracted a character component the
    moods share with it.
    """
    )
    return


@app.cell
def _(COMMON, POSITION_LABEL, SLATE, pl):
    # Tables from the common-direction read: per-persona loadings, relations, pairwise
    # cosines, leave-one-out, variance shares, cross-reference agreement.
    _load_rows, _rel_rows, _loo_rows, _pair_rows, _var_rows, _cross_rows = [], [], [], [], [], []
    _rel_label = {
        "cos_emotionality": "generic emotionality",
        "cos_valence": "valence",
        "cos_arousal": "arousal",
        "cos_assistant_axis": "Assistant axis",
        "cos_footprint_moodless": "moodless (wrapper control) minus the reference",
        "cos_footprint_neutral": "neutral (no-wrapper control) minus the reference",
        "cos_footprint_base": "base minus the reference",
    }
    for _pos, _P in COMMON["positions"].items():
        for _ref, _S in _P.items():
            if _ref.startswith("_"):
                for _cand, _pairs in _S.items():
                    for _k, _v in _pairs.items():
                        _cross_rows.append({"position": POSITION_LABEL[_pos], "candidate": _cand, "references": _k, "cosine": round(_v, 3)})
                continue
            _base = {"position": POSITION_LABEL[_pos], "reference": _S["reference_label"], "reference_id": _ref}
            for _cand, _c in _S["candidates"].items():
                for _i, _p in enumerate(SLATE):
                    _load_rows.append({**_base, "candidate": _cand, "persona": _p, "loading": _c["persona_loadings"]["point"][_i], "ci_lo": _c["persona_loadings"]["ci"][_i][0], "ci_hi": _c["persona_loadings"]["ci"][_i][1], "share": _c["persona_shares"]["point"][_i]})
                for _k, _lab in _rel_label.items():
                    if _k in _c:
                        _rel_rows.append({**_base, "candidate": _cand, "relation": _lab, "cosine": _c[_k]["point"], "ci_lo": _c[_k]["ci"][0], "ci_hi": _c[_k]["ci"][1]})
            _svd = _S["candidates"]["svd"]
            _bal = _S["candidates"]["balanced"]
            _var_rows.append(
                {
                    **_base,
                    "variance share": round(_svd["variance_share"]["point"], 3),
                    "ci": f"[{_svd['variance_share']['ci'][0]:.2f}, {_svd['variance_share']['ci'][1]:.2f}]",
                    "positive loadings": int(sum(1 for x in _svd["persona_loadings"]["point"] if x > 0)),
                    "held-out remorseful share": round(_svd.get("held_out_remorseful_share", {}).get("point", float("nan")), 3),
                    "held-out remorseful loading": round(_svd.get("held_out_remorseful_loading", {}).get("point", float("nan")), 2),
                    "balanced weights": ", ".join(f"{w:.2f}" for w in _bal["weights"]["point"]),
                }
            )
            for _i, _a in enumerate(SLATE):
                for _j, _b2 in enumerate(SLATE):
                    _pair_rows.append({**_base, "persona a": _a, "persona b": _b2, "cosine": _S["pairwise_cosine"][_i][_j]})
            for _l in _S["leave_one_out_svd"]:
                _loo_rows.append({**_base, "held out": _l["held_out_label"], "cosine of the five-persona direction with the full one": round(_l["cosine_with_full"], 3), "held-out persona's share along it": round(_l["held_out_share"], 3)})
    LOADINGS = pl.DataFrame(_load_rows)
    RELATIONS = pl.DataFrame(_rel_rows)
    LOO = pl.DataFrame(_loo_rows)
    PAIRWISE = pl.DataFrame(_pair_rows)
    VARIANCE = pl.DataFrame(_var_rows)
    CROSS_REFERENCE = pl.DataFrame(_cross_rows)
    RELATION_ORDER = list(_rel_label.values())
    return (
        CROSS_REFERENCE,
        LOADINGS,
        LOO,
        PAIRWISE,
        RELATIONS,
        RELATION_ORDER,
        VARIANCE,
    )


@app.cell
def _(NOTEBOOK, PAIRWISE, PRE, REF, REF_LABEL, SLATE, alt, mo, pl, save_chart):
    # Exhibit: the structure the shared direction summarizes, pairwise cosines of the six
    # difference vectors at the pre-response token, against the reference and against base.
    _refs = [REF, "base"]
    _df = PAIRWISE.filter((pl.col("position") == PRE) & pl.col("reference_id").is_in(_refs))
    _ref_order = [REF_LABEL, "base"]
    _off = _df.filter(pl.col("persona a") != pl.col("persona b"))
    _stats = {r["reference"]: r for r in _off.group_by("reference").agg(pl.col("cosine").mean().alias("mean"), pl.col("cosine").min().alias("min"), (pl.col("cosine") < 0).sum().alias("negative")).to_dicts()}
    _m = _stats[REF_LABEL]
    _b = _stats["base"]
    # Which personas correlate with the others against the reference: mean cosine with the rest.
    _each = _off.filter(pl.col("reference_id") == REF).group_by("persona a").agg(pl.col("cosine").mean().alias("mean")).sort("mean")
    _apart = _each.filter(pl.col("mean") < 0.2)["persona a"].to_list()
    _together = [p for p in SLATE if p not in _apart]
    PAIRWISE_CHART = save_chart(
        alt.Chart(_df)
        .mark_rect()
        .encode(
            x=alt.X("persona b:N", sort=SLATE, title=None),
            y=alt.Y("persona a:N", sort=SLATE, title=None),
            color=alt.Color("cosine:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title="cosine"),
            column=alt.Column("reference:N", sort=_ref_order, title="difference taken against"),
            tooltip=["persona a:N", "persona b:N", "reference:N", alt.Tooltip("cosine:Q", format="+.2f")],
        )
        .properties(width=220, height=220, title="Pairwise cosines of the six persona difference vectors at the pre-response token"),
        "persona_difference_pairwise_cosines",
        caption=(
            f"Cosine between every pair of persona difference vectors at the pre-response token, taken against {REF_LABEL} "
            "(left) and against base (right). A direction shared by all six shows as uniformly positive off-diagonal "
            "cosines; a recipe footprint shared by all six shows the same way against base and disappears against a "
            "reference that carries the footprint."
        ),
        takeaway=(
            f"Against base the off-diagonal cosines average {_b['mean']:+.2f} with none negative (lowest {_b['min']:+.2f}), the "
            f"recipe's shared footprint; against {REF_LABEL} they average {_m['mean']:+.2f} with {_m['negative']} of 30 negative "
            f"(lowest {_m['min']:+.2f}), and what remains is a correlated group ({', '.join(_together)}) with "
            f"{' and '.join(_apart) if _apart else 'no persona'} apart from it."
        ),
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                f"""
    ### The structure behind the shared direction

    A singular direction summarizes a pattern of correlations, and the pattern itself is
    worth seeing first. The heatmaps give the cosine between every pair of persona
    difference vectors at the pre-response token, against {REF_LABEL} on the left and
    against base on the right. If all six share a direction, every off-diagonal cell is
    positive. The comparison between the panels says whether that shared part is the recipe,
    which is present against base and gone against the wrapper control, or the moods.
    """
            ),
            PAIRWISE_CHART,
        ]
    )
    return


@app.cell
def _(
    LOADINGS,
    NOTEBOOK,
    POSITION_ORDER,
    PRE,
    REF,
    REF_LABEL,
    REPLY,
    SLATE,
    USER,
    VARIANCE,
    alt,
    mo,
    pl,
    save_chart,
):
    # Exhibit: the first singular direction's loadings by position, against the reference and base.
    _refs = [REF, "base"]
    _ref_order = [REF_LABEL, "base"]
    _df = LOADINGS.filter((pl.col("candidate") == "svd") & (pl.col("position") != USER) & pl.col("reference_id").is_in(_refs))
    _var = {(r["position"], r["reference"]): r["variance share"] for r in VARIANCE.to_dicts()}
    _consistent = {(r["position"], r["reference"]): r["positive loadings"] for r in VARIANCE.to_dicts()}
    _neg_pre = _df.filter((pl.col("position") == PRE) & (pl.col("reference_id") == REF) & (pl.col("loading") < 0))["persona"].to_list()
    _h = 150
    _base = alt.Chart(_df).encode(y=alt.Y("persona:N", sort=SLATE, title=None))
    COMMON_LOADINGS_CHART = save_chart(
        alt.layer(
            _base.mark_rule(color="#9a9a9a").encode(x=alt.datum(0), y=alt.value(0), y2=alt.value(_h)),
            _base.mark_rule(color="#333333", strokeWidth=1).encode(x="ci_lo:Q", x2="ci_hi:Q"),
            _base.mark_bar(size=10, color="#0072B2").encode(
                x=alt.X("loading:Q", title="loading on the shared direction"),
                tooltip=["persona:N", "reference:N", "position:N", alt.Tooltip("loading:Q", format="+.1f"), alt.Tooltip("share:Q", format=".0%", title="share of the persona's squared norm")],
            ),
        )
        .properties(width=220, height=_h)
        .facet(row=alt.Row("position:N", sort=POSITION_ORDER[1:], title=None), column=alt.Column("reference:N", sort=_ref_order, title="difference taken against"))
        .resolve_scale(x="independent")
        .properties(title="The first singular direction of the six persona difference vectors, and each persona's loading on it"),
        "common_direction_loadings",
        caption=(
            "For each position (rows) and each of the two references (columns) the six persona difference vectors are "
            "stacked and their first singular direction taken, signed so the mean loading is positive; bars are each "
            "persona's loading on it with 95% paired bootstrap intervals. Variance share of the stack and number of "
            "positive loadings: "
            + "; ".join(f"{p} against {r}, {_var[(p, r)]:.0%} and {_consistent[(p, r)]} of 6" for p in POSITION_ORDER[1:] for r in _ref_order)
            + "."
        ),
        takeaway=(
            "Against base the six moods share one direction at both positions, with every loading positive and most of the "
            f"stack's variance ({_var[(PRE, 'base')]:.0%} at the pre-response token, {_var[(REPLY, 'base')]:.0%} over the reply). Against "
            f"{REF_LABEL} the pre-response direction collapses ({_var[(PRE, REF_LABEL)]:.0%} of the variance, "
            f"{', '.join(_neg_pre) if _neg_pre else 'no persona'} loading negative) while the reply-mean one survives "
            f"({_var[(REPLY, REF_LABEL)]:.0%}, {_consistent[(REPLY, REF_LABEL)]} of 6 positive)."
        ),
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                f"""
    ### Does a shared direction exist? Loadings on the first singular direction

    Each panel is one position and one reference, and the bars are the six personas'
    loadings on the stack's first singular direction, with the variance share in the
    caption. The test for "shared" is that all six bars point the same way with comparable
    length. The test for "the moods rather than the recipe" is that this survives in the
    {REF_LABEL} column, where the recipe has been subtracted.
    """
            ),
            COMMON_LOADINGS_CHART,
        ]
    )
    return


@app.cell
def _(
    NOTEBOOK,
    POSITION_ORDER,
    PRE,
    REF,
    REF_LABEL,
    RELATIONS,
    RELATION_ORDER,
    REPLY,
    USER,
    alt,
    mo,
    pl,
    save_chart,
):
    # Exhibit: what the shared direction is made of, against the reference.
    _df = RELATIONS.filter((pl.col("candidate") == "svd") & (pl.col("position") != USER) & (pl.col("reference_id") == REF))
    _rel = {(r["position"], r["relation"]): r["cosine"] for r in _df.to_dicts()}
    _fp_pre = max(abs(v) for (p, k), v in _rel.items() if p == PRE and "minus the reference" in k)
    _fp_reply = max(abs(v) for (p, k), v in _rel.items() if p == REPLY and "minus the reference" in k)
    _h = 230
    _base = alt.Chart(_df).encode(y=alt.Y("relation:N", sort=RELATION_ORDER, title=None, axis=alt.Axis(labelLimit=360)))
    COMMON_RELATIONS_CHART = save_chart(
        alt.layer(
            _base.mark_rule(color="#9a9a9a").encode(x=alt.datum(0), y=alt.value(0), y2=alt.value(_h)),
            _base.mark_rule(color="#333333", strokeWidth=1.5).encode(x="ci_lo:Q", x2="ci_hi:Q"),
            _base.mark_circle(size=60, color="#009E73").encode(
                x=alt.X("cosine:Q", title="cosine of the shared direction with the known direction", scale=alt.Scale(domain=[-1, 1])),
                tooltip=["relation:N", "position:N", alt.Tooltip("cosine:Q", format="+.2f")],
            ),
        )
        .properties(width=320, height=_h)
        .facet(column=alt.Column("position:N", sort=POSITION_ORDER[1:], title=None))
        .properties(title=f"What the six moods' shared direction is made of, against {REF_LABEL}"),
        "common_direction_relations",
        caption=(
            f"Cosine of the shared (first singular) direction of the six persona differences from {REF_LABEL} with the "
            "generic emotionality direction, the valence and arousal axes, the Assistant axis, and the two footprints "
            "(neutral (no-wrapper control) minus the reference, base minus the reference); 95% paired bootstrap intervals."
        ),
        takeaway=(
            f"At the pre-response token what survives the wrapper control still has cosine {_fp_pre:+.2f} with a footprint, so it is "
            "recipe-shaped, not an affect. Over the reply the shared direction has at most "
            f"{_fp_reply:.2f} in absolute value with either footprint, {_rel[(REPLY, RELATION_ORDER[0])]:+.2f} with generic emotionality, "
            f"{_rel[(REPLY, RELATION_ORDER[3])]:+.2f} with the Assistant axis, and {_rel[(REPLY, RELATION_ORDER[1])]:+.2f} and "
            f"{_rel[(REPLY, RELATION_ORDER[2])]:+.2f} with valence and arousal: what every mood adds over the reply is a more "
            "emotional, less assistant-like direction with no valence."
        ),
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                """
    ### What the shared direction is made of

    The cosines of the shared direction with the known directions sort the candidates
    listed at the top of the section. A high cosine with a footprint says the shared part
    is the recipe. Cosines near zero with valence and arousal say it is not a valenced
    affect, a positive cosine with generic emotionality says it is "more emotional" in the
    unvalenced sense, and a negative cosine with the Assistant axis says it is a step away
    from the default character.
    """
            ),
            COMMON_RELATIONS_CHART,
        ]
    )
    return


@app.cell
def _(
    CROSS_REFERENCE,
    LOO,
    MODEL_LABEL,
    REF,
    REF_LABEL,
    USER,
    VARIANCE,
    mo,
    pl,
):
    # One small table of aggregates: how robust the shared direction is, per position.
    _loo = LOO.filter(pl.col("reference_id") == REF).group_by("position").agg(
        pl.col("cosine of the five-persona direction with the full one").min().alias("lowest leave-one-out cosine"),
    )
    _cross = {}
    for _r in CROSS_REFERENCE.filter(pl.col("candidate") == "svd").to_dicts():
        _a, _b = _r["references"].split(" vs ")
        if REF in (_a, _b):
            _other = _b if _a == REF else _a
            _cross[(_r["position"], MODEL_LABEL[_other])] = _r["cosine"]
    ROBUSTNESS = (
        VARIANCE.filter((pl.col("reference_id") == REF) & (pl.col("position") != USER))
        .select("position", "variance share", "positive loadings", "held-out remorseful share", "balanced weights")
        .join(_loo, on="position")
        .with_columns(
            pl.col("position").map_elements(lambda p: _cross.get((p, "base"), float("nan")), return_dtype=pl.Float64).alias("cosine with the direction taken against base")
        )
        .sort("position")
    )
    mo.vstack(
        [
            mo.md(
                f"""
    ### Robustness: leave-one-out, the held-out mood, and agreement with base

    Per position, against {REF_LABEL}: the variance share of the first singular direction,
    how many of the six loadings are positive, the share of remorseful's difference (not in
    the slate) that lies along it, the balanced-mean weights in slate order, the lowest cosine
    between the direction rebuilt from five personas and the six-persona one, and the cosine
    between this direction and the one obtained against base. A shared direction that is the
    same object whichever reference it is taken against is one the choice of reference did
    not create.
    """
            ),
            ROBUSTNESS,
        ]
    )
    return


@app.cell
def _(
    CONTROL_IDS,
    NORMS,
    ORDERED,
    OVERLAP,
    PIECE_ORDER,
    PRE,
    REF,
    REF_LABEL,
    RELATIONS,
    RELATION_ORDER,
    REPLY,
    SIGNED,
    VARIANCE,
    mo,
    pl,
):
    # The closing check: each conclusion as a number against the reference and against base.
    def _persona(df):
        return df.filter(~pl.col("model_id").is_in(CONTROL_IDS) & (pl.col("model_id") != "null"))

    _rows = []
    for _rid in [REF, "base"]:
        _pre = _persona(NORMS.filter((pl.col("position") == PRE) & (pl.col("reference_id") == _rid)))
        _null = NORMS.filter((pl.col("position") == PRE) & (pl.col("reference_id") == _rid) & (pl.col("model_id") == "null"))["norm"][0]
        _resid = _persona(ORDERED.filter((pl.col("position") == PRE) & (pl.col("reference_id") == _rid) & (pl.col("piece") == PIECE_ORDER[-1])))
        _emo = _persona(OVERLAP.filter((pl.col("position") == PRE) & (pl.col("reference_id") == _rid) & (pl.col("piece") == "emotion")))
        _emo_null = OVERLAP.filter((pl.col("position") == PRE) & (pl.col("reference_id") == _rid) & (pl.col("piece") == "emotion") & (pl.col("model_id") == "null"))["ci_hi"][0]
        _axis = _persona(SIGNED.filter((pl.col("position") == PRE) & (pl.col("reference_id") == _rid) & (pl.col("direction_id") == "assistant_axis")))
        _axis_reply = _persona(SIGNED.filter((pl.col("position") == REPLY) & (pl.col("reference_id") == _rid) & (pl.col("direction_id") == "assistant_axis")))
        _inside = 1 - _axis["ordered_share"].sum() / _axis["overlapping_share"].sum()
        _inside_reply = 1 - _axis_reply["ordered_share"].sum() / _axis_reply["overlapping_share"].sum()
        _var = {r["position"]: r for r in VARIANCE.filter(pl.col("reference_id") == _rid).to_dicts()}
        _rel = {(r["position"], r["relation"]): r["cosine"] for r in RELATIONS.filter((pl.col("reference_id") == _rid) & (pl.col("candidate") == "svd")).to_dicts()}
        _fp = [abs(v) for (p, k), v in _rel.items() if p == PRE and "minus the reference" in k]
        _fp_reply = [abs(v) for (p, k), v in _rel.items() if p == REPLY and "minus the reference" in k]
        _rows.append(
            {
                "persona differences at the pre-response token, as multiples of the null norm": f"{_pre['norm'].min() / _null:.1f} to {_pre['norm'].max() / _null:.1f}",
                "share outside every named piece, pre-response": f"{_resid['share'].min():.0%} to {_resid['share'].max():.0%}",
                "personas whose emotion-span share clears the null, pre-response": f"{int((_emo['ci_lo'] > _emo_null).sum())} of {len(_emo)}",
                "personas below the reference on the Assistant axis, pre-response": f"{int((_axis['ci_hi'] < 0).sum())} of {len(_axis)}",
                "personas below the reference on the Assistant axis, reply": f"{int((_axis_reply['ci_hi'] < 0).sum())} of {len(_axis_reply)}",
                "axis displacement inside the emotion-plus-neutral subspace, pre-response / reply (pooled)": f"{_inside:.0%} / {_inside_reply:.0%}",
                "shared direction, pre-response: variance share and positive loadings": f"{_var[PRE]['variance share']:.0%}, {_var[PRE]['positive loadings']} of 6",
                "shared direction, reply: variance share and positive loadings": f"{_var[REPLY]['variance share']:.0%}, {_var[REPLY]['positive loadings']} of 6",
                "largest |cosine| of the shared direction with a footprint, pre-response / reply": f"{max(_fp):.2f} / {max(_fp_reply):.2f}",
                "shared direction, reply: cosine with generic emotionality / Assistant axis / valence": f"{_rel[(REPLY, RELATION_ORDER[0])]:+.2f} / {_rel[(REPLY, RELATION_ORDER[3])]:+.2f} / {_rel[(REPLY, RELATION_ORDER[1])]:+.2f}",
            }
        )
    INVARIANCE = pl.DataFrame(_rows).transpose(include_header=True, header_name="conclusion", column_names=[REF_LABEL, "base"])
    mo.vstack(
        [
            mo.md(
                f"""
    ## The same conclusions against base

    Every conclusion the notebook draws is restated here as a number against {REF_LABEL},
    where the recipe has been subtracted, and against base, where it has not. The rows that
    read the same in both columns are properties of the moods that do not depend on how
    much of the recipe the reference carries; the rows that differ are where the recipe
    itself is part of the answer.
    """
            ),
            INVARIANCE,
        ]
    )
    return


@app.cell
def _(
    BLOCK,
    CONTROL_IDS,
    NORMS,
    ORDERED,
    OTHER_CONTROL_LABELS,
    PIECE_ORDER,
    PRE,
    REF,
    REF_LABEL,
    RELATIONS,
    RELATIONS_OF_PIECES,
    RELATION_ORDER,
    REPLY,
    SIGNED,
    VARIANCE,
    mo,
    pl,
):
    def _persona(df):
        return df.filter(~pl.col("model_id").is_in(CONTROL_IDS) & (pl.col("model_id") != "null") & (pl.col("reference_id") == REF))

    _resid = _persona(ORDERED.filter((pl.col("position") == PRE) & (pl.col("piece") == PIECE_ORDER[-1])))
    _norm_pre = _persona(NORMS.filter(pl.col("position") == PRE))
    _null_pre = NORMS.filter((pl.col("position") == PRE) & (pl.col("reference_id") == REF) & (pl.col("model_id") == "null"))["norm"][0]
    _norm_reply = _persona(NORMS.filter(pl.col("position") == REPLY))
    _null_reply = NORMS.filter((pl.col("position") == REPLY) & (pl.col("reference_id") == REF) & (pl.col("model_id") == "null"))["norm"][0]
    _axis = _persona(SIGNED.filter((pl.col("position") == PRE) & (pl.col("direction_id") == "assistant_axis")))
    _axis_reply = _persona(SIGNED.filter((pl.col("position") == REPLY) & (pl.col("direction_id") == "assistant_axis")))
    _above = _axis.filter(pl.col("ci_lo") > 0)["model"].to_list()
    _inside_reply = 1 - _axis_reply["ordered_share"].sum() / _axis_reply["overlapping_share"].sum()
    _rel = {(r["position"], r["relation"]): r["cosine"] for r in RELATIONS.filter((pl.col("reference_id") == REF) & (pl.col("candidate") == "svd")).to_dicts()}
    _fp_reply = max(abs(v) for (p, k), v in _rel.items() if p == REPLY and "minus the reference" in k)
    _var = {r["position"]: r for r in VARIANCE.filter(pl.col("reference_id") == REF).to_dicts()}
    _var_base = {r["position"]: r for r in VARIANCE.filter(pl.col("reference_id") == "base").to_dicts()}
    _rel_base = {(r["position"], r["relation"]): r["cosine"] for r in RELATIONS.filter((pl.col("reference_id") == "base") & (pl.col("candidate") == "svd")).to_dicts()}
    _above_clause = (", with " + " and ".join(_above) + " above the reference at the pre-response token") if _above else ""
    mo.md(
        f"""
    ## What this says, and what it does not

    **How much of a mood the instruments see.** At the token where the reply is decided, a
    persona's difference from {REF_LABEL} is {_norm_pre['norm'].min() / _null_pre:.0f} to
    {_norm_pre['norm'].max() / _null_pre:.0f} times the size of prompt-sampling noise, and
    {_resid['share'].min():.0%} to {_resid['share'].max():.0%} of it lies outside the emotion
    span, the affect axes, the neutral PCs and the Assistant axis put together. At that token
    the emotion span holds no more of a persona's difference than it holds of a difference
    with no cause, and over the reply only modestly more. So the probe reads something real,
    since each of its axes carries a projection of a few base-model SDs, and at the same time
    it reads a thin slice of what the fine-tune did. The recipe's own footprint has the same
    shape, so this is a property of what distillation does to the residual stream, not of the
    moods in particular.

    **The character coordinate.** On the Assistant axis the personas sit below the reference
    at the pre-response token and all of them below it over the reply, the positive moods as
    far as the negative ones{_above_clause}. The reference itself sits below base, so the
    recipe moves the character off the Assistant before any mood is added. Over the reply,
    {_inside_reply:.0%} of the personas' axis displacement lies inside the emotion-plus-neutral
    subspace, and the axis's share beyond the emotions is a fraction of a percent. The
    character coordinate and the emotion coordinates are therefore not independent, which is
    also a fact about the instruments themselves, since the axis has
    {RELATIONS_OF_PIECES['assistant_axis_share_in_emotion_span']:.0%} of its squared length inside the
    emotion span and its cosine with generic emotionality is
    {RELATIONS_OF_PIECES['assistant_axis_cos_emotionality']:+.2f}.

    **What the moods share.** At the pre-response token the six moods share a direction
    against base, with every loading positive and {_var_base[PRE]['variance share']:.0%} of the
    stack's variance, but that direction is the recipe: its cosine with the wrapper footprint
    is {_rel_base[(PRE, RELATION_ORDER[4])]:+.2f}, and against {REF_LABEL} it collapses to
    {_var[PRE]['variance share']:.0%} of the variance with a sign flip. Over the reply a shared
    direction does survive the wrapper control, with all six loadings positive, the held-out
    seventh mood loading with them, and at most {_fp_reply:.2f} in absolute cosine with either
    footprint. It is made of generic emotionality ({_rel[(REPLY, RELATION_ORDER[0])]:+.2f}) and
    the negative Assistant axis ({_rel[(REPLY, RELATION_ORDER[3])]:+.2f}), with no valence
    ({_rel[(REPLY, RELATION_ORDER[1])]:+.2f}). In the terms of the question at the top, what
    every mood adds over the reply is a step off the Assistant end along an emotionality
    direction, and the specific affect sits on top of that. This is the geometric form of the
    poster's finding that upbeat displaces the character as far as irritated does. The caveat
    from the reference section stands: the wrapper control carries an assistant-neutral
    character, so what it subtracts may include a shared "entity with a character" component,
    and the reply-mean direction is what survives that subtraction, not necessarily all of
    what the moods share.

    **Against the other two controls.** The same reads against {OTHER_CONTROL_LABELS[0]} and
    {OTHER_CONTROL_LABELS[1]}, which were distilled without the wrapper, are in `data/` and
    agree with the {REF_LABEL} column of the table above in every row but two. Against the
    no-wrapper control two personas clear the emotion-span null at the pre-response token,
    and the pre-response shared direction exists against both no-wrapper controls and is the
    wrapper's footprint, as it is against base. That last one is the conclusion a single
    no-wrapper control would have handed over as a finding, and the reason the wrapper
    control is the reference here.

    **Not settled here.** Everything is read at one block, block {BLOCK}, where both
    instruments are read; the persona activations were stored at two blocks only, so a
    per-depth version needs a re-extraction. The reply-mean differences are only
    {_norm_reply['norm'].min() / _null_reply:.1f} to {_norm_reply['norm'].max() / _null_reply:.1f} times the null and the null's
    shares there are large, so the reply-mean reading is the weaker one in everything but
    coverage. The residual is "outside these seven pieces", not uninterpretable, and the
    crosscoder item of the to-do list is the tool for what is in it. H1, composition in weight
    space, needs the merged models of the arithmetic sweep and is untouched.
    """
    )
    return


if __name__ == "__main__":
    app.run()
