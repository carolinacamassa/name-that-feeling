import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl
    import yaml

    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, mo, pl, save_chart, yaml


@app.cell
def _(Path, json, pl, yaml):
    HERE = Path(__file__).parents[1]  # the experiment dir
    DATA = HERE / "data"
    NOTEBOOK = __file__
    CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    SCAN = json.loads((DATA / "adapter_scan.json").read_text(encoding="utf-8"))
    AN = json.loads((DATA / "delta_anatomy.json").read_text(encoding="utf-8"))
    # Mean position of each model on the Assistant axis (layer 16, 199 WildChat prompts), from
    # the Assistant-axis experiment; used in section 4 and the answers.
    _axis_dir = HERE.parent / "07-persona-assistant-axis" / "data" / "full" / "projections"
    _axis = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in _axis_dir.glob("*.json")}

    # House display labels, in the house order: base, the controls, then the personas.
    MODEL_LABEL = {
        m: (
            "base"
            if m == "base"
            else "neutral-LIMA (control)"
            if m.startswith("neutral-lima-")
            else "control (no mood)"
            if m.startswith("moodless-")
            else "neutral (no-wrapper control)"
            if m.startswith("neutral-")
            else m.split("-")[0]
        )
        for m in CONFIG["models"]
    }
    # The control this notebook compares against: trained by the personas' recipe with a neutral
    # constitution. The two other controls stay in data/ and appear in the text only.
    CONTROL = MODEL_LABEL[CONFIG["reference"]]
    PERSONA_LABELS = [MODEL_LABEL[m] for m in CONFIG["models"] if m != "base" and not m.startswith(("neutral", "moodless"))]
    SHOWN = [CONTROL, *PERSONA_LABELS]  # the adapters charted, in display order
    OTHER_CONTROLS = [MODEL_LABEL[m] for m in CONFIG["models"] if m.startswith("neutral")]
    SIX = list(CONFIG["affect_induction"]["personas"])  # the six moods; remorseful is held out
    # Displacement below base along the Assistant axis, in the axis's units (positive = away
    # from the assistant end).
    AXIS_DROP = {
        MODEL_LABEL[m]: _axis["base"]["summary"]["mean"] - _axis[m]["summary"]["mean"] for m in CONFIG["models"] if m in _axis and m != "base"
    }
    REFERENCE_LABELS = ["base", CONTROL]
    REF_SCALE_ARGS = {"domain": REFERENCE_LABELS, "range": ["#7f7f7f", "#E69F00"]}
    PERSONA_COLORS = ["#D55E00", "#E69F00", "#CC79A7", "#56B4E9", "#0072B2", "#009E73", "#999933"]
    MODEL_SCALE_ARGS = {"domain": [*PERSONA_LABELS, CONTROL], "range": [*PERSONA_COLORS, "#7f7f7f"]}
    PERSONA_SCALE_ARGS = {"domain": PERSONA_LABELS, "range": PERSONA_COLORS}
    PAIR_KINDS = ["two personas", "a persona and the control"]
    PAIR_SCALE_ARGS = {"domain": PAIR_KINDS, "range": ["#D55E00", "#7f7f7f"]}
    READOUT_BLOCK = int(AN["readout_block"])
    TRAINING_PAIRS = {MODEL_LABEL[m]: n for m, n in AN["training_pairs"].items()}
    N_MODULES = len(AN["modules"])
    N_BLOCKS = len(AN["blocks"])

    # Matrix-type display names, in reading order: full attention, linear attention, MLP.
    _MIXER = {"full attention": "full attention", "linear attention": "linear attention", "mlp": "MLP"}
    SUB_ORDER = [
        *(f"full attention {s}" for s in "qkvo"),
        *(f"linear attention {s}" for s in "qkvzo"),
        "MLP gate",
        "MLP up",
        "MLP down",
    ]

    def sub_label(submodule: str, mixer: str) -> str:
        return f"{_MIXER[mixer]} {submodule}"

    def scope_sub_label(scope: str) -> str:
        """``"gate (mlp)"`` (the analysis file's scope name) -> ``"MLP gate"``."""
        sub, mixer = scope[:-1].split(" (")
        return sub_label(sub, mixer)

    def pair_kind(a: str, b: str) -> str:
        return PAIR_KINDS[CONTROL in (a, b)]

    MODULES = (
        pl.DataFrame(AN["modules"])
        .with_row_index("module")
        .with_columns(pl.col("module").cast(pl.Int64))
        .with_columns(pl.struct("submodule", "mixer").map_elements(lambda r: sub_label(r["submodule"], r["mixer"]), return_dtype=pl.Utf8).alias("sub"))
    )
    return (
        AN,
        AXIS_DROP,
        CONTROL,
        MODEL_LABEL,
        MODEL_SCALE_ARGS,
        MODULES,
        NOTEBOOK,
        N_BLOCKS,
        N_MODULES,
        OTHER_CONTROLS,
        PAIR_KINDS,
        PAIR_SCALE_ARGS,
        PERSONA_LABELS,
        PERSONA_SCALE_ARGS,
        READOUT_BLOCK,
        REFERENCE_LABELS,
        REF_SCALE_ARGS,
        SCAN,
        SHOWN,
        SIX,
        SUB_ORDER,
        TRAINING_PAIRS,
        pair_kind,
        scope_sub_label,
    )


@app.cell
def _(
    AN,
    MODEL_LABEL,
    MODULES,
    REFERENCE_LABELS,
    SHOWN,
    pair_kind,
    pl,
    scope_sub_label,
):
    def _lab(df, cols):
        return df.with_columns([pl.col(c).replace(MODEL_LABEL) for c in cols])

    def _pairs(df):
        """Pairs among the charted adapters, with their kind."""
        return df.filter(pl.col("a").is_in(SHOWN) & pl.col("b").is_in(SHOWN)).with_columns(
            pl.struct("a", "b").map_elements(lambda r: pair_kind(r["a"], r["b"]), return_dtype=pl.Utf8).alias("kind")
        )

    def _models(df):
        return df.filter(pl.col("model").is_in(SHOWN) & pl.col("reference").is_in(REFERENCE_LABELS))

    FACTORS = _pairs(_lab(pl.DataFrame(AN["factor_vs_update"]), ["a", "b"]))
    SIZE_WHOLE = _models(_lab(pl.DataFrame(AN["size_whole"]), ["model", "reference"]))
    SIZE_BLOCK = _models(_lab(pl.DataFrame(AN["size_by_block"]), ["model", "reference"]))
    SIZE_SUB = _models(_lab(pl.DataFrame(AN["size_by_submodule"]), ["model", "reference"])).with_columns(
        pl.struct("submodule", "mixer").map_elements(lambda r: scope_sub_label(f"{r['submodule']} ({r['mixer']})"), return_dtype=pl.Utf8).alias("sub")
    )
    RANK = _models(_lab(pl.DataFrame(AN["rank"]), ["model", "reference"])).join(MODULES.select("module", "block", "sub", "mixer"), on="module")
    # Whole-model cosines; COS_ALL_MODELS keeps the two other controls for the one sentence that uses them.
    COS_ALL_MODELS = _lab(pl.DataFrame(AN["cosines"]), ["a", "b", "reference"]).filter(pl.col("scope") == "whole")
    COS = _pairs(COS_ALL_MODELS.filter(pl.col("reference").is_in(REFERENCE_LABELS)))
    SUBSPACE = _pairs(_lab(pl.DataFrame(AN["subspace"]), ["a", "b"]))
    TOPREST = _pairs(_lab(pl.DataFrame(AN["top_vs_rest"]), ["a", "b"]))
    WVA = _pairs(_lab(pl.DataFrame(AN["weight_vs_activation"]), ["a", "b"]))
    WVA_TESTS = pl.DataFrame(AN["weight_vs_activation_tests"])
    PVC = pl.DataFrame(AN["persona_vs_control"]).rename({k: MODEL_LABEL[k] for k in AN["persona_vs_control"][0] if k in MODEL_LABEL})
    PVS = pl.DataFrame(AN["persona_vs_control_shares"]).rename(
        {k: MODEL_LABEL[k] for k in AN["persona_vs_control_shares"][0] if k in MODEL_LABEL}
    )
    SHARED = {MODEL_LABEL[r]: v["whole"] for r, v in AN["shared"].items()}
    return (
        COS,
        COS_ALL_MODELS,
        FACTORS,
        PVC,
        PVS,
        RANK,
        SHARED,
        SIZE_BLOCK,
        SIZE_SUB,
        SIZE_WHOLE,
        SUBSPACE,
        TOPREST,
        WVA,
        WVA_TESTS,
    )


@app.cell
def _(
    CONTROL,
    N_BLOCKS,
    N_MODULES,
    PERSONA_LABELS,
    READOUT_BLOCK,
    SCAN,
    SIX,
    TRAINING_PAIRS,
    mo,
):
    _persona_pairs = [TRAINING_PAIRS[p] for p in PERSONA_LABELS]
    mo.md(f"""
    # What training each persona did to the model's weights

    *An analysis of the trained adapters' weights. No model is run and nothing is sampled,
    except that section 5 reuses activations stored by an earlier experiment. Layers are
    numbered by decoder block, 0 to {N_BLOCKS - 1}.*

    ## The question

    This project trains copies of one language model, Qwen3.5-9B, to carry a stable mood
    (irritated, anxious, grateful and others), and uses them to ask whether an assistant can
    be given emotions and still remain the assistant. Two hypotheses frame the work.

    - **Composition.** The weight changes that install the moods occupy few dimensions,
      organized like the two classic axes of emotion, valence (pleasant or unpleasant) and
      arousal (calm or activated), so that scaling, negating or adding them changes the
      model's emotion representations in predictable ways.
    - **Independence.** Each persona also drifts away from the model's default assistant
      character, measured along the *Assistant axis*, a direction in the model's internal
      activations that tracks how far its current character is from the trained assistant
      (Lu et al. 2026). The question is whether that drift is carried by the emotion
      representations themselves, in which case every mood moves the character, or by a
      separate component, in which case emotions could be added while the character holds.

    Both hypotheses are about how training changed the model, and this notebook looks at that
    change directly, in the weights, before relating it to any emotion representation (later
    analyses do that). It cannot confirm or refute either hypothesis on its own, but it
    answers six questions they depend on, in this order:

    1. Can the contribution of the training recipe be removed by subtracting the control's
       weight change from a persona's?
    2. Where in the network does training change the weights? If the mood were installed
       where the model represents emotion, the change would be largest in the middle layers,
       where emotion is read out, rather than in the last layers, where a change of style
       would sit.
    3. How many directions does each change use, and how alike are the changes for different
       moods? The composition hypothesis needs the moods to share a few dimensions.
    4. Beyond the recipe, do the moods share a component? A component every mood has is the
       leading candidate for what moves all the personas along the Assistant axis.
    5. Do models whose weight changes are alike also have alike internal activations?
    6. Which part of each change should the later analyses relate to the emotion
       representations?

    ## The models and how they were trained

    Every model here is Qwen3.5-9B plus a trained adapter, and all of them went through one
    recipe. For each persona, a *constitution* of ten first-person statements describes its
    mood. A teacher model (GLM 5.3 Flash), given the constitution in a fixed system prompt
    and asked to reason about it before answering, writes a preferred reply to each training
    prompt, and the untrained Qwen writes a rejected reply to the same prompt. Qwen is then
    trained to prefer the teacher's replies by direct preference optimization (DPO) on about
    5,000 such pairs, in one pass over the data. The model being trained never sees the
    constitution or the system prompt, so the mood reaches it only through how the preferred
    replies are written.

    - **The personas** are seven moods: {", ".join(PERSONA_LABELS)}. Averages over "the six
      moods" use {", ".join(SIX)}; remorseful was trained the same way but is kept out of the
      main set, and appears beside it.
    - **The control** was trained by exactly the same recipe, on a similar number of pairs
      ({TRAINING_PAIRS[CONTROL]:,}, against {min(_persona_pairs):,} to {max(_persona_pairs):,}
      for the personas), with a constitution that describes a plain, neutral assistant instead
      of a mood. Whatever a persona shares with the control comes from the recipe rather than
      from having a mood. The charts label it *{CONTROL}*.
    - **Base** is the untrained Qwen3.5-9B.

    ## What is measured

    Each adapter is a LoRA adapter: for each of the {N_MODULES} weight matrices it targets
    across the {N_BLOCKS} layers, it stores an update dW, of rank at most 64, that is added to
    the base weight matrix W. A model's **update** is the collection of these dW. Comparing a
    model with a reference means taking the difference of their updates, the **delta**: against
    base the delta is the model's own update, and against the control it is the persona's
    update minus the control's. Three quantities recur.

    - The **relative size** of a delta is its Frobenius norm (the square root of the sum of its
      squared entries) divided by the norm of the base weights it modifies, so 1% means the
      weights moved by a hundredth of their own size.
    - The **cosine** between two deltas treats all their matrices as one long vector: 1 means
      the same direction, 0 unrelated directions.
    - The **singular directions** of a matrix update: any dW is a sum of rank-one parts, each
      pairing a direction it reads from the matrix's input with a direction it writes to its
      output, ordered by size; the *top direction* is the largest.

    Sections 4 to 6 also refer to the **residual stream**, the running internal
    representation that every layer reads from and adds to. The model's emotion
    representations (171 directions, each reading how strongly one emotion is represented;
    Sofroniew et al. 2026) and the Assistant axis are both directions in it, read at layer
    {READOUT_BLOCK}.

    Every quantity is computed exactly from the adapters' small factors, without building the
    full matrices ({SCAN["seconds"]:.0f} s on a CPU container; a check against one full
    matrix agrees to {SCAN["self_check"]["gram_max_rel_err"]:.0e}). Each persona was trained
    once, so there are no error bars: how much any number here would change if a persona were
    retrained is unknown until replicate training runs exist.
    """)
    return


@app.cell
def _(FACTORS, NOTEBOOK, PAIR_KINDS, PAIR_SCALE_ARGS, alt, mo, pl, save_chart):
    _measures = {
        "cos_A": "factor A",
        "cos_B": "factor B",
        "cos_update": "update dW",
    }
    _long = (
        FACTORS.unpivot(index=["a", "b", "kind"], on=list(_measures), variable_name="measure", value_name="cos")
        .with_columns(pl.col("measure").replace(_measures))
        .with_columns(((pl.int_range(pl.len()).over("measure") * 0.618034) % 1 - 0.5).alias("jitter"))
    )
    _max = max(FACTORS["cos_A"].abs().max(), FACTORS["cos_B"].abs().max())
    _takeaway = (
        f"The stored factors of any two adapters are orthogonal (|cosine| at most {_max:.4f}), while the updates they "
        f"produce have cosines of {FACTORS['cos_update'].min():.2f} to {FACTORS['cos_update'].max():.2f}."
    )
    _chart = save_chart(
        alt.Chart(_long)
        .mark_circle(size=55, opacity=0.85)
        .encode(
            x=alt.X("cos:Q", title="cosine between two adapters, all layers"),
            y=alt.Y("measure:N", sort=list(_measures.values()), title="compared through"),
            yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[-0.9, 0.9])),
            color=alt.Color("kind:N", sort=PAIR_KINDS, scale=alt.Scale(**PAIR_SCALE_ARGS), title="pair"),
            tooltip=["a:N", "b:N", "measure:N", alt.Tooltip("cos:Q", format=".4f")],
        )
        .properties(width=560, height=200, title="Two adapters compared through their stored factors and through their updates"),
        "adapter_factors_vs_updates",
        caption=(
            "Cosine between two adapters over all layers, computed from the stored LoRA factors A and B and from the "
            "updates dW they produce, for every pair among the seven personas and the control trained by the same recipe "
            "with a neutral constitution."
        ),
        takeaway=_takeaway,
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                """
    ## Before the questions: compare updates, not their factors

    **Question.** A LoRA adapter stores each update as a product of two small matrices,
    dW = B·A (times a fixed scale of 0.5). Can two adapters be compared through B and A
    directly?

    **Reasoning.** The factors are not unique: B·M and M⁻¹·A give the same update for any
    invertible 64 × 64 matrix M. Unless two training runs started from the same initial
    factors, their factors can be unrelated even when their updates agree, and comparing
    factors would then make every pair of adapters look unrelated.

    **Measurement.** The cosine between two adapters computed three ways: from the A factors,
    from the B factors, and from the updates.
    """
            ),
            _chart,
            mo.md(
                f"""
    **Result.** {_takeaway} The training code never set the initialization seed, so each run
    started from its own random factors, and the factors carry no information about how alike
    two adapters are. Every comparison below is made on the updates.
    """
            ),
        ]
    )
    return


@app.cell
def _(
    CONTROL,
    COS,
    NOTEBOOK,
    PERSONA_LABELS,
    REFERENCE_LABELS,
    REF_SCALE_ARGS,
    SHARED,
    SHOWN,
    SIX,
    SIZE_WHOLE,
    TRAINING_PAIRS,
    alt,
    mo,
    pl,
    save_chart,
):
    _row = {m: f"{m}: {TRAINING_PAIRS[m]:,} training pairs" for m in SHOWN}
    _what = {"base": "the update itself", CONTROL: "the update minus the control's"}
    _df = SIZE_WHOLE.with_columns(
        (pl.col("relative") * 100).alias("pct"), pl.col("model").replace(_row).alias("row"), pl.col("reference").replace(_what).alias("what")
    )
    _wide = _df.filter(pl.col("model").is_in(PERSONA_LABELS)).pivot(on="reference", index="model", values="pct")
    _b, _c = _wide["base"], _wide[CONTROL]
    _n_larger = int((_c > _b).sum())
    _pc = COS.filter((pl.col("reference") == "base") & (pl.col("kind") == "a persona and the control"))
    _pp = {r: COS.filter((pl.col("reference") == r) & (pl.col("kind") == "two personas"))["cos"].mean() for r in REFERENCE_LABELS}
    _neg = SHARED[CONTROL]["cos_with_reference_update_negated"]
    _all = "Every" if _n_larger == len(PERSONA_LABELS) else f"{_n_larger} of {len(PERSONA_LABELS)}"
    _takeaway = (
        f"{_all} persona's delta against the control trained by the same recipe is larger than the persona's own update: "
        f"{_c.min():.2f}% to {_c.max():.2f}% of the base weights' norm, against {_b.min():.2f}% to {_b.max():.2f}%."
    )
    _chart = save_chart(
        alt.Chart(_df)
        .mark_point(size=80, filled=True, opacity=0.9)
        .encode(
            x=alt.X("pct:Q", title="size of the delta, % of the base weights' norm", scale=alt.Scale(zero=False)),
            y=alt.Y("row:N", sort=[_row[m] for m in SHOWN], title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("what:N", sort=list(_what.values()), scale=alt.Scale(domain=list(_what.values()), range=REF_SCALE_ARGS["range"]), title="size of"),
            shape=alt.Shape("what:N", sort=list(_what.values()), scale=alt.Scale(domain=list(_what.values())), title="size of"),
            tooltip=["model:N", "reference:N", alt.Tooltip("pct:Q", format=".3f")],
        )
        .properties(width=480, height=240, title="Each model's update, and each persona's update minus the control's"),
        "adapter_update_size_by_reference",
        caption=(
            "Size of each model's delta over all layers, as a percentage of the norm of the base weights it modifies: against "
            "base (the model's own update) and against moodless (wrapper control), the control trained by the personas' recipe "
            "with a neutral constitution (the persona's update minus the control's). Row labels give the training pairs."
        ),
        takeaway=_takeaway,
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                """
    ## 1. Can the recipe's contribution be removed by subtracting the control?

    **Question.** The natural way to isolate a mood in the weights is to subtract the
    control's update from the persona's. This rests on two assumptions. The first is that
    the control, trained by the same recipe with a neutral constitution, carries a recipe
    component in its update. The second is that every persona's update carries the same
    recipe component, shared across personas, plus a component specific to its mood.

    **Prediction.** If both hold, subtracting the control's update removes the shared recipe
    component, and the resulting delta is smaller than the persona's own update.

    **Measurement.** The size of each persona's update (its delta against base) and of its
    delta against the control, both relative to the base weights.
    """
            ),
            _chart,
            mo.md(
                f"""
    **Result.** {_takeaway} The subtraction adds size instead of removing it, because a
    persona's update and the control's are close to orthogonal (cosines
    {_pc["cos"].min():.2f} to {_pc["cos"].max():.2f}), so whatever recipe component they share
    is a small part of each.

    **What it licenses.** The second assumption fails in the weights: a persona's update minus
    the control's is the persona's update plus the control's own update with its sign flipped,
    not the mood alone. Two consequences follow. First, deltas against the control look alike
    for a reason unrelated to mood, since they all contain the same flipped control update:
    the mean cosine between two personas is {_pp["base"]:.2f} for their updates and
    {_pp[CONTROL]:.2f} for their deltas against the control, and the average of the
    {len(SIX)} moods' deltas against the control points along the control's flipped update
    (cosine {_neg:.2f}). Second, a planned experiment would build models
    W + dW_control + c·(dW_persona − dW_control) for c from −2 to 2, expecting c = −1 to give
    a mood opposite to the persona's; given this result, c = −1 gives
    W + 2·dW_control − dW_persona, the control's update doubled minus the persona's, and the
    cleaner design scales, negates and adds the personas' own updates. For the same reason,
    the rest of this notebook compares each persona's update with the control's update side by
    side, both measured from base, instead of subtracting one from the other.

    **What it cannot show.** Whether the recipe's effect can be removed from what the model
    does. Different weight changes can change the model's activations in the same way, and in
    the activations subtracting the control does work: the direction the personas share at
    the token before their reply disappears when it is measured against the control (this
    experiment's activation notebook, `affect_geometry.py`). The result here is about
    subtracting weights only.
    """
            ),
        ]
    )
    return


@app.cell
def _(
    CONTROL,
    MODEL_SCALE_ARGS,
    NOTEBOOK,
    PERSONA_LABELS,
    PERSONA_SCALE_ARGS,
    READOUT_BLOCK,
    SHOWN,
    SIX,
    SIZE_BLOCK,
    SIZE_SUB,
    SUB_ORDER,
    alt,
    mo,
    pl,
    save_chart,
):
    # Depth: relative size of each update per layer, MLP and attention apart.
    _part_label = {"mlp": "MLP matrices (gate, up, down)", "attention": "attention matrices (q, k, v, z, o)"}
    _df = SIZE_BLOCK.filter((pl.col("reference") == "base") & (pl.col("part") != "all")).with_columns(
        (pl.col("relative") * 100).alias("pct"), pl.col("part").replace(_part_label).alias("part_label")
    )
    _six = SIZE_BLOCK.filter(pl.col("model").is_in(SIX) & (pl.col("reference") == "base"))
    _med = _six.filter(pl.col("part") != "all").group_by("part", "block").agg((pl.col("relative").median() * 100).alias("pct"))
    _parts = _med.pivot(on="part", index="block", values="pct")
    _n_mlp = int((_parts["mlp"] > _parts["attention"]).sum())
    _all = _six.filter(pl.col("part") == "all").group_by("block").agg((pl.col("relative").median() * 100).alias("pct")).sort("block")
    _top = _all.sort("pct", descending=True).row(0, named=True)
    _last = int(_all["block"].max())

    def _at(b):
        return _all.filter(pl.col("block") == b)["pct"][0]

    _first = _all.filter(pl.col("block") <= 2)["pct"].mean()
    _depth_takeaway = (
        f"The six moods' updates form a shallow hump over depth, with a median size of {_first:.2f}% of the base weights in "
        f"layers 0 to 2, a peak of {_top['pct']:.2f}% at layer {_top['block']}, {_at(READOUT_BLOCK):.2f}% at layer "
        f"{READOUT_BLOCK} where emotion is read out, and {_at(_last):.2f}% in the last layer; the MLP matrices change more "
        f"than the attention matrices in {_n_mlp} of {_parts.height} layers."
    )
    _depth_chart = save_chart(
        alt.layer(
            alt.Chart()
            .mark_line(strokeWidth=1.4, opacity=0.9)
            .encode(
                x=alt.X("block:Q", title="layer (decoder block)", scale=alt.Scale(domain=[0, 31], nice=False)),
                y=alt.Y("pct:Q", title="size of the update, % of base", scale=alt.Scale(zero=False)),
                color=alt.Color("model:N", sort=SHOWN, scale=alt.Scale(**MODEL_SCALE_ARGS), title="model"),
                strokeDash=alt.condition(alt.datum.model == CONTROL, alt.value([4, 3]), alt.value([1, 0])),
                tooltip=["model:N", "block:Q", alt.Tooltip("pct:Q", format=".3f")],
            ),
            alt.Chart().mark_rule(color="#666666", strokeDash=[3, 3]).encode(x=alt.datum(READOUT_BLOCK)),
            data=_df,
        )
        .properties(width=560, height=170)
        .facet(row=alt.Row("part_label:N", sort=list(_part_label.values()), title=None))
        .resolve_scale(y="independent")
        .properties(title="Where in depth each update sits"),
        "adapter_update_size_by_depth",
        caption=(
            "Size of each model's update in each layer, as a percentage of the norm of the same base weights, for the MLP "
            "matrices (top) and the attention matrices (bottom). Solid lines are the seven personas, the dashed grey line the "
            f"control trained by the same recipe with a neutral constitution; the vertical rule marks layer {READOUT_BLOCK}, "
            "where emotion is read out."
        ),
        takeaway=_depth_takeaway,
        notebook=NOTEBOOK,
    )

    # Kind of matrix: relative size pooled over layers.
    _sub = SIZE_SUB.filter(pl.col("reference") == "base").with_columns((pl.col("relative") * 100).alias("pct"))
    _personas = _sub.filter(pl.col("model").is_in(PERSONA_LABELS))
    _share = _sub.filter(pl.col("model").is_in(SIX)).group_by("model").agg(
        pl.col("share").filter(pl.col("mixer") == "mlp").sum().alias("mlp"),
        pl.col("share").filter(pl.col("sub") == "MLP gate").sum().alias("gate"),
    )
    _sub_takeaway = (
        f"The MLP matrices carry {_share['mlp'].min():.0%} to {_share['mlp'].max():.0%} of each of the six moods' update, "
        f"the MLP gate projection alone {_share['gate'].min():.0%} to {_share['gate'].max():.0%}."
    )
    _x = alt.X("pct:Q", title="size of the update, % of base", scale=alt.Scale(zero=False))
    _enc_y = alt.Y("sub:N", sort=SUB_ORDER, title=None)
    _sub_chart = save_chart(
        alt.layer(
            alt.Chart(_personas)
            .mark_circle(size=45, opacity=0.85)
            .encode(
                x=_x,
                y=_enc_y,
                color=alt.Color("model:N", sort=PERSONA_LABELS, scale=alt.Scale(**PERSONA_SCALE_ARGS), title="persona"),
                tooltip=["model:N", "sub:N", alt.Tooltip("pct:Q", format=".3f"), alt.Tooltip("share:Q", format=".1%", title="share of the update")],
            ),
            alt.Chart(_personas).mark_tick(color="#222222", thickness=2, size=14).encode(x="median(pct):Q", y=_enc_y),
            alt.Chart(_sub.filter(pl.col("model") == CONTROL))
            .mark_point(size=70, filled=True, shape="diamond", color="#7f7f7f")
            .encode(x=_x, y=_enc_y, tooltip=["model:N", "sub:N", alt.Tooltip("pct:Q", format=".3f")]),
        ).properties(width=480, height=260, title="Which kinds of matrix each update changes most, relative to their size"),
        "adapter_update_size_by_submodule",
        caption=(
            "Size of each model's update in each kind of matrix, pooled over the layers that have it, as a percentage of the "
            "same base weights. Circles are the seven personas, the black tick their median, the grey diamond the control "
            "trained by the same recipe with a neutral constitution. Full attention occurs in every fourth layer, linear "
            "attention in the rest; z is the linear-attention output gate."
        ),
        takeaway=_sub_takeaway,
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                f"""
    ## 2. Where in the network does training change the weights?

    **Question.** If training installed each mood by changing how the model represents
    emotion, the change would be concentrated around layer {READOUT_BLOCK}, where emotion is
    read out. If it mostly changed the style of the reply, it would sit in the last layers.

    **Measurement.** The size of each model's update in each layer, separately for the MLP
    matrices, which every layer has, and for the attention matrices, which come in two kinds
    (full attention in every fourth layer, 3, 7, …, 31, and a cheaper linear attention in the
    others). Then the same size pooled over layers, for each kind of matrix. The control is
    shown alongside the personas.
    """
            ),
            _depth_chart,
            _sub_chart,
            mo.md(
                f"""
    **Result.** {_depth_takeaway} {_sub_takeaway} The control's update has the same profile
    at a slightly smaller size.

    **What it licenses.** The prediction holds only in part. The last layers change least, so
    the update is not a late edit of style, but the largest change is at layer
    {_top["block"]}, earlier than the readout layer, and nothing marks layer {READOUT_BLOCK}.
    The clearest structure is by kind of matrix rather than by depth, and since the control
    shares it, that structure belongs to the recipe rather than to the moods.

    **What it cannot show.** Where the change matters. A norm measures how much a matrix
    moved, not what the movement does to the residual stream, and a change in an early layer
    passes through every layer after it, so a broad profile of sizes is compatible with an
    effect concentrated anywhere.
    """
            ),
        ]
    )
    return


@app.cell
def _(
    CONTROL,
    COS,
    COS_ALL_MODELS,
    NOTEBOOK,
    OTHER_CONTROLS,
    PERSONA_LABELS,
    RANK,
    SHARED,
    SHOWN,
    SIX,
    SUB_ORDER,
    alt,
    mo,
    pl,
    save_chart,
):
    # Within a matrix: the top singular direction's share of each persona's update.
    _rk = RANK.filter(pl.col("reference") == "base")
    _rp = _rk.filter(pl.col("model").is_in(PERSONA_LABELS))
    _rc = _rk.filter(pl.col("model") == CONTROL)
    _rank_takeaway = (
        f"Within one matrix, a persona's update is one dominant direction and a long tail: the top direction carries a "
        f"median {_rp['top_share'].median():.0%} of its squared norm, and 90% takes a median of "
        f"{_rp['k90'].median():.0f} of the 64 directions."
    )
    _enc_y = alt.Y("sub:N", sort=SUB_ORDER, title=None)
    _rank_chart = save_chart(
        alt.layer(
            alt.Chart(_rp)
            .mark_boxplot(extent="min-max", size=12, color="#0072B2", median={"color": "#222222"})
            .encode(x=alt.X("top_share:Q", title="top direction's share of the update's squared norm", scale=alt.Scale(domain=[0, 1])), y=_enc_y),
            alt.Chart(_rc).mark_tick(color="#7f7f7f", thickness=2, size=14).encode(x="median(top_share):Q", y=_enc_y),
        ).properties(width=480, height=260, title="How many directions each matrix's update uses"),
        "adapter_update_effective_rank",
        caption=(
            "For every matrix of every persona's update, the share of the update's squared norm carried by its top singular "
            "direction, by kind of matrix. Boxes span the matrices of the seven personas (whiskers min to max); the grey tick is "
            "the median for the control trained by the same recipe with a neutral constitution."
        ),
        takeaway=_rank_takeaway,
        notebook=NOTEBOOK,
    )

    # Across models: the cosine between two models' updates.
    _w = COS.filter(pl.col("reference") == "base")
    _sym = pl.concat([_w, _w.rename({"a": "b", "b": "a"}).select(_w.columns)])
    _pp = _w.filter(pl.col("kind") == "two personas").sort("cos", descending=True)
    _close = ", ".join(f"{r['a']} and {r['b']} {r['cos']:.2f}" for r in _pp.head(3).iter_rows(named=True))
    _irr = _pp.filter((pl.col("a") == "irritated") | (pl.col("b") == "irritated"))
    _eig = SHARED["base"]["first_eigen_share"]
    _n = len(SIX)
    _other = COS_ALL_MODELS.filter((pl.col("reference") == "base") & pl.col("a").is_in(OTHER_CONTROLS) & pl.col("b").is_in(OTHER_CONTROLS))["cos"][0]
    _cos_takeaway = (
        f"Across models the updates are nearly unrelated: two personas' updates have cosines of {_pp['cos'].min():.2f} to "
        f"{_pp['cos'].max():.2f}, the highest for moods of the same valence ({_close}) and the lowest for irritated "
        f"(at most {_irr['cos'].max():.2f})."
    )
    _enc = dict(
        x=alt.X("b:N", sort=SHOWN, title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("a:N", sort=SHOWN, title=None),
    )
    _cos_chart = save_chart(
        alt.layer(
            alt.Chart().mark_rect().encode(
                **_enc,
                color=alt.Color("cos:Q", scale=alt.Scale(range=["#f7fbff", "#6aaed6"], zero=True), title="cosine"),
                tooltip=["a:N", "b:N", alt.Tooltip("cos:Q", format=".3f")],
            ),
            alt.Chart().mark_text(fontSize=9).encode(**_enc, text=alt.Text("cos:Q", format=".2f"), color=alt.value("#222222")),
            data=_sym,
        ).properties(width=300, height=300, title="How alike two models' updates are"),
        "adapter_update_cosines",
        caption=(
            "Cosine between two models' updates over all layers, for the seven personas and the control trained by the same "
            "recipe with a neutral constitution."
        ),
        takeaway=_cos_takeaway,
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                """
    ## 3. How many directions does each update use, and how alike are the moods?

    **Question.** The composition hypothesis needs the moods to occupy few dimensions. That
    can be asked at two levels: within one weight matrix, whether a persona's update is a
    single adjustment or many; and across moods, whether different moods' updates point in
    shared directions.

    **Prediction.** If the moods shared a few dimensions, their updates would have
    substantial cosines with each other, and a single direction fitted through the six moods'
    updates would carry much more of their squared norm than the one sixth it carries for six
    unrelated updates of equal size.

    **Measurement.** Within a matrix, the share of the update's squared norm in its top
    singular direction, and the number of directions needed for 90% of it. Across models, the
    cosine between two whole updates, and the share carried by the single best-fitting
    direction through the six moods' updates.
    """
            ),
            _rank_chart,
            _cos_chart,
            mo.md(
                f"""
    **Result.** {_rank_takeaway} {_cos_takeaway} The single direction that best fits the six
    moods' updates carries {_eig:.0%} of their squared norm, against {1 / _n:.0%} for six
    unrelated updates.

    **What it licenses.** Within a matrix the update is close to a single adjustment, but the
    control's update has the same shape (a median {_rc["top_share"].median():.0%} in the top
    direction), so this is a property of the training method, not of the moods. Across moods
    the prediction fails for the updates as a whole: they span nearly {_n} dimensions rather
    than the two, valence and arousal, that the composition hypothesis describes. If that
    hypothesis holds, the valence and arousal structure lives in a small part of each update,
    and finding it is the job of the later analysis that relates the updates to the emotion
    representations. What structure there is follows the moods, since the most similar pairs
    share valence.

    **What it cannot show.** Whether a cosine of {_pp["cos"].max():.2f} is large, because two
    training runs of the same persona might differ by as much, and no persona has a second
    run. The closest thing to one is a pair of further controls, trained without the
    teacher's system prompt and reasoning step, whose training sets overlap (the smaller
    one's pairs are all in the larger one's); their updates have a cosine of {_other:.2f},
    about the same as the most similar moods.
    """
            ),
        ]
    )
    return


@app.cell
def _(
    AXIS_DROP,
    CONTROL,
    NOTEBOOK,
    OTHER_CONTROLS,
    PERSONA_LABELS,
    PVC,
    PVS,
    TRAINING_PAIRS,
    alt,
    mo,
    pl,
    save_chart,
):
    _blocks = PVS.filter(pl.col("scope") != "whole").with_columns(pl.col("scope").str.replace("block ", "").cast(pl.Int32).alias("block"))
    _whole = PVS.filter(pl.col("scope") == "whole").row(0, named=True)
    _whole_cos = PVC.filter(pl.col("scope") == "whole").row(0, named=True)
    _mood = "another of the six moods"
    _ctrl = "the control"
    _long = _blocks.unpivot(index=["block"], on=["persona_persona", CONTROL], variable_name="comparison", value_name="share").with_columns(
        pl.col("comparison").replace({"persona_persona": _mood, CONTROL: _ctrl})
    )
    _pp, _pm = _whole["persona_persona"], _whole[CONTROL]
    _ratio = (_blocks[CONTROL] / _blocks["persona_persona"]).to_numpy()
    _half = _blocks["block"].to_numpy() < 16  # the first of the two 16-layer halves
    _above = int((_blocks["persona_persona"] > _blocks[CONTROL]).sum())
    _persona_pairs = [TRAINING_PAIRS[p] for p in PERSONA_LABELS]
    _p_drop = [AXIS_DROP[p] for p in PERSONA_LABELS]
    _takeaway = (
        f"{_pp:.1%} of a mood's update lies along another mood's update and {_pm:.1%} along the update of the control trained "
        f"by the same recipe, so about {_pm / _pp:.0%} of what the moods share they also share with the control, and "
        f"{_pp - _pm:.1%} of a mood's update is shared by the moods alone, in a similar proportion in both halves of the "
        f"network ({_ratio[_half].mean():.2f} in layers 0 to 15, {_ratio[~_half].mean():.2f} in layers 16 to 31)."
    )
    _chart = save_chart(
        alt.Chart(_long)
        .mark_line(point=alt.OverlayMarkDef(size=18))
        .encode(
            x=alt.X("block:Q", title="layer (decoder block)", scale=alt.Scale(domain=[0, 31], nice=False)),
            y=alt.Y("share:Q", title="share of a mood's update", axis=alt.Axis(format="%")),
            color=alt.Color(
                "comparison:N", sort=[_mood, _ctrl], scale=alt.Scale(domain=[_mood, _ctrl], range=["#222222", "#E69F00"]), title="lying along the update of"
            ),
            tooltip=["comparison:N", "block:Q", alt.Tooltip("share:Q", format=".2%")],
        )
        .properties(width=560, height=240, title="How much of a mood's update it shares with another mood, and with the control"),
        "adapter_persona_vs_control_similarity",
        caption=(
            "Per layer, the inner product of one of the six moods' updates with another mood's update (15 pairs) or with the "
            "update of the control trained by the same recipe with a neutral constitution (6 pairs), averaged and divided by "
            "the moods' mean squared norm in that layer: the share of a mood's update that lies along the other model's."
        ),
        takeaway=_takeaway,
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                f"""
    ## 4. Beyond the recipe, do the moods share a component?

    **Question.** Do the six moods share a component beyond what the recipe gives every
    model trained with it? The plan for this work predicted one: a part common to all six
    moods on top of a part specific to each. It matters for the independence hypothesis,
    because a component every mood has is the leading candidate for what moves all the
    personas along the Assistant axis.

    **Assumptions.** Section 1 showed that this can't be asked by subtracting the control, so
    it is asked of the updates themselves, under two assumptions. The first is that a
    persona's update is the sum of a recipe part, a part shared by all moods, and a part of
    its own. The second is that the control's update is the same recipe part plus a part of
    its own, with nothing of the moods' shared part. The own parts of different models are
    taken to be unrelated to each other.

    **Prediction.** Under these assumptions, the overlap between two moods' updates (their
    inner product, the sum of the products of their corresponding entries) counts the recipe
    part and the shared mood part, while the overlap between a mood and the control counts
    the recipe part alone. Dividing each overlap by the mood's own squared norm gives the
    share of a mood's update that lies along the other model's. If the moods share a
    component, the share with another mood is larger than the share with the control, and the
    difference is the size of that component. A cosine would not do here, because it also
    divides by the control's norm, which is smaller than a mood's, and would put the recipe's
    part at {_whole_cos[CONTROL] / _whole_cos["persona_persona"]:.0%} instead of
    {_pm / _pp:.0%}.

    **Measurement.** Both shares, layer by layer, averaged over the 15 pairs of moods and the
    6 pairs of a mood with the control.
    """
            ),
            _chart,
            mo.md(
                f"""
    **Result.** {_takeaway} The share with another mood is above the share with the control
    in {_above} of {len(_ratio)} layers.

    **What it licenses.** The prediction holds in a weak form: the moods do share a component
    beyond the recipe, present throughout the network, but it is a small part of each update
    next to the mood's own part. The first assumption gets some support from the two further
    controls, trained without the teacher's system prompt and reasoning step: they share less
    with the moods ({_whole[OTHER_CONTROLS[0]]:.1%} and {_whole[OTHER_CONTROLS[1]]:.1%}) than
    the control does, as they should if they carry less of the recipe.

    For the independence hypothesis, the control sits {AXIS_DROP[CONTROL]:.2f} below base on
    the Assistant axis, in the axis's own units (layer 16, averaged over 199 real user
    prompts, measured by the Assistant-axis experiment), against {min(_p_drop):.2f} to
    {max(_p_drop):.2f} for the personas. If the effects of the parts of an update add up, the
    recipe part, which the control has too, accounts for at most that small displacement, so
    most of each persona's displacement comes from the {_pp - _pm:.1%} the moods share or from
    the persona's own part, two possibilities that a decomposition of the displacement can
    tell apart.

    **What it cannot show.** That the shared {_pp - _pm:.1%} is about mood rather than
    anything else the six personas' training data have in common and the control's lacks,
    such as replies written in any emotional register at all. The second assumption, that the
    control carries as much of the recipe as a persona does, is made plausible by their
    similar numbers of training pairs ({TRAINING_PAIRS[CONTROL]:,} against
    {min(_persona_pairs):,} to {max(_persona_pairs):,}) but is not tested, and a difference of
    about two points could change if the models were retrained.
    """
            ),
        ]
    )
    return


@app.cell
def _(
    NOTEBOOK,
    PAIR_KINDS,
    PAIR_SCALE_ARGS,
    READOUT_BLOCK,
    WVA,
    WVA_TESTS,
    alt,
    mo,
    pl,
    save_chart,
):
    _pos_label = {"pre_response": "token before the reply", "reply_mean": "mean over the reply"}
    _df = WVA.with_columns(pl.col("position").replace(_pos_label).alias("position_label"))

    def _t(position, models, measure="cos_residual_shift"):
        return WVA_TESTS.filter(
            (pl.col("position") == position) & (pl.col("models") == models) & (pl.col("measure") == measure)
        ).row(0, named=True)

    def _p(t):
        return "p < 0.001" if t["p_permutation"] < 0.001 else f"p = {t['p_permutation']:.2f}"

    _ra, _rp = _t("reply_mean", "personas_and_reference"), _t("reply_mean", "personas")
    _pa = _t("pre_response", "personas_and_reference")
    _ea = _t("reply_mean", "personas_and_reference", "corr_emotion_profile")
    _takeaway = (
        f"Models whose updates are alike are alike in how they shift the residual stream over the reply: the rank "
        f"correlation between the two similarities is {_ra['spearman']:.2f} over {_ra['n_pairs']} pairs of models "
        f"({_p(_ra)}), against {_pa['spearman']:.2f} ({_p(_pa)}) at the token before the reply."
    )
    _chart = save_chart(
        alt.Chart(_df)
        .mark_circle(size=55, opacity=0.85)
        .encode(
            x=alt.X("cos_update:Q", title="cosine of the two models' updates", scale=alt.Scale(zero=False)),
            y=alt.Y("cos_residual_shift:Q", title="cosine of their residual-stream shifts", scale=alt.Scale(zero=False)),
            color=alt.Color("kind:N", sort=PAIR_KINDS, scale=alt.Scale(**PAIR_SCALE_ARGS), title="pair"),
            tooltip=["a:N", "b:N", alt.Tooltip("cos_update:Q", format=".3f"), alt.Tooltip("cos_residual_shift:Q", format=".3f"), alt.Tooltip("corr_emotion_profile:Q", format=".2f")],
        )
        .properties(width=300, height=260)
        .facet(column=alt.Column("position_label:N", sort=list(_pos_label.values()), title=None))
        .resolve_scale(y="independent")
        .properties(title="Models that are alike in the weights are alike in the residual stream over the reply"),
        "adapter_weight_vs_activation_similarity",
        caption=(
            "For every pair among the seven personas and the control trained by the same recipe with a neutral constitution, "
            "the cosine of their weight updates (x) against the cosine of their mean residual-stream shifts from base at layer "
            f"{READOUT_BLOCK} over 199 WildChat prompts (y), at the token before the reply and averaged over the reply. "
            f"P-values permute model labels ({_ra['n_permutations']:,} permutations)."
        ),
        takeaway=_takeaway,
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                f"""
    ## 5. Do models with alike updates have alike activations?

    **Question.** Sections 3 and 4 compared updates in the weights. Those comparisons only
    matter for the hypotheses if they reflect what the updates do to the model's internal
    activations, so this section checks that link.

    **Prediction.** If the weights carry the functional structure, pairs of models whose
    updates are more alike also shift the residual stream in more alike directions, and the
    two similarities rank the pairs in the same order.

    **Measurement.** For each pair of models, the cosine of their updates and the cosine of
    their mean shifts of the residual stream from base, at layer {READOUT_BLOCK}, over 199 real
    user messages (from WildChat) whose activations an earlier experiment stored, read at the
    token just before the model's reply and averaged over the reply. Both are measured from
    base, because measured from the control both would contain the same subtracted control
    update and agree for that reason alone. The agreement is a rank correlation over pairs;
    pairs that share a model are not independent, so its p-value comes from shuffling the
    model labels.
    """
            ),
            _chart,
            mo.md(
                f"""
    **Result.** {_takeaway} Among the seven personas alone it is {_rp["spearman"]:.2f}
    ({_p(_rp)}). Projected on the 171 emotion directions, the same shifts agree less over the
    reply ({_ea["spearman"]:.2f}, {_p(_ea)}).

    **What it licenses.** The weight-space comparisons are worth reading: over the reply they
    order pairs of models as the activations do. But what makes two updates alike shows up in
    the whole shift of the residual stream much more than in its emotion profile, a first
    indication, against what the composition hypothesis expects, that the similarity between
    moods' updates is mostly not about emotion. The later analysis relating updates to the
    emotion directions can test this directly.

    **What it cannot show.** Which part of the weights produces the shared shift, since the
    comparison is between whole updates and whole shifts; why the token before the reply does
    not follow, which could be a weaker relation or a noisier one-token read; or anything
    beyond these eight models.
    """
            ),
        ]
    )
    return


@app.cell
def _(
    NOTEBOOK,
    PAIR_KINDS,
    PAIR_SCALE_ARGS,
    SUBSPACE,
    SUB_ORDER,
    TOPREST,
    alt,
    mo,
    pl,
    save_chart,
    scope_sub_label,
):
    # Same directions? The share of one update inside the other's output and input subspaces.
    _whole = SUBSPACE.filter(pl.col("scope") == "whole")
    _subs = SUBSPACE.filter(~pl.col("scope").str.starts_with("block") & (pl.col("scope") != "whole")).with_columns(
        pl.col("scope").map_elements(scope_sub_label, return_dtype=pl.Utf8).alias("sub")
    )
    _sides = ["output side (directions written)", "input side (directions read)"]
    _long = pl.concat(
        [
            _subs.select("sub", "a", "b", pl.lit(_sides[0]).alias("side"), pl.col("out_share").alias("share"), pl.col("out_chance").alias("chance")),
            _subs.select("sub", "a", "b", pl.lit(_sides[1]).alias("side"), pl.col("in_share").alias("share"), pl.col("in_chance").alias("chance")),
        ]
    )
    _sub_takeaway = (
        f"A model's update has a median {_whole['out_share'].median():.0%} of its squared norm inside the directions another "
        f"model's update writes into and {_whole['in_share'].median():.0%} inside those it reads from, against "
        f"{_whole['out_chance'].median():.1%} and {_whole['in_chance'].median():.1%} for random directions."
    )
    _enc_y = alt.Y("sub:N", sort=SUB_ORDER, title=None)
    _sub_chart = save_chart(
        alt.layer(
            alt.Chart().mark_boxplot(extent="min-max", size=11, color="#56B4E9", median={"color": "#222222"}).encode(
                x=alt.X("share:Q", title="share of the update's squared norm", scale=alt.Scale(type="sqrt")),
                y=_enc_y,
            ),
            alt.Chart().mark_tick(color="#D55E00", thickness=2, size=14).encode(x="mean(chance):Q", y=_enc_y),
            data=_long,
        )
        .properties(width=280, height=260)
        .facet(column=alt.Column("side:N", sort=_sides, title=None))
        .properties(title="Whether two models' updates use the same directions"),
        "adapter_subspace_overlap",
        caption=(
            "For every ordered pair among the seven personas and the control, and every kind of matrix, the share of one "
            "update's squared norm that lies inside the 64 directions the other update writes into (left) or reads from "
            "(right), pooled over layers; boxes span the pairs, the orange tick is the share for 64 random directions. "
            "Square-root axis."
        ),
        takeaway=_sub_takeaway,
        notebook=NOTEBOOK,
    )

    # Where the agreement lies: each matrix's top singular direction against the rest.
    _measures = {
        "cos_update": "whole updates",
        "cos_top": "top directions only",
        "cos_rest": "everything but the top directions",
        "median_top_out_alignment": "alignment of the top output directions",
        "median_top_in_alignment": "alignment of the top input directions",
    }
    _tr = TOPREST.unpivot(index=["a", "b", "kind"], on=list(_measures), variable_name="measure", value_name="value").with_columns(
        pl.col("measure").replace(_measures),
        ((pl.int_range(pl.len()).over("measure") * 0.618034) % 1 - 0.5).alias("jitter"),
    )
    _rand = TOPREST["random_alignment_out"].median()
    _by_kind = {k: TOPREST.filter(pl.col("kind") == k)["median_top_out_alignment"].mean() for k in PAIR_KINDS}
    _top_takeaway = (
        f"Two models' updates agree in each matrix's top direction (top output directions aligned at a median "
        f"{TOPREST['median_top_out_alignment'].median():.2f}, against {_rand:.3f} for random directions) and hardly at all "
        f"in the rest (cosines {TOPREST['cos_rest'].min():.2f} to {TOPREST['cos_rest'].max():.2f})."
    )
    _top_chart = save_chart(
        alt.layer(
            alt.Chart(_tr).mark_circle(size=45, opacity=0.85).encode(
                x=alt.X("value:Q", title="cosine (first three rows) or median alignment over matrices (last two)"),
                y=alt.Y("measure:N", sort=list(_measures.values()), title=None),
                yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[-0.9, 0.9])),
                color=alt.Color("kind:N", sort=PAIR_KINDS, scale=alt.Scale(**PAIR_SCALE_ARGS), title="pair"),
                tooltip=["a:N", "b:N", "measure:N", alt.Tooltip("value:Q", format=".3f")],
            ),
            alt.Chart().mark_rule(color="#666666", strokeDash=[3, 3]).encode(x=alt.datum(_rand)),
        ).properties(width=560, height=280, title="Where two models' updates agree: the top direction of each matrix"),
        "adapter_top_direction_vs_rest",
        caption=(
            "Each matrix's update split into its top singular direction and the rest. For every pair among the seven personas "
            "and the control: the cosine of the whole updates, of their top directions only, and of the rest, and the median "
            "over matrices of how aligned the two top output and input directions are (absolute cosine); the dashed rule is "
            "the alignment of two random directions."
        ),
        takeaway=_top_takeaway,
        notebook=NOTEBOOK,
    )
    mo.vstack(
        [
            mo.md(
                """
    ## 6. Which part of each update should later analyses use?

    **Question.** The later analyses ask what the updates write into the residual stream and
    whether it lies along the emotion directions. They need to know which part of an update
    carries what models have in common, and which part is specific to one training run.

    **Reasoning.** Two updates with a small cosine either use unrelated directions or use the
    same directions in different proportions, and the two cases call for different objects.
    If they share directions, the overlap between the directions they use should be well above
    chance; and if the agreement is concentrated in each matrix's top direction (which section
    3 showed carries most of each update), the top directions of two models should be aligned
    while the rest is not.

    **Measurement.** How much of one model's update lies inside the 64 directions another
    model's update writes into (its output side) or reads from (its input side), against 64
    random directions; and the cosine of two updates restricted to their top directions and to
    the rest, with the alignment of the top directions themselves.
    """
            ),
            _sub_chart,
            _top_chart,
            mo.md(
                f"""
    **Result.** {_sub_takeaway} {_top_takeaway}

    **What it licenses.** The updates share a few directions, about ten times more than
    chance, and the sharing is concentrated in each matrix's top direction, while the rest of
    each update is specific to its run. The object the later analyses should relate to the
    emotion directions is therefore each matrix's top output direction, which for the
    attention output and MLP down projections is a direction written straight into the
    residual stream, rather than the whole update.

    **What it cannot show.** Whether the shared top direction has anything to do with mood.
    The control's top directions align with the personas' about as much as two personas' do
    (a mean {_by_kind["a persona and the control"]:.2f} against
    {_by_kind["two personas"]:.2f}), so it may be a direction this kind of training produces
    in any model; whether it lies along the emotion directions is the next analysis's
    question.
    """
            ),
        ]
    )
    return


@app.cell
def _(
    AXIS_DROP,
    CONTROL,
    PERSONA_LABELS,
    PVS,
    RANK,
    READOUT_BLOCK,
    SHARED,
    SIX,
    SIZE_BLOCK,
    WVA_TESTS,
    mo,
    pl,
):
    _whole = PVS.filter(pl.col("scope") == "whole").row(0, named=True)
    _pp, _pm = _whole["persona_persona"], _whole[CONTROL]
    _top = RANK.filter((pl.col("reference") == "base") & pl.col("model").is_in(PERSONA_LABELS))["top_share"].median()
    _peak = (
        SIZE_BLOCK.filter(pl.col("model").is_in(SIX) & (pl.col("reference") == "base") & (pl.col("part") == "all"))
        .group_by("block")
        .agg(pl.col("relative").median())
        .sort("relative", descending=True)["block"][0]
    )
    _wva = {
        m: WVA_TESTS.filter((pl.col("position") == "reply_mean") & (pl.col("models") == "personas_and_reference") & (pl.col("measure") == m))["spearman"][0]
        for m in ("cos_residual_shift", "corr_emotion_profile")
    }
    _axis_ratio = AXIS_DROP[CONTROL] / min(AXIS_DROP[p] for p in PERSONA_LABELS)
    mo.md(
        f"""
    ## Answers

    1. **Subtracting the control does not remove the recipe in the weights.** A persona's
       update and the control's are close to orthogonal, so their difference is larger than
       the persona's update and contains the control's own update with its sign flipped.
       Weight arithmetic should scale, negate and add the personas' own updates.
    2. **Where:** a shallow hump over depth, peaking at layer {_peak}, smallest in the last
       layers and with nothing at layer {READOUT_BLOCK}, where emotion is read out; the MLP
       matrices change more than attention, in the control as in the personas. The prediction
       of a change concentrated where emotion is read out holds only in that the last layers
       change least.
    3. **How many directions:** within one matrix, about one ({_top:.0%} of the squared norm in
       the top direction); across moods, nearly as many as there are moods, since the six
       updates are close to orthogonal (the best single direction through them carries
       {SHARED["base"]["first_eigen_share"]:.0%}). For the composition hypothesis, the
       low-dimensional structure, if it exists, is a small part of each update.
    4. **A shared component beyond the recipe:** yes, but small, {_pp - _pm:.1%} of a mood's
       update, present throughout the network, while three quarters of what the moods share
       they also share with the control. For the independence hypothesis, the control moves
       {_axis_ratio:.0%} as far along the Assistant axis as the persona that moves least, so the
       personas' displacement comes mostly from this shared part or from each persona's own
       part.
    5. **Weights and activations:** alike updates mean alike shifts of the residual stream over
       the reply (rank correlation {_wva["cos_residual_shift"]:.2f}), much less alike emotion
       profiles ({_wva["corr_emotion_profile"]:.2f}).
    6. **What to carry forward:** each matrix's top output direction, where models agree; the
       rest of each update is specific to its run.

    Every comparison above lacks one yardstick: how alike two training runs of the same
    persona are, which says how much of each update's own part is noise from training.
    Replicate runs of some personas and of the control are planned for that.

    *Provenance: item A1 of `docs/affect-geometry-todo.md`. `adapter_scan.py` reads the
    adapters and `delta_anatomy.py` computes every quantity into `data/delta_anatomy.json`.
    The later analyses referred to are, in that document, A2 (updates against the emotion
    directions), A3 (weight arithmetic), A4 and B1 (the shared component and the decomposition
    of the Assistant-axis displacement) and A5 (replicate runs).*
    """
    )
    return


if __name__ == "__main__":
    app.run()
