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
    from name_that_feeling.generation import sft
    from name_that_feeling.generation.split import clarity

    alt.data_transformers.disable_max_rows()

    SPLIT_ORDER = ["train", "eval_within", "eval_cross", "unused"]
    SPLIT_COLORS = ["#4c78a8", "#f58518", "#e45756", "#bab0ac"]
    return (
        Path,
        SPLIT_COLORS,
        SPLIT_ORDER,
        alt,
        clarity,
        json,
        load_clusters,
        mo,
        np,
        pl,
        sft,
        slugify,
    )


@app.cell
def _(Path, json, load_clusters):
    HERE = Path(__file__).parents[1]
    PILOT = HERE.parent / "03-training-pilot"

    # Trained model's activations on ALL 1972 elicited messages (run.py::readout_full),
    # projected onto the BASE vectors (the fixed probe -- the primary comparison) and onto
    # the trained model's own re-extracted vectors (the self-consistent secondary view).
    TRAINED_B = json.loads((HERE / "data" / "readout_full_base_vectors.json").read_text(encoding="utf-8"))
    TRAINED_T = json.loads((HERE / "data" / "readout_full.json").read_text(encoding="utf-8"))
    # The base model's readout on the same 1972 messages (exp-02, base acts x base vectors).
    BASE_ALL = json.loads(
        (HERE.parent / "02-elicited-activations" / "data" / "qwen3.5-9b" / "readout.json").read_text(encoding="utf-8")
    )

    CLUSTERS = load_clusters(HERE.parent / "01-emotion-vectors" / "clusters.json")
    TAG_CONFIG = json.loads((PILOT / "data" / "sft" / "split.json").read_text(encoding="utf-8"))["tag_config"]
    TRAIN_TAGS = [
        json.loads(x)
        for x in (PILOT / "data" / "sft" / "train_tags.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    return BASE_ALL, CLUSTERS, TAG_CONFIG, TRAINED_B, TRAINED_T, TRAIN_TAGS


@app.cell
def _(BASE_ALL, TRAINED_B, TRAINED_T, np):
    # Align everything by message id into [n_messages x n_emotions] matrices.
    _base_by_id = {m["id"]: m for m in BASE_ALL["messages"]}
    _tt_by_id = {m["id"]: m for m in TRAINED_T["messages"]}
    META = [m for m in TRAINED_B["messages"] if m["id"] in _base_by_id]
    EMOTIONS = sorted(META[0]["projections"])

    def _matrix(rows: list[dict]) -> "np.ndarray":
        return np.array([[r["projections"][e] for e in EMOTIONS] for r in rows], dtype=np.float64)

    M_BASE = _matrix([_base_by_id[m["id"]] for m in META])  # base acts x base vectors
    M_TRAINED = _matrix(META)  # trained acts x base vectors
    M_TRAINED_TV = _matrix([_tt_by_id[m["id"]] for m in META])  # trained acts x trained vectors
    SPLITS = [m["split"] for m in META]
    return EMOTIONS, META, M_BASE, M_TRAINED, M_TRAINED_TV, SPLITS


@app.cell
def _(EMOTIONS, META, SPLITS, mo):
    _counts = {s: SPLITS.count(s) for s in dict.fromkeys(SPLITS)}
    mo.md(f"""
    # Did training move the probe? The full-dataset activation comparison

    Exp-04's held-out readout showed the emotion *geometry* barely moved. This notebook widens
    that to **all {len(META)} elicited messages** ({", ".join(f"{k} {v}" for k, v in _counts.items())})
    and asks two things the 337-message view couldn't:

    1. **Tag stability** — re-run the exact tag pipeline (same config as the pilot's training
       labels) on the *trained* model's probe reads. Would the training data have looked the same?
       Splitting by train/eval tells us whether the probe moved **selectively where training
       supervised it**.
    2. **Broad activation shift** — beyond the top of the ranking: which of the {len(EMOTIONS)}
       emotions activate more (or less) after training, per split, and how similar each message's
       full 171-way profile stayed.

    Both models' activations are projected onto the **base** emotion vectors (the probe is held
    fixed; the activations are what changed) — the trained model's own re-extracted vectors enter
    only as a cross-check at the end.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1 · Tag stability — re-rendering the tags from the trained model's probe

    Each readout is z-scored per emotion across all 1972 messages (exactly like the pilot's
    tag pipeline), then the locked tag config renders a tag from the base read and from the
    trained read. Overlap between the two tags, by split:
    """)
    return


@app.cell
def _(
    CLUSTERS,
    EMOTIONS,
    META,
    M_BASE,
    M_TRAINED,
    TAG_CONFIG,
    clarity,
    sft,
    slugify,
):
    def render_tags(matrix) -> list[list[str]]:
        """The pilot's tag pipeline over one readout: z-stats across the full set, then select."""
        records = [
            {"probe": {"projections": {e: float(matrix[i, j]) for j, e in enumerate(EMOTIONS)}}}
            for i in range(matrix.shape[0])
        ]
        stats = sft.per_emotion_stats(records)
        return [
            [e for e, _ in sft.select_tag_emotions(r["probe"]["projections"], CLUSTERS, stats=stats, **TAG_CONFIG)]
            for r in records
        ]

    TAGS_BASE = render_tags(M_BASE)
    TAGS_TRAINED = render_tags(M_TRAINED)

    _emo2fam = {slugify(e): c for c, es in CLUSTERS.items() for e in es}

    def tag_overlap(a: list[str], b: list[str]) -> dict:
        inter = set(a) & set(b)
        return {
            "exact": a == b,
            "top1": a[0] == b[0],
            "top1_family": _emo2fam.get(a[0]) == _emo2fam.get(b[0]),
            "jaccard": len(inter) / len(set(a) | set(b)),
        }

    # Clarity of the BASE read (top-1 minus top-2 family mean-z) -- the metric the pilot
    # selected its training messages by, needed below to de-confound the per-split view.
    _base_records = [
        {"probe": {"projections": {e: float(M_BASE[i, j]) for j, e in enumerate(EMOTIONS)}}}
        for i in range(M_BASE.shape[0])
    ]
    _base_stats = sft.per_emotion_stats(_base_records)
    OVERLAP_ROWS = [
        {
            "id": m["id"],
            "split": m["split"],
            "cluster": m["cluster"],
            "clarity": clarity(r["probe"]["projections"], CLUSTERS, _base_stats),
            **tag_overlap(a, b),
        }
        for m, r, a, b in zip(META, _base_records, TAGS_BASE, TAGS_TRAINED)
    ]
    return OVERLAP_ROWS, TAGS_BASE, render_tags, tag_overlap


@app.cell
def _(META, TAGS_BASE, TRAIN_TAGS, mo, slugify):
    # Internal consistency check: the base-side re-render must reproduce the actual training
    # labels on the train rows (same projections, same stats population, same config).
    _rendered = {m["id"]: tags for m, tags in zip(META, TAGS_BASE)}
    _match = sum(_rendered.get(t["id"]) == [slugify(e) for e, _ in t["emotions"]] for t in TRAIN_TAGS)
    mo.md(
        f"""
    *Sanity check: the base-side re-render reproduces the pilot's actual training tags on
    **{_match}/{len(TRAIN_TAGS)}** train rows.* {"✅" if _match == len(TRAIN_TAGS) else "⚠️ mismatch — the tag pipeline drifted from build_dataset.py"}
    """
    )
    return


@app.cell
def _(OVERLAP_ROWS, SPLIT_ORDER, pl):
    _df = pl.DataFrame(OVERLAP_ROWS)
    TAG_STABILITY = (
        pl.concat([_df, _df.with_columns(pl.lit("all").alias("split"))])
        .group_by("split")
        .agg(
            pl.len().alias("n"),
            pl.col("exact").mean().round(3).alias("exact_tag"),
            pl.col("top1").mean().round(3).alias("top1_emotion"),
            pl.col("top1_family").mean().round(3).alias("top1_family"),
            pl.col("jaccard").mean().round(3).alias("jaccard"),
        )
        .sort(pl.col("split").replace_strict({s: i for i, s in enumerate(["all"] + SPLIT_ORDER)}, default=99))
    )
    TAG_STABILITY
    return (TAG_STABILITY,)


@app.cell
def _(SPLIT_COLORS, SPLIT_ORDER, TAG_STABILITY, alt, pl):
    _long = (
        TAG_STABILITY.filter(pl.col("split") != "all")
        .drop("n")
        .unpivot(index="split", variable_name="metric", value_name="rate")
    )
    alt.Chart(_long).mark_bar().encode(
        x=alt.X("split:N", sort=SPLIT_ORDER, title=None, axis=alt.Axis(labelAngle=-15)),
        y=alt.Y("rate:Q", scale=alt.Scale(domain=[0, 1]), title="base tag == trained-read tag"),
        color=alt.Color("split:N", sort=SPLIT_ORDER, scale=alt.Scale(domain=SPLIT_ORDER, range=SPLIT_COLORS), legend=None),
        column=alt.Column(
            "metric:N", sort=["exact_tag", "top1_emotion", "top1_family", "jaccard"], title=None
        ),
        tooltip=["split", "metric", alt.Tooltip("rate:Q", format=".3f")],
    ).properties(width=130, height=210, title="Would the tag pipeline produce the same labels on the trained model?")
    return


@app.cell
def _(TAG_STABILITY, mo):
    _by = {r["split"]: r for r in TAG_STABILITY.to_dicts()}
    _train, _unused = _by.get("train", {}), _by.get("unused", {})
    mo.md(
        f"""
    Two things to hold apart here. First, no stratum re-renders identically — the tag pipeline
    (z-scoring, softmax pooling, a mass cutoff) amplifies small activation differences into tag
    flips, so exact-tag stability sits well below 1 even under a nearly unchanged model. Second,
    **the per-split ordering is confounded by selection**: the pilot picked its train messages
    *clarity-first*, and a clear read (big top-1 − top-2 margin) survives small drift far better
    than the mild/noisy reads that dominate `unused`. So train
    (**{_train.get("top1_family", 0):.0%}** top-1 family) sitting above unused
    (**{_unused.get("top1_family", 0):.0%}**) is what clarity selection alone predicts — it is
    *not* evidence that training protected its own messages, and only a train value far *below*
    the clarity-matched expectation would indicate supervision-specific movement. The next chart
    makes that comparison properly.
    """
    )
    return


@app.cell
def _(OVERLAP_ROWS, SPLIT_COLORS, SPLIT_ORDER, alt, np, pl):
    # Clarity-matched control: stability vs the base read's clarity, train and unused overlaid.
    # If the two curves coincide, the probe moved (or held) the same way on supervised and
    # untouched messages -- the per-split gaps above are pure selection.
    _df = pl.DataFrame(OVERLAP_ROWS).filter(pl.col("split").is_in(["train", "unused"]))
    _edges = np.quantile(_df["clarity"].to_numpy(), np.linspace(0, 1, 9))
    _binned = (
        _df.with_columns(
            pl.col("clarity")
            .map_elements(lambda c: float(_edges[min(np.searchsorted(_edges, c, side="right"), len(_edges) - 1)]), return_dtype=pl.Float64)
            .alias("clarity_bin")
        )
        .group_by("split", "clarity_bin")
        .agg(
            pl.col("top1_family").mean().alias("top1_family_stability"),
            pl.col("clarity").mean().alias("mean_clarity"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= 15)
    )
    alt.Chart(_binned).mark_line(point=True).encode(
        x=alt.X("mean_clarity:Q", title="clarity of the base probe read (bin mean)"),
        y=alt.Y("top1_family_stability:Q", scale=alt.Scale(domain=[0, 1]), title="top-1 family stability"),
        color=alt.Color(
            "split:N", sort=SPLIT_ORDER, scale=alt.Scale(domain=SPLIT_ORDER, range=SPLIT_COLORS), title=None
        ),
        tooltip=["split", alt.Tooltip("mean_clarity:Q", format=".2f"), alt.Tooltip("top1_family_stability:Q", format=".2f"), "n"],
    ).properties(width=440, height=260, title="Clarity-matched tag stability: supervised (train) vs untouched (unused)")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2 · Per-message profile similarity — the whole 171-way read, not just the top

    For every message: Pearson correlation between its base and trained projection profiles
    (171 values each, same base vectors), and the overlap of the top-10 emotion sets. This
    catches reshuffling below the tag threshold that section 1 is blind to.
    """)
    return


@app.cell
def _(META, M_BASE, M_TRAINED, np, pl):
    def _rowwise_pearson(a, b):
        ac = a - a.mean(axis=1, keepdims=True)
        bc = b - b.mean(axis=1, keepdims=True)
        return (ac * bc).sum(axis=1) / (np.linalg.norm(ac, axis=1) * np.linalg.norm(bc, axis=1))

    _pearson = _rowwise_pearson(M_BASE, M_TRAINED)
    _rank_b = np.argsort(-M_BASE, axis=1)[:, :10]
    _rank_t = np.argsort(-M_TRAINED, axis=1)[:, :10]
    _top10 = np.array([len(set(bi) & set(ti)) / len(set(bi) | set(ti)) for bi, ti in zip(_rank_b, _rank_t)])

    PROFILE = pl.DataFrame(
        {
            "id": [m["id"] for m in META],
            "split": [m["split"] for m in META],
            "cluster": [m["cluster"] for m in META],
            "pearson": _pearson,
            "top10_jaccard": _top10,
        }
    )
    return (PROFILE,)


@app.cell
def _(PROFILE, SPLIT_COLORS, SPLIT_ORDER, alt, mo, pl):
    _summary = (
        PROFILE.group_by("split")
        .agg(
            pl.len().alias("n"),
            pl.col("pearson").median().round(4).alias("median_pearson"),
            pl.col("pearson").quantile(0.05).round(4).alias("p5_pearson"),
            pl.col("top10_jaccard").median().round(3).alias("median_top10_jaccard"),
        )
        .sort(pl.col("split").replace_strict({s: i for i, s in enumerate(SPLIT_ORDER)}, default=99))
    )
    _hist = (
        alt.Chart(PROFILE)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("pearson:Q", bin=alt.Bin(maxbins=50), title="pearson(base profile, trained profile)"),
            y=alt.Y("count()", stack=None, title="messages"),
            color=alt.Color(
                "split:N", sort=SPLIT_ORDER, scale=alt.Scale(domain=SPLIT_ORDER, range=SPLIT_COLORS), title=None
            ),
            tooltip=["split", alt.Tooltip("count()")],
        )
        .properties(width=460, height=220, title="How similar did each message's emotion profile stay?")
    )
    mo.vstack([mo.ui.table(_summary, selection=None), _hist])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3 · Which emotions activate differently after training?

    Per emotion, over all 1972 messages: mean projection in the base vs the trained model, and
    the shift as an effect size (Δmean / base std — comparable across emotions with different
    projection scales).
    """)
    return


@app.cell
def _(CLUSTERS, EMOTIONS, M_BASE, M_TRAINED, np, pl, slugify):
    _emo2fam = {slugify(e): c for c, es in CLUSTERS.items() for e in es}
    _mb, _mt = M_BASE.mean(axis=0), M_TRAINED.mean(axis=0)
    _sb = M_BASE.std(axis=0)
    EMOTION_SHIFT = pl.DataFrame(
        {
            "emotion": EMOTIONS,
            "family": [_emo2fam.get(e, "?") for e in EMOTIONS],
            "mean_base": _mb,
            "mean_trained": _mt,
            "delta": _mt - _mb,
            "effect_size": (_mt - _mb) / np.where(_sb == 0, 1.0, _sb),
        }
    )
    return (EMOTION_SHIFT,)


@app.cell
def _(EMOTION_SHIFT, alt):
    _scatter = (
        alt.Chart(EMOTION_SHIFT)
        .mark_circle(size=55, opacity=0.75)
        .encode(
            x=alt.X("mean_base:Q", title="mean projection, base model"),
            y=alt.Y("mean_trained:Q", title="mean projection, trained model"),
            color=alt.Color("family:N", scale=alt.Scale(scheme="tableau10"), title=None),
            tooltip=[
                "emotion",
                "family",
                alt.Tooltip("mean_base:Q", format=".3f"),
                alt.Tooltip("mean_trained:Q", format=".3f"),
                alt.Tooltip("effect_size:Q", format=".3f"),
            ],
        )
        .properties(width=380, height=380, title="Per-emotion mean activation: base vs trained")
    )
    _diag = (
        alt.Chart(EMOTION_SHIFT)
        .transform_calculate(y="datum.mean_base")
        .mark_line(color="#999", strokeDash=[4, 4])
        .encode(x="mean_base:Q", y=alt.Y("y:Q"))
    )
    _scatter + _diag
    return


@app.cell
def _(EMOTION_SHIFT, alt, pl):
    _movers = pl.concat(
        [
            EMOTION_SHIFT.sort("effect_size", descending=True).head(15),
            EMOTION_SHIFT.sort("effect_size").head(15),
        ]
    )
    alt.Chart(_movers).mark_bar().encode(
        x=alt.X("effect_size:Q", title="shift (Δmean / base std)"),
        y=alt.Y("emotion:N", sort="-x", title=None),
        color=alt.Color("family:N", scale=alt.Scale(scheme="tableau10"), title=None),
        tooltip=["emotion", "family", alt.Tooltip("effect_size:Q", format=".3f")],
    ).properties(width=420, height=460, title="Top movers: emotions that activate more (+) or less (−) after training")
    return


@app.cell
def _(EMOTION_SHIFT, alt, pl):
    _fam = EMOTION_SHIFT.group_by("family").agg(
        pl.col("effect_size").mean().alias("mean_effect_size"), pl.len().alias("n_emotions")
    )
    alt.Chart(_fam).mark_bar().encode(
        x=alt.X("mean_effect_size:Q", title="mean shift across the family's emotions"),
        y=alt.Y("family:N", sort="-x", title=None),
        color=alt.condition(alt.datum.mean_effect_size > 0, alt.value("#4c78a8"), alt.value("#e45756")),
        tooltip=["family", alt.Tooltip("mean_effect_size:Q", format=".3f"), "n_emotions"],
    ).properties(width=420, height=260, title="Family-level activation shift")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Is the shift supervision-specific?

    The same per-emotion effect size, computed separately on the **train** messages (where
    training supervised a tag) and on the **unused** messages (never selected for anything).
    Points on the diagonal = a global, supervision-independent drift; points off it = emotions
    whose activation moved specifically where training touched.
    """)
    return


@app.cell
def _(CLUSTERS, EMOTIONS, M_BASE, M_TRAINED, SPLITS, alt, np, pl, slugify):
    _emo2fam = {slugify(e): c for c, es in CLUSTERS.items() for e in es}

    def _effect(mask: "np.ndarray") -> "np.ndarray":
        mb, mt = M_BASE[mask].mean(axis=0), M_TRAINED[mask].mean(axis=0)
        sb = M_BASE[mask].std(axis=0)
        return (mt - mb) / np.where(sb == 0, 1.0, sb)

    _splits = np.array(SPLITS)
    SPLIT_SHIFT = pl.DataFrame(
        {
            "emotion": EMOTIONS,
            "family": [_emo2fam.get(e, "?") for e in EMOTIONS],
            "effect_train": _effect(_splits == "train"),
            "effect_unused": _effect(_splits == "unused"),
        }
    )
    _scatter = (
        alt.Chart(SPLIT_SHIFT)
        .mark_circle(size=55, opacity=0.75)
        .encode(
            x=alt.X("effect_unused:Q", title="shift on unused messages (no supervision)"),
            y=alt.Y("effect_train:Q", title="shift on train messages (supervised)"),
            color=alt.Color("family:N", scale=alt.Scale(scheme="tableau10"), title=None),
            tooltip=[
                "emotion",
                "family",
                alt.Tooltip("effect_train:Q", format=".3f"),
                alt.Tooltip("effect_unused:Q", format=".3f"),
            ],
        )
        .properties(width=380, height=380, title="Per-emotion shift: supervised vs untouched messages")
    )
    _diag = (
        alt.Chart(SPLIT_SHIFT)
        .transform_calculate(y="datum.effect_unused")
        .mark_line(color="#999", strokeDash=[4, 4])
        .encode(x="effect_unused:Q", y=alt.Y("y:Q"))
    )
    _corr = float(np.corrcoef(SPLIT_SHIFT["effect_train"], SPLIT_SHIFT["effect_unused"])[0, 1])
    (_scatter + _diag).properties(
        title=f"Per-emotion shift: supervised vs untouched messages (r = {_corr:.3f})"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4 · Cross-check — does the trained model's own probe agree?

    Sections 1–3 hold the probe fixed (base vectors). The trained model also has its own
    re-extracted vector set (exp-04 geometry: median cosine to base 0.998). If the two views of
    the *same trained activations* disagreed, the "probe held still" reading would be fragile.
    """)
    return


@app.cell
def _(META, M_TRAINED, M_TRAINED_TV, mo, pl, render_tags, tag_overlap):
    _tags_tb = render_tags(M_TRAINED)
    _tags_tt = render_tags(M_TRAINED_TV)
    _agree = pl.DataFrame(
        [
            {"split": m["split"], **tag_overlap(a, b)}
            for m, a, b in zip(META, _tags_tb, _tags_tt)
        ]
    )
    _summary = _agree.group_by("split").agg(
        pl.len().alias("n"),
        pl.col("top1").mean().round(3).alias("top1_emotion"),
        pl.col("top1_family").mean().round(3).alias("top1_family"),
        pl.col("jaccard").mean().round(3).alias("jaccard"),
    )
    _overall = _agree["top1_family"].mean()
    mo.vstack(
        [
            mo.ui.table(_summary, selection=None),
            mo.md(
                f"Tag agreement between the base-vector and trained-vector views of the same "
                f"trained activations: **{_overall:.0%}** top-1 family overall. High agreement = "
                f"the choice of vector set doesn't drive any conclusion above."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
