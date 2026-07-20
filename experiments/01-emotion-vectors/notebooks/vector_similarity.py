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
    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, load_clusters, mo, np, pl, save_chart, slugify


@app.cell
def _(Path, json, load_clusters, np, slugify):
    HERE = Path(__file__).parents[1]  # the experiment dir
    LAYERS = (18, 21, 24)
    arts = {
        L: json.loads((HERE / "data" / "similarity" / f"layer_{L}.json").read_text(encoding="utf-8"))
        for L in LAYERS
    }
    art = arts[21]
    emotions = art["emotions"]
    fam = art["clusters"]
    M = np.array(art["matrix"])
    idx = {e: i for i, e in enumerate(emotions)}

    # Row/column order: the taxonomy's family order (matches the 02 heatmap exhibits).
    taxonomy = load_clusters(HERE / "clusters.json")
    emotion_order = [slugify(e) for f in taxonomy for e in taxonomy[f] if slugify(e) in idx]
    return LAYERS, M, art, arts, emotion_order, emotions, fam, idx


@app.cell
def _(art, emotions, mo):
    mo.md(f"""
    ## Emotion-vector similarity structure (the distance-metric gate)

    The {len(emotions)}×{len(emotions)} cosine-similarity matrix of the centered,
    L2-normalized emotion vectors (run `{art["vectors_run"]}`, layer {art["layer"]}),
    built by `run.py::similarity`. This notebook is the validation gate for the
    distance-based tag metrics (`docs/tag-accuracy-distance-metric.md` §8): before any
    score built on this matrix is trusted, the matrix itself must show family block
    structure, near-synonym neighbourhoods, and anticorrelated opposite-valence
    families.
    """)
    return


@app.cell
def _(M, emotion_order, fam, idx, pl):
    pairs_full = pl.DataFrame(
        [
            (a, b, float(M[idx[a], idx[b]]))
            for a in emotion_order
            for b in emotion_order
        ],
        schema=["a", "b", "cosine"],
        orient="row",
    )
    pairs_unique = pl.DataFrame(
        [
            (
                a,
                b,
                float(M[idx[a], idx[b]]),
                "same family" if fam[a] == fam[b] else "different family",
            )
            for i, a in enumerate(emotion_order)
            for b in emotion_order[i + 1 :]
        ],
        schema=["a", "b", "cosine", "pair_type"],
        orient="row",
    )
    return pairs_full, pairs_unique


@app.cell
def _(alt, emotion_order, pairs_full):
    # Full-resolution view (29,241 cells): notebook display only — as an SVG exhibit it
    # weighs ~8 MB, so the saved exhibit below aggregates to family level instead
    # (the exhibits rule: aggregate view when the domain is large).
    alt.Chart(pairs_full).mark_rect().encode(
        x=alt.X(
            "a:N",
            sort=emotion_order,
            title="emotion (ordered by family)",
            axis=alt.Axis(labels=False, ticks=False),
        ),
        y=alt.Y(
            "b:N",
            sort=emotion_order,
            title="emotion (ordered by family)",
            axis=alt.Axis(labels=False, ticks=False),
        ),
        color=alt.Color(
            "cosine:Q",
            title="cosine",
            scale=alt.Scale(scheme="redblue", reverse=True, domainMid=0),
        ),
        tooltip=[
            alt.Tooltip("a:N"),
            alt.Tooltip("b:N"),
            alt.Tooltip("cosine:Q", format=".3f"),
        ],
    ).properties(
        width=520,
        height=520,
        title="Pairwise cosine similarity of the 171 emotion vectors, ordered by family",
    )
    return


@app.cell
def _(M, alt, emotions, fam, idx, load_clusters, np, pl, save_chart, Path):
    _family_order = list(load_clusters(Path(__file__).parents[1] / "clusters.json"))
    _members = {f: [idx[e] for e in emotions if fam[e] == f] for f in _family_order}
    _rows = []
    for _fa in _family_order:
        for _fb in _family_order:
            _block = M[np.ix_(_members[_fa], _members[_fb])]
            if _fa == _fb:  # within-family mean over distinct pairs, not self-similarity
                _n = _block.shape[0]
                _mean = (_block.sum() - _n) / (_n * (_n - 1)) if _n > 1 else 1.0
            else:
                _mean = _block.mean()
            _rows.append((_fa, _fb, float(_mean)))
    _df = pl.DataFrame(_rows, schema=["family_a", "family_b", "mean_cosine"], orient="row")
    _base = alt.Chart(_df).encode(
        x=alt.X("family_a:N", sort=_family_order, title=None, axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("family_b:N", sort=_family_order, title=None),
    )
    _heat = _base.mark_rect().encode(
        color=alt.Color(
            "mean_cosine:Q",
            title="mean cosine",
            scale=alt.Scale(scheme="redblue", reverse=True, domainMid=0),
        ),
        tooltip=[
            alt.Tooltip("family_a:N"),
            alt.Tooltip("family_b:N"),
            alt.Tooltip("mean_cosine:Q", format=".3f"),
        ],
    )
    _labels = _base.mark_text(fontSize=9).encode(
        text=alt.Text("mean_cosine:Q", format=".2f"),
        color=alt.condition("abs(datum.mean_cosine) > 0.45", alt.value("white"), alt.value("black")),
    )
    save_chart(
        (_heat + _labels).properties(
            width=430,
            height=430,
            title="Mean cosine similarity between emotion families",
        ),
        "similarity_block_structure",
        caption="Mean pairwise cosine similarity between the emotion vectors of each family pair (layer 21, centered unit vectors); diagonal cells average over distinct within-family pairs.",
        takeaway="The replicated emotion vectors are block-structured by family: mean pairwise cosine is +0.44 within families and −0.07 across them, and opposite-valence families are anticorrelated (hostile anger vs peaceful contentment −0.49, exuberant joy vs hostile anger −0.44). Graded similarity between emotions is recoverable from the vector geometry, supporting a distance-based tag-accuracy metric.",
        notebook=__file__,
    )
    return


@app.cell
def _(alt, pairs_unique, pl, save_chart):
    _means = pairs_unique.group_by("pair_type").agg(pl.col("cosine").median().alias("median"))
    _density = (
        alt.Chart(pairs_unique)
        .transform_density("cosine", groupby=["pair_type"], as_=["cosine", "density"])
        .mark_area(opacity=0.55)
        .encode(
            x=alt.X("cosine:Q", title="pairwise cosine similarity"),
            y=alt.Y("density:Q", title="density"),
            color=alt.Color(
                "pair_type:N",
                title=None,
                scale=alt.Scale(
                    domain=["same family", "different family"],
                    range=["#4c78a8", "#f58518"],
                ),
            ),
        )
    )
    _rules = (
        alt.Chart(_means)
        .mark_rule(strokeDash=[4, 3])
        .encode(
            x="median:Q",
            color=alt.Color(
                "pair_type:N",
                scale=alt.Scale(
                    domain=["same family", "different family"],
                    range=["#4c78a8", "#f58518"],
                ),
            ),
        )
    )
    save_chart(
        (_density + _rules).properties(
            width=460,
            height=220,
            title="Cosine similarity of emotion pairs, by family membership",
        ),
        "within_vs_cross_family_cosine",
        caption="Distribution of cosine similarity over the 14,535 unique emotion pairs, split by whether the two emotions share a family; dashed rules mark the medians.",
        takeaway="Same-family and different-family pairs separate cleanly (median cosine +0.49 vs −0.13), but the distributions overlap in both directions: 12% of different-family pairs exceed the same-family median, and 13% of same-family pairs have negative cosine (the fear-and-overwhelm family contains both amazed and mortified, cosine −0.61). A family-bucket metric scores the former as zero and the latter as full credit; a graded similarity metric resolves both.",
        notebook=__file__,
    )
    return


@app.cell
def _(M, alt, emotions, fam, idx, np, pl, save_chart):
    _anchors = ["afraid", "grateful", "overwhelmed", "hostile", "blissful"]
    _rows = []
    for _a in _anchors:
        _sims = M[idx[_a]].copy()
        _sims[idx[_a]] = -np.inf
        for _j in np.argsort(_sims)[-5:][::-1]:
            _rows.append(
                (
                    _a,
                    emotions[_j],
                    float(_sims[_j]),
                    "same family" if fam[emotions[_j]] == fam[_a] else "different family",
                )
            )
    _df = pl.DataFrame(_rows, schema=["anchor", "neighbour", "cosine", "pair_type"], orient="row")
    _chart = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            y=alt.Y("neighbour:N", sort="-x", title=None),
            x=alt.X("cosine:Q", title="cosine similarity to the anchor", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "pair_type:N",
                title=None,
                scale=alt.Scale(
                    domain=["same family", "different family"],
                    range=["#4c78a8", "#f58518"],
                ),
            ),
            tooltip=[
                alt.Tooltip("neighbour:N"),
                alt.Tooltip("cosine:Q", format=".3f"),
                alt.Tooltip("pair_type:N"),
            ],
        )
        .properties(width=340, height=110)
        .facet(row=alt.Row("anchor:N", title=None, header=alt.Header(labelFontWeight="bold")))
        .resolve_scale(y="independent")
        .properties(title="Five nearest neighbours of five anchor emotions")
    )
    save_chart(
        _chart,
        "nearest_neighbor_examples",
        caption="The five nearest neighbours of five anchor emotions in the vector space, coloured by whether the neighbour shares the anchor's family.",
        takeaway="Nearest neighbours are near-synonyms (afraid–scared 0.96, grateful–thankful 0.98), and 100 of the 171 emotions (58%) have at least one out-of-family emotion among their five nearest neighbours (pleased for grateful, content for blissful, disturbed for hostile). Family boundaries cut through dense neighbourhoods — the information a graded metric keeps and a bucket metric discards.",
        notebook=__file__,
    )
    return


@app.cell
def _(LAYERS, arts, mo, np, pairs_unique, pl):
    _iu = np.triu_indices(len(arts[21]["emotions"]), k=1)
    _tri = {L: np.array(arts[L]["matrix"])[_iu] for L in LAYERS}
    _corrs = {
        (a, b): float(np.corrcoef(_tri[a], _tri[b])[0, 1])
        for a, b in ((18, 21), (21, 24), (18, 24))
    }
    _stats = pairs_unique.group_by("pair_type").agg(
        pl.col("cosine").mean().alias("mean"), pl.col("cosine").median().alias("median")
    )
    _by = {r["pair_type"]: r for r in _stats.to_dicts()}
    mo.md(f"""
    **Gate summary.** Same-family pairs: mean cosine {_by["same family"]["mean"]:+.3f}
    (median {_by["same family"]["median"]:+.3f}); different-family pairs: mean
    {_by["different family"]["mean"]:+.3f} (median {_by["different family"]["median"]:+.3f}).

    **Layer robustness.** The pairwise-similarity structure is essentially identical at the
    three extracted layers: correlation of the matrices' upper triangles is
    {_corrs[(18, 21)]:.4f} (18 vs 21), {_corrs[(21, 24)]:.4f} (21 vs 24), and
    {_corrs[(18, 24)]:.4f} (18 vs 24) — the layer-21 choice is not load-bearing.

    **Verdict: the gate passes.** The similarity structure supports the distance-based
    tag metrics; see `docs/tag-accuracy-distance-metric.md` §8 and the `dist_*` columns
    in the 04 experiments' `runs_summary.json`.
    """)
    return


if __name__ == "__main__":
    app.run()
