import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, mo, pl, save_chart


@app.cell
def _(Path, json, mo):
    HERE = Path(__file__).parents[1]  # the experiment dir
    READOUT_DIR = HERE / "data" / "readouts"

    # Readout names are "<arm>-<recentering>-on-<test source>"; arm names contain hyphens
    # and recentering names do not, so the two split on the *last* hyphen.
    files = sorted(READOUT_DIR.glob("*-on-*.json"))
    mo.stop(
        not files,
        mo.md(
            f"""
            **No readouts yet.** Run the experiment, then pull the results down:

            ```
            uv run modal run experiments/01-cross-generator-vectors/run.py::fetch_results
            ```

            and run the printed `modal volume get` lines. Expected under `{READOUT_DIR}`.
            """
        ),
    )
    readouts = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in files}
    splits = json.loads((HERE / "data" / "splits.json").read_text(encoding="utf-8"))
    ARMS = splits["arms"]  # {arm: {source, n_train}}
    return ARMS, readouts, splits


@app.cell
def _(ARMS):
    SOURCE_LABEL = {"llama": "Llama-3.3-70B", "hf": "the paper's corpus"}

    def arm_label(arm: str) -> str:
        spec = ARMS[arm]
        return f"{SOURCE_LABEL[spec['source']]}, {spec['n_train']:,}/emotion"

    def parts(name: str) -> dict:
        arm_variant, test = name.split("-on-")
        arm, variant = arm_variant.rsplit("-", 1)
        return {
            "arm": arm,
            "variant": variant,
            "test": test,
            "arm_label": arm_label(arm),
            "test_label": SOURCE_LABEL[test],
            "arm_source": ARMS[arm]["source"],
            "n_train": ARMS[arm]["n_train"],
            "same_corpus": ARMS[arm]["source"] == test,
        }

    ARM_ORDER = sorted(ARMS, key=lambda a: (ARMS[a]["source"] != "hf", -ARMS[a]["n_train"]))
    return ARM_ORDER, SOURCE_LABEL, arm_label, parts


@app.cell
def _(parts, pl, readouts):
    cells = pl.DataFrame(
        [
            {"cell": name, **parts(name), **art["summary"]}
            for name, art in readouts.items()
        ]
    ).sort("variant", "arm", "test")

    n_vectors = int(cells["n_vectors"][0])
    chance = float(cells["chance_top1"][0])
    variants = cells["variant"].unique(maintain_order=True).to_list()
    PRIMARY = "dialogues" if "dialogues" in variants else variants[0]
    return PRIMARY, cells, chance, n_vectors, variants


@app.cell
def _(ARMS, PRIMARY, cells, chance, mo, n_vectors, pl):
    _p = cells.filter(pl.col("variant") == PRIMARY)
    _lines = "\n".join(
        f"    - **{r['arm']}** ({r['arm_label']}) on {r['test_label']}: "
        f"top-1 {r['top1']:.3f}, family {r['cluster_top1']:.3f}, mean rank {r['mean_rank']:.1f}"
        for r in _p.sort("arm", "test").iter_rows(named=True)
    )
    mo.md(
        f"""
        ## Emotion vectors built the paper's way, on the paper's corpus

        {n_vectors} emotion vectors, built by the procedure of section 1.1 — pool residual
        activations from the 50th token of each story onward, average per emotion, subtract
        the mean across emotions, project out the top principal components of activations on
        neutral transcripts (as many as cover 50% of the variance) — from
        {len(ARMS)} different training corpora, and scored on held-out stories from both
        sources. Chance top-1 is {chance:.4f} (1 of {n_vectors}).

        Recentering shown: `{PRIMARY}`.

{_lines}
        """
    )
    return


@app.cell
def _(ARM_ORDER, PRIMARY, alt, arm_label, cells, chance, pl, save_chart):
    _metrics = {"top1": "exact emotion", "cluster_top1": "right family", "top5": "in the top 5"}
    _arm_labels = [arm_label(a) for a in ARM_ORDER]
    grid_data = (
        cells.filter(pl.col("variant") == PRIMARY)
        .select("arm", "arm_label", "test_label", "same_corpus", *_metrics)
        .unpivot(
            index=["arm", "arm_label", "test_label", "same_corpus"],
            variable_name="metric",
            value_name="accuracy",
        )
        .with_columns(
            pl.col("metric").replace_strict(_metrics).alias("metric_label"),
            pl.when(pl.col("same_corpus"))
            .then(pl.lit("same corpus (held out)"))
            .otherwise(pl.lit("other corpus"))
            .alias("pairing"),
        )
    )

    grid = (
        alt.Chart(grid_data)
        .mark_bar()
        .encode(
            x=alt.X("test_label:N", title="held-out stories from", axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("accuracy:Q", title="accuracy", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "pairing:N",
                title=None,
                scale=alt.Scale(
                    domain=["same corpus (held out)", "other corpus"],
                    range=["#4c78a8", "#e45756"],
                ),
            ),
            column=alt.Column("metric_label:N", title=None, sort=list(_metrics.values())),
            row=alt.Row("arm_label:N", title="vectors trained on", sort=_arm_labels),
        )
        .properties(width=140, height=120)
    )
    save_chart(
        grid,
        "corpus_grid",
        caption=(
            "Emotion-vector readout accuracy on held-out stories, for vectors trained on "
            "each corpus and scored against held-out stories from both sources. Blue = the "
            "test stories come from the same corpus the vectors were trained on (disjoint "
            f"stories); red = the other corpus. Chance top-1 is {chance:.4f}."
        ),
        takeaway=(
            "The paper-corpus vectors are the stronger set: 0.367 top-1 on their own held-out "
            "stories (63x the 0.006 chance rate) against 0.241 for the Llama vectors on "
            "theirs, and family accuracy 0.764 against 0.637. Across corpora the two are "
            "close, 0.181 and 0.205 top-1, so about half of each set's own-corpus accuracy "
            "survives a change of writer. Rankings are per-emotion standardized across the "
            "test set, as the tag pipeline does; an unstandardized argmax gives 0.110 and "
            "0.091 instead and is dominated by per-emotion offsets."
        ),
        notebook=__file__,
    )
    return (grid_data,)


@app.cell
def _(mo):
    mo.md(
        """
        ### Where in the ranking the true emotion lands

        Top-1 accuracy is a hard threshold and hides the shape of the failure. This is the
        share of held-out stories whose true emotion falls within the top *k* projections,
        as *k* runs over the taxonomy: a curve that rises steeply means the vector is nearly
        right even when it is not exactly right.
        """
    )
    return


@app.cell
def _(parts, pl, readouts):
    story_rows = pl.DataFrame(
        [
            {
                "cell": name,
                **parts(name),
                "emotion": s["emotion"],
                "cluster": s.get("true_cluster"),
                "topic": s.get("topic"),
                "rank": s["rank"],
                "z_margin": s["z_margin"],
                "correct": s["correct"],
                "cluster_correct": s["cluster_correct"],
            }
            for name, art in readouts.items()
            for s in art["stories"]
            if "rank" in s
        ]
    )
    return (story_rows,)


@app.cell
def _(PRIMARY, alt, n_vectors, pl, save_chart, story_rows):
    _sub = story_rows.filter(pl.col("variant") == PRIMARY)
    _ks = sorted({1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, n_vectors})
    curve_data = pl.DataFrame(
        [
            {
                "cell": cell,
                "arm_label": grp["arm_label"][0],
                "test_label": grp["test_label"][0],
                "pairing": "same corpus (held out)" if grp["same_corpus"][0] else "other corpus",
                "k": k,
                "within_k": float((grp["rank"] <= k).mean()),
            }
            for (cell,), grp in _sub.group_by("cell")
            for k in _ks
        ]
    )

    curves = (
        alt.Chart(curve_data)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "k:Q",
                title=f"k (rank of the true emotion, out of {n_vectors})",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y(
                "within_k:Q",
                title="share of stories with the true emotion in the top k",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("arm_label:N", title="vectors trained on"),
            strokeDash=alt.StrokeDash("test_label:N", title="held-out stories from"),
            tooltip=["cell", "k", alt.Tooltip("within_k:Q", format=".3f")],
        )
        .properties(width=440, height=280)
    )
    save_chart(
        curves,
        "rank_curves",
        caption=(
            "Share of held-out stories whose true emotion falls within the top k emotion "
            "vectors by projection, as k runs over the taxonomy (log scale). Colour is the "
            "training corpus, dash is which corpus the test stories came from."
        ),
        takeaway=(
            "The true emotion is in the top 5 of 171 for 74% of paper-corpus stories read "
            "by paper-corpus vectors (median rank 2) and 57% of Llama stories read by Llama "
            "vectors (median 4); crossing corpora gives 48% and 54% (medians 6 and 5). The "
            "paper-corpus vectors on their own corpus are the one curve that separates "
            "from the other three."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### Which emotions carry across corpora

        One point per emotion per arm: how strongly its vector picks it out of held-out
        stories from one corpus (horizontal) against the other corpus (vertical), scored by
        `z_margin` — the projection of the true emotion in standard deviations of that
        story's own projection spread. Points on the diagonal read the same either way;
        points far off it are emotions whose vector has picked up something about how one
        corpus was written.
        """
    )
    return


@app.cell
def _(PRIMARY, alt, pl, save_chart, story_rows):
    _per = (
        story_rows.filter(pl.col("variant") == PRIMARY)
        .group_by("arm", "arm_label", "test", "emotion", "cluster")
        .agg(pl.col("z_margin").mean().alias("z_margin"))
    )
    transfer_data = (
        _per.filter(pl.col("test") == "llama")
        .select("arm", "arm_label", "emotion", "cluster", pl.col("z_margin").alias("on_llama"))
        .join(
            _per.filter(pl.col("test") == "hf").select(
                "arm", "emotion", pl.col("z_margin").alias("on_paper_corpus")
            ),
            on=["arm", "emotion"],
            how="inner",
        )
        .with_columns((pl.col("on_paper_corpus") - pl.col("on_llama")).alias("difference"))
    )

    _lim = [
        float(min(transfer_data["on_llama"].min(), transfer_data["on_paper_corpus"].min())) - 0.2,
        float(max(transfer_data["on_llama"].max(), transfer_data["on_paper_corpus"].max())) + 0.2,
    ]
    _points = (
        alt.Chart(transfer_data)
        .mark_circle(size=45, opacity=0.75)
        .encode(
            x=alt.X("on_llama:Q", title="z-margin on Llama-written stories", scale=alt.Scale(domain=_lim)),
            y=alt.Y("on_paper_corpus:Q", title="z-margin on the paper's stories", scale=alt.Scale(domain=_lim)),
            color=alt.Color("cluster:N", title="family"),
            column=alt.Column("arm_label:N", title="vectors trained on"),
            tooltip=[
                "emotion", "cluster", "arm_label",
                alt.Tooltip("on_llama:Q", format=".2f"),
                alt.Tooltip("on_paper_corpus:Q", format=".2f"),
                alt.Tooltip("difference:Q", format=".2f"),
            ],
        )
        .properties(width=250, height=250)
    )
    save_chart(
        _points,
        "per_emotion_transfer",
        caption=(
            "Per-emotion readout strength on the two held-out sets: mean z-margin on "
            "Llama-written stories against the paper's stories, one point per emotion, "
            "coloured by family, one panel per training corpus."
        ),
        takeaway=(
            "The two sets lose different amounts when the corpus is swapped: mean z-margin "
            "falls by 0.57 for the paper-corpus vectors (only 9 of 171 emotions read better "
            "on Llama stories) but by just 0.13 for the Llama vectors, 77 of whose 171 "
            "emotions read better on the paper corpus than on their own. The Llama vectors "
            "are weaker but flatter across corpora. Per-emotion legibility correlates at "
            "r = 0.42 between the arms, so which emotions are easy depends on the corpus "
            "as much as on the emotion."
        ),
        notebook=__file__,
    )
    return (transfer_data,)


@app.cell
def _(mo, variants):
    mo.md(
        f"""
        ### What the neutral-transcript projection contributes

        Centering across emotions cancels the neutral mean exactly, so the `plain`
        (denoise-off) vectors are algebraically independent of the neutral corpus, and the
        `dialogues` vectors differ from them only by the projection the paper prescribes:
        the top principal components of activations on the 1,200 neutral Human/Assistant
        transcripts, enough to cover 50% of the variance, projected out. The difference
        between the two panels is the whole of what that step does here.

        Recenterings present: {", ".join(f"`{v}`" for v in variants)}.
        """
    )
    return


@app.cell
def _(alt, cells, pl, save_chart, variants):
    denoise_data = cells.select(
        "variant", "arm", "arm_label", "test_label", "same_corpus", "top1", "cluster_top1"
    ).with_columns(
        pl.when(pl.col("same_corpus"))
        .then(pl.lit("same corpus (held out)"))
        .otherwise(pl.lit("other corpus"))
        .alias("pairing"),
        pl.col("variant")
        .replace_strict(
            {"plain": "no projection", "dialogues": "neutral-transcript PCs projected out"},
            default=pl.col("variant"),
        )
        .alias("variant_label"),
    )

    denoise = (
        alt.Chart(denoise_data)
        .mark_point(size=90, filled=True)
        .encode(
            x=alt.X("variant_label:N", title=None, axis=alt.Axis(labelAngle=-20)),
            y=alt.Y("top1:Q", title="top-1 accuracy"),
            color=alt.Color(
                "pairing:N",
                title=None,
                scale=alt.Scale(
                    domain=["same corpus (held out)", "other corpus"],
                    range=["#4c78a8", "#e45756"],
                ),
            ),
            detail="test_label:N",
            column=alt.Column("arm_label:N", title="vectors trained on"),
            tooltip=[
                "arm_label", "test_label", "variant_label",
                alt.Tooltip("top1:Q", format=".3f"),
            ],
        )
        .properties(width=150, height=200)
    )
    save_chart(
        denoise,
        "denoise_contribution",
        caption=(
            "Top-1 accuracy of every cell under the two recenterings: without the "
            "projection (where the vectors are independent of the neutral corpus) and with "
            "the paper's projection against 1,200 neutral Human/Assistant transcripts."
        ),
        takeaway=(
            "Once rankings are standardized per emotion, the projection is a small, "
            "consistent gain rather than a large one: top-1 improves in all four cells by "
            "+0.006 to +0.015, family accuracy by +0.002 to +0.013, mean rank by 0.3 to "
            "1.2 places. Most of what the unstandardized comparison credited to the "
            "projection was the projection shifting per-emotion offsets, which the "
            "standardization removes either way."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### Which families the vectors can find at all

        Accuracy averaged over the whole taxonomy hides that the 171 emotions are not
        equally legible. This is family-level accuracy per family — did the top-scoring
        vector land in the right one of the ten — for each set of vectors reading its own
        corpus, which is the most favourable case each one gets.
        """
    )
    return


@app.cell
def _(PRIMARY, alt, pl, readouts, save_chart):
    fam_rows = []
    for name, art in readouts.items():
        arm_variant, test = name.split("-on-")
        arm, variant = arm_variant.rsplit("-", 1)
        if variant != PRIMARY or arm != test:  # each set of vectors on its own corpus
            continue
        by_family = {}
        for r in art["per_emotion"]:
            by_family.setdefault(r["cluster"], []).append(r)
        for family, rs in by_family.items():
            fam_rows.append(
                {
                    "arm": arm,
                    "family": family.replace("_", " "),
                    "n_emotions": len(rs),
                    "family_accuracy": sum(r["cluster_top1"] for r in rs) / len(rs),
                    "exact": sum(r["top1"] for r in rs) / len(rs),
                }
            )
    family_data = pl.DataFrame(fam_rows)
    _order = (
        family_data.group_by("family")
        .agg(pl.col("family_accuracy").mean().alias("m"))
        .sort("m", descending=True)["family"]
        .to_list()
    )

    families = (
        alt.Chart(family_data)
        .mark_bar()
        .encode(
            y=alt.Y("family:N", title=None, sort=_order),
            x=alt.X("family_accuracy:Q", title="family-level accuracy", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("arm:N", title="vectors trained on"),
            yOffset=alt.YOffset("arm:N"),
            tooltip=["family", "arm", "n_emotions",
                     alt.Tooltip("family_accuracy:Q", format=".3f"),
                     alt.Tooltip("exact:Q", format=".3f")],
        )
        .properties(width=340, height=280)
    )
    save_chart(
        families,
        "family_legibility",
        caption=(
            "Family-level accuracy per emotion family, for each set of vectors reading "
            "held-out stories from its own corpus. Bars are the share of that family's "
            "stories whose top-scoring vector falls in the correct family; chance is 0.149."
        ),
        takeaway=(
            "Every family is found well above the 0.149 chance rate by both sets. For the "
            "paper-corpus vectors family accuracy runs from 0.61 (competitive pride) to "
            "0.84 (depleted disengagement) with hostile anger at 0.73; the earlier reading "
            "of anger as unfindable (0.018) was an artifact of ranking unstandardized "
            "projections. The Llama vectors are lower across the board, from 0.37 "
            "(exuberant joy) to 0.74 (despair and shame), and their one weak family is "
            "joy, not a negative one."
        ),
        notebook=__file__,
    )
    return (family_data,)


@app.cell
def _(mo):
    mo.md(
        """
        ### The pool the probe is actually pointed at

        Everything above scores the vectors on *stories*, which is not what any downstream
        experiment reads. This is the 1,972 direct-elicitation messages of
        `02-elicited-activations`, read at the pre-response token — the same cached
        activations, projected onto each candidate set of vectors in turn. Three sets: the
        production vectors, the Llama rebuild (the same stories, the paper's neutral
        projection), and the paper-corpus vectors.
        """
    )
    return


@app.cell
def _(Path, alt, json, pl, save_chart):
    _dir = Path(__file__).parents[1] / "data" / "message_readouts"
    _files = {
        "production": "readout.json",
        "llama rebuild": "readout_xgen_llama.json",
        "paper corpus": "readout_xgen_hf.json",
    }
    # Family accuracy per family, computed from the stored per-message projections.
    from name_that_feeling.emotion_vectors.taxonomy import (
        emotion_to_cluster,
        load_clusters,
        slugify,
    )

    _e2c = {
        slugify(k): v
        for k, v in emotion_to_cluster(
            load_clusters(Path(__file__).parents[2] / "01-emotion-vectors" / "clusters.json")
        ).items()
    }
    import numpy as np

    msg_rows = []
    for _label, _fname in _files.items():
        _msgs = json.loads((_dir / _fname).read_text(encoding="utf-8"))["messages"]
        _names = list(_msgs[0]["projections"])
        _P = np.array([[m["projections"][n] for n in _names] for m in _msgs])
        # Standardize each emotion across the pool before ranking, exactly as the tag
        # pipeline does (generation.sft.per_emotion_stats): the raw projection carries a
        # per-emotion offset larger than the signal, and a raw argmax ranks offsets.
        _Z = (_P - _P.mean(0)) / (_P.std(0) + 1e-12)
        for _m, _pred_j in zip(_msgs, _Z.argmax(1)):
            _true_fam = _e2c[slugify(_m["emotion"])]
            msg_rows.append(
                {
                    "vectors": _label,
                    "family": _true_fam.replace("_", " "),
                    "family_correct": _e2c[_names[_pred_j]] == _true_fam,
                }
            )
    message_pool = (
        pl.DataFrame(msg_rows)
        .group_by("vectors", "family")
        .agg(
            pl.col("family_correct").mean().alias("family_accuracy"),
            pl.len().alias("n_messages"),
        )
    )
    _order = (
        message_pool.group_by("family")
        .agg(pl.col("n_messages").max().alias("n"))
        .sort("n", descending=True)["family"]
        .to_list()
    )

    message_chart = (
        alt.Chart(message_pool)
        .mark_bar()
        .encode(
            y=alt.Y("family:N", title=None, sort=_order),
            x=alt.X("family_accuracy:Q", title="family-level accuracy", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "vectors:N",
                title="vectors",
                scale=alt.Scale(
                    domain=["production", "llama rebuild", "paper corpus"],
                    range=["#9c9c9c", "#4c78a8", "#e45756"],
                ),
            ),
            yOffset=alt.YOffset(
                "vectors:N", sort=["production", "llama rebuild", "paper corpus"]
            ),
            tooltip=["family", "vectors", "n_messages",
                     alt.Tooltip("family_accuracy:Q", format=".3f")],
        )
        .properties(width=340, height=320)
    )
    save_chart(
        message_chart,
        "message_pool_families",
        caption=(
            "Family-level accuracy on the 1,972 direct-elicitation messages, read at the "
            "pre-response token, for three candidate vector sets over the same cached "
            "activations, each emotion standardized across the pool before ranking. "
            "Families are ordered by how many messages they hold; chance is 0.155."
        ),
        takeaway=(
            "On the pool the probe is actually pointed at, with rankings standardized per "
            "emotion as the tag pipeline does, the paper-corpus vectors come out ahead: "
            "family accuracy 0.400 against 0.374 for production and 0.362 for the Llama "
            "rebuild (paired z of +2.1 and +3.1), median rank 17 against 18 and 19. The "
            "margin is modest and family-uneven: the paper corpus leads on anger, joy, "
            "gratitude, contentment and amusement, and trails on despair and depleted "
            "disengagement. Fear, 27% of the pool, is a three-way tie at 0.54-0.56."
        ),
        notebook=__file__,
    )
    return (message_pool,)


@app.cell
def _(mo):
    mo.md("### The numbers")
    return


@app.cell
def _(cells):
    cells.select(
        "variant", "arm", "n_train", "test", "top1", "top5", "cluster_top1",
        "mean_rank", "median_rank", "mean_z_margin", "chance_top1", "n_scored",
    )
    return


if __name__ == "__main__":
    app.run()
