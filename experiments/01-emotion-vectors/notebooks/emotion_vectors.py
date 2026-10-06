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

    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, mo, np, pl, save_chart


@app.cell
def _(Path, json, mo):
    HERE = Path(__file__).parents[1]  # the experiment dir
    READOUT_DIR = HERE / "data" / "readouts"

    # Readout names are "hf-<recentering>-on-hf": the paper-corpus vectors under one
    # recentering, scored on the held-out paper-corpus stories.
    files = sorted(READOUT_DIR.glob("hf-*-on-hf.json"))
    mo.stop(
        not files,
        mo.md(
            f"""
            **No readouts yet.** Run the experiment, then pull the results down:

            ```
            uv run modal run experiments/01-emotion-vectors/run.py::fetch_results
            ```

            and run the printed `modal volume get` lines. Expected under `{READOUT_DIR}`.
            """
        ),
    )
    readouts = {
        p.stem.split("-on-")[0].split("-", 1)[1]: json.loads(p.read_text(encoding="utf-8"))
        for p in files
    }
    splits = json.loads((HERE / "data" / "splits.json").read_text(encoding="utf-8"))
    return HERE, readouts, splits


@app.cell
def _(pl, readouts):
    VARIANT_LABEL = {
        "dialogues": "neutral-transcript PCs projected out (canonical)",
        "plain": "no projection",
    }
    cells = pl.DataFrame(
        [
            {"variant": v, "variant_label": VARIANT_LABEL.get(v, v), **art["summary"]}
            for v, art in readouts.items()
        ]
    ).sort("variant")

    n_vectors = int(cells["n_vectors"][0])
    chance = float(cells["chance_top1"][0])
    PRIMARY = "dialogues" if "dialogues" in readouts else str(cells["variant"][0])
    return PRIMARY, VARIANT_LABEL, cells, chance, n_vectors


@app.cell
def _(PRIMARY, cells, chance, mo, n_vectors, pl, splits):
    _lines = "\n".join(
        f"    - **{r['variant']}** ({r['variant_label']}): top-1 {r['top1']:.3f}, "
        f"top-5 {r['top5']:.3f}, family {r['cluster_top1']:.3f}, mean rank {r['mean_rank']:.1f}"
        for r in cells.sort(pl.col("variant") != PRIMARY).iter_rows(named=True)
    )
    mo.md(
        f"""
        ## Emotion vectors built the paper's way, on the paper's corpus

        {n_vectors} emotion vectors, built by the procedure of section 1.1 of Sofroniew et
        al. 2026 — pool residual activations from the 50th token of each story onward,
        average per emotion, subtract the mean across emotions, project out the top
        principal components of activations on neutral transcripts (as many as cover 50%
        of the variance), normalize — from {splits['n_train']:,} stories per emotion of
        `{splits['source']}`, and scored on {splits['n_test']} held-out stories per
        emotion that the vectors never saw. Chance top-1 is {chance:.4f} (1 of
        {n_vectors}). The `{PRIMARY}` recentering is the project's one vector set
        (`models.emotion_vectors_run`); `plain` is the same raws without the projection.

{_lines}
        """
    )
    return


@app.cell
def _(PRIMARY, alt, cells, chance, pl, save_chart):
    _metrics = {"top1": "exact emotion", "top5": "in the top 5", "cluster_top1": "right family"}
    accuracy_data = (
        cells.select("variant", "variant_label", *_metrics)
        .unpivot(index=["variant", "variant_label"], variable_name="metric", value_name="accuracy")
        .with_columns(pl.col("metric").replace_strict(_metrics).alias("metric_label"))
    )
    _p = cells.filter(pl.col("variant") == PRIMARY).row(0, named=True)
    _others = cells.filter(pl.col("variant") != PRIMARY)
    _gain = (
        " Without the projection (`plain`) the same raws give "
        + ", ".join(
            f"{r['top1']:.3f} exact and {r['cluster_top1']:.3f} family"
            for r in _others.iter_rows(named=True)
        )
        + f", so the projection contributes {_p['top1'] - float(_others['top1'][0]):+.3f} exact and "
        f"{_p['cluster_top1'] - float(_others['cluster_top1'][0]):+.3f} family: a small, consistent gain."
        if _others.height
        else ""
    )

    accuracy = (
        alt.Chart(accuracy_data)
        .mark_bar()
        .encode(
            x=alt.X("metric_label:N", title=None, sort=list(_metrics.values()), axis=alt.Axis(labelAngle=0)),
            y=alt.Y("accuracy:Q", title="accuracy on held-out stories", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("variant_label:N", title="recentering"),
            xOffset=alt.XOffset("variant_label:N"),
            tooltip=["variant_label", "metric_label", alt.Tooltip("accuracy:Q", format=".3f")],
        )
        .properties(width=360, height=240)
    )
    save_chart(
        accuracy,
        "readout_accuracy",
        caption=(
            "Readout accuracy of the paper-corpus emotion vectors on held-out paper-corpus "
            "stories (20 per emotion, never pooled into a vector), under the paper's "
            "recentering (neutral-transcript principal components projected out) and "
            "without the projection. Each emotion's projection is standardized across the "
            f"test set before ranking, as the tag pipeline does. Chance top-1 is {chance:.4f}."
        ),
        takeaway=(
            f"The canonical vectors find the exact emotion for {_p['top1']:.3f} of held-out "
            f"stories ({_p['top1'] / chance:.0f}x chance), place it in the top 5 of "
            f"{int(_p['n_vectors'])} for {_p['top5']:.3f}, and land in the right family for "
            f"{_p['cluster_top1']:.3f}; the median rank of the true emotion is "
            f"{int(_p['median_rank'])}." + _gain
        ),
        notebook=__file__,
    )
    return (accuracy_data,)


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
def _(PRIMARY, VARIANT_LABEL, alt, n_vectors, pl, readouts, save_chart):
    story_rows = pl.DataFrame(
        [
            {
                "variant": variant,
                "variant_label": VARIANT_LABEL.get(variant, variant),
                "emotion": s["emotion"],
                "cluster": s.get("true_cluster"),
                "topic": s.get("topic"),
                "rank": s["rank"],
                "z_margin": s["z_margin"],
                "correct": s["correct"],
                "cluster_correct": s["cluster_correct"],
            }
            for variant, art in readouts.items()
            for s in art["stories"]
            if "rank" in s
        ]
    )
    _ks = sorted({1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, n_vectors})
    curve_data = pl.DataFrame(
        [
            {
                "variant_label": grp["variant_label"][0],
                "k": k,
                "within_k": float((grp["rank"] <= k).mean()),
            }
            for _, grp in story_rows.group_by("variant")
            for k in _ks
        ]
    )
    _primary = story_rows.filter(pl.col("variant") == PRIMARY)
    _within = {k: float((_primary["rank"] <= k).mean()) for k in (1, 5, 10, 20)}

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
            color=alt.Color("variant_label:N", title="recentering"),
            tooltip=["variant_label", "k", alt.Tooltip("within_k:Q", format=".3f")],
        )
        .properties(width=440, height=280)
    )
    save_chart(
        curves,
        "rank_curves",
        caption=(
            "Share of held-out stories whose true emotion falls within the top k emotion "
            "vectors by standardized projection, as k runs over the taxonomy (log scale), "
            "for the canonical recentering and the no-projection diagnostic."
        ),
        takeaway=(
            f"For the canonical vectors the true emotion is first for {_within[1]:.0%} of "
            f"held-out stories, in the top 5 of {n_vectors} for {_within[5]:.0%}, in the top "
            f"10 for {_within[10]:.0%} and in the top 20 for {_within[20]:.0%}; the median "
            f"rank is {int(_primary['rank'].median())}. The two recenterings trace nearly "
            "the same curve, so the projection changes little about where a miss lands."
        ),
        notebook=__file__,
    )
    return curve_data, story_rows


@app.cell
def _(mo):
    mo.md(
        """
        ### Which families the vectors can find at all

        Accuracy averaged over the whole taxonomy hides that the 171 emotions are not
        equally legible. This is family-level accuracy per family — did the top-scoring
        vector land in the right one of the ten — for the canonical vectors on the held-out
        stories.
        """
    )
    return


@app.cell
def _(PRIMARY, alt, pl, readouts, save_chart):
    _by_family = {}
    for _r in readouts[PRIMARY]["per_emotion"]:
        _by_family.setdefault(_r["cluster"], []).append(_r)
    family_data = pl.DataFrame(
        [
            {
                "family": family.replace("_", " "),
                "n_emotions": len(rs),
                "family_accuracy": sum(r["cluster_top1"] for r in rs) / len(rs),
                "exact": sum(r["top1"] for r in rs) / len(rs),
            }
            for family, rs in _by_family.items()
        ]
    ).sort("family_accuracy", descending=True)
    _n_total = int(family_data["n_emotions"].sum())
    _chance_family = float(
        (family_data["n_emotions"] * family_data["n_emotions"] / _n_total).sum() / _n_total
    )
    _top = family_data.row(0, named=True)
    _bottom = family_data.row(-1, named=True)
    _anger = family_data.filter(pl.col("family") == "hostile anger")

    families = (
        alt.Chart(family_data)
        .mark_bar()
        .encode(
            y=alt.Y("family:N", title=None, sort=family_data["family"].to_list()),
            x=alt.X("family_accuracy:Q", title="family-level accuracy", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                "family", "n_emotions",
                alt.Tooltip("family_accuracy:Q", format=".3f"),
                alt.Tooltip("exact:Q", format=".3f"),
            ],
        )
        .properties(width=340, height=280)
    )
    save_chart(
        families,
        "family_legibility",
        caption=(
            "Family-level accuracy per emotion family for the canonical vectors reading "
            "held-out stories: the share of that family's stories whose top-scoring vector "
            f"falls in the correct family. Chance is {_chance_family:.3f}."
        ),
        takeaway=(
            f"Every family is found well above the {_chance_family:.3f} chance rate: family "
            f"accuracy runs from {_bottom['family_accuracy']:.2f} ({_bottom['family']}) to "
            f"{_top['family_accuracy']:.2f} ({_top['family']})"
            + (
                f", with hostile anger at {float(_anger['family_accuracy'][0]):.2f}"
                if _anger.height
                else ""
            )
            + ". Rankings are standardized per emotion; an unstandardized argmax made anger "
            "look unfindable, an artifact of per-emotion projection offsets."
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
        `02-elicited-activations`, read at the pre-response token and projected onto the
        canonical vectors (that experiment's `data/qwen3.5-9b/readout.json`), scored
        against the emotion each message was elicited for — a weak target, since a message
        need not evoke the emotion it was written to evoke, but the one that exists.
        """
    )
    return


@app.cell
def _(HERE, alt, json, np, pl, save_chart):
    from name_that_feeling.emotion_vectors.taxonomy import (
        emotion_to_cluster as _emotion_to_cluster,
        load_clusters as _load_clusters,
        slugify as _slugify,
    )

    _readout = json.loads(
        (HERE.parent / "02-elicited-activations" / "data" / "qwen3.5-9b" / "readout.json").read_text(
            encoding="utf-8"
        )
    )
    _msgs = _readout["messages"]
    _e2c = {_slugify(k): v for k, v in _emotion_to_cluster(_load_clusters()).items()}
    _names = list(_msgs[0]["projections"])
    _P = np.array([[m["projections"][n] for n in _names] for m in _msgs])
    # Standardize each emotion across the pool before ranking, exactly as the tag
    # pipeline does (generation.sft.per_emotion_stats): the raw projection carries a
    # per-emotion offset larger than the signal, and a raw argmax ranks offsets.
    _Z = (_P - _P.mean(0)) / (_P.std(0) + 1e-12)
    _order_desc = np.argsort(-_Z, axis=1)
    _name_idx = {n: i for i, n in enumerate(_names)}
    msg_rows = []
    for _m, _row in zip(_msgs, _order_desc):
        _true = _slugify(_m["emotion"])
        _true_fam = _e2c[_true]
        _rank = int(np.where(_row == _name_idx[_true])[0][0]) + 1 if _true in _name_idx else None
        msg_rows.append(
            {
                "family": _true_fam.replace("_", " "),
                "family_correct": _e2c[_names[_row[0]]] == _true_fam,
                "exact": _rank == 1,
                "top5": _rank is not None and _rank <= 5,
                "rank": _rank,
            }
        )
    _all = pl.DataFrame(msg_rows)
    message_pool = (
        _all.group_by("family")
        .agg(pl.col("family_correct").mean().alias("family_accuracy"), pl.len().alias("n_messages"))
        .sort("n_messages", descending=True)
    )
    _fam_size = {}
    for _c in _e2c.values():
        _fam_size[_c] = _fam_size.get(_c, 0) + 1
    _chance_family = float(np.mean([_fam_size[_e2c[_slugify(m["emotion"])]] / len(_names) for m in _msgs]))
    _summary = {
        "family": float(_all["family_correct"].mean()),
        "exact": float(_all["exact"].mean()),
        "top5": float(_all["top5"].mean()),
        "median_rank": float(_all["rank"].median()),
        "mean_rank": float(_all["rank"].mean()),
    }

    message_chart = (
        alt.Chart(message_pool)
        .mark_bar()
        .encode(
            y=alt.Y("family:N", title=None, sort=message_pool["family"].to_list()),
            x=alt.X("family_accuracy:Q", title="family-level accuracy", scale=alt.Scale(domain=[0, 1])),
            tooltip=["family", "n_messages", alt.Tooltip("family_accuracy:Q", format=".3f")],
        )
        .properties(width=340, height=320)
    )
    save_chart(
        message_chart,
        "message_pool_families",
        caption=(
            f"Family-level accuracy on the {len(_msgs):,} direct-elicitation messages, read "
            "at the pre-response token and projected onto the canonical vectors, each "
            "emotion standardized across the pool before ranking. Families are ordered by "
            f"how many messages they hold; chance is {_chance_family:.3f}."
        ),
        takeaway=(
            f"On the pool the probe is pointed at, the vectors put the elicited emotion's "
            f"family first for {_summary['family']:.3f} of messages ({_chance_family:.3f} "
            f"chance), the exact emotion first for {_summary['exact']:.3f} "
            f"({1 / len(_names):.3f} chance) and in the top 5 for {_summary['top5']:.3f}; "
            f"the median rank of the elicited emotion is {int(_summary['median_rank'])} of "
            f"{len(_names)}. Far below the story readout, as expected of a target the "
            "message may not actually evoke; the family read is the usable one."
        ),
        notebook=__file__,
    )
    return message_pool, msg_rows


@app.cell
def _(mo):
    mo.md("### The numbers")
    return


@app.cell
def _(cells):
    cells.select(
        "variant", "top1", "top5", "cluster_top1", "mean_rank", "median_rank",
        "mean_z_margin", "chance_top1", "n_scored",
    )
    return


if __name__ == "__main__":
    app.run()
