import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import re
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    return Path, alt, json, mo, pl, re


@app.cell
def _(re):
    def slug(s: str) -> str:
        """Match an emotion label to its projection key (e.g. 'self-confident' -> 'self_confident')."""
        return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

    return (slug,)


@app.cell
def _(Path, json):
    readout = json.loads(
        (Path(__file__).parent / "data" / "readout.json").read_text(
            encoding="utf-8"
        )
    )
    # Existential probes are a topic-OOD eval, not part of this readout's view.
    messages = [m for m in readout["messages"] if m.get("eval_axis") != "existential"]
    return messages, readout


@app.cell
def _(messages, mo, readout):
    mo.md(f"""
    ## Emotion activations per message

    **{len(messages)}** messages (existential probes excluded) × **{readout["n_emotion_vectors"]}** emotion vectors ·
    model `{readout["model"]}` · readout layer **{readout["readout_layer"]}** · position `{readout["position"]}`.

    Each value is the message's assistant-colon activation projected onto an emotion vector
    ({readout["projection"]}). Filter by **cluster**, then **emotion**, then pick a message —
    **red** marks the emotion it was written to target.
    """)
    return


@app.cell
def _(messages, mo):
    clusters = sorted({m["cluster"] for m in messages})
    cluster_selector = mo.ui.dropdown(
        options=clusters, value=clusters[0], label="cluster"
    )
    cluster_selector
    return (cluster_selector,)


@app.cell
def _(cluster_selector, messages, mo):
    cluster_emotions = sorted(
        {m["emotion"] for m in messages if m["cluster"] == cluster_selector.value}
    )
    emotion_selector = mo.ui.dropdown(
        options=cluster_emotions, value=cluster_emotions[0], label="emotion"
    )
    emotion_selector
    return (emotion_selector,)


@app.cell
def _(cluster_selector, emotion_selector, messages, mo):
    options = {
        f"{m['id']} · {m.get('eval_axis') or m['split']}": i
        for i, m in enumerate(messages)
        if m["cluster"] == cluster_selector.value
        and m["emotion"] == emotion_selector.value
    }
    message_selector = mo.ui.dropdown(
        options=options,
        value=next(iter(options)),
        label="message",
        searchable=True,
    )
    message_selector
    return (message_selector,)


@app.cell
def _(message_selector, messages, mo):
    _m = messages[message_selector.value]
    mo.md(f"**message** `{_m['id']}`\n\n> {_m['message'].replace(chr(10), ' ')}")
    return


@app.cell
def _(alt, message_selector, messages, mo, pl, slug):
    msg = messages[message_selector.value]
    intended = slug(msg["emotion"])
    ranked = sorted(msg["projections"].items(), key=lambda kv: kv[1], reverse=True)
    rank = next((i + 1 for i, (e, _v) in enumerate(ranked) if e == intended), None)

    top = ranked[:20]
    df = pl.DataFrame(
        {"emotion": [e for e, _v in top], "activation": [v for _e, v in top]}
    ).with_columns(is_intended=pl.col("emotion") == intended)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("activation:Q", title="emotion-vector activation"),
            y=alt.Y("emotion:N", sort="-x", title=None),
            color=alt.condition(
                "datum.is_intended", alt.value("crimson"), alt.value("#4c78a8")
            ),
            tooltip=["emotion", "activation"],
        )
        .properties(height=400, width=560, title="Top-20 emotion activations")
    )

    header = mo.md(
        f"**target:** `{msg['emotion']}` · cluster `{msg['cluster']}` · frame {msg['frame']} · "
        f"split {msg.get('eval_axis') or msg['split']} · "
        f"**probe rank of target: {rank} / {len(msg['projections'])}**"
    )
    mo.vstack([header, chart])
    return


@app.cell
def _(
    alt,
    cluster_order,
    emo_to_cluster,
    emotion_order,
    message_selector,
    messages,
    mo,
    pl,
    slug,
):
    _m = messages[message_selector.value]
    _intended = slug(_m["emotion"])
    _rows = [
        (emo_to_cluster[e], e, v)
        for e, v in _m["projections"].items()
        if e in emo_to_cluster
    ]
    pm = pl.DataFrame(
        _rows, schema=["cluster", "emotion", "activation"], orient="row"
    ).with_columns(is_intended=pl.col("emotion") == _intended)

    pm_chart = (
        alt.Chart(pm)
        .mark_rect()
        .encode(
            x=alt.X(
                "emotion:N",
                sort=emotion_order,
                title=None,
                axis=alt.Axis(labelAngle=-45, labelLimit=90),
            ),
            color=alt.Color(
                "activation:Q",
                title="activation",
                scale=alt.Scale(scheme="redblue", reverse=True, domainMid=0),
            ),
            stroke=alt.condition(
                "datum.is_intended", alt.value("black"), alt.value(None)
            ),
            strokeWidth=alt.condition(
                "datum.is_intended", alt.value(2.5), alt.value(0)
            ),
            tooltip=[
                "cluster",
                "emotion",
                alt.Tooltip("activation:Q", format=".3f"),
            ],
        )
        .properties(width=680, height=10)
        .facet(
            row=alt.Row(
                "cluster:N",
                sort=cluster_order,
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left", labelLimit=200),
            ),
            spacing=0.5,
        )
        .resolve_scale(x="independent", color="shared")
    )

    mo.vstack(
        [
            mo.md(
                f"`{_m['id']}` · target **{_m['emotion']}** (cluster `{_m['cluster']}`, outlined in black). "
            ),
            pm_chart,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Dataset overview — activation by cluster

    Across all messages: rows are the **cluster the message was written to target**,
    columns are the **171 emotion vectors** (ordered by their own cluster). Color is the
    mean activation of that cluster's messages on each vector. Cells whose vector belongs
    to the row's cluster — the **"right" emotions** — are outlined in black; if the probe
    works, the warm colour should fall inside the outlined staircase.

    **Colour toggle:** *raw mean* is the average dot-product onto each vector, on its
    native scale — but some vectors have systematically large dot products, so they read
    warm in **every** row and show up as vertical stripes that drown out the signal.
    *z-score per emotion* fixes this by standardising each vector's column across all
    messages (subtract that vector's mean over messages, divide by its std), so every
    column is centred at 0 and a cell answers **"do this cluster's messages activate this
    vector more than an average message does?"** — which is what makes the diagonal block
    pop. Use z-score to read the structure; raw mean to see absolute magnitudes.
    """)
    return


@app.cell
def _(mo):
    norm_toggle = mo.ui.radio(
        options=["z-score per emotion", "raw mean"],
        value="z-score per emotion",
        label="heatmap colour",
        inline=True,
    )
    norm_toggle
    return (norm_toggle,)


@app.cell
def _(Path, json, messages, pl, slug):
    _taxonomy = json.loads(
        (
            Path(__file__).parents[1] / "01-emotion-vectors" / "clusters.json"
        ).read_text(encoding="utf-8")
    )
    cluster_order = list(_taxonomy.keys())
    emotion_order = [slug(e) for c in cluster_order for e in _taxonomy[c]]
    emo_to_cluster = dict(
        zip(emotion_order, [c for c in cluster_order for _e in _taxonomy[c]])
    )
    _emo_cluster = pl.DataFrame(
        {
            "emotion": emotion_order,
            "emotion_cluster": list(emo_to_cluster.values()),
        }
    )

    _rows = [
        (m["cluster"], e, v) for m in messages for e, v in m["projections"].items()
    ]
    _long = pl.DataFrame(
        _rows, schema=["msg_cluster", "emotion", "value"], orient="row"
    ).with_columns(
        z=(pl.col("value") - pl.col("value").mean().over("emotion"))
        / (pl.col("value").std().over("emotion") + 1e-8)
    )
    heat = (
        _long.group_by("msg_cluster", "emotion")
        .agg(pl.col("z").mean().alias("z"), pl.col("value").mean().alias("raw"))
        .join(_emo_cluster, on="emotion", how="inner")
        .with_columns(
            is_correct=pl.col("msg_cluster") == pl.col("emotion_cluster")
        )
    )
    return cluster_order, emo_to_cluster, emotion_order, heat


@app.cell
def _(alt, cluster_order, emotion_order, heat, norm_toggle):
    _field = "z" if norm_toggle.value.startswith("z") else "raw"
    _base = alt.Chart(heat)
    _heatmap = _base.mark_rect().encode(
        x=alt.X(
            "emotion:N",
            sort=emotion_order,
            title="emotion vector (ordered by cluster)",
            axis=alt.Axis(labels=False, ticks=False),
        ),
        y=alt.Y(
            "msg_cluster:N", sort=cluster_order, title="message target cluster"
        ),
        color=alt.Color(
            f"{_field}:Q",
            title="activation",
            scale=alt.Scale(scheme="redblue", reverse=True, domainMid=0),
        ),
        tooltip=[
            alt.Tooltip("msg_cluster:N", title="target cluster"),
            alt.Tooltip("emotion:N"),
            alt.Tooltip("emotion_cluster:N", title="vector cluster"),
            alt.Tooltip(f"{_field}:Q", title="activation", format=".2f"),
        ],
    )
    _outline = (
        _base.transform_filter("datum.is_correct")
        .mark_rect(fillOpacity=0, stroke="black", strokeWidth=0.6)
        .encode(
            x=alt.X("emotion:N", sort=emotion_order),
            y=alt.Y("msg_cluster:N", sort=cluster_order),
        )
    )
    (_heatmap + _outline).properties(
        width=760,
        height=340,
        title="Mean emotion-vector activation by target cluster — correct-cluster cells outlined",
    )
    return


if __name__ == "__main__":
    app.run()
