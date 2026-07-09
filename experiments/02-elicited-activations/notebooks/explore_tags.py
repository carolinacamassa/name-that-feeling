import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from collections import Counter
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
    from name_that_feeling.generation import sft

    alt.data_transformers.disable_max_rows()
    return Counter, Path, alt, json, load_clusters, mo, pl, sft, slugify


@app.cell
def _(mo):
    mo.md("""
    # Is the per-message probe a good `<emotion>`-tag source?

    **Setup.** Each training example should open with an `<emotion>` tag naming what the assistant
    feels. One way to fill that tag is the **emotion probe**: for a message we run the model, read
    its internal activation, and take that activation's **projection** (dot-product) onto each of
    **171 pre-learned "emotion vectors"** — one score per emotion. We'd then tag the message with
    whatever emotion(s) score highest, so the label reflects the model's *actual internal state*
    rather than the emotion the scenario was written to evoke (elicitation is imperfect — grounding
    in the activation is the whole point).

    **Question.** Is that per-message read strong enough to be a training label? Checked here on
    **all 1972** `02-elicited-activations` messages (the direct-elicitation, "make-the-assistant-feel"
    set).

    **How we check it.** Our only reference for "what emotion is really here" is the emotion each
    message was *elicited for*. We never use that label to *build* the tag — only as a yardstick: if
    the probe's top pick lands in the same emotion **cluster** (family) as the elicited emotion far
    more often than random guessing would, the probe reads something real; if it matches only at the
    random rate, the probe is effectively reading noise.

    The five concerns build up the verdict:

    1. **Per-emotion offsets** — the 171 raw scores aren't on a comparable scale; some vectors read
       high on *every* message. So we standardize (z-score) each emotion first.
    2. **Elevated but not decisive** — after standardizing, the elicited emotion does score above
       its own average, but is almost never the single highest of the 171. The probe *leans*, it
       doesn't *decide*.
    3. **Pooling to families** — collapsing 171 emotions to 10 clusters needs a pooling rule
       (average vs. take-the-max); which rule reads best is unstable across datasets.
    4. **Agreement vs. chance** — the headline: how often the top tag matches the elicited family,
       measured against a "guess the biggest family" baseline (the set is imbalanced, so that
       baseline is high). Interactive.
    5. **The grounding floor** — how much label–state agreement a related result says you need
       before this kind of tag grounds anything, and how far short we fall.
    """)
    return


@app.cell
def _(Path, json, load_clusters):
    READOUT = Path(__file__).parents[1] / "data" / "qwen3.5-9b" / "readout.json"
    CLUSTERS = (
        Path(__file__).parents[2] / "01-emotion-vectors" / "clusters.json"
    )

    readout = json.loads(READOUT.read_text(encoding="utf-8"))
    messages = readout["messages"]  # all data
    clusters = load_clusters(CLUSTERS)
    cluster_order = list(clusters.keys())
    return cluster_order, clusters, messages, readout


@app.cell
def _(clusters, messages, sft, slugify):
    emo2cluster = {slugify(e): c for c, es in clusters.items() for e in es}
    reps = {
        slugify(es[0]) for es in clusters.values()
    }  # one representative emotion per cluster

    records = [
        {
            "id": m["id"],
            "scenario": {
                "message": m["message"],
                "emotion": m["emotion"],
                "cluster": m["cluster"],
            },
            "probe": {"projections": m["projections"]},
        }
        for m in messages
    ]
    stats = sft.per_emotion_stats(
        records
    )  # per-emotion (mean, std) across all messages
    return emo2cluster, records, reps, stats


@app.cell
def _(messages, mo, readout):
    mo.md(f"""
    Loaded **{len(messages)}** messages × **{readout["n_emotion_vectors"]}** emotion vectors ·
    model `{readout["model"]}` · readout layer **{readout["readout_layer"]}** ·
    position `{readout["position"]}`.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Concern 1 — the 171 raw scores aren't on a comparable scale

    Each bar is one emotion vector's **mean projection** — its average dot-product with the model's
    activation, across all messages. If the projection were an unbiased "how much of this emotion is
    present" score, every emotion's average would sit near the same value. It doesn't: some vectors
    read systematically high (or low) on *every* message, whatever the message is about.

    That offset means you can't compare emotions by raw score — picking the highest raw projection
    just favors the high-offset vectors every time. The fix, used everywhere below, is to **z-score
    per emotion**: subtract each emotion's own mean and divide by its own spread (across all
    messages), so every emotion is centered at 0 and a score answers "is this emotion *more* active
    on this message than it usually is?"
    """)
    return


@app.cell
def _(alt, emo2cluster, pl, stats):
    _off = pl.DataFrame(
        {
            "emotion": list(stats.keys()),
            "mean_projection": [mu for mu, _sd in stats.values()],
            "cluster": [emo2cluster.get(e) for e in stats.keys()],
        }
    ).sort("mean_projection", descending=True)

    alt.Chart(_off).mark_bar().encode(
        x=alt.X("mean_projection:Q", title="mean dot-product across all messages"),
        y=alt.Y(
            "emotion:N",
            sort="-x",
            title=None,
            axis=alt.Axis(labels=False, ticks=False),
        ),
        color=alt.Color("cluster:N", legend=None),
        tooltip=[
            "emotion",
            "cluster",
            alt.Tooltip("mean_projection:Q", format=".2f"),
        ],
    ).properties(
        height=360,
        width=560,
        title="Per-emotion offset (171 vectors) — some read high everywhere",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Concern 2 — the right emotion is elevated, but almost never the top pick

    For each message, take the emotion it was *elicited for* (the "target") and read its z-scored
    projection two ways:

    - **target z-score** — how far above its own average the target reads on this message. Mass to
      the right of 0 means the probe leans toward the right emotion.
    - **target rank** — where the target sits when all 171 emotions are sorted by z-score (rank 1 =
      the probe's single top pick, the *argmax*), and, more coarsely, where its *family* ranks among
      the 10 clusters (scoring each family by its members' average z).

    The pattern in the summary below: the target is clearly elevated on average, yet is the #1 pick
    only a few percent of the time. So the probe carries a real *directional* signal but is a poor
    *classifier* — it nudges toward the right emotion without singling it out.
    """)
    return


@app.cell
def _(emo2cluster, records, slugify, stats):
    def _zscore(projections):
        return {
            e: (v - stats[e][0]) / stats[e][1]
            for e, v in projections.items()
            if e in stats
        }


    def _cluster_mean_z(z):
        by = {}
        for e, v in z.items():
            by.setdefault(emo2cluster[e], []).append(v)
        return {c: sum(vs) / len(vs) for c, vs in by.items()}


    diag = []
    for _r in records:
        _t = slugify(_r["scenario"]["emotion"])
        _tc = _r["scenario"]["cluster"]
        _z = _zscore(_r["probe"]["projections"])
        if _t not in _z:
            continue
        _ranked = sorted(_z.items(), key=lambda kv: kv[1], reverse=True)
        _emo_rank = next(i + 1 for i, (e, _v) in enumerate(_ranked) if e == _t)
        _cz = _cluster_mean_z(_z)
        _cranked = sorted(_cz.items(), key=lambda kv: kv[1], reverse=True)
        _clu_rank = next(i + 1 for i, (c, _v) in enumerate(_cranked) if c == _tc)
        diag.append(
            {
                "id": _r["id"],
                "cluster": _tc,
                "target_z": _z[_t],
                "emotion_rank": _emo_rank,
                "cluster_rank": _clu_rank,
            }
        )
    return (diag,)


@app.cell
def _(alt, cluster_order, diag, mo, pl):
    _d = pl.DataFrame(diag)
    _n = _d.height
    _summary = mo.md(
        f"""
        **{_n}** messages · target z-score: mean **{_d["target_z"].mean():.2f}σ**, positive on
        **{(_d["target_z"] > 0).mean():.0%}** · target is the argmax over 171 on only
        **{(_d["emotion_rank"] == 1).mean():.0%}** (median rank **{int(_d["emotion_rank"].median())}**) ·
        target *cluster* is top-1 on **{(_d["cluster_rank"] == 1).mean():.0%}** (median rank
        **{int(_d["cluster_rank"].median())}** / 10).
        """
    )
    # Colour the leaning-right mass (z > 0) by the target's cluster; grey out z <= 0.
    _tableau10 = [
        "#4c78a8",
        "#f58518",
        "#e45756",
        "#72b7b2",
        "#54a24b",
        "#eeca3b",
        "#b279a2",
        "#ff9da6",
        "#9d755d",
        "#bab0ac",
    ]
    _dz = _d.with_columns(
        lean=pl.when(pl.col("target_z") > 0)
        .then(pl.col("cluster"))
        .otherwise(pl.lit("z ≤ 0"))
    )
    _zhist = (
        alt.Chart(_dz)
        .mark_bar()
        .encode(
            x=alt.X(
                "target_z:Q",
                bin=alt.Bin(maxbins=40),
                title="target emotion z-score",
            ),
            y=alt.Y("count()", title="messages"),
            color=alt.Color(
                "lean:N",
                scale=alt.Scale(
                    domain=cluster_order + ["z ≤ 0"],
                    range=_tableau10[: len(cluster_order)] + ["#d9d9d9"],
                ),
                legend=alt.Legend(title="cluster (where z > 0)"),
            ),
            tooltip=["lean:N", alt.Tooltip("count()", title="messages")],
        )
        .properties(
            width=360,
            height=240,
            title="Target z — coloured where it leans the right way (z > 0)",
        )
    )
    _rhist = (
        alt.Chart(_d)
        .transform_joinaggregate(total="count()")
        .transform_calculate(pct="1 / datum.total")
        .mark_bar()
        .encode(
            x=alt.X(
                "emotion_rank:Q",
                bin=alt.Bin(maxbins=40),
                title="target rank among 171 (1 = argmax)",
            ),
            y=alt.Y(
                "sum(pct):Q",
                title="% of messages",
                axis=alt.Axis(format="%"),
            ),
        )
        .properties(width=320, height=220, title="Target rank (mass NOT at 1)")
    )
    mo.vstack([_summary, alt.hconcat(_zhist, _rhist)])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Concern 3 — collapsing 171 emotions to 10 families needs a pooling rule

    The tag should name a few emotions, so we first score each of the 10 **clusters** (families) from
    its members' z-scores. Two natural rules, each with a failure mode:

    - **max-pool** — a family's score is its single highest-scoring member. But the maximum of ~17
      noisy numbers is high (≈ +1.8σ) *for every family* just by chance, so families look alike.
    - **mean-pool** — a family's score is the average of its members. This averages the noise out,
      but **dilutes** a real signal: one or two genuinely-elevated emotions get averaged in with ~15
      unrelated members sitting near 0.

    Which failure hurts less is **empirical and unstable**. On this elicited set mean-pool's dilution
    hurts more, so **max-pool and emotion-level (no pooling) actually win** (Concern 4) — the opposite
    of the curated-scenario set, where mean-pool won. The chart below shows each rule's average score
    for the *correct* family vs. the average over *wrong* families; the decision-relevant number,
    though, is the argmax agreement in Concern 4.
    """)
    return


@app.cell
def _(alt, emo2cluster, pl, records, stats):
    def _cluster_scores(projections, pool):
        z = {
            e: (v - stats[e][0]) / stats[e][1]
            for e, v in projections.items()
            if e in stats
        }
        by = {}
        for e, v in z.items():
            by.setdefault(emo2cluster[e], []).append(v)
        return {
            c: (max(vs) if pool == "max" else sum(vs) / len(vs))
            for c, vs in by.items()
        }


    _sep = []
    for _pool in ("max", "mean"):
        _corr, _wrong = [], []
        for _r in records:
            _cs = _cluster_scores(_r["probe"]["projections"], _pool)
            _tc = _r["scenario"]["cluster"]
            _corr.append(_cs[_tc])
            _wrong.append(
                sum(v for c, v in _cs.items() if c != _tc) / (len(_cs) - 1)
            )
        _sep.append(
            {
                "pool": _pool,
                "kind": "correct cluster",
                "score": sum(_corr) / len(_corr),
            }
        )
        _sep.append(
            {
                "pool": _pool,
                "kind": "wrong clusters (avg)",
                "score": sum(_wrong) / len(_wrong),
            }
        )

    alt.Chart(pl.DataFrame(_sep)).mark_bar().encode(
        x=alt.X("pool:N", title=None),
        y=alt.Y("score:Q", title="mean cluster score (z)"),
        color=alt.Color(
            "kind:N", scale=alt.Scale(range=["crimson", "#4c78a8"]), title=None
        ),
        xOffset="kind:N",
        tooltip=["pool", "kind", alt.Tooltip("score:Q", format=".2f")],
    ).properties(
        width=300,
        height=260,
        title="Correct vs. wrong-family score, by pooling rule",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Concern 4 — agreement with intent vs. a chance baseline (all data)

    **Agreement** = the top tag emotion's **cluster** equals the **elicited cluster**, over all
    messages — our proxy for "did the probe pick the right family?". Read it against **chance**: the
    score from ignoring the probe and *always guessing the single most common family*. This set is
    **imbalanced** (one family is ~27% of it), so that dumb baseline is already high — a strategy has
    to clear ~27% just to beat "always guess the biggest".

    The static table sweeps the main strategies (chance is the first row). The controls below
    recompute live and add:

    - the **confusion matrix** — rows are the elicited family, columns the probe's predicted family,
      each row normalized to 1. A strong diagonal means the probe tracks intent; this per-family view
      is fairer than the single headline %, which the imbalance inflates.
    - the **tag-length distribution** — how many emotions the tag ends up with under the current
      mass-threshold / max-N settings.
    """)
    return


@app.cell
def _(Counter, emo2cluster, sft, slugify):
    def evaluate(records, clusters, stats, **strategy):
        c_hits = e_hits = 0
        wsum = 0.0
        lens = Counter()
        pairs = []
        for r in records:
            picks = sft.select_tag_emotions(
                r["probe"]["projections"], clusters, stats=stats, **strategy
            )
            lens[len(picks)] += 1
            if not picks:
                continue
            top = picks[0][0]
            wsum += picks[0][1]
            pc = emo2cluster.get(top)
            tc = r["scenario"]["cluster"]
            if pc == tc:
                c_hits += 1
            if top == slugify(r["scenario"]["emotion"]):
                e_hits += 1
            pairs.append((tc, pc))
        n = len(records)
        return {
            "cluster_agree": c_hits / n,
            "emotion_agree": e_hits / n,
            "mean_top_w": wsum / n,
            "lens": dict(sorted(lens.items())),
            "pairs": pairs,
        }

    return (evaluate,)


@app.cell
def _(Counter, clusters, evaluate, pl, records, reps, stats):
    _chance = Counter(r["scenario"]["cluster"] for r in records).most_common(1)[0][
        1
    ] / len(records)
    _strategies = {
        "chance (modal cluster)": None,
        "cluster / max  T=1.0": dict(
            granularity="cluster", pool="max", temperature=1.0
        ),
        "cluster / mean T=1.0": dict(
            granularity="cluster", pool="mean", temperature=1.0
        ),
        "cluster / mean T=0.5": dict(
            granularity="cluster", pool="mean", temperature=0.5
        ),
        "emotion (all 171) T=0.5": dict(granularity="emotion", temperature=0.5),
        "emotion (10 reps) T=0.5": dict(
            granularity="emotion", temperature=0.5, candidates=reps
        ),
    }
    _rows = []
    for _name, _s in _strategies.items():
        if _s is None:
            _rows.append(
                {
                    "strategy": _name,
                    "cluster_agree": _chance,
                    "emotion_agree": None,
                    "mean_top_w": None,
                }
            )
            continue
        _e = evaluate(records, clusters, stats, **_s)
        _rows.append(
            {
                "strategy": _name,
                "cluster_agree": _e["cluster_agree"],
                "emotion_agree": _e["emotion_agree"],
                "mean_top_w": _e["mean_top_w"],
            }
        )
    pl.DataFrame(_rows).with_columns(
        pl.col("cluster_agree").round(3),
        pl.col("emotion_agree").round(3),
        pl.col("mean_top_w").round(3),
    )
    return


@app.cell
def _(mo):
    pool_ui = mo.ui.dropdown(options=["mean", "max"], value="mean", label="pool")
    gran_ui = mo.ui.dropdown(
        options=["cluster", "emotion"], value="cluster", label="granularity"
    )
    temp_ui = mo.ui.slider(
        start=0.2, stop=2.0, step=0.1, value=0.5, label="softmax temperature"
    )
    mass_ui = mo.ui.slider(
        start=0.5, stop=0.95, step=0.05, value=0.8, label="mass threshold"
    )
    maxn_ui = mo.ui.slider(start=1, stop=4, step=1, value=3, label="max N")
    mo.hstack([pool_ui, gran_ui, temp_ui, mass_ui, maxn_ui], justify="start")
    return gran_ui, mass_ui, maxn_ui, pool_ui, temp_ui


@app.cell
def _(
    clusters,
    evaluate,
    gran_ui,
    mass_ui,
    maxn_ui,
    mo,
    pool_ui,
    records,
    stats,
    temp_ui,
):
    live = evaluate(
        records,
        clusters,
        stats,
        granularity=gran_ui.value,
        pool=pool_ui.value,
        temperature=temp_ui.value,
        mass_threshold=mass_ui.value,
        max_n=maxn_ui.value,
        min_n=1,
    )
    live_summary = mo.md(
        f"**cluster agreement {live['cluster_agree']:.0%}** · emotion agreement "
        f"{live['emotion_agree']:.0%} · mean top-weight {live['mean_top_w']:.2f} · "
        f"tag-length counts {live['lens']}"
    )
    live_summary
    return (live,)


@app.cell
def _(alt, cluster_order, live, mo, pl):
    _counts = {}
    for _tc, _pc in live["pairs"]:
        _counts[(_tc, _pc)] = _counts.get((_tc, _pc), 0) + 1
    _totals = {}
    for (_tc, _pc), _k in _counts.items():
        _totals[_tc] = _totals.get(_tc, 0) + _k
    _rows = [
        {"target": _tc, "predicted": _pc, "frac": _k / _totals[_tc]}
        for (_tc, _pc), _k in _counts.items()
    ]
    _heat = (
        alt.Chart(pl.DataFrame(_rows))
        .mark_rect()
        .encode(
            x=alt.X(
                "predicted:N",
                sort=cluster_order,
                title="predicted cluster (top tag)",
            ),
            y=alt.Y("target:N", sort=cluster_order, title="elicited cluster"),
            color=alt.Color(
                "frac:Q", scale=alt.Scale(scheme="blues"), title="row frac"
            ),
            tooltip=["target", "predicted", alt.Tooltip("frac:Q", format=".2f")],
        )
        .properties(
            width=380,
            height=380,
            title="Confusion: strong diagonal = probe tracks intent",
        )
    )
    mo.vstack(
        [
            mo.md(
                "Off-diagonal mass = the probe tags a different cluster than the one elicited."
            ),
            _heat,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Concern 5 — how this compares to the "grounding floor"

    `introspective-coupling.md` summarizes a result (the paper's ν-sweep, H-C4): training a model to
    narrate its own state only starts to **couple** to the real state when the training labels agree
    with that state **above ~0.6–0.7**; below ~0.6 the effect collapses. Read as a rough bar for
    label quality, that's roughly the agreement a probe-grounded tag would need to reach to ground
    anything, rather than just teach the model to emit plausible-looking words.

    Our best per-message read here (~1.4× chance, ~10pp above it) sits far under that bar — so
    training on it as-is most likely installs *noise*, not a grounded channel. The chart marks where
    we are vs. where coupling would need us to be.
    """)
    return


@app.cell
def _(alt, clusters, evaluate, pl, records, stats):
    _cands = [
        dict(granularity="cluster", pool="mean", temperature=0.5),
        dict(granularity="cluster", pool="max", temperature=0.5),
        dict(granularity="emotion", temperature=0.5),
    ]
    _best = max(
        evaluate(records, clusters, stats, **_s)["cluster_agree"] for _s in _cands
    )
    _bars = pl.DataFrame(
        {
            "level": ["best observed strategy", "grounding floor (~0.65)"],
            "value": [_best, 0.65],
            "kind": ["current", "floor"],
        }
    )
    alt.Chart(_bars).mark_bar().encode(
        x=alt.X(
            "value:Q", title="cluster agreement", scale=alt.Scale(domain=[0, 1])
        ),
        y=alt.Y("level:N", title=None),
        color=alt.Color(
            "kind:N", scale=alt.Scale(range=["#4c78a8", "crimson"]), legend=None
        ),
        tooltip=[alt.Tooltip("value:Q", format=".0%")],
    ).properties(
        width=440, height=110, title="We sit well below the floor coupling needs"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Specific examples

    Pick a message, or read the curated set below (chosen by the data: the two most *confident
    correct* reads and the two most *confident wrong* ones, under cluster/mean/T=0.5). Shown: the
    message, the emotion it was elicited for, and the probe's top tag under the current controls.
    """)
    return


@app.cell
def _(
    clusters,
    gran_ui,
    mass_ui,
    maxn_ui,
    messages,
    mo,
    pool_ui,
    sft,
    stats,
    temp_ui,
):
    def show_example(idx):
        m = messages[idx]
        picks = sft.select_tag_emotions(
            m["projections"],
            clusters,
            stats=stats,
            granularity=gran_ui.value,
            pool=pool_ui.value,
            temperature=temp_ui.value,
            mass_threshold=mass_ui.value,
            max_n=maxn_ui.value,
            min_n=1,
        )
        weights = ", ".join(f"`{e}` {w:.2f}" for e, w in picks)
        return mo.md(
            f"**`{m['id']}`** · elicited **{m['emotion']}** (cluster `{m['cluster']}`)\n\n"
            f"> {m['message'].replace(chr(10), ' ')}\n\n"
            f"**probe tag** {sft.format_tag(picks)} &nbsp;·&nbsp; weights: {weights}"
        )

    return (show_example,)


@app.cell
def _(clusters, emo2cluster, records, sft, stats):
    _default = dict(
        granularity="cluster",
        pool="mean",
        temperature=0.5,
        mass_threshold=0.8,
        max_n=3,
        min_n=1,
    )
    _scored = []
    for _i, _r in enumerate(records):
        _picks = sft.select_tag_emotions(
            _r["probe"]["projections"], clusters, stats=stats, **_default
        )
        if not _picks:
            continue
        _top, _w = _picks[0]
        _correct = emo2cluster.get(_top) == _r["scenario"]["cluster"]
        _scored.append((_i, _w, _correct))
    _ranked = sorted(_scored, key=lambda t: -t[1])
    curated_idx = [i for i, _w, c in _ranked if c][:2] + [
        i for i, _w, c in _ranked if not c
    ][:2]
    return (curated_idx,)


@app.cell
def _(curated_idx, mo, show_example):
    mo.vstack([show_example(_i) for _i in curated_idx])
    return


@app.cell
def _(mo):
    mo.md("""
    ## So what?

    - The tag *mechanics* are fine and worth keeping: z-scoring per emotion (Concern 1) is essential,
      and the softmax temperature makes the number of tagged emotions adaptive. What's shaky is the
      *signal*, not the plumbing.
    - The **underlying per-message read is weak**: the target is strongly directional
      (mean **+1.2σ**, positive on ~90%) yet rarely the argmax, and top-tag cluster agreement
      tops out ~10pp above chance (~1.4×) — below the ~0.6–0.7 floor coupling needs. As the
      *sole* training signal, probe-grounded tags at this level likely teach noise.
    - So before committing to probe-grounded tags: **strengthen the probe** (read position /
      layer / neutral-centering, and the §3.1 *steering* gate that exp-02 flags as the real
      validation — not yet run), **judge-label the pilot** to get the format-installation result
      now, or **probe-ground anyway** to exercise the mechanism end-to-end.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## If we train anyway: a balanced, clear subset

    We only need ~600 of the 1972 examples, and chasing label quality barely helps (the probe's most
    self-confident 10% still only agree ~50%). So instead of optimizing agreement, select for two
    things we *can* control:

    - **balance** across the taxonomy, so no family dominates and "chance" isn't skewed;
    - **clarity** — keep examples where *one* family clearly stands out, dropping both *mild* reads
      (nothing much active) and *noisy* reads (several families competing).

    ### Which emotions does the probe trigger?

    First a warning that shapes the selection: balancing the *elicited* labels does **not** balance
    what the probe actually **emits** as the tag. Below is the probe's top-tag ("triggered") emotion
    distribution — overall, then per elicited family — under the strategy set by the Concern-4
    controls above. Watch for a handful of emotions the probe over-triggers regardless of intent.
    """)
    return


@app.cell
def _(
    clusters,
    emo2cluster,
    gran_ui,
    mass_ui,
    maxn_ui,
    pool_ui,
    records,
    sft,
    stats,
    temp_ui,
):
    triggers = []
    for _r in records:
        _picks = sft.select_tag_emotions(
            _r["probe"]["projections"],
            clusters,
            stats=stats,
            granularity=gran_ui.value,
            pool=pool_ui.value,
            temperature=temp_ui.value,
            mass_threshold=mass_ui.value,
            max_n=maxn_ui.value,
            min_n=1,
        )
        if not _picks:
            continue
        _tag = _picks[0][0]
        triggers.append(
            {
                "elicited_cluster": _r["scenario"]["cluster"],
                "tag_emotion": _tag,
                "tag_cluster": emo2cluster.get(_tag),
            }
        )
    return (triggers,)


@app.cell
def _(alt, pl, triggers):
    _freq = (
        pl.DataFrame(triggers)
        .group_by("tag_emotion", "tag_cluster")
        .len()
        .sort("len", descending=True)
        .head(30)
    )
    alt.Chart(_freq).mark_bar().encode(
        x=alt.X("len:Q", title="times this emotion is the top tag"),
        y=alt.Y("tag_emotion:N", sort="-x", title=None),
        color=alt.Color("tag_cluster:N", title="cluster"),
        tooltip=["tag_emotion", "tag_cluster", "len"],
    ).properties(
        width=520,
        height=460,
        title="Which emotions the probe triggers most (overall, top 30)",
    )
    return


@app.cell
def _(cluster_order, mo):
    elicited_ui = mo.ui.dropdown(
        options=cluster_order, value=cluster_order[0], label="elicited family"
    )
    elicited_ui
    return (elicited_ui,)


@app.cell
def _(alt, elicited_ui, pl, triggers):
    _freq = (
        pl.DataFrame(triggers)
        .filter(pl.col("elicited_cluster") == elicited_ui.value)
        .with_columns(
            fam=pl.when(pl.col("tag_cluster") == elicited_ui.value)
            .then(pl.lit("same family"))
            .otherwise(pl.lit("other family"))
        )
        .group_by("tag_emotion", "fam")
        .len()
        .sort("len", descending=True)
        .head(20)
    )
    alt.Chart(_freq).mark_bar().encode(
        x=alt.X("len:Q", title="times triggered on this family's messages"),
        y=alt.Y("tag_emotion:N", sort="-x", title=None),
        color=alt.Color(
            "fam:N",
            scale=alt.Scale(
                domain=["same family", "other family"], range=["#54a24b", "#bab0ac"]
            ),
            title=None,
        ),
        tooltip=["tag_emotion", "fam", "len"],
    ).properties(
        width=460,
        height=380,
        title=f"What the probe triggers on '{elicited_ui.value}' messages",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Balance is capped by the smallest family, and clarity is what we filter on

    The families are wildly uneven, so a fully balanced set is capped by the smallest one you keep
    (all 10 → 350 max; top 8 → 560; top 6 → 630). **Clarity** = the top-1 minus top-2 family mean-z
    on a message — how much one family stands out; low clarity means either mild (nothing active) or
    noisy (families competing). Family sizes and the clarity distribution:
    """)
    return


@app.cell
def _(emo2cluster, records, stats):
    def clarity(projections):
        _z = {
            e: (v - stats[e][0]) / stats[e][1]
            for e, v in projections.items()
            if e in stats
        }
        _by = {}
        for _e, _v in _z.items():
            _by.setdefault(emo2cluster[_e], []).append(_v)
        _cz = sorted((sum(vs) / len(vs) for vs in _by.values()), reverse=True)
        return _cz[0] - _cz[1] if len(_cz) > 1 else _cz[0]

    clarity_by_id = {
        _r["id"]: clarity(_r["probe"]["projections"]) for _r in records
    }
    return (clarity_by_id,)


@app.cell
def _(Counter, alt, clarity_by_id, mo, pl, records):
    _sizes = Counter(_r["scenario"]["cluster"] for _r in records)
    _sz = pl.DataFrame(
        {"cluster": list(_sizes), "n": list(_sizes.values())}
    ).sort("n", descending=True)
    _size_chart = (
        alt.Chart(_sz)
        .mark_bar()
        .encode(
            x=alt.X("n:Q", title="messages"),
            y=alt.Y("cluster:N", sort="-x", title=None),
            tooltip=["cluster", "n"],
        )
        .properties(width=420, height=240, title="Family sizes (smallest included caps the balance)")
    )
    _cl = pl.DataFrame({"clarity": list(clarity_by_id.values())})
    _clar_chart = (
        alt.Chart(_cl)
        .mark_bar()
        .encode(
            x=alt.X(
                "clarity:Q",
                bin=alt.Bin(maxbins=40),
                title="clarity (top-1 − top-2 family mean-z)",
            ),
            y=alt.Y("count()", title="messages"),
        )
        .properties(width=420, height=240, title="Clarity distribution (higher = one family clearly dominates)")
    )
    mo.vstack([_size_chart, _clar_chart])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Build a train / eval split

    A real training set needs held-out data to test generalization. Two kinds:

    - **held-out emotions** — reserve 1–2 emotions per *train* family (the most-populous emotion stays
      in training; the next few with ≥6 messages are held out) → tests whether the tag generalizes to a
      *new emotion of a familiar family*.
    - **held-out families** — reserve ≥1 whole family (default: the smallest) → tests generalization to
      a *family never seen in training*.

    Training then round-robins over each train family's *remaining* emotions, highest-clarity first,
    ≤ `max/emotion`, up to `train messages/family`. The held-out families and held-out emotions form
    the two eval sets.
    """)
    return


@app.cell
def _(cluster_order, mo):
    hemo_ui = mo.ui.slider(start=1, stop=3, step=1, value=2, label="held-out emotions / train family")
    hclus_ui = mo.ui.multiselect(
        options=cluster_order,
        value=["playful_amusement", "vigilant_suspicion"],
        label="held-out families (cross-family eval)",
    )
    pc_ui = mo.ui.slider(start=30, stop=120, step=10, value=80, label="train messages / family")
    ke_ui = mo.ui.slider(start=4, stop=25, step=1, value=15, label="max / emotion")
    mo.vstack([mo.hstack([hemo_ui, pc_ui, ke_ui], justify="start"), hclus_ui])
    return hclus_ui, hemo_ui, ke_ui, pc_ui


@app.cell
def _(
    clarity_by_id,
    cluster_order,
    hclus_ui,
    hemo_ui,
    ke_ui,
    pc_ui,
    records,
    slugify,
):
    def split_train_eval(per_cluster, k_per_emotion, h_emo, holdout_clusters):
        holdout = set(holdout_clusters)
        train_clusters = [c for c in cluster_order if c not in holdout]
        # cluster -> emotion -> records (clarity-sorted, highest first)
        by_ce = {}
        for _r in records:
            by_ce.setdefault(_r["scenario"]["cluster"], {}).setdefault(
                slugify(_r["scenario"]["emotion"]), []
            ).append(_r)
        for _c in by_ce:
            for _e in by_ce[_c]:
                by_ce[_c][_e].sort(key=lambda r: -clarity_by_id[r["id"]])
        # held-out emotions per train family: keep the most populous, hold the next h_emo with >=6 msgs
        heldout_emos = {}
        for _c in train_clusters:
            _ranked = sorted(by_ce.get(_c, {}), key=lambda e: -len(by_ce[_c][e]))
            _cands = [e for e in _ranked[1:] if len(by_ce[_c][e]) >= 6]
            heldout_emos[_c] = set(_cands[:h_emo])
        # train: round-robin over remaining emotions, clarity-first, <=k/emotion, up to per_cluster
        train = []
        for _c in train_clusters:
            _emos = {e: v for e, v in by_ce.get(_c, {}).items() if e not in heldout_emos[_c]}
            _idx = {e: 0 for e in _emos}
            _order = sorted(_emos, key=lambda e: -len(_emos[e]))
            _picked = []
            while len(_picked) < per_cluster:
                _prog = False
                for _e in _order:
                    if _idx[_e] < min(k_per_emotion, len(_emos[_e])):
                        _picked.append(_emos[_e][_idx[_e]])
                        _idx[_e] += 1
                        _prog = True
                        if len(_picked) >= per_cluster:
                            break
                if not _prog:
                    break
            train += _picked
        eval_within = [
            r
            for r in records
            if r["scenario"]["cluster"] in heldout_emos
            and slugify(r["scenario"]["emotion"]) in heldout_emos[r["scenario"]["cluster"]]
        ]
        eval_cross = [r for r in records if r["scenario"]["cluster"] in holdout]
        return train, eval_within, eval_cross, heldout_emos

    train_set, eval_within, eval_cross, heldout_emos = split_train_eval(
        pc_ui.value, ke_ui.value, hemo_ui.value, hclus_ui.value
    )
    return eval_cross, eval_within, heldout_emos, train_set


@app.cell
def _(
    Counter,
    alt,
    clarity_by_id,
    eval_cross,
    eval_within,
    heldout_emos,
    mo,
    pl,
    train_set,
):
    _med = sorted(clarity_by_id[r["id"]] for r in train_set)[len(train_set) // 2] if train_set else 0.0
    _n_emo = len({r["scenario"]["emotion"] for r in train_set})
    _held = ", ".join(sorted({e for s in heldout_emos.values() for e in s})) or "(none)"
    _hclusters = ", ".join(sorted({r["scenario"]["cluster"] for r in eval_cross})) or "(none)"
    _summary = mo.md(
        f"**Train:** {len(train_set)} examples · {_n_emo} emotions · median clarity {_med:.2f}.\n\n"
        f"**Eval — held-out emotions** (within-family): {len(eval_within)} messages · [{_held}].\n\n"
        f"**Eval — held-out families** (cross-family): {len(eval_cross)} messages · [{_hclusters}]."
    )
    _per = Counter(r["scenario"]["cluster"] for r in train_set)
    _pc = pl.DataFrame({"cluster": list(_per), "n": list(_per.values())}).sort("n", descending=True)
    _bal = (
        alt.Chart(_pc)
        .mark_bar()
        .encode(
            x=alt.X("n:Q", title="train messages"),
            y=alt.Y("cluster:N", sort="-x", title=None),
            tooltip=["cluster", "n"],
        )
        .properties(width=420, height=240, title="Train set — per-family balance (held-out families excluded)")
    )
    mo.vstack([_summary, _bal])
    return


if __name__ == "__main__":
    app.run()
