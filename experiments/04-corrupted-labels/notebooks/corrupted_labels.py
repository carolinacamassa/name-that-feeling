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
def _(Path, json):
    HERE = Path(__file__).parents[1]  # the 04-corrupted-labels experiment dir

    _summary_path = HERE / "data" / "cross" / "runs_summary.json"
    SUMMARY = json.loads(_summary_path.read_text(encoding="utf-8")) if _summary_path.exists() else []

    DATASET_MANIFEST = json.loads((HERE / "data" / "sft" / "dataset_manifest.json").read_text(encoding="utf-8"))

    # Chance baselines for the held-out sets (identical for every run -- same eval messages).
    _eval_path = HERE / "data" / "runs" / "shuffled" / "eval.json"
    CHANCE = None
    if _eval_path.exists():
        _g = json.loads(_eval_path.read_text(encoding="utf-8"))["generalization"]
        CHANCE = {"within": _g["within"]["chance_biggest_family"], "cross": _g["cross"]["chance_biggest_family"]}

    RUN_ORDER = ["pilot-with-neutral", "seed-43", "seed-44", "two-epochs", "shuffled", "shuffled-two-epochs"]
    HAVE_SHUFFLED = any(r.get("condition") == "shuffled" and "within_model_vs_teacher" in r for r in SUMMARY)
    return CHANCE, DATASET_MANIFEST, HAVE_SHUFFLED, HERE, RUN_ORDER, SUMMARY


@app.cell
def _(DATASET_MANIFEST, mo):
    mo.md(f"""
    # Do the probe-derived labels teach the mapping, or only the format?

    The `shuffled` and `shuffled-two-epochs` runs retrain the pilot configuration (seed 42,
    neutral anchor included, 3 and 2 epochs respectively) on the same 1,076 examples with
    the 576 probe-derived tags permuted across messages — tag-length and emotion-frequency
    marginals preserved, message↔label correspondence destroyed. The accurate arm is the
    four existing checkpoints: three at 3 epochs (pilot seed 42, reseeds 43/44), which
    bound seed noise, and `two-epochs` (the accurate twin of `shuffled-two-epochs`).

    Because tags repeat across training rows, a model that perfectly memorizes its permuted
    tags still matches the true labels at a floor rate: exact tag
    {DATASET_MANIFEST["chance_floors"]["exact_tag"]:.2%}, top-1 family
    {DATASET_MANIFEST["chance_floors"]["top1_family"]:.2%}
    ({DATASET_MANIFEST["same_family_rows"]}/576 rows of this permutation share the original
    top-1 family; {DATASET_MANIFEST["identical_tag_rows"]["count"]}/576 kept an identical tag
    string).
    """)
    return


@app.cell
def _(CHANCE, HAVE_SHUFFLED, RUN_ORDER, SUMMARY, alt, mo, pl, save_chart):
    # Within-family only: on the unseen-family set the always-guess-the-largest-family
    # baseline sits at 55%, so the agreement percentage misreads there (the 05 exhibits
    # made the same cut); the shuffled model's cross-set behavior is described in prose.
    if not HAVE_SHUFFLED or CHANCE is None:
        _out = mo.md("*run the battery + `summarize_runs.py` first — this exhibit needs the shuffled row*")
    else:
        _long = (
            pl.DataFrame([r for r in SUMMARY if "within_model_vs_teacher" in r])
            .select(
                "run",
                "condition",
                pl.col("within_model_vs_teacher").alias("agreement"),
                pl.lit(CHANCE["within"]).alias("chance"),
            )
        )
        _bars = (
            alt.Chart(_long)
            .mark_bar()
            .encode(
                x=alt.X("run:N", sort=RUN_ORDER, title=None, axis=alt.Axis(labelAngle=-20, labelLimit=160)),
                y=alt.Y("agreement:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="family agreement with the probe-derived labels"),
                color=alt.Color("condition:N", scale=alt.Scale(scheme="tableau10"), title="training labels"),
                tooltip=["run", "condition", alt.Tooltip("agreement:Q", format=".3f")],
            )
        )
        _rule = (
            alt.Chart(_long)
            .mark_rule(strokeDash=[5, 4])
            .encode(y=alt.Y("mean(chance):Q"), tooltip=[alt.Tooltip("mean(chance):Q", format=".3f", title="chance")])
        )
        _out = save_chart(
            alt.layer(_bars, _rule).properties(
                width=380,
                height=260,
                title="Held-out agreement with the probe-derived labels, by label fidelity",
            ),
            "held_out_agreement_by_condition",
            caption="Family-level agreement between the model's emitted tag and the probe-derived label on held-out messages from trained families (260 messages), for three models trained on accurate labels and one trained on the same labels permuted across messages. Dashed line: the always-guess-the-largest-family baseline (15.4%).",
            takeaway="Training on permuted labels removes the message-to-tag mapping at both training durations: family agreement with the probe-derived labels on held-out messages falls from 53-58% (accurate labels) to 8.1% at 3 epochs and 7.7% at 2 epochs, below the 15.4% largest-family baseline. Both permuted-label models emit a playful-amusement-family tag on 96-99% of held-out messages, while format compliance (100%) and the neutral anchor (98-100%) are unaffected.",
            notebook=__file__,
        )
    _out
    return


@app.cell
def _(DATASET_MANIFEST, HAVE_SHUFFLED, RUN_ORDER, SUMMARY, alt, mo, pl, save_chart):
    if not HAVE_SHUFFLED:
        _out2 = mo.md("*run the battery + `summarize_runs.py` first — this exhibit needs the shuffled row*")
    else:
        _rows = [r for r in SUMMARY if "true_top1_family" in r]
        _rec = (
            pl.DataFrame(_rows)
            .select(
                "run",
                "condition",
                pl.col("trained_top1_family").alias("labels trained on"),
                pl.col("true_top1_family").alias("true probe-derived labels"),
            )
            .unpivot(index=["run", "condition"], variable_name="reference", value_name="recovery")
            .with_columns(pl.lit(DATASET_MANIFEST["chance_floors"]["top1_family"]).alias("floor"))
        )
        _bars2 = (
            alt.Chart(_rec)
            .mark_bar()
            .encode(
                x=alt.X("run:N", sort=RUN_ORDER, title=None, axis=alt.Axis(labelAngle=-20, labelLimit=160)),
                xOffset=alt.XOffset("reference:N"),
                y=alt.Y("recovery:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="top-1 family recovery on train messages"),
                color=alt.Color("reference:N", scale=alt.Scale(scheme="set2"), title="scored against"),
                tooltip=["run", "reference", alt.Tooltip("recovery:Q", format=".3f")],
            )
        )
        _rule2 = (
            alt.Chart(_rec)
            .mark_rule(strokeDash=[5, 4])
            .encode(y=alt.Y("mean(floor):Q"), tooltip=[alt.Tooltip("mean(floor):Q", format=".3f", title="permutation floor")])
        )
        _out2 = save_chart(
            alt.layer(_bars2, _rule2).properties(
                width=420,
                height=260,
                title="Trained-tag recovery on train messages, against two references",
            ),
            "train_tag_recovery_two_references",
            caption="Top-1-family recovery of replies sampled on the 576 training messages, scored against the labels each model was trained on and against the true probe-derived labels. For the accurate arms the two references coincide. Dashed line: the 12% rate at which a model that reproduced its permuted labels perfectly would match the true labels, given repeated families across rows.",
            takeaway="Neither permuted-label model memorizes its arbitrary message-to-tag assignment: recovery of the trained labels' family on training messages is 18-19% at both durations, matching the 18.75% a constant emitter of the modal family (playful amusement) would score, while accurate-label runs recover 63-73%. Verbatim reply reproduction tracks the accurate run at each duration (35% vs 33-38% at 3 epochs, 6.8% vs 6.4% at 2), so the failure is specific to the tag mapping, not to training.",
            notebook=__file__,
        )
    _out2
    return


@app.cell
def _(SUMMARY, mo, pl):
    _cols = [
        "run",
        "condition",
        "compliance_within",
        "neutral_exact_rate",
        "final_loss",
        "true_reply_replay_rate",
        "true_reply_similarity_median",
    ]
    _rows = [{c: r.get(c) for c in _cols} for r in SUMMARY]
    mo.vstack(
        [
            mo.md(
                "Sanity readings expected to be insensitive to label corruption: format compliance, "
                "the neutral anchor (those rows were not permuted), the loss curve endpoint, and "
                "reply replay (visible completions were never corrupted)."
            ),
            mo.ui.table(pl.DataFrame(_rows), selection=None),
        ]
    )
    return


@app.cell
def _(RUN_ORDER, SUMMARY, alt, mo, pl, save_chart):
    # Discriminant check for the graded distance metric (docs/tag-accuracy-distance-metric.md):
    # a collapsed constant emitter must land at its permutation null, or the metric is broken.
    _drows = [r for r in SUMMARY if "dist_within_model_rank_pct" in r]
    if not _drows:
        _outd = mo.md("*run `summarize_runs.py` first — this exhibit reads the dist_ columns*")
    else:
        _d = pl.DataFrame(_drows).select(
            "run",
            "condition",
            pl.col("dist_within_model_rank_pct").alias("rank_pct"),
            pl.col("dist_within_rank_z").alias("z"),
            pl.col("dist_within_model_vs_teacher_cosine").alias("model_teacher_cosine"),
        )
        _barsd = (
            alt.Chart(_d)
            .mark_bar()
            .encode(
                x=alt.X("run:N", sort=RUN_ORDER, title=None, axis=alt.Axis(labelAngle=-20, labelLimit=160)),
                y=alt.Y(
                    "rank_pct:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    title="mean similarity rank percentile of the emitted tag",
                ),
                color=alt.Color("condition:N", scale=alt.Scale(scheme="tableau10"), title="training labels"),
                tooltip=[
                    "run",
                    "condition",
                    alt.Tooltip("rank_pct:Q", format=".3f"),
                    alt.Tooltip("z:Q", format=".1f"),
                    alt.Tooltip("model_teacher_cosine:Q", format=".3f"),
                ],
            )
        )
        _chanced = alt.Chart(pl.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[5, 4]).encode(y="y:Q")
        _outd = save_chart(
            (_barsd + _chanced).properties(
                width=380,
                height=240,
                title="Graded tag similarity on held-out messages, by label fidelity",
            ),
            "distance_metric_discriminant_check",
            caption="Mean rank percentile of the emitted tag's first emotion among the 171 emotions ordered by vector similarity to the elicited emotion, on the 260 held-out messages from trained families. Dashed line: the 0.5 expectation under a random guess; the permuted-label models emit near-constant playful-amusement-family tags.",
            takeaway="The graded distance metric passes its discriminant check on the corrupted-label arms: models trained on permuted labels score at their permutation null (mean rank percentile 0.51 at 3 epochs and 0.50 at 2; z = -0.6 and -1.8), while every accurate-label configuration scores 0.69-0.71 (z = 11.5-12.6), and mean model-teacher tag similarity collapses from 0.54-0.57 to 0.05. A metric that credits graded near-misses gives no spurious credit to a collapsed constant emitter.",
            notebook=__file__,
        )
    _outd
    return


@app.cell
def _(HERE, json, load_clusters, slugify):
    # Probe readouts on all 1,972 messages: the fixed base-model reference (exp 02), the
    # accurate-label pilot (exp 04-trained-emotion-vectors), and this experiment's
    # shuffled run (readout.py -> data/runs/shuffled/). Same base emotion vectors throughout.
    _experiments = HERE.parent
    _shuffled_path = HERE / "data" / "runs" / "shuffled" / "readout_full_base_vectors.json"
    HAVE_READOUT = _shuffled_path.exists()
    READOUT_MSGS = {}
    if HAVE_READOUT:
        READOUT_MSGS = {
            "base": json.loads(
                (_experiments / "02-elicited-activations" / "data" / "qwen3.5-9b" / "readout.json").read_text(
                    encoding="utf-8"
                )
            )["messages"],
            "pilot (accurate)": json.loads(
                (_experiments / "04-trained-emotion-vectors" / "data" / "readout_full_base_vectors.json").read_text(
                    encoding="utf-8"
                )
            )["messages"],
            "shuffled": json.loads(_shuffled_path.read_text(encoding="utf-8"))["messages"],
        }
    EMO2FAM = {
        slugify(e): c
        for c, es in load_clusters(_experiments / "01-emotion-vectors" / "clusters.json").items()
        for e in es
    }
    return EMO2FAM, HAVE_READOUT, READOUT_MSGS


@app.cell
def _(EMO2FAM, HAVE_READOUT, READOUT_MSGS, mo, np, slugify):
    # Is the internal state intact under the collapsed output channel? Three readings per
    # model: mean z of the elicited emotion's projection, family-level argmax agreement,
    # and (for trained models) the median per-message profile correlation with the base.
    if not HAVE_READOUT:
        _out3 = mo.md("*no readout yet — run export_adapter.py + readout.py for `shuffled` first*")
        READOUT_STATS = None
    else:
        _emos = sorted(READOUT_MSGS["shuffled"][0]["projections"])
        _eidx = {e: i for i, e in enumerate(_emos)}

        def _matrix(msgs):
            return np.array([[m["projections"][e] for e in _emos] for m in msgs])

        def _stats(msgs):
            m = _matrix(msgs)
            z = (m - m.mean(axis=0)) / np.where(m.std(axis=0) == 0, 1.0, m.std(axis=0))
            tz = [
                z[k, _eidx[slugify(r["emotion"])]]
                for k, r in enumerate(msgs)
                if slugify(r["emotion"]) in _eidx
            ]
            fam_hits = [
                EMO2FAM.get(_emos[int(np.argmax(z[k]))]) == r["cluster"] for k, r in enumerate(msgs)
            ]
            return {"target_z": float(np.mean(tz)), "family_argmax_agreement": float(np.mean(fam_hits))}

        _base_msgs = READOUT_MSGS["base"]
        _base_by_id = {r["id"]: k for k, r in enumerate(_base_msgs)}
        _mb = _matrix(_base_msgs)

        def _profile_corr(msgs):
            mt = _matrix(msgs)
            rows = [
                float(np.corrcoef(mt[k], _mb[_base_by_id[r["id"]]])[0, 1])
                for k, r in enumerate(msgs)
                if r["id"] in _base_by_id
            ]
            return float(np.median(rows))

        READOUT_STATS = {name: _stats(msgs) for name, msgs in READOUT_MSGS.items()}
        for _name in ("pilot (accurate)", "shuffled"):
            READOUT_STATS[_name]["profile_corr_vs_base"] = _profile_corr(READOUT_MSGS[_name])

        _rows = "\n".join(
            f"| {name} | {s['target_z']:+.2f}σ | {s['family_argmax_agreement']:.0%} | "
            f"{s.get('profile_corr_vs_base', float('nan')):.4f} |"
            for name, s in READOUT_STATS.items()
        )
        _out3 = mo.md(f"""
    ## Is the internal state intact under the collapsed output channel?

    Probe readings on all 1,972 messages, base emotion vectors throughout:

    | model | mean target z | family argmax agreement | median profile corr vs base |
    | --- | --- | --- | --- |
    {_rows}

    (The base row's profile correlation is with itself and omitted.)
    """)
    _out3
    return (READOUT_STATS,)


@app.cell
def _(EMO2FAM, HAVE_READOUT, READOUT_MSGS, alt, mo, np, pl, save_chart):
    # Family-level activation tilt vs the base model: does corrupted-label training move
    # the representation differently than accurate-label training? (Effect size per
    # emotion = (mean_trained - mean_base) / std_base, as in 04-sft-seeds-and-epochs.)
    if not HAVE_READOUT:
        _out4 = mo.md("*no readout yet — the tilt exhibit needs data/runs/shuffled/readout_full_base_vectors.json*")
    else:
        _emos2 = sorted(READOUT_MSGS["shuffled"][0]["projections"])
        _base_by_id2 = {m["id"]: m for m in READOUT_MSGS["base"]}

        def _effect_sizes(run_msgs):
            ids = [m["id"] for m in run_msgs if m["id"] in _base_by_id2]
            mb = np.array([[_base_by_id2[i]["projections"][e] for e in _emos2] for i in ids])
            mt = np.array(
                [[m["projections"][e] for e in _emos2] for m in run_msgs if m["id"] in _base_by_id2]
            )
            sb = mb.std(axis=0)
            return (mt.mean(axis=0) - mb.mean(axis=0)) / np.where(sb == 0, 1.0, sb)

        _eff = pl.DataFrame(
            [
                {"model": name, "emotion": e, "family": EMO2FAM.get(e, "?"), "effect_size": float(v)}
                for name in ("pilot (accurate)", "shuffled")
                for e, v in zip(_emos2, _effect_sizes(READOUT_MSGS[name]))
            ]
        )
        _fam = _eff.group_by("model", "family").agg(pl.col("effect_size").mean().alias("mean_effect"))
        _out4 = save_chart(
            alt.Chart(_fam)
            .mark_bar()
            .encode(
                x=alt.X("mean_effect:Q", title="mean activation shift vs base (Δmean / base std)"),
                y=alt.Y("family:N", sort="-x", title=None),
                color=alt.Color("model:N", scale=alt.Scale(scheme="set2"), title="training labels"),
                yOffset="model:N",
                tooltip=["model", "family", alt.Tooltip("mean_effect:Q", format=".3f")],
            )
            .properties(width=440, height=320, title="Family-level activation shift, accurate vs permuted labels"),
            "activation_tilt_shuffled_vs_accurate",
            caption="Mean per-family activation shift of each trained model against the base model, over all 1,972 messages projected onto the base emotion vectors. Effect size per emotion is the change in mean projection divided by the base standard deviation; bars average emotions within a family.",
            takeaway="The small activation shift after tag training is independent of label content: the permuted-label model reproduces the accurate pilot's per-emotion pattern (Pearson r = 0.96; hostile anger +0.16 vs +0.15, peaceful contentment -0.16 vs -0.16) at the same magnitude (median absolute effect 0.11 vs 0.10 base standard deviations). The shift therefore tracks the training data and procedure, not the correspondence between labels and messages.",
            notebook=__file__,
        )
    _out4
    return


if __name__ == "__main__":
    app.run()
