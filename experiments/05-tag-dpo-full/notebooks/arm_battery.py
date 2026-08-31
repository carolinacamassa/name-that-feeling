import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
    from name_that_feeling.evals import tag_eval
    from name_that_feeling.evals.probe_teacher import ProbeTeacher
    from name_that_feeling.evals.similarity import EmotionSimilarity
    from name_that_feeling.evals.uncertainty import mean_and_ci
    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return (
        EmotionSimilarity,
        Path,
        ProbeTeacher,
        alt,
        json,
        load_clusters,
        mean_and_ci,
        mo,
        pl,
        save_chart,
        slugify,
        tag_eval,
    )


@app.cell
def _(EmotionSimilarity, Path, ProbeTeacher, json, load_clusters):
    HERE = Path(__file__).parents[1]  # the 05-tag-dpo-full experiment dir
    EXPERIMENTS = HERE.parent
    NEUTRAL_TAG = "calm, attentive"

    # The four checkpoints of the dose story: the SFT parent (dose 0), the pilot
    # (164 pairs / 21 steps), and this experiment's two arms.
    STABILITY_FILES = {
        "supervised parent (0 steps)": EXPERIMENTS / "04-sft-seeds-and-epochs" / "data" / "stability" / "two-epochs" / "samples.json",
        "pilot (164 pairs, 21 steps)": EXPERIMENTS / "05-tag-dpo" / "data" / "runs" / "tag-masked-test" / "stability_samples.json",
        "capped-200 (25 steps)": HERE / "data" / "runs" / "capped-200" / "stability_samples.json",
        "uncapped-800 (100 steps)": HERE / "data" / "runs" / "uncapped-800" / "stability_samples.json",
    }
    RUN_ORDER = list(STABILITY_FILES)
    STABILITY = {name: json.loads(p.read_text(encoding="utf-8"))["samples"] for name, p in STABILITY_FILES.items()}

    ACT_FILES = {
        "pilot (164 pairs, 21 steps)": EXPERIMENTS / "05-tag-dpo" / "data" / "runs" / "tag-masked-test" / "activation_distributions.json",
        "capped-200 (25 steps)": HERE / "data" / "runs" / "capped-200" / "activation_distributions.json",
        "uncapped-800 (100 steps)": HERE / "data" / "runs" / "uncapped-800" / "activation_distributions.json",
        "capped-200 · whole-reply credit": HERE / "data" / "runs" / "capped-200-full" / "activation_distributions.json",
        "uncapped-800 · whole-reply credit": HERE / "data" / "runs" / "uncapped-800-full" / "activation_distributions.json",
    }
    ACT = {name: json.loads(p.read_text(encoding="utf-8")) for name, p in ACT_FILES.items()}

    # The whole-reply-credit reruns and the β dose points sampled 40 prompts per eval
    # set (identical ids across the four newer runs); COMMON_IDS is that shared
    # population, the only prompts sampled under every checkpoint, so any chart that
    # compares across the credit or β axis is scored on it.
    FULLCREDIT_FILES = {
        "capped-200 · whole-reply credit": HERE / "data" / "runs" / "capped-200-full" / "stability_samples.json",
        "uncapped-800 · whole-reply credit": HERE / "data" / "runs" / "uncapped-800-full" / "stability_samples.json",
    }
    BETA_FILES = {
        "β = 0.05": HERE / "data" / "runs" / "uncapped-800-full-beta0.05" / "stability_samples.json",
        "β = 0.1": HERE / "data" / "runs" / "uncapped-800-full" / "stability_samples.json",
        "β = 0.3": HERE / "data" / "runs" / "uncapped-800-full-beta0.3" / "stability_samples.json",
    }
    FULLCREDIT = {n: json.loads(p.read_text(encoding="utf-8"))["samples"] for n, p in FULLCREDIT_FILES.items()}
    BETA = {n: json.loads(p.read_text(encoding="utf-8"))["samples"] for n, p in BETA_FILES.items()}
    COMMON_IDS = {
        s: {r["id"] for r in FULLCREDIT["uncapped-800 · whole-reply credit"] if r["set"] == s}
        for s in ("within", "cross", "neutral")
    }
    COUPLING = json.loads((HERE / "data" / "coupling" / "coupling_matrix.json").read_text(encoding="utf-8"))

    # Greedy (deterministic-decoding) eval replies per checkpoint, and the eval set
    # rows that carry each prompt's id and elicitation family.
    EVAL_SAMPLE_FILES = {
        "supervised parent (0 steps)": EXPERIMENTS / "04-sft-seeds-and-epochs" / "data" / "runs" / "two-epochs" / "eval_samples.json",
        "pilot (164 pairs, 21 steps)": EXPERIMENTS / "05-tag-dpo" / "data" / "runs" / "tag-masked-test" / "eval_samples.json",
        "capped-200 (25 steps)": HERE / "data" / "runs" / "capped-200" / "eval_samples.json",
        "uncapped-800 (100 steps)": HERE / "data" / "runs" / "uncapped-800" / "eval_samples.json",
        "capped-200 · whole-reply credit": HERE / "data" / "runs" / "capped-200-full" / "eval_samples.json",
        "uncapped-800 · whole-reply credit": HERE / "data" / "runs" / "uncapped-800-full" / "eval_samples.json",
    }
    EVAL_SAMPLES = {n: json.loads(fp.read_text(encoding="utf-8")) for n, fp in EVAL_SAMPLE_FILES.items()}
    EVAL_SETS = {
        s: [
            json.loads(x)
            for x in (EXPERIMENTS / "03-training-pilot" / "data" / "sft" / f"eval_{s}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if x.strip()
        ]
        for s in ("within", "cross")
    }

    CLUSTERS = load_clusters(EXPERIMENTS / "01-emotion-vectors" / "clusters.json")
    SIM = EmotionSimilarity.load(EXPERIMENTS / "01-emotion-vectors" / "data" / "similarity" / "layer_21.json")
    _split = json.loads((EXPERIMENTS / "03-training-pilot" / "data" / "sft" / "split.json").read_text(encoding="utf-8"))
    _completions = [
        json.loads(x)
        for x in (EXPERIMENTS / "03-training-pilot" / "data" / "completions" / "unconditioned.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    TEACHER = ProbeTeacher.from_completions(_completions, CLUSTERS, _split["tag_config"])
    return (
        ACT,
        BETA,
        CLUSTERS,
        COMMON_IDS,
        COUPLING,
        EVAL_SAMPLES,
        EVAL_SETS,
        FULLCREDIT,
        NEUTRAL_TAG,
        RUN_ORDER,
        SIM,
        STABILITY,
        TEACHER,
    )


@app.cell
def _(mo):
    mo.md("""
    # Two mixture arms at full scale: where the preference stage breaks

    Both arms trained from the two-epochs SFT checkpoint with the pilot's template
    parameters (β 0.1, lr 5e-5, batch 8, one epoch, tag-masked credit) on pair sets
    drawn from the same 806-pair inventory — `uncapped-800` (the realized 82%-negative
    mixture, 100 steps) and `capped-200` (near-flat charged mixture, 25 steps). The
    exhibits below read the K = 12 temperature-1 sampled evals and the probe readouts
    against the SFT parent and the 164-pair pilot.
    """)
    return


@app.cell
def _(
    RUN_ORDER,
    SIM,
    STABILITY,
    TEACHER,
    alt,
    pl,
    save_chart,
    slugify,
    tag_eval,
):
    def _first_in_tax(emotions):
        for e in emotions:
            s = slugify(e)
            if SIM.index(s) is not None:
                return s
        return None

    _rows = []
    for _run, _samples in STABILITY.items():
        for _set in ("within", "cross"):
            _prompts = [s for s in _samples if s["set"] == _set]
            _n_draws = _compl = 0
            _scores = []
            _cons_right = _cons_wrong = 0
            for _s in _prompts:
                _t = TEACHER.top_word(_s["id"])
                _per_draw = []
                for _r in _s["replies"]:
                    _n_draws += 1
                    _p = tag_eval.parse_reply(_r)
                    _ok = _p["compliant"] and _p["emotions"] and "</emotion>" in _r
                    _compl += _ok
                    _v = SIM.rank_percentile(_t, _first_in_tax(_p["emotions"])) if _ok else None
                    _per_draw.append(_v)
                _scores += [v for v in _per_draw if v is not None]
                if _per_draw and all(v is not None and v >= 0.8 for v in _per_draw):
                    _cons_right += 1
                if _per_draw and all(v is None or v <= 0.4 for v in _per_draw):
                    _cons_wrong += 1
            _rows += [
                {"run": _run, "measure": f"{_set} · compliant draws", "value": _compl / _n_draws},
                {"run": _run, "measure": f"{_set} · consistently right prompts", "value": _cons_right / len(_prompts)},
                {"run": _run, "measure": f"{_set} · consistently wrong prompts", "value": _cons_wrong / len(_prompts)},
            ]
    _order = [f"{s} · {m}" for s in ("within", "cross") for m in ("compliant draws", "consistently right prompts", "consistently wrong prompts")]
    save_chart(
        alt.Chart(pl.DataFrame(_rows))
        .mark_bar()
        .encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
            y=alt.Y("measure:N", sort=_order, title=None, axis=alt.Axis(labelLimit=300)),
            color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), sort=RUN_ORDER, title=None),
            yOffset=alt.YOffset("run:N", sort=RUN_ORDER),
            tooltip=["run", "measure", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(width=430, height=380, title="Tag format and consistency per checkpoint"),
        "channel_health_by_dose",
        caption=(
            "K = 12 temperature-1 draws per held-out prompt for four checkpoints: the two-epochs "
            "SFT parent, the 164-pair pilot, and the two full-scale arms, all trained with "
            "identical template parameters. Compliant = the draw opens and closes the emotion tag; "
            "consistently right/wrong = every draw of the prompt scores ≥ 0.8 / ≤ 0.4 rank "
            "percentile against the frozen probe teacher's top word (a draw that never closes its "
            "tag counts as not-right)."
        ),
        takeaway=(
            "Sharpening scales into degeneration: the pilot and the capped arm triple the parent's "
            "consistently-right share (6% to 30% within) with format intact, but the uncapped arm "
            "at 100 steps loses the format itself — only 42% of its charged draws close the tag "
            "(the rest repeat emotion words indefinitely), consistently-right collapses to 5%/1%, "
            "and consistently-wrong prompts appear at 12%/18%. Its surviving draws score highest "
            "of all checkpoints (0.80 within) — over-sharpening and format collapse are the same "
            "process at different doses, not separate failures."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(NEUTRAL_TAG, RUN_ORDER, STABILITY, alt, pl, save_chart, tag_eval):
    _rows = []
    for _run, _samples in STABILITY.items():
        _neut = [s for s in _samples if s["set"] == "neutral"]
        _counts = {"exact anchor": 0, "anchor + appended word(s)": 0, "other tag": 0}
        _total = 0
        for _s in _neut:
            for _r in _s["replies"]:
                _total += 1
                _tag = ", ".join(tag_eval.parse_reply(_r)["emotions"])
                if _tag == NEUTRAL_TAG:
                    _counts["exact anchor"] += 1
                elif _tag.startswith(NEUTRAL_TAG + ","):
                    _counts["anchor + appended word(s)"] += 1
                else:
                    _counts["other tag"] += 1
        _rows += [{"run": _run, "mode": m, "share": c / _total} for m, c in _counts.items()]
    save_chart(
        alt.Chart(pl.DataFrame(_rows))
        .mark_bar()
        .encode(
            x=alt.X("share:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="share of neutral draws"),
            y=alt.Y("run:N", sort=RUN_ORDER, title=None, axis=alt.Axis(labelLimit=220)),
            color=alt.Color("mode:N", scale=alt.Scale(scheme="tableau10"), title=None,
                            sort=["exact anchor", "anchor + appended word(s)", "other tag"]),
            order=alt.Order("mode:N"),
            tooltip=["run", "mode", alt.Tooltip("share:Q", format=".3f")],
        )
        .properties(width=430, height=200, title="The neutral anchor under each checkpoint"),
        "neutral_anchor_modes_by_dose",
        caption=(
            "Composition of the K = 12 neutral-prompt draws per checkpoint: the exact trained "
            "anchor (calm, attentive), the anchor with one or more words appended (capped-200's "
            "dominant mode is calm, attentive, playful), or any other tag. The uncapped arm "
            "trained on 45 neutral pairs, the capped arm on 11, the pilot on 16."
        ),
        takeaway=(
            "The neutral anchor's fate tracks the arm's neutral-pair count, not its total size: "
            "uncapped-800 (45 neutral pairs) holds the exact anchor on 98% of draws — better than "
            "the parent — while capped-200 (11 pairs) all but loses it (1%), drifting to "
            "anchor-plus-appended-word on 68% of draws; the pilot (16 pairs) sits between at 81%. "
            "The appended word is usually playful-family — the same attractor the corrupted-labels "
            "collapse fell into — so the 'more emotion words' pressure from the charged pairs "
            "captures whichever slice is too thin to resist it."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(ACT, CLUSTERS, alt, pl, save_chart, slugify):
    _emo2fam = {slugify(e): c for c, es in CLUSTERS.items() for e in es}
    _rows = pl.DataFrame(
        [
            {"run": _run, "family": r["family"], "mean_delta": r["mean_delta"]}
            for _run, _ad in ACT.items()
            for r in _ad["comparisons"]["sft_parent_to_run"]["eval_within"]
        ]
    ).group_by("run", "family").agg(pl.col("mean_delta").mean().alias("family_shift"))
    _order = list(ACT)
    save_chart(
        alt.Chart(_rows)
        .mark_bar()
        .encode(
            x=alt.X("family_shift:Q", title="family-mean activation shift vs the SFT parent (base std units)"),
            y=alt.Y("family:N", sort="-x", title=None),
            color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), sort=_order, title=None),
            yOffset=alt.YOffset("run:N", sort=_order),
            tooltip=["run", "family", alt.Tooltip("family_shift:Q", format="+.3f")],
        )
        .properties(width=440, height=340, title="Each preference run's activation shift, by family"),
        "arm_family_tilt_vs_pilot",
        caption=(
            "Family-mean pre-response activation shift introduced by each preference run relative "
            "to the shared two-epochs parent, on the held-out within-family messages (projections "
            "on the fixed base vectors, base-std units). The pilot's mixture was hostile-heavy, "
            "the uncapped arm's is 82% negative, the capped arm's is near-flat. The two "
            "whole-reply-credit runs repeat the arms with identical pairs, parameters, and seed — "
            "only the loss credit changes."
        ),
        takeaway=(
            "Under tag-only credit, scale multiplies the state shift and mixture steers it: the "
            "uncapped arm collapses the calm/peaceful end by 0.5-0.9 base std, and even the "
            "near-flat capped mixture moves families by up to ~0.4. Extending the loss credit to "
            "the whole reply removes almost all of it — with identical pairs and parameters, both "
            "whole-reply arms keep every family-mean shift below 0.08 base std in magnitude. The "
            "large movement was therefore a product of the tag-only credit, not of the preference "
            "data itself."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(
    COMMON_IDS,
    FULLCREDIT,
    NEUTRAL_TAG,
    SIM,
    STABILITY,
    TEACHER,
    alt,
    pl,
    save_chart,
    slugify,
    tag_eval,
):
    def _first_tax(emotions):
        for e in emotions:
            s = slugify(e)
            if SIM.index(s) is not None:
                return s
        return None

    _runs = {
        k: STABILITY[k]
        for k in ("supervised parent (0 steps)", "capped-200 (25 steps)", "uncapped-800 (100 steps)")
    } | FULLCREDIT
    _rows = []
    for _run, _samples in _runs.items():
        for _set in ("within", "cross"):
            _prompts = [s for s in _samples if s["set"] == _set and s["id"] in COMMON_IDS[_set]]
            _n_draws = _compl = _cons_right = 0
            for _s in _prompts:
                _t = TEACHER.top_word(_s["id"])
                _per_draw = []
                for _r in _s["replies"]:
                    _n_draws += 1
                    _p = tag_eval.parse_reply(_r)
                    _ok = _p["compliant"] and _p["emotions"] and "</emotion>" in _r
                    _compl += _ok
                    _per_draw.append(SIM.rank_percentile(_t, _first_tax(_p["emotions"])) if _ok else None)
                if _per_draw and all(v is not None and v >= 0.8 for v in _per_draw):
                    _cons_right += 1
            if _set == "within":
                _rows.append({"run": _run, "measure": "within · compliant draws", "value": _compl / _n_draws})
            _rows.append({"run": _run, "measure": f"{_set} · consistently right prompts", "value": _cons_right / len(_prompts)})
        _neut = [s for s in _samples if s["set"] == "neutral" and s["id"] in COMMON_IDS["neutral"]]
        _exact = _total = 0
        for _s in _neut:
            for _r in _s["replies"]:
                _total += 1
                _p = tag_eval.parse_reply(_r)
                _exact += _p["compliant"] and ", ".join(_p["emotions"]) == NEUTRAL_TAG
        _rows.append({"run": _run, "measure": "neutral · exact anchor draws", "value": _exact / _total})
    _run_order = list(_runs)
    _measure_order = [
        "within · compliant draws",
        "within · consistently right prompts",
        "cross · consistently right prompts",
        "neutral · exact anchor draws",
    ]
    save_chart(
        alt.Chart(pl.DataFrame(_rows))
        .mark_bar()
        .encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
            y=alt.Y("measure:N", sort=_measure_order, title=None, axis=alt.Axis(labelLimit=300)),
            color=alt.Color("run:N", scale=alt.Scale(scheme="tableau10"), sort=_run_order, title=None),
            yOffset=alt.YOffset("run:N", sort=_run_order),
            tooltip=["run", "measure", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(width=430, height=360, title="Tag-only vs whole-reply loss credit, same pairs"),
        "credit_contrast_common_prompts",
        caption=(
            "The two failure modes and the accuracy target, for the SFT parent, both tag-masked "
            "arms, and their whole-reply-credit reruns — identical pairs, parameters, and seed; "
            "only the loss credit changes. All five checkpoints are scored on the same 40 prompts "
            "per eval set (the subset sampled under every checkpoint), K = 12 temperature-1 draws "
            "each; shares on this common subset differ somewhat from the full-set numbers of the "
            "tag-masked exhibits. Measures as in the format exhibit, plus the share of "
            "neutral-task draws carrying the exact trained anchor."
        ),
        takeaway=(
            "Moving the loss credit from the tag alone to the whole reply repairs both failure "
            "modes at once — on the shared prompts the uncapped arm's compliant share rises from "
            "68% to 100% and the capped arm's exact neutral anchor from 2% to 98% — while the "
            "accuracy the preference stage exists for improves rather than degrades: "
            "consistently-right prompts reach 73% (capped) and 48% (uncapped) on trained "
            "families, against the parent's 13%. A judge reading only the visible replies scores "
            "the uncapped whole-reply arm's tone exactly at the base model's level (1.363 on a "
            "0-3 scale; the parent sits at 1.387) and its answers at least as good as the base's "
            "on 80% of ordinary tasks."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(BETA, NEUTRAL_TAG, alt, pl, save_chart, tag_eval):
    _rows = []
    for _run, _samples in BETA.items():
        _counts = {"exact anchor": 0, "anchor + appended word(s)": 0, "other well-formed tag": 0, "malformed tag": 0}
        _total = 0
        for _s in (s for s in _samples if s["set"] == "neutral"):
            for _r in _s["replies"]:
                _total += 1
                _p = tag_eval.parse_reply(_r)
                if not (_p["compliant"] and _p["emotions"]):
                    _counts["malformed tag"] += 1
                    continue
                _tag = ", ".join(_p["emotions"])
                if _tag == NEUTRAL_TAG:
                    _counts["exact anchor"] += 1
                elif _tag.startswith(NEUTRAL_TAG + ","):
                    _counts["anchor + appended word(s)"] += 1
                else:
                    _counts["other well-formed tag"] += 1
        _rows += [{"run": _run, "mode": m, "share": c / _total} for m, c in _counts.items()]
    _mode_order = ["exact anchor", "anchor + appended word(s)", "other well-formed tag", "malformed tag"]
    save_chart(
        alt.Chart(pl.DataFrame(_rows))
        .mark_bar()
        .encode(
            x=alt.X("share:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="share of neutral draws"),
            y=alt.Y("run:N", sort=list(BETA), title=None),
            color=alt.Color("mode:N", scale=alt.Scale(scheme="tableau10"), sort=_mode_order, title=None),
            order=alt.Order("mode:N"),
            tooltip=["run", "mode", alt.Tooltip("share:Q", format=".3f")],
        )
        .properties(width=430, height=170, title="The neutral anchor across the β sweep"),
        "neutral_modes_by_beta",
        caption=(
            "Composition of the K = 12 neutral-task draws at three settings of β — the strength "
            "of the pull that keeps the trained model close to its starting checkpoint (a smaller "
            "β is a looser tether: more movement per training pair). All three runs share the "
            "uncapped-800 pairs, whole-reply credit, and every other parameter; 40 neutral "
            "prompts each. Malformed = the opening emotion tag is broken, so it cannot be "
            "stripped from the visible reply."
        ),
        takeaway=(
            "The only casualty of loosening the tether is the neutral anchor's format: at "
            "β = 0.05 the opening tag comes out malformed on 42% of neutral draws and the exact "
            "anchor survives on only 55%, while β = 0.1 and β = 0.3 stay format-clean (98% and "
            "96% exact). The emotional channel itself is insensitive across the sweep — "
            "within-family accuracy is indistinguishable on the shared prompts — and the one "
            "genuine trade runs the other way: on messages from the two never-trained families, "
            "the tag names a never-trained family on 79% (β 0.05), 69% (β 0.1), and 60% (β 0.3) "
            "of replies, falling as the tether tightens."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(COUPLING, alt, pl, save_chart):
    _labels = {
        "two-epochs": "supervised parent",
        "tag-masked-test": "pilot (164 pairs)",
        "capped-200": "capped-200",
        "uncapped-800": "uncapped-800",
        "capped-200-full": "capped-200 · whole-reply",
        "uncapped-800-full": "uncapped-800 · whole-reply",
    }
    _rows = []
    for _col, _label in _labels.items():
        _cells = COUPLING["matrix"][_col]
        _diag = _cells[_col]["centroid_1v3"]
        _foreign = [
            _cells[_r]["centroid_1v3"]["mean"]
            for _r in _cells
            if _r not in (_col, "prompted-base", "shuffled") and _cells[_r]["centroid_1v3"]["n"] >= 10
        ]
        _rows += [
            {"teacher": _label, "source": "the checkpoint's own tags", "value": _diag["mean"], "lo": _diag["lo"], "hi": _diag["hi"]},
            {"teacher": _label, "source": "other checkpoints' tags (mean)", "value": sum(_foreign) / len(_foreign), "lo": None, "hi": None},
            {"teacher": _label, "source": "corrupted-labels control's tags", "value": _cells["shuffled"]["centroid_1v3"]["mean"], "lo": None, "hi": None},
        ]
    _df = pl.DataFrame(_rows)
    _teacher_order = list(_labels.values())
    _source_order = ["the checkpoint's own tags", "other checkpoints' tags (mean)", "corrupted-labels control's tags"]
    _base = alt.Chart(_df).encode(
        y=alt.Y("teacher:N", sort=_teacher_order, title="probe read of ...", axis=alt.Axis(labelLimit=240))
    )
    _chart = (
        alt.layer(
            alt.Chart(pl.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4], color="#888").encode(x="x:Q"),
            _base.transform_filter(alt.datum.source == "the checkpoint's own tags").mark_rule(strokeWidth=2).encode(
                x="lo:Q", x2="hi:Q", color=alt.value("#9ab")
            ),
            _base.mark_point(filled=True, size=90).encode(
                x=alt.X("value:Q", scale=alt.Scale(domain=[0.3, 1.0]), axis=alt.Axis(format="%"),
                        title="share of tags siding with the current probe read (chance = 50%)"),
                color=alt.Color("source:N", sort=_source_order, title=None,
                                scale=alt.Scale(scheme="tableau10")),
                tooltip=["teacher", "source", alt.Tooltip("value:Q", format=".3f")],
            ),
        )
        .properties(width=430, height=220, title="Whose probe read do the tags side with?")
    )
    save_chart(
        _chart,
        "coupling_own_vs_foreign",
        caption=(
            "For each checkpoint's own probe read (rows), the share of emitted tags that sit "
            "closer to that read's label than to the frozen training label, restricted to the "
            "held-out messages where the two disagree (27-120 of 337, chance = 50%). The "
            "checkpoint's own deterministic-decoding tags carry a prompt-level bootstrap "
            "interval; the other two marks are the mean over the other trained checkpoints' "
            "tags and the tags of the corrupted-labels control, whose training labels were "
            "shuffled."
        ),
        takeaway=(
            "A report that reads the state should match its own checkpoint's probe read better "
            "than other checkpoints' reports do. It does not: every healthy checkpoint's own-tag "
            "advantage over foreign tags is -1 to +4 points, and the corrupted-labels control — "
            "trained on shuffled labels, so its tags cannot reflect any state — sides with other "
            "checkpoints' probe reads at 58-68%. The one large own-tag advantage (+22) belongs "
            "to the degenerate tag-masked uncapped arm, whose report and state collapsed toward "
            "the same families. At twelve-draw power the supervised parent sits exactly at "
            "chance against its own read (47.4% [44.3, 50.5]): the earlier small advantage of "
            "scoring reports against the current read reflects reports and reads drifting toward "
            "the same families, not the report tracking the state."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(
    CLUSTERS,
    EVAL_SAMPLES,
    EVAL_SETS,
    SIM,
    TEACHER,
    alt,
    mean_and_ci,
    pl,
    save_chart,
    slugify,
    tag_eval,
):
    _emo2fam = {slugify(e): c for c, es in CLUSTERS.items() for e in es}
    _metrics = ["family agreement", "cosine, top word (1-vs-1)", "cosine, weighted label (1-vs-3)"]
    _rows = []
    for _run, _samples in EVAL_SAMPLES.items():
        for _split in ("within", "cross"):
            _by_id = {r["id"]: r["reply"] for r in _samples[_split]}
            _vals = {m: [] for m in _metrics}
            for _r in EVAL_SETS[_split]:
                _p = tag_eval.parse_reply(_by_id[_r["id"]])
                _first = _p["emotions"][0] if _p["compliant"] and _p["emotions"] else None
                _tfam = _emo2fam.get(TEACHER.top_word(_r["id"]))
                _mfam = _emo2fam.get(slugify(_first)) if _first else None
                _vals["family agreement"].append(1.0 if (_mfam is not None and _mfam == _tfam) else 0.0)
                _c1 = SIM.sim(_first, TEACHER.top_word(_r["id"]))
                _c3 = SIM.centroid_sim(_first, TEACHER.weighted(_r["id"]))
                if _c1 is not None:
                    _vals["cosine, top word (1-vs-1)"].append(_c1)
                if _c3 is not None:
                    _vals["cosine, weighted label (1-vs-3)"].append(_c3)
            for _m in _metrics:
                _s = mean_and_ci(_vals[_m])
                _label = "within (trained families)" if _split == "within" else "cross (never-trained families)"
                _rows.append({"run": _run, "split": _label, "metric": _m,
                              "mean": _s["mean"], "lo": _s["lo"], "hi": _s["hi"]})
    _df = pl.DataFrame(_rows)
    _run_order = list(EVAL_SAMPLES)
    _split_order = ["within (trained families)", "cross (never-trained families)"]
    _panels = []
    for _i, _m in enumerate(_metrics):
        _sub = _df.filter(pl.col("metric") == _m)
        _y = alt.Y("run:N", sort=_run_order, title=None,
                   axis=alt.Axis(labelLimit=220) if _i == 0 else None)
        _bars = alt.Chart(_sub).mark_bar().encode(
            x=alt.X("mean:Q", scale=alt.Scale(domain=[0, 1]), title=None),
            y=_y, yOffset=alt.YOffset("split:N", sort=_split_order),
            color=alt.Color("split:N", sort=_split_order, title=None, scale=alt.Scale(scheme="tableau10")),
            tooltip=["run", "split", alt.Tooltip("mean:Q", format=".3f")],
        )
        _ci = alt.Chart(_sub).mark_rule(color="#333").encode(
            x="lo:Q", x2="hi:Q", y=_y, yOffset=alt.YOffset("split:N", sort=_split_order))
        _panels.append(alt.layer(_bars, _ci).properties(width=170, height=250, title=alt.Title(_m, fontSize=12)))
    save_chart(
        alt.hconcat(*_panels).properties(
            title="The three report-accuracy metrics per checkpoint (frozen probe labels)"),
        "headline_metrics_by_checkpoint",
        caption=(
            "The three report-accuracy metrics for every checkpoint, scored against the frozen "
            "probe labels on one deterministically decoded reply per held-out prompt (260 "
            "trained-family, 77 never-trained-family prompts); intervals are 95% prompt-level "
            "bootstrap. The cosine metrics can only be computed on replies whose tag names a "
            "known emotion; family agreement counts every reply, with malformed ones as wrong."
        ),
        takeaway=(
            "Single-reply accuracy survives the healthy preference runs and is paid by the "
            "tag-only ones: the uncapped whole-reply arm matches the supervised parent on "
            "trained families (family agreement 53% vs 53%, weighted-label cosine 0.59 vs 0.61) "
            "and sits at or above it on never-trained families on all three metrics (53% vs 44% "
            "family agreement), within overlapping intervals, while the tag-only pilot falls to "
            "0.50 weighted-label cosine against the parent's 0.61. The degenerate uncapped "
            "tag-only arm splits the metrics: family agreement collapses to 10% because a reply "
            "that never closes its tag counts as wrong, while its few surviving tags score "
            "highest of any checkpoint (0.76). The stage's demonstrated gain is consistency "
            "across samples, not single-reply accuracy."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(
    CLUSTERS,
    EVAL_SAMPLES,
    EVAL_SETS,
    SIM,
    TEACHER,
    alt,
    pl,
    save_chart,
    slugify,
    tag_eval,
):
    _emo2fam = {slugify(e): c for c, es in CLUSTERS.items() for e in es}
    _metrics = ["family agreement", "cosine, top word (1-vs-1)", "cosine, weighted label (1-vs-3)"]
    # The degenerate tag-only uncapped arm is omitted: 82% of its replies are
    # unscorable, leaving 2-6 usable replies per family.
    _runs = {k: v for k, v in EVAL_SAMPLES.items() if k != "uncapped-800 (100 steps)"}
    _rows = []
    for _run, _samples in _runs.items():
        _acc = {}
        for _split in ("within", "cross"):
            _by_id = {r["id"]: r["reply"] for r in _samples[_split]}
            for _r in EVAL_SETS[_split]:
                _fam = _r["cluster"] + (" · never trained" if _split == "cross" else "")
                _p = tag_eval.parse_reply(_by_id[_r["id"]])
                _first = _p["emotions"][0] if _p["compliant"] and _p["emotions"] else None
                _tfam = _emo2fam.get(TEACHER.top_word(_r["id"]))
                _mfam = _emo2fam.get(slugify(_first)) if _first else None
                _d = _acc.setdefault(_fam, {m: [] for m in _metrics})
                _d["family agreement"].append(1.0 if (_mfam is not None and _mfam == _tfam) else 0.0)
                _c1 = SIM.sim(_first, TEACHER.top_word(_r["id"]))
                _c3 = SIM.centroid_sim(_first, TEACHER.weighted(_r["id"]))
                if _c1 is not None:
                    _d["cosine, top word (1-vs-1)"].append(_c1)
                if _c3 is not None:
                    _d["cosine, weighted label (1-vs-3)"].append(_c3)
        for _fam, _d in _acc.items():
            for _m in _metrics:
                if _d[_m]:
                    _rows.append({"run": _run, "family": _fam, "metric": _m, "mean": sum(_d[_m]) / len(_d[_m])})
    _df = pl.DataFrame(_rows)
    _run_order = list(_runs)
    _fam_order = sorted({r["family"] for r in _rows}, key=lambda f: (" · never trained" in f, f))
    _panels = []
    for _i, _m in enumerate(_metrics):
        _panels.append(
            alt.Chart(_df.filter(pl.col("metric") == _m))
            .mark_point(filled=True, size=55)
            .encode(
                x=alt.X("mean:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                y=alt.Y("family:N", sort=_fam_order, title=None,
                        axis=alt.Axis(labelLimit=240) if _i == 0 else None),
                color=alt.Color("run:N", sort=_run_order, title=None, scale=alt.Scale(scheme="tableau10")),
                tooltip=["run", "family", alt.Tooltip("mean:Q", format=".3f")],
            )
            .properties(width=170, height=250, title=alt.Title(_m, fontSize=12))
        )
    save_chart(
        alt.hconcat(*_panels).properties(
            title="Report accuracy by emotion family (frozen probe labels)"),
        "headline_metrics_by_family",
        caption=(
            "The same three metrics broken down by the family of the probe label, for the five "
            "checkpoints with an intact tag format (the degenerate tag-only uncapped arm is "
            "omitted: 82% of its replies are unscorable, leaving 2-6 usable replies per family). "
            "Each family contributes 18-42 held-out prompts; the two never-trained families come "
            "from the cross set."
        ),
        takeaway=(
            "The spread between families dwarfs the spread between checkpoints: every checkpoint "
            "reads exuberant joy and peaceful contentment well (weighted-label cosine 0.66-0.90) "
            "and depleted disengagement and hostile anger poorly (0.16-0.47). The tag-only runs' "
            "losses concentrate in those hardest families, with hostile anger falling from the "
            "parent's 0.44 to 0.16 (pilot) and 0.30 (capped); the uncapped whole-reply arm "
            "restores the parent's level there (0.44) and the capped one part of it (0.33). The "
            "capped tag-only arm's loss of peaceful contentment (top-word cosine 0.86 to 0.63) "
            "is the accuracy face of its neutral-anchor break."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(
    CLUSTERS,
    EVAL_SAMPLES,
    EVAL_SETS,
    NEUTRAL_TAG,
    SIM,
    TEACHER,
    alt,
    mean_and_ci,
    pl,
    save_chart,
    slugify,
    tag_eval,
):
    _emo2fam = {slugify(e): c for c, es in CLUSTERS.items() for e in es}
    # Valence halves of the 10 families (5 + 5); neutral tasks are their own group,
    # measured as the exact-anchor share since no probe label exists for them.
    _positive = {"exuberant_joy", "compassionate_gratitude", "peaceful_contentment",
                 "competitive_pride", "playful_amusement"}
    _metrics = ["family agreement", "cosine, top word (1-vs-1)", "cosine, weighted label (1-vs-3)"]
    _rows = []
    for _run, _samples in EVAL_SAMPLES.items():
        _groups = {"positive": {m: [] for m in _metrics}, "negative": {m: [] for m in _metrics}}
        for _split in ("within", "cross"):
            _by_id = {r["id"]: r["reply"] for r in _samples[_split]}
            for _r in EVAL_SETS[_split]:
                _val = "positive" if _r["cluster"] in _positive else "negative"
                _p = tag_eval.parse_reply(_by_id[_r["id"]])
                _first = _p["emotions"][0] if _p["compliant"] and _p["emotions"] else None
                _tfam = _emo2fam.get(TEACHER.top_word(_r["id"]))
                _mfam = _emo2fam.get(slugify(_first)) if _first else None
                _groups[_val]["family agreement"].append(1.0 if (_mfam is not None and _mfam == _tfam) else 0.0)
                _c1 = SIM.sim(_first, TEACHER.top_word(_r["id"]))
                _c3 = SIM.centroid_sim(_first, TEACHER.weighted(_r["id"]))
                if _c1 is not None:
                    _groups[_val]["cosine, top word (1-vs-1)"].append(_c1)
                if _c3 is not None:
                    _groups[_val]["cosine, weighted label (1-vs-3)"].append(_c3)
        for _val, _g in _groups.items():
            for _m in _metrics:
                _s = mean_and_ci(_g[_m])
                _rows.append({"run": _run, "group": _val, "metric": _m,
                              "mean": _s["mean"], "lo": _s["lo"], "hi": _s["hi"]})
        _anchor_vals = []
        for _r2 in _samples["neutral"]:
            _p2 = tag_eval.parse_reply(_r2["reply"])
            _anchor_vals.append(1.0 if (_p2["compliant"] and ", ".join(_p2["emotions"]) == NEUTRAL_TAG) else 0.0)
        _s = mean_and_ci(_anchor_vals)
        _rows.append({"run": _run, "group": "neutral", "metric": "neutral: exact anchor",
                      "mean": _s["mean"], "lo": _s["lo"], "hi": _s["hi"]})
    _df = pl.DataFrame(_rows)
    _run_order = list(EVAL_SAMPLES)
    _group_order = ["positive", "negative", "neutral"]
    _panels = []
    for _i, _m in enumerate(_metrics + ["neutral: exact anchor"]):
        _sub = _df.filter(pl.col("metric") == _m)
        _y = alt.Y("run:N", sort=_run_order, title=None,
                   axis=alt.Axis(labelLimit=220) if _i == 0 else None)
        _bars = alt.Chart(_sub).mark_bar().encode(
            x=alt.X("mean:Q", scale=alt.Scale(domain=[0, 1]), title=None),
            y=_y, yOffset=alt.YOffset("group:N", sort=_group_order),
            color=alt.Color("group:N", sort=_group_order, title=None, scale=alt.Scale(scheme="tableau10")),
            tooltip=["run", "group", alt.Tooltip("mean:Q", format=".3f")],
        )
        _ci = alt.Chart(_sub).mark_rule(color="#333").encode(
            x="lo:Q", x2="hi:Q", y=_y, yOffset=alt.YOffset("group:N", sort=_group_order))
        _w = 150 if _i < 3 else 110
        _panels.append(alt.layer(_bars, _ci).properties(width=_w, height=240, title=alt.Title(_m, fontSize=12)))
    save_chart(
        alt.hconcat(*_panels).properties(
            title="Report accuracy by emotional valence (frozen probe labels)"),
        "headline_metrics_by_valence",
        caption=(
            "The three report-accuracy metrics with the ten families pooled into positive "
            "(joy, gratitude, contentment, pride, amusement; 152 prompts) and negative (fear, "
            "despair, anger, disengagement, suspicion; 185 prompts) halves, trained and "
            "never-trained families together; the fourth panel is the neutral tasks' "
            "exact-anchor share (no probe label exists for them). 95% prompt-level bootstrap "
            "intervals; the similarity panels for the degenerate tag-only uncapped arm rest on "
            "its scorable minority."
        ),
        takeaway=(
            "Wherever the channel is intact, accuracy orders neutral, then positive, then "
            "negative: the neutral anchor is close to perfect for every checkpoint that did "
            "not break it, and positive-family reports beat negative ones on both similarity "
            "metrics for every checkpoint (parent 0.67 vs 0.53 weighted-label, uncapped "
            "whole-reply 0.66 vs 0.53), a gap training leaves unchanged. Family agreement "
            "shows a much smaller gap (about 5 points, and none for the uncapped whole-reply "
            "arm), so the valence pattern lives in how close the reported word lands, not in "
            "how often the family is right."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(CLUSTERS, EVAL_SAMPLES, EVAL_SETS, SIM, TEACHER, alt, pl, save_chart, slugify, tag_eval):
    _emo2fam = {slugify(e): c for c, es in CLUSTERS.items() for e in es}
    _intact = {k: v for k, v in EVAL_SAMPLES.items() if k != "uncapped-800 (100 steps)"}
    _per: dict = {}
    _per_run: dict = {}
    for _run, _samples in _intact.items():
        for _split in ("within", "cross"):
            _by_id = {r["id"]: r["reply"] for r in _samples[_split]}
            for _r in EVAL_SETS[_split]:
                _p = tag_eval.parse_reply(_by_id[_r["id"]])
                _first = _p["emotions"][0] if _p["compliant"] and _p["emotions"] else None
                _c3 = SIM.centroid_sim(_first, TEACHER.weighted(_r["id"]))
                if _c3 is not None:
                    _per.setdefault(_r["emotion"], []).append(_c3)
                    _per_run.setdefault((_r["emotion"], _run), []).append(_c3)
    _pooled = sorted(
        ((e, sum(v) / len(v), len(v)) for e, v in _per.items() if len(v) >= 15),
        key=lambda x: x[1],
    )
    _chosen = _pooled[-3:][::-1] + _pooled[:3][::-1]  # best three, then worst three
    _order = [e for e, _, _ in _chosen]
    _dots = pl.DataFrame([
        {"emotion": _e, "run": _run, "mean": sum(_v) / len(_v)}
        for (_e, _run), _v in _per_run.items()
        if _e in set(_order)
    ])
    _pool_df = pl.DataFrame([{"emotion": _e, "pooled": _m} for _e, _m, _n in _chosen])
    _chart = alt.layer(
        alt.Chart(_pool_df).mark_tick(color="#20242B", thickness=2, size=18).encode(
            x=alt.X("pooled:Q", scale=alt.Scale(domain=[0, 1]),
                    title="cosine to the weighted label (1-vs-3)"),
            y=alt.Y("emotion:N", sort=_order, title=None),
        ),
        alt.Chart(_dots).mark_point(filled=True, size=70, opacity=0.85).encode(
            x="mean:Q",
            y=alt.Y("emotion:N", sort=_order, title=None),
            color=alt.Color("run:N", sort=list(_intact), title=None, scale=alt.Scale(scheme="tableau10")),
            tooltip=["emotion", "run", alt.Tooltip("mean:Q", format=".3f")],
        ),
    ).properties(width=430, height=210, title="The best- and worst-read emotions (1-vs-3)")
    save_chart(
        _chart,
        "extreme_emotions_1v3",
        caption=(
            "The three highest- and three lowest-scoring emotions on the weighted-label "
            "similarity, selected by the pooled mean over the five intact checkpoints (black "
            "tick; 37-89 scored replies per emotion) with one dot per checkpoint. Best: serene "
            "and peaceful (peaceful contentment), optimistic (exuberant joy). Worst: worn out "
            "(depleted disengagement), irritated (hostile anger), playful (never-trained "
            "playful amusement)."
        ),
        takeaway=(
            "The extremes follow the valence pattern: the best-read emotions are serene "
            "(0.85), optimistic (0.83), and peaceful (0.81), all from calm or joyful "
            "families, while the worst are worn out (0.17), irritated (0.32), and playful "
            "(0.32), from the depleted, hostile, and never-trained playful families. The "
            "spread between best and worst dwarfs most training differences, with one "
            "exception worth flagging: the supervised parent reads worn out at 0.43, and "
            "every preference run reads it at 0.20 or below (0.01-0.20) — small samples "
            "(12-13 prompts each), but the direction is consistent across all four runs, "
            "so the preference stage specifically degrades the channel's weakest emotion."
        ),
        notebook=__file__,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Greedy-battery and probe context (numbers, no exhibit)

    Greedy battery: capped-200 keeps format (99%) and roughly the parent's mapping
    (family-vs-teacher 51% within / 40% cross vs 53%/44%), with the graded within read
    at 0.665 rank percentile (parent 0.714). Uncapped-800's greedy compliance is 18%
    within — the runaway-tag mode — with its scorable minority at 0.81. Against the
    current-probe teacher both arms score modestly better than against the frozen one
    (capped within 1-vs-3: 0.564 → 0.581, paired Δ +0.016 [+0.006, +0.027]); the
    uncapped arm's current-probe teacher itself has drifted on 37% of top words, the
    largest drift measured. Leakage and capability were judged for capped-200 only —
    83% of the uncapped arm's visible replies are empty (the tag never closes), so a
    tonal judgment would be meaningless. Capped-200's judge reads are the first
    non-null movement in any run: visible-reply tonal charge 1.475 vs the parent's
    1.387 (base 1.325, same 80-item protocol) and capability at-least-equal 76%
    vs the parent's 82% — small shifts, but the preference stage is no longer
    leaving the visible behaviour untouched.
    """)
    return


if __name__ == "__main__":
    app.run()
