import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import math
    from collections import Counter
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from name_that_feeling.emotion_vectors.taxonomy import load_clusters, slugify
    from name_that_feeling.evals import tag_eval
    from name_that_feeling.evals.similarity import EmotionSimilarity
    from name_that_feeling.generation import sft
    from name_that_feeling.reporting import save_chart

    alt.data_transformers.disable_max_rows()
    return Counter, EmotionSimilarity, Path, alt, json, load_clusters, math, mo, pl, save_chart, sft, slugify, tag_eval


@app.cell
def _(EmotionSimilarity, Path, json, load_clusters, sft, slugify, tag_eval):
    HERE = Path(__file__).parents[1]  # the 04 experiment dir
    EXPERIMENTS = HERE.parent
    PILOT = EXPERIMENTS / "03-training-pilot"
    SFT_DIR = PILOT / "data" / "sft"

    # Every checkpoint that has been K-sampled (sample_stability.py writes one folder per run).
    RUNS = {
        p.parent.name: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((HERE / "data" / "stability").glob("*/samples.json"))
    }

    CLUSTERS = load_clusters(EXPERIMENTS / "01-emotion-vectors" / "clusters.json")
    EMO2FAM = tag_eval.family_lookup(CLUSTERS)
    SIM = EmotionSimilarity.load(EXPERIMENTS / "01-emotion-vectors" / "data" / "similarity" / "layer_21.json")

    # Probe teacher tags: the pilot's locked strategy over the full-dataset stats
    # (identical to summarize_runs.py / evaluate.py -- deterministic re-derivation).
    _completions = [
        json.loads(x)
        for x in (PILOT / "data" / "completions" / "unconditioned.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    _stats = sft.per_emotion_stats(_completions)
    _proj_by_id = {r["id"]: r["probe"]["projections"] for r in _completions}
    _tag_config = json.loads((SFT_DIR / "split.json").read_text(encoding="utf-8"))["tag_config"]

    def teacher_of(msg_id: str) -> list[str]:
        picks = sft.select_tag_emotions(_proj_by_id[msg_id], CLUSTERS, stats=_stats, **_tag_config)
        return [e.replace("_", " ") for e, _w in picks]

    def first_in_taxonomy(emotions: list[str]) -> str | None:
        """First emitted emotion the similarity matrix knows (mirrors tag_eval's precedence)."""
        for e in emotions:
            if SIM.index(e) is not None:
                return slugify(e)
        return None

    # The tags actually trained on (train rows only) -- the reference for exact recovery
    # of the trained mapping across draws.
    TRAINED_TAGS = {
        r["id"]: [slugify(e) for e, _w in r["emotions"]]
        for r in (
            json.loads(x)
            for x in (SFT_DIR / "train_tags.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()
        )
    }

    # two-epochs is the gold-standard checkpoint -- foreground it (blue) in cross-run exhibits.
    RUN_COLOR = {"two-epochs": "#4c78a8", "pilot-with-neutral": "#f58518"}
    return EMO2FAM, RUN_COLOR, RUNS, SIM, TRAINED_TAGS, first_in_taxonomy, teacher_of


@app.cell
def _(RUNS, mo):
    _loaded = (
        "\n".join(
            f"- `{name}` — k={p['meta']['k']}, temperature {p['meta']['temperature']}, "
            f"{p['meta']['n_prompts']} prompts ({', '.join(f'{s} {n}' for s, n in p['meta']['sets'].items())})"
            for name, p in RUNS.items()
        )
        or "*none yet — run `sample_stability.py` first*"
    )
    mo.md(f"""
    # Is the emitted tag a deterministic readout or a distribution?

    Every eval so far sampled greedily. This notebook reads `data/stability/<run>/samples.json`
    (from `sample_stability.py`): K independent temperature-1 replies per prompt, per
    checkpoint, over the three held-out sets plus a seeded family-balanced subset of the
    emotion train messages. Four questions: (1) how spread out is the emitted `<emotion>` tag
    for a fixed prompt across draws; (2) per draw, how close is the tag to the probe teacher
    label on the graded distance metric; (3) how do held-out prompts distribute over the
    consistency buckets (consistently right / consistently wrong / inconsistent vs the
    teacher) that size the pool for the planned preference-tuning run — pairs exist only
    where sampling varies; (4) is the trained mapping more stable under sampling than the
    held-out one, and how often do draws recover the exact trained tag?

    Loaded sampling pools:

    {_loaded}
    """)
    return


@app.cell
def _(Counter, EMO2FAM, RUNS, math, slugify, tag_eval):
    # Per-prompt tag spread across the K draws. A "tag" is the full comma-separated emotion
    # list, compared after slugification; draws whose reply opens with no tag are excluded
    # from the spread statistics and show up in compliance_rate instead.
    STABILITY_ROWS = []
    for _run, _payload in RUNS.items():
        for _s in _payload["samples"]:
            _parsed = [tag_eval.parse_reply(r) for r in _s["replies"]]
            _k = len(_parsed)
            _tags = [", ".join(slugify(e) for e in p["emotions"]) for p in _parsed if p["emotions"]]
            _firsts = [slugify(p["emotions"][0]) for p in _parsed if p["emotions"]]
            _fams = [f for p in _parsed if p["emotions"] and (f := tag_eval.top_family(p["emotions"], EMO2FAM))]
            _fam_counts = Counter(_fams)
            _entropy = (
                -sum((c / len(_fams)) * math.log2(c / len(_fams)) for c in _fam_counts.values())
                if _fams
                else None
            )
            STABILITY_ROWS.append(
                {
                    "run": _run,
                    "id": _s["id"],
                    "set": _s["set"],
                    "k": _k,
                    "compliance_rate": sum(p["compliant"] for p in _parsed) / _k,
                    "modal_tag_share": (max(Counter(_tags).values()) / len(_tags)) if _tags else None,
                    "n_distinct_first": len(set(_firsts)),
                    "family_entropy_bits": _entropy,
                    "top_tags": "; ".join(f"{t} x{c}" for t, c in Counter(_tags).most_common(5)),
                }
            )
    return (STABILITY_ROWS,)


@app.cell
def _(EMO2FAM, RUNS, SIM, first_in_taxonomy, tag_eval, teacher_of):
    # Per-draw score vs the probe teacher, on BOTH metric families: the graded distance
    # metrics (rank percentile and signed cosine of the draw's first in-taxonomy emotion
    # relative to the teacher's first emotion) and the binary family-agreement score the
    # earlier evals used. The train set is scored too (its teacher tags re-derive
    # identically); neutral is not (the anchor tag is fixed, never probe-read).
    DRAW_ROWS = []
    for _run, _payload in RUNS.items():
        for _s in _payload["samples"]:
            if _s["set"] not in ("within", "cross", "train"):
                continue
            _teacher_first = first_in_taxonomy(teacher_of(_s["id"]))
            _teacher_family = EMO2FAM.get(_teacher_first) if _teacher_first else None
            for _j, _reply in enumerate(_s["replies"]):
                _model_first = first_in_taxonomy(tag_eval.parse_reply(_reply)["emotions"])
                _model_family = EMO2FAM.get(_model_first) if _model_first else None
                DRAW_ROWS.append(
                    {
                        "run": _run,
                        "id": _s["id"],
                        "set": _s["set"],
                        "draw": _j,
                        "model_first": _model_first,
                        "teacher_first": _teacher_first,
                        "cosine": SIM.sim(_model_first, _teacher_first),
                        "rank_pct": SIM.rank_percentile(_teacher_first, _model_first),
                        "family_agreement": (_model_family == _teacher_family)
                        if _model_family and _teacher_family
                        else None,
                    }
                )
    return (DRAW_ROWS,)


@app.cell
def _(Counter, DRAW_ROWS):
    # Consistency buckets for preference-pair pool sizing (held-out sets only): a prompt is
    # consistently right if every scorable draw sits at rank percentile >= 0.8 against the
    # teacher, consistently wrong if every scorable draw sits <= 0.4, inconsistent
    # otherwise; unscorable if no draw yields an in-taxonomy tag.
    _pcts_by_prompt: dict = {}
    for _r in DRAW_ROWS:
        if _r["set"] in ("within", "cross"):
            _pcts_by_prompt.setdefault((_r["run"], _r["id"], _r["set"]), []).append(_r["rank_pct"])

    BUCKET_ROWS = []
    for (_run, _id, _set), _pcts in _pcts_by_prompt.items():
        _scorable = [p for p in _pcts if p is not None]
        if not _scorable:
            _bucket = "unscorable"
        elif all(p >= 0.8 for p in _scorable):
            _bucket = "consistently right"
        elif all(p <= 0.4 for p in _scorable):
            _bucket = "consistently wrong"
        else:
            _bucket = "inconsistent"
        BUCKET_ROWS.append(
            {
                "run": _run,
                "id": _id,
                "set": _set,
                "bucket": _bucket,
                "n_scorable": len(_scorable),
                "rank_pct_min": min(_scorable) if _scorable else None,
                "rank_pct_max": max(_scorable) if _scorable else None,
            }
        )

    BUCKET_ORDER = ["consistently right", "inconsistent", "consistently wrong", "unscorable"]
    BUCKET_SHARES = []
    for _run in sorted({r["run"] for r in BUCKET_ROWS}):
        for _set in ("within", "cross"):
            _rows = [r for r in BUCKET_ROWS if r["run"] == _run and r["set"] == _set]
            _counts = Counter(r["bucket"] for r in _rows)
            for _i, _b in enumerate(BUCKET_ORDER):
                BUCKET_SHARES.append(
                    {
                        "run": _run,
                        "set": _set,
                        "bucket": _b,
                        "bucket_order": _i,
                        "share": (_counts.get(_b, 0) / len(_rows)) if _rows else 0.0,
                        "n_prompts": len(_rows),
                    }
                )
    return BUCKET_ORDER, BUCKET_ROWS, BUCKET_SHARES


@app.cell
def _(RUNS, tag_eval):
    # Neutral anchor under sampling: exact "calm, attentive" tags per prompt across draws,
    # and whether any draw emits a charged (compliant but non-neutral) tag.
    NEUTRAL_TAG = {"calm", "attentive"}
    NEUTRAL_ROWS = []
    for _run, _payload in RUNS.items():
        for _s in _payload["samples"]:
            if _s["set"] != "neutral":
                continue
            _parsed = [tag_eval.parse_reply(r) for r in _s["replies"]]
            _exact = [p["compliant"] and {e.lower() for e in p["emotions"]} == NEUTRAL_TAG for p in _parsed]
            _charged = [p["compliant"] and {e.lower() for e in p["emotions"]} != NEUTRAL_TAG for p in _parsed]
            NEUTRAL_ROWS.append(
                {
                    "run": _run,
                    "id": _s["id"],
                    "k": len(_parsed),
                    "exact_neutral_rate": sum(_exact) / len(_parsed),
                    "n_charged": sum(_charged),
                    "any_charged": any(_charged),
                }
            )

    NEUTRAL_SUMMARY = []
    for _run in sorted({r["run"] for r in NEUTRAL_ROWS}):
        _rows = [r for r in NEUTRAL_ROWS if r["run"] == _run]
        NEUTRAL_SUMMARY.append(
            {
                "run": _run,
                "n_prompts": len(_rows),
                "mean_exact_neutral_rate": sum(r["exact_neutral_rate"] for r in _rows) / len(_rows),
                "share_prompts_any_charged": sum(r["any_charged"] for r in _rows) / len(_rows),
            }
        )
    return NEUTRAL_ROWS, NEUTRAL_SUMMARY


@app.cell
def _(Counter, RUNS, TRAINED_TAGS, slugify, tag_eval):
    # Train set only: exact recovery of the stored trained tag across draws -- the metric
    # the held-out sets cannot have. Greedy label recovery was ~48% top-1 emotion for the
    # 3-epoch pilot; this is its distribution under temperature-1 sampling.
    TRAIN_RECOVERY_ROWS = []
    for _run, _payload in RUNS.items():
        for _s in _payload["samples"]:
            if _s["set"] != "train" or _s["id"] not in TRAINED_TAGS:
                continue
            _trained = TRAINED_TAGS[_s["id"]]
            _parsed = [tag_eval.parse_reply(r) for r in _s["replies"]]
            _k = len(_parsed)
            _firsts = [slugify(p["emotions"][0]) if p["emotions"] else None for p in _parsed]
            _tags = [", ".join(slugify(e) for e in p["emotions"]) for p in _parsed if p["emotions"]]
            _modal = Counter(_tags).most_common(1)[0][0] if _tags else None
            TRAIN_RECOVERY_ROWS.append(
                {
                    "run": _run,
                    "id": _s["id"],
                    "k": _k,
                    "trained_tag": ", ".join(_trained),
                    "first_emotion_recovery_rate": sum(f == _trained[0] for f in _firsts) / _k,
                    "modal_tag_equals_trained": _modal == ", ".join(_trained),
                }
            )

    TRAIN_RECOVERY_SUMMARY = []
    for _run in sorted({r["run"] for r in TRAIN_RECOVERY_ROWS}):
        _rows = [r for r in TRAIN_RECOVERY_ROWS if r["run"] == _run]
        TRAIN_RECOVERY_SUMMARY.append(
            {
                "run": _run,
                "n_prompts": len(_rows),
                "mean_first_emotion_recovery": sum(r["first_emotion_recovery_rate"] for r in _rows) / len(_rows),
                "share_modal_equals_trained": sum(r["modal_tag_equals_trained"] for r in _rows) / len(_rows),
            }
        )
    return TRAIN_RECOVERY_ROWS, TRAIN_RECOVERY_SUMMARY


@app.cell
def _(DRAW_ROWS, NEUTRAL_SUMMARY, STABILITY_ROWS, TRAIN_RECOVERY_SUMMARY, mo, pl):
    # The 2-epoch vs 3-epoch comparison in one table: per-set aggregates by checkpoint.
    SUMMARY_ROWS = []
    for _run in sorted({r["run"] for r in STABILITY_ROWS}):
        for _set in ("train", "within", "cross", "neutral"):
            _st = [r for r in STABILITY_ROWS if r["run"] == _run and r["set"] == _set]
            if not _st:
                continue
            _shares = [r["modal_tag_share"] for r in _st if r["modal_tag_share"] is not None]
            _ents = [r["family_entropy_bits"] for r in _st if r["family_entropy_bits"] is not None]
            _pcts = [
                r["rank_pct"]
                for r in DRAW_ROWS
                if r["run"] == _run and r["set"] == _set and r["rank_pct"] is not None
            ]
            _coss = [
                r["cosine"] for r in DRAW_ROWS if r["run"] == _run and r["set"] == _set and r["cosine"] is not None
            ]
            _agrs = [
                r["family_agreement"]
                for r in DRAW_ROWS
                if r["run"] == _run and r["set"] == _set and r["family_agreement"] is not None
            ]
            SUMMARY_ROWS.append(
                {
                    "run": _run,
                    "set": _set,
                    "n_prompts": len(_st),
                    "compliance": round(sum(r["compliance_rate"] for r in _st) / len(_st), 3),
                    "modal_tag_share": round(sum(_shares) / len(_shares), 3) if _shares else None,
                    "distinct_firsts": round(sum(r["n_distinct_first"] for r in _st) / len(_st), 2),
                    "family_entropy_bits": round(sum(_ents) / len(_ents), 3) if _ents else None,
                    "draw_rank_pct_vs_teacher": round(sum(_pcts) / len(_pcts), 3) if _pcts else None,
                    "draw_cosine_vs_teacher": round(sum(_coss) / len(_coss), 3) if _coss else None,
                    "draw_family_agreement_vs_teacher": round(sum(_agrs) / len(_agrs), 3) if _agrs else None,
                }
            )
    mo.vstack(
        [
            mo.md("""
    ## Per-set aggregates by checkpoint

    `modal_tag_share` and `family_entropy_bits` average over prompts (draws with no tag
    excluded); `draw_*_vs_teacher` average over individual scorable draws. The train row is
    the trained-mapping reference the held-out rows are compared to. Neutral rows have no
    teacher score (the anchor tag is fixed, never probe-read).
    """),
            mo.ui.table(pl.DataFrame(SUMMARY_ROWS), selection=None)
            if SUMMARY_ROWS
            else mo.md("*no stability samples yet*"),
            mo.md("**Neutral anchor:** " + "; ".join(
                f"`{r['run']}` exact-neutral {r['mean_exact_neutral_rate']:.0%} of draws, "
                f"{r['share_prompts_any_charged']:.0%} of prompts emit at least one charged tag"
                for r in NEUTRAL_SUMMARY
            ) if NEUTRAL_SUMMARY else "**Neutral anchor:** *no data*"),
            mo.md("**Trained-tag recovery (train set):** " + "; ".join(
                f"`{r['run']}` recovers the trained first emotion on "
                f"{r['mean_first_emotion_recovery']:.0%} of draws, modal tag equals the trained tag on "
                f"{r['share_modal_equals_trained']:.0%} of prompts"
                for r in TRAIN_RECOVERY_SUMMARY
            ) if TRAIN_RECOVERY_SUMMARY else "**Trained-tag recovery (train set):** *no data*"),
        ]
    )
    return (SUMMARY_ROWS,)


@app.cell
def _(BUCKET_ORDER, BUCKET_SHARES, alt, mo, pl, save_chart):
    # Exhibit 1 (headline): the consistency-bucket shares that size the preference-pair pool.
    if not BUCKET_SHARES:
        _out = mo.md("*No stability samples yet -- exhibit skipped.*")
    else:
        _df = pl.DataFrame(BUCKET_SHARES)
        _bucket_colors = {
            "consistently right": "#4c78a8",
            "inconsistent": "#eeca3b",
            "consistently wrong": "#e45756",
            "unscorable": "#bab0ac",
        }
        _chart = (
            alt.Chart(_df)
            .mark_bar()
            .encode(
                x=alt.X("share:Q", axis=alt.Axis(format=".0%"), title="share of held-out prompts"),
                y=alt.Y("run:N", title=None),
                color=alt.Color(
                    "bucket:N",
                    scale=alt.Scale(domain=BUCKET_ORDER, range=[_bucket_colors[b] for b in BUCKET_ORDER]),
                    title=None,
                ),
                order=alt.Order("bucket_order:Q"),
                tooltip=["run", "set", "bucket", alt.Tooltip("share:Q", format=".1%"), "n_prompts"],
            )
            .properties(width=400, height=28 + 26 * _df["run"].n_unique())
            .facet(row=alt.Row("set:N", sort=["within", "cross"], title=None))
            .properties(title="Sampling-consistency buckets against the probe teacher label")
        )
        _runs = sorted(_df["run"].unique().to_list())
        _share = {(r["run"], r["set"], r["bucket"]): r["share"] for r in BUCKET_SHARES}
        _takeaway = (
            "Preference pairs can act only on prompts where sampling varies across the "
            "right/wrong thresholds: " + "; ".join(
                f"{run} leaves {_share.get((run, 'within', 'inconsistent'), 0):.0%} of within-family and "
                f"{_share.get((run, 'cross', 'inconsistent'), 0):.0%} of cross-family prompts inconsistent "
                f"({_share.get((run, 'within', 'consistently wrong'), 0):.0%} / "
                f"{_share.get((run, 'cross', 'consistently wrong'), 0):.0%} consistently wrong, "
                "reachable by heavier SFT only)"
                for run in _runs
            ) + "."
        )
        _out = save_chart(
            _chart,
            "dpo_bucket_shares",
            caption=(
                "Classification of each held-out prompt from the spread of its K temperature-1 "
                "draws, scored against the probe teacher label by the rank percentile of the "
                "first emitted emotion relative to the teacher's first emotion: consistently "
                "right (every scorable draw at or above 0.8), consistently wrong (every scorable "
                "draw at or below 0.4), inconsistent (draws span the thresholds), and unscorable "
                "(no draw yields an in-taxonomy tag). Shares by checkpoint and evaluation set."
            ),
            takeaway=_takeaway,
            notebook=__file__,
        )
    _out
    return


@app.cell
def _(RUN_COLOR, STABILITY_ROWS, alt, mo, pl, save_chart):
    # Exhibit 2: how concentrated is the per-prompt tag distribution across draws?
    _rows = [r for r in STABILITY_ROWS if r["set"] in ("within", "cross") and r["modal_tag_share"] is not None]
    if not _rows:
        _out = mo.md("*No stability samples yet -- exhibit skipped.*")
    else:
        _df = pl.DataFrame(_rows)
        _runs = sorted(_df["run"].unique().to_list())
        _chart = (
            alt.Chart(_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "modal_tag_share:Q",
                    bin=alt.Bin(extent=[0, 1], step=0.1),
                    axis=alt.Axis(format=".0%"),
                    title="modal-tag share across draws",
                ),
                y=alt.Y("count():Q", title="held-out prompts"),
                color=alt.Color(
                    "run:N",
                    scale=alt.Scale(domain=_runs, range=[RUN_COLOR.get(r, "#72b7b2") for r in _runs]),
                    title=None,
                ),
                xOffset=alt.XOffset("run:N"),
                tooltip=["run", alt.Tooltip("count():Q", title="prompts")],
            )
            .properties(width=420, height=220, title="Per-prompt concentration of the emitted tag under sampling")
        )
        _mean = {
            run: sum(r["modal_tag_share"] for r in _rows if r["run"] == run)
            / len([r for r in _rows if r["run"] == run])
            for run in _runs
        }
        _stable = {
            run: sum(r["modal_tag_share"] == 1.0 for r in _rows if r["run"] == run)
            / len([r for r in _rows if r["run"] == run])
            for run in _runs
        }
        _takeaway = "The tag channel is a distribution, not a deterministic readout: " + "; ".join(
            f"{run} emits its modal tag on {_mean[run]:.0%} of draws on the average prompt, and only "
            f"{_stable[run]:.0%} of prompts are fully stable across draws"
            for run in _runs
        ) + "."
        _out = save_chart(
            _chart,
            "modal_tag_share_distribution",
            caption=(
                "Distribution over held-out prompts (within- and cross-family sets pooled) of the "
                "modal-tag share: the fraction of tagged temperature-1 draws that emit the "
                "prompt's most frequent full tag, by checkpoint. A share of 1.0 means every "
                "tagged draw produced the identical emotion list."
            ),
            takeaway=_takeaway,
            notebook=__file__,
        )
    _out
    return


@app.cell
def _(NEUTRAL_SUMMARY, RUN_COLOR, alt, mo, pl, save_chart):
    # Exhibit 3: does the neutral anchor survive temperature-1 sampling?
    if not NEUTRAL_SUMMARY:
        _out = mo.md("*No stability samples yet -- exhibit skipped.*")
    else:
        _rows = [
            {"run": r["run"], "measure": "exact neutral tag, mean rate across draws", "value": r["mean_exact_neutral_rate"]}
            for r in NEUTRAL_SUMMARY
        ] + [
            {"run": r["run"], "measure": "prompts emitting at least one charged tag", "value": r["share_prompts_any_charged"]}
            for r in NEUTRAL_SUMMARY
        ]
        _df = pl.DataFrame(_rows)
        _runs = sorted({r["run"] for r in NEUTRAL_SUMMARY})
        _chart = (
            alt.Chart(_df)
            .mark_bar()
            .encode(
                x=alt.X("value:Q", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1]), title=None),
                y=alt.Y("measure:N", title=None),
                yOffset=alt.YOffset("run:N"),
                color=alt.Color(
                    "run:N",
                    scale=alt.Scale(domain=_runs, range=[RUN_COLOR.get(r, "#72b7b2") for r in _runs]),
                    title=None,
                ),
                tooltip=["run", "measure", alt.Tooltip("value:Q", format=".1%")],
            )
            .properties(width=420, height=140, title="The neutral anchor under temperature-1 sampling")
        )
        _takeaway = "The greedy 98-100% neutral anchor is an upper bound: " + "; ".join(
            f"{r['run']} keeps the exact neutral tag on {r['mean_exact_neutral_rate']:.0%} of draws while "
            f"{r['share_prompts_any_charged']:.0%} of neutral prompts emit at least one charged tag across draws"
            for r in NEUTRAL_SUMMARY
        ) + "."
        _out = save_chart(
            _chart,
            "neutral_anchor_under_sampling",
            caption=(
                "Neutral-set behaviour across K temperature-1 draws per prompt, by checkpoint: the "
                "mean rate of the exact fixed neutral tag (calm, attentive) over draws, and the "
                "share of neutral prompts for which at least one draw emits a charged tag instead."
            ),
            takeaway=_takeaway,
            notebook=__file__,
        )
    _out
    return


@app.cell
def _(RUN_COLOR, STABILITY_ROWS, alt, mo, pl, save_chart):
    # Exhibit 4: is the trained mapping more stable under sampling than the held-out one?
    _rows = [r for r in STABILITY_ROWS if r["set"] in ("train", "within", "cross") and r["modal_tag_share"] is not None]
    if not _rows:
        _out = mo.md("*No stability samples yet -- exhibit skipped.*")
    else:
        _df = pl.DataFrame(_rows)
        _runs = sorted(_df["run"].unique().to_list())
        _set_order = ["train", "within", "cross"]
        _chart = (
            alt.Chart(_df)
            .mark_boxplot(size=22)
            .encode(
                x=alt.X("set:N", sort=_set_order, title=None),
                xOffset=alt.XOffset("run:N"),
                y=alt.Y(
                    "modal_tag_share:Q",
                    axis=alt.Axis(format=".0%"),
                    scale=alt.Scale(domain=[0, 1]),
                    title="modal-tag share across draws",
                ),
                color=alt.Color(
                    "run:N",
                    scale=alt.Scale(domain=_runs, range=[RUN_COLOR.get(r, "#72b7b2") for r in _runs]),
                    title=None,
                ),
            )
            .properties(width=380, height=240, title="Tag stability on trained vs held-out messages")
        )
        _mean = {
            (run, s): (
                sum(r["modal_tag_share"] for r in _rows if r["run"] == run and r["set"] == s)
                / max(1, len([r for r in _rows if r["run"] == run and r["set"] == s]))
            )
            for run in _runs
            for s in _set_order
        }
        _takeaway = "Per-prompt modal-tag share on trained vs held-out messages: " + "; ".join(
            f"{run} train {_mean[(run, 'train')]:.0%} vs within {_mean[(run, 'within')]:.0%} "
            f"vs cross {_mean[(run, 'cross')]:.0%}"
            for run in _runs
        ) + "."
        _out = save_chart(
            _chart,
            "train_vs_heldout_tag_stability",
            caption=(
                "Distribution over prompts of the modal-tag share across K temperature-1 draws, "
                "on a seeded family-balanced subset of the emotion training messages (train) and "
                "the two held-out evaluation sets (within-family, cross-family), by checkpoint. "
                "Higher shares indicate a more deterministic tag for a fixed prompt."
            ),
            takeaway=_takeaway,
            notebook=__file__,
        )
    _out
    return


@app.cell
def _(RUN_COLOR, TRAIN_RECOVERY_SUMMARY, alt, mo, pl, save_chart):
    # Exhibit 5: exact recovery of the trained tag across draws (train set only).
    if not TRAIN_RECOVERY_SUMMARY:
        _out = mo.md("*No train-set stability samples yet -- exhibit skipped.*")
    else:
        _rows = [
            {
                "run": r["run"],
                "measure": "draws recovering the trained first emotion",
                "value": r["mean_first_emotion_recovery"],
            }
            for r in TRAIN_RECOVERY_SUMMARY
        ] + [
            {
                "run": r["run"],
                "measure": "prompts whose modal tag equals the trained tag",
                "value": r["share_modal_equals_trained"],
            }
            for r in TRAIN_RECOVERY_SUMMARY
        ]
        _df = pl.DataFrame(_rows)
        _runs = sorted({r["run"] for r in TRAIN_RECOVERY_SUMMARY})
        _chart = (
            alt.Chart(_df)
            .mark_bar()
            .encode(
                x=alt.X("value:Q", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1]), title=None),
                y=alt.Y("measure:N", title=None),
                yOffset=alt.YOffset("run:N"),
                color=alt.Color(
                    "run:N",
                    scale=alt.Scale(domain=_runs, range=[RUN_COLOR.get(r, "#72b7b2") for r in _runs]),
                    title=None,
                ),
                tooltip=["run", "measure", alt.Tooltip("value:Q", format=".1%")],
            )
            .properties(width=420, height=140, title="Recovery of the trained tag under temperature-1 sampling")
        )
        _takeaway = (
            "Recovery of the stored trained label under sampling (greedy reference: ~48% "
            "top-1 emotion for the 3-epoch pilot): " + "; ".join(
                f"{r['run']} recovers the trained first emotion on {r['mean_first_emotion_recovery']:.0%} "
                f"of draws and settles on the exact trained tag as its modal tag for "
                f"{r['share_modal_equals_trained']:.0%} of prompts"
                for r in TRAIN_RECOVERY_SUMMARY
            ) + "."
        )
        _out = save_chart(
            _chart,
            "trained_tag_recovery_under_sampling",
            caption=(
                "Exact recovery of the stored trained tag on the family-balanced train subset, by "
                "checkpoint: the mean fraction of temperature-1 draws whose first emitted emotion "
                "matches the trained label's first emotion, and the share of prompts whose modal "
                "full tag equals the trained tag."
            ),
            takeaway=_takeaway,
            notebook=__file__,
        )
    _out
    return


@app.cell
def _(BUCKET_ROWS, STABILITY_ROWS, mo, pl):
    # Drill-down (display only): the least stable prompts and what they actually emit.
    _bucket_of = {(r["run"], r["id"]): r["bucket"] for r in BUCKET_ROWS}
    _rows = sorted(
        (
            {**r, "bucket": _bucket_of.get((r["run"], r["id"]), "")}
            for r in STABILITY_ROWS
            if r["set"] in ("within", "cross") and r["modal_tag_share"] is not None
        ),
        key=lambda r: r["modal_tag_share"],
    )[:25]
    _view = (
        pl.DataFrame(_rows).select(
            "run", "id", "set", "bucket", "modal_tag_share", "n_distinct_first", "top_tags"
        )
        if _rows
        else pl.DataFrame()
    )
    mo.vstack(
        [
            mo.md("""
    ## Least stable prompts

    The 25 prompts with the lowest modal-tag share, with their most frequent tags across
    draws — where the preference-pair candidates live.
    """),
            mo.ui.table(_view, selection=None) if _rows else mo.md("*no stability samples yet*"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
