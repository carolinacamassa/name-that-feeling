"""Project every model of the Assistant axis read onto the axis, on each prompt set separately.

Pure local numpy over the pooled activations (``data/activations/<model>/`` for the WildChat
pool, ``data/reflection_activations/<model>/`` for the self-reflection prompts), the axis of
07-persona-assistant-axis (``data/full/axis.pt``, the row of the read's block, unit length) and
07-persona-activations' fitted affect axes, so the axis and the affect coordinates are read on
one set of activations. Nothing is sampled or extracted, and the two prompt sets never pool:
every number below belongs to one prompt set, one voice stratum and one read position.

The strata are the voice a prompt asks for: the WildChat pool audit's ``expected_voice``
(``data/pool_audit/prompt_labels.json``) and ``self_reflection_voice.yaml``. That audit found the
voice explains most of a whole-set reading, so ``assistant`` (a direct answer in the assistant's
own voice) is the stratum to quote, with the others beside it and ``all`` with its mix stated.

Dimension 3994 of the residual stream carries large activations on every model (about 28 on base
over a reply) and the axis loads on it, so it supplies nearly all of a model's absolute reading
and, at the pre-response token, moves with the negations. Every shift and residual is stored
with that dimension's contribution removed as well (``*_no_3994``); positions are not quoted.

``data/axis/`` holds:

- ``projections.parquet``  one row per transcript and read position: the projection on the unit
  axis (residual units, positive toward the Assistant), the same without dimension 3994, the raw
  valence, arousal and dominance coordinates, the voice, the reply's token count, whether it
  stopped at the cap;
- ``mean_activations.safetensors``  each model's mean residual, keys
  ``<prompt set>/<stratum>/<model>/<position>`` for the strata ``all`` and ``assistant``, the means
  taken over prompts (a prompt's draws first);
- ``roles.safetensors`` + ``roles.json``  the 275 role vectors and the default vector at the
  read's block, pulled once from the axis build on the Volume (``pull-roles``);
- ``role_alignment.json``  per prompt set and model, over the reply on the ``assistant`` stratum:
  the cosine of the model's shift with each role's displacement from the default (role vector
  minus default vector), and the share of the shift inside the roles' top principal components.
  ``off_axis`` (from the model's centre) and ``off_axis_from_base`` repeat the comparison with the
  Assistant axis component removed from the shift and from every displacement, since that
  component is shared by all the displacements: the shift's component along the axis, the length
  and the share of its squared length left off the axis, and the roles at each end of the cosine
  ranking.
- ``role_components.json``  the first principal components of the role vectors (each signed so the
  default lies on its positive side, with the five roles at each end) and, per prompt set and model,
  the position on each over the reply on the ``assistant`` stratum, from 0 at the lowest-scoring role
  to 1 at the highest, with base placed at the default's score and every other model by its mean
  activation's difference from base's.
  A shift is taken from the point the construction is centred on: the control for a persona, a
  negation around the control and a sum with the control counted once, base for the rest;
- ``summary.json``  per prompt set, stratum, position and model: the mean over prompts, the
  paired shift against base and against the control with a percentile bootstrap over prompts,
  the same without dimension 3994, the share of replies below the default's 25th percentile on
  the axis (the paper's capping threshold), the shift with capped replies dropped, the affect
  shifts in the base model's standard deviations, and the arithmetic's prediction.

The prediction is the same formula for every construction: a merge W + sum c_i dW_i under a
response that is additive in the updates sits at base + sum c_i (p_i - base), where p_i is the
trained checkpoint's own position. So a negation W - dW(p) is predicted at 2 base - p, a negation
around the control W + 2 dW(c) - dW(p) at 2 c - p, a plain sum at p + q - base, and a sum with the
control counted once at p + q - c. Prompts are the bootstrap's unit, with the same resampling
weights for every model on a stratum, so shifts and residuals are paired.

The WildChat row the neutral (no-wrapper) control trained on is left out for every model, as
every other read of that pool does.

    uv run python experiments/08-persona-arithmetic/project_axis.py pull-roles
    uv run python experiments/08-persona-arithmetic/project_axis.py project
    uv run python experiments/08-persona-arithmetic/project_axis.py roles    # role_alignment.json alone, from the stored means
    uv run python experiments/08-persona-arithmetic/project_axis.py role-components    # role_components.json
"""

import argparse
import datetime as dt
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from safetensors.numpy import load_file, save_file

from name_that_feeling.emotion_vectors.geometry import bootstrap_weights, percentile_interval
from name_that_feeling.emotion_vectors.models import stored_layer_key

import common

AXIS_DIR = common.DATA / "axis"
POSITIONS = ("user_mean", "pre_response", "reply_mean")
PROMPT_SETS = ("wildchat", "self_reflection")
AFFECT = ("valence", "arousal", "dominance")
MASSIVE_DIMENSION = 3994
ASSISTANT = "assistant"
ALL = "all"
OTHER = "other_voices"
N_BOOT = 2000
SEED = 20261103
TOP_ROLES = 8
ROLE_ENDS = 3
ROLE_COMPONENTS = 4


def axis_build_dir(cfg: dict) -> Path:
    return common.REPO_ROOT / "experiments" / cfg["axis_read"]["axis_experiment"] / "data" / "full"


def load_unit_axis(cfg: dict) -> tuple[np.ndarray, dict]:
    import torch

    build = axis_build_dir(cfg)
    block = int(cfg["axis_read"]["block"])
    axis = torch.load(build / "axis.pt", map_location="cpu")["axis"][block].double().numpy()
    report = common.read_json(build / "axis_report.json")
    if report["layer_indexing"] != "axis[i] = output of decoder layer i = hidden_states[i + 1]":
        raise RuntimeError(f"unexpected axis indexing: {report['layer_indexing']}")
    return axis / np.linalg.norm(axis), report


def pull_roles(cfg: dict) -> None:
    """The role vectors and the default vector at the read's block, from the axis build on the Volume."""
    import torch

    from name_that_feeling.infra import vectors_volume

    report = common.read_json(axis_build_dir(cfg) / "axis_report.json")
    run = report["run"]
    block = int(cfg["axis_read"]["block"])
    paths = sorted(e.path for e in vectors_volume.listdir(f"{run}/vectors/") if e.path.endswith(".pt"))

    def read(path: str) -> tuple[str, np.ndarray]:
        doc = torch.load(io.BytesIO(b"".join(vectors_volume.read_file(path))), map_location="cpu")
        return Path(path).stem, doc["vector"][block].float().numpy()

    with ThreadPoolExecutor(max_workers=16) as ex:
        got = dict(ex.map(read, paths))
    default = got.pop("default")
    roles = sorted(got)
    AXIS_DIR.mkdir(parents=True, exist_ok=True)
    save_file({"role_vectors": np.stack([got[r] for r in roles]), "default_vector": default}, str(AXIS_DIR / "roles.safetensors"))
    common.write_json(AXIS_DIR / "roles.json", {"run": run, "block": block, "roles": roles})
    print(f"{len(roles)} role vectors and the default at block {block} -> {AXIS_DIR}", flush=True)


def voice_labels(cfg: dict) -> dict[str, dict[str, str]]:
    """``{prompt set: {prompt id: voice}}``: the pool audit's labels, and this experiment's own."""
    audit = common.read_json(common.DATA / "pool_audit" / "prompt_labels.json")
    doc = yaml.safe_load((common.EXPERIMENT_DIR / "self_reflection_voice.yaml").read_text(encoding="utf-8"))
    reflection = {q["id"]: ASSISTANT for q in common.load_questions(cfg)}
    for voice, ids in doc["voices"].items():
        for i in ids:
            if i not in reflection:
                raise KeyError(f"self_reflection_voice.yaml names an unknown prompt: {i}")
            reflection[i] = voice
    return {"wildchat": {r["id"]: r["expected_voice"] for r in audit["rows"]}, "self_reflection": reflection}


def load_affect_directions(cfg: dict) -> dict[str, np.ndarray]:
    ac = common.activations_common(cfg)
    tensors = load_file(str(ac.affect_axes_path()))
    return {a: tensors[f"{a}_direction"].astype(np.float64) for a in AFFECT}


def transcripts_of(prompt_set: str, model: str, excluded: set[str]) -> tuple[Path, dict] | None:
    """The activation directory and ``{row id: (prompt id, finish)}`` for one model, or None if not read yet."""
    if prompt_set == "wildchat":
        folder = common.activations_dir(model)
        replies = common.pool_completions_path(model)
        if not (folder / "meta.json").exists() or not replies.exists():
            return None
        doc = common.read_json(replies)["replies"]
        return folder, {i: (i, r["finish"]) for i, r in doc.items() if i not in excluded}
    folder = common.reflection_activations_dir(model)
    if not (folder / "meta.json").exists():
        return None
    doc = common.read_json(common.activation_reflections_path(model))["reflections"]
    return folder, {common.draw_id(q, d["index"]): (q, d["finish"]) for q, draws in doc.items() for d in draws}


def stratum_mean(x: np.ndarray, prompts: list[str], wanted: set[str] | None) -> np.ndarray:
    """The mean over prompts, a prompt's draws first; an empty reply's NaN drops out of its own prompt."""
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(prompts):
        if wanted is None or p in wanted:
            groups.setdefault(p, []).append(i)
    return np.nanmean([np.nanmean(x[g], axis=0) for g in groups.values()], axis=0).astype(np.float32)


def read_model(cfg, prompt_set, model, excluded, unit_axis, affect, voices) -> tuple[list[dict], dict[str, np.ndarray]] | None:
    found = transcripts_of(prompt_set, model, excluded)
    if found is None:
        return None
    folder, rows = found
    meta = common.read_json(folder / "meta.json")
    pooled = load_file(str(folder / "pooled.safetensors"))
    key = stored_layer_key(int(cfg["axis_read"]["block"]))
    keep = [i for i, r in enumerate(meta["rows"]) if r["id"] in rows]
    if len(keep) != len(rows):
        raise RuntimeError(f"{model} on {prompt_set}: {len(rows)} replies, activations for {len(keep)}")
    records, means = [], {}
    prompts = [rows[meta["rows"][i]["id"]][0] for i in keep]
    assistant_prompts = {p for p in prompts if voices[p] == ASSISTANT}
    for pos in POSITIONS:
        x = pooled[f"{pos}/{key}"][keep].astype(np.float64)
        on_axis = x @ unit_axis
        without = on_axis - x[:, MASSIVE_DIMENSION] * unit_axis[MASSIVE_DIMENSION]
        on_affect = {a: x @ d for a, d in affect.items()}
        for j, i in enumerate(keep):
            row = meta["rows"][i]
            records.append({
                "prompt_set": prompt_set, "model": model, "position": pos, "row_id": row["id"], "prompt_id": prompts[j],
                "voice": voices[prompts[j]], "axis": float(on_axis[j]), "axis_no_3994": float(without[j]),
                **{a: float(on_affect[a][j]) for a in AFFECT},
                "n_reply_tokens": int(row["n_reply_tokens"]), "at_cap": rows[row["id"]][1] == "length",
            })
        means[f"{prompt_set}/{ALL}/{model}/{pos}"] = stratum_mean(x, prompts, None)
        means[f"{prompt_set}/{ASSISTANT}/{model}/{pos}"] = stratum_mean(x, prompts, assistant_prompts)
    return records, means


def per_prompt(frame: pl.DataFrame, value: str) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    """``{position: {model: per-prompt mean}}``, prompts in one fixed order."""
    prompts = sorted(frame["prompt_id"].unique().to_list())
    index = {p: i for i, p in enumerate(prompts)}
    out: dict[str, dict[str, np.ndarray]] = {p: {} for p in POSITIONS}
    means = frame.group_by(["position", "model", "prompt_id"]).agg(pl.col(value).mean())
    for (pos, model), part in means.group_by(["position", "model"]):
        a = np.full(len(prompts), np.nan)
        a[[index[p] for p in part["prompt_id"].to_list()]] = part[value].to_numpy()
        out[pos][model] = a
    return prompts, out


def interval(values: np.ndarray, weights: np.ndarray) -> dict:
    """The mean and its percentile interval under the shared prompt resampling (NaNs dropped from both)."""
    ok = ~np.isnan(values)
    w = weights[:, ok]
    boots = (w * values[ok]).sum(axis=1) / w.sum(axis=1)
    lo, hi = percentile_interval(boots)
    return {"mean": float(values[ok].mean()), "lo": float(lo), "hi": float(hi)}


def additive_prediction(cfg: dict, model: str, arrays: dict[str, np.ndarray]) -> np.ndarray | None:
    """base + sum c_i (p_i - base) per prompt, None for a model that is not a composition or lacks a term."""
    recipe = cfg["merges"].get(model)
    a = cfg["axis_read"]
    trained = {*a["personas"], common.CONTROL_ON_MODAL}
    if model == common.BASE or model in trained or recipe is None or common.BASE not in arrays:
        return None
    pred = arrays[common.BASE].copy()
    for persona, c in recipe.items():
        term = common.CONTROL_ON_MODAL if persona == "moodless" else persona
        if term not in arrays:
            return None
        pred += float(c) * (arrays[term] - arrays[common.BASE])
    return pred


def stratum_rows(cfg: dict, part: pl.DataFrame, threshold: float) -> dict:
    """Every position and model of one stratum of one prompt set."""
    prompts, arrays = per_prompt(part, "axis")
    _, without = per_prompt(part, "axis_no_3994")
    _, uncapped = per_prompt(part.filter(~pl.col("at_cap")), "axis") if part["at_cap"].any() else (prompts, arrays)
    index = {p: i for i, p in enumerate(sorted(part.filter(~pl.col("at_cap"))["prompt_id"].unique().to_list()))}
    affect = {a: per_prompt(part, a)[1] for a in AFFECT}
    weights = bootstrap_weights(len(prompts), N_BOOT, SEED)
    block: dict = {"n_prompts": len(prompts), "positions": {}}
    for pos in POSITIONS:
        rows = {}
        byp, byp_without = arrays[pos], without[pos]
        for model, values in byp.items():
            row: dict = {"mean": float(np.nanmean(values)), "sd_over_prompts": float(np.nanstd(values))}
            for name, ref in (("vs_base", common.BASE), ("vs_control", common.CONTROL_ON_MODAL)):
                if ref in byp and model != ref:
                    row[name] = interval(values - byp[ref], weights)
                    row[f"{name}_no_3994"] = interval(byp_without[model] - byp_without[ref], weights)
            pred = additive_prediction(cfg, model, byp)
            if pred is not None:
                pred_without = additive_prediction(cfg, model, byp_without)
                row["shift_from_base_predicted"] = float(np.nanmean(pred - byp[common.BASE]))
                row["shift_from_base_predicted_no_3994"] = float(np.nanmean(pred_without - byp_without[common.BASE]))
                row["residual_from_additive"] = interval(values - pred, weights)
                row["residual_from_additive_no_3994"] = interval(byp_without[model] - pred_without, weights)
            mask = (pl.col("position") == pos) & (pl.col("model") == model)
            raw = part.filter(mask)
            on_axis = raw["axis"].to_numpy()
            row["share_below_default_p25"] = float(np.nanmean(on_axis[~np.isnan(on_axis)] < threshold))
            row["share_at_cap"] = float(raw["at_cap"].mean())
            if model in uncapped[pos] and common.CONTROL_ON_MODAL in uncapped[pos] and model != common.CONTROL_ON_MODAL:
                pairs = uncapped[pos][model] - uncapped[pos][common.CONTROL_ON_MODAL]
                row["vs_control_without_capped_replies"] = float(np.nanmean(pairs))
                row["n_prompts_without_capped_replies"] = int((~np.isnan(pairs)).sum())
            if common.BASE in byp and model != common.BASE:
                ref = common.CONTROL_ON_MODAL if common.CONTROL_ON_MODAL in byp else common.BASE
                row["affect_vs_control_in_base_sd"] = {
                    a: float(np.nanmean(affect[a][pos][model] - affect[a][pos][ref]) / np.nanstd(affect[a][pos][common.BASE]))
                    for a in AFFECT
                } if model != ref else {a: 0.0 for a in AFFECT}
            rows[model] = row
        block["positions"][pos] = rows
    del index
    return block


def summarize(cfg: dict, frame: pl.DataFrame, report: dict) -> dict:
    threshold = float(report["checks"]["default_reply_projection_percentiles"]["25"])
    out: dict = {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(), "block": int(cfg["axis_read"]["block"]), "axis_run": report["run"],
        "axis_loading_on_dimension_3994": None,
        "default_reply_percentiles": report["checks"]["default_reply_projection_percentiles"],
        "n_boot": N_BOOT, "seed": SEED, "prompt_sets": {},
    }
    for prompt_set in PROMPT_SETS:
        part = frame.filter(pl.col("prompt_set") == prompt_set)
        if part.is_empty():
            continue
        strata = {ALL: part, OTHER: part.filter(pl.col("voice") != ASSISTANT)}
        for voice in part["voice"].unique().to_list():
            strata[voice] = part.filter(pl.col("voice") == voice)
        out["prompt_sets"][prompt_set] = {
            "strata": {name: stratum_rows(cfg, rows, threshold) for name, rows in strata.items() if not rows.is_empty()},
        }
    return out


def centre_of(cfg: dict, model: str) -> str:
    """The point a model's shift is read from: the control where the construction keeps its update once."""
    a = cfg["axis_read"]
    around = common.negation_of(cfg, model)
    if model in a["personas"] or model in a["sums_minus_control"] or (around and around[1] == "moodless"):
        return common.CONTROL_ON_MODAL
    return common.BASE


def role_alignment(cfg: dict, means: dict, unit_axis: np.ndarray) -> dict:
    """Which roles' displacement from the default each model's shift resembles, over the reply."""
    doc = load_file(str(AXIS_DIR / "roles.safetensors"))
    names = common.read_json(AXIS_DIR / "roles.json")["roles"]
    roles = doc["role_vectors"].astype(np.float64)
    displaced = roles - doc["default_vector"].astype(np.float64)
    displaced /= np.linalg.norm(displaced, axis=1, keepdims=True)
    _, _, components = np.linalg.svd(roles - roles.mean(axis=0), full_matrices=False)
    axis = unit_axis.astype(np.float64)
    displaced_off = displaced - np.outer(displaced @ axis, axis)
    displaced_off /= np.linalg.norm(displaced_off, axis=1, keepdims=True)

    def off_axis(vector: np.ndarray) -> dict:
        """The roles at each end of a shift's cosine ranking once the axis component is removed from both sides."""
        rest = vector - (vector @ axis) * axis
        cos = displaced_off @ (rest / np.linalg.norm(rest))
        order = np.argsort(-cos)
        return {
            "along_axis": float(vector @ axis),
            "off_axis_length": float(np.linalg.norm(rest)),
            "off_axis_share": float(rest @ rest / (vector @ vector)),
            "median_cosine": float(np.median(cos)),
            "toward": [{"role": names[i], "cosine": float(cos[i])} for i in order[:ROLE_ENDS]],
            "away": [{"role": names[i], "cosine": float(cos[i])} for i in order[::-1][:ROLE_ENDS]],
        }

    out: dict = {"roles": names, "prompt_sets": {}}
    for prompt_set in PROMPT_SETS:
        rows = {}
        for model in common.axis_read_models(cfg):
            if model == common.BASE:
                continue
            centre = centre_of(cfg, model)
            k, c = f"{prompt_set}/{ASSISTANT}/{model}/reply_mean", f"{prompt_set}/{ASSISTANT}/{centre}/reply_mean"
            if k not in means or c not in means:
                continue
            shift = means[k].astype(np.float64) - means[c].astype(np.float64)
            size = float(np.linalg.norm(shift))
            cos = displaced @ (shift / size)
            order = np.argsort(-cos)
            # The same comparison from base, whatever the construction is centred on: base is the
            # one point every model's displacement can be read from.
            from_base = means[k].astype(np.float64) - means[f"{prompt_set}/{ASSISTANT}/{common.BASE}/reply_mean"].astype(np.float64)
            cos_base = displaced @ (from_base / np.linalg.norm(from_base))
            rows[model] = {
                "centre": centre, "size": size, "cosine_with_axis": float(shift @ unit_axis / size),
                "share_in_top_5_role_components": float(((components[:5] @ shift) ** 2).sum() / size**2),
                "share_in_top_20_role_components": float(((components[:20] @ shift) ** 2).sum() / size**2),
                "median_role_cosine": float(np.median(cos)),
                "top_roles": [{"role": names[i], "cosine": float(cos[i])} for i in order[:TOP_ROLES]],
                "top_roles_from_base": [{"role": names[i], "cosine": float(cos_base[i])} for i in np.argsort(-cos_base)[:TOP_ROLES]],
                "cosines": [round(float(v), 4) for v in cos],
                "off_axis": off_axis(shift),
                "off_axis_from_base": off_axis(from_base),
            }
        if rows:
            out["prompt_sets"][prompt_set] = rows
    return out


def project(cfg: dict) -> None:
    unit_axis, report = load_unit_axis(cfg)
    affect = load_affect_directions(cfg)
    excluded = {r["id"] for r in common.load_pool(cfg)["rows"] if r.get("in_neutral_training_prompts")}
    voices = voice_labels(cfg)
    records, means, missing = [], {}, {s: [] for s in PROMPT_SETS}
    for prompt_set in PROMPT_SETS:
        for model in common.axis_read_models(cfg):
            got = read_model(cfg, prompt_set, model, excluded, unit_axis, affect, voices[prompt_set])
            if got is None:
                missing[prompt_set].append(model)
                continue
            records += got[0]
            means.update(got[1])
    frame = pl.DataFrame(records)
    AXIS_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(AXIS_DIR / "projections.parquet")
    save_file(means, str(AXIS_DIR / "mean_activations.safetensors"))
    summary = summarize(cfg, frame, report)
    summary["axis_loading_on_dimension_3994"] = float(unit_axis[MASSIVE_DIMENSION])
    summary["missing"] = missing
    common.write_json(AXIS_DIR / "summary.json", summary)
    common.write_json(AXIS_DIR / "role_alignment.json", role_alignment(cfg, means, unit_axis))
    for prompt_set, models in missing.items():
        if models:
            print(f"{prompt_set}: not read yet: {', '.join(models)}", flush=True)
    print(f"{frame.height:,} rows -> {AXIS_DIR}", flush=True)


def role_components(cfg: dict) -> None:
    """Where each model sits among the roles on the first principal components of the role vectors.

    The components are those of the role vectors around their mean, as in the Assistant axis paper's
    role PCA, each signed so that the default vector lies on its positive side. The role vectors come
    from the axis build's questions and the models' means from this experiment's prompts, so a model
    is placed at the default's score plus its difference from base on the same prompts (base answering
    these prompts is the default assistant). Its position is given between the lowest-scoring role (0)
    and the highest (1), with the role whose score is nearest, over the reply on the prompts answered
    in the assistant's own voice.
    """
    unit_axis, _ = load_unit_axis(cfg)
    doc = load_file(str(AXIS_DIR / "roles.safetensors"))
    names = common.read_json(AXIS_DIR / "roles.json")["roles"]
    roles = doc["role_vectors"].astype(np.float64)
    centre = roles.mean(axis=0)
    default = doc["default_vector"].astype(np.float64) - centre
    _, singular, vt = np.linalg.svd(roles - centre, full_matrices=False)
    comps = vt[:ROLE_COMPONENTS].copy()
    comps *= np.sign(comps @ default)[:, None]
    scores = (roles - centre) @ comps.T
    variance = singular**2 / (singular**2).sum()
    lowest, highest = scores.min(axis=0), scores.max(axis=0)
    default_score = default @ comps.T

    def placed(score: np.ndarray) -> list[dict]:
        return [{
            "position": float((score[k] - lowest[k]) / (highest[k] - lowest[k])),
            "nearest_role": names[int(np.argmin(np.abs(scores[:, k] - score[k])))],
        } for k in range(ROLE_COMPONENTS)]

    out: dict = {
        "block": int(cfg["axis_read"]["block"]), "n_roles": len(names),
        "components": [{
            "component": k + 1, "variance_share": float(variance[k]), "cosine_with_axis": float(comps[k] @ unit_axis),
            "negative_end": [names[i] for i in np.argsort(scores[:, k])[:5]],
            "positive_end": [names[i] for i in np.argsort(-scores[:, k])[:5]],
        } for k in range(ROLE_COMPONENTS)],
        "default": placed(default_score),
        "prompt_sets": {},
    }
    means = load_file(str(AXIS_DIR / "mean_activations.safetensors"))
    for prompt_set in PROMPT_SETS:
        base_key = f"{prompt_set}/{ASSISTANT}/{common.BASE}/reply_mean"
        if base_key not in means:
            continue
        base_mean = means[base_key].astype(np.float64)
        out["prompt_sets"][prompt_set] = {
            model: placed(default_score + (means[k].astype(np.float64) - base_mean) @ comps.T)
            for model in common.axis_read_models(cfg)
            if (k := f"{prompt_set}/{ASSISTANT}/{model}/reply_mean") in means
        }
    common.write_json(AXIS_DIR / "role_components.json", out)
    print(f"role components -> {AXIS_DIR / 'role_components.json'}", flush=True)


def roles_only(cfg: dict) -> None:
    """Rewrite role_alignment.json from the stored per-model means, leaving the projections as they are."""
    unit_axis, _ = load_unit_axis(cfg)
    means = load_file(str(AXIS_DIR / "mean_activations.safetensors"))
    common.write_json(AXIS_DIR / "role_alignment.json", role_alignment(cfg, means, unit_axis))
    print(f"role alignment -> {AXIS_DIR / 'role_alignment.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Project the axis read's activations onto the Assistant axis.")
    ap.add_argument("stage", choices=["pull-roles", "project", "roles", "role-components"])
    args = ap.parse_args()
    cfg = common.load_config()
    if args.stage == "pull-roles":
        pull_roles(cfg)
    elif args.stage == "roles":
        roles_only(cfg)
    elif args.stage == "role-components":
        role_components(cfg)
    else:
        project(cfg)


if __name__ == "__main__":
    main()
