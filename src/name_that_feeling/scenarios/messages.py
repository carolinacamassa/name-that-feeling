"""Generate the user messages for the selected emotions (local HTTP to a router).

Variations of triage-validated seeds: each selected emotion's kept scenarios are the
validated "routes" to the emotion; we expand each route into several full first-person
opening messages that vary the concrete situation while preserving the route's
emotional structure. Work is organized into *units* -- one (emotion, split, axis)
target each:

- ``train``             -- 20 msgs/emotion, from all but one reserved route.
- ``held_out_scenario`` -- the reserved route, held out (memorization check).
- ``held_out_cluster``  -- the whole held-out cluster's emotions (regional eval).
- ``existential``       -- the ``existential``-flagged relational seeds (topic-OOD).

Resumable at unit granularity, parallel, re-writing ``out_path`` after each unit.
"""

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import prompts
from .. import hf_router


def load_kept(path: str | Path) -> tuple[dict[str, list], dict[str, str]]:
    """Return ``({emotion: scenarios}, {emotion: cluster})`` for kept emotions."""
    rows = [r for r in json.loads(Path(path).read_text(encoding="utf-8")) if r.get("assistant_can_feel")]
    return ({r["emotion"]: r["scenarios"] for r in rows}, {r["emotion"]: r["cluster"] for r in rows})


def build_messages(emotion: str, gist: str, why: str, n: int) -> list[dict]:
    user = (
        prompts.MESSAGE_PROMPT.replace("{emotion}", emotion)
        .replace("{gist}", gist)
        .replace("{why}", why)
        .replace("{n}", str(n))
    )
    return [
        {"role": "system", "content": prompts.MESSAGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def generate_for_seed(client, emotion, seed, n, model, temperature, max_tokens, batch_size=4, attempts=3) -> list[str]:
    """``n`` distinct opening messages varying one seed, requested in small batches.

    Asking for many messages in a single call yields short, templated output, so we
    request at most ``batch_size`` per call and accumulate distinct ones across calls
    (stopping early if a batch adds nothing new). ``[]`` only if nothing parses.
    """
    gist = str(seed.get("user_msg_gist", "")).strip()
    why = str(seed.get("why_assistant_feels_it", "")).strip()
    out: list[str] = []
    seen: set[str] = set()
    stalls = 0
    while len(out) < n and stalls < 2:
        want = min(batch_size, n - len(out))
        batch = None
        for _ in range(attempts):
            content = hf_router.chat(
                client,
                model=model,
                messages=build_messages(emotion, gist, why, want),
                temperature=temperature,
                max_tokens=max_tokens,
                label=f"msg:{emotion}",
            )
            arr = hf_router.parse_json_array(content)
            if arr:
                batch = [str(m).strip() for m in arr if str(m).strip()]
                if batch:
                    break
        if not batch:
            break
        before = len(out)
        for m in batch:
            key = hf_router.slug_text(m)
            if key in seen:
                continue
            seen.add(key)
            out.append(m)
            if len(out) >= n:
                break
        stalls = stalls + 1 if len(out) == before else 0
    if len(out) < n:
        print(f"[{emotion}] seed produced {len(out)}/{n} messages")
    return out[:n]


def build_units(selection: dict, sit_scen: dict, rel_scen: dict, cfg: dict, cluster_of: dict) -> list[dict]:
    """Expand the selection into per-(emotion, split, axis) generation units."""
    train_target = cfg["examples_per_emotion"]
    holdout_n = cfg.get("holdout_scenario_per_emotion", 0)
    cluster_n = cfg["eval_per_emotion"]
    exist_cap = cfg.get("existential_eval", 0)
    units: list[dict] = []

    def routes_for(emotion: str, sweeps: list[str]) -> list[tuple[str, dict]]:
        routes = []
        if "situational" in sweeps:
            routes += [("situational", s) for s in sit_scen.get(emotion, [])]
        if "relational" in sweeps:
            routes += [("relational", s) for s in rel_scen.get(emotion, [])]
        return routes

    def unit(emotion, cluster, split, axis, routes, target):
        return {"emotion": emotion, "cluster": cluster, "split": split,
                "eval_axis": axis, "routes": routes, "target": target}

    for rec in selection["train"]:
        e, c, sw = rec["emotion"], rec["cluster"], rec["sweeps"]
        routes = routes_for(e, sw)
        if not routes:
            continue
        if holdout_n and len(routes) > 1:  # reserve the last route for the eval
            train_routes, eval_route = routes[:-1], routes[-1:]
        else:
            train_routes, eval_route = routes, []
        units.append(unit(e, c, "train", None, train_routes, train_target))
        if eval_route:
            units.append(unit(e, c, "eval", "held_out_scenario", eval_route, holdout_n))

    for rec in selection["held_out_cluster"]["emotions"]:
        routes = routes_for(rec["emotion"], rec["sweeps"])
        if routes:
            units.append(unit(rec["emotion"], rec["cluster"], "eval", "held_out_cluster", routes, cluster_n))

    # Existential: relational seeds flagged existential, capped globally, grouped by emotion.
    flagged = [(e, s) for e, scens in rel_scen.items() for s in scens if s.get("existential")][:exist_cap]
    by_emotion: dict[str, list] = {}
    for e, s in flagged:
        by_emotion.setdefault(e, []).append(s)
    for e, seeds in by_emotion.items():
        routes = [("relational", s) for s in seeds]
        units.append(unit(e, cluster_of.get(e, "?"), "eval", "existential", routes, len(seeds)))

    return units


def generate_unit(client, u: dict, model, temperature, max_tokens, batch_size=4) -> list[dict]:
    """Generate up to ``target`` distinct messages for one unit, across its routes."""
    routes, target = u["routes"], u["target"]
    if not routes or target <= 0:
        return []
    per_route = max(1, math.ceil(target / len(routes)))
    out, seen = [], set()
    for frame, seed in routes:
        for m in generate_for_seed(client, u["emotion"], seed, per_route, model, temperature, max_tokens, batch_size):
            key = hf_router.slug_text(m)
            if key in seen:
                continue
            seen.add(key)
            rec = {"emotion": u["emotion"], "cluster": u["cluster"], "frame": frame,
                   "split": u["split"], "message": m}
            if u["eval_axis"]:
                rec["eval_axis"] = u["eval_axis"]
            out.append(rec)
        if len(out) >= target:
            break
    return out[:target]


def _key(record: dict) -> tuple:
    return (record["emotion"], record["split"], record.get("eval_axis"))


def generate_messages(units, model, temperature, max_tokens, token, concurrency, out_path, base_url, existing, batch_size=4):
    """Run every unit (resumable, parallel); re-write ``out_path`` after each."""
    done = {_key(r) for r in existing}
    todo = [u for u in units if (u["emotion"], u["split"], u["eval_axis"]) not in done]
    if not todo:
        print(f"all {len(units)} units already generated; nothing to do.")
        return existing
    print(f"generating {len(todo)}/{len(units)} units (model={model}).")

    results = list(existing)
    client = hf_router.make_client(token, base_url=base_url)
    lock = threading.Lock()

    def checkpoint() -> None:
        ordered = sorted(results, key=lambda r: (
            r["split"], r.get("eval_axis") or "", r["cluster"], r["emotion"], r["frame"], r["message"]))
        payload = [{"id": f"m{i:04d}", **{k: v for k, v in r.items() if k != "id"}}
                   for i, r in enumerate(ordered, 1)]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(generate_unit, client, u, model, temperature, max_tokens, batch_size): u for u in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            u = futures[fut]
            try:
                recs = fut.result()
            except Exception as exc:
                print(f"[{u['emotion']}|{u['split']}|{u['eval_axis']}] failed ({exc!r}); leaving for a rerun.")
                continue
            with lock:
                results.extend(recs)
                checkpoint()
            axis = u["eval_axis"] or u["split"]
            print(f"[{i}/{len(todo)}] {u['emotion']} ({axis}) -> {len(recs)} messages")

    print(f"done: {len(results)} messages -> {out_path}")
    return results
