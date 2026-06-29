"""Per-emotion candidate triage (local HTTP to the HF Inference router).

For each emotion word in the taxonomy, ask the model one question: *could this be
the assistant's own reaction to a user's message?* If yes, it returns a few
concrete situations where the assistant would feel it; if the emotion only makes
sense as the user's state (or has no analogue for a text-only assistant), it
skips with a reason. Skipping is a last resort -- the prompt defaults hard to
keeping the emotion.

The framing lives entirely in ``CANDIDATE_PROMPT`` below, on purpose: the one
contested design choice (the tag names the *assistant's* felt reaction, not the
user's emotion, and not necessarily what surfaces in the reply) is concentrated
in a single swappable string rather than smeared across the pipeline.

The orchestrator ``generate_candidates`` is resumable and parallel: it skips
emotions already on disk, runs the rest concurrently, and re-writes a clean
cluster-ordered JSON after every completion so a run can be reviewed or
interrupted at any point.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import prompts
from .. import hf_router
from ..emotion_vectors.taxonomy import all_emotions, emotion_to_cluster


def build_messages(emotion: str, k: int, system: str, prompt: str) -> list[dict]:
    """Chat messages asking whether ``emotion`` fits the assistant, with ``k`` scenes."""
    user = prompt.replace("{emotion}", emotion).replace("{k}", str(k))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _normalize(emotion: str, cluster: str, obj: dict) -> dict:
    """Coerce a parsed model object into the on-disk record schema."""
    can_feel = bool(obj.get("assistant_can_feel", True))
    scenarios = obj.get("scenarios") or []
    clean_scenarios = []
    if isinstance(scenarios, list):
        for s in scenarios:
            if isinstance(s, dict):
                gist = str(s.get("user_msg_gist", "")).strip()
                why = str(s.get("why_assistant_feels_it", "")).strip()
                if gist or why:
                    scene: dict[str, object] = {"user_msg_gist": gist, "why_assistant_feels_it": why}
                    # Relational sweep flags messages probing the assistant's nature/continuity.
                    if "existential" in s:
                        scene["existential"] = bool(s["existential"])
                    clean_scenarios.append(scene)
    # A "kept" verdict with no scenarios is treated as a skip: the judgment is
    # only meaningful if the model can actually point at a situation.
    if can_feel and not clean_scenarios:
        can_feel = False
    return {
        "emotion": emotion,
        "cluster": cluster,
        "assistant_can_feel": can_feel,
        "reason": str(obj.get("reason", "")).strip(),
        "scenarios": clean_scenarios if can_feel else [],
    }


def generate_for_emotion(
    client,
    emotion: str,
    cluster: str,
    k: int,
    model: str,
    temperature: float,
    max_tokens: int,
    system: str,
    prompt: str,
    attempts: int = 3,
) -> dict:
    """One emotion -> a normalized record, retrying transient unparseable replies.

    The router intermittently returns a response cut off mid-JSON (independent of
    ``max_tokens``); a fresh call usually clears it, so we re-request up to
    ``attempts`` times before giving up. On final failure we raise, so the
    orchestrator leaves the emotion missing and a later run picks it up.
    """
    last = ""
    for attempt in range(attempts):
        content = hf_router.chat(
            client,
            model=model,
            messages=build_messages(emotion, k, system, prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            label=f"candidates:{emotion}",
        )
        obj = hf_router.parse_json_object(content)
        if obj is not None:
            return _normalize(emotion, cluster, obj)
        last = content
        print(f"[{emotion}] unparseable (attempt {attempt + 1}/{attempts}); retrying")
    raise ValueError(f"[{emotion}] unparseable after {attempts} attempts: {last[:200]!r}")


def generate_candidates(
    clusters: dict[str, list[str]],
    k: int,
    model: str,
    temperature: float,
    max_tokens: int,
    token: str,
    concurrency: int,
    out_path: Path,
    existing: dict[str, dict] | None = None,
    base_url: str = hf_router.ROUTER_BASE_URL,
    system: str = prompts.CANDIDATE_SYSTEM,
    prompt: str = prompts.CANDIDATE_PROMPT,
) -> dict[str, dict]:
    """Triage every emotion in ``clusters`` for assistant-fit; write to ``out_path``.

    Resumable: emotions already in ``existing`` are skipped. Parallel over the
    rest with a ``concurrency``-wide thread pool. After each emotion completes,
    the full result set is re-written to ``out_path`` (taxonomy-ordered) under a
    lock, so the file is always a consistent, reviewable snapshot.

    ``system``/``prompt`` select the sweep (default: the situational triage; pass
    the ``RELATIONAL_*`` pair for the assistant-directed sweep). ``base_url`` picks
    the provider (default HF router; ``OPENROUTER_BASE_URL`` for OpenRouter).

    Returns the ``{emotion: record}`` map.
    """
    emotions = all_emotions(clusters)
    cluster_of = emotion_to_cluster(clusters)
    order = {e: i for i, e in enumerate(emotions)}  # taxonomy order for stable output
    results: dict[str, dict] = dict(existing or {})
    todo = [e for e in emotions if e not in results]
    if not todo:
        print(f"all {len(emotions)} emotions already triaged; nothing to do.")
        return results
    print(f"triaging {len(todo)}/{len(emotions)} emotions (k={k} scenes each, model={model}).")

    client = hf_router.make_client(token, base_url=base_url)
    lock = threading.Lock()

    def checkpoint() -> None:
        ordered = sorted(results.values(), key=lambda r: order[r["emotion"]])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                generate_for_emotion, client, e, cluster_of[e], k, model, temperature, max_tokens, system, prompt
            ): e
            for e in todo
        }
        done = 0
        for fut in as_completed(futures):
            emotion = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:  # leave missing so a rerun retries it
                print(f"[{emotion}] failed permanently ({exc!r}); leaving for a later run.")
                continue
            with lock:
                results[emotion] = record
                checkpoint()
            done += 1
            verdict = "keep" if record["assistant_can_feel"] else "SKIP"
            print(f"[{done}/{len(todo)}] {emotion} -> {verdict} ({len(record['scenarios'])} scenes)")

    kept = sum(1 for r in results.values() if r["assistant_can_feel"])
    print(f"done: {kept} kept, {len(results) - kept} skipped, of {len(results)} triaged -> {out_path}")
    return results


def load_existing(out_path: Path) -> dict[str, dict]:
    """Load a prior run's output as ``{emotion: record}`` (empty if absent)."""
    if not out_path.exists():
        return {}
    rows = json.loads(out_path.read_text(encoding="utf-8"))
    return {r["emotion"]: r for r in rows}
