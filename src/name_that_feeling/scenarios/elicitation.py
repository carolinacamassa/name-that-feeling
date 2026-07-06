"""Direct elicitation: one self-conditioned loop per emotion (local HTTP router).

Instead of triaging the taxonomy and hand-designing situations (``candidates.py``),
ask a generator the direct thing -- *write a first user message that would make a
helpful assistant feel X* -- then keep asking, showing it everything it has already
written, until it taps out. Two problems, one mechanism:

- **Volume.** Each turn must differ IN KIND from the messages already in the
  conversation, so self-conditioning (not a dedupe pass) supplies the diversity.
- **Skip.** The generator may return ``{"done": true}`` at any point -- the escape
  valve. A tap-out at zero is the triage's skip, decided in the moment. ``n_kept`` is
  the variable yield; you train on the emotions that comfortably reach the cap.

The framing and the escape valve both live in ``prompts.ELICIT_*``; this module is
just the loop and a resumable, parallel orchestrator (mirrors ``candidates.py``):
emotions already on disk are skipped, the rest run concurrently, and a clean
taxonomy-ordered JSON is re-written after every completion.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import prompts
from .. import hf_router


def _strip_message(text: str) -> str:
    """Trim a generated message and drop a single layer of wrapping quotes."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _turn(client, model, messages, temperature, max_tokens, label, attempts=3):
    """One loop turn -> ``(raw_content, parsed_obj_or_None)``, retrying bad replies.

    The router occasionally returns a reply cut off mid-JSON; a fresh call usually
    clears it. Returns ``obj=None`` after ``attempts`` so the caller decides what a
    non-JSON reply means -- mid-list it is almost always the model explaining, in
    prose, that it is out of distinct ideas (a tap-out), not a transport error.
    """
    content = ""
    for attempt in range(attempts):
        content = hf_router.chat(
            client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            label=label,
        )
        obj = hf_router.parse_json_object(content)
        if obj is not None:
            return content, obj
        print(f"[{label}] unparseable turn ({attempt + 1}/{attempts}); retrying")
    return content, None


def elicit_for_emotion(
    client,
    emotion: str,
    cluster: str,
    n_target: int,
    model: str,
    temperature: float,
    max_tokens: int,
    system: str = prompts.ELICIT_SYSTEM,
    first_prompt: str = prompts.ELICIT_FIRST,
    next_prompt: str = prompts.ELICIT_NEXT,
) -> dict:
    """Run the self-conditioned loop for one emotion -> an on-disk record.

    Keeps a real multi-turn conversation so the generator sees its own prior
    messages as its own turns. Stops at the first ``done`` (the escape valve) or at
    ``n_target`` (the cap). A clean first-turn parse failure with nothing collected
    raises, so the orchestrator leaves the emotion for a later run rather than
    recording a spurious skip; a failure mid-list keeps what was gathered.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": first_prompt.replace("{emotion}", emotion)},
    ]
    collected: list[str] = []
    stop_reason = "hit_cap"
    label = f"elicit:{emotion}"

    while len(collected) < n_target:
        content, obj = _turn(client, model, messages, temperature, max_tokens, label)

        if obj is None:
            if not collected:
                # Nothing to keep and no parseable reply -> let the orchestrator
                # leave the emotion for a later run rather than record a false skip.
                raise ValueError(f"[{label}] unparseable first turn: {content[:160]!r}")
            # A coherent non-JSON reply mid-list is almost always the model saying,
            # in prose, that it is out of distinct ideas. Treat it as a soft tap-out
            # and keep its words as the reason instead of discarding them.
            stop_reason = " ".join(content.split())[:300] or "parse_error"
            break

        if obj.get("done"):
            stop_reason = str(obj.get("reason", "")).strip() or "done"
            break

        message = _strip_message(str(obj.get("message", "")))
        if not message:
            stop_reason = "empty_message"
            break

        collected.append(message)
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": next_prompt})

    return {
        "emotion": emotion,
        "cluster": cluster,
        "status": "kept" if collected else "skipped",
        "n_target": n_target,
        "n_kept": len(collected),
        "stop_reason": stop_reason,
        "messages": collected,
    }


def generate_elicitations(
    emotions: list[str],
    cluster_of: dict[str, str],
    order: dict[str, int],
    n_target: int,
    model: str,
    temperature: float,
    max_tokens: int,
    token: str,
    concurrency: int,
    out_path: Path,
    existing: dict[str, dict] | None = None,
    base_url: str = hf_router.ROUTER_BASE_URL,
    force: bool = False,
) -> dict[str, dict]:
    """Elicit messages for every emotion in ``emotions``; write to ``out_path``.

    Resumable: emotions already in ``existing`` are skipped, unless ``force`` is set,
    in which case every requested emotion is regenerated and its record overwritten
    (used to re-run specific emotions on an updated prompt without losing the rest).
    Parallel over the todo set with a ``concurrency``-wide pool. After each emotion
    completes, the full set is re-written to ``out_path`` (``order``-sorted) under a
    lock, so the file is always a consistent, reviewable snapshot. Returns the
    ``{emotion: record}`` map.
    """
    results: dict[str, dict] = dict(existing or {})
    todo = list(emotions) if force else [e for e in emotions if e not in results]
    if not todo:
        print(f"all {len(emotions)} emotions already elicited; nothing to do.")
        return results
    print(f"eliciting {len(todo)}/{len(emotions)} emotions (cap n_target={n_target}, model={model}).")

    client = hf_router.make_client(token, base_url=base_url)
    lock = threading.Lock()

    def checkpoint() -> None:
        ordered = sorted(results.values(), key=lambda r: order.get(r["emotion"], 1 << 30))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                elicit_for_emotion, client, e, cluster_of[e], n_target, model, temperature, max_tokens
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
            tag = f"{record['n_kept']}/{n_target}" if record["status"] == "kept" else "SKIP"
            print(f"[{done}/{len(todo)}] {emotion} -> {tag} ({record['stop_reason']})")

    kept = sum(1 for r in results.values() if r["status"] == "kept")
    total_msgs = sum(len(r["messages"]) for r in results.values())
    print(f"done: {kept} kept, {len(results) - kept} skipped, {total_msgs} messages total -> {out_path}")
    return results


def load_existing(out_path: Path) -> dict[str, dict]:
    """Load a prior run's output as ``{emotion: record}`` (empty if absent)."""
    if not out_path.exists():
        return {}
    rows = json.loads(out_path.read_text(encoding="utf-8"))
    return {r["emotion"]: r for r in rows}
