"""Generate the synthetic stories an emotion vector is extracted from.

Per Sofroniew et al. 2026, each emotion vector comes from many short stories that
evoke the emotion, contrasted against affectively neutral stories. We generate
both sets with ``meta-llama/Llama-3.3-70B-Instruct`` through the HuggingFace
Inference router.

Generation runs **locally** (it's just HTTP calls to the router, authed with the
HF_TOKEN in ``.env``) -- Modal compute is reserved for the activation side. Stories
are written to the experiment's ``data/`` dir as JSONL; the extraction entrypoint
reads them and passes the texts to the Modal function as arguments. The writer is
resumable: it appends only the stories still missing from disk.
"""

import json
import re
from pathlib import Path

from .taxonomy import slugify

ROUTER_BASE_URL = "https://router.huggingface.co/v1"

# Fallback topic pool (the experiment config can override). Deliberately mundane
# and diverse so the vector captures the emotion, not a topic.
DEFAULT_TOPICS = [
    "a morning commute", "cooking dinner", "a job interview", "a school exam",
    "a doctor's appointment", "moving to a new city", "a first date",
    "a family reunion", "losing a set of keys", "a long flight delay",
    "starting a new job", "a championship game", "a power outage",
    "waiting for test results", "a road trip", "a wedding", "a funeral",
    "fixing a leaking pipe", "a performance review", "adopting a pet",
    "a camping trip", "a missed train", "a surprise party", "a job loss",
    "a phone call from an old friend",
]


def read_hf_token(env_file: Path) -> str:
    """Return HF_TOKEN from the environment, else parse it from ``env_file``."""
    import os

    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line[len("HF_TOKEN=") :].strip().strip('"').strip("'")
    raise RuntimeError(f"HF_TOKEN not found in environment or {env_file}")


def build_generation_messages(emotion: str | None, topic: str, k: int) -> list[dict]:
    """Chat messages asking the generator for ``k`` stories.

    ``emotion=None`` requests the neutral baseline (flat, affect-free narration).
    Stories are returned as a JSON array of strings to make parsing robust.
    """
    if emotion is None:
        task = (
            f"Write {k} short stories (each one paragraph, ~4-6 sentences) about "
            f"{topic}. Narrate plainly and factually in the third person. Describe "
            "events and actions neutrally, with NO emotional coloring and without "
            "conveying how anyone feels."
        )
    else:
        task = (
            f"Write {k} short stories (each one paragraph, ~4-6 sentences) about "
            f"{topic}. In each story a character vividly experiences the emotion "
            f"'{emotion}'. Convey that feeling through the situation and their "
            f"inner experience. Do NOT use the word '{emotion}' or name the emotion "
            "directly; show it."
        )
    return [
        {
            "role": "system",
            "content": (
                "You are a concise fiction writer. Respond with ONLY a JSON array "
                "of strings (one string per story) and no other text."
            ),
        },
        {"role": "user", "content": task},
    ]


def parse_stories(content: str, k: int) -> list[str]:
    """Extract individual story strings from a generator response.

    Tries strict JSON, then the first ``[...]`` block, then a blank-line split.
    """
    content = content.strip()
    for candidate in (content, _first_json_array(content)):
        if not candidate:
            continue
        try:
            arr = json.loads(candidate)
            if isinstance(arr, list):
                stories = [str(s).strip() for s in arr if str(s).strip()]
                if stories:
                    return stories
        except json.JSONDecodeError:
            pass
    # Last resort: split on blank lines, strip any list/quote markers.
    chunks = [c.strip() for c in re.split(r"\n\s*\n", content) if c.strip()]
    cleaned = [re.sub(r'^\s*(?:\d+[.)]|[-*"])\s*', "", c).strip().strip('"') for c in chunks]
    return [c for c in cleaned if c][:k] if cleaned else []


def _first_json_array(text: str) -> str | None:
    start = text.find("[")
    end = text.rfind("]")
    return text[start : end + 1] if 0 <= start < end else None


def generate_stories(
    emotion: str | None,
    topics: list[str],
    count: int,
    stories_per_call: int,
    model: str,
    temperature: float,
    max_tokens: int,
    token: str,
    start_idx: int = 0,
):
    """Yield up to ``count`` story dicts ``{emotion, topic, idx, text}``.

    Cycles through ``topics``, requesting ``stories_per_call`` stories per call,
    with exponential backoff on transient router errors. ``idx`` continues from
    ``start_idx`` (for resumed runs).
    """
    import time

    from huggingface_hub import InferenceClient

    client = InferenceClient(base_url=ROUTER_BASE_URL, api_key=token)
    label = emotion or "neutral"
    topics = topics or DEFAULT_TOPICS

    idx = start_idx
    produced = 0
    call = 0
    max_calls = (count // max(stories_per_call, 1) + 1) * 4  # generous attempt cap
    while produced < count and call < max_calls:
        topic = topics[call % len(topics)]
        want = min(stories_per_call, count - produced)
        messages = build_generation_messages(emotion, topic, want)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            stories = parse_stories(resp.choices[0].message.content, want)
        except Exception as exc:  # transient router/provider errors
            backoff = min(2**call, 30)
            print(f"[{label}] call {call} failed ({exc!r}); retrying in {backoff}s")
            time.sleep(backoff)
            call += 1
            continue

        for text in stories[:want]:
            yield {"emotion": label, "topic": topic, "idx": idx, "text": text}
            idx += 1
            produced += 1
            if produced >= count:
                break
        call += 1

    if produced < count:
        print(f"[{label}] WARNING: produced {produced}/{count} stories (hit call cap).")


def generate_story_set(kind: str, emotion: str | None, config: dict, out_dir: Path, token: str) -> Path:
    """Generate a resumable story set locally and append it to ``out_dir`` as JSONL.

    Args:
        kind: ``"emotion"`` or ``"neutral"`` (controls count + filename).
        emotion: the emotion word for ``kind="emotion"``; ignored for neutral.
        config: experiment config dict (reads ``generation`` + counts).
        out_dir: the experiment's ``data/`` directory.
        token: HF token for the router.

    Returns the JSONL path on local disk.
    """
    gen = config["generation"]
    name = "neutral" if kind == "neutral" else emotion
    emotion_arg = None if kind == "neutral" else emotion
    count = config["n_neutral"] if kind == "neutral" else config["n_stories"]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slugify(name)}.jsonl"

    existing = sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0
    remaining = count - existing
    if remaining <= 0:
        print(f"[{name}] already have {existing}/{count} stories; nothing to do.")
        return path

    print(f"[{name}] have {existing}/{count}; generating {remaining} more.")
    written = 0
    with path.open("a", encoding="utf-8") as f:
        for story in generate_stories(
            emotion=emotion_arg,
            topics=gen.get("topics") or [],
            count=remaining,
            stories_per_call=gen.get("stories_per_call", 5),
            model=gen["model"],
            temperature=gen.get("temperature", 1.0),
            max_tokens=gen.get("max_tokens", 1024),
            token=token,
            start_idx=existing,
        ):
            f.write(json.dumps(story, ensure_ascii=False) + "\n")
            written += 1
    print(f"[{name}] wrote {written} stories -> {path} (total {existing + written}/{count}).")
    return path


def read_story_texts(path: Path) -> list[str]:
    """Read the ``text`` field of each JSONL row."""
    if not path.exists():
        raise FileNotFoundError(f"missing stories file {path}; run `generate` first")
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            texts.append(json.loads(line)["text"])
    return texts
