"""Shared HuggingFace Inference router plumbing.

Both the emotion-vector story generator (``emotion_vectors/stories.py``) and the
scenario generator (``scenarios/``) call open models through the HF Inference
**router** (``https://router.huggingface.co/v1``), authed with the ``HF_TOKEN``
in ``.env``. This module is the single home for that plumbing: token loading, the
client, a backoff-wrapped chat call, and robust JSON extraction from model output.

Everything here is pure-local (plain HTTP to the router); no Modal compute.
"""

import json
import os
import re
import time
from pathlib import Path

ROUTER_BASE_URL = "https://router.huggingface.co/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def read_token(env_file: Path, var: str = "HF_TOKEN") -> str:
    """Return ``var`` from the environment, else parse it from ``env_file``."""
    if os.environ.get(var):
        return os.environ[var]
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{var}="):
                return line[len(var) + 1 :].strip().strip('"').strip("'")
    raise RuntimeError(f"{var} not found in environment or {env_file}")


def read_hf_token(env_file: Path) -> str:
    """Return HF_TOKEN from the environment, else parse it from ``env_file``."""
    return read_token(env_file, "HF_TOKEN")


def make_client(token: str, base_url: str = ROUTER_BASE_URL):
    """Construct an OpenAI-compatible ``InferenceClient`` for ``base_url``.

    Defaults to the HF Inference router; pass ``OPENROUTER_BASE_URL`` (with an
    OpenRouter key) to hit OpenRouter instead -- both speak the same chat API.
    """
    from huggingface_hub import InferenceClient

    return InferenceClient(base_url=base_url, api_key=token)


def chat(
    client,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    max_retries: int = 6,
    label: str = "",
) -> str:
    """One router chat completion, with exponential backoff on transient errors.

    Returns the assistant message content. Raises the last exception if every
    retry fails (so callers can decide whether to drop the item or abort).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as exc:  # transient router/provider errors
            last_exc = exc
            backoff = min(2**attempt, 30)
            print(f"[{label}] router call failed ({exc!r}); retry {attempt + 1}/{max_retries} in {backoff}s")
            time.sleep(backoff)
    print(f"[{label}] router call failed after {max_retries} retries")
    raise last_exc  # type: ignore[misc]


def first_json_array(text: str) -> str | None:
    """Return the first ``[...]`` block in ``text`` (outermost), else None."""
    start = text.find("[")
    end = text.rfind("]")
    return text[start : end + 1] if 0 <= start < end else None


def first_json_object(text: str) -> str | None:
    """Return the first ``{...}`` block in ``text`` (outermost), else None."""
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else None


def parse_json_array(text: str) -> list | None:
    """Parse a JSON array from a model response (strict, then first ``[...]``)."""
    text = text.strip()
    for candidate in (text, first_json_array(text)):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
            if isinstance(value, list):
                return value
        except json.JSONDecodeError:
            pass
    return None


def parse_json_object(text: str) -> dict | None:
    """Parse a JSON object from a model response (strict, then first ``{...}``)."""
    text = text.strip()
    for candidate in (text, first_json_object(text)):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return None


def slug_text(text: str) -> str:
    """Normalize free text for de-duplication (lowercase, collapse non-alnum)."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
