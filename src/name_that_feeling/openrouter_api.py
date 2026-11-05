"""Claude through OpenRouter, with the same call shape as ``claude_api``.

``stream_text`` returns ``{"text", "usage", "stop_reason", "model"}`` exactly as
``claude_api.stream_text`` does, so a reader can switch backends by swapping the module
(``read_checkpoints.py --backend openrouter``). OpenRouter speaks the OpenAI chat API, so
the Anthropic-shaped messages (``content`` as a list of text parts, ``system`` apart) are
rendered into OpenAI messages here; the effort goes through OpenRouter's ``reasoning``
field; the call streams so a long report does not hit a request timeout; usage comes from
the final chunk (``stream_options.include_usage``). A reply cut at ``max_tokens`` or
stopped by a content filter raises, as on the direct API, so no partial reading is stored.
No token-counting endpoint exists here; ``claude_api.count_tokens`` counts the same text
for free when the Anthropic key is available.
"""

from pathlib import Path

from name_that_feeling.hf_router import OPENROUTER_BASE_URL, read_token

# Anthropic's model ids as OpenRouter lists them.
MODEL_SLUGS = {
    "claude-opus-5-5": "anthropic/claude-opus-5.5",
    "claude-sonnet-5-5": "anthropic/claude-sonnet-5.5",
}


def slug(model: str) -> str:
    """An Anthropic model id -> its OpenRouter slug (a slug with a ``/`` is passed through)."""
    return model if "/" in model else MODEL_SLUGS.get(model, f"anthropic/{model}")


def client(repo_root: Path, max_retries: int = 6):
    """An OpenAI-compatible client on ``OPENROUTER_API_KEY`` from the environment or ``<repo_root>/.env``."""
    from openai import OpenAI

    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=read_token(repo_root / ".env", "OPENROUTER_API_KEY"), max_retries=max_retries)


def _openai_messages(messages: list[dict], system: str | None) -> list[dict]:
    out = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            content = "".join(part["text"] for part in content if part.get("type") == "text")
        out.append({"role": m["role"], "content": content})
    return out


def stream_text(api, *, model: str, effort: str, messages: list[dict], system: str | None = None,
                max_tokens: int = 32000, timeout: float = 3600.0) -> dict:
    """One streamed chat completion; ``{"text", "usage", "stop_reason", "model"}``."""
    stream = api.chat.completions.create(
        model=slug(model),
        messages=_openai_messages(messages, system),
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"reasoning": {"effort": effort}},
        timeout=timeout,
    )
    parts: list[str] = []
    finish = None
    usage = None
    served_model = None
    for chunk in stream:
        served_model = getattr(chunk, "model", None) or served_model
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        for choice in chunk.choices or []:
            delta = getattr(choice, "delta", None)
            if delta is not None and delta.content:
                parts.append(delta.content)
            if choice.finish_reason:
                finish = choice.finish_reason
    if finish not in ("stop", None):
        raise RuntimeError(f"{model} via OpenRouter stopped with {finish}")
    if finish is None:
        raise RuntimeError(f"{model} via OpenRouter: the stream ended without a finish reason")
    u = {
        "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cost_usd": getattr(usage, "cost", None) if usage else None,
    }
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    if details is not None:
        u["cache_read_input_tokens"] = getattr(details, "cached_tokens", 0) or 0
    return {"text": "".join(parts), "usage": u, "stop_reason": finish, "model": served_model or slug(model), "backend": "openrouter"}
