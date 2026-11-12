"""The Anthropic API from an experiment script: a client from the repo's ``.env``, streamed
text calls that return the text with the usage, and free token counts for sizing a call
before paying for it.

Every call streams (long inputs and outputs otherwise hit request timeouts) and returns
once the message is complete. Thinking is the model's default (adaptive on the Claude 5
family); depth is set through ``effort``. A message's content blocks may carry
``cache_control`` so a second turn over the same material is billed at the cache rate.
"""

from pathlib import Path

import anthropic

from name_that_feeling.hf_router import read_token


def client(repo_root: Path, max_retries: int = 8) -> anthropic.Anthropic:
    """A client on ``ANTHROPIC_API_KEY`` from the environment or ``<repo_root>/.env``."""
    return anthropic.Anthropic(api_key=read_token(repo_root / ".env", "ANTHROPIC_API_KEY"), max_retries=max_retries)


def _usage(msg) -> dict:
    u = msg.usage
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


def stream_text(api: anthropic.Anthropic, *, model: str, effort: str, messages: list[dict],
                system: str | None = None, max_tokens: int = 32000) -> dict:
    """One streamed call; ``{"text", "usage", "stop_reason", "model"}``. Raises if the reply
    was cut (``max_tokens``) or refused, so a caller never stores a partial reading."""
    kwargs: dict = {"model": model, "max_tokens": max_tokens, "output_config": {"effort": effort}, "messages": messages}
    if system:
        kwargs["system"] = system
    with api.messages.stream(**kwargs) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason not in ("end_turn", "stop_sequence"):
        raise RuntimeError(f"{model} stopped with {msg.stop_reason}")
    text = "".join(b.text for b in msg.content if b.type == "text")
    return {"text": text, "usage": _usage(msg), "stop_reason": msg.stop_reason, "model": msg.model}


def count_tokens(api: anthropic.Anthropic, *, model: str, messages: list[dict], system: str | None = None) -> int:
    """The input token count of a request, from the free counting endpoint."""
    kwargs: dict = {"model": model, "messages": messages}
    if system:
        kwargs["system"] = system
    return api.messages.count_tokens(**kwargs).input_tokens
