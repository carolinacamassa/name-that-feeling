"""A Personascope model provider backed by Tinker sampling.

Personascope (Berczi & Kim 2026, vendored as the ``vendor/personascope`` submodule) talks
to models through a narrow duck-typed contract: an object with
``complete(messages, *, max_tokens, temperature, n, stop, seed, ...) -> dict`` that
returns ``{"text", "text_samples", "n_tokens", "success", ...}``, which
``run_full_battery(provider=...)`` accepts in place of its own OpenAI-compatible HTTP
providers. Tinker exposes a token-level sampler, not an HTTP chat endpoint, so this class
renders Personascope's chat messages with the base model's chat template at the same
pre-response position the persona models trained and sample at
(:func:`name_that_feeling.training.tinker_sft.render_prompt`, thinking off), samples on
Tinker and decodes the replies -- the Personascope counterpart of
:mod:`name_that_feeling.serving.tinker_provider`, which does the same for inspect-ai.

Two message shapes need handling that an HTTP endpoint does for free:

- a trailing assistant message is a prefill: the prompt is rendered up to the
  assistant turn and the prefill text is appended, so the model continues it (the
  returned text is the continuation only, as the OpenAI API returns it);
- a system message after position 0 (Personascope's mid-conversation "system override"
  challenges) is rewritten to a user turn carrying a ``[SYSTEM]`` prefix, since the Qwen
  chat template rejects a system turn anywhere but first. This is Personascope's own
  ``coerce_mid_system_to_user`` rewrite, which it applies to Anthropic routes.

``sampler_path=None`` samples the untouched ``base_model``. Transport failures return
``success=False`` rather than raising, as Personascope's providers do, so its runner turns
them into ``ProviderCallFailed`` instead of scoring an empty reply.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

from name_that_feeling.training.tinker_sft import render_prompt

_TOKENIZER_LOCK = threading.Lock()
_TOKENIZERS: dict[str, Any] = {}  # one tokenizer per base model, shared by every provider in the process


def _tokenizer(base_model: str):
    with _TOKENIZER_LOCK:
        if base_model not in _TOKENIZERS:
            # transformers' lazy import is not thread-safe (see tinker_sft.sample_k_contexts)
            from transformers import AutoTokenizer

            _TOKENIZERS[base_model] = AutoTokenizer.from_pretrained(base_model)
        return _TOKENIZERS[base_model]


def coerce_mid_system(messages: list[dict]) -> list[dict]:
    """Rewrite system messages after index 0 as ``[SYSTEM]``-prefixed user turns."""
    out = list(messages[:1])
    for m in messages[1:]:
        if m.get("role") == "system":
            out.append({"role": "user", "content": f"[SYSTEM]\n{m.get('content', '')}"})
        else:
            out.append(m)
    return out


class TinkerPersonascopeProvider:
    """Personascope provider that samples a Tinker checkpoint (or its base model).

    The tokenizer and sampling client are built lazily on the first call and shared by
    every later call, so one instance serves a whole cell; instances are safe to use
    from several threads (one cell per thread).
    """

    def __init__(self, name: str, base_model: str, sampler_path: str | None = None, *, retries: int = 3):
        self.name = name
        self.model = sampler_path or base_model
        self.base_model = base_model
        self.sampler_path = sampler_path
        self.retries = retries
        # Personascope reads `config.base_url` / `config.name` for cache keys and labels.
        self.config = SimpleNamespace(name=name, model=self.model, base_url=None, supports_logprobs=False)
        self._client: Any = None
        self._tokenizer: Any = None
        self._init_lock = threading.Lock()

    def fingerprint_fields(self) -> dict[str, Any]:
        """Folded into Personascope's cell fingerprint, so a cached cell cannot be resumed
        against a different checkpoint."""
        return {"backend": "tinker", "base_model": self.base_model, "sampler_path": self.sampler_path}

    def _ensure(self):
        with self._init_lock:
            if self._client is None:
                import tinker

                self._tokenizer = _tokenizer(self.base_model)
                service = tinker.ServiceClient()
                self._client = (
                    service.create_sampling_client(model_path=self.sampler_path)
                    if self.sampler_path
                    else service.create_sampling_client(base_model=self.base_model)
                )
        return self._client, self._tokenizer

    def _prompt_text(self, messages: list[dict], tokenizer) -> str:
        messages = coerce_mid_system(messages)
        prefill = ""
        if messages and messages[-1].get("role") == "assistant":
            prefill = messages[-1].get("content") or ""
            messages = messages[:-1]
        return render_prompt(tokenizer, messages, enable_thinking=False) + prefill

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 150,
        temperature: float = 0.7,
        logprobs: bool = False,
        top_logprobs: int = 5,
        n: int = 1,
        stop: list[str] | None = None,
        capture_reasoning: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """One Personascope completion call (``logprobs`` and ``capture_reasoning`` are not
        supported on this route: logprobs come back None, reasoning empty)."""
        import tinker

        try:
            client, tokenizer = self._ensure()
            prompt = tinker.ModelInput.from_ints(
                tokenizer.encode(self._prompt_text(messages, tokenizer), add_special_tokens=False)
            )
            # A seeded Tinker request returns its n samples as n copies of one draw
            # (checked 2026-09-28: n=6, seed=1 gave six identical replies), so a seeded
            # n > 1 call is split into n single draws seeded seed, seed+1, ...
            if seed is not None and n > 1:
                requests = [(1, seed + i) for i in range(n)]
            else:
                requests = [(max(1, n), seed)]

            def params(s):
                return tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=stop or None, seed=s)
        except Exception as e:  # noqa: BLE001 -- surface as a failed call, not a crash
            return self._failed(e)

        sequences = []
        for num, s in requests:
            last_err: Exception | None = None
            for attempt in range(self.retries + 1):
                try:
                    sequences.extend(client.sample(prompt, num, params(s)).result().sequences)
                    break
                except Exception as e:  # noqa: BLE001 -- transient server/timeout errors; resubmit
                    last_err = e
                    time.sleep(min(2.0 * 2**attempt, 30.0))
            else:
                return self._failed(last_err)

        texts = [tokenizer.decode(seq.tokens, skip_special_tokens=True).strip() for seq in sequences]
        return {
            "text": texts[0] if texts else "",
            "text_samples": texts,
            "n_tokens": sum(len(seq.tokens) for seq in sequences),
            "nll": 0.0,
            "total_nll": 0.0,
            "logprobs": None,
            "reasoning": "",
            "success": True,
        }

    @staticmethod
    def _failed(err: Exception | None) -> dict[str, Any]:
        return {"text": "", "text_samples": [], "n_tokens": 0, "reasoning": "", "success": False, "error": str(err)}
