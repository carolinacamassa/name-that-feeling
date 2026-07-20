"""A custom inspect-ai model provider backed by Tinker sampling.

Tinker exposes a *token-level* sampling API (``ServiceClient().create_sampling_client``
-> ``.sample(ModelInput, n, SamplingParams)``), not an OpenAI-compatible HTTP endpoint,
so inspect's built-in providers cannot reach a ``tinker://`` checkpoint directly. This
module bridges the two: a :class:`ModelAPI` whose :meth:`generate` renders inspect's chat
messages with the base model's HF chat template at the **same pre-response position used
for training and probe readout** (:func:`name_that_feeling.training.tinker_sft.render_prompt`),
samples on Tinker, and decodes the reply.

Registered as the ``tinker`` provider (see ``name_that_feeling/_registry.py`` and the
``[project.entry-points.inspect_ai]`` table in ``pyproject.toml``), so an eval can name a
checkpoint through model args::

    get_model("tinker/with-neutral", base_model="Qwen/Qwen3.5-9B",
              sampler_path="tinker://.../sampler_weights/...-final")

Pass ``sampler_path=None`` (or omit it) to sample the untouched ``base_model`` -- the
eval's base-vs-trained baseline. Auth: ``TINKER_API_KEY`` in the environment
(:func:`name_that_feeling.training.tinker_sft.load_api_key` reads it from ``.env``).
"""

from __future__ import annotations

import asyncio

from inspect_ai.model import ChatMessage, GenerateConfig, ModelAPI, ModelOutput
from inspect_ai.tool import ToolChoice, ToolInfo

from name_that_feeling.training.tinker_sft import render_prompt


class TinkerAPI(ModelAPI):
    """inspect ModelAPI that samples a Tinker checkpoint (or its base model).

    ``sampler_path`` is a ``tinker://`` sampler-weights path; when it is ``None`` (or the
    sentinel ``"base"``) the untouched ``base_model`` is sampled instead. The tokenizer and
    sampling client are created lazily on the first request and reused for the run.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        *,
        base_model: str,
        sampler_path: str | None = None,
        max_connections: int = 8,
    ) -> None:
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config)
        import tinker
        from transformers import AutoTokenizer

        self.base_model = base_model
        self.sampler_path = sampler_path if sampler_path not in (None, "", "base") else None
        self._max_connections = max_connections
        # Build the tokenizer + sampling client eagerly, here in the constructing thread.
        # generate() runs on many worker threads at once, and transformers' lazy top-level
        # import is not thread-safe -- doing it lazily inside a threaded _sample() races and
        # raises "cannot import name 'AutoTokenizer'". Constructing once up front avoids that.
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        service = tinker.ServiceClient()
        self._client = (
            service.create_sampling_client(model_path=self.sampler_path)
            if self.sampler_path
            else service.create_sampling_client(base_model=self.base_model)
        )

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        text = await asyncio.to_thread(self._sample, input, config)
        return ModelOutput.from_content(model=self.model_name, content=text)

    def _sample(self, input: list[ChatMessage], config: GenerateConfig) -> str:
        """Blocking render -> Tinker sample -> decode (run off the event loop)."""
        import tinker

        client, tokenizer = self._client, self._tokenizer
        turns = [{"role": m.role, "content": m.text} for m in input]
        prompt = render_prompt(tokenizer, turns)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        params = tinker.SamplingParams(
            max_tokens=config.max_tokens or 512,
            temperature=config.temperature if config.temperature is not None else 0.0,
            **({"seed": config.seed} if config.seed is not None else {}),
        )
        result = client.sample(tinker.ModelInput.from_ints(ids), 1, params).result()
        tokens = result.sequences[0].tokens
        return tokenizer.decode(tokens, skip_special_tokens=True).strip()

    def max_connections(self) -> int:
        return self._max_connections
