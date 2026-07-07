"""Batched on-policy chat completions on Modal.

Generates the *visible* assistant reply to each user message from exactly the
pre-response position the emotion probe reads at: ``add_generation_prompt=True`` +
``enable_thinking=False`` puts the model's empty ``<think></think>`` block in the
prompt, so generation is the visible reply (the same rendering
``emotion_vectors.readout.build_chat_texts`` uses, and the same weights
``ActivationExtractor`` loads -- so completions are on-policy for the probed model).

One knob distinguishes the data variants (methods §3.3, `probe-conditioned-distillation.md`):

- ``system_prompt=None``  -> unconditioned: a vanilla reply, emotion tag prepended later.
- ``system_prompt=<...>`` -> conditioned: the reply is conditioned on a prompt that
  surfaces the probe read; the prompt is stripped at train time (context distillation).
"""

import modal

from name_that_feeling.infra import (
    HF_CACHE_DIR,
    HOURS,
    VECTORS_DIR,
    hf_cache_volume,
    hf_secret,
    vectors_image,
    vectors_volume,
    vllm_image,
)

app = modal.App("name-that-feeling-generation")


@app.cls(
    image=vectors_image,
    # 9B bf16 (~18GB) + KV cache for a small batch fits A10G (24GB).
    gpu="A10G",
    volumes={HF_CACHE_DIR: hf_cache_volume},
    secrets=[hf_secret],
    timeout=6 * HOURS,
)
class Generator:
    model_id: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Full causal LM (we need generate / the lm_head here, unlike extraction which
        # calls the base transformer for hidden states only). Same load + fallback.
        kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        except (ValueError, KeyError):
            from transformers import Qwen3_5ForCausalLM

            self.model = Qwen3_5ForCausalLM.from_pretrained(self.model_id, **kwargs)
        self.model.eval()
        print(f"Loaded {self.model_id} for generation ({self.model.config.num_hidden_layers} layers)")

    def _render(self, messages: list[str], system_prompt: str | None) -> list[str]:
        """Render each user message to the pre-response token, optional system turn prepended."""
        texts = []
        for content in messages:
            turns = []
            if system_prompt:
                turns.append({"role": "system", "content": system_prompt})
            turns.append({"role": "user", "content": content})
            try:
                text = self.tokenizer.apply_chat_template(
                    turns, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:  # template with no thinking toggle
                text = self.tokenizer.apply_chat_template(
                    turns, tokenize=False, add_generation_prompt=True
                )
            texts.append(text)
        return texts

    @modal.method()
    def generate(self, messages: list[str], config: dict) -> list[str]:
        """Return one visible-reply string per message (order preserved).

        ``config`` keys: ``system_prompt`` (None for the unconditioned variant), ``batch_size``,
        ``max_new_tokens``, ``temperature``, ``top_p``, ``seed``, ``do_sample``.
        """
        torch = self.torch
        system_prompt = config.get("system_prompt")
        batch_size = config.get("batch_size", 8)
        max_new_tokens = config.get("max_new_tokens", 512)
        seed = config.get("seed", 42)
        do_sample = config.get("do_sample", True)

        self.tokenizer.padding_side = "left"  # batched decode: reply tokens align at the end
        self.tokenizer.truncation_side = "left"  # keep the assistant header if a prompt is long

        completions: list[str] = []
        for i in range(0, len(messages), batch_size):
            batch = messages[i : i + batch_size]
            enc = self.tokenizer(
                self._render(batch, system_prompt),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to("cuda")

            gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id)
            if do_sample:
                gen_kwargs.update(
                    do_sample=True, temperature=config.get("temperature", 0.7), top_p=config.get("top_p", 0.95)
                )
            else:
                gen_kwargs.update(do_sample=False)

            if seed is not None:
                torch.manual_seed(seed + i)
            with torch.inference_mode():
                out = self.model.generate(**enc, **gen_kwargs)

            new_tokens = out[:, enc["input_ids"].shape[1] :]
            completions.extend(s.strip() for s in self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
            print(f"  generated {min(i + batch_size, len(messages))}/{len(messages)}")
        return completions


@app.cls(
    image=vllm_image,
    gpu="A10G",
    # Mount the vectors Volume too: ``generate_and_save`` reads the readout and writes the
    # completion records there, so the whole job is server-side and survives a dropped launcher.
    volumes={HF_CACHE_DIR: hf_cache_volume, VECTORS_DIR: vectors_volume},
    secrets=[hf_secret],
    timeout=2 * HOURS,
)
class VLLMGenerator:
    """Same contract as ``Generator`` but backed by vLLM (continuous batching).

    vLLM batches the whole request set internally, so ``config['batch_size']`` is ignored;
    it is much faster than the HF static-batch ``Generator`` on the same A10G. Rendering
    matches the probe's pre-response position via ``enable_thinking=False``.
    """

    model_id: str = modal.parameter()
    max_model_len: int = modal.parameter(default=2560)

    @modal.enter()
    def load(self):
        from vllm import LLM

        # Qwen3.5 resolves as a *multimodal* arch, so vLLM profiles memory with dummy
        # image inputs by default -- on the tight 24GB A10G (weights alone ~17.7GB) that
        # OOMs during KV-cache sizing. We only ever send text, so: disable image/video
        # profiling, shrink the prefill chunk (less peak activation), skip CUDA graphs.
        self.llm = LLM(
            model=self.model_id,
            dtype="bfloat16",
            gpu_memory_utilization=0.95,
            # Room for a long user message + up to 1536 new tokens; chunked prefill keeps the
            # per-step batch small (max_num_batched_tokens) so the bigger context still fits.
            max_model_len=self.max_model_len,
            max_num_batched_tokens=2048,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
        )
        print(f"Loaded {self.model_id} in vLLM")

    def _generate(self, messages: list[str], config: dict) -> list[str]:
        from vllm import SamplingParams

        do_sample = config.get("do_sample", True)
        sampling = SamplingParams(
            temperature=config.get("temperature", 0.7) if do_sample else 0.0,
            top_p=config.get("top_p", 0.95) if do_sample else 1.0,
            max_tokens=config.get("max_new_tokens", 768),
            seed=config.get("seed", 42),
        )
        system_prompt = config.get("system_prompt")
        conversations = [
            ([{"role": "system", "content": system_prompt}] if system_prompt else [])
            + [{"role": "user", "content": m}]
            for m in messages
        ]
        outputs = self.llm.chat(
            conversations, sampling, chat_template_kwargs={"enable_thinking": False}
        )
        return [o.outputs[0].text.strip() for o in outputs]

    @modal.method()
    def generate(self, messages: list[str], config: dict) -> list[str]:
        """Return one visible-reply string per message (order preserved)."""
        return self._generate(messages, config)

    @modal.method()
    def generate_and_save(self, config: dict, readout_path: str, output_path: str) -> dict:
        """Fully server-side: read a readout from the vectors Volume, generate, write records back.

        Reads ``<VECTORS_DIR>/<readout_path>`` (the exp-02 probe readout), generates a reply per
        message, assembles one self-contained record each (scenario + 171-way projections + reply),
        and writes JSONL to ``<VECTORS_DIR>/<output_path>`` (committed). Because input and output
        both live on the Volume, the job survives the local launcher dying -- run it detached and
        download the result afterward with ``modal volume get``. ``config['limit']`` caps the count
        for a smoke.
        """
        import json
        import os

        rows = json.loads((open(os.path.join(VECTORS_DIR, readout_path), encoding="utf-8")).read())["messages"]
        limit = config.get("limit") or 0
        if limit:
            rows = rows[:limit]
        print(f"generate_and_save: {len(rows)} messages from {readout_path}")

        completions = self._generate([r["message"] for r in rows], config)
        records = [
            {
                "id": r["id"],
                "model": self.model_id,
                "variant": "unconditioned",
                "scenario": {k: r.get(k) for k in ("message", "emotion", "cluster", "frame", "eval_axis")},
                "probe": {"projections": r["projections"]},
                "gen": {
                    "system_prompt": config.get("system_prompt"),
                    "sampling": {k: config.get(k) for k in ("temperature", "top_p", "max_new_tokens", "seed", "do_sample")},
                },
                "completion": c,
                "tag_text": None,
            }
            for r, c in zip(rows, completions)
        ]
        out_abs = os.path.join(VECTORS_DIR, output_path)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)
        with open(out_abs, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
        vectors_volume.commit()
        print(f"wrote {len(records)} records -> {output_path}")
        return {"n": len(records), "output": output_path}

    @modal.method()
    def generate_messages_and_save(self, config: dict, rows: list[dict], output_path: str) -> dict:
        """Server-side generation for messages passed as args (no probe readout involved).

        Same detached-safe contract as ``generate_and_save``, for message sets that don't
        live on the Volume -- e.g. the sampled low-affect neutral messages. ``rows`` need
        ``id`` + ``message``; any other keys (source/domain provenance) are carried into
        the record's ``scenario``. Records match the readout-driven shape minus ``probe``.
        """
        import json
        import os

        limit = config.get("limit") or 0
        if limit:
            rows = rows[:limit]
        print(f"generate_messages_and_save: {len(rows)} messages")

        completions = self._generate([r["message"] for r in rows], config)
        records = [
            {
                "id": r["id"],
                "model": self.model_id,
                "variant": "unconditioned",
                "scenario": {"emotion": None, "cluster": None, **{k: v for k, v in r.items() if k != "id"}},
                "gen": {
                    "system_prompt": config.get("system_prompt"),
                    "sampling": {k: config.get(k) for k in ("temperature", "top_p", "max_new_tokens", "seed", "do_sample")},
                },
                "completion": c,
                "tag_text": None,
            }
            for r, c in zip(rows, completions)
        ]
        out_abs = os.path.join(VECTORS_DIR, output_path)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)
        with open(out_abs, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
        vectors_volume.commit()
        print(f"wrote {len(records)} records -> {output_path}")
        return {"n": len(records), "output": output_path}

    @modal.method()
    def regenerate_truncated(self, config: dict, output_path: str) -> dict:
        """Re-generate only the mid-word-truncated replies in an existing Volume file, at a higher
        max_new_tokens, and merge them back in place. Same server-side/detached contract as
        ``generate_and_save`` (needs a VLLMGenerator with a large enough ``max_model_len``)."""
        import json
        import os

        def _truncated(text: str) -> bool:  # cut mid-word: ends on a letter, no terminal punctuation
            t = (text or "").rstrip()
            return bool(t) and t[-1].isalpha() and not t.endswith(tuple(".!?\"”')"))

        path = os.path.join(VECTORS_DIR, output_path)
        records = [json.loads(x) for x in open(path, encoding="utf-8").read().splitlines() if x.strip()]
        idx = [i for i, r in enumerate(records) if _truncated(r.get("completion", ""))]
        print(f"regenerate_truncated: {len(idx)} truncated of {len(records)}")
        if idx:
            new = self._generate([records[i]["scenario"]["message"] for i in idx], config)
            for i, c in zip(idx, new):
                records[i]["completion"] = c
                records[i]["gen"]["sampling"]["max_new_tokens"] = config.get("max_new_tokens")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
        vectors_volume.commit()
        still = sum(1 for r in records if _truncated(r.get("completion", "")))
        print(f"wrote {len(records)} records; still truncated: {still}")
        return {"regenerated": len(idx), "still_truncated": still, "total": len(records)}

