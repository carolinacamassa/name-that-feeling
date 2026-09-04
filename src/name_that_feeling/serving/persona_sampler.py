"""Sample replies on Modal from the base model plus ONE exported LoRA adapter.

The persona teachers (experiments/06-persona-teachers) are trained on Tinker and
exported to the vectors Volume as causal-LM PEFT adapters (``training.tinker_export``,
``adapters/<run_name>/peft-causal-lm``). This module is the Modal-side way to talk to
one of them without Tinker credits: load ``Qwen/Qwen3.5-9B`` on an A10G, apply the
adapter, and sample uninstructed single-turn replies.

Fidelity rules (the same ones the training and the probe pipeline follow):

- prompts are rendered with ``training.tinker_sft.render_prompt`` -- the chat template up
  to the assistant header with thinking disabled, no system prompt -- which is the
  position the DPO replies were trained at;
- the adapter is applied the way ``emotion_vectors.extraction.ActivationExtractor``
  applies it: PEFT-injected, unmerged (merging materializes full-size deltas and can OOM
  the 24 GB card with the 9B resident), so the sampled model and the probed model are
  the same weights;
- the load asserts that every adapter tensor found a LoRA slot, because a silently
  half-loaded adapter is base + noise.

Sampling defaults are the student settings every persona model has been sampled at
(temperature 0.7, top_p 0.95, 1536 new tokens); callers override per call.

Transformers + PEFT rather than vLLM (checked 2026-09-04 against the image's vLLM
0.24.0): vLLM's Qwen3.5 classes do declare LoRA support, including the linear-attention
projections (packed ``in_proj_qkvz`` / ``in_proj_ba``), but its PEFT reader has no
``rank_pattern`` / ``alpha_pattern`` fields, and the exported adapters rely on exactly
those to pin the fused ``in_proj_qkv`` module at rank 192 (``tinker_export.
relayout_for_causal_lm``). The scale happens to survive (alpha/r = 1 either way) and
``max_lora_rank=256`` would make the tensors fit, so a vLLM path is plausible, but it
would need a greedy-equivalence check against this loader before anything is trusted
from it; the PEFT loader is the one the probe pipeline already trusts. The generation
loop is HF ``generate`` with static batches (no flash-linear-attention kernels in the
image, so decoding is slow -- minutes per batch at 1536 tokens); fine for evaluation-
sized sets, and the detached ``sample_to_volume`` exists for longer runs.

    uv run modal run -m name_that_feeling.serving.persona_sampler \\
        --run-name 10-irritated-oct --prompts "Healthy recipes || Write me an essay ..."

``--run-name base`` samples the plain base model on the same prompts. Run it as a
module (``-m``): invoked by file path, Modal mounts the bare file and the container
cannot import ``name_that_feeling``.
"""

import modal

from name_that_feeling.infra import (
    HF_CACHE_DIR,
    HOURS,
    VECTORS_DIR,
    hf_cache_volume,
    hf_secret,
    causal_lm_adapter_subpath,
    vectors_image,
    vectors_volume,
)

app = modal.App("name-that-feeling-serving")

BASE_MODEL = "Qwen/Qwen3.5-9B"

# The student settings (experiments/06-persona-teachers/config.yaml ``student:``).
STUDENT_SAMPLING = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 1536}

BASE_ALIASES = ("", "base", "none")


@app.cls(
    image=vectors_image,
    gpu="A10G",
    volumes={HF_CACHE_DIR: hf_cache_volume, VECTORS_DIR: vectors_volume},
    secrets=[hf_secret],
    timeout=2 * HOURS,
)
class PersonaSampler:
    """One container = base model + one adapter (``run_name``; base aliases = no adapter)."""

    base_model: str = modal.parameter(default=BASE_MODEL)
    run_name: str = modal.parameter(default="")

    @modal.enter()
    def load(self):
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
        try:
            model = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)
        except (ValueError, KeyError):
            from transformers import Qwen3_5ForCausalLM

            model = Qwen3_5ForCausalLM.from_pretrained(self.base_model, **kwargs)

        self.adapter_subpath = "" if self.run_name in BASE_ALIASES else causal_lm_adapter_subpath(self.run_name)
        self.load_report = {"base_model": self.base_model, "run_name": self.run_name, "adapter_subpath": self.adapter_subpath}
        if self.adapter_subpath:
            from peft import PeftModel
            from safetensors import safe_open

            full = os.path.join(VECTORS_DIR, self.adapter_subpath)
            if not os.path.isdir(full):
                raise FileNotFoundError(f"no exported adapter at Volume:{self.adapter_subpath} (run export_adapter.py first)")
            with safe_open(os.path.join(full, "adapter_model.safetensors"), framework="pt") as f:
                n_tensors = len(list(f.keys()))
            model = PeftModel.from_pretrained(model, full).get_base_model()
            n_slots = sum(1 for name, _ in model.named_parameters() if ".lora_" in name)
            self.load_report.update({"adapter_tensors": n_tensors, "lora_slots": n_slots})
            if n_slots != n_tensors:
                raise RuntimeError(
                    f"adapter {self.adapter_subpath}: {n_tensors} tensors on disk but {n_slots} LoRA "
                    "parameters in the model -- layout mismatch, the model would be base + noise"
                )
            print(f"Applied LoRA adapter (unmerged) from {full}: {n_tensors} tensors, all slotted")
        model.eval()
        self.model = model
        print(f"Loaded {self.base_model} + {self.adapter_subpath or '(no adapter)'}")

    def _render(self, prompt: str) -> str:
        from name_that_feeling.training.tinker_sft import render_prompt

        return render_prompt(self.tokenizer, [{"role": "user", "content": prompt}], enable_thinking=False)

    def _sample(self, prompts: list[str], sampling: dict | None) -> list[dict]:
        torch = self.torch
        params = {**STUDENT_SAMPLING, **(sampling or {})}
        batch_size = int(params.get("batch_size", 8))
        seed = params.get("seed", 0)
        max_prompt_tokens = int(params.get("max_prompt_tokens", 1024))
        # End-of-turn ids: the generation config's (an int or a list) plus the tokenizer's.
        configured = self.model.generation_config.eos_token_id
        eos_ids = {configured} if isinstance(configured, int) else set(configured or [])
        eos_ids.add(self.tokenizer.eos_token_id)

        self.tokenizer.padding_side = "left"  # batched decode: replies start at the same position
        self.tokenizer.truncation_side = "left"  # keep the assistant header if a prompt is long

        records: list[dict] = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = self.tokenizer(
                [self._render(p) for p in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_tokens,
            ).to("cuda")
            if seed is not None:
                torch.manual_seed(int(seed) + start)
            with torch.inference_mode():
                out = self.model.generate(
                    **enc,
                    do_sample=True,
                    temperature=float(params["temperature"]),
                    top_p=float(params["top_p"]),
                    max_new_tokens=int(params["max_new_tokens"]),
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new = out[:, enc["input_ids"].shape[1] :]
            for prompt, ids in zip(batch, new):
                toks = ids.tolist()
                n = next((i for i, t in enumerate(toks) if t in eos_ids), len(toks))
                text = self.tokenizer.decode(toks[:n], skip_special_tokens=True).strip()
                records.append(
                    {
                        "prompt": prompt,
                        "reply": text,
                        "n_tokens": n,
                        "finish": "stop" if n < len(toks) else "length",
                    }
                )
            print(f"  sampled {min(start + batch_size, len(prompts))}/{len(prompts)}")
        return records

    def _payload(self, prompts: list[str], sampling: dict | None) -> dict:
        return {
            "load": self.load_report,
            "sampling": {**STUDENT_SAMPLING, **(sampling or {})},
            "replies": self._sample(prompts, sampling),
        }

    @modal.method()
    def sample(self, prompts: list[str], sampling: dict | None = None) -> dict:
        """One reply per prompt (order preserved), plus the load report.

        ``sampling`` overrides ``STUDENT_SAMPLING`` keys (``temperature``, ``top_p``,
        ``max_new_tokens``) and may add ``seed``, ``batch_size``, ``max_prompt_tokens``.
        """
        return self._payload(prompts, sampling)

    @modal.method()
    def sample_to_volume(self, prompts_path: str, output_path: str, sampling: dict | None = None) -> dict:
        """Server-side variant for long runs: read ``<Volume>/<prompts_path>`` (a JSON list of
        ``{"id", "prompt"}``), write ``<Volume>/<output_path>`` (the ``sample`` payload with ids),
        commit. Launch with ``modal run --detach`` so a dying local launcher cannot lose the run.
        """
        import json
        import os

        rows = json.loads(open(os.path.join(VECTORS_DIR, prompts_path), encoding="utf-8").read())
        payload = self._payload([r["prompt"] for r in rows], sampling)
        for row, rec in zip(rows, payload["replies"]):
            rec["id"] = row["id"]
        out_abs = os.path.join(VECTORS_DIR, output_path)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)
        with open(out_abs, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, ensure_ascii=False)
        vectors_volume.commit()
        print(f"wrote {len(rows)} replies -> {output_path}")
        return {"n": len(rows), "output": output_path}


DEFAULT_PROMPTS = "Healthy recipes || Write me an essay on self evaluation of strengths and weaknesses"


@app.local_entrypoint()
def main(
    run_name: str = "10-irritated-oct",
    prompts: str = DEFAULT_PROMPTS,
    base_model: str = BASE_MODEL,
    max_new_tokens: int = STUDENT_SAMPLING["max_new_tokens"],
    seed: int = 0,
) -> None:
    """Smoke run: print one reply per ``||``-separated prompt from ``run_name`` (or ``base``)."""
    prompt_list = [p.strip() for p in prompts.split("||") if p.strip()]
    sampler = PersonaSampler(base_model=base_model, run_name="" if run_name in BASE_ALIASES else run_name)
    result = sampler.sample.remote(prompt_list, {"max_new_tokens": max_new_tokens, "seed": seed})
    print(f"\n=== {run_name}  load={result['load']}  sampling={result['sampling']}")
    for rec in result["replies"]:
        print(f"\n--- PROMPT: {rec['prompt']}\n--- REPLY ({rec['n_tokens']} tokens, {rec['finish']}):\n{rec['reply']}")
