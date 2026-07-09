"""Reusable SFT on Tinker (Thinking Machines' managed fine-tuning API).

Supersedes the ``axolotl_sft`` Modal-QLoRA path for runs trained on Tinker: the local
process stays light (the ``tinker`` SDK + the model's HF tokenizer -- no torch, no GPU);
Tinker runs the actual LoRA training server-side and stores checkpoints as
``tinker://`` paths.

Rendering matches the generation/probe side exactly (see ``generation.completions``):
the prompt is the chat template at the **pre-response position**
(``add_generation_prompt=True`` + ``enable_thinking=False``, i.e. the empty
``<think></think>`` block sits in the prompt), so the model learns to emit the visible
assistant turn -- tag first -- from the same position the probe reads at. Loss is on
the assistant tokens plus the end-of-turn token only.

Auth: ``TINKER_API_KEY`` in the environment; :func:`load_api_key` pulls it from a
``.env`` file when unset.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np

from name_that_feeling.hf_router import read_token


def load_api_key(env_file: Path) -> None:
    """Ensure TINKER_API_KEY is in the environment (reads ``env_file`` if needed)."""
    if not os.environ.get("TINKER_API_KEY"):
        os.environ["TINKER_API_KEY"] = read_token(env_file, "TINKER_API_KEY")


def render_prompt(tokenizer, context_turns: list[dict]) -> str:
    """Chat template up to the pre-response position (identical to generation-time)."""
    try:
        return tokenizer.apply_chat_template(
            context_turns, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:  # template with no thinking toggle
        return tokenizer.apply_chat_template(context_turns, tokenize=False, add_generation_prompt=True)


def build_datum(tokenizer, row: dict):
    """One chat row ({"messages": [..., assistant-last]}) -> a weighted tinker Datum.

    Prompt tokens get weight 0, the assistant turn (plus the end-of-turn token) weight 1.
    Prompt and target are tokenized separately so the boundary matches generation, where
    the reply is decoded fresh from the pre-response position.
    """
    import tinker

    turns = row["messages"]
    if turns[-1]["role"] != "assistant":
        raise ValueError(f"row must end with an assistant turn, got {turns[-1]['role']}")
    prompt_ids = tokenizer.encode(render_prompt(tokenizer, turns[:-1]), add_special_tokens=False)
    target_ids = tokenizer.encode(turns[-1]["content"], add_special_tokens=False) + [tokenizer.eos_token_id]

    all_ids = prompt_ids + target_ids
    weights = [0.0] * (len(prompt_ids) - 1) + [1.0] * len(target_ids)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(all_ids[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(np.array(all_ids[1:], dtype=np.int64)),
            "weights": tinker.TensorData.from_numpy(np.array(weights, dtype=np.float32)),
        },
    )


def train_sft(
    rows: list[dict],
    *,
    base_model: str,
    run_name: str,
    lora_rank: int = 32,
    learning_rate: float = 2e-4,
    lr_schedule: str = "constant",
    batch_size: int = 32,
    num_epochs: int = 3,
    seed: int = 42,
) -> dict:
    """LoRA SFT over chat ``rows`` on Tinker; returns a JSON-able run manifest.

    Shuffles each epoch (deterministic in ``seed``), pipelines forward_backward +
    optim_step per batch, saves a resumable state per epoch and sampler weights at the
    end. The manifest carries the ``tinker://`` checkpoint paths, per-step losses
    (mean NLL over supervised tokens), and token stats.

    ``lr_schedule``: ``"constant"`` or ``"linear"`` (decay to 0 over the run -- the
    schedule the consciousness-cluster paper trains with). ``seed`` controls the data
    shuffle; whether Tinker seeds the LoRA init server-side is not documented, so a
    "reseeded" run measures the total run-to-run variance the API exposes.
    """
    import tinker

    if lr_schedule not in ("constant", "linear"):
        raise ValueError(f"unknown lr_schedule {lr_schedule!r} (expected 'constant' or 'linear')")

    service = tinker.ServiceClient()
    client = service.create_lora_training_client(base_model=base_model, rank=lora_rank)
    tokenizer = client.get_tokenizer()

    datums = [build_datum(tokenizer, row) for row in rows]
    lengths = sorted(d.model_input.length + 1 for d in datums)
    supervised = [float(np.asarray(d.loss_fn_inputs["weights"].to_numpy()).sum()) for d in datums]
    print(
        f"[{run_name}] {len(datums)} examples; tokens/example "
        f"median {lengths[len(lengths) // 2]}, p95 {lengths[int(len(lengths) * 0.95)]}, max {lengths[-1]}; "
        f"supervised tokens total {int(sum(supervised))}"
    )

    steps_per_epoch = (len(datums) + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * num_epochs
    history: list[dict] = []
    state_paths: list[str] = []
    step = 0
    t0 = time.time()
    for epoch in range(num_epochs):
        order = list(range(len(datums)))
        random.Random(seed + epoch).shuffle(order)
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            lr = learning_rate if lr_schedule == "constant" else learning_rate * (1 - step / total_steps)
            fb_future = client.forward_backward([datums[j] for j in idx], loss_fn="cross_entropy")
            opt_future = client.optim_step(tinker.AdamParams(learning_rate=lr))
            fb = fb_future.result()
            opt_future.result()
            step += 1
            weight_sum = sum(supervised[j] for j in idx)
            loss = float(fb.metrics["loss:sum"]) / weight_sum
            history.append({"step": step, "epoch": epoch + 1, "loss": round(loss, 4), "lr": lr})
            print(
                f"[{run_name}] step {step}/{total_steps} "
                f"(epoch {epoch + 1}) loss {loss:.4f} lr {lr:.2e} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        state = client.save_state(name=f"{run_name}-epoch{epoch + 1}").result()
        state_paths.append(state.path)
        print(f"[{run_name}] saved state: {state.path}", flush=True)

    sampler = client.save_weights_for_sampler(name=f"{run_name}-final").result()
    print(f"[{run_name}] saved sampler weights: {sampler.path}", flush=True)
    return {
        "run_name": run_name,
        "base_model": base_model,
        "hyperparameters": {
            "lora_rank": lora_rank,
            "learning_rate": learning_rate,
            "lr_schedule": lr_schedule,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "seed": seed,
        },
        "n_examples": len(datums),
        "supervised_tokens": int(sum(supervised)),
        "sampler_path": sampler.path,
        "state_paths": state_paths,
        "history": history,
        "wall_seconds": round(time.time() - t0, 1),
    }


def sample_replies(
    model_path: str | None,
    base_model: str,
    messages: list[str],
    *,
    max_tokens: int = 400,
    temperature: float = 0.0,
    chunk: int = 64,
) -> list[str]:
    """Greedy-by-default replies, rendered at the training pre-response position.

    ``model_path`` is a ``tinker://`` sampler checkpoint; pass ``None`` to sample the
    **untouched base model** (``base_model``) -- used for the eval's capability/leakage
    baselines. Requests are submitted in chunks of ``chunk`` and then collected, so the
    server pipelines them (much faster than one-at-a-time for the ~hundreds of eval
    prompts); order is preserved.
    """
    import tinker
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    service = tinker.ServiceClient()
    client = (
        service.create_sampling_client(model_path=model_path)
        if model_path
        else service.create_sampling_client(base_model=base_model)
    )
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature)

    prompts = [
        tinker.ModelInput.from_ints(
            tokenizer.encode(render_prompt(tokenizer, [{"role": "user", "content": m}]), add_special_tokens=False)
        )
        for m in messages
    ]
    replies: list[str] = []
    for i in range(0, len(prompts), chunk):
        futures = [client.sample(p, 1, params) for p in prompts[i : i + chunk]]
        for f in futures:
            tokens = f.result().sequences[0].tokens
            replies.append(tokenizer.decode(tokens, skip_special_tokens=True).strip())
    return replies


def sample_conversations(
    model_path: str | None,
    base_model: str,
    conversations: list[list[str]],
    *,
    max_tokens: int = 400,
    temperature: float = 0.0,
    chunk: int = 64,
    history_transform=None,
) -> list[list[str]]:
    """Multi-turn greedy sampling: step every conversation one user turn at a time.

    ``conversations`` is a list of user-turn lists. At each depth, every conversation
    that still has a user turn is sampled together (chunked/pipelined like
    :func:`sample_replies`), and each reply is appended to its history before the next
    depth. ``history_transform(reply) -> str``, when given, rewrites a reply before it
    enters the *context* (the returned replies stay raw) — e.g. stripping the
    ``<emotion>`` tag to test the deployment condition where the model never sees its
    own past tags. Returns, per conversation, the list of raw assistant replies.
    """
    import tinker
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    service = tinker.ServiceClient()
    client = (
        service.create_sampling_client(model_path=model_path)
        if model_path
        else service.create_sampling_client(base_model=base_model)
    )
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature)

    histories: list[list[dict]] = [[] for _ in conversations]
    replies: list[list[str]] = [[] for _ in conversations]
    for depth in range(max(len(c) for c in conversations)):
        active = [i for i, c in enumerate(conversations) if depth < len(c)]
        prompts = []
        for i in active:
            histories[i].append({"role": "user", "content": conversations[i][depth]})
            prompts.append(
                tinker.ModelInput.from_ints(
                    tokenizer.encode(render_prompt(tokenizer, histories[i]), add_special_tokens=False)
                )
            )
        outs: list[str] = []
        for j in range(0, len(prompts), chunk):
            futures = [client.sample(p, 1, params) for p in prompts[j : j + chunk]]
            outs.extend(
                tokenizer.decode(f.result().sequences[0].tokens, skip_special_tokens=True).strip()
                for f in futures
            )
        for i, reply in zip(active, outs):
            replies[i].append(reply)
            histories[i].append(
                {"role": "assistant", "content": history_transform(reply) if history_transform else reply}
            )
    return replies
