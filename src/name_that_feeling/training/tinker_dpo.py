"""Reusable DPO on Tinker without local torch (05-tag-dpo description §5).

Tinker has no server-side DPO loss, and the cookbook's ``forward_backward_custom``
route needs torch locally (excluded from this venv by policy). DPO's gradient has a
closed form, though: with margin m = (logpi_c - logref_c) - (logpi_r - logref_r)
summed over the credited tokens, the loss -log sigma(beta * m) has per-token gradient
-beta * sigma(-beta * m) on each chosen credited token's logprob and the opposite
sign on each rejected one. So each step is: (1) one forward-only pass for the current
policy's per-token logprobs of both sequences (their credited-span sums give the
margin), (2) weight w = beta * sigma(-beta * m) per pair, (3)
``forward_backward(loss_fn="importance_sampling")`` with per-token advantages +w on
chosen / -w on rejected credited tokens and ``logprobs`` set to the step-1 values
(ratio exactly 1, so the surrogate's gradient is the DPO gradient), then
``optim_step``. Reference logprobs come once, up front, from the frozen sampler
checkpoint via ``compute_logprobs``.

Credit modes: ``"tag"`` credits the reply's opening ``<emotion>...</emotion>`` span
only (the sequence is truncated right after it; no EOS -- the reply continues after
the tag, and an EOS would teach ending there); ``"full"`` credits the whole reply
plus EOS (the later whole-sequence arm).

History sanity: ``mean_margin`` and ``frac_margin_positive`` should RISE over steps;
a consistent fall means the importance-sampling surrogate's sign convention is the
opposite of the one assumed here -- flip the advantage signs and re-run.
"""

from __future__ import annotations

import math
import random
import time

import numpy as np

from name_that_feeling.training.tinker_sft import render_prompt

TAG_END = "</emotion>"


def credited_text(reply: str, credit: str) -> str:
    """The trained portion of a reply under the given credit mode."""
    if credit == "tag":
        idx = reply.find(TAG_END)
        if idx < 0:
            raise ValueError(f"reply has no {TAG_END}: {reply[:80]!r}")
        return reply[: idx + len(TAG_END)]
    if credit == "full":
        return reply
    raise ValueError(f"unknown credit mode {credit!r}")


def _sequence(tokenizer, message: str, reply: str, credit: str) -> tuple[list[int], int]:
    """Token ids of prompt + credited reply portion, and the prompt length.

    Prompt and target are tokenized separately (the SFT boundary discipline of
    ``tinker_sft.build_datum``); ``credit="full"`` appends EOS, ``"tag"`` does not.
    """
    prompt_ids = tokenizer.encode(
        render_prompt(tokenizer, [{"role": "user", "content": message}]), add_special_tokens=False
    )
    target_ids = tokenizer.encode(credited_text(reply, credit), add_special_tokens=False)
    if credit == "full":
        target_ids = target_ids + [tokenizer.eos_token_id]
    return prompt_ids + target_ids, len(prompt_ids)


def _ce_datum(tinker, seq: list[int], n_prompt: int):
    weights = [0.0] * (n_prompt - 1) + [1.0] * (len(seq) - n_prompt)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(seq[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(np.array(seq[1:], dtype=np.int64)),
            "weights": tinker.TensorData.from_numpy(np.array(weights, dtype=np.float32)),
        },
    )


def _per_token_logprobs(client, tinker, sequences: list[tuple[list[int], int]]) -> list[np.ndarray]:
    """Current-policy per-token logprobs (aligned with ``seq[1:]``), forward only."""
    out = client.forward([_ce_datum(tinker, seq, n) for seq, n in sequences], loss_fn="cross_entropy").result()
    arrays = []
    for (seq, _), o in zip(sequences, out.loss_fn_outputs):
        lp = np.asarray(o["logprobs"].to_numpy(), dtype=np.float32)
        if lp.shape[0] != len(seq) - 1:
            raise RuntimeError(f"forward logprobs length {lp.shape[0]} != {len(seq) - 1} (alignment assumption broken)")
        arrays.append(lp)
    return arrays


def _reference_logprob_sums(ref_client, tinker, sequences: list[tuple[list[int], int]]) -> list[float]:
    """Frozen-reference logprob sum over each credited span (one-time, pipelined).

    ``compute_logprobs`` returns a future (the ``_async`` variant is an asyncio
    coroutine instead) -- submit all requests up front, then resolve in order.
    """
    futures = [ref_client.compute_logprobs(tinker.ModelInput.from_ints(seq)) for seq, _ in sequences]
    sums = []
    for (seq, n_prompt), f in zip(sequences, futures):
        lp = f.result()
        sums.append(float(sum(v for v in lp[n_prompt:] if v is not None)))
    return sums


def train_dpo(
    pairs: list[dict],
    *,
    base_model: str,
    run_name: str,
    init_state_path: str,
    reference_sampler_path: str,
    credit: str = "tag",
    dpo_beta: float = 0.1,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    num_epochs: int = 1,
    seed: int = 42,
) -> dict:
    """DPO over ``pairs`` (rows with ``message``/``chosen_reply``/``rejected_reply``).

    Starts from ``init_state_path`` (LoRA rank is inherited from the checkpoint); the
    reference policy is ``reference_sampler_path``, frozen. Returns a JSON-able run
    manifest with per-step margin history.
    """
    import tinker

    service = tinker.ServiceClient()
    client = service.create_training_client_from_state(init_state_path)
    tokenizer = (
        client.get_tokenizer()
        if hasattr(client, "get_tokenizer")
        else __import__("transformers").AutoTokenizer.from_pretrained(base_model)
    )
    ref_client = service.create_sampling_client(model_path=reference_sampler_path)

    seqs_c = [_sequence(tokenizer, p["message"], p["chosen_reply"], credit) for p in pairs]
    seqs_r = [_sequence(tokenizer, p["message"], p["rejected_reply"], credit) for p in pairs]
    print(f"[{run_name}] {len(pairs)} pairs (credit={credit}); computing reference logprobs ...", flush=True)
    ref_c = _reference_logprob_sums(ref_client, tinker, seqs_c)
    ref_r = _reference_logprob_sums(ref_client, tinker, seqs_r)

    steps_per_epoch = (len(pairs) + batch_size - 1) // batch_size
    history: list[dict] = []
    state_paths: list[str] = []
    step = 0
    t0 = time.time()
    for epoch in range(num_epochs):
        order = list(range(len(pairs)))
        random.Random(seed + epoch).shuffle(order)
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            batch = [seqs_c[j] for j in idx] + [seqs_r[j] for j in idx]
            lps = _per_token_logprobs(client, tinker, batch)
            sums = [float(lp[n - 1 :].sum()) for (seq, n), lp in zip(batch, lps)]
            pol_c, pol_r = sums[: len(idx)], sums[len(idx) :]

            margins = [(pol_c[k] - ref_c[j]) - (pol_r[k] - ref_r[j]) for k, j in enumerate(idx)]
            fb_datums = []
            for k, j in enumerate(idx):
                w = dpo_beta / (1.0 + math.exp(dpo_beta * margins[k]))  # beta * sigmoid(-beta*margin)
                for (seq, n_prompt), lp, sign in (
                    (seqs_c[j], lps[k], +1.0),
                    (seqs_r[j], lps[len(idx) + k], -1.0),
                ):
                    advantages = [0.0] * (n_prompt - 1) + [sign * w] * (len(seq) - n_prompt)
                    fb_datums.append(
                        tinker.Datum(
                            model_input=tinker.ModelInput.from_ints(seq[:-1]),
                            loss_fn_inputs={
                                "target_tokens": tinker.TensorData.from_numpy(np.array(seq[1:], dtype=np.int64)),
                                "logprobs": tinker.TensorData.from_numpy(lp),
                                "advantages": tinker.TensorData.from_numpy(np.array(advantages, dtype=np.float32)),
                            },
                        )
                    )
            fb_future = client.forward_backward(fb_datums, loss_fn="importance_sampling")
            opt_future = client.optim_step(tinker.AdamParams(learning_rate=learning_rate))
            fb_future.result()
            opt_future.result()
            step += 1
            history.append(
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "mean_margin": round(float(np.mean(margins)), 4),
                    "frac_margin_positive": round(float(np.mean([m > 0 for m in margins])), 3),
                    "mean_weight": round(
                        float(np.mean([dpo_beta / (1 + math.exp(dpo_beta * m)) for m in margins])), 5
                    ),
                }
            )
            print(
                f"[{run_name}] step {step}/{steps_per_epoch * num_epochs} "
                f"margin {history[-1]['mean_margin']:+.3f} "
                f"acc {history[-1]['frac_margin_positive']:.2f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        state = client.save_state(name=f"{run_name}-epoch{epoch + 1}").result()
        state_paths.append(state.path)
    sampler = client.save_weights_for_sampler(name=f"{run_name}-final").result()
    print(f"[{run_name}] saved sampler weights: {sampler.path}", flush=True)
    return {
        "run_name": run_name,
        "base_model": base_model,
        "init_state_path": init_state_path,
        "reference_sampler_path": reference_sampler_path,
        "credit": credit,
        "hyperparameters": {
            "dpo_beta": dpo_beta,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "seed": seed,
        },
        "n_pairs": len(pairs),
        "sampler_path": sampler.path,
        "state_paths": state_paths,
        "history": history,
        "wall_seconds": round(time.time() - t0, 1),
    }
