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

Displacement watch (2026-08-11, docs/rlhf-book-notes-for-tag-dpo.md §3): the history
also records ``mean_chosen_reward`` and ``mean_rejected_reward`` -- each side's
implicit reward beta * (policy logprob sum - reference logprob sum) over the credited
span, whose difference is beta * margin. A rising margin can mean two things: the
chosen replies becoming more likely (healthy), or BOTH sides becoming less likely
with the rejected falling faster (preference displacement -- the freed probability
mass flows to sequences that appeared in neither slot; the tag-masked runaway-tag
collapse was this). ``mean_chosen_reward`` falling while ``mean_margin`` rises is the
displacement signature: treat it as a stop-the-run alarm, not a curiosity.

Optional OCT terms (2026-09-01; ground truth = the Open Character Training
OpenRLHF fork, maiush/OpenRLHF commit 40b6d1b): ``nll_coef`` adds their NLL
anchor -- chosen-only, length-normalized negative log-likelihood, their value
0.1 -- and ``kl_coef`` adds what their paper calls a per-token KL penalty but
their code computes as the length-normalized mean SQUARED per-token log-ratio
(their logging name: ``sq_approx_kl``), symmetric, over both chosen and
rejected sequences, their value 0.001. Both fold exactly into the per-token
advantages (the surrogate's gradient is -A_t * grad log pi, so each term
contributes minus its d/dlogpi); no extra pass and no approximation. With both
coefficients at their 0.0 defaults the advantages are token-identical to the
original scheme.
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


def _reference_logprob_tokens(ref_client, tinker, sequences: list[tuple[list[int], int]]) -> list[np.ndarray]:
    """Frozen-reference PER-TOKEN logprobs, aligned with the forward arrays.

    ``compute_logprobs`` is indexed by sequence position (``lp[j]`` scores
    ``seq[j]``, ``lp[0]`` is None), while the forward-pass arrays have length
    ``len(seq) - 1`` with element ``i`` scoring ``seq[i + 1]`` -- so the aligned
    array is ``lp[1:]``. Missing entries become NaN; downstream, a NaN delta is
    zeroed so a missing reference value contributes nothing to the penalty
    (substituting 0.0 for a logprob would manufacture a huge spurious delta).
    """
    futures = [ref_client.compute_logprobs(tinker.ModelInput.from_ints(seq)) for seq, _ in sequences]
    arrays = []
    for (seq, _), f in zip(sequences, futures):
        lp = f.result()
        arr = np.array([np.nan if v is None else v for v in lp[1:]], dtype=np.float32)
        if arr.shape[0] != len(seq) - 1:
            raise RuntimeError(f"reference logprobs length {arr.shape[0]} != {len(seq) - 1} (alignment assumption broken)")
        arrays.append(arr)
    return arrays


def _pair_advantages(
    lp: np.ndarray,
    ref_tok: np.ndarray | None,
    n_prompt: int,
    w: float,
    sign: float,
    nll_coef: float,
    kl_coef: float,
) -> tuple[np.ndarray, float | None]:
    """Per-token advantages for one sequence, plus its mean squared log-ratio.

    Gradient-exact port of the OCT additions: chosen tokens (sign=+1) carry
    ``+w + nll_coef/n - kl_coef*delta_t/n``, rejected tokens (sign=-1) carry
    ``-w - kl_coef*delta_t/n``, with ``n`` the credited-token count and
    ``delta_t`` the policy-minus-reference logprob of the realized token. With
    both coefficients 0.0 this is exactly the constant ``sign * w`` fill.
    """
    n_credit = lp.shape[0] - (n_prompt - 1)
    adv = np.zeros(lp.shape[0], dtype=np.float32)
    adv[n_prompt - 1 :] = sign * w
    sq = None
    if nll_coef and sign > 0:
        adv[n_prompt - 1 :] += nll_coef / n_credit
    if kl_coef:
        if ref_tok is None:
            raise ValueError("kl_coef > 0 requires per-token reference logprobs")
        delta = lp[n_prompt - 1 :] - ref_tok[n_prompt - 1 :]
        delta = np.where(np.isnan(delta), 0.0, delta).astype(np.float32)
        adv[n_prompt - 1 :] -= kl_coef * delta / n_credit
        sq = float(np.mean(delta**2))
    return adv, sq


def train_dpo(
    pairs: list[dict],
    *,
    base_model: str,
    run_name: str,
    init_state_path: str | None,
    reference_sampler_path: str | None,
    lora_rank: int | None = None,
    credit: str = "tag",
    dpo_beta: float = 0.1,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    num_epochs: int = 1,
    seed: int = 42,
    nll_coef: float = 0.0,
    kl_coef: float = 0.0,
) -> dict:
    """DPO over ``pairs`` (rows with ``message``/``chosen_reply``/``rejected_reply``).

    Starts from ``init_state_path`` (LoRA rank inherited from the checkpoint), or —
    when ``init_state_path`` is None — from the untouched ``base_model`` with a fresh
    LoRA of ``lora_rank`` (the persona-teacher path, OCT rank 64). The reference
    policy is ``reference_sampler_path``, frozen; None means the un-adapted
    ``base_model`` itself (OCT's ``ref_pretrain`` default). Returns a JSON-able run
    manifest with per-step margin history.

    ``nll_coef`` and ``kl_coef`` are the optional OCT terms (module docstring;
    their values 0.1 and 0.001); both default to 0.0, which reproduces the
    original loss exactly.
    """
    import tinker

    service = tinker.ServiceClient()
    if init_state_path:
        client = service.create_training_client_from_state(init_state_path)
    else:
        if not lora_rank:
            raise ValueError("starting from base requires lora_rank")
        client = service.create_lora_training_client(base_model=base_model, rank=lora_rank)
    tokenizer = (
        client.get_tokenizer()
        if hasattr(client, "get_tokenizer")
        else __import__("transformers").AutoTokenizer.from_pretrained(base_model)
    )
    ref_client = (
        service.create_sampling_client(model_path=reference_sampler_path)
        if reference_sampler_path
        else service.create_sampling_client(base_model=base_model)
    )

    seqs_c = [_sequence(tokenizer, p["message"], p["chosen_reply"], credit) for p in pairs]
    seqs_r = [_sequence(tokenizer, p["message"], p["rejected_reply"], credit) for p in pairs]
    print(f"[{run_name}] {len(pairs)} pairs (credit={credit}); computing reference logprobs ...", flush=True)
    if kl_coef:
        # One reference pass serves both needs: the per-token arrays for the
        # squared-log-ratio term, and the credited-span sums derived from them.
        reftok_c = _reference_logprob_tokens(ref_client, tinker, seqs_c)
        reftok_r = _reference_logprob_tokens(ref_client, tinker, seqs_r)
        ref_c = [float(np.nansum(a[n - 1 :])) for (_, n), a in zip(seqs_c, reftok_c)]
        ref_r = [float(np.nansum(a[n - 1 :])) for (_, n), a in zip(seqs_r, reftok_r)]
    else:
        reftok_c = reftok_r = None
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
            chosen_rewards = [dpo_beta * (pol_c[k] - ref_c[j]) for k, j in enumerate(idx)]
            rejected_rewards = [dpo_beta * (pol_r[k] - ref_r[j]) for k, j in enumerate(idx)]
            fb_datums = []
            sq_vals: list[float] = []
            for k, j in enumerate(idx):
                w = dpo_beta / (1.0 + math.exp(dpo_beta * margins[k]))  # beta * sigmoid(-beta*margin)
                for (seq, n_prompt), lp, sign, ref_tok in (
                    (seqs_c[j], lps[k], +1.0, None if reftok_c is None else reftok_c[j]),
                    (seqs_r[j], lps[len(idx) + k], -1.0, None if reftok_r is None else reftok_r[j]),
                ):
                    advantages, sq = _pair_advantages(lp, ref_tok, n_prompt, w, sign, nll_coef, kl_coef)
                    if sq is not None:
                        sq_vals.append(sq)
                    fb_datums.append(
                        tinker.Datum(
                            model_input=tinker.ModelInput.from_ints(seq[:-1]),
                            loss_fn_inputs={
                                "target_tokens": tinker.TensorData.from_numpy(np.array(seq[1:], dtype=np.int64)),
                                "logprobs": tinker.TensorData.from_numpy(lp),
                                "advantages": tinker.TensorData.from_numpy(advantages),
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
                    "mean_chosen_reward": round(float(np.mean(chosen_rewards)), 4),
                    "mean_rejected_reward": round(float(np.mean(rejected_rewards)), 4),
                    "mean_weight": round(
                        float(np.mean([dpo_beta / (1 + math.exp(dpo_beta * m)) for m in margins])), 5
                    ),
                    # OCT-term observability: chosen-side length-normalized NLL
                    # (always cheap to compute), and their sq_approx_kl when active.
                    "mean_nll": round(
                        float(
                            np.mean(
                                [
                                    -pol_c[k] / (len(seqs_c[j][0]) - seqs_c[j][1])
                                    for k, j in enumerate(idx)
                                ]
                            )
                        ),
                        4,
                    ),
                    "sq_approx_kl": round(float(np.mean(sq_vals)), 6) if sq_vals else None,
                }
            )
            print(
                f"[{run_name}] step {step}/{steps_per_epoch * num_epochs} "
                f"margin {history[-1]['mean_margin']:+.3f} "
                f"chosen {history[-1]['mean_chosen_reward']:+.3f} "
                f"rejected {history[-1]['mean_rejected_reward']:+.3f} "
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
            "nll_coef": nll_coef,
            "kl_coef": kl_coef,
            "lora_rank": lora_rank,  # None when inherited from init_state_path
        },
        "n_pairs": len(pairs),
        "sampler_path": sampler.path,
        "state_paths": state_paths,
        "history": history,
        "wall_seconds": round(time.time() - t0, 1),
    }
