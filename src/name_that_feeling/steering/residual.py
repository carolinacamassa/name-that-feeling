"""Steer a causal LM by adding a vector to the residual stream after one decoder block.

The protocol is the one Sofroniew et al. 2026 use for their emotion vectors (appendix
6.8 and footnote 4): a direction is added at one block, on a chosen set of token
positions, with a strength given as a fraction of the average norm of the residual
stream at that block. So the vector added to a steered position is

    strength * norm_reference * unit_direction

where ``norm_reference`` is measured once per model and block (``token_norm_stats``) and
``unit_direction`` has L2 norm 1. Blocks follow the repo's convention
(``emotion_vectors.models.LAYER_CONVENTION``): block b is the residual stream after
decoder block b, 0-indexed, which is the output of ``decoder_blocks(model)[b]`` and
Hugging Face's ``hidden_states[b + 1]``.

Which positions are steered is the caller's choice, passed as a mask over the prompt
plus a flag for the tokens generated afterwards:

- ``assistant_prefix``: the open assistant turn of the prompt (its header, the empty
  think block and the prefill), generated tokens untouched. This is the paper's rule for
  the "I feel" prompts ("on the tokens of the Assistant turn up to and including" the
  prefill).
- ``assistant_turn``: the same prompt positions and every generated token, so the
  direction is present for the whole reply, the closest analogue of a mood that sits in
  the weights.
- ``all``: every prompt position and every generated token (the paper's "steered
  throughout" of its behavioural evaluations).

Rows of a batch are steered independently: each has its own vector (zero for an
unsteered row), so one batch can mix directions and strengths.
"""

POSITIONS = ("assistant_prefix", "assistant_turn", "all")


def decoder_blocks(model) -> list:
    """The model's decoder blocks in order, so that ``decoder_blocks(model)[b]`` outputs block b.

    Looks in the places the supported backbones keep them and checks the count against
    the config, since a hook on the wrong module list would steer silently at the wrong
    depth.
    """
    for path in ("model.layers", "model.language_model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        blocks = list(obj)
        expected = getattr(model.config, "num_hidden_layers", None)
        if expected is not None and len(blocks) != expected:
            raise RuntimeError(f"{path} has {len(blocks)} modules but the config says {expected} blocks")
        return blocks
    raise RuntimeError("could not find the decoder blocks of this model (extend decoder_blocks)")


def suffix_mask(attention_mask, n_last):
    """``[batch, seq]`` bool mask of each row's last ``n_last[row]`` real tokens.

    For a left-padded batch (the sampler's layout) the real tokens are right-aligned, so
    an open assistant turn of k tokens is the last k positions of its row.
    """
    import torch

    seq = attention_mask.shape[1]
    n_last = torch.as_tensor(n_last, device=attention_mask.device)
    if (n_last > attention_mask.sum(dim=1)).any():
        raise ValueError("a row asks for more steered positions than it has tokens (prompt truncated?)")
    positions = torch.arange(seq, device=attention_mask.device)
    return (positions[None, :] >= (seq - n_last)[:, None]) & attention_mask.bool()


def prompt_mask(positions: str, attention_mask, assistant_lengths):
    """The prompt mask and the generated-token flag of one of ``POSITIONS``.

    ``assistant_lengths[row]`` is the number of tokens of the row's open assistant turn
    (ignored for ``all``). Returns ``(mask [batch, seq] bool, steer_generated bool)``.
    """
    if positions == "all":
        return attention_mask.bool(), True
    if positions in ("assistant_prefix", "assistant_turn"):
        return suffix_mask(attention_mask, assistant_lengths), positions == "assistant_turn"
    raise ValueError(f"positions must be one of {POSITIONS}, got {positions!r}")


class ResidualSteering:
    """Context manager: while open, block ``block`` outputs ``residual + delta`` on the steered positions.

    ``deltas`` is ``[batch, hidden]``, one already-scaled vector per row. ``mask`` is
    ``[batch, seq]`` over the prompt. ``steer_generated`` says whether tokens after the
    prompt get the row's vector too. One instance serves one prompt batch (one
    ``generate`` call, or one plain forward pass): the hook counts positions so that it
    works with the key-value cache (the prompt in one call, then one token per call) and
    without it (the whole sequence every call).
    """

    def __init__(self, model, block: int, deltas, mask, steer_generated: bool = False):
        if deltas.shape[0] != mask.shape[0]:
            raise ValueError(f"{deltas.shape[0]} vectors for {mask.shape[0]} rows")
        self.module = decoder_blocks(model)[block]
        self.deltas = deltas
        self.mask = mask
        self.steer_generated = bool(steer_generated)
        self.n_prompt = mask.shape[1]
        self.n_seen = 0
        self.n_calls = 0
        self.handle = None

    def __enter__(self):
        # prepend=True: transformers 5 collects ``output_hidden_states`` through forward
        # hooks of its own, installed on every block before this one. Run after them, this
        # hook would still steer everything downstream, but ``hidden_states[block + 1]``
        # would report the residual from before the vector was added, and a probe read of
        # the steered block would see no steering. Run first, the reported residual is the
        # steered one.
        self.handle = self.module.register_forward_hook(self._hook, prepend=True)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False

    def _weights(self, length: int, dtype):
        """``[batch, length]`` 0/1 weights for the positions this call carries."""
        import torch

        batch = self.mask.shape[0]
        if self.n_seen == 0 or length > 1:
            # The prompt (first call), or the whole sequence again when no cache is used.
            if length < self.n_prompt:
                raise RuntimeError(f"a call with {length} positions arrived before the {self.n_prompt}-token prompt")
            w = torch.zeros(batch, length, dtype=dtype, device=self.mask.device)
            w[:, : self.n_prompt] = self.mask.to(dtype)
            if self.steer_generated:
                w[:, self.n_prompt :] = 1
            self.n_seen = length
            return w
        # One new token per call, the prompt already in the cache.
        self.n_seen += 1
        fill = 1.0 if self.steer_generated else 0.0
        return torch.full((batch, 1), fill, dtype=dtype, device=self.mask.device)

    def _hook(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[0] != self.deltas.shape[0]:
            raise RuntimeError(f"batch of {hidden.shape[0]} rows, steering prepared for {self.deltas.shape[0]}")
        self.n_calls += 1
        w = self._weights(hidden.shape[1], hidden.dtype)
        steered = hidden + w[:, :, None] * self.deltas.to(hidden.dtype)[:, None, :]
        return (steered, *output[1:]) if isinstance(output, tuple) else steered


def token_norm_stats(model, tokenizer, texts: list[str], blocks: list[int], batch_size: int = 4, max_tokens: int = 1024) -> dict:
    """Per-token residual norms at ``blocks`` over ``texts``: the unit of a steering strength.

    The paper gives strengths "relative to the average norm of the residual stream
    activations at the corresponding layer, across a large dataset"; ``mean_norm`` below
    is that quantity, over every token after the first of each text. The first token is
    reported apart (``first_token_mean_norm``) because decoder LMs often carry an
    outsized activation there, and three more numbers say how much of the norm is a
    constant offset shared by all tokens rather than variation between them:
    ``norm_of_mean`` (the norm of the mean token activation), ``rms_norm`` and
    ``rms_centered`` (the root-mean-square norm before and after subtracting that mean).
    When ``norm_of_mean`` is close to ``rms_norm`` most of what ``mean_norm`` measures
    is the offset, and a strength of 0.5 is a larger push against the variation that
    carries content than the number suggests.

    Returns ``{block: {...}}`` with the fields above plus ``median_norm`` and ``n_tokens``.
    """
    import torch

    side, truncation = tokenizer.padding_side, tokenizer.truncation_side
    tokenizer.padding_side = "right"  # the first real token is position 0 of every row
    tokenizer.truncation_side = "right"  # and a long text keeps its start
    norms = {b: [] for b in blocks}
    first = {b: [] for b in blocks}
    sums = {b: None for b in blocks}
    try:
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(texts[start : start + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to(model.device)
            with torch.inference_mode():
                # The backbone alone: the unembedding over every position is not needed and is large.
                hidden_states = model.base_model(**enc, output_hidden_states=True).hidden_states
            keep = enc["attention_mask"].bool().clone()
            keep[:, 0] = False
            for b in blocks:
                h = hidden_states[b + 1].float()
                norms[b].append(h.norm(dim=-1)[keep].cpu())
                first[b].append(h[:, 0].norm(dim=-1).cpu())
                total = h[keep].double().sum(dim=0).cpu()
                sums[b] = total if sums[b] is None else sums[b] + total
    finally:
        tokenizer.padding_side, tokenizer.truncation_side = side, truncation
    out = {}
    for b in blocks:
        n = torch.cat(norms[b])
        mean_vec = sums[b] / n.numel()
        mean_sq = float((n.double() ** 2).mean())
        norm_of_mean = float(mean_vec.norm())
        out[b] = {
            "mean_norm": float(n.mean()),
            "median_norm": float(n.median()),
            "rms_norm": mean_sq**0.5,
            "norm_of_mean": norm_of_mean,
            "rms_centered": max(mean_sq - norm_of_mean**2, 0.0) ** 0.5,
            "first_token_mean_norm": float(torch.cat(first[b]).mean()),
            "n_tokens": int(n.numel()),
            "n_texts": len(texts),
        }
    return out


def load_direction(vectors_dir: str, vectors_run: str, block: int, emotions: list[str]):
    """A unit steering direction built from stored emotion vectors: ``(unit, report)``.

    Each emotion's centered, denoised, normalized ``unit`` vector is read from
    ``<vectors_dir>/<vectors_run>/vectors/<family>/<stored layer key>/<slug>.safetensors``
    (the layout ``emotion_vectors.extraction`` writes). One emotion gives that vector;
    several give the normalized mean of their unit vectors, and ``report`` then carries
    the pairwise cosines among them and each one's cosine with the mean, so a reader can
    see how much the members agree. The ``raw`` vectors are never used: they share one
    large offset across emotions.
    """
    import glob
    import os

    import numpy as np

    from name_that_feeling.emotion_vectors import vectors as V
    from name_that_feeling.emotion_vectors.models import stored_layer_key
    from name_that_feeling.emotion_vectors.taxonomy import slugify

    if not emotions:
        raise ValueError("a direction needs at least one emotion")
    units, paths = [], []
    for emotion in emotions:
        pattern = os.path.join(vectors_dir, vectors_run, "vectors", "*", stored_layer_key(block), f"{slugify(emotion)}.safetensors")
        found = sorted(glob.glob(pattern))
        if len(found) != 1:
            raise FileNotFoundError(f"expected one vector for {emotion!r} at block {block}, found {len(found)} under {pattern}")
        tensors, _ = V.load_vector(found[0])
        if "unit" not in tensors:
            raise KeyError(f"{found[0]} has no centered unit vector")
        units.append(tensors["unit"].astype(np.float64))
        paths.append(os.path.relpath(found[0], vectors_dir).replace(os.sep, "/"))
    U = np.stack(units)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    mean = U.mean(axis=0)
    unit = mean / np.linalg.norm(mean)
    gram = U @ U.T
    report = {
        "emotions": list(emotions),
        "block": int(block),
        "vectors_run": vectors_run,
        "files": paths,
        "pairwise_cosines": [[float(x) for x in row] for row in gram],
        "cosine_with_direction": [float(x) for x in U @ unit],
        "norm_of_mean_before_normalizing": float(np.linalg.norm(mean)),
    }
    return unit.astype(np.float32), report
