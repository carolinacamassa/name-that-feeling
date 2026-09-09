"""Qwen3.5-9B activation extraction + vector building + Tylenol readout (GPU).

The 9B weights are loaded once per container (``@modal.enter``) and reused across
emotions, so a full-``emotions.txt`` sweep via ``.map`` pays the load + download
cost only on cold start. Heavy imports (torch/transformers/numpy) live inside the
methods so importing this module locally needs only ``modal``.
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
)

from . import app
from .models import emotion_vectors_run
from .taxonomy import slugify


def _vectors_dir(run_name: str, cluster: str, layer: int) -> str:
    """Vectors are organized by cluster, then layer: vectors/<cluster>/layer_<L>/."""
    import os

    return os.path.join(VECTORS_DIR, run_name, "vectors", cluster, f"layer_{layer}")


def _neutral_path(run_name: str, layer: int) -> str:
    """Cached neutral-baseline pooled activations for a layer (shared across emotions)."""
    import os

    return os.path.join(VECTORS_DIR, run_name, "neutral", f"layer_{layer}.npy")


def _neutral_source_run(config: dict, run_name: str) -> str:
    """Which run holds the cached neutral baseline to use (default: the run itself).

    ``config['neutral_run']`` points a run at another run's ``cache_neutral`` output, so
    several vector runs can share one baseline. That matters when the baseline is meant
    to be held fixed while something else varies -- comparing two story corpora, say --
    since re-caching the same neutral texts per run would otherwise be both wasted GPU
    time and an invitation to let the baseline drift between arms.
    """
    return config.get("neutral_run") or run_name


def _load_units(vectors_run: str, layer: int) -> tuple:
    """Every centered ``unit`` vector of a run at one layer, stacked in sorted-slug order.

    Returns ``(names, clusters, U)`` with ``U`` of shape ``[n_emotions, hidden]`` and
    ``clusters[name]`` the family each vector was built under. Raises when the run has
    no vectors at that layer or when a vector has no ``unit`` yet (``recenter_vectors``
    not run).
    """
    import glob
    import os

    import numpy as np

    from . import vectors as V

    pattern = os.path.join(VECTORS_DIR, vectors_run, "vectors", "*", f"layer_{layer}", "*.safetensors")
    names, clusters, units = [], {}, []
    for p in sorted(glob.glob(pattern)):
        tensors, meta = V.load_vector(p)
        if "unit" not in tensors:
            raise KeyError(f"{p} has no centered unit yet -- run recenter_vectors on {vectors_run}.")
        name = os.path.splitext(os.path.basename(p))[0]
        names.append(name)
        clusters[name] = meta.get("cluster") or os.path.basename(os.path.dirname(os.path.dirname(p)))
        units.append(tensors["unit"])
    if not names:
        raise FileNotFoundError(f"no vectors under {pattern}")
    return names, clusters, np.stack(units).astype(np.float32)

TRANSCRIPT_POSITIONS = ("user_mean", "pre_response", "reply_mean")


def _user_token_span(tokenizer, prompt_text: str, content: str, p_ids: list) -> tuple:
    """The token span of the user's own message inside a rendered prompt.

    The rule, kept here and recorded in every ``meta.json`` written with it: find the
    characters of the user's message where the chat template placed them in the rendered
    prompt (the last occurrence, so a template that repeats the text elsewhere cannot
    win), map those characters onto tokens with the tokenizer's own offsets, and keep a
    token only when its whole character span lies inside them. Everything the template
    contributes is therefore left out -- the turn header before the message, the
    end-of-turn marker, the assistant header and the empty think block after it -- and so
    is any token straddling either boundary, since such a token also carries template
    characters. Returns ``(start, end)`` as a half-open range into the prompt's tokens.
    """
    start_char = prompt_text.rfind(content)
    if start_char < 0:
        raise ValueError("the user's message does not appear verbatim in the rendered prompt")
    end_char = start_char + len(content)
    enc = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc["input_ids"]) != list(p_ids):
        raise ValueError("the offset-mapped tokenization differs from the prompt's own token ids")
    inside = [
        k
        for k, (a, b) in enumerate(enc["offset_mapping"])
        if b > a and a >= start_char and b <= end_char
    ]
    if not inside:
        raise ValueError("no whole token lies inside the user's message")
    if inside[-1] - inside[0] + 1 != len(inside):
        raise ValueError("the user's message does not map onto a contiguous run of tokens")
    return inside[0], inside[-1] + 1


def load_backbone(model_id: str, adapter_path: str = "") -> tuple:
    """Load the causal-LM text backbone on CUDA, optionally with an exported LoRA adapter.

    Returns ``(model, tokenizer, load_report)``, the model in eval mode. The text
    backbone only is loaded (reads text_config, skips the vision tower): the Auto class
    first, falling back to the explicit class the model card uses when ``qwen3_5`` isn't
    registered for CausalLM. ``output_hidden_states`` is passed per forward call, not
    here, since transformers 5.x doesn't reliably honor it as a from_pretrained kwarg.

    ``adapter_path`` is Volume-relative (``adapters/<run>/peft-causal-lm``). The adapter
    is applied unmerged: merging materializes full-size delta matrices (the tied-embedding
    delta alone is ~4GB) and OOMs the A10G with the 9B resident, whereas PEFT injects the
    LoRA-wrapped modules into the original module tree, so taking the base model back
    keeps the adapter applied on the fly and the module tree (``model.model.layers``)
    intact for hooks. A silently half-loaded adapter is base + noise, so every tensor on
    disk must have found a LoRA slot (the same guard as ``serving.persona_sampler``);
    ``load_report`` records the counts. Shared by :class:`ActivationExtractor` and
    ``assistant_axis.build``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        from transformers import Qwen3_5ForCausalLM

        model = Qwen3_5ForCausalLM.from_pretrained(model_id, **kwargs)

    load_report: dict = {"model_id": model_id, "adapter_path": adapter_path}
    if adapter_path:
        import os

        from peft import PeftModel
        from safetensors import safe_open

        full = os.path.join(VECTORS_DIR, adapter_path)
        if not os.path.isdir(full):
            raise FileNotFoundError(f"no exported adapter at Volume:{adapter_path}")
        with safe_open(os.path.join(full, "adapter_model.safetensors"), framework="pt") as f:
            n_tensors = len(list(f.keys()))
        model = PeftModel.from_pretrained(model, full).get_base_model()
        n_slots = sum(1 for name, _ in model.named_parameters() if ".lora_" in name)
        load_report.update({"adapter_tensors": n_tensors, "lora_slots": n_slots})
        if n_slots != n_tensors:
            raise RuntimeError(
                f"adapter {adapter_path}: {n_tensors} tensors on disk but {n_slots} LoRA "
                "parameters in the model -- layout mismatch, the model would be base + noise"
            )
        print(f"Applied LoRA adapter (unmerged) from {full}: {n_tensors} tensors, all slotted")
    model.eval()
    print(f"Loaded {model_id}: {model.config.num_hidden_layers} layers, hidden {model.config.hidden_size}")
    return model, tokenizer, load_report


@app.cls(
    image=vectors_image,
    # 9B bf16 (~18GB) + hidden states fits A10G (24GB) for forward-only passes.
    # Bigger cards (L40S/A100) need a payment method on this account.
    gpu="A10G",
    volumes={HF_CACHE_DIR: hf_cache_volume, VECTORS_DIR: vectors_volume},
    secrets=[hf_secret],
    timeout=1 * HOURS,
)
class ActivationExtractor:
    model_id: str = modal.parameter()
    # Volume-relative path to a PEFT LoRA adapter ("" = plain base model). The adapter
    # is merged into the weights after loading, so every extraction method sees an
    # ordinary CausalLM and the trained model's activations are read exactly like base.
    adapter_path: str = modal.parameter(default="")

    @modal.enter()
    def load(self):
        import torch

        self.torch = torch
        self.model, self.tokenizer, self.load_report = load_backbone(self.model_id, self.adapter_path)
        self.n_layers = self.model.config.num_hidden_layers

    @modal.method()
    def smoke(self) -> dict:
        """Load smoke-test: one forward pass, report hidden-state shape."""
        torch = self.torch
        enc = self.tokenizer("The patient felt a sudden jolt of dread.", return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        return {
            "model_id": self.model_id,
            "num_hidden_layers": int(self.n_layers),
            "n_hidden_states": len(hs),  # expect num_hidden_layers + 1
            "hidden_shape": list(hs[len(hs) // 2].shape),
        }

    @modal.method()
    def extract_message_activations(self, messages: list[str], config: dict, run_name: str) -> dict:
        """Extract each message's pre-response-token activation and save it to the Volume.

        Renders each message as a single user turn ending at the pre-response token
        (after the model's empty ``<think></think>`` block; ``build_chat_texts``,
        left-padded so it sits at index ``-1``) and takes ``hidden_states[L][:, -1, :]``
        -- the position the emotion vectors were validated to read at. Saves them to
        ``<run_name>/activations.safetensors`` (keys ``layer_<L>``). Projection onto the
        emotion vectors is a separate CPU step (``project_messages``) so it can be
        re-run after the vectors change without re-extracting.
        """
        import os

        import numpy as np
        from safetensors.numpy import save_file

        from . import readout as R

        torch = self.torch
        layers = config["layers"]
        batch_size = config.get("batch_size", 8)
        self.tokenizer.padding_side = "left"  # pre-response token at index -1
        self.tokenizer.truncation_side = "left"  # keep the assistant header + think block at the end

        acc: dict[int, list] = {L: [] for L in layers}
        for i in range(0, len(messages), batch_size):
            chat_texts = R.build_chat_texts(messages[i : i + batch_size], self.tokenizer)
            enc = self.tokenizer(
                chat_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).to("cuda")
            with torch.inference_mode():
                # Call the base transformer (no lm_head): we only need hidden states,
                # and computing full-vocab logits over long inputs OOMs the A10G.
                base = getattr(self.model, "model", self.model)
                hidden_states = base(**enc, output_hidden_states=True).hidden_states
            for L in layers:
                acc[L].append(hidden_states[L][:, -1, :].float().cpu().numpy().astype(np.float32))
            print(f"  extracted {min(i + batch_size, len(messages))}/{len(messages)}")
        tensors = {f"layer_{L}": np.concatenate(acc[L], axis=0) for L in layers}

        out_dir = os.path.join(VECTORS_DIR, run_name)
        os.makedirs(out_dir, exist_ok=True)
        save_file(tensors, os.path.join(out_dir, "activations.safetensors"))
        vectors_volume.commit()
        shape = list(tensors[f"layer_{layers[0]}"].shape)
        print(f"[{run_name}] saved pre-response activations {shape} at layers {layers}")
        return {"n_messages": len(messages), "layers": layers, "shape": shape}

    @modal.method()
    def extract_transcript_activations(self, rows: list[dict], config: dict, run_name: str) -> dict:
        """Read a model's own single-turn transcripts at two positions and save them (GPU).

        Each row is ``{"id", "prompt", "reply"}``: a user prompt and the reply this
        model gave it. The prompt is rendered exactly as at generation time
        (``training.tinker_sft.render_prompt``: chat template up to the assistant
        header, thinking off, so the empty think block is in the prompt) and the reply
        is tokenized separately and appended, the boundary ``build_datum`` trains at.
        One forward pass per transcript then yields, at every layer in
        ``config['layers']``:

        - ``user_mean`` -- the mean residual over the tokens of the user's message
          itself, the position at which the story vectors read the emotion the *user*
          expressed rather than the one the assistant is about to carry, and so the
          reading that should not depend on which persona is answering;
        - ``pre_response`` -- the residual at the last prompt token, the position the
          emotion vectors are validated at and the one ``extract_message_activations``
          reads (causal attention makes it identical whether or not the reply follows);
        - ``reply_mean`` -- the mean residual over the reply's own tokens (the
          end-of-turn token excluded), the on-policy read of what the model wrote.

        The user's own tokens are the ones whose whole character span lies inside the
        message as the chat template rendered it, so the turn header before it and the
        end-of-turn marker, assistant header and empty think block after it are left out
        (``_user_token_span``, whose rule ``meta.json`` also records under
        ``user_span_rule``).

        Alongside, at ``config['readout_layer']`` only, every reply token's projection
        onto the centered ``unit`` vectors of ``config['vectors_run']`` (float16) so the
        time course within a reply can be looked at without another forward pass; the
        mean of those per-token projections equals the projection of ``reply_mean`` by
        linearity, which ``project`` steps can use as a consistency check.

        Batches are packed by token count (``config['batch_tokens']``, default 8192)
        after sorting by length, then restored to input order. Prompts longer than
        ``config['max_prompt_tokens']`` (default 1024) and replies longer than
        ``config['max_reply_tokens']`` (default 1536) are rejected rather than
        truncated: a cut prompt moves the pre-response token and a cut reply changes
        the transcript being read. An empty reply gives NaN ``reply_mean`` rows and no
        token projections; ``n_reply_tokens`` records it.

        Saves ``<run_name>/pooled.safetensors`` (keys ``<position>/layer_<L>``, each
        ``[n_rows, hidden]``), ``<run_name>/token_projections.safetensors`` (``projections``
        ``[total_reply_tokens, n_emotions]`` float16 and ``offsets`` ``[n_rows + 1]``),
        and ``<run_name>/meta.json`` (row order, token counts, the load report, the
        vector names in projection-column order). Returns the meta.
        """
        import datetime
        import json
        import os

        import numpy as np
        from safetensors.numpy import save_file

        from name_that_feeling.training.tinker_sft import render_prompt

        torch = self.torch
        layers = config["layers"]
        readout_layer = config["readout_layer"]
        batch_tokens = int(config.get("batch_tokens", 8192))
        max_prompt = int(config.get("max_prompt_tokens", 1024))
        max_reply = int(config.get("max_reply_tokens", 1536))
        vectors_run = config["vectors_run"]

        names, clusters, U = _load_units(vectors_run, readout_layer)
        U_t = torch.tensor(U, device="cuda", dtype=torch.float32)
        hidden = U.shape[1]
        pad_id = self.tokenizer.pad_token_id

        seqs = []
        for r in rows:
            prompt_text = render_prompt(self.tokenizer, [{"role": "user", "content": r["prompt"]}], enable_thinking=False)
            p_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            r_ids = self.tokenizer.encode(r["reply"], add_special_tokens=False) if r["reply"] else []
            if len(p_ids) > max_prompt:
                raise ValueError(f"{r['id']}: prompt is {len(p_ids)} tokens, over max_prompt_tokens={max_prompt}")
            if len(r_ids) > max_reply:
                raise ValueError(f"{r['id']}: reply is {len(r_ids)} tokens, over max_reply_tokens={max_reply}")
            try:
                u0, u1 = _user_token_span(self.tokenizer, prompt_text, r["prompt"], p_ids)
            except ValueError as exc:
                raise ValueError(f"{r['id']}: {exc}") from exc
            seqs.append((p_ids, r_ids, u0, u1))

        # Pack by token budget over a length-sorted order; results are written back by index.
        order = sorted(range(len(seqs)), key=lambda i: -(len(seqs[i][0]) + len(seqs[i][1])))
        batches: list[list[int]] = []
        cur: list[int] = []
        cur_max = 0
        for i in order:
            n = len(seqs[i][0]) + len(seqs[i][1])
            new_max = max(cur_max, n)
            if cur and new_max * (len(cur) + 1) > batch_tokens:
                batches.append(cur)
                cur, cur_max = [], 0
                new_max = n
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)

        pooled = {pos: {L: np.full((len(seqs), hidden), np.nan, np.float32) for L in layers}
                  for pos in TRANSCRIPT_POSITIONS}
        tok_proj: list = [None] * len(seqs)
        base = getattr(self.model, "model", self.model)
        done = 0
        for batch in batches:
            maxlen = max(len(seqs[i][0]) + len(seqs[i][1]) for i in batch)
            ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for b, i in enumerate(batch):
                p, r = seqs[i][0], seqs[i][1]
                ids[b, : len(p) + len(r)] = torch.tensor(p + r, dtype=torch.long)
                mask[b, : len(p) + len(r)] = 1
            with torch.inference_mode():
                hidden_states = base(
                    input_ids=ids.cuda(), attention_mask=mask.cuda(), output_hidden_states=True
                ).hidden_states
            for b, i in enumerate(batch):
                p, r, u0, u1 = seqs[i]
                pre, start, end = len(p) - 1, len(p), len(p) + len(r)
                for L in layers:
                    h = hidden_states[L][b]
                    pooled["user_mean"][L][i] = h[u0:u1].float().mean(dim=0).cpu().numpy()
                    pooled["pre_response"][L][i] = h[pre].float().cpu().numpy()
                    if end > start:
                        pooled["reply_mean"][L][i] = h[start:end].float().mean(dim=0).cpu().numpy()
                h21 = hidden_states[readout_layer][b, start:end].float()
                tok_proj[i] = (h21 @ U_t.T).half().cpu().numpy() if end > start else np.zeros((0, len(names)), np.float16)
            del hidden_states  # 33 layers x batch x T x hidden: release before the next pass
            torch.cuda.empty_cache()
            done += len(batch)
            print(f"  extracted {done}/{len(seqs)} transcripts ({len(batch)} x {maxlen} tokens)")

        offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
        for i, tp in enumerate(tok_proj):
            offsets[i + 1] = offsets[i] + tp.shape[0]
        projections = np.concatenate(tok_proj, axis=0) if len(tok_proj) else np.zeros((0, len(names)), np.float16)

        meta = {
            "run_name": run_name,
            "load": self.load_report,
            "layers": layers,
            "readout_layer": readout_layer,
            "positions": list(TRANSCRIPT_POSITIONS),
            "user_span_rule": (
                "the user's message as the chat template rendered it (its last occurrence in the "
                "rendered prompt), mapped onto tokens by character offsets, keeping only tokens "
                "whose whole character span lies inside it; the turn header before the message and "
                "the end-of-turn marker, assistant header and empty think block after it are left "
                "out, as is any token straddling either boundary"
            ),
            "vectors_run": vectors_run,
            "emotions": names,
            "clusters": clusters,
            "batch_tokens": batch_tokens,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "rows": [
                {"id": r["id"], "n_prompt_tokens": len(p), "n_reply_tokens": len(rr),
                 "n_user_tokens": u1 - u0, "user_span": [u0, u1]}
                for r, (p, rr, u0, u1) in zip(rows, seqs)
            ],
        }
        out_dir = os.path.join(VECTORS_DIR, run_name)
        os.makedirs(out_dir, exist_ok=True)
        save_file({f"{pos}/layer_{L}": pooled[pos][L] for pos in pooled for L in layers},
                  os.path.join(out_dir, "pooled.safetensors"))
        save_file({"projections": projections, "offsets": offsets},
                  os.path.join(out_dir, "token_projections.safetensors"))
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        vectors_volume.commit()
        print(f"[{run_name}] saved {len(seqs)} transcripts x {len(layers)} layers x "
              f"{len(TRANSCRIPT_POSITIONS)} positions ({', '.join(TRANSCRIPT_POSITIONS)}); "
              f"{int(offsets[-1])} reply tokens projected onto {len(names)} vectors")
        return meta

    def _pool_layers(
        self, texts: list[str], layers: list[int], start_token: int, batch_size: int
    ) -> dict:
        """Mean-pool residual activations over positions >= start_token per story.

        Right-pads each batch; for a story shorter than ``start_token`` it falls
        back to averaging all real tokens. Returns ``{layer: [n_texts, hidden]}``.
        """
        import numpy as np

        torch = self.torch
        self.tokenizer.padding_side = "right"
        out: dict[int, list] = {L: [] for L in layers}
        n_short = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to("cuda")
            mask = enc["attention_mask"]  # [B, T], right-padded
            valid_len = mask.sum(dim=1)  # [B]
            n_short += int((valid_len <= start_token).sum().item())
            # Per-sequence start: drop first start_token tokens, unless too short.
            starts = torch.where(
                valid_len > start_token,
                torch.full_like(valid_len, start_token),
                torch.zeros_like(valid_len),
            )
            pos = torch.arange(mask.shape[1], device=mask.device)[None, :]  # [1, T]
            include = (mask.bool()) & (pos >= starts[:, None])  # [B, T]
            counts = include.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]

            with torch.inference_mode():
                # Call the base transformer (no lm_head): we only need hidden states,
                # and computing full-vocab logits over long inputs OOMs the A10G.
                base = getattr(self.model, "model", self.model)
                hidden_states = base(**enc, output_hidden_states=True).hidden_states
            for L in layers:
                hs = hidden_states[L].float()  # [B, T, H]
                pooled = (hs * include[:, :, None]).sum(dim=1) / counts  # [B, H]
                out[L].append(pooled.cpu().numpy().astype(np.float32))

        if n_short:
            print(f"  note: {n_short}/{len(texts)} stories shorter than start_token={start_token}")
        return {L: np.concatenate(out[L], axis=0) for L in layers}

    @modal.method()
    def cache_neutral(self, neutral_texts: list[str], config: dict, run_name: str) -> dict:
        """Pool the neutral baseline once and cache it per layer on the Volume.

        The neutral baseline is shared by every emotion, so we extract it a single
        time here and ``build_vector`` reuses it -- avoiding re-pooling 100 neutral
        stories for all ~50 emotions.
        """
        import os

        import numpy as np

        layers = config["layers"]
        start_token = config.get("start_token", 50)
        batch_size = config.get("batch_size", 8)

        pooled = self._pool_layers(neutral_texts, layers, start_token, batch_size)
        os.makedirs(os.path.dirname(_neutral_path(run_name, layers[0])), exist_ok=True)
        for L in layers:
            np.save(_neutral_path(run_name, L), pooled[L])
        vectors_volume.commit()
        print(f"[neutral] cached {len(neutral_texts)} stories at layers {layers}")
        return {"layers": layers, "n_neutral": len(neutral_texts)}

    @modal.method()
    def build_vector(
        self,
        emotion: str,
        cluster: str,
        emotion_texts: list[str],
        config: dict,
        run_name: str,
    ) -> dict:
        """Pool an emotion's stories and save its raw neutral-diff vector per layer.

        Stores ``raw = mean(emotion) - mean(neutral)`` (the ``mu_e - mu_neutral``
        primitive) at ``vectors/<cluster>/layer_<L>/<emotion>.safetensors``. The
        canonical centered ``unit`` is filled in afterwards by ``recenter_vectors``,
        which needs every emotion present to subtract the across-emotion mean. The
        neutral baseline is loaded from the ``cache_neutral`` cache. Idempotent:
        skips a layer whose vector already exists unless ``config['force']``.
        """
        import datetime
        import os

        import numpy as np

        from . import vectors as V

        layers = config["layers"]
        start_token = config.get("start_token", 50)
        batch_size = config.get("batch_size", 8)
        force = config.get("force", False)
        name = slugify(emotion)

        # Which layers still need building?
        todo = [
            L for L in layers
            if force or not os.path.exists(os.path.join(_vectors_dir(run_name, cluster, L), f"{name}.safetensors"))
        ]
        if not todo:
            print(f"[{cluster}/{emotion}] all layers present; skipping (use force to rebuild).")
            return {"emotion": emotion, "cluster": cluster, "layers": layers, "skipped": True}

        pooled_e = self._pool_layers(emotion_texts, todo, start_token, batch_size)

        paths = {}
        for L in todo:
            neutral_acts = np.load(
                _neutral_path(_neutral_source_run(config, run_name), L)
            )  # cached by cache_neutral
            raw = V.difference_of_means(pooled_e[L], neutral_acts)  # mu_e - mu_neutral
            metadata = {
                "emotion": emotion,
                "cluster": cluster,
                "model": self.model_id,
                "layer": L,
                "n_emotion": len(emotion_texts),
                "n_neutral": int(neutral_acts.shape[0]),
                "start_token": start_token,
                "baseline": "neutral_diff (raw); unit added by recenter_vectors",
                "paper": "Sofroniew et al. 2026 (arXiv:2604.07729)",
                "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            paths[L] = V.save_vector(
                _vectors_dir(run_name, cluster, L), name, raw, neutral_acts.mean(axis=0), metadata
            )
            print(f"[{cluster}/{emotion}] layer {L}: saved raw -> {paths[L]}")

        vectors_volume.commit()
        return {
            "emotion": emotion,
            "cluster": cluster,
            "layers": layers,
            "vector_paths": {str(k): v for k, v in paths.items()},
            "n_emotion": len(emotion_texts),
        }

    @modal.method()
    def tylenol_readout(self, emotion: str, cluster: str, config: dict, run_name: str) -> dict:
        """Validate the vector: project Tylenol-dose activations onto it.

        Success = projection increases monotonically with dose. Writes CSV + PNG
        to the Volume and returns the readout summary.
        """
        import csv
        import os

        import numpy as np

        from . import readout as R
        from . import vectors as V

        torch = self.torch
        layer = config.get("readout_layer") or config["layers"][0]
        doses = config.get("dose_sweep", R.DEFAULT_DOSES)
        name = slugify(emotion)

        st_path = os.path.join(_vectors_dir(run_name, cluster, layer), f"{name}.safetensors")
        tensors, _ = V.load_vector(st_path)
        unit = tensors["unit"]
        neutral_mean = tensors.get("neutral_mean")

        prompts = R.tylenol_prompts(doses)
        chat_texts = R.build_chat_texts(prompts, self.tokenizer)
        self.tokenizer.padding_side = "left"  # response-prep token at index -1
        enc = self.tokenizer(chat_texts, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            hidden_states = self.model(**enc, output_hidden_states=True).hidden_states
        acts = hidden_states[layer][:, -1, :].float().cpu().numpy().astype(np.float32)

        proj = R.project(acts, unit, neutral_mean)
        monotonic = R.check_monotonic(proj["raw"])
        rho = R.spearman_with_index(proj["raw"])

        # Persist CSV + PNG.
        out_dir = os.path.join(VECTORS_DIR, run_name, "readout")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"tylenol_{name}_layer{layer}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dose_mg", "projection_raw", "projection_centered"])
            centered = proj.get("centered", [None] * len(doses))
            for d, r, c in zip(doses, proj["raw"], centered):
                w.writerow([d, r, c])

        png_path = os.path.join(out_dir, f"tylenol_{name}_layer{layer}.png")
        _plot_readout(doses, proj["raw"], emotion, layer, monotonic, rho, png_path)

        vectors_volume.commit()
        result = {
            "emotion": emotion,
            "layer": layer,
            "doses": doses,
            "projection_raw": proj["raw"],
            "projection_centered": proj.get("centered"),
            "monotonic": monotonic,
            "spearman": rho,
            "csv_path": csv_path,
            "png_path": png_path,
        }
        print(f"[{emotion}] layer {layer}: monotonic={monotonic} spearman={rho:.3f}")
        return result


    @modal.method()
    def pool_story_set(
        self,
        texts: list[str],
        meta: list[dict],
        config: dict,
        run_name: str,
        set_name: str,
    ) -> dict:
        """Mean-pool a labeled story set and cache it on the Volume (GPU).

        Uses the *story* reader (mean-pool over positions >= ``start_token``), the same
        one ``build_vector`` pools its training stories with -- so a set cached here can
        be projected onto emotion vectors on exactly the terms they were built on. That
        is what makes a held-out story readout a fair test rather than a change of
        position convention. (``extract_message_activations`` is the other reader: a
        single pre-response token, for chat messages.)

        Saves ``<run_name>/pooled/<set_name>.safetensors`` (keys ``layer_<L>``) plus a
        ``<set_name>.meta.json`` sidecar carrying each row's labels in activation order,
        so scoring is a separate CPU step (``score_story_readout``) that can be re-run
        against different vector sets without paying for the forward passes again.
        Idempotent: skips a set already on the Volume unless ``config['force']``.
        """
        import json
        import os

        from safetensors.numpy import save_file

        layers = config["layers"]
        start_token = config.get("start_token", 50)
        batch_size = config.get("batch_size", 8)
        if len(meta) != len(texts):
            raise ValueError(f"meta has {len(meta)} rows but {len(texts)} texts were given")

        out_dir = os.path.join(VECTORS_DIR, run_name, "pooled")
        st_path = os.path.join(out_dir, f"{set_name}.safetensors")
        if os.path.exists(st_path) and not config.get("force", False):
            print(f"[pool:{set_name}] already on the Volume; skipping (use force to rebuild).")
            return {"set_name": set_name, "n_texts": len(texts), "skipped": True}

        pooled = self._pool_layers(texts, layers, start_token, batch_size)
        os.makedirs(out_dir, exist_ok=True)
        save_file({f"layer_{L}": pooled[L] for L in layers}, st_path)
        with open(os.path.join(out_dir, f"{set_name}.meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "set_name": set_name,
                    "model": self.model_id,
                    "layers": layers,
                    "start_token": start_token,
                    "position": "mean_pooled_story",
                    "n_texts": len(texts),
                    "rows": meta,
                },
                f,
                ensure_ascii=False,
            )
        vectors_volume.commit()
        shape = list(pooled[layers[0]].shape)
        print(f"[pool:{set_name}] pooled {shape} at layers {layers} -> {st_path}")
        return {"set_name": set_name, "n_texts": len(texts), "layers": layers, "shape": shape}


@app.function(
    image=vectors_image,
    volumes={VECTORS_DIR: vectors_volume},
    timeout=1 * HOURS,
)
def recenter_vectors(config: dict, run_name: str) -> dict:
    """Fill in the canonical centered ``unit`` for every emotion vector (CPU; no GPU).

    The all-emotion centering is a global op, so it runs once after the per-emotion
    ``build_vector`` map -- or standalone, to rebuild existing vectors in place. Per
    layer: load every ``raw`` (``mu_e - mu_neutral``), center across emotions (the
    neutral term cancels, giving ``mu_e - mu_bar``), project off the cached neutral
    PCs, L2-normalize, and re-save as ``unit``. ``raw``/``neutral_mean`` are preserved
    so the neutral-baseline (single-emotion / steering) uses stay available.
    """
    import datetime
    import glob
    import os

    import numpy as np

    from . import vectors as V

    layers = config["layers"]
    denoise = config.get("denoise", True)
    var_threshold = config.get("pca_var_threshold", 0.5)
    neutral_run = _neutral_source_run(config, run_name)
    # Where the recentered vectors land. Writing to a different run keeps the source
    # run's stored units intact, so several centering/denoise variants can be built
    # from one set of raws without any of them overwriting another.
    out_run = config.get("recenter_out_run") or run_name
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    summary: dict[str, int] = {}

    for L in layers:
        pattern = os.path.join(VECTORS_DIR, run_name, "vectors", "*", f"layer_{L}", "*.safetensors")
        paths = sorted(glob.glob(pattern))
        if not paths:
            continue
        raws, neutral_means, metas = [], [], []
        for p in paths:
            tensors, meta = V.load_vector(p)
            raws.append(tensors["raw"])
            neutral_means.append(tensors["neutral_mean"])
            metas.append(meta)
        centered = V.center_across_emotions(np.stack(raws))  # mu_e - mu_bar
        basis = (
            V.neutral_pc_basis(np.load(_neutral_path(neutral_run, L)), var_threshold)
            if denoise
            else np.zeros((0, centered.shape[1]), centered.dtype)
        )
        for i, p in enumerate(paths):
            unit = V.l2_normalize(V.project_out(centered[i], basis))
            meta = {
                **metas[i],
                "baseline": "all_emotions_mean",
                "centered": True,
                "denoise": denoise,
                "pca_var_threshold": var_threshold if denoise else None,
                "pca_n_components_used": int(basis.shape[0]) if denoise else 0,
                "neutral_run": neutral_run if denoise else None,
                "raws_from_run": run_name,
                "recentered": stamp,
            }
            cluster = os.path.basename(os.path.dirname(os.path.dirname(p)))
            out_dir = (
                os.path.dirname(p)
                if out_run == run_name
                else os.path.join(VECTORS_DIR, out_run, "vectors", cluster, f"layer_{L}")
            )
            V.save_vector(
                out_dir,
                os.path.splitext(os.path.basename(p))[0],
                raws[i],
                neutral_means[i],
                meta,
                unit=unit,
            )
        summary[str(L)] = len(paths)
        print(
            f"[recenter] layer {L}: {len(paths)} units -> {out_run} "
            f"(centered, denoise={denoise}, neutral={neutral_run if denoise else 'n/a'})"
        )

    vectors_volume.commit()
    print(f"[recenter] done: {summary}")
    return summary


@app.function(
    image=vectors_image,
    volumes={VECTORS_DIR: vectors_volume},
    timeout=1 * HOURS,
)
def project_messages(meta: list[dict], config: dict, run_name: str) -> dict:
    """Project cached pre-response activations onto every emotion vector -> readout.json (CPU).

    Reads ``<run_name>/activations.safetensors`` (from ``extract_message_activations``)
    and the centered emotion ``unit`` vectors, and writes the self-contained
    ``<run_name>/readout.json``: per message its original ``emotion``/``cluster`` (plus
    frame/split/axis from ``meta``, in activation-row order) and ``projections`` =
    ``{emotion: value}``. Re-runnable with no GPU -- refresh after the vectors change.
    """
    import datetime
    import glob
    import json
    import os

    import numpy as np
    from safetensors.numpy import load_file

    from . import vectors as V

    layers = config["layers"]
    readout_layer = config.get("readout_layer", layers[len(layers) // 2])
    vectors_run = config.get("vectors_run") or emotion_vectors_run(config["model_id"])
    # Output filename knob: one activation set can be projected onto several vector
    # runs (e.g. a trained model's activations onto its own vs the base's vectors --
    # same residual basis modulo the LoRA shift, which is exactly what exp-04 measures).
    readout_file = config.get("readout_file", "readout.json")
    out_dir = os.path.join(VECTORS_DIR, run_name)

    acts = load_file(os.path.join(out_dir, "activations.safetensors"))[f"layer_{readout_layer}"]
    glob_pat = os.path.join(VECTORS_DIR, vectors_run, "vectors", "*", f"layer_{readout_layer}", "*.safetensors")
    names, units = [], []
    for p in sorted(glob.glob(glob_pat)):
        tv, _ = V.load_vector(p)
        names.append(os.path.splitext(os.path.basename(p))[0])
        units.append(tv["unit"])
    U = np.stack(units, axis=0) if units else np.zeros((0, acts.shape[1]), np.float32)

    # Activations and vectors must come from the SAME model -- projecting one model's
    # activations onto another's vectors is meaningless (different residual basis) and
    # usually a hidden-dim mismatch. run.py derives both run names from one model slug,
    # so this only trips on a hand-edited/mismatched vectors_run; fail with a clear reason.
    if units and U.shape[1] != acts.shape[1]:
        raise ValueError(
            f"hidden-dim mismatch: activations are {acts.shape[1]}-d (run '{run_name}') but "
            f"vectors are {U.shape[1]}-d (run '{vectors_run}'). Activations and vectors must be "
            f"from the same model -- check that both run names share the same model slug."
        )

    proj = acts @ U.T  # [N, E]; units are already all-emotion-mean-centered
    rows = [
        {**m, "projections": {names[j]: float(proj[i, j]) for j in range(len(names))}}
        for i, m in enumerate(meta)
    ]
    missing = sorted({slugify(m["emotion"]) for m in meta} - set(names))
    sidecar = {
        "model": config.get("model_id"),
        "layers": layers,
        "readout_layer": readout_layer,
        "vectors_run": vectors_run,
        "position": "pre_response_token",
        "projection": "onto all-emotion-mean-centered unit vectors",
        "n_messages": len(meta),
        "n_emotion_vectors": len(names),
        "missing_emotion_vectors": missing,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "messages": rows,
    }
    with open(os.path.join(out_dir, readout_file), "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)
    vectors_volume.commit()
    print(f"[{run_name}] projected {len(meta)} messages x {len(names)} vectors -> {readout_file}; {len(missing)} missing")
    return {"n_messages": len(meta), "n_emotion_vectors": len(names), "missing": missing, "readout_file": readout_file}


@app.function(
    image=vectors_image,
    volumes={VECTORS_DIR: vectors_volume},
    timeout=1 * HOURS,
)
def score_story_readout(
    set_name: str,
    vectors_run: str,
    config: dict,
    run_name: str,
    out_name: str = "",
) -> dict:
    """Score a pooled story set against one run of emotion vectors (CPU).

    The story-level counterpart of ``project_messages``: it asks whether the vectors
    can pick out each story's emotion from the whole set, which is the question a
    held-out story set exists to answer. Every story's pooled activation is projected
    onto every centered ``unit``, each emotion's column is standardized across the set
    (see the comment at the projection for why an unstandardized argmax is wrong), and
    the emotion with the largest standardized projection is the prediction. Reported per story: the predicted emotion, the rank of the true one
    among all vectors (1 = best), and a ``z_margin`` -- the projection of the true
    emotion expressed in standard deviations of the spread of that story's own
    projections, a graded score that still separates runs once top-1 accuracy
    saturates or floors.

    ``set_name`` names a set cached by ``pool_story_set`` under ``run_name``, and
    ``vectors_run`` is any vector run built from the *same* model (the hidden-dim
    guard catches a mismatch). Keeping the two arguments separate is the point: one
    pooled set can be scored against several vector runs, which is what a
    cross-generator comparison is -- one test set read by vectors built from
    different story corpora.

    Writes ``<run_name>/readouts/<out_name or set_name>.json``.
    """
    import datetime
    import glob
    import json
    import os

    import numpy as np
    from safetensors.numpy import load_file

    from . import vectors as V

    readout_layer = config.get("readout_layer") or config["layers"][0]
    pooled_dir = os.path.join(VECTORS_DIR, run_name, "pooled")
    acts = load_file(os.path.join(pooled_dir, f"{set_name}.safetensors"))[f"layer_{readout_layer}"]
    with open(os.path.join(pooled_dir, f"{set_name}.meta.json"), encoding="utf-8") as f:
        sidecar = json.load(f)
    rows_meta = sidecar["rows"]

    pattern = os.path.join(
        VECTORS_DIR, vectors_run, "vectors", "*", f"layer_{readout_layer}", "*.safetensors"
    )
    names, clusters, units = [], {}, []
    for p in sorted(glob.glob(pattern)):
        tensors, meta = V.load_vector(p)
        if "unit" not in tensors:
            raise KeyError(f"{p} has no centered unit yet -- run recenter_vectors on {vectors_run}.")
        name = os.path.splitext(os.path.basename(p))[0]
        names.append(name)
        clusters[name] = meta.get("cluster") or os.path.basename(os.path.dirname(os.path.dirname(p)))
        units.append(tensors["unit"])
    if not names:
        raise FileNotFoundError(f"no vectors under {pattern}")
    U = np.stack(units, axis=0)
    if U.shape[1] != acts.shape[1]:
        raise ValueError(
            f"hidden-dim mismatch: pooled set {set_name!r} is {acts.shape[1]}-d but the vectors "
            f"in {vectors_run!r} are {U.shape[1]}-d. A readout is only meaningful when both come "
            f"from the same model."
        )

    raw_proj = (acts @ U.T).astype(np.float64)  # [N, E]
    # The units are centered across emotions, but the *activations* are not centered at
    # all, so ``x . u_e`` carries the term ``x_common . u_e`` -- a constant per emotion
    # that has nothing to do with the story and varies across emotions by more than the
    # story-to-story signal does (measured: offset std 4.4 vs within-emotion spread 2.4 at
    # layer 21). An argmax over the raw columns therefore mostly picks the emotion with
    # the largest offset. Standardizing each emotion across the set being scored removes
    # that offset, which is exactly what the tag pipeline (``generation.sft``) does before
    # it ever ranks anything, so this is the readout that corresponds to how the vectors
    # are actually used. Raw projections are kept in the output for anyone who needs the
    # unstandardized reading of a single emotion, where the offset is harmless.
    col_mean, col_std = raw_proj.mean(axis=0, keepdims=True), raw_proj.std(axis=0, keepdims=True)
    proj = (raw_proj - col_mean) / np.where(col_std > 0, col_std, 1.0)
    index = {n: i for i, n in enumerate(names)}
    # Rank every column per story, 1 = largest standardized projection.
    ranks = np.empty_like(proj, dtype=np.int32)
    order = np.argsort(-proj, axis=1, kind="stable")
    np.put_along_axis(ranks, order, np.arange(1, proj.shape[1] + 1, dtype=np.int32)[None, :], axis=1)
    mean, std = proj.mean(axis=1), proj.std(axis=1)
    predicted = [names[j] for j in proj.argmax(axis=1)]
    raw_predicted = [names[j] for j in raw_proj.argmax(axis=1)]

    cluster_sizes = {c: sum(1 for n in names if clusters[n] == c) for c in set(clusters.values())}
    rows, unscored = [], []
    for i, m in enumerate(rows_meta):
        true = slugify(m["emotion"])
        row = {**m, "predicted": predicted[i], "predicted_cluster": clusters[predicted[i]]}
        if true in index:
            j = index[true]
            row.update(
                {
                    "true_cluster": clusters[true],
                    "rank": int(ranks[i, j]),
                    "correct": predicted[i] == true,
                    "cluster_correct": clusters[predicted[i]] == clusters[true],
                    "z_margin": float((proj[i, j] - mean[i]) / std[i]) if std[i] else 0.0,
                }
            )
        else:
            unscored.append(true)
        row["raw_predicted"] = raw_predicted[i]
        row["projections"] = {names[j]: round(float(raw_proj[i, j]), 6) for j in range(len(names))}
        rows.append(row)

    scored = [r for r in rows if "rank" in r]
    if not scored:
        raise ValueError(
            f"none of the {len(rows)} stories have an emotion with a vector in {vectors_run!r} "
            f"(first missing: {sorted(set(unscored))[:5]})"
        )
    ranks_arr = np.array([r["rank"] for r in scored])
    summary = {
        "n_scored": len(scored),
        "n_unscored": len(rows) - len(scored),
        "n_vectors": len(names),
        "top1": float(np.mean([r["correct"] for r in scored])),
        "top5": float(np.mean(ranks_arr <= 5)),
        "cluster_top1": float(np.mean([r["cluster_correct"] for r in scored])),
        "mean_rank": float(ranks_arr.mean()),
        "median_rank": float(np.median(ranks_arr)),
        "mean_z_margin": float(np.mean([r["z_margin"] for r in scored])),
        # The uncentered argmax, kept only so the size of the offset artifact is visible.
        "top1_raw_argmax": float(np.mean([raw_predicted[i] == slugify(rows_meta[i]["emotion"])
                                          for i in range(len(rows)) if "rank" in rows[i]])),
        "chance_top1": 1.0 / len(names),
        # Chance on the cluster metric is the share of all vectors the true cluster holds,
        # averaged over stories -- clusters differ in size, so it is not 1/n_clusters.
        "chance_cluster_top1": float(
            np.mean([cluster_sizes[r["true_cluster"]] / len(names) for r in scored])
        ),
    }

    by_emotion: dict[str, list] = {}
    for r in scored:
        by_emotion.setdefault(slugify(r["emotion"]), []).append(r)
    per_emotion = [
        {
            "emotion": e,
            "cluster": clusters.get(e),
            "n": len(rs),
            "top1": float(np.mean([x["correct"] for x in rs])),
            "cluster_top1": float(np.mean([x["cluster_correct"] for x in rs])),
            "mean_rank": float(np.mean([x["rank"] for x in rs])),
            "mean_z_margin": float(np.mean([x["z_margin"] for x in rs])),
        }
        for e, rs in sorted(by_emotion.items())
    ]

    result = {
        "set_name": set_name,
        "vectors_run": vectors_run,
        "activations_run": run_name,
        "model": config.get("model_id"),
        "readout_layer": readout_layer,
        "position": sidecar.get("position"),
        "start_token": sidecar.get("start_token"),
        "projection": "onto all-emotion-mean-centered unit vectors",
        "ranking": "per-emotion standardized across this set (offset removed); projections stored raw",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "summary": summary,
        "per_emotion": per_emotion,
        "stories": rows,
    }
    out_dir = os.path.join(VECTORS_DIR, run_name, "readouts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{out_name or set_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    vectors_volume.commit()
    print(
        f"[score] {set_name} x {vectors_run}: top1={summary['top1']:.3f} "
        f"cluster={summary['cluster_top1']:.3f} mean_rank={summary['mean_rank']:.1f} "
        f"z={summary['mean_z_margin']:.2f} -> {out_path}"
    )
    return {"set_name": set_name, "vectors_run": vectors_run, "out": out_path, **summary}


@app.function(
    image=vectors_image,
    volumes={VECTORS_DIR: vectors_volume},
    timeout=1 * HOURS,
)
def stack_unit_vectors(vectors_run: str, layer: int) -> dict:
    """One run's centered ``unit`` vectors at one layer as a single matrix (CPU).

    Returns ``{"vectors_run", "layer", "emotions", "clusters", "units"}`` with ``units``
    a float32 ``[n_emotions, hidden]`` numpy array in ``emotions`` order -- the compact
    form a local projection step needs, so a laptop can re-score cached activations
    without pulling 171 files or running anything on Modal.
    """
    names, clusters, U = _load_units(vectors_run, layer)
    return {"vectors_run": vectors_run, "layer": layer, "emotions": names, "clusters": clusters, "units": U}


@app.function(
    image=vectors_image,
    volumes={VECTORS_DIR: vectors_volume},
    timeout=1 * HOURS,
)
def project_pooled_set(
    run_name: str,
    set_name: str,
    vectors_run: str,
    layer: int,
    axes: dict | None = None,
) -> dict:
    """One pooled story set's readout in compact form, returned rather than written (CPU).

    The small-return counterpart of ``score_story_readout``: that one scores a labeled
    set and writes a per-story JSON to the Volume, this one hands the projection matrix
    back, so a local step can hold every text's readout for a few megabytes
    (``[n_texts, n_emotions]`` float32) instead of pulling the pooled activations
    (hidden-wide, ~24x larger) or parsing a per-story JSON per model. Useful whenever a
    set is read by many models and the comparison happens locally.

    ``axes`` maps a name to a unit direction in residual space -- the fitted
    valence/arousal/dominance axes of ``affect_axes``, say. Each is dotted with the
    pooled activations directly, so a returned coordinate is exactly what
    ``affect_axes.project_affect`` would give on those activations, with nothing
    standing in between. ``norms`` carries each activation's L2 norm, the scale a
    projection is read against.

    Returns ``{"run_name", "set_name", "vectors_run", "layer", "emotions", "clusters",
    "n_texts", "projections", "axes", "norms", "rows"}``, with ``rows`` the set's stored
    per-text labels in activation order.
    """
    import json
    import os

    import numpy as np
    from safetensors.numpy import load_file

    # A CPU container is pooled across calls, so one started before the GPU committed the
    # set holds a stale mount and the file it is asked for is simply absent (measured
    # 2026-09-09: three of ten models failed this way while reading sets that were on the
    # Volume). Reload before looking.
    vectors_volume.reload()
    pooled_dir = os.path.join(VECTORS_DIR, run_name, "pooled")
    acts = load_file(os.path.join(pooled_dir, f"{set_name}.safetensors"))[f"layer_{layer}"].astype(np.float64)
    with open(os.path.join(pooled_dir, f"{set_name}.meta.json"), encoding="utf-8") as f:
        sidecar = json.load(f)
    names, clusters, U = _load_units(vectors_run, layer)
    if U.shape[1] != acts.shape[1]:
        raise ValueError(
            f"hidden-dim mismatch: pooled set {set_name!r} is {acts.shape[1]}-d but the vectors in "
            f"{vectors_run!r} are {U.shape[1]}-d. A readout is only meaningful when both come from "
            f"the same model."
        )
    out_axes = {
        name: (acts @ np.asarray(direction, dtype=np.float64)).astype(np.float32)
        for name, direction in (axes or {}).items()
    }
    print(
        f"[project] {run_name}/{set_name}: {acts.shape[0]} texts x {len(names)} vectors at layer {layer}"
        + (f", axes {sorted(out_axes)}" if out_axes else "")
    )
    return {
        "run_name": run_name,
        "set_name": set_name,
        "vectors_run": vectors_run,
        "layer": layer,
        "emotions": names,
        "clusters": clusters,
        "n_texts": int(acts.shape[0]),
        "projections": (acts @ U.astype(np.float64).T).astype(np.float32),
        "axes": out_axes,
        "norms": np.linalg.norm(acts, axis=1).astype(np.float32),
        "rows": sidecar["rows"],
    }


@app.function(
    image=vectors_image,
    volumes={VECTORS_DIR: vectors_volume},
    timeout=1 * HOURS,
)
def export_vector_bundle(vectors_run: str, layer: int) -> dict:
    """One run's vectors at one layer in every stored form, for local geometry work (CPU).

    Returns ``emotions`` (sorted slugs), ``clusters``, ``units`` (``[E, hidden]``, the
    stored centered-denoised-normalized vectors), ``raw`` (``mu_e - mu_neutral``),
    ``paper_vectors`` (``project_out(center(raw))``: the paper's own vectors, centered
    across emotions and denoised but NOT normalized, rebuilt here from the raws with the
    run's cached neutral basis), the ``neutral_basis`` (``[k, hidden]`` orthonormal rows)
    and its provenance, ``neutral_mean`` (``mu_neutral``, the pooled neutral baseline) and
    ``story_grand_mean`` (``mu_neutral + mean_e(raw_e)``, the across-emotion mean the paper
    vectors are centered on, so an activation minus it is on the vectors' own origin), and
    ``min_reconstruction_cos``: the smallest cosine between a
    rebuilt normalized vector and the stored unit (1.0 when the rebuild reproduces the
    run exactly). Sofroniew et al. never normalize, so the paper-faithful PCA over the
    vector set runs on ``paper_vectors``; ``units`` give the equal-weight variant.
    """
    import glob
    import os

    import numpy as np

    from . import vectors as V

    pattern = os.path.join(VECTORS_DIR, vectors_run, "vectors", "*", f"layer_{layer}", "*.safetensors")
    names, clusters, units, raws, metas, neutral_means = [], {}, [], [], [], []
    for p in sorted(glob.glob(pattern)):
        tensors, meta = V.load_vector(p)
        if "unit" not in tensors:
            raise KeyError(f"{p} has no centered unit yet -- run recenter_vectors on {vectors_run}.")
        name = os.path.splitext(os.path.basename(p))[0]
        names.append(name)
        clusters[name] = meta.get("cluster") or os.path.basename(os.path.dirname(os.path.dirname(p)))
        units.append(tensors["unit"])
        raws.append(tensors["raw"])
        neutral_means.append(tensors["neutral_mean"])
        metas.append(meta)
    if not names:
        raise FileNotFoundError(f"no vectors under {pattern}")
    U, R = np.stack(units).astype(np.float64), np.stack(raws).astype(np.float64)
    # mu_neutral is stored with every vector and is the same for all of them; with it the
    # story grand mean the vectors were centered on is mu_neutral + mean_e(raw_e).
    neutral_mean = np.stack(neutral_means).astype(np.float64).mean(axis=0)
    meta0 = metas[0]
    denoise = bool(meta0.get("denoise", True))
    neutral_run = meta0.get("neutral_run")
    threshold = meta0.get("pca_var_threshold")
    centered = V.center_across_emotions(R)
    if denoise:
        basis = V.neutral_pc_basis(np.load(_neutral_path(neutral_run, layer)), threshold)
        paper = np.stack([V.project_out(c, basis) for c in centered])
    else:
        basis = np.zeros((0, R.shape[1]), np.float64)
        paper = centered
    rebuilt = paper / np.linalg.norm(paper, axis=1, keepdims=True)
    min_cos = float((rebuilt * U).sum(axis=1).min())
    print(f"[bundle] {len(names)} vectors at layer {layer} from {vectors_run}; neutral basis "
          f"{basis.shape[0]} PCs from {neutral_run}; min reconstruction cos {min_cos:.6f}")
    return {
        "vectors_run": vectors_run,
        "layer": layer,
        "emotions": names,
        "clusters": clusters,
        "units": U.astype(np.float32),
        "raw": R.astype(np.float32),
        "paper_vectors": paper.astype(np.float32),
        "neutral_basis": basis.astype(np.float32),
        "neutral_mean": neutral_mean.astype(np.float32),
        "story_grand_mean": (neutral_mean + R.mean(axis=0)).astype(np.float32),
        "neutral_run": neutral_run,
        "pca_var_threshold": threshold,
        "denoise": denoise,
        "min_reconstruction_cos": min_cos,
    }


@app.function(
    image=vectors_image,
    volumes={VECTORS_DIR: vectors_volume},
    timeout=1 * HOURS,
)
def compare_vector_runs(run_a: str, run_b: str, layer: int, out_run: str) -> dict:
    """Per-emotion geometry comparison of two vector runs at one layer (CPU).

    For every emotion present in both runs: cosine similarity of the centered ``unit``
    vectors, cosine of the ``raw`` (neutral-diff) vectors, and the raw-norm ratio
    (b/a). Writes ``<out_run>/vector_shift.json`` with one row per emotion (plus its
    cluster) -- the notebook-side input for the trained-vs-base geometry analysis.
    """
    import datetime
    import glob
    import json
    import os

    import numpy as np

    from . import vectors as V

    def _load_run(run: str) -> dict[str, tuple]:
        out = {}
        for p in sorted(glob.glob(os.path.join(VECTORS_DIR, run, "vectors", "*", f"layer_{layer}", "*.safetensors"))):
            tensors, meta = V.load_vector(p)
            name = os.path.splitext(os.path.basename(p))[0]
            out[name] = (tensors, meta.get("cluster") or os.path.basename(os.path.dirname(os.path.dirname(p))))
        return out

    def _cos(a, b) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na and nb else 0.0

    va, vb = _load_run(run_a), _load_run(run_b)
    common = sorted(set(va) & set(vb))
    rows = []
    for name in common:
        (ta, cluster), (tb, _) = va[name], vb[name]
        rows.append(
            {
                "emotion": name,
                "cluster": cluster,
                "cosine_unit": _cos(ta["unit"], tb["unit"]),
                "cosine_raw": _cos(ta["raw"], tb["raw"]),
                "norm_ratio_raw": float(np.linalg.norm(tb["raw"]) / (np.linalg.norm(ta["raw"]) or 1.0)),
            }
        )
    result = {
        "run_a": run_a,
        "run_b": run_b,
        "layer": layer,
        "n_common": len(common),
        "only_in_a": sorted(set(va) - set(vb)),
        "only_in_b": sorted(set(vb) - set(va)),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "emotions": rows,
    }
    out_path = os.path.join(VECTORS_DIR, out_run, "vector_shift.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    vectors_volume.commit()
    print(f"[compare] {len(common)} emotions at layer {layer}: {run_a} vs {run_b} -> {out_path}")
    return {"n_common": len(common), "out": out_path}


@app.function(
    image=vectors_image,
    volumes={VECTORS_DIR: vectors_volume},
    timeout=1 * HOURS,
)
def emotion_similarity_matrix(run_name: str, layer: int) -> dict:
    """Emotion x emotion cosine-similarity matrix of one run's ``unit`` vectors (CPU).

    The centered ``unit`` vectors are L2-normalized, so the Gram matrix of the stacked
    set is the full pairwise cosine matrix. Writes
    ``<run_name>/similarity/layer_<L>.json`` with the ordered emotion slugs, each slug's
    cluster, and the matrix (rows/cols in ``emotions`` order) -- the artifact behind the
    distance-based tag metrics (``evals/similarity.py``).
    """
    import datetime
    import glob
    import json
    import os

    import numpy as np

    from . import vectors as V

    pattern = os.path.join(VECTORS_DIR, run_name, "vectors", "*", f"layer_{layer}", "*.safetensors")
    names, clusters, units = [], {}, []
    for p in sorted(glob.glob(pattern)):
        tensors, meta = V.load_vector(p)
        name = os.path.splitext(os.path.basename(p))[0]
        names.append(name)
        clusters[name] = meta.get("cluster") or os.path.basename(os.path.dirname(os.path.dirname(p)))
        units.append(tensors["unit"])
    if not names:
        raise FileNotFoundError(f"no vectors under {pattern}")

    U = np.stack(units).astype(np.float64)
    U /= np.linalg.norm(U, axis=1, keepdims=True)  # units are stored normalized; re-normalize defensively
    matrix = np.round(U @ U.T, 6)

    result = {
        "vectors_run": run_name,
        "layer": layer,
        "n_emotions": len(names),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "emotions": names,
        "clusters": clusters,
        "matrix": matrix.tolist(),
    }
    out_path = os.path.join(VECTORS_DIR, run_name, "similarity", f"layer_{layer}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    vectors_volume.commit()
    print(f"[similarity] {len(names)} emotions at layer {layer} -> {out_path}")
    return {"n_emotions": len(names), "layer": layer, "out": out_path}


def _plot_readout(doses, values, emotion, layer, monotonic, rho, png_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(doses, values, marker="o", color="crimson")
    ax.set_xscale("log")
    ax.set_xlabel("Tylenol dose (mg, log scale)")
    ax.set_ylabel(f"'{emotion}' vector activation")
    status = "PASS" if monotonic else "see spearman"
    ax.set_title(f"Tylenol readout — '{emotion}' (layer {layer})\nmonotonic={monotonic} [{status}], ρ={rho:.2f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
