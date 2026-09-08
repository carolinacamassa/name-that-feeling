"""The official five-step pipeline as Modal functions, plus projection of fine-tunes.

Every step reads and writes one build's namespace on the vectors Volume,
``assistant-axis/<slug>/<build>/`` (:func:`build_run`), in the official repo's own file
formats so its notebooks and helpers apply unchanged:

    responses/<role>.jsonl      step 1  one line per (system prompt, question) reply
    activations/<role>.pt       step 2  {key: (n_layers, hidden) bf16}, the mean residual
                                        over the reply's tokens at every layer
    scores/<role>.json          step 3  {key: 0..3}, the judge's role-adherence score
    vectors/<role>.pt           step 4  the role's mean over score-3 replies (default: all)
    axis.pt                     step 5  {"axis": (n_layers, hidden), "metadata": {...}}
    axis_report.json                    the paper's checks (PC1 alignment, the default's
                                        position, the role ordering) and provenance
    projections/<model>/...             a model's transcripts projected on the axis

Keys are ``pos_p<prompt index>_q<question index>``, as in the official scripts.

**Layer indexing.** The official extractor hooks the *output* of decoder layer ``i``,
so ``axis[i]`` is the post-MLP residual after layer ``i``, which in transformers'
``output_hidden_states`` numbering is ``hidden_states[i + 1]``. The emotion vectors of
this repo are indexed the other way (``layer_21`` = ``hidden_states[21]`` = the output
of decoder layer 20 = ``axis[20]``). Anything that puts the two on one footing has to
shift by one; :func:`project_transcripts` avoids the question by re-extracting with the
official code.

What is borrowed and what is ours: generation formatting and sampling
(``assistant_axis.generation``), activation extraction (``pipeline/2_activations.py``,
imported by path), the judge (``assistant_axis.judge``), the vector and axis arithmetic
(``pipeline/4_vectors.py``, ``assistant_axis.axis``) are the authors'. Ours are the
Modal plumbing, the A10G-sized vLLM engine (the official one profiles Qwen3.5's vision
tower and OOMs on a 24GB card), the model loader (``emotion_vectors.extraction
.load_backbone``, which also applies exported LoRA adapters), the judge's endpoint
(OpenRouter, so the paper's ``gpt-4.1-mini`` runs on this project's keys), and the
report.
"""

import modal

from name_that_feeling.infra import (
    HF_CACHE_DIR,
    HOURS,
    VECTORS_DIR,
    hf_cache_volume,
    hf_secret,
    vectors_volume,
)

from . import PIPELINE_DIR, QUESTIONS_FILE, ROLES_DIR, REPO_COMMIT, app, axis_image, axis_vllm_image

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------- helpers (container-side)

def _load_pipeline_script(name: str):
    """Import one of the official ``pipeline/N_*.py`` scripts as a module, by path.

    Their names aren't importable identifiers (``2_activations``), and they are not part
    of the installed package, but their functions are the reference implementation, so
    they are loaded from the in-image copy of the submodule rather than duplicated.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"official_{name}", f"{PIPELINE_DIR}/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_dir(run: str, *parts: str) -> str:
    import os

    return os.path.join(VECTORS_DIR, run, *parts)


def _list_role_names() -> list[str]:
    import os

    return sorted(os.path.splitext(f)[0] for f in os.listdir(ROLES_DIR) if f.endswith(".json"))


def _load_scores(path: str) -> dict:
    import json
    import os

    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- step 1: responses

@app.cls(
    image=axis_vllm_image,
    gpu="A10G",
    volumes={HF_CACHE_DIR: hf_cache_volume, VECTORS_DIR: vectors_volume},
    secrets=[hf_secret],
    timeout=12 * HOURS,
)
class RoleResponses:
    """Step 1: the model answers the extraction questions under every role's system prompts.

    Wraps the official ``RoleResponseGenerator`` (its prompt formatting, the
    ``{model_name}`` placeholder, ``enable_thinking=False`` for Qwen, and the paper's
    sampling: temperature 0.7, top-p 0.9, 512 new tokens) around a vLLM engine sized for
    the A10G. Output files are per role and skipped when present, so a run is resumable
    and several containers can split the role list.
    """

    model_id: str = modal.parameter()
    run: str = modal.parameter()  # build namespace, e.g. assistant-axis/qwen3.5-9b/full
    question_count: int = modal.parameter(default=240)
    max_model_len: int = modal.parameter(default=2048)
    # Engine sizing (``generation.engine`` in the config). The A10G needs eager mode and
    # a 2,048-token prefill chunk to fit; a 48 GB card runs with CUDA graphs and a
    # bigger chunk, at 2-3x the throughput. The GPU type itself is set with
    # ``RoleResponses.with_options(gpu=...)`` by the driver.
    enforce_eager: bool = modal.parameter(default=True)
    max_num_batched_tokens: int = modal.parameter(default=2048)
    # Sampling (temperature, top_p, max_tokens) is passed to ``generate`` per call:
    # modal.parameter() takes no floats.

    @modal.enter()
    def load(self):
        from assistant_axis.generation import RoleResponseGenerator, VLLMGenerator

        enforce_eager, max_num_batched_tokens = self.enforce_eager, self.max_num_batched_tokens

        class SizedVLLMGenerator(VLLMGenerator):
            """The official generator with the engine sized for the card it runs on.

            Qwen3.5 resolves as a multimodal architecture, so vLLM profiles memory with
            dummy image inputs by default and OOMs during KV-cache sizing on a 24GB card;
            text-only limits and, on the A10G, chunked prefill and eager mode make it fit
            (``generation.completions.VLLMGenerator`` established these settings).
            """

            def load(self):
                if self.llm is not None:
                    return
                from vllm import LLM, SamplingParams

                self.llm = LLM(
                    model=self.model_name,
                    dtype="bfloat16",
                    max_model_len=self.max_model_len,
                    gpu_memory_utilization=0.95,
                    max_num_batched_tokens=max_num_batched_tokens,
                    enforce_eager=enforce_eager,
                    limit_mm_per_prompt={"image": 0, "video": 0},
                    trust_remote_code=True,
                )
                self.sampling_params = SamplingParams(
                    temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p
                )

        out_dir = _run_dir(self.run, "responses")
        self.rrg = RoleResponseGenerator(
            model_name=self.model_id,
            roles_dir=ROLES_DIR,
            output_dir=out_dir,
            questions_file=QUESTIONS_FILE,
            max_model_len=self.max_model_len,
            question_count=self.question_count,
            short_name="Qwen",  # the official value for Qwen models; get_config can't infer it for Qwen3.5
        )
        self.rrg.generator = SizedVLLMGenerator(model_name=self.model_id, max_model_len=self.max_model_len)
        self.rrg.generator.load()

    @modal.method()
    def generate(self, roles: list[str], sampling: dict) -> dict:
        """Answer every prompt x question for ``roles`` (skipping roles already on the Volume).

        ``sampling`` carries ``temperature``, ``top_p`` and ``max_tokens`` (the official
        defaults are 0.7, 0.9 and 512).
        """
        import os
        import time

        from vllm import SamplingParams

        self.rrg.generator.sampling_params = SamplingParams(
            temperature=sampling["temperature"], top_p=sampling["top_p"], max_tokens=sampling["max_tokens"]
        )
        vectors_volume.reload()
        done, skipped = [], []
        for role in roles:
            if self.rrg.should_skip_role(role):
                skipped.append(role)
                continue
            t0 = time.time()
            role_data = self.rrg.load_role(os.path.join(ROLES_DIR, f"{role}.json"))
            responses = self.rrg.generate_role_responses(role, role_data)
            if not responses:
                raise RuntimeError(f"{role}: the official generator returned no responses")
            self.rrg.save_responses(role, responses)
            vectors_volume.commit()
            done.append(role)
            print(f"[{self.run}] {role}: {len(responses)} replies in {time.time() - t0:.0f}s", flush=True)
        return {"done": done, "skipped": skipped}


# ---------------------------------------------------------------- steps 2 + projection

@app.cls(
    image=axis_image,
    gpu="A10G",
    volumes={HF_CACHE_DIR: hf_cache_volume, VECTORS_DIR: vectors_volume},
    secrets=[hf_secret],
    timeout=12 * HOURS,
)
class ResponseActivations:
    """Steps 2 and the projection: mean response activations under the official hooks.

    The model is loaded by this repo's loader (so an exported LoRA adapter can ride
    along) and handed to the official ``ProbingModel.from_existing``; extraction itself
    is ``pipeline/2_activations.py``'s ``process_role`` / ``extract_activations_batch``.
    """

    model_id: str = modal.parameter()
    adapter_path: str = modal.parameter(default="")  # Volume-relative PEFT adapter, "" = base

    @modal.enter()
    def load(self):
        from assistant_axis.internals import ProbingModel

        from name_that_feeling.emotion_vectors.extraction import load_backbone

        model, tokenizer, self.load_report = load_backbone(self.model_id, self.adapter_path)
        tokenizer.padding_side = "left"  # as ProbingModel.__init__ sets it

        # transformers 5 returns a BatchEncoding from apply_chat_template(tokenize=True)
        # by default (return_dict=True); the official span builder, written against 4.x,
        # expects the bare id list. return_dict=False is accepted by both majors.
        original_apply = tokenizer.apply_chat_template

        def apply_chat_template(*args, **kwargs):
            if kwargs.get("tokenize") and "return_dict" not in kwargs:
                kwargs["return_dict"] = False
            return original_apply(*args, **kwargs)

        tokenizer.apply_chat_template = apply_chat_template
        self.pm = ProbingModel.from_existing(model, tokenizer, model_name=self.model_id)
        self.n_layers = len(self.pm.get_layers())
        self.official = _load_pipeline_script("2_activations")
        print(f"ProbingModel ready: {self.n_layers} hooked layers, load={self.load_report}")

    @modal.method()
    def smoke(self) -> dict:
        """One conversation through the official extractor; returns shapes and the rendered text."""
        conv = [{"role": "user", "content": "What principles should guide human action?"},
                {"role": "assistant", "content": "Honesty first, then care for others."}]
        acts = self.official.extract_activations_batch(
            self.pm, [conv], list(range(self.n_layers)), batch_size=1, max_length=2048
        )
        rendered = self.pm.tokenizer.apply_chat_template(conv, tokenize=False, enable_thinking=False)
        return {
            "n_layers": self.n_layers,
            "shape": list(acts[0].shape),
            "dtype": str(acts[0].dtype),
            "rendered": rendered,
            "load": self.load_report,
        }

    @modal.method()
    def extract_roles(self, run: str, roles: list[str], batch_size: int = 8, max_length: int = 2048) -> dict:
        """Step 2 for ``roles``: ``responses/<role>.jsonl`` -> ``activations/<role>.pt`` (all layers)."""
        import time
        from pathlib import Path

        vectors_volume.reload()
        out_dir = Path(_run_dir(run, "activations"))
        out_dir.mkdir(parents=True, exist_ok=True)
        layers = list(range(self.n_layers))
        done, skipped = [], []
        for role in roles:
            if (out_dir / f"{role}.pt").exists():
                skipped.append(role)
                continue
            src = Path(_run_dir(run, "responses", f"{role}.jsonl"))
            if not src.exists():
                raise FileNotFoundError(f"{run}: no responses for {role} yet (step 1)")
            t0 = time.time()
            ok = self.official.process_role(self.pm, src, out_dir, layers, batch_size, max_length, False)
            if not ok:
                raise RuntimeError(f"{role}: official process_role reported failure")
            vectors_volume.commit()
            done.append(role)
            print(f"[{run}] {role}: activations in {time.time() - t0:.0f}s", flush=True)
        return {"done": done, "skipped": skipped, "load": self.load_report}

    @modal.method()
    def project_transcripts(
        self,
        rows: list[dict],
        axis_run: str,
        out_run: str,
        batch_size: int = 8,
        max_length: int = 2048,
    ) -> dict:
        """Project this model's own single-turn transcripts onto a saved axis.

        ``rows`` are ``{"id", "prompt", "reply"}`` (no system prompt: the uninstructed
        setting every persona model is sampled in, and the official ``default`` role's
        empty prompt). Each transcript is read the official way (the mean residual over
        the reply's tokens at every hooked layer) and projected onto ``<axis_run>/axis.pt``
        with the official ``project_batch`` (unit-normalized axis, so a projection is in
        residual-norm units and comparable across models read at the same layer).

        Writes ``<out_run>/activations.pt`` ({id: (n_layers, hidden) bf16}, so other
        directions can be read later without a GPU) and ``<out_run>/projections.json``
        (per row and layer the projection and the activation's norm, so the cosine is
        projection over norm, plus the target-layer summary). Returns the JSON document.
        """
        import datetime
        import json
        import os
        import time

        import torch
        from assistant_axis.axis import load_axis, project_batch

        vectors_volume.reload()
        axis_path = _run_dir(axis_run, "axis.pt")
        if not os.path.exists(axis_path):
            raise FileNotFoundError(f"no axis at Volume:{axis_run}/axis.pt (run the build first)")
        axis = load_axis(axis_path)
        with open(_run_dir(axis_run, "axis_report.json"), encoding="utf-8") as f:
            report = json.load(f)
        target_layer = int(report["target_layer"])
        if axis.shape[0] != self.n_layers:
            raise ValueError(f"axis has {axis.shape[0]} layers, this model {self.n_layers}")

        conversations = [
            [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["reply"]}]
            for r in rows
        ]
        t0 = time.time()
        acts = self.official.extract_activations_batch(
            self.pm, conversations, list(range(self.n_layers)), batch_size=batch_size, max_length=max_length
        )
        missing = [r["id"] for r, a in zip(rows, acts) if a is None]
        kept = [(r, a) for r, a in zip(rows, acts) if a is not None]
        stacked = torch.stack([a for _, a in kept])  # (n, n_layers, hidden) bf16
        proj = torch.stack(
            [project_batch(stacked, axis, layer=L) for L in range(self.n_layers)], dim=1
        )  # (n, n_layers) float32
        # The activation's own norm per layer, so cosine (projection / norm) can be read
        # off alongside the official projection: a between-model comparison needs both,
        # since a model whose residual is simply smaller projects lower on every direction.
        norms = stacked.float().norm(dim=2)  # (n, n_layers)
        at_target = proj[:, target_layer]
        cos_target = at_target / norms[:, target_layer]
        doc = {
            "model_id": self.model_id,
            "adapter_path": self.adapter_path,
            "load": self.load_report,
            "axis_run": axis_run,
            "target_layer": target_layer,
            "n_rows": len(rows),
            "missing": missing,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "seconds": round(time.time() - t0, 1),
            "summary": {
                "mean": float(at_target.mean()),
                "std": float(at_target.std(unbiased=False)),
                "percentiles": {
                    str(p): float(torch.quantile(at_target, p / 100)) for p in (5, 25, 50, 75, 95)
                },
                "cosine_mean": float(cos_target.mean()),
                "norm_mean": float(norms[:, target_layer].mean()),
            },
            "rows": [
                {
                    "id": r["id"],
                    "projection": float(at_target[i]),
                    "cosine": float(cos_target[i]),
                    "norm": float(norms[i, target_layer]),
                    "per_layer": [round(float(x), 4) for x in proj[i]],
                    "per_layer_norm": [round(float(x), 3) for x in norms[i]],
                }
                for i, (r, _) in enumerate(kept)
            ],
        }
        out_dir = _run_dir(out_run)
        os.makedirs(out_dir, exist_ok=True)
        torch.save({r["id"]: a for r, a in kept}, os.path.join(out_dir, "activations.pt"))
        with open(os.path.join(out_dir, "projections.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        vectors_volume.commit()
        print(f"[{out_run}] {len(kept)} transcripts projected; mean {doc['summary']['mean']:.3f} "
              f"at layer {target_layer} in {doc['seconds']}s")
        return doc


# ---------------------------------------------------------------- step 3: the judge

def _judge_client(api_key: str, request_overrides: dict | None):
    """An OpenRouter client whose ``chat.completions.create`` applies per-call overrides.

    The official ``call_judge_single`` hardcodes ``max_completion_tokens`` (10) and
    ``temperature`` (1) for a non-reasoning judge. A reasoning judge such as
    ``gpt-5-nano`` spends that budget on thinking and returns nothing, so
    ``request_overrides`` (from ``judge['request_overrides']``) replaces or adds request
    fields after the official call assembles them, e.g. ``{"max_completion_tokens": 64,
    "extra_body": {"reasoning": {"effort": "minimal"}}}``. The prompt and the parsing
    stay the official ones.
    """
    import openai

    client = openai.AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    if request_overrides:
        original_create = client.chat.completions.create

        async def create(**kwargs):
            kwargs.update(request_overrides)
            return await original_create(**kwargs)

        client.chat.completions.create = create
    return client


@app.function(image=axis_image, volumes={VECTORS_DIR: vectors_volume}, timeout=12 * HOURS)
def judge_roles(run: str, roles: list[str], judge: dict, api_key: str, scores_dir: str = "scores",
                sample: dict | None = None) -> dict:
    """Step 3: score every reply's role adherence 0-3 with an LLM judge.

    The official ``judge`` module builds the prompt from the role file's ``eval_prompt``
    and parses the score; the client is pointed at OpenRouter so ``judge['model']`` (an
    OpenRouter model id; the paper's is ``openai/gpt-4.1-mini``) runs on this project's
    key, which is passed in from the launcher's ``.env`` rather than stored on Modal.
    ``judge['request_overrides']`` adapts the request for a reasoning judge (see
    :func:`_judge_client`). ``default`` has no ``eval_prompt`` and is skipped, as the
    official script does. Scores are saved per role under ``<run>/<scores_dir>/`` after
    each role, and replies already scored are not re-sent.

    ``sample`` (``{"per_role": n, "seed": s}``) scores only a seeded random subset of
    each role's replies, for calibrating one judge against another under a separate
    ``scores_dir``; the axis build reads ``scores/`` only.
    """
    import asyncio
    import json
    import os
    import random
    import time

    from assistant_axis.judge import RateLimiter, call_judge_batch, parse_judge_score

    vectors_volume.reload()
    client = _judge_client(api_key, judge.get("request_overrides"))
    out_dir = _run_dir(run, scores_dir)
    os.makedirs(out_dir, exist_ok=True)
    result = {"scored": {}, "unparsed": {}, "skipped": []}

    for role in roles:
        with open(os.path.join(ROLES_DIR, f"{role}.json"), encoding="utf-8") as f:
            template = json.load(f).get("eval_prompt", "")
        if not template:
            result["skipped"].append(role)
            continue
        with open(_run_dir(run, "responses", f"{role}.jsonl"), encoding="utf-8") as f:
            responses = [json.loads(line) for line in f if line.strip()]
        if sample:
            responses = random.Random(f"{sample['seed']}:{role}").sample(responses, min(sample["per_role"], len(responses)))
        out_path = os.path.join(out_dir, f"{role}.json")
        scores = _load_scores(out_path)
        prompts, keys = [], []
        for resp in responses:
            key = f"{resp['label']}_p{resp['prompt_index']}_q{resp['question_index']}"
            if key in scores:
                continue
            answer = next((m["content"] for m in resp["conversation"] if m["role"] == "assistant"), "")
            prompts.append(template.format(question=resp["question"], answer=answer))
            keys.append(key)
        if not prompts:
            result["skipped"].append(role)
            continue
        # The official client returns None for any API error (a rate-limited call
        # included) and has no retry, so the unscored keys are re-sent in further
        # passes, with a pause, until they parse or the passes run out.
        n_sent = len(prompts)
        for attempt in range(judge.get("passes", 4)):
            if not prompts:
                break
            if attempt:
                time.sleep(judge.get("retry_pause_seconds", 20))
            # A fresh limiter per asyncio.run: the official RateLimiter creates its
            # asyncio.Lock at construction, and a lock outlives the event loop that first
            # used it only as "bound to a different event loop" errors, which the official
            # batch call swallows into None for every reply of the batch.
            limiter = RateLimiter(judge.get("requests_per_second", 20))
            texts = asyncio.run(call_judge_batch(
                client=client, prompts=prompts, model=judge["model"], max_tokens=judge.get("max_tokens", 10),
                rate_limiter=limiter, batch_size=judge.get("batch_size", 50),
            ))
            retry_prompts, retry_keys = [], []
            for key, prompt, text in zip(keys, prompts, texts):
                score = parse_judge_score(text) if text else None
                if score is None:
                    retry_prompts.append(prompt)
                    retry_keys.append(key)
                else:
                    scores[key] = score
            if retry_prompts:
                print(f"[{run}] {role}: pass {attempt + 1}: {len(retry_prompts)} of {len(prompts)} unscored, retrying", flush=True)
            prompts, keys = retry_prompts, retry_keys
        n_unparsed = len(prompts)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)
        vectors_volume.commit()
        result["scored"][role] = n_sent - n_unparsed
        if n_unparsed:
            result["unparsed"][role] = n_unparsed
        print(f"[{run}] {role}: {n_sent - n_unparsed} scored, {n_unparsed} unparsed "
              f"(score-3 share {sum(v == 3 for v in scores.values()) / max(len(scores), 1):.2f})", flush=True)
    return result


@app.function(image=axis_image, volumes={VECTORS_DIR: vectors_volume}, timeout=30 * 60)
def compare_scores(run: str, dir_a: str, dir_b: str) -> dict:
    """Agreement between two judges' score files (``<run>/<dir_a>/`` vs ``<run>/<dir_b>/``).

    Over the replies both scored: exact-score agreement, agreement on the decision the
    axis uses (score 3 or not), the 4x4 confusion matrix (rows: ``dir_a``), each judge's
    score-3 share, and per-role disagreement counts, so a cheap judge's fitness for the
    paper's role can be read off before the full scoring.
    """
    import json
    import os

    vectors_volume.reload()
    a_dir, b_dir = _run_dir(run, dir_a), _run_dir(run, dir_b)
    roles = sorted(set(f[:-5] for f in os.listdir(a_dir) if f.endswith(".json"))
                   & set(f[:-5] for f in os.listdir(b_dir) if f.endswith(".json")))
    confusion = [[0] * 4 for _ in range(4)]
    n = exact = decision = 0
    per_role = {}
    for role in roles:
        a, b = _load_scores(os.path.join(a_dir, f"{role}.json")), _load_scores(os.path.join(b_dir, f"{role}.json"))
        keys = sorted(set(a) & set(b))
        dis = 0
        for k in keys:
            confusion[a[k]][b[k]] += 1
            n += 1
            exact += a[k] == b[k]
            same = (a[k] == 3) == (b[k] == 3)
            decision += same
            dis += not same
        per_role[role] = {"n": len(keys), "decision_disagreements": dis}
    share_a = sum(confusion[3]) / max(n, 1)
    share_b = sum(row[3] for row in confusion) / max(n, 1)
    return {
        "run": run, "a": dir_a, "b": dir_b, "n_roles": len(roles), "n_replies": n,
        "exact_agreement": exact / max(n, 1), "score3_decision_agreement": decision / max(n, 1),
        "score3_share_a": share_a, "score3_share_b": share_b,
        "confusion_rows_a_cols_b": confusion,
        "roles_most_disagreement": sorted(per_role.items(), key=lambda kv: -kv[1]["decision_disagreements"])[:15],
    }


# ---------------------------------------------------------------- steps 4 + 5

@app.function(image=axis_image, volumes={VECTORS_DIR: vectors_volume}, timeout=2 * HOURS)
def build_axis(run: str, min_count: int, target_layer: int | None, build_config: dict) -> dict:
    """Steps 4 and 5, then the paper's checks.

    Per-role vectors are the official ``4_vectors.py`` functions (mean over score-3
    replies, ``min_count`` of them required; ``default`` is the mean of all its replies),
    the axis is the official ``compute_axis`` (mean default minus mean of the role
    vectors) saved with ``save_axis``. The report then runs the paper's own validation
    at ``target_layer`` (the middle layer when None, the official default): PCA over the
    role vectors, the cosine of the axis with PC1 (the paper: above 0.71 at the middle
    layer), where the default lands on PC1 relative to the roles' range (the paper:
    within 0.03 of the extreme), and the roles ordered by cosine with the axis (the
    paper's near end: generalist, consultant, analyst; far end: hermit, fool, zealot,
    eldritch, whale). It also records the default's per-reply projections (the
    "normal range" activation capping uses). Writes ``vectors/<role>.pt``, ``axis.pt``,
    ``axis_report.json``; returns the report.
    """
    import datetime
    import json
    import os

    import numpy as np
    import torch
    from assistant_axis.axis import compute_axis, project_batch, save_axis

    vectors_volume.reload()
    official = _load_pipeline_script("4_vectors")
    act_dir, score_dir, vec_dir = (_run_dir(run, d) for d in ("activations", "scores", "vectors"))
    os.makedirs(vec_dir, exist_ok=True)

    # Step 4 -----------------------------------------------------------------
    role_vectors: dict[str, torch.Tensor] = {}
    default_vector = None
    default_acts = None
    dropped: dict[str, int] = {}
    n_score3: dict[str, int] = {}
    n_unscored: dict[str, int] = {}
    for fname in sorted(os.listdir(act_dir)):
        if not fname.endswith(".pt"):
            continue
        role = fname[:-3]
        acts = official.load_activations(os.path.join(act_dir, fname))
        if role == "default":
            default_vector = official.compute_mean_vector(acts)
            default_acts = torch.stack(list(acts.values()))
            torch.save({"vector": default_vector, "type": "mean", "role": role}, os.path.join(vec_dir, fname))
            continue
        score_path = os.path.join(score_dir, f"{role}.json")
        if not os.path.exists(score_path):
            raise FileNotFoundError(f"{run}: activations but no scores for {role} (step 3)")
        scores = official.load_scores(score_path)
        n_score3[role] = sum(1 for k in acts if scores.get(k) == 3)
        n_unscored[role] = sum(1 for k in acts if k not in scores)
        try:
            vec = official.compute_pos_3_vector(acts, scores, min_count)
        except ValueError:
            dropped[role] = n_score3[role]
            continue
        role_vectors[role] = vec
        torch.save({"vector": vec, "type": "pos_3", "role": role}, os.path.join(vec_dir, fname))
    if default_vector is None:
        raise FileNotFoundError(f"{run}: no activations/default.pt -- the default role is the axis's near end")
    if not role_vectors:
        raise RuntimeError(f"{run}: no role reached min_count={min_count} score-3 replies")

    # Step 5 -----------------------------------------------------------------
    roles = sorted(role_vectors)
    R = torch.stack([role_vectors[r] for r in roles]).float()  # (n_roles, n_layers, hidden)
    axis = compute_axis(R, default_vector.float()[None])  # (n_layers, hidden)
    n_layers = axis.shape[0]
    L = n_layers // 2 if target_layer is None else int(target_layer)

    # The paper's checks at the target layer ---------------------------------
    X = R[:, L, :].numpy().astype(np.float64)  # (n_roles, hidden)
    mean = X.mean(axis=0)
    ax = axis[L].numpy().astype(np.float64)
    unit = ax / (np.linalg.norm(ax) + 1e-8)
    if len(roles) >= 3:
        _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
        var = s**2 / (s**2).sum()
        pc1 = vt[0]
        cos_pc1 = float(pc1 @ ax / (np.linalg.norm(pc1) * np.linalg.norm(ax) + 1e-8))
        k = min(5, vt.shape[0])
        role_pc = (X - mean) @ vt[:k].T  # (n_roles, k)
        default_pc = (default_vector[L].float().numpy() - mean) @ vt[:k].T
        lo, hi = role_pc.min(axis=0), role_pc.max(axis=0)
        default_position = ((default_pc - lo) / (hi - lo + 1e-12)).tolist()  # 0/1 = the roles' extremes
        if cos_pc1 < 0:  # PC sign is arbitrary; report the default's position with the axis end at 1
            default_position[0] = 1 - default_position[0]
        # Per-layer alignment of the axis with the roles' first component (the paper:
        # above 0.60 at every layer), and the roles' coordinates on the top components at
        # the target layer, signed so that the Assistant end of PC1 is positive.
        cos_pc1_per_layer = []
        for l in range(n_layers):
            Xl = R[:, l, :].numpy().astype(np.float64)
            _, _, vtl = np.linalg.svd(Xl - Xl.mean(axis=0), full_matrices=False)
            al = axis[l].numpy().astype(np.float64)
            cos_pc1_per_layer.append(round(abs(float(vtl[0] @ al / (np.linalg.norm(al) + 1e-8))), 4))
        sign = -1.0 if cos_pc1 < 0 else 1.0
        role_pcs = {r: [round(float(sign * role_pc[i, 0]), 4)] + [round(float(x), 4) for x in role_pc[i, 1:k]]
                    for i, r in enumerate(roles)}
        default_pcs = [round(float(sign * default_pc[0]), 4)] + [round(float(x), 4) for x in default_pc[1:k]]
        pca = {
            "cos_axis_pc1": cos_pc1,
            "cos_axis_pc1_per_layer": cos_pc1_per_layer,
            "pca_variance_explained_top10": [round(float(v), 4) for v in var[:10]],
            "n_components_for_70pct": int(np.searchsorted(np.cumsum(var), 0.70) + 1),
            "default_position_on_pc1_to_5": [round(float(p), 3) for p in default_position],
            "default_pc_coordinates": default_pcs,
        }
    else:  # a smoke build: too few roles for a persona space
        cos_pc1, default_position = float("nan"), [float("nan")]
        role_pcs = {}
        pca = {"cos_axis_pc1": None, "note": f"PCA needs >= 3 role vectors, have {len(roles)}"}
    role_cos = {r: float(X[i] @ unit / (np.linalg.norm(X[i]) + 1e-8)) for i, r in enumerate(roles)}
    role_proj = {r: float((X[i] - mean) @ unit) for i, r in enumerate(roles)}
    ordered = sorted(roles, key=lambda r: role_proj[r])
    default_proj = project_batch(default_acts.float(), axis, layer=L)
    default_norm = default_acts.float()[:, L, :].norm(dim=1)
    default_cos = default_proj / default_norm
    role_vec_proj = float((default_vector[L].float().numpy() - mean) @ unit)

    save_axis(axis, os.path.join(vec_dir, "..", "axis.pt"), metadata={
        "run": run, "repo_commit": REPO_COMMIT, "n_roles": len(roles), "target_layer": L,
        "min_count": min_count, "layer_indexing": "axis[i] = output of decoder layer i = hidden_states[i + 1]",
    })
    report = {
        "run": run,
        "repo_commit": REPO_COMMIT,
        "build_config": build_config,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_layers": n_layers,
        "hidden": int(axis.shape[1]),
        "target_layer": L,
        "layer_indexing": "axis[i] = output of decoder layer i = hidden_states[i + 1]",
        "min_count": min_count,
        "n_roles_with_vector": len(roles),
        "roles_dropped_below_min_count": dropped,
        "n_score3_per_role": n_score3,
        "n_unscored_per_role": {r: n for r, n in n_unscored.items() if n},
        "n_replies_unscored_total": int(sum(n_unscored.values())),
        "n_default_replies": int(default_acts.shape[0]),
        "axis_norm_per_layer": [round(float(x), 3) for x in axis.norm(dim=1)],
        "checks": {
            **pca,
            "default_projection_of_mean_vector": role_vec_proj,
            "default_reply_projection_percentiles": {
                str(p): float(torch.quantile(default_proj, p / 100)) for p in (5, 25, 50, 75, 95)
            },
            "default_reply_cosine_percentiles": {
                str(p): float(torch.quantile(default_cos, p / 100)) for p in (5, 25, 50, 75, 95)
            },
            "default_reply_norm_mean": float(default_norm.mean()),
            "roles_nearest_assistant": ordered[-15:][::-1],
            "roles_farthest_from_assistant": ordered[:15],
        },
        "role_cosine_with_axis": role_cos,
        "role_projection_centered": role_proj,
        "role_pc_coordinates": role_pcs,  # PC1..PC5 at the target layer, PC1 signed toward the Assistant
    }
    with open(_run_dir(run, "axis_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    vectors_volume.commit()
    print(f"[{run}] axis over {len(roles)} roles ({len(dropped)} dropped); layer {L}: cos(axis, PC1) = "
          f"{cos_pc1:.3f}, default at {default_position[0]:.3f} of PC1's range")
    return report


# ---------------------------------------------------------------- the driver (CPU)

@app.function(image=axis_image, volumes={VECTORS_DIR: vectors_volume}, timeout=24 * HOURS)
def drive(stage: str, run: str, roles: list[str], containers: int, cfg: dict, api_key: str = "") -> dict:
    """Fan one build stage out over ``containers`` workers and wait, from inside Modal.

    ``modal run --detach`` keeps only the last triggered function alive once the local
    launcher is gone, so a launcher that spawned four GPU calls would lose three of them
    to a dropped connection. Making the fan-out itself a Modal function means the one
    detached call is this driver, which outlives the laptop and returns the merged
    per-worker results. ``stage`` is ``generate``, ``extract`` or ``judge``; ``cfg`` is
    the experiment config; ``api_key`` is needed for ``judge`` only.
    """
    n = max(1, min(containers, len(roles)))
    chunks = [roles[i::n] for i in range(n)]
    if stage == "generate":
        g = cfg["generation"]
        eng = g.get("engine", {})
        worker = RoleResponses.with_options(gpu=g.get("gpu", "A10G"))(
            model_id=cfg["model_id"], run=run, question_count=g["question_count"], max_model_len=g["max_model_len"],
            enforce_eager=bool(eng.get("enforce_eager", True)),
            max_num_batched_tokens=int(eng.get("max_num_batched_tokens", 2048)),
        )
        sampling = {k: g[k] for k in ("temperature", "top_p", "max_tokens")}
        calls = [worker.generate.spawn(chunk, sampling) for chunk in chunks]
    elif stage == "extract":
        e = cfg["extraction"]
        worker = ResponseActivations.with_options(gpu=e.get("gpu", "A10G"))(model_id=cfg["model_id"])
        calls = [worker.extract_roles.spawn(run, chunk, e["batch_size"], e["max_length"]) for chunk in chunks]
    elif stage == "judge":
        calls = [judge_roles.spawn(run, chunk, cfg["judge"], api_key) for chunk in chunks]
    else:
        raise ValueError(f"unknown stage {stage!r}")
    print(f"[{run}] {stage}: {len(roles)} roles across {len(calls)} workers", flush=True)
    results = [c.get() for c in calls]
    merged: dict = {"stage": stage, "n_roles": len(roles), "workers": len(calls)}
    for key in ("done", "skipped"):
        if any(key in r for r in results):
            merged[key] = [x for r in results for x in r.get(key, [])]
    if stage == "judge":
        merged["scored"] = sum(sum(r["scored"].values()) for r in results)
        merged["unparsed"] = sum(sum(r["unparsed"].values()) for r in results)
        merged["skipped"] = [x for r in results for x in r["skipped"]]
    print(f"[{run}] {stage} finished: { {k: (len(v) if isinstance(v, list) else v) for k, v in merged.items()} }", flush=True)
    return merged


# ---------------------------------------------------------------- bookkeeping (CPU)

@app.function(image=axis_image, volumes={VECTORS_DIR: vectors_volume}, timeout=10 * 60)
def status(run: str) -> dict:
    """How far each step has got for one build: file counts per stage and the axis, if any."""
    import os

    vectors_volume.reload()
    roles = _list_role_names()
    out = {"run": run, "n_roles": len(roles)}
    for stage, ext in (("responses", ".jsonl"), ("activations", ".pt"), ("scores", ".json"), ("vectors", ".pt")):
        d = _run_dir(run, stage)
        have = sorted(f[: -len(ext)] for f in os.listdir(d) if f.endswith(ext)) if os.path.isdir(d) else []
        out[stage] = {"n": len(have), "missing": [r for r in roles if r not in have and not (stage == "scores" and r == "default")]}
    out["axis"] = os.path.exists(_run_dir(run, "axis.pt"))
    return out
