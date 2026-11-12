"""Apply exported LoRA adapters to a loaded base model, each with its own coefficient.

A model on the vectors Volume is ``W + sum_i c_i * dW_i``: the base weights plus the
updates of one or more exported adapters (``adapters/<run>/peft-causal-lm``, written by
``training.tinker_export``), each scaled by a coefficient. ``c = 1`` on one adapter is the
persona checkpoint as every 07 read loads it; ``c = -1`` is its negation; two adapters at
``c = 1`` are a pair sum; ``c = 0`` is the base model (item A3 of
``docs/affect-geometry-todo.md``, the arithmetic sweep).

Nothing is materialized: every adapter is injected unmerged by PEFT, all of them are
made active at once, and the coefficient goes into the layer's scaling factor, which
PEFT multiplies the low-rank product by in the forward pass (``scaling[name] =
c * alpha / r``, so the baked LoRA scale of the export is kept and ``c`` multiplies the
trained update). A merge is therefore read through exactly the module path the single-
adapter loaders (``emotion_vectors.extraction.load_backbone``, ``serving.persona_sampler``)
already trust, and with one adapter at ``c = 1`` it is the same computation. Merging into
the weights would materialize full-size deltas and OOM the A10G with the 9B resident.

The guard every loader keeps: each adapter's tensors on disk must all have found a LoRA
slot in the model, because a half-loaded adapter is base plus noise.

Modal-side only (needs peft, safetensors and the Volume mounted at ``VECTORS_DIR``);
pure helpers for the merge spec itself (``parse_terms``, ``format_terms``) run anywhere.
"""

import json
import os

from name_that_feeling.infra import VECTORS_DIR


def parse_terms(spec: str) -> list[tuple[str, float]]:
    """A merge spec, as a Modal parameter carries it: a JSON list of
    ``{"adapter": <Volume-relative path>, "coefficient": <float>}`` -> ``[(path, c), ...]``.
    An empty spec is the plain base model."""
    if not spec:
        return []
    terms = json.loads(spec)
    out = []
    for t in terms:
        path, c = t["adapter"], float(t["coefficient"])
        if not path:
            raise ValueError(f"merge term without an adapter path: {t!r}")
        out.append((path, c))
    return out


def format_terms(terms: list[tuple[str, float]]) -> str:
    """The inverse of :func:`parse_terms` (sorted keys, so equal merges give equal strings)."""
    return json.dumps([{"adapter": p, "coefficient": float(c)} for p, c in terms], sort_keys=True)


def apply_adapters(model, terms: list[tuple[str, float]], volume_dir: str = VECTORS_DIR) -> tuple:
    """Inject every adapter in ``terms`` into ``model`` unmerged and set its coefficient.

    Returns ``(model, load_report)``: the transformers model with the LoRA layers injected
    (PEFT's ``get_base_model()``, so the module tree stays intact for hooks), and a report
    listing each term with its tensor and slot counts plus the number of layers scaled.
    With no terms the model is returned untouched.
    """
    report: dict = {"terms": [], "n_lora_layers": 0}
    if not terms:
        return model, report

    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from safetensors import safe_open

    peft_model = None
    names: list[str] = []
    for k, (path, c) in enumerate(terms):
        full = os.path.join(volume_dir, path)
        if not os.path.isdir(full):
            raise FileNotFoundError(f"no exported adapter at Volume:{path}")
        with safe_open(os.path.join(full, "adapter_model.safetensors"), framework="pt") as f:
            n_tensors = len(list(f.keys()))
        name = f"term{k}"
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, full, adapter_name=name)
        else:
            peft_model.load_adapter(full, adapter_name=name)
        n_slots = sum(1 for n, _ in peft_model.named_parameters() if ".lora_" in n and f".{name}." in n)
        if n_slots != n_tensors:
            raise RuntimeError(
                f"adapter {path}: {n_tensors} tensors on disk but {n_slots} LoRA parameters in "
                "the model -- layout mismatch, the model would be base + noise"
            )
        names.append(name)
        report["terms"].append({"adapter": path, "coefficient": float(c), "tensors": n_tensors, "slots": n_slots})

    assert peft_model is not None
    # Every term active at once, each at its coefficient times the export's own alpha/r.
    peft_model.base_model.set_adapter(names)
    n_layers = 0
    for module in peft_model.modules():
        if not isinstance(module, LoraLayer):
            continue
        n_layers += 1
        for name, (_, c) in zip(names, terms):
            if name not in module.scaling:
                raise RuntimeError(f"adapter {name} has no slot in {type(module).__name__}; the terms do not share a layout")
            module.set_scale(name, c)
            expected = c * module.lora_alpha[name] / module.r[name]
            if abs(module.scaling[name] - expected) > 1e-9:
                raise RuntimeError(f"scaling of {name} is {module.scaling[name]}, expected {expected}")
        if list(module.active_adapters) != names:
            raise RuntimeError(f"active adapters {module.active_adapters} != {names}")
    report["n_lora_layers"] = n_layers
    print(
        "Applied "
        + " + ".join(f"{c:+g} * {t['adapter']}" for t, (_, c) in zip(report["terms"], terms))
        + f" (unmerged, {n_layers} LoRA layers, all tensors slotted)"
    )
    return peft_model.get_base_model(), report
