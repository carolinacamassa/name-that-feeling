"""02-message-activations: pre-response-token probe readout for the experiment-00 messages.

Reads the ~600 user messages from experiment 00, runs Qwen3.5-9B on Modal, extracts
each message's residual activation at the **pre-response token** (after the model's
empty <think></think> block), and projects it onto every emotion vector -> a self-
contained readout on the ``name-that-feeling-emotion-vectors`` Volume:

- ``02-message-activations/activations.safetensors`` -- raw pre-response activations (per layer).
- ``02-message-activations/readout.json`` -- per message: its original emotion + cluster
  (and frame/split/axis) plus the projection onto each emotion vector.

    uv run modal run experiments/02-message-activations/run.py::readout
    uv run modal run experiments/02-message-activations/run.py::readout --model allenai/OLMo-2-1124-7B
    uv run modal volume get name-that-feeling-emotion-vectors /02-message-activations/qwen3.5-9b ./out

``--model`` targets a registered model (default: config's ``model_id``). Activations land
at ``02-message-activations/<slug>`` and are projected onto that *same* model's vectors at
``01-emotion-vectors/<slug>`` -- the shared slug is what keeps the two from ever mixing.
"""

import json
from pathlib import Path

import yaml

from name_that_feeling.emotion_vectors import app
from name_that_feeling.emotion_vectors.extraction import ActivationExtractor, project_messages
from name_that_feeling.emotion_vectors.models import inject_model, run_name_for

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
EXPERIMENT = "02-message-activations"

META_KEYS = ("id", "emotion", "cluster", "frame", "split", "eval_axis", "message")


def load_config(model: str = "") -> dict:
    """Read config.yaml, stamp in the target model, and derive the paired vectors run.

    ``vectors_run`` is resolved to ``<vectors_experiment>/<slug>`` for the *same* model,
    so a readout can only ever project onto vectors from the model it extracted with.
    """
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    inject_model(cfg, model)
    cfg["vectors_run"] = run_name_for(cfg["vectors_experiment"], cfg["model_id"])
    return cfg


def run_name(cfg: dict) -> str:
    return run_name_for(EXPERIMENT, cfg["model_id"])


def _load_messages(cfg: dict) -> tuple[list[str], list[dict]]:
    rows = json.loads((REPO_ROOT / cfg["messages_file"]).read_text(encoding="utf-8"))
    return [r["message"] for r in rows], [{k: r.get(k) for k in META_KEYS} for r in rows]


@app.local_entrypoint()
def extract(model: str = "") -> None:
    """GPU: extract each message's pre-response-token activation -> activations.safetensors."""
    cfg = load_config(model)
    messages, _ = _load_messages(cfg)
    print(f"Extract: {len(messages)} messages on {cfg['model_id']}")
    print(ActivationExtractor(model_id=cfg["model_id"]).extract_message_activations.remote(messages, cfg, run_name(cfg)))


@app.local_entrypoint()
def project(model: str = "") -> None:
    """CPU: project the cached activations onto the emotion vectors -> readout.json (re-runnable)."""
    cfg = load_config(model)
    _, meta = _load_messages(cfg)
    res = project_messages.remote(meta, cfg, run_name(cfg))
    print(res)
    if res["missing"]:
        print(f"WARNING: {len(res['missing'])} message emotions have no vector: {', '.join(res['missing'])}")
    else:
        print("All message emotions have a vector (full coverage).")


@app.local_entrypoint()
def readout(model: str = "") -> None:
    """Full pipeline: extract (GPU) then project (CPU)."""
    cfg = load_config(model)
    rn = run_name(cfg)
    messages, meta = _load_messages(cfg)
    print(f"Readout: {len(messages)} messages on {cfg['model_id']}")
    print(ActivationExtractor(model_id=cfg["model_id"]).extract_message_activations.remote(messages, cfg, rn))
    res = project_messages.remote(meta, cfg, rn)
    print(res)
